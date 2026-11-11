"""Stage-7 protocol, auditable contrasts and validation-only selection (no models)."""
from __future__ import annotations

from copy import deepcopy
import math
from .artifacts import digest

REGIMES = ('official', 'leave_paraguay_out')


def config_diff(a, b, prefix=''):
    if isinstance(a, dict) and isinstance(b, dict):
        result = {}
        for key in sorted(a.keys() | b.keys()):
            path = prefix+'.'+key if prefix else key
            if key not in a or key not in b:
                result[path] = {'before': a.get(key), 'after': b.get(key)}
            else:
                result.update(config_diff(a[key], b[key], path))
        return result
    return {} if a == b else {prefix: {'before': a, 'after': b}}


def validate_matrix(matrix):
    if (matrix['stage'] != 'phase1_stage7' or matrix['roles'] != ['val'] or
            matrix['regimes'] != list(REGIMES) or matrix['seed'] != 42 or
            matrix['ndwi']['included'] or matrix['calibration']['execute'] or
            matrix['positive_sweep'] != [1, 3, 5, 10] or
            matrix['negative_sweep'] != [0, 1, 3, 5] or
            matrix['fixed_positive'] != 3 or matrix['fixed_negative'] != 3 or
            matrix['tie_break'] != ['water_iou_desc', 'uncached_median_ms_asc', 'id_asc']):
        raise ValueError('Stage-7 fixed scope/search changed; version the protocol')
    p = matrix['profiling']
    if (p['batch_size'] != 1 or p['precision'] != 'float32' or p['autocast'] or
            p['warmup'] != 5 or p['repetitions'] != 20 or p['validation_chips'] != 1):
        raise ValueError('Stage-7 common profiling conditions changed')
    if set(matrix['hypotheses']) != {'A1', 'A2+', 'A2-', 'A3'} or any(
            not h['date'] or not h['text'] for h in matrix['hypotheses'].values()):
        raise ValueError('Dated prior hypotheses required')


def resolve_matrix(matrix, bundle, lineage):
    """One conceptual field per contrast; no threshold fitting in this stage."""
    validate_matrix(matrix)
    if (lineage['regime'] not in REGIMES or bundle['regime'] != lineage['regime'] or
            bundle['ndwi']['included'] or
            bundle['prompts']['positive_per_component'] != matrix['fixed_positive'] or
            bundle['prompts']['negative_per_component'] != matrix['fixed_negative']):
        raise ValueError('Calibration regime/budget/NDWI mismatch')
    base = dict(deepcopy(lineage), seed=matrix['seed'], role='val',
        source='dino_points', decoder='frozen', route='predictor',
        probe_sha256=digest(bundle['probe']), policy=deepcopy(bundle['prompts']),
        positive_threshold=bundle['positive_threshold'], grid_water_threshold=bundle['grid_water_threshold'])
    rows = {}

    def add(key, config, automatic=True, purpose='ablation'):
        rows[key] = {'id': key, 'config': config, 'automatic': automatic, 'purpose': purpose,
                     'config_sha256': digest(config), 'retain_for_final': True}

    add('a1_points', base)
    for source in ('assisted', 'whole_box', 'grid'):
        config = deepcopy(base); config['source'] = source
        add('a1_'+source, config, source != 'assisted')
    # Stage 3 is a descriptive historical reference, not the matched cost contrast.
    historical = deepcopy(base); historical['source'] = 'assisted_historical'
    add('historical_assisted', historical, False, 'historical_not_controlled_contrast')
    for field, values, fixed in (
            ('positive_per_component', matrix['positive_sweep'], matrix['fixed_positive']),
            ('negative_per_component', matrix['negative_sweep'], matrix['fixed_negative'])):
        for k in values:
            if k == fixed:
                continue
            config = deepcopy(base); config['policy'][field] = k
            add(('a2_positive_' if field.startswith('positive') else 'a2_negative_')+str(k), config)
    direct = deepcopy(base); direct['route'] = 'stage6_direct'
    add('a3_frozen', direct)
    tuned = deepcopy(direct); tuned['decoder'] = 'stage6_best'
    add('a3_tuned', tuned)
    contrasts = []

    def contrast(group, reference, candidate, factor):
        diff = config_diff(rows[reference]['config'], rows[candidate]['config'])
        if set(diff) != {factor}:
            raise ValueError('More than one factor in '+candidate+': '+str(diff))
        controls = deepcopy(rows[reference]['config'])
        parent = controls
        parts = factor.split('.')
        for part in parts[:-1]:
            parent = parent[part]
        parent.pop(parts[-1])
        contrasts.append({'id': group+'__'+candidate, 'group': group,
            'hypothesis': matrix['hypotheses'][group], 'reference': reference,
            'candidate': candidate, 'factor': factor, 'diff': diff,
            'controls_sha256': digest(controls)})

    for source in ('assisted', 'whole_box', 'grid'):
        contrast('A1', 'a1_points', 'a1_'+source, 'source')
    for key in rows:
        if key.startswith('a2_'):
            positive = key.startswith('a2_positive_')
            contrast('A2+' if positive else 'A2-', 'a1_points', key,
                     'policy.positive_per_component' if positive else 'policy.negative_per_component')
    contrast('A3', 'a3_frozen', 'a3_tuned', 'decoder')
    return {'rows': list(rows.values()), 'contrasts': contrasts,
        'calibration': matrix['calibration'], 'budget': {'unique_rows': len(rows),
            'validation_chips': len(lineage['validation_ids']),
            'row_chip_units': len(rows)*len(lineage['validation_ids'])}}


def require_validation(regime, ids, records):
    if regime not in REGIMES or not ids or len(set(ids)) != len(ids):
        raise ValueError('Unique validation IDs required')
    if [r['id'] for r in records] != ids or any(r['role'] != 'val' or
            r['event'] == 'Bolivia' or (regime == 'leave_paraguay_out' and r['event'] == 'Paraguay')
            for r in records):
        raise ValueError('Stage 7 accepts exactly its own validation set')


def choose_automatic(rows, expected_ids):
    """Exact IoU tie -> full image latency in identical conditions -> stable ID."""
    eligible = []
    for r in rows:
        if r['status'] != 'ok' or not r['automatic']:
            continue
        if r['validation_ids'] != expected_ids or r['metrics']['chips'] != len(expected_ids):
            raise ValueError('Selection must use complete validation, never partial/reserved metrics')
        iou = r['metrics']['pooled']['values']['water_iou']
        latency = r['efficiency']['uncached']['median_ms']
        if iou is not None and math.isfinite(iou) and 0 <= iou <= 1 and math.isfinite(latency) and latency > 0:
            eligible.append(r)
    if not eligible:
        raise ValueError('No successful automatic row with full validation and efficiency')
    conditions = {digest(r['efficiency']['conditions']) for r in eligible}
    if len(conditions) != 1:
        raise ValueError('Latency tie break requires identical hardware/precision/profile conditions')
    return min(eligible, key=lambda r: (-r['metrics']['pooled']['values']['water_iou'],
                                      r['efficiency']['uncached']['median_ms'], r['id']))


def prompt_cost(assisted, automatic):
    a, b = assisted['config'], automatic['config']
    if (a['source'] != 'assisted' or b['source'] != 'dino_points' or
            set(config_diff(a, b)) != {'source'} or
            assisted['validation_ids'] != automatic['validation_ids']):
        raise ValueError('Prompt cost needs matched frozen decoder, budget, geometry and chips')
    x = assisted['metrics']['pooled']['values']['water_iou']
    y = automatic['metrics']['pooled']['values']['water_iou']
    def subtract(left, right):
        return left-right if left is not None and right is not None else None
    by_event = {}
    for event, metric in assisted['metrics'].get('by_event', {}).items():
        left = metric['scores']['values']['water_iou']
        right = automatic['metrics']['by_event'][event]['scores']['values']['water_iou']
        delta = subtract(left, right)
        by_event[event] = {'iou_difference': delta, 'percentage_points': 100*delta if delta is not None else None}
    automatic_chips = {r['id']: r for r in automatic.get('chip_metrics', [])}
    by_chip = []
    for row in assisted.get('chip_metrics', []):
        other = automatic_chips[row['id']]
        delta = subtract(row['individual_water_iou'], other['individual_water_iou'])
        by_chip.append({'id': row['id'], 'iou_difference': delta,
            'percentage_points': 100*delta if delta is not None else None,
            'undefined_reason': row['individual_iou_exclusion'] if delta is None else None})
    return {'assisted_row': assisted['id'], 'automatic_row': automatic['id'],
            'iou_difference': x-y if x is not None and y is not None else None,
            'percentage_points': 100*(x-y) if x is not None and y is not None else None,
            'by_event': by_event, 'by_chip': by_chip,
            'budget': 'same_configured_k_per_component_actual_components_and_shortfalls_reported',
            'not_a_mathematical_upper_bound': True}


def seal(matrix, resolved, results, lineage):
    """Computational freeze only. External evidence review remains mandatory."""
    if [r['id'] for r in results] != [r['id'] for r in resolved['rows']]:
        raise ValueError('Every planned row, including failures, must be retained in order')
    for planned, result in zip(resolved['rows'], results):
        if (any(result.get(k) != v for k, v in planned.items()) or
                result['config_sha256'] != digest(result['config']) or
                result['validation_ids'] != lineage['validation_ids'] or
                result['status'] not in ('ok', 'failed') or
                (result['status'] == 'failed' and not result.get('failure'))):
            raise ValueError('Missing/changed row or unjustified failure')
    by_id = {r['id']: r for r in results}
    a, b = by_id['a3_frozen'], by_id['a3_tuned']
    if a['status'] == b['status'] == 'ok' and a['prompt_hashes'] != b['prompt_hashes']:
        raise ValueError('A3 points changed')
    winner = choose_automatic(results, lineage['validation_ids'])
    cost = (prompt_cost(by_id['a1_assisted'], by_id['a1_points'])
            if all(by_id[k]['status'] == 'ok' for k in ('a1_assisted', 'a1_points')) else
            {'undefined_reason': 'matched_contrast_has_failed_row'})
    selection = {'schema_version': 1, 'stage': 'phase1_stage7', 'regime': lineage['regime'],
        'status': 'computed_pending_external_review', 'lineage': lineage,
        'matrix_sha256': digest(matrix), 'resolved_matrix_sha256': digest(resolved),
        'rule': matrix['tie_break'], 'selected_id': winner['id'], 'selected_config': winner['config'],
        'validation_ids': lineage['validation_ids'], 'rows': results, 'prompt_cost': cost,
        'ndwi': matrix['ndwi'], 'reserved_access': [], 'external_review_required': True}
    lock = {'schema_version': 1, 'stage': 'phase1_stage7', 'regime': lineage['regime'],
        'status': 'frozen_pending_external_review', 'selection_sha256': digest(selection),
        'matrix_sha256': digest(matrix), 'lineage': lineage, 'selected_id': winner['id'],
        'final_rows': [{'id': r['id'], 'config': r['config'], 'automatic': r['automatic'],
            'config_sha256': r['config_sha256'], 'validation_status': r['status'],
            'purpose': r['purpose'], 'failure_policy': 'attempt_exact_configuration_or_record_failure'} for r in results],
        'contrasts': resolved['contrasts'], 'ndwi': matrix['ndwi'],
        'final_sets': ({'test': 90, 'bolivia': 15} if lineage['regime'] == 'official' else
                       {'test': 76, 'paraguay_reserved': 67}),
        'special_cases': {'empty_prompts': 'empty_mask_no_fallback', 'invalid': 'metrics_only_label_ne_minus_one',
            'no_water': 'exclude_individual_iou_keep_pooled_and_fp_area',
            'multimask': 'highest_predicted_iou_first_tie', 'seed': 42,
            'historical_assisted': 'stage3_policy_and_no_input_valid_clipping',
            'matched_assisted': 'same_configured_k_and_spacing_as_automatic_input_valid_clipped'},
        'reserved_access': [], 'reserved_metrics': {},
        'gate': 'stage8_requires_both_regime_locks_and_external_evidence_review',
        'stage8_authorized': False}
    return selection, lock


def verify_pair(selections, locks):
    if set(selections) != set(REGIMES) or set(locks) != set(REGIMES):
        raise ValueError('Both regimes required before any final evaluation')
    for regime in REGIMES:
        s, lock = selections[regime], locks[regime]
        if (s['regime'] != regime or lock['regime'] != regime or
                lock['selection_sha256'] != digest(s) or s['reserved_access'] or
                lock['reserved_access'] or lock['reserved_metrics'] or
                lock['final_rows'] == [] or s['matrix_sha256'] != lock['matrix_sha256'] or
                s['selected_id'] != lock['selected_id'] or s['lineage'] != lock['lineage'] or
                [r['id'] for r in s['rows']] != [r['id'] for r in lock['final_rows']]):
            raise ValueError('Broken, mixed or contaminated experimental closure')
        for row, final in zip(s['rows'], lock['final_rows']):
            if (row['config'] != final['config'] or final['config_sha256'] != digest(final['config']) or
                    row['automatic'] != final['automatic'] or row['status'] != final['validation_status']):
                raise ValueError('Final campaign rows differ from validation freeze')
    if len({s['matrix_sha256'] for s in selections.values()}) != 1:
        raise ValueError('Common preregistered matrix required')
    return {'both_computational_locks_verified': True, 'external_review': 'pending',
            'stage8_authorized': False, 'reserved_access': []}


def validation_tables(resolved, results):
    """All differences, negative outcomes and A2 curve coordinates; no refitting."""
    by_id = {r['id']: r for r in results}
    contrasts = []
    for edge in resolved['contrasts']:
        a, b = by_id[edge['reference']], by_id[edge['candidate']]
        item = dict(edge, status='ok' if a['status'] == b['status'] == 'ok' else 'undefined_failed_row')
        if item['status'] == 'ok':
            av, bv = a['metrics']['pooled']['values'], b['metrics']['pooled']['values']
            item['candidate_minus_reference'] = {k: bv[k]-av[k] if bv[k] is not None and av[k] is not None else None
                for k in ('water_iou', 'precision', 'recall', 'dice', 'miou')}
            item['fp_pixels_delta'] = b['metrics']['confusion']['fp']-a['metrics']['confusion']['fp']
        contrasts.append(item)
    curves = {}
    for kind, field in [('positive', 'positive_per_component'), ('negative', 'negative_per_component')]:
        rows = [r for r in results if r['id'] == 'a1_points' or r['id'].startswith('a2_'+kind+'_')]
        curves[kind] = [{'id': r['id'], 'k': r['config']['policy'][field], 'status': r['status'],
            'water_iou': r['metrics']['pooled']['values']['water_iou'] if r['status'] == 'ok' else None,
            'fp_pixels': r['metrics']['confusion']['fp'] if r['status'] == 'ok' else None} for r in
            sorted(rows, key=lambda r: r['config']['policy'][field])]
    points, assisted, grid = (by_id[k] for k in ('a1_points', 'a1_assisted', 'a1_grid'))
    h1 = None
    if all(r['status'] == 'ok' for r in (points, assisted, grid)):
        p, a, g = (r['metrics']['pooled']['values']['water_iou'] for r in (points, assisted, grid))
        if all(v is not None for v in (p, a, g)):
            h1 = g < p < a
    return {'contrasts': contrasts, 'a2_curves': curves, 'H1_order_observed_on_validation': h1,
        'H2': 'descriptive_curves_no_preregistered_numeric_saturation_test_no_category_claim',
        'H3': 'validation_delta_only_geographic_hypothesis_pending_stage8',
        'inference': 'one_seed_no_significance_claims'}
