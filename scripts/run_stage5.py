"""Stage 5: user-run Colab GPU only. Train probes; validate frozen SAM; recover."""
from __future__ import annotations

import argparse
import gc
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from auto_prompt_sam2.artifacts import ArtifactStore, Run, canonical_json, digest, sha256_file, utc_now
from auto_prompt_sam2.sam_adapter import require_colab_gpu


def source_paths():
    from auto_prompt_sam2.runtime import source_files
    return sorted(set(source_files(REPO)) | {REPO / p for p in (
        'configs/data.yaml', 'configs/baseline.yaml', 'configs/profiling.yaml',
        'configs/probe.yaml', 'configs/prompts.yaml', 'requirements-stage2.txt',
        'notebooks/05_prompts_automaticos.ipynb', 'docs/project/validacion_etapa5.md')})


def snapshot():
    from auto_prompt_sam2.runtime import provenance
    source = provenance(REPO)
    source['source_sha256'] = {p.relative_to(REPO).as_posix(): sha256_file(p) for p in source_paths()}
    source['source_tree_sha256'] = digest(source['source_sha256'])
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in source_paths():
            z.write(p, p.relative_to(REPO).as_posix())
    return source, output.getvalue()


def export_evidence(root, run, report):
    """Compact transport: features stay on Drive; complete hash inventory included.

    Includes one val feature shard/regime for independent recovery checks, all
    prediction/probe/target payloads, all markers, and every source file. Does not
    package licensed base weights or raw rasters. Never claims full feature audit.
    """
    output = root / 'stage5/evidence' / run.run_id
    output.mkdir(parents=True, exist_ok=True)
    for name in ('validation', 'recovery_validation', 'config', 'environment', 'failure'):
        try:
            (output / (name+'.json')).write_bytes(run.store.read(name, run.identity)[0])
        except FileNotFoundError:
            pass
    target = output.with_suffix('.zip')
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, item in report['artifacts'].items():
            base = run.store.root / name / item['generation']
            z.write(base / 'COMPLETE.json', (base / 'COMPLETE.json').relative_to(root).as_posix())
            if item['category'] != 'features' or name.endswith('_val_000'):
                z.write(base / 'payload', (base / 'payload').relative_to(root).as_posix())
        # Reports, run state and immutable source snapshot have run-wide identities.
        for name in ('validation', 'recovery_validation', 'config', 'environment', 'failure', 'state', 'source_snapshot'):
            folder = run.store.root / name
            if folder.exists():
                for p in sorted(folder.rglob('*')):
                    if p.is_file():
                        z.write(p, p.relative_to(root).as_posix())
        for p in output.glob('*.json'):
            z.write(p, p.relative_to(root).as_posix())
    print(json.dumps({'evidence_zip': str(target), 'sha256': sha256_file(target),
        'full_feature_shards': 'Remain on Drive; only first val shard per regime included.'}, indent=2), flush=True)


class Units:
    def __init__(self, run, report):
        self.run, self.report = run, report

    def identity(self, regime):
        return {'stage5_identity_sha256': digest(self.run.identity), 'regime': regime,
                **self.run.identity['regimes'][regime]}

    def read(self, name, regime, category):
        payload, marker = self.run.store.read(name, self.identity(regime))
        self.report['artifacts'][name] = {'generation': marker['generation'], 'sha256': marker['sha256'],
            'size': marker['size'], 'identity': marker['identity'], 'category': category}
        return payload

    def get(self, name, regime, category, compute):
        try:
            return self.read(name, regime, category)
        except FileNotFoundError:
            payload = compute()
            self.run.store.publish(name, payload, self.identity(regime))
            restored = self.read(name, regime, category)
            if restored != payload:
                raise ValueError('Roundtrip mismatch')
            return restored


def feature_name(regime, role, index):
    return f'{regime}_features_{role}_{index:03}'


def target_name(regime, role, index):
    return f'{regime}_targets_{role}_{index:03}'


def run_stage5(root, mode):
    require_colab_gpu()  # Before imports, downloads, fitting, image I/O or GPU work.
    import numpy as np
    import torch
    from auto_prompt_sam2.baseline import stage2_inputs
    from auto_prompt_sam2.features import load_frozen_dino, token_layout, tokens_to_grid
    from auto_prompt_sam2.preprocessing import prepare, sam_predictor_image, Coordinates
    from auto_prompt_sam2.probe import patch_targets, fit_probe, probabilities
    from auto_prompt_sam2.automatic_prompts import probability_map, infer_points
    from auto_prompt_sam2.mask_selection import make_grid_generator, select_grid_masks
    from auto_prompt_sam2.metrics import chip_metrics, pixel_areas_m2, aggregate
    from auto_prompt_sam2.stage5 import (validate_config, plan, read_image, supervised_labels,
                                        no_label_inference, pack, unpack, choose)
    from auto_prompt_sam2.runtime import load_config, seed_everything, inventory, verify_installed_pins
    from auto_prompt_sam2.sam_adapter import load_frozen_sam, assert_frozen, state_fingerprint

    root = Path(root)
    os.environ['AUTO_PROMPT_PERSIST_ROOT'] = str(root)
    env = load_config(REPO)
    pc = json.loads((REPO / 'configs/probe.yaml').read_text('utf-8'))
    pp = json.loads((REPO / 'configs/prompts.yaml').read_text('utf-8'))
    dc = json.loads((REPO / 'configs/data.yaml').read_text('utf-8'))
    validate_config(pc, pp)
    seed_everything(pc['seed'])
    torch.set_num_threads(1)
    inputs = stage2_inputs(root, dc)
    plans = {r: {role: plan(v['manifest'], role) for role in pc['roles']} for r, v in inputs.items()}
    source, source_zip = snapshot()
    identity = {'stage': 'phase1_stage5', 'probe_config_sha256': digest(pc), 'prompts_config_sha256': digest(pp),
        'source_tree_sha256': source['source_tree_sha256'], 'models': env['models'],
        'regimes': {r: {'manifest_sha256': digest(v['manifest']), 'preprocessing_sha256': digest(v['params']),
            'manifest_generation': v['manifest_generation'], 'parameter_generation': v['parameter_generation'],
            'ids': {role: [row['id'] for row in plans[r][role]] for role in pc['roles']}}
            for r, v in inputs.items()}}
    active = ArtifactStore(root / 'stage5')
    prior = None
    if mode == 'prepare':
        run = Run.create(root / 'stage5', identity)
        active.publish('active_run', canonical_json({'run_id': run.run_id}), identity)
        run.transition('running')
        run.store.publish('source_snapshot', source_zip, identity)
        run.store.publish('config', canonical_json({'identity': identity, 'source': source,
            'probe': pc, 'prompts': pp, 'data': dc, 'environment': env}), identity)
    else:
        run = Run(root / 'stage5', json.loads(active.read('active_run', identity)[0])['run_id'], identity)
        if mode == 'recover':
            if run.state != 'complete':
                raise ValueError('Recovery requires completed computation')
            prior = json.loads(run.store.read('validation', identity)[0])
        else:
            if run.state == 'complete':
                raise ValueError('Use recover for a completed run')
            run.resume()  # User explicitly selects resume after stopping previous writer.
    report = {'stage': 'phase1_stage5', 'created_at': utc_now(), 'run_id': run.run_id, 'mode': mode,
        'identity': identity, 'models': {}, 'regimes': {}, 'artifacts': {}, 'access_log': [],
        'reserved_access': [], 'decoder_training': False, 'final_evaluations': [],
        'probe_training_executed': False, 'probe_training_units': [],
        'ndwi': pc['ndwi'], 'runtime_boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'closure': 'pending_external_review_of_user_notebook_json_and_artifacts',
        'license': 'publisher_reply_pending_historical_salvedad_retained',
        'ai_use': 'Codex prepared implementation/tests/notebook; human Colab execution and external review required.'}
    units = Units(run, report)

    def train_payload(name, x, y, lineage, candidate):
        result = fit_probe(x, y, lineage=lineage, candidate=candidate, config=pc)
        report['probe_training_executed'] = True
        report['probe_training_units'].append(name)
        return canonical_json(result)

    def access(regime, row, kind):
        report['access_log'].append({'regime': regime, 'id': row['id'], 'role': row['role'], 'kind': kind})

    def extract(model, regime, role, row):
        rgb, valid = read_image(root, inputs[regime]['manifest'], row['id'], role)
        access(regime, row, 'image_features_no_labels')
        tensor = prepare(rgb, valid, inputs[regime]['params'], model='dino')['pixel_values'][None].cuda()
        layout = token_layout((512, 512), 16, model.config.num_register_tokens, model.config.hidden_size)
        with torch.inference_mode():
            grid = tokens_to_grid(model(pixel_values=tensor).last_hidden_state, layout)
            features = grid[0].permute(1, 2, 0).float().cpu().numpy().copy()
        if features.shape != (32, 32, 1024) or not np.isfinite(features).all():
            raise ValueError('DINO feature geometry/value mismatch')
        return pack({'features': features, 'input_valid': valid}, {'id': row['id'], 'role': role,
            'regime': regime, 'image_sha256': row['image_sha256'], 'token_layout': layout,
            'coordinates': Coordinates((512, 512), (512, 512)).metadata(),
            'feature_layer': pc['features'], 'preprocessing_sha256': digest(inputs[regime]['params'])})

    def targets(regime, role, row):
        labels = supervised_labels(root, inputs[regime]['manifest'], row['id'], role)
        access(regime, row, 'labels_targets_only')
        target, balance = patch_targets(labels, pc['targets'])
        return pack({'targets': target}, {'id': row['id'], 'role': role, 'regime': regime,
            'label_sha256': row['label_sha256'], 'balance': balance, 'policy': pc['targets']})

    def evaluate(regime, row, mask):
        # Invoked only AFTER automatic prediction returns; validity never flows back.
        labels = supervised_labels(root, inputs[regime]['manifest'], row['id'], 'val')
        access(regime, row, 'labels_post_inference_metrics')
        return chip_metrics(labels, mask, chip_id=row['id'], event=row['event'],
                            pixel_area_m2=pixel_areas_m2(row['geometry']))

    def point_prediction(predictor, regime, row, feat, probe, threshold, policy=None):
        rgb, valid = read_image(root, inputs[regime]['manifest'], row['id'], 'val')
        access(regime, row, 'image_automatic_points')
        if not np.array_equal(valid, feat['input_valid']):
            raise ValueError('Image validity differs from feature shard')
        p = probability_map(probabilities(feat['features'], probe, regime=regime))
        image = sam_predictor_image(rgb, valid, inputs[regime]['params'])
        result, prompts = no_label_inference(infer_points, predictor, image, p, valid, policy or pp, threshold)
        component_map = prompts.pop('component_map')
        metadata = {'id': row['id'], 'regime': regime, 'role': 'val', 'event': row['event'],
            'image_sha256': row['image_sha256'], 'probe_sha256': digest(probe),
            'prompts': prompts, 'inference': {k: v for k, v in result.items() if k != 'mask'},
            'metrics': evaluate(regime, row, result['mask'])}
        return pack({'mask': result['mask'], 'probability': p, 'component_map': component_map,
                     'input_valid': valid}, metadata)

    try:
        report['pins'] = verify_installed_pins(REPO, env, linux_lock=True)
        from importlib.metadata import version
        for line in (REPO / 'requirements-stage2.txt').read_text().splitlines():
            if line and not line.startswith('#'):
                name, expected = line.split('==')
                if version(name) != expected:
                    raise ValueError('Stage-2 dependency mismatch: '+name)
        for args in (['-m', 'pip', 'check'], ['-m', 'unittest', 'discover', '-s', 'tests', '-v']):
            result = subprocess.run([sys.executable, *args], cwd=REPO, capture_output=True, text=True)
            report.setdefault('checks', []).append({'args': args, 'exit_code': result.returncode,
                'stdout': result.stdout, 'stderr': result.stderr})
            print(result.stdout, result.stderr, flush=True)
            result.check_returncode()
        report['environment'] = inventory(root, REPO)
        run.store.publish('environment', canonical_json(report['environment']), identity)
        if prior:
            if report['runtime_boot_id'] == prior['runtime_boot_id']:
                raise ValueError('Recover requires a newly created Colab runtime')
            for name, item in prior['artifacts'].items():
                payload, marker = run.store.read(name, item['identity'], item['generation'])
                if marker['sha256'] != item['sha256']:
                    raise ValueError('Recovered artifact mismatch')
            report['artifacts'] = prior['artifacts'].copy()
            report['regimes'] = prior['regimes'].copy()
            report['different_runtime_boot'] = True
        # DINO once, then release before allocating the train matrix / SAM.
        expected = prior['models']['dino']['checkpoint_sha256'] if prior else None
        dino, dino_info = load_frozen_dino(env['models']['dinov3'], root, offline=bool(prior), expected_sha256=expected)
        report['models']['dino'] = dino_info
        for regime in pc['regimes']:
            if prior:
                row = plans[regime]['val'][0]
                name = feature_name(regime, 'val', 0)
                original, _ = unpack(units.read(name, regime, 'features'))
                repeated, _ = unpack(extract(dino, regime, 'val', row))
                if not np.array_equal(original['input_valid'], repeated['input_valid']) or not np.allclose(
                        original['features'], repeated['features'], atol=1e-6, rtol=1e-5):
                    raise ValueError('Feature recovery mismatch')
                report['regimes'][regime]['feature_recovery'] = 'first_val_shard_matches_at_1e-6_1e-5'
            else:
                balances = {}
                for role in pc['roles']:
                    balances[role] = {k: 0 for k in ('water', 'nonwater', 'ignored', 'no_valid')}
                    for i, row in enumerate(plans[regime][role]):
                        units.get(feature_name(regime, role, i), regime, 'features',
                                  lambda: extract(dino, regime, role, row))
                        target = units.get(target_name(regime, role, i), regime, 'targets',
                                           lambda: targets(regime, role, row))
                        _, meta = unpack(target)
                        for k in balances[role]:
                            balances[role][k] += meta['balance'][k]
                        if i % 20 == 0:
                            print(regime, role, i, '/', len(plans[regime][role]), 'shards persisted', flush=True)
                report['regimes'][regime] = {'patch_balance': balances}
        assert_frozen(dino)
        dino_info['state_sha256_after'] = state_fingerprint(dino)
        if dino_info['state_sha256_after'] != dino_info['state_sha256']:
            raise ValueError('DINO weights changed')
        if prior and dino_info['state_sha256'] != prior['models']['dino']['state_sha256']:
            raise ValueError('DINO reload state mismatch')
        del dino
        gc.collect()
        torch.cuda.empty_cache()

        candidates = [{'C': c, 'class_weight': w} for c in pc['search']['C'] for w in pc['search']['class_weight']]
        if not prior:
            for regime in pc['regimes']:
                xs, ys = [], []
                for i, row in enumerate(plans[regime]['train']):
                    f, fm = unpack(units.read(feature_name(regime, 'train', i), regime, 'features'))
                    t, tm = unpack(units.read(target_name(regime, 'train', i), regime, 'targets'))
                    if fm['id'] != row['id'] or tm['id'] != row['id']:
                        raise ValueError('Feature/target ID mismatch')
                    use = t['targets'] >= 0
                    xs.append(f['features'][use])
                    ys.append(t['targets'][use])
                x, y = np.concatenate(xs), np.concatenate(ys)
                del xs, ys, f, t
                lineage = {'regime': regime, 'role': 'train', 'ids': identity['regimes'][regime]['ids']['train'],
                    'image_hashes': {r['id']: r['image_sha256'] for r in plans[regime]['train']},
                    'label_hashes': {r['id']: r['label_sha256'] for r in plans[regime]['train']},
                    'manifest_sha256': identity['regimes'][regime]['manifest_sha256'],
                    'preprocessing_sha256': identity['regimes'][regime]['preprocessing_sha256'],
                    'dino_checkpoint_sha256': dino_info['checkpoint_sha256'], 'feature_layer': pc['features']}
                for ci, candidate in enumerate(candidates):
                    name = f'{regime}_probe_{ci}'
                    probe = json.loads(units.get(name, regime, 'probe',
                        lambda: train_payload(name, x, y, lineage, candidate)))
                    print(regime, candidate, probe['fit'], flush=True)
                del x, y
                gc.collect()
                torch.cuda.empty_cache()

        expected = prior['models']['sam']['checkpoint_sha256'] if prior else None
        predictor, sam_info = load_frozen_sam(env['models']['sam2'], root, offline=bool(prior), expected_sha256=expected)
        report['models']['sam'] = sam_info
        for regime in pc['regimes']:
            if not prior:
                search_rows = []
                for ci, candidate in enumerate(candidates):
                    probe = json.loads(units.read(f'{regime}_probe_{ci}', regime, 'probe'))
                    for ti, threshold in enumerate(pp['positive_thresholds']):
                        metrics, diagnostics = [], []
                        aux_correct, aux_n, aux_logloss = 0, 0, 0.
                        for i, row in enumerate(plans[regime]['val']):
                            feat, _ = unpack(units.read(feature_name(regime, 'val', i), regime, 'features'))
                            payload = units.get(f'{regime}_points_{ci}_{ti}_{i:03}', regime, 'prediction',
                                lambda: point_prediction(predictor, regime, row, feat, probe, threshold))
                            arrays, meta = unpack(payload)
                            metrics.append(meta['metrics'])
                            target, _ = unpack(units.read(target_name(regime, 'val', i), regime, 'targets'))
                            use = target['targets'] >= 0
                            yp = probabilities(feat['features'], probe, regime=regime)[use]
                            yt = target['targets'][use]
                            aux_correct += int(((yp >= .5) == yt).sum())
                            aux_n += len(yt)
                            yp = np.clip(yp.astype(np.float64), 1e-12, 1-1e-12)
                            aux_logloss += float(-(yt*np.log(yp)+(1-yt)*np.log1p(-yp)).sum())
                            diagnostics.append({'id': row['id'], 'components': len(meta['prompts']['components']),
                                'positive_points': sum(sum(c['point_labels']) for c in meta['prompts']['components']),
                                'negative_points': sum(len(c['point_labels'])-sum(c['point_labels']) for c in meta['prompts']['components'])})
                        # Probe patch diagnostics remain auxiliary (no candidate choice by these).
                        search_rows.append({'candidate_id': f'{ci}_{ti}', 'probe_index': ci, 'candidate': candidate,
                            'positive_threshold': threshold, 'eligible': probe['fit']['converged'],
                            'fit': probe['fit'], 'metrics': aggregate(metrics), 'point_counts': diagnostics,
                            'auxiliary_probe_validation': {'patches': aux_n,
                                'accuracy_at_0_5': aux_correct/aux_n if aux_n else None,
                                'log_loss': aux_logloss/aux_n if aux_n else None,
                                'used_for_selection': False},
                            'validation_ids': identity['regimes'][regime]['ids']['val']})
                        print(regime, ci, ti, search_rows[-1]['metrics']['pooled']['values'], flush=True)
                winner = choose(search_rows)
                selection = {'regime': regime, 'selection_rule': pc['selection'], 'search_rows': search_rows,
                    'selected': winner, 'validation_ids': identity['regimes'][regime]['ids']['val'],
                    'calibration_only_not_stage7_lock': True}
                units.get(f'{regime}_selection', regime, 'selection', lambda: canonical_json(selection))
                report['regimes'][regime]['selection'] = selection
            selection = report['regimes'][regime]['selection']
            winner = selection['selected']
            ci = winner['probe_index']
            ti = pp['positive_thresholds'].index(winner['positive_threshold'])
            probe = json.loads(units.read(f'{regime}_probe_{ci}', regime, 'probe'))
            # Persist a portable bundle: scaler/weights, policy and train/val lineage.
            bundle = {'probe': probe, 'prompts': pp, 'positive_threshold': winner['positive_threshold'],
                'regime': regime, 'preprocessing': inputs[regime]['params'],
                'models': env['models'], 'base_checkpoint_sha256': {
                    'dino': dino_info['checkpoint_sha256'], 'sam': sam_info['checkpoint_sha256']},
                'validation_ids': selection['validation_ids'], 'ndwi': pc['ndwi']}
            if not prior:
                units.get(f'{regime}_inference_bundle', regime, 'bundle', lambda: canonical_json(bundle))
                generator = make_grid_generator(predictor, pp['grid'])
                grid_metrics = [[] for _ in pp['grid']['water_thresholds']]
                for i, row in enumerate(plans[regime]['val']):
                    feat, _ = unpack(units.read(feature_name(regime, 'val', i), regime, 'features'))
                    p = probability_map(probabilities(feat['features'], probe, regime=regime))
                    # Cache quality-filtered class-agnostic masks ONCE per chip.
                    def grid_candidates():
                        rgb, valid = read_image(root, inputs[regime]['manifest'], row['id'], 'val')
                        access(regime, row, 'image_grid_no_labels')
                        image = sam_predictor_image(rgb, valid, inputs[regime]['params'])
                        generated = no_label_inference(generator.generate, image)
                        masks = np.stack([c['segmentation'] for c in generated]) if generated else np.zeros((0, 512, 512), bool)
                        return pack({'masks': masks}, {'candidates': [{k: float(c[k]) for k in
                            ('predicted_iou', 'stability_score')} for c in generated], 'grid_positions': 1024,
                            'quality_filters_and_nms': pp['grid']})
                    ga, gm = unpack(units.get(f'{regime}_grid_candidates_{i:03}', regime, 'grid_candidates', grid_candidates))
                    for gi, threshold in enumerate(pp['grid']['water_thresholds']):
                        def grid_prediction():
                            cs = [dict(c, segmentation=m) for c, m in zip(gm['candidates'], ga['masks'])]
                            result = no_label_inference(select_grid_masks, cs, p, feat['input_valid'], pp['grid'], threshold)
                            mask = result.pop('mask')
                            return pack({'mask': mask, 'probability': p, 'input_valid': feat['input_valid']}, {'id': row['id'], 'regime': regime,
                                'role': 'val', 'event': row['event'], 'probe_sha256': digest(probe),
                                'grid': result, 'metrics': evaluate(regime, row, mask)})
                        _, meta = unpack(units.get(f'{regime}_grid_{gi}_{i:03}', regime, 'prediction', grid_prediction))
                        grid_metrics[gi].append(meta['metrics'])
                    print(regime, 'grid', i+1, '/', len(plans[regime]['val']), flush=True)
                rows = [{'water_threshold': t, 'eligible': True, 'metrics': aggregate(m)}
                        for t, m in zip(pp['grid']['water_thresholds'], grid_metrics)]
                grid_selection = {'rows': rows, 'selected': choose(rows), 'probe_index_fixed_from_points': ci,
                                  'validation_ids': selection['validation_ids'], 'tie': 'first_threshold'}
                units.get(f'{regime}_grid_selection', regime, 'selection', lambda: canonical_json(grid_selection))
                report['regimes'][regime]['grid_selection'] = grid_selection
                # Bundle includes independently calibrated grid threshold.
                bundle['grid_water_threshold'] = grid_selection['selected']['water_threshold']
                units.get(f'{regime}_complete_bundle', regime, 'bundle', lambda: canonical_json(bundle))
                generator.predictor.reset_predictor()
                del generator

            # Reload everything from persistent generations, never hidden notebook variables.
            loaded = json.loads(units.read(f'{regime}_complete_bundle', regime, 'bundle'))
            row = plans[regime]['val'][0]
            feat, _ = unpack(units.read(feature_name(regime, 'val', 0), regime, 'features'))
            old, old_meta = unpack(units.read(f'{regime}_points_{ci}_{ti}_000', regime, 'prediction'))
            rep, meta = unpack(point_prediction(predictor, regime, row, feat, loaded['probe'],
                                                loaded['positive_threshold'], loaded['prompts']))
            tol = pp['repeat_tolerance']
            if (not np.allclose(old['probability'], rep['probability'], atol=tol['probability_atol'], rtol=tol['probability_rtol']) or
                    np.count_nonzero(old['mask'] != rep['mask']) > tol['mask_disagreement_pixels'] or
                    old_meta['prompts'] != meta['prompts']):
                raise ValueError('Disk reconstruction probability/prompt/mask mismatch')
            if not np.allclose([c['predicted_iou'] for c in old_meta['inference']['components']],
                               [c['predicted_iou'] for c in meta['inference']['components']], atol=1e-6, rtol=1e-5):
                raise ValueError('Disk reconstruction SAM scores mismatch')
            report['regimes'][regime]['disk_reconstruction'] = {'passed': True, 'id': row['id'],
                'mask_disagreement_pixels': 0, 'probability_tolerance': tol,
                'label_reader_disabled_during_inference': True, 'scope': 'first_val_chip'}
            # Grid reconstruction from persisted class-agnostic masks is also label-free.
            ga, gm = unpack(units.read(f'{regime}_grid_candidates_000', regime, 'grid_candidates'))
            cs = [dict(c, segmentation=m) for c, m in zip(gm['candidates'], ga['masks'])]
            gr = no_label_inference(select_grid_masks, cs, rep['probability'], feat['input_valid'],
                                    loaded['prompts']['grid'], loaded['grid_water_threshold'])
            gi = pp['grid']['water_thresholds'].index(loaded['grid_water_threshold'])
            old_grid, old_grid_meta = unpack(units.read(f'{regime}_grid_{gi}_000', regime, 'prediction'))
            if not np.array_equal(gr['mask'], old_grid['mask']) or gr['decisions'] != old_grid_meta['grid']['decisions']:
                raise ValueError('Grid semantic reconstruction mismatch')
            report['regimes'][regime]['disk_reconstruction']['grid_from_persisted_candidates'] = True
        assert_frozen(predictor.model)
        sam_info['state_sha256_after'] = state_fingerprint(predictor.model)
        if sam_info['state_sha256_after'] != sam_info['state_sha256']:
            raise ValueError('SAM weights/decoder changed')
        if prior and sam_info['state_sha256'] != prior['models']['sam']['state_sha256']:
            raise ValueError('SAM reload state mismatch')
        report['status'] = 'computation_passed_pending_external_review'
        report['all_artifacts_hash_verified'] = True
        run.store.publish('recovery_validation' if prior else 'validation', canonical_json(report), identity)
        if not prior:
            run.transition('complete')
        export_evidence(root, run, report)
        return report
    except BaseException as error:
        run.store.publish('failure', canonical_json({'at': utc_now(), 'type': type(error).__name__,
            'message': str(error), 'partial_report': report}), identity)
        if not prior and run.state == 'running':
            run.transition('interrupted')
        export_evidence(root, run, report)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'resume', 'recover'))
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    run_stage5(args.root, args.mode)
