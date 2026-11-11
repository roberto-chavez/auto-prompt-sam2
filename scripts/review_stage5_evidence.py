"""CPU review of user-supplied evidence; never executes archived code or models."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from auto_prompt_sam2.artifacts import digest, sha256_file, utc_now
from auto_prompt_sam2.baseline import stage2_inputs
from auto_prompt_sam2.metrics import aggregate, chip_metrics, pixel_areas_m2
from auto_prompt_sam2.stage5 import plan, supervised_labels, unpack, choose, validate_config
from auto_prompt_sam2.probe import probabilities
from auto_prompt_sam2.automatic_prompts import probability_map, generate_prompts
from auto_prompt_sam2.mask_selection import select_grid_masks


def require(condition, message):
    if not condition:
        raise ValueError(message)


def review(evidence, notebook, root, output):
    report = {'reviewed_at': utc_now(), 'status': 'failed', 'scope': 'CPU_arrays_hashes_no_models_no_training',
              'models_executed': False, 'human_review_required': True,
              'evidence_sha256': sha256_file(evidence), 'notebook_sha256': sha256_file(notebook)}
    try:
        nb = json.loads(notebook.read_text('utf-8'))
        code = [c for c in nb['cells'] if c['cell_type'] == 'code']
        require(code and all(c.get('execution_count') is not None for c in code), 'Notebook cells not executed')
        require(not any(o.get('output_type') == 'error' for c in code for o in c.get('outputs', [])),
                'Notebook errors require explicit investigation')
        with zipfile.ZipFile(evidence) as z:
            names = z.namelist()
            require(len(names) == len(set(names)) and z.testzip() is None, 'Corrupt/duplicate archive')
            require(all(not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts
                        and ':' not in n and '\\' not in n for n in names), 'Unsafe ZIP path')
            def read_report(name):
                matches = [n for n in names if n.endswith('/'+name+'.json')]
                require(len(matches) == 1, 'Missing/ambiguous report '+name)
                return json.loads(z.read(matches[0]))
            original, recovery, config = (read_report(n) for n in ('validation', 'recovery_validation', 'config'))
            identity = original['identity']
            require(identity == recovery['identity'] == config['identity'], 'Identity mismatch')
            require(original['run_id'] == recovery['run_id'], 'Recovery run mismatch')
            require(original['runtime_boot_id'] != recovery['runtime_boot_id'], 'Same runtime boot')
            validate_config(config['probe'], config['prompts'])
            require(digest(config['probe']) == identity['probe_config_sha256'] and
                    digest(config['prompts']) == identity['prompts_config_sha256'], 'Config hash mismatch')
            for local, key in (('configs/probe.yaml', 'probe'), ('configs/prompts.yaml', 'prompts')):
                require(json.loads((REPO/local).read_text('utf-8')) == config[key], 'Reviewed policy differs')
            generations, omitted = {}, []
            for name in names:
                if not name.endswith('/COMPLETE.json'):
                    continue
                marker = json.loads(z.read(name))
                base = name.rsplit('/', 1)[0]
                artifact, generation = base.rsplit('/', 2)[-2:]
                require(marker['generation'] == generation and marker['identity_sha256'] == digest(marker['identity']),
                        'Marker generation/identity mismatch')
                if base+'/payload' not in names:
                    require('_features_' in artifact and not artifact.endswith('_val_000'),
                            'Unexpected missing payload')
                    omitted.append(artifact)
                    generations[(artifact, generation)] = (None, marker)
                    continue
                payload = z.read(base+'/payload')
                require(marker['size'] == len(payload) and marker['sha256'] == hashlib.sha256(payload).hexdigest(),
                        'Payload checksum mismatch')
                generations[(artifact, generation)] = (payload, marker)
            for name, value in (('validation', original), ('recovery_validation', recovery), ('config', config)):
                require(any(p is not None and json.loads(p) == value for (a, _), (p, _) in generations.items() if a == name),
                        'Report is not bound to a verified generation')
            sources = [p for (n, _), (p, _) in generations.items() if n == 'source_snapshot']
            require(len(sources) == 1, 'Missing source snapshot')
            with zipfile.ZipFile(io.BytesIO(sources[0])) as source:
                hashes = config['source']['source_sha256']
                require(set(source.namelist()) == set(hashes) and digest(hashes) == identity['source_tree_sha256'],
                        'Source inventory mismatch')
                for name, sha in hashes.items():
                    require(not PurePosixPath(name).is_absolute() and '..' not in PurePosixPath(name).parts
                            and ':' not in name and '\\' not in name, 'Unsafe source path')
                    require(hashlib.sha256(source.read(name)).hexdigest() == sha and
                            (REPO/name).read_bytes() == source.read(name), 'Source differs: '+name)
            require(original['artifacts'] == recovery['artifacts'], 'Recovery artifact index changed')
            for name, item in original['artifacts'].items():
                _, marker = generations[(name, item['generation'])]
                require(marker['identity'] == item['identity'] and marker['sha256'] == item['sha256']
                        and marker['size'] == item['size'], 'Indexed artifact mismatch')
                require(item['identity']['stage5_identity_sha256'] == digest(identity), 'Unit from another run')
                regime = item['identity']['regime']
                require(name.startswith(regime+'_') and all(item['identity'][k] == v
                    for k, v in identity['regimes'][regime].items()), 'Cross-regime unit lineage')
            def unit(name):
                item = original['artifacts'][name]
                payload = generations[(name, item['generation'])][0]
                require(payload is not None, 'Needed feature not included in compact transport')
                return payload
            data = stage2_inputs(root, config['data'])
            verified_predictions = 0
            report['regimes'] = {}
            pp = config['prompts']
            for regime, inputs in data.items():
                plans = {role: plan(inputs['manifest'], role) for role in ('train', 'val')}
                expected = (252, 89) if regime == 'official' else (213, 75)
                require(tuple(len(plans[r]) for r in ('train', 'val')) == expected, 'Wrong role counts')
                lineage = identity['regimes'][regime]
                require(digest(inputs['manifest']) == lineage['manifest_sha256'] and
                        digest(inputs['params']) == lineage['preprocessing_sha256'], 'Stage-2 lineage differs')
                require({r: [x['id'] for x in rows] for r, rows in plans.items()} == lineage['ids'], 'IDs differ')
                allowed = {(r, x['id']) for r, rows in plans.items() for x in rows}
                for rep in (original, recovery):
                    require(all((a['role'], a['id']) in allowed for a in rep['access_log'] if a['regime'] == regime),
                            'Unpermitted access')
                    require(rep['regimes'][regime]['disk_reconstruction']['passed'], 'Disk reconstruction missing')
                search = original['regimes'][regime]['selection']
                candidates = [{'C': c, 'class_weight': w} for c in config['probe']['search']['C']
                              for w in config['probe']['search']['class_weight']]
                require(len(search['search_rows']) == len(candidates)*len(pp['positive_thresholds']), 'Search coverage differs')
                labels = {x['id']: supervised_labels(root, inputs['manifest'], x['id'], 'val') for x in plans['val']}
                areas = {x['id']: pixel_areas_m2(x['geometry']) for x in plans['val']}
                def verify_metrics(arrays, meta, row):
                    nonlocal verified_predictions
                    require(meta['regime'] == regime and meta['id'] == row['id'] and meta['role'] == 'val', 'Prediction ID mismatch')
                    require(arrays['input_valid'].dtype == bool and arrays['input_valid'].shape == (512, 512)
                        and not (arrays['mask'] & ~arrays['input_valid']).any(), 'Input validity/output clipping differs')
                    require(arrays['probability'].shape == (512, 512) and np.isfinite(arrays['probability']).all()
                        and ((arrays['probability'] >= 0) & (arrays['probability'] <= 1)).all(), 'Invalid probability map')
                    metric = chip_metrics(labels[row['id']], arrays['mask'], chip_id=row['id'], event=row['event'],
                                          pixel_area_m2=areas[row['id']])
                    require(metric == meta['metrics'], 'Prediction confusion/area mismatch')
                    verified_predictions += 1
                    return metric
                for ci, candidate in enumerate(candidates):
                    probe = json.loads(unit(f'{regime}_probe_{ci}'))
                    require(probe['candidate'] == candidate and probe['lineage']['regime'] == regime and
                        probe['lineage']['role'] == 'train' and probe['lineage']['ids'] == lineage['ids']['train'],
                        'Probe training provenance mismatch')
                    require(probe['lineage']['label_hashes'] == {x['id']: x['label_sha256'] for x in plans['train']} and
                            probe['lineage']['image_hashes'] == {x['id']: x['image_sha256'] for x in plans['train']}, 'Probe train hash mismatch')
                    require(probe['lineage']['dino_checkpoint_sha256'] == original['models']['dino']['checkpoint_sha256'],
                            'Feature model lineage differs')
                    for ti, threshold in enumerate(pp['positive_thresholds']):
                        rows = []
                        for i, row in enumerate(plans['val']):
                            arrays, meta = unpack(unit(f'{regime}_points_{ci}_{ti}_{i:03}'))
                            require(meta['probe_sha256'] == digest(probe), 'Prediction uses another probe')
                            rows.append(verify_metrics(arrays, meta, row))
                            if i == 0:
                                feat, fm = unpack(unit(f'{regime}_features_val_000'))
                                require(fm['id'] == row['id'] and fm['regime'] == regime, 'Feature provenance differs')
                                probability = probability_map(probabilities(feat['features'], probe, regime=regime))
                                require(np.array_equal(arrays['probability'], probability), 'Probability reconstruction differs')
                                generated = generate_prompts(probability, feat['input_valid'], pp, threshold)
                                component_map = generated.pop('component_map')
                                require(np.array_equal(arrays['component_map'], component_map) and generated == meta['prompts'],
                                        'Prompt reconstruction differs')
                        sr = search['search_rows'][ci*len(pp['positive_thresholds'])+ti]
                        require(sr['candidate'] == candidate and sr['positive_threshold'] == threshold and
                                sr['eligible'] == probe['fit']['converged'] and sr['metrics'] == aggregate(rows), 'Selection row differs')
                require(choose(search['search_rows']) == search['selected'], 'Point winner differs')
                grid_rows = []
                for gi, threshold in enumerate(pp['grid']['water_thresholds']):
                    rows = []
                    for i, row in enumerate(plans['val']):
                        arrays, meta = unpack(unit(f'{regime}_grid_{gi}_{i:03}'))
                        rows.append(verify_metrics(arrays, meta, row))
                        ga, gm = unpack(unit(f'{regime}_grid_candidates_{i:03}'))
                        cs = [dict(c, segmentation=m) for c, m in zip(gm['candidates'], ga['masks'])]
                        selected = select_grid_masks(cs, arrays['probability'], arrays['input_valid'], pp['grid'], threshold)
                        require(np.array_equal(selected['mask'], arrays['mask']) and
                            selected['decisions'] == meta['grid']['decisions'], 'Grid reconstruction differs')
                    grid_rows.append({'water_threshold': threshold, 'eligible': True, 'metrics': aggregate(rows)})
                grid_selection = original['regimes'][regime]['grid_selection']
                require(grid_selection['rows'] == grid_rows and grid_selection['selected'] == choose(grid_rows), 'Grid winner differs')
                report['regimes'][regime] = {'train_chips': expected[0], 'val_chips': expected[1],
                    'selection_recomputed': True, 'confusions_verified': True,
                    'first_val_probabilities_prompts_grid_reconstructed': True}
            for rep in (original, recovery):
                require(rep['reserved_access'] == [] and rep['final_evaluations'] == [] and
                    rep['decoder_training'] is False and rep['ndwi']['included'] is False, 'Scope violation')
                require(all(c['exit_code'] == 0 for c in rep['checks']), 'Colab checks failed')
                for kind in ('dino', 'sam'):
                    require(rep['models'][kind]['state_sha256'] == rep['models'][kind]['state_sha256_after'] ==
                        original['models'][kind]['state_sha256'], 'Base state changed')
                    require(rep['models'][kind]['checkpoint_sha256'] == original['models'][kind]['checkpoint_sha256'], 'Base checkpoint changed')
            require(recovery['probe_training_executed'] is False, 'Recovery retrained probe')
            report.update(status='automated_audit_passed_pending_human_review', predictions_verified=verified_predictions,
                omitted_feature_payloads=len(omitted), limitations=[
                    'Compact ZIP omits most feature payloads: their markers and Colab full-hash checks are audited, not independently recomputed.',
                    'Only first val shard per regime is reconstructed without models. Human notebook/output interpretation remains required.'])
    finally:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', 'utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--notebook', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True, help='Existing stage-2 root, development labels only')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(review(args.evidence, args.notebook, args.root, args.output), indent=2))
