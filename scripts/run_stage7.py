"""Stage 7 only: user-run Colab validation ablations and computational locks."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
import zipfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'src'))
from auto_prompt_sam2.artifacts import ArtifactStore, canonical_json, digest, sha256_file, utc_now
from auto_prompt_sam2.experiments import (resolve_matrix, validate_matrix, require_validation,
                                         seal, verify_pair, validation_tables)


def source_paths():
    from auto_prompt_sam2.runtime import source_files
    return sorted(set(source_files(REPO)) | set((REPO/'configs').glob('*.yaml')) | {
        REPO/p for p in ('configs/stage6_dependencies.json', 'configs/stage7_dependencies.json',
        'requirements-stage2.txt', 'notebooks/07_ablaciones_y_seleccion.ipynb',
        'docs/project/validacion_etapa7.md', 'docs/WAIMLAP/Abstract_CostOfThePrompt.md',
        'docs/project/cuaderno_decisiones.md')})


def export_evidence(root, store, identity):
    """Complete stage-7 generations/predictions, sources, JSON. No raw data/weights."""
    target = store.root/'evidence.zip'
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(store.root.glob('*/*/COMPLETE.json')):
            marker = json.loads(path.read_text('utf-8'))
            name = path.parent.parent.name
            store.read(name, identity, marker['generation'])
            for filename in ('COMPLETE.json', 'payload'):
                file = path.parent/filename
                archive.write(file, 'generations/'+file.relative_to(store.root).as_posix())
        for path in sorted((store.root/'evidence').glob('*')):
            archive.write(path, 'evidence/'+path.name)
        for path in source_paths():
            archive.write(path, 'source/'+path.relative_to(REPO).as_posix())
    print('EVIDENCE', target, 'SHA256', sha256_file(target), flush=True)


def main(root, mode):
    from auto_prompt_sam2.sam_adapter import require_colab_gpu
    require_colab_gpu()  # Before data reads, model imports, allocations or downloads.
    import numpy as np
    import torch
    from auto_prompt_sam2.baseline import stage2_inputs
    from auto_prompt_sam2.runtime import load_config, seed_everything, inventory, verify_installed_pins
    from auto_prompt_sam2.sam_adapter import load_frozen_sam, assert_frozen, state_fingerprint
    from auto_prompt_sam2.features import load_frozen_dino
    from auto_prompt_sam2.stage5 import plan, pack, unpack
    from auto_prompt_sam2.stage7 import checked_read, stage6_inputs, ValidationEngine, prompt_diagnostics
    from auto_prompt_sam2.metrics import aggregate
    from auto_prompt_sam2.training import deserialize, fingerprints, audit_changes
    from run_stage6 import read_dependency, verify_bundle

    root = Path(root).resolve()
    if not root.is_relative_to(Path('/content/drive').resolve()):
        raise ValueError('Mounted persistent Drive root required')
    os.environ['AUTO_PROMPT_PERSIST_ROOT'] = str(root)
    matrix = json.loads((REPO/'configs/ablations.yaml').read_text('utf-8'))
    validate_matrix(matrix)
    env = load_config(REPO)
    verify_installed_pins(REPO, env, linux_lock=True)
    from importlib.metadata import version
    for line in (REPO/'requirements-stage2.txt').read_text('utf-8').splitlines():
        if line and not line.startswith('#'):
            name, expected = line.split('==')
            if version(name) != expected:
                raise ValueError('Stage-2 dependency changed: '+name)
    seed_everything(42)
    torch.set_num_threads(1)
    inputs = stage2_inputs(root, json.loads((REPO/'configs/data.yaml').read_text('utf-8')))
    deps = json.loads((REPO/'configs/stage6_dependencies.json').read_text('utf-8'))
    seventh = json.loads((REPO/'configs/stage7_dependencies.json').read_text('utf-8'))
    first = deps['stage1']
    first_store = ArtifactStore(root/'environment/runs'/first['run_id'])
    for name, markers in first['markers'].items():
        for marker in markers:
            checked_read(first_store, name, marker)
    prior_stages = {}
    for stage in ('stage3', 'stage4', 'stage5'):
        store, report, marker = read_dependency(root, stage, deps[stage])
        for regime, inp in inputs.items():
            previous = report['identity']['regimes'][regime]
            if (previous['manifest_sha256'] != digest(inp['manifest']) or
                    previous['preprocessing_sha256'] != digest(inp['params'])):
                raise ValueError('Prior data lineage differs: '+stage)
        prior_stages[stage] = {'store': store, 'report': report, 'marker': marker}
    prior_store, prior = prior_stages['stage5']['store'], prior_stages['stage5']['report']
    source = {p.relative_to(REPO).as_posix(): sha256_file(p) for p in source_paths()}
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    selections, locks = {}, {}
    stores = {}
    for regime in matrix['regimes']:
        seed_everything(42)
        inp = inputs[regime]
        records = plan(inp['manifest'], 'val')
        ids = [r['id'] for r in records]
        require_validation(regime, ids, records)
        bundle = json.loads(checked_read(prior_store, regime+'_complete_bundle', prior['artifacts'][regime+'_complete_bundle']))
        old_identity = seventh['stage6'][regime]['identity']
        verify_bundle(bundle, regime, inp, old_identity['ids'], env['models'])
        if (bundle['base_checkpoint_sha256']['sam'] != prior['models']['sam']['checkpoint_sha256'] or
                bundle['base_checkpoint_sha256']['dino'] != prior['models']['dino']['checkpoint_sha256'] or
                bundle['probe']['lineage']['dino_checkpoint_sha256'] != bundle['base_checkpoint_sha256']['dino']):
            raise ValueError('Model checkpoint/probe/feature lineage differs')
        spec = seventh['stage6'][regime]
        sixth = stage6_inputs(root, spec, inp, bundle, REPO)
        sixth['best_epoch'] = spec['best_epoch']
        lineage = {'regime': regime, 'validation_ids': ids,
            'manifest_sha256': digest(inp['manifest']), 'preprocessing_sha256': digest(inp['params']),
            'bundle_sha256': digest(bundle), 'base_checkpoint_sha256': bundle['base_checkpoint_sha256'],
            'stage6_decoder_sha256': spec['artifacts']['decoder_export']['sha256'],
            'stage6_decoder_artifact': spec['artifacts']['decoder_export'], 'stage6_store_path': spec['store_path'],
            'stage6_validation_sha256': spec['artifacts']['validation']['sha256'],
            'dependencies_sha256': digest({'stage1_6': deps, 'stage6_reviewed': spec}),
            'source_sha256': source}
        resolved = resolve_matrix(matrix, bundle, lineage)
        identity = {'stage': 'phase1_stage7', 'regime': regime,
            'matrix_sha256': digest(matrix), 'resolved_sha256': digest(resolved), 'source_sha256': source}
        store = ArtifactStore(root/'regimes'/regime/'stage7'/digest(identity))
        stores[regime] = (store, identity)
        evidence = store.root/'evidence'; evidence.mkdir(exist_ok=True)

        def publish(name, value):
            payload = canonical_json(value)
            store.publish(name, payload, identity)
            (evidence/(name+'.json')).write_bytes(payload)
        try:
            if mode == 'prepare':
                try:
                    store.read('prepared', identity)
                except FileNotFoundError:
                    publish('prepared', {'at': utc_now(), 'identity': identity})
                else:
                    raise ValueError('Already prepared; stop previous writer and use resume or recover')
            else:
                store.read('prepared', identity)
            if mode == 'recover':
                old_validation = json.loads(store.read('validation', identity)[0])
                if old_validation['boot_id'] == boot:
                    raise ValueError('Offline recover requires a new Colab runtime')
            publish('resolved_matrix', resolved)
            publish('inference_bundle', bundle)
            publish('validation_records', records)
            publish('build_provenance', json.loads((REPO/'stage7_build_provenance.json').read_text('utf-8')))
            publish('calibration', {'executed': False, 'bundle_sha256': digest(bundle),
                'positive_threshold': bundle['positive_threshold'], 'grid_water_threshold': bundle['grid_water_threshold'],
                'probe_sha256': digest(bundle['probe']), 'decoder_best_epoch': spec['best_epoch'],
                'source': 'reviewed_stage5_and_stage6_no_recalibration'})
            import platform
            import psutil
            cpu_info = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                             if line.startswith('model name')), platform.processor())
            hardware = {'gpu': torch.cuda.get_device_name(), 'capability': list(torch.cuda.get_device_capability()),
                'total_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
                'torch': torch.__version__, 'cuda': torch.version.cuda,
                'cudnn': torch.backends.cudnn.version(), 'cpu_threads': torch.get_num_threads(),
                'cpu_model': cpu_info, 'cpu_count': os.cpu_count(), 'ram_total_bytes': psutil.virtual_memory().total,
                'platform': platform.platform()}
            conditions = {'hardware': hardware, 'profile': matrix['profiling'], 'profile_ids': ids[:1]}
            publish('environment', {'inventory': inventory(root, REPO), 'boot_id': boot, 'hardware': hardware})
            load_times = {}
            torch.cuda.synchronize(); start = time.perf_counter()
            predictor, sam_info = load_frozen_sam(env['models']['sam2'], root, offline=True,
                expected_sha256=bundle['base_checkpoint_sha256']['sam'])
            torch.cuda.synchronize(); load_times['sam_seconds'] = time.perf_counter()-start
            before = fingerprints(predictor.model)
            base_decoder = {k: v.detach().cpu().clone() for k, v in predictor.model.sam_mask_decoder.state_dict().items()}
            tuned_decoder = deserialize(sixth['payloads']['decoder_export'])
            torch.cuda.synchronize(); start = time.perf_counter()
            dino, dino_info = load_frozen_dino(env['models']['dinov3'], root, offline=True,
                expected_sha256=bundle['base_checkpoint_sha256']['dino'])
            torch.cuda.synchronize(); load_times['dino_seconds'] = time.perf_counter()-start
            publish('model_loading', {'seconds': load_times, 'scope': 'offline_Drive_weights_load_plus_state_hash_audit',
                'sam': sam_info, 'dino': dino_info, 'excluded_from_pipeline_latency': True})
            engine = ValidationEngine(root, regime, inp, bundle, prior_store, prior, sixth, predictor, dino,
                json.loads((REPO/'configs/baseline.yaml').read_text('utf-8'))['prompts'])
            profile_feature = engine.feature(0, records[0])
            results = []
            for planned in resolved['rows']:
                row_id, config = planned['id'], planned['config']
                predictor.model.sam_mask_decoder.load_state_dict(
                    tuned_decoder if config['decoder'] == 'stage6_best' else base_decoder, strict=True)
                predictor.model.eval().requires_grad_(False)
                intended_decoder_sha256 = state_fingerprint(predictor.model.sam_mask_decoder)
                start = time.perf_counter()
                cached_result, predictions, metrics = None, [], []
                try:
                    try:
                        cached_result = json.loads(store.read('result_'+row_id, identity)[0])
                    except FileNotFoundError:
                        cached_result = None
                    if mode == 'recover' and cached_result is None:
                        raise ValueError('Recover cannot fill missing result')
                    if cached_result and cached_result['status'] == 'failed':
                        results.append(cached_result)
                        continue  # Failure retained, never silently erased by resume.
                    metrics, prompt_hashes, diagnostics = [], {}, []
                    predictions = []
                    for i, chip in enumerate(records):
                        name = f'prediction_{row_id}_{i:03}'
                        try:
                            payload, marker = store.read(name, identity)
                            arrays, meta = unpack(payload)
                        except FileNotFoundError:
                            if mode == 'recover':
                                raise ValueError('Recover found missing/corrupt prediction')
                            reused = engine.reuse(row_id, config, chip, i, prior_stages['stage3'])
                            if reused is None:
                                feat = profile_feature if i == 0 else engine.feature(i, chip)
                                arrays, meta = engine.run_pipeline(config, chip, feat)
                            else:
                                arrays, meta = reused
                            meta.update({'id': chip['id'], 'event': chip['event'], 'role': 'val',
                                'regime': regime, 'image_sha256': chip['image_sha256'],
                                'config_sha256': digest(config), 'probe_sha256': digest(bundle['probe'])})
                            prompts = meta.get('prompts', {})
                            meta['prompt_sha256'] = digest(prompts)
                            if 'counts' not in meta:
                                meta['counts'] = prompt_diagnostics(config, meta)
                            meta['metrics'] = engine.metric(chip, arrays['mask'])
                            payload = pack(arrays, meta)
                            store.publish(name, payload, identity)
                            payload, marker = store.read(name, identity)
                            arrays, meta = unpack(payload)
                        if (meta['id'] != chip['id'] or meta['regime'] != regime or meta['role'] != 'val' or
                                meta['config_sha256'] != digest(config) or meta['probe_sha256'] != digest(bundle['probe'])):
                            raise ValueError('Stage-7 prediction identity mismatch')
                        recomputed = engine.metric(chip, arrays['mask'])
                        if recomputed['confusion'] != meta['metrics']['confusion']:
                            raise ValueError('Saved validation confusion differs from persisted mask')
                        metrics.append(recomputed)
                        prompt_hashes[chip['id']] = meta['prompt_sha256']
                        if row_id in ('a3_frozen', 'a3_tuned') and meta['prompts'] != sixth['fixed'][chip['id']]:
                            raise ValueError('A3 prompts differ from fixed stage6 prompts')
                        diagnostics.append({'id': chip['id'], 'counts': meta.get('counts'),
                            'prompts': meta.get('prompts'), 'inference': meta.get('inference'),
                            'grid': meta.get('grid'), 'origin': meta.get('origin')})
                        predictions.append({'name': name, 'generation': marker['generation'], 'sha256': marker['sha256']})
                        if i == 0:
                            first_mask = arrays['mask'].copy()
                        if i % 20 == 0:
                            print(regime, row_id, i+1, '/', len(records), flush=True)
                    if cached_result:
                        if mode != 'recover' and cached_result['efficiency']['conditions'] != conditions:
                            raise ValueError('Hardware/profile changed; do not mix timings on resume')
                        efficiency = cached_result['efficiency']
                        if mode == 'recover':
                            repeated, _ = engine.run_pipeline(config, records[0], None)
                            if not np.array_equal(repeated['mask'], first_mask):
                                raise ValueError('Offline first-val reconstruction differs')
                    else:
                        efficiency = engine.profile(config, records[0], profile_feature,
                                                    matrix['profiling'], conditions, first_mask)
                    result = dict(planned, status='ok', validation_ids=ids, metrics=aggregate(metrics),
                        chip_metrics=metrics, prompt_hashes=prompt_hashes, diagnostics=diagnostics,
                        predictions=predictions, efficiency=efficiency)
                    if cached_result and digest(result) != digest(cached_result):
                        raise ValueError('Reconstructed result differs from immutable prior result')
                except Exception as error:
                    if mode == 'recover' or cached_result:
                        raise  # Broken persisted evidence is not a new scientific failure row.
                    result = dict(planned, status='failed', validation_ids=ids,
                        failure={'type': type(error).__name__, 'message': str(error),
                                 'at': utc_now(), 'partial_predictions': predictions,
                                 'partial_validation_metrics': aggregate(metrics) if metrics else None,
                                 'elapsed_seconds': time.perf_counter()-start})
                    torch.cuda.empty_cache()
                if state_fingerprint(predictor.model.sam_mask_decoder) != intended_decoder_sha256:
                    raise ValueError('Decoder state changed during inference: '+row_id)
                publish('result_'+row_id, result)
                results.append(result)
                print(regime, row_id, result['status'],
                    result.get('metrics', {}).get('pooled', {}).get('values', {}), flush=True)
            # Verify that inference never modified any model, including the selected decoder.
            assert_frozen(dino); assert_frozen(predictor.model)
            if state_fingerprint(dino) != dino_info['state_sha256']:
                raise ValueError('DINO state changed')
            audit_changes(before, fingerprints(predictor.model))
            predictor.model.sam_mask_decoder.load_state_dict(base_decoder, strict=True)
            if fingerprints(predictor.model) != before:
                raise ValueError('SAM base state changed outside explicit checkpoint load')
            selection, lock = seal(matrix, resolved, results, lineage)
            publish('validation_tables', validation_tables(resolved, results))
            if mode == 'recover':
                if (digest(selection) != digest(json.loads(store.read('selection', identity)[0])) or
                        digest(lock) != digest(json.loads(store.read('evaluation_lock', identity)[0]))):
                    raise ValueError('Recomputed selection/lock differs')
            else:
                publish('selection', selection); publish('evaluation_lock', lock)
            publish('recovery_validation' if mode == 'recover' else 'validation', {
                'stage': 'phase1_stage7', 'regime': regime, 'identity': identity, 'boot_id': boot,
                'mode': mode, 'status': 'computed_pending_external_review', 'reserved_access': [],
                'access_log': engine.access_log, 'training_executed': False, 'calibration_executed': False,
                'rows': len(results), 'failed': [r['id'] for r in results if r['status'] == 'failed'],
                'selection_sha256': digest(selection), 'evaluation_lock_sha256': digest(lock),
                'models_unchanged_after_explicit_decoder_restoration': True,
                'recovery': 'first_val_all_successful_rows_uncached_bitwise' if mode == 'recover' else None,
                'closure': 'pending_user_outputs_and_external_review_no_stage8'})
            selections[regime], locks[regime] = selection, lock
            del engine, predictor, dino, base_decoder, tuned_decoder
            gc.collect(); torch.cuda.empty_cache()
        except Exception as error:
            publish('failure', {'at': utc_now(), 'mode': mode, 'type': type(error).__name__,
                'message': str(error), 'reserved_access': [], 'closure': 'not_closed'})
            export_evidence(root, store, identity)
            raise
        export_evidence(root, store, identity)
    pair = verify_pair(selections, locks)
    pair['locks_sha256'] = {r: digest(v) for r, v in locks.items()}
    pair['selections_sha256'] = {r: digest(v) for r, v in selections.items()}
    pair_identity = {'stage': 'phase1_stage7', 'locks_sha256': pair['locks_sha256']}
    pair_store = ArtifactStore(root/'stage7')
    pair_store.publish('both_regimes', canonical_json(pair), pair_identity)
    for store, identity in stores.values():
        (store.root/'evidence/both_regimes.json').write_bytes(canonical_json(pair))
        export_evidence(root, store, identity)
    print(json.dumps(pair, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'resume', 'recover'))
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.mode)
