"""CPU read-only stage-7 evidence review. No notebook execution or checkpoints."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'src'))
from auto_prompt_sam2.artifacts import digest, sha256_file
from auto_prompt_sam2.experiments import (resolve_matrix, seal, verify_pair,
    require_validation, validation_tables)
from auto_prompt_sam2.metrics import aggregate, scores, chip_metrics, pixel_areas_m2
from auto_prompt_sam2.stage5 import unpack, supervised_labels


def review(path, data_root=None):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or archive.testzip() is not None:
            raise ValueError('Duplicate/corrupt ZIP entries')
        generations = {}
        identity = None
        for name in names:
            if not name.startswith('generations/') or not name.endswith('/COMPLETE.json'):
                continue
            marker = json.loads(archive.read(name))
            if digest(marker['identity']) != marker['identity_sha256']:
                raise ValueError('Broken generation identity')
            payload = archive.read(name.rsplit('/', 1)[0]+'/payload')
            if len(payload) != marker['size'] or hashlib.sha256(payload).hexdigest() != marker['sha256']:
                raise ValueError('Broken generation payload')
            key, gen = name.split('/')[1:3]
            if gen != marker['generation']:
                raise ValueError('Generation path mismatch')
            generations.setdefault(key, {})[gen] = (payload, marker)
            if identity is not None and identity != marker['identity']:
                raise ValueError('Mixed stage-7 identities')
            identity = marker['identity']
        if identity is None:
            raise ValueError('No verified completed generations')

        def read(name, gen=None):
            items = generations[name]
            return items[gen if gen else sorted(items)[-1]]

        def value(name):
            return json.loads(read(name)[0])

        for rel, expected in identity['source_sha256'].items():
            if hashlib.sha256(archive.read('source/'+rel)).hexdigest() != expected:
                raise ValueError('Source differs: '+rel)
        for name in names:
            if name.startswith('evidence/') and name.endswith('.json') and not name.endswith('/both_regimes.json'):
                key = Path(name).stem
                payload = archive.read(name)
                if not any(payload == v[0] for v in generations.get(key, {}).values()):
                    raise ValueError('Exported JSON lacks a matching generation: '+key)
        selection, lock = value('selection'), value('evaluation_lock')
        matrix = json.loads(archive.read('source/configs/ablations.yaml'))
        resolved = value('resolved_matrix')
        if resolve_matrix(matrix, value('inference_bundle'), selection['lineage']) != resolved:
            raise ValueError('Resolved matrix cannot be reproduced')
        records = value('validation_records')
        require_validation(selection['regime'], selection['validation_ids'], records)
        by_id = {r['id']: r for r in records}
        manifest = None
        if data_root:
            from auto_prompt_sam2.baseline import stage2_inputs
            inp = stage2_inputs(Path(data_root), json.loads(archive.read('source/configs/data.yaml')))[selection['regime']]
            manifest = inp['manifest']
            if (digest(manifest) != selection['lineage']['manifest_sha256'] or
                    digest(inp['params']) != selection['lineage']['preprocessing_sha256']):
                raise ValueError('Local stage-2 artifacts differ')
        rows = []
        audited_masks = 0
        for planned in resolved['rows']:
            row = value('result_'+planned['id'])
            if row['status'] == 'ok':
                metrics = []
                for i, prediction in enumerate(row['predictions']):
                    payload, marker = read(prediction['name'], prediction['generation'])
                    if marker['sha256'] != prediction['sha256']:
                        raise ValueError('Prediction reference differs')
                    arrays, meta = unpack(payload)
                    chip = by_id[meta['id']]
                    if (meta['id'] != selection['validation_ids'][i] or meta['role'] != 'val' or
                            meta['regime'] != selection['regime'] or
                            meta['image_sha256'] != chip['image_sha256'] or
                            meta['config_sha256'] != row['config_sha256'] or
                            meta['probe_sha256'] != row['config']['probe_sha256'] or
                            digest(meta.get('prompts', {})) != row['prompt_hashes'][chip['id']] or
                            arrays['mask'].shape != (512, 512) or arrays['mask'].dtype != np.bool_):
                        raise ValueError('Prediction scope/config/prompt/geometry differs')
                    metric = meta['metrics']
                    if scores(metric['confusion']) != metric['scores']:
                        raise ValueError('Per-chip scores differ from confusion')
                    if manifest:
                        labels = supervised_labels(Path(data_root), manifest, chip['id'], 'val')
                        recomputed = chip_metrics(labels, arrays['mask'], chip_id=chip['id'], event=chip['event'],
                            pixel_area_m2=pixel_areas_m2(chip['geometry']))
                        if (recomputed['confusion'] != metric['confusion'] or
                                not np.isclose(recomputed['fp_area_m2'], metric['fp_area_m2'], rtol=1e-6, atol=.01)):
                            raise ValueError('Mask/label audit differs')
                        audited_masks += 1
                    metrics.append(metric)
                if metrics != row['chip_metrics'] or aggregate(metrics) != row['metrics']:
                    raise ValueError('Aggregation differs from saved predictions')
                efficiency = row['efficiency']
                for cache in ('cached', 'uncached'):
                    p = efficiency[cache]
                    if (len(p['samples_ms']) != matrix['profiling']['repetitions'] or p['warmup'] != 5 or
                            float(np.median(p['samples_ms'])) != p['median_ms'] or p['batch_size'] != 1 or
                            p['precision'] != 'float32' or p['autocast'] or
                            p['peak_allocated_bytes'] < p['baseline_allocated_bytes'] or
                            len(p['stages_ms_samples']) != len(p['samples_ms']) or
                            len(p['counts']) != len(p['samples_ms']) or
                            p['mask_reconstruction'] != 'bitwise_equal_all_warmup_and_measured_repetitions'):
                        raise ValueError('Profile metadata/samples inconsistent')
            rows.append(row)
        reconstructed_selection, reconstructed_lock = seal(matrix, resolved, rows, selection['lineage'])
        if reconstructed_selection != selection or reconstructed_lock != lock:
            raise ValueError('Selection/lock cannot be reproduced')
        if validation_tables(resolved, rows) != value('validation_tables'):
            raise ValueError('Contrasts/curves differ')
        validation = value('validation')
        if (validation['reserved_access'] or validation['training_executed'] or validation['calibration_executed'] or
                validation['selection_sha256'] != digest(selection) or validation['evaluation_lock_sha256'] != digest(lock)):
            raise ValueError('Invalid validation report')
        recovery = value('recovery_validation') if 'recovery_validation' in generations else None
        if recovery and (recovery['boot_id'] == validation['boot_id'] or recovery['reserved_access'] or
                         recovery['recovery'] != 'first_val_all_successful_rows_uncached_bitwise'):
            raise ValueError('Recovery evidence inconsistent')
        return {'regime': selection['regime'], 'selection': selection, 'lock': lock,
            'review': {'status': 'portable_consistency_passed_not_stage_closure',
                'zip_sha256': sha256_file(path), 'generation_count': sum(map(len, generations.values())),
                'rows': len(rows), 'selected_id': selection['selected_id'],
                'failures': [r['id'] for r in rows if r['status'] == 'failed'],
                'validation_masks_independently_audited': audited_masks,
                'offline_recovery_present': recovery is not None, 'models_executed': False,
                'reserved_access': [], 'pending': ['Human review of executed notebook, logs, hardware, failures and interpretation']+
                    ([] if recovery else ['User-run offline recover on a different Colab runtime'])+
                    ([] if data_root else ['Independent confusion audit using validation labels only'])}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('zips', nargs=2, type=Path)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    reviewed = [review(path, args.data_root) for path in args.zips]
    selections = {r['regime']: r['selection'] for r in reviewed}
    locks = {r['regime']: r['lock'] for r in reviewed}
    result = {'pair': verify_pair(selections, locks),
              'regimes': {r['regime']: r['review'] for r in reviewed}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n', 'utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False))
