"""Scientific selection/lineage tests: small synthetic CPU data, no models."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO/'src'), str(REPO/'scripts')]
from auto_prompt_sam2.artifacts import ArtifactStore, canonical_json, digest
from auto_prompt_sam2.experiments import (resolve_matrix, config_diff, choose_automatic,
    prompt_cost, seal, verify_pair, require_validation, validate_matrix)
from auto_prompt_sam2.stage7 import checked_read, stage6_inputs, prompt_diagnostics


def setup_protocol(regime='official'):
    matrix = json.loads((REPO/'configs/ablations.yaml').read_text('utf-8'))
    bundle = {'regime': regime, 'probe': {'test': 'no_model'},
        'prompts': json.loads((REPO/'configs/prompts.yaml').read_text('utf-8')),
        'positive_threshold': .5, 'grid_water_threshold': .7, 'ndwi': {'included': False}}
    lineage = {'regime': regime, 'validation_ids': ['a', 'b'], 'manifest_sha256': 'manifest',
               'preprocessing_sha256': 'preprocessing', 'decoder_sha256': 'decoder'}
    resolved = resolve_matrix(matrix, bundle, lineage)
    results = [dict(r, status='ok', validation_ids=['a', 'b'],
        metrics={'chips': 2, 'pooled': {'values': {'water_iou': .3}}},
        prompt_hashes={'a': 'one', 'b': 'two'}, efficiency={'conditions': {'gpu': 'same'},
        'uncached': {'median_ms': 20.}}) for r in resolved['rows']]
    return matrix, bundle, lineage, resolved, results


class Stage7Tests(unittest.TestCase):
    def test_all_contrasts_one_factor_and_separate_sweeps(self):
        matrix, _, _, resolved, _ = setup_protocol()
        self.assertEqual(len(resolved['rows']), 13)
        self.assertEqual(len(resolved['contrasts']), 10)
        for contrast in resolved['contrasts']:
            self.assertEqual(list(contrast['diff']), [contrast['factor']])
        for row in resolved['rows']:
            if row['id'].startswith('a2_positive_'):
                self.assertEqual(row['config']['policy']['negative_per_component'], 3)
            if row['id'].startswith('a2_negative_'):
                self.assertEqual(row['config']['policy']['positive_per_component'], 3)
        self.assertFalse(matrix['calibration']['execute'])

    def test_protocol_rejects_retuning_and_scope_changes(self):
        for field, value in [('roles', ['test']), ('fixed_positive', 5), ('negative_sweep', [0, 3])]:
            matrix, *_ = setup_protocol(); matrix[field] = value
            with self.assertRaises(ValueError):
                validate_matrix(matrix)

    def test_diff_detects_second_nested_control_change(self):
        _, _, _, resolved, _ = setup_protocol()
        a = resolved['rows'][0]['config']; b = copy.deepcopy(a)
        b['policy']['positive_per_component'] = 5; b['positive_threshold'] = .7
        self.assertEqual(set(config_diff(a, b)), {'policy.positive_per_component', 'positive_threshold'})

    def test_selection_iou_then_latency_then_id_not_assisted(self):
        *_, rows = setup_protocol()
        for r in rows:
            r['metrics']['pooled']['values']['water_iou'] = .99 if not r['automatic'] else .1
        rows[0]['metrics']['pooled']['values']['water_iou'] = .5
        rows[-1]['metrics']['pooled']['values']['water_iou'] = .5
        rows[-1]['efficiency']['uncached']['median_ms'] = 10.
        self.assertEqual(choose_automatic(rows, ['a', 'b'])['id'], 'a3_tuned')
        rows[-1]['efficiency']['uncached']['median_ms'] = 20.
        self.assertEqual(choose_automatic(rows, ['a', 'b'])['id'], 'a1_points')
        rows[0]['metrics']['pooled']['values']['water_iou'] += 1e-12
        rows[-1]['efficiency']['uncached']['median_ms'] = 1.
        self.assertEqual(choose_automatic(rows, ['a', 'b'])['id'], 'a1_points')

    def test_selection_rejects_partial_or_mixed_hardware(self):
        *_, rows = setup_protocol()
        rows[0]['validation_ids'] = ['test']
        with self.assertRaises(ValueError): choose_automatic(rows, ['a', 'b'])
        *_, rows = setup_protocol()
        rows[0]['efficiency']['conditions']['gpu'] = 'different'
        with self.assertRaises(ValueError): choose_automatic(rows, ['a', 'b'])

    def test_negative_prompt_cost_retained_and_budget_mismatch_rejected(self):
        *_, rows = setup_protocol()
        a, b = rows[1], rows[0]
        a['metrics']['pooled']['values']['water_iou'] = .1
        b['metrics']['pooled']['values']['water_iou'] = .4
        self.assertAlmostEqual(prompt_cost(a, b)['percentage_points'], -30.)
        a['config']['policy']['positive_per_component'] = 1
        with self.assertRaises(ValueError): prompt_cost(a, b)

    def test_failure_and_nonwinner_rows_survive_lock(self):
        matrix, _, lineage, resolved, results = setup_protocol()
        results[2].update(status='failed', failure={'type': 'OOM', 'message': 'preserved'})
        selection, lock = seal(matrix, resolved, results, lineage)
        self.assertEqual(len(lock['final_rows']), 13)
        self.assertEqual(lock['final_rows'][2]['validation_status'], 'failed')
        self.assertEqual(lock['selection_sha256'], digest(selection))
        self.assertFalse(lock['stage8_authorized'])
        with self.assertRaises(ValueError): seal(matrix, resolved, results[:-1], lineage)
        results[2].pop('failure')
        with self.assertRaises(ValueError): seal(matrix, resolved, results, lineage)

    def test_a3_prompt_hash_invariant(self):
        matrix, _, lineage, resolved, results = setup_protocol()
        results[-1]['prompt_hashes']['a'] = 'modified'
        with self.assertRaises(ValueError): seal(matrix, resolved, results, lineage)

    def test_assisted_eligibility_cannot_be_changed(self):
        matrix, _, lineage, resolved, results = setup_protocol()
        results[1]['automatic'] = True
        with self.assertRaises(ValueError): seal(matrix, resolved, results, lineage)

    def test_both_regimes_gate_and_hash_tamper(self):
        selections, locks = {}, {}
        for regime in ('official', 'leave_paraguay_out'):
            matrix, _, lineage, resolved, results = setup_protocol(regime)
            selections[regime], locks[regime] = seal(matrix, resolved, results, lineage)
        self.assertFalse(verify_pair(selections, locks)['stage8_authorized'])
        with self.assertRaises(ValueError): verify_pair({'official': selections['official']}, locks)
        selections['official']['selected_id'] = 'tampered'
        with self.assertRaises(ValueError): verify_pair(selections, locks)

    def test_validation_membership_tripwire(self):
        require_validation('official', ['a'], [{'id': 'a', 'role': 'val', 'event': 'Paraguay'}])
        for regime, role, event in [('official','test','Ghana'), ('official','val','Bolivia'),
                                   ('leave_paraguay_out','val','Paraguay')]:
            with self.assertRaises(ValueError):
                require_validation(regime, ['a'], [{'id': 'a', 'role': role, 'event': event}])

    def test_checked_cache_detects_payload_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ArtifactStore(Path(tmp)); identity = {'regime': 'official'}
            generation = store.publish('prediction', b'valid', identity)
            item = store.read('prediction', identity)[1]
            self.assertEqual(checked_read(store, 'prediction', item), b'valid')
            (store.root/'prediction'/generation/'payload').write_bytes(b'corrupt')
            with self.assertRaises(FileNotFoundError): checked_read(store, 'prediction', item)

    def test_stage6_pins_match_reviewed_payloads_without_loading_models(self):
        specs = json.loads((REPO/'configs/stage7_dependencies.json').read_text('utf-8'))['stage6']
        for regime, spec in specs.items():
            p = REPO/'docs/project/evidence/stage6/colab_review_20261101'/regime
            self.assertEqual(spec['identity'], json.loads((p/'config.json').read_text('utf-8')))
            import hashlib
            self.assertEqual(spec['artifacts']['validation']['sha256'], hashlib.sha256((p/'validation.json').read_bytes()).hexdigest())
            inp = {'manifest': {}, 'params': {}}
            with self.assertRaises(ValueError): stage6_inputs(Path('.'), spec, inp, {'regime': regime}, REPO)

    def test_cached_counts_use_actual_points_and_shortfalls(self):
        meta = {'prompts': {'components': [{'point_labels': [1, 0], 'positive_shortfall': 2, 'negative_shortfall': 2}]}}
        c = prompt_diagnostics({'source': 'dino_points'}, meta)
        self.assertEqual((c['positive_points'], c['negative_points'], c['mask_decoder']), (1, 1, 1))
        self.assertEqual(c['positive_shortfall'], 2)

    def test_pc_guard_before_any_model_or_data(self):
        from run_stage7 import main
        with self.assertRaises(RuntimeError): main(Path('nonexistent'), 'prepare')

    def test_portable_review_roundtrip_and_corruption(self):
        """Exercise exported masks/metrics/selection end to end without any model."""
        import hashlib
        import zipfile
        from auto_prompt_sam2.metrics import chip_metrics, aggregate
        from auto_prompt_sam2.stage5 import pack
        from auto_prompt_sam2.experiments import validation_tables
        from review_stage7_evidence import review
        matrix, bundle, lineage, resolved, rows = setup_protocol()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = canonical_json(matrix)
            identity = {'stage': 'phase1_stage7', 'regime': 'official',
                'source_sha256': {'configs/ablations.yaml': hashlib.sha256(source).hexdigest()}}
            store = ArtifactStore(root/'store')
            metrics, predictions = [], []
            records = [{'id': key, 'role': 'val', 'event': 'Ghana', 'image_sha256': 'image'} for key in ('a', 'b')]
            for i, record in enumerate(records):
                y = np.zeros((512, 512), np.int8); y[0, 0] = 1
                mask = y.astype(bool)
                metric = chip_metrics(y, mask, chip_id=record['id'], event='Ghana', pixel_area_m2=100.)
                metrics.append(metric)
                prompts = {'components': []}
                name = f'prediction_a1_points_{i:03}'
                metadata = dict(record, regime='official', config_sha256=rows[0]['config_sha256'],
                    probe_sha256=rows[0]['config']['probe_sha256'], prompts=prompts, metrics=metric)
                gen = store.publish(name, pack({'mask': mask}, metadata), identity)
                marker = store.read(name, identity, gen)[1]
                predictions.append({'name': name, 'generation': gen, 'sha256': marker['sha256']})
            rows[0].update(chip_metrics=metrics, metrics=aggregate(metrics), predictions=predictions,
                prompt_hashes={key: digest(prompts) for key in ('a', 'b')})
            sample = {'samples_ms': [10.]*20, 'median_ms': 10., 'warmup': 5, 'batch_size': 1,
                'precision': 'float32', 'autocast': False, 'peak_allocated_bytes': 100,
                'baseline_allocated_bytes': 50, 'stages_ms_samples': [{'sam': 5.}]*20,
                'counts': [{}]*20, 'mask_reconstruction': 'bitwise_equal_all_warmup_and_measured_repetitions'}
            rows[0]['efficiency'].update(cached=copy.deepcopy(sample), uncached=copy.deepcopy(sample))
            for row in rows[1:]:
                row.update(status='failed', failure={'type': 'synthetic', 'message': 'intentional fixture'})
            selection, lock = seal(matrix, resolved, rows, lineage)
            objects = {'selection': selection, 'evaluation_lock': lock, 'resolved_matrix': resolved,
                'inference_bundle': bundle, 'validation_records': records,
                'validation_tables': validation_tables(resolved, rows),
                'validation': {'reserved_access': [], 'training_executed': False,
                    'calibration_executed': False, 'selection_sha256': digest(selection),
                    'evaluation_lock_sha256': digest(lock), 'boot_id': 'first'}}
            objects.update({'result_'+r['id']: r for r in rows})
            for name, obj in objects.items(): store.publish(name, canonical_json(obj), identity)
            target = root/'evidence.zip'
            def export(corrupt=False):
                with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
                    archive.writestr('source/configs/ablations.yaml', source if not corrupt else b'bad source')
                    for p in store.root.glob('*/*/*'):
                        archive.write(p, 'generations/'+p.relative_to(store.root).as_posix())
            export()
            audited = review(target)
            self.assertEqual(audited['review']['selected_id'], 'a1_points')
            self.assertEqual(len(audited['review']['failures']), 12)
            self.assertFalse(audited['lock']['stage8_authorized'])
            export(corrupt=True)
            with self.assertRaises(ValueError): review(target)


if __name__ == '__main__': unittest.main()
