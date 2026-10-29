"""Read-only CPU audit of stage-3 user evidence. Never execute archived code/models."""
import io
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from auto_prompt_sam2.artifacts import digest, sha256_file, utc_now
from auto_prompt_sam2.baseline import decode_prediction, summaries, development_plan
from auto_prompt_sam2.data import read_labels
from auto_prompt_sam2.metrics import chip_metrics, scores
from auto_prompt_sam2.reference_prompts import simulate_points


def audit(zip_path, notebook_path, raw_root, output):
    output.mkdir(parents=True, exist_ok=True)
    review = {'reviewed_at': utc_now(), 'scope': 'CPU artifact and label audit; no models',
              'models_executed': False, 'reserved_access': [], 'checks': {}, 'regimes': {}}
    for p in (zip_path, notebook_path):
        dest = output / p.name
        if not dest.exists():
            shutil.copyfile(p, dest)
        assert sha256_file(dest) == sha256_file(p)
    review['originals'] = {p.name: {'sha256': sha256_file(p), 'size': p.stat().st_size}
                           for p in (zip_path, notebook_path)}
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        assert len(names) == len(set(names))
        assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in names)
        assert z.testzip() is None
        def report(name):
            n = next(n for n in names if n.endswith('/' + name + '.json'))
            return json.loads(z.read(n))
        prepare, recover, config = report('validation'), report('recovery_validation'), report('config')
        identity = prepare['identity']
        assert identity == recover['identity'] == config['identity']
        markers = {}
        for n in names:
            if n.endswith('/COMPLETE.json'):
                m = json.loads(z.read(n)); base = n.rsplit('/', 1)[0]; payload = z.read(base + '/payload')
                assert m['size'] == len(payload)
                import hashlib
                assert m['sha256'] == hashlib.sha256(payload).hexdigest()
                assert m['identity_sha256'] == digest(m['identity'])
                assert m['generation'] == base.rsplit('/', 1)[-1]
                markers[base] = m
        review['checks']['intact_artifact_generations'] = len(markers)
        source_path = next(n for n in names if '/source_snapshot/' in n and n.endswith('/payload'))
        with zipfile.ZipFile(io.BytesIO(z.read(source_path))) as source:
            hashes = config['source']['source_sha256']
            assert set(hashes) == set(source.namelist())
            for n in source.namelist():
                assert hashlib.sha256(source.read(n)).hexdigest() == hashes[n]
                assert source.read(n) == (REPO / n).read_bytes(), n
            assert digest(hashes) == identity['source_tree_sha256']
        review['checks']['source_files_match_delivered_workspace'] = len(hashes)
        for name in ('baseline', 'data', 'environment'):
            file = {'baseline': 'baseline.yaml', 'data': 'data.yaml', 'environment': 'environment.yaml'}[name]
            assert config[name] == json.loads((REPO / 'configs' / file).read_text('utf-8'))
        for r in (prepare, recover):
            assert r['reserved_access'] == [] and r['final_evaluations'] == [] and not r['training_executed']
            assert r['frozen_unchanged'] and r['model']['trainable_parameters'] == 0
            assert r['model']['all_modules_eval'] and r['model']['all_parameters_frozen']
            assert r['model']['state_sha256'] == r['state_sha256_after']
            assert r['tests']['exit_code'] == 0 and 'Ran 40 tests' in r['tests']['stderr']
            assert r['environment']['cuda_available'] and r['environment']['gpus'][0]['name'] == 'Tesla T4'
            assert 'No broken requirements found.' in r['pip_check']
        assert prepare['model'] == recover['model']
        assert prepare['runtime_boot_id'] != recover['runtime_boot_id']
        assert recover['different_runtime_boot'] and recover['reloaded_from_persistent_checkpoint']
        review['checks']['different_runtime_boots'] = [prepare['runtime_boot_id'], recover['runtime_boot_id']]
        review['model'] = prepare['model']
        stage2 = json.loads((REPO / 'docs/project/evidence/stage2/colab_run_20261029T041823Z/validation.json').read_text('utf-8'))
        for regime in ('official', 'leave_paraguay_out'):
            manifest = json.loads((REPO / 'docs/project/evidence/stage2' / (regime + '_manifest.json')).read_text('utf-8'))
            rows = {r['id']: r for r in development_plan(manifest)}
            a, b = prepare['regimes'][regime], recover['regimes'][regime]
            assert a['inputs']['manifest_sha256'] == digest(manifest)
            for k in ('manifest_sha256', 'preprocessing_sha256', 'manifest_generation', 'parameter_generation'):
                assert a['inputs'][k] == stage2['regimes'][regime][k]
            assert a['prediction_index'] == b['prediction_index']
            assert a['summary'] == b['summary'] and b['all_saved_predictions_recovered']
            assert a['planned_ids'] == [{'id': r['id'], 'role': r['role']} for r in rows.values()]
            assert {e['id'] for e in a['prediction_index']} == set(rows)
            records, components, max_area_error = [], 0, 0.
            for entry in a['prediction_index']:
                row = rows[entry['id']]
                assert row['role'] in ('train', 'val') and row['event'] != 'Bolivia'
                if regime == 'leave_paraguay_out': assert row['event'] != 'Paraguay'
                base = f"regimes/{regime}/runs/{prepare['run_id']}/assisted/{row['id']}/{entry['generation']}"
                assert markers[base]['sha256'] == entry['sha256']
                assert markers[base]['identity'] == entry['identity']
                mask, meta = decode_prediction(z.read(base + '/payload'))
                assert meta['image_sha256'] == row['image_sha256'] and meta['label_sha256'] == row['label_sha256']
                labels_path = raw_root / row['label_path']
                assert sha256_file(labels_path) == row['label_sha256']
                labels, _ = read_labels(labels_path)
                prompts = meta['prompts']; policy = config['baseline']['prompts']
                assert prompts['policy'] == policy and prompts['seed'] == 42
                expected_seed = int(digest({'seed': 42, 'regime': regime, 'chip_id': row['id']})[:16], 16)
                assert prompts['chip_seed'] == expected_seed
                assert prompts['water_pixels'] == int((labels == 1).sum())
                assert prompts['valid_pixels'] == int((labels != -1).sum())
                cs = prompts['components']; decisions = meta['inference']['components']
                assert len(cs) == len(decisions) == meta['inference']['predict_calls']
                assert sum(c['water_pixels'] for c in cs) == prompts['water_pixels']
                for c, decision in zip(cs, decisions):
                    xy = np.asarray(c['point_coords']); signs = c['point_labels']
                    assert ((xy >= 0) & (xy <= 511)).all()
                    assert [int(labels[y, x]) for x, y in xy] == signs
                    assert signs.count(1) + c['positive_shortfall'] == 1
                    assert signs.count(0) + c['negative_shortfall'] == 1
                    if len(xy) == 2: assert np.linalg.norm(xy[0] - xy[1]) >= 8
                    assert decision['selected_index'] == int(np.argmax(decision['predicted_iou']))
                if not cs: assert not mask.any()
                # Recompute confusion/ratios from native masks and trusted labels.
                # Physical area is checked against per-row PROJ transformation.
                from auto_prompt_sam2.metrics import pixel_areas_m2
                computed = chip_metrics(labels, mask, chip_id=row['id'], event=row['event'],
                    pixel_area_m2=pixel_areas_m2(row['geometry']))
                saved = meta['metrics']
                for k in computed:
                    if k == 'fp_area_m2':
                        max_area_error = max(max_area_error, abs(computed[k] - saved[k]))
                        assert np.isclose(computed[k], saved[k], atol=1e-5, rtol=1e-10)
                    else: assert computed[k] == saved[k], (regime, row['id'], k)
                if row['id'] == a['repeat_chip']:
                    assert prompts == simulate_points(labels, regime=regime, chip_id=row['id'], seed=42, policy=policy)
                records.append(meta); components += len(cs)
            assert summaries(records) == a['summary']
            for r in (a, b):
                assert r['repeat']['mask_disagreement_pixels'] == 0 and r['repeat']['scores_within_tolerance']
                assert r['points_identical'] and r['repeat']['tolerance'] == config['baseline']['repeat_tolerance']
                assert all(x['id'] in rows and x['role'] == rows[x['id']]['role'] for x in r['access_log'])
            review['regimes'][regime] = {'predictions_audited': len(records), 'components': components,
                'counts': a['counts'], 'val': a['summary']['val'], 'repeat_chip': a['repeat_chip'],
                'max_area_recompute_error_m2': max_area_error, 'stage2_inputs_identical': True,
                'all_points_have_correct_valid_label': True, 'confusions_recomputed_from_saved_masks': True}
        # Convenience reports must equal the corresponding immutable generations.
        for label, original in (('validation', prepare), ('recovery_validation', recover)):
            candidates = [n for n in names if '/'+label+'/' in n and n.endswith('/payload')]
            assert any(json.loads(z.read(n)) == original for n in candidates)
            (output / (label + '.json')).write_bytes(z.read(next(n for n in names if n.endswith('/'+label+'.json'))))
    notebook = json.loads(notebook_path.read_text('utf-8'))
    cells = []
    for i, c in enumerate(notebook['cells']):
        if c['cell_type'] != 'code': continue
        assert not any(o.get('output_type') == 'error' for o in c.get('outputs', []))
        text = ''.join(''.join(o.get('text', [])) for o in c.get('outputs', []))
        cells.append({'cell': i, 'execution_count': c.get('execution_count'), 'output_chars': len(text),
                      'contains_run_id': prepare['run_id'] in text})
    assert cells[1]['contains_run_id'] and cells[2]['contains_run_id']
    review['notebook'] = {'code_cells': cells, 'python_errors': [],
                          'prepare_and_recover_outputs_present': True}
    review['checks']['notebook_sources_review'] = 'Human review required alongside this structural audit'
    review['status'] = 'technical_audit_passed'
    review['limitations'] = ['No SAM re-execution on PC; freeze/repeat rely on integrity-checked run records and inspected source, not independent GPU rerun.',
        'Candidate masks are not retained: selection verified against saved scores and inspected adapter; union implementation covered by synthetic tests.',
        'Per-chip label/metrics and point validity audited for development rows only; no reserved labels opened.']
    (output / 'review.json').write_text(json.dumps(review, indent=2, ensure_ascii=False, allow_nan=False)+'\n', 'utf-8')
    print(json.dumps({'status': review['status'], 'checks': review['checks'],
                      'predictions': {r:v['predictions_audited'] for r,v in review['regimes'].items()}}, indent=2))


if __name__ == '__main__':
    audit(*(Path(p) for p in sys.argv[1:]))
