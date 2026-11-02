"""Local syntax, transport and small synthetic CPU tests. No training/models."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from auto_prompt_sam2.artifacts import digest, sha256_file, utc_now


def main():
    paths = [*sorted((REPO / 'src').rglob('*.py')), *sorted((REPO / 'scripts').glob('*.py')),
             *sorted((REPO / 'tests').glob('*.py'))]
    for path in paths:
        ast.parse(path.read_text('utf-8'))
    nbpath = REPO / 'notebooks/07_ablaciones_y_seleccion.ipynb'
    notebook = json.loads(nbpath.read_text('utf-8'))
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            ast.parse(''.join(cell['source']))
            assert cell['execution_count'] is None and cell['outputs'] == []
    report = {'stage': 'phase1_stage7', 'created_at': utc_now(),
        'scope': 'local_cpu_synthetic_no_models_no_training', 'models_executed': False,
        'training_executed': False, 'raw_data_read': False, 'reserved_access': [], 'syntax': 'passed',
        'notebook': {'cells': len(notebook['cells']), 'syntax': 'passed', 'execution': 'not_run'}, 'checks': []}
    from auto_prompt_sam2.experiments import resolve_matrix
    matrix = json.loads((REPO/'configs/ablations.yaml').read_text('utf-8'))
    specs = json.loads((REPO/'configs/stage7_dependencies.json').read_text('utf-8'))['stage6']
    preview = {'stage': 'phase1_stage7', 'status': 'cpu_metadata_preview_not_execution_or_lock', 'regimes': {}}
    for regime, spec in specs.items():
        bundle = json.loads((REPO/'docs/project/evidence/stage5/colab_run_20261029T194653Z'/
                             (regime+'_complete_bundle.json')).read_text('utf-8'))
        assert digest(bundle) == spec['identity']['bundle_sha256']
        prior = spec['identity']
        lineage = {'regime': regime, 'validation_ids': prior['ids']['val'],
            'manifest_sha256': prior['manifest_sha256'], 'preprocessing_sha256': prior['preprocessing_sha256'],
            'bundle_sha256': digest(bundle), 'base_checkpoint_sha256': prior['base_checkpoint_sha256'],
            'stage6_decoder_sha256': spec['artifacts']['decoder_export']['sha256'],
            'preview_only_runtime_source_identity_not_resolved': True}
        preview['regimes'][regime] = resolve_matrix(matrix, bundle, lineage)
    report['matrix_preview'] = {r: {'rows': len(v['rows']), 'contrasts': len(v['contrasts']),
        'budget': v['budget']} for r, v in preview['regimes'].items()}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               PYTHONHASHSEED='42', PYTHONIOENCODING='utf-8')
    for args in (['-m', 'pip', 'check'], ['-m', 'unittest', 'discover', '-s', 'tests', '-v']):
        result = subprocess.run([sys.executable, *args], cwd=REPO, env=env, text=True,
                                encoding='utf-8', capture_output=True)
        report['checks'].append({'command': ['python', *args], 'exit_code': result.returncode,
                                 'stdout': result.stdout, 'stderr': result.stderr})
        print(result.stdout, result.stderr, flush=True)
        if result.returncode:
            break
    from export_stage7_bundle import export
    from run_stage7 import source_paths
    target = REPO / 'artifacts/auto-prompt-sam2-stage7.zip'
    sha = export(target)
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        assert set(z.namelist()) == {p.relative_to(REPO).as_posix() for p in source_paths()} | {'stage7_build_provenance.json'}
        for path in source_paths():
            assert z.read(path.relative_to(REPO).as_posix()) == path.read_bytes()
    bundle = {'stage': 'phase1_stage7', 'created_at': utc_now(), 'status': 'passed_source_only_transport',
              'path': str(target), 'sha256': sha, 'size_bytes': target.stat().st_size,
              'files': [p.relative_to(REPO).as_posix() for p in source_paths()], 'models_or_data_included': False}
    report['source_sha256'] = {p.relative_to(REPO).as_posix(): sha256_file(p) for p in source_paths()}
    report['source_tree_sha256'] = digest(report['source_sha256'])
    report['status'] = 'passed_local_only' if all(c['exit_code'] == 0 for c in report['checks']) else 'failed'
    report['pending'] = ['User-run Colab GPU validation-only prepare and offline recover on different runtime',
        'External evidence audit and human notebook/output interpretation before stage closure']
    folder = REPO / 'docs/project/evidence/stage7'
    folder.mkdir(parents=True, exist_ok=True)
    for name, value in (('local_validation', report), ('bundle_validation', bundle), ('matrix_preview', preview)):
        (folder / (name+'.json')).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', 'utf-8')
    if report['status'] == 'failed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
