"""Source-only Colab transport for stage 7, with exact Git provenance."""
from pathlib import Path
import json
import subprocess
import zipfile
from run_stage7 import REPO, source_paths
from auto_prompt_sam2.artifacts import sha256_file


def export(target):
    target = Path(target); target.parent.mkdir(parents=True, exist_ok=True)
    git_errors = []
    def git(*args):
        try:
            result = subprocess.run(['git', *args], cwd=REPO, capture_output=True, text=True, encoding='utf-8', errors='replace')
        except OSError as error:
            git_errors.append({'command': ['git', *args], 'error': str(error)})
            return None
        if result.returncode:
            git_errors.append({'command': ['git', *args], 'exit_code': result.returncode,
                               'stderr': result.stderr})
        return result.stdout if result.returncode == 0 else None
    provenance = {'commit': git('rev-parse', 'HEAD'), 'status': git('status', '--porcelain'),
        'tracked_patch': git('diff', 'HEAD', '--no-ext-diff'),
        'note': 'Source snapshot includes untracked stage-7 files; no data/weights.'}
    provenance['git_errors'] = git_errors
    provenance['git_provenance_complete'] = not git_errors
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in source_paths():
            archive.write(path, path.relative_to(REPO).as_posix())
        archive.writestr('stage7_build_provenance.json', json.dumps(provenance, indent=2, ensure_ascii=False))
    return sha256_file(target)


if __name__ == '__main__':
    target = REPO/'artifacts/auto-prompt-sam2-stage7.zip'
    print(target)
    print('SHA256:', export(target))
