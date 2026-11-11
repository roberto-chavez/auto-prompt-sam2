"""Source-only Colab transport for stage 5. No data or weights."""
from pathlib import Path
import zipfile

from run_stage5 import REPO, source_paths
from auto_prompt_sam2.artifacts import sha256_file


def export(target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in source_paths():
            archive.write(path, path.relative_to(REPO).as_posix())
    return sha256_file(target)


if __name__ == '__main__':
    target = REPO / 'artifacts/auto-prompt-sam2-stage5.zip'
    print(target)
    print('sha256:', export(target))
