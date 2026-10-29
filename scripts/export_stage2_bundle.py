"""Transport code and audit metadata only; raster/weight bytes are excluded."""
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from auto_prompt_sam2.artifacts import sha256_file
from auto_prompt_sam2.runtime import source_files


def export(target):
    paths = set(source_files(REPO))
    paths.update(REPO / p for p in ("configs/data.yaml", "requirements-stage2.txt", "notebooks/02_datos_y_splits.ipynb"))
    paths.update((REPO / "docs/project/evidence/stage1/colab_run_20261028T221055Z").glob("*.json"))
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            archive.write(path, path.relative_to(REPO).as_posix())
    return sha256_file(target)


if __name__ == "__main__":
    target = REPO / "artifacts/auto-prompt-sam2-stage2.zip"
    print(target)
    print("sha256:", export(target))
