"""E2 source-only ZIP transport (no data, weights, checkpoints or notebooks).

Notebooks are deliberately NOT inside the ZIP: a notebook that carries the ZIP's SHA-256
cannot also be an input to that SHA-256 (circular dependency). Build order is
code -> this ZIP -> SHA-256 -> notebooks (scripts/build_e2_notebooks.py).
"""
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from auto_prompt_sam2.artifacts import sha256_file
from auto_prompt_sam2.runtime import source_files

EXTRA = ("configs/data.yaml", "configs/baseline.yaml", "configs/profiling.yaml", "configs/e2_intervention.yaml",
         "requirements-stage2.txt")


def paths():
    found = {p for p in source_files(REPO) if p.suffix != ".ipynb"}
    found.update(REPO / p for p in EXTRA)
    missing = [p for p in found if not p.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    return sorted(found)


def export(target):
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths():
            info = zipfile.ZipInfo(path.relative_to(REPO).as_posix(), date_time=(2026, 1, 1, 0, 0, 0))  # reproducible bytes
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    return sha256_file(target)


if __name__ == "__main__":
    target = REPO / "artifacts/auto-prompt-sam2-e2.zip"
    print(target)
    print("sha256:", export(target))
