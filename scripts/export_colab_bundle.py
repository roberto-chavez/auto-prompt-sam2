"""Export only stage-1 code/config/notebook, no local secrets, data or weights."""
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from auto_prompt_sam2.runtime import source_files
from auto_prompt_sam2.artifacts import sha256_file

if __name__ == "__main__":
    target = REPO / "artifacts/auto-prompt-sam2-stage1.zip"
    target.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in source_files(REPO):
            archive.write(path, path.relative_to(REPO).as_posix())
    print(target)
    print("sha256:", sha256_file(target))
