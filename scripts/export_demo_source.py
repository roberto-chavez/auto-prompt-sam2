"""Package only the demo source and public reference JSON for a Colab GPU run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

REPO = Path(__file__).resolve().parents[1]
FILES = [
    Path("app.py"), Path("requirements-demo.txt"), Path("requirements-demo-inference.txt"),
    Path("configs/demo.json"),
    Path("configs/environment.yaml"),
    Path("configs/stage7_dependencies.json"),
    Path("docs/project/evidence/e2/colab_run_20261030T013250Z/run/report.json"),
    Path("docs/project/evidence/stage2/official_preprocessing.json"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/demo-colab-source.zip")
    args = parser.parse_args()
    output = Path(args.output)
    if not output.is_absolute():
        output = REPO / output
    output.parent.mkdir(parents=True, exist_ok=True)
    files = FILES + sorted(path.relative_to(REPO) for path in (REPO / "src/auto_prompt_sam2").glob("*.py"))
    missing = [str(rel) for rel in files if not (REPO / rel).is_file()]
    if missing:
        raise FileNotFoundError("Missing demo source files: " + ", ".join(missing))
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, check=True,
                                 capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    manifest = {"schema_version": 1, "commit_HEAD": commit,
                "sha256": {rel.as_posix(): hashlib.sha256((REPO / rel).read_bytes()).hexdigest()
                           for rel in files}}
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for rel in files:
            archive.write(REPO / rel, rel.as_posix())
        archive.writestr("demo_source_manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(output), "bytes": output.stat().st_size,
                      "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                      "commit_HEAD": commit, "contains_model_weights": False,
                      "source_files": len(files)}, indent=2))


if __name__ == "__main__":
    main()
