"""Evidence from an existing clean local venv; never claim a Colab execution."""
import ast
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from auto_prompt_sam2.artifacts import ArtifactStore, sha256_file, utc_now
from auto_prompt_sam2.runtime import load_config, run_identity


def main():
    output = REPO / "docs/project/evidence/stage1/local_validation.json"
    config = load_config(REPO)
    root = REPO / "artifacts" / ("local-validation-" + uuid.uuid4().hex[:8])
    root.mkdir(parents=True)
    env = dict(os.environ, AUTO_PROMPT_PERSIST_ROOT=str(root), PYTHONHASHSEED=str(config["seed"]),
               CUBLAS_WORKSPACE_CONFIG=":4096:8", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    report = {"started_at": utc_now(), "scope": "local_windows_not_colab", "checks": []}
    notebook = json.loads((REPO / "notebooks/01_entorno_colab.ipynb").read_text("utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell["execution_count"] is None and cell["outputs"] == []
    report["notebook"] = {"cells": len(notebook["cells"]), "code_syntax": "passed", "execution": "not_run"}
    commands = [
        ["-m", "pip", "check"],
        ["-m", "unittest", "discover", "-s", "tests", "-v"],
        ["scripts/validate_environment.py", "prepare"],
        ["scripts/validate_environment.py", "recover"],
    ]
    for command in commands:
        completed = subprocess.run([sys.executable, *command], cwd=REPO, env=env,
                                   text=True, capture_output=True)
        report["checks"].append({"command": ["python", *command], "exit_code": completed.returncode,
                                 "stdout": completed.stdout, "stderr": completed.stderr})
        if completed.returncode:
            report["result"] = "failed"
            output.write_text(json.dumps(report, indent=2) + "\n", "utf-8")
            print("Validation failed:", " ".join(command), completed.stderr)
            raise SystemExit(completed.returncode)
    identity = run_identity(REPO, config)
    ticket = json.loads(ArtifactStore(root / "environment").read("recovery_ticket", identity)[0])
    store = ArtifactStore(root / "environment/runs" / ticket["run_id"])
    report["environment"] = json.loads(store.read("environment", identity)[0])
    report["recovery"] = json.loads(store.read("recovery_validation", identity)[0])
    report["requirements_colab_sha256"] = sha256_file(REPO / "requirements-colab.txt")
    report["result"] = "passed_local_only"
    report["finished_at"] = utc_now()
    output.write_text(json.dumps(report, indent=2) + "\n", "utf-8")
    print("Local checks passed. Colab, CUDA execution and authenticated access remain pending.")


if __name__ == "__main__":
    main()
