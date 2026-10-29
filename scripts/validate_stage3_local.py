"""CPU-only synthetic validation evidence. Never downloads or constructs models."""
import ast
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from auto_prompt_sam2.artifacts import digest, sha256_file, utc_now


def main():
    report = {"stage": "phase1_stage3", "scope": "local_cpu_synthetic_no_models",
              "created_at": utc_now(), "models_executed": False,
              "raw_data_read": False, "reserved_access": [], "checks": []}
    paths = [*sorted((REPO / "src").rglob("*.py")), *sorted((REPO / "scripts").glob("*.py")),
             *sorted((REPO / "tests").glob("*.py"))]
    for path in paths:
        ast.parse(path.read_text("utf-8"))
    nbpath = REPO / "notebooks/03_linea_base_asistida.ipynb"
    notebook = json.loads(nbpath.read_text("utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell["execution_count"] is None and cell["outputs"] == []
    report["syntax"] = "passed"
    report["notebook"] = {"cells": len(notebook["cells"]), "syntax": "passed", "execution": "not_run"}
    env = dict(os.environ, PYTHONHASHSEED="42", CUDA_VISIBLE_DEVICES="",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    for args in (["-m", "pip", "check"], ["-m", "unittest", "discover", "-s", "tests", "-v"]):
        result = subprocess.run([sys.executable, *args], cwd=REPO, env=env, capture_output=True, text=True)
        report["checks"].append({"command": ["python", *args], "exit_code": result.returncode,
                                 "stdout": result.stdout, "stderr": result.stderr})
        print(result.stdout, result.stderr, flush=True)
        if result.returncode:
            break
    sources = [*paths, nbpath, *(REPO / p for p in ("configs/baseline.yaml", "configs/data.yaml",
        "configs/environment.yaml", "requirements-colab.in", "requirements-colab.txt", "requirements-stage2.txt"))]
    report["source_sha256"] = {p.relative_to(REPO).as_posix(): sha256_file(p) for p in sorted(sources)}
    report["source_tree_sha256"] = digest(report["source_sha256"])
    report["status"] = "passed_local_only" if all(c["exit_code"] == 0 for c in report["checks"]) else "failed"
    report["pending"] = ["User Colab prepare/recover with GPU and real frozen SAM",
                         "External review of notebook, JSON and artifact evidence before stage closure"]
    output = REPO / "docs/project/evidence/stage3/local_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", "utf-8")
    if report["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
