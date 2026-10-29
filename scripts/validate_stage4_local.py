"""Local syntax/import and CPU synthetic tests only, never constructs models."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from auto_prompt_sam2.artifacts import digest, sha256_file, utc_now


def main():
    paths = [*sorted((REPO / "src").rglob("*.py")), *sorted((REPO / "scripts").glob("*.py")),
             *sorted((REPO / "tests").glob("*.py"))]
    for path in paths:
        ast.parse(path.read_text("utf-8"))
    nbpath = REPO / "notebooks/04_diseccion_modelos.ipynb"
    notebook = json.loads(nbpath.read_text("utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell["execution_count"] is None and cell["outputs"] == []
    report = {"stage": "phase1_stage4", "created_at": utc_now(),
        "scope": "local_cpu_synthetic_no_models", "models_executed": False,
        "raw_data_read": False, "reserved_access": [], "syntax": "passed",
        "notebook": {"cells": len(notebook["cells"]), "syntax": "passed", "execution": "not_run"}, "checks": []}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               PYTHONHASHSEED="42", PYTHONIOENCODING="utf-8")
    for args in (["-m", "pip", "check"], ["-m", "unittest", "discover", "-s", "tests", "-v"]):
        result = subprocess.run([sys.executable, *args], cwd=REPO, env=env, text=True,
                                encoding="utf-8", capture_output=True)
        report["checks"].append({"command": ["python", *args], "exit_code": result.returncode,
                                 "stdout": result.stdout, "stderr": result.stderr})
        print(result.stdout, result.stderr, flush=True)
        if result.returncode:
            break
    sources = [*paths, nbpath, *(REPO / p for p in ("configs/profiling.yaml", "configs/data.yaml",
        "configs/environment.yaml", "requirements-colab.txt", "requirements-stage2.txt"))]
    report["source_sha256"] = {p.relative_to(REPO).as_posix(): sha256_file(p) for p in sorted(sources)}
    report["source_tree_sha256"] = digest(report["source_sha256"])
    report["status"] = "passed_local_only" if all(c["exit_code"] == 0 for c in report["checks"]) else "failed"
    report["pending"] = ["User-run Colab GPU prepare and offline recover on different runtime",
        "External notebook/JSON/ZIP audit, visual inspection and interpretation before closure"]
    target = REPO / "docs/project/evidence/stage4/local_validation.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", "utf-8")
    if report["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
