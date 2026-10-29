"""Local (PC/CPU) validation of the E2 deliverables. Nothing here is a real experiment.

Writes docs/project/evidence/e2/local_validation.json.
Usage: python scripts/validate_e2_local.py [--smoke]
"""
import argparse
import ast
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from auto_prompt_sam2.artifacts import sha256_file, utc_now

NOTEBOOKS = ("E2_01_datos_baseline_teardown.ipynb", "E2_02_intervencion_decoder.ipynb")


def check_notebook(path):
    nb = json.loads(path.read_text("utf-8"))
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    for i, c in enumerate(cells):
        ast.parse("".join(c["source"]), filename=f"{path.name}:code{i}")
        if c["outputs"] or c["execution_count"] is not None:
            raise ValueError("Notebook carries outputs/execution counts: " + path.name)
    text = "".join("".join(c["source"]) for c in cells)
    for forbidden in ("hf_", "EXPECTED_BUNDLE_SHA256", "BUNDLE ="):
        if forbidden in text:
            raise ValueError("Unexpected content %r in %s" % (forbidden, path.name))
    return {"code_cells": len(cells), "all_cells_parse": True, "no_outputs_no_execution_counts": True,
            "no_embedded_tokens_or_bundle_hashes": True, "sha256": sha256_file(path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    sources = sorted([*(REPO / "src").rglob("*.py"), *(REPO / "scripts").rglob("*.py"), *(REPO / "tests").rglob("*.py")])
    report = {"created_at": utc_now(), "kind": "local_cpu_validation_not_a_real_experiment", "python": sys.version.split()[0],
              "syntax": {}, "notebooks": {}}
    for p in sources:
        ast.parse(p.read_text("utf-8"), filename=str(p))
        report["syntax"][p.relative_to(REPO).as_posix()] = "syntax_ok"
    for name in NOTEBOOKS:
        report["notebooks"][name] = check_notebook(REPO / "notebooks" / name)
    # Everything the notebooks call must exist in the repository (a fresh clone is all Colab has).
    needed = ["scripts/bootstrap_colab.py", "scripts/prepare_stage2.py", "scripts/run_stage3.py", "scripts/run_stage4.py",
              "scripts/run_e2.py", "scripts/e2_decision_log_draft.py", "requirements-colab.txt", "requirements-stage2.txt",
              "configs/environment.yaml", "configs/data.yaml", "configs/baseline.yaml", "configs/profiling.yaml",
              "configs/e2_intervention.yaml", "docs/project/validacion_etapa4.md", "notebooks/01_entorno_colab.ipynb",
              "notebooks/02_datos_y_splits.ipynb", "notebooks/04_diseccion_modelos.ipynb"]
    missing = [n for n in needed if not (REPO / n).is_file()]
    report["files_required_by_notebooks_present_in_repo"] = {"checked": needed, "missing": missing}
    if missing:
        raise SystemExit("Missing files: %s" % missing)
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    unit = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"], cwd=REPO, text=True, capture_output=True, env=env)
    report["unit_tests"] = {"exit_code": unit.returncode, "summary": unit.stderr.strip().splitlines()[-3:],
                            "nature": "synthetic_and_random_weight_cpu_only"}
    pip = subprocess.run([sys.executable, "-m", "pip", "check"], text=True, capture_output=True)
    report["pip_check"] = {"exit_code": pip.returncode, "output": pip.stdout.strip()[-300:],
                           "note": "local environment, not the pinned Colab venv"}
    if args.smoke:
        smoke = subprocess.run([sys.executable, "tests/smoke_e2_pipeline.py"], cwd=REPO, text=True, capture_output=True)
        report["pipeline_smoke"] = {"exit_code": smoke.returncode, "last_lines": smoke.stdout.strip().splitlines()[-3:],
                                    "nature": "SYNTHETIC plumbing: random SAM weights, synthetic chips, no GPU, no real data"}
    report["not_validated_locally"] = ["real SAM weights", "CUDA/T4 behaviour and timing", "Google Drive persistence",
                                       "Sen1Floods11 download", "Colab clone/bootstrap of this repository",
                                       "any segmentation quality or ablation result"]
    out = REPO / "docs/project/evidence/e2/local_validation.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", "utf-8")
    print(json.dumps({k: report[k] for k in ("unit_tests", "pip_check")}, indent=2))
    if unit.returncode or pip.returncode or (args.smoke and report["pipeline_smoke"]["exit_code"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
