"""Local (PC/CPU) validation of the E2 deliverables. Nothing here is a real experiment.

Writes docs/project/evidence/e2/local_validation.json. Run after exporting the bundle and
building the notebooks:  python scripts/validate_e2_local.py [--notebooks DIR] [--smoke]
"""
import argparse
import ast
import json
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from auto_prompt_sam2.artifacts import sha256_file, utc_now


def check_python(paths):
    out = {}
    for p in paths:
        ast.parse(p.read_text("utf-8"), filename=str(p))
        out[p.relative_to(REPO).as_posix()] = "syntax_ok"
    return out


def check_notebook(path, bundle_sha):
    nb = json.loads(path.read_text("utf-8"))
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    for i, c in enumerate(cells):
        ast.parse("".join(c["source"]), filename=f"{path.name}:code{i}")
        if c["outputs"] or c["execution_count"] is not None:
            raise ValueError("Notebook carries outputs/execution counts: " + path.name)
    text = "".join("".join(c["source"]) for c in cells)
    hashes = [bundle_sha in text]
    return {"code_cells": len(cells), "all_cells_parse": True, "no_outputs_no_execution_counts": True,
            "embeds_bundle_sha256": hashes[0], "sha256": sha256_file(path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebooks", type=Path, default=REPO / "artifacts/notebooks")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    bundle = REPO / "artifacts/auto-prompt-sam2-e2.zip"
    bundle_sha = sha256_file(bundle)
    with zipfile.ZipFile(bundle) as z:
        names = z.namelist()
    sources = [p for p in sorted([*(REPO / "src").rglob("*.py"), *(REPO / "scripts").rglob("*.py"), *(REPO / "tests").rglob("*.py")])]
    report = {"created_at": utc_now(), "kind": "local_cpu_validation_not_a_real_experiment",
              "python": sys.version.split()[0], "bundle": {"path": "artifacts/auto-prompt-sam2-e2.zip", "sha256": bundle_sha,
              "files": len(names), "contains_notebooks": any(n.endswith(".ipynb") for n in names),
              "contains_weights_or_data": any(n.endswith((".pt", ".pth", ".tif", ".npz", ".zip")) for n in names)},
              "syntax": check_python(sources), "notebooks": {}}
    for nb in sorted(args.notebooks.glob("*.ipynb")):
        report["notebooks"][nb.name] = check_notebook(nb, bundle_sha)
    if not all(v["embeds_bundle_sha256"] for v in report["notebooks"].values()):
        raise SystemExit("A notebook does not embed the current bundle SHA-256: rebuild notebooks")
    unit = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"], cwd=REPO, text=True, capture_output=True,
                          env={**__import__("os").environ, "PYTHONPATH": str(REPO / "src")})
    tail = unit.stderr.strip().splitlines()
    report["unit_tests"] = {"exit_code": unit.returncode, "summary": tail[-3:], "nature": "synthetic_and_random_weight_cpu_only"}
    pip = subprocess.run([sys.executable, "-m", "pip", "check"], text=True, capture_output=True)
    report["pip_check"] = {"exit_code": pip.returncode, "output": pip.stdout.strip()[-500:],
                           "note": "local environment, not the pinned Colab venv"}
    if args.smoke:
        smoke = subprocess.run([sys.executable, "tests/smoke_e2_pipeline.py"], cwd=REPO, text=True, capture_output=True)
        report["pipeline_smoke"] = {"exit_code": smoke.returncode, "last_lines": smoke.stdout.strip().splitlines()[-3:],
                                    "nature": "SYNTHETIC plumbing: random SAM weights, synthetic chips, no GPU, no real data"}
    report["not_validated_locally"] = ["real SAM weights", "CUDA/T4 behaviour and timing", "Google Drive persistence",
                                       "Sen1Floods11 data", "any segmentation quality or ablation result"]
    out = REPO / "docs/project/evidence/e2/local_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", "utf-8")
    print(json.dumps({k: report[k] for k in ("unit_tests", "pip_check")}, indent=2))
    if unit.returncode or pip.returncode or (args.smoke and report["pipeline_smoke"]["exit_code"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
