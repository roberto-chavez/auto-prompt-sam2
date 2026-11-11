"""Validate phase 2 on CPU without importing models or changing phase-1 evidence."""
import ast
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from auto_prompt_sam2.demo_config import inspect_artifacts, load_demo_config
from auto_prompt_sam2.artifacts import sha256_file


def main():
    report = {"scope": "phase2_local_no_models", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "python": platform.python_version(), "platform": platform.platform(),
              "models_executed": False, "training_executed": False,
              "gpu_inference": "not_run", "docker_build": "not_run", "deployment": "not_done"}
    files = [REPO / "app.py", *sorted((REPO / "src/auto_prompt_sam2").glob("demo_*.py")),
             REPO / "tests/demo/test_demo.py", REPO / "scripts/check_demo_artifacts.py", Path(__file__)]
    for file in files:
        ast.parse(file.read_text(encoding="utf-8"))
    report["syntax"] = "passed"
    report["source_sha256"] = {file.relative_to(REPO).as_posix(): sha256_file(file) for file in files}
    # Force the repository's preview configuration, independent of user shell overrides.
    report["default_artifacts"] = inspect_artifacts(load_demo_config(environ={}))
    report["versions"] = {name: importlib.metadata.version(name)
                          for name in ("gradio", "numpy", "pillow", "rasterio", "huggingface-hub")}
    env = dict(os.environ, PYTHONIOENCODING="utf-8", GRADIO_ANALYTICS_ENABLED="False")
    # A regression that tries to import a model framework fails immediately.
    guarded_tests = '''import importlib.abc, sys, unittest
class BlockModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torchvision', 'sam2', 'transformers'}:
            raise RuntimeError('Model framework import forbidden in local demo tests: ' + fullname)
sys.meta_path.insert(0, BlockModels())
suite = unittest.defaultTestLoader.discover('tests/demo', pattern='test_demo.py')
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
'''
    commands = [
        ("dependencies", [sys.executable, "-m", "pip", "check"]),
        ("cpu_contracts_and_http", [sys.executable, "-c", guarded_tests]),
        ("phase1_tracked_files_unchanged_against_HEAD", ["git", "diff", "--no-renames", "--diff-filter=MDT", "--exit-code", "HEAD", "--",
            "docs/project/evidence", "notebooks", "src/auto_prompt_sam2", "configs/environment.yaml",
            "configs/e2_intervention.yaml", "configs/data.yaml", "configs/baseline.yaml", "configs/profiling.yaml",
            "requirements-colab.txt", "requirements-colab.in", "requirements-stage2.txt",
            "docs/project/resultados_fase1.md"]),
        ("whitespace", ["git", "diff", "--check"]),
    ]
    report["checks"] = []
    for name, args in commands:
        run = subprocess.run(args, cwd=REPO, env=env, text=True, encoding="utf-8", capture_output=True)
        report["checks"].append({"name": name, "exit_code": run.returncode,
                                 "stdout": run.stdout, "stderr": run.stderr})
        print(name + ": " + ("passed" if run.returncode == 0 else "FAILED"))
    report["result"] = "passed_local_only" if all(c["exit_code"] == 0 for c in report["checks"]) else "failed"
    output = REPO / "artifacts/demo-local-validation.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output)
    return 0 if report["result"] == "passed_local_only" else 1


if __name__ == "__main__":
    raise SystemExit(main())
