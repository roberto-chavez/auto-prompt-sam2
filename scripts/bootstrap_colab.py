"""Idempotent, isolated Linux/CPython 3.12 setup. No data/model downloads."""
import argparse
import json
import os
import platform
import subprocess
import sys
import venv
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from auto_prompt_sam2.artifacts import sha256_file
from auto_prompt_sam2.runtime import load_config


def install_tooling(python: Path, env: dict):
    """Host pip can provision a pip-less venv (e.g. Colab without ensurepip)."""
    subprocess.run([sys.executable, "-m", "pip", "--python", str(python),
                    "install", "--index-url", "https://pypi.org/simple",
                    "pip==25.2", "setuptools==80.9.0", "wheel==0.45.1"],
                   check=True, env=env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--venv", type=Path, default=Path("/content/auto-prompt-venv"))
    args = parser.parse_args()
    config = load_config(REPO)
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise SystemExit("This lock targets Linux x86_64 only; local tests are separate.")
    if f"{sys.version_info.major}.{sys.version_info.minor}" != config["python_minor"]:
        raise SystemExit("Select a Colab Python 3.12 runtime; do not silently change the lock.")
    import shutil
    if shutil.disk_usage(REPO).free < config["minimum_local_free_gib"] * 1024**3:
        raise SystemExit("Not enough local disk for the installation budget.")
    target = args.venv.resolve()
    python = target / "bin/python"
    if not python.exists():
        venv.EnvBuilder(with_pip=False).create(target)
    minor = subprocess.check_output([str(python), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"], text=True).strip()
    if minor != config["python_minor"]:
        raise SystemExit("Existing venv has another Python version; choose a new empty venv path.")
    # A successful stamp is not a substitute for package checks.
    env = dict(os.environ, SAM2_BUILD_CUDA="0", PIP_DISABLE_PIP_VERSION_CHECK="1")
    def run(*args):
        subprocess.run([str(python), "-m", "pip", *args], check=True, env=env)
    install_tooling(python, env)
    run("install", "--index-url", "https://pypi.org/simple", "--require-hashes", "--no-build-isolation",
        "-r", str(REPO / "requirements-colab.txt"))
    sam = config["models"]["sam2"]
    run("install", "--no-deps", "--no-build-isolation",
        f'SAM-2 @ git+{sam["source_repo"]}@{sam["source_revision"]}')
    run("check")
    stamp = {"requirements_sha256": sha256_file(REPO / "requirements-colab.txt"),
             "sam_source_revision": sam["source_revision"], "python": str(python)}
    (target / "stage1-installed.json").write_text(json.dumps(stamp, indent=2), "utf-8")
    print("Isolated environment ready:", python)


if __name__ == "__main__":
    main()
