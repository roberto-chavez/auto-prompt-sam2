"""Runtime inventory and reproducibility, without model construction."""
from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import random
import shutil
import subprocess
import sys
from pathlib import Path

from .artifacts import digest, sha256_file, utc_now


def load_config(repo: Path) -> dict:
    # JSON is a YAML 1.2 subset: bootstrap needs only Python's standard library.
    return json.loads((Path(repo) / "configs/environment.yaml").read_text("utf-8"))


def persistent_root(config: dict, *, colab: bool = False) -> Path:
    root = Path(os.environ.get(config["persistent_root_env"],
                               config["persistent_root_default"])).expanduser().resolve()
    if colab and root.is_relative_to(Path("/content").resolve()):
        if not root.is_relative_to(Path("/content/drive/MyDrive").resolve()):
            raise ValueError("Persistent root must not be Colab temporary storage")
        # Do not accidentally create a fake Drive directory on ephemeral disk.
        if not os.path.ismount("/content/drive"):
            raise RuntimeError("Mount Google Drive before creating persistent root")
    elif colab:
        mounts = [p for p in (root, *root.parents) if p != Path(p.anchor) and os.path.ismount(p)]
        if not mounts:
            raise ValueError("Custom persistent root requires an explicitly mounted volume")
    root.mkdir(parents=True, exist_ok=True)
    return root


def seed_everything(seed: int, strict: bool = True):
    import numpy as np
    import torch
    if os.environ.get("PYTHONHASHSEED") != str(seed):
        raise RuntimeError("Set PYTHONHASHSEED before launching this Python process")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(strict)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = strict
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def source_files(repo: Path):
    paths = [repo / "configs/environment.yaml", repo / "requirements-colab.in",
             repo / "requirements-colab.txt", repo / "notebooks/01_entorno_colab.ipynb"]
    for folder in ("src", "scripts", "tests"):
        paths.extend((repo / folder).rglob("*.py"))
    return sorted(p for p in paths if p.is_file())


def provenance(repo: Path):
    repo = Path(repo).resolve()
    files = {p.relative_to(repo).as_posix(): sha256_file(p) for p in source_files(repo)}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"commit": commit, "working_tree_dirty": dirty, "source_sha256": files,
            "source_tree_sha256": digest(files)}


def run_identity(repo: Path, config: dict):
    return {"stage": config["stage"], "config_sha256": digest(config),
            "source": provenance(repo), "seed": config["seed"],
            "precision": config["precision"], "models": config["models"],
            "dataset": config["dataset"], "split_sha256": None,
            "regime": "environment_only"}


def inventory(root: Path, repo: Path):
    import psutil
    import torch
    memory = psutil.virtual_memory()
    disks = {}
    for name, path in (("persistent", root), ("local", repo)):
        usage = shutil.disk_usage(path)
        disks[name] = {"total_bytes": usage.total, "free_bytes": usage.free}
    gpus = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(index)
            prop = torch.cuda.get_device_properties(index)
            gpus.append({"index": index, "name": prop.name, "total_bytes": total,
                         "free_bytes": free, "capability": list(torch.cuda.get_device_capability(index))})
    try:
        raw = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version",
                                      "--format=csv,noheader,nounits"], text=True, stderr=subprocess.DEVNULL,
                                     timeout=15).strip().splitlines()
        nvidia_devices = []
        for line in raw:
            name, total, free, driver = [item.strip() for item in line.split(",")]
            nvidia_devices.append({"name": name, "total_mib": int(total), "free_mib": int(free),
                                   "driver_version": driver})
    except (OSError, subprocess.SubprocessError):
        nvidia_devices = None
    packages = sorted({d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}.items())
    return {"recorded_at": utc_now(), "platform": platform.platform(),
            "python": sys.version, "cpu_count": os.cpu_count(),
            "ram_total_bytes": memory.total, "ram_available_bytes": memory.available,
            "disk": disks, "persistent_root": str(root), "cuda_available": torch.cuda.is_available(),
            "torch_cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "nvidia_smi_devices": nvidia_devices, "gpus": gpus, "packages": dict(packages)}


def import_checks():
    import torch
    import torchvision
    from sam2.build_sam import build_sam2
    from transformers import DINOv3ViTModel, AutoImageProcessor
    # Only import symbols. Do not instantiate, download or call a model.
    cuda_smoke = "not_run_unavailable"
    if torch.cuda.is_available():
        value = torch.tensor([1.0, 2.0], device="cuda").sum().item()
        torch.cuda.synchronize()
        if value != 3.0:
            raise RuntimeError("CUDA scalar arithmetic failed")
        cuda_smoke = "passed_scalar_arithmetic_no_model"
    return {"torch": torch.__version__, "torchvision": torchvision.__version__,
            "sam2_build_import": callable(build_sam2),
            "dinov3_import": DINOv3ViTModel.__name__,
            "processor_import": AutoImageProcessor.__name__, "cuda_smoke": cuda_smoke}


def verify_installed_pins(repo: Path, config: dict, *, linux_lock: bool):
    """Validate installed versions, including SAM's immutable VCS provenance."""
    import re
    from packaging.version import Version
    filename = "requirements-colab.txt" if linux_lock else "requirements-colab.in"
    pins = re.findall(r"^([A-Za-z0-9_-]+)==([^\s;\\]+)", (repo / filename).read_text("utf-8"), re.M)
    mismatches = []
    for name, expected in pins:
        try:
            installed = importlib.metadata.version(name)
            if Version(installed).base_version != Version(expected).base_version:
                mismatches.append(name)
        except importlib.metadata.PackageNotFoundError:
            mismatches.append(name)
    sam = importlib.metadata.distribution("SAM-2")
    direct = json.loads(sam.read_text("direct_url.json") or "{}")
    if direct.get("vcs_info", {}).get("commit_id") != config["models"]["sam2"]["source_revision"]:
        mismatches.append("SAM-2 source revision")
    if mismatches:
        raise RuntimeError("Installed pins differ: " + ", ".join(mismatches))
    return {"verified_pins": len(pins), "requirements_file": filename,
            "sam_source_revision": direct["vcs_info"]["commit_id"]}


def resource_blockers(config: dict, report: dict):
    blockers = []
    if config["require_cuda_for_stage_closure"] and not report["cuda_available"]:
        blockers.append("CUDA unavailable; GPU validation remains pending")
    for name, gib in (("persistent", sum(config["disk_budget_gib"].values())),
                      ("local", config["minimum_local_free_gib"])):
        if report["disk"][name]["free_bytes"] < gib * 1024**3:
            blockers.append(f"Insufficient {name} free space for configured budget: {gib} GiB")
    # No claim that observed VRAM is sufficient for later inference/training.
    return blockers
