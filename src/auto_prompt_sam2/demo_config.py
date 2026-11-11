"""Demo configuration and artifact checks. Standard library only; never loads weights."""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from .artifacts import digest, sha256_file

REPO = Path(__file__).resolve().parents[2]
SAM_SOURCE_REVISION = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
ENV_FIELDS = {
    "DEMO_MODE": "mode",
    "DEMO_REPORT_PATH": "report_path",
    "DEMO_PREPROCESSING_PATH": "preprocessing_path",
    "DEMO_BASE_CHECKPOINT": "base_checkpoint_path",
    "DEMO_DECODER_CHECKPOINT": "decoder_checkpoint_path",
    "DEMO_DECODER_REPO_ID": "decoder_repo_id",
    "DEMO_DECODER_REVISION": "decoder_revision",
    "DEMO_DECODER_FILENAME": "decoder_filename",
    "DEMO_CACHE_DIR": "cache_dir",
}


def resolve_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO / path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_demo_config(path=None, *, environ=None):
    env = os.environ if environ is None else environ
    config = read_json(resolve_path(path or env.get("DEMO_CONFIG", "configs/demo.json")))
    for key, field in ENV_FIELDS.items():
        if key in env:
            config[field] = env[key] or None
    if "DEMO_ALLOW_DOWNLOAD" in env:
        if env["DEMO_ALLOW_DOWNLOAD"] not in ("0", "1"):
            raise ValueError("DEMO_ALLOW_DOWNLOAD debe ser 0 o 1")
        config["allow_download"] = env["DEMO_ALLOW_DOWNLOAD"] == "1"
    if config["mode"] not in ("preview", "inference"):
        raise ValueError("DEMO_MODE debe ser preview o inference")
    if type(config["allow_download"]) is not bool:
        raise ValueError("allow_download debe ser booleano")
    for key, ceiling in (("max_points", 256), ("max_upload_mb", 128)):
        if type(config[key]) is not int or not 1 <= config[key] <= ceiling:
            raise ValueError(f"{key} fuera del rango permitido")
    remote = [config.get(k) for k in ("decoder_repo_id", "decoder_revision", "decoder_filename")]
    if any(remote) and not all(remote):
        raise ValueError("Decoder remoto: indicar repo_id, revision y filename juntos")
    if all(remote) and not re.fullmatch(r"[0-9a-f]{40}", remote[1]):
        raise ValueError("La revisión HF del decoder debe ser un commit de 40 caracteres")
    if all(remote) and config.get("decoder_checkpoint_path"):
        raise ValueError("Elegir decoder local o remoto, no ambos")
    return config


def artifact_contract(config):
    report = read_json(resolve_path(config["report_path"]))
    params = read_json(resolve_path(config["preprocessing_path"]))
    model = report["identity"]["model"]
    if (model["source_revision"] != SAM_SOURCE_REVISION or
            model["architecture_config"] != "configs/sam2.1/sam2.1_hiera_t.yaml" or
            model["repo_id"] != "facebook/sam2.1-hiera-tiny"):
        raise ValueError("El reporte no corresponde a la arquitectura/revisión SAM soportada")
    if digest(params) != report["identity"]["preprocessing_sha256"]:
        raise ValueError("El preprocesamiento no coincide con el reporte E2")
    if (params["source_role"] != "train" or params["regime"] != "official" or
            params["config"]["rgb_bands_one_based"] != [4, 3, 2] or
            params["upper_dn"] <= 0):
        raise ValueError("Preprocesamiento incompatible con la demo")
    for value in (report["model"]["checkpoint_sha256"], report["checkpoints"]["best"]["sha256"]):
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("SHA-256 ausente o inválido en el reporte")
    return report, params


def verify_file(path, sha256, size=None):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Falta el checkpoint configurado; consultar docs/project/fase2_demo.md")
    if size is not None and path.stat().st_size != size:
        raise ValueError("Tamaño de checkpoint distinto al reporte")
    if sha256_file(path) != sha256:
        raise ValueError("SHA-256 de checkpoint distinto al reporte")
    return path


def require_remote_runtime(config, *, environ=None, platform=None):
    """Fail BEFORE importing torch/SAM, downloading or reading checkpoint payloads."""
    env = os.environ if environ is None else environ
    platform = sys.platform if platform is None else platform
    if config["mode"] != "inference":
        raise RuntimeError("Modo preparación: no se ejecutan modelos ni se generan máscaras")
    colab = (Path("/content").is_dir() and
             env.get("AUTO_PROMPT_EXECUTION_CONTEXT") == "google_colab_user_notebook")
    if platform != "linux" or not (env.get("SPACE_ID") or colab):
        raise RuntimeError("Inferencia permitida solo en Colab o Hugging Face Spaces con GPU")


def configured_decoder(config):
    return bool(config.get("decoder_checkpoint_path") or config.get("decoder_repo_id"))


def inspect_artifacts(config):
    """Read-only preflight; hashes local bytes but never deserializes or downloads."""
    report, params = artifact_contract(config)
    result = {"mode": config["mode"], "run_id": report["run_id"],
              "preprocessing_matches_report": True, "upper_dn": params["upper_dn"],
              "models_executed": False, "weights_deserialized": False}
    for arm, field, sha, size in (
        ("base", "base_checkpoint_path", report["model"]["checkpoint_sha256"], None),
        ("decoder", "decoder_checkpoint_path", report["checkpoints"]["best"]["sha256"],
         report["checkpoints"]["best"]["bytes"]),
    ):
        if config.get(field):
            verify_file(resolve_path(config[field]), sha, size)
            result[arm] = "local_hash_verified"
        else:
            result[arm] = "remote_configured_unverified" if arm == "base" or configured_decoder(config) else "not_configured"
    result["remote_validation"] = "pending_GPU_inference_and_checkpoint_schema"
    return result
