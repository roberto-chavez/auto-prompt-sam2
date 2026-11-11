"""Lazy GPU-only inference. No torch/SAM imports until an authorized remote call."""
from __future__ import annotations

import logging
import threading

import numpy as np

from .demo_config import (artifact_contract, configured_decoder, require_remote_runtime,
                          resolve_path, verify_file)
from .demo_images import validate_points
from .sam_adapter import select_multimask

LOG = logging.getLogger(__name__)


def best_decoder_state(payload, report):
    """E2 best payload schema; do not interpret a full SAM state or last epoch as best."""
    if not isinstance(payload, dict) or payload.get("config_sha256") != report["identity"]["config_sha256"]:
        raise ValueError("El checkpoint no corresponde a la configuración del reporte")
    best = payload.get("best", {})
    epoch = report["checkpoints"]["best"]["epoch"]
    if payload.get("epoch") != epoch or best.get("epoch") != epoch or not isinstance(best.get("state"), dict):
        raise ValueError("Se requiere el payload best de la época registrada")
    if not best["state"]:
        raise ValueError("El decoder está vacío")
    return best["state"]


def predict_manual(predictor, image, points):
    """Single manual object prompt, same native (x,y) and multimask rule as E2."""
    validate_points(points, image, max_points=256)
    try:
        # Use original float32 pixels, never the uint8 canvas with painted markers.
        predictor.set_image(image.rgb)
        masks, scores, _ = predictor.predict(
            point_coords=np.array([p[:2] for p in points], dtype=np.float32),
            point_labels=np.array([p[2] for p in points], dtype=np.int32),
            multimask_output=True, return_logits=False, normalize_coords=True)
        mask, decision = select_multimask(masks, scores, image.valid.shape)
        return {"mask": mask & image.valid, "selection": decision}
    finally:
        # A shared predictor must not retain an image from another session.
        predictor.reset_predictor()


class DemoBackend:
    def __init__(self, config):
        self.config = config
        self.report, self.params = artifact_contract(config)
        self._predictors = {}
        self._lock = threading.Lock()

    def _checkpoint(self, arm):
        config, report = self.config, self.report
        if arm == "frozen":
            local = config.get("base_checkpoint_path")
            model = report["identity"]["model"]
            repo, revision, filename = model["repo_id"], model["revision"], model["filename"]
            sha, size = report["model"]["checkpoint_sha256"], None
        else:
            local = config.get("decoder_checkpoint_path")
            repo, revision, filename = (config.get(k) for k in
                                        ("decoder_repo_id", "decoder_revision", "decoder_filename"))
            entry = report["checkpoints"]["best"]
            sha, size = entry["sha256"], entry["bytes"]
        if local:
            path = resolve_path(local)
        else:
            if not config["allow_download"]:
                raise FileNotFoundError("Faltan pesos locales; habilitar descargas solo en Colab/Spaces")
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(repo_id=repo, revision=revision, filename=filename,
                                   cache_dir=str(resolve_path(config["cache_dir"])))
        return verify_file(path, sha, size)

    def _predictor(self, arm):
        require_remote_runtime(self.config)
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("Se necesita GPU CUDA; no se permite inferencia en CPU")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        if arm in self._predictors:
            return self._predictors[arm]
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        base_path = self._checkpoint("frozen")
        state = None
        if arm == "tuned":
            payload = torch.load(self._checkpoint("tuned"), map_location="cpu", weights_only=True)
            state = best_decoder_state(payload, self.report)
        # Separate model instances keep the baseline immutable after decoder loading.
        model = build_sam2(self.report["identity"]["model"]["architecture_config"],
                           str(base_path), device="cuda", mode="eval", apply_postprocessing=False)
        if state is not None:
            model.sam_mask_decoder.load_state_dict(state, strict=True)
        model.float().eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        predictor = SAM2ImagePredictor(model, mask_threshold=0.0,
                                       max_hole_area=0.0, max_sprinkle_area=0.0)
        self._predictors[arm] = predictor
        return predictor

    def segment(self, image, points, compare=True):
        require_remote_runtime(self.config)
        validate_points(points, image, self.config["max_points"])
        # Serialize load + set_image + predict + reset across ALL sessions and arms.
        with self._lock:
            import torch
            with torch.inference_mode():
                frozen = predict_manual(self._predictor("frozen"), image, points)
                tuned = None
                status = "Comparación desactivada por el usuario"
                if compare and not configured_decoder(self.config):
                    status = "Decoder ajustado no configurado: se muestra únicamente SAM congelado"
                elif compare:
                    try:
                        tuned = predict_manual(self._predictor("tuned"), image, points)
                        status = f"Comparación con best, época {self.report['checkpoints']['best']['epoch']}"
                    except Exception:
                        LOG.exception("No se pudo ejecutar el decoder ajustado")
                        status = "Decoder ajustado no disponible o inválido; revisar configuración y logs del servidor"
                return {"frozen": frozen, "tuned": tuned, "status": status}
