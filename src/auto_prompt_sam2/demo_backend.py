"""Lazy GPU-only inference. No torch/SAM imports until an authorized remote call."""
from __future__ import annotations

import logging
import threading
import json
from pathlib import Path

import numpy as np

from .demo_config import (artifact_contract, configured_decoder, decoder_expectation, decoder_kind,
                          require_remote_runtime, resolve_path, stage6_decoder_pin, verify_file, read_json)
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


def stage6_decoder_state(payload):
    """Stage-6 `decoder_export`: a plain mask-decoder state_dict of tensors (a3_tuned)."""
    if not isinstance(payload, dict) or not payload:
        raise ValueError("El payload de la etapa 6 debe ser un state_dict no vacío")
    if "best" in payload or "config_sha256" in payload:
        raise ValueError("Este es un payload E2; usar decoder_kind=e2_best")
    if not all(isinstance(k, str) and hasattr(v, "shape") and hasattr(v, "dtype") for k, v in payload.items()):
        raise ValueError("El state_dict contiene entradas que no son tensores")
    return payload


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
        self._dino = None
        self._automatic_bundle = None
        self._lock = threading.Lock()

    def _load_automatic_bundle(self):
        if self._automatic_bundle is not None:
            return self._automatic_bundle
        path = self.config.get("automatic_bundle_path")
        if not path:
            raise FileNotFoundError("Configura DEMO_AUTO_BUNDLE con el payload oficial_complete_bundle de Stage 5")
        path = resolve_path(path)
        payload = path.read_bytes()
        bundle = json.loads(payload)
        # Stage 6/7 pin the complete Stage-5 bundle by canonical content hash.
        pinned = read_json(Path(__file__).resolve().parents[2] /
                           "configs/stage7_dependencies.json")["stage6"]["official"]["identity"]["bundle_sha256"]
        from .artifacts import digest
        if digest(bundle) != pinned:
            raise ValueError("El bundle automático no coincide con el pin de Stage 7")
        if (bundle.get("regime") != "official" or
                bundle.get("base_checkpoint_sha256", {}).get("sam") != self.report["model"]["checkpoint_sha256"]):
            raise ValueError("El bundle automático usa otro régimen o checkpoint SAM")
        self._automatic_bundle = bundle
        return bundle

    def _dino_model(self, bundle):
        if self._dino is not None:
            return self._dino
        from .features import load_frozen_dino
        cache = resolve_path(self.config["cache_dir"])
        self._dino, _ = load_frozen_dino(bundle["models"]["dinov3"], cache,
            expected_sha256=bundle["base_checkpoint_sha256"]["dino"])
        return self._dino

    def _predict_automatic(self, image, bundle):
        import torch
        import torch.nn.functional as F
        from .automatic_prompts import probability_map, infer_points
        from .features import token_layout, tokens_to_grid
        from .probe import probabilities

        params = bundle["preprocessing"]
        model = self._dino_model(bundle)
        dino = params["config"]["dino"]
        rgb = torch.from_numpy(np.moveaxis(image.rgb, -1, 0).copy()).float()
        rgb = F.interpolate(rgb[None], size=(dino["size"], dino["size"]), mode="bilinear",
                             align_corners=False, antialias=True)[0]
        mean = torch.tensor(dino["mean"], dtype=torch.float32)[:, None, None]
        std = torch.tensor(dino["std"], dtype=torch.float32)[:, None, None]
        tensor = ((rgb - mean) / std)[None].cuda()
        layout = token_layout((512, 512), 16, model.config.num_register_tokens, model.config.hidden_size)
        with torch.inference_mode():
            grid = tokens_to_grid(model(pixel_values=tensor).last_hidden_state, layout)
            features = grid[0].permute(1, 2, 0).float().cpu().numpy().copy()
        if features.shape != (32, 32, 1024) or not np.isfinite(features).all():
            raise ValueError("DINOv3 feature geometry/value mismatch")
        patch_probabilities = probabilities(features, bundle["probe"], regime="official")
        probability = probability_map(patch_probabilities, output_hw=image.valid.shape)
        policy = bundle["prompts"]
        threshold = bundle["positive_threshold"]
        result, prompts = infer_points(self._predictor("frozen"), image.rgb.astype(np.float32),
                                       probability, image.valid, policy, threshold)
        predictor = self._predictors.get("frozen")
        if predictor is not None:
            predictor.reset_predictor()
        summary = {k: v for k, v in prompts.items() if k != "component_map"}
        summary["component_map"] = None
        summary["points"] = sum(len(component["point_coords"]) for component in prompts["components"])
        summary["components"] = len(prompts["components"])
        summary["positive_threshold"] = threshold
        summary["probe_source"] = "pinned Stage-5 official_complete_bundle"
        return result, summary, probability

    def segment_automatic(self, image, compare=True):
        """DINOv3 probe -> deterministic points -> frozen SAM; optionally the pinned a3_tuned arm."""
        require_remote_runtime(self.config)
        with self._lock:
            import torch
            with torch.inference_mode():
                bundle = self._load_automatic_bundle()
                generated, prompt_info, probability = self._predict_automatic(image, bundle)
                frozen = generated
                tuned = None
                status = (f"Automático: {prompt_info['components']} componentes, {prompt_info['points']} puntos; "
                          f"umbral {prompt_info['positive_threshold']:.2f}; DINOv3 → SAM congelado")
                if compare and configured_decoder(self.config):
                    try:
                        from .automatic_prompts import infer_points
                        tuned, _ = infer_points(self._predictor("tuned"), image.rgb.astype(np.float32),
                            probability, image.valid, bundle["prompts"], bundle["positive_threshold"])
                        predictor = self._predictors.get("tuned")
                        if predictor is not None:
                            predictor.reset_predictor()
                        status += ("; comparado con a3_tuned official (mismo prompt automático; "
                                   "no implica evaluación independiente)")
                    except Exception:
                        LOG.exception("No se pudo ejecutar a3_tuned en la ruta automática")
                        status += "; a3_tuned no disponible (revisar log)"
                return {"frozen": frozen, "tuned": tuned, "status": status,
                        "automatic_prompts": prompt_info}

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
            sha, size = decoder_expectation(config, report)
            if decoder_kind(config) == "stage6_a3_tuned" and not local:
                raise FileNotFoundError("a3_tuned requiere decoder_checkpoint_path local (Drive/Colab); sin descarga remota")
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
            state = (stage6_decoder_state(payload) if decoder_kind(self.config) == "stage6_a3_tuned"
                     else best_decoder_state(payload, self.report))
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
                        if decoder_kind(self.config) == "stage6_a3_tuned":
                            status = ("Comparación con a3_tuned (etapa 6/7, régimen official, generación "
                                      f"{stage6_decoder_pin()['generation']}); ajustado con puntos automáticos, "
                                      "no evaluado con clics manuales")
                        else:
                            status = f"Comparación con best, época {self.report['checkpoints']['best']['epoch']}"
                    except Exception:
                        LOG.exception("No se pudo ejecutar el decoder ajustado")
                        status = "Decoder ajustado no disponible o inválido; revisar configuración y logs del servidor"
                return {"frozen": frozen, "tuned": tuned, "status": status}
