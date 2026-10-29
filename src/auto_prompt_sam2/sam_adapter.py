"""Image + prompts only. No label reader, label validity or ground-truth scoring."""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np

from .artifacts import sha256_file


def require_colab_gpu():
    # The notebook supplies context to the isolated child venv. Fail before any
    # import/download/construction of SAM on a workstation or CPU-only runtime.
    if (sys.platform != "linux" or not Path("/content").is_dir() or
            os.environ.get("AUTO_PROMPT_EXECUTION_CONTEXT") != "google_colab_user_notebook" or
            not os.path.ismount("/content/drive")):
        raise RuntimeError("Real SAM execution is allowed only in the user-run Google Colab notebook with mounted Drive")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Colab GPU required; do not fall back to local/CPU inference")


def state_fingerprint(model):
    """Hash every parameter AND persistent buffer; only used with SAM in Colab."""
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str((tuple(value.shape), value.dtype)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def assert_frozen(model):
    if any(module.training for module in model.modules()):
        raise ValueError("All SAM modules must be in eval mode")
    if any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise ValueError("All SAM parameters must be frozen with no gradients")


def load_frozen_sam(model_config, root, *, offline=False, expected_sha256=None):
    require_colab_gpu()
    from huggingface_hub import hf_hub_download
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    # Explicit revision: build_sam2_hf/from_pretrained would use a floating default.
    path = Path(hf_hub_download(repo_id=model_config["repo_id"],
        revision=model_config["revision"], filename=model_config["filename"],
        cache_dir=str(Path(root) / "base_models/huggingface"), local_files_only=offline))
    sha = sha256_file(path)
    if expected_sha256 is not None and sha != expected_sha256:
        raise ValueError("SAM checkpoint hash changed")
    model = build_sam2(model_config["architecture_config"], str(path), device="cuda",
                       mode="eval", apply_postprocessing=False)
    model.float().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    assert_frozen(model)
    predictor = SAM2ImagePredictor(model, mask_threshold=0.0, max_hole_area=0.0, max_sprinkle_area=0.0)
    metadata = {"model": dict(model_config), "checkpoint_sha256": sha,
                "checkpoint_path": str(path.relative_to(Path(root))),
                "state_sha256": state_fingerprint(model), "all_parameters_frozen": True,
                "all_modules_eval": True, "precision": "float32", "autocast": False,
                "apply_postprocessing": False,
                "parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    return predictor, metadata


def select_multimask(masks, predicted_scores, output_hw):
    masks, scores = np.asarray(masks), np.asarray(predicted_scores)
    if (masks.ndim != 3 or tuple(masks.shape[1:]) != tuple(output_hw) or
            scores.ndim != 1 or len(scores) != len(masks) or not len(scores) or
            not np.isfinite(scores).all() or not np.isin(masks, [0, 1]).all()):
        raise ValueError("Finite predicted scores and native binary candidate masks required")
    index = int(np.argmax(scores))  # First index resolves ties. Never sees labels.
    return masks[index].astype(bool), {"predicted_iou": scores.astype(float).tolist(),
        "selected_index": index, "selection": "highest_predicted_iou_first_index_on_tie"}


def segment_components(predictor, image, components):
    """Native HWC [0,1] image; one multimask call per prompted component.

    The official predictor maps native (x,y) to 1024 and restores logits to native
    shape before thresholding. No categorical resize, label clipping or oracle IoU.
    `predictor` may be a plain synthetic test double in CPU contract tests.
    """
    if (not isinstance(image, np.ndarray) or image.shape != (512, 512, 3) or
            image.dtype != np.float32 or not np.isfinite(image).all() or
            image.min() < 0 or image.max() > 1):
        raise ValueError("Native 512x512 HWC float32 RGB in [0,1] required")
    prepared = []
    ids = set()
    for component in components:
        xy = np.asarray(component["point_coords"], dtype=np.float32)
        labels = np.asarray(component["point_labels"])
        cid = component["component_id"]
        if (cid in ids or xy.ndim != 2 or xy.shape[1] != 2 or len(xy) != len(labels) or
                labels.ndim != 1 or not np.isin(labels, [0, 1]).all() or
                not (labels == 1).any() or not np.isfinite(xy).all() or
                (xy < 0).any() or (xy > 511).any()):
            raise ValueError("Unique components and finite native (x,y) prompts with a positive required")
        ids.add(cid)
        prepared.append((cid, xy, labels.astype(np.int32)))
    union, decisions = np.zeros((512, 512), bool), []
    if prepared:
        predictor.set_image(image)
        for cid, xy, labels in prepared:
            masks, predicted_iou, _ = predictor.predict(point_coords=xy, point_labels=labels,
                multimask_output=True, return_logits=False, normalize_coords=True)
            selected, decision = select_multimask(masks, predicted_iou, union.shape)
            union |= selected
            decisions.append(dict(decision, component_id=cid))
    return {"mask": union, "components": decisions,
            "output_hw": [512, 512], "coordinate_order": "x,y",
            "predict_calls": len(prepared), "image_embedding_calls": int(bool(prepared))}


def compare_predictions(first, second, tolerance):
    disagreement = int(np.count_nonzero(first["mask"] != second["mask"]))
    a, b = first["components"], second["components"]
    if disagreement > tolerance["mask_disagreement_pixels"] or len(a) != len(b):
        raise ValueError("Prediction repeat mask/component mismatch")
    for ca, cb in zip(a, b):
        if (ca["component_id"] != cb["component_id"] or ca["selected_index"] != cb["selected_index"] or
                not np.allclose(ca["predicted_iou"], cb["predicted_iou"],
                    atol=tolerance["scores_atol"], rtol=tolerance["scores_rtol"])):
            raise ValueError("Prediction repeat score/selection mismatch")
    return {"mask_disagreement_pixels": disagreement, "scores_within_tolerance": True,
            "tolerance": tolerance}
