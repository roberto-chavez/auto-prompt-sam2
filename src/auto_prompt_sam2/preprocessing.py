"""Train-only radiometry and reversible geometry; image path never sees labels."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode

from .artifacts import ArtifactStore, canonical_json, digest
from .data import development_records, input_rgb


def histogram_percentile(hist, percentile):
    """Exact np.percentile(method='linear') on histogram bin indices."""
    count = int(hist.sum())
    if count == 0 or not 0 < percentile <= 100:
        raise ValueError("Empty population or invalid percentile")
    rank = (count - 1) * percentile / 100
    cumulative = np.cumsum(hist)
    lo = np.searchsorted(cumulative, np.floor(rank), side="right")
    hi = np.searchsorted(cumulative, np.ceil(rank), side="right")
    return float(lo + (hi - lo) * (rank - np.floor(rank)))


def fit_preprocessing(root, manifest, config):
    regime = manifest["regime"]
    rows = development_records(manifest, regime=regime, role="train")
    # Covers int16 v1.1 rasters and the uint16 type stated in the README.
    offset = 32768
    hist = np.zeros(98304, dtype=np.int64)
    for row in rows:
        rgb, valid = input_rgb(Path(root) / row["image_path"])
        hist += np.bincount(rgb[:, valid].astype(np.int32).ravel() + offset, minlength=len(hist))
    upper_dn = histogram_percentile(hist, config["clip_percentile"]) - offset
    if upper_dn <= 0:
        raise ValueError("Nonpositive train clipping limit")
    params = {"schema_version": 1, "regime": regime, "config": config,
              "manifest_sha256": digest(manifest), "source_role": "train",
              "source_ids": sorted(r["id"] for r in rows),
              "source_image_hashes": {r["id"]: r["image_sha256"] for r in rows},
              "valid_rgb_values": int(hist.sum()), "upper_dn": upper_dn,
              "upper_reflectance": upper_dn / config["reflectance_divisor"],
              "fit_validity": "input_rgb_masks_and_finite_only_no_labels",
              "histogram_sha256": digest(hist.tolist())}
    return params


def validate_params(params, manifest):
    rows = development_records(manifest, regime=params["regime"], role="train")
    if (params["manifest_sha256"] != digest(manifest) or params["source_role"] != "train" or
            params["source_ids"] != sorted(r["id"] for r in rows) or
            params["source_image_hashes"] != {r["id"]: r["image_sha256"] for r in rows}):
        raise ValueError("Preprocessing source differs from permitted train")
    if params["upper_dn"] <= 0 or not np.isfinite(params["upper_dn"]):
        raise ValueError("Invalid clipping limit")


def save_preprocessing(folder, params):
    identity = {"regime": params["regime"], "manifest_sha256": params["manifest_sha256"],
                "config_sha256": digest(params["config"])}
    return ArtifactStore(Path(folder)).publish("parameters", canonical_json(params), identity)


def load_preprocessing(folder, manifest, config, generation=None):
    identity = {"regime": manifest["regime"], "manifest_sha256": digest(manifest),
                "config_sha256": digest(config)}
    params = json.loads(ArtifactStore(Path(folder)).read("parameters", identity, generation)[0])
    if params["config"] != config:
        raise ValueError("Saved preprocessing config mismatch")
    validate_params(params, manifest)
    return params


@dataclass(frozen=True)
class Coordinates:
    """(x,y) coordinates; SAM uses x*W'/W, sampling uses pixel centers.

    Native point (x,y) maps to a patch-center index via (x+.5)/patch-.5.
    Patch k covers original pixel centers [16*k,16*k+15] at native 512.
    """
    original_hw: tuple[int, int]
    target_hw: tuple[int, int]
    patch_size: int = 16

    def points(self, xy, *, inverse=False, pixel_centers=False):
        a = np.asarray(xy, dtype=np.float64)
        if a.shape[-1] != 2 or not np.isfinite(a).all():
            raise ValueError("Finite (x,y) coordinates required")
        oh, ow = self.original_hw
        th, tw = self.target_hw
        scale = np.array([tw / ow, th / oh])
        offset = 0.5 if pixel_centers else 0.0
        return (a + offset) / scale - offset if inverse else (a + offset) * scale - offset

    def to_patch(self, xy):
        return (self.points(xy, pixel_centers=True) + 0.5) / self.patch_size - 0.5

    def from_patch(self, xy):
        centers = (np.asarray(xy, dtype=np.float64) + 0.5) * self.patch_size - 0.5
        return self.points(centers, inverse=True, pixel_centers=True)

    def metadata(self):
        oh, ow = self.original_hw
        th, tw = self.target_hw
        return {"original_hw": list(self.original_hw), "target_hw": list(self.target_hw),
                "padding_ltrb": [0, 0, 0, 0], "sam_xy_scale": [tw / ow, th / oh],
                "sam_xy_offset": [0, 0], "sampling_xy_offset": [0.5 * tw / ow - 0.5, 0.5 * th / oh - 0.5],
                "coordinate_order": "x,y", "patch_size": self.patch_size,
                "patch_grid_hw": [th // self.patch_size, tw // self.patch_size],
                "patch_centers": "(index+0.5)*patch_size-0.5", "inverse": "analytic"}


def rgb_unit(rgb, input_valid, params):
    """Returns float32 CHW [0,1]. No labels or label-validity parameter."""
    if not isinstance(rgb, np.ndarray) or rgb.dtype not in (np.dtype('uint16'), np.dtype('int16')) or rgb.ndim != 3 or rgb.shape[0] != 3:
        raise ValueError("Raw int16/uint16 RGB CHW required; normalized tensors cannot enter twice")
    if input_valid.shape != rgb.shape[1:] or input_valid.dtype != np.bool_:
        raise ValueError("Boolean input validity required")
    divisor = params["config"]["reflectance_divisor"]
    reflectance = rgb.astype(np.float32) / divisor
    result = np.clip(reflectance, 0, params["upper_reflectance"]) / params["upper_reflectance"]
    result[:, ~input_valid] = params["config"]["invalid_rgb_fill_before_normalize"]
    return result.astype(np.float32)


def prepare(rgb, input_valid, params, *, model):
    if model not in ("sam", "dino"):
        raise ValueError("Unknown preprocessing branch")
    config = params["config"][model]
    unit = rgb_unit(rgb, input_valid, params)
    tensor = resize(torch.from_numpy(unit), [config["size"], config["size"]],
                    interpolation=InterpolationMode.BILINEAR, antialias=True)
    mean = torch.tensor(config["mean"], dtype=torch.float32)[:, None, None]
    std = torch.tensor(config["std"], dtype=torch.float32)[:, None, None]
    tensor = (tensor - mean) / std
    valid = resize_categorical(input_valid, (config["size"], config["size"]))
    coords = Coordinates(tuple(rgb.shape[1:]), (config["size"], config["size"]))
    return {"pixel_values": tensor, "input_valid": valid, "coordinates": coords.metadata(),
            "preprocessing_sha256": digest(params), "normalization_applied": True,
            "integration": "already_normalized_direct_model_tensor_skip_official_processor"}


def sam_predictor_image(rgb, input_valid, params):
    """Float32 HWC [0,1] for SAM2ImagePredictor.set_image: it normalizes once.

    Do not send prepare(..., model='sam')['pixel_values'] to set_image.
    The official ToTensor scales only uint8; float32 retains [0,1].
    """
    return np.moveaxis(rgb_unit(rgb, input_valid, params), 0, -1).copy()


def resize_categorical(mask, target_hw):
    if mask.ndim != 2:
        raise ValueError("2D categorical mask required")
    # PyTorch nearest matches the documented categorical path; preserves -1.
    value = F.interpolate(torch.from_numpy(mask.astype(np.float32))[None, None],
                          size=target_hw, mode="nearest")[0, 0].numpy()
    return value.astype(mask.dtype)
