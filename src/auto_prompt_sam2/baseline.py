"""Stage 3 orchestration: reuse stage-2 generations; development roles only."""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np

from .artifacts import ArtifactStore, canonical_json, digest, sha256_file
from .data import development_records, input_rgb, read_labels
from .metrics import aggregate, chip_metrics, pixel_areas_m2
from .preprocessing import Coordinates, load_preprocessing, sam_predictor_image
from .reference_prompts import simulate_points
from .sam_adapter import segment_components


REGIMES = ("official", "leave_paraguay_out")


def validate_config(config):
    """This stage is a fixed development run, not a search or final evaluation."""
    if (config["branch"] != "reference_assisted_not_deployable" or
            config["regimes"] != list(REGIMES) or config["roles"] != ["train", "val"] or
            config["validation_scope"] != "all" or config["train_smoke_chips"] != 1 or
            config["inference"] != {
                "precision": "float32", "autocast": False, "apply_postprocessing": False,
                "mask_threshold": 0.0, "multimask_output": True,
                "selection": "highest_predicted_iou_first_index_on_tie", "output_hw": [512, 512],
                "no_water": "empty_mask_no_sam_call_label_assisted_advantage"}):
        raise ValueError("Stage-3 development configuration contract changed")
    policy = config["prompts"]
    for key, expected in {
        "sampling": "seeded_permutation_greedy_without_replacement",
        "component_order": "first_pixel_row_major", "negative_pool": "all_valid_label_zero_pixels",
        "distance_scope": "all_points_within_each_component_call",
        "shortfall": "use_available_no_relaxation_no_component_drop"}.items():
        if policy[key] != expected:
            raise ValueError("Unsupported prompt policy: " + key)
    if (type(config["seed"]) is not int or config["seed"] < 0 or
            policy["connectivity"] not in (4, 8) or
            type(policy["positive_per_component"]) is not int or policy["positive_per_component"] < 1 or
            type(policy["negative_per_component"]) is not int or policy["negative_per_component"] < 0 or
            not np.isfinite(policy["minimum_distance_px"]) or policy["minimum_distance_px"] < 0):
        raise ValueError("Invalid seed or prompt geometry/budget")


def stage2_inputs(root, data_config):
    """Read completed generations only. No fitting, split rebuilding or acquisition."""
    root = Path(root)
    acquisition = json.loads(ArtifactStore(root / "raw").read(
        "acquisition", {"dataset": "Sen1Floods11-v1.1"})[0])
    identity = {"config_sha256": digest(data_config), "acquisition_sha256": digest(acquisition)}
    report = json.loads(ArtifactStore(root / "stage2").read("validation", identity)[0])
    result = {}
    for regime in REGIMES:
        prior = report["regimes"][regime]
        base = root / "regimes" / regime
        manifest = json.loads(ArtifactStore(base / "manifests").read("manifest",
            {"regime": regime, "manifest_sha256": prior["manifest_sha256"]},
            prior["manifest_generation"])[0])
        if manifest["regime"] != regime or digest(manifest) != prior["manifest_sha256"]:
            raise ValueError("Stage-2 manifest changed")
        params = load_preprocessing(base / "preprocessing", manifest, data_config,
                                    prior["parameter_generation"])
        if digest(params) != prior["preprocessing_sha256"]:
            raise ValueError("Stage-2 preprocessing changed")
        result[regime] = {"manifest": manifest, "params": params,
            "manifest_generation": prior["manifest_generation"],
            "parameter_generation": prior["parameter_generation"]}
    return result


def development_plan(manifest):
    regime = manifest["regime"]
    train = sorted(development_records(manifest, regime=regime, role="train"), key=lambda r: r["id"])
    val = sorted(development_records(manifest, regime=regime, role="val"), key=lambda r: r["id"])
    if not train or not val:
        raise ValueError("Both development roles required")
    return [train[0], *val]


def permitted_row(manifest, *, chip_id, role):
    rows = development_records(manifest, regime=manifest["regime"], role=role)
    matches = [r for r in rows if r["id"] == chip_id]
    if len(matches) != 1 or matches[0]["event"] == "Bolivia":
        raise ValueError("Chip is not a permitted unique development example")
    return matches[0]


def read_development_chip(root, manifest, *, chip_id, role):
    # Reject roles and membership BEFORE touching either image or labels.
    row = permitted_row(manifest, chip_id=chip_id, role=role)
    root = Path(root).resolve()
    for kind in ("image", "label"):
        path = (root / row[kind + "_path"]).resolve()
        if not path.is_relative_to(root) or sha256_file(path) != row[kind + "_sha256"]:
            raise ValueError("Development input path/hash mismatch")
    rgb, input_valid = input_rgb(root / row["image_path"])
    labels, _ = read_labels(root / row["label_path"])
    if rgb.shape != (3, 512, 512) or labels.shape != (512, 512):
        raise ValueError("Native geometry required")
    return row, rgb, input_valid, labels


def predict_development(predictor, root, inputs, row, config):
    manifest, params = inputs["manifest"], inputs["params"]
    row, rgb, valid, labels = read_development_chip(root, manifest, chip_id=row["id"], role=row["role"])
    prompts = simulate_points(labels, regime=manifest["regime"], chip_id=row["id"],
                              seed=config["seed"], policy=config["prompts"])
    image = sam_predictor_image(rgb, valid, params)
    # Only the image and the minimal spatial prompts cross the adapter boundary.
    components = [{k: c[k] for k in ("component_id", "point_coords", "point_labels")}
                  for c in prompts["components"]]
    result = segment_components(predictor, image, components)
    metric = chip_metrics(labels, result["mask"], chip_id=row["id"], event=row["event"],
                          pixel_area_m2=pixel_areas_m2(row["geometry"]))
    metadata = {"id": row["id"], "event": row["event"], "role": row["role"],
        "regime": manifest["regime"], "image_sha256": row["image_sha256"],
        "label_sha256": row["label_sha256"], "prompts": prompts,
        "inference": {k: v for k, v in result.items() if k != "mask"},
        "metrics": metric, "geometry": row["geometry"],
        "coordinates": Coordinates((512, 512), (1024, 1024)).metadata(),
        "preprocessing_sha256": digest(params), "config_sha256": digest(config)}
    return result, metadata


def encode_prediction(mask, metadata):
    if np.asarray(mask).dtype != np.bool_ or mask.shape != (512, 512):
        raise ValueError("Native boolean prediction required")
    buffer = io.BytesIO()
    np.savez_compressed(buffer, mask=mask,
                        metadata=np.frombuffer(canonical_json(metadata), dtype=np.uint8))
    return buffer.getvalue()


def decode_prediction(payload):
    with np.load(io.BytesIO(payload), allow_pickle=False) as saved:
        mask = saved["mask"].copy()
        metadata = json.loads(saved["metadata"].tobytes())
    if mask.dtype != np.bool_ or mask.shape != (512, 512):
        raise ValueError("Invalid saved native mask")
    return mask, metadata


def prediction_result(mask, metadata):
    return {"mask": mask, **metadata["inference"]}


def summaries(records):
    return {role: aggregate([r["metrics"] for r in records if r["role"] == role])
            for role in ("train", "val")}
