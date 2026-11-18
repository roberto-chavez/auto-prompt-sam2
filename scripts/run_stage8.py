#!/usr/bin/env python
"""Run the frozen, evaluation-only geographic transfer protocol in Colab GPU.

Reserved image/label reads happen only after the review gate and frozen protocol hash pass.
Raw rasters and masks remain in private Drive; the JSON report contains metrics and hashes only.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _find_stage7(root, regime, expected_zip_sha):
    from auto_prompt_sam2.artifacts import sha256_file
    base = Path(root) / "regimes" / regime / "stage7"
    matches = [p for p in base.glob("*/evidence.zip") if sha256_file(p) == expected_zip_sha]
    if len(matches) != 1:
        raise ValueError(f"{regime}: expected exactly one stage-7 evidence.zip matching the reviewed SHA")
    evidence = matches[0].parent / "evidence"
    required = ("selection.json", "evaluation_lock.json", "resolved_matrix.json", "inference_bundle.json",
                "validation.json", "recovery_validation.json")
    if any(not (evidence / name).is_file() for name in required):
        raise ValueError(f"{regime}: incomplete stage-7 evidence directory")
    values = {Path(name).stem: _json(evidence / name) for name in required}
    if (values["selection"]["selected_id"] != "a3_tuned" or
            values["validation"].get("reserved_access") or
            values["recovery_validation"].get("reserved_access") or
            values["validation"].get("failed") or values["recovery_validation"].get("failed")):
        raise ValueError(f"{regime}: selected row, failures or reserved-access record differs from review")
    if values["recovery_validation"].get("recovery") != "first_val_all_successful_rows_uncached_bitwise":
        raise ValueError(f"{regime}: offline recovery record is absent or incomplete")
    return matches[0].parent, values


def _reserved_records(manifest, regime, set_name, expected):
    roles = {"test_seen": "test", "bolivia_heldout": "bolivia", "paraguay_heldout": "paraguay_reserved"}
    if set_name not in roles:
        raise ValueError("Unknown preregistered reserved set")
    role = roles[set_name]
    rows = sorted((r for r in manifest["records"] if r["role"] == role), key=lambda r: r["id"])
    if len(rows) != expected or any(r["regime"] != regime for r in rows):
        raise ValueError(f"{regime}/{set_name}: exact frozen count/identity mismatch")
    if regime == "official" and set_name == "bolivia_heldout" and any(r["event"] != "Bolivia" for r in rows):
        raise ValueError("Bolivia heldout contains a non-Bolivia chip")
    if regime == "leave_paraguay_out" and set_name == "paraguay_heldout" and any(r["event"] != "Paraguay" for r in rows):
        raise ValueError("Paraguay heldout contains a non-Paraguay chip")
    if regime == "leave_paraguay_out" and set_name == "test_seen" and any(r["event"] == "Paraguay" for r in rows):
        raise ValueError("Paraguay leaked into test_seen")
    return rows


def main(root, review_path, frozen_sha256, resume=False):
    from auto_prompt_sam2.sam_adapter import require_colab_gpu
    require_colab_gpu()  # Fail before any reserved data, model import, or download.
    import numpy as np
    import torch
    from auto_prompt_sam2.artifacts import ArtifactStore, canonical_json, digest, sha256_file, utc_now
    from auto_prompt_sam2.baseline import stage2_inputs
    from auto_prompt_sam2.data import input_rgb, read_labels
    from auto_prompt_sam2.evaluation import (GateClosed, bootstrap_ci, protocol_sha256,
                                              require_gate)
    from auto_prompt_sam2.experiments import resolve_matrix, verify_pair
    from auto_prompt_sam2.metrics import aggregate, chip_metrics, pixel_areas_m2
    from auto_prompt_sam2.runtime import load_config, seed_everything, verify_installed_pins
    from auto_prompt_sam2.sam_adapter import load_frozen_sam, assert_frozen, state_fingerprint
    from auto_prompt_sam2.features import load_frozen_dino
    from auto_prompt_sam2.training import deserialize
    from run_stage6 import read_dependency, verify_bundle
    from auto_prompt_sam2.stage7 import stage6_inputs
    import auto_prompt_sam2.stage7 as stage7_module

    root = Path(root).resolve()
    drive = Path("/content/drive").resolve()
    if not root.is_relative_to(drive):
        raise ValueError("Persistent Google Drive root required")
    os.environ["AUTO_PROMPT_PERSIST_ROOT"] = str(root)

    # Enforce human review, explicit authorization, empty previous reserved access,
    # and the exact protocol freeze BEFORE reading the reserved image/label files.
    protocol = _json(REPO / "configs/stage8_protocol.json")
    review = _json(review_path)
    require_gate(review, protocol, frozen_sha256)
    actual_sha = protocol_sha256(protocol)
    if actual_sha != frozen_sha256:
        raise ValueError("Frozen Stage-8 protocol hash mismatch")
    if not str(protocol.get("status", "")).startswith("FROZEN_BEFORE_RESERVED_ACCESS"):
        raise ValueError("Stage-8 protocol has not been marked frozen")

    env = load_config(REPO)
    verify_installed_pins(REPO, env, linux_lock=True)
    deps6 = _json(REPO / "configs/stage6_dependencies.json")
    deps7 = _json(REPO / "configs/stage7_dependencies.json")
    data_config = _json(REPO / "configs/data.yaml")
    inputs = stage2_inputs(root, data_config)  # Reads manifests/preprocessing only after gate.
    stage7 = {}
    for regime in protocol["regimes"]:
        stage7[regime] = _find_stage7(root, regime, review["regimes"][regime]["zip_sha256"])
    selections = {r: stage7[r][1]["selection"] for r in stage7}
    locks = {r: stage7[r][1]["evaluation_lock"] for r in stage7}
    verify_pair(selections, locks)
    if any(s["selected_id"] != "a3_tuned" for s in selections.values()):
        raise ValueError("Stage-7 selected checkpoint differs from the reviewed lock")

    # Stage 8 extracts fresh DINO features for every reserved chip. It never calls
    # ValidationEngine.reuse() and does not consume Stage-5 cached feature payloads.
    # Verify the pinned Stage-5 report and completed state, but do not read/hash every
    # unused cache object through DriveFS (which can stall on large remote folders).
    stage5_store, stage5_report, _ = read_dependency(
        root, "stage5", deps6["stage5"], verify_artifacts=False)
    model_config = env["models"]
    outputs = {}
    stage7_matrix = _json(REPO / "configs/ablations.yaml")
    pair_sha = digest({"selections": selections, "locks": locks})
    run_identity = {"stage": "phase1_stage8", "protocol_sha256": actual_sha,
                    "stage7_pair_sha256": pair_sha, "source_commit": os.environ.get("CODE_COMMIT", "unknown")}
    run_dir = root / "stage8" / digest(run_identity)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_identity.json").write_bytes(canonical_json(run_identity))
    if not resume and any(run_dir.glob("predictions/**/*.npz")):
        raise ValueError("Stage-8 predictions already exist; use --resume with the same identity")

    for regime, spec in protocol["regimes"].items():
        evidence_dir, ev = stage7[regime]
        bundle = ev["inference_bundle"]
        resolved = ev["resolved_matrix"]
        if digest(bundle) != selections[regime]["lineage"]["bundle_sha256"]:
            raise ValueError(f"{regime}: inference bundle differs from the selection lock")
        regenerated = resolve_matrix(stage7_matrix, bundle, selections[regime]["lineage"])
        if resolved != regenerated or resolved != _json(evidence_dir / "evidence/resolved_matrix.json"):
            raise ValueError(f"{regime}: stage-7 matrix/config lineage differs from its reviewed lock")
        row_configs = {r["id"]: copy.deepcopy(r["config"]) for r in resolved["rows"]
                       if r["id"] in protocol["rows"]}
        if set(row_configs) != set(protocol["rows"]):
            raise ValueError(f"{regime}: frozen protocol rows not present in resolved stage-7 matrix")
        inp = inputs[regime]
        stage6_spec = deps7["stage6"][regime]
        verify_bundle(bundle, regime, inp, stage6_spec["identity"]["ids"], model_config)
        sixth = stage6_inputs(root, stage6_spec, inp, bundle, REPO)
        sixth["best_epoch"] = stage6_spec["best_epoch"]

        torch.set_num_threads(1)
        seed_everything(protocol["uncertainty"]["seed"])
        predictor, sam_info = load_frozen_sam(model_config["sam2"], root, offline=True,
            expected_sha256=bundle["base_checkpoint_sha256"]["sam"])
        model_before = state_fingerprint(predictor.model)
        decoder_base = {k: v.detach().cpu().clone() for k, v in predictor.model.sam_mask_decoder.state_dict().items()}
        decoder_tuned = deserialize(sixth["payloads"]["decoder_export"])
        dino, dino_info = load_frozen_dino(model_config["dinov3"], root, offline=True,
            expected_sha256=bundle["base_checkpoint_sha256"]["dino"])
        dino_before = dino_info["state_sha256"]
        baseline_policy = _json(REPO / "configs/baseline.yaml")["prompts"]
        engine = stage7_module.ValidationEngine(root, regime, inp, bundle, stage5_store,
            stage5_report, sixth, predictor, dino, baseline_policy)

        records_by_id, allowed_sets = {}, {}
        for set_name, count in spec["sets"].items():
            rows = _reserved_records(inp["manifest"], regime, set_name, count)
            allowed_sets[set_name] = rows
            records_by_id.update({r["id"]: r for r in rows})
        if len(records_by_id) != sum(spec["sets"].values()):
            raise ValueError(f"{regime}: duplicate ID across frozen reserved sets")

        # Stage-7's image/prompt path is reused byte-for-byte. Its reader is temporarily
        # routed to only the exact reserved IDs above; no development or arbitrary path.
        old_read_image, old_labels = stage7_module.read_image, stage7_module.supervised_labels
        label_reads = []
        def reserved_image(data_root, manifest, chip_id, role):
            if role != "val" or chip_id not in records_by_id:
                raise ValueError("Stage-8 image read outside its frozen reserved IDs")
            row = records_by_id[chip_id]
            p = (root / row["image_path"]).resolve()
            if not p.is_relative_to(root) or sha256_file(p) != row["image_sha256"]:
                raise ValueError("Reserved image path/hash mismatch")
            return input_rgb(p)
        def assisted_labels(data_root, manifest, chip_id, role):
            if role != "val" or chip_id not in records_by_id:
                raise ValueError("Stage-8 label read outside its frozen reserved IDs")
            row = records_by_id[chip_id]
            p = (root / row["label_path"]).resolve()
            if not p.is_relative_to(root) or sha256_file(p) != row["label_sha256"]:
                raise ValueError("Reserved label path/hash mismatch")
            label_reads.append({"id": chip_id, "purpose": "assisted_prompt_generation"})
            return read_labels(p)[0]

        stage7_module.read_image = reserved_image
        stage7_module.supervised_labels = assisted_labels
        try:
            regime_result = {"sets": {}, "model_identity": {"sam": sam_info, "dino": dino_info},
                             "label_reads_for_assisted_prompts": label_reads}
            for set_name, rows in allowed_sets.items():
                set_result = {"chips": len(rows), "rows": {}}
                for row_id in protocol["rows"]:
                    config = row_configs[row_id]
                    is_tuned = row_id == "a3_tuned"
                    predictor.model.sam_mask_decoder.load_state_dict(decoder_tuned if is_tuned else decoder_base, strict=True)
                    intended = state_fingerprint(predictor.model.sam_mask_decoder)
                    metrics_rows = []
                    prompt_hashes = {}
                    pred_dir = run_dir / "predictions" / regime / set_name / row_id
                    pred_dir.mkdir(parents=True, exist_ok=True)
                    for index, chip in enumerate(rows):
                        target = pred_dir / (chip["id"] + ".npz")
                        meta = None
                        if resume and target.is_file():
                            with np.load(target, allow_pickle=False) as z:
                                mask = z["mask"].copy()
                                meta = json.loads(z["metadata"].tobytes())
                            if (meta["protocol_sha256"] != actual_sha or meta["stage7_pair_sha256"] != pair_sha or
                                    meta["config_sha256"] != resolved_row_hash(resolved, row_id) or
                                    meta["id"] != chip["id"] or mask.dtype != np.bool_ or mask.shape != (512, 512) or
                                    hashlib.sha256(mask.tobytes()).hexdigest() != meta.get("prediction_mask_sha256")):
                                raise ValueError("Cached stage-8 prediction identity mismatch")
                        else:
                            arrays, meta = engine.run_pipeline(config, chip, cached_features=None)
                            mask = arrays["mask"]
                            if mask.dtype != np.bool_ or mask.shape != (512, 512):
                                raise ValueError("Prediction mask geometry/type mismatch")
                            meta.update({"id": chip["id"], "event": chip["event"], "role": set_name,
                                "regime": regime, "protocol_sha256": actual_sha, "stage7_pair_sha256": pair_sha,
                                "config_sha256": resolved_row_hash(resolved, row_id),
                                "stage7_source_role_for_engine": "val"})
                            meta["prediction_mask_sha256"] = hashlib.sha256(mask.tobytes()).hexdigest()
                            buffer = __import__("io").BytesIO()
                            np.savez_compressed(buffer, mask=mask, metadata=canonical_json(meta))
                            temp = target.with_suffix(".pending")
                            temp.write_bytes(buffer.getvalue()); temp.replace(target)
                        # For automatic rows, prediction is complete before ground-truth labels are read.
                        label_path = (root / chip["label_path"]).resolve()
                        if not label_path.is_relative_to(root) or sha256_file(label_path) != chip["label_sha256"]:
                            raise ValueError("Reserved label path/hash mismatch")
                        labels = read_labels(label_path)[0]
                        metric = chip_metrics(labels, mask, chip_id=chip["id"], event=chip["event"],
                                              pixel_area_m2=pixel_areas_m2(chip["geometry"]))
                        metric["prediction_sha256"] = hashlib.sha256(mask.tobytes()).hexdigest()
                        metrics_rows.append(metric)
                        prompt_hashes[chip["id"]] = digest(meta.get("prompts", {}))
                        if index % 10 == 0:
                            print(regime, set_name, row_id, index + 1, "/", len(rows), flush=True)
                    if state_fingerprint(predictor.model.sam_mask_decoder) != intended:
                        raise ValueError("Decoder state changed during evaluation")
                    set_result["rows"][row_id] = {"metrics": aggregate(metrics_rows),
                        "chip_metrics": metrics_rows, "prompt_hashes": prompt_hashes,
                        "prediction_files": [str((pred_dir / (r["id"] + ".npz")).relative_to(run_dir)) for r in rows]}
                regime_result["sets"][set_name] = set_result
            if label_reads and set({x["id"] for x in label_reads}) != set(records_by_id):
                raise ValueError("Assisted prompt-label audit does not cover exactly the frozen reserved IDs")
            # Compute preregistered paired comparisons and descriptive seen/heldout shifts.
            from auto_prompt_sam2.evaluation import bootstrap_ci
            for set_name, set_result in regime_result["sets"].items():
                comps = {}
                for c in protocol["comparisons"]:
                    if c["type"] != "paired_same_chips": continue
                    a = set_result["rows"][c["a"]]["chip_metrics"]
                    b = set_result["rows"][c["b"]]["chip_metrics"]
                    comps[c["id"]] = {
                        "difference_pooled_water_iou": set_result["rows"][c["a"]]["metrics"]["pooled"]["values"]["water_iou"] - set_result["rows"][c["b"]]["metrics"]["pooled"]["values"]["water_iou"],
                        "bootstrap_ci": bootstrap_ci(a, b, **{k: protocol["uncertainty"][k] for k in ("resamples", "seed", "level")})}
                set_result["comparisons"] = comps
            heldout = next(s for s in regime_result["sets"] if s != "test_seen")
            regime_result["geographic_shift_descriptive"] = {
                row_id: {"seen": regime_result["sets"]["test_seen"]["rows"][row_id]["metrics"]["pooled"]["values"]["water_iou"],
                         "heldout": regime_result["sets"][heldout]["rows"][row_id]["metrics"]["pooled"]["values"]["water_iou"],
                         "difference_seen_minus_heldout": regime_result["sets"]["test_seen"]["rows"][row_id]["metrics"]["pooled"]["values"]["water_iou"] - regime_result["sets"][heldout]["rows"][row_id]["metrics"]["pooled"]["values"]["water_iou"]}
                for row_id in protocol["rows"]}
            regime_result["reserved_access"] = sorted(records_by_id)
            regime_result["access_log"] = {"labels_used_for_assisted_prompts": sorted({x["id"] for x in label_reads}),
                "automatic_rows_read_labels_for_inference": False}
            outputs[regime] = regime_result
            assert_frozen(dino); assert_frozen(predictor.model)
            predictor.model.sam_mask_decoder.load_state_dict(decoder_base, strict=True)
            if state_fingerprint(predictor.model) != model_before or state_fingerprint(dino) != dino_before:
                raise ValueError("Base model state changed during Stage 8")
            del engine, predictor, dino
            torch.cuda.empty_cache()
        finally:
            stage7_module.read_image = old_read_image
            stage7_module.supervised_labels = old_labels

    report = {"schema_version": 1, "stage": "phase1_stage8", "created_at": utc_now(),
              "status": "complete", "protocol": protocol, "protocol_sha256": actual_sha,
              "stage7_pair_sha256": pair_sha, "run_identity": run_identity,
              "human_review_receipt_sha256": review["pair"].get("human_review_receipt_sha256"),
              "reserved_access_by_regime": {r: v["reserved_access"] for r, v in outputs.items()},
              "results": outputs,
              "limitations": ["single seed", "no retuning on reserved sets", "assisted row uses labels by design and is not deployable"]}
    (run_dir / "stage8_report.json").write_bytes(canonical_json(report))
    print(json.dumps({"stage8_report": str(run_dir / 'stage8_report.json'),
        "protocol_sha256": actual_sha, "run_identity": run_identity,
        "result_counts": {r: {s: v["chips"] for s, v in x["sets"].items()} for r, x in outputs.items()}}, indent=2))


def resolved_row_hash(resolved, row_id):
    return next(r["config_sha256"] for r in resolved["rows"] if r["id"] == row_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--frozen-protocol-sha256", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    main(args.root, args.review, args.frozen_protocol_sha256, args.resume)
