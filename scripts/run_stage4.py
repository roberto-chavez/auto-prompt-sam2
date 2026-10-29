"""User-run Colab GPU dissection; no training, automatic prompts or reserved data."""
from __future__ import annotations

import argparse
import csv
import gc
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from auto_prompt_sam2.artifacts import ArtifactStore, Run, canonical_json, digest, sha256_file, utc_now
from auto_prompt_sam2.sam_adapter import require_colab_gpu


def source_paths():
    from auto_prompt_sam2.runtime import source_files
    return sorted(set(source_files(REPO)) | {REPO / p for p in (
        "configs/data.yaml", "configs/baseline.yaml", "configs/profiling.yaml",
        "requirements-stage2.txt", "notebooks/04_diseccion_modelos.ipynb",
        "docs/project/validacion_etapa4.md")})


def source_snapshot():
    from auto_prompt_sam2.runtime import provenance
    paths = source_paths()
    info = provenance(REPO)
    info["source_sha256"] = {p.relative_to(REPO).as_posix(): sha256_file(p) for p in paths}
    info["source_tree_sha256"] = digest(info["source_sha256"])
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for p in paths:
            archive.write(p, p.relative_to(REPO).as_posix())
    return info, buffer.getvalue()


def export_evidence(root, run):
    output = root / "stage4/evidence" / run.run_id
    output.mkdir(parents=True, exist_ok=True)
    for name in ("validation", "recovery_validation", "config", "environment", "failure"):
        try:
            (output / (name + ".json")).write_bytes(run.store.read(name, run.identity)[0])
        except FileNotFoundError:
            pass
    # Friendly names for inspecting tables/figures; canonical generations remain authoritative.
    try:
        validation = json.loads(run.store.read("validation", run.identity)[0])
        for name, item in validation["artifacts"].items():
            if item["extension"] in ("png", "csv", "json"):
                payload, _ = run.store.read(name, run.identity, item["generation"])
                (output / (name + "." + item["extension"])).write_bytes(payload)
    except FileNotFoundError:
        pass
    target = output.with_suffix(".zip")
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for folder in (run.store.root, output):
            for p in sorted(folder.rglob("*")):
                if p.is_file():
                    archive.write(p, p.relative_to(root).as_posix())
    print(json.dumps({"evidence_zip": str(target), "sha256": sha256_file(target)}, indent=2), flush=True)


def persist(run, name, payload, index, extension):
    generation = run.store.publish(name, payload, run.identity)
    restored, marker = run.store.read(name, run.identity, generation)
    if restored != payload:
        raise ValueError("Persistent roundtrip failed")
    index[name] = {"generation": generation, "sha256": marker["sha256"], "extension": extension}


def csv_payload(rows):
    output = io.StringIO(newline="")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def inspect_chip(kind, model, predictor, tensor, rgb, valid, params, config):
    import numpy as np
    import torch
    from auto_prompt_sam2.dissection import sam_static_path
    from auto_prompt_sam2.features import capture_features, token_layout, tokens_to_grid, effective_receptive_field
    from auto_prompt_sam2.preprocessing import sam_predictor_image

    metadata, grids, arrays = {}, {}, {}
    if kind == "sam":
        count = len(model.image_encoder.trunk.blocks)
        names = ["image_encoder.trunk.blocks." + str(i) for i in sorted({0, count // 2, count - 1})]
        with torch.no_grad(), capture_features(model, names) as captured:
            direct_logits, direct_scores = sam_static_path(model, tensor, config["fixed_prompts"])
        for name, value in captured.items():
            # Pinned Hiera block outputs BHWC; preserve raw layout and explicit BCHW view.
            if value.ndim != 4:
                raise ValueError("Expected Hiera BHWC feature")
            grids[name] = value.transpose(0, 3, 1, 2)
            arrays[name] = value
            metadata[name] = {"raw_shape": list(value.shape), "raw_layout": "BHWC",
                "visual_grid_shape": list(grids[name].shape), "transformation": "transpose(0,3,1,2)",
                "tokens": "no CLS/register tokens; hierarchical spatial grid"}
        predictor.set_image(sam_predictor_image(rgb, valid, params))
        logits, scores, _ = predictor.predict(
            point_coords=np.asarray(config["fixed_prompts"]["point_coords"], np.float32),
            point_labels=np.asarray(config["fixed_prompts"]["point_labels"], np.int32),
            multimask_output=True, return_logits=True, normalize_coords=True)
        tol = config["repeat_tolerance"]
        if (not np.allclose(logits, direct_logits[0].cpu().numpy(), **tol) or
                not np.allclose(scores, direct_scores[0].cpu().numpy(), **tol)):
            raise ValueError("Differentiable SAM tensor path differs from reused official predictor")
        predictor.reset_predictor()
        del direct_logits, direct_scores
        metadata["sam_predictor_equivalence"] = {"passed": True, "tolerance": tol}
        erf, erf_meta = effective_receptive_field(lambda x: model.forward_image(x)["vision_features"], tensor)
    else:
        layout = token_layout((512, 512), 16, model.config.num_register_tokens, model.config.hidden_size)
        names = ["layer." + str(i) for i in config["dino_feature_layers"]]
        with torch.no_grad(), capture_features(model, names) as captured:
            output = model(pixel_values=tensor).last_hidden_state
        for name, value in captured.items():
            grids[name] = tokens_to_grid(value, layout)
            arrays[name] = value
            metadata[name] = {"raw_shape": list(value.shape), "raw_layout": "BNC_pre_final_norm",
                "visual_grid_shape": list(grids[name].shape), "token_layout": layout}
        arrays["last_hidden_state"] = output.detach().cpu().numpy()
        del output
        metadata["last_hidden_state"] = {"raw_shape": list(arrays["last_hidden_state"].shape),
                                         "token_layout": layout, "normalization": "model_final_LayerNorm"}
        erf, erf_meta = effective_receptive_field(
            lambda x: tokens_to_grid(model(pixel_values=x).last_hidden_state, layout), tensor)
    arrays["erf"] = erf
    metadata["erf"] = dict(erf_meta, method=config["erf"],
        limitation="single output channel/location and chip; gradient magnitude is local sensitivity, not theoretical RF or causal proof")
    return arrays, grids, metadata


def run_stage4(root, mode, models=("sam", "dino")):
    require_colab_gpu()
    if not models or any(m not in ("sam", "dino") for m in models):
        raise ValueError("models must be a non-empty subset of sam,dino")
    import numpy as np
    import torch
    from auto_prompt_sam2.baseline import stage2_inputs
    from auto_prompt_sam2.dissection import train_plan, read_train_image, sam_static_path, sam_points, arrays_payload, read_arrays, figure_png
    from auto_prompt_sam2.features import load_frozen_dino, hook_inventory
    from auto_prompt_sam2.preprocessing import prepare, rgb_unit
    from auto_prompt_sam2.profiling import validate_config, block_names, parameter_table, count_flops, benchmark, benchmark_blocks
    from auto_prompt_sam2.runtime import load_config, persistent_root, seed_everything, inventory, verify_installed_pins
    from auto_prompt_sam2.sam_adapter import load_frozen_sam, assert_frozen, state_fingerprint

    os.environ["AUTO_PROMPT_PERSIST_ROOT"] = str(root)
    environment_config = load_config(REPO)
    root = persistent_root(environment_config, colab=True)
    config = json.loads((REPO / "configs/profiling.yaml").read_text("utf-8"))
    data_config = json.loads((REPO / "configs/data.yaml").read_text("utf-8"))
    validate_config(config)
    if config["seed"] != environment_config["seed"] or environment_config["precision"] != config["precision"]:
        raise ValueError("Environment seed/precision mismatch")
    if data_config["sam"]["size"] != 1024 or data_config["dino"]["size"] != 512:
        raise ValueError("Stage-2 geometry mismatch")
    seed_everything(config["seed"])
    torch.set_num_threads(1)
    pins = verify_installed_pins(REPO, environment_config, linux_lock=True)
    from importlib.metadata import version
    for line in (REPO / "requirements-stage2.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            name, expected = line.split("==")
            if version(name) != expected:
                raise ValueError("Stage-2 package pin mismatch: " + name)
    inputs = stage2_inputs(root, data_config)
    source, snapshot = source_snapshot()
    plans = {r: train_plan(v["manifest"], config["chips_per_regime"]) for r, v in inputs.items()}
    identity = {"stage": "phase1_stage4", "config_sha256": digest(config),
        "source_tree_sha256": source["source_tree_sha256"], "models": environment_config["models"],
        "models_run": list(models),
        "regimes": {r: {"manifest_sha256": digest(v["manifest"]), "preprocessing_sha256": digest(v["params"]),
            "manifest_generation": v["manifest_generation"], "parameter_generation": v["parameter_generation"],
            "train_ids": [row["id"] for row in plans[r]]} for r, v in inputs.items()}}
    active = ArtifactStore(root / "stage4")
    if mode == "prepare":
        run = Run.create(root / "stage4", identity)
        active.publish("active_run", canonical_json({"run_id": run.run_id}), identity)
        run.transition("running")
        run.store.publish("source_snapshot", snapshot, identity)
        run.store.publish("config", canonical_json({"profiling": config, "environment": environment_config,
            "data": data_config, "source": source, "identity": identity}), identity)
        prior = None
    else:
        run = Run(root / "stage4", json.loads(active.read("active_run", identity)[0])["run_id"], identity)
        if run.state != "complete":
            raise ValueError("Recover requires a complete run; interrupted runs remain preserved, use prepare for a new run")
        prior = json.loads(run.store.read("validation", identity)[0])
    report = {"stage": "phase1_stage4", "run_id": run.run_id, "mode": mode, "created_at": utc_now(),
        "identity": identity, "pins": pins, "models": {}, "regimes": {}, "artifacts": {},
        "access_log": [], "reserved_access": [], "training_executed": False,
        "automatic_prompts_generated": False, "final_evaluations": [],
        "closure": "pending_external_review_of_user_notebook_json_tables_and_figures",
        "license": "publisher_reply_pending_historical_salvedad_retained",
        "runtime_boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    try:
        tests = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                               cwd=REPO, text=True, capture_output=True)
        report["tests"] = {"exit_code": tests.returncode, "stdout": tests.stdout, "stderr": tests.stderr}
        print(tests.stdout, tests.stderr, flush=True)
        tests.check_returncode()
        pip = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
        pip.check_returncode()
        report["pip_check"] = pip.stdout
        report["environment"] = inventory(root, REPO)
        run.store.publish("environment", canonical_json(report["environment"]), identity)
        if prior:
            for name, item in prior["artifacts"].items():
                _, marker = run.store.read(name, identity, item["generation"])
                if marker["sha256"] != item["sha256"]:
                    raise ValueError("Recovery artifact hash mismatch")
            report["artifacts"] = prior["artifacts"]
        for kind, model_key in (("sam", "sam2"), ("dino", "dinov3")):
            if kind not in models:
                continue  # E2 teardown prioritises SAM; DINOv3 (gated) must not block it.
            expected = prior["models"][kind]["checkpoint_sha256"] if prior else None
            if kind == "sam":
                predictor, model_info = load_frozen_sam(environment_config["models"][model_key], root,
                    offline=bool(prior), expected_sha256=expected)
                model = predictor.model
            else:
                predictor = None
                model, model_info = load_frozen_dino(environment_config["models"][model_key], root,
                    offline=bool(prior), expected_sha256=expected)
            before = hook_inventory(model)
            if prior and model_info["state_sha256"] != prior["models"][kind]["state_sha256"]:
                raise ValueError("Offline reloaded state differs")
            blocks = block_names(model, kind)
            model_info["parameters_by_block"] = parameter_table(model, blocks)
            report["models"][kind] = model_info
            if not prior:
                persist(run, kind + "_parameters", csv_payload([
                    {k: v for k, v in row.items() if k != "parameter_names"}
                    for row in model_info["parameters_by_block"]["rows"]]), report["artifacts"], "csv")
            for regime, value in inputs.items():
                result = {"chips": {}, "inputs": identity["regimes"][regime]}
                report["regimes"].setdefault(regime, {})[kind] = result
                for i, row in enumerate(plans[regime]):
                    _, rgb, valid = read_train_image(root, value["manifest"], row["id"])
                    report["access_log"].append({"regime": regime, "model": kind, "id": row["id"],
                        "role": "train", "event": row["event"], "image_sha256": row["image_sha256"]})
                    prepared = prepare(rgb, valid, value["params"], model=kind)
                    tensor = prepared["pixel_values"][None].cuda()
                    prefix = f"{kind}_{regime}_{i}"
                    arrays, grids, metadata = inspect_chip(kind, model, predictor, tensor, rgb, valid, value["params"], config)
                    chip = {"id": row["id"], "event": row["event"], "role": "train",
                        "image_sha256": row["image_sha256"], "preprocessing_sha256": prepared["preprocessing_sha256"],
                        "coordinates": prepared["coordinates"], "input_shape": list(tensor.shape), "features": metadata}
                    result["chips"][row["id"]] = chip
                    if prior:
                        old = read_arrays(run.store.read(prefix + "_arrays", identity,
                            prior["artifacts"][prefix + "_arrays"]["generation"])[0])
                        if set(arrays) != set(old) or any(not np.allclose(arrays[k], old[k],
                                **config["repeat_tolerance"]) for k in arrays):
                            raise ValueError("Recovered feature/ERF values differ at predefined tolerance")
                        chip["recovery_features_and_erf_within_tolerance"] = True
                    else:
                        persist(run, prefix + "_arrays", arrays_payload(arrays), report["artifacts"], "npz")
                        persist(run, prefix + "_figure", figure_png(rgb_unit(rgb, valid, value["params"]),
                            grids, arrays["erf"], config["feature_channels"]), report["artifacts"], "png")
                        persist(run, prefix + "_features", canonical_json(chip), report["artifacts"], "json")
                    del arrays, grids
                    if i == 0 and not prior:
                        points = sam_points(tensor, config["fixed_prompts"]) if kind == "sam" else None
                        forward = (lambda: sam_static_path(model, tensor, config["fixed_prompts"], preloaded_points=points)) if kind == "sam" else (
                            lambda: model(pixel_values=tensor))
                        result["flops"] = count_flops(model, blocks, forward)
                        result["full_path_timing"] = benchmark(forward, config)
                        result["block_timings"] = benchmark_blocks(model, blocks, forward, config)
                        result["timing_chip"] = row["id"]
                        result["timing_exclusions"] = "Downloads, model construction, TIFF IO, CPU radiometry/resize/normalization, H2D/D2H, feature hooks/capture, gradients, figure generation, persistence and automatic pipeline excluded. SAM includes forward_image (conv_s0/s1), spatial reshape/no_mem, fixed prompt encoding/dense PE, decoder and native logits bilinear resize. DINO includes embeddings/RoPE/all layers/final norm. Block replay is isolated and not additive; conv_s0/s1 called outside decoder are residual in full FLOP accounting."
                        persist(run, prefix + "_profile", canonical_json(result), report["artifacts"], "json")
                        persist(run, prefix + "_flops", csv_payload(result["flops"]["rows"]), report["artifacts"], "csv")
                        times = {"full_static_path": result["full_path_timing"], **result["block_timings"]}
                        persist(run, prefix + "_timings", csv_payload([
                            {"block": name, **{k: v for k, v in values.items()
                                if k not in ("samples_ms", "latency", "memory", "shapes")}}
                            for name, values in times.items()]), report["artifacts"], "csv")
                        del points, forward
                    del tensor
                print(kind, regime, "train dissection/recovery complete", flush=True)
            assert_frozen(model)
            model_info["state_sha256_after"] = state_fingerprint(model)
            model_info["hooks_removed"] = hook_inventory(model) == before
            model_info["frozen_unchanged"] = model_info["state_sha256_after"] == model_info["state_sha256"]
            if not model_info["hooks_removed"] or not model_info["frozen_unchanged"]:
                raise ValueError("Weights/buffers changed or hooks leaked")
            del model, predictor
            if "forward" in locals():
                del forward
            gc.collect()
            torch.cuda.empty_cache()  # Between models only; never between latency samples.
        measured = prior if prior else report
        observed = []
        for regime, models in measured["regimes"].items():
            for kind, value in models.items():
                flop, timing = value["flops"], value["full_path_timing"]
                observed.append({"regime": regime, "model": kind,
                    "counted_forward_flops_partial": flop["counted_total"],
                    "median_ms": timing["median_ms"], "peak_allocated_bytes": timing["peak_allocated_bytes"],
                    "arithmetic_share_by_block": {row["block"]: (row["counted_flops"] or 0) / flop["counted_total"]
                        for row in flop["rows"] if row["counted_flops"] is not None},
                    "erf_mass_outside_central_half_by_train_chip": {
                        chip_id: chip["features"]["erf"]["mass_outside_central_half_input"]
                        for chip_id, chip in value["chips"].items()}})
        report["interpretation"] = {
            "hypotheses": ["Image encoder dominates SAM static arithmetic; memory modules remain present.",
                "Deep global-attention DINO features can have nonuniform distant sensitivity.",
                "Latency is not determined by counted FLOPs alone."],
            "observed": observed,
            "findings": "Observed block arithmetic shares quantify where covered work occurs; null blocks were not executed. ERF outside-center mass measures nonuniform distant sensitivity for these fixed targets. Cross-model timing uses different resolutions/outputs and cannot rank water quality; human/assistant visual review and hypothesis conclusions pending.",
            "limits": "Two lexicographic train chips per regime; fixed diagnostic spatial points are not semantic prompts. No water quality metric, learned projection, probe, decoder training or automatic flow. Gradients are local channel-0 sensitivity. Partial FLOPs, warm allocator, isolated replay not additive.",
            "ai_use": "Codex prepared instrumentation, notebook, contracts and documentation; no GPU result fabricated. Human execution and evidence review required."}
        persist(run, "interpretation_" + mode, canonical_json(report["interpretation"]), report["artifacts"], "json")
        if prior:
            report["different_runtime_boot"] = report["runtime_boot_id"] != prior["runtime_boot_id"]
            if not report["different_runtime_boot"]:
                raise ValueError("Recovery must run after disconnect/delete runtime")
            report["original_measurements_verified_not_retimed"] = True
        report["status"] = "computation_passed_pending_external_review"
        run.store.publish("recovery_validation" if prior else "validation", canonical_json(report), identity)
        if not prior:
            run.transition("complete")
        export_evidence(root, run)
        return report
    except BaseException as error:
        run.store.publish("failure", canonical_json({"type": type(error).__name__,
            "message": str(error), "at": utc_now(), "partial_report": report}), identity)
        if not prior and run.state == "running":
            run.transition("interrupted")
        export_evidence(root, run)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "recover"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--models", default="sam,dino", help="comma list: sam | sam,dino (use the same value for recover)")
    args = parser.parse_args()
    run_stage4(args.root, args.mode, tuple(m for m in args.models.split(",") if m))
