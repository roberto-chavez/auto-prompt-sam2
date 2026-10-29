"""User-run Colab GPU entry point. Never run real SAM on the workstation."""
from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from auto_prompt_sam2.artifacts import ArtifactStore, Run, canonical_json, digest, sha256_file, utc_now
from auto_prompt_sam2.sam_adapter import require_colab_gpu


def source_snapshot():
    from auto_prompt_sam2.runtime import source_files, provenance
    paths = set(source_files(REPO))
    paths.update(REPO / p for p in ("configs/data.yaml", "configs/baseline.yaml",
        "requirements-stage2.txt", "notebooks/03_linea_base_asistida.ipynb"))
    info = provenance(REPO)
    info["source_sha256"] = {p.relative_to(REPO).as_posix(): sha256_file(p) for p in sorted(paths)}
    info["source_tree_sha256"] = digest(info["source_sha256"])
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(paths):
            archive.write(p, p.relative_to(REPO).as_posix())
    return info, output.getvalue()


def export_evidence(root, run):
    """All run generations, masks, prompts, config and source; no raw data/weights."""
    output = root / "stage3/evidence" / run.run_id
    output.mkdir(parents=True, exist_ok=True)
    for name in ("validation", "recovery_validation", "config", "model", "environment"):
        try:
            payload, _ = run.store.read(name, run.identity)
        except FileNotFoundError:
            continue
        (output / (name + ".json")).write_bytes(payload)
    folders = [run.store.root, *(root / "regimes" / r / "runs" / run.run_id for r in run.identity["regimes"])]
    target = output.with_suffix(".zip")
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for folder in [*folders, output]:
            for p in sorted(folder.rglob("*")):
                if p.is_file():
                    archive.write(p, p.relative_to(root).as_posix())
    print(json.dumps({"run_id": run.run_id, "evidence_zip": str(target),
                      "sha256": sha256_file(target)}, indent=2), flush=True)


def run_stage3(root, mode):
    # This guard precedes model imports, downloads and construction.
    require_colab_gpu()
    import numpy as np
    import torch
    from auto_prompt_sam2.baseline import (REGIMES, validate_config, stage2_inputs,
        development_plan, predict_development, encode_prediction, decode_prediction,
        prediction_result, summaries, permitted_row)
    from auto_prompt_sam2.runtime import (load_config, persistent_root, seed_everything,
        verify_installed_pins, inventory)
    from auto_prompt_sam2.sam_adapter import (load_frozen_sam, state_fingerprint,
        assert_frozen, compare_predictions)

    environment_config = load_config(REPO)
    os.environ[environment_config["persistent_root_env"]] = str(root)
    root = persistent_root(environment_config, colab=True)
    config = json.loads((REPO / "configs/baseline.yaml").read_text("utf-8"))
    data_config = json.loads((REPO / "configs/data.yaml").read_text("utf-8"))
    validate_config(config)
    if config["seed"] != environment_config["seed"] or environment_config["precision"] != "float32":
        raise ValueError("Environment/baseline seed or precision mismatch")
    seed_everything(config["seed"])
    torch.set_num_threads(1)
    pins = verify_installed_pins(REPO, environment_config, linux_lock=True)
    # No additional dependencies or unlocked package upgrades in this stage.
    for line in (REPO / "requirements-stage2.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            import importlib.metadata
            name, version = line.split("==")
            if importlib.metadata.version(name) != version:
                raise ValueError("Stage-2 package pin mismatch: " + name)
    inputs = stage2_inputs(root, data_config)
    source, source_zip = source_snapshot()
    identity = {"stage": "phase1_stage3", "config_sha256": digest(config),
        "source_tree_sha256": source["source_tree_sha256"],
        "model": environment_config["models"]["sam2"],
        "regimes": {regime: {"manifest_sha256": digest(value["manifest"]),
            "preprocessing_sha256": digest(value["params"]),
            "manifest_generation": value["manifest_generation"],
            "parameter_generation": value["parameter_generation"]} for regime, value in inputs.items()}}
    ticket_store = ArtifactStore(root / "stage3")
    if mode == "prepare":
        run = Run.create(root / "stage3", identity)
        ticket_store.publish("active_run", canonical_json({"run_id": run.run_id}), identity)
        run.transition("running")
        run.store.publish("source_snapshot", source_zip, identity)
        run.store.publish("config", canonical_json({"baseline": config, "data": data_config,
            "environment": environment_config, "source": source, "identity": identity}), identity)
        prior = None
    else:
        ticket = json.loads(ticket_store.read("active_run", identity)[0])
        run = Run(root / "stage3", ticket["run_id"], identity)
        if mode == "resume":
            run.resume()  # User must stop the prior writer before this mode.
            if run.state == "complete":
                raise ValueError("Run complete; use recover")
            prior = None
        else:
            if run.state != "complete":
                raise ValueError("Recovery requires a complete baseline run")
            prior = json.loads(run.store.read("validation", identity)[0])
    print("Stage 3", mode, run.run_id, flush=True)
    report = {"stage": "phase1_stage3", "mode": mode, "run_id": run.run_id,
        "created_at": utc_now(), "scope": "colab_gpu_development_only",
        "branch": config["branch"], "identity": identity, "regimes": {},
        "reserved_access": [], "training_executed": False, "final_evaluations": [],
        "closure": "pending_external_review_of_user_notebook_and_json",
        "license": "publisher_reply_pending_historical_salvedad_retained"}
    try:
        completed = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                                  cwd=REPO, text=True, capture_output=True)
        tests = {"exit_code": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}
        run.store.publish("tests_" + mode, canonical_json(tests), identity)
        print(completed.stdout, completed.stderr, flush=True)
        completed.check_returncode()
        pip = subprocess.run([sys.executable, "-m", "pip", "check"], text=True, capture_output=True)
        pip.check_returncode()
        env = inventory(root, REPO)
        report["environment"] = env
        report["pins"] = pins
        report["tests"] = tests
        report["pip_check"] = pip.stdout
        report["process_id"] = os.getpid()
        if Path("/proc/sys/kernel/random/boot_id").exists():
            report["runtime_boot_id"] = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        environment_generation = run.store.publish("environment", canonical_json(env), identity)
        try:
            model_prior = json.loads(run.store.read("model", identity)[0])
        except FileNotFoundError:
            if mode == "recover":
                raise
            model_prior = None
        predictor, model_info = load_frozen_sam(identity["model"], root,
            offline=model_prior is not None, expected_sha256=model_prior["checkpoint_sha256"] if model_prior else None)
        if model_prior and model_info["state_sha256"] != model_prior["state_sha256"]:
            raise ValueError("Reloaded SAM state differs")
        if not model_prior:
            run.store.publish("model", canonical_json(model_info), identity)
        report["model"] = model_info
        for regime in REGIMES:
            value = inputs[regime]
            manifest = value["manifest"]
            store = ArtifactStore(root / "regimes" / regime / "runs" / run.run_id / "assisted")
            records, index, accesses = [], [], []
            plan = development_plan(manifest)
            for row in plan:
                permitted_row(manifest, chip_id=row["id"], role=row["role"])
                key = dict(identity, regime=regime, id=row["id"], role=row["role"],
                           image_sha256=row["image_sha256"], label_sha256=row["label_sha256"],
                           checkpoint_sha256=model_info["checkpoint_sha256"])
                try:
                    payload, marker = store.read(row["id"], key)
                    mask, metadata = decode_prediction(payload)
                except FileNotFoundError:
                    if mode == "recover":
                        raise  # Recovery must not silently regenerate lost predictions.
                    result, metadata = predict_development(predictor, root, value, row, config)
                    accesses.append({"id": row["id"], "role": row["role"], "purpose": "baseline"})
                    metadata["execution"] = {"run_id": run.run_id, "mode": mode,
                        "recorded_at": utc_now(), "environment_generation": environment_generation,
                        "environment_sha256": digest(env), "model_state_sha256": model_info["state_sha256"],
                        "checkpoint_sha256": model_info["checkpoint_sha256"]}
                    mask = result["mask"]
                    generation = store.publish(row["id"], encode_prediction(mask, metadata), key)
                    payload, marker = store.read(row["id"], key, generation)
                    restored, restored_meta = decode_prediction(payload)
                    if not np.array_equal(restored, mask) or restored_meta != metadata:
                        raise ValueError("Prediction persistence roundtrip failed")
                # Metadata is bound by marker identity; check logical identity too.
                if (metadata["regime"] != regime or metadata["id"] != row["id"] or
                        metadata["role"] != row["role"] or metadata["config_sha256"] != digest(config)):
                    raise ValueError("Saved prediction identity mismatch")
                records.append(metadata)
                index.append({"id": row["id"], "role": row["role"], "identity": key,
                              "generation": marker["generation"], "sha256": marker["sha256"]})
                print(regime, row["role"], row["id"], "components", len(metadata["inference"]["components"]), flush=True)
            # A real prompted chip, not merely a label-assisted empty prediction.
            repeat_meta = next((r for r in records if r["inference"]["predict_calls"]), None)
            if repeat_meta is None:
                raise ValueError("No prompted development chip available for SAM repeat check")
            repeat_row = permitted_row(manifest, chip_id=repeat_meta["id"], role=repeat_meta["role"])
            saved_entry = next(e for e in index if e["id"] == repeat_meta["id"])
            saved_mask, _ = decode_prediction(store.read(repeat_meta["id"], saved_entry["identity"], saved_entry["generation"])[0])
            repeated, repeated_meta = predict_development(predictor, root, value, repeat_row, config)
            accesses.append({"id": repeat_meta["id"], "role": repeat_meta["role"], "purpose": "repeat_" + mode})
            if repeated_meta["prompts"] != repeat_meta["prompts"]:
                raise ValueError("Same-seed prompts changed")
            repeat = compare_predictions(prediction_result(saved_mask, repeat_meta), repeated,
                                         config["repeat_tolerance"])
            summary = summaries(records)
            regime_report = {"planned_ids": [{"id": r["id"], "role": r["role"]} for r in plan],
                "counts": {role: sum(r["role"] == role for r in records) for role in ("train", "val")},
                "summary": summary, "prediction_index": index, "repeat_chip": repeat_meta["id"],
                "repeat": repeat, "points_identical": True, "access_log": accesses,
                "preprocessing_reused": True, "inputs": identity["regimes"][regime]}
            if prior:
                expected = prior["regimes"][regime]
                if index != expected["prediction_index"] or summary != expected["summary"]:
                    raise ValueError("Recovered prediction index/summary differs")
                regime_report["all_saved_predictions_recovered"] = True
            store.publish("summary_" + mode, canonical_json(regime_report), identity)
            report["regimes"][regime] = regime_report
        assert_frozen(predictor.model)
        report["state_sha256_after"] = state_fingerprint(predictor.model)
        if report["state_sha256_after"] != model_info["state_sha256"]:
            raise ValueError("SAM state changed during inference")
        report["frozen_unchanged"] = True
        report["status"] = "development_checks_passed_pending_review"
        if prior:
            report["reloaded_from_persistent_checkpoint"] = True
            report["different_process"] = report["process_id"] != prior["process_id"]
            report["different_runtime_boot"] = report.get("runtime_boot_id") != prior.get("runtime_boot_id")
        name = "recovery_validation" if mode == "recover" else "validation"
        run.store.publish(name, canonical_json(report), identity)
        if mode != "recover":
            run.transition("complete")  # Computation complete; stage closure still requires review.
        export_evidence(root, run)
        print(json.dumps({r: v["summary"]["val"] for r, v in report["regimes"].items()}, indent=2), flush=True)
        return report
    except BaseException as error:
        run.store.publish("failure_" + mode, canonical_json({"type": type(error).__name__,
            "message": str(error), "at": utc_now()}), identity)
        if mode != "recover" and run.state == "running":
            run.transition("interrupted")
        export_evidence(root, run)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "resume", "recover"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    run_stage3(args.root, args.mode)


if __name__ == "__main__":
    main()
