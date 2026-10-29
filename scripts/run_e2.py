"""User-run Colab GPU entry point for the E2 preliminary decoder intervention.

Modes: prepare (full run), resume (continue an interrupted training run), recover
(new runtime: reload SAM + best decoder from Drive and re-evaluate). Never run on the PC.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import time
import warnings
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from auto_prompt_sam2.artifacts import ArtifactStore, Run, canonical_json, digest, sha256_file, utc_now
from auto_prompt_sam2.sam_adapter import require_colab_gpu

LIMITATIONS = [
    "Preliminary: official regime, small label-free subsets of train/val, one seed. No claim about unseen events, Bolivia or Paraguay-out.",
    "Prompts are label-simulated (reference-assisted, not deployable). The automatic DINOv3 -> SAM flow is NOT implemented here.",
    "Chips without water have no components: empty prediction and FP=0 by construction in BOTH arms (label-assisted advantage); they produce no gradient.",
    "Validation is used both to pick the best epoch and to report the result: the best-epoch number is optimistic. The last-epoch number is also reported.",
    "The IoU-prediction head receives no gradient from BCE+Dice, so selection by predicted IoU may drift away from true mask quality after tuning.",
    "No significance test, no repeated seeds. Per-chip deltas are descriptive.",
]


def source_snapshot():
    from auto_prompt_sam2.runtime import source_files, provenance
    paths = set(source_files(REPO))
    paths.update(REPO / p for p in ("configs/data.yaml", "configs/baseline.yaml", "configs/e2_intervention.yaml",
                                    "requirements-stage2.txt"))
    paths = {p for p in paths if p.is_file()}
    info = provenance(REPO)
    info["source_sha256"] = {p.relative_to(REPO).as_posix(): sha256_file(p) for p in sorted(paths)}
    info["source_tree_sha256"] = digest(info["source_sha256"])
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(paths):
            archive.write(p, p.relative_to(REPO).as_posix())
    return info, output.getvalue()


def setup(root):
    import torch
    from auto_prompt_sam2 import intervention as iv
    from auto_prompt_sam2.baseline import stage2_inputs
    from auto_prompt_sam2.runtime import load_config, persistent_root, seed_everything, verify_installed_pins
    environment_config = load_config(REPO)
    os.environ[environment_config["persistent_root_env"]] = str(root)
    root = persistent_root(environment_config, colab=True)
    config = json.loads((REPO / "configs/e2_intervention.yaml").read_text("utf-8"))
    baseline = json.loads((REPO / "configs/baseline.yaml").read_text("utf-8"))
    data_config = json.loads((REPO / "configs/data.yaml").read_text("utf-8"))
    iv.validate_config(config, baseline)
    if config["seed"] != environment_config["seed"] or environment_config["precision"] != "float32":
        raise ValueError("Environment/E2 seed or precision mismatch")
    seed_everything(config["seed"])
    # Training needs backward through bilinear upsampling, which has no deterministic CUDA kernel.
    # Do not hide it: warn (recorded in the report) instead of raising.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(1)
    pins = verify_installed_pins(REPO, environment_config, linux_lock=True)
    import importlib.metadata
    for line in (REPO / "requirements-stage2.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            name, version = line.split("==")
            if importlib.metadata.version(name) != version:
                raise ValueError("Stage-2 package pin mismatch: " + name)
    # Stage-2 artifacts are read, never rebuilt. Missing -> FileNotFoundError with a clear stop.
    try:
        inputs = stage2_inputs(root, data_config)
    except FileNotFoundError as error:
        raise SystemExit("PREREQUISITE MISSING: stage-2 data/manifests not found under %s. Run notebook 02 first "
                         "(or share the Drive folder). Original error: %s" % (root, error))
    value = inputs[config["regime"]]
    source, snapshot = source_snapshot()
    identity = {"stage": config["stage"], "config_sha256": digest(config),
                "source_tree_sha256": source["source_tree_sha256"],
                "model": environment_config["models"]["sam2"], "regime": config["regime"],
                "manifest_sha256": digest(value["manifest"]), "preprocessing_sha256": digest(value["params"]),
                "manifest_generation": value["manifest_generation"], "parameter_generation": value["parameter_generation"]}
    return {"root": root, "env_config": environment_config, "config": config, "baseline": baseline,
            "data_config": data_config, "pins": pins, "value": value, "source": source, "snapshot": snapshot,
            "identity": identity}


def load_model(ctx, *, offline):
    from auto_prompt_sam2 import intervention as iv
    from auto_prompt_sam2.sam_adapter import load_frozen_sam, state_fingerprint
    expected = ctx["config"]["model_integrity"]
    predictor, info = load_frozen_sam(ctx["identity"]["model"], ctx["root"], offline=offline,
                                      expected_sha256=expected["expected_checkpoint_sha256"])
    if info["state_sha256"] != expected["expected_state_sha256"]:
        raise ValueError("SAM state hash differs from the audited stage-3/4 state; do not continue")
    runner = iv.Sam2Runner(predictor)
    return predictor, runner, info, state_fingerprint


def build_chips(ctx, runner, rows, *, keep_image):
    from auto_prompt_sam2 import intervention as iv
    from auto_prompt_sam2.baseline import permitted_row, read_development_chip
    from auto_prompt_sam2.preprocessing import sam_predictor_image
    from auto_prompt_sam2.artifacts import digest as dg
    from auto_prompt_sam2.reference_prompts import simulate_points
    manifest, params, config = ctx["value"]["manifest"], ctx["value"]["params"], ctx["config"]
    chips = []
    for row in rows:
        permitted_row(manifest, chip_id=row["id"], role=row["role"])  # membership before any read
        row, rgb, valid, labels = read_development_chip(ctx["root"], manifest, chip_id=row["id"], role=row["role"])
        prompts = simulate_points(labels, regime=manifest["regime"], chip_id=row["id"], seed=config["seed"],
                                  policy=config["prompts"])
        image = sam_predictor_image(rgb, valid, params)
        chip = iv.build_chip(runner, image, labels, row, prompts, dg(params), device="cuda")
        if keep_image:
            chip["image"] = image
        chips.append(chip)
    return chips


def chip_stats(chips):
    comps = [len(c["components"]) for c in chips]
    events = {}
    for c in chips:
        events[c["event"]] = events.get(c["event"], 0) + 1
    return {"chips": len(chips), "per_event": dict(sorted(events.items())), "components_total": sum(comps),
            "components_min": min(comps), "components_max": max(comps),
            "chips_without_components": sum(1 for n in comps if n == 0)}


def evaluation_view(ev):
    return {"summary": ev["summary"], "chips": ev["chips"]}


def per_chip_rows(frozen, tuned, chips):
    comps = {c["id"]: len(c["components"]) for c in chips}
    t = {r["id"]: r for r in tuned["chips"]}
    rows = []
    for f in frozen["chips"]:
        r = t[f["id"]]
        fv, tv = f["scores"]["values"], r["scores"]["values"]
        rows.append({"id": f["id"], "event": f["event"], "components": comps[f["id"]],
                     "iou_frozen": f["individual_water_iou"], "iou_tuned": r["individual_water_iou"],
                     "delta": None if f["individual_water_iou"] is None or r["individual_water_iou"] is None
                     else r["individual_water_iou"] - f["individual_water_iou"],
                     "precision_frozen": fv["precision"], "precision_tuned": tv["precision"],
                     "recall_frozen": fv["recall"], "recall_tuned": tv["recall"],
                     "dice_frozen": fv["dice"], "dice_tuned": tv["dice"],
                     "fp_pixels_frozen": f["fp_pixels"], "fp_pixels_tuned": r["fp_pixels"],
                     "exclusion": f["individual_iou_exclusion"]})
    return rows


def write_evidence(ctx, run, report, frozen_ev, tuned_ev, val_chips, train_chips, history, mode):
    """Plain files for review (no checkpoints, raw data or tokens) + a ZIP; private arrays stay in Drive."""
    import numpy as np
    from auto_prompt_sam2 import e2_report as rep
    root, run_id = ctx["root"], run.run_id
    out = root / "e2/evidence" / run_id
    (out / "examples").mkdir(parents=True, exist_ok=True)
    columns = ["id", "event", "components", "iou_frozen", "iou_tuned", "delta", "precision_frozen", "precision_tuned",
               "recall_frozen", "recall_tuned", "dice_frozen", "dice_tuned", "fp_pixels_frozen", "fp_pixels_tuned", "exclusion"]
    rows = per_chip_rows(frozen_ev, tuned_ev, val_chips)
    (out / "per_chip_val.csv").write_text(rep.csv_text(rows, columns), "utf-8")
    hist_cols = ["epoch", "train_bce", "train_dice", "train_loss", "train_chips_with_gradient", "train_chips_without_gradient",
                 "optimizer_steps", "val_water_iou_pooled", "val_event_mean_water_iou", "seconds"]
    (out / "history.csv").write_text(rep.csv_text(history, hist_cols), "utf-8")
    frozen_iou = frozen_ev["summary"]["pooled"]["values"]["water_iou"]
    (out / "curves.png").write_bytes(rep.curves_png(history, frozen_iou))
    ids = [c["id"] for c in val_chips]
    np.savez_compressed(out / "predictions_val.npz",
                        ids=np.array(ids), frozen=np.stack([np.packbits(frozen_ev["masks"][i].ravel()) for i in ids]),
                        tuned=np.stack([np.packbits(tuned_ev["masks"][i].ravel()) for i in ids]),
                        note=np.array("bit-packed 512x512 boolean masks (np.unpackbits(...)[:262144].reshape(512,512)); rows follow ids"))
    (out / "points_val.json").write_bytes(canonical_json({c["id"]: c["prompts"] for c in val_chips}))
    seen, examples = set(), []
    for c in val_chips:  # first val chip per event in the subset: label-free, no cherry-picking
        if c["event"] in seen:
            continue
        seen.add(c["event"])
        fm, tm = frozen_ev["masks"][c["id"]], tuned_ev["masks"][c["id"]]
        fr = {r["id"]: r for r in frozen_ev["chips"]}[c["id"]]
        tr = {r["id"]: r for r in tuned_ev["chips"]}[c["id"]]
        fmt = lambda r: "IoU=%s Dice=%s" % tuple("n/a" if v is None else "%.3f" % v for v in (r["individual_water_iou"], r["scores"]["values"]["dice"]))
        png = rep.example_png(c["image"], c["labels"], c["prompts"]["components"], fm, tm, "%s (%s)" % (c["id"], c["event"]),
                              fmt(fr), fmt(tr))
        (out / "examples" / (c["id"] + ".png")).write_bytes(png)
        examples.append(c)
    private = root / "e2/runs" / run_id / "private"
    private.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(private / "examples_private_NOT_FOR_GIT.npz",
                        ids=np.array([c["id"] for c in examples]),
                        images=np.stack([c["image"] for c in examples]).astype(np.float16),
                        labels=np.stack([c["labels"] for c in examples]).astype(np.int8),
                        frozen=np.stack([frozen_ev["masks"][c["id"]] for c in examples]),
                        tuned=np.stack([tuned_ev["masks"][c["id"]] for c in examples]))
    (out / "report.json").write_bytes(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8"))
    return out


def zip_evidence(root, run, out):
    target = out.with_suffix(".zip")
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(out.rglob("*")):
            if p.is_file():
                archive.write(p, p.relative_to(out.parent).as_posix())
        for name in ("source_snapshot", "config", "hypothesis", "environment", "model", "tests_prepare"):
            try:
                payload, marker = run.store.read(name, run.identity)
            except FileNotFoundError:
                continue
            archive.writestr("%s/store/%s_%s" % (run.run_id, name, marker["generation"]), payload)
    print(json.dumps({"run_id": run.run_id, "evidence_zip": str(target), "sha256": sha256_file(target),
                      "bytes": target.stat().st_size}, indent=2), flush=True)
    return target


def run_e2(root, mode):
    require_colab_gpu()
    import numpy as np
    import torch
    from auto_prompt_sam2 import intervention as iv
    from auto_prompt_sam2 import e2_report as rep
    from auto_prompt_sam2.data import development_records
    from auto_prompt_sam2.runtime import inventory
    from auto_prompt_sam2.sam_adapter import segment_components

    started_at = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ctx = setup(root)
        root, config, identity = ctx["root"], ctx["config"], ctx["identity"]
        ticket_store = ArtifactStore(root / "e2")
        if mode == "prepare":
            run = Run.create(root / "e2", identity)
            ticket_store.publish("active_run", canonical_json({"run_id": run.run_id}), identity)
            run.transition("running")
            run.store.publish("source_snapshot", ctx["snapshot"], identity)
            run.store.publish("config", canonical_json({"e2": config, "environment": ctx["env_config"], "source": ctx["source"],
                                                         "identity": identity}), identity)
            # Hypothesis is persisted BEFORE any model is loaded or trained.
            hypothesis = dict(config["hypothesis"], recorded_at=utc_now(), config_sha256=digest(config))
            run.store.publish("hypothesis", canonical_json(hypothesis), identity)
            print("HYPOTHESIS (recorded before training):\n" + json.dumps(config["hypothesis"], indent=2), flush=True)
        else:
            ticket = json.loads(ticket_store.read("active_run", identity)[0])
            run = Run(root / "e2", ticket["run_id"], identity)
            if mode == "resume":
                run.resume()
                if run.state == "complete":
                    raise ValueError("Run already complete; use recover")
            elif run.state != "complete":
                raise ValueError("recover requires a complete run")
        print("E2", mode, run.run_id, flush=True)
        outputs = ArtifactStore(root / "e2/runs" / run.run_id / "outputs")
        checkpoints = ArtifactStore(root / "e2/runs" / run.run_id / "checkpoints")
        report = {"stage": config["stage"], "mode": mode, "run_id": run.run_id, "created_at": utc_now(),
                  "scope": config["scope_note"], "regime": config["regime"], "hypothesis": config["hypothesis"],
                  "identity": identity, "reserved_access": [], "automatic_prompts": False, "dinov3_used": False,
                  "limitations": LIMITATIONS, "closure": "pending_external_review_of_user_notebook_and_json"}
        try:
            if mode != "recover":
                completed = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                                           cwd=REPO, text=True, capture_output=True)
                tests = {"exit_code": completed.returncode, "stdout": completed.stdout[-20000:],
                         "stderr": completed.stderr[-20000:]}
                run.store.publish("tests_" + mode, canonical_json(tests), identity)
                print(completed.stderr[-3000:], flush=True)
                completed.check_returncode()
                pip = subprocess.run([sys.executable, "-m", "pip", "check"], text=True, capture_output=True)
                pip.check_returncode()
                report["tests"] = {"exit_code": 0, "pip_check": pip.stdout}
                env = inventory(root, REPO)
                run.store.publish("environment", canonical_json(env), identity)
                report["environment"] = env
            report["pins"] = ctx["pins"]

            manifest = ctx["value"]["manifest"]
            plan_train = iv.select_subset(development_records(manifest, regime=config["regime"], role="train"),
                                          config["subset"]["train_chips_per_event"])
            plan_val = iv.select_subset(development_records(manifest, regime=config["regime"], role="val"),
                                        config["subset"]["val_chips_per_event"])
            if any(r["event"] == "Bolivia" for r in [*plan_train, *plan_val]):
                raise ValueError("Reserved Bolivia chip in development plan")
            report["subsets"] = {"train_ids": [r["id"] for r in plan_train], "val_ids": [r["id"] for r in plan_val]}

            predictor, runner, info, state_fingerprint = load_model(ctx, offline=(mode != "prepare"))
            if mode == "prepare":
                run.store.publish("model", canonical_json(info), identity)
            report["model"] = info
            report["parameters_before"] = iv.parameter_report(predictor.model)
            state_before_all = info["state_sha256"]
            chunk = config["inference"]["component_chunk"]

            # ---- frozen control (epoch 0) -------------------------------------------------
            val_chips = build_chips(ctx, runner, plan_val, keep_image=True)
            report["subsets"]["val_stats"] = chip_stats(val_chips)
            print("val chips ready", report["subsets"]["val_stats"], flush=True)
            if mode == "recover":
                prior = json.loads((root / "e2/evidence" / run.run_id / "report.json").read_text("utf-8"))
                if prior["subsets"]["val_ids"] != report["subsets"]["val_ids"]:
                    raise ValueError("Recovered subset differs")
                frozen_ev = iv.evaluate_chips(runner, val_chips, chunk, keep_masks=True)
                saved_masks, saved_meta = rep.decode_eval(outputs.read("epoch0_val", dict(identity, kind="epoch0_val"))[0])
                tol = config["verification"]["recover_water_iou_atol"]
                a = frozen_ev["summary"]["pooled"]["values"]["water_iou"]
                b = saved_meta["summary"]["pooled"]["values"]["water_iou"]
                same_conf = all(x["confusion"] == y["confusion"] for x, y in zip(frozen_ev["chips"], saved_meta["chips"]))
                disagreement = sum(int(np.count_nonzero(frozen_ev["masks"][i] != saved_masks[i])) for i in saved_masks)
                frozen_check = {"pooled_iou_now": a, "pooled_iou_saved": b, "abs_diff": abs(a - b), "tolerance": tol,
                                "per_chip_confusions_identical": same_conf, "mask_disagreement_pixels": disagreement,
                                "passed": abs(a - b) <= tol and same_conf}
                best_ref = prior["checkpoints"]["best"]
                blob, _ = checkpoints.read("best", dict(identity, kind="best"), best_ref["generation"])
                if digest_bytes(blob) != best_ref["sha256"]:
                    raise ValueError("Best checkpoint hash differs from report")
                ckpt = iv.unpack_checkpoint(blob)
                before_frozen = iv.frozen_fingerprint(predictor.model)
                if before_frozen != prior["freezing"]["frozen_fingerprint_before"]:
                    raise ValueError("Reloaded frozen blocks differ from the original run")
                iv.load_decoder_state(predictor.model, ckpt["best"]["state"])
                tuned_ev = iv.evaluate_chips(runner, val_chips, chunk, keep_masks=True)
                saved_masks_t, saved_meta_t = rep.decode_eval(outputs.read("tuned_val", dict(identity, kind="tuned_val"))[0])
                a2 = tuned_ev["summary"]["pooled"]["values"]["water_iou"]
                b2 = saved_meta_t["summary"]["pooled"]["values"]["water_iou"]
                same_conf2 = all(x["confusion"] == y["confusion"] for x, y in zip(tuned_ev["chips"], saved_meta_t["chips"]))
                dis2 = sum(int(np.count_nonzero(tuned_ev["masks"][i] != saved_masks_t[i])) for i in saved_masks_t)
                tuned_check = {"pooled_iou_now": a2, "pooled_iou_saved": b2, "abs_diff": abs(a2 - b2), "tolerance": tol,
                               "per_chip_confusions_identical": same_conf2, "mask_disagreement_pixels": dis2,
                               "passed": abs(a2 - b2) <= tol and same_conf2}
                report.update({"status": "recovery_checks_passed_pending_review" if frozen_check["passed"] and tuned_check["passed"] else "recovery_checks_FAILED",
                               "recovered_in_new_process": True, "frozen_arm_check": frozen_check, "tuned_arm_check": tuned_check,
                               "best_epoch_loaded": ckpt["best"]["epoch"], "frozen_fingerprint_matches": True,
                               "process_id": os.getpid(), "created_at": utc_now()})
                if Path("/proc/sys/kernel/random/boot_id").exists():
                    report["runtime_boot_id"] = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
                run.store.publish("recovery_validation", canonical_json(report), identity)
                print(json.dumps({"frozen_arm_check": frozen_check, "tuned_arm_check": tuned_check}, indent=2), flush=True)
                if report["status"].endswith("FAILED"):
                    raise RuntimeError("Recovery check failed; do not change tolerances")
                return report

            # parity of my differentiable path against the stage-3 predictor path (frozen decoder)
            parity = []
            for chip in [c for c in val_chips if c["components"]][:config["verification"]["parity_chips"]]:
                official = segment_components(predictor, chip["image"], chip["components"])
                mine, details = iv.predict_chip(runner, chip["feats"], chip["components"], chunk)
                mine_by_id = {d["component_id"]: d for d in details["components"]}
                diff = int(np.count_nonzero(official["mask"] != mine))
                idx_ok = all(o["selected_index"] == mine_by_id[o["component_id"]]["selected_index"] for o in official["components"])
                parity.append({"id": chip["id"], "mask_disagreement_pixels": diff, "selected_indices_identical": idx_ok})
            if (not parity or any(p["mask_disagreement_pixels"] > config["verification"]["parity_max_disagreement_pixels"]
                                  or not p["selected_indices_identical"] for p in parity)):
                raise ValueError("Differentiable path does not reproduce the stage-3 predictor: " + json.dumps(parity))
            report["parity_with_stage3_predictor"] = parity
            print("parity ok", parity, flush=True)

            if mode == "prepare":
                frozen_ev = iv.evaluate_chips(runner, val_chips, chunk, keep_masks=True)
                outputs.publish("epoch0_val", rep.encode_eval(frozen_ev, [c["id"] for c in val_chips]),
                                dict(identity, kind="epoch0_val"))
            else:
                masks, meta = rep.decode_eval(outputs.read("epoch0_val", dict(identity, kind="epoch0_val"))[0])
                frozen_ev = {"summary": meta["summary"], "chips": meta["chips"], "masks": masks, "details": meta["details"]}
            print("epoch 0 (frozen) val pooled IoU:", frozen_ev["summary"]["pooled"]["values"]["water_iou"], flush=True)

            # ---- decoder-only training ------------------------------------------------------
            parameters_after = iv.configure_decoder_only(predictor.model)
            report["parameters_after_configure"] = parameters_after
            frozen_fp, decoder_fp0 = iv.frozen_fingerprint(predictor.model), iv.decoder_fingerprint(predictor.model)
            train_chips = build_chips(ctx, runner, plan_train, keep_image=False)
            report["subsets"]["train_stats"] = chip_stats(train_chips)
            print("train chips ready", report["subsets"]["train_stats"], flush=True)
            checkpoint_index = {}

            def on_epoch(epoch, payload):
                blob = iv.pack_checkpoint({"epoch": epoch, "history": payload["history"], "best": payload["best"],
                                           "optimizer": payload["optimizer"], "decoder_state": payload["decoder_state"],
                                           "finished": payload["finished"]}, digest(config))
                gen = checkpoints.publish("last", blob, dict(identity, kind="last"))
                checkpoint_index["last"] = {"generation": gen, "sha256": digest_bytes(blob), "epoch": epoch, "bytes": len(blob)}
                best = payload["best"]
                if best["epoch"] == epoch:
                    bblob = iv.pack_checkpoint({"epoch": epoch, "best": best}, digest(config))
                    bgen = checkpoints.publish("best", bblob, dict(identity, kind="best"))
                    checkpoint_index["best"] = {"generation": bgen, "sha256": digest_bytes(bblob), "epoch": epoch, "bytes": len(bblob)}

            resume_args = {}
            if mode == "resume":
                ckpt = iv.unpack_checkpoint(checkpoints.read("last", dict(identity, kind="last"))[0])
                if ckpt["finished"]:
                    raise ValueError("Training already finished in this run")
                iv.load_decoder_state(predictor.model, ckpt["decoder_state"])
                resume_args = {"history": ckpt["history"], "start_epoch": ckpt["epoch"] + 1, "best": ckpt["best"],
                               "optimizer_state": ckpt["optimizer"]}
                print("resuming after epoch", ckpt["epoch"], flush=True)
            train_out = iv.train_decoder(runner, train_chips, val_chips, config, on_epoch=on_epoch, **resume_args)
            history, best = train_out["history"], train_out["best"]
            if best["epoch"] is None:
                raise RuntimeError("No fine-tuned epoch completed")
            report["history"], report["best_epoch"] = history, best["epoch"]
            report["gradient_audit_first_step"] = train_out["gradient_audit_first_step"]
            report["epochs_run"] = len(history)
            last_iou = history[-1]["val_water_iou_pooled"]

            # ---- final evaluation with the BEST decoder, then verification -------------------
            iv.load_decoder_state(predictor.model, best["state"])
            tuned_ev = iv.evaluate_chips(runner, val_chips, chunk, keep_masks=True)
            outputs.publish("tuned_val", rep.encode_eval(tuned_ev, [c["id"] for c in val_chips]), dict(identity, kind="tuned_val"))
            frozen_after, decoder_fp1 = iv.frozen_fingerprint(predictor.model), iv.decoder_fingerprint(predictor.model)
            if frozen_after != frozen_fp:
                raise RuntimeError("Frozen blocks changed during training")
            if decoder_fp1 == decoder_fp0:
                raise RuntimeError("Decoder did not change: training had no effect")
            # checkpoint round-trip from Drive
            blob, marker = checkpoints.read("best", dict(identity, kind="best"))
            best_ref = {"generation": marker["generation"], "sha256": digest_bytes(blob), "epoch": best["epoch"], "bytes": len(blob)}
            restored = iv.unpack_checkpoint(blob)["best"]["state"]
            roundtrip = all(torch.equal(restored[k], best["state"][k]) for k in best["state"])
            if not roundtrip:
                raise RuntimeError("Best checkpoint round-trip from Drive failed")
            lblob, lmarker = checkpoints.read("last", dict(identity, kind="last"))
            last_ref = {"generation": lmarker["generation"], "sha256": digest_bytes(lblob), "bytes": len(lblob)}
            report["checkpoints"] = {"best": best_ref, "last": last_ref,
                                     "location": str(root / "e2/runs" / run.run_id / "checkpoints"), "roundtrip_equal": roundtrip,
                                     "note": "Stored in Drive only; never commit to Git."}
            report["freezing"] = {"frozen_fingerprint_before": frozen_fp, "frozen_fingerprint_after": frozen_after,
                                  "frozen_blocks_unchanged": True, "decoder_fingerprint_before": decoder_fp0,
                                  "decoder_fingerprint_best": decoder_fp1, "decoder_changed": True,
                                  "full_state_hash_before_training": state_before_all}
            fv = frozen_ev["summary"]["pooled"]["values"]["water_iou"]
            tv = tuned_ev["summary"]["pooled"]["values"]["water_iou"]
            report["arms"] = {"frozen_epoch0": iv.arm_row("frozen_decoder", frozen_ev),
                              "tuned_best_epoch": iv.arm_row("finetuned_decoder_best_epoch_%d" % best["epoch"], tuned_ev),
                              "tuned_last_epoch_val_pooled_iou": last_iou}
            report["paired_deltas_val"] = iv.paired_deltas(frozen_ev["chips"], tuned_ev["chips"])
            report["hypothesis_check"] = {"criterion": config["hypothesis"]["refuted_if"], "frozen_val_pooled_iou": fv,
                                          "tuned_best_val_pooled_iou": tv,
                                          "mechanical_outcome": "not_refuted_on_this_subset" if tv > fv else "refuted_on_this_subset",
                                          "interpretation": "pending: the team must interpret, including precision/recall and failures"}
            report["status"] = "executed_pending_review"
            val_eval_frozen = frozen_ev
            run.store.publish("validation", canonical_json(report), identity)
            out = write_evidence(ctx, run, report, val_eval_frozen, tuned_ev, val_chips, train_chips, history, mode)
            if mode != "recover" and run.state == "running":
                run.transition("complete")
            report["nondeterminism_warnings"] = sorted({str(w.message)[:240] for w in caught})[:20]
            (out / "report.json").write_bytes(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8"))
            zip_evidence(root, run, out)
            print(json.dumps({"frozen_val_pooled_iou": fv, "tuned_best_val_pooled_iou": tv, "best_epoch": best["epoch"],
                              "tuned_last_epoch_val_pooled_iou": last_iou, "seconds_total": round(time.time() - started_at, 1)}, indent=2), flush=True)
            return report
        except BaseException as error:
            run.store.publish("failure_" + mode, canonical_json({"type": type(error).__name__, "message": str(error), "at": utc_now()}), identity)
            if mode != "recover" and run.state == "running":
                run.transition("interrupted")
            raise


def digest_bytes(blob):
    import hashlib
    return hashlib.sha256(blob).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "resume", "recover"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    run_e2(args.root, args.mode)


if __name__ == "__main__":
    main()
