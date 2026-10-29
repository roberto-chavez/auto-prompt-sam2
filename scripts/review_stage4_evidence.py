"""Read-only CPU artifact audit. Never imports/executes archived code or models."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import zipfile

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from auto_prompt_sam2.artifacts import canonical_json, digest, sha256_file, utc_now
from auto_prompt_sam2.dissection import read_arrays, train_plan
from auto_prompt_sam2.baseline import stage2_inputs
from auto_prompt_sam2.profiling import validate_config


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(zip_path, notebook_path, root, output):
    output.mkdir(parents=True, exist_ok=True)
    review = {"reviewed_at": utc_now(), "scope": "local_CPU_artifact_audit_no_models",
        "models_executed": False, "reserved_access": [], "status": "review_failed",
        "human_interpretation_required": True}
    try:
        review["originals"] = {}
        for path in (zip_path, notebook_path):
            dest = output / path.name
            if not dest.exists():
                shutil.copyfile(path, dest)
            require(sha256_file(path) == sha256_file(dest), "Archived original differs")
            review["originals"][path.name] = {"sha256": sha256_file(path), "size": path.stat().st_size}
        notebook = json.loads(notebook_path.read_text("utf-8"))
        code = [c for c in notebook["cells"] if c["cell_type"] == "code"]
        require(code and all(c.get("execution_count") is not None for c in code), "Unexecuted notebook cells")
        require(not any(o.get("output_type") == "error" for c in code for o in c.get("outputs", [])), "Notebook contains errors")
        with zipfile.ZipFile(zip_path) as z:
            names = z.namelist()
            require(len(names) == len(set(names)) and z.testzip() is None, "Corrupt/duplicate ZIP entries")
            require(all(not PurePosixPath(n).is_absolute() and ".." not in PurePosixPath(n).parts
                and "\\" not in n and ":" not in n for n in names), "Unsafe archive paths")
            def report(name):
                matches = [n for n in names if n.endswith("/" + name + ".json")]
                require(len(matches) == 1, "Missing/ambiguous report: " + name)
                return json.loads(z.read(matches[0]))
            prepare, recover, config = report("validation"), report("recovery_validation"), report("config")
            identity = prepare["identity"]
            require(identity == recover["identity"] == config["identity"], "Run identity differs")
            require(prepare["run_id"] == recover["run_id"], "Recovery run differs")
            validate_config(config["profiling"])
            require(digest(config["profiling"]) == identity["config_sha256"], "Configuration hash differs")
            require(json.loads((REPO / "configs/profiling.yaml").read_text("utf-8")) == config["profiling"], "Local protocol differs")
            generations = {}
            for name in names:
                if not name.endswith("/COMPLETE.json"):
                    continue
                marker = json.loads(z.read(name))
                base = name.rsplit("/", 1)[0]
                payload = z.read(base + "/payload")
                require(marker["size"] == len(payload) and marker["sha256"] == hashlib.sha256(payload).hexdigest(), "Artifact checksum differs")
                require(marker["identity"] == identity and marker["identity_sha256"] == digest(identity), "Marker identity differs")
                require(marker["generation"] == base.rsplit("/", 1)[-1], "Marker generation differs")
                generations[(base.rsplit("/", 2)[-2], marker["generation"])] = (payload, marker)
            require(generations, "No complete persistent generations")
            for name, value in (("validation", prepare), ("recovery_validation", recover), ("config", config)):
                require(any(json.loads(payload) == value for (key, _), (payload, _) in generations.items()
                    if key == name), "Exported report is not bound to an intact generation: " + name)
            require(any(json.loads(payload)["state"] == "complete" for (key, _), (payload, _) in
                        generations.items() if key == "state"), "Computation never completed")
            sources = [p for (n, g), (p, m) in generations.items() if n == "source_snapshot"]
            require(len(sources) == 1, "Missing source snapshot")
            with zipfile.ZipFile(io.BytesIO(sources[0])) as source:
                hashes = config["source"]["source_sha256"]
                require(set(source.namelist()) == set(hashes), "Source file inventory differs")
                for name, sha in hashes.items():
                    p = PurePosixPath(name)
                    require(not p.is_absolute() and ".." not in p.parts and ":" not in name and "\\" not in name, "Unsafe source path")
                    payload = source.read(name)
                    require(hashlib.sha256(payload).hexdigest() == sha, "Source hash differs")
                    require((REPO / name).read_bytes() == payload, "Current reviewed source differs: " + name)
                require(digest(hashes) == identity["source_tree_sha256"], "Source identity differs")
            originals = {name: value for name, value in prepare["artifacts"].items()}
            # Expose verified figures/tables for the mandatory subsequent visual review.
            from auto_prompt_sam2.artifacts import safe_name
            figure_dir = output / "verified_artifacts"
            figure_dir.mkdir(exist_ok=True)
            for name, entry in originals.items():
                safe_name(name)
                payload, marker = generations[(name, entry["generation"])]
                require(marker["sha256"] == entry["sha256"], "Indexed artifact differs")
                if entry["extension"] == "npz":
                    arrays = read_arrays(payload)
                    require(all(np.isfinite(a).all() for a in arrays.values()), "Nonfinite features")
                    require(arrays["erf"].ndim == 2 and (arrays["erf"] >= 0).all()
                        and np.isclose(arrays["erf"].sum(), 1), "Invalid ERF mass")
                elif entry["extension"] == "png":
                    with Image.open(io.BytesIO(payload)) as image:
                        image.verify()
                if entry["extension"] in ("png", "csv", "json"):
                    (figure_dir / (name + "." + entry["extension"])).write_bytes(payload)
            data_config = json.loads((REPO / "configs/data.yaml").read_text("utf-8"))
            inputs = stage2_inputs(root, data_config)  # manifests/stats only, no raster reads
            for regime, value in inputs.items():
                lineage = identity["regimes"][regime]
                require(digest(value["manifest"]) == lineage["manifest_sha256"] and
                    digest(value["params"]) == lineage["preprocessing_sha256"], "Stage-2 lineage differs")
                planned = train_plan(value["manifest"], config["profiling"]["chips_per_regime"])
                require([r["id"] for r in planned] == lineage["train_ids"], "Training selection differs")
                allowed = {r["id"]: r for r in planned}
                for mode_report in (prepare, recover):
                    entries = [a for a in mode_report["access_log"] if a["regime"] == regime]
                    require(len(entries) == len(planned) * 2, "Access log count differs")
                    require(sorted((a["model"], a["id"]) for a in entries) ==
                        sorted((kind, row["id"]) for kind in ("sam", "dino") for row in planned),
                        "Per-model training access differs")
                    for a in entries:
                        require(a["id"] in allowed and a["role"] == "train" and
                            a["image_sha256"] == allowed[a["id"]]["image_sha256"], "Unpermitted image access")
                for kind in ("sam", "dino"):
                    result = prepare["regimes"][regime][kind]
                    require(set(result["chips"]) == set(allowed), "Feature chips differ")
                    for chip_id, chip in result["chips"].items():
                        require(chip["role"] == "train" and chip["preprocessing_sha256"] == lineage["preprocessing_sha256"], "Feature provenance differs")
                        require(recover["regimes"][regime][kind]["chips"][chip_id]["recovery_features_and_erf_within_tolerance"], "Feature recovery missing")
                    flop = result["flops"]
                    require(sum(r["counted_flops"] or 0 for r in flop["rows"]) == flop["counted_total"] > 0, "FLOP sum differs")
                    require(flop["operations"] and flop["coverage"], "FLOP coverage missing")
                    for timing in [result["full_path_timing"], *result["block_timings"].values()]:
                        if timing.get("status") == "not_executed_static_image_path":
                            continue
                        require(len(timing["samples_ms"]) == timing["repetitions"] == config["profiling"]["repetitions"], "Timing repetitions differ")
                        require(timing["warmup"] == config["profiling"]["warmup"] and timing["batch_size"] == 1 and timing["precision"] == "float32", "Timing setup differs")
                        require(np.isclose(np.median(timing["samples_ms"]), timing["median_ms"])
                            and min(timing["samples_ms"]) > 0, "Timing median differs")
                        require(timing["peak_allocated_bytes"] >= timing["baseline_allocated_bytes"] and
                            timing["peak_reserved_bytes"] >= timing["peak_allocated_bytes"], "Invalid peak memory")
            for r in (prepare, recover):
                require(not r["reserved_access"] and not r["training_executed"] and not r["automatic_prompts_generated"] and not r["final_evaluations"], "Scope violation")
                require(r["tests"]["exit_code"] == 0 and r["environment"]["cuda_available"], "Tests/CUDA missing")
                for kind in ("sam", "dino"):
                    model = r["models"][kind]
                    table = model["parameters_by_block"]
                    require(sum(row["parameters"] for row in table["rows"]) == table["total"] == model["parameters"], "Parameter sum differs")
                    require(table["trainable"] == 0 and model["state_sha256"] == model["state_sha256_after"] and model["hooks_removed"] and model["frozen_unchanged"], "Frozen/hook integrity missing")
                    require(model["checkpoint_sha256"] == prepare["models"][kind]["checkpoint_sha256"] and
                        model["state_sha256"] == prepare["models"][kind]["state_sha256"], "Reload identity differs")
            require(prepare["runtime_boot_id"] != recover["runtime_boot_id"] and recover["different_runtime_boot"], "Independent runtime recovery missing")
            review.update(status="artifact_checks_passed_pending_visual_and_scientific_review",
                run_id=prepare["run_id"], verified_generations=len(generations), verified_artifacts=len(originals),
                next_step="Assistant must inspect actual PNGs and measured tables, contrast hypotheses, archive notebook outputs and write interpretation before stage closure.")
    except Exception as error:
        review["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        (output / "review.json").write_bytes(canonical_json(review))
    return review


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip", type=Path)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--root", type=Path, required=True, help="Persistent stage-2 artifact root for lineage audit")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.zip, args.notebook, args.root, args.output), indent=2))
