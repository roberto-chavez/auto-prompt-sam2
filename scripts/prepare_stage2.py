"""Acquire/audit/prepare stage 2 only. Uses the existing persistent ArtifactStore."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import torch
from auto_prompt_sam2.artifacts import ArtifactStore, canonical_json, digest, sha256_file, utc_now
from auto_prompt_sam2.data import (acquire, audit, manifests, development_records,
                                   input_rgb, read_labels, get_bytes)
from auto_prompt_sam2.preprocessing import (fit_preprocessing, save_preprocessing,
    load_preprocessing, prepare, Coordinates, resize_categorical)
from auto_prompt_sam2.runtime import load_config, persistent_root, provenance, source_files


EVIDENCE1 = REPO / "docs/project/evidence/stage1/colab_run_20261028T221055Z"
SOURCE_URLS = {
    "dataset_readme": "https://raw.githubusercontent.com/cloudtostreet/Sen1Floods11/b9e460dabe70230111a3c3d36cd0c5ce03606ee9/README.md",
    "dataset_structure": "https://raw.githubusercontent.com/cloudtostreet/Sen1Floods11/b9e460dabe70230111a3c3d36cd0c5ce03606ee9/docs/README.md",
    "sam_transforms": "https://raw.githubusercontent.com/facebookresearch/sam2/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/utils/transforms.py",
    "dino_readme": "https://raw.githubusercontent.com/facebookresearch/dinov3/6876159a11b4df116f30f667f8c9888617df0751/README.md",
}


def verify_stage1():
    """Teammate's stage-1 Colab evidence is an ANTECEDENT only.

    It documents a different person's run and never accredits the current user's execution.
    A missing/changed antecedent is recorded, not fatal: E2 neither needs DINOv3 nor that run.
    """
    try:
        return _verify_stage1_antecedent()
    except (OSError, KeyError, ValueError) as error:
        return {"status": "antecedent_unavailable_not_gating", "reason": type(error).__name__ + ": " + str(error)[:200],
                "accredits_this_execution": False,
                "license": "pending_publisher_reply; user_explicitly_requested_stage2_acquisition"}


def _verify_stage1_antecedent():
    markers = json.loads((EVIDENCE1 / "artifact_markers.json").read_text("utf-8"))
    result = {}
    for name in ("environment", "recovery_validation"):
        path = EVIDENCE1 / (name + ".json")
        sha = sha256_file(path)
        matches = [m for m in markers[name] if m["sha256"] == sha and m["size"] == path.stat().st_size
                   and digest(m["identity"]) == m["identity_sha256"]]
        if not matches:
            raise ValueError("Stage-1 evidence is not intact: " + name)
        result[name] = {"sha256": sha, "identity_sha256": matches[0]["identity_sha256"]}
    recovery = json.loads((EVIDENCE1 / "recovery_validation.json").read_text("utf-8"))
    if not (recovery["persistence_smoke"] == "passed" and recovery["rejected_incomplete"]
            and recovery["rejected_mismatch"] and recovery["seed_probe_reproduced"]
            and not recovery["resource_blockers"]):
        raise ValueError("Stage-1 recovery contract failed")
    access = json.loads((EVIDENCE1 / "access.json").read_text("utf-8"))
    access_markers = json.loads((EVIDENCE1 / "access_markers.json").read_text("utf-8"))
    sha = sha256_file(EVIDENCE1 / "access.json")
    if not any(m["sha256"] == sha and m["size"] == (EVIDENCE1 / "access.json").stat().st_size
               and digest(m["identity"]) == m["identity_sha256"]
               and m["identity_sha256"] == result["environment"]["identity_sha256"] for m in access_markers):
        raise ValueError("Stage-1 access marker mismatch")
    if not all(v["access"]["authorized"] and v["metadata_matches"] for v in access["models"].values()):
        raise ValueError("Stage-1 model access failed")
    result["access"] = {"sha256": sha}
    result["status"] = "antecedent_verified"
    result["accredits_this_execution"] = False
    result["license"] = "pending_publisher_reply; user_explicitly_requested_stage2_acquisition"
    return result


def sources(root, offline):
    report = {}
    for name, url in SOURCE_URLS.items():
        path = root / "raw/sources" / (name + ".txt")
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            if offline:
                raise FileNotFoundError("Missing official source snapshot: " + name)
            path.write_bytes(get_bytes(url))
        report[name] = {"url": url, "sha256": sha256_file(path), "path": path.relative_to(root).as_posix()}
    data = (root / report["dataset_readme"]["path"]).read_text("utf-8")
    sam = (root / report["sam_transforms"]["path"]).read_text("utf-8")
    dino = (root / report["dino_readme"]["path"]).read_text("utf-8")
    if not all(s in data for s in ("1: B2", "2: B3", "3: B4", "scaled by 10000")):
        raise ValueError("Official band documentation changed")
    if not all(s in sam for s in ("[0.485, 0.456, 0.406]", "[0.229, 0.224, 0.225]")):
        raise ValueError("SAM statistics differ")
    if not all(s in dino for s in ("(0.430, 0.411, 0.296)", "(0.213, 0.156, 0.143)")):
        raise ValueError("DINO satellite statistics differ")
    return report


def source_snapshot(root):
    info = provenance(REPO)
    paths = set(source_files(REPO))
    paths.update(REPO / p for p in ("configs/data.yaml", "requirements-stage2.txt", "notebooks/02_datos_y_splits.ipynb"))
    info["source_sha256"] = {p.relative_to(REPO).as_posix(): sha256_file(p) for p in sorted(paths)}
    info["source_tree_sha256"] = digest(info["source_sha256"])
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            archive.write(path, path.relative_to(REPO).as_posix())
    # Contains current uncommitted source bytes as well as commit/dirty provenance.
    ArtifactStore(root / "stage2").publish("source_snapshot", buffer.getvalue(),
                                          {"source_tree_sha256": info["source_tree_sha256"]})
    return info


def assert_reload(root, manifest, config, folder, generation, row):
    """New interpreter reads saved parameters/data; no inherited Python state."""
    params = load_preprocessing(folder, manifest, config, generation)
    rgb, valid = input_rgb(root / row["image_path"])
    checks = {}
    for model in ("sam", "dino"):
        output = prepare(rgb, valid, params, model=model)["pixel_values"].numpy()
        checks[model] = hashlib.sha256(output.tobytes()).hexdigest()
    context = {"root": str(root), "manifest": manifest, "config": config,
               "folder": str(folder), "generation": generation, "row": row, "expected": checks}
    context_path = folder / "reload_context.json"
    context_path.write_bytes(canonical_json(context))
    code = """
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import torch
torch.set_num_threads(1)
from auto_prompt_sam2.data import input_rgb
from auto_prompt_sam2.preprocessing import load_preprocessing,prepare
c=json.loads(Path(sys.argv[2]).read_text('utf-8'))
p=load_preprocessing(c['folder'],c['manifest'],c['config'],c['generation'])
rgb,valid=input_rgb(Path(c['root'])/c['row']['image_path'])
for model,expected in c['expected'].items():
 value=prepare(rgb,valid,p,model=model)['pixel_values'].numpy()
 assert hashlib.sha256(value.tobytes()).hexdigest()==expected
print('new_interpreter_exact_equality')
"""
    subprocess.run([sys.executable, "-I", "-c", code, str(REPO / "src"), str(context_path)], check=True, capture_output=True, text=True)
    return {"chip": row["id"], "tensor_sha256": checks, "new_interpreter_exact_equality": True}


def run(root, *, offline=False, evidence=None):
    torch.set_num_threads(1)
    stage1 = verify_stage1()
    config = json.loads((REPO / "configs/data.yaml").read_text("utf-8"))
    primary = sources(root, offline)
    raw_store = ArtifactStore(root / "raw")
    acquisition = (json.loads(raw_store.read("acquisition", {"dataset": "Sen1Floods11-v1.1"})[0])
                   if offline else acquire(root))
    print("Auditing 446 image/label pairs (structure only on reserved sets)", flush=True)
    rows, audit_report = audit(root, acquisition, event_aliases=config["event_metadata_aliases"])
    regimes = manifests(rows)
    report = {"stage": "phase1_stage2", "created_at": utc_now(), "persistent_root": str(root),
              "scope": ("colab" if sys.platform == "linux" and Path("/content").is_dir() else "local_cpu")
                       + "_data_and_preprocessing_no_models", "stage1": stage1,
              "sources": primary, "raw_objects": len(acquisition["objects"]),
              "raw_bytes": sum(o["size"] for o in acquisition["objects"]),
              "acquisition_sha256": digest(acquisition), "audit": audit_report, "regimes": {},
              "packages": {p: importlib.metadata.version(p) for p in
                           ("numpy", "torch", "torchvision", "rasterio", "affine", "attrs", "click", "cligj", "click-plugins", "pyparsing")},
              "source": source_snapshot(root), "config_sha256": digest(config),
              "models_executed": False, "final_evaluations": [], "reserved_pixel_caches": [],
              "colab_runtime_restart_verified": False,
              "closure_pending": ["Colab runtime restart and recovery evidence"],
              "retained_salvedades": ["publisher license reply; stage1 exception retained"]}
    if evidence:
        evidence.mkdir(parents=True, exist_ok=True)
        (evidence / "acquisition.json").write_bytes(canonical_json(acquisition))
        (evidence / "sources.json").write_bytes(canonical_json(primary))
    for regime, manifest in regimes.items():
        print("Preparing", regime, manifest["counts"], flush=True)
        base = root / "regimes" / regime
        identity = {"regime": regime, "manifest_sha256": digest(manifest)}
        manifest_generation = ArtifactStore(base / "manifests").publish("manifest", canonical_json(manifest), identity)
        folder = base / "preprocessing"
        # Reuse intact train statistics; never silently recompute a changed config/split.
        try:
            params = load_preprocessing(folder, manifest, config)
            reused = True
        except FileNotFoundError:
            params = fit_preprocessing(root, manifest, config)
            reused = False
        generation = save_preprocessing(folder, params)
        params = load_preprocessing(folder, manifest, config, generation)
        masks_store = ArtifactStore(folder / "validity")
        masks = []
        for role in ("train", "val"):
            for row in development_records(manifest, regime=regime, role=role):
                rgb, valid = input_rgb(root / row["image_path"])
                labels, label_valid = read_labels(root / row["label_path"])
                out = io.BytesIO()
                np.savez_compressed(out, input_valid=valid, label_valid=label_valid)
                key = dict(identity, image_sha256=row["image_sha256"], label_sha256=row["label_sha256"])
                try:
                    payload, marker = masks_store.read(row["id"], key)
                    gen = marker["generation"]
                except FileNotFoundError:
                    gen = masks_store.publish(row["id"], out.getvalue(), key)
                    payload = masks_store.read(row["id"], key, gen)[0]
                restored = np.load(io.BytesIO(payload))
                if not np.array_equal(restored["input_valid"], valid) or not np.array_equal(restored["label_valid"], labels != -1):
                    raise ValueError("Validity mask roundtrip failed")
                masks.append({"id": row["id"], "role": role, "generation": gen, "identity": key})
        coords = {model: Coordinates((512, 512), (config[model]["size"], config[model]["size"])).metadata()
                  for model in ("sam", "dino")}
        ArtifactStore(folder).publish("coordinates", canonical_json(coords), dict(identity, config_sha256=digest(config)))
        ArtifactStore(folder).publish("validity_index", canonical_json(masks), identity)
        sample = development_records(manifest, regime=regime, role="train")[0]
        reload = assert_reload(root, manifest, config, folder, generation, sample)
        xy = np.array([[0, 0], [511, 511], [128.25, 200.75]])
        for model in ("sam", "dino"):
            c = Coordinates((512, 512), (config[model]["size"],) * 2)
            if not np.allclose(c.points(c.points(xy), inverse=True), xy, atol=config["coordinate_tolerance_px"], rtol=0):
                raise ValueError("Point inverse failed")
            if not np.allclose(c.from_patch(c.to_patch(xy)), xy, atol=config["coordinate_tolerance_px"], rtol=0):
                raise ValueError("Patch inverse failed")
        if not set(np.unique(resize_categorical(np.array([[-1, 0], [1, -1]], dtype=np.int16), (19, 23)))).issubset({-1, 0, 1}):
            raise ValueError("Categorical resize introduced labels")
        report["regimes"][regime] = {"counts": manifest["counts"], "manifest_sha256": digest(manifest),
            "manifest_generation": manifest_generation, "preprocessing_sha256": digest(params),
            "parameter_generation": generation, "parameters_reused": reused,
            "upper_dn": params["upper_dn"], "upper_reflectance": params["upper_reflectance"],
            "fit_train_chips": len(params["source_ids"]), "validity_masks": len(masks),
            "reload": reload, "coordinate_and_label_checks": "passed"}
        if evidence:
            (evidence / (regime + "_manifest.json")).write_bytes(canonical_json(manifest))
            (evidence / (regime + "_preprocessing.json")).write_bytes(canonical_json(params))
            (evidence / (regime + "_coordinates.json")).write_bytes(canonical_json(coords))
    report["status"] = "data_preprocessing_checks_passed"
    store = ArtifactStore(root / "stage2")
    report_identity = {"config_sha256": digest(config), "acquisition_sha256": digest(acquisition)}
    generation = store.publish("validation", canonical_json(report), report_identity)
    if json.loads(store.read("validation", report_identity, generation)[0]) != report:
        raise ValueError("Report recovery mismatch")
    if evidence:
        (evidence / "validation.json").write_bytes(canonical_json(report))
    print(json.dumps({"status": report["status"], "regimes": report["regimes"]}, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args()
    if args.root:
        os.environ["AUTO_PROMPT_PERSIST_ROOT"] = str(args.root.resolve())
    root = persistent_root(load_config(REPO), colab="google.colab" in sys.modules or Path("/content").exists())
    run(root, offline=args.offline, evidence=args.evidence)


if __name__ == "__main__":
    main()
