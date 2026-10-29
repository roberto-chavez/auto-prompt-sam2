"""CPU plumbing smoke for scripts/run_e2.py: random SAM weights + synthetic chips + tmp 'Drive'.

SYNTHETIC. Verifies artifact store, run state, evidence files, checkpoint round-trip and
`recover`; says nothing about segmentation quality. Run: python tests/smoke_e2_pipeline.py
(not discovered by unittest; needs the sam2 package and ~1-3 min on CPU).
"""
import copy, importlib.util, json, sys, tempfile, zipfile
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
spec = importlib.util.spec_from_file_location("run_e2", REPO / "scripts/run_e2.py")
run_e2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(run_e2)
from auto_prompt_sam2 import intervention as iv
from auto_prompt_sam2.artifacts import digest
from auto_prompt_sam2.reference_prompts import simulate_points
import auto_prompt_sam2.runtime as rt

GEOM = {"height": 512, "width": 512, "crs": "EPSG:4326", "transform": [0.0009, 0, 10.0, 0, -0.0009, 10.0]}
rows = [{"id": "%s_%d" % (ev, k), "event": ev, "role": role, "geometry": GEOM}
        for role, evs, n in (("train", ("Ghana", "India"), 2), ("val", ("Ghana", "India"), 2)) for ev in evs for k in range(n)]
rows = [dict(r, id="%s_%s_%d" % (r["role"], r["event"], i)) for i, r in enumerate(rows)]


def fake_chip_inputs(row_id):
    rng = np.random.default_rng(abs(hash(row_id)) % 2**32)
    labels = np.zeros((512, 512), np.int64)
    if not row_id.endswith("_3"):  # one chip without water
        for _ in range(2):
            y, x = rng.integers(60, 440, 2); labels[y:y + 50, x:x + 70] = 1
    labels[:6] = -1
    return rng.random((512, 512, 3), dtype=np.float32), labels


def main():
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    tmp = Path(tempfile.mkdtemp())
    config = json.loads((REPO / "configs/e2_intervention.yaml").read_text())
    config["training"]["max_epochs"] = 2
    baseline = json.loads((REPO / "configs/baseline.yaml").read_text())
    manifest = {"regime": "official", "records": rows}
    torch.manual_seed(0)
    model = build_sam2("configs/sam2.1/sam2.1_hiera_t.yaml", None, device="cpu", mode="eval", apply_postprocessing=False).float().eval()
    for p in model.parameters(): p.requires_grad_(False)
    predictor = SAM2ImagePredictor(model, mask_threshold=0.0, max_hole_area=0.0, max_sprinkle_area=0.0)
    from auto_prompt_sam2.sam_adapter import state_fingerprint
    info = {"state_sha256": state_fingerprint(model), "checkpoint_sha256": "synthetic-random-weights", "synthetic": True}
    source, snap = run_e2.source_snapshot()
    ctx = {"root": tmp, "env_config": {"synthetic": True}, "config": config, "baseline": baseline, "data_config": {}, "pins": {"synthetic": True},
           "value": {"manifest": manifest, "params": {"p": 1}}, "source": source, "snapshot": snap,
           "identity": {"stage": config["stage"], "config_sha256": digest(config), "source_tree_sha256": source["source_tree_sha256"],
                        "model": {"synthetic": True}, "regime": "official"}}

    def build(ctx_, runner, plan, keep_image):
        out = []
        for r in plan:
            image, labels = fake_chip_inputs(r["id"])
            prompts = simulate_points(labels, regime="official", chip_id=r["id"], seed=42, policy=config["prompts"])
            chip = iv.build_chip(runner, image, labels, r, prompts, "synthetic")
            if keep_image: chip["image"] = image
            out.append(chip)
        return out

    run_e2.require_colab_gpu = lambda: None
    run_e2.setup = lambda root: ctx
    run_e2.load_model = lambda c, offline: (predictor, iv.Sam2Runner(predictor), info, state_fingerprint)
    run_e2.build_chips = build
    rt.inventory = lambda root, repo: {"synthetic": True}
    # tiny subset sizes keep CPU time short
    config["subset"]["train_chips_per_event"], config["subset"]["val_chips_per_event"] = 2, 2
    report = run_e2.run_e2(tmp, "prepare")
    assert report["status"] == "executed_pending_review", report["status"]
    ev = sorted((tmp / "e2/evidence").glob("*.zip"))
    assert ev, "no evidence zip"
    names = zipfile.ZipFile(ev[0]).namelist()
    for needed in ("report.json", "per_chip_val.csv", "history.csv", "curves.png", "predictions_val.npz", "points_val.json"):
        assert any(n.endswith(needed) for n in names), needed
    assert not any(n.endswith((".pt", ".pth")) or "private" in n for n in names)
    print("PREPARE OK; best epoch", report["best_epoch"], "frozen/tuned", report["hypothesis_check"]["frozen_val_pooled_iou"],
          report["hypothesis_check"]["tuned_best_val_pooled_iou"])
    # fresh decoder weights (as in a new runtime) -> recover must reproduce
    model2 = build_sam2("configs/sam2.1/sam2.1_hiera_t.yaml", None, device="cpu", mode="eval", apply_postprocessing=False).float().eval()
    torch.manual_seed(0)
    model2 = build_sam2("configs/sam2.1/sam2.1_hiera_t.yaml", None, device="cpu", mode="eval", apply_postprocessing=False).float().eval()
    for p in model2.parameters(): p.requires_grad_(False)
    pred2 = SAM2ImagePredictor(model2, mask_threshold=0.0, max_hole_area=0.0, max_sprinkle_area=0.0)
    run_e2.load_model = lambda c, offline: (pred2, iv.Sam2Runner(pred2), info, state_fingerprint)
    rec = run_e2.run_e2(tmp, "recover")
    print("RECOVER", rec["status"], rec["frozen_arm_check"]["passed"], rec["tuned_arm_check"]["passed"])
    assert rec["status"] == "recovery_checks_passed_pending_review"
    print("SMOKE PASSED (synthetic plumbing only)")


if __name__ == "__main__":
    main()
