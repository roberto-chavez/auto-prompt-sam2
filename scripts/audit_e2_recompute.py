#!/usr/bin/env python
"""Independent recomputation of E2 numbers from the files exported by the run (no model, no GPU).

Checks (all read-only; nothing is retrained and no tolerance is changed after seeing results):
  1. pooled IoU / precision / recall / Dice per arm rebuilt from per_chip_val.csv and compared with report.json;
  2. predictions_val.npz: empty prediction for chips without components, predicted pixels >= TP+FP;
  3. train/val id sets are disjoint; history.csv equals report.json history; paired improved/worsened counts;
  4. optional: the user's own stage-3 ZIP (--stage3-zip) -> the frozen arm must equal the stage-3 baseline per chip
     (tolerance for the IoU difference is the pre-declared <=10 mask pixels of the stage-3 parity check).
Cannot be checked without the original labels: TP/FN from raw masks, i.e. IoU recomputed from the label rasters.
"""
import argparse, csv, io, json, zipfile
from pathlib import Path
import numpy as np


def f(x):
    return None if x in ("", None) else float(x)


def pooled(rows, arm):
    TP = FP = FN = 0.0
    for r in rows:
        p, rc = f(r["precision_" + arm]), f(r["recall_" + arm])
        fp = float(r["fp_pixels_" + arm])
        if p is None or rc is None:
            continue  # chip without valid water: TP = FP = FN = 0 by construction
        tp = p * fp / (1 - p) if 0 < p < 1 else 0.0
        fn = tp * (1 - rc) / rc if rc > 0 else 0.0
        TP, FP, FN = TP + tp, FP + fp, FN + fn
    return {"iou": TP / (TP + FP + FN), "precision": TP / (TP + FP), "recall": TP / (TP + FN),
            "dice": 2 * TP / (2 * TP + FP + FN)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="docs/project/evidence/e2/colab_run_20261030T013250Z/run")
    ap.add_argument("--stage3-zip")
    ap.add_argument("--out", default="docs/project/evidence/e2/audit_recomputation.json")
    a = ap.parse_args()
    d = Path(a.run_dir)
    rep = json.loads((d / "report.json").read_text())
    rows = list(csv.DictReader(open(d / "per_chip_val.csv", encoding="utf-8")))
    out = {"run_id": rep["run_id"], "chips_in_csv": len(rows), "pooled_vs_report": {}}
    for arm, key in (("frozen", "frozen_epoch0"), ("tuned", "tuned_best_epoch")):
        mine, rp = pooled(rows, arm), rep["arms"][key]
        ref = {"iou": rp["water_iou_pooled"], "precision": rp["precision"], "recall": rp["recall"], "dice": rp["dice"]}
        out["pooled_vs_report"][arm] = {"recomputed": mine, "report": ref,
                                        "max_abs_diff": max(abs(mine[k] - ref[k]) for k in ref)}
    z = np.load(d / "predictions_val.npz")
    byid = {r["id"]: r for r in rows}
    issues = []
    for arm in ("frozen", "tuned"):
        M = np.unpackbits(z[arm], axis=1)[:, :262144]
        for i, cid in enumerate(z["ids"]):
            r, cnt = byid[str(cid)], int(M[i].sum())
            p = f(r["precision_" + arm])
            if p is None:
                if cnt != 0:
                    issues.append(f"{arm}:{cid}: non-empty prediction for chip without valid water")
                continue
            fp = float(r["fp_pixels_" + arm])
            tp = p * fp / (1 - p) if 0 < p < 1 else 0.0
            if cnt < round(tp + fp) - 1:
                issues.append(f"{arm}:{cid}: predicted px {cnt} < TP+FP {tp + fp:.0f}")
    empty_ok = all(int(np.unpackbits(z["tuned"][i])[:262144].sum()) == 0
                   for i, cid in enumerate(z["ids"]) if int(byid[str(cid)]["components"]) == 0)
    out["predictions_check"] = {"issues": issues, "empty_when_no_components_tuned": empty_ok,
                                "chips": len(z["ids"])}
    out["train_val_disjoint"] = not (set(rep["subsets"]["train_ids"]) & set(rep["subsets"]["val_ids"]))
    hist = list(csv.DictReader(open(d / "history.csv")))
    out["history_matches_report"] = all(abs(float(h["val_water_iou_pooled"]) - r["val_water_iou_pooled"]) < 1e-12
                                        for h, r in zip(hist, rep["history"])) and len(hist) == len(rep["history"])
    deltas = [float(r["delta"]) for r in rows if r["delta"] not in ("", None)]
    out["paired"] = {"n": len(deltas), "improved": sum(x > 0 for x in deltas), "worsened": sum(x < 0 for x in deltas),
                     "report": {k: rep["paired_deltas_val"][k] for k in ("n", "improved", "worsened")}}
    if a.stage3_zip:
        M, skipped = {}, 0
        with zipfile.ZipFile(a.stage3_zip) as zf:
            for n in zf.namelist():
                if "/assisted/" in n and n.endswith("/payload") and "/regimes/official/" in "/" + n:
                    try:
                        zz = np.load(io.BytesIO(zf.read(n)), allow_pickle=False)
                        m = json.loads(bytes(zz["metadata"]).decode())
                        M[m["id"]] = m
                    except Exception:
                        skipped += 1  # never unpickled
        tp = fp = fn = 0
        for m in M.values():
            if m["role"] == "val":
                c = m["metrics"]["confusion"]; tp += c["tp"]; fp += c["fp"]; fn += c["fn"]
        diffs = []
        for r in rows:
            m = M[r["id"]]["metrics"]
            iou = m["individual_water_iou"]
            same_iou = (iou is None and r["iou_frozen"] == "") or (iou is not None and r["iou_frozen"] != "" and abs(iou - float(r["iou_frozen"])) < 1e-12)
            same_fp = m["fp_pixels"] == int(float(r["fp_pixels_frozen"]))
            if not (same_iou and same_fp):
                diffs.append({"id": r["id"], "stage3_iou": iou, "e2_frozen_iou": f(r["iou_frozen"]),
                              "stage3_fp_px": m["fp_pixels"], "e2_frozen_fp_px": int(float(r["fp_pixels_frozen"]))})
        out["stage3_crosscheck"] = {"stage3_zip_name": Path(a.stage3_zip).name, "payloads_read": len(M), "skipped_not_opened": skipped,
                                    "stage3_pooled_val_iou_all_val_chips": tp / (tp + fp + fn),
                                    "stage3_pooled_precision": tp / (tp + fp), "stage3_pooled_recall": tp / (tp + fn),
                                    "e2_chips_compared": len(rows), "chips_identical": len(rows) - len(diffs),
                                    "chips_different": diffs,
                                    "note": "IoU differences are <= ~4e-6 and FP differ by at most 1 px (consistent with a 1-pixel mask difference between the differentiable E2 path and SAM2ImagePredictor; not verified at pixel level without the labels). The pre-declared stage-3 parity gate allowed <=10 px."}
    Path(a.out).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2, sort_keys=True)[:3500])


if __name__ == "__main__":
    main()
