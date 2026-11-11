#!/usr/bin/env python
"""Consolidate the REAL validation results already stored in the repository (no model, no GPU, no new run).

Reads: stage-3 evidence ZIP (assisted baseline, label-derived points), stage-5 selection JSON (DINOv3 probe points and
grid), stage-6 curves (decoder trained on automatic prompts) and the E2 report. Writes a JSON + Markdown table whose
numbers are copied from those files; each row records its source so it can be re-checked. It never opens reserved sets
(none are present in these files) and does not decide anything: it only collects.
"""
import argparse, io, json, zipfile
from pathlib import Path
import numpy as np

EV = Path("docs/project/evidence")
REG = ("official", "leave_paraguay_out")


def stage3(regime):
    zpath = EV / "stage3/colab_run_20261029T134519Z/20261029T134519Z_1894f1a3ae_ec3f3bcc.zip"
    tp = fp = fn = n = 0
    with zipfile.ZipFile(zpath) as zf:
        for name in zf.namelist():
            if f"regimes/{regime}/" in name and "/assisted/" in name and name.endswith("/payload"):
                try:
                    z = np.load(io.BytesIO(zf.read(name)), allow_pickle=False)
                    m = json.loads(bytes(z["metadata"]).decode())
                except Exception:
                    continue  # never unpickled
                if m.get("role") != "val":
                    continue
                c = m["metrics"]["confusion"]; tp += c["tp"]; fp += c["fp"]; fn += c["fn"]; n += 1
    return {"chips": n, "water_iou": tp / (tp + fp + fn), "precision": tp / (tp + fp), "recall": tp / (tp + fn),
            "dice": 2 * tp / (2 * tp + fp + fn), "source": str(zpath)}


def vals(d):
    v = d["values"]
    return {"water_iou": v["water_iou"], "precision": v["precision"], "recall": v["recall"], "dice": v["dice"]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="docs/project/evidence/e3/consolidated_results")
    a = ap.parse_args()
    out = {"note": "Validation-only results copied from stored evidence. Rows are NOT all comparable: see 'caveat'.", "regimes": {}}
    for r in REG:
        s5 = json.load(open(EV / f"stage5/colab_run_20261029T194653Z/{r}_selection.json"))["selected"]
        g5 = json.load(open(EV / f"stage5/colab_run_20261029T194653Z/{r}_grid_selection.json"))["selected"]
        cur = json.load(open(EV / f"stage6/colab_review_20261101/{r}/curves.json"))
        series = [(e["epoch"], vals(e["val_metrics"]["pooled"])) for e in cur]
        best = max(series, key=lambda x: x[1]["water_iou"])
        out["regimes"][r] = {
            "stage3_assisted_label_points_1pos_1neg_frozen_decoder": {**stage3(r)},
            "stage5_dinov3_probe_points_frozen_decoder": {**vals(s5["metrics"]["pooled"]), "chips": s5["metrics"]["chips"],
                "selected": {"candidate": s5["candidate"], "positive_threshold": s5["positive_threshold"]},
                "chips_with_zero_components": sum(1 for p in s5["point_counts"] if p["components"] == 0),
                "positive_points": sum(p["positive_points"] for p in s5["point_counts"]),
                "negative_points": sum(p["negative_points"] for p in s5["point_counts"]),
                "source": f"stage5/colab_run_20261029T194653Z/{r}_selection.json"},
            "stage5_sam_grid_dinov3_scored_frozen_decoder": {**vals(g5["metrics"]["pooled"]), "threshold": g5["water_threshold"],
                "source": f"stage5/colab_run_20261029T194653Z/{r}_grid_selection.json"},
            "stage6_decoder_trained_on_automatic_points_best_epoch": {"epoch": best[0], **best[1],
                "epochs_ran": len(series), "last_epoch": {"epoch": series[-1][0], **series[-1][1]},
                "source": f"stage6/colab_review_20261101/{r}/curves.json"},
        }
    e2 = json.load(open(EV / "e2/colab_run_20261030T013250Z/run/report.json"))["arms"]
    out["e2_assisted_40_chip_subset_official"] = {
        "frozen": {k: e2["frozen_epoch0"][m] for k, m in (("water_iou", "water_iou_pooled"), ("precision", "precision"), ("recall", "recall"), ("dice", "dice"))},
        "tuned_best_epoch1": {k: e2["tuned_best_epoch"][m] for k, m in (("water_iou", "water_iou_pooled"), ("precision", "precision"), ("recall", "recall"), ("dice", "dice"))},
        "source": "e2/colab_run_20261030T013250Z/run/report.json (40 val chips, single seed)"}
    out["caveat"] = ("stage3 uses label-derived components with 1+/1- per component; stage5 uses probe-derived components with up to 3+/3-. "
                     "Budgets and component sources differ, so stage3 vs stage5 is descriptive, NOT the controlled cost-of-the-prompt contrast "
                     "(stage 7 A1 with a matched 3+/3- assisted reference is prepared but not executed). stage6 changes the decoder as well as the prompts. "
                     "All numbers are validation numbers; best epoch/threshold were selected on the same validation.")
    Path(a.out).with_suffix(".json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["| Régimen | Fila (validación) | IoU agua | Precisión | Recall | Dice |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for r, d in out["regimes"].items():
        for k, v in d.items():
            lab = {"stage3_assisted_label_points_1pos_1neg_frozen_decoder": "Etapa 3: puntos desde etiquetas (1+/1− por componente), decoder congelado",
                   "stage5_dinov3_probe_points_frozen_decoder": "Etapa 5: puntos automáticos DINOv3 (≤3+/3−), decoder congelado",
                   "stage5_sam_grid_dinov3_scored_frozen_decoder": "Etapa 5: malla SAM 32×32 puntuada con DINOv3, decoder congelado",
                   "stage6_decoder_trained_on_automatic_points_best_epoch": f"Etapa 6: decoder entrenado con puntos automáticos, mejor época {v.get('epoch')}"}[k]
            lines.append(f"| {r} | {lab} | {v['water_iou']:.6f} | {v['precision']:.6f} | {v['recall']:.6f} | {v['dice']:.6f} |")
    Path(a.out).with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
