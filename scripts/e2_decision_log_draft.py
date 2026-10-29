"""Draft the RESULTS block of the E2 decision-log entry from a finished run's report.json.

Only measured values are copied. Interpretation is left as PENDING for the team.
Usage: python scripts/e2_decision_log_draft.py <evidence_dir> [output.md]
"""
import json
import sys
from pathlib import Path


def fmt(v):
    return "n/d" if v is None else ("%.4f" % v if isinstance(v, float) else str(v))


def draft(report):
    arms, check, hist = report["arms"], report["hypothesis_check"], report["history"]
    frozen, tuned = arms["frozen_epoch0"], arms["tuned_best_epoch"]
    lines = ["### Resultados medidos (E2) — run `%s`" % report["run_id"], "",
             "_Generado automáticamente desde `report.json`. Solo cifras medidas; la interpretación es del equipo._", "",
             "| Brazo | IoU agua agrupado | IoU medio por evento | Precisión | Recall | Dice | Mediana IoU por chip |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for a in (frozen, tuned):
        lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (a["arm"], fmt(a["water_iou_pooled"]), fmt(a["event_mean_water_iou"]),
                     fmt(a["precision"]), fmt(a["recall"]), fmt(a["dice"]), fmt(a["per_chip_median"])))
    lines += ["", "Mejor época (por IoU agrupado en val): **%s**; IoU agrupado de la última época: %s." % (
        report["best_epoch"], fmt(arms["tuned_last_epoch_val_pooled_iou"])), "",
        "Criterio prefijado de refutación: %s" % check["criterion"],
        "Resultado mecánico del criterio: **%s** (congelado %s vs ajustado %s). Interpretación: **PENDIENTE**." % (
            check["mechanical_outcome"], fmt(check["frozen_val_pooled_iou"]), fmt(check["tuned_best_val_pooled_iou"])), "",
        "| Época | BCE | Dice | Pérdida | IoU val agrupado | s |", "| --- | --- | --- | --- | --- | --- |"]
    for h in hist:
        lines.append("| %d | %s | %s | %s | %s | %.1f |" % (h["epoch"], fmt(h["train_bce"]), fmt(h["train_dice"]),
                     fmt(h["train_loss"]), fmt(h["val_water_iou_pooled"]), h["seconds"]))
    d = report["paired_deltas_val"]
    lines += ["", "Chips con IoU definido en ambos brazos: %d (mejoran %d, empeoran %d, igual %d; mediana del delta %s). Descriptivo, una semilla, sin significancia." % (
        d["n"], d["improved"], d["worsened"], d["unchanged"], fmt(d["median_delta"])), "",
        "Congelamiento verificado: bloques congelados sin cambio = %s; decoder cambió = %s." % (
            report["freezing"]["frozen_blocks_unchanged"], report["freezing"]["decoder_changed"]),
        "Subconjuntos: train %s, val %s." % (json.dumps(report["subsets"].get("train_stats")), json.dumps(report["subsets"].get("val_stats"))), "",
        "**Interpretación, fallos y limitaciones observadas: PENDIENTE (completar tras revisar curvas y ejemplos).**", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    folder = Path(sys.argv[1])
    text = draft(json.loads((folder / "report.json").read_text("utf-8")))
    target = Path(sys.argv[2]) if len(sys.argv) > 2 else folder / "e2_resultados_para_cuaderno_decisiones.md"
    target.write_text(text, "utf-8")
    print(text)
