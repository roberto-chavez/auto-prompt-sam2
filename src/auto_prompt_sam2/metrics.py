"""Common valid-pixel confusion metrics. Undefined values are JSON null + reason."""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def confusion(labels, prediction):
    labels, prediction = np.asarray(labels), np.asarray(prediction)
    if (labels.ndim != 2 or labels.shape != prediction.shape or
            not np.isin(labels, [-1, 0, 1]).all() or
            not np.isin(prediction, [0, 1]).all()):
        raise ValueError("Aligned 2D -1/0/1 labels and binary predictions required")
    valid, water, pred = labels != -1, labels == 1, prediction.astype(bool)
    return {"tp": int((valid & water & pred).sum()),
            "fp": int((valid & ~water & pred).sum()),
            "fn": int((valid & water & ~pred).sum()),
            "tn": int((valid & ~water & ~pred).sum())}


def scores(counts):
    if set(counts) != {"tp", "fp", "fn", "tn"} or any(
            type(v) is not int or v < 0 for v in counts.values()):
        raise ValueError("Nonnegative integer TP/FP/FN/TN required")
    tp, fp, fn, tn = (counts[k] for k in ("tp", "fp", "fn", "tn"))
    values, reasons = {}, {}
    for name, numerator, denominator in (
        ("water_iou", tp, tp + fp + fn), ("precision", tp, tp + fp),
        ("recall", tp, tp + fn), ("dice", 2 * tp, 2 * tp + fp + fn),
        ("nonwater_iou", tn, tn + fp + fn)):
        values[name] = numerator / denominator if denominator else None
        if not denominator:
            reasons[name] = "zero_denominator"
    defined = [values[k] for k in ("water_iou", "nonwater_iou") if values[k] is not None]
    values["miou"] = float(np.mean(defined)) if defined else None
    if not defined:
        reasons["miou"] = "no_defined_class_iou"
    return {"values": values, "undefined": reasons,
            "miou_defined_classes": [k for k in ("water_iou", "nonwater_iou") if values[k] is not None]}


def pixel_areas_m2(geometry):
    """Georeferenced row areas for the audited EPSG:4326 north-up rasters.

    EPSG:6933 is a WGS84 equal-area projection. Never interpret degrees as metres
    or silently substitute the nominal 100 m². Rotated/other CRS fail explicitly.
    """
    from rasterio.warp import transform
    h, w = geometry["height"], geometry["width"]
    a, b, c, d, e, f = geometry["transform"]
    if (geometry["crs"] != "EPSG:4326" or b != 0 or d != 0 or
            not np.isfinite([a, b, c, d, e, f]).all() or a <= 0 or e >= 0 or
            not -180 <= c < c + w * a <= 180 or not -86 < f + h * e < f < 86):
        raise ValueError("Expected audited north-up EPSG:4326 geometry within EPSG:6933 domain")
    xs, _ = transform("EPSG:4326", "EPSG:6933", [c, c + a], [f, f])
    _, ys = transform("EPSG:4326", "EPSG:6933", [c] * (h + 1),
                      (f + np.arange(h + 1) * e).tolist())
    areas = abs(xs[1] - xs[0]) * np.abs(np.diff(ys))
    if not np.isfinite(areas).all() or (areas <= 0).any():
        raise ValueError("Invalid physical pixel area")
    return np.broadcast_to(areas[:, None], (h, w))


def chip_metrics(labels, prediction, *, chip_id, event, pixel_area_m2):
    counts = confusion(labels, prediction)
    area = np.asarray(pixel_area_m2)
    if ((area.ndim != 0 and area.shape != labels.shape) or
            not np.isfinite(area).all() or (area <= 0).any()):
        raise ValueError("Positive finite scalar or aligned m² area required")
    water = counts["tp"] + counts["fn"]
    valid = sum(counts.values())
    excluded = "all_invalid" if not valid else "no_valid_water" if not water else None
    metric = scores(counts)
    fp_mask = (labels == 0) & np.asarray(prediction).astype(bool)
    fp_area = float(counts["fp"] * area) if area.ndim == 0 else float(area[fp_mask].sum())
    return {"id": chip_id, "event": event, "confusion": counts, "scores": metric,
            "valid_pixels": valid, "invalid_pixels": int(labels.size - valid),
            "individual_water_iou": None if excluded else metric["values"]["water_iou"],
            "individual_iou_exclusion": excluded,
            "no_valid_water": bool(valid and not water), "all_invalid": not valid,
            "fp_pixels": counts["fp"], "fp_area_m2": fp_area}


def aggregate(chips):
    if len({r["id"] for r in chips}) != len(chips):
        raise ValueError("Duplicate chip ID in aggregation")
    total = {k: sum(r["confusion"][k] for r in chips) for k in ("tp", "fp", "fn", "tn")}
    events = defaultdict(list)
    for row in chips:
        events[row["event"]].append(row)
    by_event = {}
    for event, rows in sorted(events.items()):
        counts = {k: sum(r["confusion"][k] for r in rows) for k in total}
        by_event[event] = {"chips": len(rows), "confusion": counts, "scores": scores(counts)}
    event_ious = [r["scores"]["values"]["water_iou"] for r in by_event.values()
                  if r["scores"]["values"]["water_iou"] is not None]
    individual = [r["individual_water_iou"] for r in chips if r["individual_water_iou"] is not None]
    quantiles = np.quantile(individual, [0.25, 0.5, 0.75], method="linear").tolist() if individual else [None] * 3
    no_water = [r for r in chips if r["no_valid_water"]]
    return {"chips": len(chips), "confusion": total, "pooled": scores(total), "by_event": by_event,
            "event_mean_water_iou": float(np.mean(event_ious)) if event_ious else None,
            "event_mean_undefined_reason": None if event_ious else "no_defined_event_water_iou",
            "excluded_events": [e for e, r in by_event.items() if r["scores"]["values"]["water_iou"] is None],
            "per_chip": {"included": len(individual), "excluded": len(chips) - len(individual),
                         "q25": quantiles[0], "median": quantiles[1], "q75": quantiles[2],
                         "iqr": quantiles[2] - quantiles[0] if individual else None,
                         "undefined_reason": None if individual else "no_chips_with_valid_water",
                         "quantile_method": "linear"},
            "all_invalid_chips": sum(r["all_invalid"] for r in chips),
            "no_water": {"chips": len(no_water), "fp_pixels": sum(r["fp_pixels"] for r in no_water),
                         "fp_area_m2": sum(r["fp_area_m2"] for r in no_water)},
            "fp_area_m2": sum(r["fp_area_m2"] for r in chips)}
