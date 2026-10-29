"""PIL-only figures/tables for E2 (the pinned Colab environment has no matplotlib/pandas).

Nothing here computes a metric: it draws numbers and masks that were already measured.
"""
from __future__ import annotations

import csv
import io

import numpy as np
from PIL import Image, ImageDraw

PANEL = 256


def csv_text(rows, columns):
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in columns})
    return out.getvalue()


def _png(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _axes(draw, box, title, xs, ymin, ymax, xlabel="epoch"):
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline="black")
    draw.text((x0, y0 - 16), title, fill="black")
    for v in (ymin, (ymin + ymax) / 2, ymax):
        y = y1 - (v - ymin) / max(ymax - ymin, 1e-12) * (y1 - y0)
        draw.line([(x0 - 3, y), (x0, y)], fill="black")
        draw.text((x0 - 52, y - 6), "%.3g" % v, fill="black")
    for x in xs:
        px = x0 + (x - min(xs)) / max(max(xs) - min(xs), 1) * (x1 - x0)
        draw.line([(px, y1), (px, y1 + 3)], fill="black")
        draw.text((px - 4, y1 + 5), str(x), fill="black")
    draw.text(((x0 + x1) / 2 - 14, y1 + 20), xlabel, fill="black")


def _line(draw, box, xs, ys, ymin, ymax, color, dash=False):
    x0, y0, x1, y1 = box
    pts = [(x0 + (x - min(xs)) / max(max(xs) - min(xs), 1) * (x1 - x0),
            y1 - (y - ymin) / max(ymax - ymin, 1e-12) * (y1 - y0)) for x, y in zip(xs, ys) if y is not None]
    if dash and len(pts) >= 2:
        for a, b in zip(pts[::2], pts[1::2]):
            draw.line([a, b], fill=color, width=2)
    elif len(pts) >= 2:
        draw.line(pts, fill=color, width=2)
    for p in pts:
        draw.ellipse([p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3], fill=color)


def curves_png(history, frozen_val_iou):
    """Left: training BCE/Dice/total per epoch. Right: val pooled/event-mean IoU + frozen control."""
    epochs = [h["epoch"] for h in history]
    image = Image.new("RGB", (900, 360), "white")
    d = ImageDraw.Draw(image)
    lbox, rbox = (70, 40, 420, 290), (520, 40, 870, 290)
    series = [("train_bce", "#d62728"), ("train_dice", "#1f77b4"), ("train_loss", "black")]
    lo = min(min(h[k] for h in history) for k, _ in series)
    hi = max(max(h[k] for h in history) for k, _ in series)
    _axes(d, lbox, "Train loss per epoch (BCE red, Dice blue, total black)", epochs, lo, hi)
    for k, c in series:
        _line(d, lbox, epochs, [h[k] for h in history], lo, hi, c)
    vals = [h["val_water_iou_pooled"] for h in history] + [h["val_event_mean_water_iou"] for h in history]
    vals = [v for v in vals if v is not None] + ([frozen_val_iou] if frozen_val_iou is not None else [])
    vlo, vhi = min(vals), max(vals)
    pad = max((vhi - vlo) * 0.1, 1e-3)
    vlo, vhi = vlo - pad, vhi + pad
    _axes(d, rbox, "Validation water IoU (pooled blue, event-mean red)", epochs, vlo, vhi)
    _line(d, rbox, epochs, [h["val_water_iou_pooled"] for h in history], vlo, vhi, "#1f77b4")
    _line(d, rbox, epochs, [h["val_event_mean_water_iou"] for h in history], vlo, vhi, "#d62728")
    if frozen_val_iou is not None:
        y = rbox[3] - (frozen_val_iou - vlo) / (vhi - vlo) * (rbox[3] - rbox[1])
        d.line([(rbox[0], y), (rbox[2], y)], fill="gray", width=2)
        d.text((rbox[0] + 4, y - 14), "frozen decoder (epoch 0) pooled IoU = %.4f" % frozen_val_iou, fill="gray")
    d.text((10, 335), "Preliminary E2 subset, one seed. Measured values only; no smoothing.", fill="black")
    return _png(image)


def _overlay(base, prediction, labels):
    """TP green, FP red, FN blue on a dimmed image; invalid pixels gray."""
    out = (base * 0.55 * 255).astype(np.uint8).copy()
    valid, water, pred = labels != -1, labels == 1, prediction.astype(bool)
    out[valid & water & pred] = (0, 200, 0)
    out[valid & ~water & pred] = (220, 0, 0)
    out[valid & water & ~pred] = (0, 90, 255)
    out[~valid] = (128, 128, 128)
    return out


def example_png(image_unit_hwc, labels, points, frozen_mask, tuned_mask, title, text_frozen, text_tuned):
    """image | label | simulated points | frozen overlay | tuned overlay (identical points, same evaluator)."""
    labels = np.asarray(labels)
    label_rgb = np.zeros(labels.shape + (3,), np.uint8)
    label_rgb[labels == 1] = (255, 255, 255)
    label_rgb[labels == -1] = (128, 128, 128)
    rgb = (np.clip(image_unit_hwc, 0, 1) * 255).astype(np.uint8)
    pts = Image.fromarray(rgb)
    pd = ImageDraw.Draw(pts)
    for comp in points:
        for (x, y), lab in zip(comp["point_coords"], comp["point_labels"]):
            color = (0, 255, 0) if lab == 1 else (255, 0, 0)
            pd.ellipse([x - 3, y - 3, x + 3, y + 3], outline=color, width=2)
    panels = [("Sentinel-2 RGB", Image.fromarray(rgb)), ("label: water white, invalid gray", Image.fromarray(label_rgb)),
              ("simulated points (+ green, - red)", pts),
              ("frozen decoder\n" + text_frozen, Image.fromarray(_overlay(image_unit_hwc, frozen_mask, labels))),
              ("fine-tuned decoder\n" + text_tuned, Image.fromarray(_overlay(image_unit_hwc, tuned_mask, labels)))]
    canvas = Image.new("RGB", (PANEL * len(panels), PANEL + 70), "white")
    d = ImageDraw.Draw(canvas)
    d.text((4, 2), title + "   overlay: TP green, FP red, FN blue", fill="black")
    for i, (name, img) in enumerate(panels):
        canvas.paste(img.resize((PANEL, PANEL), Image.Resampling.NEAREST), (i * PANEL, 70))
        d.text((i * PANEL + 4, 18), name, fill="black")
    return _png(canvas)


# --------------------------------------------------------------------------- persistence helpers
def encode_eval(evaluation, ids):
    """Masks (bit-packed) + metrics/decisions JSON in one NPZ (no pickle)."""
    from .artifacts import canonical_json
    masks = np.stack([np.packbits(np.asarray(evaluation["masks"][i], bool).ravel()) for i in ids])
    meta = {"ids": list(ids), "summary": evaluation["summary"], "chips": evaluation["chips"],
            "details": evaluation["details"]}
    buffer = io.BytesIO()
    np.savez_compressed(buffer, masks=masks, metadata=np.frombuffer(canonical_json(meta), dtype=np.uint8))
    return buffer.getvalue()


def decode_eval(payload, hw=(512, 512)):
    import json
    with np.load(io.BytesIO(payload), allow_pickle=False) as saved:
        packed, meta = saved["masks"].copy(), json.loads(saved["metadata"].tobytes())
    n = hw[0] * hw[1]
    masks = {i: np.unpackbits(packed[k])[:n].reshape(hw).astype(bool) for k, i in enumerate(meta["ids"])}
    return masks, meta
