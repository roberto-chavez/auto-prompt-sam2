"""Stage-4 train-only data, static SAM path and persistent visual artifacts."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .artifacts import sha256_file
from .data import development_records, input_rgb
from .preprocessing import Coordinates


def train_plan(manifest, count):
    if type(count) is not int or count < 1:
        raise ValueError("Positive train chip count required")
    rows = sorted(development_records(manifest, regime=manifest["regime"], role="train"),
                  key=lambda r: r["id"])
    if len({r["id"] for r in rows}) != len(rows) or len(rows) < count:
        raise ValueError("Unique and sufficient train examples required")
    if any(r["event"] == "Bolivia" or
           (manifest["regime"] == "leave_paraguay_out" and r["event"] == "Paraguay") for r in rows):
        raise ValueError("Reserved event in train")
    return rows[:count]


def read_train_image(root, manifest, chip_id):
    # Never reads labels; validate membership and hash before opening RGB.
    matches = [r for r in train_plan(manifest, len(development_records(
        manifest, regime=manifest["regime"], role="train"))) if r["id"] == chip_id]
    if len(matches) != 1:
        raise ValueError("Not a permitted training chip")
    row = matches[0]
    root = Path(root).resolve()
    path = (root / row["image_path"]).resolve()
    if not path.is_relative_to(root) or sha256_file(path) != row["image_sha256"]:
        raise ValueError("Training image path/hash mismatch")
    rgb, valid = input_rgb(path)
    if rgb.shape != (3, 512, 512):
        raise ValueError("Native 512 RGB required")
    return row, rgb, valid


def sam_points(tensor, fixed_prompts):
    import torch
    coords = Coordinates((512, 512), (1024, 1024)).points(fixed_prompts["point_coords"])
    return (torch.tensor(coords, dtype=torch.float32, device=tensor.device)[None],
            torch.tensor(fixed_prompts["point_labels"], dtype=torch.int32, device=tensor.device)[None])


def sam_static_path(model, tensor, fixed_prompts, *, preloaded_points=None):
    """Same tensor route as pinned SAM2ImagePredictor, with input autograd intact.

    CPU radiometry/normalization and H2D have already happened. No model cache.
    Outputs all three logits at native 512, not a semantic water prediction.
    """
    import torch
    import torch.nn.functional as F
    backbone = model.forward_image(tensor)
    _, vision, _, sizes = model._prepare_backbone_features(backbone)
    if model.directly_add_no_mem_embed:
        vision[-1] = vision[-1] + model.no_mem_embed
    spatial = [v.permute(1, 2, 0).reshape(1, -1, *hw) for v, hw in zip(vision, sizes)]
    points, labels = preloaded_points if preloaded_points is not None else sam_points(tensor, fixed_prompts)
    sparse, dense = model.sam_prompt_encoder(points=(points, labels), boxes=None, masks=None)
    low, scores, _, _ = model.sam_mask_decoder(image_embeddings=spatial[-1],
        image_pe=model.sam_prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense, multimask_output=True, repeat_image=False,
        high_res_features=spatial[:-1])
    return F.interpolate(low.float(), (512, 512), mode="bilinear", align_corners=False), scores


def arrays_payload(arrays):
    output = io.BytesIO()
    if any(not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("Nonfinite feature/ERF arrays")
    np.savez_compressed(output, **arrays)
    return output.getvalue()


def read_arrays(payload):
    with np.load(io.BytesIO(payload), allow_pickle=False) as saved:
        return {k: saved[k].copy() for k in saved.files}


def figure_png(rgb_unit, grids, erf, channels):
    """No fitted projection. Native RGB + fixed feature channels + log ERF overlay.

    Display scaling is per panel min/max, not a fitted normalizer or feature cache.
    Raw tensors and ERF are persisted separately for quantitative inspection.
    """
    panels = [("RGB train", np.moveaxis(rgb_unit, 0, -1))]
    for name, grid in grids.items():
        for channel in channels:
            panels.append((f"{name} c{channel}", grid[0, channel]))
    panels.append(("ERF log10 mass (relative color)", np.log10(np.maximum(erf, 1e-12))))
    width, tile, columns = 260, 256, 3
    canvas = Image.new("RGB", (width * columns, 290 * ((len(panels) + columns - 1) // columns)), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (title, values) in enumerate(panels):
        lo, hi = float(values.min()), float(values.max())
        if values.ndim == 2:
            unit = (values - lo) / max(hi - lo, 1e-12)
            values = np.stack((unit, np.sqrt(unit), 1 - unit), axis=-1)
        image = Image.fromarray((np.clip(values, 0, 1) * 255).astype(np.uint8)).resize(
            (tile, tile), Image.Resampling.NEAREST)
        x, y = (i % columns) * width, (i // columns) * 290
        canvas.paste(image, (x, y + 30))
        draw.text((x + 2, y + 1), title, fill="black")
        draw.text((x + 2, y + 14), f"min={lo:.3g} max={hi:.3g}", fill="black")
    # Add an actual native-coordinate ERF overlay alongside its model-grid panel.
    native = Image.fromarray((np.moveaxis(rgb_unit, 0, -1) * 255).astype(np.uint8))
    log = np.log10(np.maximum(erf, 1e-12))
    unit = (log - log.min()) / max(float(log.max() - log.min()), 1e-12)
    heat = Image.fromarray((np.stack((unit, np.zeros_like(unit), 1 - unit), -1) * 255).astype(np.uint8))
    heat = heat.resize(native.size, Image.Resampling.BILINEAR)
    overlay = Image.blend(native, heat, .45)
    out = Image.new("RGB", (canvas.width, canvas.height + 290), "white")
    out.paste(canvas, (0, 0))
    out.paste(overlay.resize((256, 256)), (0, canvas.height + 30))
    ImageDraw.Draw(out).text((2, canvas.height + 5), "ERF overlay on native train RGB", fill="black")
    output = io.BytesIO()
    out.save(output, format="PNG")
    return output.getvalue()
