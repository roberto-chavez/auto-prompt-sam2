"""E2 preliminary intervention: SAM 2.1 mask decoder frozen vs fine-tuned.

Scope: label-simulated points (the stage-3 reference-assisted policy), official regime,
train/val development roles only. This is NOT the automatic DINOv3 -> SAM flow.

The official ``SAM2ImagePredictor.predict/_predict`` are decorated with ``torch.no_grad``
(pinned revision 2b90b9f), so they can never train anything. This module re-implements the
same tensor route (prompt encoder -> mask decoder -> official ``postprocess_masks``) with
autograd enabled ONLY for the mask decoder. Labels supervise the loss; they never choose
prompts for inference (prompt simulation stays in ``reference_prompts``).

Union and gradients: a chip prediction is ``max`` over components of the logits of the mask
selected by PREDICTED IoU (never by true IoU). ``max(logits) > 0`` equals the OR of the binary
masks used by stage 3. A chip may have hundreds of components, so training uses an exact
two-pass scheme: pass 1 (no grad) finds, per pixel, which component wins the max and the
loss gradient w.r.t. the union logits; pass 2 recomputes each chunk with grad and
back-propagates only through the pixels that chunk wins. The gradient equals the one-pass
gradient (unit-tested against plain autograd) with bounded memory.
"""
from __future__ import annotations

import hashlib
import io
import time

import numpy as np

from .artifacts import digest
from .metrics import aggregate, chip_metrics, pixel_areas_m2

DECODER_PREFIX = "sam_mask_decoder."
BLOCKS = ("image_encoder", "sam_prompt_encoder", "sam_mask_decoder")


# --------------------------------------------------------------------------- config / subsets
def validate_config(config, baseline_config):
    """Fixed preliminary contract; prompts must be identical to the stage-3 policy."""
    if config["regime"] != "official" or config["subset"]["reserved_sets"] != "forbidden":
        raise ValueError("E2 preliminary uses the official regime and no reserved sets")
    if config["prompts"] != baseline_config["prompts"]:
        raise ValueError("E2 prompts must be identical to configs/baseline.yaml prompts")
    inf, base = config["inference"], baseline_config["inference"]
    for key in ("precision", "autocast", "mask_threshold", "multimask_output", "selection", "output_hw"):
        if inf[key] != base[key]:
            raise ValueError("E2 inference contract differs from stage 3: " + key)
    t = config["training"]
    if (t["trainable"] != "sam_mask_decoder_only" or t["optimizer"] != "AdamW" or
            type(config["seed"]) is not int or t["lr"] <= 0 or t["max_epochs"] < 1 or
            t["accumulation_chips"] < 1 or t["patience"] < 1 or t["max_grad_norm"] <= 0 or
            inf["component_chunk"] < 1):
        raise ValueError("Invalid E2 training configuration")
    for key in ("statement", "mechanism", "refuted_if", "recorded_before_training"):
        if key not in config["hypothesis"]:
            raise ValueError("Hypothesis field missing: " + key)


def select_subset(rows, per_event):
    """Deterministic, label-free rule: events alphabetical, first `per_event` ids each.

    per_event=None keeps every chip. Returns rows sorted by (event, id).
    """
    ordered = sorted(rows, key=lambda r: (r["event"], r["id"]))
    if per_event is None:
        return ordered
    taken, out = {}, []
    for row in ordered:
        taken[row["event"]] = taken.get(row["event"], 0) + 1
        if taken[row["event"]] <= per_event:
            out.append(row)
    return out


# --------------------------------------------------------------------------- freezing helpers
def block_of(name):
    for block in BLOCKS:
        if name == block or name.startswith(block + "."):
            return block
    return "other"


def parameter_report(model):
    """Total/trainable parameters per block, from requires_grad (not from intent)."""
    table = {}
    for name, p in model.named_parameters():
        row = table.setdefault(block_of(name), {"parameters": 0, "trainable": 0})
        row["parameters"] += p.numel()
        row["trainable"] += p.numel() if p.requires_grad else 0
    return table


def configure_decoder_only(model):
    """Freeze everything, then unfreeze exactly ``sam_mask_decoder``. Returns the report."""
    for p in model.parameters():
        p.requires_grad_(False)
        p.grad = None
    for p in model.sam_mask_decoder.parameters():
        p.requires_grad_(True)
    model.eval()
    report = parameter_report(model)
    outside = [n for n, p in model.named_parameters()
               if p.requires_grad and not n.startswith(DECODER_PREFIX)]
    if outside:
        raise ValueError("Trainable parameters outside the mask decoder: " + ", ".join(outside[:5]))
    if not report["sam_mask_decoder"]["trainable"]:
        raise ValueError("Mask decoder has no trainable parameters")
    return report


def _hash_state(items):
    h = hashlib.sha256()
    for name, tensor in sorted(items):
        value = tensor.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str((tuple(value.shape), value.dtype)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def frozen_fingerprint(model):
    """Hash of every parameter AND buffer outside the mask decoder."""
    return _hash_state((n, t) for n, t in model.state_dict().items() if not n.startswith(DECODER_PREFIX))


def decoder_fingerprint(model):
    return _hash_state((n, t) for n, t in model.state_dict().items() if n.startswith(DECODER_PREFIX))


def gradient_audit(model):
    """After a backward pass: frozen blocks must have NO grad; decoder grads finite."""
    import torch
    outside = [n for n, p in model.named_parameters()
               if not n.startswith(DECODER_PREFIX) and (p.requires_grad or p.grad is not None)]
    with_grad, without_grad, nonfinite, zero = [], [], [], []
    for n, p in model.named_parameters():
        if not n.startswith(DECODER_PREFIX):
            continue
        if p.grad is None:
            without_grad.append(n)
            continue
        with_grad.append(n)
        if not torch.isfinite(p.grad).all():
            nonfinite.append(n)
        elif not bool(p.grad.abs().sum() > 0):
            zero.append(n)
    return {"outside_decoder_with_grad_or_requires_grad": outside,
            "decoder_params_with_grad": len(with_grad), "decoder_params_without_grad": without_grad,
            "decoder_params_nonfinite_grad": nonfinite, "decoder_params_zero_grad": zero,
            "ok": not outside and not nonfinite and bool(with_grad)}


# --------------------------------------------------------------------------- SAM runner
class Sam2Runner:
    """Differentiable counterpart of SAM2ImagePredictor._predict (same tensor route)."""

    def __init__(self, predictor):
        self.predictor = predictor
        self.model = predictor.model
        self.transforms = predictor._transforms
        self.decoder = self.model.sam_mask_decoder

    def encode(self, image):
        """Image encoder under the official no_grad set_image; features cached detached."""
        self.predictor.set_image(image)
        f = self.predictor._features
        return {"image_embed": f["image_embed"].detach().clone(),
                "high_res_feats": [x.detach().clone() for x in f["high_res_feats"]],
                "orig_hw": tuple(int(v) for v in self.predictor._orig_hw[-1])}

    def run(self, feats, coords, labels, *, grad):
        import torch
        device = feats["image_embed"].device
        c = torch.as_tensor(np.asarray(coords, dtype=np.float32), dtype=torch.float, device=device)
        l = torch.as_tensor(np.asarray(labels), dtype=torch.int, device=device)
        c = self.transforms.transform_coords(c, normalize=True, orig_hw=feats["orig_hw"])
        with torch.no_grad():  # prompt encoder is frozen; its outputs are constants for the decoder
            sparse, dense = self.model.sam_prompt_encoder(points=(c, l), boxes=None, masks=None)
            pe = self.model.sam_prompt_encoder.get_dense_pe()
        with torch.enable_grad() if grad else torch.no_grad():
            low, iou, _, _ = self.decoder(
                image_embeddings=feats["image_embed"], image_pe=pe,
                sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                multimask_output=True, repeat_image=c.shape[0] > 1,
                high_res_features=list(feats["high_res_feats"]))
            logits = self.transforms.postprocess_masks(low, feats["orig_hw"])
        return logits, iou


# --------------------------------------------------------------------------- union logic
def ordered_chunks(components, chunk):
    """Deterministic order: (#points, component_id); chunks hold equal #points (no padding)."""
    order = sorted(components, key=lambda c: (len(c["point_labels"]), c["component_id"]))
    chunks, i = [], 0
    while i < len(order):
        n, j = len(order[i]["point_labels"]), i
        while j < len(order) and len(order[j]["point_labels"]) == n and j - i < chunk:
            j += 1
        chunks.append((i, order[i:j]))
        i = j
    return chunks


def selected_logits(runner, feats, comps, *, grad):
    """Logits of the mask with the highest PREDICTED IoU per component (first index on ties)."""
    import torch
    coords = np.asarray([c["point_coords"] for c in comps], dtype=np.float32)
    labels = np.asarray([c["point_labels"] for c in comps])
    logits, iou = runner.run(feats, coords, labels, grad=grad)
    if logits.ndim != 4 or logits.shape[1] != 3 or iou.shape != (len(comps), 3):
        raise ValueError("Three multimask candidates per component expected")
    index = iou.detach().argmax(dim=1)
    sel = logits[torch.arange(len(comps), device=logits.device), index]
    return sel, index, iou.detach()


def union_pass(runner, feats, components, chunk):
    """No-grad pass: union logits, winner component per pixel, per-component decisions."""
    import torch
    union, winner, decisions = None, None, []
    with torch.no_grad():
        for start, comps in ordered_chunks(components, chunk):
            sel, index, iou = selected_logits(runner, feats, comps, grad=False)
            cm, ci = sel.max(dim=0)
            if union is None:
                union, winner = cm.clone(), (ci + start)
            else:
                upd = cm > union
                union = torch.where(upd, cm, union)
                winner = torch.where(upd, ci + start, winner)
            for k, c in enumerate(comps):
                decisions.append({"component_id": c["component_id"], "selected_index": int(index[k]),
                                  "predicted_iou": [float(v) for v in iou[k].cpu()]})
    return union, winner, decisions


def predict_chip(runner, feats, components, chunk, threshold=0.0, hw=(512, 512)):
    """Binary prediction (union of per-component masks). No components -> empty, no SAM call."""
    if not components:
        return np.zeros(hw, bool), {"components": [], "predict_calls": 0}
    union, _, decisions = union_pass(runner, feats, components, chunk)
    return (union > threshold).cpu().numpy(), {"components": decisions, "predict_calls": len(components)}


def bce_dice_loss(logits, labels, cfg):
    """BCE + Dice over VALID pixels only (-1 never contributes). labels: int tensor (-1/0/1)."""
    import torch
    import torch.nn.functional as F
    valid = (labels != -1).float()
    n = valid.sum()
    if not bool(n > 0):
        raise ValueError("Chip without valid pixels has no loss")
    y = (labels == 1).float()
    bce = (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * valid).sum() / n
    p = torch.sigmoid(logits)
    inter = (p * y * valid).sum()
    dice = 1.0 - (2.0 * inter + cfg["dice_eps"]) / ((p * valid).sum() + (y * valid).sum() + cfg["dice_eps"])
    total = cfg["bce_weight"] * bce + cfg["dice_weight"] * dice
    return total, {"bce": float(bce.detach()), "dice": float(dice.detach()), "total": float(total.detach())}


def backward_chip(runner, feats, components, labels, loss_cfg, chunk, scale=1.0):
    """Exact two-pass backward of the soft-union loss. Returns parts or None (no components)."""
    import torch
    if not components:
        return None  # No automatic/simulated component -> empty prediction, no update (counted).
    union, winner, _ = union_pass(runner, feats, components, chunk)
    leaf = union.detach().clone().requires_grad_(True)
    loss, parts = bce_dice_loss(leaf, labels, loss_cfg)
    (loss * scale).backward()
    g = leaf.grad.detach()
    for start, comps in ordered_chunks(components, chunk):
        sel, _, _ = selected_logits(runner, feats, comps, grad=True)
        ids = torch.arange(start, start + len(comps), device=sel.device)[:, None, None]
        ((sel * (winner[None] == ids).to(sel.dtype) * g[None]).sum()).backward()
    return parts


def reference_backward(runner, feats, components, labels, loss_cfg, chunk, scale=1.0):
    """One-pass autograd over every component (memory-hungry). Used only to test backward_chip."""
    import torch
    union = None
    for _, comps in ordered_chunks(components, chunk):
        sel, _, _ = selected_logits(runner, feats, comps, grad=True)
        cm = sel.max(dim=0).values
        union = cm if union is None else torch.maximum(union, cm)
    loss, parts = bce_dice_loss(union, labels, loss_cfg)
    (loss * scale).backward()
    return parts


# --------------------------------------------------------------------------- chip cache / evaluation
def build_chip(runner, image, labels, row, prompts, params_sha, device="cpu"):
    """Encode once (frozen encoder) and keep labels/prompts for loss and metrics."""
    import torch
    comps = [{k: c[k] for k in ("component_id", "point_coords", "point_labels")}
             for c in prompts["components"]]
    feats = runner.encode(image) if comps else None  # no components -> no SAM call at all
    return {"id": row["id"], "event": row["event"], "role": row["role"], "labels": labels,
            "labels_t": torch.as_tensor(np.asarray(labels).astype(np.int64)).to(device),
            "components": comps, "prompts": prompts, "feats": feats,
            "area": pixel_areas_m2(row["geometry"]), "preprocessing_sha256": params_sha}


def evaluate_chips(runner, chips, chunk, *, keep_masks=False):
    """Common evaluator for BOTH arms: same chips, same prompts, same metrics code."""
    records, masks, details = [], {}, {}
    for chip in chips:
        mask, info = predict_chip(runner, chip["feats"], chip["components"], chunk,
                                  hw=tuple(chip["labels"].shape))
        records.append(chip_metrics(chip["labels"], mask, chip_id=chip["id"], event=chip["event"],
                                    pixel_area_m2=chip["area"]))
        details[chip["id"]] = info
        if keep_masks:
            masks[chip["id"]] = mask
    return {"chips": records, "summary": aggregate(records), "masks": masks, "details": details}


# --------------------------------------------------------------------------- training
def train_decoder(runner, train_chips, val_chips, config, *, history=None, start_epoch=1,
                  best=None, optimizer_state=None, on_epoch=None,
                  log=lambda m: print(m, flush=True)):
    """Decoder-only AdamW loop. Epoch 0 (frozen control) is evaluated by the caller.

    `on_epoch(epoch, payload)` persists last/best; resume passes history/best/optimizer_state.
    """
    import torch
    t, chunk = config["training"], config["inference"]["component_chunk"]
    params = [p for p in runner.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=t["lr"], weight_decay=t["weight_decay"], betas=tuple(t["betas"]))
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    history = list(history or [])
    best = dict(best) if best else {"epoch": None, "val_iou": None, "state": None}
    stale = sum(1 for h in history if h["epoch"] >= 1 and (best["epoch"] is None or h["epoch"] > best["epoch"]))
    loss_cfg = t["loss"]
    audits = []
    for epoch in range(start_epoch, t["max_epochs"] + 1):
        started = time.time()
        rng = np.random.Generator(np.random.PCG64([config["seed"], epoch]))
        order = rng.permutation(len(train_chips))
        runner.decoder.train()
        sums, used, no_grad_chips, steps, pending = {"bce": 0.0, "dice": 0.0, "total": 0.0}, 0, [], 0, 0
        optimizer.zero_grad(set_to_none=True)

        def step():
            nonlocal steps, pending
            if not pending:
                return
            audit = gradient_audit(runner.model)
            if not audit["ok"]:
                raise RuntimeError("Gradient audit failed: " + str(audit))
            if not audits:
                audits.append(audit)  # first real step is kept as evidence
            norm = torch.nn.utils.clip_grad_norm_(params, t["max_grad_norm"])
            if not bool(torch.isfinite(norm)):
                raise RuntimeError("Non-finite gradient norm")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            steps, pending = steps + 1, 0

        for i in order:
            chip = train_chips[int(i)]
            parts = backward_chip(runner, chip["feats"], chip["components"], chip["labels_t"],
                                  loss_cfg, chunk, scale=1.0 / t["accumulation_chips"])
            if parts is None:
                no_grad_chips.append(chip["id"])
                continue
            for k in sums:
                sums[k] += parts[k]
            used, pending = used + 1, pending + 1
            if pending == t["accumulation_chips"]:
                step()
        step()  # flush a partial accumulation group
        runner.decoder.eval()
        val = evaluate_chips(runner, val_chips, chunk)
        pooled = val["summary"]["pooled"]["values"]["water_iou"]
        entry = {"epoch": epoch, "train_bce": sums["bce"] / max(used, 1), "train_dice": sums["dice"] / max(used, 1),
                 "train_loss": sums["total"] / max(used, 1), "train_chips_with_gradient": used,
                 "train_chips_without_gradient": len(no_grad_chips), "optimizer_steps": steps,
                 "val_water_iou_pooled": pooled, "val_event_mean_water_iou": val["summary"]["event_mean_water_iou"],
                 "seconds": time.time() - started}
        history.append(entry)
        improved = pooled is not None and (best["val_iou"] is None or pooled > best["val_iou"])
        if improved:  # strict: ties keep the earlier epoch
            best = {"epoch": epoch, "val_iou": pooled,
                    "state": {k: v.detach().cpu().clone() for k, v in runner.decoder.state_dict().items()}}
            stale = 0
        else:
            stale += 1
        log("epoch %d | %s | best_epoch=%s" % (epoch, {k: (round(v, 6) if isinstance(v, float) else v)
                                                       for k, v in entry.items()}, best["epoch"]))
        stop = stale >= t["patience"]
        if on_epoch:
            on_epoch(epoch, {"history": history, "best": best, "optimizer": optimizer.state_dict(),
                             "decoder_state": {k: v.detach().cpu().clone() for k, v in runner.decoder.state_dict().items()},
                             "stopped_early": stop, "finished": stop or epoch == t["max_epochs"]})
        if stop:
            break
    return {"history": history, "best": best, "gradient_audit_first_step": audits[0] if audits else None}


# --------------------------------------------------------------------------- checkpoints
def pack_checkpoint(payload, config_sha256):
    import torch
    buffer = io.BytesIO()
    torch.save({"config_sha256": config_sha256, **payload}, buffer)
    return buffer.getvalue()


def unpack_checkpoint(blob):
    import torch
    return torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)


def load_decoder_state(model, state):
    missing = model.sam_mask_decoder.load_state_dict(state, strict=True)
    return missing


# --------------------------------------------------------------------------- reporting
def paired_deltas(frozen_chips, tuned_chips):
    """Per-chip individual-IoU deltas (tuned - frozen) for chips with defined IoU in both arms."""
    a = {r["id"]: r for r in frozen_chips}
    rows = []
    for r in tuned_chips:
        f = a[r["id"]]
        if f["individual_water_iou"] is None or r["individual_water_iou"] is None:
            continue
        rows.append({"id": r["id"], "event": r["event"], "iou_frozen": f["individual_water_iou"],
                     "iou_tuned": r["individual_water_iou"],
                     "delta": r["individual_water_iou"] - f["individual_water_iou"]})
    d = np.array([x["delta"] for x in rows]) if rows else np.array([])
    return {"chips": rows, "n": len(rows), "improved": int((d > 0).sum()), "worsened": int((d < 0).sum()),
            "unchanged": int((d == 0).sum()),
            "median_delta": float(np.median(d)) if len(d) else None,
            "note": "descriptive only; one seed, small subset, no significance claim"}


def arm_row(name, evaluation):
    s = evaluation["summary"]
    v = s["pooled"]["values"]
    return {"arm": name, "chips": s["chips"], "water_iou_pooled": v["water_iou"],
            "event_mean_water_iou": s["event_mean_water_iou"], "precision": v["precision"],
            "recall": v["recall"], "dice": v["dice"], "miou": v["miou"],
            "per_chip_median": s["per_chip"]["median"], "per_chip_iqr": s["per_chip"]["iqr"],
            "chips_in_individual_iou": s["per_chip"]["included"], "fp_area_m2_total": s["fp_area_m2"]}


def config_digest(config):
    return digest(config)
