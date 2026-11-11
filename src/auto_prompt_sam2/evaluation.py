"""Stage-8 evaluator: gate, frozen protocol, per-set metrics, paired and shift comparisons.

No models and no data access live here. Predictions come from caller-supplied callables (the stage-7
engine in Colab, or synthetic stand-ins in tests). Reserved sets are only reachable after `require_gate`
passes and the caller supplies the SHA-256 recorded for the protocol BEFORE opening them.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from .metrics import aggregate, chip_metrics, scores


class GateClosed(RuntimeError):
    pass


def protocol_sha256(protocol):
    return hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def gate_problems(review):
    """Same conditions as scripts/check_stage8_gate.py, on an already loaded revision JSON."""
    p, bad = review["pair"], []
    if not p.get("both_computational_locks_verified"): bad.append("locks not verified")
    if p.get("external_review") == "pending": bad.append("external/human review pending")
    if not p.get("stage8_authorized"): bad.append("stage8_authorized is false")
    if p.get("reserved_access"): bad.append("reserved_access not empty")
    for name, g in review["regimes"].items():
        if g.get("failures"): bad.append(f"{name}: failures")
        if not g.get("offline_recovery_present"): bad.append(f"{name}: no offline recover on another runtime")
        if g.get("pending"): bad.append(f"{name}: pending items")
    return bad


def require_gate(review, protocol, expected_protocol_sha256):
    bad = gate_problems(review)
    if protocol_sha256(protocol) != expected_protocol_sha256:
        bad.append("protocol hash differs from the one frozen before opening")
    if bad:
        raise GateClosed("Stage 8 closed: " + "; ".join(bad))


def bootstrap_ci(chip_rows_a, chip_rows_b=None, *, resamples=2000, seed=42, level=0.95):
    """Percentile bootstrap over chips of pooled water IoU (difference a-b if b is given, same chips)."""
    def pooled(rows, idx):
        tp = sum(rows[i]["confusion"]["tp"] for i in idx); fp = sum(rows[i]["confusion"]["fp"] for i in idx)
        fn = sum(rows[i]["confusion"]["fn"] for i in idx)
        return tp / (tp + fp + fn) if tp + fp + fn else float("nan")
    n = len(chip_rows_a)
    if chip_rows_b is not None and [r["id"] for r in chip_rows_a] != [r["id"] for r in chip_rows_b]:
        raise ValueError("Paired comparison needs identical chip IDs in identical order")
    rng, vals = np.random.default_rng(seed), []
    for _ in range(resamples):
        idx = rng.integers(0, n, n)
        v = pooled(chip_rows_a, idx) - (pooled(chip_rows_b, idx) if chip_rows_b is not None else 0.0)
        vals.append(v)
    vals = np.array(vals); vals = vals[~np.isnan(vals)]
    lo, hi = np.quantile(vals, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {"lo": float(lo), "hi": float(hi), "resamples": int(len(vals)), "seed": seed, "level": level}


def evaluate(protocol, sets, predictors):
    """sets: {regime: {set_name: [chip dict(id,event,labels,pixel_area_m2)]}}; predictors: {regime: {row: fn(chip)->mask}}.
    Returns per regime/set/row aggregates and the pre-declared comparisons. Chips are never dropped."""
    out = {"protocol_sha256": protocol_sha256(protocol), "regimes": {}}
    for regime, spec in protocol["regimes"].items():
        res = {}
        for set_name, expected in spec["sets"].items():
            chips = sets[regime][set_name]
            if len(chips) != expected:
                raise ValueError(f"{regime}/{set_name}: expected {expected} chips, got {len(chips)}")
            rows = {}
            for row in protocol["rows"]:
                rows[row] = [chip_metrics(c["labels"], predictors[regime][row](c), chip_id=c["id"], event=c["event"],
                                          pixel_area_m2=c["pixel_area_m2"]) for c in chips]
            agg = {r: aggregate(v) for r, v in rows.items()}
            comps = {}
            for comp in protocol["comparisons"]:
                if comp["type"] == "paired_same_chips":
                    d = agg[comp["a"]]["pooled"]["values"]["water_iou"] - agg[comp["b"]]["pooled"]["values"]["water_iou"]
                    comps[comp["id"]] = {"difference_pooled_water_iou": d,
                                         "bootstrap_ci": bootstrap_ci(rows[comp["a"]], rows[comp["b"]], **{
                                             k: protocol["uncertainty"][k] for k in ("resamples", "seed", "level")})}
            res[set_name] = {"chips": len(chips), "rows": agg, "comparisons": comps,
                             "fp_area_m2_no_water_chips": {r: float(sum(x["fp_area_m2"] for x in v if x["no_valid_water"])) for r, v in rows.items()}}
        shift = {}
        seen = res["test_seen"]; held = [k for k in res if k != "test_seen"][0]
        for row in protocol["rows"]:
            a = seen["rows"][row]["pooled"]["values"]["water_iou"]; b = res[held]["rows"][row]["pooled"]["values"]["water_iou"]
            shift[row] = {"seen": a, "heldout": b, "difference_seen_minus_heldout": a - b}
        res["geographic_shift_descriptive"] = shift
        out["regimes"][regime] = res
    return out
