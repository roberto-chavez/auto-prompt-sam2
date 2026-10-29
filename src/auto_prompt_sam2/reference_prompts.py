"""Label-assisted simulation only. This module must never enter automatic inference."""
from __future__ import annotations

from collections import deque

import numpy as np

from .artifacts import digest


def validate_labels(labels):
    labels = np.asarray(labels)
    if labels.ndim != 2 or not np.isin(labels, [-1, 0, 1]).all():
        raise ValueError("2D labels in -1/0/1 required")
    return labels


def connected_water(labels, connectivity=8):
    """All true-water components, including singletons; invalid pixels cannot bridge."""
    labels = validate_labels(labels)
    if connectivity not in (4, 8):
        raise ValueError("Connectivity must be 4 or 8")
    water = labels == 1
    seen = np.zeros(labels.shape, bool)
    h, w = labels.shape
    offsets = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
               if (dy or dx) and (connectivity == 8 or abs(dy) + abs(dx) == 1)]
    for y, x in np.argwhere(water):
        if seen[y, x]:
            continue
        seen[y, x] = True
        queue, pixels = deque([(int(y), int(x))]), []
        while queue:
            py, px = queue.popleft()
            pixels.append((px, py))
            for dy, dx in offsets:
                ny, nx = py + dy, px + dx
                if 0 <= ny < h and 0 <= nx < w and water[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    queue.append((ny, nx))
        # Candidate order is independent of traversal and scipy/platform versions.
        yield np.array(sorted(pixels, key=lambda xy: (xy[1], xy[0])), dtype=np.int32)


def _sample(pool, budget, rng, distance, existing):
    chosen = []
    for i in rng.permutation(len(pool)):
        if len(chosen) == budget:
            break
        xy = pool[i]
        if all(np.sum((xy.astype(float) - p) ** 2) >= distance ** 2
               for p in [*existing, *chosen]):
            chosen.append(xy.tolist())
    return chosen


def simulate_points(labels, *, regime, chip_id, seed, policy):
    """Returns serializable native (x,y) prompts, never an oracle-selected mask.

    Positives are label==1, negatives label==0 (outside ALL water), both valid.
    Missing negative candidates do not discard a component or relax spacing.
    """
    labels = validate_labels(labels)
    kp, kn = policy["positive_per_component"], policy["negative_per_component"]
    distance = policy["minimum_distance_px"]
    if (type(kp) is not int or kp < 1 or type(kn) is not int or kn < 0 or
            not np.isfinite(distance) or distance < 0):
        raise ValueError("Invalid point budget/distance")
    chip_seed = int(digest({"seed": seed, "regime": regime, "chip_id": chip_id})[:16], 16)
    rng = np.random.Generator(np.random.PCG64(chip_seed))
    negative_pool = np.argwhere(labels == 0)[:, ::-1]
    components = []
    for i, pool in enumerate(connected_water(labels, policy["connectivity"]), start=1):
        positive = _sample(pool, kp, rng, distance, [])
        negative = _sample(negative_pool, kn, rng, distance, positive)
        components.append({"component_id": i, "water_pixels": len(pool),
                           "point_coords": positive + negative,
                           "point_labels": [1] * len(positive) + [0] * len(negative),
                           "positive_shortfall": kp - len(positive),
                           "negative_shortfall": kn - len(negative)})
    return {"branch": "reference_assisted_not_deployable", "regime": regime,
            "chip_id": chip_id, "seed": seed, "chip_seed": chip_seed,
            "rng": "numpy.PCG64", "coordinate_order": "x,y",
            "original_hw": list(labels.shape), "policy": dict(policy),
            "valid_pixels": int((labels != -1).sum()),
            "water_pixels": int((labels == 1).sum()), "components": components,
            "no_water": len(components) == 0,
            "assistance": "true_components_and_absence_of_water_known_from_labels"}
