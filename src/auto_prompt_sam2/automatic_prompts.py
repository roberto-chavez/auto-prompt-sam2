"""Label-free probability geometry and deterministic points. No label I/O."""
from __future__ import annotations

import numpy as np


def probability_map(patches, output_hw=(512, 512)):
    """Bilinear at pixel centres; replicate boundary values (no padding/cropping)."""
    p = np.asarray(patches)
    if p.ndim != 2 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Finite 2D probabilities required')
    h, w = output_hw
    if min(h, w) < 1:
        raise ValueError('Positive output geometry required')
    yy = np.clip((np.arange(h) + .5) * p.shape[0] / h - .5, 0, p.shape[0] - 1)
    xx = np.clip((np.arange(w) + .5) * p.shape[1] / w - .5, 0, p.shape[1] - 1)
    y0, x0 = yy.astype(int), xx.astype(int)
    y1, x1 = np.minimum(y0 + 1, p.shape[0]-1), np.minimum(x0 + 1, p.shape[1]-1)
    fy, fx = (yy-y0)[:, None], (xx-x0)[None, :]
    return ((1-fy)*((1-fx)*p[y0[:, None], x0] + fx*p[y0[:, None], x1]) +
            fy*((1-fx)*p[y1[:, None], x0] + fx*p[y1[:, None], x1])).astype(np.float32)


def connected_regions(binary, connectivity):
    """Row-major component IDs, independent of platform/library traversal order."""
    # Independent implementation avoids importing the label-assisted simulator.
    from collections import deque
    binary = np.asarray(binary)
    if binary.ndim != 2 or binary.dtype != bool or connectivity not in (4, 8):
        raise ValueError('Boolean 2D mask and connectivity 4/8 required')
    seen = np.zeros(binary.shape, bool)
    offsets = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
               if (dy or dx) and (connectivity == 8 or abs(dy)+abs(dx) == 1)]
    h, w = binary.shape
    for y, x in np.argwhere(binary):
        if seen[y, x]:
            continue
        seen[y, x] = True
        queue, pixels = deque([(int(y), int(x))]), []
        while queue:
            y, x = queue.popleft()
            pixels.append((y, x))
            for dy, dx in offsets:
                ny, nx = y+dy, x+dx
                if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    queue.append((ny, nx))
        yield np.asarray(sorted(pixels), dtype=np.int32)


def _peaks(pool, probability, budget, distance, existing=(), low=False):
    if not budget or not len(pool):
        return []
    values = probability[pool[:, 0], pool[:, 1]]
    order = np.lexsort((pool[:, 1], pool[:, 0], values if low else -values))
    # Greedy maximum after radius suppression; includes flat plateaus and edges.
    chosen = []
    for index in order:
        y, x = pool[index]
        point = [int(x), int(y)]
        if all((x-a)**2 + (y-b)**2 >= distance**2 for a, b in [*existing, *chosen]):
            chosen.append(point)
            if len(chosen) == budget:
                break
    return chosen


def generate_prompts(probability, input_valid, policy, positive_threshold):
    """input_valid comes solely from the image's nodata/masks/finite values."""
    p, valid = np.asarray(probability), np.asarray(input_valid)
    kp, kn, d = (policy[k] for k in ('positive_per_component', 'negative_per_component', 'minimum_distance_px'))
    if (p.shape != (512, 512) or valid.shape != p.shape or valid.dtype != bool or
            not np.isfinite(p).all() or ((p < 0) | (p > 1)).any() or
            not 0 <= policy['negative_threshold'] < positive_threshold <= 1 or
            type(kp) is not int or kp < 1 or type(kn) is not int or kn < 0 or
            not np.isfinite(d) or d < 0 or policy['minimum_component_pixels'] < 1):
        raise ValueError('Invalid automatic geometry, probabilities or budget')
    negative_pool = np.argwhere(valid & (p <= policy['negative_threshold']))
    components, dropped = [], 0
    component_map = np.zeros(p.shape, np.int32)
    for cid, pool in enumerate(connected_regions(valid & (p >= positive_threshold), policy['connectivity']), 1):
        if len(pool) < policy['minimum_component_pixels']:
            dropped += 1
            continue
        component_map[pool[:, 0], pool[:, 1]] = cid
        positive = _peaks(pool, p, kp, d)
        negative = _peaks(negative_pool, p, kn, d, positive, low=True)
        components.append({'component_id': cid, 'pixels': len(pool), 'point_coords': positive+negative,
            'point_labels': [1]*len(positive)+[0]*len(negative),
            'positive_shortfall': kp-len(positive), 'negative_shortfall': kn-len(negative)})
    return {'components': components, 'component_map': component_map, 'dropped_small': dropped,
        'positive_threshold': positive_threshold, 'policy': policy,
        'empty': not components, 'validity_source': 'image_only'}


def infer_points(predictor, image, probability, input_valid, policy, positive_threshold):
    from .sam_adapter import segment_components
    prompts = generate_prompts(probability, input_valid, policy, positive_threshold)
    result = segment_components(predictor, image, prompts['components'])
    result['mask'] &= input_valid  # Never clip with label validity.
    return result, prompts
