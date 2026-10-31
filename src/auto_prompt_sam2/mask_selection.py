"""Classify class-agnostic SAM automatic masks using DINO probabilities only."""
from __future__ import annotations

import numpy as np

from .sam_adapter import require_colab_gpu


def make_grid_generator(predictor, policy):
    require_colab_gpu()
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    if policy['points_per_side'] != 32 or policy['crop_n_layers'] != 0:
        raise ValueError('Stage 5 requires exactly a single 32x32 grid')
    keys = ('points_per_side', 'points_per_batch', 'pred_iou_thresh', 'stability_score_thresh',
            'stability_score_offset', 'box_nms_thresh', 'crop_n_layers', 'min_mask_region_area')
    return SAM2AutomaticMaskGenerator(predictor.model, **{k: policy[k] for k in keys},
                                     output_mode='binary_mask', use_m2m=False)


def select_grid_masks(candidates, probability, input_valid, policy, water_threshold):
    p, valid = np.asarray(probability), np.asarray(input_valid)
    if (p.shape != (512, 512) or valid.shape != p.shape or valid.dtype != bool or
            not np.isfinite(p).all() or ((p < 0) | (p > 1)).any() or not 0 <= water_threshold <= 1):
        raise ValueError('Aligned native probabilities/input validity required')
    union, decisions, accepted = np.zeros(p.shape, bool), [], []
    for index, c in enumerate(candidates):
        mask = np.asarray(c['segmentation'])
        if mask.shape != p.shape or not np.isin(mask, [0, 1]).all():
            raise ValueError('Native binary grid candidate required')
        iou, stability = float(c['predicted_iou']), float(c['stability_score'])
        if not np.isfinite([iou, stability]).all():
            raise ValueError('Finite predicted quality required')
        region = mask.astype(bool) & valid
        mean = float(p[region].mean(dtype=np.float64)) if region.any() else None
        # Match the official AMG strict IoU / inclusive stability filters.
        keep = (mean is not None and iou > policy['pred_iou_thresh'] and
                stability >= policy['stability_score_thresh'] and mean >= water_threshold)
        decisions.append({'candidate_index': index, 'predicted_iou': iou, 'stability_score': stability,
            'input_valid_pixels': int(region.sum()), 'mean_probability': mean, 'accepted': keep})
        if keep:
            union |= region
            accepted.append(index)
    return {'mask': union, 'decisions': decisions, 'accepted_indices': accepted,
            'water_threshold': water_threshold, 'points_per_side': 32, 'grid_positions': 1024}
