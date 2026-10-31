"""Portable binary probe; fitting is guarded before touching arrays or models."""
from __future__ import annotations

import json
import numpy as np

from .artifacts import canonical_json
from .sam_adapter import require_colab_gpu


def patch_targets(labels, policy, patch_size=16):
    labels = np.asarray(labels)
    if (labels.ndim != 2 or not np.isin(labels, [-1, 0, 1]).all() or
            any(n % patch_size for n in labels.shape) or
            not 0 <= policy['nonwater_max'] < policy['water_min'] <= 1 or
            not 1 <= policy['minimum_valid_pixels'] <= patch_size ** 2):
        raise ValueError('Invalid patch targets/geometry')
    h, w = labels.shape
    def blocks(a):
        return a.reshape(h // patch_size, patch_size, w // patch_size, patch_size).sum((1, 3))
    valid, water = blocks(labels != -1), blocks(labels == 1)
    fraction = np.divide(water, valid, out=np.zeros_like(water, dtype=float), where=valid > 0)
    targets = np.full(valid.shape, -1, np.int8)
    enough = valid >= policy['minimum_valid_pixels']
    targets[enough & (fraction <= policy['nonwater_max'])] = 0
    targets[enough & (fraction >= policy['water_min'])] = 1
    return targets, {'water': int((targets == 1).sum()), 'nonwater': int((targets == 0).sum()),
        'ignored': int((targets == -1).sum()), 'no_valid': int((valid == 0).sum()),
        'valid_pixel_count': valid.tolist()}


def fit_probe(features, targets, *, lineage, candidate, config):
    require_colab_gpu()  # Includes even scaler fitting. Never fit locally.
    import torch
    import torch.nn.functional as F
    if (lineage['role'] != 'train' or lineage['regime'] not in config['regimes'] or
            len(set(lineage['ids'])) != len(lineage['ids']) or not lineage['ids']):
        raise ValueError('Unique train-only lineage required')
    x = torch.as_tensor(features, dtype=torch.float32, device='cuda')
    y = torch.as_tensor(targets, dtype=torch.float32, device='cuda')
    if (x.ndim != 2 or y.shape != (x.shape[0],) or not torch.isfinite(x).all() or
            set(y.unique().tolist()) != {0., 1.}):
        raise ValueError('Finite features with both train classes required')
    mean = x.mean(0)
    scale = x.std(0, correction=0)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    x = (x - mean) / scale
    n = len(y)
    if candidate['class_weight'] == 'balanced':
        weight = torch.where(y == 1, n / (2 * (y == 1).sum()), n / (2 * (y == 0).sum()))
    elif candidate['class_weight'] == 'none':
        weight = torch.ones_like(y)
    else:
        raise ValueError('Unsupported class weight')
    w = torch.zeros(x.shape[1], device='cuda', requires_grad=True)
    b = torch.zeros((), device='cuda', requires_grad=True)
    s = config['solver']
    optimizer = torch.optim.LBFGS([w, b], lr=1., max_iter=s['max_iter'],
        tolerance_grad=s['tolerance_grad'], tolerance_change=s['tolerance_change'],
        history_size=s['history_size'], line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad()
        loss = (F.binary_cross_entropy_with_logits(x @ w + b, y, reduction='none') * weight).mean()
        loss = loss + w.square().sum() / (2 * candidate['C'] * n)
        loss.backward()
        return loss
    optimizer.step(closure)
    loss = float(closure().detach())
    grad = max(float(w.grad.abs().max()), float(b.grad.abs()))
    iterations = int(optimizer.state[w].get('n_iter', 0))
    converged = grad <= s['tolerance_grad'] or iterations < s['max_iter']
    if not np.isfinite([loss, grad]).all():
        raise ValueError('Nonfinite probe optimization')
    return {'schema_version': 1, 'classes': [0, 1], 'coef': w.detach().cpu().tolist(),
        'intercept': float(b.detach()), 'scaler_mean': mean.cpu().tolist(), 'scaler_scale': scale.cpu().tolist(),
        'lineage': lineage, 'candidate': candidate, 'config': config,
        'fit': {'samples': n, 'iterations': iterations, 'objective': loss,
            'gradient_inf_norm': grad, 'converged': converged,
            'convergence_rule': 'gradient_tolerance_or_lbfgs_stop_before_max_iter',
            'class_counts': {'0': int((y == 0).sum()), '1': int((y == 1).sum())}}}


def probabilities(features, probe, *, regime):
    if probe['lineage']['regime'] != regime or probe['classes'] != [0, 1]:
        raise ValueError('Cross-regime probe or wrong class order')
    x = np.asarray(features, dtype=np.float64)
    w, mean, scale = (np.asarray(probe[k], dtype=np.float64) for k in
                      ('coef', 'scaler_mean', 'scaler_scale'))
    if (x.shape[-1] != len(w) or w.shape != mean.shape or w.shape != scale.shape or
            (scale <= 0).any() or not all(np.isfinite(a).all() for a in (x, w, mean, scale)) or
            not np.isfinite(probe['intercept'])):
        raise ValueError('Incompatible/nonfinite portable probe')
    z = np.clip(((x - mean) / scale) @ w + probe['intercept'], -700, 700)
    return (1 / (1 + np.exp(-z))).astype(np.float32)


def save_probe(store, name, probe, identity):
    if identity['regime'] != probe['lineage']['regime']:
        raise ValueError('Probe identity mismatch')
    return store.publish(name, canonical_json(probe), identity)


def load_probe(store, name, identity, generation=None):
    probe = json.loads(store.read(name, identity, generation)[0])
    if probe['lineage']['regime'] != identity['regime']:
        raise ValueError('Probe regime mismatch')
    probabilities(np.zeros((1, len(probe['coef']))), probe, regime=identity['regime'])
    return probe
