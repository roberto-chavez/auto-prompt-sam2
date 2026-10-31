"""Stage-5 development I/O and policy validation; labels stay outside inference."""
from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from .artifacts import sha256_file
from .data import development_records, input_rgb


def validate_config(probe, prompts):
    if (probe['regimes'] != ['official', 'leave_paraguay_out'] or probe['roles'] != ['train', 'val'] or
            probe['grid_hw'] != [32, 32] or probe['hidden_size'] != 1024 or probe['seed'] != 42 or
            probe['targets'] != {'water_min': .5, 'nonwater_max': .05, 'minimum_valid_pixels': 1} or
            probe['search'] != {'C': [.1, 1.], 'class_weight': ['none', 'balanced']} or
            probe['ndwi']['included'] or prompts['positive_thresholds'] != [.5, .7] or
            prompts['grid']['water_thresholds'] != [.5, .7] or prompts['grid']['points_per_side'] != 32):
        raise ValueError('Stage-5 preregistered search/scope changed; requires a versioned decision')
    # Exercise policy validation with an empty synthetic map, without a model.
    from .automatic_prompts import generate_prompts
    for threshold in prompts['positive_thresholds']:
        generate_prompts(np.zeros((512, 512)), np.ones((512, 512), bool), prompts, threshold)


def plan(manifest, role):
    rows = sorted(development_records(manifest, regime=manifest['regime'], role=role), key=lambda r: r['id'])
    if not rows or len({r['id'] for r in rows}) != len(rows) or any(
            r['event'] == 'Bolivia' or (manifest['regime'] == 'leave_paraguay_out' and r['event'] == 'Paraguay')
            for r in rows):
        raise ValueError('Invalid development plan')
    return rows


def permitted_file(root, manifest, chip_id, role, kind):
    matches = [r for r in plan(manifest, role) if r['id'] == chip_id]
    if len(matches) != 1 or kind not in ('image', 'label'):
        raise ValueError('Not a unique permitted development input')
    row = matches[0]
    root = Path(root).resolve()
    path = (root / row[kind+'_path']).resolve()
    if not path.is_relative_to(root) or sha256_file(path) != row[kind+'_sha256']:
        raise ValueError('Input path/hash mismatch')
    return path


def read_image(root, manifest, chip_id, role):
    rgb, valid = input_rgb(permitted_file(root, manifest, chip_id, role, 'image'))
    if rgb.shape != (3, 512, 512) or valid.shape != (512, 512):
        raise ValueError('Native RGB geometry required')
    return rgb, valid


def supervised_labels(root, manifest, chip_id, role):
    from .data import read_labels
    labels, _ = read_labels(permitted_file(root, manifest, chip_id, role, 'label'))
    return labels


def no_label_inference(function, *args, **kwargs):
    """Runtime tripwire for the common readers, beyond the label-free API boundary."""
    with patch('auto_prompt_sam2.data.read_labels', side_effect=AssertionError('Labels forbidden in inference')), \
         patch(__name__+'.supervised_labels', side_effect=AssertionError('Labels forbidden in inference')):
        return function(*args, **kwargs)


def pack(arrays, metadata):
    from .artifacts import canonical_json
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays, metadata=np.frombuffer(canonical_json(metadata), np.uint8))
    return buffer.getvalue()


def unpack(payload):
    with np.load(io.BytesIO(payload), allow_pickle=False) as saved:
        return ({k: saved[k].copy() for k in saved.files if k != 'metadata'},
                json.loads(saved['metadata'].tobytes()))


def choose(rows):
    """Pooled validation segmentation IoU; exact tie keeps preregistered order."""
    eligible = [r for r in rows if r['eligible'] and r['metrics']['pooled']['values']['water_iou'] is not None]
    if not eligible:
        raise ValueError('No converged candidate with defined validation IoU')
    return max(eligible, key=lambda r: r['metrics']['pooled']['values']['water_iou'])
