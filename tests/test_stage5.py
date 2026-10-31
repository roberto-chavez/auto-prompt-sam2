"""Small synthetic CPU contracts; no model construction, downloads or training."""
import copy
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from auto_prompt_sam2.artifacts import ArtifactStore
from auto_prompt_sam2.automatic_prompts import probability_map, generate_prompts, connected_regions, infer_points
from auto_prompt_sam2.mask_selection import select_grid_masks, make_grid_generator
from auto_prompt_sam2.probe import patch_targets, probabilities, save_probe, load_probe, fit_probe
from auto_prompt_sam2.stage5 import pack, unpack, choose, validate_config, no_label_inference, permitted_file
from auto_prompt_sam2.preprocessing import Coordinates


class PredictorStub:
    def __init__(self):
        self.calls = 0
    def set_image(self, image):
        self.image = image
    def predict(self, **kwargs):
        self.calls += 1
        masks = np.zeros((3, 512, 512), bool)
        masks[1, 100:104, 100:104] = True
        masks[2, 200:204, 200:204] = True
        return masks, np.array([.1, .9, .9]), None


class Stage5Contracts(unittest.TestCase):
    def setUp(self):
        self.pc = json.loads((REPO / 'configs/probe.yaml').read_text())
        self.pp = json.loads((REPO / 'configs/prompts.yaml').read_text())
        self.valid = np.ones((512, 512), bool)

    def portable(self, regime='official'):
        # Hand-written coefficients; this test does not fit anything.
        return {'schema_version': 1, 'classes': [0, 1], 'coef': [1., -2.], 'intercept': .3,
            'scaler_mean': [2., 3.], 'scaler_scale': [1., 2.], 'lineage': {'regime': regime, 'role': 'train'}}

    def test_registered_scope(self):
        validate_config(self.pc, self.pp)
        for field, wrong in [('roles', ['test']), ('regimes', ['official']), ('seed', 1)]:
            bad = dict(self.pc, **{field: wrong})
            with self.assertRaises(ValueError):
                validate_config(bad, self.pp)

    def test_target_boundaries_ignored_and_invalid(self):
        labels = np.zeros((16, 16*6), np.int8)
        labels[:, 16:32] = 1  # fully water
        labels[:, 32:48] = -1
        labels[:, 48:64] = 0
        labels[:8, 48:64] = 1  # exactly 50%
        labels[:4, 64:80] = 1  # ambiguous
        labels[:, 80:96] = -1
        labels[0, 80:96] = 0
        labels[0, 80] = 1  # 1/16 >5% => ignored
        targets, counts = patch_targets(labels, self.pc['targets'])
        np.testing.assert_array_equal(targets, [[0, 1, -1, 1, -1, -1]])
        self.assertEqual(counts['no_valid'], 1)
        self.assertEqual(counts['ignored'], 3)
        exact = np.full((20, 20), -1, np.int8)
        exact.flat[:20] = 0
        exact.flat[0] = 1  # exactly 5% of 20 valid
        self.assertEqual(patch_targets(exact, self.pc['targets'], patch_size=20)[0].item(), 0)

    def test_targets_no_implicit_crop(self):
        with self.assertRaises(ValueError):
            patch_targets(np.zeros((17, 16)), self.pc['targets'])

    def test_probability_geometry_pixel_centers_borders(self):
        p = probability_map(np.array([[0., 1.], [0., 1.]]), (4, 4))
        np.testing.assert_allclose(p[0], [0., .25, .75, 1.])
        self.assertEqual(float(p[-1, -1]), 1.)
        coords = Coordinates((512, 512), (512, 512))
        np.testing.assert_allclose(coords.from_patch([[0, 0], [31, 31]]), [[7.5, 7.5], [503.5, 503.5]])
        patch = np.zeros((32, 32)); patch[2, 3] = 1
        full = probability_map(patch)
        self.assertEqual(full.shape, (512, 512))
        y, x = np.unravel_index(full.argmax(), full.shape)
        self.assertEqual((x, y), (55, 39))  # tied native centers choose top-left
        for bad in (np.array([[np.nan]]), np.array([[-.1]])):
            with self.assertRaises(ValueError):
                probability_map(bad)

    def test_connectivity_order_and_invalid_bridges(self):
        binary = np.eye(3, dtype=bool)
        self.assertEqual(len(list(connected_regions(binary, 4))), 3)
        self.assertEqual(len(list(connected_regions(binary, 8))), 1)
        binary[1, 1] = False
        self.assertEqual(len(list(connected_regions(binary, 8))), 2)

    def test_empty_and_all_input_invalid_no_sam(self):
        stub = PredictorStub()
        image = np.zeros((512, 512, 3), np.float32)
        for p, valid in ((np.zeros((512, 512)), self.valid), (np.ones((512, 512)), ~self.valid)):
            result, prompts = no_label_inference(infer_points, stub, image, p, valid, self.pp, .5)
            self.assertTrue(prompts['empty'])
            self.assertFalse(result['mask'].any())
        self.assertEqual(stub.calls, 0)

    def test_budgets_distances_signs_ties_and_components(self):
        p = np.full((512, 512), .05)
        p[40:90, 50:100] = .9
        p[200:230, 300:330] = .8
        self.valid[40, 50] = False
        prompts = generate_prompts(p, self.valid, self.pp, .5)
        self.assertEqual(len(prompts['components']), 2)
        for c in prompts['components']:
            points = np.asarray(c['point_coords'])
            labels = np.asarray(c['point_labels'])
            self.assertEqual(int((labels == 1).sum()), 3)
            self.assertEqual(int((labels == 0).sum()), 3)
            self.assertTrue(((points >= 0) & (points <= 511)).all())
            self.assertTrue(all(np.linalg.norm(a-b) >= 16 for i, a in enumerate(points) for b in points[i+1:]))
            self.assertTrue(self.valid[points[:, 1], points[:, 0]].all())
            self.assertTrue((p[points[labels == 0, 1], points[labels == 0, 0]] <= .1).all())
        self.assertEqual(prompts['components'][0]['point_coords'][0], [51, 40])
        again = generate_prompts(p, self.valid, self.pp, .5)
        self.assertEqual(prompts['components'], again['components'])

    def test_shortfall_no_relaxation_and_small_drop(self):
        p = np.full((512, 512), .3)
        p[5:9, 5:9] = .9
        p[50, 50] = .9
        result = generate_prompts(p, self.valid, self.pp, .5)
        self.assertEqual(result['dropped_small'], 1)
        c = result['components'][0]
        self.assertEqual(c['positive_shortfall'], 2)
        self.assertEqual(c['negative_shortfall'], 3)
        self.assertEqual(c['point_coords'], [[5, 5]])
        zero_neg = dict(self.pp, negative_per_component=0)
        result = generate_prompts(p, self.valid, zero_neg, .5)
        self.assertEqual(result['components'][0]['point_labels'], [1])

    def test_multimask_predicted_iou_first_tie_and_union(self):
        p = np.zeros((512, 512))
        p[20:24, 20:24] = p[40:44, 40:44] = .9
        stub = PredictorStub()
        result, _ = no_label_inference(infer_points, stub, np.zeros((512, 512, 3), np.float32),
                                      p, self.valid, self.pp, .5)
        self.assertEqual(stub.calls, 2)
        self.assertEqual(int(result['mask'].sum()), 16)
        self.assertTrue(all(c['selected_index'] == 1 for c in result['components']))

    def test_grid_quality_semantics_and_no_input_valid(self):
        a = np.zeros((512, 512), bool); a[:2, :2] = True
        b = np.zeros_like(a); b[4:6, 4:6] = True
        p = np.zeros(a.shape); p[a] = .9
        def c(mask, iou=.9, stability=.96):
            return {'segmentation': mask, 'predicted_iou': iou, 'stability_score': stability}
        candidates = [c(a), c(b), c(a, .8), c(a, stability=.94), c(np.zeros_like(a))]
        result = no_label_inference(select_grid_masks, candidates, p, self.valid, self.pp['grid'], .5)
        self.assertEqual(result['accepted_indices'], [0])
        self.assertEqual(result['grid_positions'], 1024)
        np.testing.assert_array_equal(result['mask'], a)
        result = select_grid_masks([c(a)], p, ~self.valid, self.pp['grid'], .5)
        self.assertFalse(result['mask'].any())
        self.assertIsNone(result['decisions'][0]['mean_probability'])
        self.assertFalse(select_grid_masks([], p, self.valid, self.pp['grid'], .5)['mask'].any())

    def test_portable_probe_scaler_disk_and_regime_guard(self):
        x = np.array([[2., 3.], [4., 4.]])
        probe = self.portable()
        before = probabilities(x, probe, regime='official')
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory))
            identity = {'regime': 'official', 'train_ids': ['synthetic']}
            g = save_probe(store, 'probe', probe, identity)
            loaded = load_probe(store, 'probe', identity, g)
            np.testing.assert_array_equal(before, probabilities(x, loaded, regime='official'))
            with self.assertRaises(FileNotFoundError):
                load_probe(store, 'probe', {'regime': 'leave_paraguay_out'}, g)
        with self.assertRaises(ValueError):
            probabilities(x, probe, regime='leave_paraguay_out')
        bad = copy.deepcopy(probe); bad['scaler_scale'][0] = 0
        with self.assertRaises(ValueError):
            probabilities(x, bad, regime='official')

    def test_prediction_archive_no_pickle(self):
        arrays = {'mask': np.eye(8, dtype=bool), 'probability': np.full((8, 8), .7, np.float32)}
        meta = {'regime': 'official', 'role': 'val', 'count': 3}
        restored, restored_meta = unpack(pack(arrays, meta))
        for key in arrays:
            np.testing.assert_array_equal(restored[key], arrays[key])
        self.assertEqual(meta, restored_meta)

    def test_selection_segmentation_iou_eligibility_and_first_tie(self):
        def row(value, eligible=True):
            return {'eligible': eligible, 'metrics': {'pooled': {'values': {'water_iou': value}}}}
        rows = [row(.2), row(.5), row(.5), row(.9, False), row(None)]
        self.assertIs(choose(rows), rows[1])
        with self.assertRaises(ValueError):
            choose([row(None), row(.9, False)])

    def test_workstation_guards_precede_training_or_model_creation(self):
        # No actual fitting, including tiny synthetic fitting, takes place on PC.
        with patch('auto_prompt_sam2.probe.require_colab_gpu', side_effect=RuntimeError('Colab only')):
            with self.assertRaises(RuntimeError):
                fit_probe(None, None, lineage=None, candidate=None, config=None)
        with patch('auto_prompt_sam2.mask_selection.require_colab_gpu', side_effect=RuntimeError('Colab only')):
            with self.assertRaises(RuntimeError):
                make_grid_generator(None, None)

    def test_reserved_rejected_before_file_io(self):
        manifest = {'regime': 'official', 'records': []}
        with patch('auto_prompt_sam2.stage5.sha256_file', side_effect=AssertionError('No IO')):
            for role in ('test', 'bolivia', 'paraguay_holdout', 'val'):
                with self.assertRaises(ValueError):
                    permitted_file(Path('nonexistent'), manifest, 'reserved', role, 'image')
        manifest = {'regime': 'leave_paraguay_out', 'records': [{'id': 'p', 'event': 'Paraguay', 'role': 'train'}]}
        with self.assertRaises(ValueError):
            permitted_file(Path('nonexistent'), manifest, 'p', 'train', 'image')

    def test_inference_public_boundary_has_no_label_parameter(self):
        for function in (generate_prompts, infer_points, probability_map, select_grid_masks):
            self.assertFalse(any('label' in k for k in inspect.signature(function).parameters))


if __name__ == '__main__':
    unittest.main()
