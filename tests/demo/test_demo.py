"""CPU-only demo contracts: synthetic arrays and recorders, never neural models."""
import builtins
import copy
from contextlib import contextmanager, nullcontext
import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import Mock, patch

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_origin

from auto_prompt_sam2.artifacts import sha256_file
from auto_prompt_sam2.demo_config import (artifact_contract, inspect_artifacts, load_demo_config,
                                         require_remote_runtime, verify_file)
from auto_prompt_sam2.demo_images import (AUTO_LAYOUT, DISPLAY_LAYOUT, RGB_LAYOUT, S2_LAYOUT, DemoImage,
                                         draw_points, load_image, validate_points)
from auto_prompt_sam2.demo_backend import best_decoder_state, predict_manual, DemoBackend
from auto_prompt_sam2.demo_ui import DemoController, build_demo


@contextmanager
def forbid_model_imports():
    original_import = builtins.__import__

    def checked_import(name, *args, **kwargs):
        if name.partition(".")[0] in {"torch", "sam2"}:
            raise AssertionError(f"Unexpected model import: {name}")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", new=checked_import):
        yield


def synthetic_image():
    return DemoImage(np.full((512, 512, 3), .4, np.float32), np.ones((512, 512), bool), {})


class DemoContracts(unittest.TestCase):
    def setUp(self):
        self.config = load_demo_config(environ={})
        self.report, self.params = artifact_contract(self.config)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def raster(self, array, nodata=None):
        path = self.root / "chip.tif"
        with rasterio.open(path, "w", driver="GTiff", width=array.shape[2], height=array.shape[1],
                           count=array.shape[0], dtype=array.dtype, nodata=nodata, crs="EPSG:4326",
                           transform=from_origin(-1, 10, .0001, .0001)) as ds:
            ds.write(array)
        return path

    def test_preprocessing_matches_real_report_and_detects_change(self):
        result = inspect_artifacts(self.config)
        self.assertTrue(result["preprocessing_matches_report"])
        self.assertEqual(result["decoder"], "not_configured")
        altered = copy.deepcopy(self.params)
        altered["upper_dn"] += 1
        path = self.root / "params.json"
        path.write_text(json.dumps(altered))
        with self.assertRaisesRegex(ValueError, "no coincide"):
            artifact_contract(dict(self.config, preprocessing_path=str(path)))

    def test_file_integrity_missing_size_and_hash(self):
        path = self.root / "payload"
        with self.assertRaises(FileNotFoundError):
            verify_file(path, "0" * 64)
        path.write_bytes(b"synthetic bytes, not weights")
        sha = sha256_file(path)
        self.assertEqual(verify_file(path, sha, path.stat().st_size), path)
        with self.assertRaisesRegex(ValueError, "Tama"):
            verify_file(path, sha, 1)
        with self.assertRaisesRegex(ValueError, "SHA"):
            verify_file(path, "0" * 64)

    def test_environment_and_remote_revision_validation(self):
        config = load_demo_config(environ={"DEMO_MODE": "inference", "DEMO_ALLOW_DOWNLOAD": "1",
                                          "DEMO_DECODER_CHECKPOINT": "weights/best/payload"})
        self.assertTrue(config["allow_download"])
        for env in ({"DEMO_ALLOW_DOWNLOAD": "yes"}, {"DEMO_MODE": "train"},
                    {"DEMO_DECODER_REPO_ID": "team/model"},
                    {"DEMO_DECODER_REPO_ID": "team/model", "DEMO_DECODER_REVISION": "main",
                     "DEMO_DECODER_FILENAME": "payload"}):
            with self.subTest(env=env), self.assertRaises(ValueError):
                load_demo_config(environ=env)

    def test_runtime_guard_rejects_pc_before_torch_import(self):
        with forbid_model_imports():
            for mode in ("preview", "inference"):
                with self.subTest(mode=mode), self.assertRaises(RuntimeError):
                    require_remote_runtime(dict(self.config, mode=mode), environ={}, platform="win32")
            with self.assertRaises(RuntimeError):
                require_remote_runtime(dict(self.config, mode="inference"), environ={}, platform="linux")

    def test_geotiff_band_order_radiometry_and_nodata(self):
        raw = np.full((13, 512, 512), 100, np.int16)
        raw[3], raw[2], raw[1] = 4796, 2398, -100
        raw[:, 8, 9] = -9999
        im = load_image(self.raster(raw, -9999), S2_LAYOUT, self.params)
        np.testing.assert_array_equal(im.rgb[0, 0], [1, .5, 0])
        np.testing.assert_array_equal(im.rgb[8, 9], [0, 0, 0])
        self.assertFalse(im.valid[8, 9])
        self.assertEqual(im.rgb.dtype, np.float32)
        np.testing.assert_array_equal(load_image(self.raster(raw[[3, 2, 1]], -9999), RGB_LAYOUT, self.params).rgb, im.rgb)

    def test_reject_wrong_band_count_shape_dtype_empty_and_oversize(self):
        for array in (np.ones((3, 512, 512), np.int16), np.ones((13, 16, 16), np.int16),
                      np.ones((13, 512, 512), np.float32), np.zeros((13, 512, 512), np.int16)):
            with self.subTest(shape=array.shape, dtype=array.dtype), self.assertRaises(ValueError):
                load_image(self.raster(array, 0), S2_LAYOUT, self.params)
        with self.assertRaises(ValueError):
            load_image(self.raster(np.ones((13, 512, 512), np.int16)), S2_LAYOUT, self.params, max_upload_mb=.001)

    def test_rgb_preview_is_not_dn_and_transparent_pixels_are_invalid(self):
        path = self.root / "render.png"
        Image.fromarray(np.full((512, 512, 3), 128, np.uint8)).save(path)
        im = load_image(path, DISPLAY_LAYOUT, self.params)
        self.assertAlmostEqual(float(im.rgb[0, 0, 0]), 128 / 255)
        self.assertEqual(im.metadata["radiometry"], "rendered_RGB_not_equivalent_to_raw_E2")
        Image.new("RGBA", (512, 512)).save(path)
        with self.assertRaises(ValueError):
            load_image(path, DISPLAY_LAYOUT, self.params)

    def test_rendered_rgba_and_grayscale_resize_without_distortion(self):
        path = self.root / "figure.png"
        Image.new("RGBA", (636, 658), (128, 128, 128, 255)).save(path)
        image = load_image(path, AUTO_LAYOUT, self.params)
        self.assertEqual(image.rgb.shape, (512, 512, 3))
        self.assertEqual(image.metadata["original_wh"], [636, 658])
        self.assertEqual(image.metadata["resized_wh"], [495, 512])
        self.assertEqual(image.metadata["padding_left_top"], [8, 0])
        self.assertFalse(image.valid[:, :8].any())
        self.assertTrue(image.valid[:, 8:503].all())
        with self.assertRaises(ValueError):
            validate_points([[0, 20, 1]], image)
        Image.new("L", (1000, 500), 128).save(path)
        gray = load_image(path, AUTO_LAYOUT, self.params)
        self.assertEqual(gray.metadata["resized_wh"], [512, 256])
        np.testing.assert_array_equal(gray.rgb[:, :, 0], gray.rgb[:, :, 1])
        controller = DemoController(self.config, backend=Mock())
        self.assertIsNotNone(controller.upload(path, AUTO_LAYOUT)[0])

    def test_points_bounds_positive_nodata_duplicates(self):
        im = synthetic_image()
        im.valid[8, 9] = False
        for points in ([], [[2, 2, 0]], [[512, 2, 1]], [[-1, 2, 1]], [[9, 8, 1]],
                       [[1, 2, 1], [1, 2, 0]], [[1., 2, 1]]):
            with self.subTest(points=points), self.assertRaises(ValueError):
                validate_points(points, im)
        validate_points([[511, 0, 1], [0, 511, 0]], im)
        with self.assertRaises(ValueError):
            validate_points([[1, 2, 1], [3, 4, 0]], im, max_points=1)

    def test_selected_mask_by_score_tie_and_original_pixels(self):
        im = synthetic_image()
        im.valid[0, 0] = False
        masks = np.zeros((3, 512, 512), bool)
        masks[1, 0:2, 0:2] = True
        recorder = Mock()
        recorder.predict.return_value = (masks, np.array([.1, .9, .9]), None)
        points = [[511, 0, 1], [0, 511, 0]]
        marked = draw_points(im, points)
        result = predict_manual(recorder, im, points)
        self.assertIs(recorder.set_image.call_args.args[0], im.rgb)
        self.assertFalse(np.array_equal(marked, im.preview()))
        self.assertEqual(result["selection"]["selected_index"], 1)
        self.assertEqual(int(result["mask"].sum()), 3)
        np.testing.assert_array_equal(recorder.predict.call_args.kwargs["point_coords"], [[511, 0], [0, 511]])
        recorder.reset_predictor.assert_called_once()

    def test_predictor_is_cleared_on_failure(self):
        recorder = Mock()
        recorder.predict.side_effect = RuntimeError("synthetic failure")
        with self.assertRaises(RuntimeError):
            predict_manual(recorder, synthetic_image(), [[1, 1, 1]])
        recorder.reset_predictor.assert_called_once()

    def test_checkpoint_schema_best_only(self):
        payload = {"config_sha256": self.report["identity"]["config_sha256"], "epoch": 1,
                   "best": {"epoch": 1, "state": {"synthetic": "not a tensor"}}}
        self.assertEqual(best_decoder_state(payload, self.report), payload["best"]["state"])
        for altered in ({}, dict(payload, epoch=4), dict(payload, config_sha256="wrong"),
                        dict(payload, best={"epoch": 1, "state": {}})):
            with self.assertRaises(ValueError):
                best_decoder_state(altered, self.report)

    def test_preview_never_calls_backend_and_edits_clear_results(self):
        backend = Mock()
        controller = DemoController(self.config, backend=backend)
        im = synthetic_image()
        self.assertTrue(all(v is None for v in controller.segment(im, [[1, 1, 1]], True)[:4]))
        backend.segment.assert_not_called()
        before = []
        output = controller.edit(im, before, "add", [100, 80], True)
        self.assertEqual(before, [])
        self.assertEqual(output[0], [[100, 80, 1]])
        self.assertEqual(output[3:7], (None, None, None, None))
        self.assertEqual(controller.edit(im, output[0], "undo")[0], [])
        self.assertIsNone(controller.upload("missing.tif", S2_LAYOUT)[0])

    def test_controller_partial_result_has_no_fabricated_tuned_mask(self):
        backend = Mock()
        backend.segment.return_value = {"frozen": {"mask": np.zeros((512, 512), bool), "selection": {}},
                                        "tuned": None, "status": "Decoder no configurado"}
        controller = DemoController(dict(self.config, mode="inference"), backend)
        out = controller.segment(synthetic_image(), [[1, 1, 1]], True)
        self.assertIsNone(out[1])
        self.assertIsNone(out[3])
        self.assertIsNone(out[4]["tuned_selection"])

    def test_backend_compares_identical_inputs_with_separate_predictors(self):
        backend = DemoBackend(dict(self.config, mode="inference", decoder_checkpoint_path="not-read-in-test"))
        im, points = synthetic_image(), [[20, 30, 1], [70, 80, 0]]
        recorders = {}
        for arm in ("frozen", "tuned"):
            recorder = Mock()
            recorder.predict.return_value = (np.zeros((3, 512, 512), bool), np.array([.2, .8, .1]), None)
            recorders[arm] = recorder
        # Only a plain context manager is injected: no model framework is imported.
        with patch("auto_prompt_sam2.demo_backend.require_remote_runtime"), \
                patch.dict(sys.modules, {"torch": SimpleNamespace(inference_mode=nullcontext)}), \
                patch.object(backend, "_predictor", side_effect=lambda arm: recorders[arm]):
            result = backend.segment(im, points, True)
        self.assertIsNotNone(result["tuned"])
        for recorder in recorders.values():
            self.assertIs(recorder.set_image.call_args.args[0], im.rgb)
            np.testing.assert_array_equal(recorder.predict.call_args.kwargs["point_labels"], [1, 0])
            np.testing.assert_array_equal(recorder.predict.call_args.kwargs["point_coords"], [[20, 30], [70, 80]])
            recorder.reset_predictor.assert_called_once()

    def test_backend_missing_or_bad_decoder_keeps_baseline_without_substitution(self):
        for decoder in (None, "bad-payload"):
            backend = DemoBackend(dict(self.config, mode="inference", decoder_checkpoint_path=decoder))
            recorder = Mock()
            recorder.predict.return_value = (np.zeros((3, 512, 512), bool), np.array([.2, .8, .1]), None)
            def get_predictor(arm):
                if arm == "tuned":
                    raise ValueError("synthetic integrity failure")
                return recorder
            with patch("auto_prompt_sam2.demo_backend.require_remote_runtime"), \
                    patch.dict(sys.modules, {"torch": SimpleNamespace(inference_mode=nullcontext)}), \
                    patch.object(backend, "_predictor", side_effect=get_predictor) as loader, \
                    patch("auto_prompt_sam2.demo_backend.LOG"):
                result = backend.segment(synthetic_image(), [[2, 3, 1]], True)
            self.assertIsNotNone(result["frozen"])
            self.assertIsNone(result["tuned"])
            self.assertEqual(loader.call_count, 1 if decoder is None else 2)

    def test_gradio_build_and_local_http_without_models(self):
        with forbid_model_imports():
            from fastapi.testclient import TestClient
            import gradio as gr
            from fastapi import FastAPI
            demo = build_demo(self.config)
            try:
                app = gr.mount_gradio_app(FastAPI(), demo, path="/")
                with TestClient(app) as client:
                    self.assertEqual(client.get("/").status_code, 200)
                    config = client.get("/config").json()
                    self.assertTrue(any(c["type"] == "image" for c in config["components"]))
                    self.assertTrue(any(d["targets"][0][1] == "select" for d in config["dependencies"]))
            finally:
                demo.close()


if __name__ == "__main__":
    unittest.main()
