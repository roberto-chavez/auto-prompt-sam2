"""Light CPU contracts with arrays/plain stubs only; no model construction/weights."""
import copy
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
from auto_prompt_sam2.artifacts import ArtifactStore, canonical_json, digest
from auto_prompt_sam2.baseline import (validate_config, development_plan, read_development_chip,
    encode_prediction, decode_prediction, stage2_inputs)
from auto_prompt_sam2.metrics import confusion, scores, chip_metrics, aggregate, pixel_areas_m2
from auto_prompt_sam2.reference_prompts import connected_water, simulate_points
from auto_prompt_sam2.sam_adapter import (segment_components, select_multimask, compare_predictions,
    require_colab_gpu, assert_frozen, state_fingerprint)


class PlainPredictorStub:
    """A function contract recorder, not a neural network or SAM model."""
    def __init__(self):
        self.images, self.calls = [], []

    def set_image(self, image):
        self.images.append(image)

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        masks = np.zeros((3, 512, 512), bool)
        # Highest-predicted-score mask deliberately does not follow positive point.
        masks[0, 1, 1] = True
        masks[1, 20 + len(self.calls), 30] = True
        masks[2, :, :] = True
        return masks, np.array([0.1, 0.9, 0.2]), None


class Stage3Metrics(unittest.TestCase):
    def metric(self, labels, pred, chip="one", event="A"):
        return chip_metrics(np.array(labels), np.array(pred), chip_id=chip, event=event, pixel_area_m2=100.)

    def test_hand_calculated_confusion_and_scores(self):
        label = np.array([[1, 1, 0], [0, -1, 1]])
        pred = np.array([[1, 0, 1], [0, 1, 1]])
        c = confusion(label, pred)
        self.assertEqual(c, {"tp": 2, "fp": 1, "fn": 1, "tn": 1})
        values = scores(c)["values"]
        for name, expected in {"water_iou": .5, "precision": 2/3, "recall": 2/3,
                               "dice": 2/3, "nonwater_iou": 1/3, "miou": 5/12}.items():
            self.assertAlmostEqual(values[name], expected)
        pred[1, 1] = 0
        self.assertEqual(confusion(label, pred), c)  # -1 cannot affect any denominator.

    def test_perfect_and_total_error(self):
        label = [[1, 0], [-1, 1]]
        perfect = self.metric(label, [[1, 0], [1, 1]])
        self.assertTrue(all(v == 1 for v in perfect["scores"]["values"].values()))
        wrong = self.metric(label, [[0, 1], [1, 0]])
        self.assertTrue(all(v == 0 for v in wrong["scores"]["values"].values()))

    def test_all_invalid_zero_denominators(self):
        row = self.metric([[-1, -1]], [[1, 0]])
        self.assertEqual(row["confusion"], {k: 0 for k in ("tp", "fp", "fn", "tn")})
        self.assertTrue(row["all_invalid"])
        self.assertFalse(row["no_valid_water"])
        self.assertEqual(row["individual_iou_exclusion"], "all_invalid")
        self.assertTrue(all(v is None for v in row["scores"]["values"].values()))
        summary = aggregate([row])
        self.assertEqual(summary["all_invalid_chips"], 1)
        self.assertIsNone(summary["event_mean_water_iou"])
        self.assertEqual(summary["excluded_events"], ["A"])
        self.assertEqual(summary["per_chip"]["undefined_reason"], "no_chips_with_valid_water")
        json.dumps(summary, allow_nan=False)
        self.assertIsNone(aggregate([])["pooled"]["values"]["miou"])

    def test_waterless_with_false_positives_excluded_individually_retained_pooled(self):
        water = self.metric([[1, 0]], [[1, 0]], "water")
        empty = self.metric([[0, -1]], [[1, 1]], "empty")
        self.assertIsNone(empty["individual_water_iou"])
        self.assertEqual(empty["scores"]["values"]["water_iou"], 0)
        self.assertEqual(empty["fp_area_m2"], 100)
        summary = aggregate([water, empty])
        self.assertEqual(summary["pooled"]["values"]["water_iou"], .5)
        self.assertEqual(summary["per_chip"]["median"], 1)
        self.assertEqual(summary["per_chip"]["included"], 1)
        self.assertEqual(summary["no_water"], {"chips": 1, "fp_pixels": 1, "fp_area_m2": 100.})

    def test_no_prediction_no_truth_and_single_defined_class(self):
        empty = self.metric([[0, 0]], [[0, 0]])
        values = empty["scores"]["values"]
        self.assertIsNone(values["water_iou"])
        self.assertIsNone(values["precision"])
        self.assertIsNone(values["recall"])
        self.assertEqual(values["miou"], 1)
        self.assertEqual(empty["scores"]["miou_defined_classes"], ["nonwater_iou"])
        only_water = self.metric([[1, 1]], [[1, 1]])
        self.assertEqual(only_water["scores"]["miou_defined_classes"], ["water_iou"])
        missed = self.metric([[1]], [[0]])
        self.assertIsNone(missed["scores"]["values"]["precision"])
        self.assertEqual(missed["scores"]["values"]["recall"], 0)

    def test_pool_first_then_equal_event_mean_and_linear_iqr(self):
        a = self.metric([[1]], [[1]], "a", "A")
        b = self.metric([[1, 1, 1]], [[0, 0, 0]], "b", "B")
        c = self.metric([[1, 1]], [[1, 0]], "c", "B")
        summary = aggregate([a, b, c])
        self.assertAlmostEqual(summary["pooled"]["values"]["water_iou"], 2/6)
        self.assertAlmostEqual(summary["event_mean_water_iou"], (1 + 1/5)/2)
        self.assertEqual(summary["per_chip"]["median"], .5)
        self.assertEqual(summary["per_chip"]["q25"], .25)
        self.assertEqual(summary["per_chip"]["q75"], .75)
        self.assertEqual(summary["per_chip"]["iqr"], .5)
        with self.assertRaises(ValueError):
            aggregate([a, a])

    def test_invalid_inputs_rejected(self):
        for label, pred in (([[2]], [[1]]), ([[1]], [[2]]), ([[1]], [[float("nan")]]),
                            ([[1]], [[1, 0]]), ([1], [1])):
            with self.assertRaises(ValueError):
                confusion(label, pred)
        for area in (0, -100, float("nan"), np.ones((2, 2))):
            with self.assertRaises(ValueError):
                chip_metrics(np.array([[0]]), np.array([[1]]), chip_id="x", event="A", pixel_area_m2=area)

    def test_physical_area_from_degrees_and_row_weights(self):
        geo = {"height": 2, "width": 3, "crs": "EPSG:4326",
               "transform": [0.0001, 0, -70, 0, -0.0001, 0.0002]}
        areas = pixel_areas_m2(geo)
        self.assertEqual(areas.shape, (2, 3))
        # ~11.13 x 11.06 m at equator, not 1e-8 m² nor nominal 100 m².
        self.assertTrue(np.all((areas > 122) & (areas < 124)))
        metric = chip_metrics(np.array([[0, 0, -1], [1, 0, 0]]), np.ones((2, 3), bool),
            chip_id="x", event="A", pixel_area_m2=areas)
        self.assertAlmostEqual(metric["fp_area_m2"], float(areas[0, :2].sum() + areas[1, 1:].sum()))
        for change in ({"crs": "EPSG:3857"}, {"transform": [.001, 1, 0, 0, -.001, 0]}):
            with self.assertRaises(ValueError):
                pixel_areas_m2(dict(geo, **change))


class Stage3PromptsAndAdapter(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((REPO / "configs/baseline.yaml").read_text("utf-8"))

    def simulate(self, labels, **changes):
        policy = dict(self.config["prompts"], **changes)
        return simulate_points(labels, regime="leave_paraguay_out", chip_id="Ghana_1", seed=42, policy=policy)

    def test_connectivity_invalid_bridge_singletons_and_no_drop(self):
        labels = np.array([[1, -1, 1], [0, 1, 0], [1, 0, 1]])
        self.assertEqual(len(list(connected_water(labels, 8))), 1)
        self.assertEqual(len(list(connected_water(labels, 4))), 5)
        self.assertEqual(len(list(connected_water(np.array([[1, -1, 1]]), 8))), 2)
        with self.assertRaises(ValueError):
            list(connected_water(labels, 6))

    def test_point_sign_validity_spacing_budget_seed_and_shortfall(self):
        labels = np.zeros((32, 32), np.int16)
        labels[2:6, 3:7] = 1
        labels[24:30, 24:30] = 1
        labels[0, :] = -1
        a = self.simulate(labels, positive_per_component=3, negative_per_component=4, minimum_distance_px=3.)
        self.assertEqual(a, self.simulate(labels, positive_per_component=3, negative_per_component=4, minimum_distance_px=3.))
        self.assertEqual(len(a["components"]), 2)
        for c in a["components"]:
            points = c["point_coords"]
            for (x, y), sign in zip(points, c["point_labels"]):
                self.assertEqual(labels[y, x], sign)
            for i, p in enumerate(points):
                for q in points[i + 1:]:
                    self.assertGreaterEqual(np.linalg.norm(np.array(p) - q), 3)
            self.assertEqual(c["positive_shortfall"] + c["point_labels"].count(1), 3)
            self.assertEqual(c["negative_shortfall"] + c["point_labels"].count(0), 4)
        singleton = self.simulate(np.ones((1, 1), np.int16), positive_per_component=2)
        self.assertEqual(singleton["components"][0]["positive_shortfall"], 1)
        self.assertEqual(singleton["components"][0]["negative_shortfall"], 1)
        with self.assertRaises(ValueError):
            self.simulate(labels, positive_per_component=0)

    def test_waterless_and_all_invalid_empty_no_predictor_call(self):
        predictor = PlainPredictorStub()
        for fill in (0, -1):
            prompts = self.simulate(np.full((512, 512), fill, np.int16))
            self.assertTrue(prompts["no_water"])
            result = segment_components(predictor, np.zeros((512, 512, 3), np.float32), prompts["components"])
            self.assertFalse(result["mask"].any())
            self.assertEqual(result["predict_calls"], 0)
        self.assertEqual(predictor.images, [])
        self.assertEqual(predictor.calls, [])

    def test_native_xy_multimask_predicted_score_union(self):
        predictor = PlainPredictorStub()
        components = [{"component_id": i, "point_coords": [[511, 7], [0, 511]],
                       "point_labels": [1, 0]} for i in (1, 2)]
        result = segment_components(predictor, np.zeros((512, 512, 3), np.float32), components)
        self.assertEqual(len(predictor.images), 1)
        self.assertEqual(len(predictor.calls), 2)
        np.testing.assert_array_equal(predictor.calls[0]["point_coords"], [[511, 7], [0, 511]])
        np.testing.assert_array_equal(predictor.calls[0]["point_labels"], [1, 0])
        self.assertTrue(predictor.calls[0]["multimask_output"])
        self.assertTrue(predictor.calls[0]["normalize_coords"])
        self.assertEqual(result["mask"].sum(), 2)
        self.assertTrue(result["mask"][21, 30] and result["mask"][22, 30])
        self.assertFalse(result["mask"][7, 511])  # No oracle overlap ranking.
        self.assertEqual([c["selected_index"] for c in result["components"]], [1, 1])
        self.assertNotIn("labels", inspect.signature(segment_components).parameters)

    def test_ties_nonfinite_scores_and_wrong_geometry(self):
        masks = np.zeros((3, 512, 512), bool)
        _, decision = select_multimask(masks, [0.8, 0.8, 0.1], (512, 512))
        self.assertEqual(decision["selected_index"], 0)
        for scores_ in ([np.nan, 1, 0], [], [1], [[1, 2, 3]]):
            with self.assertRaises(ValueError):
                select_multimask(masks, scores_, (512, 512))
        with self.assertRaises(ValueError):
            select_multimask(masks, [0, 1, 2], (1024, 1024))
        for point in ([512, 0], [0, -1], [np.nan, 0]):
            with self.assertRaises(ValueError):
                segment_components(PlainPredictorStub(), np.zeros((512, 512, 3), np.float32),
                    [{"component_id": 1, "point_coords": [point], "point_labels": [1]}])
        with self.assertRaises(ValueError):
            segment_components(PlainPredictorStub(), np.zeros((1024, 1024, 3), np.float32), [])

    def test_repeat_tolerance_and_native_artifact_reload(self):
        a = segment_components(PlainPredictorStub(), np.zeros((512, 512, 3), np.float32),
            [{"component_id": 1, "point_coords": [[1, 1]], "point_labels": [1]}])
        b = copy.deepcopy(a)
        b["components"][0]["predicted_iou"][1] += 1e-7
        compare_predictions(a, b, self.config["repeat_tolerance"])
        b["mask"][0, 0] = True
        with self.assertRaises(ValueError):
            compare_predictions(a, b, self.config["repeat_tolerance"])
        with tempfile.TemporaryDirectory() as folder:
            store = ArtifactStore(Path(folder))
            metadata = {"inference": {k: v for k, v in a.items() if k != "mask"},
                        "seed": 42, "branch": self.config["branch"]}
            gen = store.publish("chip", encode_prediction(a["mask"], metadata), {"regime": "official"})
            mask, restored = decode_prediction(ArtifactStore(Path(folder)).read("chip", {"regime": "official"}, gen)[0])
            np.testing.assert_array_equal(mask, a["mask"])
            self.assertEqual(metadata, restored)
            with self.assertRaises(FileNotFoundError):
                store.read("chip", {"regime": "leave_paraguay_out"}, gen)

    def test_freeze_and_state_hash_contract_with_plain_tensor_container(self):
        import torch
        class StateStub:
            training = False
            def modules(self): return [self]
            def parameters(self): return [self.weight]
            def state_dict(self): return {"weight": self.weight, "buffer": self.buffer}
        stub = StateStub()
        stub.weight = torch.tensor([1.])
        stub.buffer = torch.tensor([2.])
        assert_frozen(stub)
        first = state_fingerprint(stub)
        stub.buffer += 1
        self.assertNotEqual(first, state_fingerprint(stub))
        stub.weight.requires_grad_(True)
        with self.assertRaises(ValueError): assert_frozen(stub)
        stub.weight.requires_grad_(False)
        stub.training = True
        with self.assertRaises(ValueError): assert_frozen(stub)

    def test_native_points_match_official_transform_without_model(self):
        import torch
        from sam2.utils.transforms import SAM2Transforms
        from auto_prompt_sam2.preprocessing import Coordinates
        xy = np.array([[0, 0], [511, 511], [5.25, 30.75]], np.float32)
        transformed = SAM2Transforms(1024, 0.).transform_coords(torch.from_numpy(xy),
                                                              normalize=True, orig_hw=(512, 512))
        np.testing.assert_allclose(transformed.numpy(), Coordinates((512, 512), (1024, 1024)).points(xy), atol=1e-6)

    def test_development_only_reject_before_any_file_read(self):
        rows = [{"id": role, "role": role, "event": event} for role, event in
                (("train", "Ghana"), ("val", "Ghana"), ("test", "Ghana"),
                 ("bolivia", "Bolivia"), ("paraguay_reserved", "Paraguay"))]
        manifest = {"regime": "leave_paraguay_out", "records": rows}
        self.assertEqual([r["id"] for r in development_plan(manifest)], ["train", "val"])
        with patch("auto_prompt_sam2.baseline.sha256_file", side_effect=AssertionError("unexpected file read")):
            for role in ("test", "bolivia", "paraguay_reserved"):
                with self.assertRaises(ValueError):
                    read_development_chip(".", manifest, chip_id=role, role=role)
            with self.assertRaises(ValueError):
                read_development_chip(".", manifest, chip_id="test", role="val")
        leaked = copy.deepcopy(manifest)
        leaked["records"][0]["event"] = "Paraguay"
        with self.assertRaises(ValueError):
            development_plan(leaked)
        validate_config(self.config)
        changed = copy.deepcopy(self.config)
        changed["roles"].append("test")
        with self.assertRaises(ValueError): validate_config(changed)
        with patch("auto_prompt_sam2.sam_adapter.sys.platform", "win32"):
            with self.assertRaisesRegex(RuntimeError, "Colab"):
                require_colab_gpu()

    def test_stage2_generations_reused_without_fitting_or_raw_reads(self):
        from auto_prompt_sam2.preprocessing import save_preprocessing
        data_config = json.loads((REPO / "configs/data.yaml").read_text("utf-8"))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            acquisition = {"objects": []}
            ArtifactStore(root / "raw").publish("acquisition", canonical_json(acquisition), {"dataset": "Sen1Floods11-v1.1"})
            report = {"regimes": {}}
            for regime in ("official", "leave_paraguay_out"):
                manifest = {"regime": regime, "records": [{"id": "Ghana_1", "event": "Ghana",
                    "role": "train", "image_sha256": "abc"}]}
                base = root / "regimes" / regime
                generation = ArtifactStore(base / "manifests").publish("manifest", canonical_json(manifest),
                    {"regime": regime, "manifest_sha256": digest(manifest)})
                params = {"regime": regime, "manifest_sha256": digest(manifest), "source_role": "train",
                    "source_ids": ["Ghana_1"], "source_image_hashes": {"Ghana_1": "abc"},
                    "config": data_config, "upper_dn": 4796 if regime == "official" else 4907}
                pg = save_preprocessing(base / "preprocessing", params)
                report["regimes"][regime] = {"manifest_sha256": digest(manifest), "manifest_generation": generation,
                    "parameter_generation": pg, "preprocessing_sha256": digest(params)}
            ArtifactStore(root / "stage2").publish("validation", canonical_json(report),
                {"config_sha256": digest(data_config), "acquisition_sha256": digest(acquisition)})
            with patch("auto_prompt_sam2.preprocessing.fit_preprocessing", side_effect=AssertionError("no refitting")):
                loaded = stage2_inputs(root, data_config)
            self.assertEqual(loaded["official"]["params"]["upper_dn"], 4796)
            self.assertEqual(loaded["leave_paraguay_out"]["params"]["upper_dn"], 4907)
            # A missing compatible generation is an error, never a request to fit.
            marker = root / "regimes/official/preprocessing/parameters" / report["regimes"]["official"]["parameter_generation"] / "COMPLETE.json"
            marker.unlink()
            with self.assertRaises(FileNotFoundError):
                stage2_inputs(root, data_config)

    def test_prepare_and_recover_artifacts_with_plain_stub_only(self):
        # Integration of the run lifecycle, not SAM execution. Every model entry
        # point is replaced by a plain recorder; no weights or neural nets exist.
        import subprocess
        from contextlib import ExitStack, redirect_stdout
        import io
        sys.path.insert(0, str(REPO / "scripts"))
        import run_stage3
        data_config = json.loads((REPO / "configs/data.yaml").read_text("utf-8"))
        inputs = {}
        for regime in ("official", "leave_paraguay_out"):
            rows = [{"id": "Ghana_" + role, "role": role, "event": "Ghana",
                     "image_sha256": "image", "label_sha256": "label",
                     "geometry": {"height": 512, "width": 512, "crs": "EPSG:4326",
                         "transform": [.0001, 0, -70, 0, -.0001, -10]}} for role in ("train", "val")]
            inputs[regime] = {"manifest": {"regime": regime, "records": rows},
                "params": {"config": data_config, "upper_reflectance": .5},
                "manifest_generation": "saved", "parameter_generation": "saved"}
        def read_stub(root, manifest, *, chip_id, role):
            row = next(r for r in manifest["records"] if r["id"] == chip_id and r["role"] == role)
            labels = np.zeros((512, 512), np.int16)
            labels[1, 1] = 1
            return row, np.full((3, 512, 512), 1000, np.int16), np.ones((512, 512), bool), labels
        def load_stub(*args, **kwargs):
            predictor = PlainPredictorStub()
            # Make repeated calls independent of the recorder's invocation count.
            original_predict = predictor.predict
            def predict(**kw):
                predictor.calls.clear()
                return original_predict(**kw)
            predictor.predict = predict
            predictor.model = object()
            return predictor, {"state_sha256": "unchanged_stub", "checkpoint_sha256": "no_checkpoint_stub"}
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            stack.enter_context(patch.object(run_stage3, "require_colab_gpu"))
            stack.enter_context(patch.object(run_stage3, "source_snapshot", return_value=(
                {"source_tree_sha256": "stub", "source_sha256": {}}, b"stub")))
            stack.enter_context(patch.object(run_stage3.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "stubbed", "")))
            stack.enter_context(patch("auto_prompt_sam2.runtime.persistent_root", return_value=Path(folder)))
            stack.enter_context(patch("auto_prompt_sam2.runtime.verify_installed_pins", return_value={"stub": True}))
            stack.enter_context(patch("auto_prompt_sam2.runtime.inventory", return_value={"scope": "stub_no_gpu"}))
            stack.enter_context(patch("auto_prompt_sam2.baseline.stage2_inputs", return_value=inputs))
            stack.enter_context(patch("auto_prompt_sam2.baseline.read_development_chip", side_effect=read_stub))
            stack.enter_context(patch("auto_prompt_sam2.sam_adapter.load_frozen_sam", side_effect=load_stub))
            stack.enter_context(patch("auto_prompt_sam2.sam_adapter.assert_frozen"))
            stack.enter_context(patch("auto_prompt_sam2.sam_adapter.state_fingerprint", return_value="unchanged_stub"))
            stack.enter_context(redirect_stdout(io.StringIO()))
            # Existing CPU tensors are allowed; no CUDA or model construction.
            with patch.dict("os.environ", {"PYTHONHASHSEED": "42"}):
                prepare = run_stage3.run_stage3(Path(folder), "prepare")
                recover = run_stage3.run_stage3(Path(folder), "recover")
            self.assertEqual(prepare["run_id"], recover["run_id"])
            self.assertTrue(recover["frozen_unchanged"])
            for regime in inputs:
                self.assertTrue(recover["regimes"][regime]["all_saved_predictions_recovered"])
                self.assertEqual(prepare["regimes"][regime]["summary"], recover["regimes"][regime]["summary"])
                self.assertEqual(prepare["regimes"][regime]["counts"], {"train": 1, "val": 1})
            self.assertTrue((Path(folder) / "stage3/evidence" / (prepare["run_id"] + ".zip")).is_file())


if __name__ == "__main__":
    unittest.main()
