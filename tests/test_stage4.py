"""Light CPU contracts only: arrays and plain records/stubs, no neural models."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from auto_prompt_sam2.dissection import train_plan, read_train_image, arrays_payload, read_arrays, figure_png
from auto_prompt_sam2.features import token_layout, tokens_to_grid, normalize_erf, capture_features, hook_inventory, load_frozen_dino
from auto_prompt_sam2.profiling import validate_config, parameter_table, timing_summary, block_instrumentation, count_flops, benchmark


class ParameterRecord:
    def __init__(self, count, trainable=False):
        self.count, self.requires_grad = count, trainable
    def numel(self):
        return self.count


class HookRecord:
    """Hook registry only; deliberately has no forward method or tensor compute."""
    def __init__(self):
        self._forward_hooks, self._forward_pre_hooks = {}, {}
    def register(self, registry, hook):
        key = len(registry)
        registry[key] = hook
        class Handle:
            def remove(self):
                registry.pop(key, None)
        return Handle()
    def register_forward_hook(self, hook, **kwargs):
        return self.register(self._forward_hooks, hook)
    def register_forward_pre_hook(self, hook, **kwargs):
        return self.register(self._forward_pre_hooks, hook)
    def named_modules(self):
        return [("", self), ("block", self)]


class Stage4Contracts(unittest.TestCase):
    def config(self):
        return json.loads((REPO / "configs/profiling.yaml").read_text("utf-8"))

    def test_fixed_protocol_and_reject_scope_changes(self):
        config = self.config()
        validate_config(config)
        for key, wrong in (("roles", ["val"]), ("precision", "float16"),
                           ("dino_size", 224), ("projection", "pca"),
                           ("repetitions", 1), ("warmup", 0),
                           ("repeat_tolerance", {"atol": 1, "rtol": 1})):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config(dict(config, **{key: wrong}))

    def test_tokens_special_positions_and_row_major_coordinates(self):
        layout = token_layout((32, 48), 16, 4, 2)
        tokens = np.arange(1 * 11 * 2).reshape(1, 11, 2)
        grid = tokens_to_grid(tokens, layout)
        self.assertEqual(grid.shape, (1, 2, 2, 3))
        np.testing.assert_array_equal(grid[0, :, 0, 0], tokens[0, 5])
        np.testing.assert_array_equal(grid[0, :, 1, 2], tokens[0, 10])
        with self.assertRaises(ValueError):
            tokens_to_grid(tokens[:, 1:], layout)
        with self.assertRaises(ValueError):
            token_layout((513, 512), 16, 4, 1024)

    def test_erf_unit_mass_and_invalid_gradients(self):
        grad = np.zeros((3, 4, 4))
        grad[0, 1, 2], grad[2, 0, 0] = -3, 1
        erf = normalize_erf(grad)
        self.assertEqual(float(erf.sum()), 1.)
        self.assertEqual(float(erf[1, 2]), .75)
        for bad in (np.zeros_like(grad), np.full_like(grad, np.nan)):
            with self.assertRaises(ValueError):
                normalize_erf(bad)

    def test_parameter_sum_unique_and_explicit_residual(self):
        a, b, c = ParameterRecord(10), ParameterRecord(20), ParameterRecord(3)
        class Records:
            def named_parameters(self):
                return [("encoder.weight", a), ("encoder.shared_alias", a),
                        ("decoder.weight", b), ("video_token", c)]
            def parameters(self):
                return [a, b, c]
        table = parameter_table(Records(), ["encoder", "decoder"])
        self.assertEqual(table["total"], 33)
        self.assertEqual([r["parameters"] for r in table["rows"]], [10, 20, 3])
        self.assertEqual(table["trainable"], 0)
        with self.assertRaises(ValueError):
            parameter_table(Records(), ["encoder", "encoder.child"])

    def test_hooks_removed_after_normal_and_exceptional_exit(self):
        record = HookRecord()
        before = hook_inventory(record)
        with capture_features(record, ["block"]):
            self.assertTrue(record._forward_hooks)
        self.assertEqual(hook_inventory(record), before)
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with capture_features(record, ["block"]):
                raise RuntimeError("intentional")
        self.assertEqual(hook_inventory(record), before)
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with block_instrumentation(record, ["block"], inputs=True):
                raise RuntimeError("intentional")
        self.assertEqual(hook_inventory(record), before)
        with self.assertRaises(ValueError):
            with capture_features(record, ["block", "missing"]):
                pass
        self.assertEqual(hook_inventory(record), before)

    def manifest(self, regime="official"):
        return {"regime": regime, "records": [
            {"id": "b", "role": "train", "event": "Ghana"},
            {"id": "a", "role": "train", "event": "India"},
            {"id": "held", "role": "test", "event": "Bolivia"},
            {"id": "validation", "role": "val", "event": "Ghana"}]}

    def test_train_selection_does_not_open_validation_or_reserved(self):
        self.assertEqual([r["id"] for r in train_plan(self.manifest(), 2)], ["a", "b"])
        with tempfile.TemporaryDirectory() as directory:
            with patch("auto_prompt_sam2.dissection.input_rgb") as reader:
                for chip in ("held", "validation", "absent"):
                    with self.assertRaises(ValueError):
                        read_train_image(directory, self.manifest(), chip)
                reader.assert_not_called()
        for regime, event in (("official", "Bolivia"), ("leave_paraguay_out", "Paraguay")):
            manifest = self.manifest(regime)
            manifest["records"][0]["event"] = event
            with self.assertRaises(ValueError):
                train_plan(manifest, 1)

    def test_workstation_guards_precede_model_or_cuda_work(self):
        import auto_prompt_sam2.sam_adapter as adapter
        with patch.object(adapter.sys, "platform", "win32"):
            for call in (lambda: load_frozen_dino({}, Path(".")),
                         lambda: count_flops(None, [], None), lambda: benchmark(None, {})):
                with self.assertRaisesRegex(RuntimeError, "Google Colab"):
                    call()

    def test_latency_median_samples_and_bad_values(self):
        summary = timing_summary([4., 1., 9., 2.], 5)
        self.assertEqual(summary["median_ms"], 3.)
        for samples in ([1.], [1., 0.], [1., float("nan")]):
            with self.assertRaises(ValueError):
                timing_summary(samples, 5)

    def test_flop_audit_on_tiny_cpu_arrays_without_any_model(self):
        import torch
        # Only 2x3 @ 3x4 arithmetic, no nn.Module, weights or data/model forward.
        # Bypass location guard solely for this synthetic counter contract.
        with patch("auto_prompt_sam2.profiling.require_colab_gpu"):
            result = count_flops(HookRecord(), [], lambda: torch.ones(2, 3) @ torch.ones(3, 4))
        self.assertEqual(result["counted_total"], 48)  # 2 FLOPs per multiply-add
        self.assertEqual(result["rows"], [{"block": "residual", "executed_calls": None, "counted_flops": 48}])
        self.assertTrue(any(o["operation"] == "aten.mm" and o["formula_registered"] for o in result["operations"]))
        self.assertTrue(any(not o["formula_registered"] for o in result["operations"]))

    def test_arrays_and_figure_roundtrip_without_models(self):
        arrays = {"layer.0": np.arange(24, dtype=np.float32).reshape(1, 3, 2, 4),
                  "erf": np.ones((8, 8), np.float32) / 64}
        restored = read_arrays(arrays_payload(arrays))
        for name, value in arrays.items():
            np.testing.assert_array_equal(value, restored[name])
        png = figure_png(np.zeros((3, 8, 8), np.float32), {"layer.0": arrays["layer.0"]},
                         arrays["erf"], [0, 1, 2])
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))


if __name__ == "__main__":
    unittest.main()
