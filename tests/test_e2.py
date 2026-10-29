"""E2 contracts. Synthetic tiny decoder (always) + random-weight SAM 2.1 architecture (if sam2 is installed).

Nothing here loads pretrained weights, reads Sen1Floods11 or produces a segmentation result.
Random-weight tests only prove code paths (shapes, gradients, freezing, parity), not quality.
"""
import copy
import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import torch
from torch import nn

from auto_prompt_sam2 import intervention as iv
from auto_prompt_sam2.metrics import confusion

CFG = json.loads((REPO / "configs/e2_intervention.yaml").read_text("utf-8"))
BASE = json.loads((REPO / "configs/baseline.yaml").read_text("utf-8"))
LOSS = CFG["training"]["loss"]


class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.tensor([0.5, -1.0, 2.0]))
        self.b = nn.Parameter(torch.tensor([0.1, 0.2, -0.3]))
        self.iou_w = nn.Parameter(torch.tensor([0.3, -0.2, 0.1]))  # no gradient from the loss


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = nn.Linear(1, 1)
        self.sam_prompt_encoder = nn.Linear(2, 2)
        self.sam_mask_decoder = TinyDecoder()
        self.register_buffer("frozen_buffer", torch.arange(3.0))


class TinyRunner:
    """Same interface as Sam2Runner; a plain differentiable function, not SAM."""
    def __init__(self):
        torch.manual_seed(0)
        self.model = TinyModel()
        self.decoder = self.model.sam_mask_decoder

    def encode(self, image):
        return {"feat": torch.as_tensor(image, dtype=torch.float32)}

    def run(self, feats, coords, labels, *, grad):
        d, c = self.decoder, torch.as_tensor(np.asarray(coords), dtype=torch.float32)
        shift = c.mean(dim=(1, 2)) * 0.01
        with torch.enable_grad() if grad else torch.no_grad():
            logits = (d.w[None, :, None, None] * feats["feat"][None, None] + d.b[None, :, None, None]
                      + shift[:, None, None, None])
            iou = torch.sigmoid(d.iou_w[None] * c.mean(dim=(1, 2))[:, None] / 10 + 0.01 * shift[:, None])
        return logits, iou


def make_components(n, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        points = rng.integers(0, 16, size=(2 if i % 3 else 1 + (i % 2), 2)).tolist()
        labels = [1] + [0] * (len(points) - 1)
        out.append({"component_id": i + 1, "point_coords": points, "point_labels": labels})
    return out


def make_chip(runner, components, seed=0, hw=(16, 16)):
    rng = np.random.default_rng(seed)
    image = rng.normal(size=hw).astype(np.float32)
    labels = rng.integers(0, 2, size=hw).astype(np.int64)
    labels[0, :3] = -1
    return {"id": "c%d" % seed, "event": "E", "role": "train", "labels": labels,
            "labels_t": torch.as_tensor(labels), "components": components,
            "feats": runner.encode(image), "area": 100.0}


class Losses(unittest.TestCase):
    def test_hand_calculated_bce_dice(self):
        logits = torch.tensor([[2.0, -1.0], [0.0, 3.0]])
        labels = torch.tensor([[1, 0], [-1, 1]])
        total, parts = iv.bce_dice_loss(logits, labels, LOSS)
        p = torch.sigmoid(logits)
        bce = -(torch.log(p[0, 0]) + torch.log(1 - p[0, 1]) + torch.log(p[1, 1])) / 3
        inter, denom = p[0, 0] + p[1, 1], p[0, 0] + p[0, 1] + p[1, 1] + 2
        dice = 1 - (2 * inter + 1) / (denom + 1)
        self.assertAlmostEqual(parts["bce"], float(bce), places=6)
        self.assertAlmostEqual(parts["dice"], float(dice), places=6)
        self.assertAlmostEqual(float(total), float(bce + dice), places=6)

    def test_invalid_pixels_never_change_loss_or_gradient(self):
        labels = torch.tensor([[1, -1, 0], [0, -1, 1]])
        a = torch.randn(2, 3, requires_grad=True)
        b = a.detach().clone()
        b[:, 1] = 1e3  # only the invalid column differs
        b.requires_grad_(True)
        la, _ = iv.bce_dice_loss(a, labels, LOSS)
        lb, _ = iv.bce_dice_loss(b, labels, LOSS)
        la.backward()
        lb.backward()
        self.assertAlmostEqual(float(la.detach()), float(lb.detach()), places=6)
        self.assertTrue(torch.equal(a.grad[:, 1], torch.zeros(2)))
        self.assertTrue(torch.equal(b.grad[:, 1], torch.zeros(2)))

    def test_all_invalid_chip_has_no_loss(self):
        with self.assertRaises(ValueError):
            iv.bce_dice_loss(torch.zeros(2, 2), -torch.ones(2, 2, dtype=torch.long), LOSS)


class Subsets(unittest.TestCase):
    def test_label_free_deterministic_subset(self):
        rows = [{"id": "%s_%d" % (e, i), "event": e} for e in ("B", "A") for i in (3, 1, 2)]
        first = iv.select_subset(rows, 2)
        self.assertEqual([r["id"] for r in first], ["A_1", "A_2", "B_1", "B_2"])
        self.assertEqual(first, iv.select_subset(list(reversed(rows)), 2))
        self.assertEqual(len(iv.select_subset(rows, None)), 6)

    def test_chunks_group_equal_point_counts_without_padding(self):
        comps = make_components(20)
        for start, chunk in iv.ordered_chunks(comps, 4):
            self.assertLessEqual(len(chunk), 4)
            self.assertEqual(len({len(c["point_labels"]) for c in chunk}), 1)
        flat = [c["component_id"] for _, ch in iv.ordered_chunks(comps, 4) for c in ch]
        self.assertEqual(sorted(flat), [c["component_id"] for c in comps])


class UnionAndGradients(unittest.TestCase):
    def test_selection_uses_predicted_iou_not_truth_and_union_is_or(self):
        runner, comps = TinyRunner(), make_components(7)
        chip = make_chip(runner, comps)
        mask, info = iv.predict_chip(runner, chip["feats"], comps, 3, hw=(16, 16))
        manual = np.zeros((16, 16), bool)
        for c in comps:
            logits, iou = runner.run(chip["feats"], [c["point_coords"]], [c["point_labels"]], grad=False)
            manual |= (logits[0, int(iou[0].argmax())] > 0).numpy()
        self.assertTrue(np.array_equal(mask, manual))
        self.assertEqual(len(info["components"]), 7)
        # Selected index must equal argmax of the PREDICTED scores stored in the decision.
        for d in info["components"]:
            self.assertEqual(d["selected_index"], int(np.argmax(d["predicted_iou"])))

    def test_two_pass_backward_equals_plain_autograd(self):
        for chunk in (1, 2, 8):
            runner, comps = TinyRunner(), make_components(9, seed=3)
            chip = make_chip(runner, comps, seed=4)
            for p in runner.decoder.parameters():
                p.grad = None
            ref = iv.reference_backward(runner, chip["feats"], comps, chip["labels_t"], LOSS, chunk, scale=0.5)
            g_ref = {n: p.grad.clone() for n, p in runner.decoder.named_parameters() if p.grad is not None}
            for p in runner.decoder.parameters():
                p.grad = None
            got = iv.backward_chip(runner, chip["feats"], comps, chip["labels_t"], LOSS, chunk, scale=0.5)
            self.assertAlmostEqual(ref["total"], got["total"], places=6)
            for n, g in g_ref.items():
                self.assertTrue(torch.allclose(g, dict(runner.decoder.named_parameters())[n].grad, atol=1e-6, rtol=1e-5), n)
            self.assertIsNone(runner.decoder.iou_w.grad)  # predicted-IoU head gets no loss gradient

    def test_chip_without_components_is_empty_and_has_no_update(self):
        runner = TinyRunner()
        chip = make_chip(runner, [])
        self.assertIsNone(iv.backward_chip(runner, chip["feats"], [], chip["labels_t"], LOSS, 4))
        mask, info = iv.predict_chip(runner, chip["feats"], [], 4, hw=(16, 16))
        self.assertFalse(mask.any())
        self.assertEqual(info["predict_calls"], 0)
        self.assertTrue(all(p.grad is None for p in runner.model.parameters()))


class Freezing(unittest.TestCase):
    def test_only_decoder_trainable_and_frozen_hash_stable(self):
        runner = TinyRunner()
        report = iv.configure_decoder_only(runner.model)
        self.assertEqual(report["image_encoder"]["trainable"], 0)
        self.assertEqual(report["sam_prompt_encoder"]["trainable"], 0)
        self.assertEqual(report["sam_mask_decoder"]["trainable"], report["sam_mask_decoder"]["parameters"])
        before_frozen, before_dec = iv.frozen_fingerprint(runner.model), iv.decoder_fingerprint(runner.model)
        comps = make_components(6)
        chip = make_chip(runner, comps)
        iv.backward_chip(runner, chip["feats"], comps, chip["labels_t"], LOSS, 2)
        audit = iv.gradient_audit(runner.model)
        self.assertTrue(audit["ok"])
        self.assertIn("sam_mask_decoder.iou_w", audit["decoder_params_without_grad"])
        opt = torch.optim.AdamW([p for p in runner.model.parameters() if p.requires_grad], lr=0.1)
        opt.step()
        self.assertEqual(before_frozen, iv.frozen_fingerprint(runner.model))
        self.assertNotEqual(before_dec, iv.decoder_fingerprint(runner.model))

    def test_audit_detects_gradient_outside_decoder_and_configure_refreezes(self):
        runner = TinyRunner()
        iv.configure_decoder_only(runner.model)
        runner.model.image_encoder.weight.requires_grad_(True)
        self.assertFalse(iv.gradient_audit(runner.model)["ok"])
        report = iv.configure_decoder_only(runner.model)
        self.assertEqual(report["image_encoder"]["trainable"], 0)
        self.assertEqual(iv.gradient_audit(runner.model)["outside_decoder_with_grad_or_requires_grad"], [])

    def test_configure_rejects_nothing_to_train(self):
        runner = TinyRunner()
        for p in runner.decoder.parameters():
            p.requires_grad_(False)
        report = iv.configure_decoder_only(runner.model)  # re-enables decoder only
        self.assertGreater(report["sam_mask_decoder"]["trainable"], 0)


class TrainingLoop(unittest.TestCase):
    def setUp(self):
        self.cfg = copy.deepcopy(CFG)
        self.cfg["training"].update({"max_epochs": 4, "lr": 0.05, "accumulation_chips": 2, "patience": 10})
        self.cfg["inference"]["component_chunk"] = 3

    def chips(self, runner):
        return ([make_chip(runner, make_components(5 + k, seed=k), seed=k) for k in range(4)],
                [make_chip(runner, make_components(4 + k, seed=10 + k), seed=10 + k) for k in range(3)])

    def test_train_improves_logged_loss_and_keeps_frozen_state(self):
        runner = TinyRunner()
        iv.configure_decoder_only(runner.model)
        frozen = iv.frozen_fingerprint(runner.model)
        train, val = self.chips(runner)
        out = iv.train_decoder(runner, train, val, self.cfg, log=lambda m: None)
        self.assertEqual(len(out["history"]), 4)
        self.assertEqual(frozen, iv.frozen_fingerprint(runner.model))
        self.assertTrue(out["gradient_audit_first_step"]["ok"])
        self.assertLess(out["history"][-1]["train_loss"], out["history"][0]["train_loss"])
        self.assertIsNotNone(out["best"]["epoch"])

    def test_interrupt_and_resume_matches_uninterrupted_run(self):
        full = TinyRunner()
        iv.configure_decoder_only(full.model)
        train, val = self.chips(full)
        saved = {}
        out_full = iv.train_decoder(full, train, val, self.cfg, log=lambda m: None,
                                    on_epoch=lambda e, p: saved.__setitem__(e, copy.deepcopy(p)))
        resumed = TinyRunner()
        iv.configure_decoder_only(resumed.model)
        train2, val2 = self.chips(resumed)
        iv.load_decoder_state(resumed.model, saved[2]["decoder_state"])
        out_res = iv.train_decoder(resumed, train2, val2, self.cfg, log=lambda m: None, history=saved[2]["history"],
                                   start_epoch=3, best=saved[2]["best"], optimizer_state=saved[2]["optimizer"])
        self.assertEqual(iv.decoder_fingerprint(full.model), iv.decoder_fingerprint(resumed.model))
        self.assertEqual([h["val_water_iou_pooled"] for h in out_full["history"]],
                         [h["val_water_iou_pooled"] for h in out_res["history"]])

    def test_checkpoint_roundtrip(self):
        runner = TinyRunner()
        blob = iv.pack_checkpoint({"epoch": 1, "decoder_state": runner.decoder.state_dict()}, "abc")
        loaded = iv.unpack_checkpoint(blob)
        self.assertEqual(loaded["config_sha256"], "abc")
        other = TinyRunner()
        other.decoder.w.data.zero_()
        iv.load_decoder_state(other.model, loaded["decoder_state"])
        self.assertTrue(torch.equal(other.decoder.w, runner.decoder.w))


class ConfigContract(unittest.TestCase):
    def test_current_config_is_valid_and_matches_stage3(self):
        iv.validate_config(CFG, BASE)
        self.assertTrue(CFG["hypothesis"]["recorded_before_training"])
        self.assertEqual(CFG["subset"]["reserved_sets"], "forbidden")

    def test_rejects_different_prompts_or_regime(self):
        for edit in (lambda c: c["prompts"].__setitem__("positive_per_component", 3),
                     lambda c: c.__setitem__("regime", "leave_paraguay_out"),
                     lambda c: c["inference"].__setitem__("selection", "oracle_iou"),
                     lambda c: c["training"].__setitem__("trainable", "all")):
            bad = copy.deepcopy(CFG)
            edit(bad)
            with self.assertRaises(ValueError):
                iv.validate_config(bad, BASE)

    def test_paired_deltas(self):
        mk = lambda i, v: {"id": i, "event": "E", "individual_water_iou": v}
        out = iv.paired_deltas([mk("a", .2), mk("b", .5), mk("c", None)], [mk("a", .4), mk("b", .4), mk("c", .9)])
        self.assertEqual((out["n"], out["improved"], out["worsened"]), (2, 1, 1))


try:
    import sam2  # noqa: F401
    from hydra import initialize_config_module  # noqa: F401
    HAVE_SAM2 = True
except Exception:  # pragma: no cover
    HAVE_SAM2 = False


@unittest.skipUnless(HAVE_SAM2, "sam2 package not installed")
class RandomWeightSam2Architecture(unittest.TestCase):
    """Real SAM 2.1 tiny architecture with RANDOM weights on CPU: wiring only, never quality."""

    @classmethod
    def setUpClass(cls):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        torch.manual_seed(0)
        cls.model = build_sam2("configs/sam2.1/sam2.1_hiera_t.yaml", None, device="cpu", mode="eval",
                               apply_postprocessing=False).float().eval()
        cls.predictor = SAM2ImagePredictor(cls.model, mask_threshold=0.0, max_hole_area=0.0, max_sprinkle_area=0.0)
        cls.runner = iv.Sam2Runner(cls.predictor)
        rng = np.random.default_rng(0)
        cls.image = rng.random((512, 512, 3), dtype=np.float32)
        cls.feats = cls.runner.encode(cls.image)
        cls.comps = [{"component_id": i + 1, "point_coords": pts, "point_labels": lab} for i, (pts, lab) in enumerate([
            ([[40, 50], [300, 300]], [1, 0]), ([[100, 400]], [1]), ([[400, 100], [10, 10]], [1, 0]),
            ([[250, 260], [30, 480]], [1, 0]), ([[480, 480]], [1])])]
        cls.labels = (rng.random((512, 512)) > 0.7).astype(np.int64)
        cls.labels[:8] = -1

    def test_parameter_blocks_match_stage4_measurements(self):
        report = iv.parameter_report(self.model)
        self.assertEqual(sum(v["parameters"] for v in report.values()), 38_962_498)
        self.assertEqual(report["image_encoder"]["parameters"], 27_219_136)
        self.assertEqual(report["sam_prompt_encoder"]["parameters"], 6_220)
        self.assertEqual(report["sam_mask_decoder"]["parameters"], 4_215_109)

    def test_parity_with_stage3_predictor_path(self):
        from auto_prompt_sam2.sam_adapter import segment_components
        official = segment_components(self.predictor, self.image, self.comps)
        mine, info = iv.predict_chip(self.runner, self.feats, self.comps, 2)
        self.assertLessEqual(int(np.count_nonzero(official["mask"] != mine)), 10)
        for a, b in zip(official["components"], sorted(info["components"], key=lambda d: d["component_id"])):
            self.assertEqual(a["component_id"], b["component_id"])
            self.assertEqual(a["selected_index"], b["selected_index"])
            np.testing.assert_allclose(a["predicted_iou"], b["predicted_iou"], atol=1e-4)

    def test_decoder_only_gradients_and_two_pass_equivalence(self):
        model = self.model
        report = iv.configure_decoder_only(model)
        self.assertEqual(report["sam_mask_decoder"]["trainable"], 4_215_109)
        self.assertEqual(sum(v["trainable"] for v in report.values()), 4_215_109)
        frozen = iv.frozen_fingerprint(model)
        labels_t = torch.as_tensor(self.labels)
        for p in model.parameters():
            p.grad = None
        ref = iv.reference_backward(self.runner, self.feats, self.comps, labels_t, LOSS, 2)
        g_ref = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        self.assertTrue(iv.gradient_audit(model)["ok"])
        for p in model.parameters():
            p.grad = None
        got = iv.backward_chip(self.runner, self.feats, self.comps, labels_t, LOSS, 2)
        self.assertAlmostEqual(ref["total"], got["total"], places=4)
        worst = 0.0
        for n, g in g_ref.items():
            other = dict(model.named_parameters())[n].grad
            denom = max(float(g.abs().max()), 1e-12)
            worst = max(worst, float((g - other).abs().max()) / denom)
        self.assertLess(worst, 1e-3)
        audit = iv.gradient_audit(model)
        self.assertTrue(audit["ok"])
        self.assertTrue(all(n.startswith("sam_mask_decoder.") for n in g_ref))
        # An optimizer step must change decoder weights and nothing else.
        before_dec = iv.decoder_fingerprint(model)
        torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3).step()
        self.assertEqual(frozen, iv.frozen_fingerprint(model))
        self.assertNotEqual(before_dec, iv.decoder_fingerprint(model))


if __name__ == "__main__":
    unittest.main()
