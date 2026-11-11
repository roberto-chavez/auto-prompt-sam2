import copy, json, sys, unittest
from pathlib import Path
sys.path[:0] = [str(Path(__file__).parents[1] / "src")]
import numpy as np
from auto_prompt_sam2.evaluation import GateClosed, bootstrap_ci, evaluate, gate_problems, protocol_sha256, require_gate

PROTO = json.loads((Path(__file__).parents[1] / "configs/stage8_protocol.json").read_text("utf-8"))
CLOSED = {"pair": {"both_computational_locks_verified": True, "external_review": "pending", "stage8_authorized": False, "reserved_access": []},
          "regimes": {r: {"failures": [], "offline_recovery_present": False, "pending": ["x"]} for r in ("official", "leave_paraguay_out")}}
OPEN = {"pair": {"both_computational_locks_verified": True, "external_review": "done", "stage8_authorized": True, "reserved_access": []},
        "regimes": {r: {"failures": [], "offline_recovery_present": True, "pending": []} for r in ("official", "leave_paraguay_out")}}


def synthetic(regime_sizes, seed=0):
    rng = np.random.default_rng(seed); sets = {}
    for regime, spec in regime_sizes.items():
        sets[regime] = {}
        for name, n in spec["sets"].items():
            chips = []
            for i in range(n):
                lab = (rng.random((16, 16)) < 0.3).astype(int)
                if i % 7 == 0: lab[:] = 0          # chip without water
                lab[0, :3] = -1                     # invalid pixels
                chips.append({"id": f"{regime}-{name}-{i}", "event": "E%d" % (i % 3), "labels": lab, "pixel_area_m2": 100.0})
            sets[regime][name] = chips
    return sets


class Stage8Evaluator(unittest.TestCase):
    def test_gate_blocks_with_current_review_and_opens_only_with_hash(self):
        self.assertTrue(gate_problems(CLOSED))
        with self.assertRaises(GateClosed): require_gate(CLOSED, PROTO, protocol_sha256(PROTO))
        with self.assertRaises(GateClosed): require_gate(OPEN, PROTO, "0" * 64)
        require_gate(OPEN, PROTO, protocol_sha256(PROTO))

    def test_evaluate_synthetic_perfect_and_degraded_rows(self):
        sets = synthetic(PROTO["regimes"])
        def perfect(c): return (c["labels"] == 1).astype(int)
        def noisy(c): p = perfect(c).copy(); p[1, :] = 1; return p
        def empty(c): return np.zeros_like(c["labels"])
        preds = {r: {"a1_assisted": empty, "a1_points": noisy, "a3_tuned": perfect} for r in PROTO["regimes"]}
        out = evaluate(PROTO, sets, preds)
        s = out["regimes"]["official"]["test_seen"]
        self.assertEqual(s["chips"], 90)
        self.assertEqual(s["rows"]["a3_tuned"]["pooled"]["values"]["water_iou"], 1.0)
        self.assertEqual(s["rows"]["a1_assisted"]["pooled"]["values"]["water_iou"], 0.0)
        self.assertGreater(s["comparisons"]["decoder"]["difference_pooled_water_iou"], 0)
        lo, hi = (s["comparisons"]["decoder"]["bootstrap_ci"][k] for k in ("lo", "hi"))
        self.assertLessEqual(lo, hi); self.assertGreater(lo, 0)
        self.assertEqual(out["regimes"]["leave_paraguay_out"]["paraguay_heldout"]["chips"], 67)
        self.assertGreater(out["regimes"]["official"]["geographic_shift_descriptive"]["a3_tuned"]["seen"], 0.99)

    def test_wrong_chip_count_fails_instead_of_dropping(self):
        sets = synthetic(PROTO["regimes"]); sets["official"]["bolivia_heldout"].pop()
        with self.assertRaises(ValueError):
            evaluate(PROTO, sets, {r: {k: (lambda c: (c["labels"] == 1).astype(int)) for k in PROTO["rows"]} for r in PROTO["regimes"]})

    def test_paired_bootstrap_requires_same_ids(self):
        a = [{"id": "1", "confusion": {"tp": 1, "fp": 0, "fn": 0, "tn": 0}}]; b = [{"id": "2", "confusion": a[0]["confusion"]}]
        with self.assertRaises(ValueError): bootstrap_ci(a, b)


if __name__ == "__main__":
    unittest.main()
