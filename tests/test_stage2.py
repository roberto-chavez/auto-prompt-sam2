"""Scientific contracts: leakage, validity isolation, geometry and persistence."""
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import torch
from auto_prompt_sam2.artifacts import digest
from auto_prompt_sam2.data import manifests, development_records, parse_splits, input_rgb, read_labels
from auto_prompt_sam2.preprocessing import (Coordinates, histogram_percentile,
    rgb_unit, prepare, resize_categorical, fit_preprocessing, save_preprocessing,
    load_preprocessing, sam_predictor_image)


def synthetic_rows():
    rows = []
    n = 0
    for split, count, paraguay in (("train", 252, 39), ("val", 89, 14), ("test", 90, 14), ("bolivia", 15, 0)):
        for i in range(count):
            n += 1
            event = "Bolivia" if split == "bolivia" else "Paraguay" if i < paraguay else "Ghana"
            chip = f"{event}_{n}"
            rows.append({"id": chip, "event": event, "original_split": split,
                         "image_path": chip, "image_sha256": str(n)})
    return rows


class Stage2Contracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = json.loads((REPO / "configs/data.yaml").read_text("utf-8"))
        self.regimes = manifests(synthetic_rows())
        self.manifest = self.regimes["leave_paraguay_out"]

    def params(self):
        with patch("auto_prompt_sam2.preprocessing.input_rgb", return_value=(np.full((3, 2, 2), 2000, np.uint16), np.ones((2, 2), bool))):
            return fit_preprocessing(Path("."), self.manifest, self.config)

    def test_splits_disjoint_and_paraguay_excluded(self):
        for regime, manifest in self.regimes.items():
            sets = [{r["id"] for r in manifest["records"] if r["role"] == role} for role in manifest["counts"]]
            self.assertEqual(len(set.union(*sets)), 446)
            for i, a in enumerate(sets):
                for b in sets[i + 1:]:
                    self.assertFalse(a & b)
        self.assertEqual(self.manifest["counts"], {"train": 213, "val": 75, "test": 76, "paraguay_reserved": 67, "bolivia": 15})
        for role in ("train", "val"):
            self.assertFalse(any(r["event"] in ("Paraguay", "Bolivia") for r in development_records(self.manifest, regime="leave_paraguay_out", role=role)))
        for regime, role in (("official", "train"), ("leave_paraguay_out", "test"), ("leave_paraguay_out", "paraguay_reserved")):
            with self.assertRaises(ValueError):
                development_records(self.manifest, regime=regime, role=role)

    def test_fit_reads_only_allowed_train_without_labels(self):
        accessed = []
        def reader(path):
            accessed.append(str(path))
            return np.array([[[100, 200]], [[300, 400]], [[500, 65535]]], dtype=np.uint16), np.array([[True, False]])
        with patch("auto_prompt_sam2.preprocessing.input_rgb", side_effect=reader):
            p = fit_preprocessing(Path("."), self.manifest, self.config)
        self.assertEqual(len(accessed), 213)
        self.assertTrue(all("Paraguay" not in s and "Bolivia" not in s for s in accessed))
        self.assertEqual(p["valid_rgb_values"], 213 * 3)
        self.assertEqual(p["upper_dn"], 500)
        self.assertEqual(set(accessed), set(p["source_ids"]))

    def test_exact_percentile_histogram(self):
        x = np.array([0, 0, 1, 5, 200, 65535], np.uint16)
        hist = np.bincount(x, minlength=65536)
        for q in (1, 50, 99, 100):
            self.assertAlmostEqual(histogram_percentile(hist, q), np.percentile(x, q, method="linear"))
        with self.assertRaises(ValueError):
            histogram_percentile(np.zeros(65536, np.int64), 99)

    def test_real_raster_band_order_signed_dtype_nodata_and_labels(self):
        import rasterio
        from rasterio.transform import from_origin
        with tempfile.TemporaryDirectory() as folder:
            image = Path(folder) / "image.tif"
            label = Path(folder) / "label.tif"
            profile = dict(driver="GTiff", height=2, width=2, crs="EPSG:4326",
                           transform=from_origin(100, 20, 0.001, 0.001), dtype="int16")
            data = np.stack([np.full((2, 2), b * 100, np.int16) for b in range(1, 14)])
            data[3, 0, 0] = 0
            with rasterio.open(image, "w", count=13, nodata=0, **profile) as ds:
                ds.write(data)
            rgb, valid = input_rgb(image)
            np.testing.assert_array_equal(rgb[:, 1, 1], [400, 300, 200])
            self.assertFalse(valid[0, 0])
            self.assertTrue(valid[1, 1])
            p = self.params()
            self.assertTrue(np.isfinite(rgb_unit(rgb, valid, p)).all())
            for values in ([-1, 0, 1, 0], [-1] * 4, [0] * 4):
                with rasterio.open(label, "w", count=1, **profile) as ds:
                    ds.write(np.array(values, np.int16).reshape(1, 2, 2))
                labels, label_valid = read_labels(label)
                np.testing.assert_array_equal(label_valid, labels != -1)
                # Label validity cannot affect image validity or raw RGB.
                rgb2, valid2 = input_rgb(image)
                np.testing.assert_array_equal(rgb, rgb2)
                np.testing.assert_array_equal(valid, valid2)
            with rasterio.open(label, "w", count=1, **profile) as ds:
                ds.write(np.full((1, 2, 2), 2, np.int16))
            with self.assertRaises(ValueError):
                read_labels(label)

    def test_official_csv_id_pairing_and_hyphenated_event(self):
        import csv
        rows = synthetic_rows()
        rows[-20]["event"] = "Sri-Lanka"
        number = rows[-20]["id"].split("_")[-1]
        rows[-20]["id"] = "Sri-Lanka_" + number
        with tempfile.TemporaryDirectory() as folder:
            dest = Path(folder) / "raw/v1.1/splits/flood_handlabeled"
            dest.mkdir(parents=True)
            for role, csv_role in (("train", "train"), ("val", "valid"), ("test", "test"), ("bolivia", "bolivia")):
                with (dest / f"flood_{csv_role}_data.csv").open("w", newline="") as out:
                    csv.writer(out).writerows([[r["id"] + "_S1Hand.tif", r["id"] + "_LabelHand.tif"]
                                               for r in rows if r["original_split"] == role])
            parsed = parse_splits(folder)
            self.assertEqual(len(parsed), 446)
            self.assertTrue(any(r["event"] == "Sri-Lanka" for r in parsed))
            path = dest / "flood_train_data.csv"
            path.write_text(path.read_text().replace("Paraguay_1_LabelHand", "Paraguay_2_LabelHand"))
            with self.assertRaises(ValueError):
                parse_splits(folder)

    def test_persistent_reload_and_wrong_regime_rejected(self):
        p = self.params()
        with tempfile.TemporaryDirectory() as folder:
            generation = save_preprocessing(folder, p)
            got = load_preprocessing(folder, self.manifest, self.config, generation)
            self.assertEqual(got, p)
            rgb = np.full((3, 2, 2), 1000, np.uint16)
            valid = np.ones((2, 2), bool)
            for model in ("sam", "dino"):
                self.assertTrue(torch.equal(prepare(rgb, valid, p, model=model)["pixel_values"], prepare(rgb, valid, got, model=model)["pixel_values"]))
            with self.assertRaises(FileNotFoundError):
                load_preprocessing(folder, self.regimes["official"], self.config)
            changed = copy.deepcopy(self.config)
            changed["clip_percentile"] = 95
            with self.assertRaises(FileNotFoundError):
                load_preprocessing(folder, self.manifest, changed)

    def test_input_validity_is_independent_from_labels_and_invalid_fill(self):
        p = self.params()
        rgb = np.full((3, 2, 2), 1000, np.uint16)
        valid = np.array([[True, False], [True, True]])
        a = rgb_unit(rgb, valid, p)
        # Label changes have no path into this function or its validity mask.
        changed = rgb.copy()
        changed[:, ~valid] = 65535
        np.testing.assert_array_equal(a, rgb_unit(changed, valid, p))
        self.assertTrue((a[:, ~valid] == 0).all())
        self.assertTrue((a[:, valid] == 0.5).all())
        with self.assertRaises(ValueError):
            rgb_unit(a, valid, p)
        all_invalid = prepare(rgb, np.zeros((2, 2), bool), p, model="dino")
        self.assertFalse(all_invalid["input_valid"].any())
        self.assertTrue(torch.isfinite(all_invalid["pixel_values"]).all())

    def test_sam_exact_official_transform_without_constructing_model(self):
        from sam2.utils.transforms import SAM2Transforms
        p = self.params()
        rgb = np.arange(3 * 7 * 5, dtype=np.uint16).reshape(3, 7, 5) * 100
        valid = np.ones((7, 5), bool)
        official = SAM2Transforms(1024, 0.0)
        expected = official(sam_predictor_image(rgb, valid, p))
        actual = prepare(rgb, valid, p, model="sam")["pixel_values"]
        self.assertTrue(torch.equal(actual, expected))

    def test_dino_normalization_and_no_crop(self):
        p = self.params()
        rgb = np.full((3, 512, 512), 1000, np.uint16)
        out = prepare(rgb, np.ones((512, 512), bool), p, model="dino")
        self.assertEqual(tuple(out["pixel_values"].shape), (3, 512, 512))
        np.testing.assert_allclose(out["pixel_values"][:, 0, 0].numpy(),
            (0.5 - np.array([0.430, 0.411, 0.296])) / np.array([0.213, 0.156, 0.143]), rtol=1e-6)

    def test_coordinate_inverse_non_square_patch_centers_and_labels(self):
        xy = np.array([[0., 0.], [4., 6.], [1.25, 2.75]])
        c = Coordinates((7, 5), (1024, 1024))
        np.testing.assert_allclose(c.points([[4, 6]]), [[819.2, 6 * 1024 / 7]])
        for centers in (False, True):
            np.testing.assert_allclose(c.points(c.points(xy, pixel_centers=centers), inverse=True, pixel_centers=centers), xy, atol=1e-6)
        np.testing.assert_allclose(c.from_patch(c.to_patch(xy)), xy, atol=1e-6)
        native = Coordinates((512, 512), (512, 512))
        np.testing.assert_array_equal(native.from_patch([[0, 0], [31, 31]]), [[7.5, 7.5], [503.5, 503.5]])
        labels = np.array([[-1, 0], [1, 0]], np.int16)
        self.assertEqual(set(np.unique(resize_categorical(labels, (17, 21)))), {-1, 0, 1})
        waterless = np.zeros((512, 512), np.int16)
        self.assertEqual(set(np.unique(resize_categorical(waterless, (1024, 1024)))), {0})


if __name__ == "__main__":
    unittest.main()
