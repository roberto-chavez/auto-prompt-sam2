"""Sen1Floods11 acquisition and structural audit. No model or reserved-set EDA."""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import re
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import bounds as feature_bounds

from .artifacts import ArtifactStore, canonical_json, digest, sha256_file, utc_now

BUCKET = "https://storage.googleapis.com/storage/v1/b/sen1floods11/o"
BASE = "v1.1/data/flood_events/HandLabeled/"
SPLITS = {"train": 252, "val": 89, "test": 90, "bolivia": 15}
CSV_NAMES = {"train": "train", "val": "valid", "test": "test", "bolivia": "bolivia"}


def get_bytes(url):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=90) as stream:
                return stream.read()
        except (OSError, TimeoutError):
            if attempt == 3:
                raise
            time.sleep(1 + attempt)


def list_objects(prefix):
    items, token = [], None
    while True:
        params = {"prefix": prefix, "maxResults": 1000}
        if token:
            params["pageToken"] = token
        page = json.loads(get_bytes(BUCKET + "?" + urllib.parse.urlencode(params)))
        items.extend(page.get("items", []))
        token = page.get("nextPageToken")
        if not token:
            return items


def acquire(root: Path, workers=8):
    """Download only the two manual layers, official CSVs and event metadata.

    Pins GCS generations, checks publisher MD5/size, and records SHA-256.
    Existing bytes are reused only after verification against the same metadata.
    """
    root = Path(root)
    objects = []
    for prefix in (BASE + "S2Hand/", BASE + "LabelHand/", "v1.1/splits/flood_handlabeled/"):
        objects.extend(list_objects(prefix))
    objects.append(json.loads(get_bytes(BUCKET + "/" + urllib.parse.quote(
        "v1.1/Sen1Floods11_Metadata.geojson", safe=""))))
    for layer in ("S2Hand", "LabelHand"):
        if sum(o["name"].endswith("_" + layer + ".tif") for o in objects) != 446:
            raise ValueError("Official manual-layer inventory changed")

    def download(obj):
        name = obj["name"]
        if ".." in Path(name).parts or not name.startswith("v1.1/"):
            raise ValueError("Unsafe object path")
        path = root / "raw" / name
        path.parent.mkdir(parents=True, exist_ok=True)

        def valid(p):
            return (p.is_file() and p.stat().st_size == int(obj["size"]) and
                    base64.b64encode(hashlib.md5(p.read_bytes()).digest()).decode() == obj["md5Hash"])

        url = ("https://storage.googleapis.com/download/storage/v1/b/sen1floods11/o/" +
               urllib.parse.quote(name, safe="") + "?alt=media&generation=" + obj["generation"])
        if not valid(path):
            pending = path.with_suffix(path.suffix + ".pending")
            pending.write_bytes(get_bytes(url))
            if not valid(pending):
                raise IOError("Publisher checksum mismatch: " + name)
            pending.replace(path)
        return {"object": name, "path": path.relative_to(root).as_posix(), "url": url,
                "generation": obj["generation"], "publisher_md5": obj["md5Hash"],
                "publisher_crc32c": obj.get("crc32c"), "size": int(obj["size"]),
                "sha256": sha256_file(path)}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(download, objects))
    result = {"acquired_at": utc_now(), "objects": sorted(records, key=lambda x: x["object"])}
    ArtifactStore(root / "raw").publish("acquisition", canonical_json(result), {"dataset": "Sen1Floods11-v1.1"})
    return result


def parse_splits(root):
    rows, ids, numeric_ids = [], set(), set()
    for split, expected in SPLITS.items():
        path = Path(root) / "raw/v1.1/splits/flood_handlabeled" / f"flood_{CSV_NAMES[split]}_data.csv"
        parsed = [r for r in csv.reader(io.StringIO(path.read_text("utf-8"))) if r]
        if len(parsed) != expected:
            raise ValueError("Unexpected official split count")
        for pair in parsed:
            if len(pair) != 2:
                raise ValueError("Invalid CSV pair")
            match = re.fullmatch(r"([A-Za-z]+(?:-[A-Za-z]+)*)_([0-9]+)_S1Hand\.tif", pair[0])
            if match is None:
                raise ValueError("Invalid chip ID")
            event, number = match.groups()
            chip = event + "_" + number
            if pair[1] != chip + "_LabelHand.tif" or chip in ids or number in numeric_ids:
                raise ValueError("Duplicate ID or unmatched label")
            if (event == "Bolivia") != (split == "bolivia"):
                raise ValueError("Bolivia leaked")
            ids.add(chip)
            numeric_ids.add(number)
            rows.append({"id": chip, "event": event, "original_split": split,
                         "image_path": "raw/" + BASE + "S2Hand/" + chip + "_S2Hand.tif",
                         "label_path": "raw/" + BASE + "LabelHand/" + pair[1]})
    if Counter(r["original_split"] for r in rows if r["event"] == "Paraguay") != {"train": 39, "val": 14, "test": 14}:
        raise ValueError("Unexpected Paraguay membership")
    return rows


def geometry(ds):
    return {"height": ds.height, "width": ds.width, "crs": str(ds.crs),
            "transform": list(ds.transform)[:6], "bounds": list(ds.bounds),
            "resolution_crs_units": list(ds.res), "nodata": ds.nodata,
            "count": ds.count, "dtypes": list(ds.dtypes), "band_descriptions": list(ds.descriptions)}


def input_rgb(path):
    """No label argument: automatic inference validity derives only from input."""
    with rasterio.open(path) as ds:
        rgb = ds.read([4, 3, 2])
        valid = (ds.read_masks([4, 3, 2]) > 0).all(axis=0) & np.isfinite(rgb).all(axis=0)
    return rgb, valid


def read_labels(path):
    with rasterio.open(path) as ds:
        labels = ds.read(1)
    if not np.isin(labels, [-1, 0, 1]).all():
        raise ValueError("Labels outside -1/0/1")
    return labels, labels != -1


def audit(root, acquisition, *, event_aliases=None):
    root = Path(root)
    rows = parse_splits(root)
    by_path = {o["path"]: o for o in acquisition["objects"]}
    events = json.loads((root / "raw/v1.1/Sen1Floods11_Metadata.geojson").read_text("utf-8"))
    event_metadata = {f["properties"]["location"]: f for f in events["features"]}
    event_aliases = event_aliases or {}
    image_hashes, label_hashes, pixel_hashes = defaultdict(list), defaultdict(list), defaultdict(list)
    max_alignment_error_px = 0.0
    for row in rows:
        location = event_aliases.get(row["event"], row["event"])
        if location not in event_metadata:
            raise ValueError("Unknown event")
        event = event_metadata[location]
        props = event["properties"]
        row["event_metadata"] = {k: props[k] for k in ("ID", "location", "ISO_CC", "s2_date")}
        for kind in ("image", "label"):
            path = row[kind + "_path"]
            source = by_path[path]
            if sha256_file(root / path) != source["sha256"]:
                raise ValueError("Raw bytes changed")
            row[kind + "_sha256"] = source["sha256"]
            row[kind + "_generation"] = source["generation"]
        with rasterio.open(root / row["image_path"]) as image, rasterio.open(root / row["label_path"]) as label:
            a, b = geometry(image), geometry(label)
            if (a["height"], a["width"], a["count"]) != (512, 512, 13) or b["count"] != 1:
                raise ValueError("Unexpected raster dimensions")
            if a["dtypes"] not in (["uint16"] * 13, ["int16"] * 13) or b["dtypes"] != ["int16"]:
                raise ValueError("Unexpected dtype")
            if a["band_descriptions"] != ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]:
                raise ValueError("GeoTIFF band order differs from official documentation")
            for key in ("height", "width", "crs"):
                if a[key] != b[key]:
                    raise ValueError("Image/label geometry mismatch: " + row["id"])
            error = max(abs(z - q) for xy in ((0, 0), (512, 0), (0, 512), (512, 512))
                        for z, q in zip((~image.transform) * (label.transform * xy), xy))
            max_alignment_error_px = max(max_alignment_error_px, error)
            if error > 1e-6:
                raise ValueError("Image/label alignment exceeds 1e-6 pixels")
            if a["crs"] != "EPSG:4326" or image.transform.b != 0 or image.transform.d != 0:
                raise ValueError("Unexpected CRS/rotation")
            row["geometry"] = a
            row["label_transform"] = b["transform"]
            row["alignment_error_px"] = error
            row["label_nodata"] = label.nodata
            ex1, ey1, ex2, ey2 = feature_bounds(event)
            x1, y1, x2, y2 = a["bounds"]
            if not (ex1 <= (x1 + x2) / 2 <= ex2 and ey1 <= (y1 + y2) / 2 <= ey2):
                raise ValueError("Chip center outside declared event footprint envelope")
            # Structural read only: no class balance, visualisation or quality statistics.
            if not np.isin(label.read(1), [-1, 0, 1]).all():
                raise ValueError("Invalid categorical label")
            pixel_hashes[hashlib.sha256(image.read().tobytes()).hexdigest()].append(row["id"])
        image_hashes[row["image_sha256"]].append(row["id"])
        label_hashes[row["label_sha256"]].append(row["id"])
    duplicates = {"image_files": [v for v in image_hashes.values() if len(v) > 1],
                  "image_pixels": [v for v in pixel_hashes.values() if len(v) > 1],
                  "label_files": [v for v in label_hashes.values() if len(v) > 1]}
    if duplicates["image_files"] or duplicates["image_pixels"]:
        raise ValueError("Duplicate images require protocol review: " + str(duplicates))
    spatial = []
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            if a["event"] != b["event"]:
                continue
            x1, y1, x2, y2 = a["geometry"]["bounds"]
            u1, v1, u2, v2 = b["geometry"]["bounds"]
            dx, dy = min(x2, u2) - max(x1, u1), min(y2, v2) - max(y1, v1)
            if dx >= -1e-9 and dy >= -1e-9:
                spatial.append({"a": a["id"], "b": b["id"], "cross_split": a["original_split"] != b["original_split"],
                                "relation": "overlap" if dx > 1e-9 and dy > 1e-9 else "touch"})
    return rows, {"chips": len(rows), "duplicates": duplicates, "spatial_relations": spatial,
                  "labels_allowed": [-1, 0, 1], "reserved_access": "structural_only",
                  "max_alignment_error_px": max_alignment_error_px, "alignment_tolerance_px": 1e-6,
                  "event_metadata_aliases": event_aliases, "event_footprint_envelopes": "all_chip_centers_verified",
                  "reserved_exploration": [], "events": sorted({r["event"] for r in rows})}


def manifests(rows):
    result = {}
    for regime in ("official", "leave_paraguay_out"):
        records = []
        for row in rows:
            role = row["original_split"]
            if regime == "leave_paraguay_out" and row["event"] == "Paraguay":
                role = "paraguay_reserved"
            records.append(dict(row, regime=regime, role=role))
        expected = ({"train": 252, "val": 89, "test": 90, "bolivia": 15} if regime == "official" else
                    {"train": 213, "val": 75, "test": 76, "paraguay_reserved": 67, "bolivia": 15})
        if Counter(r["role"] for r in records) != expected or len({r["id"] for r in records}) != 446:
            raise ValueError("Regime contract failed")
        result[regime] = {"regime": regime, "counts": expected, "records": records}
    return result


def development_records(manifest, *, regime, role):
    if regime != manifest["regime"] or role not in ("train", "val"):
        raise ValueError("Explicit matching regime and development role required")
    rows = [r for r in manifest["records"] if r["role"] == role]
    if regime == "leave_paraguay_out" and any(r["event"] == "Paraguay" for r in rows):
        raise ValueError("Paraguay leakage")
    return rows
