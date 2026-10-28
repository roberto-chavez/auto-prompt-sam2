"""Read bounded public metadata and HEAD protected weights; never GET weights."""
from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request

from .artifacts import utc_now


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Never forward a token to a CDN or store signed URLs.


def public_metadata(url: str):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "auto-prompt-sam2-stage1"}), timeout=30) as response:
        content = response.read(2_000_001)
        if len(content) > 2_000_000:
            raise ValueError("Metadata exceeds size limit")
    return content


def head_access(repo_id: str, revision: str, filename: str, token: str | None = None):
    url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"
    headers = {"User-Agent": "auto-prompt-sam2-stage1"}
    if token:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(url, method="HEAD", headers=headers)
    try:
        with urllib.request.build_opener(NoRedirect).open(request, timeout=30) as response:
            status = response.status
    except urllib.error.HTTPError as exc:
        status = exc.code
    except (urllib.error.URLError, TimeoutError):
        return {"status": "network_error", "authorized": False}
    # HF returns a signed CDN redirect only after its authorization check.
    return {"http_status": status, "authorized": status in (200, 302, 303, 307),
            "method": "HEAD_no_redirect", "body_downloaded": False}


def check_access(config: dict, token: str | None = None):
    report = {"checked_at": utc_now(), "models": {}, "dataset": {}, "blockers": []}
    for name, model in config["models"].items():
        try:
            url = f'https://huggingface.co/api/models/{model["repo_id"]}/revision/{model["revision"]}'
            metadata = json.loads(public_metadata(url))
            files = [f["rfilename"] for f in metadata.get("siblings", [])]
            valid = metadata.get("sha") == model["revision"] and model["filename"] in files
            access = head_access(model["repo_id"], model["revision"], model["filename"], token)
            report["models"][name] = {"repo_id": model["repo_id"], "revision": metadata.get("sha"),
                                      "filename": model["filename"], "metadata_matches": valid,
                                      "gated": metadata.get("gated"), "access": access,
                                      "license": metadata.get("cardData", {}).get("license")}
            if not valid or not access["authorized"]:
                report["blockers"].append(f"{name}: revision/file mismatch or HEAD access denied; confirm account grant and HF_TOKEN read permission")
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError):
            report["models"][name] = {"status": "metadata_unavailable"}
            report["blockers"].append(f"{name}: metadata could not be verified")
    for kind in ("source", "label"):
        url = f'https://storage.googleapis.com/sen1floods11/v1.1/catalog/sen1floods11_hand_labeled_{kind}/collection.json'
        try:
            raw = public_metadata(url)
            meta = json.loads(raw)
            report["dataset"][kind] = {"url": url, "id": meta.get("id"), "license": meta.get("license"),
                                        "metadata_sha256": hashlib.sha256(raw).hexdigest(), "accessible": True}
        except (urllib.error.URLError, TimeoutError, ValueError):
            report["dataset"][kind] = {"url": url, "accessible": False}
            report["blockers"].append(f"Sen1Floods11 {kind}: catalog inaccessible")
    for layer in ("S2Hand", "LabelHand"):
        prefix = f"v1.1/data/flood_events/HandLabeled/{layer}/"
        query = urllib.parse.urlencode({"prefix": prefix, "maxResults": 1,
                                        "fields": "items(name,generation,size,md5Hash,crc32c)"})
        try:
            listing = json.loads(public_metadata(f"https://storage.googleapis.com/storage/v1/b/sen1floods11/o?{query}"))
            item = listing["items"][0]
            url = "https://storage.googleapis.com/sen1floods11/" + urllib.parse.quote(item["name"], safe="/")
            with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=30) as response:
                status = response.status
            report["dataset"][layer] = {"object_metadata": item, "http_status": status,
                                         "method": "HEAD", "body_downloaded": False}
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError, IndexError):
            report["dataset"][layer] = {"status": "access_unverified"}
            report["blockers"].append(f"Sen1Floods11 {layer}: sample object HEAD unverified")
    if config["dataset"]["license_status"] != "confirmed":
        report["blockers"].append("Sen1Floods11: license grant for S2Hand and LabelHand unresolved; obtain publisher clarification before acquisition")
    return report
