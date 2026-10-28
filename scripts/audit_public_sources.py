"""Record reproducible primary-source metadata without acquiring model/data bytes."""
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from auto_prompt_sam2.access import public_metadata, check_access
from auto_prompt_sam2.artifacts import utc_now
from auto_prompt_sam2.runtime import load_config


def main():
    config = load_config(REPO)
    sam = config["models"]["sam2"]
    data = config["dataset"]
    sources = {
        "sam_setup": f'https://raw.githubusercontent.com/facebookresearch/sam2/{sam["source_revision"]}/setup.py',
        "sam_license": f'https://raw.githubusercontent.com/facebookresearch/sam2/{sam["source_revision"]}/LICENSE',
        "data_readme": f'https://raw.githubusercontent.com/{data["documentation_repo"]}/{data["documentation_revision"]}/README.md',
        "dinov3_license": "https://raw.githubusercontent.com/facebookresearch/dinov3/6876159a11b4df116f30f667f8c9888617df0751/LICENSE.md",
    }
    report = {"checked_at": utc_now(), "sources": {}, "access": check_access(config)}
    for name, url in sources.items():
        content = public_metadata(url)
        report["sources"][name] = {"url": url, "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
    for name, repo, revision in (("sam_source", "facebookresearch/sam2", sam["source_revision"]),
                                 ("data_documentation", data["documentation_repo"], data["documentation_revision"])):
        url = f"https://api.github.com/repos/{repo}/commits/{revision}"
        meta = json.loads(public_metadata(url))
        assert meta["sha"] == revision
        report["sources"][name] = {"url": url, "revision": meta["sha"], "confirmed": True}
    path = REPO / "docs/project/evidence/stage1/public_sources.json"
    path.write_text(json.dumps(report, indent=2) + "\n", "utf-8")
    print("Public-source audit saved; no data/weights acquired.")


if __name__ == "__main__":
    main()
