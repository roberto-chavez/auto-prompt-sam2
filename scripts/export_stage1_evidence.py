"""Export existing stage-1 JSON artifacts without creating a new run.

Works with the old bundle too: identities come from the saved, verified marker,
not from the current source tree. No source snapshots, tokens or weights exported.
"""
import argparse
import json
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from auto_prompt_sam2.artifacts import ArtifactStore, canonical_json, safe_name, sha256_file, utc_now


def existing_run(store: ArtifactStore):
    folder = store.root / "environment"
    for generation in sorted(folder.iterdir(), reverse=True) if folder.exists() else []:
        try:
            marker = json.loads((generation / "COMPLETE.json").read_text("utf-8"))
            identity = marker["identity"]
            payload, verified = store.read("environment", identity, generation.name)
            environment = json.loads(payload)
            if environment["identity"] != identity:
                continue
            return identity, payload, verified
        except (OSError, ValueError, KeyError, TypeError):
            continue
    raise FileNotFoundError("No verified environment artifact for this run")


def export_evidence(root: Path, run_id: str, output: Path):
    store = ArtifactStore(root / "environment/runs" / safe_name(run_id))
    identity, environment, marker = existing_run(store)
    files = {"environment.json": environment}
    markers = {"environment": marker}
    missing = []
    for name in ("recovery_validation", "state"):
        try:
            payload, markers[name] = store.read(name, identity)
            files[name + ".json"] = payload
        except FileNotFoundError:
            missing.append(name)
    try:
        payload, markers["access"] = ArtifactStore(root / "environment").read("access", identity)
        files["access.json"] = payload
    except FileNotFoundError:
        missing.append("access")
    manifest = {"schema_version": 1, "exported_at": utc_now(), "run_id": run_id,
                "missing_artifacts": missing, "stage1_closed": False,
                "note": "Export does not establish a runtime restart or grant any license"}
    files["artifact_markers.json"] = canonical_json(markers)
    files["export_manifest.json"] = canonical_json(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in files.items():
            archive.writestr(name, payload)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = export_evidence(args.root, args.run_id, args.output)
    print(json.dumps(manifest, indent=2))
    print("Evidence ZIP:", args.output)
    print("SHA256:", sha256_file(args.output))
