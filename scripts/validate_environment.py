"""Two independent processes for prepare/recover; restart Colab between them."""
import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from auto_prompt_sam2.artifacts import ArtifactStore, Run, canonical_json, digest
from auto_prompt_sam2.runtime import (import_checks, inventory, load_config, persistent_root,
                                    resource_blockers, run_identity, seed_everything, source_files,
                                    verify_installed_pins)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "recover"])
    parser.add_argument("--run-id")
    parser.add_argument("--colab", action="store_true")
    args = parser.parse_args()
    config = load_config(REPO)
    root = persistent_root(config, colab=args.colab)
    seed_everything(config["seed"], config["deterministic_algorithms"])
    imports = import_checks()
    pins = verify_installed_pins(REPO, config, linux_lock=args.colab)
    identity = run_identity(REPO, config)
    env_report = inventory(root, REPO)
    env_report.update({"imports": imports, "pins": pins, "seed": config["seed"], "precision": config["precision"],
                       "identity": identity, "scope": "colab" if args.colab else "local"})
    blockers = resource_blockers(config, env_report)
    sample = canonical_json({"message": "persistencia etapa 1", "seed": config["seed"], "values": [1, 2, 3]})
    import random
    import numpy as np
    import torch
    rng_probe = {"python": random.random(), "numpy": float(np.random.random()),
                 "torch_cpu": torch.rand(3).tolist()}
    registry = ArtifactStore(root / "environment")
    if args.mode == "prepare":
        run = Run.create(root, identity)
        run.transition("running")
        run.store.publish("environment", canonical_json(env_report), identity)
        run.store.publish("smoke", sample, identity)
        run.store.publish("seed_probe", canonical_json(rng_probe), identity)
        # Simulate a disconnect before publication: even an existing payload is invalid.
        incomplete = run.store.root / "incomplete" / "aborted-generation"
        incomplete.mkdir(parents=True)
        (incomplete / "payload").write_bytes(b"partial")
        # Preserve exact local source (including uncommitted files) outside /content.
        import io
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in source_files(REPO):
                archive.write(path, path.relative_to(REPO).as_posix())
        run.store.publish("source_snapshot", buffer.getvalue(), identity)
        registry.publish("recovery_ticket", canonical_json({"run_id": run.run_id}), identity)
        print(json.dumps({"run_id": run.run_id, "state": run.state, "blockers": blockers,
                          "next": "Restart Colab, remount Drive, rerun setup and recover"}, indent=2))
        return
    if args.run_id:
        run_id = args.run_id
    else:
        run_id = json.loads(registry.read("recovery_ticket", identity)[0])["run_id"]
    run = Run(root, run_id, identity)
    original = json.loads(run.store.read("environment", identity)[0])
    # Exact versions must match; hardware may change but is recorded again.
    if original["packages"] != env_report["packages"]:
        raise RuntimeError("Installed packages changed; rebuild exact environment before recovery")
    recovered, marker = run.store.read("smoke", identity)
    if recovered != sample:
        raise RuntimeError("Recovered test payload differs")
    if json.loads(run.store.read("seed_probe", identity)[0]) != rng_probe:
        raise RuntimeError("Seed probe differs across fresh processes")
    try:
        run.store.read("incomplete", identity)
    except FileNotFoundError:
        rejected_incomplete = True
    else:
        raise RuntimeError("Incomplete write was accepted")
    try:
        run.store.read("smoke", dict(identity, seed=config["seed"] + 1))
    except FileNotFoundError:
        rejected_mismatch = True
    else:
        raise RuntimeError("Incompatible identity was accepted")
    run.resume()
    result = {"run_id": run.run_id, "persistence_smoke": "passed", "artifact_sha256": marker["sha256"],
              "rejected_incomplete": rejected_incomplete, "rejected_mismatch": rejected_mismatch,
              "seed_probe_reproduced": True,
              "scope": env_report["scope"], "hardware_after_recovery": env_report,
              "resource_blockers": blockers, "stage1_closed": False,
              "note": "Smoke completion does not certify access/licenses or a real Colab restart"}
    run.store.publish("recovery_validation", canonical_json(result), identity)
    if run.state != "complete":
        run.transition("complete")  # Only the persistence test run, not stage closure.
    print(json.dumps({k: v for k, v in result.items() if k != "hardware_after_recovery"}, indent=2))


if __name__ == "__main__":
    main()
