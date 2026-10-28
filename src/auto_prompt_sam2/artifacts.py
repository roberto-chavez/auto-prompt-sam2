"""Immutable generations, verified completion markers and compatible recovery.

One writer per run is required. Remote rename/fsync atomicity is not assumed.
Readers verify size and SHA-256 even when a completion marker exists.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path


def canonical_json(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_name(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name):
        raise ValueError("Invalid artifact/run name")
    return name


def _write(path: Path, payload: bytes):
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        try:
            os.fsync(stream.fileno())
        except OSError:
            # Drive FUSE may not support fsync; verification remains mandatory.
            pass


class ArtifactStore:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def publish(self, name: str, payload: bytes, identity: dict) -> str:
        folder = self.root / safe_name(name)
        folder.mkdir(exist_ok=True)
        generation = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:12]
        target = folder / generation
        target.mkdir()  # Never overwrite a prior generation, even if interrupted.
        _write(target / "payload.pending", payload)
        expected = hashlib.sha256(payload).hexdigest()
        if sha256_file(target / "payload.pending") != expected:
            raise IOError("Write verification failed")
        # Readers ignore this directory until COMPLETE.json is valid.
        (target / "payload.pending").rename(target / "payload")
        marker = {"schema_version": 1, "generation": generation,
                  "sha256": expected, "size": len(payload), "identity": identity,
                  "identity_sha256": digest(identity), "created_at": utc_now()}
        _write(target / "COMPLETE.json", canonical_json(marker))
        self.read(name, identity, generation)
        return generation

    def read(self, name: str, identity: dict, generation: str | None = None):
        folder = self.root / safe_name(name)
        candidates = ([folder / safe_name(generation)] if generation else
                      sorted(folder.iterdir(), reverse=True) if folder.exists() else [])
        for candidate in candidates:
            try:
                marker = json.loads((candidate / "COMPLETE.json").read_text("utf-8"))
                if marker["identity_sha256"] != digest(marker["identity"]):
                    continue
                if marker["identity_sha256"] != digest(identity):
                    continue
                payload = (candidate / "payload").read_bytes()
                if (marker["schema_version"] == 1 and marker["generation"] == candidate.name
                        and len(payload) == marker["size"]
                        and hashlib.sha256(payload).hexdigest() == marker["sha256"]):
                    return payload, marker
            except (OSError, ValueError, KeyError, TypeError):
                continue
        raise FileNotFoundError("No complete, intact artifact with matching identity")


TRANSITIONS = {
    "planned": {"running", "failed"},
    "running": {"interrupted", "complete", "failed"},
    "interrupted": {"running", "failed"},
    "failed": set(), "complete": set(),
}


class Run:
    def __init__(self, root: Path, run_id: str, identity: dict):
        self.run_id = safe_name(run_id)
        self.identity = identity
        self.store = ArtifactStore(Path(root) / "environment" / "runs" / self.run_id)

    @classmethod
    def create(cls, root: Path, identity: dict):
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + digest(identity)[:10] + "_" + uuid.uuid4().hex[:8]
        run = cls(root, run_id, identity)
        run.store.publish("state", canonical_json({"state": "planned", "at": utc_now()}), identity)
        return run

    @property
    def state(self):
        return json.loads(self.store.read("state", self.identity)[0])["state"]

    def transition(self, state: str):
        if state not in TRANSITIONS[self.state]:
            raise ValueError("Invalid run state transition")
        self.store.publish("state", canonical_json({"state": state, "at": utc_now()}), self.identity)

    def resume(self):
        # Call only after confirming the old writer/session has stopped.
        if self.state == "running":
            self.transition("interrupted")
        if self.state == "interrupted":
            self.transition("running")
        elif self.state != "complete":
            raise ValueError("Run cannot be resumed")
        return self
