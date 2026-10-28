import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import urllib.error

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from auto_prompt_sam2.artifacts import ArtifactStore, Run, digest
from auto_prompt_sam2.access import head_access
from auto_prompt_sam2.runtime import load_config, persistent_root
sys.path.insert(0, str(REPO / "scripts"))
from export_stage1_evidence import export_evidence


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = ArtifactStore(self.root)
        self.identity = {"seed": 42, "config": "abc", "models": "def", "splits": None}

    def test_recover_in_new_interpreter(self):
        generation = self.store.publish("sample", b"small artifact", self.identity)
        code = ("import sys,json; from pathlib import Path; sys.path.insert(0,sys.argv[1]); "
                "from auto_prompt_sam2.artifacts import ArtifactStore; "
                "assert ArtifactStore(Path(sys.argv[2])).read('sample',json.loads(sys.argv[3]))[0] == b'small artifact'")
        subprocess.run([sys.executable, "-I", "-c", code, str(REPO / "src"),
                        str(self.root), json.dumps(self.identity)], check=True)
        self.assertTrue((self.root / "sample" / generation / "COMPLETE.json").is_file())

    def test_incomplete_and_corrupt_generation_preserve_previous(self):
        old = self.store.publish("sample", b"old", self.identity)
        new = self.store.publish("sample", b"new", self.identity)
        (self.root / "sample" / new / "payload").write_bytes(b"bad")
        self.assertEqual(self.store.read("sample", self.identity)[1]["generation"], old)
        (self.root / "sample" / new / "COMPLETE.json").write_text('{"broken":', "utf-8")
        self.assertEqual(self.store.read("sample", self.identity)[0], b"old")
        with self.assertRaises(FileNotFoundError):
            self.store.read("sample", self.identity, new)

    def test_payload_without_marker_is_rejected(self):
        path = self.root / "sample" / "unfinished"
        path.mkdir(parents=True)
        (path / "payload").write_bytes(b"data")
        with self.assertRaises(FileNotFoundError):
            self.store.read("sample", self.identity)

    def test_each_identity_change_is_rejected(self):
        self.store.publish("sample", b"test", self.identity)
        for key in self.identity:
            with self.subTest(key=key), self.assertRaises(FileNotFoundError):
                self.store.read("sample", dict(self.identity, **{key: "changed"}))

    def test_no_path_escape(self):
        for name in ("../other", "/tmp/a", "a/b", "a\\b", "..", ""):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.store.publish(name, b"test", self.identity)

    def test_state_and_compatible_resume(self):
        run = Run.create(self.root, self.identity)
        self.assertEqual(run.state, "planned")
        run.transition("running")
        recovered = Run(self.root, run.run_id, self.identity).resume()
        self.assertEqual(recovered.state, "running")
        recovered.transition("complete")
        with self.assertRaises(ValueError):
            recovered.transition("running")
        with self.assertRaises(FileNotFoundError):
            Run(self.root, run.run_id, {"other": 1}).resume()

    def test_canonical_hash(self):
        self.assertEqual(digest({"a": 1, "b": 2}), digest({"b": 2, "a": 1}))
        with self.assertRaises(ValueError):
            digest({"nan": float("nan")})

    def test_colab_temporary_root_blocked(self):
        config = load_config(REPO)
        with patch.dict("os.environ", {"AUTO_PROMPT_PERSIST_ROOT": "/content/ephemeral"}):
            with self.assertRaises(ValueError):
                persistent_root(config, colab=True)

    def test_head_never_logs_token_or_follows_redirect(self):
        secret = "fake-secret-for-test-only"
        class FakeOpener:
            def open(inner, request, timeout):
                self.assertEqual(request.get_method(), "HEAD")
                self.assertEqual(request.get_header("Authorization"), "Bearer " + secret)
                raise urllib.error.HTTPError(request.full_url, 302, secret, {}, None)
        output = io.StringIO()
        with patch("urllib.request.build_opener", return_value=FakeOpener()), contextlib.redirect_stdout(output):
            result = head_access("facebook/example", "revision", "weights.pt", secret)
        self.assertTrue(result["authorized"])
        self.assertNotIn(secret, json.dumps(result) + output.getvalue())

    def test_export_uses_saved_identity_and_only_verified_json(self):
        import zipfile
        run = Run.create(self.root, self.identity)
        run.store.publish("environment", json.dumps({"identity": self.identity}).encode(), self.identity)
        run.store.publish("recovery_validation", b'{"persistence_smoke":"passed"}', self.identity)
        run.store.publish("source_snapshot", b"not exported", self.identity)
        ArtifactStore(self.root / "environment").publish("access", b'{"models":{}}', self.identity)
        output = self.root / "evidence.zip"
        manifest = export_evidence(self.root, run.run_id, output)
        self.assertEqual(manifest["missing_artifacts"], [])
        with zipfile.ZipFile(output) as archive:
            self.assertEqual(set(archive.namelist()), {"environment.json", "recovery_validation.json",
                "state.json", "access.json", "artifact_markers.json", "export_manifest.json"})

    def test_export_reports_absent_access_instead_of_claiming_authorization(self):
        run = Run.create(self.root, self.identity)
        run.store.publish("environment", json.dumps({"identity": self.identity}).encode(), self.identity)
        manifest = export_evidence(self.root, run.run_id, self.root / "evidence.zip")
        self.assertEqual(set(manifest["missing_artifacts"]), {"access", "recovery_validation"})
        self.assertFalse(manifest["stage1_closed"])


if __name__ == "__main__":
    unittest.main()
