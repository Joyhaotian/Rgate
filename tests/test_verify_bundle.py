from __future__ import annotations

import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import verify_bundle


SOURCE_ROOT = Path(__file__).resolve().parents[1]


class BundleVerifierRegressionTest(unittest.TestCase):
    def candidate(self):
        temporary = TemporaryDirectory(prefix="rgate-bundle-test-")
        root = Path(temporary.name) / "candidate"
        shutil.copytree(
            SOURCE_ROOT,
            root,
            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", ".mypy_cache"),
        )
        # The published candidate may already contain normalized public models.
        # These regression fixtures intentionally start from the private-source
        # state so they can test unregistered-file and missing-artifact paths.
        manifest_path = root / "ARTIFACT_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in manifest.get("artifacts", []):
            relative = item.get("relative_path")
            if isinstance(relative, str):
                target = root / relative
                if target.is_file() and not target.is_symlink():
                    target.unlink()
        receipts = root / "normalization_receipts"
        if receipts.is_dir():
            shutil.rmtree(receipts)
        # Runtime artifacts are intentionally outside the published file
        # closure, but the regression fixture exercises that boundary by
        # placing one file below the directory.
        (root / "artifacts").mkdir(parents=True, exist_ok=True)
        manifest["public_artifacts"] = []
        manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        return temporary, root

    def test_unregistered_private_source_at_public_path_fails(self):
        temporary, root = self.candidate()
        self.addCleanup(temporary.cleanup)
        target = root / "models/seed_00/radar_model.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / "ARTIFACT_MANIFEST.json", target)
        errors, warnings = [], []
        with mock.patch.object(verify_bundle, "ROOT", root):
            verify_bundle.verify_artifacts(True, errors, warnings)
        self.assertTrue(any("unregistered file at public artifact path" in item for item in errors))

    def test_absolute_config_value_fails(self):
        temporary, root = self.candidate()
        self.addCleanup(temporary.cleanup)
        path = root / "configs/repro.example.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["paths"]["nuscenes_root"] = "/opt/private/data"
        path.write_text(json.dumps(value), encoding="utf-8")
        errors = []
        with mock.patch.object(verify_bundle, "ROOT", root):
            verify_bundle.verify_json_and_relative_paths(errors)
        self.assertIn("non-relative configured path at paths.nuscenes_root", errors)

    def test_runtime_and_interpreter_caches_are_outside_release_closure(self):
        temporary, root = self.candidate()
        self.addCleanup(temporary.cleanup)
        cache = root / "tests/__pycache__/transient.pyc"
        cache.parent.mkdir(parents=True)
        cache.write_bytes(b"transient")
        runtime = root / "outputs/result.json"
        runtime.parent.mkdir(parents=True)
        runtime.write_text("{}", encoding="utf-8")
        covered_artifact = root / "artifacts/leak.json"
        covered_artifact.write_text("{}", encoding="utf-8")
        with mock.patch.object(verify_bundle, "ROOT", root):
            files = verify_bundle.regular_files()
        self.assertNotIn("tests/__pycache__/transient.pyc", files)
        self.assertNotIn("outputs/result.json", files)
        self.assertIn("artifacts/leak.json", files)

    def test_import_closure_does_not_require_python310_stdlib_index(self):
        errors = []
        with mock.patch.object(verify_bundle.sys, "stdlib_module_names", ()):
            verify_bundle.verify_import_and_syntax_closure(errors)
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
