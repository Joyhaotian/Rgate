from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import assemble_normalized_release as assembler  # noqa: E402
import normalize_learned_artifact as normalizer  # noqa: E402


def canonical(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def synthetic_model(seed):
    return {
        "schema_version": "bge_af_mlp_arbiter_v1",
        "numeric_features": ["signal", "offset"],
        "categorical_levels": {"kind": ["alpha", "beta"]},
        "stats": {
            "signal": {"mean": 1.0, "std": 2.0},
            "offset": {"mean": -1.0, "std": 0.5},
        },
        "feature_names": ["intercept", "signal", "offset", "kind=alpha", "kind=beta"],
        "hidden_dim": 2,
        "activation": "tanh",
        "input_hidden_weights": [
            [0.1, 0.2, -0.3, 0.4, -0.5],
            [-0.2, 0.3, 0.1, -0.4, 0.5],
        ],
        "hidden_bias": [0.01, -0.02],
        "output_weights": [0.6, -0.7],
        "output_bias": 0.03 + seed * 0.001,
        "score_blend_default": 1.0,
        "training": {
            "cache_jsonl": "/private/synthetic-cache-%d.jsonl" % seed,
            "model_kind": "mlp",
            "seed": seed,
        },
    }


def synthetic_calibration(seed):
    return {
        "schema_version": "bge_af_score_calibration_table_v1",
        "cache_jsonl": "/private/synthetic-cache-%d.jsonl" % seed,
        "model": "/private/synthetic-model-%d.json" % seed,
        "mode": "model",
        "grouping": {
            "class_key": "class_name",
            "range_edges": [0.0, 20.0, 40.0],
            "source_key": "source_signature",
            "min_bin_rows": 2,
        },
        "global": {"temperature": 1.25, "power": 0.75, "cap": 0.95},
        "bins": [
            {
                "key": "class=car|range=0_20|source=a+b",
                "class_name": "car",
                "range_bin": "0_20",
                "source_signature": "a+b",
                "temperature": 0.75,
                "power": 1.25,
                "cap": 0.9,
            },
            {
                "key": "class=truck|range=40_inf|source=c",
                "class_name": "truck",
                "range_bin": "40_inf",
                "source_signature": "c",
                "temperature": 1.5,
                "power": 1.0,
                "cap": 0.98,
            },
        ],
    }


def refresh_hashes(root):
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "SHA256SUMS":
            continue
        if any(part in {"__pycache__", ".pytest_cache", ".mypy_cache"} for part in path.parts):
            continue
        relative = path.relative_to(root).as_posix()
        rows.append("%s  %s" % (hashlib.sha256(path.read_bytes()).hexdigest(), relative))
    (root / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")


def private_readme_fixture(root):
    """Return the audited private-source README for copied-candidate tests.

    The assembler test suite is also run against an already assembled public
    candidate.  Reversing only the assembler's documented README substitutions
    keeps that copied tree a valid private-source fixture without depending on
    an external absolute path or changing production assembly behavior.
    """
    text = (root / "README.md").read_text(encoding="utf-8")
    if "Twenty small learned artifacts\nare deliberately absent" in text:
        return text.encode("utf-8")
    replacements = (
        (
            "The candidate is **not ready for public release**. Twenty public-normalized learned\n"
            "artifacts and their receipts are present, but no project license has been selected, and the\n",
            "The candidate is **not ready for public release**. Twenty small learned artifacts\n"
            "are deliberately absent, the project license has not been selected, and the\n",
        ),
        (
            "- separate private-source and public-normalized identities for twenty learned\n"
            "  artifacts across five random seeds, with canonical receipts;\n",
            "- exact expected identities for the twenty missing learned artifacts across\n"
            "  five random seeds;\n",
        ),
        (
            "1. **Structure audit (works now).** Checks file closure, hashes, import closure,\n"
            "   relative paths, anonymous-release rules, Python syntax, and all twenty\n"
            "   normalized artifact/receipt registrations.\n",
            "1. **Structure audit (works now).** Checks file closure, hashes, import closure,\n"
            "   relative paths, anonymous-release rules, and Python syntax while explicitly\n"
            "   acknowledging that the learned artifacts are absent.\n",
        ),
        (
            "3. **Core scientific replay (external inputs required).** The learned JSONs are\n"
            "   present; replay additionally requires nuScenes data and four expert result JSONs.\n",
            "3. **Core scientific replay (blocked in this candidate).** Requires the exact\n"
            "   learned JSON artifacts, nuScenes data, and four expert result JSONs.\n",
        ),
        (
            "python3 -B verify_bundle.py\n",
            "python3 -B verify_bundle.py --allow-missing-models\n",
        ),
        (
            "  tests.test_normalize_learned_artifact \\\n"
            "  tests.test_assemble_normalized_release\n",
            "  tests.test_normalize_learned_artifact\n",
        ),
        (
            "The verifier must return zero with no missing-artifact warnings. Every\n"
            "public-normalized artifact identity, receipt, and parameter/fixture-equivalence\n"
            "proof is registered and present. ",
            "The first command should return zero and report twenty acknowledged missing\n"
            "artifacts. The default command is intentionally stricter:\n\n"
            "```bash\n"
            "python3 -B verify_bundle.py\n"
            "```\n\n"
            "It must return nonzero until every public-normalized artifact identity and its\n"
            "parameter/fixture-equivalence proof have been registered and the corresponding\n"
            "files are present. ",
        ),
    )
    for public, private in replacements:
        if text.count(public) != 1:
            raise AssertionError("assembled README cannot be reversed into the private fixture")
        text = text.replace(public, private)
    return text.encode("utf-8")


class NormalizedReleaseAssemblerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = TemporaryDirectory(prefix="rgate-assembler-baseline-")
        base = Path(cls.fixture.name)
        cls.baseline_candidate = base / "candidate"
        cls.baseline_stages = base / "stages"
        private = base / "private"
        shutil.copytree(
            ROOT,
            cls.baseline_candidate,
            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", ".mypy_cache"),
        )
        (cls.baseline_candidate / "README.md").write_bytes(private_readme_fixture(ROOT))
        # The public candidate contains real normalized models; this fixture
        # builds a clean private-source baseline before injecting synthetic
        # registrations and stages.
        for path in cls.baseline_candidate.glob("models/seed_*/*.json"):
            if path.is_file() and not path.is_symlink():
                path.unlink()
        receipts = cls.baseline_candidate / "normalization_receipts"
        if receipts.is_dir():
            shutil.rmtree(receipts)
        cls.baseline_stages.mkdir()
        private.mkdir()

        manifest_path = cls.baseline_candidate / "ARTIFACT_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        registrations = []
        private_paths = {}
        for index, (artifact_id, relative) in enumerate(sorted(assembler.EXPECTED_LAYOUT.items())):
            value = (
                synthetic_calibration(index)
                if relative.endswith("_calibration.json")
                else synthetic_model(index)
            )
            payload = json.dumps(value, indent=2, sort_keys=True).encode("utf-8")
            source = private / (artifact_id + ".json")
            source.write_bytes(payload)
            private_paths[artifact_id] = source
            registrations.append(
                {
                    "id": artifact_id,
                    "relative_path": relative,
                    "required": True,
                    "availability": "missing_pending_verified_copy",
                    "size_bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        manifest["artifacts"] = registrations
        manifest["public_artifacts"] = []
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        refresh_hashes(cls.baseline_candidate)

        for artifact_id in sorted(private_paths):
            normalizer.normalize_artifact(
                manifest_path=manifest_path,
                artifact_id=artifact_id,
                source_path=private_paths[artifact_id],
                staging_dir=cls.baseline_stages / artifact_id,
                bundle_root=cls.baseline_candidate,
            )

    @classmethod
    def tearDownClass(cls):
        cls.fixture.cleanup()

    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="rgate-assembler-test-")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.source = self.base / "source"
        self.stages = self.base / "stages"
        self.output = self.base / "output"
        shutil.copytree(self.baseline_candidate, self.source)
        shutil.copytree(self.baseline_stages, self.stages)

    def receipt_path(self, artifact_id="seed_00_radar_model"):
        return self.stages / artifact_id / "normalization_receipts" / (
            artifact_id + ".normalization_receipt.json"
        )

    def public_path(self, artifact_id="seed_00_radar_model"):
        relative = assembler.EXPECTED_LAYOUT[artifact_id]
        return self.stages / artifact_id / relative

    def mutate_receipt(self, mutate, artifact_id="seed_00_radar_model"):
        path = self.receipt_path(artifact_id)
        value = json.loads(path.read_text(encoding="utf-8"))
        mutate(value)
        path.write_bytes(canonical(value))

    def assemble(self):
        return assembler.assemble_candidate(
            source_candidate=self.source,
            stages_root=self.stages,
            output_candidate=self.output,
        )

    def test_exact_twenty_stage_set_builds_fresh_strictly_verifiable_clone(self):
        source_before = assembler._snapshot_source_candidate(self.source)
        stage_before = {
            path.relative_to(self.stages).as_posix(): path.read_bytes()
            for path in self.stages.rglob("*")
            if path.is_file()
        }

        result = self.assemble()

        self.assertEqual(result["artifact_count"], 20)
        self.assertEqual(result["receipt_count"], 20)
        self.assertEqual(assembler._snapshot_source_candidate(self.source), source_before)
        self.assertEqual(
            {
                path.relative_to(self.stages).as_posix(): path.read_bytes()
                for path in self.stages.rglob("*")
                if path.is_file()
            },
            stage_before,
        )
        manifest = json.loads((self.output / "ARTIFACT_MANIFEST.json").read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["public_artifacts"]), 20)
        self.assertTrue(all(item["availability"] == "present_verified" for item in manifest["public_artifacts"]))
        self.assertTrue(all(item["availability"] == "missing_pending_verified_copy" for item in manifest["artifacts"]))
        self.assertEqual(len(list((self.output / "normalization_receipts").glob("*.json"))), 20)
        verifier = subprocess.run(
            [sys.executable, "-B", "verify_bundle.py"],
            cwd=self.output,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(verifier.returncode, 0, verifier.stdout + verifier.stderr)
        self.assertIn("NOT_READY_FOR_PUBLIC_RELEASE", (self.output / "RELEASE_STATUS.md").read_text(encoding="utf-8"))
        self.assertIn("pending_human_selection", json.dumps(manifest))
        self.assertIn("private_local_candidate_only", json.dumps(manifest))

    def test_missing_stage_is_rejected_without_output(self):
        shutil.rmtree(self.stages / "seed_04_radar_model")
        with self.assertRaisesRegex(assembler.AssemblyError, "exactly twenty"):
            self.assemble()
        self.assertFalse(self.output.exists())

    def test_extra_stage_is_rejected_without_output(self):
        (self.stages / "unexpected").mkdir()
        with self.assertRaisesRegex(assembler.AssemblyError, "exactly twenty"):
            self.assemble()
        self.assertFalse(self.output.exists())

    def test_receipt_artifact_id_swapped_between_stages_is_rejected(self):
        self.mutate_receipt(
            lambda value: value.__setitem__("artifact_id", "seed_01_radar_model")
        )
        with self.assertRaisesRegex(assembler.AssemblyError, "artifact id"):
            self.assemble()
        self.assertFalse(self.output.exists())

    def test_receipt_source_identity_mismatch_is_rejected(self):
        self.mutate_receipt(
            lambda value: value["source_artifact_identity"].__setitem__("sha256", "0" * 64)
        )
        with self.assertRaisesRegex(assembler.AssemblyError, "source identity"):
            self.assemble()
        self.assertFalse(self.output.exists())

    def test_receipt_public_identity_mismatch_is_rejected(self):
        self.mutate_receipt(
            lambda value: value["public_artifact_identity"].__setitem__("sha256", "1" * 64)
        )
        with self.assertRaisesRegex(assembler.AssemblyError, "public identity"):
            self.assemble()
        self.assertFalse(self.output.exists())

    def test_changed_pointer_reordering_is_rejected(self):
        artifact_id = "seed_00_radar_calibration"
        self.mutate_receipt(
            lambda value: value["normalization"].__setitem__(
                "changed_json_pointers", ["/model", "/cache_jsonl"]
            ),
            artifact_id,
        )
        with self.assertRaisesRegex(assembler.AssemblyError, "changed-pointer"):
            self.assemble()
        self.assertFalse(self.output.exists())

    def test_parameter_hash_inequality_is_rejected(self):
        self.mutate_receipt(
            lambda value: value["normalization"].__setitem__(
                "source_inference_parameter_sha256", "2" * 64
            )
        )
        with self.assertRaisesRegex(assembler.AssemblyError, "parameters differ"):
            self.assemble()
        self.assertFalse(self.output.exists())

    def test_fixture_output_hash_inequality_is_rejected(self):
        self.mutate_receipt(
            lambda value: value["fixture_equivalence"].__setitem__(
                "public_output_sha256", "3" * 64
            )
        )
        with self.assertRaisesRegex(assembler.AssemblyError, "fixture outputs differ"):
            self.assemble()
        self.assertFalse(self.output.exists())

    def test_receipt_path_escape_is_rejected(self):
        self.mutate_receipt(
            lambda value: value.__setitem__("intended_public_relative_path", "../escape.json")
        )
        with self.assertRaises(assembler.AssemblyError):
            self.assemble()
        self.assertFalse(self.output.exists())

    def test_sensitive_absolute_path_in_public_json_is_rejected(self):
        path = self.public_path()
        value = json.loads(path.read_text(encoding="utf-8"))
        value["unexpected_provenance"] = "/private/machine/run.json"
        path.write_bytes(canonical(value))
        with self.assertRaisesRegex(assembler.AssemblyError, "sensitive"):
            self.assemble()
        self.assertFalse(self.output.exists())

    def test_stage_symlink_is_rejected(self):
        path = self.receipt_path()
        copy_path = path.with_suffix(".copy")
        copy_path.write_bytes(path.read_bytes())
        path.unlink()
        try:
            os.symlink(copy_path.name, path)
        except OSError as exc:
            self.skipTest("symlink creation unavailable: %s" % exc)
        with self.assertRaisesRegex(assembler.AssemblyError, "symbolic link"):
            self.assemble()
        self.assertFalse(self.output.exists())

    def test_existing_output_is_never_modified(self):
        self.output.mkdir()
        marker = self.output / "owner.txt"
        marker.write_text("unchanged", encoding="utf-8")
        with self.assertRaisesRegex(assembler.AssemblyError, "must not already exist"):
            self.assemble()
        self.assertEqual(marker.read_text(encoding="utf-8"), "unchanged")

    def test_destination_race_is_never_overwritten(self):
        real_rename = assembler._rename_noreplace_at

        def destination_appears(parent_descriptor, source_name, destination_name):
            os.mkdir(destination_name, dir_fd=parent_descriptor)
            descriptor = os.open(
                destination_name + "/owner.txt",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=parent_descriptor,
            )
            os.write(descriptor, b"unchanged")
            os.close(descriptor)
            return real_rename(parent_descriptor, source_name, destination_name)

        with mock.patch.object(assembler, "_rename_noreplace_at", side_effect=destination_appears):
            with self.assertRaisesRegex(assembler.AssemblyError, "appeared"):
                self.assemble()
        self.assertEqual((self.output / "owner.txt").read_bytes(), b"unchanged")
        self.assertFalse((self.output / "ARTIFACT_MANIFEST.json").exists())

    def test_unavailable_atomic_no_replace_refuses_publication(self):
        with mock.patch.object(
            assembler,
            "_rename_noreplace_at",
            side_effect=assembler.AssemblyError("atomic no-replace publication is unavailable"),
        ):
            with self.assertRaisesRegex(assembler.AssemblyError, "unavailable"):
                self.assemble()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
