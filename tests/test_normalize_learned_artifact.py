from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import normalize_learned_artifact as normalizer  # noqa: E402


def canonical(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def synthetic_model():
    return {
        "schema_version": "bge_af_mlp_arbiter_v1",
        "numeric_features": ["signal", "offset"],
        "categorical_levels": {"kind": ["alpha", "beta"]},
        "stats": {
            "signal": {"mean": 1.0, "std": 2.0},
            "offset": {"mean": -1.0, "std": 0.5},
        },
        "feature_names": [
            "intercept",
            "signal",
            "offset",
            "kind=alpha",
            "kind=beta",
        ],
        "hidden_dim": 2,
        "activation": "tanh",
        "input_hidden_weights": [
            [0.1, 0.2, -0.3, 0.4, -0.5],
            [-0.2, 0.3, 0.1, -0.4, 0.5],
        ],
        "hidden_bias": [0.01, -0.02],
        "output_weights": [0.6, -0.7],
        "output_bias": 0.03,
        "score_blend_default": 1.0,
        "training": {
            "cache_jsonl": "/private/synthetic-cache.jsonl",
            "model_kind": "mlp",
            "seed": 7,
        },
    }


def synthetic_calibration():
    return {
        "schema_version": "bge_af_score_calibration_table_v1",
        "cache_jsonl": "/private/synthetic-cache.jsonl",
        "model": "/private/synthetic-model.json",
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


class LearnedArtifactNormalizerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="rgate-normalizer-test-")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.bundle = self.base / "candidate"
        self.private = self.base / "private"
        self.stages = self.base / "stages"
        self.bundle.mkdir()
        self.private.mkdir()
        self.stages.mkdir()

    def write_source(self, name, value, raw=None):
        path = self.private / name
        payload = raw if raw is not None else json.dumps(value, indent=2, sort_keys=True).encode("utf-8")
        path.write_bytes(payload)
        return path, payload

    def entry(self, artifact_id, relative_path, payload):
        return {
            "id": artifact_id,
            "relative_path": relative_path,
            "required": True,
            "availability": "missing_pending_verified_copy",
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    def write_manifest(self, entries, public=None):
        manifest = {
            "schema_version": "rgate_release_artifact_manifest_v2",
            "normalization_contract": {
                "schema_version": "rgate_release_metadata_normalization_v1",
                "allowed_changed_json_pointers": copy.deepcopy(normalizer.EXPECTED_POINTERS),
                "required_public_proofs": list(normalizer.EXPECTED_PUBLIC_PROOFS),
            },
            "artifacts": entries,
            "public_artifacts": [] if public is None else public,
        }
        path = self.bundle / "ARTIFACT_MANIFEST.json"
        path.write_bytes(canonical(manifest))
        return path

    def model_registration(self, payload):
        return self.entry(
            "seed_00_radar_model",
            "models/seed_00/radar_model.json",
            payload,
        )

    def normalize(self, manifest, source, stage_name="stage"):
        return normalizer.normalize_artifact(
            manifest_path=manifest,
            artifact_id="seed_00_radar_model",
            source_path=source,
            staging_dir=self.stages / stage_name,
            bundle_root=self.bundle,
        )

    def test_model_success_is_atomic_canonical_and_does_not_edit_registry(self):
        source_value = synthetic_model()
        source, payload = self.write_source("model.json", source_value)
        manifest = self.write_manifest([self.model_registration(payload)])
        manifest_before = manifest.read_bytes()
        source_before = source.read_bytes()

        result = self.normalize(manifest, source)

        stage = self.stages / "stage"
        public_path = stage / "models/seed_00/radar_model.json"
        receipt_path = stage / "normalization_receipts/seed_00_radar_model.normalization_receipt.json"
        self.assertTrue(public_path.is_file())
        self.assertTrue(receipt_path.is_file())
        public = json.loads(public_path.read_text(encoding="utf-8"))
        receipt_payload = receipt_path.read_bytes()
        receipt = json.loads(receipt_payload.decode("utf-8"))
        self.assertEqual(public_path.read_bytes(), canonical(public))
        self.assertEqual(receipt_payload, canonical(receipt))
        self.assertEqual(
            public["training"]["cache_jsonl"],
            "artifacts/cache/fullval_cache.jsonl",
        )
        self.assertEqual(receipt["normalization"]["changed_json_pointers"], ["/training/cache_jsonl"])
        self.assertEqual(
            receipt["normalization"]["source_inference_parameter_sha256"],
            receipt["normalization"]["public_inference_parameter_sha256"],
        )
        self.assertEqual(
            receipt["fixture_equivalence"]["source_output_sha256"],
            receipt["fixture_equivalence"]["public_output_sha256"],
        )
        self.assertIn("model_score_exact", receipt["fixture_equivalence"]["method"])
        self.assertEqual(
            hashlib.sha256(receipt_payload).hexdigest(),
            result["normalization_receipt_identity"]["sha256"],
        )
        self.assertNotIn("/private/", receipt_payload.decode("utf-8"))
        self.assertEqual(manifest.read_bytes(), manifest_before)
        self.assertEqual(source.read_bytes(), source_before)
        self.assertFalse((self.bundle / "models/seed_00/radar_model.json").exists())

    def test_calibration_exercises_bins_and_global_path(self):
        calibration = synthetic_calibration()
        source, payload = self.write_source("calibration.json", calibration)
        model_payload = b"synthetic registry identity only"
        entries = [
            self.entry(
                "seed_00_radar_model",
                "models/seed_00/radar_model.json",
                model_payload,
            ),
            self.entry(
                "seed_00_radar_calibration",
                "models/seed_00/radar_calibration.json",
                payload,
            ),
        ]
        manifest = self.write_manifest(entries)
        result = normalizer.normalize_artifact(
            manifest_path=manifest,
            artifact_id="seed_00_radar_calibration",
            source_path=source,
            staging_dir=self.stages / "calibration-stage",
            bundle_root=self.bundle,
        )
        public_path = self.stages / "calibration-stage/models/seed_00/radar_calibration.json"
        receipt_path = self.stages / (
            "calibration-stage/normalization_receipts/"
            "seed_00_radar_calibration.normalization_receipt.json"
        )
        public = json.loads(public_path.read_text(encoding="utf-8"))
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        self.assertEqual(public["cache_jsonl"], "artifacts/cache/fullval_cache.jsonl")
        self.assertEqual(public["model"], "models/seed_00/radar_model.json")
        self.assertEqual(
            receipt["normalization"]["changed_json_pointers"],
            ["/cache_jsonl", "/model"],
        )
        self.assertEqual(receipt["fixture_equivalence"]["probe_count"], 15)
        self.assertIn("calibration_path_exact", receipt["fixture_equivalence"]["method"])
        self.assertEqual(result["fixture_sha256"], receipt["fixture_equivalence"]["fixture_sha256"])

    def test_wrong_private_sha_leaves_no_stage(self):
        source, payload = self.write_source("model.json", synthetic_model())
        manifest = self.write_manifest([self.model_registration(payload)])
        altered = bytearray(source.read_bytes())
        altered[-2] = ord(" ") if altered[-2] != ord(" ") else ord("\n")
        source.write_bytes(bytes(altered))
        with self.assertRaisesRegex(normalizer.NormalizationError, "SHA-256"):
            self.normalize(manifest, source)
        self.assertFalse((self.stages / "stage").exists())

    def test_missing_required_pointer_leaves_no_stage(self):
        model = synthetic_model()
        del model["training"]["cache_jsonl"]
        source, payload = self.write_source("model.json", model)
        manifest = self.write_manifest([self.model_registration(payload)])
        with self.assertRaisesRegex(normalizer.NormalizationError, "pointer is absent"):
            self.normalize(manifest, source)
        self.assertFalse((self.stages / "stage").exists())

    def test_other_absolute_metadata_is_not_exported(self):
        model = synthetic_model()
        model["run_note"] = "/private/another-machine/run"
        source, payload = self.write_source("model.json", model)
        manifest = self.write_manifest([self.model_registration(payload)])
        with self.assertRaisesRegex(normalizer.NormalizationError, "absolute path"):
            self.normalize(manifest, source)
        self.assertFalse((self.stages / "stage").exists())

    def test_embedded_double_slash_and_unc_paths_are_not_exported(self):
        private_strings = (
            "run used [/scratch/private-user/run/model.json]",
            "//private-server/share/run/model.json",
            "\\\\private-server\\share\\run\\model.json",
        )
        for index, private_string in enumerate(private_strings):
            with self.subTest(private_string=private_string):
                model = synthetic_model()
                model["run_note"] = private_string
                source, payload = self.write_source("private-path-%d.json" % index, model)
                manifest = self.write_manifest([self.model_registration(payload)])
                stage_name = "private-path-stage-%d" % index
                with self.assertRaisesRegex(normalizer.NormalizationError, "absolute path"):
                    self.normalize(manifest, source, stage_name=stage_name)
                self.assertFalse((self.stages / stage_name).exists())

    def test_invalid_model_dimensions_leave_no_stage(self):
        model = synthetic_model()
        model["input_hidden_weights"][0].pop()
        source, payload = self.write_source("model.json", model)
        manifest = self.write_manifest([self.model_registration(payload)])
        with self.assertRaisesRegex(normalizer.NormalizationError, "input dimension"):
            self.normalize(manifest, source)
        self.assertFalse((self.stages / "stage").exists())

    def test_duplicate_json_key_is_rejected(self):
        raw = (
            b'{"schema_version":"bge_af_mlp_arbiter_v1",'
            b'"training":{"cache_jsonl":"/private/a"},'
            b'"training":{"cache_jsonl":"/private/b"}}'
        )
        source, payload = self.write_source("duplicate.json", None, raw=raw)
        manifest = self.write_manifest([self.model_registration(payload)])
        with self.assertRaisesRegex(normalizer.NormalizationError, "duplicate JSON key"):
            self.normalize(manifest, source)
        self.assertFalse((self.stages / "stage").exists())

    def test_existing_stage_is_not_modified(self):
        source, payload = self.write_source("model.json", synthetic_model())
        manifest = self.write_manifest([self.model_registration(payload)])
        stage = self.stages / "stage"
        stage.mkdir()
        marker = stage / "keep.txt"
        marker.write_text("unchanged", encoding="utf-8")
        with self.assertRaisesRegex(normalizer.NormalizationError, "must not already exist"):
            self.normalize(manifest, source)
        self.assertEqual(marker.read_text(encoding="utf-8"), "unchanged")

    def test_stage_inside_candidate_is_rejected(self):
        source, payload = self.write_source("model.json", synthetic_model())
        manifest = self.write_manifest([self.model_registration(payload)])
        (self.bundle / "outputs").mkdir()
        with self.assertRaisesRegex(normalizer.NormalizationError, "outside the candidate"):
            normalizer.normalize_artifact(
                manifest_path=manifest,
                artifact_id="seed_00_radar_model",
                source_path=source,
                staging_dir=self.bundle / "outputs/stage",
                bundle_root=self.bundle,
            )
        self.assertFalse((self.bundle / "outputs/stage").exists())

    def test_source_symlink_is_rejected(self):
        source, payload = self.write_source("model.json", synthetic_model())
        link = self.private / "model-link.json"
        try:
            os.symlink(source.name, link)
        except OSError as exc:
            self.skipTest("symlink creation unavailable: %s" % exc)
        manifest = self.write_manifest([self.model_registration(payload)])
        with self.assertRaisesRegex(normalizer.NormalizationError, "symbolic-link component"):
            self.normalize(manifest, link)
        self.assertFalse((self.stages / "stage").exists())

    def test_staging_ancestor_symlink_is_rejected(self):
        source, payload = self.write_source("model.json", synthetic_model())
        manifest = self.write_manifest([self.model_registration(payload)])
        real_parent = self.base / "real-stages"
        real_parent.mkdir()
        linked_parent = self.base / "linked-stages"
        try:
            os.symlink(real_parent.name, linked_parent, target_is_directory=True)
        except OSError as exc:
            self.skipTest("symlink creation unavailable: %s" % exc)
        with self.assertRaisesRegex(normalizer.NormalizationError, "symbolic-link component"):
            normalizer.normalize_artifact(
                manifest_path=manifest,
                artifact_id="seed_00_radar_model",
                source_path=source,
                staging_dir=linked_parent / "stage",
                bundle_root=self.bundle,
            )
        self.assertFalse((real_parent / "stage").exists())

    def test_source_ancestor_symlink_is_rejected(self):
        source, payload = self.write_source("model.json", synthetic_model())
        manifest = self.write_manifest([self.model_registration(payload)])
        linked_private = self.base / "linked-private"
        try:
            os.symlink(self.private.name, linked_private, target_is_directory=True)
        except OSError as exc:
            self.skipTest("symlink creation unavailable: %s" % exc)
        with self.assertRaisesRegex(normalizer.NormalizationError, "symbolic-link component"):
            normalizer.normalize_artifact(
                manifest_path=manifest,
                artifact_id="seed_00_radar_model",
                source_path=linked_private / source.name,
                staging_dir=self.stages / "stage",
                bundle_root=self.bundle,
            )
        self.assertFalse((self.stages / "stage").exists())

    def test_concurrent_destination_is_not_replaced(self):
        stage = self.stages / "concurrent-stage"
        real_publish = normalizer._rename_noreplace_at

        def destination_appears(parent_descriptor, source_name, destination_name):
            stage.mkdir()
            (stage / "other-owner.txt").write_text("unchanged", encoding="utf-8")
            return real_publish(parent_descriptor, source_name, destination_name)

        with mock.patch.object(normalizer, "_rename_noreplace_at", side_effect=destination_appears):
            with self.assertRaisesRegex(normalizer.NormalizationError, "appeared"):
                normalizer._atomic_stage(
                    stage,
                    "models/seed_00/radar_model.json",
                    b"{}",
                    "receipt.json",
                    b"{}",
                )
        self.assertEqual((stage / "other-owner.txt").read_text(encoding="utf-8"), "unchanged")
        self.assertFalse((stage / "models").exists())
        self.assertEqual(len(list(self.stages.glob(".rgate-normalize-*"))), 1)

    def test_swapped_transaction_directory_is_not_published_or_deleted(self):
        stage = self.stages / "swapped-stage"
        real_verify = normalizer._verify_name_matches_directory_fd
        swap = {}
        transaction_checks = 0

        def swap_before_identity_check(parent_descriptor, name, descriptor, label):
            nonlocal transaction_checks
            if label == "transaction":
                transaction_checks += 1
            if label == "transaction" and transaction_checks == 2:
                stolen_name = ".swapped-original"
                os.rename(
                    name,
                    stolen_name,
                    src_dir_fd=parent_descriptor,
                    dst_dir_fd=parent_descriptor,
                )
                os.mkdir(name, dir_fd=parent_descriptor)
                swap["replacement"] = self.stages / name
                swap["stolen"] = self.stages / stolen_name
            return real_verify(parent_descriptor, name, descriptor, label)

        with mock.patch.object(
            normalizer,
            "_verify_name_matches_directory_fd",
            side_effect=swap_before_identity_check,
        ):
            with self.assertRaisesRegex(normalizer.NormalizationError, "identity changed"):
                normalizer._atomic_stage(
                    stage,
                    "models/seed_00/radar_model.json",
                    b"{}",
                    "receipt.json",
                    b"{}",
                )
        self.assertFalse(stage.exists())
        self.assertTrue(swap["replacement"].is_dir())
        self.assertTrue((swap["stolen"] / "models/seed_00/radar_model.json").is_file())
        self.assertTrue((swap["stolen"] / "normalization_receipts/receipt.json").is_file())

    def test_staging_parent_exchange_cannot_publish_a_fake_stage(self):
        stage = self.stages / "parent-exchange-stage"
        moved_parent = self.base / "stages-moved"
        real_publish = normalizer._rename_noreplace_at

        def exchange_parent_then_publish(parent_descriptor, source_name, destination_name):
            os.rename(self.stages, moved_parent)
            self.stages.mkdir()
            return real_publish(parent_descriptor, source_name, destination_name)

        with mock.patch.object(
            normalizer,
            "_rename_noreplace_at",
            side_effect=exchange_parent_then_publish,
        ):
            with self.assertRaisesRegex(normalizer.NormalizationError, "staging parent directory identity"):
                normalizer._atomic_stage(
                    stage,
                    "models/seed_00/radar_model.json",
                    b"{}",
                    "receipt.json",
                    b"{}",
                )
        self.assertFalse(stage.exists())
        published = moved_parent / "parent-exchange-stage"
        self.assertEqual((published / "models/seed_00/radar_model.json").read_bytes(), b"{}")
        self.assertEqual((published / "normalization_receipts/receipt.json").read_bytes(), b"{}")

    def test_ancestor_exchange_after_boundary_check_cannot_redirect_into_candidate(self):
        source, payload = self.write_source("model.json", synthetic_model())
        manifest = self.write_manifest([self.model_registration(payload)])
        outer = self.base / "external-outer"
        external_stages = outer / "stages"
        external_stages.mkdir(parents=True)
        moved_outer = self.base / "external-outer-moved"
        redirect_target = self.bundle / "redirect-target"
        redirected_stages = redirect_target / "stages"
        redirected_stages.mkdir(parents=True)
        stage = external_stages / "stage"
        real_atomic = normalizer._atomic_stage
        exchanged = False

        def exchange_ancestor_then_stage(*args, **kwargs):
            nonlocal exchanged
            if not exchanged:
                os.rename(outer, moved_outer)
                os.symlink(redirect_target, outer, target_is_directory=True)
                exchanged = True
            return real_atomic(*args, **kwargs)

        with mock.patch.object(normalizer, "_atomic_stage", side_effect=exchange_ancestor_then_stage):
            with self.assertRaisesRegex(normalizer.NormalizationError, "staging parent directory identity"):
                normalizer.normalize_artifact(
                    manifest_path=manifest,
                    artifact_id="seed_00_radar_model",
                    source_path=source,
                    staging_dir=stage,
                    bundle_root=self.bundle,
                )
        self.assertFalse((redirected_stages / "stage").exists())
        self.assertFalse((moved_outer / "stages/stage").exists())

    def test_held_staging_parent_moved_inside_candidate_is_rejected(self):
        source, payload = self.write_source("model.json", synthetic_model())
        manifest = self.write_manifest([self.model_registration(payload)])
        outer = self.base / "movable-outer"
        external_stages = outer / "stages"
        external_stages.mkdir(parents=True)
        redirect_target = self.bundle / "moved-stage-root"
        redirect_target.mkdir()
        moved_outer = redirect_target / "movable-outer"
        stage = external_stages / "stage"
        real_atomic = normalizer._atomic_stage
        exchanged = False

        def move_held_parent_inside_then_stage(*args, **kwargs):
            nonlocal exchanged
            if not exchanged:
                os.rename(outer, moved_outer)
                os.symlink(moved_outer, outer, target_is_directory=True)
                exchanged = True
            return real_atomic(*args, **kwargs)

        with mock.patch.object(
            normalizer,
            "_atomic_stage",
            side_effect=move_held_parent_inside_then_stage,
        ):
            with self.assertRaisesRegex(normalizer.NormalizationError, "moved inside the candidate"):
                normalizer.normalize_artifact(
                    manifest_path=manifest,
                    artifact_id="seed_00_radar_model",
                    source_path=source,
                    staging_dir=stage,
                    bundle_root=self.bundle,
                )
        self.assertFalse((moved_outer / "stages/stage").exists())

    def test_unexpected_internal_symlink_closes_transaction(self):
        stage = self.stages / "internal-symlink-stage"
        real_verify = normalizer._verify_transaction_tree
        injected = False

        def inject_then_verify(root_descriptor, expected_payloads, prefix=""):
            nonlocal injected
            if not injected:
                os.symlink("models", "unexpected-link", dir_fd=root_descriptor)
                injected = True
            return real_verify(root_descriptor, expected_payloads, prefix)

        with mock.patch.object(
            normalizer,
            "_verify_transaction_tree",
            side_effect=inject_then_verify,
        ):
            with self.assertRaisesRegex(normalizer.NormalizationError, "non-regular object"):
                normalizer._atomic_stage(
                    stage,
                    "models/seed_00/radar_model.json",
                    b"{}",
                    "receipt.json",
                    b"{}",
                )
        self.assertFalse(stage.exists())

    def test_registered_public_entry_closes_renormalization(self):
        source, payload = self.write_source("model.json", synthetic_model())
        entry = self.model_registration(payload)
        manifest = self.write_manifest(
            [entry],
            public=[{"id": entry["id"], "relative_path": entry["relative_path"]}],
        )
        with self.assertRaisesRegex(normalizer.NormalizationError, "public registration"):
            self.normalize(manifest, source)
        self.assertFalse((self.stages / "stage").exists())


if __name__ == "__main__":
    unittest.main()
