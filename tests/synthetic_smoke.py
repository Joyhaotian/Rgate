#!/usr/bin/env python3
"""Pure-standard-library smoke tests; no real data or learned artifact needed."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from apply_bge_af_arbiter import (  # noqa: E402
    calibrate_score,
    range_bin_for_value,
    sigmoid,
)
from check_nuscenes_result_json_scope_stream import iter_result_arrays  # noqa: E402
from fuse_nuscenes_expert_results import Box, fuse_cluster  # noqa: E402
from rgate_deterministic_result_archive import (  # noqa: E402
    DeterministicGzipResultWriter,
    GzipNuScenesResultReader,
    verify_archive,
)


def synthetic_box(source: str, x: float, score: float) -> Box:
    return Box(
        sample_token="synthetic-token",
        detection_name="car",
        score=score,
        translation=[x, 0.0, 0.0],
        size=[4.0, 1.8, 1.5],
        rotation=[1.0, 0.0, 0.0, 0.0],
        velocity=[0.0, 0.0],
        attribute_name="vehicle.parked",
        source=source,
        group=source,
        mode="full",
        score_weight=1.0,
        geometry_weight=1.0,
    )


class SyntheticCoreSmoke(unittest.TestCase):
    def test_score_and_range_helpers(self) -> None:
        self.assertAlmostEqual(sigmoid(0.0), 0.5)
        calibrated = calibrate_score(
            score=0.8, temperature=1.0, power=1.0, cap=0.75
        )
        self.assertAlmostEqual(calibrated, 0.75)
        self.assertEqual(range_bin_for_value(23.0, [0.0, 20.0, 40.0]), "20_40")

    def test_two_expert_fusion(self) -> None:
        fused = fuse_cluster(
            [synthetic_box("expert-a", 0.0, 0.8), synthetic_box("expert-b", 0.2, 0.7)],
            gamma=0.1,
            tau=0.5,
            score_power=1.0,
            score_cap=1.0,
            score_mode="weighted_max",
            support_mode="source",
        )
        self.assertEqual(fused["sample_token"], "synthetic-token")
        self.assertEqual(fused["detection_name"], "car")
        self.assertTrue(0.0 < fused["translation"][0] < 0.2)
        self.assertTrue(math.isfinite(fused["detection_score"]))

    def test_stream_parser(self) -> None:
        payload = {
            "meta": {"use_camera": True},
            "results": {
                "synthetic-token": [
                    {
                        "detection_name": "car",
                        "detection_score": 0.5,
                        "translation": [0.0, 0.0, 0.0],
                    }
                ]
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            rows = list(iter_result_arrays(path))
        self.assertEqual(rows[0][0], "synthetic-token")
        self.assertEqual(len(rows[0][1]), 1)

    def test_deterministic_archive_round_trip(self) -> None:
        row = {
            "sample_token": "synthetic-token",
            "translation": [0.0, 0.0, 0.0],
            "size": [4.0, 1.8, 1.5],
            "rotation": [1.0, 0.0, 0.0, 0.0],
            "velocity": [0.0, 0.0],
            "detection_name": "car",
            "detection_score": 0.5,
            "attribute_name": "vehicle.parked",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json.gz"
            writer = DeterministicGzipResultWriter(path, {"use_camera": True})
            writer.write_sample("synthetic-token", [row])
            record = writer.close()
            checked = verify_archive(
                path,
                expected_archive_size_bytes=record["archive_size_bytes"],
                expected_archive_sha256=record["archive_sha256"],
                expected_raw_size_bytes=record["raw_size_bytes"],
                expected_raw_sha256=record["raw_sha256"],
            )
            reader = GzipNuScenesResultReader(path, read_chunk_bytes=32)
            token, restored = reader.read_sample()
            self.assertEqual(token, "synthetic-token")
            self.assertEqual(restored, [row])
            reader.finish(
                expected_raw_size_bytes=record["raw_size_bytes"],
                expected_raw_sha256=record["raw_sha256"],
                expected_sample_count=1,
                expected_box_count=1,
            )
        self.assertTrue(checked["decompression_identity_closed"])


if __name__ == "__main__":
    unittest.main()
