#!/usr/bin/env python3
"""Streaming-ish scope check for very large nuScenes result JSON files.

``check_nuscenes_result_json_scope.py`` uses ``json.load`` and is convenient for
small files, but current full-val teacher JSONs can be around 1GB.  This script
memory-maps the file, decodes one ``results[sample_token]`` array at a time, and
keeps only aggregate counters in memory.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import mmap
from pathlib import Path
from statistics import mean
from typing import Any, Iterator


def skip_ws(mm: mmap.mmap, idx: int) -> int:
    n = len(mm)
    while idx < n and mm[idx] in b" \t\r\n":
        idx += 1
    return idx


def find_value_end(mm: mmap.mmap, start: int) -> int:
    """Return the exclusive end index for a JSON object/array value."""

    stack: list[int] = []
    idx = start
    n = len(mm)
    specials = (0x22, 0x5B, 0x5D, 0x7B, 0x7D)  # " [ ] { }
    while idx < n:
        positions = [mm.find(bytes([char]), idx) for char in specials]
        positions = [pos for pos in positions if pos >= 0]
        if not positions:
            break
        idx = min(positions)
        char = mm[idx]
        if char == 0x22:
            idx = find_string_end(mm, idx)
            continue
        if char in (0x5B, 0x7B):  # [ {
            stack.append(char)
        elif char == 0x5D:  # ]
            if not stack or stack[-1] != 0x5B:
                raise ValueError(f"mismatched ] at byte {idx}")
            stack.pop()
            if not stack:
                return idx + 1
        elif char == 0x7D:  # }
            if not stack or stack[-1] != 0x7B:
                raise ValueError(f"mismatched }} at byte {idx}")
            stack.pop()
            if not stack:
                return idx + 1
        idx += 1
    raise ValueError(f"unterminated JSON value from byte {start}")


def parse_json_string(mm: mmap.mmap, start: int) -> tuple[str, int]:
    if mm[start] != 0x22:
        raise ValueError(f"expected string at byte {start}")
    end = find_string_end(mm, start)
    return json.loads(mm[start:end].decode("utf-8")), end


def find_string_end(mm: mmap.mmap, start: int) -> int:
    idx = start + 1
    n = len(mm)
    while idx < n:
        idx = mm.find(b'"', idx)
        if idx < 0:
            break
        backslashes = 0
        probe = idx - 1
        while probe > start and mm[probe] == 0x5C:
            backslashes += 1
            probe -= 1
        if backslashes % 2 == 0:
            return idx + 1
        idx += 1
    raise ValueError(f"unterminated string from byte {start}")


def iter_result_arrays(path: Path) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    with path.open("rb") as handle:
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            key_idx = mm.find(b'"results"')
            if key_idx < 0:
                raise ValueError(f"{path} has no top-level-looking results key")
            colon = mm.find(b":", key_idx)
            if colon < 0:
                raise ValueError(f"{path}: malformed results key")
            idx = skip_ws(mm, colon + 1)
            if mm[idx] != 0x7B:
                raise ValueError(f"{path}: results value is not an object")
            idx += 1
            while True:
                idx = skip_ws(mm, idx)
                if mm[idx] == 0x7D:
                    return
                token, idx = parse_json_string(mm, idx)
                idx = skip_ws(mm, idx)
                if mm[idx] != 0x3A:
                    raise ValueError(f"{path}: expected ':' after sample token {token!r}")
                idx = skip_ws(mm, idx + 1)
                if mm[idx] != 0x5B:
                    raise ValueError(f"{path}: results[{token!r}] is not an array")
                end = find_value_end(mm, idx)
                rows = json.loads(mm[idx:end])
                if not isinstance(rows, list):
                    raise ValueError(f"{path}: results[{token!r}] decoded as non-list")
                yield token, rows
                idx = skip_ws(mm, end)
                if mm[idx] == 0x2C:
                    idx += 1
                    continue
                if mm[idx] == 0x7D:
                    return
                raise ValueError(f"{path}: expected ',' or '}}' after sample {token!r}")


def score_stats(scores: list[float]) -> dict[str, float | int | None]:
    if not scores:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(scores),
        "min": min(scores),
        "max": max(scores),
        "mean": mean(scores),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-json", required=True, type=Path)
    parser.add_argument("--label", default="result")
    parser.add_argument("--expected-samples", type=int)
    parser.add_argument("--min-samples", type=int)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    if not args.result_json.is_file():
        raise SystemExit(f"missing result json: {args.result_json}")

    sample_count = 0
    total_boxes = 0
    nonempty_samples = 0
    min_boxes_per_sample: int | None = None
    max_boxes_per_sample = 0
    malformed_rows = 0
    class_counts: Counter[str] = Counter()
    scores: list[float] = []

    for _, rows in iter_result_arrays(args.result_json):
        sample_count += 1
        row_count = len(rows)
        total_boxes += row_count
        nonempty_samples += int(row_count > 0)
        max_boxes_per_sample = max(max_boxes_per_sample, row_count)
        min_boxes_per_sample = row_count if min_boxes_per_sample is None else min(min_boxes_per_sample, row_count)
        for row in rows:
            if not isinstance(row, dict):
                malformed_rows += 1
                continue
            class_counts[str(row.get("detection_name", ""))] += 1
            try:
                scores.append(float(row.get("detection_score", 0.0)))
            except (TypeError, ValueError):
                malformed_rows += 1

    summary = {
        "schema_version": "nuscenes_result_scope_stream_v1",
        "label": args.label,
        "result_json": str(args.result_json),
        "sample_count": sample_count,
        "expected_samples": args.expected_samples,
        "min_samples": args.min_samples,
        "is_expected_sample_count": None
        if args.expected_samples is None
        else sample_count == args.expected_samples,
        "meets_min_samples": None if args.min_samples is None else sample_count >= args.min_samples,
        "total_boxes": total_boxes,
        "nonempty_samples": nonempty_samples,
        "min_boxes_per_sample": min_boxes_per_sample or 0,
        "max_boxes_per_sample": max_boxes_per_sample,
        "class_counts": dict(sorted(class_counts.items())),
        "score_stats": score_stats(scores),
        "malformed_rows": malformed_rows,
    }
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.expected_samples is not None and sample_count != args.expected_samples:
        raise SystemExit(
            f"{args.label}: sample_count={sample_count}, expected={args.expected_samples}"
        )
    if args.min_samples is not None and sample_count < args.min_samples:
        raise SystemExit(f"{args.label}: sample_count={sample_count}, min={args.min_samples}")
    if malformed_rows:
        raise SystemExit(f"{args.label}: malformed_rows={malformed_rows}")


if __name__ == "__main__":
    main()
