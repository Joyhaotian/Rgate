#!/usr/bin/env python3
"""Independent, stdlib-only verifier for RQ2 stage materialization outputs.

This module intentionally does not import the materializer or any of its helper
modules.  It verifies frozen identities, the committed-directory contract,
streamed nuScenes result structure, cache/source closure, exact E1 semantics,
and independently rebuilds the raw four-expert naive WBF.  Bounded smoke runs
also independently rebuild all three learned arms, including one-sweep radar
features and ego-relative calibration.  A full run is eligible for official
evaluation only after the fixed 6019-sample / 4,732,657-row closure passes.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import pickle
import re
import struct
import tempfile
from typing import Any, BinaryIO, Iterator, List, Dict, Optional, Tuple


CONFIG_SCHEMA = "rgate_rq2_stage_execution_config_v1"
PLAN_SCHEMA = "rgate_rq2_cpu_controls_plan_v1"
SUMMARY_SCHEMA = "rgate_rq2_stage_materialization_summary_v1"
REPORT_SCHEMA = "rgate_rq2_stage_independent_check_v1"
CACHE_SCHEMA = "bge_af_arbiter_cache_row_v1"
OUTPUTS = {
    "naive_equal_wbf": "naive_equal_wbf_results_nusc.json",
    "rgate_rescore_only": "rgate_rescore_only_results_nusc.json",
    "learned_radar_uncalibrated": "learned_radar_uncalibrated_results_nusc.json",
    "learned_radar_calibrated": "learned_radar_calibrated_results_nusc.json",
    "learned_no_radar_calibrated": "learned_no_radar_calibrated_results_nusc.json",
}
CLASSES = {
    "barrier", "bicycle", "bus", "car", "construction_vehicle",
    "motorcycle", "pedestrian", "traffic_cone", "trailer", "truck",
}
RADII = {
    "barrier": 0.4, "bicycle": 0.5, "bus": 1.0, "car": 0.7,
    "construction_vehicle": 1.0, "motorcycle": 0.5, "pedestrian": 0.4,
    "traffic_cone": 0.25, "trailer": 1.0, "truck": 1.0,
}
MAX_BOXES = 500
FULL_SAMPLES = 6019
FULL_ROWS = 4732657
HASH_CHUNK = 8 * 1024 * 1024
TOKEN_RE = re.compile(rb'"token"\s*:\s*"([^"\\]+)"')
SAMPLE_TOKEN_RE = re.compile(rb'"sample_token"\s*:\s*"([^"\\]+)"')


class AuditError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(HASH_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    resolved_before = path.resolve(strict=True)
    stat = resolved_before.stat()
    signature_before = (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
    )
    digest = sha256_file(resolved_before)
    resolved_after = path.resolve(strict=True)
    stat_after = resolved_after.stat()
    signature_after = (
        stat_after.st_dev,
        stat_after.st_ino,
        stat_after.st_size,
        stat_after.st_mtime_ns,
    )
    require(
        resolved_after == resolved_before and signature_after == signature_before,
        f"file identity changed while hashing: {path}",
    )
    return {
        "path": str(resolved_before),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest,
    }


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def same_path(left: Any, right: Any) -> bool:
    return Path(str(left)).resolve() == Path(str(right)).resolve()


def require_no_symlink_chain(path: Path, label: str) -> None:
    absolute = path.absolute()
    parts = absolute.parts
    cursor = Path(parts[0])
    for part in parts[1:]:
        cursor = cursor / part
        if cursor.is_symlink():
            raise AuditError(f"symlink path component forbidden for {label}: {cursor}")
        if not cursor.exists():
            break


def mount_source(path: Path) -> str:
    target=path.resolve();candidates=[]
    escapes={"\\040":" ","\\011":"\t","\\012":"\n","\\134":"\\"}
    with Path("/proc/self/mountinfo").open("r",encoding="utf-8") as fh:
        for line in fh:
            left,sep,right=line.rstrip("\n").partition(" - ")
            if not sep:continue
            before=left.split();after=right.split()
            if len(before)<5 or len(after)<2:continue
            point=before[4];source=after[1]
            for encoded,decoded in escapes.items():point=point.replace(encoded,decoded);source=source.replace(encoded,decoded)
            if below(target,Path(point)):candidates.append((len(Path(point).parts),source))
    require(bool(candidates),f"cannot resolve mount source: {path}")
    return max(candidates)[1]


def below(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def check_identity(
    label: str,
    path: Path,
    expected: Any,
    *,
    allow_registered_symlink: bool = False,
) -> dict[str, Any]:
    # Some preregistered large inputs use a stable configured logical alias
    # whose target lives on a separate storage volume. Allow that alias only at
    # call sites that close the logical path against the frozen plan/config and
    # still hash the stable resolved inode.
    if not allow_registered_symlink:
        require_no_symlink_chain(path, label)
    require(isinstance(expected, dict), f"missing configured identity: {label}")
    require(same_path(path, expected.get("path")), f"path drift: {label}")
    actual = identity(path)
    require(actual["size_bytes"] == expected.get("size_bytes"), f"size drift: {label}")
    require(actual["sha256"] == expected.get("sha256"), f"SHA drift: {label}")
    return actual


def validate_config_against_plan(config: dict[str, Any], plan: dict[str, Any]) -> None:
    registered=plan.get("registered_inputs");inputs=config.get("inputs")
    require(isinstance(registered,dict) and isinstance(inputs,dict),"config/plan input contract missing")
    def registered_file(config_key:str,plan_key:str,path_key:str="path",size:bool=False)->None:
        actual=inputs.get(config_key);expected=registered.get(plan_key)
        require(isinstance(actual,dict) and isinstance(expected,dict),f"config/plan lacks {config_key}")
        require(same_path(actual.get("path"),expected.get(path_key)),f"unregistered config path: {config_key}")
        require(actual.get("sha256")==expected.get("sha256"),f"unregistered config SHA: {config_key}")
        if size:require(actual.get("size_bytes")==expected.get("size_bytes"),f"unregistered config size: {config_key}")
    registered_file("cache_jsonl","merged_cluster_cache",size=True)
    registered_file("sample_info_pkl","val_infos_for_radar_repair","resolved_path")
    for config_key,plan_key in (
        ("radar_model","radar_mlp"),
        ("radar_calibration","radar_calibration"),
        ("no_radar_model","no_radar_mlp"),
        ("no_radar_calibration","no_radar_calibration"),
    ):
        registered_file(config_key,plan_key)
    experts=inputs.get("naive_experts");planned=registered.get("expert_pool")
    require(isinstance(experts,list) and isinstance(planned,list) and len(experts)==len(planned)==4,"unregistered expert pool")
    for idx,(actual,expected) in enumerate(zip(experts,planned)):
        require(actual.get("name")==expected.get("name") and actual.get("fixed_ordinal")==idx,f"unregistered expert order: {idx}")
        require(same_path(actual.get("path"),expected.get("path")) and actual.get("sha256")==expected.get("sha256") and actual.get("size_bytes")==expected.get("size_bytes"),f"unregistered expert identity: {idx}")
    nusc=inputs.get("nuscenes")
    require(isinstance(nusc,dict) and same_path(nusc.get("root"),registered.get("nuscenes_root")),"unregistered nuScenes root")
    storage=plan.get("storage_and_safety");full=config.get("scopes",{}).get("full")
    require(isinstance(storage,dict) and isinstance(full,dict),"config/plan storage contract missing")
    require(same_path(full.get("output_dir"),storage.get("large_output_root")),"unregistered full output dir")
    require(int(full.get("minimum_free_bytes",-1))>=int(storage.get("minimum_D_free_bytes_before_start",-1)),"config weakens registered free floor")
    counts=config.get("registered_counts",{})
    require(counts.get("samples")==FULL_SAMPLES and counts.get("cache_rows")==FULL_ROWS,"fixed full counts missing")


def finite_tree(value: Any, where: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        require(math.isfinite(float(value)), f"non-finite number at {where}")
        return
    if isinstance(value, list):
        for idx, child in enumerate(value):
            finite_tree(child, f"{where}[{idx}]")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            finite_tree(child, f"{where}.{key}")
        return
    raise AuditError(f"non-JSON value at {where}: {type(value).__name__}")


def strict_json_loads(raw: Any, where: str) -> Any:
    def reject(value: str) -> None:
        raise AuditError(f"non-finite JSON literal {value} at {where}")
    return json.loads(raw, parse_constant=reject)


def skip_ws(mm: mmap.mmap, idx: int) -> int:
    while idx < len(mm) and mm[idx] in b" \t\r\n":
        idx += 1
    return idx


def value_end(mm: mmap.mmap, idx: int) -> int:
    idx = skip_ws(mm, idx)
    require(idx < len(mm), "truncated JSON value")
    first = mm[idx]
    if first == ord('"'):
        escaped = False
        cursor = idx + 1
        while cursor < len(mm):
            byte = mm[cursor]
            if escaped:
                escaped = False
            elif byte == ord('\\'):
                escaped = True
            elif byte == ord('"'):
                return cursor + 1
            cursor += 1
        raise AuditError("unterminated JSON string")
    if first in (ord("["), ord("{")):
        opening, closing = (first, ord("]") if first == ord("[") else ord("}"))
        depth = 0
        in_string = escaped = False
        cursor = idx
        while cursor < len(mm):
            byte = mm[cursor]
            if in_string:
                if escaped:
                    escaped = False
                elif byte == ord('\\'):
                    escaped = True
                elif byte == ord('"'):
                    in_string = False
            elif byte == ord('"'):
                in_string = True
            elif byte == opening:
                depth += 1
            elif byte == closing:
                depth -= 1
                if depth == 0:
                    return cursor + 1
            cursor += 1
        raise AuditError("unterminated JSON container")
    cursor = idx
    while cursor < len(mm) and mm[cursor] not in b",}] \t\r\n":
        cursor += 1
    return cursor


class ResultIndex:
    """Independent mmap index over a standard nuScenes result JSON."""

    def __init__(self, path: Path):
        self.path = path
        self.fh: BinaryIO = path.open("rb")
        self.mm = mmap.mmap(self.fh.fileno(), 0, access=mmap.ACCESS_READ)
        self.meta: dict[str, Any] = {}
        self.ranges: dict[str, tuple[int, int]] = {}
        self.order: list[str] = []
        self._parse()

    def close(self) -> None:
        self.mm.close()
        self.fh.close()

    def __enter__(self) -> "ResultIndex":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _parse(self) -> None:
        require(self.mm[:1] == b"{", f"result is not an object: {self.path}")
        meta_key = self.mm.find(b'"meta"')
        results_key = self.mm.find(b'"results"')
        require(meta_key >= 0 and results_key >= 0, f"missing meta/results: {self.path}")
        meta_colon = self.mm.find(b":", meta_key)
        meta_start = skip_ws(self.mm, meta_colon + 1)
        meta_end = value_end(self.mm, meta_start)
        self.meta = strict_json_loads(self.mm[meta_start:meta_end],str(self.path))
        require(isinstance(self.meta, dict), f"meta is not an object: {self.path}")

        colon = self.mm.find(b":", results_key)
        idx = skip_ws(self.mm, colon + 1)
        require(self.mm[idx] == ord("{"), f"results is not an object: {self.path}")
        idx += 1
        while True:
            idx = skip_ws(self.mm, idx)
            if self.mm[idx] == ord("}"):
                idx += 1
                break
            end_key = value_end(self.mm, idx)
            token = strict_json_loads(self.mm[idx:end_key],str(self.path))
            require(isinstance(token, str) and token, f"invalid sample token: {self.path}")
            require(token not in self.ranges, f"duplicate sample token {token}: {self.path}")
            idx = skip_ws(self.mm, end_key)
            require(self.mm[idx] == ord(":"), f"missing token colon: {self.path}")
            start = skip_ws(self.mm, idx + 1)
            require(self.mm[start] == ord("["), f"sample rows are not an array: {token}")
            end = value_end(self.mm, start)
            self.ranges[token] = (start, end)
            self.order.append(token)
            idx = skip_ws(self.mm, end)
            if self.mm[idx] == ord(","):
                idx += 1
                continue
            require(self.mm[idx] == ord("}"), f"malformed results object: {self.path}")
            idx += 1
            break
        idx = skip_ws(self.mm, idx)
        require(idx < len(self.mm) and self.mm[idx] == ord("}"), f"missing top-level close: {self.path}")
        require(not self.mm[idx + 1:].strip(), f"trailing JSON data: {self.path}")

    def rows(self, token: str) -> list[dict[str, Any]]:
        start, end = self.ranges[token]
        value = strict_json_loads(self.mm[start:end],f"{self.path}/{token}")
        require(isinstance(value, list), f"rows not list: {self.path}/{token}")
        require(all(isinstance(row, dict) for row in value), f"non-object result row: {token}")
        return value


def validate_box(row: dict[str, Any], token: str, where: str) -> None:
    required = {
        "sample_token", "translation", "size", "rotation", "velocity",
        "detection_name", "detection_score", "attribute_name",
    }
    require(required <= set(row), f"missing box fields at {where}: {sorted(required-set(row))}")
    require(row["sample_token"] == token, f"box sample_token drift at {where}")
    require(row["detection_name"] in CLASSES, f"unknown class at {where}")
    for key, width in (("translation", 3), ("size", 3), ("rotation", 4), ("velocity", 2)):
        require(isinstance(row[key], list) and len(row[key]) == width, f"bad {key} at {where}")
    score = float(row["detection_score"])
    require(math.isfinite(score) and 0.0 <= score <= 1.0, f"bad score at {where}")
    finite_tree(row, where)


def geometry(row: dict[str, Any]) -> bytes:
    return canonical({
        "translation": row["translation"], "size": row["size"],
        "rotation": row["rotation"], "velocity": row["velocity"],
        "detection_name": row["detection_name"],
        "attribute_name": row.get("attribute_name", ""),
    })


def assert_rows_equal(actual: list[dict[str, Any]], expected: list[dict[str, Any]], where: str) -> None:
    require(len(actual) == len(expected), f"row-count mismatch at {where}")
    for idx, (left, right) in enumerate(zip(actual, expected)):
        require(canonical(left) == canonical(right), f"numeric/content mismatch at {where}[{idx}]")


def assert_learned_rows_close(
    actual: list[dict[str, Any]], expected: list[dict[str, Any]], where: str
) -> tuple[float, int]:
    require(len(actual) == len(expected), f"row-count mismatch at {where}")
    max_difference=0.0;max_index=-1
    for idx, (left, right) in enumerate(zip(actual, expected)):
        require(geometry(left)==geometry(right),f"selected geometry/order mismatch at {where}[{idx}]")
        require(left.get("sample_token")==right.get("sample_token"),f"token mismatch at {where}[{idx}]")
        difference=abs(float(left["detection_score"])-float(right["detection_score"]))
        if difference>max_difference:max_difference=difference;max_index=idx
        require(difference<=1e-7,f"learned score mismatch at {where}[{idx}]: abs={difference:.9g}")
    return max_difference,max_index


@dataclass(frozen=True)
class RawCandidate:
    source: int
    row_ordinal: int
    score: float
    row: dict[str, Any]

    @property
    def key(self) -> tuple[float, int, int]:
        return (-self.score, self.source, self.row_ordinal)


@dataclass
class RawCluster:
    ordinal: int
    members: list[RawCandidate] = field(default_factory=list)
    sources: set[int] = field(default_factory=set)

    def center(self) -> tuple[float, float]:
        return (
            sum(float(item.row["translation"][0]) for item in self.members) / len(self.members),
            sum(float(item.row["translation"][1]) for item in self.members) / len(self.members),
        )


def yaw(q: list[Any]) -> float:
    w, x, y, z = (float(value) for value in q)
    return math.atan2(2 * (w*z + x*y), 1 - 2*(y*y + z*z))


def naive_sample(token: str, expert_rows: list[list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_class: dict[str, list[RawCandidate]] = defaultdict(list)
    input_counts: list[int] = []
    for source, rows in enumerate(expert_rows):
        input_counts.append(len(rows))
        for row_ordinal, row in enumerate(rows):
            validate_box(row, token, f"raw[{source}]/{token}/{row_ordinal}")
            by_class[row["detection_name"]].append(
                RawCandidate(source, row_ordinal, max(0.0, float(row["detection_score"])), row)
            )
    ordinal = 0
    fused: list[tuple[int, dict[str, Any], int]] = []
    supports: Counter[int] = Counter()
    for class_name in sorted(by_class):
        clusters: list[RawCluster] = []
        for candidate in sorted(by_class[class_name], key=lambda item: item.key):
            legal: list[tuple[float, int, RawCluster]] = []
            x, y = (float(value) for value in candidate.row["translation"][:2])
            for cluster in clusters:
                if candidate.source in cluster.sources:
                    continue
                cx, cy = cluster.center()
                distance = math.hypot(x-cx, y-cy)
                if distance <= RADII[class_name]:
                    legal.append((distance, cluster.ordinal, cluster))
            if legal:
                selected = min(legal, key=lambda item: (item[0], item[1]))[2]
            else:
                selected = RawCluster(ordinal)
                ordinal += 1
                clusters.append(selected)
            selected.members.append(candidate)
            selected.sources.add(candidate.source)
        for cluster in clusters:
            members = cluster.members
            count = len(members)
            best = min(members, key=lambda item: item.key)
            yaws = [yaw(item.row["rotation"]) for item in members]
            mean_yaw = math.atan2(sum(math.sin(v) for v in yaws), sum(math.cos(v) for v in yaws))
            def mean_vec(key: str, width: int, default: Optional[List[float]] = None) -> List[float]:
                vectors = []
                for item in members:
                    vector = list(item.row.get(key, default or []))[:width]
                    vector += [0.0] * (width-len(vector))
                    vectors.append(vector)
                return [sum(float(v[i]) for v in vectors)/count for i in range(width)]
            box = {
                "sample_token": token,
                "translation": mean_vec("translation", 3),
                "size": mean_vec("size", 3),
                "rotation": [math.cos(mean_yaw/2), 0.0, 0.0, math.sin(mean_yaw/2)],
                "velocity": mean_vec("velocity", 2, [0.0, 0.0]),
                "detection_name": class_name,
                "detection_score": sum(item.score for item in members)/count,
                "attribute_name": str(best.row.get("attribute_name", "")),
            }
            supports[count] += 1
            fused.append((cluster.ordinal, box, count))
    fused.sort(key=lambda item: (-float(item[1]["detection_score"]), item[0]))
    return [item[1] for item in fused[:MAX_BOXES]], {
        "inputs": input_counts, "clusters": len(fused), "supports": supports,
        "truncated": len(fused) > MAX_BOXES,
    }


def clamp(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = 0.0
    if not math.isfinite(parsed):
        parsed = 0.0
    return min(max(parsed, 0.0), 1.0)


def iter_table_objects(path: Path) -> Iterator[bytes]:
    current: Optional[List[bytes]] = None
    depth=0;in_string=False;structural=re.compile(rb'\\.|["{}]')
    with path.open("rb") as fh:
        require(fh.readline().strip() == b"[", f"unsupported table layout: {path}")
        for line in fh:
            stripped = line.lstrip()
            if current is None:
                if stripped.startswith(b"]"):
                    return
                require(stripped.startswith(b"{") or not stripped.strip(), f"bad table record: {path}")
                if not stripped.strip():
                    continue
                current = [line]
            else:
                current.append(line)
            for match in structural.finditer(line):
                token=match.group(0)
                if token.startswith(b"\\"):continue
                if token==b'"':in_string=not in_string
                elif not in_string and token==b"{":depth+=1
                elif not in_string and token==b"}":depth-=1
                require(depth>=0,f"unbalanced table object: {path}")
            if current is not None and depth==0:
                require(not in_string,f"unterminated table string: {path}")
                yield b"".join(current).rstrip().rstrip(b",")
                current = None
    require(current is None and depth==0 and not in_string, f"truncated table: {path}")


def filtered_table(path: Path, regex: re.Pattern[bytes], wanted: set[str]) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    encoded = {token.encode("ascii") for token in wanted}
    for raw in iter_table_objects(path):
        match = regex.search(raw)
        if match is None or match.group(1) not in encoded:
            continue
        row = json.loads(raw)
        token = str(row["token"])
        require(token not in selected, f"duplicate table token: {path}/{token}")
        selected[token] = row
    return selected


def table_context(root: Path, version: str, sample_tokens: set[str]) -> dict[str, dict[str, dict[str, Any]]]:
    base = root/version
    samples = filtered_table(base/"sample.json", TOKEN_RE, sample_tokens)
    require(set(samples) == sample_tokens, "sample table coverage mismatch")
    data: dict[str, dict[str, Any]] = {}
    encoded = {token.encode("ascii") for token in sample_tokens}
    for raw in iter_table_objects(base/"sample_data.json"):
        match = SAMPLE_TOKEN_RE.search(raw)
        if match is None or match.group(1) not in encoded:
            continue
        row = json.loads(raw)
        if row.get("is_key_frame"):
            data[str(row["token"])] = row
    cal_tokens = {str(row["calibrated_sensor_token"]) for row in data.values()}
    pose_tokens = {str(row["ego_pose_token"]) for row in data.values()}
    cal = filtered_table(base/"calibrated_sensor.json", TOKEN_RE, cal_tokens)
    pose = filtered_table(base/"ego_pose.json", TOKEN_RE, pose_tokens)
    sensor_tokens = {str(row["sensor_token"]) for row in cal.values()}
    sensors = filtered_table(base/"sensor.json", TOKEN_RE, sensor_tokens)
    require(set(cal)==cal_tokens and set(pose)==pose_tokens and set(sensors)==sensor_tokens, "table FK closure")
    for row in data.values():
        channel = str(sensors[str(cal[str(row["calibrated_sensor_token"])] ["sensor_token"])] ["channel"])
        samples[str(row["sample_token"])].setdefault("data", {})[channel] = str(row["token"])
    return {"sample": samples, "sample_data": data, "calibrated_sensor": cal, "ego_pose": pose, "sensor": sensors}


def quat_rotate(q: list[Any], p: list[Any]) -> list[float]:
    w,x,y,z = (float(v) for v in q); px,py,pz = (float(v) for v in p)
    return [
        (1-2*(y*y+z*z))*px + 2*(x*y-z*w)*py + 2*(x*z+y*w)*pz,
        2*(x*y+z*w)*px + (1-2*(x*x+z*z))*py + 2*(y*z-x*w)*pz,
        2*(x*z-y*w)*px + 2*(y*z+x*w)*py + (1-2*(x*x+y*y))*pz,
    ]


def transform(q: list[Any], t: list[Any], p: list[Any]) -> list[float]:
    value = quat_rotate(q, p)
    return [value[i]+float(t[i]) for i in range(3)]


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def read_pcd(path: Path) -> list[dict[str, float]]:
    with path.open("rb") as fh:
        header: dict[str, list[str]] = {}
        while True:
            line = fh.readline()
            require(line, f"truncated PCD header: {path}")
            text = line.decode("ascii").strip()
            if not text or text.startswith("#"):
                continue
            parts = text.split()
            header[parts[0].upper()] = parts[1:]
            if parts[0].upper() == "DATA":
                require(parts[1].lower() == "binary", f"unsupported PCD DATA: {path}")
                break
        fields = header["FIELDS"]
        sizes = [int(v) for v in header["SIZE"]]
        types = header["TYPE"]
        counts = [int(v) for v in header.get("COUNT", ["1"]*len(fields))]
        points = int(header["POINTS"][0])
        formats = {("F",4):"f", ("F",8):"d", ("I",1):"b", ("I",2):"h", ("I",4):"i", ("U",1):"B", ("U",2):"H", ("U",4):"I"}
        fmt = "<" + "".join(formats[(typ,size)]*count for typ,size,count in zip(types,sizes,counts))
        stride = struct.calcsize(fmt)
        payload = fh.read()
        require(len(payload) >= stride*points, f"truncated PCD data: {path}")
        rows: list[dict[str,float]] = []
        for idx in range(points):
            values = struct.unpack_from(fmt, payload, idx*stride)
            cursor=0; row: dict[str,float]={}
            for name,count in zip(fields,counts):
                row[name]=float(values[cursor]); cursor += count
            if (
                int(row.get("invalid_state", 0.0)) in {0}
                # Match nuscenes-devkit RadarPointCloud defaults exactly.
                # ``{0, 2, 6}`` is the optional moving-object-only filter;
                # the locked materializer calls ``from_file`` without that
                # override, whose default is range(0, 7).
                and 0 <= int(row.get("dyn_prop", 0.0)) < 7
                and int(row.get("ambig_state", 3.0)) in {3}
            ):
                rows.append(row)
        return rows


def read_npy(path: Path) -> list[list[float]]:
    """Read the small C-order float NPY arrays used by synthetic smoke tests."""
    with path.open("rb") as fh:
        require(fh.read(6) == b"\x93NUMPY", f"bad NPY magic: {path}")
        major, minor = struct.unpack("BB", fh.read(2))
        require((major, minor) in {(1, 0), (2, 0), (3, 0)}, f"unsupported NPY version: {path}")
        header_len = struct.unpack("<H" if major == 1 else "<I", fh.read(2 if major == 1 else 4))[0]
        header = ast.literal_eval(fh.read(header_len).decode("latin1"))
        require(header.get("fortran_order") is False, f"Fortran NPY unsupported: {path}")
        shape = tuple(int(v) for v in header.get("shape", ()))
        require(len(shape) == 2, f"2-D NPY required: {path}")
        descr = str(header.get("descr"))
        formats = {"<f4": "f", "=f4": "f", "<f8": "d", "=f8": "d"}
        require(descr in formats, f"float NPY required: {path}/{descr}")
        fmt = "<" + formats[descr] * shape[1]
        stride = struct.calcsize(fmt)
        payload = fh.read()
        require(len(payload) == stride * shape[0], f"NPY payload size mismatch: {path}")
        return [list(struct.unpack_from(fmt, payload, idx * stride)) for idx in range(shape[0])]


def embedded_radar_points(info: dict[str, Any]) -> tuple[list[list[float]], tuple[float, float]]:
    names = ["x", "y", "z", "dyn_prop", "id", "rcs", "vx", "vy", "vx_comp", "vy_comp"]
    points: list[list[float]] = []
    for channel in sorted(info.get("radars", {})):
        sweeps = info["radars"].get(channel)
        if not isinstance(sweeps, list):
            continue
        for sweep in sweeps[:1]:
            path = Path(str(sweep["data_path"]))
            for values in read_npy(path):
                row = {name: float(values[idx]) for idx, name in enumerate(names) if idx < len(values)}
                xyz = transform(
                    sweep["sensor2ego_rotation"], sweep["sensor2ego_translation"],
                    [row.get("x", 0.0), row.get("y", 0.0), row.get("z", 0.0)],
                )
                xyz = transform(sweep["ego2global_rotation"], sweep["ego2global_translation"], xyz)
                velocity = quat_rotate(
                    sweep["sensor2ego_rotation"],
                    [row.get("vx_comp", row.get("vx", 0.0)), row.get("vy_comp", row.get("vy", 0.0)), 0.0],
                )
                velocity = quat_rotate(sweep["ego2global_rotation"], velocity)
                lag = (
                    (safe(info.get("timestamp")) - safe(sweep.get("timestamp"))) / 1e6
                    if info.get("timestamp") is not None and sweep.get("timestamp") is not None
                    else 0.0
                )
                points.append([f32(v) for v in [*xyz, row.get("rcs", 0.0), velocity[0], velocity[1], lag]])
    ego_value = info.get("ego2global_translation")
    require(isinstance(ego_value, (list, tuple)) and len(ego_value) >= 2, "embedded smoke lacks ego pose")
    return points, (float(ego_value[0]), float(ego_value[1]))


def radar_points(token: str, tables: dict[str, dict[str, dict[str, Any]]], root: Path) -> tuple[list[list[float]], tuple[float,float]]:
    sample = tables["sample"][token]
    lidar = tables["sample_data"][sample["data"]["LIDAR_TOP"]]
    ego = tables["ego_pose"][lidar["ego_pose_token"]]
    points: list[list[float]]=[]
    for channel in sorted(k for k in sample["data"] if k.startswith("RADAR_")):
        sd=tables["sample_data"][sample["data"][channel]]
        cal=tables["calibrated_sensor"][sd["calibrated_sensor_token"]]
        pose=tables["ego_pose"][sd["ego_pose_token"]]
        path=Path(sd["filename"]); path=path if path.is_absolute() else root/path
        for row in read_pcd(path):
            xyz=transform(cal["rotation"],cal["translation"],[row["x"],row["y"],row["z"]])
            xyz=transform(pose["rotation"],pose["translation"],xyz)
            vel=quat_rotate(cal["rotation"],[row.get("vx_comp",row.get("vx",0.0)),row.get("vy_comp",row.get("vy",0.0)),0.0])
            vel=quat_rotate(pose["rotation"],vel)
            lag = (safe(sample.get("timestamp")) - safe(sd.get("timestamp"))) / 1e6
            points.append([f32(v) for v in [*xyz,row.get("rcs",0.0),vel[0],vel[1],lag]])
    return points,(float(ego["translation"][0]),float(ego["translation"][1]))


def radar_features(box: dict[str, Any], points: list[list[float]]) -> dict[str,float]:
    x,y=map(float,box["translation"][:2]); width,length=map(float,box["size"][:2]); angle=yaw(box["rotation"])
    chosen: list[tuple[list[float],float]]=[]
    for point in points:
        dx,dy=point[0]-x,point[1]-y; dist=math.hypot(dx,dy)
        lx=dx*math.cos(-angle)-dy*math.sin(-angle); ly=dx*math.sin(-angle)+dy*math.cos(-angle)
        if abs(lx)<=width/2+1 and abs(ly)<=length/2+1 and dist<=4:
            chosen.append((point,dist))
    if not chosen:
        return {"radar_point_count":0.0,"radar_point_density":0.0,"radar_min_center_dist":4.0,"radar_mean_center_dist":4.0,"radar_rcs_mean":0.0,"radar_rcs_max":0.0,"radar_vx_mean":0.0,"radar_vy_mean":0.0,"radar_speed_mean":0.0,"radar_velocity_delta_mean":0.0,"radar_velocity_support_count":0.0,"radar_time_lag_mean":0.0,"radar_time_lag_min":0.0,"radar_time_lag_max":0.0,"radar_time_lag_std":0.0}
    vals=[v for v,_ in chosen]; dists=[d for _,d in chosen]; n=len(vals)
    vx=[v[4] for v in vals if math.isfinite(v[4])]; vy=[v[5] for v in vals if math.isfinite(v[5])]
    speeds=[math.hypot(v[4],v[5]) for v in vals if math.isfinite(v[4]) and math.isfinite(v[5])]
    bv=list(box.get("velocity",[0.0,0.0])); delta=[math.hypot(v[4]-float(bv[0]),v[5]-float(bv[1])) for v in vals]
    lag=[v[6] for v in vals if math.isfinite(v[6])]
    mean=lambda seq: sum(seq)/len(seq) if seq else 0.0
    return {"radar_point_count":float(n),"radar_point_density":n/max((width+2)*(length+2),1e-6),"radar_min_center_dist":min(dists),"radar_mean_center_dist":mean(dists),"radar_rcs_mean":mean([0 if not math.isfinite(v[3]) else v[3] for v in vals]),"radar_rcs_max":max([0 if not math.isfinite(v[3]) else v[3] for v in vals]),"radar_vx_mean":mean(vx),"radar_vy_mean":mean(vy),"radar_speed_mean":mean(speeds),"radar_velocity_delta_mean":mean(delta),"radar_velocity_support_count":float(len(delta)),"radar_time_lag_mean":mean(lag),"radar_time_lag_min":min(lag) if lag else 0.0,"radar_time_lag_max":max(lag) if lag else 0.0,"radar_time_lag_std":math.sqrt(mean([(v-mean(lag))**2 for v in lag])) if lag else 0.0}


def safe(value: Any, default: float=0.0) -> float:
    try: parsed=float(value)
    except (TypeError,ValueError): return default
    return parsed if math.isfinite(parsed) else default


def sigmoid(v: float) -> float:
    if v>=0: return 1/(1+math.exp(-v))
    z=math.exp(v); return z/(1+z)


def model_score(row: dict[str,Any], model: dict[str,Any]) -> float:
    features=row["features"]; categorical=row.get("categorical",{})
    x=[1.0]
    for name in model["numeric_features"]:
        stat=model["stats"][name]
        x.append((safe(features.get(name))-safe(stat.get("mean")))/max(safe(stat.get("std"),1.0),1e-6))
    for key in sorted(model.get("categorical_levels",{})):
        value=str(categorical.get(key,"")); x += [1.0 if value==level else 0.0 for level in model["categorical_levels"][key]]
    schema=model["schema_version"]
    if schema=="bge_af_linear_arbiter_v1": return sigmoid(sum(safe(w)*v for w,v in zip(model["weights"],x)))
    require(schema=="bge_af_mlp_arbiter_v1", f"unsupported smoke model: {schema}")
    hidden=[]
    for j,w2 in enumerate(model["output_weights"]):
        hidden.append(math.tanh(safe(model["hidden_bias"][j])+sum(safe(model["input_hidden_weights"][j][i])*x[i] for i in range(len(x)))))
    return sigmoid(safe(model.get("output_bias"))+sum(safe(w)*v for w,v in zip(model["output_weights"],hidden)))


def range_bin(value: float, edges: list[float]) -> str:
    def fmt(v:float)->str: return str(int(round(v))) if abs(v-round(v))<1e-9 else f"{v:g}".replace(".","p")
    edges=sorted(set(safe(v) for v in edges))
    if value<edges[0]: return f"lt_{fmt(edges[0])}"
    for lo,hi in zip(edges,edges[1:]):
        if lo<=value<hi:return f"{fmt(lo)}_{fmt(hi)}"
    return f"{fmt(edges[-1])}_inf"


def learned(row: Dict[str,Any], idx:int, model:Dict[str,Any], table:Optional[Dict[str,Any]])->Tuple[int,Dict[str,Any]]:
    raw=model_score(row,model); temperature=power=cap=1.0
    if table is not None:
        edges=table.get("grouping",{}).get("range_edges",[0,20,40,60])
        key=f'class={row["class_name"]}|range={range_bin(max(0.0,safe(row["features"]["ego_range_xy"])),edges)}|source={row["source_signature"]}'
        block=next((v for v in table["bins"] if v.get("key")==key),table.get("global",{}))
        temperature=safe(block.get("temperature"),1.0); power=safe(block.get("power"),1.0); cap=min(max(safe(block.get("cap"),1.0),0.0),1.0)
    score=min(max(raw,0.0),1.0)
    if temperature!=1.0:
        p=min(max(score,1e-6),1-1e-6); score=sigmoid(math.log(p/(1-p))/temperature)
    if power!=1.0: score=score**power
    score=min(score,cap)
    out=dict(row["output_box"]); out["sample_token"]=row["sample_token"]
    out["detection_score"]=clamp(0.9*safe(row.get("base_score",out.get("detection_score")))+0.1*score)
    return idx,out


def parse_args(argv: Optional[List[str]]=None)->argparse.Namespace:
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-dir",required=True,type=Path)
    parser.add_argument("--execution-config-json",required=True,type=Path)
    parser.add_argument("--expected-execution-config-sha256",required=True)
    parser.add_argument("--report-json",required=True,type=Path)
    args=parser.parse_args(argv)
    require(not args.report_json.exists() and not args.report_json.is_symlink(),f"refusing to overwrite/symlink report: {args.report_json}")
    require(args.run_dir.is_dir(),f"missing run dir: {args.run_dir}")
    return args


def verify(args: argparse.Namespace)->dict[str,Any]:
    require_no_symlink_chain(args.run_dir,"run_dir")
    require_no_symlink_chain(args.execution_config_json,"execution_config_json")
    require_no_symlink_chain(args.report_json.parent,"report_parent")
    config_id=identity(args.execution_config_json)
    require(config_id["sha256"]==args.expected_execution_config_sha256,"execution config SHA mismatch")
    config=json.loads(args.execution_config_json.read_text())
    require(config.get("schema_version")==CONFIG_SCHEMA and config.get("status")=="frozen_for_execution","bad execution config")
    plan=check_identity("plan",Path(config["plan"]["path"]),config["plan"])
    plan_data=json.loads(Path(plan["path"]).read_text())
    require(plan_data.get("schema_version")==PLAN_SCHEMA and plan_data.get("status")=="preregistered_before_full_run","bad plan")
    validate_config_against_plan(config,plan_data)
    summary_path=args.run_dir/"materialization_summary.json"
    summary=json.loads(summary_path.read_text())
    require(summary.get("schema_version")==SUMMARY_SCHEMA,"bad materialization summary")
    scope=summary.get("scope"); require(scope in {"bounded_smoke","full"},"bad scope")
    scopes=config["scopes"]; scope_cfg=scopes[scope]
    require(same_path(summary.get("output_dir"),args.run_dir),"summary/run-dir path drift")
    if scope=="full":
        require(same_path(scope_cfg["output_dir"],args.run_dir),"full output path contract failed")
        storage_mount=Path(scope_cfg["storage_mount"])
        require_no_symlink_chain(storage_mount,"storage_mount")
        require(mount_source(storage_mount)==scope_cfg.get("expected_mount_source"),"full storage mount source drift")
    else:
        require(below(args.run_dir,Path(scope_cfg["output_root"])),"smoke output root contract failed")
    report_root=Path(scope_cfg["report_root"])
    require(below(args.report_json,report_root),"report path is outside configured report root")
    require(not below(args.report_json,args.run_dir),"checker report may not mutate committed run dir")
    require(summary.get("evaluation_performed") is False and summary.get("training_performed") is False,"materializer performed forbidden work")
    require(summary.get("atomic_directory_commit") is True and summary.get("overwrite_allowed") is False,"atomic/no-overwrite contract failed")

    inputs=config["inputs"]
    input_paths={key:Path(inputs[key]["path"]) for key in ("cache_jsonl","cache_summary_json","sample_info_pkl","radar_model","radar_calibration","no_radar_model","no_radar_calibration","meta_from_result_json")}
    checked_inputs={
        key:check_identity(
            key,path,inputs[key],allow_registered_symlink=True
        )
        for key,path in input_paths.items()
    }
    for idx,expert in enumerate(inputs["naive_experts"]):
        require(expert.get("fixed_ordinal")==idx,"expert ordinal drift")
        check_identity(
            f"expert[{idx}]",
            Path(expert["path"]),
            expert,
            allow_registered_symlink=True,
        )
    for name,expected in config["implementation"].items():
        check_identity(f"implementation.{name}",Path(expected["path"]),expected)
    nusc=inputs["nuscenes"]
    for name,expected in nusc["tables"].items():
        check_identity(f"nuscenes.{name}",Path(expected["path"]),expected)
    lock=summary.get("execution_lock",{})
    require(lock.get("execution_config",{}).get("sha256")==config_id["sha256"],"summary config lock drift")
    require(lock.get("plan",{}).get("sha256")==plan["sha256"],"summary plan lock drift")
    require(lock.get("scope")==scope,"summary execution scope drift")

    output_indexes: dict[str,ResultIndex]={}
    output_checks: dict[str,Any]={}
    try:
        for stage,filename in OUTPUTS.items():
            path=args.run_dir/filename
            declared=summary["outputs"][stage]
            actual=identity(path)
            require(actual["sha256"]==declared.get("sha256") and actual["size_bytes"]==declared.get("size_bytes"),f"output identity drift: {stage}")
            index=ResultIndex(path); output_indexes[stage]=index
            require(index.meta.get("use_lidar") is False,f"meta.use_lidar must be false: {stage}")
            output_checks[stage]={"sha256":actual["sha256"],"samples":len(index.order),"boxes":0}
        orders=[index.order for index in output_indexes.values()]
        require(all(order==orders[0] for order in orders[1:]),"five-arm token order mismatch")
        require(all(index.meta==next(iter(output_indexes.values())).meta for index in output_indexes.values()),"five-arm meta mismatch")

        expert_indexes=[ResultIndex(Path(item["path"])) for item in inputs["naive_experts"]]
        try:
            require(all(set(index.order)==set(expert_indexes[0].order) for index in expert_indexes),"raw expert coverage mismatch")
            cache_digest=hashlib.sha256(); cluster_digest=hashlib.sha256(); token_digest=hashlib.sha256()
            cache_rows=0; tokens=[]; current=None; grouped=[]; closed=set(); candidate_counts=Counter(); truncated=Counter()
            max_score_difference=0.0;max_score_witness=None
            naive_inputs=[0,0,0,0]; naive_clusters=0; naive_support=Counter(); naive_truncated=0
            tables=None
            info_by_token={}
            radar_model=json.loads(input_paths["radar_model"].read_text()); radar_cal=json.loads(input_paths["radar_calibration"].read_text())
            no_model=json.loads(input_paths["no_radar_model"].read_text()); no_cal=json.loads(input_paths["no_radar_calibration"].read_text())
            payload=pickle.loads(input_paths["sample_info_pkl"].read_bytes()); items=payload.get("infos",payload.get("data_list")) if isinstance(payload,dict) else payload
            info_by_token={str(item.get("token") or item.get("sample_token")):item for item in items}
            if scope=="full" or not summary["storage_guard"].get("mount_guard_bypassed_for_smoke"):
                tables=table_context(Path(nusc["root"]),nusc["version"],set(orders[0]))

            def process(token:str, rows:list[tuple[int,dict[str,Any]]])->None:
                nonlocal naive_clusters,naive_truncated,tables,max_score_difference,max_score_witness
                require(token in output_indexes["naive_equal_wbf"].ranges,f"cache token absent from outputs: {token}")
                actual_by_stage={stage:index.rows(token) for stage,index in output_indexes.items()}
                for stage,actual in actual_by_stage.items():
                    require(len(actual)<=MAX_BOXES,f"top500 violated: {stage}/{token}")
                    previous=math.inf
                    for idx,box in enumerate(actual):
                        validate_box(box,token,f"{stage}/{token}/{idx}")
                        score=float(box["detection_score"]); require(score<=previous+1e-15,f"score order violated: {stage}/{token}"); previous=score
                    output_checks[stage]["boxes"] += len(actual)
                expected_naive,telemetry=naive_sample(token,[index.rows(token) for index in expert_indexes])
                assert_rows_equal(actual_by_stage["naive_equal_wbf"],expected_naive,f"naive/{token}")
                for i,count in enumerate(telemetry["inputs"]):naive_inputs[i]+=count
                naive_clusters+=telemetry["clusters"];naive_support.update(telemetry["supports"]);naive_truncated+=int(telemetry["truncated"])
                e1=[]; all_geometry=Counter()
                enriched=[]
                points=[]; ego=None
                if scope in {"bounded_smoke","full"}:
                    if tables is not None: points,ego=radar_points(token,tables,Path(nusc["root"]))
                    else:
                        points,ego=embedded_radar_points(info_by_token[token])
                for idx,row in rows:
                    require(row.get("schema_version")==CACHE_SCHEMA,f"bad cache schema: {token}/{idx}")
                    features=row.get("features");require(isinstance(features,dict),f"missing cache features: {token}/{idx}")
                    require(safe(features.get("radar_point_count"))==0 and "ego_range_xy" not in features,f"source cache radar/range defect contract violated: {token}/{idx}")
                    full=safe(features.get("full_mode_count"),float("nan")); require(math.isfinite(full),f"missing full_mode_count: {token}/{idx}")
                    cluster_full=sum(1 for box in row.get("cluster_boxes",[]) if isinstance(box,dict) and box.get("mode")=="full")
                    require(abs(full-cluster_full)<=1e-9,f"full_mode_count closure failed: {token}/{idx}")
                    base=dict(row["output_box"]);base["sample_token"]=token;base["detection_score"]=clamp(row.get("base_score",base.get("detection_score")))
                    if full>0:e1.append((idx,base))
                    all_geometry[geometry(base)]+=1
                    if scope in {"bounded_smoke","full"}:
                        copied=dict(row); f=dict(features); f["ego_range_xy"]=math.hypot(float(base["translation"][0])-ego[0],float(base["translation"][1])-ego[1])
                        f.update(radar_features(base,points))
                        copied["features"]=f; enriched.append((idx,copied))
                e1.sort(key=lambda item:(-float(item[1]["detection_score"]),item[0])); expected_e1=[v for _,v in e1[:MAX_BOXES]]
                assert_rows_equal(actual_by_stage["rgate_rescore_only"],expected_e1,f"E1/{token}")
                candidate_counts["rgate_rescore_only"]+=len(e1);truncated["rgate_rescore_only"]+=int(len(e1)>MAX_BOXES)
                for stage in ("learned_radar_uncalibrated","learned_radar_calibrated","learned_no_radar_calibrated"):
                    candidate_counts[stage]+=len(rows);truncated[stage]+=int(len(rows)>MAX_BOXES)
                if scope in {"bounded_smoke","full"}:
                    specs=(("learned_radar_uncalibrated",radar_model,None),("learned_radar_calibrated",radar_model,radar_cal),("learned_no_radar_calibrated",no_model,no_cal))
                    for stage,model,table in specs:
                        values=[learned(row,idx,model,table) for idx,row in enriched];values.sort(key=lambda item:(-float(item[1]["detection_score"]),item[0]))
                        difference,index=assert_learned_rows_close(actual_by_stage[stage],[v for _,v in values[:MAX_BOXES]],f"{stage}/{token}")
                        if difference>max_score_difference:
                            max_score_difference=difference;max_score_witness={"stage":stage,"sample_token":token,"output_index":index,"absolute_score_difference":difference}
                else:
                    for stage in ("learned_radar_uncalibrated","learned_radar_calibrated","learned_no_radar_calibrated"):
                        chosen=Counter(geometry(v) for v in actual_by_stage[stage]);require(all(chosen[k]<=all_geometry[k] for k in chosen),f"learned geometry escaped cache: {stage}/{token}")
                        require(len(actual_by_stage[stage])==min(MAX_BOXES,len(rows)),f"learned count closure failed: {stage}/{token}")

            max_samples=int(summary["coverage"].get("max_samples",0))
            with input_paths["cache_jsonl"].open("rb") as fh:
                for line in fh:
                    if current is None and max_samples and len(tokens)>=max_samples: break
                    if not line.strip():
                        cache_digest.update(line)
                        continue
                    row=json.loads(line);token=str(row.get("sample_token",""));require(token,"empty cache token")
                    if current is None:current=token
                    elif token!=current:
                        require(token not in closed,f"cache token regrouped: {token}");closed.add(current);process(current,grouped);tokens.append(current);token_digest.update((current+"\n").encode());grouped=[];current=token
                        if max_samples and len(tokens)>=max_samples:break
                    cache_digest.update(line)
                    cluster_payload={key:row.get(key) for key in ("sample_token","cluster_id","class_name","sources","groups","source_signature","group_signature","base_score","output_box","cluster_boxes")}
                    cluster_digest.update(canonical(cluster_payload)+b"\n");grouped.append((cache_rows,row));cache_rows+=1
                else:
                    if current is not None:process(current,grouped);tokens.append(current);token_digest.update((current+"\n").encode())
            require(tokens==orders[0],"cache/output sample order mismatch")
            coverage=summary["coverage"]
            require(cache_rows==coverage["processed_row_count"] and len(tokens)==coverage["processed_sample_count"],"summary coverage count drift")
            require(token_digest.hexdigest()==coverage["sample_token_sha256"],"sample token digest drift")
            require(cluster_digest.hexdigest()==summary["read_only_cache_invariants"]["processed_cluster_identity_sha256"],"cluster identity digest drift")
            cache_summary=summary["input_identities"]["cache_jsonl"]
            require(cache_digest.hexdigest()==cache_summary["processed_prefix_sha256"],"cache prefix SHA drift")
            require(cache_rows==summary["read_only_cache_invariants"]["nonradar_feature_rows_checked"],"nonradar check coverage drift")
            require(naive_clusters==summary["naive_telemetry"]["cluster_count"] and naive_truncated==summary["naive_telemetry"]["truncated_sample_count"],"naive telemetry drift")
            require({str(k):v for k,v in sorted(naive_support.items())}==summary["naive_telemetry"]["support_histogram"],"naive support telemetry drift")
            configured_names=[item["name"] for item in inputs["naive_experts"]]
            require(dict(sorted(zip(configured_names,naive_inputs)))==summary["naive_telemetry"]["input_box_counts"],"naive input telemetry drift")
            for stage in candidate_counts:
                require(candidate_counts[stage]==summary["outputs"][stage]["candidate_box_count"],f"candidate telemetry drift: {stage}")
                require(truncated[stage]==summary["outputs"][stage]["truncated_sample_count"],f"truncation telemetry drift: {stage}")
            for stage in OUTPUTS:
                require(output_checks[stage]["boxes"]==summary["outputs"][stage]["output_box_count"],f"output box count drift: {stage}")
            if scope=="full":
                require(summary["coverage"].get("cache_reached_eof") is True,"full cache did not reach EOF")
                require(len(tokens)==FULL_SAMPLES and cache_rows==FULL_ROWS,"full fixed coverage failed")
                require(cache_digest.hexdigest()==inputs["cache_jsonl"]["sha256"],"full cache SHA closure failed")
                status="passed_for_official_eval";eligible=True
            else:
                # A bounded smoke is defined by the explicit non-zero
                # ``max_samples`` scope, not by whether a tiny synthetic
                # fixture happens to end at that boundary.  Real smoke uses a
                # prefix of the 6,019-sample cache; unit fixtures may contain
                # exactly one sample and legitimately reach EOF.
                require(0<len(tokens)<=int(scope_cfg["max_samples_max"]),"smoke bound failed")
                status="bounded_smoke_verified";eligible=False
        finally:
            for index in expert_indexes:index.close()
    finally:
        for index in output_indexes.values():index.close()
    return {"schema_version":REPORT_SCHEMA,"status":status,"passed_for_official_eval":eligible,"scope":scope,"run_dir":str(args.run_dir.resolve()),"execution_config":config_id,"plan":plan,"summary":identity(summary_path),"coverage":{"samples":len(tokens),"cache_rows":cache_rows},"outputs":output_checks,"numeric_closure":{"learned_score_absolute_tolerance":1e-7,"maximum_absolute_score_difference":max_score_difference,"maximum_difference_witness":max_score_witness,"selected_geometry_and_order_exact":True},"checker":identity(Path(__file__).resolve()),"checks":{"identity_lock":True,"no_overwrite_and_path_contract":True,"five_arm_schema_and_finite":True,"top500_and_score_order":True,"raw_four_expert_naive_exact":True,"E1_full_mode_exact":True,"learned_cache_geometry_closure":True,"learned_numeric_exact":True,"cache_source_closure":True}}


def write_report(path:Path,payload:dict[str,Any])->None:
    missing=[]; cursor=path.parent
    while not cursor.exists():
        require(not cursor.is_symlink(),f"report parent component is a symlink: {cursor}")
        missing.append(cursor);cursor=cursor.parent
    require(cursor.is_dir() and not cursor.is_symlink(),f"unsafe report parent: {cursor}")
    path.parent.mkdir(parents=True,exist_ok=True)
    cursor=path.parent
    while True:
        require(not cursor.is_symlink(),f"report parent component is a symlink: {cursor}")
        if cursor==cursor.parent:break
        cursor=cursor.parent
    require(not path.exists() and not path.is_symlink(),f"refusing to overwrite/symlink report: {path}")
    fd,temp=tempfile.mkstemp(prefix=f".{path.name}.tmp.",dir=path.parent)
    try:
        with os.fdopen(fd,"wb") as fh:
            fh.write(json.dumps(payload,indent=2,sort_keys=True,ensure_ascii=False).encode()+b"\n");fh.flush();os.fsync(fh.fileno())
        if path.exists() or path.is_symlink():raise AuditError(f"report appeared during check: {path}")
        # link() is the portable stdlib no-replace primitive: it fails if an
        # attacker or concurrent verifier creates the destination first.
        os.link(temp,path,follow_symlinks=False)
        os.unlink(temp)
        temp=""
        parent_fd=os.open(str(path.parent),os.O_RDONLY)
        try:os.fsync(parent_fd)
        finally:os.close(parent_fd)
    finally:
        if temp and os.path.exists(temp):os.unlink(temp)


def main(argv:Optional[List[str]]=None)->None:
    args=parse_args(argv)
    try:
        report=verify(args)
    except Exception as exc:
        report={"schema_version":REPORT_SCHEMA,"status":"failed","passed_for_official_eval":False,"error":f"{type(exc).__name__}: {exc}","run_dir":str(args.run_dir.resolve()),"checker":identity(Path(__file__).resolve())}
        write_report(args.report_json,report)
        raise SystemExit(report["error"])
    write_report(args.report_json,report)
    print(json.dumps(report,indent=2,sort_keys=True,ensure_ascii=False))


if __name__=="__main__":main()
