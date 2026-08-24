#!/usr/bin/env python3
"""Deterministic gzip storage for large standard nuScenes result JSON streams.

The scientific runners use this module only as a byte sink.  It does not rank,
filter, score, or otherwise inspect predictions.  Every byte is hashed before
compression and can later be recovered and checked without retaining a large
uncompressed file.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Tuple


HASH_CHUNK_BYTES = 8 * 1024 * 1024


class ArchiveError(RuntimeError):
    pass


def _reject_nonfinite(value: Any, where: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ArchiveError("non-finite value at %s" % where)
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_nonfinite(child, "%s[%d]" % (where, index))
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_nonfinite(child, "%s.%s" % (where, key))
        return
    raise ArchiveError("non-JSON value at %s: %s" % (where, type(value).__name__))


def canonical_bytes(value: Any) -> bytes:
    _reject_nonfinite(value, "root")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DeterministicGzipResultWriter:
    """Write compact nuScenes results directly to a deterministic gzip file."""

    def __init__(
        self,
        path: Path,
        meta: Mapping[str, Any],
        *,
        compresslevel: int = 6,
        max_per_sample: int = 500,
    ) -> None:
        if os.path.lexists(str(path)):
            raise ArchiveError("refusing to overwrite archive: %s" % path)
        if not 0 <= int(compresslevel) <= 9:
            raise ArchiveError("gzip compresslevel must be in [0,9]")
        self.path = path
        self.max_per_sample = int(max_per_sample)
        self.compresslevel = int(compresslevel)
        self.raw_digest = hashlib.sha256()
        self.token_digest = hashlib.sha256()
        self.raw_size = 0
        self.sample_count = 0
        self.box_count = 0
        self.first = True
        self.closed = False
        self._raw_handle = path.open("xb")
        self._gzip = gzip.GzipFile(
            filename="",
            mode="wb",
            compresslevel=int(compresslevel),
            fileobj=self._raw_handle,
            mtime=0,
        )
        self._write(b'{"meta":')
        self._write(canonical_bytes(dict(meta)))
        self._write(b',"results":{')

    def _write(self, value: bytes) -> None:
        self._gzip.write(value)
        self.raw_digest.update(value)
        self.raw_size += len(value)

    def write_sample(self, token: str, rows: Sequence[Mapping[str, Any]]) -> None:
        if self.closed:
            raise ArchiveError("result archive is already closed")
        if not isinstance(token, str) or not token:
            raise ArchiveError("sample token must be a non-empty string")
        if len(rows) > self.max_per_sample:
            raise ArchiveError(
                "sample %s exceeds top-%d: %d"
                % (token, self.max_per_sample, len(rows))
            )
        if not self.first:
            self._write(b",")
        self.first = False
        self._write(canonical_bytes(token))
        self._write(b":")
        self._write(canonical_bytes(list(rows)))
        self.token_digest.update((token + "\n").encode("utf-8"))
        self.sample_count += 1
        self.box_count += len(rows)

    def close(self) -> Dict[str, Any]:
        if not self.closed:
            self._write(b"}}")
            self._gzip.close()
            self._raw_handle.flush()
            os.fsync(self._raw_handle.fileno())
            self._raw_handle.close()
            self.closed = True
        compressed_stat = self.path.stat()
        return {
            "filename": self.path.name,
            "archive_format": "gzip",
            "gzip_compresslevel": self.compresslevel,
            "gzip_mtime": 0,
            "archive_size_bytes": compressed_stat.st_size,
            "archive_sha256": sha256_file(self.path),
            "raw_size_bytes": self.raw_size,
            "raw_sha256": self.raw_digest.hexdigest(),
            "sample_token_newline_sha256": self.token_digest.hexdigest(),
            "sample_count": self.sample_count,
            "output_box_count": self.box_count,
            "max_per_sample": self.max_per_sample,
            "standard_nuscenes_result_json": True,
        }

    def abort(self) -> None:
        if not self.closed:
            try:
                self._gzip.close()
            finally:
                self._raw_handle.close()
                self.closed = True


def verify_archive(
    path: Path,
    *,
    expected_archive_size_bytes: int,
    expected_archive_sha256: str,
    expected_raw_size_bytes: int,
    expected_raw_sha256: str,
) -> Dict[str, Any]:
    """Stream-decompress an archive and prove both compressed and raw identity."""

    if not path.is_file() or path.is_symlink():
        raise ArchiveError("missing or symlink archive: %s" % path)
    stat = path.stat()
    if stat.st_size != int(expected_archive_size_bytes):
        raise ArchiveError("archive size mismatch: %s" % path)
    archive_sha = sha256_file(path)
    if archive_sha != expected_archive_sha256:
        raise ArchiveError("archive SHA mismatch: %s" % path)
    raw_digest = hashlib.sha256()
    raw_size = 0
    with gzip.open(str(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            raw_digest.update(chunk)
            raw_size += len(chunk)
    if raw_size != int(expected_raw_size_bytes):
        raise ArchiveError("decompressed size mismatch: %s" % path)
    raw_sha = raw_digest.hexdigest()
    if raw_sha != expected_raw_sha256:
        raise ArchiveError("decompressed SHA mismatch: %s" % path)
    return {
        "archive_size_bytes": stat.st_size,
        "archive_sha256": archive_sha,
        "raw_size_bytes": raw_size,
        "raw_sha256": raw_sha,
        "decompression_identity_closed": True,
    }


class GzipNuScenesResultReader:
    """Incrementally decode one compact result archive in registered order.

    Only one sample array is materialized at a time.  The reader also hashes
    every decompressed byte, allowing a checker to close serialization identity
    while independently reconstructing predictions in the same cache pass.
    """

    def __init__(self, path: Path, *, read_chunk_bytes: int = 1024 * 1024) -> None:
        if not path.is_file() or path.is_symlink():
            raise ArchiveError("missing or symlink archive: %s" % path)
        self.path = path
        self.read_chunk_bytes = max(8, int(read_chunk_bytes))
        self.handle = gzip.open(str(path), "rb")
        self.buffer = bytearray()
        self.position = 0
        self.eof = False
        self.raw_digest = hashlib.sha256()
        self.raw_size = 0
        self.sample_count = 0
        self.box_count = 0
        self.finished = False
        self.meta = self._read_header()

    def _fill(self) -> bool:
        if self.eof:
            return False
        chunk = self.handle.read(self.read_chunk_bytes)
        if not chunk:
            self.eof = True
            return False
        self.buffer.extend(chunk)
        self.raw_digest.update(chunk)
        self.raw_size += len(chunk)
        return True

    def _compact(self) -> None:
        if self.position >= self.read_chunk_bytes * 2:
            del self.buffer[: self.position]
            self.position = 0

    def _ensure(self) -> None:
        if self.position >= len(self.buffer) and not self._fill():
            raise ArchiveError("unexpected end of decompressed JSON: %s" % self.path)

    def _skip_ws(self) -> None:
        while True:
            self._ensure()
            if self.buffer[self.position] not in b" \t\r\n":
                return
            self.position += 1
            self._compact()

    def _expect(self, value: int) -> None:
        self._skip_ws()
        if self.buffer[self.position] != value:
            raise ArchiveError(
                "expected byte %r at decompressed offset near %d: %s"
                % (chr(value), self.raw_size - len(self.buffer) + self.position, self.path)
            )
        self.position += 1

    def _value_bytes(self) -> bytes:
        self._skip_ws()
        start = self.position
        first = self.buffer[start]
        if first == ord('"'):
            cursor, escaped = start + 1, False
            while True:
                if cursor >= len(self.buffer):
                    if not self._fill():
                        raise ArchiveError("unterminated JSON string: %s" % self.path)
                    continue
                byte = self.buffer[cursor]
                if escaped:
                    escaped = False
                elif byte == ord("\\"):
                    escaped = True
                elif byte == ord('"'):
                    end = cursor + 1
                    break
                cursor += 1
        elif first in (ord("["), ord("{")):
            stack = [first]
            cursor, in_string, escaped = start + 1, False, False
            while stack:
                if cursor >= len(self.buffer):
                    if not self._fill():
                        raise ArchiveError("unterminated compound JSON value: %s" % self.path)
                    continue
                byte = self.buffer[cursor]
                if in_string:
                    if escaped:
                        escaped = False
                    elif byte == ord("\\"):
                        escaped = True
                    elif byte == ord('"'):
                        in_string = False
                elif byte == ord('"'):
                    in_string = True
                elif byte in (ord("["), ord("{")):
                    stack.append(byte)
                elif byte in (ord("]"), ord("}")):
                    expected = ord("[") if byte == ord("]") else ord("{")
                    if not stack or stack.pop() != expected:
                        raise ArchiveError("malformed JSON nesting: %s" % self.path)
                cursor += 1
            end = cursor
        else:
            raise ArchiveError("unsupported top-level JSON value: %s" % self.path)
        raw = bytes(self.buffer[start:end])
        self.position = end
        self._compact()
        return raw

    def _json_value(self) -> Any:
        return json.loads(
            self._value_bytes(),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ArchiveError("non-finite JSON literal: %s" % value)
            ),
        )

    def _read_header(self) -> Dict[str, Any]:
        self._expect(ord("{"))
        if self._json_value() != "meta":
            raise ArchiveError("first result key must be meta: %s" % self.path)
        self._expect(ord(":"))
        meta = self._json_value()
        if not isinstance(meta, dict):
            raise ArchiveError("result meta is not an object: %s" % self.path)
        self._expect(ord(","))
        if self._json_value() != "results":
            raise ArchiveError("second result key must be results: %s" % self.path)
        self._expect(ord(":"))
        self._expect(ord("{"))
        return meta

    def read_sample(self) -> Optional[Tuple[str, Sequence[Mapping[str, Any]]]]:
        if self.finished:
            return None
        self._skip_ws()
        if self.buffer[self.position] == ord("}"):
            self.position += 1
            self._expect(ord("}"))
            while True:
                while self.position < len(self.buffer) and self.buffer[self.position] in b" \t\r\n":
                    self.position += 1
                if self.position < len(self.buffer):
                    raise ArchiveError("unexpected trailing JSON bytes: %s" % self.path)
                if not self._fill():
                    break
            self.finished = True
            return None
        if self.sample_count:
            self._expect(ord(","))
        token = self._json_value()
        if not isinstance(token, str) or not token:
            raise ArchiveError("invalid result sample token: %s" % self.path)
        self._expect(ord(":"))
        rows = self._json_value()
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ArchiveError("result sample is not a box list: %s/%s" % (self.path, token))
        self.sample_count += 1
        self.box_count += len(rows)
        return token, rows

    def finish(
        self,
        *,
        expected_raw_size_bytes: int,
        expected_raw_sha256: str,
        expected_sample_count: Optional[int] = None,
        expected_box_count: Optional[int] = None,
    ) -> Dict[str, Any]:
        while self.read_sample() is not None:
            pass
        self.handle.close()
        if self.raw_size != int(expected_raw_size_bytes):
            raise ArchiveError("streamed raw size mismatch: %s" % self.path)
        digest = self.raw_digest.hexdigest()
        if digest != expected_raw_sha256:
            raise ArchiveError("streamed raw SHA mismatch: %s" % self.path)
        if expected_sample_count is not None and self.sample_count != int(expected_sample_count):
            raise ArchiveError("streamed sample count mismatch: %s" % self.path)
        if expected_box_count is not None and self.box_count != int(expected_box_count):
            raise ArchiveError("streamed box count mismatch: %s" % self.path)
        return {
            "raw_size_bytes": self.raw_size,
            "raw_sha256": digest,
            "sample_count": self.sample_count,
            "output_box_count": self.box_count,
            "stream_reached_exact_eof": True,
        }

    def close(self) -> None:
        self.handle.close()
