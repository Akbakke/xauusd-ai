#!/usr/bin/env python3
"""Publish one immutable direct-M5 OHLCV source that ends before sealed TEST.

This is a narrow provenance bridge, not a feature builder.  It reads only a
complete canonical native-M5 bundle whose declared end is exactly the 2026-07-01
TEST boundary, preserves its direct OANDA M5 axis, and writes the six OHLCV
columns needed by the five-clock cache owner.  It rejects a source or an output
row at/after that boundary; it neither opens nor accepts a TEST split file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from gx1.contracts.entry_model_native_bundle_commit_v1 import (
    publish_bundle_directory_noreplace,
)
from gx1.contracts.gx1_scope_v1 import require_offline_scope
from gx1.contracts.xau_tape_provenance_v1 import (
    XAU_INSTRUMENT,
    validate_canonical_native_source_bundle,
)

TEST_BOUNDARY_UTC = "2026-07-01T00:00:00+00:00"
PRETEST_M5_SOURCE_SCHEMA_VERSION = "gx1_direct_m5_pretest_source_v1"
OUTPUT_NAME = "m5_ohlcv.parquet"
OUTPUT_COLUMNS = ("time", "open", "high", "low", "close", "volume")
OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("time", pa.timestamp("ns", tz="UTC"), nullable=False),
        *(pa.field(name, pa.float64(), nullable=False) for name in OUTPUT_COLUMNS[1:-1]),
        pa.field("volume", pa.int64(), nullable=False),
    ]
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _write_json_fsync(path: Path, payload: dict[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_clean_repository(repo_root: Path) -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout.strip():
        raise RuntimeError("DIRECT_M5_PRETEST_REPOSITORY_DIRTY")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise RuntimeError("DIRECT_M5_PRETEST_REPOSITORY_COMMIT_INVALID")
    return commit


def _require_exact_directory(path: Path, *, label: str) -> Path:
    candidate = path.expanduser()
    if (
        not candidate.is_absolute()
        or candidate.is_symlink()
        or not candidate.is_dir()
        or candidate.resolve() != candidate
    ):
        raise RuntimeError(f"DIRECT_M5_PRETEST_{label}_INVALID: {candidate}")
    return candidate


def _require_new_output_directory(path: Path) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute() or candidate.exists() or candidate.is_symlink():
        raise RuntimeError(f"DIRECT_M5_PRETEST_OUTPUT_INVALID: {candidate}")
    parent = _require_exact_directory(candidate.parent, label="OUTPUT_PARENT")
    if candidate.resolve(strict=False) != candidate or parent != candidate.parent:
        raise RuntimeError(f"DIRECT_M5_PRETEST_OUTPUT_INVALID: {candidate}")
    return candidate


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"DIRECT_M5_PRETEST_{label}_MISSING")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"DIRECT_M5_PRETEST_{label}_INVALID") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"DIRECT_M5_PRETEST_{label}_INVALID")
    return value


def _native_part_paths(root: Path) -> Iterable[Path]:
    paths = tuple(sorted(root.glob("year=*/part-000.parquet")))
    if not paths:
        raise RuntimeError("DIRECT_M5_PRETEST_NATIVE_PARTS_MISSING")
    for path in paths:
        if path.is_symlink() or not path.is_file() or path.resolve() != path:
            raise RuntimeError("DIRECT_M5_PRETEST_NATIVE_PART_INVALID")
    return paths


def _preflight_native_m5(root: Path) -> tuple[dict[str, Any], Path, str]:
    validate_canonical_native_source_bundle(
        root,
        timeframe="M5",
        expected_declared_root=root,
    )
    manifest_path = root / "MANIFEST.json"
    manifest = _read_object(manifest_path, label="NATIVE_MANIFEST")
    boundary = pd.Timestamp(TEST_BOUNDARY_UTC)
    if (
        manifest.get("instrument") != XAU_INSTRUMENT
        or manifest.get("timeframe") != "M5"
        or pd.Timestamp(manifest.get("requested_end_utc_exclusive")) != boundary
        or pd.Timestamp(manifest.get("time_max_utc")) >= boundary
        or not isinstance(manifest.get("row_count"), int)
        or int(manifest["row_count"]) <= 0
    ):
        raise RuntimeError("DIRECT_M5_PRETEST_NATIVE_BOUNDARY_INVALID")
    return manifest, manifest_path, _sha256_file(manifest_path)


def materialize_direct_m5_pretest_source(
    *,
    native_m5_root: Path,
    out_dir: Path,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Write a hash-bound direct OHLCV file from an all-pre-TEST native bundle."""

    require_offline_scope("featurebase_build")
    native_root = _require_exact_directory(native_m5_root, label="NATIVE_ROOT")
    destination = _require_new_output_directory(out_dir)
    repository = (
        Path(__file__).resolve().parents[2]
        if repo_root is None
        else _require_exact_directory(repo_root, label="REPOSITORY")
    )
    if repository == destination or repository in destination.parents:
        raise RuntimeError("DIRECT_M5_PRETEST_OUTPUT_INSIDE_REPOSITORY")
    initial_commit = _require_clean_repository(repository)
    native_manifest, native_manifest_path, native_manifest_sha256 = _preflight_native_m5(
        native_root
    )
    source_parts = tuple(_native_part_paths(native_root))
    boundary = pd.Timestamp(TEST_BOUNDARY_UTC)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging.", dir=destination.parent))
    published = False
    try:
        output = stage / OUTPUT_NAME
        writer = pq.ParquetWriter(output, OUTPUT_SCHEMA, compression="zstd")
        row_count = 0
        first_time: pd.Timestamp | None = None
        last_time: pd.Timestamp | None = None
        previous_time: pd.Timestamp | None = None
        try:
            for source_part in source_parts:
                parquet = pq.ParquetFile(source_part)
                if any(name not in parquet.schema_arrow.names for name in OUTPUT_COLUMNS):
                    raise RuntimeError("DIRECT_M5_PRETEST_NATIVE_SCHEMA_INVALID")
                for row_group in range(parquet.metadata.num_row_groups):
                    table = parquet.read_row_group(row_group, columns=list(OUTPUT_COLUMNS))
                    if not table.schema.equals(OUTPUT_SCHEMA, check_metadata=False):
                        table = table.cast(OUTPUT_SCHEMA, safe=True)
                    timestamps = pd.DatetimeIndex(
                        pd.to_datetime(table["time"].to_pandas(), utc=True, errors="raise")
                    )
                    if (
                        len(timestamps) == 0
                        or not timestamps.is_monotonic_increasing
                        or (previous_time is not None and timestamps[0] <= previous_time)
                        or bool((timestamps >= boundary).any())
                    ):
                        raise RuntimeError("DIRECT_M5_PRETEST_TIMESTAMP_BOUNDARY_INVALID")
                    writer.write_table(table)
                    row_count += len(timestamps)
                    first_time = timestamps[0] if first_time is None else first_time
                    last_time = timestamps[-1]
                    previous_time = last_time
        finally:
            writer.close()
        if (
            row_count != int(native_manifest["row_count"])
            or first_time is None
            or last_time is None
            or last_time >= boundary
        ):
            raise RuntimeError("DIRECT_M5_PRETEST_OUTPUT_BOUNDARY_INVALID")
        _fsync_file(output)
        verified = pq.ParquetFile(output)
        if (
            verified.metadata.num_rows != row_count
            or not verified.schema_arrow.equals(OUTPUT_SCHEMA, check_metadata=False)
        ):
            raise RuntimeError("DIRECT_M5_PRETEST_OUTPUT_VERIFY_FAILED")
        output_sha256 = _sha256_file(output)
        manifest: dict[str, Any] = {
            "schema_version": PRETEST_M5_SOURCE_SCHEMA_VERSION,
            "instrument": XAU_INSTRUMENT,
            "timeframe": "M5",
            "timestamp_semantics": "bar_start_utc",
            "test_boundary_utc": TEST_BOUNDARY_UTC,
            "test_accessed": False,
            "source_native_root": str(native_root),
            "source_native_manifest_path": str(native_manifest_path),
            "source_native_manifest_sha256": native_manifest_sha256,
            "source_native_manifest_payload_sha256": native_manifest[
                "manifest_payload_sha256"
            ],
            "source_requested_start_utc": native_manifest["requested_start_utc"],
            "source_requested_end_utc_exclusive": native_manifest[
                "requested_end_utc_exclusive"
            ],
            "output_parquet": str(destination / OUTPUT_NAME),
            "output_parquet_sha256": output_sha256,
            "row_count": row_count,
            "time_min_utc": first_time.isoformat(),
            "time_max_utc": last_time.isoformat(),
            "producer_git_commit": initial_commit,
            "producer_repository_clean": True,
        }
        manifest["manifest_payload_sha256"] = _canonical_sha256(manifest)
        _write_json_fsync(stage / "manifest.json", manifest)
        _fsync_directory(stage)
        if _require_clean_repository(repository) != initial_commit:
            raise RuntimeError("DIRECT_M5_PRETEST_REPOSITORY_CHANGED_BEFORE_PUBLISH")
        if _sha256_file(native_manifest_path) != native_manifest_sha256:
            raise RuntimeError("DIRECT_M5_PRETEST_NATIVE_MANIFEST_CHANGED_BEFORE_PUBLISH")
        publish_bundle_directory_noreplace(stage, destination)
        published = True
        return manifest
    finally:
        if not published and stage.exists() and stage.parent == destination.parent:
            shutil.rmtree(stage)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-m5-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize_direct_m5_pretest_source(**vars(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
