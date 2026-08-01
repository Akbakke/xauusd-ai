#!/usr/bin/env python3
"""Publish one immutable native OANDA XAU_USD M1 or M5 source bundle.

This is the sole canonical native-M1/M5 producer. Bootstrap requests a complete
interval. Successor publication CAS-binds an immutable parent, reuses its
verified raw chunks, refetches only one bounded overlap plus the new tail, and
requires byte-exact overlap before appending. Both modes accept only OANDA
``complete=true`` MBA candles, preserve every normalized source response,
re-derive every parquet row, validate the hidden bundle, and publish without
replacement. There is no mutable append, alternate provider, resampling,
synthesis, repair fallback, historical rewrite, or empty-success path.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from gx1.contracts.entry_model_native_bundle_commit_v1 import (
    publish_bundle_directory_noreplace,
)
from gx1.contracts.xau_tape_provenance_v1 import (
    CANONICAL_NATIVE_CLOSURE_CONTRACT,
    CANONICAL_NATIVE_PRODUCER_OWNER,
    CANONICAL_NATIVE_PRODUCER_SOURCE_FILES,
    CANONICAL_NATIVE_REQUEST_INTERVAL_SEMANTICS,
    CANONICAL_NATIVE_REQUIRED_COLUMNS,
    CANONICAL_NATIVE_SOURCE_CHUNK_SCHEMA,
    CANONICAL_NATIVE_SOURCE_RESPONSE_ENCODING,
    CANONICAL_NATIVE_SOURCE_SCHEMA,
    CANONICAL_NATIVE_SUCCESSOR_MODE,
    CANONICAL_NATIVE_SUCCESSOR_SOURCE_SCHEMA,
    XAU_INSTRUMENT,
    canonical_native_parent_binding_v1,
    canonical_json_sha256,
    canonical_native_frame_from_oanda_response,
    canonical_native_rows_bytes,
    canonical_xau_source_descriptor_v1,
    native_timeframe_policy,
    sha256_file,
    validate_canonical_native_frame,
    validate_canonical_native_source_bundle,
)
from gx1.execution.oanda_client import OandaClient, OandaClientConfig
from gx1.execution.oanda_credentials import load_oanda_credentials
from gx1.utils.env_loader import load_dotenv_if_present
from gx1_guards.gates import require_retrain_vedtak


log = logging.getLogger(__name__)
INSTRUMENT = XAU_INSTRUMENT
SOURCE_ENDPOINT = f"/instruments/{INSTRUMENT}/candles"
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_PARQUET_SCHEMA = pa.schema(
    [
        pa.field("time", pa.timestamp("ns", tz="UTC"), nullable=False),
        *[
            pa.field(name, pa.float64(), nullable=False)
            for name in CANONICAL_NATIVE_REQUIRED_COLUMNS
            if name not in {"time", "volume"}
        ],
        pa.field("volume", pa.int64(), nullable=False),
    ]
)


def _canonical_json_bytes(value: Any, *, pretty: bool) -> bytes:
    if pretty:
        return (
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _write_bytes_fsync(path: Path, raw: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o644)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError(f"short write: {path}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _utc_native(
    raw: Any,
    *,
    timeframe: str,
    label: str,
) -> pd.Timestamp:
    normalized, policy = native_timeframe_policy(timeframe)
    try:
        value = pd.Timestamp(pd.to_datetime(raw, utc=True, errors="raise"))
    except Exception as exc:
        raise RuntimeError(
            f"[NATIVE_{normalized}_{label}_INVALID] {raw!r}"
        ) from exc
    if pd.isna(value):
        raise RuntimeError(f"[NATIVE_{normalized}_{label}_INVALID] {raw!r}")
    if value.value % (policy["bar_seconds"] * 1_000_000_000) != 0:
        raise RuntimeError(
            f"[NATIVE_{normalized}_{label}_NOT_{normalized}_ALIGNED] {value}"
        )
    return value


def _request_timestamp(value: pd.Timestamp) -> str:
    return value.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")


def _require_clean_repository(repo_root: Path, *, timeframe: str) -> str:
    normalized, _policy = native_timeframe_policy(timeframe)
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"[NATIVE_{normalized}_REPOSITORY_IDENTITY_UNAVAILABLE]"
        ) from exc
    if _GIT_COMMIT_RE.fullmatch(commit) is None:
        raise RuntimeError(f"[NATIVE_{normalized}_REPOSITORY_COMMIT_INVALID]")
    if status:
        raise RuntimeError(
            f"[NATIVE_{normalized}_REPOSITORY_NOT_CLEAN] "
            "commit producer changes first"
        )
    return commit


def _hash_canonical_parent_file(path: Path, *, label: str) -> str:
    """Hash one immutable parent file without admitting symlinked bytes."""

    path = Path(path)
    if (
        path.is_symlink()
        or not path.is_file()
        or path.resolve(strict=True) != path
    ):
        raise RuntimeError(f"[NATIVE_PARENT_{label}_PATH_INVALID] {path}")
    return sha256_file(path)


def _load_parent_descriptor_cas(
    parent_root: Path,
    *,
    timeframe: str,
    expected_manifest_sha256: str,
    vedtak: str,
) -> dict[str, Any]:
    """Admit an existing parent by manifest and complete byte-CAS evidence.

    The parent was already semantically admitted when it was published.  A
    successor only needs to prove that the exact manifest, producer snapshot,
    source chunks and year partitions are still the admitted bytes.  Decoding
    every historical OANDA response again would add latency without adding
    identity authority; the child is fully semantically validated before its
    own publication.
    """

    normalized, policy = native_timeframe_policy(timeframe)
    root = Path(parent_root)
    manifest_path = root / "MANIFEST.json"
    observed_manifest_sha = _hash_canonical_parent_file(
        manifest_path,
        label=f"{normalized}_MANIFEST",
    )
    if observed_manifest_sha != expected_manifest_sha256:
        raise RuntimeError(
            f"[NATIVE_{normalized}_SUCCESSOR_PARENT_MANIFEST_CAS_MISMATCH]"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(
            f"[NATIVE_{normalized}_SUCCESSOR_PARENT_MANIFEST_INVALID]"
        ) from exc
    if not isinstance(manifest, dict):
        raise RuntimeError(
            f"[NATIVE_{normalized}_SUCCESSOR_PARENT_MANIFEST_INVALID]"
        )

    schema = manifest.get("schema_version")
    if schema not in {
        CANONICAL_NATIVE_SOURCE_SCHEMA,
        CANONICAL_NATIVE_SUCCESSOR_SOURCE_SCHEMA,
    }:
        raise RuntimeError(
            f"[NATIVE_{normalized}_SUCCESSOR_PARENT_SCHEMA_INVALID]"
        )
    exact = {
        "schema_version": schema,
        "producer_owner": CANONICAL_NATIVE_PRODUCER_OWNER,
        "source_kind": "oanda_native_mba_candles",
        "source_endpoint": SOURCE_ENDPOINT,
        "source_granularity": normalized,
        "prices": "MBA",
        "timestamp_semantics": "bar_start_utc",
        "bar_duration_seconds": policy["bar_seconds"],
        "decision_available_offset_seconds": policy["bar_seconds"],
        "completion_field": "complete",
        "completion_value": True,
        "market_closure_contract": CANONICAL_NATIVE_CLOSURE_CONTRACT,
        "request_interval_semantics": (
            CANONICAL_NATIVE_REQUEST_INTERVAL_SEMANTICS
        ),
        "source_response_encoding": CANONICAL_NATIVE_SOURCE_RESPONSE_ENCODING,
        "source_chunk_schema": CANONICAL_NATIVE_SOURCE_CHUNK_SCHEMA,
        "producer_repository_clean": True,
    }
    for name, expected in exact.items():
        if manifest.get(name) != expected:
            raise RuntimeError(
                f"[NATIVE_{normalized}_SUCCESSOR_PARENT_POLICY_MISMATCH]"
            )
    if (
        manifest.get("instrument") != INSTRUMENT
        or manifest.get("timeframe") != normalized
        or manifest.get("out_root") != str(root)
        or manifest.get("explicit_vedtak_id") != vedtak
    ):
        raise RuntimeError(
            f"[NATIVE_{normalized}_SUCCESSOR_PARENT_IDENTITY_MISMATCH]"
        )
    environment = manifest.get("source_environment")
    base_url = manifest.get("source_base_url")
    expected_base_url = {
        "practice": "https://api-fxpractice.oanda.com/v3",
        "live": "https://api-fxtrade.oanda.com/v3",
    }.get(environment)
    if expected_base_url is None or base_url != expected_base_url:
        raise RuntimeError(
            f"[NATIVE_{normalized}_SUCCESSOR_PARENT_ENVIRONMENT_INVALID]"
        )
    start = _utc_native(
        manifest.get("requested_start_utc"),
        timeframe=normalized,
        label="SUCCESSOR_PARENT_START",
    )
    end = _utc_native(
        manifest.get("requested_end_utc_exclusive"),
        timeframe=normalized,
        label="SUCCESSOR_PARENT_END",
    )
    if end <= start or manifest.get("request_chunk_days") != policy[
        "request_chunk_days"
    ]:
        raise RuntimeError(
            f"[NATIVE_{normalized}_SUCCESSOR_PARENT_INTERVAL_INVALID]"
        )

    def require_sha(value: object, label: str) -> str:
        digest = str(value or "")
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise RuntimeError(
                f"[NATIVE_{normalized}_SUCCESSOR_PARENT_{label}_SHA256_INVALID]"
            )
        return digest

    payload_sha = require_sha(
        manifest.get("manifest_payload_sha256"),
        "MANIFEST_PAYLOAD",
    )
    without_payload = dict(manifest)
    without_payload.pop("manifest_payload_sha256", None)
    if canonical_json_sha256(without_payload) != payload_sha:
        raise RuntimeError(
            f"[NATIVE_{normalized}_SUCCESSOR_PARENT_MANIFEST_PAYLOAD_MISMATCH]"
        )

    source_chunks = manifest.get("source_chunks")
    if not isinstance(source_chunks, list) or not source_chunks:
        raise RuntimeError(
            f"[NATIVE_{normalized}_SUCCESSOR_PARENT_CHUNKS_INVALID]"
        )
    if canonical_json_sha256(source_chunks) != require_sha(
        manifest.get("source_chunks_sha256"),
        "SOURCE_CHUNKS",
    ):
        raise RuntimeError(
            f"[NATIVE_{normalized}_SUCCESSOR_PARENT_CHUNKS_HASH_MISMATCH]"
        )
    expected_chunk_paths: list[Path] = []
    for sequence, metadata in enumerate(source_chunks):
        if not isinstance(metadata, dict):
            raise RuntimeError(
                f"[NATIVE_{normalized}_SUCCESSOR_PARENT_CHUNK_METADATA_INVALID]"
            )
        relative = Path(str(metadata.get("relative_path") or ""))
        expected_relative = (
            Path("source_chunks") / f"chunk-{sequence:06d}.json.gz"
        )
        size = metadata.get("size_bytes")
        if (
            relative != expected_relative
            or relative.is_absolute()
            or ".." in relative.parts
            or "." in relative.parts
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size <= 0
        ):
            raise RuntimeError(
                f"[NATIVE_{normalized}_SUCCESSOR_PARENT_CHUNK_METADATA_INVALID]"
            )
        expected = require_sha(metadata.get("sha256"), "CHUNK")
        chunk = root / relative
        if (
            chunk.stat().st_size != size
            or _hash_canonical_parent_file(
                chunk,
                label=f"{normalized}_CHUNK_{sequence:06d}",
            )
            != expected
        ):
            raise RuntimeError(
                f"[NATIVE_{normalized}_SUCCESSOR_PARENT_CHUNK_CAS_MISMATCH]"
            )
        expected_chunk_paths.append(chunk)
    actual_chunk_paths = sorted(
        path
        for path in (root / "source_chunks").glob("chunk-*.json.gz")
        if path.is_file()
    )
    if actual_chunk_paths != sorted(expected_chunk_paths):
        raise RuntimeError(
            f"[NATIVE_{normalized}_SUCCESSOR_PARENT_CHUNK_FILESYSTEM_MISMATCH]"
        )

    year_sha256 = manifest.get("year_sha256")
    year_rows = manifest.get("year_rows")
    if (
        not isinstance(year_sha256, dict)
        or not year_sha256
        or not isinstance(year_rows, dict)
        or set(year_sha256) != set(year_rows)
    ):
        raise RuntimeError(
            f"[NATIVE_{normalized}_SUCCESSOR_PARENT_YEARS_INVALID]"
        )
    for key, expected_value in sorted(year_sha256.items()):
        if (
            not isinstance(key, str)
            or re.fullmatch(r"year=[0-9]{4}", key) is None
            or not isinstance(year_rows[key], int)
            or isinstance(year_rows[key], bool)
            or year_rows[key] <= 0
        ):
            raise RuntimeError(
                f"[NATIVE_{normalized}_SUCCESSOR_PARENT_YEARS_INVALID]"
            )
        part = root / key / "part-000.parquet"
        if _hash_canonical_parent_file(
            part,
            label=f"{normalized}_{key}",
        ) != require_sha(expected_value, "YEAR"):
            raise RuntimeError(
                f"[NATIVE_{normalized}_SUCCESSOR_PARENT_YEAR_CAS_MISMATCH]"
            )

    descriptor: dict[str, Any] = {
        **exact,
        "root": str(root),
        "manifest_path": str(manifest_path),
        "manifest_sha256": observed_manifest_sha,
        "instrument": INSTRUMENT,
        "instrument_observed": str(manifest.get("instrument")),
        "timeframe": normalized,
        "explicit_vedtak_id": str(manifest["explicit_vedtak_id"]),
        "source_environment": str(environment),
        "source_base_url": str(base_url),
        "schema_required_cols": list(CANONICAL_NATIVE_REQUIRED_COLUMNS),
        "schema_optional_cols": [],
        "request_chunk_days": policy["request_chunk_days"],
        "requested_start_utc": start.isoformat(),
        "requested_end_utc_exclusive": end.isoformat(),
        "row_count": int(manifest["row_count"]),
        "time_min_utc": str(manifest["time_min_utc"]),
        "time_max_utc": str(manifest["time_max_utc"]),
        "canonical_rows_sha256": require_sha(
            manifest.get("canonical_rows_sha256"),
            "CANONICAL_ROWS",
        ),
        "source_chunks_sha256": require_sha(
            manifest.get("source_chunks_sha256"),
            "SOURCE_CHUNKS",
        ),
        "producer_git_commit": str(manifest["producer_git_commit"]),
        "producer_source_inventory_sha256": require_sha(
            manifest.get("producer_source_inventory_sha256"),
            "PRODUCER_SOURCE_INVENTORY",
        ),
        "manifest_payload_sha256": payload_sha,
        "year_sha256": dict(sorted(year_sha256.items())),
        "year_rows": dict(sorted(year_rows.items())),
    }
    if schema == CANONICAL_NATIVE_SUCCESSOR_SOURCE_SCHEMA:
        descriptor.update(
            {
                "publication_mode": manifest["publication_mode"],
                "parent_source": json.loads(
                    json.dumps(manifest["parent_source"], sort_keys=True)
                ),
                "successor_append": json.loads(
                    json.dumps(manifest["successor_append"], sort_keys=True)
                ),
            }
        )
    return descriptor


def _snapshot_producer_sources(
    *,
    timeframe: str,
    repo_root: Path,
    stage: Path,
) -> list[dict[str, Any]]:
    normalized, _policy = native_timeframe_policy(timeframe)
    inventory: list[dict[str, Any]] = []
    for relative_text in CANONICAL_NATIVE_PRODUCER_SOURCE_FILES:
        relative = Path(relative_text)
        source = repo_root / relative
        if source.is_symlink() or not source.is_file():
            raise RuntimeError(
                f"[NATIVE_{normalized}_PRODUCER_SOURCE_INVALID] {relative_text}"
            )
        raw = source.read_bytes()
        destination = stage / "producer_source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        _write_bytes_fsync(destination, raw)
        inventory.append(
            {
                "repo_relative_path": relative_text,
                "snapshot_relative_path": str(
                    Path("producer_source") / relative
                ),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw),
            }
        )
    return inventory


def _verify_producer_sources_unchanged(
    *,
    timeframe: str,
    repo_root: Path,
    inventory: list[dict[str, Any]],
) -> None:
    normalized, _policy = native_timeframe_policy(timeframe)
    for item in inventory:
        source = repo_root / str(item["repo_relative_path"])
        if (
            source.is_symlink()
            or not source.is_file()
            or source.stat().st_size != item["size_bytes"]
            or sha256_file(source) != item["sha256"]
        ):
            raise RuntimeError(
                f"[NATIVE_{normalized}_PRODUCER_SOURCE_CHANGED_DURING_RUN] "
                f"{item['repo_relative_path']}"
            )


def _parquet_table(frame: pd.DataFrame) -> pa.Table:
    return pa.Table.from_pandas(
        frame.loc[:, list(CANONICAL_NATIVE_REQUIRED_COLUMNS)],
        schema=_PARQUET_SCHEMA,
        preserve_index=False,
        safe=True,
    )


def _year_parquet_writer(path: Path) -> pq.ParquetWriter:
    return pq.ParquetWriter(
        path,
        _PARQUET_SCHEMA,
        compression="zstd",
        use_dictionary=False,
        write_statistics=True,
        data_page_version="2.0",
    )


def _copy_file_fsync(source: Path, destination: Path) -> None:
    """Copy one immutable file without following links or replacing a target."""

    if source.is_symlink() or not source.is_file():
        raise RuntimeError(f"NATIVE_SUCCESSOR_COPY_SOURCE_INVALID: {source}")
    source_stat = source.stat()
    source_before = (
        source_stat.st_dev,
        source_stat.st_ino,
        source_stat.st_size,
        source_stat.st_mtime_ns,
        source_stat.st_ctime_ns,
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(destination, flags, 0o644)
    try:
        with source.open("rb") as input_handle:
            while True:
                chunk = input_handle.read(1024 * 1024)
                if not chunk:
                    break
                view = memoryview(chunk)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise OSError(f"short copy write: {destination}")
                    view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    source_stat = source.stat()
    source_after = (
        source_stat.st_dev,
        source_stat.st_ino,
        source_stat.st_size,
        source_stat.st_mtime_ns,
        source_stat.st_ctime_ns,
    )
    if source_after != source_before or sha256_file(source) != sha256_file(destination):
        raise RuntimeError(
            f"NATIVE_SUCCESSOR_COPY_IDENTITY_MISMATCH: {source}"
        )


def _load_parent_year(
    parent_root: Path,
    *,
    key: str,
    expected_sha256: str,
    timeframe: str,
) -> pd.DataFrame:
    source = parent_root / key / "part-000.parquet"
    if (
        source.is_symlink()
        or not source.is_file()
        or sha256_file(source) != expected_sha256
    ):
        raise RuntimeError(
            f"[NATIVE_{timeframe}_SUCCESSOR_PARENT_YEAR_CAS_MISMATCH] {key}"
        )
    frame = validate_canonical_native_frame(
        pd.read_parquet(source),
        timeframe=timeframe,
        label=f"SUCCESSOR_PARENT_{key}",
    )
    if sha256_file(source) != expected_sha256:
        raise RuntimeError(
            f"[NATIVE_{timeframe}_SUCCESSOR_PARENT_YEAR_CHANGED] {key}"
        )
    return frame


def _write_year_frame(path: Path, frame: pd.DataFrame) -> None:
    writer = _year_parquet_writer(path)
    try:
        writer.write_table(_parquet_table(frame.reset_index(drop=True)))
    finally:
        writer.close()
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _source_chunk(
    *,
    client: OandaClient,
    timeframe: str,
    stage: Path,
    sequence: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    normalized, _policy = native_timeframe_policy(timeframe)
    request = {
        "instrument": INSTRUMENT,
        "from": _request_timestamp(start),
        "to": _request_timestamp(end),
        "granularity": normalized,
        "price": "MBA",
    }
    response = client._request(
        "GET",
        SOURCE_ENDPOINT,
        params={
            "from": request["from"],
            "to": request["to"],
            "granularity": request["granularity"],
            "price": request["price"],
        },
    )
    if not isinstance(response, Mapping):
        raise RuntimeError(
            f"[NATIVE_{normalized}_OANDA_RESPONSE_OBJECT_REQUIRED]"
        )
    payload = {
        "schema_version": CANONICAL_NATIVE_SOURCE_CHUNK_SCHEMA,
        "request": request,
        "response": dict(response),
    }
    encoded = gzip.compress(
        _canonical_json_bytes(payload, pretty=False),
        compresslevel=9,
        mtime=0,
    )
    relative = Path("source_chunks") / f"chunk-{sequence:06d}.json.gz"
    destination = stage / relative
    _write_bytes_fsync(destination, encoded)
    frame, stats = canonical_native_frame_from_oanda_response(
        response,
        timeframe=normalized,
        request_start=start,
        request_end=end,
    )
    metadata = {
        "sequence": sequence,
        "request_from_utc": start.isoformat(),
        "request_to_utc_exclusive": end.isoformat(),
        "relative_path": str(relative),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "size_bytes": len(encoded),
        **stats,
    }
    return frame, metadata


def materialize_native_xau_snapshot(
    *,
    client: OandaClient,
    timeframe: str,
    vedtak_id: str,
    start_utc: Any,
    end_utc: Any,
    out_root: Path | str,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Materialize and atomically publish one complete requested interval."""

    normalized, policy = native_timeframe_policy(timeframe)
    chunk_days = policy["request_chunk_days"]
    vedtak = require_retrain_vedtak(vedtak_id)
    start = _utc_native(
        start_utc,
        timeframe=normalized,
        label="START_UTC",
    )
    end = _utc_native(
        end_utc,
        timeframe=normalized,
        label="END_UTC",
    )
    if end <= start:
        raise RuntimeError(f"[NATIVE_{normalized}_INTERVAL_INVALID]")
    latest_safe_end = pd.Timestamp.now(tz="UTC").floor(
        f"{policy['bar_seconds']}s"
    )
    if end > latest_safe_end:
        raise RuntimeError(
            f"[NATIVE_{normalized}_END_NOT_COMPLETE] "
            f"requested={end} latest_safe_exclusive_end={latest_safe_end}"
        )
    output_arg = Path(out_root).expanduser()
    if not output_arg.is_absolute():
        raise RuntimeError(
            f"[NATIVE_{normalized}_OUTPUT_NOT_ABSOLUTE] {output_arg}"
        )
    if output_arg.exists() or output_arg.is_symlink():
        raise RuntimeError(
            f"[NATIVE_{normalized}_IMMUTABLE_OUTPUT_EXISTS] {output_arg}"
        )
    if (
        output_arg.parent.is_symlink()
        or not output_arg.parent.is_dir()
        or output_arg.parent.resolve() != output_arg.parent
        or output_arg.resolve(strict=False) != output_arg
    ):
        raise RuntimeError(
            f"[NATIVE_{normalized}_OUTPUT_PARENT_INVALID] {output_arg.parent}"
        )

    repository = (
        Path(__file__).resolve().parents[2]
        if repo_root is None
        else Path(repo_root).expanduser()
    )
    if (
        not repository.is_absolute()
        or repository.is_symlink()
        or not repository.is_dir()
        or repository.resolve() != repository
    ):
        raise RuntimeError(
            f"[NATIVE_{normalized}_REPOSITORY_ROOT_INVALID] {repository}"
        )
    if repository == output_arg or repository in output_arg.parents:
        raise RuntimeError(
            f"[NATIVE_{normalized}_OUTPUT_INSIDE_REPOSITORY_FORBIDDEN]"
        )
    initial_commit = _require_clean_repository(
        repository,
        timeframe=normalized,
    )

    environment = str(getattr(client, "env", "") or "")
    base_url = str(getattr(client, "base_url", "") or "")
    if environment not in {"practice", "live"}:
        raise RuntimeError(
            f"[NATIVE_{normalized}_OANDA_ENVIRONMENT_INVALID]"
        )
    expected_base_url = {
        "practice": "https://api-fxpractice.oanda.com/v3",
        "live": "https://api-fxtrade.oanda.com/v3",
    }[environment]
    if base_url != expected_base_url:
        raise RuntimeError(f"[NATIVE_{normalized}_OANDA_BASE_URL_INVALID]")

    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{output_arg.name}.staging.",
            dir=str(output_arg.parent),
        )
    )
    try:
        (stage / "source_chunks").mkdir()
        producer_sources = _snapshot_producer_sources(
            timeframe=normalized,
            repo_root=repository,
            stage=stage,
        )
        source_chunks: list[dict[str, Any]] = []
        source_digest = hashlib.sha256()
        complete_rows = 0
        complete_time_min: str | None = None
        complete_time_max: str | None = None
        year_writers: dict[str, pq.ParquetWriter] = {}
        year_destinations: dict[str, Path] = {}
        year_rows: dict[str, int] = {}
        year_time_bounds: dict[str, dict[str, str]] = {}
        cursor = start
        previous_complete: pd.Timestamp | None = None
        sequence = 0
        try:
            while cursor < end:
                chunk_end = min(cursor + pd.Timedelta(days=chunk_days), end)
                frame, metadata = _source_chunk(
                    client=client,
                    timeframe=normalized,
                    stage=stage,
                    sequence=sequence,
                    start=cursor,
                    end=chunk_end,
                )
                if not frame.empty:
                    first = pd.Timestamp(frame["time"].iloc[0])
                    last = pd.Timestamp(frame["time"].iloc[-1])
                    if previous_complete is not None and first <= previous_complete:
                        raise RuntimeError(
                            f"[NATIVE_{normalized}_CROSS_CHUNK_TIME_CONFLICT] "
                            f"previous={previous_complete} current={first}"
                        )
                    if complete_time_min is None:
                        complete_time_min = first.isoformat()
                    complete_time_max = last.isoformat()
                    previous_complete = last
                    complete_rows += len(frame)
                    source_digest.update(
                        canonical_native_rows_bytes(
                            frame,
                            timeframe=normalized,
                        )
                    )
                    for year, year_frame in frame.groupby(
                        frame["time"].dt.year,
                        sort=True,
                    ):
                        key = f"year={int(year)}"
                        if key not in year_writers:
                            directory = stage / key
                            directory.mkdir()
                            destination = directory / "part-000.parquet"
                            year_destinations[key] = destination
                            year_writers[key] = _year_parquet_writer(destination)
                            year_rows[key] = 0
                            year_time_bounds[key] = {
                                "time_min_utc": pd.Timestamp(
                                    year_frame["time"].iloc[0]
                                ).isoformat(),
                                "time_max_utc": pd.Timestamp(
                                    year_frame["time"].iloc[-1]
                                ).isoformat(),
                            }
                        year_writers[key].write_table(
                            _parquet_table(year_frame.reset_index(drop=True))
                        )
                        year_rows[key] += len(year_frame)
                        year_time_bounds[key]["time_max_utc"] = pd.Timestamp(
                            year_frame["time"].iloc[-1]
                        ).isoformat()
                source_chunks.append(metadata)
                cursor = chunk_end
                sequence += 1
        finally:
            for writer in year_writers.values():
                writer.close()

        if complete_rows <= 0 or complete_time_min is None or complete_time_max is None:
            raise RuntimeError(
                f"[NATIVE_{normalized}_COMPLETE_SOURCE_EMPTY]"
            )
        canonical_rows_sha = source_digest.hexdigest()
        year_sha256: dict[str, str] = {}
        for key, destination in sorted(year_destinations.items()):
            descriptor = os.open(destination, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            year_sha256[key] = sha256_file(destination)
        manifest: dict[str, Any] = {
            "schema_version": CANONICAL_NATIVE_SOURCE_SCHEMA,
            "producer_owner": CANONICAL_NATIVE_PRODUCER_OWNER,
            "instrument": INSTRUMENT,
            "timeframe": normalized,
            "out_root": str(output_arg),
            "explicit_vedtak_id": vedtak,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_kind": "oanda_native_mba_candles",
            "source_environment": environment,
            "source_base_url": base_url,
            "source_endpoint": SOURCE_ENDPOINT,
            "source_granularity": normalized,
            "prices": "MBA",
            "timestamp_semantics": "bar_start_utc",
            "bar_duration_seconds": policy["bar_seconds"],
            "decision_available_offset_seconds": policy["bar_seconds"],
            "completion_field": "complete",
            "completion_value": True,
            "market_closure_contract": CANONICAL_NATIVE_CLOSURE_CONTRACT,
            "request_interval_semantics": CANONICAL_NATIVE_REQUEST_INTERVAL_SEMANTICS,
            "requested_start_utc": start.isoformat(),
            "requested_end_utc_exclusive": end.isoformat(),
            "request_chunk_days": chunk_days,
            "source_response_encoding": CANONICAL_NATIVE_SOURCE_RESPONSE_ENCODING,
            "source_chunk_schema": CANONICAL_NATIVE_SOURCE_CHUNK_SCHEMA,
            "source_chunks": source_chunks,
            "source_chunks_sha256": canonical_json_sha256(source_chunks),
            "producer_git_commit": initial_commit,
            "producer_repository_clean": True,
            "producer_source_files": producer_sources,
            "producer_source_inventory_sha256": canonical_json_sha256(
                producer_sources
            ),
            "runtime_versions": {
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "pyarrow": pa.__version__,
                "python": sys.version.split()[0],
            },
            "schema_required_cols": list(CANONICAL_NATIVE_REQUIRED_COLUMNS),
            "schema_optional_cols": [],
            "row_count": complete_rows,
            "time_min_utc": complete_time_min,
            "time_max_utc": complete_time_max,
            "canonical_rows_sha256": canonical_rows_sha,
            "year_sha256": year_sha256,
            "year_rows": year_rows,
            "year_time_bounds": year_time_bounds,
        }
        manifest["manifest_payload_sha256"] = canonical_json_sha256(manifest)
        _write_bytes_fsync(
            stage / "MANIFEST.json",
            _canonical_json_bytes(manifest, pretty=True),
        )
        for directory, _, _ in os.walk(stage, topdown=False):
            _fsync_directory(Path(directory))

        validate_canonical_native_source_bundle(
            stage,
            timeframe=normalized,
            expected_declared_root=output_arg,
        )
        final_commit = _require_clean_repository(
            repository,
            timeframe=normalized,
        )
        if final_commit != initial_commit:
            raise RuntimeError(
                f"[NATIVE_{normalized}_REPOSITORY_COMMIT_CHANGED_BEFORE_PUBLISH]"
            )
        _verify_producer_sources_unchanged(
            timeframe=normalized,
            repo_root=repository,
            inventory=producer_sources,
        )
        publish_bundle_directory_noreplace(stage, output_arg)
        return manifest
    except Exception:
        if (
            stage.exists()
            and stage.parent == output_arg.parent
            and stage.name.startswith(f".{output_arg.name}.staging.")
        ):
            shutil.rmtree(stage)
        raise


def materialize_native_xau_successor(
    *,
    client: OandaClient,
    timeframe: str,
    vedtak_id: str,
    end_utc: Any,
    out_root: Path | str,
    parent_root: Path | str,
    expected_parent_manifest_sha256: str,
    start_utc: Any | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Publish one strict child while fetching only the bounded overlap and tail."""

    normalized, policy = native_timeframe_policy(timeframe)
    chunk_days = policy["request_chunk_days"]
    vedtak = require_retrain_vedtak(vedtak_id)
    expected_parent_sha = str(expected_parent_manifest_sha256 or "").strip()
    if re.fullmatch(r"[0-9a-f]{64}", expected_parent_sha) is None:
        raise RuntimeError(
            f"[NATIVE_{normalized}_SUCCESSOR_PARENT_MANIFEST_SHA256_INVALID]"
        )
    parent_arg = Path(parent_root).expanduser()
    if (
        not parent_arg.is_absolute()
        or parent_arg.is_symlink()
        or not parent_arg.is_dir()
        or parent_arg.resolve() != parent_arg
    ):
        raise RuntimeError(
            f"[NATIVE_{normalized}_SUCCESSOR_PARENT_ROOT_INVALID] {parent_arg}"
        )
    parent_descriptor = _load_parent_descriptor_cas(
        parent_arg,
        timeframe=normalized,
        expected_manifest_sha256=expected_parent_sha,
        vedtak=vedtak,
    )
    if parent_descriptor["explicit_vedtak_id"] != vedtak:
        raise RuntimeError(
            f"[NATIVE_{normalized}_SUCCESSOR_PARENT_VEDTAK_MISMATCH]"
        )
    parent_start = _utc_native(
        parent_descriptor["requested_start_utc"],
        timeframe=normalized,
        label="SUCCESSOR_PARENT_START",
    )
    parent_end = _utc_native(
        parent_descriptor["requested_end_utc_exclusive"],
        timeframe=normalized,
        label="SUCCESSOR_PARENT_END",
    )
    if start_utc is not None:
        offered_start = _utc_native(
            start_utc,
            timeframe=normalized,
            label="START_UTC",
        )
        if offered_start != parent_start:
            raise RuntimeError(
                f"[NATIVE_{normalized}_SUCCESSOR_START_MISMATCH]"
            )
    end = _utc_native(
        end_utc,
        timeframe=normalized,
        label="END_UTC",
    )
    if end <= parent_end:
        raise RuntimeError(
            f"[NATIVE_{normalized}_SUCCESSOR_INTERVAL_NOT_ADVANCING]"
        )
    latest_safe_end = pd.Timestamp.now(tz="UTC").floor(
        f"{policy['bar_seconds']}s"
    )
    if end > latest_safe_end:
        raise RuntimeError(
            f"[NATIVE_{normalized}_END_NOT_COMPLETE] "
            f"requested={end} latest_safe_exclusive_end={latest_safe_end}"
        )

    output_arg = Path(out_root).expanduser()
    if not output_arg.is_absolute():
        raise RuntimeError(
            f"[NATIVE_{normalized}_OUTPUT_NOT_ABSOLUTE] {output_arg}"
        )
    if output_arg.exists() or output_arg.is_symlink():
        raise RuntimeError(
            f"[NATIVE_{normalized}_IMMUTABLE_OUTPUT_EXISTS] {output_arg}"
        )
    if (
        output_arg.parent.is_symlink()
        or not output_arg.parent.is_dir()
        or output_arg.parent.resolve() != output_arg.parent
        or output_arg.resolve(strict=False) != output_arg
        or output_arg == parent_arg
        or output_arg in parent_arg.parents
        or parent_arg in output_arg.parents
    ):
        raise RuntimeError(
            f"[NATIVE_{normalized}_OUTPUT_PARENT_INVALID] {output_arg.parent}"
        )

    repository = (
        Path(__file__).resolve().parents[2]
        if repo_root is None
        else Path(repo_root).expanduser()
    )
    if (
        not repository.is_absolute()
        or repository.is_symlink()
        or not repository.is_dir()
        or repository.resolve() != repository
    ):
        raise RuntimeError(
            f"[NATIVE_{normalized}_REPOSITORY_ROOT_INVALID] {repository}"
        )
    if repository == output_arg or repository in output_arg.parents:
        raise RuntimeError(
            f"[NATIVE_{normalized}_OUTPUT_INSIDE_REPOSITORY_FORBIDDEN]"
        )
    initial_commit = _require_clean_repository(
        repository,
        timeframe=normalized,
    )

    environment = str(getattr(client, "env", "") or "")
    base_url = str(getattr(client, "base_url", "") or "")
    expected_base_url = {
        "practice": "https://api-fxpractice.oanda.com/v3",
        "live": "https://api-fxtrade.oanda.com/v3",
    }.get(environment)
    if expected_base_url is None:
        raise RuntimeError(
            f"[NATIVE_{normalized}_OANDA_ENVIRONMENT_INVALID]"
        )
    if (
        base_url != expected_base_url
        or environment != parent_descriptor["source_environment"]
        or base_url != parent_descriptor["source_base_url"]
    ):
        raise RuntimeError(
            f"[NATIVE_{normalized}_SUCCESSOR_SOURCE_IDENTITY_MISMATCH]"
        )

    parent_manifest_path = parent_arg / "MANIFEST.json"
    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(parent_manifest, dict)
        or sha256_file(parent_manifest_path) != expected_parent_sha
    ):
        raise RuntimeError(
            f"[NATIVE_{normalized}_SUCCESSOR_PARENT_CHANGED_BEFORE_BUILD]"
        )
    parent_chunks = parent_manifest.get("source_chunks")
    if not isinstance(parent_chunks, list) or not parent_chunks:
        raise RuntimeError(
            f"[NATIVE_{normalized}_SUCCESSOR_PARENT_CHUNKS_INVALID]"
        )
    nonempty_chunk_positions = [
        index
        for index, metadata in enumerate(parent_chunks)
        if isinstance(metadata, dict)
        and isinstance(metadata.get("complete_candles"), int)
        and not isinstance(metadata.get("complete_candles"), bool)
        and metadata["complete_candles"] > 0
    ]
    if not nonempty_chunk_positions:
        raise RuntimeError(
            f"[NATIVE_{normalized}_SUCCESSOR_PARENT_OVERLAP_UNAVAILABLE]"
        )
    reused_chunk_count = nonempty_chunk_positions[-1]
    overlap_start = _utc_native(
        parent_chunks[reused_chunk_count]["request_from_utc"],
        timeframe=normalized,
        label="SUCCESSOR_OVERLAP_START",
    )

    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{output_arg.name}.staging.",
            dir=str(output_arg.parent),
        )
    )
    try:
        (stage / "source_chunks").mkdir()
        producer_sources = _snapshot_producer_sources(
            timeframe=normalized,
            repo_root=repository,
            stage=stage,
        )
        source_chunks: list[dict[str, Any]] = []
        for sequence, metadata in enumerate(parent_chunks[:reused_chunk_count]):
            relative = Path(str(metadata["relative_path"]))
            expected_relative = (
                Path("source_chunks") / f"chunk-{sequence:06d}.json.gz"
            )
            if relative != expected_relative:
                raise RuntimeError(
                    f"[NATIVE_{normalized}_SUCCESSOR_PARENT_CHUNK_PATH_INVALID]"
                )
            source = parent_arg / relative
            if (
                source.is_symlink()
                or not source.is_file()
                or source.stat().st_size != metadata["size_bytes"]
                or sha256_file(source) != metadata["sha256"]
            ):
                raise RuntimeError(
                    f"[NATIVE_{normalized}_SUCCESSOR_PARENT_CHUNK_CAS_MISMATCH]"
                )
            _copy_file_fsync(source, stage / relative)
            source_chunks.append(json.loads(json.dumps(metadata)))

        refetched_frames: list[pd.DataFrame] = []
        cursor = overlap_start
        sequence = reused_chunk_count
        while cursor < end:
            chunk_end = min(cursor + pd.Timedelta(days=chunk_days), end)
            frame, metadata = _source_chunk(
                client=client,
                timeframe=normalized,
                stage=stage,
                sequence=sequence,
                start=cursor,
                end=chunk_end,
            )
            source_chunks.append(metadata)
            if not frame.empty:
                refetched_frames.append(frame)
            cursor = chunk_end
            sequence += 1
        if not refetched_frames:
            raise RuntimeError(
                f"[NATIVE_{normalized}_SUCCESSOR_REFETCH_EMPTY]"
            )
        refetched = validate_canonical_native_frame(
            pd.concat(refetched_frames, ignore_index=True),
            timeframe=normalized,
            label="SUCCESSOR_REFETCH",
        )
        refetched_overlap = refetched.loc[refetched["time"] < parent_end]
        appended = refetched.loc[refetched["time"] >= parent_end]
        if refetched_overlap.empty or appended.empty:
            raise RuntimeError(
                f"[NATIVE_{normalized}_SUCCESSOR_STRICT_APPEND_MISSING]"
            )

        parent_year_hashes = parent_descriptor["year_sha256"]
        parent_overlap_frames: list[pd.DataFrame] = []
        refetched_by_year = {
            int(year): frame.reset_index(drop=True)
            for year, frame in refetched.groupby(
                refetched["time"].dt.year,
                sort=True,
            )
        }
        output_frames: dict[int, pd.DataFrame] = {}
        parent_years = {
            int(str(key).split("=", 1)[1]): key
            for key in parent_year_hashes
        }
        all_years = sorted(set(parent_years) | set(refetched_by_year))
        for year in all_years:
            parent_year: pd.DataFrame | None = None
            if year in parent_years:
                key = parent_years[year]
                parent_year = _load_parent_year(
                    parent_arg,
                    key=key,
                    expected_sha256=parent_year_hashes[key],
                    timeframe=normalized,
                )
                overlap = parent_year.loc[
                    parent_year["time"] >= overlap_start
                ]
                if not overlap.empty:
                    parent_overlap_frames.append(overlap)
                prefix = parent_year.loc[
                    parent_year["time"] < overlap_start
                ]
            else:
                prefix = pd.DataFrame(
                    columns=list(CANONICAL_NATIVE_REQUIRED_COLUMNS)
                )
            suffix = refetched_by_year.get(year)
            pieces = [
                frame
                for frame in (prefix, suffix)
                if frame is not None and not frame.empty
            ]
            if not pieces:
                continue
            output_frames[year] = validate_canonical_native_frame(
                pd.concat(pieces, ignore_index=True),
                timeframe=normalized,
                label=f"SUCCESSOR_OUTPUT_YEAR_{year}",
            )
        if not parent_overlap_frames:
            raise RuntimeError(
                f"[NATIVE_{normalized}_SUCCESSOR_PARENT_OVERLAP_EMPTY]"
            )
        parent_overlap = validate_canonical_native_frame(
            pd.concat(parent_overlap_frames, ignore_index=True),
            timeframe=normalized,
            label="SUCCESSOR_PARENT_OVERLAP",
        )
        parent_overlap_bytes = canonical_native_rows_bytes(
            parent_overlap,
            timeframe=normalized,
        )
        if parent_overlap_bytes != canonical_native_rows_bytes(
            refetched_overlap,
            timeframe=normalized,
        ):
            raise RuntimeError(
                f"[NATIVE_{normalized}_SUCCESSOR_OVERLAP_REWRITE]"
            )
        parent_time_max = pd.Timestamp(parent_descriptor["time_max_utc"])
        if pd.Timestamp(appended["time"].iloc[-1]) <= parent_time_max:
            raise RuntimeError(
                f"[NATIVE_{normalized}_SUCCESSOR_NOT_STRICTLY_ADVANCING]"
            )

        source_digest = hashlib.sha256()
        year_sha256: dict[str, str] = {}
        year_rows: dict[str, int] = {}
        year_time_bounds: dict[str, dict[str, str]] = {}
        complete_rows = 0
        complete_time_min: str | None = None
        complete_time_max: str | None = None
        for year, frame in sorted(output_frames.items()):
            key = f"year={year}"
            directory = stage / key
            directory.mkdir()
            destination = directory / "part-000.parquet"
            parent_year_key = parent_years.get(year)
            if (
                parent_year_key is not None
                and frame["time"].iloc[-1] < overlap_start
            ):
                _copy_file_fsync(
                    parent_arg / parent_year_key / "part-000.parquet",
                    destination,
                )
            else:
                _write_year_frame(destination, frame)
            year_sha256[key] = sha256_file(destination)
            year_rows[key] = len(frame)
            first = pd.Timestamp(frame["time"].iloc[0]).isoformat()
            last = pd.Timestamp(frame["time"].iloc[-1]).isoformat()
            year_time_bounds[key] = {
                "time_min_utc": first,
                "time_max_utc": last,
            }
            if complete_time_min is None:
                complete_time_min = first
            complete_time_max = last
            complete_rows += len(frame)
            source_digest.update(
                canonical_native_rows_bytes(
                    frame,
                    timeframe=normalized,
                )
            )
        if (
            complete_rows <= int(parent_descriptor["row_count"])
            or complete_time_min is None
            or complete_time_max is None
        ):
            raise RuntimeError(
                f"[NATIVE_{normalized}_SUCCESSOR_OUTPUT_NOT_ADVANCING]"
            )

        parent_binding = canonical_native_parent_binding_v1(
            parent_descriptor
        )
        manifest: dict[str, Any] = {
            "schema_version": CANONICAL_NATIVE_SUCCESSOR_SOURCE_SCHEMA,
            "publication_mode": CANONICAL_NATIVE_SUCCESSOR_MODE,
            "parent_source": parent_binding,
            "successor_append": {
                "overlap_start_utc": overlap_start.isoformat(),
                "parent_end_utc_exclusive": parent_end.isoformat(),
                "reused_source_chunks": reused_chunk_count,
                "refetched_source_chunks": (
                    len(source_chunks) - reused_chunk_count
                ),
                "parent_overlap_rows": len(parent_overlap),
                "appended_rows": len(appended),
                "overlap_rows_sha256": hashlib.sha256(
                    parent_overlap_bytes
                ).hexdigest(),
            },
            "producer_owner": CANONICAL_NATIVE_PRODUCER_OWNER,
            "instrument": INSTRUMENT,
            "timeframe": normalized,
            "out_root": str(output_arg),
            "explicit_vedtak_id": vedtak,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_kind": "oanda_native_mba_candles",
            "source_environment": environment,
            "source_base_url": base_url,
            "source_endpoint": SOURCE_ENDPOINT,
            "source_granularity": normalized,
            "prices": "MBA",
            "timestamp_semantics": "bar_start_utc",
            "bar_duration_seconds": policy["bar_seconds"],
            "decision_available_offset_seconds": policy["bar_seconds"],
            "completion_field": "complete",
            "completion_value": True,
            "market_closure_contract": CANONICAL_NATIVE_CLOSURE_CONTRACT,
            "request_interval_semantics": (
                CANONICAL_NATIVE_REQUEST_INTERVAL_SEMANTICS
            ),
            "requested_start_utc": parent_start.isoformat(),
            "requested_end_utc_exclusive": end.isoformat(),
            "request_chunk_days": chunk_days,
            "source_response_encoding": (
                CANONICAL_NATIVE_SOURCE_RESPONSE_ENCODING
            ),
            "source_chunk_schema": CANONICAL_NATIVE_SOURCE_CHUNK_SCHEMA,
            "source_chunks": source_chunks,
            "source_chunks_sha256": canonical_json_sha256(source_chunks),
            "producer_git_commit": initial_commit,
            "producer_repository_clean": True,
            "producer_source_files": producer_sources,
            "producer_source_inventory_sha256": canonical_json_sha256(
                producer_sources
            ),
            "runtime_versions": {
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "pyarrow": pa.__version__,
                "python": sys.version.split()[0],
            },
            "schema_required_cols": list(CANONICAL_NATIVE_REQUIRED_COLUMNS),
            "schema_optional_cols": [],
            "row_count": complete_rows,
            "time_min_utc": complete_time_min,
            "time_max_utc": complete_time_max,
            "canonical_rows_sha256": source_digest.hexdigest(),
            "year_sha256": year_sha256,
            "year_rows": year_rows,
            "year_time_bounds": year_time_bounds,
        }
        manifest["manifest_payload_sha256"] = canonical_json_sha256(manifest)
        _write_bytes_fsync(
            stage / "MANIFEST.json",
            _canonical_json_bytes(manifest, pretty=True),
        )
        for directory, _, _ in os.walk(stage, topdown=False):
            _fsync_directory(Path(directory))

        validate_canonical_native_source_bundle(
            stage,
            timeframe=normalized,
            expected_declared_root=output_arg,
        )
        if _require_clean_repository(
            repository,
            timeframe=normalized,
        ) != initial_commit:
            raise RuntimeError(
                f"[NATIVE_{normalized}_REPOSITORY_COMMIT_CHANGED_BEFORE_PUBLISH]"
            )
        _verify_producer_sources_unchanged(
            timeframe=normalized,
            repo_root=repository,
            inventory=producer_sources,
        )
        try:
            observed_parent = _load_parent_descriptor_cas(
                parent_arg,
                timeframe=normalized,
                expected_manifest_sha256=expected_parent_sha,
                vedtak=vedtak,
            )
        except Exception as exc:
            raise RuntimeError(
                f"[NATIVE_{normalized}_SUCCESSOR_PARENT_CHANGED_BEFORE_PUBLISH]"
            ) from exc
        if (
            observed_parent != parent_descriptor
            or observed_parent["manifest_sha256"] != expected_parent_sha
        ):
            raise RuntimeError(
                f"[NATIVE_{normalized}_SUCCESSOR_PARENT_CHANGED_BEFORE_PUBLISH]"
            )
        publish_bundle_directory_noreplace(stage, output_arg)
        return manifest
    except Exception:
        if (
            stage.exists()
            and stage.parent == output_arg.parent
            and stage.name.startswith(f".{output_arg.name}.staging.")
        ):
            shutil.rmtree(stage)
        raise


def _load_oanda_client() -> OandaClient:
    credentials = load_oanda_credentials(prod_baseline=False)
    return OandaClient(
        OandaClientConfig(
            api_key=credentials.api_token,
            account_id=credentials.account_id,
            env=credentials.env,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--publication-mode",
        choices=("bootstrap", CANONICAL_NATIVE_SUCCESSOR_MODE),
        required=True,
    )
    parser.add_argument(
        "--vedtak",
        required=True,
        help="Explicit decision ID authorizing the external-data publication",
    )
    parser.add_argument(
        "--timeframe",
        choices=tuple(sorted(("M1", "M5"))),
    )
    parser.add_argument("--start-utc")
    parser.add_argument("--end-utc")
    parser.add_argument("--out-root", type=Path)
    parser.add_argument("--parent-root", type=Path)
    parser.add_argument("--expected-parent-manifest-sha256")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    # The authorization gate precedes environment loading, credentials,
    # networking, staging, and every external-data write.
    require_retrain_vedtak(args.vedtak)
    common_missing = [
        flag
        for flag, value in (
            ("--timeframe", args.timeframe),
            ("--end-utc", args.end_utc),
            ("--out-root", args.out_root),
        )
        if value is None
    ]
    if common_missing:
        parser.error(
            "the following arguments are required: "
            + ", ".join(common_missing)
        )
    if args.publication_mode == "bootstrap":
        if args.start_utc is None:
            parser.error("bootstrap requires --start-utc")
        if (
            args.parent_root is not None
            or args.expected_parent_manifest_sha256 is not None
        ):
            parser.error(
                "bootstrap forbids --parent-root and "
                "--expected-parent-manifest-sha256"
            )
    else:
        successor_missing = [
            flag
            for flag, value in (
                ("--parent-root", args.parent_root),
                (
                    "--expected-parent-manifest-sha256",
                    args.expected_parent_manifest_sha256,
                ),
            )
            if value is None
        ]
        if successor_missing:
            parser.error(
                "successor requires " + ", ".join(successor_missing)
            )
        if (
            re.fullmatch(
                r"[0-9a-f]{64}",
                str(args.expected_parent_manifest_sha256),
            )
            is None
        ):
            parser.error(
                "successor requires a lowercase SHA-256 "
                "--expected-parent-manifest-sha256"
            )
    load_dotenv_if_present()
    client = _load_oanda_client()
    if args.publication_mode == "bootstrap":
        report = materialize_native_xau_snapshot(
            client=client,
            timeframe=args.timeframe,
            vedtak_id=args.vedtak,
            start_utc=args.start_utc,
            end_utc=args.end_utc,
            out_root=args.out_root,
        )
    else:
        report = materialize_native_xau_successor(
            client=client,
            timeframe=args.timeframe,
            vedtak_id=args.vedtak,
            start_utc=args.start_utc,
            end_utc=args.end_utc,
            out_root=args.out_root,
            parent_root=args.parent_root,
            expected_parent_manifest_sha256=(
                args.expected_parent_manifest_sha256
            ),
        )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
