#!/usr/bin/env python3
"""Publish one immutable native OANDA XAU_USD M1 or M5 source bundle.

This is the sole canonical native-M1/M5 producer.  It requests only OANDA
``complete=true`` MBA candles, preserves every normalized source response,
re-derives every parquet row from those responses, validates the complete
bundle while it is hidden, and atomically publishes without replacement.
Request failures, malformed candles, duplicate timestamps, repository drift,
and publication races are fatal.  There is no incremental mutation, alternate
provider, resampling, synthesis, repair fallback, or empty-success path.
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
    XAU_INSTRUMENT,
    canonical_json_sha256,
    canonical_native_frame_from_oanda_response,
    canonical_native_rows_bytes,
    native_timeframe_policy,
    sha256_file,
    validate_canonical_native_source_bundle_v3,
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

        validate_canonical_native_source_bundle_v3(
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
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    # The authorization gate precedes environment loading, credentials,
    # networking, staging, and every external-data write.
    require_retrain_vedtak(args.vedtak)
    missing = [
        flag
        for flag, value in (
            ("--timeframe", args.timeframe),
            ("--start-utc", args.start_utc),
            ("--end-utc", args.end_utc),
            ("--out-root", args.out_root),
        )
        if value is None
    ]
    if missing:
        parser.error(f"the following arguments are required: {', '.join(missing)}")
    load_dotenv_if_present()
    client = _load_oanda_client()
    report = materialize_native_xau_snapshot(
        client=client,
        timeframe=args.timeframe,
        vedtak_id=args.vedtak,
        start_utc=args.start_utc,
        end_utc=args.end_utc,
        out_root=args.out_root,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
