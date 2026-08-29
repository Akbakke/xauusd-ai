#!/usr/bin/env python3
"""Materialize one immutable M5 OHLCV source which is physically before TEST.

The current V46 M5-enriched parquet was published after the TEST boundary.  A
normal parquet filter is not an acceptable solution here: its final row group
crosses the boundary, so a reader could decompress TEST rows while selecting a
prefix.  This utility deliberately reads only whole row groups whose *maximum*
timestamp is before the declared prefix boundary.  The short remaining tail is
rebuilt from separately stored, day-bounded M1 reports, also admitted only when
their complete parquet time range is before TEST.

The output is a new, immutable, six-column M5 candidate source.  It is not a
replacement for any existing source or cache: a candidate built from M1 must
also pass the independent MTF timestamp-axis proof before it can be used for a
V4 cache.  The companion manifest records exactly which safe inputs were
admitted and makes the boundary proof machine-readable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


_OHLCV = ("time", "open", "high", "low", "close", "volume")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc(value: str | pd.Timestamp, *, label: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise RuntimeError(f"PRETEST_MTF_SOURCE_{label}_UTC_REQUIRED")
    return stamp.tz_convert("UTC")


def _regular(path: Path, *, label: str) -> Path:
    path = path.expanduser()
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise RuntimeError(f"PRETEST_MTF_SOURCE_{label}_REGULAR_FILE_REQUIRED:{path}")
    return path.resolve(strict=True)


def _timestamp_stat(
    parquet: pq.ParquetFile,
    row_group: int,
    *,
    time_column: int,
    label: str,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    statistics = parquet.metadata.row_group(row_group).column(time_column).statistics
    if statistics is None or statistics.min is None or statistics.max is None:
        raise RuntimeError(f"PRETEST_MTF_SOURCE_{label}_TIME_STATS_REQUIRED:{row_group}")
    lower = _utc(statistics.min, label=f"{label}_MIN")
    upper = _utc(statistics.max, label=f"{label}_MAX")
    if upper < lower:
        raise RuntimeError(f"PRETEST_MTF_SOURCE_{label}_TIME_STATS_INVALID:{row_group}")
    return lower, upper


def _read_whole_safe_prefix(
    path: Path,
    *,
    prefix_end_inclusive: pd.Timestamp,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Read only row groups wholly before ``prefix_end_inclusive``.

    A row group which intersects the boundary is rejected rather than filtered;
    this is the key guarantee that the materializer never loads a TEST row.
    """

    source = _regular(path, label="PREFIX")
    parquet = pq.ParquetFile(source)
    names = tuple(parquet.schema_arrow.names)
    missing = [column for column in _OHLCV if column not in names]
    if missing:
        raise RuntimeError(f"PRETEST_MTF_SOURCE_PREFIX_COLUMNS_MISSING:{missing}")
    time_column = names.index("time")
    tables: list[pa.Table] = []
    evidence: list[dict[str, Any]] = []
    for row_group in range(parquet.metadata.num_row_groups):
        lower, upper = _timestamp_stat(
            parquet,
            row_group,
            time_column=time_column,
            label="PREFIX_ROW_GROUP",
        )
        if upper <= prefix_end_inclusive:
            tables.append(parquet.read_row_group(row_group, columns=list(_OHLCV)))
            evidence.append(
                {
                    "row_group": row_group,
                    "rows": int(parquet.metadata.row_group(row_group).num_rows),
                    "time_min_utc": lower.isoformat(),
                    "time_max_utc": upper.isoformat(),
                    "admitted": True,
                }
            )
            continue
        if lower > prefix_end_inclusive:
            evidence.append(
                {
                    "row_group": row_group,
                    "rows": int(parquet.metadata.row_group(row_group).num_rows),
                    "time_min_utc": lower.isoformat(),
                    "time_max_utc": upper.isoformat(),
                    "admitted": False,
                }
            )
            continue
        raise RuntimeError(
            "PRETEST_MTF_SOURCE_PREFIX_ROW_GROUP_CROSSES_BOUNDARY:"
            f"row_group={row_group}:min={lower.isoformat()}:max={upper.isoformat()}:"
            f"boundary={prefix_end_inclusive.isoformat()}"
        )
    if not tables:
        raise RuntimeError("PRETEST_MTF_SOURCE_PREFIX_EMPTY")
    frame = pa.concat_tables(tables).to_pandas()
    return _normalise_m5(frame, context="PREFIX"), evidence


def _normalise_m5(frame: pd.DataFrame, *, context: str) -> pd.DataFrame:
    missing = [column for column in _OHLCV if column not in frame.columns]
    if missing:
        raise RuntimeError(f"PRETEST_MTF_SOURCE_{context}_COLUMNS_MISSING:{missing}")
    result = frame.loc[:, _OHLCV].copy()
    result["time"] = pd.to_datetime(result["time"], utc=True, errors="raise")
    if result["time"].duplicated().any() or not result["time"].is_monotonic_increasing:
        raise RuntimeError(f"PRETEST_MTF_SOURCE_{context}_TIME_AXIS_INVALID")
    for column in _OHLCV[1:]:
        result[column] = pd.to_numeric(result[column], errors="raise").astype(np.float32)
    values = result.loc[:, _OHLCV[1:]].to_numpy(dtype=np.float32, copy=False)
    if not np.isfinite(values).all():
        raise RuntimeError(f"PRETEST_MTF_SOURCE_{context}_NONFINITE")
    if (result["high"] < result[["open", "close", "low"]].max(axis=1)).any() or (
        result["low"] > result[["open", "close", "high"]].min(axis=1)
    ).any():
        raise RuntimeError(f"PRETEST_MTF_SOURCE_{context}_OHLC_INVALID")
    return result


def _m1_tail_paths(
    directory: Path,
    *,
    tail_start: pd.Timestamp,
    test_start: pd.Timestamp,
) -> Iterable[Path]:
    root = directory.expanduser()
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"PRETEST_MTF_SOURCE_TAIL_DIR_INVALID:{root}")
    for day in pd.date_range(tail_start.normalize(), test_start.normalize(), freq="D"):
        if day >= test_start:
            break
        candidate = root / f"xauusd_m1_{day.strftime('%Y%m%d')}.parquet"
        if candidate.is_file() and not candidate.is_symlink():
            yield candidate.resolve(strict=True)


def _read_safe_m1_tail(
    directory: Path,
    *,
    tail_start: pd.Timestamp,
    test_start: pd.Timestamp,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    evidence: list[dict[str, Any]] = []
    for path in _m1_tail_paths(
        directory, tail_start=tail_start, test_start=test_start
    ):
        parquet = pq.ParquetFile(path)
        names = tuple(parquet.schema_arrow.names)
        missing = [column for column in _OHLCV if column not in names]
        if missing:
            raise RuntimeError(f"PRETEST_MTF_SOURCE_TAIL_COLUMNS_MISSING:{path}:{missing}")
        time_column = names.index("time")
        ranges = [
            _timestamp_stat(
                parquet,
                row_group,
                time_column=time_column,
                label="TAIL_ROW_GROUP",
            )
            for row_group in range(parquet.metadata.num_row_groups)
        ]
        if not ranges:
            raise RuntimeError(f"PRETEST_MTF_SOURCE_TAIL_EMPTY:{path}")
        observed_min = min(lower for lower, _ in ranges)
        observed_max = max(upper for _, upper in ranges)
        if observed_min < tail_start.normalize() or observed_max >= test_start:
            raise RuntimeError(
                "PRETEST_MTF_SOURCE_TAIL_OUTSIDE_SAFE_WINDOW:"
                f"path={path}:min={observed_min.isoformat()}:max={observed_max.isoformat()}"
            )
        frame = pq.read_table(path, columns=list(_OHLCV)).to_pandas()
        frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="raise")
        frame = frame.loc[(frame["time"] >= tail_start) & (frame["time"] < test_start)]
        if not frame.empty:
            frames.append(frame)
        evidence.append(
            {
                "path": str(path),
                "rows_loaded": int(len(frame)),
                "time_min_utc": observed_min.isoformat(),
                "time_max_utc": observed_max.isoformat(),
                "test_rows_loaded": False,
            }
        )
    if not frames:
        raise RuntimeError("PRETEST_MTF_SOURCE_TAIL_EMPTY")
    minute = pd.concat(frames, ignore_index=True).sort_values("time")
    if minute["time"].duplicated().any():
        raise RuntimeError("PRETEST_MTF_SOURCE_TAIL_DUPLICATE_M1_TIME")
    minute = _normalise_m5(minute, context="TAIL_M1")
    minute = minute.set_index("time")
    count = minute["close"].resample("5min", label="left", closed="left").count()
    tail = minute.resample("5min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    tail = tail.loc[count.eq(5)].reset_index()
    return _normalise_m5(tail, context="TAIL_M5"), evidence


def _write_immutable_parquet(path: Path, frame: pd.DataFrame) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"PRETEST_MTF_SOURCE_OUTPUT_EXISTS:{path}")
    parent = path.parent
    if not parent.is_absolute() or parent.is_symlink() or not parent.is_dir():
        raise RuntimeError(f"PRETEST_MTF_SOURCE_OUTPUT_PARENT_INVALID:{parent}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(parent))
    temporary = Path(temporary_name)
    try:
        os.close(descriptor)
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError as exc:
        raise RuntimeError(f"PRETEST_MTF_SOURCE_OUTPUT_EXISTS:{path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def materialize_pretest_m5_source(
    *,
    prefix_parquet: Path,
    prefix_end_inclusive: str | pd.Timestamp,
    m1_tail_dir: Path,
    test_start_utc: str | pd.Timestamp,
    output_parquet: Path,
    output_manifest: Path,
) -> dict[str, Any]:
    """Build a sealed M5 source without loading any TEST row."""

    prefix_end = _utc(prefix_end_inclusive, label="PREFIX_END")
    test_start = _utc(test_start_utc, label="TEST_START")
    if prefix_end >= test_start:
        raise RuntimeError("PRETEST_MTF_SOURCE_BOUNDARY_ORDER_INVALID")
    prefix, prefix_evidence = _read_whole_safe_prefix(
        prefix_parquet, prefix_end_inclusive=prefix_end
    )
    if prefix["time"].iloc[-1] != prefix_end:
        raise RuntimeError(
            "PRETEST_MTF_SOURCE_PREFIX_END_MISMATCH:"
            f"observed={prefix['time'].iloc[-1].isoformat()}:expected={prefix_end.isoformat()}"
        )
    tail_start = prefix_end + pd.Timedelta(minutes=5)
    tail, tail_evidence = _read_safe_m1_tail(
        m1_tail_dir, tail_start=tail_start, test_start=test_start
    )
    if tail.empty or tail["time"].iloc[0] != tail_start:
        observed = None if tail.empty else tail["time"].iloc[0].isoformat()
        raise RuntimeError(f"PRETEST_MTF_SOURCE_TAIL_START_MISMATCH:{observed}")
    merged = _normalise_m5(pd.concat([prefix, tail], ignore_index=True), context="OUTPUT")
    if (merged["time"] >= test_start).any():
        raise RuntimeError("PRETEST_MTF_SOURCE_TEST_ROW_BLOCKED")
    output = output_parquet.expanduser()
    manifest = output_manifest.expanduser()
    if not output.is_absolute() or not manifest.is_absolute():
        raise RuntimeError("PRETEST_MTF_SOURCE_OUTPUT_ABSOLUTE_REQUIRED")
    _write_immutable_parquet(output, merged)
    payload: dict[str, Any] = {
        "schema_version": "gx1_pretest_mtf_source_v1",
        "decision": "PASS",
        "test_accessed": False,
        "test_start_utc": test_start.isoformat(),
        "output_parquet": str(output),
        "output_sha256": _sha256_file(output),
        "rows": int(len(merged)),
        "time_min_utc": merged["time"].iloc[0].isoformat(),
        "time_max_utc": merged["time"].iloc[-1].isoformat(),
        "prefix": {
            "path": str(Path(prefix_parquet).expanduser()),
            "read_through_utc": prefix_end.isoformat(),
            "row_groups": prefix_evidence,
        },
        "m1_tail": {"directory": str(Path(m1_tail_dir).expanduser()), "files": tail_evidence},
    }
    if manifest.exists() or manifest.is_symlink():
        raise RuntimeError(f"PRETEST_MTF_SOURCE_MANIFEST_EXISTS:{manifest}")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(manifest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix-parquet", type=Path, required=True)
    parser.add_argument("--prefix-end-inclusive", required=True)
    parser.add_argument("--m1-tail-dir", type=Path, required=True)
    parser.add_argument("--test-start-utc", required=True)
    parser.add_argument("--output-parquet", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()
    payload = materialize_pretest_m5_source(
        prefix_parquet=args.prefix_parquet,
        prefix_end_inclusive=args.prefix_end_inclusive,
        m1_tail_dir=args.m1_tail_dir,
        test_start_utc=args.test_start_utc,
        output_parquet=args.output_parquet,
        output_manifest=args.output_manifest,
    )
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "test_accessed": payload["test_accessed"],
                "rows": payload["rows"],
                "time_min_utc": payload["time_min_utc"],
                "time_max_utc": payload["time_max_utc"],
                "output_parquet": payload["output_parquet"],
                "output_sha256": payload["output_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
