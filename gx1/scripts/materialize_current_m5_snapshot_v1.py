"""Materialize one immutable current M5 tape from base M5 + live OANDA M1.

The live collector is a mutable external source.  This producer snapshots the
exact parquet bytes it reads, rejects schema/geometry/nonfinite/conflicting
duplicate values, admits only M1-complete or overlap-proven session-reopen M5
buckets, and requires bit-exact overlap with the already repaired event-local
M5 tape.  Unsupported partial buckets are omitted with exact evidence, never
filled or passed through.  The complete output directory is published
atomically and carries hashes for every year and every collector snapshot.  No
canonical or live file is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.contracts.entry_run_lineage_v1 import require_entry_run_id
from gx1.execution.v12_m1_to_m5_downsample import m1_to_m5


SCHEMA_VERSION = "m5_tape_current_snapshot_v1"
REQUIRED_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "bid_open",
    "bid_high",
    "bid_low",
    "bid_close",
    "ask_open",
    "ask_high",
    "ask_low",
    "ask_close",
)
COLLECTOR_PATTERN = "xauusd_m1_*.parquet"


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc(raw: Any, *, label: str) -> pd.Timestamp:
    try:
        parsed = pd.to_datetime(raw, utc=True, errors="raise")
    except Exception as exc:
        raise RuntimeError(f"CURRENT_M5_{label}_TIMESTAMP_INVALID: {raw!r}") from exc
    if pd.isna(parsed):
        raise RuntimeError(f"CURRENT_M5_{label}_TIMESTAMP_INVALID: {raw!r}")
    return pd.Timestamp(parsed)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"CURRENT_M5_{label}_MISSING_OR_SYMLINK: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"CURRENT_M5_{label}_INVALID_JSON: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"CURRENT_M5_{label}_OBJECT_REQUIRED: {path}")
    return value


def _atomic_bytes(path: Path, raw: bytes) -> None:
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


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    _atomic_bytes(path, raw)


def _validate_price_frame(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    expected = ["time", *REQUIRED_COLUMNS]
    observed = list(frame.columns)
    if len(observed) != len(expected) or len(set(observed)) != len(observed) or set(observed) != set(expected):
        raise RuntimeError(
            f"CURRENT_M5_{label}_SCHEMA_INVALID: got={observed} "
            f"expected={expected}"
        )
    out = frame.loc[:, expected].copy()
    out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")
    if out["time"].isna().any():
        raise RuntimeError(f"CURRENT_M5_{label}_TIME_INVALID")
    values = out.loc[:, list(REQUIRED_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError(f"CURRENT_M5_{label}_NONFINITE")
    for prefix in ("", "bid_", "ask_"):
        high = out[f"{prefix}high"].to_numpy(dtype=np.float64)
        low = out[f"{prefix}low"].to_numpy(dtype=np.float64)
        opened = out[f"{prefix}open"].to_numpy(dtype=np.float64)
        closed = out[f"{prefix}close"].to_numpy(dtype=np.float64)
        if np.any(high < np.maximum(opened, closed)) or np.any(
            low > np.minimum(opened, closed)
        ) or np.any(high < low):
            raise RuntimeError(f"CURRENT_M5_{label}_OHLC_GEOMETRY_INVALID")
    for suffix in ("open", "high", "low", "close"):
        if np.any(
            out[f"ask_{suffix}"].to_numpy(dtype=np.float64)
            < out[f"bid_{suffix}"].to_numpy(dtype=np.float64)
        ):
            raise RuntimeError(f"CURRENT_M5_{label}_BID_ASK_GEOMETRY_INVALID")
    return out


def _snapshot_collector_file(path: Path, destination: Path) -> tuple[pd.DataFrame, str]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"CURRENT_M5_COLLECTOR_FILE_INVALID: {path}")
    last_error: Exception | None = None
    for _ in range(5):
        try:
            raw = path.read_bytes()
            frame = pd.read_parquet(io.BytesIO(raw))
            frame = _validate_price_frame(frame, label="COLLECTOR")
            _atomic_bytes(destination, raw)
            return frame, _sha256_bytes(raw)
        except Exception as exc:
            last_error = exc
            time.sleep(0.05)
    raise RuntimeError(f"CURRENT_M5_COLLECTOR_SNAPSHOT_FAILED: {path}: {last_error}")


def _reject_conflicting_duplicates(frame: pd.DataFrame) -> int:
    duplicate = frame[frame.duplicated("time", keep=False)]
    conflicts: list[str] = []
    for timestamp, group in duplicate.groupby("time", sort=False):
        values = group.loc[:, REQUIRED_COLUMNS].to_numpy(dtype=np.float64)
        if not np.array_equal(values, np.repeat(values[:1], len(values), axis=0)):
            conflicts.append(pd.Timestamp(timestamp).isoformat())
            if len(conflicts) == 10:
                break
    if conflicts:
        raise RuntimeError(f"CURRENT_M5_COLLECTOR_DUPLICATE_CONFLICT: {conflicts}")
    return int(duplicate["time"].nunique())


def _filter_supported_m5_buckets(
    m1: pd.DataFrame, aggregated: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Keep only M5 buckets whose M1 coverage has an evidenced interpretation.

    Normal liquid-market buckets need all five minute bars.  OANDA's daily
    XAUUSD reopen is a stable exception: the 22:00 UTC M5 candle can consist of
    the sole 22:04 M1 candle.  That sparse convention is separately required
    to match native M5 exactly across the overlap before any tail is admitted.
    Other partial buckets are collector holes and are omitted explicitly.
    """

    work = m1.loc[:, ["time"]].copy()
    work["bucket"] = work["time"].dt.floor("5min")
    offsets_by_bucket: dict[pd.Timestamp, tuple[int, ...]] = {}
    for bucket, group in work.groupby("bucket", sort=False):
        offsets = tuple(
            int(value)
            for value in ((group["time"] - bucket) / pd.Timedelta(minutes=1)).tolist()
        )
        offsets_by_bucket[pd.Timestamp(bucket)] = offsets

    admitted: list[pd.Timestamp] = []
    reopen_timestamps: list[str] = []
    dense_rows = 0
    reopen_rows = 0
    dropped: list[dict[str, Any]] = []
    for raw_timestamp in aggregated["time"]:
        timestamp = pd.Timestamp(raw_timestamp)
        offsets = offsets_by_bucket.get(timestamp, ())
        if offsets == (0, 1, 2, 3, 4):
            dense_rows += 1
            admitted.append(timestamp)
        elif timestamp.hour == 22 and timestamp.minute == 0 and offsets == (4,):
            reopen_rows += 1
            admitted.append(timestamp)
            reopen_timestamps.append(timestamp.isoformat())
        else:
            dropped.append(
                {
                    "time_utc": timestamp.isoformat(),
                    "m1_offsets_minutes": list(offsets),
                    "reason": "unsupported_partial_m1_bucket",
                }
            )

    mask = aggregated["time"].isin(admitted)
    filtered = aggregated.loc[mask].reset_index(drop=True)
    return filtered, {
        "policy": "five_exact_minutes_or_22utc_reopen_at_offset4",
        "dense_m5_rows": dense_rows,
        "session_reopen_sparse_m5_rows": reopen_rows,
        "session_reopen_sparse_m5_buckets": reopen_timestamps,
        "dropped_unsupported_partial_m5_rows": len(dropped),
        "dropped_unsupported_partial_m5_buckets": dropped,
    }


def _copy_file(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_id = require_entry_run_id(getattr(args, "run_id", ""))
    base_root_arg = Path(args.base_m5_root).expanduser()
    collector_arg = Path(args.collector_dir).expanduser()
    out_arg = Path(args.out_root).expanduser()
    if base_root_arg.is_symlink() or not base_root_arg.is_dir():
        raise RuntimeError(f"CURRENT_M5_BASE_ROOT_INVALID: {base_root_arg}")
    if collector_arg.is_symlink() or not collector_arg.is_dir():
        raise RuntimeError(f"CURRENT_M5_COLLECTOR_ROOT_INVALID: {collector_arg}")
    base_root = base_root_arg.resolve()
    collector = collector_arg.resolve()
    out_root = out_arg.resolve()
    if out_arg.exists() or out_arg.is_symlink():
        raise RuntimeError(f"CURRENT_M5_OUTPUT_NOT_FRESH: {out_arg}")
    cutoff = _utc(args.cutoff_utc, label="CUTOFF")
    if cutoff.second or cutoff.microsecond:
        raise RuntimeError("CURRENT_M5_CUTOFF_NOT_MINUTE_ALIGNED")
    if int(args.minimum_overlap_bars) < 1:
        raise RuntimeError("CURRENT_M5_MINIMUM_OVERLAP_INVALID")

    base_manifest_path = base_root / "REPAIR_MANIFEST.json"
    base_manifest = _read_json(base_manifest_path, label="BASE_MANIFEST")
    if base_manifest.get("schema_version") != "m5_tape_dec2024_repair_manifest_v1":
        raise RuntimeError("CURRENT_M5_BASE_MANIFEST_SCHEMA_INVALID")
    if base_manifest.get("explicit_vedtak_id") != run_id:
        raise RuntimeError("CURRENT_M5_BASE_RUN_ID_MISMATCH")
    if base_manifest.get("geometry_bad_total_after") != 0:
        raise RuntimeError("CURRENT_M5_BASE_GEOMETRY_NOT_PROVEN")
    base_years = base_manifest.get("years")
    if not isinstance(base_years, dict) or not base_years:
        raise RuntimeError("CURRENT_M5_BASE_YEARS_INVALID")

    stage = out_root.parent / f".{out_root.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    if stage.exists() or stage.is_symlink():
        raise RuntimeError(f"CURRENT_M5_STAGE_COLLISION: {stage}")
    stage.mkdir(parents=True)
    try:
        collector_snapshot = stage / "collector_snapshot"
        collector_snapshot.mkdir()
        collector_frames: list[pd.DataFrame] = []
        collector_sources: list[dict[str, Any]] = []
        collector_files = sorted(collector.glob(COLLECTOR_PATTERN))
        if not collector_files:
            raise RuntimeError("CURRENT_M5_COLLECTOR_FILES_MISSING")
        for source in collector_files:
            destination = collector_snapshot / source.name
            frame, digest = _snapshot_collector_file(source, destination)
            collector_frames.append(frame)
            collector_sources.append(
                {
                    "source_path": str(source),
                    "snapshot_path": str((out_root / "collector_snapshot" / source.name)),
                    "sha256": digest,
                    "rows": int(len(frame)),
                    "time_min_utc": pd.Timestamp(frame["time"].min()).isoformat(),
                    "time_max_utc": pd.Timestamp(frame["time"].max()).isoformat(),
                }
            )

        collected = pd.concat(collector_frames, ignore_index=True).sort_values("time")
        duplicate_timestamps = _reject_conflicting_duplicates(collected)
        collected = (
            collected.drop_duplicates("time", keep="last")
            .sort_values("time")
            .reset_index(drop=True)
        )
        collected = collected.loc[collected["time"] <= cutoff].copy()
        if collected.empty or pd.Timestamp(collected["time"].iloc[-1]) != cutoff:
            observed = None if collected.empty else pd.Timestamp(collected["time"].iloc[-1])
            raise RuntimeError(
                f"CURRENT_M5_CUTOFF_NOT_PRESENT: cutoff={cutoff} observed={observed}"
            )
        if collected["time"].duplicated().any() or not collected["time"].is_monotonic_increasing:
            raise RuntimeError("CURRENT_M5_COLLECTOR_TIME_ORDER_INVALID")

        m5_from_collector = m1_to_m5(collected, tape_end=cutoff)
        m5_from_collector = _validate_price_frame(
            m5_from_collector.loc[:, ["time", *REQUIRED_COLUMNS]],
            label="AGGREGATED",
        )
        m5_from_collector, coverage_proof = _filter_supported_m5_buckets(
            collected, m5_from_collector
        )
        if m5_from_collector.empty:
            raise RuntimeError("CURRENT_M5_SUPPORTED_BUCKETS_EMPTY")
        expected_last_m5 = (cutoff - pd.Timedelta(minutes=4)).floor("5min")
        actual_last_m5 = pd.Timestamp(m5_from_collector["time"].iloc[-1])
        if actual_last_m5 != expected_last_m5:
            raise RuntimeError(
                f"CURRENT_M5_LAST_COMPLETE_MISMATCH: actual={actual_last_m5} "
                f"expected={expected_last_m5}"
            )

        base_hashes: dict[str, str] = {}
        output_years: dict[str, dict[str, Any]] = {}
        final_year = max(int(str(key).split("=", 1)[1]) for key in base_years)
        if actual_last_m5.year != final_year:
            raise RuntimeError(
                "CURRENT_M5_CROSS_YEAR_APPEND_UNSUPPORTED: "
                f"base_final_year={final_year} current_year={actual_last_m5.year}"
            )
        for key in sorted(base_years):
            source = base_root / key / "part-000.parquet"
            if source.is_symlink() or not source.is_file():
                raise RuntimeError(f"CURRENT_M5_BASE_PART_INVALID: {source}")
            source_sha = _sha256_file(source)
            if (base_years.get(key) or {}).get("output_sha256") != source_sha:
                raise RuntimeError(f"CURRENT_M5_BASE_PART_HASH_MISMATCH: {key}")
            base_hashes[key] = source_sha
            destination_dir = stage / key
            destination_dir.mkdir()
            destination = destination_dir / "part-000.parquet"
            year = int(str(key).split("=", 1)[1])
            if year != final_year:
                _copy_file(source, destination)
            else:
                base = _validate_price_frame(pd.read_parquet(source), label="BASE_FINAL_YEAR")
                if base["time"].duplicated().any() or not base["time"].is_monotonic_increasing:
                    raise RuntimeError("CURRENT_M5_BASE_FINAL_YEAR_TIME_ORDER_INVALID")
                overlap_times = np.intersect1d(
                    base["time"].to_numpy(), m5_from_collector["time"].to_numpy()
                )
                if len(overlap_times) < int(args.minimum_overlap_bars):
                    raise RuntimeError(
                        f"CURRENT_M5_OVERLAP_INSUFFICIENT: got={len(overlap_times)} "
                        f"required={args.minimum_overlap_bars}"
                    )
                left = base.set_index("time").loc[overlap_times, list(REQUIRED_COLUMNS)]
                right = m5_from_collector.set_index("time").loc[
                    overlap_times, list(REQUIRED_COLUMNS)
                ]
                left_values = left.to_numpy(dtype=np.float64)
                right_values = right.to_numpy(dtype=np.float64)
                if not np.array_equal(left_values, right_values):
                    delta = np.abs(left_values - right_values)
                    raise RuntimeError(
                        f"CURRENT_M5_OVERLAP_MISMATCH: max_abs_diff={float(delta.max())}"
                    )
                base_min = pd.Timestamp(base["time"].iloc[0])
                base_max = pd.Timestamp(base["time"].iloc[-1])
                sparse_timestamps = {
                    pd.Timestamp(value)
                    for value in coverage_proof["session_reopen_sparse_m5_buckets"]
                }
                sparse_overlap = {
                    pd.Timestamp(value) for value in overlap_times
                } & sparse_timestamps
                sparse_tail = {
                    timestamp for timestamp in sparse_timestamps if timestamp > base_max
                }
                if sparse_tail and not sparse_overlap:
                    raise RuntimeError("CURRENT_M5_SPARSE_REOPEN_OVERLAP_PROOF_MISSING")
                dropped_overlap = [
                    row
                    for row in coverage_proof[
                        "dropped_unsupported_partial_m5_buckets"
                    ]
                    if base_min <= pd.Timestamp(row["time_utc"]) <= base_max
                ]
                dropped_tail = [
                    row
                    for row in coverage_proof[
                        "dropped_unsupported_partial_m5_buckets"
                    ]
                    if pd.Timestamp(row["time_utc"]) > base_max
                ]
                tail = m5_from_collector.loc[m5_from_collector["time"] > base_max]
                if tail.empty:
                    raise RuntimeError("CURRENT_M5_NEW_TAIL_EMPTY")
                merged = pd.concat([base, tail], ignore_index=True).sort_values("time")
                if merged["time"].duplicated().any():
                    raise RuntimeError("CURRENT_M5_MERGED_TIME_DUPLICATE")
                merged.to_parquet(destination, index=False)
                overlap_proof = {
                    "rows": int(len(overlap_times)),
                    "time_min_utc": pd.Timestamp(overlap_times[0]).isoformat(),
                    "time_max_utc": pd.Timestamp(overlap_times[-1]).isoformat(),
                    "max_abs_diff": 0.0,
                    "new_tail_rows": int(len(tail)),
                    "base_time_max_utc": base_max.isoformat(),
                    "session_reopen_sparse_rows": len(sparse_overlap),
                    "session_reopen_sparse_tail_rows": len(sparse_tail),
                    "unsupported_overlap_buckets_omitted": len(dropped_overlap),
                    "unsupported_tail_buckets_omitted": len(dropped_tail),
                }
            output = pd.read_parquet(destination, columns=["time"])
            output_time = pd.to_datetime(output["time"], utc=True, errors="coerce")
            output_years[key] = {
                "rows": int(len(output_time)),
                "time_min_utc": pd.Timestamp(output_time.iloc[0]).isoformat(),
                "time_max_utc": pd.Timestamp(output_time.iloc[-1]).isoformat(),
                "output_sha256": _sha256_file(destination),
            }

        report = {
            "schema_version": SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "entry_run_id": run_id,
            "method": "immutable_live_collector_snapshot_exact_m5_overlap",
            "cutoff_complete_m1_utc": cutoff.isoformat(),
            "last_complete_m5_utc": actual_last_m5.isoformat(),
            "base_tape_root": str(base_root),
            "base_manifest_path": str(base_manifest_path),
            "base_manifest_sha256": _sha256_file(base_manifest_path),
            "base_year_sha256": base_hashes,
            "collector_root": str(collector),
            "collector_sources": collector_sources,
            "collector_unique_rows_through_cutoff": int(len(collected)),
            "collector_duplicate_timestamps_identical": duplicate_timestamps,
            "m1_coverage_proof": coverage_proof,
            "overlap_exact": True,
            "overlap_proof": overlap_proof,
            "geometry_bad_total_after": 0,
            "years": output_years,
        }
        _atomic_json(stage / "REPAIR_MANIFEST.json", report)
        os.replace(stage, out_root)
        return report
    except Exception:
        if stage.exists() and stage.parent == out_root.parent and stage.name.startswith(
            f".{out_root.name}.tmp-"
        ):
            shutil.rmtree(stage)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-m5-root", type=Path, required=True)
    parser.add_argument("--collector-dir", type=Path, required=True)
    parser.add_argument("--cutoff-utc", required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--minimum-overlap-bars", type=int, default=12)
    return parser


def main() -> int:
    report = run(build_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
