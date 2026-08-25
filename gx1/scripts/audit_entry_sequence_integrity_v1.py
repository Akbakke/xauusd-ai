#!/usr/bin/env python3
"""Prove full seq/snap integrity for causally filtered model-native splits.

Unlike the strict sequence-roll reconstruction proof, this audit permits gaps
between *emitted* rows.  It binds every timestamp to the immutable source tape,
proves an exact physical overlap whenever fewer than 96 source events elapsed,
and records source-indexed boundaries where two 96-bar windows cannot overlap.
This preserves genuine market closures and causal M1 label exclusions without
accepting synthetic zero-filled bars or an unbound sequence reset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_SEQ_LEN,
    MODEL_NATIVE_SIGNAL_DIM,
)
from gx1.contracts.entry_sequence_integrity_v1 import (
    AUTHORITY,
    REQUIRED_CHECKS,
    SCHEMA_VERSION,
)


_SHA256_BUFFER_BYTES = 1024 * 1024
_ARROW_BATCH_ROWS = 256
_M5_NS = 5 * 60 * 1_000_000_000


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_SHA256_BUFFER_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _exact_regular_file(raw: Path, *, label: str) -> Path:
    supplied = Path(raw).expanduser()
    if not supplied.is_absolute() or supplied.is_symlink():
        raise RuntimeError(f"[ENTRY_SEQUENCE_INTEGRITY_{label}_PATH_INVALID]")
    try:
        resolved = supplied.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"[ENTRY_SEQUENCE_INTEGRITY_{label}_PATH_INVALID]") from exc
    if resolved != supplied or not resolved.is_file():
        raise RuntimeError(f"[ENTRY_SEQUENCE_INTEGRITY_{label}_PATH_INVALID]")
    return resolved


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _time_ns(column: Any, *, row_start: int) -> np.ndarray:
    try:
        values = column.to_numpy(zero_copy_only=False).astype("datetime64[ns]")
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"[ENTRY_SEQUENCE_INTEGRITY_TIME_INVALID] rows={row_start}"
        ) from exc
    values = values.astype("int64", copy=False)
    if np.any(values == np.iinfo(np.int64).min):
        raise RuntimeError(f"[ENTRY_SEQUENCE_INTEGRITY_TIME_NULL] rows={row_start}")
    return values


def _source_frame(manifest: dict[str, Any]) -> tuple[Path, str, np.ndarray]:
    extra = manifest.get("extra")
    source_frame = extra.get("source_frame") if isinstance(extra, dict) else None
    if not isinstance(source_frame, dict):
        raise RuntimeError("[ENTRY_SEQUENCE_INTEGRITY_SOURCE_FRAME_MISSING]")
    raw_path = source_frame.get("parquet_path")
    declared_sha256 = source_frame.get("parquet_sha256")
    if not isinstance(raw_path, str) or not isinstance(declared_sha256, str):
        raise RuntimeError("[ENTRY_SEQUENCE_INTEGRITY_SOURCE_FRAME_INVALID]")
    source_path = _exact_regular_file(Path(raw_path), label="SOURCE")
    observed_sha256 = _sha256_file(source_path)
    if observed_sha256 != declared_sha256:
        raise RuntimeError("[ENTRY_SEQUENCE_INTEGRITY_SOURCE_FRAME_HASH_MISMATCH]")
    source_pf = pq.ParquetFile(source_path)
    if "time" not in source_pf.schema_arrow.names:
        raise RuntimeError("[ENTRY_SEQUENCE_INTEGRITY_SOURCE_TIME_MISSING]")
    source_time_ns = _time_ns(
        source_pf.read(columns=["time"]).column("time"), row_start=0
    )
    if len(source_time_ns) < MODEL_NATIVE_SEQ_LEN:
        raise RuntimeError("[ENTRY_SEQUENCE_INTEGRITY_SOURCE_ROWS_INVALID]")
    source_deltas = np.diff(source_time_ns)
    if np.any(source_deltas <= 0):
        raise RuntimeError("[ENTRY_SEQUENCE_INTEGRITY_SOURCE_TIME_NOT_INCREASING]")
    if np.any(source_deltas % _M5_NS != 0):
        raise RuntimeError("[ENTRY_SEQUENCE_INTEGRITY_SOURCE_TIME_NOT_M5_ALIGNED]")
    return source_path, observed_sha256, source_time_ns


def audit_sequence_integrity(*, parquet_path: Path, manifest_path: Path) -> dict[str, Any]:
    """Audit every emitted row and transition in one exact split parquet."""

    parquet_path = _exact_regular_file(parquet_path, label="PARQUET")
    manifest_path = _exact_regular_file(manifest_path, label="MANIFEST")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("[ENTRY_SEQUENCE_INTEGRITY_MANIFEST_JSON_INVALID]") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError("[ENTRY_SEQUENCE_INTEGRITY_MANIFEST_JSON_INVALID]")
    source_path, source_sha256, source_time_ns = _source_frame(manifest)

    pf = pq.ParquetFile(parquet_path)
    required = {"time", "seq", "snap"}
    missing = sorted(required - set(pf.schema_arrow.names))
    if missing:
        raise RuntimeError(f"[ENTRY_SEQUENCE_INTEGRITY_COLUMNS_MISSING] {missing}")
    rows = int(pf.metadata.num_rows)
    if rows < 2:
        raise RuntimeError("[ENTRY_SEQUENCE_INTEGRITY_ROWS_INVALID]")

    previous_sequence: np.ndarray | None = None
    previous_time_ns: int | None = None
    previous_source_position: int | None = None
    observed_rows = 0
    summary = {
        "pairs": 0,
        "calendar_one_bar_pairs": 0,
        "calendar_gap_pairs": 0,
        "physical_one_bar_pairs": 0,
        "physical_multi_bar_pairs": 0,
        "calendar_elapsed_bars_total": 0,
        "physical_event_bars_total": 0,
        "nontrading_calendar_bars_total": 0,
        "source_overlap_eligible_pairs": 0,
        "source_nonoverlap_boundary_pairs": 0,
    }
    event_hash = hashlib.sha256()
    event_hash.update(b"entry_model_native_sequence_integrity_event_chain_v2\0")
    event_hash.update(
        np.asarray([rows, MODEL_NATIVE_SEQ_LEN, MODEL_NATIVE_SIGNAL_DIM], dtype="<i8").tobytes()
    )
    for batch in pf.iter_batches(
        batch_size=_ARROW_BATCH_ROWS,
        columns=["time", "seq", "snap"],
    ):
        count = int(batch.num_rows)
        time_ns = _time_ns(batch.column("time"), row_start=observed_rows)
        source_positions = np.searchsorted(source_time_ns, time_ns)
        sequence = (
            batch.column("seq")
            .flatten()
            .flatten()
            .to_numpy(zero_copy_only=False)
            .reshape(count, MODEL_NATIVE_SEQ_LEN, MODEL_NATIVE_SIGNAL_DIM)
            .astype(np.float32, copy=False)
        )
        snapshot = (
            batch.column("snap")
            .flatten()
            .to_numpy(zero_copy_only=False)
            .reshape(count, MODEL_NATIVE_SIGNAL_DIM)
            .astype(np.float32, copy=False)
        )
        if not np.isfinite(sequence).all() or not np.isfinite(snapshot).all():
            raise RuntimeError(
                f"[ENTRY_SEQUENCE_INTEGRITY_NONFINITE] rows={observed_rows}:{observed_rows + count}"
            )
        if not np.array_equal(sequence[:, -1, :], snapshot):
            raise RuntimeError(
                f"[ENTRY_SEQUENCE_INTEGRITY_LAST_SNAPSHOT_MISMATCH] rows={observed_rows}:{observed_rows + count}"
            )
        for index in range(count):
            current_time_ns = int(time_ns[index])
            current_sequence = sequence[index]
            current_source_position = int(source_positions[index])
            if previous_sequence is not None and previous_time_ns is not None:
                delta_ns = current_time_ns - previous_time_ns
                row = observed_rows + index
                if delta_ns <= 0:
                    raise RuntimeError(
                        f"[ENTRY_SEQUENCE_INTEGRITY_TIME_NOT_INCREASING] row={row} delta_ns={delta_ns}"
                    )
                if delta_ns % _M5_NS != 0:
                    raise RuntimeError(
                        f"[ENTRY_SEQUENCE_INTEGRITY_TIME_NOT_M5_ALIGNED] row={row} delta_ns={delta_ns}"
                    )
            if (
                current_source_position >= len(source_time_ns)
                or source_time_ns[current_source_position] != current_time_ns
            ):
                raise RuntimeError(
                    "[ENTRY_SEQUENCE_INTEGRITY_TIME_NOT_IN_SOURCE] "
                    f"row={observed_rows + index}"
                )
            if previous_sequence is not None and previous_time_ns is not None:
                elapsed_m5_bars = delta_ns // _M5_NS
                if previous_source_position is None:
                    raise RuntimeError("[ENTRY_SEQUENCE_INTEGRITY_SOURCE_POSITION_INVALID]")
                physical_shift = current_source_position - previous_source_position
                if physical_shift <= 0:
                    raise RuntimeError(
                        "[ENTRY_SEQUENCE_INTEGRITY_SOURCE_POSITION_NOT_INCREASING] "
                        f"row={row} source_step={physical_shift}"
                    )
                if physical_shift > elapsed_m5_bars:
                    raise RuntimeError(
                        "[ENTRY_SEQUENCE_INTEGRITY_SOURCE_STEP_EXCEEDS_CALENDAR] "
                        f"row={row} source_step={physical_shift} elapsed_m5_bars={elapsed_m5_bars}"
                    )
                overlap_eligible = physical_shift < MODEL_NATIVE_SEQ_LEN
                if overlap_eligible:
                    if not np.array_equal(
                        current_sequence[: MODEL_NATIVE_SEQ_LEN - physical_shift],
                        previous_sequence[physical_shift:],
                    ):
                        raise RuntimeError(
                            "[ENTRY_SEQUENCE_INTEGRITY_OVERLAP_MISMATCH] "
                            f"row={row} source_step={physical_shift}"
                        )
                    summary["source_overlap_eligible_pairs"] += 1
                else:
                    summary["source_nonoverlap_boundary_pairs"] += 1
                summary["pairs"] += 1
                summary["calendar_elapsed_bars_total"] += int(elapsed_m5_bars)
                summary["physical_event_bars_total"] += physical_shift
                summary["nontrading_calendar_bars_total"] += int(elapsed_m5_bars) - physical_shift
                summary[
                    "calendar_one_bar_pairs" if elapsed_m5_bars == 1 else "calendar_gap_pairs"
                ] += 1
                summary[
                    "physical_one_bar_pairs" if physical_shift == 1 else "physical_multi_bar_pairs"
                ] += 1
                event_hash.update(
                    np.asarray(
                        [delta_ns, physical_shift, int(overlap_eligible)], dtype="<i8"
                    ).tobytes()
                )
            previous_sequence = current_sequence.copy()
            previous_time_ns = current_time_ns
            previous_source_position = current_source_position
        event_hash.update(np.ascontiguousarray(time_ns, dtype="<i8").tobytes(order="C"))
        event_hash.update(np.ascontiguousarray(sequence, dtype="<f4").tobytes(order="C"))
        event_hash.update(np.ascontiguousarray(snapshot, dtype="<f4").tobytes(order="C"))
        observed_rows += count
    if observed_rows != rows:
        raise RuntimeError("[ENTRY_SEQUENCE_INTEGRITY_ROW_COUNT_MISMATCH]")

    return {
        "schema_version": SCHEMA_VERSION,
        "decision": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "parquet_path": str(parquet_path),
        "parquet_sha256": _sha256_file(parquet_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "source_parquet_path": str(source_path),
        "source_parquet_sha256": source_sha256,
        "source_rows": int(len(source_time_ns)),
        "rows": rows,
        "sequence_shape": [rows, MODEL_NATIVE_SEQ_LEN, MODEL_NATIVE_SIGNAL_DIM],
        "snapshot_shape": [rows, MODEL_NATIVE_SIGNAL_DIM],
        "checks": dict(REQUIRED_CHECKS),
        "transition_summary": summary,
        "sequence_event_chain_sha256": event_hash.hexdigest(),
        "authority": dict(AUTHORITY),
    }


def _write_new_json(path: Path, payload: dict[str, Any]) -> None:
    supplied = Path(path).expanduser()
    if not supplied.is_absolute() or supplied.is_symlink() or supplied.exists():
        raise RuntimeError("[ENTRY_SEQUENCE_INTEGRITY_OUTPUT_INVALID]")
    parent = supplied.parent
    if not parent.is_absolute() or parent.is_symlink() or not parent.is_dir():
        raise RuntimeError("[ENTRY_SEQUENCE_INTEGRITY_OUTPUT_PARENT_INVALID]")
    descriptor = os.open(supplied, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json(payload))
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            supplied.unlink(missing_ok=True)
        finally:
            raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("audit exact physical event-chain sequence integrity")
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = audit_sequence_integrity(
        parquet_path=args.parquet,
        manifest_path=args.manifest_json,
    )
    _write_new_json(args.out_json, payload)
    print(
        "[ENTRY_SEQUENCE_INTEGRITY_AUDIT_PASS] "
        f"rows={payload['rows']} chain_sha256={payload['sequence_event_chain_sha256']} "
        f"out={args.out_json}"
    )


if __name__ == "__main__":
    main()
