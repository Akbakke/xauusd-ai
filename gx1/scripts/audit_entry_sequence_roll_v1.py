#!/usr/bin/env python3
"""Prove the nested Entry sequence is exactly the rolling snapshot chain.

The model-native split parquet stores both ``seq`` (96 consecutive rows) and
``snap`` (the final row).  A memory-efficient attended-smoke path may
reconstruct its sequence view from snapshots only *after* this auditor proves
every emitted row is the next M5 event and every sequence rolls by exactly one
snapshot.  Causally filtered splits with omitted M1-lifecycle rows are not
eligible for that optimisation; use ``audit_entry_sequence_integrity_v1`` to
verify their physical event chain instead.  Sampling is forbidden: this reads
the full supplied split and source-binds the conclusion to the exact parquet
and manifest bytes.

The audit has no feature transformation, target fit or model authority.  Its
only output is an immutable PASS proof suitable for a later source-bound
memory optimisation; candidate/promotion consumers must never treat it as
edge, TEST or live evidence.
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


SCHEMA_VERSION = "entry_model_native_sequence_roll_audit_v1"
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
        raise RuntimeError(f"[ENTRY_SEQUENCE_ROLL_{label}_PATH_INVALID]")
    try:
        resolved = supplied.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"[ENTRY_SEQUENCE_ROLL_{label}_PATH_INVALID]") from exc
    if resolved != supplied or not resolved.is_file():
        raise RuntimeError(f"[ENTRY_SEQUENCE_ROLL_{label}_PATH_INVALID]")
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


def _require_emitted_rows_contiguous(pf: pq.ParquetFile, *, rows: int) -> None:
    """Fail before nested reads when a split cannot be snap-reconstructed."""

    previous_time_ns: int | None = None
    observed_rows = 0
    for batch in pf.iter_batches(batch_size=8192, columns=["time"]):
        try:
            time_ns = batch.column("time").to_numpy(zero_copy_only=False)
            time_ns = time_ns.astype("datetime64[ns]").astype("int64", copy=False)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("[ENTRY_SEQUENCE_ROLL_TIME_INVALID]") from exc
        if np.any(time_ns == np.iinfo(np.int64).min):
            raise RuntimeError("[ENTRY_SEQUENCE_ROLL_TIME_NULL]")
        if previous_time_ns is not None:
            delta_ns = int(time_ns[0]) - previous_time_ns
            if delta_ns != _M5_NS:
                raise RuntimeError(
                    "[ENTRY_SEQUENCE_ROLL_EMITTED_ROWS_NONCONTIGUOUS] "
                    f"row={observed_rows} delta_ns={delta_ns}"
                )
        if len(time_ns) > 1:
            deltas = np.diff(time_ns)
            invalid = np.flatnonzero(deltas != _M5_NS)
            if len(invalid):
                row = observed_rows + int(invalid[0]) + 1
                raise RuntimeError(
                    "[ENTRY_SEQUENCE_ROLL_EMITTED_ROWS_NONCONTIGUOUS] "
                    f"row={row} delta_ns={int(deltas[int(invalid[0])])}"
                )
        previous_time_ns = int(time_ns[-1])
        observed_rows += len(time_ns)
    if observed_rows != rows:
        raise RuntimeError("[ENTRY_SEQUENCE_ROLL_ROW_COUNT_MISMATCH]")


def audit_sequence_roll(*, parquet_path: Path, manifest_path: Path) -> dict[str, Any]:
    """Audit every sequence transition in one exact split parquet."""

    parquet_path = _exact_regular_file(parquet_path, label="PARQUET")
    manifest_path = _exact_regular_file(manifest_path, label="MANIFEST")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("[ENTRY_SEQUENCE_ROLL_MANIFEST_JSON_INVALID]") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError("[ENTRY_SEQUENCE_ROLL_MANIFEST_JSON_INVALID]")

    pf = pq.ParquetFile(parquet_path)
    required = {"time", "seq", "snap"}
    columns = set(pf.schema_arrow.names)
    if not required.issubset(columns):
        raise RuntimeError(
            f"[ENTRY_SEQUENCE_ROLL_NESTED_COLUMNS_MISSING] {sorted(required - columns)}"
        )
    rows = int(pf.metadata.num_rows)
    if rows < 2:
        raise RuntimeError("[ENTRY_SEQUENCE_ROLL_ROWS_INVALID]")
    _require_emitted_rows_contiguous(pf, rows=rows)

    previous_sequence: np.ndarray | None = None
    observed_rows = 0
    sequence_hash = hashlib.sha256()
    sequence_hash.update(b"entry_model_native_sequence_roll_exact_v1\0")
    sequence_hash.update(np.asarray([rows, MODEL_NATIVE_SEQ_LEN, MODEL_NATIVE_SIGNAL_DIM], dtype="<i8").tobytes())
    for batch in pf.iter_batches(
        batch_size=_ARROW_BATCH_ROWS,
        columns=["seq", "snap"],
    ):
        count = int(batch.num_rows)
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
                f"[ENTRY_SEQUENCE_ROLL_NONFINITE] rows={observed_rows}:{observed_rows + count}"
            )
        if not np.array_equal(sequence[:, -1, :], snapshot):
            raise RuntimeError(
                f"[ENTRY_SEQUENCE_ROLL_LAST_SNAPSHOT_MISMATCH] rows={observed_rows}:{observed_rows + count}"
            )
        if previous_sequence is not None and not np.array_equal(
            sequence[0, :-1, :], previous_sequence[1:, :]
        ):
            raise RuntimeError(
                f"[ENTRY_SEQUENCE_ROLL_BATCH_BOUNDARY_MISMATCH] row={observed_rows}"
            )
        if count > 1 and not np.array_equal(
            sequence[1:, :-1, :], sequence[:-1, 1:, :]
        ):
            raise RuntimeError(
                f"[ENTRY_SEQUENCE_ROLL_ADJACENT_MISMATCH] rows={observed_rows}:{observed_rows + count}"
            )
        previous_sequence = sequence[-1].copy()
        sequence_hash.update(
            np.ascontiguousarray(sequence, dtype="<f4").tobytes(order="C")
        )
        sequence_hash.update(
            np.ascontiguousarray(snapshot, dtype="<f4").tobytes(order="C")
        )
        observed_rows += count
    if observed_rows != rows:
        raise RuntimeError("[ENTRY_SEQUENCE_ROLL_ROW_COUNT_MISMATCH]")

    return {
        "schema_version": SCHEMA_VERSION,
        "decision": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "parquet_path": str(parquet_path),
        "parquet_sha256": _sha256_file(parquet_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "rows": rows,
        "sequence_shape": [rows, MODEL_NATIVE_SEQ_LEN, MODEL_NATIVE_SIGNAL_DIM],
        "snapshot_shape": [rows, MODEL_NATIVE_SIGNAL_DIM],
        "checks": {
            "all_values_finite_float32": True,
            "every_seq_last_equals_snap_bit_identical": True,
            "every_adjacent_sequence_rolls_one_snapshot_bit_identical": True,
            "batch_boundary_rolls_bit_identical": True,
        },
        "sequence_snapshot_chain_sha256": sequence_hash.hexdigest(),
        "authority": {
            "data_reconstruction_only": True,
            "candidate": False,
            "test": False,
            "promotion": False,
            "paper": False,
            "live": False,
        },
    }


def _write_new_json(path: Path, payload: dict[str, Any]) -> None:
    supplied = Path(path).expanduser()
    if not supplied.is_absolute() or supplied.is_symlink() or supplied.exists():
        raise RuntimeError("[ENTRY_SEQUENCE_ROLL_OUTPUT_INVALID]")
    parent = supplied.parent
    if not parent.is_absolute() or parent.is_symlink() or not parent.is_dir():
        raise RuntimeError("[ENTRY_SEQUENCE_ROLL_OUTPUT_PARENT_INVALID]")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(supplied, flags, 0o644)
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
    parser = argparse.ArgumentParser("audit exact rolling seq/snap identity")
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = audit_sequence_roll(
        parquet_path=args.parquet,
        manifest_path=args.manifest_json,
    )
    _write_new_json(args.out_json, payload)
    print(
        "[ENTRY_SEQUENCE_ROLL_AUDIT_PASS] "
        f"rows={payload['rows']} chain_sha256={payload['sequence_snapshot_chain_sha256']} "
        f"out={args.out_json}"
    )


if __name__ == "__main__":
    main()
