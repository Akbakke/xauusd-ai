#!/usr/bin/env python3
"""Prove that an Entry split's seq/snap tensors equal its M5 feature surface.

This is deliberately distinct from the emitted-row roll audit.  Causal M1
supervision filtering can remove output rows, while the underlying M5 feature
surface remains the original consecutive event timeline.  The audit exhaustively
compares every stored sequence and snapshot with that immutable source before a
trainer may use source-backed windows as a storage optimisation.
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
    MODEL_NATIVE_CTX_CAT_DIM,
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_SEQ_LEN,
    MODEL_NATIVE_SIGNAL_DIM,
)
from gx1.contracts.entry_sequence_source_reconstruction_v1 import (
    AUTHORITY,
    REQUIRED_CHECKS,
    SCHEMA_VERSION,
    feature_surface_binding_from_split_manifest,
)


_SHA256_BUFFER_BYTES = 1024 * 1024
_ARROW_BATCH_ROWS = 256
_M5_NS = 5 * 60 * 1_000_000_000
_SURFACE_COLUMNS = ("time", "signal", "ctx_cont", "ctx_cat")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_SHA256_BUFFER_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _exact_regular_file(raw: Path, *, label: str) -> Path:
    supplied = Path(raw).expanduser()
    if not supplied.is_absolute() or supplied.is_symlink():
        raise RuntimeError(
            f"[ENTRY_SEQUENCE_SOURCE_RECONSTRUCTION_{label}_PATH_INVALID]"
        )
    try:
        resolved = supplied.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(
            f"[ENTRY_SEQUENCE_SOURCE_RECONSTRUCTION_{label}_PATH_INVALID]"
        ) from exc
    if resolved != supplied or not resolved.is_file():
        raise RuntimeError(
            f"[ENTRY_SEQUENCE_SOURCE_RECONSTRUCTION_{label}_PATH_INVALID]"
        )
    return resolved


def _time_ns(column: Any, *, context: str) -> np.ndarray:
    try:
        values = column.to_numpy(zero_copy_only=False).astype("datetime64[ns]")
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"[ENTRY_SEQUENCE_SOURCE_RECONSTRUCTION_{context}_TIME_INVALID]"
        ) from exc
    values = values.astype(np.int64, copy=False)
    if np.any(values == np.iinfo(np.int64).min):
        raise RuntimeError(
            f"[ENTRY_SEQUENCE_SOURCE_RECONSTRUCTION_{context}_TIME_NULL]"
        )
    return values


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f"duplicate key {key}")
            out[key] = value
        return out

    try:
        payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_keys)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise RuntimeError(
            f"[ENTRY_SEQUENCE_SOURCE_RECONSTRUCTION_{label}_JSON_INVALID]"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"[ENTRY_SEQUENCE_SOURCE_RECONSTRUCTION_{label}_JSON_INVALID]"
        )
    return payload


def _feature_surface_from_manifest(manifest: dict[str, Any]) -> tuple[dict[str, Any], Path, Path]:
    binding = feature_surface_binding_from_split_manifest(manifest)
    surface_path = _exact_regular_file(Path(binding["path"]), label="FEATURE_SURFACE")
    surface_manifest_path = _exact_regular_file(
        Path(binding["manifest_path"]), label="FEATURE_SURFACE_MANIFEST"
    )
    if _sha256_file(surface_path) != binding["sha256"]:
        raise RuntimeError(
            "[ENTRY_SEQUENCE_SOURCE_RECONSTRUCTION_FEATURE_SURFACE_HASH_MISMATCH]"
        )
    if _sha256_file(surface_manifest_path) != binding["manifest_sha256"]:
        raise RuntimeError(
            "[ENTRY_SEQUENCE_SOURCE_RECONSTRUCTION_FEATURE_SURFACE_MANIFEST_HASH_MISMATCH]"
        )
    surface_manifest = _read_json(surface_manifest_path, label="FEATURE_SURFACE_MANIFEST")
    if (
        surface_manifest.get("output_parquet") != str(surface_path)
        or surface_manifest.get("output_parquet_sha256") != binding["sha256"]
        or surface_manifest.get("rows") != binding["rows"]
        or surface_manifest.get("signal_dim") != MODEL_NATIVE_SIGNAL_DIM
        or surface_manifest.get("ctx_cont_dim") != MODEL_NATIVE_CTX_CONT_DIM
        or surface_manifest.get("ctx_cat_dim") != MODEL_NATIVE_CTX_CAT_DIM
        or surface_manifest.get("schema_version") != binding["schema_version"]
    ):
        raise RuntimeError(
            "[ENTRY_SEQUENCE_SOURCE_RECONSTRUCTION_FEATURE_SURFACE_MANIFEST_CONTRACT_INVALID]"
        )
    return binding, surface_path, surface_manifest_path


def _load_surface_signal(surface_path: Path, *, expected_rows: int) -> tuple[np.ndarray, np.ndarray]:
    feature_surface = pq.ParquetFile(surface_path)
    if tuple(feature_surface.schema_arrow.names) != _SURFACE_COLUMNS:
        raise RuntimeError(
            "[ENTRY_SEQUENCE_SOURCE_RECONSTRUCTION_FEATURE_SURFACE_SCHEMA_INVALID]"
        )
    rows = int(feature_surface.metadata.num_rows)
    if rows != expected_rows or rows < MODEL_NATIVE_SEQ_LEN:
        raise RuntimeError(
            "[ENTRY_SEQUENCE_SOURCE_RECONSTRUCTION_FEATURE_SURFACE_ROWS_INVALID]"
        )
    time_table = feature_surface.read(columns=["time"]).combine_chunks()
    time_ns = _time_ns(time_table.column("time"), context="FEATURE_SURFACE")
    if (
        time_ns.shape != (rows,)
        or np.any(np.diff(time_ns) <= 0)
        or np.any(np.diff(time_ns) % _M5_NS != 0)
    ):
        raise RuntimeError(
            "[ENTRY_SEQUENCE_SOURCE_RECONSTRUCTION_FEATURE_SURFACE_TIME_INVALID]"
        )
    signal = np.empty((rows, MODEL_NATIVE_SIGNAL_DIM), dtype=np.float32)
    offset = 0
    for batch in feature_surface.iter_batches(
        batch_size=_ARROW_BATCH_ROWS,
        columns=["signal"],
        use_threads=False,
    ):
        count = int(batch.num_rows)
        values = batch.column("signal")
        if not hasattr(values, "values"):
            raise RuntimeError(
                "[ENTRY_SEQUENCE_SOURCE_RECONSTRUCTION_FEATURE_SURFACE_SIGNAL_DECODE_INVALID]"
            )
        flat = values.values.to_numpy(zero_copy_only=False)
        if flat.shape != (count * MODEL_NATIVE_SIGNAL_DIM,):
            raise RuntimeError(
                "[ENTRY_SEQUENCE_SOURCE_RECONSTRUCTION_FEATURE_SURFACE_SIGNAL_WIDTH_INVALID]"
            )
        decoded = np.asarray(flat, dtype=np.float32).reshape(
            count, MODEL_NATIVE_SIGNAL_DIM
        )
        if not np.isfinite(decoded).all():
            raise RuntimeError(
                "[ENTRY_SEQUENCE_SOURCE_RECONSTRUCTION_FEATURE_SURFACE_SIGNAL_NONFINITE]"
            )
        signal[offset : offset + count] = decoded
        offset += count
    if offset != rows:
        raise RuntimeError(
            "[ENTRY_SEQUENCE_SOURCE_RECONSTRUCTION_FEATURE_SURFACE_ROW_COUNT_MISMATCH]"
        )
    return time_ns, signal


def audit_sequence_source_reconstruction(
    *, parquet_path: Path, manifest_path: Path
) -> dict[str, Any]:
    """Exhaustively bind stored windows to the declared M5 feature surface."""

    parquet_path = _exact_regular_file(parquet_path, label="PARQUET")
    manifest_path = _exact_regular_file(manifest_path, label="MANIFEST")
    manifest = _read_json(manifest_path, label="MANIFEST")
    binding, surface_path, surface_manifest_path = _feature_surface_from_manifest(manifest)
    source_time_ns, source_signal = _load_surface_signal(
        surface_path, expected_rows=int(binding["rows"])
    )

    split = pq.ParquetFile(parquet_path)
    if not {"time", "seq", "snap"}.issubset(split.schema_arrow.names):
        raise RuntimeError("[ENTRY_SEQUENCE_SOURCE_RECONSTRUCTION_SPLIT_COLUMNS_MISSING]")
    rows = int(split.metadata.num_rows)
    if rows < 2:
        raise RuntimeError("[ENTRY_SEQUENCE_SOURCE_RECONSTRUCTION_SPLIT_ROWS_INVALID]")

    digest = hashlib.sha256()
    digest.update(b"entry_model_native_sequence_source_reconstruction_v1\0")
    digest.update(
        np.asarray([rows, MODEL_NATIVE_SEQ_LEN, MODEL_NATIVE_SIGNAL_DIM], dtype="<i8").tobytes()
    )
    observed = 0
    history_offsets = np.arange(MODEL_NATIVE_SEQ_LEN, dtype=np.int64) - (
        MODEL_NATIVE_SEQ_LEN - 1
    )
    for batch in split.iter_batches(
        batch_size=_ARROW_BATCH_ROWS,
        columns=["time", "seq", "snap"],
        use_threads=False,
    ):
        count = int(batch.num_rows)
        times = _time_ns(batch.column("time"), context="SPLIT")
        positions = np.searchsorted(source_time_ns, times).astype(np.int64, copy=False)
        if (
            np.any(positions < MODEL_NATIVE_SEQ_LEN - 1)
            or np.any(positions >= len(source_time_ns))
            or not np.array_equal(source_time_ns[positions], times)
            or (observed > 0 and int(times[0]) <= previous_time)
            or (count > 1 and np.any(np.diff(times) <= 0))
        ):
            raise RuntimeError(
                "[ENTRY_SEQUENCE_SOURCE_RECONSTRUCTION_SPLIT_TIME_MAPPING_INVALID]"
            )
        sequence = (
            batch.column("seq").flatten().flatten().to_numpy(zero_copy_only=False)
            .reshape(count, MODEL_NATIVE_SEQ_LEN, MODEL_NATIVE_SIGNAL_DIM)
            .astype(np.float32, copy=False)
        )
        snapshot = (
            batch.column("snap").flatten().to_numpy(zero_copy_only=False)
            .reshape(count, MODEL_NATIVE_SIGNAL_DIM)
            .astype(np.float32, copy=False)
        )
        if not np.isfinite(sequence).all() or not np.isfinite(snapshot).all():
            raise RuntimeError(
                f"[ENTRY_SEQUENCE_SOURCE_RECONSTRUCTION_SPLIT_NONFINITE] rows={observed}:{observed + count}"
            )
        expected = source_signal[positions[:, None] + history_offsets[None, :]]
        if not np.array_equal(sequence, expected):
            raise RuntimeError(
                f"[ENTRY_SEQUENCE_SOURCE_RECONSTRUCTION_SEQUENCE_MISMATCH] rows={observed}:{observed + count}"
            )
        if not np.array_equal(snapshot, source_signal[positions]):
            raise RuntimeError(
                f"[ENTRY_SEQUENCE_SOURCE_RECONSTRUCTION_SNAPSHOT_MISMATCH] rows={observed}:{observed + count}"
            )
        digest.update(np.ascontiguousarray(times, dtype="<i8").tobytes(order="C"))
        digest.update(np.ascontiguousarray(positions, dtype="<i8").tobytes(order="C"))
        digest.update(np.ascontiguousarray(sequence, dtype="<f4").tobytes(order="C"))
        digest.update(np.ascontiguousarray(snapshot, dtype="<f4").tobytes(order="C"))
        previous_time = int(times[-1])
        observed += count
    if observed != rows:
        raise RuntimeError("[ENTRY_SEQUENCE_SOURCE_RECONSTRUCTION_SPLIT_ROW_COUNT_MISMATCH]")
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "parquet_path": str(parquet_path),
        "parquet_sha256": _sha256_file(parquet_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "feature_surface_path": str(surface_path),
        "feature_surface_sha256": binding["sha256"],
        "feature_surface_manifest_path": str(surface_manifest_path),
        "feature_surface_manifest_sha256": binding["manifest_sha256"],
        "feature_surface_rows": int(binding["rows"]),
        "rows": rows,
        "sequence_shape": [rows, MODEL_NATIVE_SEQ_LEN, MODEL_NATIVE_SIGNAL_DIM],
        "snapshot_shape": [rows, MODEL_NATIVE_SIGNAL_DIM],
        "checks": dict(REQUIRED_CHECKS),
        "sequence_source_chain_sha256": digest.hexdigest(),
        "authority": dict(AUTHORITY),
    }


def _write_new_json(path: Path, payload: dict[str, Any]) -> None:
    target = Path(path).expanduser()
    if not target.is_absolute() or target.exists() or target.is_symlink():
        raise RuntimeError("[ENTRY_SEQUENCE_SOURCE_RECONSTRUCTION_OUTPUT_PATH_INVALID]")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("audit exact source-backed Entry sequences")
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = audit_sequence_source_reconstruction(
        parquet_path=args.parquet, manifest_path=args.manifest_json
    )
    _write_new_json(args.out_json, payload)
    print(
        "[ENTRY_SEQUENCE_SOURCE_RECONSTRUCTION_PASS] "
        f"rows={payload['rows']} feature_surface_rows={payload['feature_surface_rows']} "
        f"out={args.out_json}"
    )


if __name__ == "__main__":
    main()
