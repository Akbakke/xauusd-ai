#!/usr/bin/env python3
"""Materialize the exact full-input liveness contract for one fresh dataset.

The scan is deliberately exhaustive.  Signal statistics are computed from the
513-wide snapshot surface (one value per emitted row), while every value in the
96x513 sequence is also checked for shape, finiteness and exact last-step parity
with ``snap``.  The artifact binds all split manifests, the dataset build proof
and the stat identity of every fully scanned parquet.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from gx1.contracts.entry_full_input_liveness_v1 import (
    INTEGER_TOLERANCE,
    PASS_DECISION,
    SPLITS,
    build_full_input_liveness_artifact,
    sha256_file,
    validate_full_input_liveness_artifact,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_CTX_CAT_DIM,
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_DIRECTION_LOGIT_MODE,
    MODEL_NATIVE_SEQ_LEN,
    MODEL_NATIVE_SIGNAL_DIM,
    MODEL_NATIVE_SIGNAL_SCHEMA_VERSION,
    MODEL_NATIVE_SPLIT_MANIFEST_SCHEMA_VERSION,
    require_model_native_signal_contract,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CTX_CAT_FIELDS,
    MODEL_NATIVE_CTX_CONT_FIELDS,
)
from gx1.contracts.entry_run_lineage_v1 import require_entry_run_id
from gx1.features.htf_features import (
    load_multi_tf_v4_cache,
    require_multi_tf_v4_liveness_contract,
)


OUTPUT_PREFIX = "ENTRY_FULL_INPUT_LIVENESS_CONTRACT"
OUTPUT_FILENAME_RE = re.compile(
    rf"{OUTPUT_PREFIX}_\d{{8}}T\d{{6}}(?:\d{{6}})?Z\.json"
)
PRODUCER_SCHEMA_VERSION = "entry_full_input_liveness_materializer_v5"
DEFAULT_BATCH_SIZE = 512
REQUIRED_COLUMNS = ("seq", "snap", "ctx_cont", "ctx_cat")


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{label}_MISSING_REGULAR_FILE: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label}_INVALID_JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label}_ROOT_NOT_OBJECT: {path}")
    return payload


def _resolved_path(value: object) -> Path:
    text = str(value or "").strip()
    if not text:
        raise RuntimeError("EMPTY_PATH_IN_PROVENANCE")
    return Path(text).expanduser().resolve()


class _ColumnStats:
    """Numerically stable streaming statistics for one two-dimensional surface."""

    def __init__(self, width: int, *, categorical: bool = False) -> None:
        self.width = int(width)
        self.categorical = bool(categorical)
        self.row_count = 0
        self.finite_count = np.zeros(self.width, dtype=np.int64)
        self.nonfinite_count = np.zeros(self.width, dtype=np.int64)
        self.mean = np.zeros(self.width, dtype=np.float64)
        self.m2 = np.zeros(self.width, dtype=np.float64)
        self.minimum = np.full(self.width, np.inf, dtype=np.float64)
        self.maximum = np.full(self.width, -np.inf, dtype=np.float64)
        self.active_count = np.zeros(self.width, dtype=np.int64)
        self.integer_like_count = np.zeros(self.width, dtype=np.int64)
        self.unique_values = [set() for _ in range(self.width)] if categorical else []

    def update(self, values: np.ndarray) -> None:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != self.width:
            raise RuntimeError(
                f"STATS_MATRIX_SHAPE_INVALID: got={list(matrix.shape)} "
                f"expected=[rows,{self.width}]"
            )
        rows = int(matrix.shape[0])
        self.row_count += rows
        finite = np.isfinite(matrix)
        batch_count = finite.sum(axis=0, dtype=np.int64)
        self.finite_count += batch_count
        self.nonfinite_count += rows - batch_count
        self.active_count += (finite & (np.abs(matrix) > 1e-7)).sum(axis=0, dtype=np.int64)

        safe = np.where(finite, matrix, 0.0)
        batch_sum = safe.sum(axis=0, dtype=np.float64)
        batch_mean = np.divide(
            batch_sum,
            batch_count,
            out=np.zeros(self.width, dtype=np.float64),
            where=batch_count > 0,
        )
        centered = np.where(finite, matrix - batch_mean, 0.0)
        batch_m2 = np.square(centered).sum(axis=0, dtype=np.float64)
        previous_count = self.finite_count - batch_count
        combined_count = previous_count + batch_count
        delta = batch_mean - self.mean
        valid = batch_count > 0
        self.mean[valid] += delta[valid] * batch_count[valid] / combined_count[valid]
        self.m2[valid] += (
            batch_m2[valid]
            + np.square(delta[valid])
            * previous_count[valid]
            * batch_count[valid]
            / combined_count[valid]
        )

        batch_min = np.min(np.where(finite, matrix, np.inf), axis=0)
        batch_max = np.max(np.where(finite, matrix, -np.inf), axis=0)
        self.minimum = np.minimum(self.minimum, batch_min)
        self.maximum = np.maximum(self.maximum, batch_max)

        if self.categorical:
            rounded = np.rint(matrix)
            self.integer_like_count += (
                finite & (np.abs(matrix - rounded) <= INTEGER_TOLERANCE)
            ).sum(axis=0, dtype=np.int64)
            for index in range(self.width):
                if batch_count[index]:
                    self.unique_values[index].update(matrix[finite[:, index], index].tolist())

    def finalize(self, fields: list[str]) -> dict[str, dict[str, Any]]:
        if len(fields) != self.width:
            raise RuntimeError(
                f"STATS_FIELD_WIDTH_INVALID: fields={len(fields)} expected={self.width}"
            )
        result: dict[str, dict[str, Any]] = {}
        for index, field in enumerate(fields):
            count = int(self.finite_count[index])
            minimum = float(self.minimum[index]) if count else 0.0
            maximum = float(self.maximum[index]) if count else 0.0
            variance = float(self.m2[index]) / count if count else 0.0
            row = {
                "row_count": int(self.row_count),
                "finite_count": count,
                "nonfinite_count": int(self.nonfinite_count[index]),
                "mean": float(self.mean[index]) if count else 0.0,
                "std": math.sqrt(max(variance, 0.0)),
                "min": minimum,
                "max": maximum,
                "value_range": max(maximum - minimum, 0.0),
                "active_count": int(self.active_count[index]),
                "active_rate": (
                    float(self.active_count[index]) / self.row_count
                    if self.row_count > 0
                    else 0.0
                ),
            }
            if self.categorical:
                row.update(
                    {
                        "unique_count": len(self.unique_values[index]),
                        "integer_like_count": int(self.integer_like_count[index]),
                        "unique_values": sorted(
                            int(round(value)) for value in self.unique_values[index]
                        ),
                    }
                )
            result[str(field)] = row
        return result


def _stack_column(
    batch: Any,
    name: str,
    *,
    dtype: np.dtype[Any],
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    """Expose a regular nested Arrow list column as one validated NumPy tensor.

    ``to_pylist()`` materialized every scalar as a Python object before NumPy
    converted it back to a number.  On the 96x513 signal surface that made the
    exhaustive liveness scan CPU-bound for more than an hour.  Parquet already
    decoded the primitive child buffer, so validate every list offset and read
    that buffer directly instead.  This is still a full scan: no row, timestep,
    signal, context value, finiteness check, or seq/snap parity check is sampled.
    """

    index = batch.schema.get_field_index(name)
    if index < 0:
        raise RuntimeError(f"PARQUET_COLUMN_MISSING_DURING_SCAN: {name}")
    if not expected_shape or expected_shape[0] != int(batch.num_rows):
        raise RuntimeError(
            f"PARQUET_COLUMN_EXPECTED_SHAPE_INVALID: {name}: {list(expected_shape)}"
        )

    values = batch.column(index)
    try:
        for depth, width in enumerate(expected_shape[1:], start=1):
            if not (pa.types.is_list(values.type) or pa.types.is_large_list(values.type)):
                raise RuntimeError(
                    f"PARQUET_COLUMN_LIST_DEPTH_INVALID: {name}: depth={depth} "
                    f"type={values.type}"
                )
            if values.null_count:
                raise RuntimeError(
                    f"PARQUET_COLUMN_LIST_NULL_INVALID: {name}: depth={depth}"
                )
            offsets = values.offsets.to_numpy(zero_copy_only=False).astype(
                np.int64, copy=False
            )
            expected_offsets = offsets[0] + np.arange(
                len(offsets), dtype=np.int64
            ) * int(width)
            if not np.array_equal(offsets, expected_offsets):
                raise RuntimeError(
                    f"PARQUET_COLUMN_LIST_WIDTH_INVALID: {name}: depth={depth} "
                    f"expected_width={width}"
                )
            start = int(offsets[0])
            stop = int(offsets[-1])
            values = values.values.slice(start, stop - start)

        if values.null_count:
            raise RuntimeError(f"PARQUET_COLUMN_VALUE_NULL_INVALID: {name}")
        flat = values.to_numpy(zero_copy_only=False)
        expected_size = int(np.prod(expected_shape, dtype=np.int64))
        if int(flat.size) != expected_size:
            raise RuntimeError(
                f"PARQUET_COLUMN_VALUE_COUNT_INVALID: {name}: "
                f"got={flat.size} expected={expected_size}"
            )
        return np.asarray(flat, dtype=dtype).reshape(expected_shape)
    except RuntimeError:
        raise
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"PARQUET_COLUMN_STACK_FAILED: {name}: {exc}") from exc


def _validate_split_manifest(
    *,
    manifest_path: Path,
    parquet_path: Path,
    entry_run_id: str,
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    if not parquet_path.is_file() or parquet_path.is_symlink():
        raise RuntimeError(f"SPLIT_PARQUET_MISSING_REGULAR_FILE: {parquet_path}")
    manifest = _load_json(manifest_path, label="SPLIT_MANIFEST")
    if manifest.get("schema_version") != MODEL_NATIVE_SPLIT_MANIFEST_SCHEMA_VERSION:
        raise RuntimeError(
            "SPLIT_MANIFEST_SCHEMA_INVALID: "
            f"{manifest_path}: {manifest.get('schema_version')!r}"
        )
    if manifest.get("manifest_variant") != MODEL_NATIVE_CONTRACT_MODE:
        raise RuntimeError(f"SPLIT_MANIFEST_MODE_INVALID: {manifest_path}")
    if int(manifest.get("expected_seq_snap_width") or -1) != MODEL_NATIVE_SIGNAL_DIM:
        raise RuntimeError(f"SPLIT_MANIFEST_SIGNAL_WIDTH_INVALID: {manifest_path}")
    if _resolved_path(manifest.get("output_data_path")) != parquet_path:
        raise RuntimeError(f"SPLIT_MANIFEST_OUTPUT_PATH_MISMATCH: {manifest_path}")

    extra = manifest.get("extra") if isinstance(manifest.get("extra"), dict) else {}
    if extra.get("contract_mode") != MODEL_NATIVE_CONTRACT_MODE:
        raise RuntimeError(f"SPLIT_EXTRA_MODE_INVALID: {manifest_path}")
    if extra.get("direction_logit_mode") != MODEL_NATIVE_DIRECTION_LOGIT_MODE:
        raise RuntimeError(f"SPLIT_DIRECTION_MODE_INVALID: {manifest_path}")
    if str(extra.get("entry_run_id") or "").strip() != entry_run_id:
        raise RuntimeError(f"SPLIT_RUN_ID_MISMATCH: {manifest_path}")
    state = (
        extra.get("model_native_state_contract")
        if isinstance(extra.get("model_native_state_contract"), dict)
        else {}
    )
    if str(state.get("entry_run_id") or "").strip() != entry_run_id:
        raise RuntimeError(f"SPLIT_STATE_RUN_ID_MISMATCH: {manifest_path}")

    signal = (
        extra.get("model_native_signal_contract")
        if isinstance(extra.get("model_native_signal_contract"), dict)
        else {}
    )
    require_model_native_signal_contract(signal, context="FULL_INPUT_LIVENESS")
    fields = [str(name) for name in signal.get("fields", [])]
    bridge = extra.get("signal_bridge") if isinstance(extra.get("signal_bridge"), dict) else {}
    if (
        bridge.get("id") != MODEL_NATIVE_SIGNAL_SCHEMA_VERSION
        or bridge.get("fields") != fields
        or int(bridge.get("seq_input_dim") or -1) != MODEL_NATIVE_SIGNAL_DIM
        or int(bridge.get("snap_input_dim") or -1) != MODEL_NATIVE_SIGNAL_DIM
        or bridge.get("bridge_dim") != 0
        or bridge.get("bridge_source") is not None
    ):
        raise RuntimeError(f"SPLIT_SIGNAL_SURFACE_INVALID: {manifest_path}")

    ctx = extra.get("ctx_contract") if isinstance(extra.get("ctx_contract"), dict) else {}
    ctx_cont = [str(name) for name in ctx.get("ctx_cont_names", [])]
    ctx_cat = [str(name) for name in ctx.get("ctx_cat_names", [])]
    if (
        ctx_cont != list(MODEL_NATIVE_CTX_CONT_FIELDS)
        or ctx_cat != list(MODEL_NATIVE_CTX_CAT_FIELDS)
        or int(ctx.get("ctx_cont_dim") or -1) != MODEL_NATIVE_CTX_CONT_DIM
        or int(ctx.get("ctx_cat_dim") or -1) != MODEL_NATIVE_CTX_CAT_DIM
    ):
        raise RuntimeError(f"SPLIT_CTX_SURFACE_INVALID: {manifest_path}")
    return manifest, {"signal": fields, "ctx_cont": ctx_cont, "ctx_cat": ctx_cat}


def _scan_split(
    parquet_path: Path,
    *,
    field_order: Mapping[str, list[str]],
    batch_size: int,
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any], dict[str, Any]]:
    parquet = pq.ParquetFile(parquet_path)
    schema_names = set(parquet.schema_arrow.names)
    missing = sorted(set(REQUIRED_COLUMNS) - schema_names)
    if missing:
        raise RuntimeError(f"FULL_INPUT_SCAN_COLUMNS_MISSING: {parquet_path}: {missing}")
    total_rows = int(parquet.metadata.num_rows or 0)
    signal_stats = _ColumnStats(MODEL_NATIVE_SIGNAL_DIM)
    cont_stats = _ColumnStats(MODEL_NATIVE_CTX_CONT_DIM)
    cat_stats = _ColumnStats(MODEL_NATIVE_CTX_CAT_DIM, categorical=True)
    errors: list[str] = []
    scanned_rows = 0
    seq_values_scanned = 0

    for batch_index, batch in enumerate(
        parquet.iter_batches(batch_size=int(batch_size), columns=list(REQUIRED_COLUMNS))
    ):
        try:
            rows = int(batch.num_rows)
            expected_shapes = {
                "seq": (rows, MODEL_NATIVE_SEQ_LEN, MODEL_NATIVE_SIGNAL_DIM),
                "snap": (rows, MODEL_NATIVE_SIGNAL_DIM),
                "ctx_cont": (rows, MODEL_NATIVE_CTX_CONT_DIM),
                "ctx_cat": (rows, MODEL_NATIVE_CTX_CAT_DIM),
            }
            seq = _stack_column(
                batch,
                "seq",
                dtype=np.float64,
                expected_shape=expected_shapes["seq"],
            )
            snap = _stack_column(
                batch,
                "snap",
                dtype=np.float64,
                expected_shape=expected_shapes["snap"],
            )
            ctx_cont = _stack_column(
                batch,
                "ctx_cont",
                dtype=np.float64,
                expected_shape=expected_shapes["ctx_cont"],
            )
            ctx_cat = _stack_column(
                batch,
                "ctx_cat",
                dtype=np.float64,
                expected_shape=expected_shapes["ctx_cat"],
            )
            observed = {
                "seq": tuple(seq.shape),
                "snap": tuple(snap.shape),
                "ctx_cont": tuple(ctx_cont.shape),
                "ctx_cat": tuple(ctx_cat.shape),
            }
            for surface, expected in expected_shapes.items():
                if observed[surface] != expected:
                    raise RuntimeError(
                        f"{surface}_SHAPE_INVALID: got={list(observed[surface])} "
                        f"expected={list(expected)}"
                    )
            if not np.isfinite(seq).all():
                errors.append(f"batch={batch_index}: seq contains nonfinite values")
            if not np.array_equal(seq[:, -1, :], snap):
                errors.append(f"batch={batch_index}: seq last step is not exactly snap")
            signal_stats.update(snap)
            cont_stats.update(ctx_cont)
            cat_stats.update(ctx_cat)
            scanned_rows += rows
            seq_values_scanned += int(seq.size)
        except RuntimeError as exc:
            errors.append(f"batch={batch_index}: {exc}")
            break

    scan_complete = bool(scanned_rows == total_rows and total_rows > 0 and not errors)
    stat = parquet_path.stat()
    proof = {
        "parquet_path": str(parquet_path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "total_rows": total_rows,
        "scanned_rows": scanned_rows,
        "fullscan": bool(scanned_rows == total_rows and total_rows > 0),
        "scan_complete": scan_complete,
    }
    semantic_proof = {
        "parquet_path": str(parquet_path),
        "total_rows": total_rows,
        "scanned_rows": scanned_rows,
        "seq_values_scanned": seq_values_scanned,
        "seq_len": MODEL_NATIVE_SEQ_LEN,
        "signal_width": MODEL_NATIVE_SIGNAL_DIM,
        "signal_stats_source": "snap_one_value_per_emitted_row",
        "all_seq_values_finite": not any("nonfinite" in error for error in errors),
        "seq_last_exactly_equals_snap": not any("exactly snap" in error for error in errors),
        "scan_complete": scan_complete,
        "errors": errors,
    }
    stats = {
        "signal": signal_stats.finalize(list(field_order["signal"])),
        "ctx_cont": cont_stats.finalize(list(field_order["ctx_cont"])),
        "ctx_cat": cat_stats.finalize(list(field_order["ctx_cat"])),
    }
    return stats, proof, semantic_proof


def _validate_build_proof(
    *,
    proof_path: Path,
    dataset_dir: Path,
    stem: str,
    entry_run_id: str,
    field_order: Mapping[str, list[str]],
) -> dict[str, Any]:
    proof = _load_json(proof_path, label="DATASET_BUILD_PROOF")
    if str(proof.get("entry_run_id") or "").strip() != entry_run_id:
        raise RuntimeError("DATASET_BUILD_PROOF_RUN_ID_MISMATCH")
    if proof.get("contract_mode") != MODEL_NATIVE_CONTRACT_MODE:
        raise RuntimeError("DATASET_BUILD_PROOF_MODE_MISMATCH")
    if _resolved_path(proof.get("output_path")) != dataset_dir / f"{stem}.parquet":
        raise RuntimeError("DATASET_BUILD_PROOF_OUTPUT_PATH_MISMATCH")
    signal = proof.get("model_native_signal_contract")
    if not isinstance(signal, dict):
        raise RuntimeError("DATASET_BUILD_PROOF_SIGNAL_CONTRACT_MISSING")
    require_model_native_signal_contract(signal, context="FULL_INPUT_LIVENESS_BUILD_PROOF")
    if signal.get("fields") != field_order["signal"]:
        raise RuntimeError("DATASET_BUILD_PROOF_SIGNAL_ORDER_MISMATCH")
    ctx = proof.get("ctx_contract") if isinstance(proof.get("ctx_contract"), dict) else {}
    if (
        ctx.get("ctx_cont_names") != field_order["ctx_cont"]
        or ctx.get("ctx_cat_names") != field_order["ctx_cat"]
    ):
        raise RuntimeError("DATASET_BUILD_PROOF_CTX_ORDER_MISMATCH")
    state = (
        proof.get("model_native_state_contract")
        if isinstance(proof.get("model_native_state_contract"), dict)
        else {}
    )
    if str(state.get("entry_run_id") or "").strip() != entry_run_id:
        raise RuntimeError("DATASET_BUILD_PROOF_STATE_RUN_ID_MISMATCH")
    return proof


def run(args: argparse.Namespace) -> dict[str, Any]:
    entry_run_id = require_entry_run_id(getattr(args, "run_id", None))
    dataset_dir = Path(str(args.dataset_dir)).expanduser().resolve()
    stem = str(args.stem or "").strip()
    out_path = Path(str(args.out_json)).expanduser().resolve()
    mtf_cache_dir = Path(str(args.mtf_cache_dir)).expanduser().resolve()
    batch_size = int(getattr(args, "batch_size", DEFAULT_BATCH_SIZE))
    if not dataset_dir.is_dir() or dataset_dir.is_symlink():
        raise RuntimeError(f"DATASET_DIR_MISSING_REGULAR_DIRECTORY: {dataset_dir}")
    if not stem or Path(stem).name != stem or stem.endswith(".parquet"):
        raise RuntimeError(f"DATASET_STEM_INVALID: {stem!r}")
    if OUTPUT_FILENAME_RE.fullmatch(out_path.name) is None:
        raise RuntimeError(
            f"LIVENESS_OUTPUT_FILENAME_INVALID: got={out_path.name!r} "
            f"expected={OUTPUT_PREFIX}_<UTC_TIMESTAMP>.json"
        )
    if out_path.exists() or out_path.is_symlink():
        raise RuntimeError(f"LIVENESS_OUTPUT_ALREADY_EXISTS: {out_path}")
    if batch_size < 1 or batch_size > 2048:
        raise RuntimeError(f"LIVENESS_BATCH_SIZE_INVALID: {batch_size}")
    if not mtf_cache_dir.is_dir() or mtf_cache_dir.is_symlink():
        raise RuntimeError(
            f"MULTI_TF_CACHE_DIR_MISSING_REGULAR_DIRECTORY: {mtf_cache_dir}"
        )

    manifests: dict[str, Path] = {}
    parquets: dict[str, Path] = {}
    manifest_payloads: dict[str, dict[str, Any]] = {}
    manifest_bindings: dict[str, dict[str, str]] = {}
    canonical_field_order: dict[str, list[str]] | None = None
    for split in SPLITS:
        parquet_path = (dataset_dir / f"{stem}_{split}.parquet").resolve()
        manifest_path = (dataset_dir / f"{stem}_{split}.manifest.json").resolve()
        manifest, field_order = _validate_split_manifest(
            manifest_path=manifest_path,
            parquet_path=parquet_path,
            entry_run_id=entry_run_id,
        )
        if canonical_field_order is None:
            canonical_field_order = field_order
        elif field_order != canonical_field_order:
            raise RuntimeError(f"SPLIT_FIELD_ORDER_MISMATCH: {split}")
        manifests[split] = manifest_path
        parquets[split] = parquet_path
        manifest_payloads[split] = manifest
        manifest_bindings[split] = {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        }
    if canonical_field_order is None:  # pragma: no cover - SPLITS is immutable and non-empty
        raise RuntimeError("NO_SPLIT_FIELD_ORDER")

    proof_path = (dataset_dir / "DATASET_BUILD_PROOF.json").resolve()
    _validate_build_proof(
        proof_path=proof_path,
        dataset_dir=dataset_dir,
        stem=stem,
        entry_run_id=entry_run_id,
        field_order=canonical_field_order,
    )

    stats_by_split: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    scan_proof: dict[str, dict[str, Any]] = {}
    semantic_scan: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        stats, proof, semantics = _scan_split(
            parquets[split],
            field_order=canonical_field_order,
            batch_size=batch_size,
        )
        stats_by_split[split] = stats
        scan_proof[split] = proof
        semantic_scan[split] = semantics

    # The cache loader already verifies every array byte, recomputes the exact
    # 5-per-TF (MULTI_TF_FEATURE_COUNT_V4-wide) liveness contract, and rejects false manifest claims. Reuse that
    # single owner instead of scanning and defining the same proof again here.
    load_multi_tf_v4_cache(mtf_cache_dir)
    mtf_manifest_path = (mtf_cache_dir / "manifest.json").resolve()
    mtf_manifest = _load_json(
        mtf_manifest_path,
        label="MULTI_TF_CACHE_MANIFEST",
    )
    mtf_cache_binding = {
        "manifest_path": str(mtf_manifest_path),
        "manifest_sha256": sha256_file(mtf_manifest_path),
        "cache_identity_sha256": str(
            mtf_manifest.get("cache_identity_sha256") or ""
        ),
    }
    mtf_liveness_contract = require_multi_tf_v4_liveness_contract(
        mtf_manifest.get("full_input_liveness")
    )

    artifact = build_full_input_liveness_artifact(
        dataset_dir=dataset_dir,
        contract_mode=MODEL_NATIVE_CONTRACT_MODE,
        field_order=canonical_field_order,
        stats_by_split=stats_by_split,
        manifest_bindings=manifest_bindings,
        scan_proof_by_split=scan_proof,
        multi_tf_liveness_contract=mtf_liveness_contract,
        multi_tf_cache_binding=mtf_cache_binding,
        created_utc=datetime.now(timezone.utc).isoformat(),
    )
    artifact["materializer_provenance"] = {
        "schema_version": PRODUCER_SCHEMA_VERSION,
        "producer": "gx1.scripts.materialize_entry_full_input_liveness_v1",
        "entry_run_id": entry_run_id,
        "dataset_build_proof": {
            "path": str(proof_path),
            "sha256": sha256_file(proof_path),
        },
        "split_contracts": {
            split: {
                "manifest_path": str(manifests[split]),
                "manifest_sha256": manifest_bindings[split]["sha256"],
                "parquet_path": str(parquets[split]),
                "manifest_declared_output_data_path": str(
                    manifest_payloads[split].get("output_data_path") or ""
                ),
            }
            for split in SPLITS
        },
        "semantic_fullscan": semantic_scan,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with out_path.open("x", encoding="utf-8") as handle:
            json.dump(artifact, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise RuntimeError(f"LIVENESS_OUTPUT_ALREADY_EXISTS: {out_path}") from exc

    if artifact["decision"] == PASS_DECISION:
        validation = validate_full_input_liveness_artifact(
            out_path,
            expected_dataset_dir=dataset_dir,
            expected_contract_mode=MODEL_NATIVE_CONTRACT_MODE,
            expected_field_order=canonical_field_order,
            expected_manifest_bindings=manifest_bindings,
        )
        if not validation["ok"]:
            raise RuntimeError(
                "MATERIALIZED_LIVENESS_SELF_VALIDATION_FAILED: "
                + json.dumps(validation["failures"], sort_keys=True)
            )
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Full-scan and bind exact seq513 + ctx142+5 input liveness."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--stem", required=True)
    parser.add_argument("--mtf-cache-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifact = run(args)
    if not args.quiet:
        print(json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False))
    if artifact.get("decision") != PASS_DECISION:
        print(
            json.dumps(
                {
                    "event": "FULL_INPUT_LIVENESS_FAIL",
                    "artifact": str(Path(args.out_json).expanduser().resolve()),
                    "failure_count": len(artifact.get("failures") or []),
                    "failures": artifact.get("failures") or [],
                },
                sort_keys=True,
                allow_nan=False,
            ),
            file=os.sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
