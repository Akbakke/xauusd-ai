"""ThinRecordDataset — reader and authority checks for exact V3 thin records.

The original producer is retired: it used non-model-native Entry side fallback
and lacked the required user-vedtak write gate. This module preserves exact
layout/reconstruction provenance and owns the exact per-trade materializer, but
it is not authority to create a fresh dataset. Fresh V3 rebuilding remains
fail-closed until an end-to-end, model-native, vedtak-gated producer binds every
input listed by ``V3_TRAINING_PRODUCER_INPUT_NAMES``.

Background
----------
The current V8 contract stores io_features once as a shared M1 feature matrix
(N × 173 float32) plus thin per-(trade, bar) records that reference it.
At training time we reconstruct the (window_len, 173) io_features tensor by:

    io = matrix[m1_idx_now-(W-1) : m1_idx_now+1].copy()         # (W, 173)
    overlay_rows = overlays[overlay_offset + overlay_start_row :
                            overlay_offset + overlay_start_row + n_in_trade_bars]
    io[in_trade_start_in_win : in_trade_start_in_win + n_in_trade_bars,
       trade_state_indices] = overlay_rows                       # in-trade fill

The dataset is laid out as:

    <dataset_dir>/
    ├── m1_feature_matrix.npy            (N × 173 float32, mmap-friendly)
    ├── m1_time_ns.npy                   (2.2M int64)
    ├── trade_state_overlays.f32         (raw float32, shape (N, 19))
    ├── overlay_index.parquet            (trade_uid → overlay_offset, overlay_length)
    ├── records.jsonl                    (one record per (trade, bar))
    └── manifest.json                    (io_version, input_dim, window_len, ...)

Each record (one JSON object per line) carries:
    ts, run_id, trade_uid, trade_id, side, m1_idx_now, in_trade_start_in_win,
    n_in_trade_bars, overlay_start_row, scalars, teacher_final_pnl_bps,
    teacher_final_mfe_bps, teacher_final_mae_bps, teacher_duration_bars

Usage
-----
    from gx1.exits.training.thin_record_dataset import ThinRecordDataset

    ds = ThinRecordDataset("/absolute/path/to/authoritative_v3_dataset")
    sample = ds[0]
    # sample["io_features"]: (window_len, input_dim) float32 tensor
    # sample["scalars"]: dict[str, float]
    # sample["teacher"]: dict[str, float]
    # sample["meta"]: dict[str, str]

The dataset can be wrapped with a label-attaching transform to produce
training-ready (x, y) pairs — see `_attach_labels_to_thin_records` below.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from gx1.exits.contracts.exit_io_v8_regime_m1l512 import (
    EXIT_IO_V8_CONTEXT_CADENCE_CONTRACT,
    EXIT_IO_V8_REGIME_M1L512_FEATURE_COUNT,
    EXIT_IO_V8_REGIME_M1L512_FEATURE_NAMES_HASH,
    EXIT_IO_V8_REGIME_M1L512_FEATURES,
    EXIT_IO_V8_REGIME_M1L512_IO_VERSION,
)
from gx1.features.trade_overlay import (
    N_OVERLAY_COLS,
    OVERLAY_COL_NAMES,
    compute_trade_overlay,
)

V3_TRAINING_LINEAGE_SCHEMA_VERSION = "exit_v3_training_lineage_v1"
V3_TRAINING_DATASET_PRODUCER_CONTRACT = (
    "model_native_exit_v3_exact_t5_frozen_entry_state_v2"
)
V3_THIN_RECORD_SCHEMA_VERSION = "model_native_exit_v3_thin_record_v2"
V3_TRAINING_TEACHER_HORIZON_BARS = 240
V3_TRAINING_RECORD_EMIT_STRIDE_BARS = 1
V3_TRAINING_PRODUCER_INPUT_NAMES = frozenset(
    {
        "prediction_parquet",
        "prediction_report",
        "entry_bundle_metadata",
        "entry_model_state",
        "source_tape",
        "canonical_v3",
        "base_m1",
        "prebuilt_pair_manifest",
        "materializer_event",
    }
)
V3_XGB_BRIDGE_SOURCE_SCHEMA_VERSION = "exit_xgb_bridge_source_identity_v1"
V3_XGB_SIGNAL_BRIDGE_CONTRACT = "signal_bridge_v1_7"
V3_XGB_FEATURE_CONTRACT_FILENAME = "xgb_input_features.json"
V3_XGB_SANITIZER_CONFIG_FILENAME = "xgb_input_sanitizer.json"
V3_TRAINING_DATASET_REQUIRED_FILES = frozenset(
    {
        "m1_feature_matrix",
        "m1_time_ns",
        "trade_state_overlays",
        "overlay_index",
        "records",
    }
)
V3_TRAINING_SOURCE_CODE_FILES = frozenset(
    {
        "gx1/contracts/entry_model_native_signal_v1.py",
        "gx1/contracts/entry_model_native_runtime_evidence_v1.py",
        "gx1/contracts/signal_bridge_v1.py",
        "gx1/contracts/signal_bridge_v3.py",
        "gx1/execution/model_native_entry_replay_v1.py",
        "gx1/execution/v12_canonical_incremental.py",
        "gx1/execution/v12_ctx_augment_live.py",
        "gx1/execution/v12_m1_to_m5_downsample.py",
        "gx1/execution/v12_trade_state.py",
        "gx1/execution/v12_v3_live.py",
        "gx1/execution/v12_xgb_live.py",
        "gx1/exits/contracts/exit_io_v1_ctx36.py",
        "gx1/exits/contracts/exit_io_v1_ctx36_features.py",
        "gx1/exits/contracts/exit_io_v3_ctx36_m1l512_phase5.py",
        "gx1/exits/contracts/exit_io_v4_ctx_extended_m1l512.py",
        "gx1/exits/contracts/exit_io_v5_ctx_extended_smc_m1l512.py",
        "gx1/exits/contracts/exit_io_v6_ctx_v3canonical_m1l512.py",
        "gx1/exits/contracts/exit_io_v7_volume_dipstruct_m1l512.py",
        "gx1/exits/contracts/exit_io_v8_regime_m1l512.py",
        "gx1/exits/contracts/registry.py",
        "gx1/exits/training/disk_labeled_dataset.py",
        "gx1/exits/training/thin_record_dataset.py",
        "gx1/features/htf_features.py",
        "gx1/features/regime_v4_features.py",
        "gx1/features/smc_v1.py",
        "gx1/features/trade_overlay.py",
        "gx1/features/volume_features.py",
        "gx1/policy/exit_transformer_v0.py",
        "gx1/scripts/train_exit_v5_thin_records.py",
        "gx1/scripts/train_exit_v6_disk_thin.py",
        "gx1/scripts/train_exit_v6_thin_records.py",
        "gx1/scripts/entry_candidate_prediction_evidence_v1.py",
        "gx1/utils/fast_train.py",
        "gx1/xgb/multihead/xgb_multihead_model_v1.py",
        "gx1/xgb/preprocess/xgb_input_sanitizer.py",
    }
)
_V3_LINEAGE_KEYS = frozenset(
    {
        "schema_version",
        "production_allowed_v1",
        "dataset_producer_contract_v1",
        "dataset_root",
        "dataset_files",
        "dataset_inventory_sha256",
        "m5_prebuilt",
        "xgb_bridge_source",
        "source_code_files",
        "source_code_inventory_sha256",
        "split_uid_sha256",
        "training_recipe_sha256",
        "transformer_config_sha256",
        "initialization",
    }
)
_V3_DATASET_SEMANTIC_FIELDS = {
    "producer_contract_v1": V3_TRAINING_DATASET_PRODUCER_CONTRACT,
    "production_allowed_v1": True,
    "model_native_entry_snapshot_v1": True,
    "exact_t5_fill_v1": True,
    "frozen_entry_snapshot_complete_v1": True,
    "canonical_m1_base_mtf_state_complete_v1": True,
    "record_schema_version_v1": V3_THIN_RECORD_SCHEMA_VERSION,
    "teacher_horizon_bars_v1": V3_TRAINING_TEACHER_HORIZON_BARS,
    "emit_stride_bars_v1": V3_TRAINING_RECORD_EMIT_STRIDE_BARS,
    "direction_authority_v1": "runtime_head_calibrated_direction_argmax",
    "flat_handling_v1": "explicit_no_order_no_exit_records",
    "io_context_cadence_contract_v1": EXIT_IO_V8_CONTEXT_CADENCE_CONTRACT,
    "io_version": EXIT_IO_V8_REGIME_M1L512_IO_VERSION,
    "input_dim": EXIT_IO_V8_REGIME_M1L512_FEATURE_COUNT,
    "window_len": 512,
    "feature_names_hash": EXIT_IO_V8_REGIME_M1L512_FEATURE_NAMES_HASH,
    "trade_state_feature_indices": [
        EXIT_IO_V8_REGIME_M1L512_FEATURES.index(name)
        for name in OVERLAY_COL_NAMES
    ],
}
_V3_THIN_RECORD_KEYS = frozenset(
    {
        "schema_version",
        "ts",
        "decision_ts",
        "entry_fill_time",
        "runtime_head_evidence_sha256",
        "run_id",
        "trade_uid",
        "trade_id",
        "side",
        "m1_idx_now",
        "in_trade_start_in_win",
        "n_in_trade_bars",
        "overlay_start_row",
        "scalars",
        "teacher_final_pnl_bps",
        "teacher_final_mfe_bps",
        "teacher_final_mae_bps",
        "teacher_duration_bars",
    }
)
_V3_THIN_SCALAR_NAMES = (
    "pnl_bps_now",
    "mfe_bps",
    "mae_bps",
    "dd_from_mfe_bps",
    "distance_from_peak_mfe_bps",
    "giveback_ratio",
    "bars_held",
    "time_since_mfe_bars",
    "atr_bps_now",
    "rolling_slope_since_entry",
)
_V3_THIN_SCALAR_KEYS = frozenset(_V3_THIN_SCALAR_NAMES)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _exact_sha256(value: object, *, context: str) -> str:
    parsed = value if isinstance(value, str) else ""
    if (
        parsed != parsed.strip().lower()
        or len(parsed) != 64
        or any(ch not in "0123456789abcdef" for ch in parsed)
    ):
        raise RuntimeError(f"{context}: not an exact SHA-256")
    return parsed


def _read_regular_file_exact(path: Path, *, context: str) -> tuple[bytes, os.stat_result]:
    raw_path = Path(path).expanduser()
    if not raw_path.is_absolute() or raw_path.is_symlink():
        raise RuntimeError(f"{context}: path must be absolute and non-symlinked")
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(raw_path, flags)
    except OSError as exc:
        raise RuntimeError(f"{context}: file cannot be opened exactly") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"{context}: path is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after:
        raise RuntimeError(f"{context}: file changed while being read")
    payload = b"".join(chunks)
    if len(payload) != after.st_size:
        raise RuntimeError(f"{context}: file size changed while being read")
    return payload, after


def _regular_file_payload_binding(
    path: Path,
    *,
    context: str,
) -> tuple[bytes, dict[str, Any]]:
    payload, state = _read_regular_file_exact(path, context=context)
    binding = {
        "path": str(Path(path).expanduser().resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": int(state.st_size),
    }
    return payload, binding


def v3_regular_file_binding(path: Path, *, context: str) -> dict[str, Any]:
    _, binding = _regular_file_payload_binding(path, context=context)
    return binding


def _directory_file_inventory(
    root: Path,
    *,
    context: str,
) -> list[dict[str, Any]]:
    raw_root = Path(root).expanduser()
    if (
        not raw_root.is_absolute()
        or raw_root.is_symlink()
        or not raw_root.is_dir()
    ):
        raise RuntimeError(f"{context}: root must be an absolute regular directory")
    root = raw_root.resolve()
    inventory: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"{context}: symlink is forbidden: {path}")
        if path.is_dir():
            continue
        binding = v3_regular_file_binding(path, context=f"{context}[{path}]")
        inventory.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "sha256": binding["sha256"],
                "size_bytes": binding["size_bytes"],
            }
        )
    if not inventory:
        raise RuntimeError(f"{context}: artifact directory is empty")
    return inventory


def build_v3_xgb_bridge_source_identity(
    *,
    bundle_dir: Path,
    feature_contract_path: Path,
    sanitizer_config_path: Path,
) -> dict[str, Any]:
    """Bind the exact XGB bytes and ordered feature semantics feeding V3."""

    bundle_root = Path(bundle_dir).expanduser()
    if (
        not bundle_root.is_absolute()
        or bundle_root.is_symlink()
        or not bundle_root.is_dir()
    ):
        raise RuntimeError("V3_XGB_BUNDLE_ROOT_INVALID")
    bundle_root = bundle_root.resolve()
    expected_feature_path = bundle_root / V3_XGB_FEATURE_CONTRACT_FILENAME
    expected_sanitizer_path = bundle_root / V3_XGB_SANITIZER_CONFIG_FILENAME
    if (
        Path(feature_contract_path).expanduser().resolve()
        != expected_feature_path
        or Path(sanitizer_config_path).expanduser().resolve()
        != expected_sanitizer_path
    ):
        raise RuntimeError("V3_XGB_CONTRACTS_NOT_BUNDLE_OWNED")
    bundle_files = _directory_file_inventory(
        bundle_root,
        context="V3_XGB_BUNDLE",
    )
    feature_bytes, feature_contract = _regular_file_payload_binding(
        Path(feature_contract_path),
        context="V3_XGB_FEATURE_CONTRACT",
    )
    sanitizer_bytes, sanitizer_config = _regular_file_payload_binding(
        Path(sanitizer_config_path),
        context="V3_XGB_SANITIZER_CONFIG",
    )
    bundle_inventory = {
        item["relative_path"]: {
            "sha256": item["sha256"],
            "size_bytes": item["size_bytes"],
        }
        for item in bundle_files
    }
    for relative, binding in (
        (V3_XGB_FEATURE_CONTRACT_FILENAME, feature_contract),
        (V3_XGB_SANITIZER_CONFIG_FILENAME, sanitizer_config),
    ):
        if bundle_inventory.get(relative) != {
            "sha256": binding["sha256"],
            "size_bytes": binding["size_bytes"],
        }:
            raise RuntimeError("V3_XGB_BUNDLE_CHANGED_DURING_IDENTITY_READ")
    try:
        feature_payload = json.loads(feature_bytes)
        sanitizer_payload = json.loads(sanitizer_bytes)
    except Exception as exc:
        raise RuntimeError("V3_XGB_CONTRACT_JSON_INVALID") from exc
    ordered_features = (
        feature_payload.get("features")
        if isinstance(feature_payload, Mapping)
        else None
    )
    sanitizer_features = (
        sanitizer_payload.get("feature_list")
        if isinstance(sanitizer_payload, Mapping)
        else None
    )
    if (
        not isinstance(ordered_features, list)
        or not ordered_features
        or any(not isinstance(name, str) or not name for name in ordered_features)
        or len(ordered_features) != len(set(ordered_features))
    ):
        raise RuntimeError("V3_XGB_ORDERED_FEATURES_INVALID")
    if sanitizer_features != ordered_features:
        raise RuntimeError("V3_XGB_SANITIZER_FEATURE_ORDER_MISMATCH")
    payload = {
        "schema_version": V3_XGB_BRIDGE_SOURCE_SCHEMA_VERSION,
        "signal_bridge_contract": V3_XGB_SIGNAL_BRIDGE_CONTRACT,
        "bundle_root": str(bundle_root),
        "bundle_files": bundle_files,
        "bundle_inventory_sha256": _canonical_sha256(bundle_files),
        "feature_contract": feature_contract,
        "sanitizer_config": sanitizer_config,
        "ordered_feature_names": list(ordered_features),
        "ordered_feature_names_sha256": _canonical_sha256(ordered_features),
    }
    return {
        **payload,
        "identity_sha256": _canonical_sha256(payload),
    }


def require_v3_xgb_bridge_source_identity(
    value: object,
) -> dict[str, Any]:
    """Rehash and require one exact XGB→seven-field bridge source."""

    expected_keys = {
        "schema_version",
        "signal_bridge_contract",
        "bundle_root",
        "bundle_files",
        "bundle_inventory_sha256",
        "feature_contract",
        "sanitizer_config",
        "ordered_feature_names",
        "ordered_feature_names_sha256",
        "identity_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise RuntimeError("V3_XGB_BRIDGE_SOURCE_IDENTITY_INVALID")
    observed = build_v3_xgb_bridge_source_identity(
        bundle_dir=Path(str(value["bundle_root"] or "")),
        feature_contract_path=Path(
            str(
                value["feature_contract"].get("path")
                if isinstance(value["feature_contract"], Mapping)
                else ""
            )
        ),
        sanitizer_config_path=Path(
            str(
                value["sanitizer_config"].get("path")
                if isinstance(value["sanitizer_config"], Mapping)
                else ""
            )
        ),
    )
    if dict(value) != observed:
        raise RuntimeError("V3_XGB_BRIDGE_SOURCE_BYTES_OR_SEMANTICS_MISMATCH")
    return observed


def _require_v3_regular_file_binding(
    value: object,
    *,
    context: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "sha256",
        "size_bytes",
    }:
        raise RuntimeError(f"{context}: exact file binding keys are required")
    path = Path(str(value["path"] or "")).expanduser()
    expected = {
        "path": str(path.resolve()),
        "sha256": _exact_sha256(value["sha256"], context=f"{context}.sha256"),
        "size_bytes": value["size_bytes"],
    }
    if (
        isinstance(expected["size_bytes"], bool)
        or not isinstance(expected["size_bytes"], int)
        or expected["size_bytes"] < 0
        or dict(value) != expected
    ):
        raise RuntimeError(f"{context}: noncanonical file binding")
    actual = v3_regular_file_binding(path, context=context)
    if actual != expected:
        raise RuntimeError(f"{context}: file bytes differ from binding")
    return actual


def _require_v3_producer_inputs(
    manifest: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Rehash every external byte source consumed by dataset materialization."""

    raw_inputs = manifest.get("producer_inputs_v1")
    if not isinstance(raw_inputs, Mapping) or set(raw_inputs) != set(
        V3_TRAINING_PRODUCER_INPUT_NAMES
    ):
        raise RuntimeError("V3_TRAINING_PRODUCER_INPUTS_INVALID")
    inputs = {
        name: _require_v3_regular_file_binding(
            raw_inputs[name],
            context=f"V3_TRAINING_PRODUCER_INPUT[{name}]",
        )
        for name in sorted(V3_TRAINING_PRODUCER_INPUT_NAMES)
    }
    if (
        manifest.get("producer_inputs_inventory_sha256_v1")
        != _canonical_sha256(inputs)
    ):
        raise RuntimeError("V3_TRAINING_PRODUCER_INPUT_INVENTORY_MISMATCH")
    return inputs


def _require_finite_array_chunks(
    values: np.ndarray,
    *,
    context: str,
    chunk_rows: int = 16_384,
) -> None:
    for start in range(0, len(values), chunk_rows):
        if not np.isfinite(values[start : start + chunk_rows]).all():
            raise RuntimeError(f"{context}_NONFINITE")


def _require_v3_dataset_storage(
    root: Path,
    manifest: Mapping[str, Any],
) -> None:
    """Validate matrix/time/overlay/record identity, not just file presence."""

    files = manifest["files"]
    if (
        set(files)
        != {
            *V3_TRAINING_DATASET_REQUIRED_FILES,
            "trade_state_overlays_cols",
        }
        or files["trade_state_overlays_cols"] != N_OVERLAY_COLS
    ):
        raise RuntimeError("V3_TRAINING_DATASET_FILE_SCHEMA_INVALID")

    matrix = np.load(
        root / str(files["m1_feature_matrix"]),
        mmap_mode="r",
        allow_pickle=False,
    )
    times = np.load(
        root / str(files["m1_time_ns"]),
        mmap_mode="r",
        allow_pickle=False,
    )
    if (
        matrix.dtype != np.dtype("float32")
        or matrix.ndim != 2
        or matrix.shape[1] != EXIT_IO_V8_REGIME_M1L512_FEATURE_COUNT
        or matrix.shape[0] < 512
    ):
        raise RuntimeError("V3_TRAINING_M1_FEATURE_MATRIX_INVALID")
    if (
        times.dtype != np.dtype("int64")
        or times.ndim != 1
        or times.shape[0] != matrix.shape[0]
    ):
        raise RuntimeError("V3_TRAINING_M1_TIME_AXIS_INVALID")
    _require_finite_array_chunks(matrix, context="V3_TRAINING_M1_FEATURE_MATRIX")
    if len(times) > 1 and not np.all(np.diff(times) > 0):
        raise RuntimeError("V3_TRAINING_M1_TIME_AXIS_NOT_STRICTLY_ORDERED")
    if np.any(
        np.remainder(times, pd.Timedelta(minutes=1).value) != 0
    ):
        raise RuntimeError("V3_TRAINING_M1_TIME_AXIS_NOT_MINUTE_ALIGNED")
    trade_state_indices = np.asarray(
        _V3_DATASET_SEMANTIC_FIELDS["trade_state_feature_indices"],
        dtype=np.int64,
    )
    signal_bridge_indices = np.asarray(
        [
            EXIT_IO_V8_REGIME_M1L512_FEATURES.index(name)
            for name in (
                "p_long",
                "p_short",
                "p_flat",
                "p_hat",
                "uncertainty_score",
                "margin_top1_top2",
                "entropy",
            )
        ],
        dtype=np.int64,
    )
    from gx1.xgb.multihead.xgb_multihead_model_v1 import (
        proba_to_signal_bridge_v1,
    )

    for start in range(0, len(matrix), 16_384):
        matrix_chunk = matrix[start : start + 16_384]
        if np.any(matrix_chunk[:, trade_state_indices] != 0.0):
            raise RuntimeError("V3_TRAINING_BASE_MATRIX_TRADE_STATE_NOT_ZERO")
        observed_bridge = np.asarray(
            matrix_chunk[:, signal_bridge_indices],
            dtype=np.float32,
        )
        probabilities = observed_bridge[:, :3]
        if (
            np.any(probabilities < 0.0)
            or np.any(probabilities > 1.0)
            or not np.allclose(
                probabilities.sum(axis=1),
                1.0,
                rtol=0.0,
                atol=1e-6,
            )
            or not np.array_equal(
                observed_bridge,
                np.asarray(
                    proba_to_signal_bridge_v1(probabilities),
                    dtype=np.float32,
                ),
            )
        ):
            raise RuntimeError("V3_TRAINING_XGB_SIGNAL_BRIDGE_INVALID")

    overlay_path = root / str(files["trade_state_overlays"])
    overlay_item_count = overlay_path.stat().st_size // np.dtype("float32").itemsize
    if (
        overlay_path.stat().st_size % np.dtype("float32").itemsize
        or overlay_item_count <= 0
        or overlay_item_count % N_OVERLAY_COLS
    ):
        raise RuntimeError("V3_TRAINING_OVERLAY_STORAGE_INVALID")
    overlays = np.memmap(
        overlay_path,
        dtype=np.float32,
        mode="r",
    ).reshape(-1, N_OVERLAY_COLS)
    _require_finite_array_chunks(overlays, context="V3_TRAINING_OVERLAYS")

    overlay_index = pd.read_parquet(root / str(files["overlay_index"]))
    required_overlay_columns = {
        "trade_uid",
        "overlay_offset",
        "overlay_length",
    }
    if (
        set(overlay_index.columns) != required_overlay_columns
        or overlay_index.empty
        or overlay_index["trade_uid"].isna().any()
        or overlay_index["trade_uid"].astype(str).duplicated().any()
    ):
        raise RuntimeError("V3_TRAINING_OVERLAY_INDEX_INVALID")
    offsets = pd.to_numeric(
        overlay_index["overlay_offset"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    lengths = pd.to_numeric(
        overlay_index["overlay_length"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    if (
        not np.isfinite(offsets).all()
        or not np.isfinite(lengths).all()
        or not np.equal(offsets, np.floor(offsets)).all()
        or not np.equal(lengths, np.floor(lengths)).all()
        or np.any(offsets < 0)
        or np.any(lengths <= 0)
    ):
        raise RuntimeError("V3_TRAINING_OVERLAY_INDEX_BOUNDS_INVALID")
    normalized_index = pd.DataFrame(
        {
            "trade_uid": overlay_index["trade_uid"].astype(str),
            "overlay_offset": offsets.astype(np.int64),
            "overlay_length": lengths.astype(np.int64),
        }
    ).sort_values("overlay_offset", kind="stable")
    expected_offsets = np.concatenate(
        [
            np.asarray([0], dtype=np.int64),
            np.cumsum(
                normalized_index["overlay_length"].to_numpy(dtype=np.int64)
            )[:-1],
        ]
    )
    if (
        not np.array_equal(
            normalized_index["overlay_offset"].to_numpy(dtype=np.int64),
            expected_offsets,
        )
        or int(
            normalized_index["overlay_offset"].iloc[-1]
            + normalized_index["overlay_length"].iloc[-1]
        )
        != len(overlays)
    ):
        raise RuntimeError("V3_TRAINING_OVERLAY_INDEX_NOT_EXACT_PARTITION")
    overlay_lookup = {
        row.trade_uid: (int(row.overlay_offset), int(row.overlay_length))
        for row in normalized_index.itertuples(index=False)
    }

    records_path = root / str(files["records"])
    record_count = 0
    previous_by_trade: dict[
        str,
        tuple[int, pd.Timestamp, int, tuple[Any, ...]],
    ] = {}
    records_per_trade: dict[str, int] = {}
    scalar_positions = {
        name: OVERLAY_COL_NAMES.index(name) for name in _V3_THIN_SCALAR_NAMES
    }
    with records_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise RuntimeError(
                    f"V3_TRAINING_RECORD_EMPTY_LINE: {line_number}"
                )
            try:
                raw_record = json.loads(line)
            except Exception as exc:
                raise RuntimeError(
                    f"V3_TRAINING_RECORD_JSON_INVALID: {line_number}"
                ) from exc
            record = require_exact_v3_thin_record(
                raw_record,
                context=f"V3_TRAINING_RECORD[{line_number}]",
            )
            m1_index = int(record["m1_idx_now"])
            if (
                m1_index < 511
                or m1_index >= len(times)
                or int(times[m1_index])
                != pd.Timestamp(record["ts"]).value
            ):
                raise RuntimeError(
                    f"V3_TRAINING_RECORD_TIME_INDEX_MISMATCH: {line_number}"
                )
            in_trade_start = int(record["in_trade_start_in_win"])
            n_in_trade = int(record["n_in_trade_bars"])
            if (
                in_trade_start >= 512
                or in_trade_start + n_in_trade > 512
            ):
                raise RuntimeError(
                    f"V3_TRAINING_RECORD_WINDOW_BOUNDS_INVALID: {line_number}"
                )
            trade_uid = str(record["trade_uid"])
            if trade_uid not in overlay_lookup:
                raise RuntimeError(
                    f"V3_TRAINING_RECORD_OVERLAY_ID_MISSING: {line_number}"
                )
            overlay_offset, overlay_length = overlay_lookup[trade_uid]
            if overlay_length != V3_TRAINING_TEACHER_HORIZON_BARS:
                raise RuntimeError(
                    f"V3_TRAINING_OVERLAY_HORIZON_INVALID: {line_number}"
                )
            overlay_start = int(record["overlay_start_row"])
            if overlay_start + n_in_trade > overlay_length:
                raise RuntimeError(
                    f"V3_TRAINING_RECORD_OVERLAY_BOUNDS_INVALID: {line_number}"
                )
            relative_overlay_row = overlay_start + n_in_trade - 1
            current_overlay_row = overlay_offset + relative_overlay_row
            for name, position in scalar_positions.items():
                if not np.isclose(
                    float(record["scalars"][name]),
                    float(overlays[current_overlay_row, position]),
                    rtol=0.0,
                    atol=0.0,
                ):
                    raise RuntimeError(
                        "V3_TRAINING_RECORD_SCALAR_OVERLAY_MISMATCH: "
                        f"line={line_number} field={name}"
                    )
            timestamp = pd.Timestamp(record["ts"])
            immutable_trade_facts = (
                record["decision_ts"],
                record["entry_fill_time"],
                record["runtime_head_evidence_sha256"],
                record["run_id"],
                record["trade_id"],
                record["side"],
                record["teacher_final_pnl_bps"],
                record["teacher_final_mfe_bps"],
                record["teacher_final_mae_bps"],
                record["teacher_duration_bars"],
            )
            previous = previous_by_trade.get(trade_uid)
            if previous is None:
                if (
                    relative_overlay_row != 0
                    or timestamp != pd.Timestamp(record["entry_fill_time"])
                ):
                    raise RuntimeError(
                        f"V3_TRAINING_RECORD_FIRST_BAR_INVALID: {line_number}"
                    )
            elif (
                m1_index
                != previous[0] + V3_TRAINING_RECORD_EMIT_STRIDE_BARS
                or timestamp
                != previous[1]
                + pd.Timedelta(
                    minutes=V3_TRAINING_RECORD_EMIT_STRIDE_BARS,
                )
                or relative_overlay_row
                != previous[2] + V3_TRAINING_RECORD_EMIT_STRIDE_BARS
                or immutable_trade_facts != previous[3]
            ):
                raise RuntimeError(
                    f"V3_TRAINING_RECORD_CADENCE_INVALID: {line_number}"
                )
            fill_m1_index = m1_index - relative_overlay_row
            window_start = m1_index - 511
            expected_in_trade_start = max(0, fill_m1_index - window_start)
            expected_overlay_start = max(0, window_start - fill_m1_index)
            expected_n_in_trade = min(
                relative_overlay_row + 1,
                512 - expected_in_trade_start,
            )
            if (
                int(times[fill_m1_index])
                != pd.Timestamp(record["entry_fill_time"]).value
                or in_trade_start != expected_in_trade_start
                or overlay_start != expected_overlay_start
                or n_in_trade != expected_n_in_trade
                or trade_uid
                != (
                    f"{record['run_id']}:{record['trade_id']}:"
                    f"{record['side']}"
                )
            ):
                raise RuntimeError(
                    f"V3_TRAINING_RECORD_RECONSTRUCTION_INVALID: {line_number}"
                )
            previous_by_trade[trade_uid] = (
                m1_index,
                timestamp,
                relative_overlay_row,
                immutable_trade_facts,
            )
            records_per_trade[trade_uid] = (
                records_per_trade.get(trade_uid, 0) + 1
            )
            record_count += 1
    if (
        record_count == 0
        or set(previous_by_trade) != set(overlay_lookup)
        or any(
            records_per_trade.get(trade_uid) != overlay_length
            for trade_uid, (_, overlay_length) in overlay_lookup.items()
        )
    ):
        raise RuntimeError("V3_TRAINING_RECORD_COVERAGE_INVALID")
    for trade_uid, (overlay_offset, overlay_length) in overlay_lookup.items():
        trade_overlay = overlays[
            overlay_offset : overlay_offset + overlay_length
        ]
        entry_snapshot = {
            name: float(
                trade_overlay[0, OVERLAY_COL_NAMES.index(name)]
            )
            for name in OVERLAY_COL_NAMES[:5]
        }
        probability_evidence = np.asarray(
            [
                entry_snapshot["p_long_entry"],
                entry_snapshot["p_hat_entry"],
                entry_snapshot["uncertainty_entry"],
                entry_snapshot["entropy_entry"],
                entry_snapshot["margin_entry"],
            ],
            dtype=np.float64,
        )
        if (
            not (0.0 <= probability_evidence[0] <= 1.0)
            or not (1.0 / 3.0 <= probability_evidence[1] <= 1.0)
            or not np.isclose(
                probability_evidence[1] + probability_evidence[2],
                1.0,
                rtol=0.0,
                atol=1e-7,
            )
            or not (0.0 <= probability_evidence[3] <= np.log(3.0))
            or not (0.0 <= probability_evidence[4] <= probability_evidence[1])
            or probability_evidence[0] > probability_evidence[1]
        ):
            raise RuntimeError(
                f"V3_TRAINING_OVERLAY_ENTRY_SNAPSHOT_INVALID: {trade_uid}"
            )
        recomputed_overlay = compute_trade_overlay(
            np.asarray(
                trade_overlay[:, OVERLAY_COL_NAMES.index("mfe_bps")],
                dtype=np.float64,
            ),
            np.asarray(
                trade_overlay[:, OVERLAY_COL_NAMES.index("mae_bps")],
                dtype=np.float64,
            ),
            np.asarray(
                trade_overlay[:, OVERLAY_COL_NAMES.index("pnl_bps_now")],
                dtype=np.float64,
            ),
            np.asarray(
                trade_overlay[:, OVERLAY_COL_NAMES.index("atr_bps_now")],
                dtype=np.float64,
            ),
            entry_snapshot,
        )
        time_since_mfe_index = OVERLAY_COL_NAMES.index("time_since_mfe_bars")
        exact_overlay_indices = [
            index
            for index in range(N_OVERLAY_COLS)
            if index != time_since_mfe_index
        ]
        time_since_mfe = trade_overlay[:, time_since_mfe_index]
        mfe = trade_overlay[:, OVERLAY_COL_NAMES.index("mfe_bps")]
        time_since_valid = (
            time_since_mfe[0] == 0.0
            and np.equal(time_since_mfe, np.floor(time_since_mfe)).all()
            and np.all(time_since_mfe >= 0.0)
        )
        if time_since_valid:
            for index in range(1, overlay_length):
                if mfe[index] > mfe[index - 1]:
                    time_since_valid = time_since_mfe[index] == 0.0
                else:
                    time_since_valid = time_since_mfe[index] in {
                        0.0,
                        time_since_mfe[index - 1] + 1.0,
                    }
                if not time_since_valid:
                    break
        if (
            not np.array_equal(
                recomputed_overlay[:, exact_overlay_indices],
                trade_overlay[:, exact_overlay_indices],
            )
            or not time_since_valid
        ):
            raise RuntimeError(
                f"V3_TRAINING_OVERLAY_PATH_INVALID: {trade_uid}"
            )
        terminal_facts = previous_by_trade[trade_uid][3]
        if (
            np.float32(terminal_facts[6])
            != float(
                trade_overlay[-1, OVERLAY_COL_NAMES.index("pnl_bps_now")]
            )
            or np.float32(terminal_facts[7])
            != float(trade_overlay[-1, OVERLAY_COL_NAMES.index("mfe_bps")])
            or np.float32(terminal_facts[8])
            != float(trade_overlay[-1, OVERLAY_COL_NAMES.index("mae_bps")])
        ):
            raise RuntimeError(
                f"V3_TRAINING_TEACHER_OVERLAY_MISMATCH: {trade_uid}"
            )


def require_authoritative_v3_training_dataset(
    dataset_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Require one self-contained exact-T+5 model-native V3 dataset."""

    supplied_root = Path(dataset_dir).expanduser()
    if (
        not supplied_root.is_absolute()
        or supplied_root.is_symlink()
        or not supplied_root.is_dir()
    ):
        raise RuntimeError("V3_TRAINING_DATASET_ROOT_INVALID")
    root = supplied_root.resolve()
    if root != supplied_root:
        raise RuntimeError("V3_TRAINING_DATASET_ROOT_NONCANONICAL")
    manifest_path = root / "manifest.json"
    payload, _ = _read_regular_file_exact(
        manifest_path,
        context="V3_TRAINING_DATASET_MANIFEST",
    )
    try:
        manifest = json.loads(payload)
    except Exception as exc:
        raise RuntimeError("V3_TRAINING_DATASET_MANIFEST_INVALID") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError("V3_TRAINING_DATASET_MANIFEST_INVALID")
    semantic_failures = [
        name
        for name, expected in _V3_DATASET_SEMANTIC_FIELDS.items()
        if isinstance(expected, bool)
        and (
            type(manifest.get(name)) is not bool
            or manifest.get(name) is not expected
        )
    ] + [
        name
        for name, expected in _V3_DATASET_SEMANTIC_FIELDS.items()
        if not isinstance(expected, bool) and manifest.get(name) != expected
    ]
    if semantic_failures:
        raise RuntimeError(
            "V3_TRAINING_DATASET_AUTHORITY_MISSING: "
            f"{semantic_failures}"
        )
    _require_v3_producer_inputs(manifest)
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise RuntimeError("V3_TRAINING_DATASET_FILES_INVALID")
    require_v3_xgb_bridge_source_identity(
        manifest.get("xgb_bridge_source_v1")
    )
    missing = sorted(V3_TRAINING_DATASET_REQUIRED_FILES - set(files))
    if missing:
        raise RuntimeError(
            f"V3_TRAINING_DATASET_FILES_MISSING: {missing}"
        )
    relative_paths = ["manifest.json"]
    for field in sorted(V3_TRAINING_DATASET_REQUIRED_FILES):
        relative = files[field]
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise RuntimeError(
                f"V3_TRAINING_DATASET_FILE_PATH_INVALID: {field}"
            )
        relative_paths.append(Path(relative).as_posix())
    if len(set(relative_paths)) != len(relative_paths):
        raise RuntimeError("V3_TRAINING_DATASET_FILE_PATH_DUPLICATE")

    def dataset_inventory() -> list[dict[str, Any]]:
        inventory: list[dict[str, Any]] = []
        for relative in sorted(relative_paths):
            member = root / relative
            if (
                member.is_symlink()
                or member.resolve(strict=False) != member
                or not member.is_file()
            ):
                raise RuntimeError(
                    f"V3_TRAINING_DATASET_FILE_NONCANONICAL: {relative}"
                )
            binding = v3_regular_file_binding(
                member,
                context=f"V3_TRAINING_DATASET_FILE[{relative}]",
            )
            inventory.append(
                {
                    "relative_path": relative,
                    "sha256": binding["sha256"],
                    "size_bytes": binding["size_bytes"],
                }
            )
        return inventory

    inventory = dataset_inventory()
    _require_v3_dataset_storage(root, manifest)
    if dataset_inventory() != inventory:
        raise RuntimeError("V3_TRAINING_DATASET_CHANGED_DURING_VALIDATION")
    return manifest, inventory


def require_reproducible_v3_training_lineage(
    *,
    bundle_dir: Path,
    config: Mapping[str, Any],
    state_path: Path,
) -> dict[str, Any]:
    """Verify every byte and recipe component needed to reproduce V3 training."""

    root = Path(bundle_dir).expanduser()
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise RuntimeError("V3_BUNDLE_ROOT_INVALID")
    root = root.resolve()
    manifest_path = root / "manifest.json"
    payload, _ = _read_regular_file_exact(
        manifest_path,
        context="V3_TRAINING_MANIFEST",
    )
    try:
        manifest = json.loads(payload)
    except Exception as exc:
        raise RuntimeError("V3_TRAINING_MANIFEST_INVALID") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError("V3_TRAINING_MANIFEST_INVALID")
    lineage = manifest.get("training_lineage_v1")
    if not isinstance(lineage, Mapping) or set(lineage) != _V3_LINEAGE_KEYS:
        raise RuntimeError(
            "V3_TRAINING_LINEAGE_MISSING_OR_NONCANONICAL"
        )
    lineage = dict(lineage)
    if (
        lineage["schema_version"] != V3_TRAINING_LINEAGE_SCHEMA_VERSION
        or lineage["production_allowed_v1"] is not True
        or lineage["dataset_producer_contract_v1"]
        != V3_TRAINING_DATASET_PRODUCER_CONTRACT
    ):
        raise RuntimeError("V3_TRAINING_LINEAGE_NOT_PRODUCTION_ELIGIBLE")

    state_binding = v3_regular_file_binding(
        state_path,
        context="V3_STATE_DICT",
    )
    if (
        _exact_sha256(
            manifest.get("model_state_dict_sha256"),
            context="V3_STATE_DICT_MANIFEST",
        )
        != state_binding["sha256"]
    ):
        raise RuntimeError("V3_STATE_DICT_HASH_MISMATCH")
    if manifest.get("exit_io_version") != config.get("exit_ml_io_version"):
        raise RuntimeError("V3_TRAINING_CONFIG_IO_VERSION_MISMATCH")

    dataset_root = Path(str(lineage["dataset_root"] or "")).expanduser()
    dataset_manifest, dataset_inventory = require_authoritative_v3_training_dataset(
        dataset_root
    )
    if lineage["dataset_files"] != dataset_inventory:
        raise RuntimeError("V3_TRAINING_DATASET_INVENTORY_MISMATCH")
    if (
        _exact_sha256(
            lineage["dataset_inventory_sha256"],
            context="V3_TRAINING_DATASET_INVENTORY_SHA",
        )
        != _canonical_sha256(dataset_inventory)
    ):
        raise RuntimeError("V3_TRAINING_DATASET_INVENTORY_HASH_MISMATCH")
    if (
        dataset_manifest.get("producer_contract_v1")
        != lineage["dataset_producer_contract_v1"]
    ):
        raise RuntimeError("V3_TRAINING_DATASET_PRODUCER_MISMATCH")
    xgb_bridge_source = require_v3_xgb_bridge_source_identity(
        lineage["xgb_bridge_source"]
    )
    if dataset_manifest.get("xgb_bridge_source_v1") != xgb_bridge_source:
        raise RuntimeError("V3_TRAINING_XGB_BRIDGE_SOURCE_MISMATCH")

    m5_prebuilt = _require_v3_regular_file_binding(
        lineage["m5_prebuilt"],
        context="V3_TRAINING_M5_PREBUILT",
    )
    if (
        m5_prebuilt
        != dataset_manifest["producer_inputs_v1"]["canonical_v3"]
    ):
        raise RuntimeError("V3_TRAINING_M5_DATASET_SOURCE_MISMATCH")
    source_files = lineage["source_code_files"]
    if not isinstance(source_files, list):
        raise RuntimeError("V3_TRAINING_SOURCE_CODE_INVENTORY_INVALID")
    observed_source_names = {
        str(item.get("relative_path") or "")
        for item in source_files
        if isinstance(item, Mapping)
    }
    if observed_source_names != V3_TRAINING_SOURCE_CODE_FILES:
        raise RuntimeError("V3_TRAINING_SOURCE_CODE_SET_MISMATCH")
    canonical_source_files: list[dict[str, Any]] = []
    for item in source_files:
        if not isinstance(item, Mapping) or set(item) != {
            "relative_path",
            "path",
            "sha256",
            "size_bytes",
        }:
            raise RuntimeError("V3_TRAINING_SOURCE_CODE_BINDING_INVALID")
        relative = str(item["relative_path"])
        binding = _require_v3_regular_file_binding(
            {
                "path": item["path"],
                "sha256": item["sha256"],
                "size_bytes": item["size_bytes"],
            },
            context=f"V3_TRAINING_SOURCE_CODE[{relative}]",
        )
        source_path = Path(binding["path"])
        if (
            source_path.parent == root
            or root not in source_path.parents
            or source_path.relative_to(root).as_posix()
            != f"training_source_v1/{relative}"
        ):
            raise RuntimeError("V3_TRAINING_SOURCE_CODE_NOT_BUNDLE_OWNED")
        canonical_source_files.append(
            {"relative_path": relative, **binding}
        )
    canonical_source_files.sort(key=lambda item: item["relative_path"])
    if source_files != canonical_source_files:
        raise RuntimeError("V3_TRAINING_SOURCE_CODE_ORDER_INVALID")
    if (
        _exact_sha256(
            lineage["source_code_inventory_sha256"],
            context="V3_TRAINING_SOURCE_CODE_INVENTORY_SHA",
        )
        != _canonical_sha256(source_files)
    ):
        raise RuntimeError("V3_TRAINING_SOURCE_CODE_INVENTORY_HASH_MISMATCH")

    split_hashes = lineage["split_uid_sha256"]
    if not isinstance(split_hashes, Mapping) or set(split_hashes) != {
        "train",
        "val",
        "test",
    }:
        raise RuntimeError("V3_TRAINING_SPLIT_HASHES_INVALID")
    for split, value in split_hashes.items():
        _exact_sha256(value, context=f"V3_TRAINING_SPLIT[{split}]")
    if (
        _exact_sha256(
            lineage["training_recipe_sha256"],
            context="V3_TRAINING_RECIPE_SHA",
        )
        != _canonical_sha256(manifest.get("training"))
    ):
        raise RuntimeError("V3_TRAINING_RECIPE_HASH_MISMATCH")
    config_binding = v3_regular_file_binding(
        root / "transformer_config.json",
        context="V3_TRANSFORMER_CONFIG",
    )
    if (
        _exact_sha256(
            lineage["transformer_config_sha256"],
            context="V3_TRANSFORMER_CONFIG_SHA",
        )
        != config_binding["sha256"]
    ):
        raise RuntimeError("V3_TRANSFORMER_CONFIG_HASH_MISMATCH")

    initialization = lineage["initialization"]
    if not isinstance(initialization, Mapping) or set(initialization) != {
        "mode",
        "source_state_dict",
    }:
        raise RuntimeError("V3_TRAINING_INITIALIZATION_INVALID")
    if initialization["mode"] == "cold":
        if initialization["source_state_dict"] is not None:
            raise RuntimeError("V3_TRAINING_COLD_SOURCE_FORBIDDEN")
    elif initialization["mode"] == "warm_start":
        _require_v3_regular_file_binding(
            initialization["source_state_dict"],
            context="V3_TRAINING_WARM_START",
        )
    else:
        raise RuntimeError("V3_TRAINING_INITIALIZATION_MODE_INVALID")
    return lineage


def require_exact_v3_thin_record(
    value: Mapping[str, Any] | Any,
    *,
    context: str,
) -> dict[str, Any]:
    """Validate one fully materialized record without neutral defaults."""

    if not isinstance(value, Mapping) or set(value) != _V3_THIN_RECORD_KEYS:
        raise RuntimeError(f"{context}: exact thin-record schema mismatch")
    record = dict(value)
    if record["schema_version"] != V3_THIN_RECORD_SCHEMA_VERSION:
        raise RuntimeError(f"{context}: thin-record schema_version mismatch")
    for name in ("ts", "decision_ts", "entry_fill_time"):
        parsed = pd.to_datetime(record[name], utc=True, errors="coerce")
        if (
            not isinstance(record[name], str)
            or pd.isna(parsed)
            or parsed != parsed.floor("min")
            or record[name] != parsed.isoformat()
        ):
            raise RuntimeError(f"{context}: {name} is not an exact UTC minute")
    if pd.Timestamp(record["entry_fill_time"]) != (
        pd.Timestamp(record["decision_ts"]) + pd.Timedelta(minutes=5)
    ):
        raise RuntimeError(f"{context}: entry_fill_time is not exact T+5")
    _exact_sha256(
        record["runtime_head_evidence_sha256"],
        context=f"{context}.runtime_head_evidence_sha256",
    )
    for name in ("run_id", "trade_uid", "trade_id"):
        if (
            not isinstance(record[name], str)
            or not record[name]
            or record[name].strip() != record[name]
        ):
            raise RuntimeError(f"{context}: {name} is invalid")
    if record["side"] not in {"long", "short"}:
        raise RuntimeError(f"{context}: side must be long/short")
    for name in (
        "m1_idx_now",
        "in_trade_start_in_win",
        "n_in_trade_bars",
        "overlay_start_row",
        "teacher_duration_bars",
    ):
        parsed = record[name]
        if isinstance(parsed, bool) or not isinstance(parsed, int) or parsed < 0:
            raise RuntimeError(f"{context}: {name} must be a non-negative integer")
    if record["n_in_trade_bars"] <= 0:
        raise RuntimeError(f"{context}: n_in_trade_bars must be positive")
    if record["teacher_duration_bars"] != V3_TRAINING_TEACHER_HORIZON_BARS:
        raise RuntimeError(f"{context}: teacher duration contract mismatch")
    scalars = record["scalars"]
    if not isinstance(scalars, Mapping) or set(scalars) != _V3_THIN_SCALAR_KEYS:
        raise RuntimeError(f"{context}: scalar schema mismatch")
    numeric_values = [
        *scalars.values(),
        record["teacher_final_pnl_bps"],
        record["teacher_final_mfe_bps"],
        record["teacher_final_mae_bps"],
    ]
    if any(isinstance(value, (bool, np.bool_)) for value in numeric_values):
        raise RuntimeError(f"{context}: numeric evidence cannot be boolean")
    try:
        numeric = np.asarray(numeric_values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"{context}: numeric evidence is invalid") from exc
    if not np.isfinite(numeric).all():
        raise RuntimeError(f"{context}: numeric evidence is non-finite")
    if float(record["teacher_final_mfe_bps"]) < float(
        record["teacher_final_pnl_bps"]
    ) or float(record["teacher_final_mae_bps"]) > float(
        record["teacher_final_pnl_bps"]
    ):
        raise RuntimeError(f"{context}: teacher MFE/MAE bounds are invalid")
    return record


def materialize_model_native_v3_trade_records(
    *,
    prediction_row: Mapping[str, Any],
    source_tape: Any,
    base_m1_time_ns: np.ndarray,
    run_id: str,
    window_len: int = 512,
    teacher_horizon_bars: int = V3_TRAINING_TEACHER_HORIZON_BARS,
    emit_stride_bars: int = V3_TRAINING_RECORD_EMIT_STRIDE_BARS,
) -> dict[str, Any]:
    """Materialize one calibrated argmax row into exact T+5 V3 records.

    The finite teacher horizon is a training-label contract only. It grants no
    active-Exit replay or live horizon-cap authority. FLAT remains an explicit
    no-order result and emits no trade records.
    """

    from gx1.contracts.entry_model_native_runtime_evidence_v1 import (
        decode_model_native_runtime_head_evidence,
        require_model_native_exit_replay_entry_time,
    )
    from gx1.features.trade_overlay import OVERLAY_COL_NAMES, compute_trade_overlay

    if (
        teacher_horizon_bars != V3_TRAINING_TEACHER_HORIZON_BARS
        or emit_stride_bars != V3_TRAINING_RECORD_EMIT_STRIDE_BARS
    ):
        raise RuntimeError("V3_TRAINING_LABEL_OR_EMIT_CONTRACT_MISMATCH")
    if (
        not isinstance(window_len, int)
        or isinstance(window_len, bool)
        or window_len != 512
    ):
        raise RuntimeError("V3_TRAINING_WINDOW_CONTRACT_MISMATCH")
    if not isinstance(run_id, str) or not run_id.strip() or run_id != run_id.strip():
        raise RuntimeError("V3_TRAINING_RUN_ID_INVALID")
    if not isinstance(prediction_row, Mapping):
        raise RuntimeError("V3_TRAINING_PREDICTION_ROW_INVALID")
    head = decode_model_native_runtime_head_evidence(
        prediction_row.get("runtime_head_evidence_json"),
        prediction_row.get("runtime_head_evidence_sha256"),
        context="V3_TRAINING_RUNTIME_HEAD",
    )
    decision_time = pd.Timestamp(head["decision_ts"])
    entry_fill_time = decision_time + pd.Timedelta(minutes=5)
    require_model_native_exit_replay_entry_time(
        head,
        entry_fill_time,
        context="V3_TRAINING_T5_FILL",
    )
    if "time" in prediction_row and pd.Timestamp(prediction_row["time"]) != decision_time:
        raise RuntimeError("V3_TRAINING_PREDICTION_TIME_MISMATCH")
    if "pred_direction" in prediction_row and int(
        prediction_row["pred_direction"]
    ) != int(head["model_direction_index"]):
        raise RuntimeError("V3_TRAINING_PREDICTION_DIRECTION_MISMATCH")

    direction = int(head["model_direction_index"])
    direction_name = str(head["model_direction"])
    head_sha = str(prediction_row["runtime_head_evidence_sha256"])
    identity_payload = {
        "run_id": run_id,
        "decision_ts": decision_time.isoformat(),
        "runtime_head_evidence_sha256": head_sha,
    }
    trade_id = _canonical_sha256(identity_payload)
    if direction == 2:
        return {
            "status": "FLAT_NO_ORDER",
            "trade_uid": None,
            "trade_id": trade_id,
            "decision_ts": decision_time.isoformat(),
            "entry_fill_time": None,
            "side": None,
            "overlay": None,
            "records": [],
        }
    if direction not in (0, 1) or direction_name not in {"LONG", "SHORT"}:
        raise RuntimeError("V3_TRAINING_DIRECTION_INVALID")
    side = "long" if direction == 0 else "short"

    tape_positions = source_tape.indices_for_times(
        pd.Series([entry_fill_time])
    )
    fill_position = int(tape_positions[0])
    end_position = fill_position + teacher_horizon_bars
    if end_position > len(source_tape.times):
        raise RuntimeError("V3_TRAINING_SOURCE_TAPE_HORIZON_MISSING")
    path_times = pd.DatetimeIndex(source_tape.index[fill_position:end_position])
    expected_times = pd.date_range(
        start=entry_fill_time,
        periods=teacher_horizon_bars,
        freq="min",
    )
    if not np.array_equal(path_times.asi8, expected_times.asi8):
        raise RuntimeError("V3_TRAINING_SOURCE_TAPE_M1_CADENCE_GAP")

    base_times = pd.to_datetime(
        np.asarray(base_m1_time_ns, dtype=np.int64),
        utc=True,
        errors="coerce",
    )
    if (
        base_times.hasnans
        or not base_times.is_monotonic_increasing
        or not base_times.is_unique
    ):
        raise RuntimeError("V3_TRAINING_BASE_M1_TIME_INDEX_INVALID")
    base_positions = base_times.get_indexer(path_times)
    if np.any(base_positions < 0) or not np.array_equal(
        base_positions,
        np.arange(
            int(base_positions[0]),
            int(base_positions[0]) + teacher_horizon_bars,
            dtype=np.int64,
        ),
    ):
        raise RuntimeError("V3_TRAINING_BASE_M1_PATH_MISMATCH")
    if int(base_positions[0]) < window_len - 1:
        raise RuntimeError("V3_TRAINING_BASE_M1_HISTORY_MISSING")

    entry_bid = float(source_tape.bid_open[fill_position])
    entry_ask = float(source_tape.ask_open[fill_position])
    path = slice(fill_position, end_position)
    bid_close = np.asarray(source_tape.bid_close[path], dtype=np.float64)
    ask_close = np.asarray(source_tape.ask_close[path], dtype=np.float64)
    bid_high = np.asarray(source_tape.bid_high[path], dtype=np.float64)
    bid_low = np.asarray(source_tape.bid_low[path], dtype=np.float64)
    ask_high = np.asarray(source_tape.ask_high[path], dtype=np.float64)
    ask_low = np.asarray(source_tape.ask_low[path], dtype=np.float64)
    if side == "long":
        cur_pnl = (bid_close - entry_ask) / entry_ask * 10_000.0
        peak = (bid_high - entry_ask) / entry_ask * 10_000.0
        trough = (bid_low - entry_ask) / entry_ask * 10_000.0
    else:
        cur_pnl = (entry_bid - ask_close) / entry_bid * 10_000.0
        peak = (entry_bid - ask_low) / entry_bid * 10_000.0
        trough = (entry_bid - ask_high) / entry_bid * 10_000.0
    mid_close = (bid_close + ask_close) / 2.0
    atr_bps = (ask_high - bid_low) / mid_close * 10_000.0
    # Quantize the four path primitives before deriving the float32 overlay.
    # This makes every stored derivative exactly reproducible from the stored
    # path values instead of depending on discarded float64 low bits.
    peak = np.asarray(peak, dtype=np.float32).astype(np.float64)
    trough = np.asarray(trough, dtype=np.float32).astype(np.float64)
    cur_pnl = np.asarray(cur_pnl, dtype=np.float32).astype(np.float64)
    atr_bps = np.asarray(atr_bps, dtype=np.float32).astype(np.float64)
    direction_probs = np.asarray(head["direction_probs"], dtype=np.float64)
    sorted_probs = np.sort(direction_probs)
    overlay = compute_trade_overlay(
        peak,
        trough,
        cur_pnl,
        atr_bps,
        {
            "p_long_entry": float(direction_probs[0]),
            "p_hat_entry": float(direction_probs.max()),
            "uncertainty_entry": float(1.0 - direction_probs.max()),
            "entropy_entry": float(
                -np.sum(direction_probs * np.log(np.maximum(direction_probs, 1e-12)))
            ),
            "margin_entry": float(sorted_probs[-1] - sorted_probs[-2]),
        },
    )
    scalar_indices = {
        name: OVERLAY_COL_NAMES.index(name) for name in _V3_THIN_SCALAR_NAMES
    }
    teacher_final_pnl = float(
        overlay[-1, OVERLAY_COL_NAMES.index("pnl_bps_now")]
    )
    teacher_final_mfe = float(
        overlay[-1, OVERLAY_COL_NAMES.index("mfe_bps")]
    )
    teacher_final_mae = float(
        overlay[-1, OVERLAY_COL_NAMES.index("mae_bps")]
    )
    trade_uid = f"{run_id}:{trade_id}:{side}"
    records: list[dict[str, Any]] = []
    fill_base_position = int(base_positions[0])
    for offset in range(0, teacher_horizon_bars, emit_stride_bars):
        m1_idx_now = int(base_positions[offset])
        window_start = m1_idx_now - window_len + 1
        in_trade_start = max(0, fill_base_position - window_start)
        overlay_start = max(0, window_start - fill_base_position)
        n_in_trade = min(offset + 1, window_len - in_trade_start)
        record = {
            "schema_version": V3_THIN_RECORD_SCHEMA_VERSION,
            "ts": path_times[offset].isoformat(),
            "decision_ts": decision_time.isoformat(),
            "entry_fill_time": entry_fill_time.isoformat(),
            "runtime_head_evidence_sha256": head_sha,
            "run_id": run_id,
            "trade_uid": trade_uid,
            "trade_id": trade_id,
            "side": side,
            "m1_idx_now": m1_idx_now,
            "in_trade_start_in_win": int(in_trade_start),
            "n_in_trade_bars": int(n_in_trade),
            "overlay_start_row": int(overlay_start),
            "scalars": {
                name: float(overlay[offset, index])
                for name, index in scalar_indices.items()
            },
            "teacher_final_pnl_bps": teacher_final_pnl,
            "teacher_final_mfe_bps": teacher_final_mfe,
            "teacher_final_mae_bps": teacher_final_mae,
            "teacher_duration_bars": teacher_horizon_bars,
        }
        records.append(
            require_exact_v3_thin_record(
                record,
                context=f"V3_TRAINING_RECORD[{offset}]",
            )
        )
    return {
        "status": "MATERIALIZED",
        "trade_uid": trade_uid,
        "trade_id": trade_id,
        "decision_ts": decision_time.isoformat(),
        "entry_fill_time": entry_fill_time.isoformat(),
        "side": side,
        "overlay": overlay,
        "records": records,
    }


class ThinRecordDataset(Dataset):
    """Memory-efficient PyTorch dataset over the V8 thin-record layout."""

    def __init__(
        self,
        dataset_dir: str | Path,
        *,
        memmap_matrix: bool = True,
        memmap_overlays: bool = True,
        records_offset: int = 0,
        records_limit: Optional[int] = None,
        load_records_eagerly: bool = True,
        build_offset_index: bool = False,
    ) -> None:
        self.dataset_dir = Path(dataset_dir).expanduser().resolve()
        self.manifest = self._load_manifest()
        self.records_path: Optional[Path] = None
        self.uid_to_offsets: Optional[Dict[str, List[Tuple[int, int]]]] = None

        # Load shared tensors (mmap'd by default)
        matrix_path = self.dataset_dir / self.manifest["files"]["m1_feature_matrix"]
        overlays_path = self.dataset_dir / self.manifest["files"]["trade_state_overlays"]
        time_ns_path = self.dataset_dir / self.manifest["files"]["m1_time_ns"]
        index_path = self.dataset_dir / self.manifest["files"]["overlay_index"]
        records_path = self.dataset_dir / self.manifest["files"]["records"]

        self.matrix = np.load(
            str(matrix_path),
            mmap_mode="r" if memmap_matrix else None,
            allow_pickle=False,
        )
        self.m1_time_ns = np.load(
            str(time_ns_path),
            mmap_mode="r" if memmap_matrix else None,
            allow_pickle=False,
        )

        # overlays.f32 is a raw binary file: shape (N, 19) float32
        n_overlay_cols = int(self.manifest["files"].get("trade_state_overlays_cols", 19))
        if memmap_overlays:
            self.overlays = np.memmap(
                overlays_path, dtype=np.float32, mode="r"
            ).reshape(-1, n_overlay_cols)
        else:
            self.overlays = np.fromfile(overlays_path, dtype=np.float32).reshape(-1, n_overlay_cols)

        # overlay_index: trade_uid → (overlay_offset, overlay_length)
        oi = pd.read_parquet(index_path)
        self.overlay_index: Dict[str, Tuple[int, int]] = {
            row["trade_uid"]: (int(row["overlay_offset"]), int(row["overlay_length"]))
            for _, row in oi.iterrows()
        }

        # Records: load JSONL into memory or stream
        self.input_dim = int(self.manifest["input_dim"])
        self.window_len = int(self.manifest["window_len"])
        self.trade_state_indices = list(self.manifest["trade_state_feature_indices"])

        self.records_path = records_path
        if load_records_eagerly:
            if build_offset_index:
                self.records, self.uid_to_offsets = self._load_records_jsonl_with_offsets(
                    records_path, offset=records_offset, limit=records_limit,
                )
            else:
                self.records = self._load_records_jsonl(
                    records_path, offset=records_offset, limit=records_limit,
                )
        else:
            self.records = []
            if build_offset_index:
                self.uid_to_offsets = self._build_offset_index(records_path)

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    def _load_manifest(self) -> Dict[str, Any]:
        mp = self.dataset_dir / "manifest.json"
        if not mp.exists():
            raise FileNotFoundError(f"[THIN_RECORD_DATASET] manifest.json not found at {mp}")
        return json.loads(mp.read_text())

    def _load_records_jsonl(
        self, path: Path, *, offset: int = 0, limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i < offset:
                    continue
                if limit is not None and (i - offset) >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
        return out

    def _load_records_jsonl_with_offsets(
        self, path: Path, *, offset: int = 0, limit: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, List[Tuple[int, int]]]]:
        """Load records AND build a per-trade byte-offset index in a single pass.

        The index maps trade_uid -> list of (byte_offset, byte_length) tuples
        so workers can later seek+read the JSONL slice for a specific trade
        without parent-process inheritance of the records dict.
        """
        out: List[Dict[str, Any]] = []
        from collections import defaultdict
        uid_to_offsets: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        i = 0
        with path.open("rb") as f:
            byte_off = 0
            while True:
                line = f.readline()
                if not line:
                    break
                ll = len(line)
                if i >= offset and (limit is None or (i - offset) < limit):
                    stripped = line.strip()
                    if stripped:
                        rec = json.loads(stripped)
                        out.append(rec)
                        uid = rec.get("trade_uid") or rec.get("trade_id")
                        if uid:
                            uid_to_offsets[uid].append((byte_off, ll))
                byte_off += ll
                i += 1
                if limit is not None and (i - offset) >= limit:
                    break
        return out, dict(uid_to_offsets)

    def _build_offset_index(self, path: Path) -> Dict[str, List[Tuple[int, int]]]:
        """Build trade_uid -> [(byte_offset, byte_length)] index without storing records.

        Used for spawn-worker labeling: parent stays small (just the index, ~30 MB
        for 2.5M records); workers seek+read their slice on demand.
        """
        from collections import defaultdict
        uid_to_offsets: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        with path.open("rb") as f:
            byte_off = 0
            while True:
                line = f.readline()
                if not line:
                    break
                ll = len(line)
                stripped = line.strip()
                if stripped:
                    rec = json.loads(stripped)
                    uid = rec.get("trade_uid") or rec.get("trade_id")
                    if uid:
                        uid_to_offsets[uid].append((byte_off, ll))
                byte_off += ll
        return dict(uid_to_offsets)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = require_exact_v3_thin_record(
            self.records[idx],
            context=f"THIN_RECORD_DATASET[{idx}]",
        )
        io = self._reconstruct_io_features(rec)
        return {
            "io_features": torch.from_numpy(io),  # (W, 173) float32
            "scalars": dict(rec.get("scalars", {})),
            "teacher": {
                "final_pnl_bps": float(rec["teacher_final_pnl_bps"]),
                "final_mfe_bps": float(rec["teacher_final_mfe_bps"]),
                "final_mae_bps": float(rec["teacher_final_mae_bps"]),
                "duration_bars": int(rec["teacher_duration_bars"]),
            },
            "meta": {
                "ts": str(rec.get("ts", "")),
                "trade_uid": str(rec.get("trade_uid", "")),
                "trade_id": str(rec.get("trade_id", "")),
                "side": str(rec.get("side", "")),
                "run_id": str(rec.get("run_id", "")),
            },
        }

    # ------------------------------------------------------------------
    # Reconstruction
    # ------------------------------------------------------------------

    def _reconstruct_io_features(self, rec: Dict[str, Any]) -> np.ndarray:
        """Reconstruct the (window_len, input_dim) io_features tensor for a record."""
        rec = require_exact_v3_thin_record(
            rec,
            context="THIN_RECORD_RECONSTRUCTION",
        )
        m1_idx_now = int(rec["m1_idx_now"])
        win_start = m1_idx_now - self.window_len + 1
        win_end = m1_idx_now + 1
        if win_start < 0 or win_end > int(self.matrix.shape[0]):
            raise IndexError(
                f"[THIN_RECORD_DATASET] m1_idx_now={m1_idx_now} window out of bounds "
                f"(matrix len={self.matrix.shape[0]}, window_len={self.window_len})"
            )
        io = np.array(self.matrix[win_start:win_end], dtype=np.float32, copy=True)

        # Apply trade-state overlay over the in-trade portion of the window
        n_in_trade = int(rec["n_in_trade_bars"])
        if n_in_trade > 0:
            in_trade_start = int(rec["in_trade_start_in_win"])
            overlay_start_row = int(rec["overlay_start_row"])
            trade_uid = rec["trade_uid"]
            if trade_uid not in self.overlay_index:
                raise KeyError(f"[THIN_RECORD_DATASET] trade_uid {trade_uid} not in overlay_index")
            ovl_off, ovl_len = self.overlay_index[trade_uid]
            slice_start = ovl_off + overlay_start_row
            slice_end = slice_start + n_in_trade
            if slice_end > ovl_off + ovl_len:
                raise IndexError(
                    f"[THIN_RECORD_DATASET] overlay slice out of range for trade {trade_uid}: "
                    f"slice_end={slice_end} bound={ovl_off + ovl_len}"
                )
            overlay_rows = np.array(self.overlays[slice_start:slice_end], dtype=np.float32, copy=False)
            io[in_trade_start: in_trade_start + n_in_trade, self.trade_state_indices] = overlay_rows
        return io


def attach_labels_to_thin_records(
    records: List[Dict[str, Any]],
    *,
    derive_should_exit_fn=None,
) -> None:
    """Attach training labels to thin records in-place using teacher hindsight.

    The V3 trainer expects each record to carry:
      - should_exit              (binary main label)
      - profit_protect_should_exit  (binary)
      - profit_protect_train_mask   (binary)
      - sample_weight (optional, float)

    These are derived from teacher_final_pnl_bps / teacher_final_mfe_bps /
    teacher_final_mae_bps / teacher_duration_bars + per-bar scalars.

    The actual derivation logic should be imported from
    `gx1.policy.exit_transformer_v0._attach_labels_to_exit_records` once the
    trainer-side adapter is integrated. This stub provides the hook point.
    """
    if derive_should_exit_fn is None:
        # Import lazily — avoids circular import + keeps this module standalone.
        from gx1.policy import exit_transformer_v0
        derive_should_exit_fn = exit_transformer_v0._attach_labels_to_exit_records  # type: ignore[attr-defined]
    derive_should_exit_fn(records)


__all__ = [
    "ThinRecordDataset",
    "V3_THIN_RECORD_SCHEMA_VERSION",
    "V3_TRAINING_DATASET_PRODUCER_CONTRACT",
    "V3_TRAINING_TEACHER_HORIZON_BARS",
    "V3_TRAINING_RECORD_EMIT_STRIDE_BARS",
    "attach_labels_to_thin_records",
    "materialize_model_native_v3_trade_records",
    "require_authoritative_v3_training_dataset",
    "require_exact_v3_thin_record",
]
