"""ThinRecordDataset — reader for the admitted V3 artifact's legacy thin-record
exit-transformer dataset layout.

The original producer is retired: it used non-model-native Entry side fallback
and lacked the required user-vedtak write gate. This module preserves exact
layout/reconstruction provenance for auditing the active artifact; it is not an
authority to create a fresh dataset. Fresh V3 rebuilding remains
``BLOCKED_PENDING_NEW_EXACT_BUILDER`` until a model-native, fail-closed,
vedtak-gated producer is implemented.

Background
----------
The V5 dataset stores io_features ONCE as a shared M1 feature matrix
(2.2M × 89 float32) plus thin per-(trade, bar) records that reference it.
At training time we reconstruct the (window_len, 89) io_features tensor by:

    io = matrix[m1_idx_now-(W-1) : m1_idx_now+1].copy()          # (W, 89)
    overlay_rows = overlays[overlay_offset + overlay_start_row :
                            overlay_offset + overlay_start_row + n_in_trade_bars]
    io[in_trade_start_in_win : in_trade_start_in_win + n_in_trade_bars,
       trade_state_indices] = overlay_rows                       # in-trade fill

The dataset is laid out as:

    <dataset_dir>/
    ├── m1_feature_matrix.npy            (2.2M × 89 float32, mmap-friendly)
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

    ds = ThinRecordDataset("/path/to/exit_v3_v7_training_2020_2026_canonical_v3")
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


V3_TRAINING_LINEAGE_SCHEMA_VERSION = "exit_v3_training_lineage_v1"
V3_TRAINING_DATASET_PRODUCER_CONTRACT = (
    "model_native_exit_v3_exact_t5_frozen_entry_state_v1"
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
        "gx1/contracts/signal_bridge_v1.py",
        "gx1/contracts/signal_bridge_v3.py",
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
}


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


def require_authoritative_v3_training_dataset(
    dataset_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Require one self-contained exact-T+5 model-native V3 dataset."""

    root = Path(dataset_dir).expanduser()
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise RuntimeError("V3_TRAINING_DATASET_ROOT_INVALID")
    root = root.resolve()
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
    inventory: list[dict[str, Any]] = []
    for relative in sorted(relative_paths):
        binding = v3_regular_file_binding(
            root / relative,
            context=f"V3_TRAINING_DATASET_FILE[{relative}]",
        )
        inventory.append(
            {
                "relative_path": relative,
                "sha256": binding["sha256"],
                "size_bytes": binding["size_bytes"],
            }
        )
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

    _require_v3_regular_file_binding(
        lineage["m5_prebuilt"],
        context="V3_TRAINING_M5_PREBUILT",
    )
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


class ThinRecordDataset(Dataset):
    """Memory-efficient PyTorch dataset over the V5 thin-record format."""

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

        self.matrix = np.load(str(matrix_path), mmap_mode="r" if memmap_matrix else None)
        self.m1_time_ns = np.load(str(time_ns_path), mmap_mode="r" if memmap_matrix else None)

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
        rec = self.records[idx]
        io = self._reconstruct_io_features(rec)
        return {
            "io_features": torch.from_numpy(io),  # (W, 89) float32
            "scalars": dict(rec.get("scalars", {})),
            "teacher": {
                "final_pnl_bps": float(rec.get("teacher_final_pnl_bps", 0.0) or 0.0),
                "final_mfe_bps": float(rec.get("teacher_final_mfe_bps", 0.0) or 0.0),
                "final_mae_bps": float(rec.get("teacher_final_mae_bps", 0.0) or 0.0),
                "duration_bars": int(rec.get("teacher_duration_bars", 0) or 0),
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
        n_in_trade = int(rec.get("n_in_trade_bars", 0) or 0)
        if n_in_trade > 0:
            in_trade_start = int(rec.get("in_trade_start_in_win", 0))
            overlay_start_row = int(rec.get("overlay_start_row", 0))
            trade_uid = rec.get("trade_uid")
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


__all__ = ["ThinRecordDataset", "attach_labels_to_thin_records"]
