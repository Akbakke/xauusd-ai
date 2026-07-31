#!/usr/bin/env python3
"""Fail-closed XAU future-outcome target audit before model-native training."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.contracts.entry_model_native_signal_v1 import (
    FORBIDDEN_LEGACY_BRIDGE_FIELDS,
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_DIRECTION_LOGIT_MODE,
    model_native_signal_contract_failures,
)
from gx1.contracts.entry_model_native_state_v2 import (
    validate_state_contract_metadata_v2,
)
from gx1.contracts.xau_tape_provenance_v1 import validate_xau_tape_provenance_v1
from gx1.scripts.build_entry_v10_ctx_training_dataset_v3 import (
    DIRECTION_DATASET_STEM_SUFFIX,
    V12_DIRECTION_UTILITY_MAE_WEIGHT,
    V12_DIRECTION_UTILITY_MFE_WEIGHT,
    V12_DIRECTION_UTILITY_MIN_BPS,
    V12_DIRECTION_UTILITY_MIN_SIDE_MARGIN_BPS,
    V12_DIRECTION_UTILITY_PATH_WEIGHT,
)
import pyarrow.parquet as pq
from gx1.contracts.entry_pretrain_polarity_signal_v1 import (
    CHANNEL_POSITION_FEATURE,
    PRETRAIN_POLARITY_SIGNAL_REQUIRED_FIELDS,
    RESISTANCE_STACK_FEATURE,
    SUPPORT_MINUS_RESISTANCE_FEATURE,
    SUPPORT_STACK_FEATURE,
)
from gx1.features.entry_chart_geometry_v1 import (
    CHART_GEOMETRY_MODEL_NATIVE_FEATURE_NAMES,
)


DEFAULT_STEM = f"v10_6yr_dataset{DIRECTION_DATASET_STEM_SUFFIX}"
REQUIRED_POLARITY_FEATURES = PRETRAIN_POLARITY_SIGNAL_REQUIRED_FIELDS
REQUIRED_RAIL_FEATURES = tuple(
    name
    for name in CHART_GEOMETRY_MODEL_NATIVE_FEATURE_NAMES
    if "_rail_" in name
)
REQUIRED_XAU_TARGET_COLUMNS = (
    "y_direction",
    "y_bad_path",
    "y_trade",
    "y_tradable",
    "y_side",
    "y_side_mask",
    "mae_first_n_bps",
    "mfe_first_n_bps",
    "path_quality_bps",
    "y_position_size_target",
    "mfe_long_first_n_bps",
    "mae_long_first_n_bps",
    "mfe_short_first_n_bps",
    "mae_short_first_n_bps",
    "bad_path_long_first_n",
    "bad_path_short_first_n",
    "y_long_final_pnl_at_direction_horizon_bps",
    "y_short_final_pnl_at_direction_horizon_bps",
    "y_direction_target_mode_id",
    "y_direction_long_score_bps",
    "y_direction_short_score_bps",
    "y_long_path_utility_bps",
    "y_short_path_utility_bps",
    "y_long_bad_path",
    "y_short_bad_path",
    "y_long_expected_mae_bps",
    "y_short_expected_mae_bps",
    "y_rising_channel_support_touch",
    "y_falling_channel_resistance_touch",
    "y_support_retest_continuation",
    "y_resistance_retest_continuation",
    "y_countertrend_short_trap",
    "y_countertrend_long_trap",
    "y_long_high_mae_low_mfe_early_failure",
    "y_short_high_mae_low_mfe_early_failure",
)
TARGET_CONTRACT_IDENTITY_COLUMNS = frozenset({"y_direction_target_mode_id"})


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj) if np.isfinite(obj) else None
    return str(obj)


def _parse_csv(raw: str) -> list[str]:
    return [part.strip() for part in str(raw or "").split(",") if part.strip()]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _feature_fields(manifest: dict[str, Any]) -> list[str]:
    extra = manifest.get("extra") if isinstance(manifest.get("extra"), dict) else {}
    bridge = extra.get("signal_bridge") if isinstance(extra.get("signal_bridge"), dict) else {}
    for source in (
        bridge.get("fields"),
        bridge.get("snap_fields"),
        (manifest.get("feature_contract") or {}).get("signal_bridge_fields")
        if isinstance(manifest.get("feature_contract"), dict)
        else None,
    ):
        if isinstance(source, list) and all(isinstance(item, str) for item in source):
            return list(source)
    raise RuntimeError("manifest lacks signal_bridge feature fields")


def _seq_structure_mode(manifest: dict[str, Any]) -> str | None:
    extra = manifest.get("extra") if isinstance(manifest.get("extra"), dict) else {}
    bridge = extra.get("signal_bridge") if isinstance(extra.get("signal_bridge"), dict) else {}
    for source in (
        bridge.get("seq_structure_extension_v1"),
        extra.get("seq_structure_extension_v1"),
    ):
        if isinstance(source, dict) and str(source.get("mode") or "").strip():
            return str(source.get("mode")).strip()
    return None


def _manifest_provenance(manifest: dict[str, Any]) -> dict[str, Any]:
    extra = manifest.get("extra") if isinstance(manifest.get("extra"), dict) else {}
    inputs = manifest.get("inputs") if isinstance(manifest.get("inputs"), dict) else {}
    signal_surface = (
        extra.get("signal_bridge")
        if isinstance(extra.get("signal_bridge"), dict)
        else {}
    )
    state_contract = extra.get("model_native_state_contract")
    return {
        "tape_root": str(
            manifest.get("tape_root")
            or extra.get("tape_root")
            or inputs.get("tape_root")
            or ""
        ),
        "contract_mode": str(extra.get("contract_mode") or ""),
        "direction_logit_mode": str(extra.get("direction_logit_mode") or ""),
        "model_native_signal_contract": extra.get("model_native_signal_contract"),
        "xau_tape_provenance": extra.get("xau_tape_provenance"),
        "fields": list(signal_surface.get("fields") or []),
        "splits": manifest.get("splits") if isinstance(manifest.get("splits"), dict) else {},
        "model_native_state_contract": (
            dict(state_contract) if isinstance(state_contract, dict) else {}
        ),
    }


def _state_contract_failures(contract: dict[str, Any], *, split: str) -> list[str]:
    if not isinstance(contract, dict) or not contract:
        return [f"{split}: XAU repair requires model_native_state_contract provenance"]
    try:
        validate_state_contract_metadata_v2(contract, require_artifact=True)
    except (RuntimeError, TypeError, ValueError, OSError) as exc:
        return [f"{split}: model_native_state_contract v2 invalid: {exc}"]
    return []


def _split_artifacts(dataset_dir: Path, stem: str, split: str) -> tuple[Path, Path]:
    parquet_path = dataset_dir / f"{stem}_{split}.parquet"
    manifest_path = dataset_dir / f"{stem}_{split}.manifest.json"
    if not parquet_path.exists():
        raise RuntimeError(f"missing parquet for split={split}: {parquet_path}")
    if not manifest_path.exists():
        raise RuntimeError(f"missing manifest for split={split}: {manifest_path}")
    return parquet_path, manifest_path


def _stem_from_split_filename(name: str, *, split: str, suffix: str) -> str | None:
    marker = f"_{split}{suffix}"
    if not name.endswith(marker):
        return None
    stem = name[: -len(marker)]
    return stem or None


def _resolve_stem(dataset_dir: Path, requested_stem: str, splits: list[str]) -> str:
    requested = str(requested_stem or "").strip()
    if not requested or requested.lower() == "auto":
        raise RuntimeError("an explicit immutable --stem is required; discovery is forbidden")
    return requested


def _column_liveness(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    unique_count = int(np.unique(finite).size) if finite.size else 0
    std = float(np.std(finite)) if finite.size else None
    return {
        "finite_rate": float(finite.size / max(arr.size, 1)),
        "unique_count": unique_count,
        "std": std,
        "min": float(np.min(finite)) if finite.size else None,
        "max": float(np.max(finite)) if finite.size else None,
        "live": bool(finite.size == arr.size and unique_count >= 2 and std is not None and std > 1e-9),
    }


def _read_sample(
    parquet_path: Path,
    *,
    columns: list[str],
    max_rows: int,
    max_row_groups: int,
) -> dict[str, np.ndarray]:
    pf = pq.ParquetFile(parquet_path)
    if pf.num_row_groups <= 0:
        raise RuntimeError(f"{parquet_path} has no row groups")
    row_groups = list(range(pf.num_row_groups))
    if max_row_groups > 0 and len(row_groups) > max_row_groups:
        positions = np.linspace(0, len(row_groups) - 1, num=max_row_groups)
        row_groups = sorted({int(round(pos)) for pos in positions})

    parts: dict[str, list[np.ndarray]] = {name: [] for name in columns}
    remaining = int(max_rows)
    for rg in row_groups:
        if remaining <= 0:
            break
        table = pf.read_row_group(rg, columns=columns)
        take = min(int(table.num_rows), remaining)
        if take <= 0:
            continue
        if take < table.num_rows:
            table = table.slice(0, take)
        for name in columns:
            col = table[name].combine_chunks()
            if name == "snap":
                parts[name].append(np.asarray(col.to_pylist(), dtype=np.float32))
            else:
                parts[name].append(np.asarray(col.to_numpy(zero_copy_only=False)))
        remaining -= take

    out: dict[str, np.ndarray] = {}
    for name, chunks in parts.items():
        if not chunks:
            out[name] = np.empty((0, 0), dtype=np.float32) if name == "snap" else np.empty((0,), dtype=np.float32)
        else:
            out[name] = np.concatenate(chunks, axis=0)
    return out


def _safe_mean(values: np.ndarray, mask: np.ndarray) -> float | None:
    if int(mask.sum()) <= 0:
        return None
    arr = np.asarray(values[mask], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else None


def _safe_rate(mask: np.ndarray) -> float | None:
    if mask.size <= 0:
        return None
    return float(np.asarray(mask, dtype=bool).mean())


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if int(ok.sum()) < 3:
        return None
    x = x[ok]
    y = y[ok]
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _audit_split(
    *,
    split: str,
    parquet_path: Path,
    manifest_path: Path,
    max_rows: int,
    max_row_groups: int,
    support_dominance_min: float,
    min_pocket_rows: int,
) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    fields = _feature_fields(manifest)
    feature_index = {name: (fields.index(name) if name in fields else None) for name in REQUIRED_POLARITY_FEATURES}
    schema_names = set(pq.read_schema(parquet_path).names)
    missing_targets = [name for name in REQUIRED_XAU_TARGET_COLUMNS if name not in schema_names]
    sample_columns = (
        ["snap"]
        + [name for name in REQUIRED_XAU_TARGET_COLUMNS if name in schema_names]
    )
    sample = _read_sample(
        parquet_path,
        columns=sample_columns,
        max_rows=max_rows,
        max_row_groups=max_row_groups,
    )
    snap = np.asarray(sample["snap"], dtype=np.float32)
    if snap.ndim != 2:
        raise RuntimeError(f"{parquet_path} snap sample is not 2D: {snap.shape}")
    missing_features = [name for name, idx in feature_index.items() if idx is None or idx >= snap.shape[1]]
    polarity: dict[str, Any] = {"available": False}
    if not missing_features:
        support = snap[:, int(feature_index[SUPPORT_STACK_FEATURE])]
        resistance = snap[:, int(feature_index[RESISTANCE_STACK_FEATURE])]
        support_minus_resistance = snap[
            :, int(feature_index[SUPPORT_MINUS_RESISTANCE_FEATURE])
        ]
        channel_position = snap[:, int(feature_index[CHANNEL_POSITION_FEATURE])]
        support_dom = (support - resistance) > float(support_dominance_min)
        resistance_dom = (resistance - support) > float(support_dominance_min)
        support_mean = _safe_mean(channel_position, support_dom)
        resistance_mean = _safe_mean(channel_position, resistance_dom)
        delta = (
            None
            if support_mean is None or resistance_mean is None
            else float(resistance_mean - support_mean)
        )
        corr = _safe_corr(channel_position, support_minus_resistance)
        polarity = {
            "available": True,
            "support_dominance_min": float(support_dominance_min),
            "support_dominant_rows": int(support_dom.sum()),
            "resistance_dominant_rows": int(resistance_dom.sum()),
            "support_dominant_channel_position_mean": support_mean,
            "resistance_dominant_channel_position_mean": resistance_mean,
            "resistance_minus_support_channel_position_mean": delta,
            "channel_position_vs_support_minus_resistance_corr": corr,
            "support_dominant_channel_position_lt_042_rate": (
                _safe_rate(channel_position[support_dom] < 0.42)
                if int(support_dom.sum())
                else None
            ),
            "resistance_dominant_channel_position_gt_058_rate": (
                _safe_rate(channel_position[resistance_dom] > 0.58)
                if int(resistance_dom.sum())
                else None
            ),
            "enough_pocket_rows": bool(
                int(support_dom.sum()) >= int(min_pocket_rows)
                and int(resistance_dom.sum()) >= int(min_pocket_rows)
            ),
        }

    target_liveness = {
        name: _column_liveness(np.asarray(sample[name]))
        for name in REQUIRED_XAU_TARGET_COLUMNS
        if name in sample and name not in TARGET_CONTRACT_IDENTITY_COLUMNS
    }
    y_direction = np.asarray(sample.get("y_direction", np.empty((0,), dtype=np.int32)), dtype=np.int32)
    y_trade = np.asarray(sample.get("y_trade", np.empty((0,), dtype=np.float32)), dtype=np.float32)
    y_tradable = np.asarray(sample.get("y_tradable", np.empty((0,), dtype=np.float32)), dtype=np.float32)
    y_side = np.asarray(sample.get("y_side", np.empty((0,), dtype=np.int32)), dtype=np.int32)
    y_side_mask = np.asarray(sample.get("y_side_mask", np.empty((0,), dtype=np.float32)), dtype=np.float32)
    y_bad_path = np.asarray(sample.get("y_bad_path", np.empty((0,), dtype=np.float32)), dtype=np.float32)
    mae_first = np.asarray(sample.get("mae_first_n_bps", np.empty((0,), dtype=np.float32)), dtype=np.float32)
    mfe_first = np.asarray(sample.get("mfe_first_n_bps", np.empty((0,), dtype=np.float32)), dtype=np.float32)
    path_quality = np.asarray(sample.get("path_quality_bps", np.empty((0,), dtype=np.float32)), dtype=np.float32)
    y_long_utility = np.asarray(sample.get("y_long_path_utility_bps", np.empty((0,), dtype=np.float32)), dtype=np.float32)
    y_short_utility = np.asarray(sample.get("y_short_path_utility_bps", np.empty((0,), dtype=np.float32)), dtype=np.float32)
    y_long_bad_path = np.asarray(sample.get("y_long_bad_path", np.empty((0,), dtype=np.float32)), dtype=np.float32)
    y_short_bad_path = np.asarray(sample.get("y_short_bad_path", np.empty((0,), dtype=np.float32)), dtype=np.float32)
    y_long_mae = np.asarray(sample.get("y_long_expected_mae_bps", np.empty((0,), dtype=np.float32)), dtype=np.float32)
    y_short_mae = np.asarray(sample.get("y_short_expected_mae_bps", np.empty((0,), dtype=np.float32)), dtype=np.float32)
    mfe_long = np.asarray(sample.get("mfe_long_first_n_bps", np.empty((0,), dtype=np.float32)), dtype=np.float32)
    mae_long = np.asarray(sample.get("mae_long_first_n_bps", np.empty((0,), dtype=np.float32)), dtype=np.float32)
    mfe_short = np.asarray(sample.get("mfe_short_first_n_bps", np.empty((0,), dtype=np.float32)), dtype=np.float32)
    mae_short = np.asarray(sample.get("mae_short_first_n_bps", np.empty((0,), dtype=np.float32)), dtype=np.float32)
    raw_long_bad = np.asarray(sample.get("bad_path_long_first_n", np.empty((0,), dtype=np.float32)), dtype=np.float32)
    raw_short_bad = np.asarray(sample.get("bad_path_short_first_n", np.empty((0,), dtype=np.float32)), dtype=np.float32)
    pnl_long = np.asarray(
        sample.get("y_long_final_pnl_at_direction_horizon_bps", np.empty((0,), dtype=np.float32)),
        dtype=np.float32,
    )
    pnl_short = np.asarray(
        sample.get("y_short_final_pnl_at_direction_horizon_bps", np.empty((0,), dtype=np.float32)),
        dtype=np.float32,
    )
    target_mode_id = np.asarray(
        sample.get("y_direction_target_mode_id", np.empty((0,), dtype=np.float32)),
        dtype=np.float32,
    )
    y_position_size = np.asarray(sample.get("y_position_size_target", np.empty((0,), dtype=np.float32)), dtype=np.float32)
    y_direction_long_score = np.asarray(
        sample.get("y_direction_long_score_bps", np.empty((0,), dtype=np.float32)),
        dtype=np.float32,
    )
    y_direction_short_score = np.asarray(
        sample.get("y_direction_short_score_bps", np.empty((0,), dtype=np.float32)),
        dtype=np.float32,
    )
    target_consistency: dict[str, Any] = {"available": False}
    if (
        y_direction.size
        and y_trade.size == y_direction.size
        and y_tradable.size == y_direction.size
        and y_side.size == y_direction.size
        and y_side_mask.size == y_direction.size
        and y_bad_path.size == y_direction.size
        and mae_first.size == y_direction.size
        and mfe_first.size == y_direction.size
        and path_quality.size == y_direction.size
        and y_long_utility.size == y_direction.size
        and y_short_utility.size == y_direction.size
        and y_long_bad_path.size == y_direction.size
        and y_short_bad_path.size == y_direction.size
        and y_long_mae.size == y_direction.size
        and y_short_mae.size == y_direction.size
        and mfe_long.size == y_direction.size
        and mae_long.size == y_direction.size
        and mfe_short.size == y_direction.size
        and mae_short.size == y_direction.size
        and raw_long_bad.size == y_direction.size
        and raw_short_bad.size == y_direction.size
        and pnl_long.size == y_direction.size
        and pnl_short.size == y_direction.size
        and target_mode_id.size == y_direction.size
        and y_position_size.size == y_direction.size
    ):
        rounded_mode = np.rint(target_mode_id)
        invalid_mode = (
            ~np.isfinite(target_mode_id)
            | (np.abs(target_mode_id - 1.0) > 1e-6)
        )
        mode_contract_valid = bool((~invalid_mode).all() and np.unique(rounded_mode).size == 1)
        expected_long_utility = (
            pnl_long
            + float(V12_DIRECTION_UTILITY_MFE_WEIGHT) * mfe_long
            - float(V12_DIRECTION_UTILITY_MAE_WEIGHT) * mae_long
            + float(V12_DIRECTION_UTILITY_PATH_WEIGHT) * (mfe_long - mae_long)
        ).astype(np.float32)
        expected_short_utility = (
            pnl_short
            + float(V12_DIRECTION_UTILITY_MFE_WEIGHT) * mfe_short
            - float(V12_DIRECTION_UTILITY_MAE_WEIGHT) * mae_short
            + float(V12_DIRECTION_UTILITY_PATH_WEIGHT) * (mfe_short - mae_short)
        ).astype(np.float32)
        utility_margin = expected_long_utility - expected_short_utility
        tradable_long = (
            mode_contract_valid
            & (expected_long_utility >= float(V12_DIRECTION_UTILITY_MIN_BPS))
            & (utility_margin >= float(V12_DIRECTION_UTILITY_MIN_SIDE_MARGIN_BPS))
        )
        tradable_short = (
            mode_contract_valid
            & (expected_short_utility >= float(V12_DIRECTION_UTILITY_MIN_BPS))
            & ((-utility_margin) >= float(V12_DIRECTION_UTILITY_MIN_SIDE_MARGIN_BPS))
        )
        expected_direction = np.full_like(y_direction, 2)
        only_long = tradable_long & (~tradable_short)
        only_short = tradable_short & (~tradable_long)
        both = tradable_long & tradable_short
        expected_direction[only_long] = 0
        expected_direction[only_short] = 1
        expected_direction[both & (expected_long_utility >= expected_short_utility)] = 0
        expected_direction[both & (expected_short_utility > expected_long_utility)] = 1

        expected_scalar_bad = np.zeros_like(y_bad_path, dtype=np.float32)
        long_rows = (y_trade > 0.5) & (y_side == 0)
        short_rows = (y_trade > 0.5) & (y_side == 1)
        flat_rows = y_trade <= 0.5
        expected_scalar_bad[long_rows] = y_long_bad_path[long_rows]
        expected_scalar_bad[short_rows] = y_short_bad_path[short_rows]
        expected_mfe = np.zeros_like(mfe_first, dtype=np.float32)
        expected_mae = np.zeros_like(mae_first, dtype=np.float32)
        expected_mfe[long_rows] = mfe_long[long_rows]
        expected_mae[long_rows] = mae_long[long_rows]
        expected_mfe[short_rows] = mfe_short[short_rows]
        expected_mae[short_rows] = mae_short[short_rows]
        expected_path = (expected_mfe - expected_mae).astype(np.float32)
        bad_path_mismatch = np.abs(y_bad_path - expected_scalar_bad) > 1e-5
        tradable_mismatch = np.abs(y_tradable - y_trade) > 1e-5
        mfe_mismatch = np.abs(mfe_first - expected_mfe) > 1e-5
        mae_mismatch = np.abs(mae_first - expected_mae) > 1e-5
        path_mismatch = np.abs(path_quality - expected_path) > 1e-5
        flat_size_mismatch = np.abs(y_position_size[flat_rows] - 0.5) > 1e-5 if int(flat_rows.sum()) else np.zeros(0, dtype=bool)
        long_alias_mismatch = (
            np.abs(y_direction_long_score - y_long_utility) > 1e-5
            if y_direction_long_score.size == y_direction.size
            else np.zeros(0, dtype=bool)
        )
        short_alias_mismatch = (
            np.abs(y_direction_short_score - y_short_utility) > 1e-5
            if y_direction_short_score.size == y_direction.size
            else np.zeros(0, dtype=bool)
        )
        target_consistency = {
            "available": True,
            "bad_path_side_consistent_rate": float(1.0 - np.mean(bad_path_mismatch)) if bad_path_mismatch.size else None,
            "bad_path_side_mismatch_count": int(bad_path_mismatch.sum()),
            "tradable_trade_mismatch_count": int(tradable_mismatch.sum()),
            "selected_mfe_mismatch_count": int(mfe_mismatch.sum()),
            "selected_mae_mismatch_count": int(mae_mismatch.sum()),
            "selected_path_quality_mismatch_count": int(path_mismatch.sum()),
            "flat_position_size_neutral_rate": float(1.0 - np.mean(flat_size_mismatch)) if flat_size_mismatch.size else None,
            "flat_position_size_mismatch_count": int(flat_size_mismatch.sum()),
            "flat_rows": int(flat_rows.sum()),
            "target_mode_contract_invalid_count": int((~np.isfinite(rounded_mode) | invalid_mode).sum())
            + int(np.unique(rounded_mode).size != 1),
            "direction_outcome_mismatch_count": int(np.sum(y_direction != expected_direction)),
            "long_utility_formula_mismatch_count": int(
                np.sum(np.abs(y_long_utility - expected_long_utility) > 1e-4)
            ),
            "short_utility_formula_mismatch_count": int(
                np.sum(np.abs(y_short_utility - expected_short_utility) > 1e-4)
            ),
            "long_bad_path_raw_mismatch_count": int(
                np.sum(np.abs(y_long_bad_path - raw_long_bad) > 1e-5)
            ),
            "short_bad_path_raw_mismatch_count": int(
                np.sum(np.abs(y_short_bad_path - raw_short_bad) > 1e-5)
            ),
            "long_mae_raw_mismatch_count": int(np.sum(np.abs(y_long_mae - mae_long) > 1e-5)),
            "short_mae_raw_mismatch_count": int(np.sum(np.abs(y_short_mae - mae_short) > 1e-5)),
            "direction_long_score_alias_mismatch_count": int(long_alias_mismatch.sum()),
            "direction_short_score_alias_mismatch_count": int(short_alias_mismatch.sum()),
        }

    return {
        "split": split,
        "parquet_path": str(parquet_path),
        "manifest_path": str(manifest_path),
        "rows_sampled": int(snap.shape[0]),
        "feature_count": int(len(fields)),
        "seq_structure_extension_mode": _seq_structure_mode(manifest),
        "provenance": _manifest_provenance(manifest),
        "feature_index": feature_index,
        "missing_polarity_features": missing_features,
        "missing_target_columns": missing_targets,
        "polarity": polarity,
        "target_liveness": target_liveness,
        "target_consistency": target_consistency,
        "core_target_policy": "future_path_and_utility_outcomes_only",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = _parse_csv(args.data_splits)
    failures: list[str] = []
    requested_stem = str(args.stem)
    stem: str | None = None
    try:
        stem = _resolve_stem(dataset_dir, requested_stem, splits)
    except Exception as exc:
        failures.append(f"stem resolution failed: {exc}")

    split_reports: list[dict[str, Any]] = []
    manifest_fields: list[str] | None = None
    missing_rail_features: list[str] = []
    if stem is not None:
        for split in splits:
            try:
                parquet_path, manifest_path = _split_artifacts(dataset_dir, stem, split)
                manifest = _load_json(manifest_path)
                fields = _feature_fields(manifest)
                if manifest_fields is None:
                    manifest_fields = fields
                    missing_rail_features = [name for name in REQUIRED_RAIL_FEATURES if name not in fields]
                split_report = _audit_split(
                    split=split,
                    parquet_path=parquet_path,
                    manifest_path=manifest_path,
                    max_rows=int(args.max_rows_per_split),
                    max_row_groups=int(args.max_row_groups_per_split),
                    support_dominance_min=float(args.support_dominance_min),
                    min_pocket_rows=int(args.min_pocket_rows),
                )
                split_reports.append(split_report)
            except Exception as exc:
                failures.append(f"{split}: audit failed: {exc}")

    if missing_rail_features:
        failures.append(f"missing required XAU rail features in manifest: {missing_rail_features}")

    stale_markers = ("utilityrepair", "20260710", "smart_candidate_20260630", "julyext")
    dataset_dir_text = str(dataset_dir).lower()
    for marker in stale_markers:
        if marker in dataset_dir_text:
            failures.append(f"dataset_dir contains known stale XAU repair marker {marker!r}: {dataset_dir}")

    tape_provenance_cache: dict[tuple[str, str], dict[str, Any]] = {}
    tape_provenance_by_split: dict[str, dict[str, Any]] = {}
    for row in split_reports:
        split = str(row.get("split"))
        seq_mode = row.get("seq_structure_extension_mode")
        if seq_mode != "mandatory_inline_common_causal_history_v1":
            failures.append(
                f"{split}: XAU repair requires inline seq-structure features; observed mode={seq_mode}"
            )
        provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
        if str(provenance.get("contract_mode") or "") != MODEL_NATIVE_CONTRACT_MODE:
            failures.append(f"{split}: contract_mode must be {MODEL_NATIVE_CONTRACT_MODE}")
        if (
            str(provenance.get("direction_logit_mode") or "")
            != MODEL_NATIVE_DIRECTION_LOGIT_MODE
        ):
            failures.append(
                f"{split}: direction_logit_mode must be {MODEL_NATIVE_DIRECTION_LOGIT_MODE}"
            )
        signal_contract = provenance.get("model_native_signal_contract")
        signal_failures = model_native_signal_contract_failures(
            signal_contract if isinstance(signal_contract, dict) else {}
        )
        failures.extend(f"{split}: {failure}" for failure in signal_failures)
        forbidden = sorted(
            set(provenance.get("fields") or ()) & set(FORBIDDEN_LEGACY_BRIDGE_FIELDS)
        )
        if forbidden:
            failures.append(f"{split}: forbidden legacy bridge fields: {forbidden}")
        state_contract = provenance.get("model_native_state_contract")
        state_contract = state_contract if isinstance(state_contract, dict) else {}
        failures.extend(_state_contract_failures(state_contract, split=split))
        tape_root = str(provenance.get("tape_root") or "").strip()
        expected_run_id = str(state_contract.get("entry_run_id") or "").strip()
        cache_key = (tape_root, expected_run_id)
        try:
            if cache_key not in tape_provenance_cache:
                tape_provenance_cache[cache_key] = validate_xau_tape_provenance_v1(
                    tape_root,
                    expected_run_id=expected_run_id,
                    require_current=True,
                )
            tape_provenance_by_split[split] = tape_provenance_cache[cache_key]
            declared_tape_provenance = provenance.get("xau_tape_provenance")
            if declared_tape_provenance != tape_provenance_by_split[split]:
                failures.append(
                    f"{split}: dataset manifest XAU_USD tape binding differs from "
                    "the revalidated immutable tape lineage"
                )
        except (RuntimeError, OSError, ValueError) as exc:
            failures.append(f"{split}: immutable XAU_USD tape provenance invalid: {exc}")
        splits_contract = provenance.get("splits")
        train_window = (
            splits_contract.get("train")
            if isinstance(splits_contract, dict)
            and isinstance(splits_contract.get("train"), dict)
            else {}
        )
        if not train_window:
            failures.append(f"{split}: manifest missing exact TRAIN split window")
        elif isinstance(state_contract, dict):
            try:
                train_start = pd.Timestamp(pd.to_datetime(train_window.get("start"), utc=True))
                train_end = pd.Timestamp(pd.to_datetime(train_window.get("end"), utc=True))
                rank_fit_start = pd.Timestamp(
                    pd.to_datetime(state_contract.get("rank_fit_start_utc"), utc=True)
                )
                rank_fit_end = pd.Timestamp(
                    pd.to_datetime(state_contract.get("rank_fit_end_utc"), utc=True)
                )
            except Exception:
                failures.append(f"{split}: TRAIN/state rank-fit timestamps are invalid")
            else:
                if rank_fit_start != train_start or rank_fit_end != train_end:
                    failures.append(
                        f"{split}: TRAIN-only rank fit {rank_fit_start}..{rank_fit_end} "
                        f"does not equal TRAIN window {train_start}..{train_end}"
                    )
        for name in row.get("missing_polarity_features") or []:
            failures.append(f"{split}: missing channel-polarity feature: {name}")
        for name in row.get("missing_target_columns") or []:
            failures.append(f"{split}: missing XAU future-outcome target column: {name}")
        for name, live in (row.get("target_liveness") or {}).items():
            if not bool(live.get("live")):
                failures.append(f"{split}: XAU future-outcome target column is not live: {name}")
        consistency = row.get("target_consistency") if isinstance(row.get("target_consistency"), dict) else {}
        if not bool(consistency.get("available")):
            failures.append(f"{split}: XAU future-outcome target consistency audit unavailable")
        else:
            if int(consistency.get("bad_path_side_mismatch_count") or 0):
                failures.append(
                    f"{split}: scalar y_bad_path mismatches repaired side-specific bad-path targets: "
                    f"mismatches={consistency.get('bad_path_side_mismatch_count')}"
                )
            if int(consistency.get("flat_position_size_mismatch_count") or 0):
                failures.append(
                    f"{split}: y_position_size_target is not neutral for FLAT/no-trade rows: "
                    f"mismatches={consistency.get('flat_position_size_mismatch_count')}"
                )
            hard_consistency_checks = (
                ("target_mode_contract_invalid_count", "direction target mode contract is invalid/mixed"),
                ("direction_outcome_mismatch_count", "y_direction mismatches future outcome side selection"),
                ("long_utility_formula_mismatch_count", "long utility is not the declared future-outcome formula"),
                ("short_utility_formula_mismatch_count", "short utility is not the declared future-outcome formula"),
                ("long_bad_path_raw_mismatch_count", "long bad-path target differs from raw future outcome"),
                ("short_bad_path_raw_mismatch_count", "short bad-path target differs from raw future outcome"),
                ("long_mae_raw_mismatch_count", "long expected MAE differs from raw future MAE"),
                ("short_mae_raw_mismatch_count", "short expected MAE differs from raw future MAE"),
                ("direction_long_score_alias_mismatch_count", "y_direction_long_score_bps mismatches outcome long utility"),
                ("direction_short_score_alias_mismatch_count", "y_direction_short_score_bps mismatches outcome short utility"),
                ("tradable_trade_mismatch_count", "y_tradable mismatches y_trade"),
                ("selected_mfe_mismatch_count", "mfe_first_n_bps mismatches selected side-specific MFE"),
                ("selected_mae_mismatch_count", "mae_first_n_bps mismatches selected side-specific MAE"),
                ("selected_path_quality_mismatch_count", "path_quality_bps mismatches selected side-specific path"),
            )
            for key, reason in hard_consistency_checks:
                count = int(consistency.get(key) or 0)
                if count:
                    failures.append(f"{split}: {reason}: mismatches={count}")
        polarity = row.get("polarity") if isinstance(row.get("polarity"), dict) else {}
        if not bool(polarity.get("available")):
            continue
        if not bool(polarity.get("enough_pocket_rows")):
            failures.append(
                f"{split}: insufficient support/resistance pocket rows for polarity audit "
                f"(support={polarity.get('support_dominant_rows')} resistance={polarity.get('resistance_dominant_rows')})"
            )
            continue
        delta = polarity.get("resistance_minus_support_channel_position_mean")
        if delta is None or float(delta) < float(args.min_channel_position_delta):
            failures.append(
                f"{split}: channel_position polarity stale/inverted; expected resistance_mean > support_mean "
                f"by >= {args.min_channel_position_delta}, got delta={delta}"
            )
        corr = polarity.get("channel_position_vs_support_minus_resistance_corr")
        if corr is None or float(corr) > float(args.max_channel_position_support_corr):
            failures.append(
                f"{split}: channel_position correlates wrong with support_minus_resistance; "
                f"expected <= {args.max_channel_position_support_corr}, got {corr}"
            )

    split_state_contracts: dict[str, dict[str, Any]] = {}
    for row in split_reports:
        provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
        contract = provenance.get("model_native_state_contract")
        if isinstance(contract, dict) and contract:
            split_state_contracts[str(row.get("split"))] = contract
    if len(split_state_contracts) > 1:
        baseline_split = next(iter(split_state_contracts))
        baseline = split_state_contracts[baseline_split]
        for split, contract in split_state_contracts.items():
            if contract != baseline:
                failures.append(
                    f"{split}: model_native_state_contract differs from {baseline_split}; "
                    "TRAIN/VAL/TEST must share one immutable common-history/TRAIN-rank contract"
                )

    if len(tape_provenance_by_split) > 1:
        baseline_split = next(iter(tape_provenance_by_split))
        baseline = tape_provenance_by_split[baseline_split]
        for split, proof in tape_provenance_by_split.items():
            if proof != baseline:
                failures.append(
                    f"{split}: immutable XAU_USD tape provenance differs from "
                    f"{baseline_split}; TRAIN/VAL/TEST must share one exact tape lineage"
                )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    report = {
        "schema_version": "xau_direction_repair_pretrain_audit_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "PASS" if not failures else "FAIL",
        "dataset_dir": str(dataset_dir),
        "requested_stem": requested_stem,
        "stem": str(stem or requested_stem),
        "data_splits": splits,
        "require_rail_features": True,
        "require_inline_seq_structure": True,
        "require_xau_provenance": True,
        "required_rail_features": list(REQUIRED_RAIL_FEATURES),
        "missing_rail_features": missing_rail_features,
        "required_xau_target_columns": list(REQUIRED_XAU_TARGET_COLUMNS),
        "tape_provenance": tape_provenance_by_split,
        "thresholds": {
            "support_dominance_min": float(args.support_dominance_min),
            "min_pocket_rows": int(args.min_pocket_rows),
            "min_channel_position_delta": float(args.min_channel_position_delta),
            "max_channel_position_support_corr": float(args.max_channel_position_support_corr),
        },
        "splits": split_reports,
        "failures": failures,
    }
    json_path = out_dir / f"XAU_DIRECTION_REPAIR_PRETRAIN_AUDIT_{timestamp}.json"
    report["json_path"] = str(json_path)
    with json_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n")

    if not args.quiet:
        print(
            json.dumps(
                {
                    "decision": report["decision"],
                    "failures": failures,
                    "json_path": str(json_path),
                },
                indent=2,
                default=_json_default,
            )
        )
    if failures:
        raise SystemExit(2)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Explicit exact model-native XAU dataset directory; no stale default is allowed.",
    )
    parser.add_argument("--stem", required=True, help="Explicit immutable dataset split stem; discovery is forbidden.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--data-splits", default="train,val,test")
    parser.add_argument("--max-rows-per-split", type=int, default=25000)
    parser.add_argument("--max-row-groups-per-split", type=int, default=5)
    parser.add_argument("--support-dominance-min", type=float, default=0.25)
    parser.add_argument("--min-pocket-rows", type=int, default=30)
    parser.add_argument("--min-channel-position-delta", type=float, default=0.05)
    parser.add_argument("--max-channel-position-support-corr", type=float, default=-0.05)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
