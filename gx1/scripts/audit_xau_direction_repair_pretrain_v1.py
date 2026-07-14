#!/usr/bin/env python3
"""Fail-closed XAU direction-repair dataset audit before smart training."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


DEFAULT_DATASET_DIR = Path(
    "/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/"
    "v10_6yr_rebuild_20260626_spreadfix/"
    "v10_dataset_6yr_smartctx_xau_direction_repair"
)
DEFAULT_STEM = "v10_6yr_dataset__HOLD_03B"
DEFAULT_OUT_DIR = Path(
    "/home/andre2/GX1_DATA/reports/xau_direction_repair_pretrain_audit_20260713_v1"
)

CHANNEL_POSITION_FEATURE = "chart.geometry_channel_position_low_to_high"
SUPPORT_STACK_FEATURE = "chart.geometry_support_line_proximity_stack"
RESISTANCE_STACK_FEATURE = "chart.geometry_resistance_line_proximity_stack"
SUPPORT_MINUS_RESISTANCE_FEATURE = "chart.geometry_support_minus_resistance_stack"

REQUIRED_POLARITY_FEATURES = (
    SUPPORT_STACK_FEATURE,
    RESISTANCE_STACK_FEATURE,
    SUPPORT_MINUS_RESISTANCE_FEATURE,
    CHANNEL_POSITION_FEATURE,
)
REQUIRED_RAIL_FEATURES = (
    "chart.geometry_rising_support_rail_long_pressure",
    "chart.geometry_rising_support_rail_short_trap_pressure",
    "chart.geometry_falling_resistance_rail_short_pressure",
    "chart.geometry_falling_resistance_rail_long_trap_pressure",
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
OPTIONAL_XAU_AUDIT_COLUMNS = (
    "y_direction_long_score_bps",
    "y_direction_short_score_bps",
)


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
    bridge = extra.get("signal_bridge") if isinstance(extra.get("signal_bridge"), dict) else {}
    state_contract = extra.get("smart520_state_contract")
    return {
        "neutral_xgb_bridge": bool(
            manifest.get("neutral_xgb_bridge", False)
            or bridge.get("neutral_xgb_bridge", False)
        ),
        "xgb_bridge_source": str(
            manifest.get("xgb_bridge_source")
            or bridge.get("bridge_source")
            or extra.get("xgb_bridge_source")
            or ""
        ),
        "tape_root": str(manifest.get("tape_root") or extra.get("tape_root") or ""),
        "smart520_state_contract": dict(state_contract) if isinstance(state_contract, dict) else {},
    }


def _state_contract_failures(contract: dict[str, Any], *, split: str) -> list[str]:
    failures: list[str] = []
    required = {
        "schema_version",
        "frame_anchor_utc",
        "model_range_start_utc",
        "rank_reference_end_utc",
        "rank_reference_npz",
        "rank_reference_npz_sha256",
    }
    if not isinstance(contract, dict) or not contract:
        return [f"{split}: XAU repair requires smart520_state_contract provenance"]
    missing = sorted(required - set(contract))
    if missing:
        failures.append(f"{split}: smart520_state_contract missing fields: {','.join(missing)}")
    if str(contract.get("schema_version") or "") != "smart520_state_contract_v1":
        failures.append(
            f"{split}: smart520_state_contract schema_version invalid: {contract.get('schema_version')!r}"
        )
    rank_ref = str(contract.get("rank_reference_npz") or "").strip()
    rank_ref_low = rank_ref.lower()
    if not rank_ref:
        failures.append(f"{split}: smart520_state_contract rank_reference_npz missing")
    elif not Path(rank_ref).expanduser().is_file():
        failures.append(f"{split}: smart520_state_contract rank_reference_npz missing on disk: {rank_ref}")
    else:
        rank_ref_path = Path(rank_ref).expanduser()
        expected_sha = str(contract.get("rank_reference_npz_sha256") or "").strip().lower()
        actual_sha = _sha256_file(rank_ref_path)
        if expected_sha != actual_sha:
            failures.append(
                f"{split}: smart520_state_contract rank_reference_npz_sha256 mismatch: "
                f"metadata={expected_sha!r} actual={actual_sha} path={rank_ref}"
            )
        sidecar_path = rank_ref_path.with_suffix(rank_ref_path.suffix + ".json")
        if not sidecar_path.is_file():
            failures.append(f"{split}: smart520_state_contract rank reference sidecar missing: {sidecar_path}")
        else:
            try:
                sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
                sidecar_sha = str(sidecar.get("out_npz_sha256") or "").strip().lower()
                if sidecar_sha != expected_sha:
                    failures.append(
                        f"{split}: smart520_state_contract sidecar out_npz_sha256 mismatch: "
                        f"sidecar={sidecar_sha!r} metadata={expected_sha!r}"
                    )
            except Exception as exc:
                failures.append(f"{split}: smart520_state_contract rank reference sidecar unreadable: {sidecar_path}: {exc}")
    for marker in ("julyext", "smart_candidate_20260630", "utilityrepair", "20260710"):
        if marker in rank_ref_low:
            failures.append(
                f"{split}: smart520_state_contract rank_reference_npz references stale marker "
                f"{marker!r}: {rank_ref}"
            )
    parsed_ts: dict[str, pd.Timestamp] = {}
    for key in ("frame_anchor_utc", "model_range_start_utc", "rank_reference_end_utc"):
        try:
            ts = pd.to_datetime(str(contract.get(key) or ""), utc=True, errors="coerce")
            if pd.isna(ts):
                raise ValueError("NaT")
            parsed_ts[key] = ts
        except Exception:
            failures.append(f"{split}: smart520_state_contract {key} is not a valid timestamp")
    if {"frame_anchor_utc", "model_range_start_utc", "rank_reference_end_utc"} <= set(parsed_ts):
        if parsed_ts["frame_anchor_utc"] < parsed_ts["model_range_start_utc"]:
            failures.append(f"{split}: smart520_state_contract frame_anchor_utc precedes model_range_start_utc")
        if parsed_ts["rank_reference_end_utc"] < parsed_ts["model_range_start_utc"]:
            failures.append(f"{split}: smart520_state_contract rank_reference_end_utc precedes model_range_start_utc")
        if parsed_ts["frame_anchor_utc"] > parsed_ts["rank_reference_end_utc"]:
            failures.append(f"{split}: smart520_state_contract frame_anchor_utc exceeds rank_reference_end_utc")
    return failures


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
    if requested and requested.lower() != "auto":
        return requested
    candidates: set[str] | None = None
    for split in splits:
        parquet_stems = {
            stem
            for path in dataset_dir.glob(f"*_{split}.parquet")
            if (stem := _stem_from_split_filename(path.name, split=split, suffix=".parquet"))
        }
        manifest_stems = {
            stem
            for path in dataset_dir.glob(f"*_{split}.manifest.json")
            if (stem := _stem_from_split_filename(path.name, split=split, suffix=".manifest.json"))
        }
        split_candidates = parquet_stems & manifest_stems
        candidates = split_candidates if candidates is None else candidates & split_candidates
    resolved = sorted(candidates or [])
    if len(resolved) != 1:
        raise RuntimeError(
            "auto stem resolution expected exactly one common train/val/test stem under "
            f"{dataset_dir}, got {resolved}"
        )
    return resolved[0]


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
        + [name for name in OPTIONAL_XAU_AUDIT_COLUMNS if name in schema_names]
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
    if missing_features:
        return {
            "split": split,
            "parquet_path": str(parquet_path),
            "manifest_path": str(manifest_path),
            "rows_sampled": int(snap.shape[0]),
            "feature_count": int(len(fields)),
            "seq_structure_extension_mode": _seq_structure_mode(manifest),
            "provenance": _manifest_provenance(manifest),
            "missing_polarity_features": missing_features,
            "missing_target_columns": missing_targets,
            "polarity": {"available": False},
            "target_liveness": {},
            "target_consistency": {"available": False},
        }

    support = snap[:, int(feature_index[SUPPORT_STACK_FEATURE])]
    resistance = snap[:, int(feature_index[RESISTANCE_STACK_FEATURE])]
    support_minus_resistance = snap[:, int(feature_index[SUPPORT_MINUS_RESISTANCE_FEATURE])]
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

    target_liveness = {
        name: _column_liveness(np.asarray(sample[name]))
        for name in REQUIRED_XAU_TARGET_COLUMNS
        if name in sample
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
        and y_position_size.size == y_direction.size
    ):
        anti_short = np.zeros_like(y_direction, dtype=bool)
        anti_long = np.zeros_like(y_direction, dtype=bool)
        for name in (
            "y_rising_channel_support_touch",
            "y_support_retest_continuation",
            "y_countertrend_short_trap",
            "y_short_high_mae_low_mfe_early_failure",
        ):
            if name in sample:
                anti_short |= np.asarray(sample[name], dtype=np.float32) > 0.5
        for name in (
            "y_falling_channel_resistance_touch",
            "y_resistance_retest_continuation",
            "y_countertrend_long_trap",
            "y_long_high_mae_low_mfe_early_failure",
        ):
            if name in sample:
                anti_long |= np.asarray(sample[name], dtype=np.float32) > 0.5
        anti_short_only = anti_short & (~anti_long)
        anti_long_only = anti_long & (~anti_short)
        conflict_rows = anti_short & anti_long
        repaired_scalar_bad = np.zeros_like(y_bad_path, dtype=np.float32)
        long_rows = (y_trade > 0.5) & (y_side == 0)
        short_rows = (y_trade > 0.5) & (y_side == 1)
        flat_rows = y_trade <= 0.5
        repaired_scalar_bad[long_rows] = y_long_bad_path[long_rows]
        repaired_scalar_bad[short_rows] = y_short_bad_path[short_rows]
        expected_mfe = np.zeros_like(mfe_first, dtype=np.float32)
        expected_mae = np.zeros_like(mae_first, dtype=np.float32)
        expected_mfe[long_rows] = mfe_long[long_rows]
        expected_mae[long_rows] = mae_long[long_rows]
        expected_mfe[short_rows] = mfe_short[short_rows]
        expected_mae[short_rows] = mae_short[short_rows]
        expected_path = (expected_mfe - expected_mae).astype(np.float32)
        bad_path_mismatch = np.abs(y_bad_path - repaired_scalar_bad) > 1e-5
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
            "anti_short_only_rows": int(anti_short_only.sum()),
            "anti_long_only_rows": int(anti_long_only.sum()),
            "conflict_rows": int(conflict_rows.sum()),
            "anti_short_direction_short_count": int(np.sum(anti_short_only & (y_direction == 1))),
            "anti_long_direction_long_count": int(np.sum(anti_long_only & (y_direction == 0))),
            "anti_short_masked_short_count": int(np.sum(anti_short_only & (y_side_mask > 0.5) & (y_side == 1))),
            "anti_long_masked_long_count": int(np.sum(anti_long_only & (y_side_mask > 0.5) & (y_side == 0))),
            "conflict_not_flat_count": int(
                np.sum(conflict_rows & ((y_direction != 2) | (y_trade > 0.5) | (y_side_mask > 0.5)))
            ),
            "anti_short_short_utility_favorable_count": int(
                np.sum(anti_short_only & (y_short_utility >= y_long_utility))
            ),
            "anti_long_long_utility_favorable_count": int(
                np.sum(anti_long_only & (y_long_utility >= y_short_utility))
            ),
            "anti_short_short_bad_path_not_forced_count": int(
                np.sum(anti_short_only & (y_short_bad_path < 0.999))
            ),
            "anti_long_long_bad_path_not_forced_count": int(
                np.sum(anti_long_only & (y_long_bad_path < 0.999))
            ),
            "anti_short_short_mae_not_higher_count": int(
                np.sum(anti_short_only & (y_short_mae <= y_long_mae))
            ),
            "anti_long_long_mae_not_higher_count": int(
                np.sum(anti_long_only & (y_long_mae <= y_short_mae))
            ),
            "direction_long_score_alias_mismatch_count": int(long_alias_mismatch.sum()),
            "direction_short_score_alias_mismatch_count": int(short_alias_mismatch.sum()),
        }
    anti_short = np.zeros_like(y_direction, dtype=bool)
    anti_long = np.zeros_like(y_direction, dtype=bool)
    for name in (
        "y_rising_channel_support_touch",
        "y_support_retest_continuation",
        "y_countertrend_short_trap",
        "y_short_high_mae_low_mfe_early_failure",
    ):
        if name in sample:
            anti_short |= np.asarray(sample[name], dtype=np.float32) > 0.5
    for name in (
        "y_falling_channel_resistance_touch",
        "y_resistance_retest_continuation",
        "y_countertrend_long_trap",
        "y_long_high_mae_low_mfe_early_failure",
    ):
        if name in sample:
            anti_long |= np.asarray(sample[name], dtype=np.float32) > 0.5
    anti_short_wrong_rate = _safe_rate((y_direction == 1) & anti_short) if y_direction.size else None
    anti_long_wrong_rate = _safe_rate((y_direction == 0) & anti_long) if y_direction.size else None

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
        "polarity": {
            "available": True,
            "support_dominance_min": float(support_dominance_min),
            "support_dominant_rows": int(support_dom.sum()),
            "resistance_dominant_rows": int(resistance_dom.sum()),
            "support_dominant_channel_position_mean": support_mean,
            "resistance_dominant_channel_position_mean": resistance_mean,
            "resistance_minus_support_channel_position_mean": delta,
            "channel_position_vs_support_minus_resistance_corr": corr,
            "support_dominant_channel_position_lt_042_rate": _safe_rate(channel_position[support_dom] < 0.42)
            if int(support_dom.sum())
            else None,
            "resistance_dominant_channel_position_gt_058_rate": _safe_rate(channel_position[resistance_dom] > 0.58)
            if int(resistance_dom.sum())
            else None,
            "enough_pocket_rows": bool(
                int(support_dom.sum()) >= int(min_pocket_rows)
                and int(resistance_dom.sum()) >= int(min_pocket_rows)
            ),
        },
        "target_liveness": target_liveness,
        "target_consistency": target_consistency,
        "anti_wrong_side_rates": {
            "anti_short_rows_labeled_short_rate_all_rows": anti_short_wrong_rate,
            "anti_long_rows_labeled_long_rate_all_rows": anti_long_wrong_rate,
        },
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

    if args.require_rail_features and missing_rail_features:
        failures.append(f"missing required XAU rail features in manifest: {missing_rail_features}")

    stale_markers = ("utilityrepair", "20260710", "smart_candidate_20260630", "julyext")
    dataset_dir_text = str(dataset_dir).lower()
    for marker in stale_markers:
        if marker in dataset_dir_text:
            failures.append(f"dataset_dir contains known stale XAU repair marker {marker!r}: {dataset_dir}")

    require_inline_seq_structure = bool(getattr(args, "require_inline_seq_structure", True))
    require_xau_provenance = bool(getattr(args, "require_xau_provenance", True))
    for row in split_reports:
        split = str(row.get("split"))
        seq_mode = row.get("seq_structure_extension_mode")
        if require_inline_seq_structure and seq_mode != "inline_from_merged3":
            failures.append(
                f"{split}: XAU repair requires inline seq-structure features; observed mode={seq_mode}"
            )
        provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
        if require_xau_provenance:
            if provenance.get("neutral_xgb_bridge") is not True:
                failures.append(f"{split}: XAU repair requires neutral_xgb_bridge=true provenance")
            if str(provenance.get("xgb_bridge_source") or "") != "neutral_uniform_proba":
                failures.append(
                    f"{split}: XAU repair requires xgb_bridge_source=neutral_uniform_proba; "
                    f"observed={provenance.get('xgb_bridge_source')!r}"
                )
            tape_root = str(provenance.get("tape_root") or "").lower()
            if "xauusd" not in tape_root:
                failures.append(f"{split}: XAU repair requires XAUUSD tape_root provenance; observed={tape_root!r}")
            failures.extend(
                _state_contract_failures(
                    provenance.get("smart520_state_contract") if isinstance(provenance.get("smart520_state_contract"), dict) else {},
                    split=split,
                )
            )
        for name in row.get("missing_polarity_features") or []:
            failures.append(f"{split}: missing channel-polarity feature: {name}")
        for name in row.get("missing_target_columns") or []:
            failures.append(f"{split}: missing XAU repair target column: {name}")
        for name, live in (row.get("target_liveness") or {}).items():
            if not bool(live.get("live")):
                failures.append(f"{split}: XAU repair target column is not live: {name}")
        consistency = row.get("target_consistency") if isinstance(row.get("target_consistency"), dict) else {}
        if not bool(consistency.get("available")):
            failures.append(f"{split}: XAU repair target consistency audit unavailable")
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
                ("anti_short_direction_short_count", "anti-short structural rows still labeled SHORT"),
                ("anti_long_direction_long_count", "anti-long structural rows still labeled LONG"),
                ("anti_short_masked_short_count", "anti-short structural rows still teach SHORT through side mask"),
                ("anti_long_masked_long_count", "anti-long structural rows still teach LONG through side mask"),
                ("conflict_not_flat_count", "conflict structural rows are not FLAT/no-trade"),
                ("anti_short_short_utility_favorable_count", "anti-short structural rows still have SHORT utility >= LONG utility"),
                ("anti_long_long_utility_favorable_count", "anti-long structural rows still have LONG utility >= SHORT utility"),
                ("anti_short_short_bad_path_not_forced_count", "anti-short structural rows do not force SHORT bad-path target"),
                ("anti_long_long_bad_path_not_forced_count", "anti-long structural rows do not force LONG bad-path target"),
                ("anti_short_short_mae_not_higher_count", "anti-short structural rows do not make SHORT expected MAE worse"),
                ("anti_long_long_mae_not_higher_count", "anti-long structural rows do not make LONG expected MAE worse"),
                ("direction_long_score_alias_mismatch_count", "y_direction_long_score_bps mismatches repaired long utility"),
                ("direction_short_score_alias_mismatch_count", "y_direction_short_score_bps mismatches repaired short utility"),
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

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "schema_version": "xau_direction_repair_pretrain_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "PASS" if not failures else "FAIL",
        "dataset_dir": str(dataset_dir),
        "requested_stem": requested_stem,
        "stem": str(stem or requested_stem),
        "data_splits": splits,
        "require_rail_features": bool(args.require_rail_features),
        "require_inline_seq_structure": require_inline_seq_structure,
        "require_xau_provenance": require_xau_provenance,
        "required_rail_features": list(REQUIRED_RAIL_FEATURES),
        "missing_rail_features": missing_rail_features,
        "required_xau_target_columns": list(REQUIRED_XAU_TARGET_COLUMNS),
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
    latest_path = out_dir / "XAU_DIRECTION_REPAIR_PRETRAIN_AUDIT_latest.json"
    report["json_path"] = str(json_path)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    latest_path.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    report["latest_json_path"] = str(latest_path)

    if not args.quiet:
        print(
            json.dumps(
                {
                    "decision": report["decision"],
                    "failures": failures,
                    "json_path": str(json_path),
                    "latest_json_path": str(latest_path),
                },
                indent=2,
                default=_json_default,
            )
        )
    if args.fail_on_audit_fail and failures:
        raise SystemExit(2)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--stem", default=DEFAULT_STEM, help="Dataset split stem, or 'auto' to discover it.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--data-splits", default="train,val,test")
    parser.add_argument("--max-rows-per-split", type=int, default=25000)
    parser.add_argument("--max-row-groups-per-split", type=int, default=5)
    parser.add_argument("--support-dominance-min", type=float, default=0.25)
    parser.add_argument("--min-pocket-rows", type=int, default=30)
    parser.add_argument("--min-channel-position-delta", type=float, default=0.05)
    parser.add_argument("--max-channel-position-support-corr", type=float, default=-0.05)
    parser.add_argument("--require-rail-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-inline-seq-structure", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-xau-provenance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fail-on-audit-fail", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
