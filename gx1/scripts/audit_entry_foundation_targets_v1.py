#!/usr/bin/env python3
"""Machine-check Entry target foundation before training."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from gx1.contracts.signal_bridge_v3 import ORDERED_CTX_CAT_NAMES_V3
from gx1.scripts.evaluate_entry_selective_edge_v1 import CLASS_NAMES, SESSION_NAMES, _split_files


DEFAULT_FOUNDATION_DATASET_DIR = Path(
    "/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/"
    "v10_6yr_rebuild_20260628_foundation_seq146/v10_dataset_foundation_seq146_neutral"
)
DEFAULT_OUT_DIR = Path(
    "/home/andre2/GX1_DATA/reports/entry_target_foundation_audit_20260628_v1/foundation_seq146"
)

BASE_TARGET_COLUMNS = [
    "y_direction",
    "y_tradable",
    "y_bad_path",
    "path_quality_bps",
    "mae_first_n_bps",
    "mfe_first_n_bps",
    "label_horizon_bars",
    "path_quality_horizon_bars",
]
SIDE_TARGET_COLUMNS = [
    "y_clean_edge_long",
    "y_survival_long",
    "y_tail_mae_long_K48",
    "y_tail_mae_long_K96",
    "y_clean_edge_short",
    "y_survival_short",
    "y_tail_mae_short_K48",
    "y_tail_mae_short_K96",
    "y_clean_edge_bidir",
    "y_survival_bidir",
]
OPTIONAL_TARGET_COLUMNS = [
    "y_early_move",
    "y_quality_score",
    "y_tf_agreement_score",
    "y_position_size_target",
    "y_hold_horizon_target",
    "y_forecast_ret_K1",
    "y_forecast_ret_K5",
    "y_forecast_ret_K12",
    "y_forecast_ret_K24",
    "y_vol_fwd_K12",
    "y_vol_fwd_K48",
    "y_vol_fwd_K96",
]
DEEP_AUX_TARGET_COLUMNS = (
    [f"y_dip_mae_{side}_K{h}" for side in ("long", "short") for h in (12, 48, 96)]
    + [f"y_dip_mfe_{side}_K{h}" for side in ("long", "short") for h in (12, 48, 96)]
    + [f"y_dip_bottom_frac_{side}_K{h}" for side in ("long", "short") for h in (12, 48, 96)]
    + [f"y_tail_mae_{side}_K{h}" for side in ("long", "short") for h in (12, 48, 96)]
)
XAU_DIRECTION_REPAIR_TARGET_COLUMNS = [
    "y_trade",
    "y_side",
    "y_side_mask",
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
]
ALL_TARGET_COLUMNS = list(dict.fromkeys(
    BASE_TARGET_COLUMNS
    + SIDE_TARGET_COLUMNS
    + OPTIONAL_TARGET_COLUMNS
    + list(DEEP_AUX_TARGET_COLUMNS)
    + XAU_DIRECTION_REPAIR_TARGET_COLUMNS
))

BASE_ACTIVE_TRAINING_HEADS = (
    "direction",
    "tradable",
    "path_quality",
    "mfe_first_n",
    "bad_path",
    "clean_edge",
    "survival",
)
EXPECTED_ACTIVE_OPTIONAL_HEADS = (
    "tf_agreement",
    "path_quality_log_var",
    "position_size",
    "dip",
    "forecast",
    "timing",
    "tail_risk",
    "vol_forecast",
    "mtf_direction",
)
EXPECTED_BLOCKED_OPTIONAL_HEADS = ("hold_horizon",)
HEAD_TARGET_COLUMNS = {
    "tf_agreement": ("y_tf_agreement_score",),
    "path_quality_log_var": ("path_quality_bps",),
    "position_size": ("y_position_size_target",),
    "hold_horizon": ("y_hold_horizon_target",),
    "dip": tuple(
        [f"y_dip_mae_{side}_K{h}" for side in ("long", "short") for h in (12, 48, 96)]
        + [f"y_dip_mfe_{side}_K{h}" for side in ("long", "short") for h in (12, 48, 96)]
    ),
    "forecast": tuple(f"y_forecast_ret_K{h}" for h in (1, 5, 12, 24)),
    "timing": tuple(f"y_dip_bottom_frac_{side}_K{h}" for side in ("long", "short") for h in (12, 48, 96)),
    "tail_risk": tuple(f"y_tail_mae_{side}_K{h}" for side in ("long", "short") for h in (12, 48, 96)),
    "vol_forecast": tuple(f"y_vol_fwd_K{h}" for h in (12, 48, 96)),
    "mtf_direction": ("y_direction",),
}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _parse_csv(raw: str) -> list[str]:
    return [p.strip() for p in str(raw or "").split(",") if p.strip()]


def _safe_rate(series: pd.Series, value: int | float | bool) -> float | None:
    if series.empty:
        return None
    arr = pd.to_numeric(series, errors="coerce")
    arr = arr[np.isfinite(arr)]
    if arr.empty:
        return None
    return float((arr == value).mean())


def _safe_bool_rate(df: pd.DataFrame, col: str) -> float | None:
    if col not in df or df.empty:
        return None
    arr = pd.to_numeric(df[col], errors="coerce")
    arr = arr[np.isfinite(arr)]
    if arr.empty:
        return None
    return float((arr > 0.5).mean())


def _safe_quantile(df: pd.DataFrame, col: str, q: float) -> float | None:
    if col not in df or df.empty:
        return None
    arr = pd.to_numeric(df[col], errors="coerce")
    arr = arr[np.isfinite(arr)]
    if arr.empty:
        return None
    return float(arr.quantile(q))


def _safe_mean(df: pd.DataFrame, col: str) -> float | None:
    if col not in df or df.empty:
        return None
    arr = pd.to_numeric(df[col], errors="coerce")
    arr = arr[np.isfinite(arr)]
    if arr.empty:
        return None
    return float(arr.mean())


def _safe_spearman(df: pd.DataFrame, a: str, b: str) -> float | None:
    if a not in df or b not in df or len(df) < 3:
        return None
    x = pd.to_numeric(df[a], errors="coerce")
    y = pd.to_numeric(df[b], errors="coerce")
    ok = x.notna() & y.notna()
    if int(ok.sum()) < 3 or x[ok].nunique() < 2 or y[ok].nunique() < 2:
        return None
    val = x[ok].corr(y[ok], method="spearman")
    return float(val) if np.isfinite(val) else None


def _ctx_cat_frame(ctx_cat: pd.Series) -> pd.DataFrame:
    arr = np.stack(ctx_cat.to_list()).astype(np.int64, copy=False)
    out: dict[str, np.ndarray] = {}
    for i, name in enumerate(ORDERED_CTX_CAT_NAMES_V3):
        if i < arr.shape[1]:
            out[name] = arr[:, i]
    df = pd.DataFrame(out)
    if "session_id" in df:
        df["session"] = df["session_id"].map(SESSION_NAMES).fillna("UNKNOWN")
    else:
        df["session"] = "UNKNOWN"
    if "vol_regime_id" in df:
        df["vol_regime"] = df["vol_regime_id"].astype(str)
    else:
        df["vol_regime"] = "UNKNOWN"
    if "H4_trend_sign_cat" in df:
        df["h4_trend_regime"] = df["H4_trend_sign_cat"].astype(str)
    else:
        df["h4_trend_regime"] = "UNKNOWN"
    return df


def _load_split(path: Path, split: str) -> tuple[pd.DataFrame, list[str]]:
    schema = pq.read_schema(path)
    available = set(schema.names)
    required = set(BASE_TARGET_COLUMNS) | {"ctx_cat"}
    missing_required = sorted(required - available)
    cols = [c for c in ALL_TARGET_COLUMNS + ["ctx_cat"] if c in available]
    df = pd.read_parquet(path, columns=cols)
    df["split"] = split
    if "ctx_cat" in df:
        ctx = _ctx_cat_frame(df["ctx_cat"])
        df = pd.concat([df.drop(columns=["ctx_cat"]).reset_index(drop=True), ctx.reset_index(drop=True)], axis=1)
    return df, missing_required


def _column_liveness(df: pd.DataFrame, col: str) -> dict[str, Any]:
    if col not in df:
        return {
            "column": col,
            "present": False,
            "finite_rate": 0.0,
            "std": None,
            "unique_count": 0,
            "min": None,
            "max": None,
            "live": False,
        }
    values = pd.to_numeric(df[col], errors="coerce")
    finite = values[np.isfinite(values)]
    std = float(finite.std(ddof=0)) if len(finite) else None
    unique_count = int(finite.nunique(dropna=True)) if len(finite) else 0
    return {
        "column": col,
        "present": True,
        "finite_rate": float(len(finite) / max(len(values), 1)),
        "std": std,
        "unique_count": unique_count,
        "min": float(finite.min()) if len(finite) else None,
        "max": float(finite.max()) if len(finite) else None,
        "live": bool(len(finite) == len(values) and unique_count >= 2 and (std is not None and std > 1e-9)),
    }


def _head_liveness(frames: list[pd.DataFrame]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for head, columns in HEAD_TARGET_COLUMNS.items():
        split_rows: list[dict[str, Any]] = []
        for df in frames:
            split = str(df["split"].iloc[0]) if "split" in df and len(df) else "UNKNOWN"
            column_rows = [_column_liveness(df, col) for col in columns]
            split_rows.append(
                {
                    "split": split,
                    "columns": column_rows,
                    "missing_columns": [row["column"] for row in column_rows if not row["present"]],
                    "dead_columns": [row["column"] for row in column_rows if row["present"] and not row["live"]],
                    "live": all(bool(row["live"]) for row in column_rows),
                }
            )
        live_all = bool(split_rows) and all(bool(row["live"]) for row in split_rows)
        out[head] = {
            "target_columns": list(columns),
            "live_all_splits": live_all,
            "splits": split_rows,
        }
    return out


def _head_contract(frames: list[pd.DataFrame]) -> dict[str, Any]:
    head_liveness = _head_liveness(frames)
    expected_active = list(BASE_ACTIVE_TRAINING_HEADS + EXPECTED_ACTIVE_OPTIONAL_HEADS)
    expected_blocked = list(EXPECTED_BLOCKED_OPTIONAL_HEADS)
    blocked_reasons: dict[str, str] = {}
    for head in expected_blocked:
        live = bool((head_liveness.get(head) or {}).get("live_all_splits"))
        blocked_reasons[head] = (
            "target is intentionally blocked until liveness is non-constant in every split"
            if not live
            else "target is live but wrapper contract still blocks this head pending explicit approval"
        )
    return {
        "base_active_heads": list(BASE_ACTIVE_TRAINING_HEADS),
        "expected_active_optional_heads": list(EXPECTED_ACTIVE_OPTIONAL_HEADS),
        "expected_blocked_optional_heads": expected_blocked,
        "active_training_heads": expected_active,
        "blocked_heads": expected_blocked,
        "blocked_head_reasons": blocked_reasons,
        "head_target_liveness": head_liveness,
    }


def _xau_direction_repair_liveness(frames: list[pd.DataFrame]) -> dict[str, Any]:
    present_columns = sorted(
        {
            col
            for df in frames
            for col in XAU_DIRECTION_REPAIR_TARGET_COLUMNS
            if col in df.columns
        }
    )
    split_rows: list[dict[str, Any]] = []
    for df in frames:
        split = str(df["split"].iloc[0]) if "split" in df and len(df) else "UNKNOWN"
        column_rows = [_column_liveness(df, col) for col in present_columns]
        split_rows.append(
            {
                "split": split,
                "columns": column_rows,
                "missing_columns": [col for col in XAU_DIRECTION_REPAIR_TARGET_COLUMNS if col not in df.columns],
                "dead_columns": [row["column"] for row in column_rows if row["present"] and not row["live"]],
                "live": all(bool(row["live"]) for row in column_rows) if column_rows else False,
            }
        )
    missing_any = sorted(
        {
            col
            for row in split_rows
            for col in row["missing_columns"]
        }
    )
    dead_any = sorted(
        {
            col
            for row in split_rows
            for col in row["dead_columns"]
        }
    )
    return {
        "expected_columns": list(XAU_DIRECTION_REPAIR_TARGET_COLUMNS),
        "present_columns": present_columns,
        "enabled": bool(present_columns),
        "missing_columns_any_split": missing_any,
        "dead_columns_any_split": dead_any,
        "all_expected_columns_present_all_splits": bool(present_columns)
        and not missing_any
        and set(present_columns) == set(XAU_DIRECTION_REPAIR_TARGET_COLUMNS),
        "live_all_present_columns_all_splits": bool(present_columns) and all(bool(row["live"]) for row in split_rows),
        "live_all_expected_columns_all_splits": bool(present_columns)
        and not missing_any
        and not dead_any
        and set(present_columns) == set(XAU_DIRECTION_REPAIR_TARGET_COLUMNS),
        "splits": split_rows,
    }


def _xau_side_quality_row(df: pd.DataFrame, *, split: str, side: str) -> dict[str, Any]:
    bad_col = f"y_{side}_bad_path"
    utility_col = f"y_{side}_path_utility_bps"
    mae_col = f"y_{side}_expected_mae_bps"
    bad = pd.to_numeric(df.get(bad_col, pd.Series(dtype=float)), errors="coerce")
    utility = pd.to_numeric(df.get(utility_col, pd.Series(dtype=float)), errors="coerce")
    mae = pd.to_numeric(df.get(mae_col, pd.Series(dtype=float)), errors="coerce")
    bad_mask = bad > 0.5
    clean_mask = bad <= 0.5
    utility_bad = utility[bad_mask & utility.notna()]
    utility_clean = utility[clean_mask & utility.notna()]
    mae_bad = mae[bad_mask & mae.notna()]
    mae_clean = mae[clean_mask & mae.notna()]
    utility_corr = _safe_spearman(df, bad_col, utility_col)
    mae_corr = _safe_spearman(df, bad_col, mae_col)
    missing = [col for col in (bad_col, utility_col, mae_col) if col not in df.columns]
    bad_live = int(bad.dropna().nunique()) >= 2
    return {
        "split": split,
        "side": side,
        "missing_columns": missing,
        "bad_path_rate": None if bad.dropna().empty else float(bad_mask[bad.notna()].mean()),
        "bad_path_unique_count": int(bad.dropna().nunique()) if not bad.dropna().empty else 0,
        "bad_path_vs_utility_spearman": utility_corr,
        "bad_path_vs_expected_mae_spearman": mae_corr,
        "utility_mean_clean_path_bps": float(utility_clean.mean()) if len(utility_clean) else None,
        "utility_mean_bad_path_bps": float(utility_bad.mean()) if len(utility_bad) else None,
        "expected_mae_mean_clean_path_bps": float(mae_clean.mean()) if len(mae_clean) else None,
        "expected_mae_mean_bad_path_bps": float(mae_bad.mean()) if len(mae_bad) else None,
        "ok": (
            not missing
            and bad_live
            and utility_corr is not None
            and mae_corr is not None
            and float(utility_corr) <= -0.10
            and float(mae_corr) >= 0.10
            and len(utility_bad) > 0
            and len(utility_clean) > 0
            and float(utility_bad.mean()) < float(utility_clean.mean())
            and len(mae_bad) > 0
            and len(mae_clean) > 0
            and float(mae_bad.mean()) > float(mae_clean.mean())
        ),
    }


def _xau_direction_repair_side_quality_contract(frames: list[pd.DataFrame]) -> dict[str, Any]:
    enabled = any(
        any(col in df.columns for col in XAU_DIRECTION_REPAIR_TARGET_COLUMNS)
        for df in frames
    )
    rows: list[dict[str, Any]] = []
    for df in frames:
        split = str(df["split"].iloc[0]) if "split" in df and len(df) else "UNKNOWN"
        for side in ("long", "short"):
            rows.append(_xau_side_quality_row(df, split=split, side=side))
    return {
        "enabled": bool(enabled),
        "description": (
            "XAU direction repair validates side-specific bad-path targets against "
            "side utility and expected MAE. Scalar y_bad_path is selected-side sparse "
            "and is not the primary monotonic quality surface."
        ),
        "rows": rows,
        "all_side_quality_checks_pass": bool(enabled) and all(bool(row["ok"]) for row in rows),
        "failures": [
            (
                f"{row['split']} {row['side']}: side bad-path must be negatively related "
                "to side utility and positively related to expected MAE"
            )
            for row in rows
            if enabled and not bool(row["ok"])
        ],
    }


def _target_metrics(df: pd.DataFrame, *, split: str, scope: str, value: str, side: str = "ALL") -> dict[str, Any]:
    if side == "LONG":
        clean_col, survival_col, tail48_col, tail96_col = (
            "y_clean_edge_long",
            "y_survival_long",
            "y_tail_mae_long_K48",
            "y_tail_mae_long_K96",
        )
    elif side == "SHORT":
        clean_col, survival_col, tail48_col, tail96_col = (
            "y_clean_edge_short",
            "y_survival_short",
            "y_tail_mae_short_K48",
            "y_tail_mae_short_K96",
        )
    else:
        clean_col, survival_col, tail48_col, tail96_col = (
            "y_clean_edge_bidir",
            "y_survival_bidir",
            "y_tail_mae_long_K48",
            "y_tail_mae_short_K48",
        )
    y = pd.to_numeric(df["y_direction"], errors="coerce") if "y_direction" in df else pd.Series(dtype=float)
    class_rates = {
        CLASS_NAMES.get(int(cls), str(cls)).lower(): _safe_rate(y, int(cls))
        for cls in sorted(CLASS_NAMES)
    }
    majority = max([rate for rate in class_rates.values() if rate is not None] or [0.0])
    return {
        "split": split,
        "scope": scope,
        "value": value,
        "side": side,
        "n": int(len(df)),
        "y_direction_rates": class_rates,
        "majority_label_baseline_acc": float(majority),
        "neutral_skip_flat_rate": class_rates.get("flat"),
        "trade_label_rate": None if class_rates.get("flat") is None else float(1.0 - class_rates["flat"]),
        "y_tradable_rate": _safe_bool_rate(df, "y_tradable"),
        "y_bad_path_rate": _safe_bool_rate(df, "y_bad_path"),
        "path_quality_mean_bps": _safe_mean(df, "path_quality_bps"),
        "path_quality_p10_bps": _safe_quantile(df, "path_quality_bps", 0.10),
        "path_quality_p50_bps": _safe_quantile(df, "path_quality_bps", 0.50),
        "path_quality_p90_bps": _safe_quantile(df, "path_quality_bps", 0.90),
        "mae_first_n_mean_bps": _safe_mean(df, "mae_first_n_bps"),
        "mae_first_n_p90_bps": _safe_quantile(df, "mae_first_n_bps", 0.90),
        "mfe_first_n_mean_bps": _safe_mean(df, "mfe_first_n_bps"),
        "mfe_first_n_p50_bps": _safe_quantile(df, "mfe_first_n_bps", 0.50),
        "mfe_first_n_p90_bps": _safe_quantile(df, "mfe_first_n_bps", 0.90),
        "clean_edge_rate": _safe_bool_rate(df, clean_col),
        "survival_rate": _safe_bool_rate(df, survival_col),
        "tail_mae_k48_p90": _safe_quantile(df, tail48_col, 0.90),
        "tail_mae_k96_p90": _safe_quantile(df, tail96_col, 0.90),
        "bad_path_vs_path_quality_spearman": _safe_spearman(df, "y_bad_path", "path_quality_bps"),
    }


def _group_metrics(df: pd.DataFrame, split: str) -> list[dict[str, Any]]:
    rows = [_target_metrics(df, split=split, scope="split", value="ALL", side="ALL")]
    for col, scope in [("session", "session"), ("vol_regime", "vol_regime"), ("h4_trend_regime", "h4_trend_regime")]:
        if col in df:
            for value, part in df.groupby(col, dropna=False):
                rows.append(_target_metrics(part, split=split, scope=scope, value=str(value), side="ALL"))
    for side in ("LONG", "SHORT"):
        rows.append(_target_metrics(df, split=split, scope="side", value=side, side=side))
    return rows


def _drift(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    key_metrics = [
        "y_tradable_rate",
        "y_bad_path_rate",
        "path_quality_mean_bps",
        "majority_label_baseline_acc",
        "trade_label_rate",
    ]
    by_key = {(m["split"], m["scope"], m["value"], m["side"]): m for m in metrics}
    out: list[dict[str, Any]] = []
    for key, train in by_key.items():
        split, scope, value, side = key
        if split != "train":
            continue
        for other_split in ("val", "test"):
            other = by_key.get((other_split, scope, value, side))
            if not other:
                continue
            row: dict[str, Any] = {"scope": scope, "value": value, "side": side, "split": other_split}
            for metric in key_metrics:
                a = train.get(metric)
                b = other.get(metric)
                row[f"{metric}_delta_vs_train"] = None if a is None or b is None else float(b) - float(a)
            out.append(row)
    return out


def _unique_numeric(df: pd.DataFrame, col: str) -> list[float]:
    if col not in df:
        return []
    vals = pd.to_numeric(df[col], errors="coerce").dropna().unique()
    return sorted(float(v) for v in vals)


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Entry Target Foundation Audit",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Failure count: `{len(report['failures'])}`",
        f"- Direction target: `{report['target_contract']['direction_target']}`",
        f"- Bad-path role: `{report['target_contract']['bad_path_role']}`",
        f"- Objective: `{report['target_contract']['trading_objective']}`",
        "",
        "## Failures",
        "",
    ]
    if report["failures"]:
        lines.extend([f"- {failure}" for failure in report["failures"]])
    else:
        lines.append("- None")
    lines.extend(["", "## Split Metrics", ""])
    for row in report["metrics"]:
        if row["scope"] == "split":
            lines.append(
                f"- `{row['split']}` n={row['n']} tradable={row['y_tradable_rate']} "
                f"bad_path={row['y_bad_path_rate']} path_mean={row['path_quality_mean_bps']} "
                f"bad_path_corr={row['bad_path_vs_path_quality_spearman']}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = _parse_csv(args.data_splits)

    failures: list[str] = []
    frames: list[pd.DataFrame] = []
    split_paths: dict[str, str] = {}
    try:
        files = _split_files(dataset_dir, splits)
    except Exception as exc:
        files = {}
        failures.append(f"dataset split resolution failed: {exc}")

    missing_required_by_split: dict[str, list[str]] = {}
    for split in splits:
        path = files.get(split)
        if path is None:
            missing_required_by_split[split] = list(BASE_TARGET_COLUMNS)
            continue
        split_paths[split] = str(path)
        try:
            df, missing = _load_split(path, split)
        except Exception as exc:
            failures.append(f"{split}: target load failed: {exc}")
            missing_required_by_split[split] = list(BASE_TARGET_COLUMNS)
            continue
        missing_required_by_split[split] = missing
        if missing:
            failures.append(f"{split}: missing required target columns: {missing}")
        frames.append(df)

    metrics: list[dict[str, Any]] = []
    label_horizons: dict[str, list[float]] = {}
    path_quality_horizons: dict[str, list[float]] = {}
    for df in frames:
        split = str(df["split"].iloc[0])
        metrics.extend(_group_metrics(df, split))
        label_horizons[split] = _unique_numeric(df, "label_horizon_bars")
        path_quality_horizons[split] = _unique_numeric(df, "path_quality_horizon_bars")

    for split, values in label_horizons.items():
        if values != [24.0]:
            failures.append(f"{split}: label_horizon_bars expected [24], observed {values}")
    for split, values in path_quality_horizons.items():
        if values != [10.0]:
            failures.append(f"{split}: path_quality_horizon_bars expected [10], observed {values}")
    xau_side_quality_contract = _xau_direction_repair_side_quality_contract(frames)
    for row in metrics:
        if row["scope"] == "split":
            corr = row.get("bad_path_vs_path_quality_spearman")
            if not xau_side_quality_contract["enabled"]:
                if corr is None:
                    failures.append(f"{row['split']}: y_bad_path/path_quality correlation unavailable")
                elif float(corr) >= 0.0:
                    failures.append(f"{row['split']}: y_bad_path should be negatively related to path_quality_bps, got {corr}")
            majority = row.get("majority_label_baseline_acc")
            if majority is not None and float(majority) > float(args.max_majority_rate):
                failures.append(f"{row['split']}: y_direction majority label collapsed: {majority}")
            tradable = row.get("y_tradable_rate")
            if tradable is not None and (float(tradable) < 0.01 or float(tradable) > 0.99):
                failures.append(f"{row['split']}: y_tradable near-constant: {tradable}")

    target_head_contract = _head_contract(frames)
    head_liveness = target_head_contract["head_target_liveness"]
    xau_repair_liveness = _xau_direction_repair_liveness(frames)
    if xau_repair_liveness["enabled"] and not xau_repair_liveness["all_expected_columns_present_all_splits"]:
        failures.append(
            "xau direction-repair target columns are present but missing expected columns in at least one split: "
            f"{xau_repair_liveness.get('missing_columns_any_split')}"
        )
    if xau_repair_liveness["enabled"] and not xau_repair_liveness["live_all_expected_columns_all_splits"]:
        failures.append("xau direction-repair target columns are present but not live in all splits")
    if xau_side_quality_contract["enabled"] and not xau_side_quality_contract["all_side_quality_checks_pass"]:
        failures.extend(xau_side_quality_contract["failures"])
    for head in EXPECTED_ACTIVE_OPTIONAL_HEADS:
        if not bool((head_liveness.get(head) or {}).get("live_all_splits")):
            failures.append(f"expected active optional head target is not live in all splits: {head}")
    for head in EXPECTED_BLOCKED_OPTIONAL_HEADS:
        if head not in set(target_head_contract["blocked_heads"]):
            failures.append(f"expected blocked optional head missing from blocked_heads: {head}")

    drift = _drift(metrics)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target_contract = {
        "direction_target": "H=24 direction label; threshold bps must remain explicit in dataset builder/config before training",
        "direction_horizon_expected_bars": 24,
        "path_quality_horizon_expected_bars": 10,
        "bad_path_role": "separate head plus sizing/gating diagnostic; do not fold into direction accuracy",
        "trading_objective": "offline replay/PnL/drawdown/tail-risk, not validation accuracy alone",
        "active_training_heads": target_head_contract["active_training_heads"],
        "blocked_heads": target_head_contract["blocked_heads"],
        "xau_direction_repair_side_quality_contract": xau_side_quality_contract,
        "approval_status": "MACHINE_AUDITED_NOT_HUMAN_APPROVED",
    }
    report = {
        "schema_version": "entry_target_foundation_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "PASS" if not failures else "FAIL",
        "dataset_dir": str(dataset_dir),
        "data_splits": splits,
        "split_paths": split_paths,
        "missing_required_by_split": missing_required_by_split,
        "label_horizons": label_horizons,
        "path_quality_horizons": path_quality_horizons,
        "target_contract": target_contract,
        "target_head_contract": target_head_contract,
        "xau_direction_repair_target_liveness": xau_repair_liveness,
        "xau_direction_repair_side_quality_contract": xau_side_quality_contract,
        "metrics": metrics,
        "drift": drift,
        "failures": failures,
    }

    json_path = out_dir / f"ENTRY_TARGET_FOUNDATION_AUDIT_{timestamp}.json"
    md_path = out_dir / f"ENTRY_TARGET_FOUNDATION_AUDIT_{timestamp}.md"
    report["json_path"] = str(json_path)
    report["md_path"] = str(md_path)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    _write_markdown(md_path, report)
    latest_json = out_dir / "ENTRY_TARGET_FOUNDATION_AUDIT_latest.json"
    latest_md = out_dir / "ENTRY_TARGET_FOUNDATION_AUDIT_latest.md"
    latest_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    latest_md.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")

    if not args.quiet:
        print(json.dumps({k: report[k] for k in ["decision", "failures", "json_path", "md_path"]}, indent=2, default=_json_default))
    if args.fail_on_audit_fail and failures:
        raise SystemExit(2)
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", default=str(DEFAULT_FOUNDATION_DATASET_DIR))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--data-splits", default="train,val,test")
    ap.add_argument("--max-majority-rate", type=float, default=0.90)
    ap.add_argument("--fail-on-audit-fail", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
