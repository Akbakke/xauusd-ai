"""Offline chart-structure feature ablation for Entry no-XGB models.

This runner is intentionally research-only. It builds deterministic chart-
structure features from the existing no-XGB matrix, trains several tabular model
families on 2026 walk-forward folds, and replays policy variants with session
filters and exit grids. It does not touch live, shadow, OANDA, or candidate
promotion wiring.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from gx1.scripts.evaluate_entry_selective_edge_v1 import _json_default, _parse_float_list
from gx1.scripts.evaluate_entry_tabular_no_xgb_baseline_v1 import (
    _check_no_xgb_feature_names,
    _train_lightgbm,
)
from gx1.scripts.evaluate_entry_tabular_no_xgb_walkforward_v1 import (
    FoldSpec,
    _fold_indices,
    _load_all_data,
    _maybe_cap_train_indices,
    _parse_folds,
)
from gx1.scripts.replay_entry_tabular_no_xgb_policy_v1 import (
    SourceTape,
    _cost_label,
    _decision_arrays,
    _frac_label,
    _metrics_row,
    _policy_hash,
    _run_policy,
    _threshold_from_scores,
)
from gx1.features.entry_foundation_structure_v1 import build_entry_foundation_structure_layer
from gx1.features.entry_chart_geometry_v1 import build_entry_chart_geometry_layer
from gx1.features.entry_candlestick_patterns_v1 import build_entry_candlestick_pattern_layer


DEFAULT_DATASET_DIR = Path(
    "/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/"
    "v10_6yr_rebuild_20260626_spreadfix/v10_dataset_6yr_smartctx_xgbfixed_h24_cwp030"
)
DEFAULT_SOURCE_PARQUET = Path(
    "/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/"
    "v10_6yr_rebuild_20260626_spreadfix/FULL_PLUS_CTX_v3src.parquet"
)
DEFAULT_OUT_DIR = Path("/home/andre2/GX1_DATA/reports/entry_chart_structure_ablation_20260627_v1")
DEFAULT_FOLDS = "2026YTD=2026-01-01:2026-05-01,2026HOLDOUT=2026-05-01:2026-06-13"
DEFAULT_BASELINE_METRICS = Path(
    "/home/andre2/GX1_DATA/reports/"
    "entry_selective_edge_20260627_tabular_no_xgb_policy_replay_2026_only_slip5/"
    "replay_policy_metrics.csv"
)


CHART_NAME_PARTS = (
    "struct",
    "smc_",
    "swing",
    "bos",
    "choch",
    "sweep",
    "wick",
    "pivot",
    "sr_",
    "liquidity",
    "pullback",
    "impulse",
    "compression",
    "retracement",
    "premium",
    "discount",
    "geometry",
    "fib_",
    "trendline",
    "channel",
    "triangle",
    "flag_pullback",
    "regime",
    "trend_age",
    "dist_last_swing",
)
VOL_NAME_PARTS = ("atr", "vol", "range", "squeeze", "bandwidth")
SESSION_NAME_PARTS = ("session", "hour", "dow", "is_asia", "is_eu", "is_us", "overlap")
FEATURE_LAYER_VERSION = "chart_structure_deep_policy_hash_v2_20260628"
PRICE_EMA_FEATURE_LAYER_VERSION = "price_ema_v3_20260628"
VETO_ONLY_PRICE_EMA_FEATURE_LAYER_VERSION = "veto_only_price_ema_v1_20260628"


@dataclass(frozen=True)
class FeatureSet:
    name: str
    indices: np.ndarray
    names: list[str]
    categorical_idx: list[int]


@dataclass(frozen=True)
class ExitSpec:
    name: str
    exit_mode: str
    take_profit_bps: float
    stop_loss_bps: float
    same_bar_policy: str


@dataclass(frozen=True)
class SessionPolicy:
    name: str
    include: frozenset[str] | None = None
    exclude: frozenset[str] | None = None


@dataclass(frozen=True)
class EntryVetoCondition:
    feature: str
    skip_side: str
    quantile: float
    decision_side: str | None = None
    trade_session: str | None = None


@dataclass(frozen=True)
class EntryVetoRule:
    conditions: tuple[EntryVetoCondition, ...]


@dataclass(frozen=True)
class EntryVetoSet:
    name: str
    rules: tuple[EntryVetoRule, ...]


SESSION_POLICIES: dict[str, SessionPolicy] = {
    "ALL": SessionPolicy("ALL"),
    "NO_EU": SessionPolicy("NO_EU", exclude=frozenset({"EU"})),
    "OVERLAP_US": SessionPolicy("OVERLAP_US", include=frozenset({"OVERLAP", "US"})),
    "OVERLAP_ONLY": SessionPolicy("OVERLAP_ONLY", include=frozenset({"OVERLAP"})),
    "US_ONLY": SessionPolicy("US_ONLY", include=frozenset({"US"})),
    "ASIA_ONLY": SessionPolicy("ASIA_ONLY", include=frozenset({"ASIA"})),
    "EU_ONLY": SessionPolicy("EU_ONLY", include=frozenset({"EU"})),
}


def _parse_csv(raw: str) -> list[str]:
    return [p.strip() for p in str(raw).split(",") if p.strip()]


def _model_slug(parts: Iterable[Any]) -> str:
    raw = "__".join(str(p) for p in parts)
    clean = []
    for ch in raw:
        clean.append(ch if ch.isalnum() or ch in ("_", "-", ".") else "_")
    return "".join(clean)


def _name_index(names: list[str]) -> dict[str, int]:
    return {name: i for i, name in enumerate(names)}


def _has_name(index: dict[str, int], name: str) -> bool:
    return name in index


def _col(x: np.ndarray, index: dict[str, int], name: str, default: float = 0.0) -> np.ndarray:
    if name not in index:
        return np.full(x.shape[0], float(default), dtype=np.float32)
    arr = x[:, index[name]].astype(np.float32, copy=False)
    return np.nan_to_num(arr, nan=float(default), posinf=float(default), neginf=float(default))


def _clip(arr: np.ndarray, lo: float = -25.0, hi: float = 25.0) -> np.ndarray:
    return np.clip(np.nan_to_num(arr, nan=0.0, posinf=hi, neginf=lo), lo, hi).astype(np.float32, copy=False)


def _pos(arr: np.ndarray) -> np.ndarray:
    return np.maximum(arr, 0.0).astype(np.float32, copy=False)


def _neg(arr: np.ndarray) -> np.ndarray:
    return np.maximum(-arr, 0.0).astype(np.float32, copy=False)


def _prox_abs(arr: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.abs(arr))).astype(np.float32, copy=False)


def _recency(arr: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.maximum(arr, 0.0))).astype(np.float32, copy=False)


def _tanh(arr: np.ndarray, scale: float = 1.0) -> np.ndarray:
    return np.tanh(arr / max(float(scale), 1e-6)).astype(np.float32, copy=False)


def _add_feature(
    arrays: list[np.ndarray],
    names: list[str],
    name: str,
    arr: np.ndarray,
    *,
    lo: float = -25.0,
    hi: float = 25.0,
) -> None:
    clean = _clip(np.asarray(arr, dtype=np.float32), lo, hi)
    if clean.ndim != 1:
        raise RuntimeError(f"generated feature {name} is not 1D: {clean.shape}")
    if not np.isfinite(clean).all():
        raise RuntimeError(f"generated feature {name} contains non-finite values")
    if float(np.nanstd(clean)) <= 1e-9:
        return
    arrays.append(clean)
    names.append(f"chart.{name}")


def _build_price_derived_layer(sample_df: pd.DataFrame, source_parquet: Path) -> tuple[np.ndarray, list[str]]:
    """Build past-only price-derived features that are not in the tabular matrix."""
    cols = ["time"]
    try:
        import pyarrow.parquet as pq

        available = set(pq.read_schema(source_parquet).names)
    except Exception:
        available = set(pd.read_parquet(source_parquet, columns=["time"], engine="pyarrow").columns)
    for col in ("mid", "close", "atr", "atr50", "_v1_atr14"):
        if col in available:
            cols.append(col)
    src = pd.read_parquet(source_parquet, columns=list(dict.fromkeys(cols)), engine="pyarrow")
    src["time"] = pd.to_datetime(src["time"], utc=True)
    src = src.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    price_col = "mid" if "mid" in src.columns else "close"
    if price_col not in src.columns:
        return np.empty((len(sample_df), 0), dtype=np.float32), []

    close = pd.to_numeric(src[price_col], errors="coerce").astype(float)
    ema50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
    ema200 = close.ewm(span=200, adjust=False, min_periods=200).mean()
    spread = ema50 - ema200
    denom = close.replace(0.0, np.nan).abs()
    spread_bps = (spread / denom * 1e4).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    price_vs_ema50 = ((close - ema50) / denom * 1e4).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    price_vs_ema200 = ((close - ema200) / denom * 1e4).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ema50_slope = (ema50.diff() / denom * 1e4).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ema200_slope = (ema200.diff() / denom * 1e4).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    spread_delta = spread_bps.diff().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    spread_accel = spread_delta.diff().replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if "atr" in src.columns:
        atr = pd.to_numeric(src["atr"], errors="coerce").astype(float)
    elif "atr50" in src.columns:
        atr = pd.to_numeric(src["atr50"], errors="coerce").astype(float)
    elif "_v1_atr14" in src.columns:
        atr = pd.to_numeric(src["_v1_atr14"], errors="coerce").astype(float)
    else:
        atr = denom * 0.001
    spread_atr = (spread / atr.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    raw = pd.DataFrame(
        {
            "time": src["time"],
            "ema50_200_spread_bps": spread_bps,
            "ema50_200_spread_atr": spread_atr,
            "ema50_200_bull_state": (spread > 0).astype(float),
            "ema50_200_cross_up": ((spread > 0) & (spread.shift(1) <= 0)).astype(float),
            "ema50_200_cross_down": ((spread < 0) & (spread.shift(1) >= 0)).astype(float),
            "price_vs_ema50_bps": price_vs_ema50,
            "price_vs_ema200_bps": price_vs_ema200,
            "ema50_slope_bps": ema50_slope,
            "ema200_slope_bps": ema200_slope,
            "ema50_200_spread_delta": spread_delta,
            "ema50_200_spread_accel": spread_accel,
        }
    ).set_index("time")

    sample_times = pd.to_datetime(sample_df["time"], utc=True)
    aligned = raw.reindex(sample_times).fillna(0.0)
    arrays: list[np.ndarray] = []
    names: list[str] = []
    clip_ranges = {
        "ema50_200_spread_bps": (-250.0, 250.0),
        "price_vs_ema50_bps": (-250.0, 250.0),
        "price_vs_ema200_bps": (-300.0, 300.0),
        "ema50_slope_bps": (-80.0, 80.0),
        "ema200_slope_bps": (-40.0, 40.0),
        "ema50_200_spread_delta": (-80.0, 80.0),
        "ema50_200_spread_accel": (-80.0, 80.0),
    }
    for col in aligned.columns:
        lo, hi = clip_ranges.get(col, (-25.0, 25.0))
        _add_feature(arrays, names, f"m5_{col}", aligned[col].to_numpy(np.float32), lo=lo, hi=hi)
    if not arrays:
        return np.empty((len(sample_df), 0), dtype=np.float32), []
    return np.column_stack(arrays).astype(np.float32, copy=False), names


def _build_candlestick_derived_layer(sample_df: pd.DataFrame, source_parquet: Path) -> tuple[np.ndarray, list[str]]:
    """Build closed-bar candlestick pattern features aligned to sample times."""
    required = ["time", "open", "high", "low", "close"]
    try:
        import pyarrow.parquet as pq

        available = set(pq.read_schema(source_parquet).names)
    except Exception:
        available = set(pd.read_parquet(source_parquet, columns=["time"], engine="pyarrow").columns)
    missing = [name for name in required if name not in available]
    if missing:
        raise RuntimeError(f"CANDLESTICK_SOURCE_FIELDS_MISSING: {missing} parquet={source_parquet}")
    src = pd.read_parquet(source_parquet, columns=required, engine="pyarrow")
    src["time"] = pd.to_datetime(src["time"], utc=True)
    src = src.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    candle_x, candle_names = build_entry_candlestick_pattern_layer(src)
    if not candle_names:
        return np.empty((len(sample_df), 0), dtype=np.float32), []
    candle_df = pd.DataFrame(candle_x, columns=candle_names)
    candle_df["time"] = src["time"].to_numpy()
    candle_df = candle_df.set_index("time")
    sample_times = pd.to_datetime(sample_df["time"], utc=True)
    aligned = candle_df.reindex(sample_times).fillna(0.0)
    return aligned[candle_names].to_numpy(np.float32), candle_names


def _build_chart_layer(x: np.ndarray, feature_names: list[str]) -> tuple[np.ndarray, list[str]]:
    idx = _name_index(feature_names)
    arrays: list[np.ndarray] = []
    names: list[str] = []

    h1_trend = _tanh(_col(x, idx, "ctx_cont._v1h1_ema_diff"))
    h4_trend = _tanh(_col(x, idx, "ctx_cont._v1h4_ema_diff"))
    d1_slope = _tanh(_col(x, idx, "ctx_cont.d1_ema_slope_20_canon_v2"))
    m15_trend = _tanh(_col(x, idx, "ctx_cont.m15_trend_sign_canon_v2"))
    regime_stack = _tanh(_col(x, idx, "ctx_cont.regime_stack_sum_v3"), scale=3.0)
    trend_proxy = _clip(0.35 * h1_trend + 0.30 * h4_trend + 0.20 * d1_slope + 0.10 * m15_trend + 0.05 * regime_stack)
    up = _pos(trend_proxy)
    down = _neg(trend_proxy)
    _add_feature(arrays, names, "trend_proxy_h1h4d1", trend_proxy)
    _add_feature(arrays, names, "trend_up_pressure", up)
    _add_feature(arrays, names, "trend_down_pressure", down)

    near_high = _prox_abs(_col(x, idx, "ctx_cont.dist_last_swing_high_atr"))
    near_low = _prox_abs(_col(x, idx, "ctx_cont.dist_last_swing_low_atr"))
    recent_high = _recency(_col(x, idx, "ctx_cont.bars_since_swing_high"))
    recent_low = _recency(_col(x, idx, "ctx_cont.bars_since_swing_low"))
    high_context = _clip(near_high * (1.0 + recent_high))
    low_context = _clip(near_low * (1.0 + recent_low))
    _add_feature(arrays, names, "near_recent_swing_high", high_context)
    _add_feature(arrays, names, "near_recent_swing_low", low_context)

    bos_up = _clip(_col(x, idx, "snap.smc_bos_up") + 0.5 * _col(x, idx, "ctx_cont.smc_bos_pressure_last12"))
    bos_down = _clip(_col(x, idx, "snap.smc_bos_down") + 0.5 * _col(x, idx, "ctx_cont.smc_bos_pressure_last12"))
    bos_pressure = _clip(_col(x, idx, "ctx_cont.smc_bos_pressure_last48") + _col(x, idx, "ctx_cont.smc_bos_pressure_last12"))
    choch = _clip(
        _col(x, idx, "snap.smc_choch")
        + _col(x, idx, "ctx_cont.smc_choch_recent_tau12")
        + 0.5 * _col(x, idx, "ctx_cont.smc_choch_recent_tau24")
    )
    pullback_h1 = _col(x, idx, "ctx_cont.struct_pullback_depth_h1_v3")
    pullback_h4 = _col(x, idx, "ctx_cont.struct_pullback_depth_h4_v3")
    pullback = _clip(0.6 * pullback_h1 + 0.4 * pullback_h4)
    _add_feature(arrays, names, "bos_pressure_combo", bos_pressure)
    _add_feature(arrays, names, "choch_recent_combo", choch)
    _add_feature(arrays, names, "pullback_depth_h1h4", pullback)

    _add_feature(arrays, names, "hh_breakout_proxy", high_context * up * (1.0 + bos_up))
    _add_feature(arrays, names, "hl_pullback_proxy", low_context * up * (1.0 + pullback))
    _add_feature(arrays, names, "lh_pullback_proxy", high_context * down * (1.0 + pullback))
    _add_feature(arrays, names, "ll_breakdown_proxy", low_context * down * (1.0 + bos_down))
    _add_feature(arrays, names, "bos_x_choch_instability", bos_pressure * choch)
    _add_feature(arrays, names, "bos_x_tf_agreement", bos_pressure * _col(x, idx, "ctx_cont.struct_tf_agree_count_v3"))
    _add_feature(arrays, names, "choch_x_regime_divergence", choch * _col(x, idx, "ctx_cont.regime_divergence_flag_v3"))

    sweep_up = _clip(_col(x, idx, "snap.smc_sweep_up") + _col(x, idx, "ctx_cont.smc_sweep_bull_pressure_last12"))
    sweep_down = _clip(_col(x, idx, "snap.smc_sweep_down") + _col(x, idx, "ctx_cont.smc_sweep_bull_pressure_last48"))
    sweep_recent = _clip(_recency(_col(x, idx, "snap.smc_bars_since_sweep")) + _col(x, idx, "ctx_cont.smc_sweep_recency_tau24"))
    sweep_size = _clip(_col(x, idx, "snap.smc_sweep_size_atr") + _col(x, idx, "ctx_cont.smc_sweep_size_recent_tau12"))
    wick_ratio = _clip(_col(x, idx, "ctx_cont.wick_ratio") + np.abs(_col(x, idx, "snap.wick_asym")))
    support_prox = _col(x, idx, "ctx_cont.sr_support_proximity_exp")
    resistance_prox = _col(x, idx, "ctx_cont.sr_resistance_proximity_exp")
    _add_feature(arrays, names, "sweep_recent_combo", sweep_recent)
    _add_feature(arrays, names, "sweep_size_combo", sweep_size)
    _add_feature(arrays, names, "false_breakout_high_reject", sweep_up * sweep_recent * wick_ratio * resistance_prox * (1.0 + down))
    _add_feature(arrays, names, "false_breakout_low_reject", sweep_down * sweep_recent * wick_ratio * support_prox * (1.0 + up))
    _add_feature(arrays, names, "sweep_high_into_resistance", sweep_up * resistance_prox * high_context)
    _add_feature(arrays, names, "sweep_low_into_support", sweep_down * support_prox * low_context)
    _add_feature(arrays, names, "sweep_size_x_wick", sweep_size * wick_ratio)
    _add_feature(arrays, names, "sweep_x_choch", sweep_recent * choch)

    h1_compression = _clip(_col(x, idx, "ctx_cont.H1_range_compression_ratio"))
    m15_compression = _clip(_col(x, idx, "ctx_cont.M15_range_compression_ratio"))
    squeeze = _clip(_col(x, idx, "snap._v1_bb_squeeze_20_2"))
    atr_z = _clip(_col(x, idx, "snap.atr_z"))
    rvol = _clip(_col(x, idx, "snap.rvol_20"))
    vol_ratio = _clip(_col(x, idx, "snap.vol_ratio_5_20"))
    d1_range_z = _clip(_col(x, idx, "ctx_cont.d1_range_z_20_canon_v2"))
    compression = _clip(0.45 * h1_compression + 0.35 * m15_compression + 0.20 * squeeze)
    expansion = _clip(compression * (0.45 * atr_z + 0.35 * rvol + 0.20 * vol_ratio))
    _add_feature(arrays, names, "compression_h1_m15_bb", compression)
    _add_feature(arrays, names, "compression_to_expansion_proxy", expansion)
    _add_feature(arrays, names, "compression_x_bos", compression * bos_pressure)
    _add_feature(arrays, names, "compression_x_choch", compression * choch)
    _add_feature(arrays, names, "d1_range_x_expansion", d1_range_z * expansion)

    retracement = _clip(_col(x, idx, "ctx_cont.retracement_from_last_impulse"))
    trend_age_h1 = _clip(_col(x, idx, "ctx_cont.h1_trend_age_bars_norm_v2"))
    trend_age_h4 = _clip(_col(x, idx, "ctx_cont.h4_trend_age_bars_norm_v2"))
    mature_d1 = _col(x, idx, "ctx_cont.d1_trend_age_mature_flag_v3")
    _add_feature(arrays, names, "impulse_pullback_up", retracement * up * (1.0 + trend_age_h1))
    _add_feature(arrays, names, "impulse_pullback_down", retracement * down * (1.0 + trend_age_h1))
    _add_feature(arrays, names, "mature_trend_pullback_risk", pullback * (trend_age_h1 + trend_age_h4 + mature_d1))
    _add_feature(arrays, names, "trend_age_x_choch", (trend_age_h1 + trend_age_h4) * choch)

    nearest_pivot = _prox_abs(_col(x, idx, "ctx_cont.sr_nearest_pivot_abs_atr"))
    dists = [
        _prox_abs(_col(x, idx, name))
        for name in (
            "ctx_cont.dist_to_R1_atr",
            "ctx_cont.dist_to_R2_atr",
            "ctx_cont.dist_to_S1_atr",
            "ctx_cont.dist_to_S2_atr",
            "ctx_cont.dist_to_h1_hi_atr",
            "ctx_cont.dist_to_h1_lo_atr",
            "ctx_cont.dist_to_h4_hi_atr",
            "ctx_cont.dist_to_h4_lo_atr",
            "ctx_cont.dist_to_d1_hi_atr",
            "ctx_cont.dist_to_d1_lo_atr",
        )
    ]
    stacked_prox = np.vstack(dists)
    level_prox_max = stacked_prox.max(axis=0).astype(np.float32, copy=False)
    level_prox_mean = stacked_prox.mean(axis=0).astype(np.float32, copy=False)
    _add_feature(arrays, names, "pivot_nearest_proximity", nearest_pivot)
    _add_feature(arrays, names, "major_level_proximity_max", level_prox_max)
    _add_feature(arrays, names, "major_level_proximity_mean", level_prox_mean)
    _add_feature(arrays, names, "wick_x_major_level", wick_ratio * level_prox_max)
    _add_feature(arrays, names, "pullback_x_support_resistance", pullback * level_prox_max)
    _add_feature(arrays, names, "premium_discount_x_level", _col(x, idx, "ctx_cont.sr_support_minus_resistance_prox") * level_prox_max)

    d1_loc = _col(x, idx, "ctx_cont.d1_close_pct_in_20day_range_canon_v2")
    d1_boundary = _prox_abs(_col(x, idx, "ctx_cont.d1_dist_to_boundary_v3"))
    _add_feature(arrays, names, "d1_upper_range_pressure", d1_loc * up * high_context)
    _add_feature(arrays, names, "d1_lower_range_pressure", (1.0 - d1_loc) * down * low_context)
    _add_feature(arrays, names, "d1_boundary_x_sweep", d1_boundary * sweep_recent)
    _add_feature(arrays, names, "d1_boundary_x_wick", d1_boundary * wick_ratio)

    session_cols = [
        "ctx_cont.is_ASIA",
        "ctx_cont.is_asia_eu_overlap",
        "ctx_cont.is_eu_us_overlap",
        "ctx_cont.is_eu_only",
        "ctx_cont.is_us_only",
    ]
    session_signals = [
        ("trend_proxy", trend_proxy),
        ("bos", bos_pressure),
        ("choch", choch),
        ("sweep_recent", sweep_recent),
        ("compression", compression),
        ("expansion", expansion),
        ("wick_level", wick_ratio * level_prox_max),
        ("pullback", pullback),
        ("d1_loc", d1_loc),
    ]
    for s_name in session_cols:
        if not _has_name(idx, s_name):
            continue
        s = _col(x, idx, s_name)
        short = s_name.replace("ctx_cont.", "")
        for sig_name, sig in session_signals:
            _add_feature(arrays, names, f"{short}_x_{sig_name}", s * sig)

    vol_signals = [
        ("d1_atr_pct", _col(x, idx, "ctx_cont.D1_atr_percentile_252")),
        ("h1_vol_pct", _col(x, idx, "ctx_cont.vol_pct_h1_1yr")),
        ("m5_vol_pct", _col(x, idx, "ctx_cont.vol_pct_m5_1yr")),
        ("atr_ratio_h1_d1", _col(x, idx, "ctx_cont.atr_ratio_h1_d1")),
        ("atr_ratio_m15_d1", _col(x, idx, "ctx_cont.atr_ratio_m15_d1")),
    ]
    struct_signals = [
        ("hh", high_context * up),
        ("hl", low_context * up),
        ("lh", high_context * down),
        ("ll", low_context * down),
        ("sweep", sweep_recent),
        ("choch", choch),
        ("bos", bos_pressure),
        ("wick_level", wick_ratio * level_prox_max),
    ]
    for vol_name, vol in vol_signals:
        for sig_name, sig in struct_signals:
            _add_feature(arrays, names, f"{sig_name}_x_{vol_name}", sig * vol)

    foundation_x, foundation_names = build_entry_foundation_structure_layer(x, feature_names)
    if foundation_x.shape[1]:
        arrays.extend([foundation_x[:, i] for i in range(foundation_x.shape[1])])
        names.extend(foundation_names)
    geometry_x, geometry_names = build_entry_chart_geometry_layer(x, feature_names)
    if geometry_x.shape[1]:
        arrays.extend([geometry_x[:, i] for i in range(geometry_x.shape[1])])
        names.extend(geometry_names)

    if not arrays:
        return np.empty((x.shape[0], 0), dtype=np.float32), []
    out = np.column_stack(arrays).astype(np.float32, copy=False)
    return out, names


def _lag1(arr: np.ndarray) -> np.ndarray:
    out = np.empty_like(arr, dtype=np.float32)
    out[0] = 0.0
    out[1:] = arr[:-1]
    return out


def _cross_up(arr: np.ndarray) -> np.ndarray:
    prev = _lag1(arr)
    return ((arr > 0.0) & (prev <= 0.0)).astype(np.float32)


def _cross_down(arr: np.ndarray) -> np.ndarray:
    prev = _lag1(arr)
    return ((arr < 0.0) & (prev >= 0.0)).astype(np.float32)


def _delta(arr: np.ndarray) -> np.ndarray:
    return _clip(arr - _lag1(arr))


def _build_deep_interaction_layer(
    x: np.ndarray,
    feature_names: list[str],
    df: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    idx = _name_index(feature_names)
    arrays: list[np.ndarray] = []
    names: list[str] = []

    def c(name: str, default: float = 0.0) -> np.ndarray:
        return _col(x, idx, name, default=default)

    ema_fast = _tanh(c("snap._v1_ema_diff"))
    ema_h1 = _tanh(c("ctx_cont._v1h1_ema_diff"))
    ema_h4 = _tanh(c("ctx_cont._v1h4_ema_diff"))
    pos_ema200 = _tanh(c("snap.pos_vs_ema200"))
    ema20_slope = _tanh(c("snap.ema20_slope"))
    d1_slope = _tanh(c("ctx_cont.d1_ema_slope_20_canon_v2"))
    h1_slope = _tanh(c("ctx_cont._v1h1_slope5"))
    h4_slope = _tanh(c("ctx_cont._v1h4_slope5"))
    trend_stack = _clip(0.20 * ema_fast + 0.20 * ema_h1 + 0.20 * ema_h4 + 0.15 * pos_ema200 + 0.10 * ema20_slope + 0.10 * d1_slope + 0.05 * h1_slope)
    trend_delta = _delta(trend_stack)
    price_ema_available = _has_name(idx, "chart.m5_ema50_200_spread_bps")
    ema50_200_spread = _tanh(c("chart.m5_ema50_200_spread_bps"), scale=50.0)
    ema50_200_atr = _tanh(c("chart.m5_ema50_200_spread_atr"), scale=2.0)
    ema50_200_bull = c("chart.m5_ema50_200_bull_state")
    ema50_200_bear = (1.0 - ema50_200_bull) if price_ema_available else np.zeros_like(ema50_200_bull)
    ema50_200_cross = _clip(c("chart.m5_ema50_200_cross_up") - c("chart.m5_ema50_200_cross_down"))
    price_vs_ema200 = _tanh(c("chart.m5_price_vs_ema200_bps"), scale=80.0)
    ema50_slope = _tanh(c("chart.m5_ema50_slope_bps"), scale=12.0)
    ema200_slope = _tanh(c("chart.m5_ema200_slope_bps"), scale=6.0)
    _add_feature(arrays, names, "ema_stack_alignment", trend_stack)
    _add_feature(arrays, names, "ema_stack_delta", trend_delta)
    _add_feature(arrays, names, "ema_stack_acceleration", _delta(trend_delta))
    _add_feature(arrays, names, "ema_stack_cross_up", _cross_up(trend_stack))
    _add_feature(arrays, names, "ema_stack_cross_down", _cross_down(trend_stack))
    _add_feature(arrays, names, "true_ema50_200_alignment", ema50_200_spread)
    _add_feature(arrays, names, "true_ema50_200_atr_alignment", ema50_200_atr)
    _add_feature(arrays, names, "true_ema50_200_cross_pressure", ema50_200_cross)
    _add_feature(arrays, names, "true_ema50_slope_pressure", ema50_slope)
    _add_feature(arrays, names, "true_ema200_slope_pressure", ema200_slope)
    _add_feature(arrays, names, "price_vs_true_ema200_pressure", price_vs_ema200)

    for raw_name, short in [
        ("snap._v1_ema_diff", "m5_ema_fast_slow"),
        ("snap.pos_vs_ema200", "m5_pos_ema200"),
        ("ctx_cont._v1h1_ema_diff", "h1_ema_fast_slow"),
        ("ctx_cont._v1h4_ema_diff", "h4_ema_fast_slow"),
        ("ctx_cont.d1_ema_slope_20_canon_v2", "d1_ema_slope"),
    ]:
        sig = _tanh(c(raw_name))
        _add_feature(arrays, names, f"{short}_cross_up", _cross_up(sig))
        _add_feature(arrays, names, f"{short}_cross_down", _cross_down(sig))
        _add_feature(arrays, names, f"{short}_delta", _delta(sig))

    compression = _clip(0.45 * c("chart.compression_h1_m15_bb") + 0.30 * c("ctx_cont.H1_range_compression_ratio") + 0.25 * c("ctx_cont.M15_range_compression_ratio"))
    expansion = _clip(c("chart.compression_to_expansion_proxy") + c("snap.rvol_20") + c("snap.vol_ratio_5_20") + c("snap.atr_z"))
    expansion_delta = _delta(expansion)
    _add_feature(arrays, names, "expansion_delta", expansion_delta)
    _add_feature(arrays, names, "compression_release", compression * _pos(expansion_delta))
    _add_feature(arrays, names, "compression_release_downtrend", compression * _pos(expansion_delta) * _neg(trend_stack))
    _add_feature(arrays, names, "compression_release_uptrend", compression * _pos(expansion_delta) * _pos(trend_stack))

    regime_stack = _tanh(c("ctx_cont.regime_stack_sum_v3"), scale=3.0)
    regime_agree = _clip(c("ctx_cont.regime_tf_agreement_v3"))
    regime_div = _clip(c("ctx_cont.regime_divergence_flag_v3"))
    d1_changed = _clip(c("ctx_cont.d1_regime_changed_flag_v3"))
    bars_since_d1_change = _recency(c("ctx_cont.bars_since_d1_regime_change_v3"))
    _add_feature(arrays, names, "regime_stack_delta", _delta(regime_stack))
    _add_feature(arrays, names, "fresh_d1_regime_change_pressure", d1_changed + bars_since_d1_change)
    _add_feature(arrays, names, "regime_divergence_x_trend_delta", regime_div * np.abs(trend_delta))
    _add_feature(arrays, names, "regime_agreement_x_trend_stack", regime_agree * trend_stack)
    _add_feature(arrays, names, "regime_agreement_x_true_ema50_200", regime_agree * ema50_200_spread)
    _add_feature(arrays, names, "regime_divergence_x_true_ema_cross", regime_div * np.abs(ema50_200_cross))
    _add_feature(arrays, names, "fresh_regime_x_true_ema_cross", (d1_changed + bars_since_d1_change) * np.abs(ema50_200_cross))

    sweep = _clip(c("chart.sweep_recent_combo") + c("snap.smc_sweep_up") + c("snap.smc_sweep_down"))
    sweep_size = _clip(c("chart.sweep_size_combo") + c("snap.smc_sweep_size_atr"))
    choch = _clip(c("chart.choch_recent_combo") + c("snap.smc_choch"))
    bos = _clip(c("chart.bos_pressure_combo") + c("ctx_cont.smc_bos_pressure_last48"))
    wick_level = _clip(c("chart.wick_x_major_level") + c("ctx_cont.wick_ratio"))
    pullback = _clip(c("chart.pullback_depth_h1h4") + c("ctx_cont.struct_pullback_depth_h1_v3") + c("ctx_cont.struct_pullback_depth_h4_v3"))
    hh = _clip(c("chart.hh_breakout_proxy"))
    hl = _clip(c("chart.hl_pullback_proxy"))
    lh = _clip(c("chart.lh_pullback_proxy"))
    ll = _clip(c("chart.ll_breakdown_proxy"))
    d1_loc = _clip(c("ctx_cont.d1_close_pct_in_20day_range_canon_v2"), 0.0, 1.0)
    level_prox = _clip(c("chart.major_level_proximity_max") + c("ctx_cont.sr_support_proximity_exp") + c("ctx_cont.sr_resistance_proximity_exp"))
    spread = _clip(c("ctx_cont.spread_bps"))
    session_open = _clip(c("ctx_cont.minutes_since_session_open") / 240.0)
    session_boundary = _recency(c("ctx_cont.minutes_to_next_session_boundary"))
    h1_vol = _clip(c("ctx_cont.vol_pct_h1_1yr"), 0.0, 1.0)
    m5_vol = _clip(c("ctx_cont.vol_pct_m5_1yr"), 0.0, 1.0)
    atr_pct = _clip(c("ctx_cont.D1_atr_percentile_252"), 0.0, 1.0)
    vol_stack = _clip(0.35 * h1_vol + 0.25 * m5_vol + 0.20 * atr_pct + 0.10 * c("ctx_cont.atr_ratio_h1_d1") + 0.10 * c("ctx_cont.atr_ratio_m15_d1"))
    _add_feature(arrays, names, "vol_stack", vol_stack)
    _add_feature(arrays, names, "vol_stack_delta", _delta(vol_stack))

    tail_pressure = _clip(
        0.22 * sweep_size
        + 0.18 * wick_level
        + 0.16 * choch
        + 0.14 * regime_div
        + 0.12 * np.abs(trend_delta)
        + 0.10 * vol_stack
        + 0.08 * spread
    )
    _add_feature(arrays, names, "entry_tail_pressure_combo", tail_pressure)
    _add_feature(arrays, names, "tail_pressure_x_session_boundary", tail_pressure * session_boundary)
    _add_feature(arrays, names, "tail_pressure_x_regime_fresh", tail_pressure * (d1_changed + bars_since_d1_change))
    _add_feature(arrays, names, "tail_pressure_x_compression_release", tail_pressure * compression * _pos(expansion_delta))

    session_cols = [
        ("ctx_cont.is_ASIA", "asia"),
        ("ctx_cont.is_asia_eu_overlap", "asia_eu"),
        ("ctx_cont.is_eu_us_overlap", "eu_us"),
        ("ctx_cont.is_eu_only", "eu"),
        ("ctx_cont.is_us_only", "us"),
    ]
    struct_signals = [
        ("sweep", sweep),
        ("sweep_size", sweep_size),
        ("choch", choch),
        ("bos", bos),
        ("wick_level", wick_level),
        ("pullback", pullback),
        ("hh", hh),
        ("hl", hl),
        ("lh", lh),
        ("ll", ll),
        ("trend_delta", trend_delta),
        ("tail_pressure", tail_pressure),
    ]
    context_signals = [
        ("vol_stack", vol_stack),
        ("regime_div", regime_div),
        ("regime_agree", regime_agree),
        ("level_prox", level_prox),
        ("d1_upper", d1_loc),
        ("d1_lower", 1.0 - d1_loc),
        ("ema50_200", ema50_200_spread),
        ("ema50_200_atr", ema50_200_atr),
        ("ema50_200_cross", ema50_200_cross),
        ("price_vs_ema200", price_vs_ema200),
        ("session_boundary", session_boundary),
        ("session_age", session_open),
    ]

    for s_name, s_short in session_cols:
        s = c(s_name)
        for sig_name, sig in struct_signals:
            _add_feature(arrays, names, f"{s_short}_x_{sig_name}", s * sig)
        for ctx_name, ctx_sig in context_signals:
            _add_feature(arrays, names, f"{s_short}_x_{ctx_name}", s * ctx_sig)

    for sig_name, sig in struct_signals:
        for ctx_name, ctx_sig in context_signals:
            _add_feature(arrays, names, f"{sig_name}_x_{ctx_name}", sig * ctx_sig)

    _add_feature(arrays, names, "long_breakout_tail_risk", hh * sweep * wick_level * vol_stack)
    _add_feature(arrays, names, "short_breakdown_tail_risk", ll * sweep * wick_level * vol_stack)
    _add_feature(arrays, names, "late_trend_choch_tail_risk", choch * (c("ctx_cont.h1_trend_age_bars_norm_v2") + c("ctx_cont.h4_trend_age_bars_norm_v2")) * vol_stack)
    _add_feature(arrays, names, "range_extreme_reversal_risk", (d1_loc * hh + (1.0 - d1_loc) * ll) * wick_level * level_prox)
    _add_feature(arrays, names, "pullback_quality_trend_agree", pullback * regime_agree * np.abs(trend_stack))
    _add_feature(arrays, names, "pullback_bad_regime_divergence", pullback * regime_div * vol_stack)
    _add_feature(arrays, names, "true_ema_bull_pullback_quality", ema50_200_bull * pullback * regime_agree)
    _add_feature(arrays, names, "true_ema_bear_short_reversal_risk", ema50_200_bear * hh * wick_level * level_prox)
    _add_feature(arrays, names, "true_ema_cross_liquidity_sweep_risk", np.abs(ema50_200_cross) * sweep * wick_level * level_prox)

    if not arrays:
        return np.empty((x.shape[0], 0), dtype=np.float32), []
    out = np.column_stack(arrays).astype(np.float32, copy=False)
    return out, names


def _is_chart_name(name: str) -> bool:
    low = name.lower()
    return any(part in low for part in CHART_NAME_PARTS)


def _is_vol_or_session_name(name: str) -> bool:
    low = name.lower()
    return any(part in low for part in VOL_NAME_PARTS) or any(part in low for part in SESSION_NAME_PARTS)


def _feature_sets(
    base_x: np.ndarray,
    base_names: list[str],
    categorical_idx: list[int],
    chart_x: np.ndarray,
    chart_names: list[str],
    deep_x: np.ndarray,
    deep_names: list[str],
    requested: list[str],
) -> tuple[np.ndarray, list[str], list[FeatureSet]]:
    pieces = [base_x]
    if chart_x.shape[1]:
        pieces.append(chart_x)
    if deep_x.shape[1]:
        pieces.append(deep_x)
    all_x = np.concatenate(pieces, axis=1).astype(np.float32, copy=False) if len(pieces) > 1 else base_x
    all_names = list(base_names) + list(chart_names) + list(deep_names)
    base_n = len(base_names)
    chart_n = len(chart_names)
    deep_start = base_n + chart_n
    chart_original = np.asarray([i for i, n in enumerate(base_names) if _is_chart_name(n)], dtype=np.int64)
    vol_session = np.asarray([i for i, n in enumerate(base_names) if _is_vol_or_session_name(n)], dtype=np.int64)
    generated = np.arange(base_n, len(all_names), dtype=np.int64)
    chart_generated = np.arange(base_n, deep_start, dtype=np.int64)
    deep_generated = np.arange(deep_start, len(all_names), dtype=np.int64)
    base_all = np.arange(base_n, dtype=np.int64)
    no_chart = np.asarray([i for i, n in enumerate(base_names) if not _is_chart_name(n)], dtype=np.int64)
    chart_layer_only = np.unique(np.concatenate([chart_original, vol_session, generated])).astype(np.int64)
    chart_only_no_deep = np.unique(np.concatenate([chart_original, vol_session, chart_generated])).astype(np.int64)
    chart_deep_only = np.unique(np.concatenate([chart_original, vol_session, chart_generated, deep_generated])).astype(np.int64)

    raw_sets: dict[str, np.ndarray] = {
        "base": base_all,
        "base_plus_chart": np.arange(deep_start, dtype=np.int64),
        "base_plus_chart_deep": np.arange(len(all_names), dtype=np.int64),
        "chart_layer_only": chart_only_no_deep,
        "chart_deep_only": chart_deep_only,
        "deep_only": deep_generated,
        "base_without_chart": no_chart,
    }
    out: list[FeatureSet] = []
    for name in requested:
        if name not in raw_sets:
            raise SystemExit(f"unknown feature set {name!r}; valid={sorted(raw_sets)}")
        indices = raw_sets[name]
        names = [all_names[int(i)] for i in indices]
        cat_positions = [pos for pos, original_i in enumerate(indices) if int(original_i) in set(categorical_idx)]
        out.append(FeatureSet(name=name, indices=indices, names=names, categorical_idx=cat_positions))
    return all_x, all_names, out


def _predict_proba_aligned(model: Any, x: np.ndarray) -> np.ndarray:
    probs = np.asarray(model.predict_proba(x), dtype=np.float64)
    if probs.ndim != 2:
        raise RuntimeError(f"unexpected probability shape: {probs.shape}")
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "steps"):
        classes = getattr(model.steps[-1][1], "classes_", None)
    if classes is not None:
        aligned = np.zeros((probs.shape[0], 3), dtype=np.float64)
        for src_i, cls in enumerate(classes):
            cls_i = int(cls)
            if 0 <= cls_i < 3:
                aligned[:, cls_i] = probs[:, src_i]
        probs = aligned
    if probs.shape[1] != 3:
        raise RuntimeError(f"unexpected probability shape after class alignment: {probs.shape}")
    row_sum = probs.sum(axis=1, keepdims=True)
    return np.divide(probs, np.maximum(row_sum, 1e-12))


def _fit_model(
    *,
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    categorical_idx: list[int],
    args: argparse.Namespace,
) -> Any:
    if model_name == "lightgbm":
        return _train_lightgbm(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            categorical_idx=categorical_idx,
            seed=int(args.seed),
            n_estimators=int(args.lgbm_n_estimators),
            learning_rate=float(args.lgbm_learning_rate),
            num_leaves=int(args.lgbm_num_leaves),
            max_depth=int(args.lgbm_max_depth),
            min_child_samples=int(args.lgbm_min_child_samples),
            n_jobs=int(args.n_jobs),
            early_stopping_rounds=int(args.lgbm_early_stopping_rounds),
        )
    if model_name == "extratrees":
        model = ExtraTreesClassifier(
            n_estimators=int(args.extra_trees_n_estimators),
            max_depth=None if int(args.extra_trees_max_depth) <= 0 else int(args.extra_trees_max_depth),
            min_samples_leaf=int(args.extra_trees_min_samples_leaf),
            max_features=str(args.extra_trees_max_features),
            class_weight="balanced_subsample",
            random_state=int(args.seed),
            n_jobs=int(args.n_jobs),
        )
        model.fit(x_train, y_train)
        return model
    if model_name == "histgb":
        model = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=float(args.histgb_learning_rate),
            max_iter=int(args.histgb_max_iter),
            max_leaf_nodes=int(args.histgb_max_leaf_nodes),
            min_samples_leaf=int(args.histgb_min_samples_leaf),
            l2_regularization=float(args.histgb_l2_regularization),
            early_stopping=True,
            validation_fraction=None,
            random_state=int(args.seed),
        )
        weights = compute_sample_weight(class_weight="balanced", y=y_train)
        model.fit(x_train, y_train, sample_weight=weights)
        return model
    if model_name == "logreg":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=float(args.logreg_c),
                class_weight="balanced",
                max_iter=int(args.logreg_max_iter),
                solver="lbfgs",
                n_jobs=int(args.n_jobs),
            ),
        )
        model.fit(x_train, y_train)
        return model
    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(f"xgboost requested but unavailable: {exc}") from exc
        model = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            n_estimators=int(args.xgb_n_estimators),
            learning_rate=float(args.xgb_learning_rate),
            max_depth=int(args.xgb_max_depth),
            subsample=float(args.xgb_subsample),
            colsample_bytree=float(args.xgb_colsample_bytree),
            reg_lambda=float(args.xgb_reg_lambda),
            tree_method="hist",
            random_state=int(args.seed),
            n_jobs=int(args.n_jobs),
        )
        weights = compute_sample_weight(class_weight="balanced", y=y_train)
        model.fit(x_train, y_train, sample_weight=weights, eval_set=[(x_val, y_val)], verbose=False)
        return model
    raise SystemExit(f"unknown model {model_name!r}")


def _feature_importance_rows(model: Any, feature_names: list[str], meta: dict[str, Any]) -> list[dict[str, Any]]:
    values = None
    if hasattr(model, "feature_importances_"):
        values = getattr(model, "feature_importances_")
    elif hasattr(model, "steps"):
        last = model.steps[-1][1]
        if hasattr(last, "coef_"):
            values = np.mean(np.abs(np.asarray(last.coef_, dtype=np.float64)), axis=0)
    if values is None:
        return []
    arr = np.asarray(values, dtype=np.float64).ravel()
    if len(arr) != len(feature_names):
        return []
    return [
        {**meta, "feature": str(name), "importance": float(value)}
        for name, value in zip(feature_names, arr)
    ]


def _parse_exit_specs(raw: str, default_same_bar_policy: str) -> list[ExitSpec]:
    specs: list[ExitSpec] = []
    for part in _parse_csv(raw):
        if part == "horizon":
            specs.append(
                ExitSpec(
                    name="horizon",
                    exit_mode="horizon",
                    take_profit_bps=0.0,
                    stop_loss_bps=0.0,
                    same_bar_policy=default_same_bar_policy,
                )
            )
            continue
        pieces = part.split(":")
        if len(pieces) not in (3, 4) or pieces[0] != "stop_tp":
            raise SystemExit(f"exit spec must be horizon or stop_tp:TP:SL[:same_bar], got {part!r}")
        same_bar = pieces[3] if len(pieces) == 4 else default_same_bar_policy
        if same_bar not in {"stop_first", "target_first"}:
            raise SystemExit(f"invalid same bar policy in {part!r}")
        tp = float(pieces[1])
        sl = float(pieces[2])
        specs.append(
            ExitSpec(
                name=f"stoptp{tp:g}_{sl:g}_{same_bar}",
                exit_mode="stop_tp",
                take_profit_bps=tp,
                stop_loss_bps=sl,
                same_bar_policy=same_bar,
            )
        )
    if not specs:
        raise SystemExit("no exit specs parsed")
    return specs


def _parse_entry_veto_rule_sets(raw: str) -> list[EntryVetoSet]:
    text = str(raw or "").strip()
    if not text or text.lower() in {"none", "no_veto", "off"}:
        return [EntryVetoSet(name="none", rules=tuple())]
    out: list[EntryVetoSet] = []
    seen: set[str] = set()
    for raw_set in [p.strip() for p in text.split(";") if p.strip()]:
        if raw_set.lower() in {"none", "no_veto", "off"}:
            name = "none"
            rules_text = ""
        elif "=" in raw_set:
            name, rules_text = raw_set.split("=", 1)
            name = _model_slug([name.strip()])
        else:
            rules_text = raw_set
            name = "veto_" + hashlib.sha256(rules_text.encode("utf-8")).hexdigest()[:8]
        if not name:
            raise SystemExit(f"invalid empty veto set name in {raw_set!r}")
        if name in seen:
            raise SystemExit(f"duplicate veto set name {name!r}")
        seen.add(name)
        if name == "none":
            out.append(EntryVetoSet(name="none", rules=tuple()))
            continue
        rules: list[EntryVetoRule] = []
        for raw_rule in [p.strip() for p in rules_text.split("+") if p.strip()]:
            conditions: list[EntryVetoCondition] = []
            for raw_condition in [p.strip() for p in raw_rule.split("&") if p.strip()]:
                pieces = raw_condition.split(":")
                if len(pieces) not in (3, 4, 5):
                    raise SystemExit(
                        "entry veto rules must be feature:low|high:quantile, optionally "
                        "scoped as feature:side:q:LONG|SHORT[:SESSION] and joined as "
                        "feature:side:q&feature:side:q; "
                        f"got {raw_condition!r} in set {name!r}"
                    )
                feature, skip_side, quantile_raw = pieces[:3]
                skip_side = skip_side.strip().lower()
                if skip_side not in {"low", "high"}:
                    raise SystemExit(f"entry veto skip_side must be low/high, got {skip_side!r}")
                quantile = float(quantile_raw)
                if not 0.0 < quantile < 1.0:
                    raise SystemExit(f"entry veto quantile must be inside (0,1), got {quantile!r}")
                feature = feature.strip()
                if not feature:
                    raise SystemExit(f"entry veto feature is empty in rule {raw_condition!r}")
                decision_side = None
                trade_session = None
                if len(pieces) >= 4:
                    side_text = pieces[3].strip().upper()
                    if side_text not in {"", "*", "ANY", "LONG", "SHORT"}:
                        raise SystemExit(
                            f"entry veto decision side must be LONG/SHORT/ANY, got {pieces[3]!r}"
                        )
                    decision_side = None if side_text in {"", "*", "ANY"} else side_text
                if len(pieces) >= 5:
                    session_text = pieces[4].strip().upper()
                    if session_text not in {"", "*", "ANY", "ASIA", "EU", "US", "OVERLAP"}:
                        raise SystemExit(
                            "entry veto trade session must be ASIA/EU/US/OVERLAP/ANY, "
                            f"got {pieces[4]!r}"
                        )
                    trade_session = None if session_text in {"", "*", "ANY"} else session_text
                conditions.append(
                    EntryVetoCondition(
                        feature=feature,
                        skip_side=skip_side,
                        quantile=quantile,
                        decision_side=decision_side,
                        trade_session=trade_session,
                    )
                )
            if not conditions:
                raise SystemExit(f"empty entry veto rule {raw_rule!r} in set {name!r}")
            rules.append(EntryVetoRule(conditions=tuple(conditions)))
        if not rules:
            raise SystemExit(f"entry veto set {name!r} has no rules")
        out.append(EntryVetoSet(name=name, rules=tuple(rules)))
    if not out:
        return [EntryVetoSet(name="none", rules=tuple())]
    return out


def _entry_veto_set_to_dict(veto_set: EntryVetoSet) -> dict[str, Any]:
    def condition_to_dict(condition: EntryVetoCondition) -> dict[str, Any]:
        return {
            "feature": condition.feature,
            "skip_side": condition.skip_side,
            "quantile": float(condition.quantile),
            "decision_side": condition.decision_side,
            "trade_session": condition.trade_session,
        }

    def rule_to_dict(rule: EntryVetoRule) -> dict[str, Any]:
        out = {"conditions": [condition_to_dict(condition) for condition in rule.conditions]}
        if len(rule.conditions) == 1:
            out.update(condition_to_dict(rule.conditions[0]))
        return out

    return {
        "name": veto_set.name,
        "rules": [rule_to_dict(rule) for rule in veto_set.rules],
    }


def _model_signature(model_name: str, args: argparse.Namespace) -> dict[str, Any]:
    include_price_ema = bool(getattr(args, "include_price_ema_features", False))
    common = {
        "feature_layer_version": (
            f"{FEATURE_LAYER_VERSION}+{PRICE_EMA_FEATURE_LAYER_VERSION}"
            if include_price_ema
            else FEATURE_LAYER_VERSION
        ),
        "include_price_ema_features": include_price_ema,
        "model": model_name,
        "seed": int(args.seed),
        "max_train_rows": int(args.max_train_rows),
        "data_splits": _parse_csv(args.data_splits),
        "folds": str(args.folds),
        "val_tail_days": int(args.val_tail_days),
        "min_val_rows": int(args.min_val_rows),
        "min_train_rows": int(args.min_train_rows),
    }
    if model_name == "lightgbm":
        common["params"] = {
            "n_estimators": int(args.lgbm_n_estimators),
            "learning_rate": float(args.lgbm_learning_rate),
            "num_leaves": int(args.lgbm_num_leaves),
            "max_depth": int(args.lgbm_max_depth),
            "min_child_samples": int(args.lgbm_min_child_samples),
            "early_stopping_rounds": int(args.lgbm_early_stopping_rounds),
        }
    elif model_name == "extratrees":
        common["params"] = {
            "n_estimators": int(args.extra_trees_n_estimators),
            "max_depth": int(args.extra_trees_max_depth),
            "min_samples_leaf": int(args.extra_trees_min_samples_leaf),
            "max_features": str(args.extra_trees_max_features),
        }
    elif model_name == "histgb":
        common["params"] = {
            "max_iter": int(args.histgb_max_iter),
            "learning_rate": float(args.histgb_learning_rate),
            "max_leaf_nodes": int(args.histgb_max_leaf_nodes),
            "min_samples_leaf": int(args.histgb_min_samples_leaf),
            "l2_regularization": float(args.histgb_l2_regularization),
        }
    elif model_name == "logreg":
        common["params"] = {
            "c": float(args.logreg_c),
            "max_iter": int(args.logreg_max_iter),
            "solver": "lbfgs",
            "class_weight": "balanced",
        }
    elif model_name == "xgboost":
        common["params"] = {
            "n_estimators": int(args.xgb_n_estimators),
            "learning_rate": float(args.xgb_learning_rate),
            "max_depth": int(args.xgb_max_depth),
            "subsample": float(args.xgb_subsample),
            "colsample_bytree": float(args.xgb_colsample_bytree),
            "reg_lambda": float(args.xgb_reg_lambda),
            "tree_method": "hist",
        }
    else:
        common["params"] = {}
    return common


def _veto_feature_layer_signature(args: argparse.Namespace) -> dict[str, Any]:
    include_model_price_ema = bool(getattr(args, "include_price_ema_features", False))
    veto_only_price_ema = bool(getattr(args, "veto_only_price_ema_features", False))
    versions = [FEATURE_LAYER_VERSION]
    if include_model_price_ema:
        versions.append(PRICE_EMA_FEATURE_LAYER_VERSION)
    if veto_only_price_ema:
        versions.append(VETO_ONLY_PRICE_EMA_FEATURE_LAYER_VERSION)
        versions.append(PRICE_EMA_FEATURE_LAYER_VERSION)
    return {
        "feature_layer_version": "+".join(versions),
        "include_model_price_ema_features": include_model_price_ema,
        "veto_only_price_ema_features": veto_only_price_ema,
    }


def _entry_veto_masks(
    *,
    veto_set: EntryVetoSet,
    feature_index: dict[str, int],
    val_x: np.ndarray,
    eval_x: np.ndarray,
    val_decision_side: np.ndarray,
    eval_decision_side: np.ndarray,
    val_sessions: np.ndarray,
    eval_sessions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    val_keep = np.ones(val_x.shape[0], dtype=bool)
    eval_keep = np.ones(eval_x.shape[0], dtype=bool)
    details: list[dict[str, Any]] = []
    if not veto_set.rules:
        return val_keep, eval_keep, details

    for rule_i, rule in enumerate(veto_set.rules):
        rule_val_skip = np.ones(val_x.shape[0], dtype=bool)
        rule_eval_skip = np.ones(eval_x.shape[0], dtype=bool)
        condition_details: list[dict[str, Any]] = []
        for condition in rule.conditions:
            if condition.feature not in feature_index:
                raise SystemExit(f"entry veto feature {condition.feature!r} not found in generated feature matrix")
            col_i = int(feature_index[condition.feature])
            val_values = val_x[:, col_i].astype(np.float64, copy=False)
            val_scope = np.ones(val_x.shape[0], dtype=bool)
            eval_scope = np.ones(eval_x.shape[0], dtype=bool)
            if condition.decision_side is not None:
                side_i = 0 if condition.decision_side == "LONG" else 1
                val_scope &= np.asarray(val_decision_side, dtype=np.int64) == side_i
                eval_scope &= np.asarray(eval_decision_side, dtype=np.int64) == side_i
            if condition.trade_session is not None:
                val_scope &= np.asarray(val_sessions, dtype=str) == condition.trade_session
                eval_scope &= np.asarray(eval_sessions, dtype=str) == condition.trade_session
            finite_val = val_values[val_scope & np.isfinite(val_values)]
            if finite_val.size == 0:
                condition_val_skip = np.zeros(val_x.shape[0], dtype=bool)
                condition_eval_skip = np.zeros(eval_x.shape[0], dtype=bool)
                rule_val_skip &= condition_val_skip
                rule_eval_skip &= condition_eval_skip
                condition_details.append(
                    {
                        "feature": condition.feature,
                        "skip_side": condition.skip_side,
                        "quantile": float(condition.quantile),
                        "decision_side": condition.decision_side,
                        "trade_session": condition.trade_session,
                        "threshold": None,
                        "calibration_skipped": "no_finite_validation_values_in_scope",
                        "val_scope_rows": int(val_scope.sum()),
                        "eval_scope_rows": int(eval_scope.sum()),
                        "val_condition_skipped": 0,
                        "eval_condition_skipped": 0,
                    }
                )
                continue
            threshold = float(np.quantile(finite_val, float(condition.quantile)))
            eval_values = eval_x[:, col_i].astype(np.float64, copy=False)
            val_cmp = np.nan_to_num(val_values, nan=threshold, posinf=threshold, neginf=threshold)
            eval_cmp = np.nan_to_num(eval_values, nan=threshold, posinf=threshold, neginf=threshold)
            if condition.skip_side == "low":
                condition_val_skip = val_cmp < threshold
                condition_eval_skip = eval_cmp < threshold
            else:
                condition_val_skip = val_cmp > threshold
                condition_eval_skip = eval_cmp > threshold
            condition_val_skip &= val_scope
            condition_eval_skip &= eval_scope
            rule_val_skip &= condition_val_skip
            rule_eval_skip &= condition_eval_skip
            condition_details.append(
                {
                    "feature": condition.feature,
                    "skip_side": condition.skip_side,
                    "quantile": float(condition.quantile),
                    "decision_side": condition.decision_side,
                    "trade_session": condition.trade_session,
                    "threshold": threshold,
                    "val_scope_rows": int(val_scope.sum()),
                    "eval_scope_rows": int(eval_scope.sum()),
                    "val_condition_skipped": int(condition_val_skip.sum()),
                    "eval_condition_skipped": int(condition_eval_skip.sum()),
                }
            )
        rule_val_keep = ~rule_val_skip
        rule_eval_keep = ~rule_eval_skip
        val_keep &= rule_val_keep
        eval_keep &= rule_eval_keep
        details.append(
            {
                "rule_index": int(rule_i),
                "conditions": condition_details,
                "compound": len(rule.conditions) > 1,
                "val_rule_skipped": int((~rule_val_keep).sum()),
                "eval_rule_skipped": int((~rule_eval_keep).sum()),
            }
        )
    return val_keep, eval_keep, details


def _session_mask(frame: pd.DataFrame, policy: SessionPolicy) -> np.ndarray:
    sessions = frame["session"].astype(str)
    mask = np.ones(len(frame), dtype=bool)
    if policy.include is not None:
        mask &= sessions.isin(set(policy.include)).to_numpy()
    if policy.exclude is not None:
        mask &= ~sessions.isin(set(policy.exclude)).to_numpy()
    return mask


def _hash_train_config(config: dict[str, Any]) -> str:
    raw = json.dumps(config, sort_keys=True, default=_json_default).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _policy_args(args: argparse.Namespace, exit_spec: ExitSpec) -> argparse.Namespace:
    ns = copy.copy(args)
    ns.exit_mode = exit_spec.exit_mode
    ns.take_profit_bps = exit_spec.take_profit_bps
    ns.stop_loss_bps = exit_spec.stop_loss_bps
    ns.same_bar_policy = exit_spec.same_bar_policy
    return ns


def _aggregate_replay_outputs(
    *,
    trades: pd.DataFrame,
    decisions: pd.DataFrame,
    thresholds: pd.DataFrame,
    out_dir: Path,
    policy_meta: dict[str, dict[str, Any]],
) -> dict[str, str]:
    trades_path = out_dir / "ablation_policy_trades.csv"
    decisions_path = out_dir / "ablation_policy_decisions.csv"
    thresholds_path = out_dir / "ablation_policy_thresholds.csv"
    metrics_path = out_dir / "ablation_policy_metrics.csv"
    daily_path = out_dir / "ablation_policy_daily.csv"
    monthly_path = out_dir / "ablation_policy_monthly.csv"
    leaderboard_path = out_dir / "leaderboard.csv"

    trades.to_csv(trades_path, index=False)
    decisions.to_csv(decisions_path, index=False)
    thresholds.to_csv(thresholds_path, index=False)

    metric_rows: list[dict[str, Any]] = []
    if not trades.empty:
        for (policy_id, fold), frame in trades.groupby(["policy_id", "fold"], sort=True):
            row = _metrics_row("fold", str(fold), str(policy_id), frame)
            row.update(policy_meta.get(str(policy_id), {}))
            metric_rows.append(row)
        for policy_id, frame in trades.groupby("policy_id", sort=True):
            row = _metrics_row("all", "ALL", str(policy_id), frame)
            row.update(policy_meta.get(str(policy_id), {}))
            metric_rows.append(row)
    metrics = pd.DataFrame(metric_rows)

    if trades.empty:
        daily = pd.DataFrame()
        monthly = pd.DataFrame()
    else:
        daily = (
            trades.groupby(["policy_id", "entry_day"], as_index=False)
            .agg(
                n_trades=("net_pnl_bps", "size"),
                net_sum_bps=("net_pnl_bps", "sum"),
                net_mean_bps=("net_pnl_bps", "mean"),
                wins=("net_pnl_bps", lambda s: int((s > 0).sum())),
            )
        )
        daily["win_rate"] = daily["wins"] / daily["n_trades"].clip(lower=1)

        monthly = (
            trades.groupby(["policy_id", "entry_month"], as_index=False)
            .agg(
                n_trades=("net_pnl_bps", "size"),
                net_sum_bps=("net_pnl_bps", "sum"),
                net_mean_bps=("net_pnl_bps", "mean"),
                wins=("net_pnl_bps", lambda s: int((s > 0).sum())),
            )
        )
        monthly["win_rate"] = monthly["wins"] / monthly["n_trades"].clip(lower=1)

    if not metrics.empty and not monthly.empty:
        pos_months = (
            monthly.assign(positive_month=lambda d: d["net_sum_bps"] > 0)
            .groupby("policy_id", as_index=False)
            .agg(positive_months=("positive_month", "sum"), total_months=("entry_month", "nunique"))
        )
        metrics = metrics.merge(pos_months, on="policy_id", how="left")

    if not metrics.empty:
        metrics["return_to_dd"] = metrics["net_sum_bps"] / metrics["max_drawdown_bps"].clip(lower=1.0)
        metrics["tail_loss_abs_bps"] = metrics["max_loss_bps"].abs()
        metrics["risk_adjusted_score"] = (
            metrics["net_sum_bps"].fillna(0.0)
            - 2.0 * metrics["max_drawdown_bps"].fillna(0.0)
            - 1.0 * metrics["tail_loss_abs_bps"].fillna(0.0)
            + 75.0 * metrics["profit_factor"].fillna(0.0)
        )
    metrics.to_csv(metrics_path, index=False)

    if metrics.empty:
        leaderboard = pd.DataFrame()
    else:
        leaderboard = (
            metrics[metrics["scope"] == "all"]
            .sort_values(
                ["risk_adjusted_score", "return_to_dd", "net_mean_bps", "max_drawdown_bps"],
                ascending=[False, False, False, True],
            )
            .reset_index(drop=True)
        )
    leaderboard.to_csv(leaderboard_path, index=False)
    daily.to_csv(daily_path, index=False)
    monthly.to_csv(monthly_path, index=False)

    return {
        "trades_csv": str(trades_path),
        "decisions_csv": str(decisions_path),
        "thresholds_csv": str(thresholds_path),
        "metrics_csv": str(metrics_path),
        "daily_csv": str(daily_path),
        "monthly_csv": str(monthly_path),
        "leaderboard_csv": str(leaderboard_path),
    }


def _baseline_reference(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty or "scope" not in df:
        return None
    candidates = df[(df["scope"] == "all") & (df["cost_stress_bps"] == 10.0)].copy()
    if "threshold_top_frac" in candidates:
        exact = candidates[np.isclose(candidates["threshold_top_frac"], 0.10)]
        if not exact.empty:
            candidates = exact
    if candidates.empty:
        return None
    numeric_cols = [
        "threshold_top_frac",
        "cost_stress_bps",
        "slippage_bps",
        "size_multiplier",
        "n_trades",
        "net_sum_bps",
        "net_mean_bps",
        "win_rate",
        "profit_factor",
        "max_loss_bps",
        "max_drawdown_bps",
    ]
    for col in numeric_cols:
        if col in candidates:
            candidates[col] = pd.to_numeric(candidates[col], errors="coerce")
    row = candidates.sort_values("net_sum_bps", ascending=False).iloc[0].to_dict()
    return {k: _json_default(v) for k, v in row.items()}


def _run_fold_model(
    *,
    fold: FoldSpec,
    feature_set: FeatureSet,
    model_name: str,
    all_x: np.ndarray,
    all_feature_names: list[str],
    veto_x: np.ndarray,
    veto_feature_names: list[str],
    y: np.ndarray,
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    eval_idx: np.ndarray,
    source_idx: np.ndarray,
    tape: SourceTape,
    session_policies: list[SessionPolicy],
    entry_veto_sets: list[EntryVetoSet],
    exit_specs: list[ExitSpec],
    threshold_top_fracs: list[float],
    cost_stress_bps_values: list[float],
    args: argparse.Namespace,
    model_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    x_view = all_x[:, feature_set.indices]
    veto_feature_index = _name_index(veto_feature_names)
    capped_train_idx = _maybe_cap_train_indices(train_idx, int(args.max_train_rows), int(args.seed))
    train_config = {
        "fold": fold.fold_id,
        "feature_set": feature_set.name,
        "model": model_name,
        "n_features": len(feature_set.names),
        "model_signature": _model_signature(model_name, args),
        "train_rows": int(len(capped_train_idx)),
        "val_rows": int(len(val_idx)),
        "eval_rows": int(len(eval_idx)),
        "max_train_rows": int(args.max_train_rows),
        "seed": int(args.seed),
    }
    train_hash = _hash_train_config(train_config)
    model_path = model_dir / f"{fold.fold_id}__{feature_set.name}__{model_name}__{train_hash}.joblib"

    t0 = perf_counter()
    model = _fit_model(
        model_name=model_name,
        x_train=x_view[capped_train_idx],
        y_train=y[capped_train_idx],
        x_val=x_view[val_idx],
        y_val=y[val_idx],
        categorical_idx=feature_set.categorical_idx,
        args=args,
    )
    train_seconds = perf_counter() - t0
    joblib.dump(model, model_path)

    val_probs = _predict_proba_aligned(model, x_view[val_idx])
    eval_probs = _predict_proba_aligned(model, x_view[eval_idx])
    val_side, _val_prob, val_score = _decision_arrays(val_probs)
    eval_side, _eval_prob, _eval_score = _decision_arrays(eval_probs)
    eval_df = df.iloc[eval_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    model_meta = {
        **train_config,
        "train_hash": train_hash,
        "model_path": str(model_path),
        "train_seconds": float(train_seconds),
        "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
        "fold_start": str(fold.start),
        "fold_end": str(fold.end),
    }
    importance_rows = _feature_importance_rows(model, feature_set.names, model_meta)
    trades_all: list[dict[str, Any]] = []
    decisions_all: list[dict[str, Any]] = []
    thresholds_all: list[dict[str, Any]] = []
    policy_meta: dict[str, dict[str, Any]] = {}

    for session_policy in session_policies:
        val_mask = _session_mask(val_df, session_policy)
        eval_mask = _session_mask(eval_df, session_policy)
        if int(val_mask.sum()) < int(args.min_threshold_rows) or int(eval_mask.sum()) == 0:
            thresholds_all.append(
                {
                    **model_meta,
                    "session_policy": session_policy.name,
                    "status": "SKIPPED_INSUFFICIENT_ROWS",
                    "val_session_rows": int(val_mask.sum()),
                    "eval_session_rows": int(eval_mask.sum()),
                }
            )
            continue

        session_val_score_raw = val_score[val_mask]
        session_eval_df_raw = eval_df.loc[eval_mask].reset_index(drop=True)
        session_eval_probs_raw = eval_probs[eval_mask]
        session_source_idx_raw = source_idx[eval_mask]
        session_val_x = veto_x[val_idx][val_mask]
        session_eval_x = veto_x[eval_idx][eval_mask]
        session_val_side = val_side[val_mask]
        session_eval_side = eval_side[eval_mask]
        session_val_names = val_df.loc[val_mask, "session"].astype(str).to_numpy()
        session_eval_names = eval_df.loc[eval_mask, "session"].astype(str).to_numpy()

        for veto_set in entry_veto_sets:
            val_veto_keep, eval_veto_keep, veto_details = _entry_veto_masks(
                veto_set=veto_set,
                feature_index=veto_feature_index,
                val_x=session_val_x,
                eval_x=session_eval_x,
                val_decision_side=session_val_side,
                eval_decision_side=session_eval_side,
                val_sessions=session_val_names,
                eval_sessions=session_eval_names,
            )
            session_val_score = session_val_score_raw[val_veto_keep]
            if int(len(session_val_score)) < int(args.min_threshold_rows) or int(eval_veto_keep.sum()) == 0:
                thresholds_all.append(
                    {
                        **model_meta,
                        "session_policy": session_policy.name,
                        "entry_veto_set": veto_set.name,
                        "entry_veto_rules": json.dumps(veto_details, default=_json_default),
                        "status": "SKIPPED_VETO_INSUFFICIENT_ROWS",
                        "val_session_rows": int(val_mask.sum()),
                        "eval_session_rows": int(eval_mask.sum()),
                        "val_veto_kept_rows": int(len(session_val_score)),
                        "eval_veto_kept_rows": int(eval_veto_keep.sum()),
                        "val_veto_skipped_rows": int((~val_veto_keep).sum()),
                        "eval_veto_skipped_rows": int((~eval_veto_keep).sum()),
                    }
                )
                continue

            session_eval_df = session_eval_df_raw.loc[eval_veto_keep].reset_index(drop=True)
            session_eval_probs = session_eval_probs_raw[eval_veto_keep]
            session_source_idx = session_source_idx_raw[eval_veto_keep]
            veto_rules_json = json.dumps(veto_details, sort_keys=True, default=_json_default)
            veto_meta = {
                "entry_veto_set": veto_set.name,
                "entry_veto_rules": veto_rules_json,
                "val_veto_kept_rows": int(len(session_val_score)),
                "eval_veto_kept_rows": int(eval_veto_keep.sum()),
                "val_veto_skipped_rows": int((~val_veto_keep).sum()),
                "eval_veto_skipped_rows": int((~eval_veto_keep).sum()),
            }

            for top_frac in threshold_top_fracs:
                threshold = _threshold_from_scores(session_val_score, top_frac, float(args.min_score_floor))
                thresholds_all.append(
                    {
                        **model_meta,
                        "session_policy": session_policy.name,
                        **veto_meta,
                        "status": "OK",
                        "threshold_top_frac": float(top_frac),
                        "score_threshold": float(threshold),
                        "val_session_rows": int(val_mask.sum()),
                        "eval_session_rows": int(eval_mask.sum()),
                        "val_score_mean": float(np.mean(session_val_score)),
                        "val_score_p50": float(np.percentile(session_val_score, 50)),
                        "val_score_p90": float(np.percentile(session_val_score, 90)),
                        "val_score_p95": float(np.percentile(session_val_score, 95)),
                    }
                )
                for cost_bps in cost_stress_bps_values:
                    for exit_spec in exit_specs:
                        policy_args = _policy_args(args, exit_spec)
                        config = {
                            "policy_hash_version": 2,
                            "feature_set": feature_set.name,
                            "model": model_name,
                            "model_signature": _model_signature(model_name, args),
                            "veto_feature_layer_signature": _veto_feature_layer_signature(args),
                            "session_policy": session_policy.name,
                            "threshold_top_frac": float(top_frac),
                            "cost_stress_bps": float(cost_bps),
                            "exit": exit_spec.__dict__,
                            "cooldown_bars": int(args.cooldown_bars),
                            "max_trades_per_day": int(args.max_trades_per_day),
                            "daily_loss_limit_bps": float(args.daily_loss_limit_bps),
                            "min_direction_prob": float(args.min_direction_prob),
                            "min_score_floor": float(args.min_score_floor),
                            "slippage_bps": float(args.slippage_bps),
                            "size_multiplier": float(args.size_multiplier),
                        }
                        policy_parts: list[Any] = [feature_set.name, model_name, session_policy.name]
                        if veto_set.rules:
                            config["entry_veto_set"] = _entry_veto_set_to_dict(veto_set)
                            policy_parts.append(veto_set.name)
                        config_hash = _policy_hash(config)
                        policy_parts.extend(
                            [
                                _frac_label(float(top_frac)),
                                _cost_label(float(cost_bps)),
                                exit_spec.name,
                                config_hash,
                            ]
                        )
                        policy_id = _model_slug(policy_parts)
                        meta = {
                            "feature_set": feature_set.name,
                            "model": model_name,
                            "session_policy": session_policy.name,
                            "exit_spec": exit_spec.name,
                            "exit_mode": exit_spec.exit_mode,
                            "take_profit_bps": float(exit_spec.take_profit_bps),
                            "stop_loss_bps": float(exit_spec.stop_loss_bps),
                            "same_bar_policy": exit_spec.same_bar_policy,
                            "n_features": int(len(feature_set.names)),
                            "n_veto_features": int(len(veto_feature_names)),
                            "train_rows": int(len(capped_train_idx)),
                            "val_rows": int(len(val_idx)),
                            "eval_rows": int(len(eval_idx)),
                            "train_seconds": float(train_seconds),
                            **veto_meta,
                        }
                        policy_meta[policy_id] = meta
                        trades, decisions = _run_policy(
                            fold_id=fold.fold_id,
                            eval_df=session_eval_df,
                            probs=session_eval_probs,
                            source_idx=session_source_idx,
                            tape=tape,
                            threshold_top_frac=float(top_frac),
                            score_threshold=float(threshold),
                            cost_stress_bps=float(cost_bps),
                            args=policy_args,
                            policy_id=policy_id,
                            policy_config_hash=config_hash,
                        )
                        for trade in trades:
                            trade.update(meta)
                        decisions.update(meta)
                        decisions["policy_id"] = policy_id
                        decisions["policy_config_hash"] = config_hash
                        decisions_all.append(decisions)
                        trades_all.extend(trades)

    return trades_all, decisions_all, thresholds_all, importance_rows, {"model_meta": model_meta, "policy_meta": policy_meta}


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    source_parquet = Path(args.source_parquet).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    model_dir = out_dir / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    data_splits = _parse_csv(args.data_splits)
    folds = _parse_folds(args.folds)
    model_names = _parse_csv(args.models)
    requested_feature_sets = _parse_csv(args.feature_sets)
    threshold_top_fracs = _parse_float_list(args.threshold_top_fracs)
    cost_stress_bps_values = _parse_float_list(args.cost_stress_bps)
    exit_specs = _parse_exit_specs(args.exit_specs, args.same_bar_policy)
    entry_veto_sets = _parse_entry_veto_rule_sets(args.entry_veto_rule_sets)
    session_policy_names = _parse_csv(args.session_policies)
    invalid_session_policies = [name for name in session_policy_names if name not in SESSION_POLICIES]
    if invalid_session_policies:
        raise SystemExit(f"unknown session policies {invalid_session_policies}; valid={sorted(SESSION_POLICIES)}")
    session_policies = [SESSION_POLICIES[name] for name in session_policy_names]
    if bool(args.include_price_ema_features) and bool(args.veto_only_price_ema_features):
        raise SystemExit("--include-price-ema-features and --veto-only-price-ema-features are mutually exclusive")

    x, y, df, base_feature_names, categorical_idx = _load_all_data(dataset_dir, data_splits)
    _check_no_xgb_feature_names(base_feature_names)
    base_chart_x, base_chart_feature_names = _build_chart_layer(x, base_feature_names)
    chart_x = base_chart_x
    chart_feature_names = list(base_chart_feature_names)
    price_x = np.empty((len(df), 0), dtype=np.float32)
    price_feature_names: list[str] = []
    if bool(args.include_price_ema_features) or bool(args.veto_only_price_ema_features):
        price_x, price_feature_names = _build_price_derived_layer(df, source_parquet)
        if bool(args.include_price_ema_features) and price_x.shape[1]:
            chart_x = (
                np.concatenate([chart_x, price_x], axis=1).astype(np.float32, copy=False)
                if chart_x.shape[1]
                else price_x
            )
            chart_feature_names = list(chart_feature_names) + list(price_feature_names)
    chart_all_x = np.concatenate([x, chart_x], axis=1).astype(np.float32, copy=False) if chart_x.shape[1] else x
    chart_all_names = list(base_feature_names) + list(chart_feature_names)
    deep_x, deep_feature_names = _build_deep_interaction_layer(chart_all_x, chart_all_names, df)
    all_x, all_feature_names, feature_sets = _feature_sets(
        x,
        base_feature_names,
        categorical_idx,
        chart_x,
        chart_feature_names,
        deep_x,
        deep_feature_names,
        requested_feature_sets,
    )
    veto_x = all_x
    veto_feature_names = list(all_feature_names)
    veto_deep_feature_names: list[str] = list(deep_feature_names)
    if bool(args.veto_only_price_ema_features) and price_x.shape[1]:
        veto_chart_x = (
            np.concatenate([base_chart_x, price_x], axis=1).astype(np.float32, copy=False)
            if base_chart_x.shape[1]
            else price_x
        )
        veto_chart_feature_names = list(base_chart_feature_names) + list(price_feature_names)
        veto_chart_all_x = (
            np.concatenate([x, veto_chart_x], axis=1).astype(np.float32, copy=False)
            if veto_chart_x.shape[1]
            else x
        )
        veto_chart_all_names = list(base_feature_names) + list(veto_chart_feature_names)
        veto_deep_x, veto_deep_feature_names = _build_deep_interaction_layer(
            veto_chart_all_x,
            veto_chart_all_names,
            df,
        )
        veto_pieces = [x]
        if veto_chart_x.shape[1]:
            veto_pieces.append(veto_chart_x)
        if veto_deep_x.shape[1]:
            veto_pieces.append(veto_deep_x)
        veto_x = np.concatenate(veto_pieces, axis=1).astype(np.float32, copy=False)
        veto_feature_names = list(base_feature_names) + list(veto_chart_feature_names) + list(veto_deep_feature_names)
    tape = SourceTape.load(source_parquet)

    all_trades: list[dict[str, Any]] = []
    all_decisions: list[dict[str, Any]] = []
    all_thresholds: list[dict[str, Any]] = []
    all_importance: list[dict[str, Any]] = []
    model_runs: list[dict[str, Any]] = []
    policy_meta: dict[str, dict[str, Any]] = {}

    for fold in folds:
        train_idx, val_idx, eval_idx = _fold_indices(
            times=df["time"],
            fold=fold,
            val_tail_days=int(args.val_tail_days),
            min_val_rows=int(args.min_val_rows),
            min_train_rows=int(args.min_train_rows),
        )
        eval_df = df.iloc[eval_idx].reset_index(drop=True)
        source_idx = tape.indices_for_times(eval_df["time"])
        for feature_set in feature_sets:
            for model_name in model_names:
                print(
                    json.dumps(
                        {
                            "event": "train_fold_model",
                            "fold": fold.fold_id,
                            "feature_set": feature_set.name,
                            "model": model_name,
                            "n_features": len(feature_set.names),
                        },
                        default=_json_default,
                    ),
                    flush=True,
                )
                trades, decisions, thresholds, importance, meta = _run_fold_model(
                    fold=fold,
                    feature_set=feature_set,
                    model_name=model_name,
                    all_x=all_x,
                    all_feature_names=all_feature_names,
                    veto_x=veto_x,
                    veto_feature_names=veto_feature_names,
                    y=y,
                    df=df,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    eval_idx=eval_idx,
                    source_idx=source_idx,
                    tape=tape,
                    session_policies=session_policies,
                    entry_veto_sets=entry_veto_sets,
                    exit_specs=exit_specs,
                    threshold_top_fracs=threshold_top_fracs,
                    cost_stress_bps_values=cost_stress_bps_values,
                    args=args,
                    model_dir=model_dir,
                )
                all_trades.extend(trades)
                all_decisions.extend(decisions)
                all_thresholds.extend(thresholds)
                all_importance.extend(importance)
                model_runs.append(meta["model_meta"])
                policy_meta.update(meta["policy_meta"])

    trades_df = pd.DataFrame(all_trades)
    decisions_df = pd.DataFrame(all_decisions)
    thresholds_df = pd.DataFrame(all_thresholds)
    importance_df = pd.DataFrame(all_importance)
    importance_path = out_dir / "feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    if not importance_df.empty:
        importance_mean = (
            importance_df.groupby(["feature_set", "model", "feature"], as_index=False)["importance"]
            .mean()
            .sort_values(["feature_set", "model", "importance"], ascending=[True, True, False])
        )
        importance_mean.to_csv(out_dir / "feature_importance_mean.csv", index=False)

    outputs = _aggregate_replay_outputs(
        trades=trades_df,
        decisions=decisions_df,
        thresholds=thresholds_df,
        out_dir=out_dir,
        policy_meta=policy_meta,
    )

    metrics_df = pd.read_csv(outputs["metrics_csv"]) if Path(outputs["metrics_csv"]).exists() else pd.DataFrame()
    leaderboard_df = pd.read_csv(outputs["leaderboard_csv"]) if Path(outputs["leaderboard_csv"]).exists() else pd.DataFrame()
    baseline = _baseline_reference(Path(args.baseline_metrics_csv).expanduser().resolve())
    if baseline is not None and not leaderboard_df.empty:
        leaderboard_df["baseline_net_sum_bps"] = float(baseline.get("net_sum_bps") or 0.0)
        leaderboard_df["baseline_max_drawdown_bps"] = float(baseline.get("max_drawdown_bps") or 0.0)
        leaderboard_df["baseline_max_loss_bps"] = float(baseline.get("max_loss_bps") or 0.0)
        leaderboard_df["drawdown_delta_vs_baseline_bps"] = (
            leaderboard_df["max_drawdown_bps"] - leaderboard_df["baseline_max_drawdown_bps"]
        )
        leaderboard_df["max_loss_delta_vs_baseline_bps"] = (
            leaderboard_df["max_loss_bps"] - leaderboard_df["baseline_max_loss_bps"]
        )
        leaderboard_df.to_csv(outputs["leaderboard_csv"], index=False)

    best_rows = []
    if not leaderboard_df.empty:
        best_rows = leaderboard_df.head(int(args.summary_top_n)).to_dict(orient="records")

    summary = {
        "dataset_dir": str(dataset_dir),
        "source_parquet": str(source_parquet),
        "out_dir": str(out_dir),
        "data_splits": data_splits,
        "folds": [{"fold_id": f.fold_id, "start": str(f.start), "end": str(f.end)} for f in folds],
        "models": model_names,
        "feature_sets": requested_feature_sets,
        "session_policies": [p.name for p in session_policies],
        "entry_veto_rule_sets": [_entry_veto_set_to_dict(veto_set) for veto_set in entry_veto_sets],
        "exit_specs": [s.__dict__ for s in exit_specs],
        "threshold_top_fracs": threshold_top_fracs,
        "cost_stress_bps": cost_stress_bps_values,
        "slippage_bps": float(args.slippage_bps),
        "feature_counts": {
            "base": len(base_feature_names),
            "generated_price_ema": len(price_feature_names),
            "generated_chart": len(chart_feature_names),
            "generated_deep": len(deep_feature_names),
            "all": len(all_feature_names),
            "veto_all": len(veto_feature_names),
            "veto_generated_deep": len(veto_deep_feature_names),
        },
        "include_price_ema_features": bool(args.include_price_ema_features),
        "veto_only_price_ema_features": bool(args.veto_only_price_ema_features),
        "veto_feature_layer_signature": _veto_feature_layer_signature(args),
        "generated_chart_features": chart_feature_names,
        "generated_price_ema_features": price_feature_names,
        "generated_deep_features": deep_feature_names,
        "veto_generated_deep_features": veto_deep_feature_names,
        "model_runs": model_runs,
        "baseline_reference": baseline,
        "n_trades_total": int(len(trades_df)),
        "n_decision_runs": int(len(decisions_df)),
        "n_metric_rows": int(len(metrics_df)),
        "outputs": {**outputs, "feature_importance_csv": str(importance_path)},
        "best_rows": best_rows,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=_json_default))
    return summary


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    ap.add_argument("--source-parquet", default=str(DEFAULT_SOURCE_PARQUET))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--baseline-metrics-csv", default=str(DEFAULT_BASELINE_METRICS))
    ap.add_argument("--data-splits", default="train,val,test")
    ap.add_argument("--folds", default=DEFAULT_FOLDS)
    ap.add_argument("--models", default="lightgbm,histgb,extratrees,logreg")
    ap.add_argument("--feature-sets", default="base,base_plus_chart,chart_layer_only,base_without_chart")
    ap.add_argument(
        "--include-price-ema-features",
        action="store_true",
        help="Opt in to source-parquet-derived EMA50/EMA200 price features. Default off because v3 tail risk failed 10-seed stress.",
    )
    ap.add_argument(
        "--veto-only-price-ema-features",
        action="store_true",
        help=(
            "Build source-parquet-derived EMA50/EMA200 features only for fold-calibrated "
            "entry veto rules, without adding them to the model training feature set."
        ),
    )
    ap.add_argument(
        "--session-policies",
        default="ALL,NO_EU,OVERLAP_US,OVERLAP_ONLY,US_ONLY,ASIA_ONLY,EU_ONLY",
        help=f"Comma-separated policies from {sorted(SESSION_POLICIES)}",
    )
    ap.add_argument("--exit-specs", default="horizon,stop_tp:60:45,stop_tp:80:50,stop_tp:100:60")
    ap.add_argument("--threshold-top-fracs", default="0.03,0.05,0.075,0.10")
    ap.add_argument("--cost-stress-bps", default="10")
    ap.add_argument(
        "--entry-veto-rule-sets",
        default="none",
        help=(
            "Semicolon-separated fold-calibrated veto sets. Use 'none' or "
            "name=feature:low|high:quantile[+feature:low|high:quantile]. "
            "Use '&' inside a rule for compound vetoes that skip only when all "
            "conditions match, e.g. feature1:high:0.90&feature2:low:0.10. "
            "Thresholds are fit on each fold/session validation slice."
        ),
    )
    ap.add_argument("--val-tail-days", type=int, default=30)
    ap.add_argument("--min-val-rows", type=int, default=2500)
    ap.add_argument("--min-train-rows", type=int, default=50000)
    ap.add_argument("--min-threshold-rows", type=int, default=250)
    ap.add_argument("--max-train-rows", type=int, default=180000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--n-jobs", type=int, default=8)

    ap.add_argument("--cooldown-bars", type=int, default=6)
    ap.add_argument("--max-trades-per-day", type=int, default=8)
    ap.add_argument("--daily-loss-limit-bps", type=float, default=150.0)
    ap.add_argument("--min-direction-prob", type=float, default=0.0)
    ap.add_argument("--min-score-floor", type=float, default=0.0)
    ap.add_argument("--slippage-bps", type=float, default=5.0)
    ap.add_argument("--size-multiplier", type=float, default=1.0)
    ap.add_argument("--same-bar-policy", choices=("stop_first", "target_first"), default="stop_first")

    ap.add_argument("--lgbm-n-estimators", type=int, default=450)
    ap.add_argument("--lgbm-learning-rate", type=float, default=0.035)
    ap.add_argument("--lgbm-num-leaves", type=int, default=63)
    ap.add_argument("--lgbm-max-depth", type=int, default=-1)
    ap.add_argument("--lgbm-min-child-samples", type=int, default=250)
    ap.add_argument("--lgbm-early-stopping-rounds", type=int, default=60)

    ap.add_argument("--extra-trees-n-estimators", type=int, default=360)
    ap.add_argument("--extra-trees-max-depth", type=int, default=18)
    ap.add_argument("--extra-trees-min-samples-leaf", type=int, default=80)
    ap.add_argument("--extra-trees-max-features", default="sqrt")

    ap.add_argument("--histgb-max-iter", type=int, default=220)
    ap.add_argument("--histgb-learning-rate", type=float, default=0.045)
    ap.add_argument("--histgb-max-leaf-nodes", type=int, default=31)
    ap.add_argument("--histgb-min-samples-leaf", type=int, default=180)
    ap.add_argument("--histgb-l2-regularization", type=float, default=0.05)

    ap.add_argument("--logreg-c", type=float, default=0.25)
    ap.add_argument("--logreg-max-iter", type=int, default=300)

    ap.add_argument("--xgb-n-estimators", type=int, default=260)
    ap.add_argument("--xgb-learning-rate", type=float, default=0.04)
    ap.add_argument("--xgb-max-depth", type=int, default=4)
    ap.add_argument("--xgb-subsample", type=float, default=0.85)
    ap.add_argument("--xgb-colsample-bytree", type=float, default=0.85)
    ap.add_argument("--xgb-reg-lambda", type=float, default=1.0)

    ap.add_argument("--summary-top-n", type=int, default=30)
    return ap


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
