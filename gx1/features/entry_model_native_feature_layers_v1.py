"""Strict causal feature construction for the model-native Entry signal stack.

This module owns the deterministic chart, price, candlestick and deep
interaction layers consumed by the seq513 dataset builder.  It deliberately
contains no research evaluator, policy, model or artifact-default coupling.
Every declared source is required and finite; missing rows or malformed market
data are contract failures, never synthetic zero evidence.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from gx1.features.entry_candlestick_patterns_v1 import (
    CANDLESTICK_PATTERN_FEATURE_NAMES,
    CANDLESTICK_PATTERN_SOURCE_FIELDS,
    build_entry_candlestick_pattern_layer,
)
from gx1.features.entry_chart_geometry_v1 import (
    CHART_GEOMETRY_MODEL_NATIVE_FEATURE_NAMES,
    CHART_GEOMETRY_SOURCE_FIELDS,
    build_entry_chart_geometry_layer,
)
from gx1.features.entry_foundation_structure_v1 import (
    FOUNDATION_STRUCTURE_SOURCE_FIELDS,
    build_entry_foundation_structure_layer,
)
from gx1.features.entry_momentum_flow_v1 import MOMENTUM_FLOW_FEATURE_NAMES
from gx1.features.entry_mtf_confluence_v1 import MTF_CONFLUENCE_FEATURE_NAMES
from gx1.features.entry_session_regime_interactions_v1 import (
    SESSION_REGIME_INTERACTION_FEATURE_NAMES,
)
from gx1.features.entry_smc_liquidity_quality_v1 import (
    SMC_LIQUIDITY_QUALITY_FEATURE_NAMES,
)
from gx1.features.entry_structure_swing_derivations_v1 import (
    STRUCTURE_SWING_DERIVATION_FEATURE_NAMES,
)
from gx1.features.entry_support_resistance_memory_v1 import (
    SUPPORT_RESISTANCE_MEMORY_FEATURE_NAMES,
)
from gx1.features.entry_trend_ema_v1 import TREND_EMA_FEATURE_NAMES
from gx1.features.entry_vol_compression_v1 import VOL_COMPRESSION_FEATURE_NAMES


# Exact raw M5 trend-state evidence.  The offline builder and live adapter
# both supply these columns from the TRAIN-ranked common-history frame.
PRICE_DERIVED_SOURCE_PRICE_FIELD = "close"
PRICE_DERIVED_SOURCE_ATR_FIELD = "atr"
PRICE_DERIVED_FEATURE_NAMES = (
    "chart.m5_ema50_200_spread_bps",
    "chart.m5_ema50_200_spread_atr",
    "chart.m5_ema50_200_bull_state",
    "chart.m5_ema50_200_cross_up",
    "chart.m5_ema50_200_cross_down",
    "chart.m5_price_vs_ema50_bps",
    "chart.m5_price_vs_ema200_bps",
    "chart.m5_ema50_slope_bps",
    "chart.m5_ema200_slope_bps",
    "chart.m5_ema50_200_spread_delta",
    "chart.m5_ema50_200_spread_accel",
)


# Ordered ownership registry for every generated specialist layer that the
# canonical seq513 builder may materialize.  This belongs beside the builders,
# not in a report/materializer script with mutable historical artifact paths.
MODEL_NATIVE_SPECIALIST_LAYER_FEATURES: tuple[
    tuple[str, tuple[str, ...]], ...
] = (
    ("trend_ema_smart_layer", TREND_EMA_FEATURE_NAMES),
    ("smc_liquidity_quality_layer", SMC_LIQUIDITY_QUALITY_FEATURE_NAMES),
    (
        "structure_swing_derivation_layer",
        STRUCTURE_SWING_DERIVATION_FEATURE_NAMES,
    ),
    ("momentum_flow_smart_layer", MOMENTUM_FLOW_FEATURE_NAMES),
    (
        "session_regime_interaction_layer",
        SESSION_REGIME_INTERACTION_FEATURE_NAMES,
    ),
    ("vol_compression_smart_layer", VOL_COMPRESSION_FEATURE_NAMES),
    ("chart_geometry_smart2_layer", CHART_GEOMETRY_MODEL_NATIVE_FEATURE_NAMES),
    (
        "price_action_candle_smart3_layer",
        CANDLESTICK_PATTERN_FEATURE_NAMES[28:],
    ),
    ("support_resistance_memory_layer", SUPPORT_RESISTANCE_MEMORY_FEATURE_NAMES),
    ("mtf_confluence_layer", MTF_CONFLUENCE_FEATURE_NAMES),
    ("price_ema50_200_layer", PRICE_DERIVED_FEATURE_NAMES),
)

# This is the code-owned full-stack retention contract.  A feature-selection
# artifact may rank additional evidence, but it may never rank away one of
# these registered causal layer outputs.  Keep the flattened order stable: it
# becomes part of the immutable seq513 signal identity.
MODEL_NATIVE_MANDATORY_FAMILY_FEATURES = MODEL_NATIVE_SPECIALIST_LAYER_FEATURES
MODEL_NATIVE_MANDATORY_SELECTED_FIELDS = tuple(
    feature
    for _family, features in MODEL_NATIVE_MANDATORY_FAMILY_FEATURES
    for feature in features
)
MODEL_NATIVE_MANDATORY_FAMILY_COUNT = 11
MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT = 316

_mandatory_family_labels = tuple(
    family for family, _features in MODEL_NATIVE_MANDATORY_FAMILY_FEATURES
)
if len(_mandatory_family_labels) != MODEL_NATIVE_MANDATORY_FAMILY_COUNT:
    raise RuntimeError(
        "MODEL_NATIVE_MANDATORY_FAMILY_COUNT_MISMATCH: "
        f"observed={len(_mandatory_family_labels)} "
        f"expected={MODEL_NATIVE_MANDATORY_FAMILY_COUNT}"
    )
if len(set(_mandatory_family_labels)) != len(_mandatory_family_labels):
    raise RuntimeError("MODEL_NATIVE_MANDATORY_FAMILY_LABEL_DUPLICATE")
if any(not family or not features for family, features in MODEL_NATIVE_MANDATORY_FAMILY_FEATURES):
    raise RuntimeError("MODEL_NATIVE_MANDATORY_FAMILY_EMPTY")
if len(MODEL_NATIVE_MANDATORY_SELECTED_FIELDS) != MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT:
    raise RuntimeError(
        "MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT_MISMATCH: "
        f"observed={len(MODEL_NATIVE_MANDATORY_SELECTED_FIELDS)} "
        f"expected={MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT}"
    )
if len(set(MODEL_NATIVE_MANDATORY_SELECTED_FIELDS)) != len(
    MODEL_NATIVE_MANDATORY_SELECTED_FIELDS
):
    raise RuntimeError("MODEL_NATIVE_MANDATORY_SELECTED_FIELD_DUPLICATE")
if any(
    not isinstance(feature, str) or not feature.strip()
    for feature in MODEL_NATIVE_MANDATORY_SELECTED_FIELDS
):
    raise RuntimeError("MODEL_NATIVE_MANDATORY_SELECTED_FIELD_INVALID")


def _ordered_unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values))


CHART_LAYER_SOURCE_FIELDS = _ordered_unique(
    (
        *FOUNDATION_STRUCTURE_SOURCE_FIELDS,
        *CHART_GEOMETRY_SOURCE_FIELDS,
        "ctx_cont.struct_tf_agree_count_v3",
        "ctx_cont.d1_range_z_20_canon_v2",
        "ctx_cont.d1_trend_age_mature_flag_v3",
        "ctx_cont.d1_dist_to_boundary_v3",
        "ctx_cont.vol_pct_h1_1yr",
        "ctx_cont.vol_pct_m5_1yr",
        "ctx_cont.atr_ratio_h1_d1",
        "ctx_cont.atr_ratio_m15_d1",
    )
)

DEEP_INTERACTION_SOURCE_FIELDS = (
    "snap._v1_ema_diff",
    "ctx_cont._v1h1_ema_diff",
    "ctx_cont._v1h4_ema_diff",
    "snap.pos_vs_ema200",
    "snap.ema20_slope",
    "ctx_cont.d1_ema_slope_20_canon_v2",
    "ctx_cont._v1h1_slope5",
    "chart.m5_ema50_200_spread_bps",
    "chart.m5_ema50_200_spread_atr",
    "chart.m5_ema50_200_bull_state",
    "chart.m5_ema50_200_cross_up",
    "chart.m5_ema50_200_cross_down",
    "chart.m5_price_vs_ema200_bps",
    "chart.m5_ema50_slope_bps",
    "chart.m5_ema200_slope_bps",
    "chart.compression_h1_m15_bb",
    "ctx_cont.H1_range_compression_ratio",
    "ctx_cont.M15_range_compression_ratio",
    "chart.compression_to_expansion_proxy",
    "snap.rvol_20",
    "snap.vol_ratio_5_20",
    "snap.atr_z",
    "ctx_cont.regime_stack_sum_v3",
    "ctx_cont.regime_tf_agreement_v3",
    "ctx_cont.regime_divergence_flag_v3",
    "ctx_cont.d1_regime_changed_flag_v3",
    "ctx_cont.bars_since_d1_regime_change_v3",
    "chart.sweep_recent_combo",
    "snap.smc_sweep_up",
    "snap.smc_sweep_down",
    "chart.sweep_size_combo",
    "snap.smc_sweep_size_atr",
    "chart.choch_recent_combo",
    "snap.smc_choch",
    "chart.bos_pressure_combo",
    "ctx_cont.smc_bos_pressure_last48",
    "chart.wick_x_major_level",
    "ctx_cont.wick_ratio",
    "chart.pullback_depth_h1h4",
    "ctx_cont.struct_pullback_depth_h1_v3",
    "ctx_cont.struct_pullback_depth_h4_v3",
    "chart.hh_breakout_proxy",
    "chart.hl_pullback_proxy",
    "chart.lh_pullback_proxy",
    "chart.ll_breakdown_proxy",
    "ctx_cont.d1_close_pct_in_20day_range_canon_v2",
    "chart.major_level_proximity_max",
    "ctx_cont.sr_support_proximity_exp",
    "ctx_cont.sr_resistance_proximity_exp",
    "ctx_cont.spread_bps",
    "ctx_cont.minutes_since_session_open",
    "ctx_cont.minutes_to_next_session_boundary",
    "ctx_cont.vol_pct_h1_1yr",
    "ctx_cont.vol_pct_m5_1yr",
    "ctx_cont.D1_atr_percentile_252",
    "ctx_cont.atr_ratio_h1_d1",
    "ctx_cont.atr_ratio_m15_d1",
    "ctx_cont.is_ASIA",
    "ctx_cont.is_asia_eu_overlap",
    "ctx_cont.is_eu_us_overlap",
    "ctx_cont.is_eu_only",
    "ctx_cont.is_us_only",
    "ctx_cont.h1_trend_age_bars_norm_v2",
    "ctx_cont.h4_trend_age_bars_norm_v2",
)


def _require_matrix_contract(
    x: np.ndarray,
    feature_names: list[str],
    required_fields: Iterable[str],
    *,
    context: str,
) -> tuple[np.ndarray, dict[str, int]]:
    matrix = np.asarray(x, dtype=np.float32)
    if matrix.ndim != 2:
        raise RuntimeError(f"{context}_MATRIX_NOT_2D: shape={matrix.shape}")
    if matrix.shape[0] == 0:
        raise RuntimeError(f"{context}_ROWS_EMPTY")
    if matrix.shape[1] != len(feature_names):
        raise RuntimeError(
            f"{context}_NAME_WIDTH_MISMATCH: matrix={matrix.shape[1]} names={len(feature_names)}"
        )
    if len(feature_names) != len(set(feature_names)):
        duplicates = sorted({name for name in feature_names if feature_names.count(name) > 1})
        raise RuntimeError(f"{context}_DUPLICATE_FEATURE_NAMES: {duplicates[:30]}")
    index = {name: i for i, name in enumerate(feature_names)}
    missing = [name for name in required_fields if name not in index]
    if missing:
        raise RuntimeError(f"{context}_SOURCE_FIELDS_MISSING: {missing[:30]} total={len(missing)}")
    if not np.isfinite(matrix).all():
        bad_rows, bad_cols = np.where(~np.isfinite(matrix))
        examples = [
            {"row": int(row), "feature": feature_names[int(col)]}
            for row, col in zip(bad_rows[:10], bad_cols[:10])
        ]
        raise RuntimeError(f"{context}_SOURCE_NONFINITE: {examples}")
    return matrix, index


def _require_sample_times(frame: pd.DataFrame, *, context: str) -> pd.DatetimeIndex:
    if not isinstance(frame, pd.DataFrame):
        raise RuntimeError(f"{context}_SAMPLE_FRAME_INVALID: {type(frame).__name__}")
    if "time" not in frame.columns:
        raise RuntimeError(f"{context}_SAMPLE_TIME_MISSING")
    if len(frame) == 0:
        raise RuntimeError(f"{context}_SAMPLE_ROWS_EMPTY")
    try:
        times = pd.DatetimeIndex(pd.to_datetime(frame["time"], utc=True, errors="raise"))
    except Exception as exc:
        raise RuntimeError(f"{context}_SAMPLE_TIME_INVALID") from exc
    if times.isna().any():
        raise RuntimeError(f"{context}_SAMPLE_TIME_INVALID")
    if times.duplicated().any():
        raise RuntimeError(f"{context}_SAMPLE_TIME_DUPLICATE: count={int(times.duplicated().sum())}")
    return times


def _read_source_schema(path: Path, *, context: str) -> set[str]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise RuntimeError(f"{context}_SOURCE_PARQUET_MISSING: {source}")
    try:
        import pyarrow.parquet as pq

        return set(pq.read_schema(source).names)
    except Exception as exc:
        raise RuntimeError(f"{context}_SOURCE_SCHEMA_INVALID: {source}") from exc


def _normalize_source_times(frame: pd.DataFrame, *, context: str) -> pd.DataFrame:
    try:
        times = pd.to_datetime(frame["time"], utc=True, errors="raise")
    except Exception as exc:
        raise RuntimeError(f"{context}_SOURCE_TIME_INVALID") from exc
    if times.isna().any():
        raise RuntimeError(f"{context}_SOURCE_TIME_INVALID")
    if times.duplicated().any():
        raise RuntimeError(f"{context}_SOURCE_TIME_DUPLICATE: count={int(times.duplicated().sum())}")
    out = frame.copy()
    out["time"] = times
    return out.sort_values("time", kind="mergesort").reset_index(drop=True)


def _require_finite_positive_column(frame: pd.DataFrame, name: str, *, context: str) -> np.ndarray:
    try:
        values = pd.to_numeric(frame[name], errors="raise").to_numpy(dtype=np.float64)
    except Exception as exc:
        raise RuntimeError(f"{context}_SOURCE_COLUMN_INVALID: {name}") from exc
    if not np.isfinite(values).all():
        raise RuntimeError(f"{context}_SOURCE_NONFINITE: {name}")
    if np.any(values <= 0.0):
        raise RuntimeError(f"{context}_SOURCE_NONPOSITIVE: {name}")
    return values


def _col(x: np.ndarray, index: dict[str, int], name: str) -> np.ndarray:
    try:
        return x[:, index[name]].astype(np.float32, copy=False)
    except KeyError as exc:
        raise RuntimeError(f"MODEL_NATIVE_FEATURE_SOURCE_MISSING: {name}") from exc


def _clip(arr: np.ndarray, lo: float = -25.0, hi: float = 25.0) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float32)
    if not np.isfinite(values).all():
        raise RuntimeError("MODEL_NATIVE_GENERATED_FEATURE_NONFINITE")
    return np.clip(values, lo, hi).astype(np.float32, copy=False)


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


def add_chart_feature(
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
    arrays.append(clean)
    names.append(f"chart.{name}")


def build_price_derived_layer(
    sample_df: pd.DataFrame,
    source_parquet: Path,
) -> tuple[np.ndarray, list[str]]:
    """Build the exact past-only price/EMA layer with strict source alignment."""

    context = "PRICE_DERIVED"
    sample_times = _require_sample_times(sample_df, context=context)
    source = Path(source_parquet).expanduser().resolve()
    available = _read_source_schema(source, context=context)
    if "time" not in available:
        raise RuntimeError(f"{context}_SOURCE_FIELDS_MISSING: ['time']")
    price_field = PRICE_DERIVED_SOURCE_PRICE_FIELD
    atr_field = PRICE_DERIVED_SOURCE_ATR_FIELD
    if price_field not in available:
        raise RuntimeError(f"{context}_SOURCE_PRICE_MISSING: required={price_field}")
    if atr_field not in available:
        raise RuntimeError(f"{context}_SOURCE_ATR_MISSING: required={atr_field}")
    src = pd.read_parquet(source, columns=["time", price_field, atr_field], engine="pyarrow")
    src = _normalize_source_times(src, context=context)
    close_values = _require_finite_positive_column(src, price_field, context=context)
    atr_values = _require_finite_positive_column(src, atr_field, context=context)

    source_index = pd.DatetimeIndex(src["time"])
    missing_times = sample_times.difference(source_index)
    if len(missing_times):
        raise RuntimeError(
            f"{context}_SOURCE_ROW_GAP: missing={len(missing_times)} first={missing_times[0]}"
        )

    close = pd.Series(close_values, index=source_index, dtype=np.float64)
    atr = pd.Series(atr_values, index=source_index, dtype=np.float64)
    ema50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
    ema200 = close.ewm(span=200, adjust=False, min_periods=200).mean()
    spread = ema50 - ema200
    denom = close.abs()
    both_ema = ema50.notna() & ema200.notna()

    def zero_before_ready(values: pd.Series, ready: pd.Series) -> pd.Series:
        out = pd.Series(np.zeros(len(values), dtype=np.float64), index=values.index)
        out.loc[ready] = values.loc[ready]
        if not np.isfinite(out.to_numpy(dtype=np.float64)).all():
            raise RuntimeError(f"{context}_GENERATED_NONFINITE")
        return out

    spread_bps = zero_before_ready(spread / denom * 1e4, both_ema)
    spread_atr = zero_before_ready(spread / atr, both_ema)
    price_vs_ema50 = zero_before_ready((close - ema50) / denom * 1e4, ema50.notna())
    price_vs_ema200 = zero_before_ready((close - ema200) / denom * 1e4, ema200.notna())
    ema50_slope = zero_before_ready(ema50.diff() / denom * 1e4, ema50.notna() & ema50.shift(1).notna())
    ema200_slope = zero_before_ready(ema200.diff() / denom * 1e4, ema200.notna() & ema200.shift(1).notna())
    def causal_delta(values: pd.Series) -> pd.Series:
        raw = values.to_numpy(dtype=np.float64)
        out = np.zeros_like(raw)
        out[1:] = raw[1:] - raw[:-1]
        return pd.Series(out, index=values.index)

    spread_delta = causal_delta(spread_bps)
    spread_accel = causal_delta(spread_delta)

    raw = pd.DataFrame(
        {
            "ema50_200_spread_bps": spread_bps,
            "ema50_200_spread_atr": spread_atr,
            "ema50_200_bull_state": (spread > 0).astype(np.float64),
            "ema50_200_cross_up": ((spread > 0) & (spread.shift(1) <= 0)).astype(np.float64),
            "ema50_200_cross_down": ((spread < 0) & (spread.shift(1) >= 0)).astype(np.float64),
            "price_vs_ema50_bps": price_vs_ema50,
            "price_vs_ema200_bps": price_vs_ema200,
            "ema50_slope_bps": ema50_slope,
            "ema200_slope_bps": ema200_slope,
            "ema50_200_spread_delta": spread_delta,
            "ema50_200_spread_accel": spread_accel,
        },
        index=source_index,
    )
    aligned = raw.loc[sample_times]
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
    for column in aligned.columns:
        lo, hi = clip_ranges.get(column, (-25.0, 25.0))
        add_chart_feature(
            arrays,
            names,
            f"m5_{column}",
            aligned[column].to_numpy(dtype=np.float32),
            lo=lo,
            hi=hi,
        )
    out = np.column_stack(arrays).astype(np.float32, copy=False)
    if tuple(names) != PRICE_DERIVED_FEATURE_NAMES:
        raise RuntimeError("PRICE_DERIVED_FEATURE_ORDER_INVALID")
    return out, names


def build_candlestick_derived_layer(
    sample_df: pd.DataFrame,
    source_parquet: Path,
) -> tuple[np.ndarray, list[str]]:
    """Build the closed-bar candlestick layer with exact timestamp/OHLC proof."""

    context = "CANDLESTICK_DERIVED"
    sample_times = _require_sample_times(sample_df, context=context)
    source = Path(source_parquet).expanduser().resolve()
    available = _read_source_schema(source, context=context)
    missing = [name for name in CANDLESTICK_PATTERN_SOURCE_FIELDS if name not in available]
    if missing:
        raise RuntimeError(f"{context}_SOURCE_FIELDS_MISSING: {missing}")
    src = pd.read_parquet(source, columns=list(CANDLESTICK_PATTERN_SOURCE_FIELDS), engine="pyarrow")
    src = _normalize_source_times(src, context=context)
    ohlc = {}
    for name in ("open", "high", "low", "close"):
        ohlc[name] = _require_finite_positive_column(src, name, context=context)
    invalid_geometry = (
        (ohlc["high"] < ohlc["low"])
        | (ohlc["high"] < ohlc["open"])
        | (ohlc["high"] < ohlc["close"])
        | (ohlc["low"] > ohlc["open"])
        | (ohlc["low"] > ohlc["close"])
    )
    if invalid_geometry.any():
        raise RuntimeError(
            f"{context}_SOURCE_OHLC_GEOMETRY_INVALID: count={int(invalid_geometry.sum())}"
        )
    source_index = pd.DatetimeIndex(src["time"])
    missing_times = sample_times.difference(source_index)
    if len(missing_times):
        raise RuntimeError(
            f"{context}_SOURCE_ROW_GAP: missing={len(missing_times)} first={missing_times[0]}"
        )

    candle_x, candle_names = build_entry_candlestick_pattern_layer(src)
    candle_x = np.asarray(candle_x, dtype=np.float32)
    if candle_x.ndim != 2 or candle_x.shape[0] != len(src) or candle_x.shape[1] != len(candle_names):
        raise RuntimeError(
            f"{context}_OUTPUT_SHAPE_INVALID: shape={candle_x.shape} names={len(candle_names)}"
        )
    if not candle_names:
        raise RuntimeError(f"{context}_OUTPUT_EMPTY")
    if not np.isfinite(candle_x).all():
        raise RuntimeError(f"{context}_OUTPUT_NONFINITE")
    candle_df = pd.DataFrame(candle_x, columns=candle_names, index=source_index)
    aligned = candle_df.loc[sample_times]
    return aligned.to_numpy(dtype=np.float32), list(candle_names)


def build_chart_layer(x: np.ndarray, feature_names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Build the stable chart/foundation/geometry layer from exact sources."""

    x, idx = _require_matrix_contract(
        x,
        feature_names,
        CHART_LAYER_SOURCE_FIELDS,
        context="CHART_LAYER",
    )
    arrays: list[np.ndarray] = []
    names: list[str] = []

    h1_trend = _tanh(_col(x, idx, "ctx_cont._v1h1_ema_diff"))
    h4_trend = _tanh(_col(x, idx, "ctx_cont._v1h4_ema_diff"))
    d1_slope = _tanh(_col(x, idx, "ctx_cont.d1_ema_slope_20_canon_v2"))
    m15_trend = _tanh(_col(x, idx, "ctx_cont.m15_trend_sign_canon_v2"))
    regime_stack = _tanh(_col(x, idx, "ctx_cont.regime_stack_sum_v3"), scale=3.0)
    trend_proxy = _clip(
        0.35 * h1_trend
        + 0.30 * h4_trend
        + 0.20 * d1_slope
        + 0.10 * m15_trend
        + 0.05 * regime_stack
    )
    up = _pos(trend_proxy)
    down = _neg(trend_proxy)
    add_chart_feature(arrays, names, "trend_proxy_h1h4d1", trend_proxy)
    add_chart_feature(arrays, names, "trend_up_pressure", up)
    add_chart_feature(arrays, names, "trend_down_pressure", down)

    near_high = _prox_abs(_col(x, idx, "ctx_cont.dist_last_swing_high_atr"))
    near_low = _prox_abs(_col(x, idx, "ctx_cont.dist_last_swing_low_atr"))
    recent_high = _recency(_col(x, idx, "ctx_cont.bars_since_swing_high"))
    recent_low = _recency(_col(x, idx, "ctx_cont.bars_since_swing_low"))
    high_context = _clip(near_high * (1.0 + recent_high))
    low_context = _clip(near_low * (1.0 + recent_low))
    add_chart_feature(arrays, names, "near_recent_swing_high", high_context)
    add_chart_feature(arrays, names, "near_recent_swing_low", low_context)

    bos_pressure_12 = _col(x, idx, "ctx_cont.smc_bos_pressure_last12")
    bos_pressure_48 = _col(x, idx, "ctx_cont.smc_bos_pressure_last48")
    bos_up = _clip(
        _col(x, idx, "snap.smc_bos_up")
        + 0.5 * _pos(bos_pressure_12)
        + 0.25 * _pos(bos_pressure_48),
        0.0,
        1.0,
    )
    bos_down = _clip(
        _col(x, idx, "snap.smc_bos_down")
        + 0.5 * _neg(bos_pressure_12)
        + 0.25 * _neg(bos_pressure_48),
        0.0,
        1.0,
    )
    bos_pressure = _clip(bos_pressure_48 + bos_pressure_12)
    choch = _clip(
        _col(x, idx, "snap.smc_choch")
        + _col(x, idx, "ctx_cont.smc_choch_recent_tau12")
        + 0.5 * _col(x, idx, "ctx_cont.smc_choch_recent_tau24")
    )
    pullback_h1 = _col(x, idx, "ctx_cont.struct_pullback_depth_h1_v3")
    pullback_h4 = _col(x, idx, "ctx_cont.struct_pullback_depth_h4_v3")
    pullback = _clip(0.6 * pullback_h1 + 0.4 * pullback_h4)
    add_chart_feature(arrays, names, "bos_pressure_combo", bos_pressure)
    add_chart_feature(arrays, names, "choch_recent_combo", choch)
    add_chart_feature(arrays, names, "pullback_depth_h1h4", pullback)

    add_chart_feature(arrays, names, "hh_breakout_proxy", high_context * up * (1.0 + bos_up))
    add_chart_feature(arrays, names, "hl_pullback_proxy", low_context * up * (1.0 + pullback))
    add_chart_feature(arrays, names, "lh_pullback_proxy", high_context * down * (1.0 + pullback))
    add_chart_feature(arrays, names, "ll_breakdown_proxy", low_context * down * (1.0 + bos_down))
    add_chart_feature(arrays, names, "bos_x_choch_instability", bos_pressure * choch)
    add_chart_feature(
        arrays,
        names,
        "bos_x_tf_agreement",
        bos_pressure * _col(x, idx, "ctx_cont.struct_tf_agree_count_v3"),
    )
    add_chart_feature(
        arrays,
        names,
        "choch_x_regime_divergence",
        choch * _col(x, idx, "ctx_cont.regime_divergence_flag_v3"),
    )

    sweep_bull_pressure_12 = _col(
        x, idx, "ctx_cont.smc_sweep_bull_pressure_last12"
    )
    sweep_bull_pressure_48 = _col(
        x, idx, "ctx_cont.smc_sweep_bull_pressure_last48"
    )
    sweep_up = _clip(
        _col(x, idx, "snap.smc_sweep_up")
        + 0.5 * _neg(sweep_bull_pressure_12)
        + 0.25 * _neg(sweep_bull_pressure_48),
        0.0,
        1.0,
    )
    sweep_down = _clip(
        _col(x, idx, "snap.smc_sweep_down")
        + 0.5 * _pos(sweep_bull_pressure_12)
        + 0.25 * _pos(sweep_bull_pressure_48),
        0.0,
        1.0,
    )
    sweep_recent = _clip(
        _recency(_col(x, idx, "snap.smc_bars_since_sweep"))
        + _col(x, idx, "ctx_cont.smc_sweep_recency_tau24")
    )
    sweep_size = _clip(
        _col(x, idx, "snap.smc_sweep_size_atr")
        + _col(x, idx, "ctx_cont.smc_sweep_size_recent_tau12")
    )
    wick_ratio = _clip(
        _col(x, idx, "ctx_cont.wick_ratio") + np.abs(_col(x, idx, "snap.wick_asym"))
    )
    support_prox = _col(x, idx, "ctx_cont.sr_support_proximity_exp")
    resistance_prox = _col(x, idx, "ctx_cont.sr_resistance_proximity_exp")
    add_chart_feature(arrays, names, "sweep_recent_combo", sweep_recent)
    add_chart_feature(arrays, names, "sweep_size_combo", sweep_size)
    add_chart_feature(
        arrays,
        names,
        "false_breakout_high_reject",
        sweep_up * sweep_recent * wick_ratio * resistance_prox * (1.0 + down),
    )
    add_chart_feature(
        arrays,
        names,
        "false_breakout_low_reject",
        sweep_down * sweep_recent * wick_ratio * support_prox * (1.0 + up),
    )
    add_chart_feature(
        arrays, names, "sweep_high_into_resistance", sweep_up * resistance_prox * high_context
    )
    add_chart_feature(arrays, names, "sweep_low_into_support", sweep_down * support_prox * low_context)
    add_chart_feature(arrays, names, "sweep_size_x_wick", sweep_size * wick_ratio)
    add_chart_feature(arrays, names, "sweep_x_choch", sweep_recent * choch)

    h1_compression = _clip(_col(x, idx, "ctx_cont.H1_range_compression_ratio"))
    m15_compression = _clip(_col(x, idx, "ctx_cont.M15_range_compression_ratio"))
    squeeze = _clip(_col(x, idx, "snap._v1_bb_squeeze_20_2"))
    atr_z = _clip(_col(x, idx, "snap.atr_z"))
    rvol = _clip(_col(x, idx, "snap.rvol_20"))
    vol_ratio = _clip(_col(x, idx, "snap.vol_ratio_5_20"))
    d1_range_z = _clip(_col(x, idx, "ctx_cont.d1_range_z_20_canon_v2"))
    compression = _clip(0.45 * h1_compression + 0.35 * m15_compression + 0.20 * squeeze)
    expansion = _clip(compression * (0.45 * atr_z + 0.35 * rvol + 0.20 * vol_ratio))
    add_chart_feature(arrays, names, "compression_h1_m15_bb", compression)
    add_chart_feature(arrays, names, "compression_to_expansion_proxy", expansion)
    add_chart_feature(arrays, names, "compression_x_bos", compression * bos_pressure)
    add_chart_feature(arrays, names, "compression_x_choch", compression * choch)
    add_chart_feature(arrays, names, "d1_range_x_expansion", d1_range_z * expansion)

    retracement = _clip(_col(x, idx, "ctx_cont.retracement_from_last_impulse"))
    trend_age_h1 = _clip(_col(x, idx, "ctx_cont.h1_trend_age_bars_norm_v2"))
    trend_age_h4 = _clip(_col(x, idx, "ctx_cont.h4_trend_age_bars_norm_v2"))
    mature_d1 = _col(x, idx, "ctx_cont.d1_trend_age_mature_flag_v3")
    add_chart_feature(arrays, names, "impulse_pullback_up", retracement * up * (1.0 + trend_age_h1))
    add_chart_feature(arrays, names, "impulse_pullback_down", retracement * down * (1.0 + trend_age_h1))
    add_chart_feature(
        arrays,
        names,
        "mature_trend_pullback_risk",
        pullback * (trend_age_h1 + trend_age_h4 + mature_d1),
    )
    add_chart_feature(arrays, names, "trend_age_x_choch", (trend_age_h1 + trend_age_h4) * choch)

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
    add_chart_feature(arrays, names, "pivot_nearest_proximity", nearest_pivot)
    add_chart_feature(arrays, names, "major_level_proximity_max", level_prox_max)
    add_chart_feature(arrays, names, "major_level_proximity_mean", level_prox_mean)
    add_chart_feature(arrays, names, "wick_x_major_level", wick_ratio * level_prox_max)
    add_chart_feature(arrays, names, "pullback_x_support_resistance", pullback * level_prox_max)
    add_chart_feature(
        arrays,
        names,
        "premium_discount_x_level",
        _col(x, idx, "ctx_cont.sr_support_minus_resistance_prox") * level_prox_max,
    )

    d1_loc = _col(x, idx, "ctx_cont.d1_close_pct_in_20day_range_canon_v2")
    d1_boundary = _prox_abs(_col(x, idx, "ctx_cont.d1_dist_to_boundary_v3"))
    add_chart_feature(arrays, names, "d1_upper_range_pressure", d1_loc * up * high_context)
    add_chart_feature(arrays, names, "d1_lower_range_pressure", (1.0 - d1_loc) * down * low_context)
    add_chart_feature(arrays, names, "d1_boundary_x_sweep", d1_boundary * sweep_recent)
    add_chart_feature(arrays, names, "d1_boundary_x_wick", d1_boundary * wick_ratio)

    session_cols = (
        "ctx_cont.is_ASIA",
        "ctx_cont.is_asia_eu_overlap",
        "ctx_cont.is_eu_us_overlap",
        "ctx_cont.is_eu_only",
        "ctx_cont.is_us_only",
    )
    session_signals = (
        ("trend_proxy", trend_proxy),
        ("bos", bos_pressure),
        ("choch", choch),
        ("sweep_recent", sweep_recent),
        ("compression", compression),
        ("expansion", expansion),
        ("wick_level", wick_ratio * level_prox_max),
        ("pullback", pullback),
        ("d1_loc", d1_loc),
    )
    for session_name in session_cols:
        session = _col(x, idx, session_name)
        short = session_name.removeprefix("ctx_cont.")
        for signal_name, signal in session_signals:
            add_chart_feature(arrays, names, f"{short}_x_{signal_name}", session * signal)

    vol_signals = (
        ("d1_atr_pct", _col(x, idx, "ctx_cont.D1_atr_percentile_252")),
        ("h1_vol_pct", _col(x, idx, "ctx_cont.vol_pct_h1_1yr")),
        ("m5_vol_pct", _col(x, idx, "ctx_cont.vol_pct_m5_1yr")),
        ("atr_ratio_h1_d1", _col(x, idx, "ctx_cont.atr_ratio_h1_d1")),
        ("atr_ratio_m15_d1", _col(x, idx, "ctx_cont.atr_ratio_m15_d1")),
    )
    struct_signals = (
        ("hh", high_context * up),
        ("hl", low_context * up),
        ("lh", high_context * down),
        ("ll", low_context * down),
        ("sweep", sweep_recent),
        ("choch", choch),
        ("bos", bos_pressure),
        ("wick_level", wick_ratio * level_prox_max),
    )
    for vol_name, vol in vol_signals:
        for signal_name, signal in struct_signals:
            add_chart_feature(arrays, names, f"{signal_name}_x_{vol_name}", signal * vol)

    foundation_x, foundation_names = build_entry_foundation_structure_layer(x, feature_names)
    if foundation_x.shape != (x.shape[0], len(foundation_names)) or not foundation_names:
        raise RuntimeError(
            f"CHART_LAYER_FOUNDATION_OUTPUT_INVALID: shape={foundation_x.shape} names={len(foundation_names)}"
        )
    geometry_x, geometry_names = build_entry_chart_geometry_layer(x, feature_names)
    if geometry_x.shape != (x.shape[0], len(geometry_names)) or not geometry_names:
        raise RuntimeError(
            f"CHART_LAYER_GEOMETRY_OUTPUT_INVALID: shape={geometry_x.shape} names={len(geometry_names)}"
        )
    if not np.isfinite(foundation_x).all() or not np.isfinite(geometry_x).all():
        raise RuntimeError("CHART_LAYER_CHILD_OUTPUT_NONFINITE")
    arrays.extend([foundation_x[:, i] for i in range(foundation_x.shape[1])])
    names.extend(foundation_names)
    arrays.extend([geometry_x[:, i] for i in range(geometry_x.shape[1])])
    names.extend(geometry_names)

    out = np.column_stack(arrays).astype(np.float32, copy=False)
    if out.shape[1] != len(names) or len(names) != len(set(names)):
        raise RuntimeError(f"CHART_LAYER_OUTPUT_CONTRACT_INVALID: shape={out.shape} names={len(names)}")
    if not np.isfinite(out).all():
        raise RuntimeError("CHART_LAYER_OUTPUT_NONFINITE")
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


def build_deep_interaction_layer(
    x: np.ndarray,
    feature_names: list[str],
    sample_df: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """Build the stable deep interaction layer from exact chart/base sources."""

    sample_times = _require_sample_times(sample_df, context="DEEP_INTERACTION")
    x, idx = _require_matrix_contract(
        x,
        feature_names,
        DEEP_INTERACTION_SOURCE_FIELDS,
        context="DEEP_INTERACTION",
    )
    if len(sample_times) != x.shape[0]:
        raise RuntimeError(
            f"DEEP_INTERACTION_ROW_MISMATCH: matrix={x.shape[0]} timestamps={len(sample_times)}"
        )
    arrays: list[np.ndarray] = []
    names: list[str] = []

    def c(name: str) -> np.ndarray:
        return _col(x, idx, name)

    ema_fast = _tanh(c("snap._v1_ema_diff"))
    ema_h1 = _tanh(c("ctx_cont._v1h1_ema_diff"))
    ema_h4 = _tanh(c("ctx_cont._v1h4_ema_diff"))
    pos_ema200 = _tanh(c("snap.pos_vs_ema200"))
    ema20_slope = _tanh(c("snap.ema20_slope"))
    d1_slope = _tanh(c("ctx_cont.d1_ema_slope_20_canon_v2"))
    h1_slope = _tanh(c("ctx_cont._v1h1_slope5"))
    trend_stack = _clip(
        0.20 * ema_fast
        + 0.20 * ema_h1
        + 0.20 * ema_h4
        + 0.15 * pos_ema200
        + 0.10 * ema20_slope
        + 0.10 * d1_slope
        + 0.05 * h1_slope
    )
    trend_delta = _delta(trend_stack)
    ema50_200_spread = _tanh(c("chart.m5_ema50_200_spread_bps"), scale=50.0)
    ema50_200_atr = _tanh(c("chart.m5_ema50_200_spread_atr"), scale=2.0)
    ema50_200_bull = c("chart.m5_ema50_200_bull_state")
    ema50_200_bear = 1.0 - ema50_200_bull
    ema50_200_cross = _clip(
        c("chart.m5_ema50_200_cross_up") - c("chart.m5_ema50_200_cross_down")
    )
    price_vs_ema200 = _tanh(c("chart.m5_price_vs_ema200_bps"), scale=80.0)
    ema50_slope = _tanh(c("chart.m5_ema50_slope_bps"), scale=12.0)
    ema200_slope = _tanh(c("chart.m5_ema200_slope_bps"), scale=6.0)
    add_chart_feature(arrays, names, "ema_stack_alignment", trend_stack)
    add_chart_feature(arrays, names, "ema_stack_delta", trend_delta)
    add_chart_feature(arrays, names, "ema_stack_acceleration", _delta(trend_delta))
    add_chart_feature(arrays, names, "ema_stack_cross_up", _cross_up(trend_stack))
    add_chart_feature(arrays, names, "ema_stack_cross_down", _cross_down(trend_stack))
    add_chart_feature(arrays, names, "true_ema50_200_alignment", ema50_200_spread)
    add_chart_feature(arrays, names, "true_ema50_200_atr_alignment", ema50_200_atr)
    add_chart_feature(arrays, names, "true_ema50_200_cross_pressure", ema50_200_cross)
    add_chart_feature(arrays, names, "true_ema50_slope_pressure", ema50_slope)
    add_chart_feature(arrays, names, "true_ema200_slope_pressure", ema200_slope)
    add_chart_feature(arrays, names, "price_vs_true_ema200_pressure", price_vs_ema200)

    for raw_name, short in (
        ("snap._v1_ema_diff", "m5_ema_fast_slow"),
        ("snap.pos_vs_ema200", "m5_pos_ema200"),
        ("ctx_cont._v1h1_ema_diff", "h1_ema_fast_slow"),
        ("ctx_cont._v1h4_ema_diff", "h4_ema_fast_slow"),
        ("ctx_cont.d1_ema_slope_20_canon_v2", "d1_ema_slope"),
    ):
        signal = _tanh(c(raw_name))
        add_chart_feature(arrays, names, f"{short}_cross_up", _cross_up(signal))
        add_chart_feature(arrays, names, f"{short}_cross_down", _cross_down(signal))
        add_chart_feature(arrays, names, f"{short}_delta", _delta(signal))

    compression = _clip(
        0.45 * c("chart.compression_h1_m15_bb")
        + 0.30 * c("ctx_cont.H1_range_compression_ratio")
        + 0.25 * c("ctx_cont.M15_range_compression_ratio")
    )
    expansion = _clip(
        c("chart.compression_to_expansion_proxy")
        + c("snap.rvol_20")
        + c("snap.vol_ratio_5_20")
        + c("snap.atr_z")
    )
    expansion_delta = _delta(expansion)
    add_chart_feature(arrays, names, "expansion_delta", expansion_delta)
    add_chart_feature(arrays, names, "compression_release", compression * _pos(expansion_delta))
    add_chart_feature(
        arrays,
        names,
        "compression_release_downtrend",
        compression * _pos(expansion_delta) * _neg(trend_stack),
    )
    add_chart_feature(
        arrays,
        names,
        "compression_release_uptrend",
        compression * _pos(expansion_delta) * _pos(trend_stack),
    )

    regime_stack = _tanh(c("ctx_cont.regime_stack_sum_v3"), scale=3.0)
    regime_agree = _clip(c("ctx_cont.regime_tf_agreement_v3"))
    regime_div = _clip(c("ctx_cont.regime_divergence_flag_v3"))
    d1_changed = _clip(c("ctx_cont.d1_regime_changed_flag_v3"))
    bars_since_d1_change = _recency(c("ctx_cont.bars_since_d1_regime_change_v3"))
    add_chart_feature(arrays, names, "regime_stack_delta", _delta(regime_stack))
    add_chart_feature(
        arrays,
        names,
        "fresh_d1_regime_change_pressure",
        d1_changed + bars_since_d1_change,
    )
    add_chart_feature(
        arrays,
        names,
        "regime_divergence_x_trend_delta",
        regime_div * np.abs(trend_delta),
    )
    add_chart_feature(
        arrays,
        names,
        "regime_agreement_x_trend_stack",
        regime_agree * trend_stack,
    )
    add_chart_feature(
        arrays,
        names,
        "regime_agreement_x_true_ema50_200",
        regime_agree * ema50_200_spread,
    )
    add_chart_feature(
        arrays,
        names,
        "regime_divergence_x_true_ema_cross",
        regime_div * np.abs(ema50_200_cross),
    )
    add_chart_feature(
        arrays,
        names,
        "fresh_regime_x_true_ema_cross",
        (d1_changed + bars_since_d1_change) * np.abs(ema50_200_cross),
    )

    sweep = _clip(c("chart.sweep_recent_combo") + c("snap.smc_sweep_up") + c("snap.smc_sweep_down"))
    sweep_size = _clip(c("chart.sweep_size_combo") + c("snap.smc_sweep_size_atr"))
    choch = _clip(c("chart.choch_recent_combo") + c("snap.smc_choch"))
    bos = _clip(c("chart.bos_pressure_combo") + c("ctx_cont.smc_bos_pressure_last48"))
    wick_level = _clip(c("chart.wick_x_major_level") + c("ctx_cont.wick_ratio"))
    pullback = _clip(
        c("chart.pullback_depth_h1h4")
        + c("ctx_cont.struct_pullback_depth_h1_v3")
        + c("ctx_cont.struct_pullback_depth_h4_v3")
    )
    hh = _clip(c("chart.hh_breakout_proxy"))
    hl = _clip(c("chart.hl_pullback_proxy"))
    lh = _clip(c("chart.lh_pullback_proxy"))
    ll = _clip(c("chart.ll_breakdown_proxy"))
    d1_loc = _clip(c("ctx_cont.d1_close_pct_in_20day_range_canon_v2"), 0.0, 1.0)
    level_prox = _clip(
        c("chart.major_level_proximity_max")
        + c("ctx_cont.sr_support_proximity_exp")
        + c("ctx_cont.sr_resistance_proximity_exp")
    )
    spread = _clip(c("ctx_cont.spread_bps"))
    session_open = _clip(c("ctx_cont.minutes_since_session_open") / 240.0)
    session_boundary = _recency(c("ctx_cont.minutes_to_next_session_boundary"))
    h1_vol = _clip(c("ctx_cont.vol_pct_h1_1yr"), 0.0, 1.0)
    m5_vol = _clip(c("ctx_cont.vol_pct_m5_1yr"), 0.0, 1.0)
    atr_pct = _clip(c("ctx_cont.D1_atr_percentile_252"), 0.0, 1.0)
    vol_stack = _clip(
        0.35 * h1_vol
        + 0.25 * m5_vol
        + 0.20 * atr_pct
        + 0.10 * c("ctx_cont.atr_ratio_h1_d1")
        + 0.10 * c("ctx_cont.atr_ratio_m15_d1")
    )
    add_chart_feature(arrays, names, "vol_stack", vol_stack)
    add_chart_feature(arrays, names, "vol_stack_delta", _delta(vol_stack))

    tail_pressure = _clip(
        0.22 * sweep_size
        + 0.18 * wick_level
        + 0.16 * choch
        + 0.14 * regime_div
        + 0.12 * np.abs(trend_delta)
        + 0.10 * vol_stack
        + 0.08 * spread
    )
    add_chart_feature(arrays, names, "entry_tail_pressure_combo", tail_pressure)
    add_chart_feature(
        arrays,
        names,
        "tail_pressure_x_session_boundary",
        tail_pressure * session_boundary,
    )
    add_chart_feature(
        arrays,
        names,
        "tail_pressure_x_regime_fresh",
        tail_pressure * (d1_changed + bars_since_d1_change),
    )
    add_chart_feature(
        arrays,
        names,
        "tail_pressure_x_compression_release",
        tail_pressure * compression * _pos(expansion_delta),
    )

    session_cols = (
        ("ctx_cont.is_ASIA", "asia"),
        ("ctx_cont.is_asia_eu_overlap", "asia_eu"),
        ("ctx_cont.is_eu_us_overlap", "eu_us"),
        ("ctx_cont.is_eu_only", "eu"),
        ("ctx_cont.is_us_only", "us"),
    )
    struct_signals = (
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
    )
    context_signals = (
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
    )

    for session_name, session_short in session_cols:
        session = c(session_name)
        for signal_name, signal in struct_signals:
            add_chart_feature(arrays, names, f"{session_short}_x_{signal_name}", session * signal)
        for context_name, context_signal in context_signals:
            add_chart_feature(
                arrays,
                names,
                f"{session_short}_x_{context_name}",
                session * context_signal,
            )

    for signal_name, signal in struct_signals:
        for context_name, context_signal in context_signals:
            # This exact interaction is emitted once above.  The retired
            # research builder appended the same name/value a second time;
            # keeping one canonical column preserves selected-feature values
            # while making the available feature contract unambiguous.
            if signal_name == "tail_pressure" and context_name == "session_boundary":
                continue
            add_chart_feature(arrays, names, f"{signal_name}_x_{context_name}", signal * context_signal)

    add_chart_feature(arrays, names, "long_breakout_tail_risk", hh * sweep * wick_level * vol_stack)
    add_chart_feature(arrays, names, "short_breakdown_tail_risk", ll * sweep * wick_level * vol_stack)
    add_chart_feature(
        arrays,
        names,
        "late_trend_choch_tail_risk",
        choch
        * (c("ctx_cont.h1_trend_age_bars_norm_v2") + c("ctx_cont.h4_trend_age_bars_norm_v2"))
        * vol_stack,
    )
    add_chart_feature(
        arrays,
        names,
        "range_extreme_reversal_risk",
        (d1_loc * hh + (1.0 - d1_loc) * ll) * wick_level * level_prox,
    )
    add_chart_feature(
        arrays,
        names,
        "pullback_quality_trend_agree",
        pullback * regime_agree * np.abs(trend_stack),
    )
    add_chart_feature(
        arrays,
        names,
        "pullback_bad_regime_divergence",
        pullback * regime_div * vol_stack,
    )
    add_chart_feature(
        arrays,
        names,
        "true_ema_bull_pullback_quality",
        ema50_200_bull * pullback * regime_agree,
    )
    add_chart_feature(
        arrays,
        names,
        "true_ema_bear_short_reversal_risk",
        ema50_200_bear * hh * wick_level * level_prox,
    )
    add_chart_feature(
        arrays,
        names,
        "true_ema_cross_liquidity_sweep_risk",
        np.abs(ema50_200_cross) * sweep * wick_level * level_prox,
    )

    out = np.column_stack(arrays).astype(np.float32, copy=False)
    if out.shape[1] != len(names) or len(names) != len(set(names)):
        raise RuntimeError(
            f"DEEP_INTERACTION_OUTPUT_CONTRACT_INVALID: shape={out.shape} names={len(names)}"
        )
    if not np.isfinite(out).all():
        raise RuntimeError("DEEP_INTERACTION_OUTPUT_NONFINITE")
    return out, names
