"""Strict causal feature construction for the model-native Entry signal stack.

This module owns the deterministic chart, price and candlestick layers
consumed by the seq513 dataset builder.  The chart layer dispatches the
registered foundation and chart-geometry child layers; it emits nothing of
its own.  The module deliberately contains no research evaluator, policy,
model or artifact-default coupling.  Every declared source is required and
finite; missing rows or malformed market data are contract failures, never
synthetic zero evidence.
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
    FOUNDATION_STRUCTURE_FEATURE_NAMES,
    FOUNDATION_STRUCTURE_SOURCE_FIELDS,
    build_entry_foundation_structure_layer,
)
from gx1.features.entry_momentum_flow_v1 import MOMENTUM_FLOW_FEATURE_NAMES
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
from gx1.features.htf_features import (
    MULTI_TF_V4_MOMENTUM_EVENT_FEATURES,
    _atr as _htf_atr_v4,
    compute_v29_momentum_event_block_from_ohlc,
)
from gx1.features.level_registry_v1 import (
    LEVEL_REGISTRY_M5_FEATURE_NAMES,
    compute_level_registry_m5_block_v1,
)
from gx1.features.regime_v4_features import (
    REGIME_V4_V29_ADDITION_COLS,
    REGIME_V4_V29_FLIP_TFS,
    compute_regime_v29_flip_frame,
)
from gx1.features.swing_structure_v1 import (
    SWING_V29_ADDITION_NAMES_V1,
    compute_swing_structure_features,
)
from gx1.features.trendline_registry_v1 import (
    TRENDLINE_REGISTRY_FEATURE_NAMES_V1,
    compute_trendline_registry_features_v1,
)


# Exact local-resolution trend-state evidence.  The same formulas run once on
# native M5 for Entry and native M1 for Exit; neither route relabels or copies
# the other route's values.
PRICE_DERIVED_SOURCE_PRICE_FIELD = "close"
PRICE_DERIVED_SOURCE_ATR_FIELD = "atr"

# Leading rows of a source frame on which the price-derived layer is undefined.
# ema200 carries min_periods=200 so its first valid row is index 199; the first
# derivative (ema50_200_spread_delta) moves that to 200 and the second
# (ema50_200_spread_accel) to 201. Sample rows must therefore begin at source
# index 201 or later. Verified against the native M1 surface: index 200 fails
# the layer's own finiteness gate and 201 passes.
PRICE_DERIVED_CAUSAL_WARMUP_ROWS = 201

PRICE_DERIVED_FEATURE_NAMES = (
    "chart.local_ema50_200_spread_bps",
    "chart.local_ema50_200_spread_atr",
    "chart.local_ema50_200_bull_state",
    "chart.local_ema50_200_cross_up",
    "chart.local_ema50_200_cross_down",
    "chart.local_price_vs_ema50_bps",
    "chart.local_price_vs_ema200_bps",
    "chart.local_ema50_slope_bps",
    "chart.local_ema200_slope_bps",
    "chart.local_ema50_200_spread_delta",
    "chart.local_ema50_200_spread_accel",
)


# The price-action mandatory block is the exact candlestick smart3 suffix of
# the candlestick layer.  Derive its start from the block's first feature name
# instead of a bare integer: an insertion before the boundary keeps mandatory
# membership anchored to the marker, and a removed or renamed marker fails
# loudly at import (ValueError) instead of silently re-pointing the mandatory
# set.  The 32-count guard in the smart-family contract enforces the suffix
# identity end-to-end.
CANDLESTICK_SMART3_FIRST_FEATURE_NAME = "candle.pattern_close_pressure_signed"
CANDLESTICK_SMART3_START_INDEX = CANDLESTICK_PATTERN_FEATURE_NAMES.index(
    CANDLESTICK_SMART3_FIRST_FEATURE_NAME
)

# V29 Phase A stage 2 — exact emitted names of the five new mandatory event
# families (docs/V29_EVENT_SURFACE_DESIGN_20260811.md §§1-3, block E kept per
# operator decision).  The name tuples are owned by the producing modules;
# this owner only sequences and (for the trendline block) applies the seq513
# ``chart.`` family prefix declared by the design (§2: ``chart.geomline_*``).
LEVEL_REGISTRY_M5_LAYER_FEATURE_NAMES = tuple(LEVEL_REGISTRY_M5_FEATURE_NAMES)
TRENDLINE_REGISTRY_M5_LAYER_FEATURE_NAMES = tuple(
    f"chart.{name}" for name in TRENDLINE_REGISTRY_FEATURE_NAMES_V1
)
SWING_EVENT_LAYER_FEATURE_NAMES = tuple(SWING_V29_ADDITION_NAMES_V1)
MOMENTUM_EVENT_M5_LAYER_FEATURE_NAMES = tuple(
    MULTI_TF_V4_MOMENTUM_EVENT_FEATURES
)
REGIME_FLIP_EVENT_LAYER_FEATURE_NAMES = tuple(REGIME_V4_V29_ADDITION_COLS)

# The two registry layers carry TRAIN-fitted constants with no legitimate
# default (rule 2a); the inline extension fails closed when either family is
# requested without this explicit payload.
V29_REGISTRY_LAYER_PARAM_KEYS = (
    "level_tol_atr",
    "trendline_band_atr",
    "trendline_seq_len",
)

# Ordered ownership registry for every generated specialist layer that the
# canonical seq513 builder may materialize.  This belongs beside the builders,
# not in a report/materializer script with mutable historical artifact paths.
# The five V29 event families are appended AFTER the pre-V29 families so the
# existing mandatory prefix order is byte-stable.
MODEL_NATIVE_SPECIALIST_LAYER_FEATURES: tuple[
    tuple[str, tuple[str, ...]], ...
] = (
    ("foundation_cross_family_layer", FOUNDATION_STRUCTURE_FEATURE_NAMES),
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
        CANDLESTICK_PATTERN_FEATURE_NAMES[CANDLESTICK_SMART3_START_INDEX:],
    ),
    ("support_resistance_memory_layer", SUPPORT_RESISTANCE_MEMORY_FEATURE_NAMES),
    ("price_ema50_200_layer", PRICE_DERIVED_FEATURE_NAMES),
    ("level_registry_m5_layer", LEVEL_REGISTRY_M5_LAYER_FEATURE_NAMES),
    (
        "trendline_registry_m5_layer",
        TRENDLINE_REGISTRY_M5_LAYER_FEATURE_NAMES,
    ),
    ("swing_structure_event_layer", SWING_EVENT_LAYER_FEATURE_NAMES),
    ("momentum_event_m5_layer", MOMENTUM_EVENT_M5_LAYER_FEATURE_NAMES),
    ("regime_flip_event_layer", REGIME_FLIP_EVENT_LAYER_FEATURE_NAMES),
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
# Both counts are DERIVED from the declared registry (rule 13: a repeated
# literal in a consumer is not ownership proof).  V29 Phase A stage 2 grew
# the pre-V29 11-family/346-field registry by the five event families above.
MODEL_NATIVE_MANDATORY_FAMILY_COUNT = len(MODEL_NATIVE_MANDATORY_FAMILY_FEATURES)
MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT = len(
    MODEL_NATIVE_MANDATORY_SELECTED_FIELDS
)

_mandatory_family_labels = tuple(
    family for family, _features in MODEL_NATIVE_MANDATORY_FAMILY_FEATURES
)
if len(set(_mandatory_family_labels)) != len(_mandatory_family_labels):
    raise RuntimeError("MODEL_NATIVE_MANDATORY_FAMILY_LABEL_DUPLICATE")
if any(not family or not features for family, features in MODEL_NATIVE_MANDATORY_FAMILY_FEATURES):
    raise RuntimeError("MODEL_NATIVE_MANDATORY_FAMILY_EMPTY")
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


# The chart layer is a pure dispatcher: its required sources are exactly the
# union of its two registered child layers' declared sources.  The children
# receive their candle inputs through the separately materialized candlestick
# layer, never through this base matrix.
CHART_LAYER_SOURCE_FIELDS = _ordered_unique(
    (
        *(
            name
            for name in FOUNDATION_STRUCTURE_SOURCE_FIELDS
            if not name.startswith("candle.")
        ),
        *(
            name
            for name in CHART_GEOMETRY_SOURCE_FIELDS
            if not name.startswith("candle.")
        ),
    )
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


def _clip(arr: np.ndarray, lo: float = -25.0, hi: float = 25.0) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float32)
    if not np.isfinite(values).all():
        raise RuntimeError("MODEL_NATIVE_GENERATED_FEATURE_NONFINITE")
    return np.clip(values, lo, hi).astype(np.float32, copy=False)


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
    """Build the exact past-only local-resolution EMA layer."""

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

    spread_bps = spread / denom * 1e4
    spread_atr = spread / atr
    price_vs_ema50 = (close - ema50) / denom * 1e4
    price_vs_ema200 = (close - ema200) / denom * 1e4
    ema50_slope = ema50.diff() / denom * 1e4
    ema200_slope = ema200.diff() / denom * 1e4

    def causal_delta(values: pd.Series) -> pd.Series:
        return values.diff()

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
    if not np.isfinite(aligned.to_numpy(dtype=np.float64)).all():
        raise RuntimeError(
            f"{context}_LOCAL_EMA_WARMUP_INCOMPLETE: "
            "sample rows must start after the exact causal EMA/derivative warmup"
        )
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
            f"local_{column}",
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


def _read_v29_price_source(
    source_parquet: Path,
    *,
    context: str,
    columns: tuple[str, ...] = ("time", "high", "low", "close"),
) -> pd.DataFrame:
    """Read and normalize the exact causal source columns for one V29 layer."""

    source = Path(source_parquet).expanduser().resolve()
    available = _read_source_schema(source, context=context)
    missing = [name for name in columns if name not in available]
    if missing:
        raise RuntimeError(f"{context}_SOURCE_FIELDS_MISSING: {missing}")
    src = pd.read_parquet(source, columns=list(columns), engine="pyarrow")
    return _normalize_source_times(src, context=context)


def _align_v29_layer_frame(
    raw: pd.DataFrame,
    sample_times: pd.DatetimeIndex,
    expected_names: tuple[str, ...],
    *,
    context: str,
) -> tuple[np.ndarray, list[str]]:
    """Align one full-history V29 layer frame to the exact sample rows.

    The layer is computed over the complete causal source history and then
    row-aligned (the ``build_price_derived_layer`` pattern), so bounded-chunk
    processing is exact by construction.  Any non-finite value at a sample
    row is a hard failure: the sample window must start after the layer's
    honest warmup prefix (rule 2e — no sentinel substitution).
    """

    if tuple(raw.columns) != tuple(expected_names):
        raise RuntimeError(f"{context}_FEATURE_ORDER_INVALID")
    missing_times = sample_times.difference(raw.index)
    if len(missing_times):
        raise RuntimeError(
            f"{context}_SOURCE_ROW_GAP: missing={len(missing_times)} "
            f"first={missing_times[0]}"
        )
    aligned = raw.loc[sample_times]
    values = aligned.to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise RuntimeError(
            f"{context}_WARMUP_INCOMPLETE: sample rows must start after the "
            "layer's causal warmup prefix"
        )
    return values, list(expected_names)


def v29_layer_first_complete_time(raw: pd.DataFrame, *, context: str):
    """First row at which EVERY column of a full-history V29 layer is finite.

    A V29 layer's honest warmup must be one chronological prefix (the registry
    emission contract); the surface materializer measures this floor on the
    exact declared source bytes and excludes the leading rows it cannot
    honestly produce (the established leading-exclusion doctrine). Any
    non-finite value after the first complete row is a computation defect,
    not warmup, and fails closed.
    """

    values = raw.to_numpy(dtype=np.float64)
    finite = np.isfinite(values).all(axis=1)
    if not bool(finite.any()):
        raise RuntimeError(f"{context}_NEVER_COMPLETE: no fully finite row")
    first = int(np.argmax(finite))
    if not bool(finite[first:].all()):
        raise RuntimeError(
            f"{context}_WARMUP_NOT_PREFIX: non-finite rows after the first "
            "complete row"
        )
    return raw.index[first]


def align_v29_layer_frame(
    raw: pd.DataFrame,
    sample_df: pd.DataFrame,
    expected_names,
    *,
    context: str,
) -> tuple[np.ndarray, list[str]]:
    """Row-align a raw full-history V29 layer frame to the sample rows."""

    sample_times = _require_sample_times(sample_df, context=context)
    return _align_v29_layer_frame(
        raw,
        sample_times,
        tuple(expected_names),
        context=context,
    )


def build_level_registry_m5_layer(
    sample_df: pd.DataFrame,
    source_parquet: Path,
    *,
    tol_level_atr: float,
    raw_frame: bool = False,
) -> tuple[np.ndarray, list[str]] | tuple[pd.DataFrame, list[str]]:
    """Entry-M5/513-lane level-registry block (design doc §1.2, 22 fields).

    ``tol_level_atr`` is the TRAIN-fitted frozen M5 cluster tolerance
    (``fit_level_registry_tolerance``); it has no default (rule 2a).
    ``raw_frame=True`` returns the unaligned full-history frame so the
    caller can measure the layer's warmup floor before choosing sample rows.
    """

    context = "LEVEL_REGISTRY_M5_LAYER"
    sample_times = (
        None if raw_frame else _require_sample_times(sample_df, context=context)
    )
    src = _read_v29_price_source(source_parquet, context=context)
    source_index = pd.DatetimeIndex(src["time"])
    registry_source = pd.DataFrame(
        {
            "high": _require_finite_positive_column(src, "high", context=context),
            "low": _require_finite_positive_column(src, "low", context=context),
            "close": _require_finite_positive_column(src, "close", context=context),
        },
        index=source_index,
    )
    registry_source["atr"] = _htf_atr_v4(
        registry_source["high"], registry_source["low"], registry_source["close"], 14
    )
    matrix, names = compute_level_registry_m5_block_v1(
        registry_source,
        tol_level_atr=tol_level_atr,
    )
    if tuple(names) != LEVEL_REGISTRY_M5_LAYER_FEATURE_NAMES:
        raise RuntimeError(f"{context}_FEATURE_ORDER_INVALID")
    raw = pd.DataFrame(
        np.asarray(matrix, dtype=np.float64),
        index=source_index,
        columns=list(names),
    )
    if raw_frame:
        return raw, list(LEVEL_REGISTRY_M5_LAYER_FEATURE_NAMES)
    return _align_v29_layer_frame(
        raw,
        sample_times,
        LEVEL_REGISTRY_M5_LAYER_FEATURE_NAMES,
        context=context,
    )


def build_trendline_registry_m5_layer(
    sample_df: pd.DataFrame,
    source_parquet: Path,
    *,
    band_atr: float,
    seq_len: int,
    raw_frame: bool = False,
) -> tuple[np.ndarray, list[str]] | tuple[pd.DataFrame, list[str]]:
    """Entry-M5/513-lane trendline/channel block (design doc §2/§4.1 block E).

    ``band_atr`` is the TRAIN-fitted frozen M5 band for the Entry candidate
    window ``seq_len`` (the Entry model sequence length — an explicit recipe
    input); neither has a default (rule 2a). ``raw_frame=True`` returns the
    unaligned full-history frame for warmup-floor measurement.
    """

    context = "TRENDLINE_REGISTRY_M5_LAYER"
    sample_times = (
        None if raw_frame else _require_sample_times(sample_df, context=context)
    )
    src = _read_v29_price_source(source_parquet, context=context)
    source_index = pd.DatetimeIndex(src["time"])
    registry_source = pd.DataFrame(
        {
            "high": _require_finite_positive_column(src, "high", context=context),
            "low": _require_finite_positive_column(src, "low", context=context),
            "close": _require_finite_positive_column(src, "close", context=context),
        },
        index=source_index,
    )
    registry_source["atr"] = _htf_atr_v4(
        registry_source["high"], registry_source["low"], registry_source["close"], 14
    )
    frame, _state = compute_trendline_registry_features_v1(
        registry_source,
        timeframe="M5",
        seq_len=seq_len,
        band_atr=band_atr,
    )
    if tuple(frame.columns) != tuple(TRENDLINE_REGISTRY_FEATURE_NAMES_V1):
        raise RuntimeError(f"{context}_FEATURE_ORDER_INVALID")
    raw = frame.astype(np.float64)
    raw.columns = list(TRENDLINE_REGISTRY_M5_LAYER_FEATURE_NAMES)
    if raw_frame:
        return raw, list(TRENDLINE_REGISTRY_M5_LAYER_FEATURE_NAMES)
    return _align_v29_layer_frame(
        raw,
        sample_times,
        TRENDLINE_REGISTRY_M5_LAYER_FEATURE_NAMES,
        context=context,
    )


def build_swing_event_m5_layer(
    sample_df: pd.DataFrame,
    source_parquet: Path,
    *,
    raw_frame: bool = False,
) -> tuple[np.ndarray, list[str]] | tuple[pd.DataFrame, list[str]]:
    """Entry-M5/513-lane structure_swing G1/G2/G4 event block (9 fields).

    One formula owner: ``swing_structure_v1.compute_swing_structure_features``
    with ``include_v29_additions=True`` on the complete causal source history.
    ``raw_frame=True`` returns the unaligned full-history frame for
    warmup-floor measurement.
    """

    context = "SWING_EVENT_M5_LAYER"
    sample_times = (
        None if raw_frame else _require_sample_times(sample_df, context=context)
    )
    src = _read_v29_price_source(source_parquet, context=context)
    source_index = pd.DatetimeIndex(src["time"])
    computed = compute_swing_structure_features(
        _require_finite_positive_column(src, "high", context=context),
        _require_finite_positive_column(src, "low", context=context),
        _require_finite_positive_column(src, "close", context=context),
        include_v29_additions=True,
    )
    raw = pd.DataFrame(
        {
            name: np.asarray(computed[name], dtype=np.float64)
            for name in SWING_EVENT_LAYER_FEATURE_NAMES
        },
        index=source_index,
    )
    if raw_frame:
        return raw, list(SWING_EVENT_LAYER_FEATURE_NAMES)
    return _align_v29_layer_frame(
        raw,
        sample_times,
        SWING_EVENT_LAYER_FEATURE_NAMES,
        context=context,
    )


def build_momentum_event_m5_layer(
    sample_df: pd.DataFrame,
    source_parquet: Path,
    *,
    raw_frame: bool = False,
) -> tuple[np.ndarray, list[str]] | tuple[pd.DataFrame, list[str]]:
    """Entry-M5/513-lane momentum G1/G2 event block (design §4.1 block E).

    One formula owner:
    ``htf_features.compute_v29_momentum_event_block_from_ohlc`` — the same
    function backing the per-TF V4 lane, run here on the entry M5 clock.
    ``raw_frame=True`` returns the unaligned full-history frame for
    warmup-floor measurement.
    """

    context = "MOMENTUM_EVENT_M5_LAYER"
    sample_times = (
        None if raw_frame else _require_sample_times(sample_df, context=context)
    )
    src = _read_v29_price_source(source_parquet, context=context)
    source_index = pd.DatetimeIndex(src["time"])
    ohlc = pd.DataFrame(
        {
            "high": _require_finite_positive_column(src, "high", context=context),
            "low": _require_finite_positive_column(src, "low", context=context),
            "close": _require_finite_positive_column(src, "close", context=context),
        },
        index=source_index,
    )
    raw = compute_v29_momentum_event_block_from_ohlc(ohlc).astype(np.float64)
    if tuple(raw.columns) != MOMENTUM_EVENT_M5_LAYER_FEATURE_NAMES:
        raise RuntimeError(f"{context}_FEATURE_ORDER_INVALID")
    if raw_frame:
        return raw, list(MOMENTUM_EVENT_M5_LAYER_FEATURE_NAMES)
    return _align_v29_layer_frame(
        raw,
        sample_times,
        MOMENTUM_EVENT_M5_LAYER_FEATURE_NAMES,
        context=context,
    )


def build_regime_flip_event_layer(
    sample_df: pd.DataFrame,
    source_parquet: Path,
    *,
    raw_frame: bool = False,
) -> tuple[np.ndarray, list[str]] | tuple[pd.DataFrame, list[str]]:
    """Entry-M5/513-lane session_regime G2 per-TF flip block (8 fields).

    One formula owner: ``regime_v4_features.compute_regime_v29_flip_frame``
    on the exact ``{tf}_regime_class_id_v2`` columns of the complete causal
    source history (base M5 clock). ``raw_frame=True`` returns the unaligned
    full-history frame for warmup-floor measurement (the flip-age fields are
    honestly NaN until each timeframe's first observed flip — a
    data-dependent warmup no fixed row constant can bound).
    """

    context = "REGIME_FLIP_EVENT_LAYER"
    sample_times = (
        None if raw_frame else _require_sample_times(sample_df, context=context)
    )
    class_columns = tuple(
        f"{tf}_regime_class_id_v2" for tf in REGIME_V4_V29_FLIP_TFS
    )
    src = _read_v29_price_source(
        source_parquet,
        context=context,
        columns=("time", *class_columns),
    )
    source_index = pd.DatetimeIndex(src["time"])
    class_frame = src.drop(columns=["time"])
    class_frame.index = source_index
    raw = compute_regime_v29_flip_frame(class_frame).astype(np.float64)
    if raw_frame:
        return raw, list(REGIME_FLIP_EVENT_LAYER_FEATURE_NAMES)
    return _align_v29_layer_frame(
        raw,
        sample_times,
        REGIME_FLIP_EVENT_LAYER_FEATURE_NAMES,
        context=context,
    )


def build_chart_layer(x: np.ndarray, feature_names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Dispatch the registered foundation and chart-geometry child layers.

    The retired chart-core interaction emissions were registered in no
    feature-name constant, discoverable by no ranker and consumed by no
    specialist layer; the two registered children below are the only chart
    outputs that can reach the seq513 signal.  Both children read exclusively
    from the base matrix, so their outputs are unchanged by the removal.
    """

    x, _idx = _require_matrix_contract(
        x,
        feature_names,
        CHART_LAYER_SOURCE_FIELDS,
        context="CHART_LAYER",
    )
    arrays: list[np.ndarray] = []
    names: list[str] = []

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

