"""Strict causal feature construction for the model-native Entry signal stack.

This module owns deterministic foundation, price and raw candle layers
consumed by the model-native dataset builder.  It deliberately contains no
research evaluator, policy, hand-written pattern score or artifact default.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from gx1.features.entry_candle_primitives_v1 import (
    CANDLE_PRIMITIVE_FEATURE_NAMES,
    CANDLE_PRIMITIVE_SOURCE_FIELDS,
    build_entry_candle_primitive_layer,
)
from gx1.features.entry_foundation_structure_v1 import (
    FOUNDATION_STRUCTURE_FEATURE_NAMES,
    FOUNDATION_STRUCTURE_SOURCE_FIELDS,
    build_entry_foundation_structure_layer,
)
from gx1.features.htf_features import (
    LOCAL_MOMENTUM_V30_PRIMITIVE_FEATURES,
    MULTI_TF_V4_MOMENTUM_EVENT_FEATURES,
    _atr as _htf_atr_v4,
    _cross_down_event as _htf_cross_down_event_v4,
    _cross_up_event as _htf_cross_up_event_v4,
    compute_v29_momentum_event_block_from_ohlc,
)
from gx1.features.event_age_v1 import raw_state_age_bars
from gx1.features.level_registry_v1 import (
    LEVEL_REGISTRY_M5_FEATURE_NAMES,
    compute_level_registry_m5_block_v1,
)
from gx1.features.smc_v1 import (
    SMC_V30_ADDITION_NAMES_V1,
    compute_smc_features,
)
from gx1.features.swing_structure_v1 import (
    SWING_V29_ADDITION_NAMES_V1,
    compute_swing_structure_features,
)
from gx1.features.technical_indicators_v1 import (
    ema50_200_spread_atr_block,
    technical_indicator_contract_metadata,
)
from gx1.features.trendline_registry_v1 import (
    TRENDLINE_REGISTRY_FEATURE_NAMES_V1,
    compute_trendline_registry_features_v1,
)
from gx1.features.volatility_squeeze_state_v1 import (
    VOLATILITY_SQUEEZE_FEATURE_NAMES,
    VolatilitySqueezeArtifactSet,
    compute_volatility_squeeze_state,
    require_volatility_squeeze_artifact_set,
)


# Exact local-resolution trend-state evidence.  The same formulas run once on
# native M5 for Entry and native M1 for Exit; neither route relabels or copies
# the other route's values.
PRICE_DERIVED_SOURCE_PRICE_FIELD = "close"
PRICE_DERIVED_SOURCE_OHLC_FIELDS = ("high", "low", "close")

# Leading rows of a source frame on which the price-derived layer is undefined.
# classic EMA200 seeds from 200 closes so its first valid row is index 199; the first
# derivative (ema50_200_spread_delta) moves that to 200 and the second
# (ema50_200_spread_accel) to 201. Sample rows must therefore begin at source
# index 201 or later. Verified against the native M1 surface: index 200 fails
# the layer's own finiteness gate and 201 passes.  The V30
# local_kama_efficiency_30 addition needs only 30 rows (window 30), and the V30
# GAP-2/3 age fields inherit their EMA source's first valid row (index 199 for
# the ema200-backed pair, 49 for the ema50 side), so this floor is unchanged —
# re-verified 2026-08-13 on the full 15-column layer: index 200 still fails the
# layer's own finiteness gate and 201 still passes.  The V30 package-3
# price-vs-EMA cross events add one shift(1) on top of their EMA source (first
# finite row: index 200 for the ema200 pair, 50 for the ema50 pair), still
# inside 201 — re-verified 2026-08-13 on the full 19-column layer: index 200
# still fails the layer's own finiteness gate and 201 still passes.
PRICE_DERIVED_CAUSAL_WARMUP_ROWS = 201

# V30 (2026-08-13): ``local_kama_efficiency_30`` is the Kaufman efficiency
# ratio ER = |close[t] - close[t-30]| / sum_{i=t-29..t} |close[i] - close[i-1]|
# that basic_v1.kama_np already computes internally (window 30 — the window
# `_v1_kama30_change_5_atr`'s `_kama(close, 30)` owns) and discards.  It is emitted
# HERE, not as a new BASE field: MODEL_NATIVE_BASE_FIELDS is the
# accepted contract's code-owned base tuple (bound into
# MODEL_NATIVE_STATIC_CONTRACT_SHA256),
# while V29/V30 additions live in the mandatory causal layers.
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
    "chart.local_kama_efficiency_30",
    # V30 Phase-A completion (2026-08-13): trend_ema GAP-2/GAP-3 on the LOCAL
    # clock.  The per-TF lane carries the matching raw EMA-state durations,
    # while
    # this layer emitted the crosses and the state as 1-bar spikes with no
    # duration — so Entry saw the spike on M5 and only the aged version on
    # M15+ (review C.1).  Values come from the SAME two htf_features helpers
    # that produces the per-TF fields: one raw uncapped native-clock owner.
    "chart.local_ema50_200_state_age_bars",
    "chart.local_price_vs_ema50_state_age_bars",
    "chart.local_price_vs_ema200_state_age_bars",
    # V30 package 3 (2026-08-13): the recorded Phase-A remainder of trend_ema
    # GAP-3.  Package 2 landed the three age fields and left the four
    # price-vs-EMA cross EVENTS open (see the package-2 message: "trend_ema
    # GAP-2/3 is 3 of 7 (four M5-local cross events still open)").  The per-TF
    # lane has emitted ``price_x_ema{50,200}_cross_{up,down}`` since the V29
    # stage-2 wiring (htf_features.MULTI_TF_V4_TREND_EMA_EVENT_FEATURES), so
    # without these the local M5/M1 clock carried the ema50/200 cross but not
    # the price-through-EMA cross that the higher timeframes already had.
    # Values come from the SAME two htf_features helpers that produce the
    # per-TF fields (`_cross_up_event` / `_cross_down_event`, imported above):
    # one formula owner, imported not duplicated.
    "chart.local_price_x_ema50_cross_up",
    "chart.local_price_x_ema50_cross_down",
    "chart.local_price_x_ema200_cross_up",
    "chart.local_price_x_ema200_cross_down",
)
PRICE_DERIVED_FORMULA_SCHEMA_VERSION = "entry_local_price_raw_primitives_v2"
PRICE_DERIVED_FORMULA_CONTRACT = (
    "ema50_200=shared_classic_sma_seeded_technical_owner",
    "spread_atr=raw_spread_over_positive_wilder_atr_no_clip",
    "price_and_slope_bps=raw_over_positive_closed_bar_close_no_clip",
    "kama_efficiency30=exact_change_over_positive_realized_path_else_unavailable",
    "events=closed_bar_cross_edges_and_causal_age_from_shared_owner",
    "warmup=causal_nan_prefix_then_exact_sample_alignment",
)
PRICE_DERIVED_FORMULA_SHA256 = hashlib.sha256(
    "\n".join(PRICE_DERIVED_FORMULA_CONTRACT).encode("utf-8")
).hexdigest()
PRICE_DERIVED_FEATURE_NAMES_SHA256 = hashlib.sha256(
    "\n".join(PRICE_DERIVED_FEATURE_NAMES).encode("utf-8")
).hexdigest()


def price_derived_contract_metadata() -> dict[str, object]:
    """Immutable local-price formula identity for signal artifacts."""

    return {
        "owner": "gx1.features.entry_model_native_feature_layers_v1",
        "schema_version": PRICE_DERIVED_FORMULA_SCHEMA_VERSION,
        "formula_sha256": PRICE_DERIVED_FORMULA_SHA256,
        "ordered_feature_names_sha256": PRICE_DERIVED_FEATURE_NAMES_SHA256,
        "ordered_feature_names": list(PRICE_DERIVED_FEATURE_NAMES),
        "technical_indicator_owner": technical_indicator_contract_metadata(),
    }


# Every raw candle primitive is mandatory.  There is no pattern suffix and no
# TRAIN-ranked hand-written score: the temporal candle specialist learns the
# multi-bar pattern from the exact local sequence.
CANDLE_PRIMITIVE_MANDATORY_FEATURE_NAMES = tuple(
    CANDLE_PRIMITIVE_FEATURE_NAMES
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
) + tuple(
    LOCAL_MOMENTUM_V30_PRIMITIVE_FEATURES
)
SMC_LOCAL_EVENT_LAYER_FEATURE_NAMES = tuple(SMC_V30_ADDITION_NAMES_V1)
VOLATILITY_SQUEEZE_LOCAL_LAYER_FEATURE_NAMES = tuple(
    VOLATILITY_SQUEEZE_FEATURE_NAMES
)

# Raw, independently clocked trend evidence needed by the structural
# auxiliary labels. These are existing ctx fields, not a generated score:
# the inline builder sees them in ``all_names`` and therefore never computes a
# second copy. Keeping the exact sources mandatory prevents TRAIN ranking from
# changing label semantics between rebuilds.
RAW_MTF_TREND_LAYER_FEATURE_NAMES = (
    "ctx_cont.m15_ema5_20_spread_atr_canon_v2",
    "ctx_cont._v1h1_ema_diff",
    "ctx_cont._v1h4_ema_diff",
    "ctx_cont.d1_ema_slope_20_canon_v2",
)

# The two registry layers carry TRAIN-fitted constants with no legitimate
# default (rule 2a); the inline extension fails closed when either family is
# requested without this explicit payload.
V29_REGISTRY_LAYER_PARAM_KEYS = (
    "level_recurrence_threshold_atr",
    "level_expiry_bars",
    "trendline_band_atr",
    "trendline_expiry_bars",
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
    ("raw_mtf_trend_layer", RAW_MTF_TREND_LAYER_FEATURE_NAMES),
    (
        "price_action_candle_raw_layer",
        CANDLE_PRIMITIVE_MANDATORY_FEATURE_NAMES,
    ),
    ("price_ema50_200_layer", PRICE_DERIVED_FEATURE_NAMES),
    ("level_registry_m5_layer", LEVEL_REGISTRY_M5_LAYER_FEATURE_NAMES),
    (
        "trendline_registry_m5_layer",
        TRENDLINE_REGISTRY_M5_LAYER_FEATURE_NAMES,
    ),
    ("swing_structure_event_layer", SWING_EVENT_LAYER_FEATURE_NAMES),
    ("momentum_event_m5_layer", MOMENTUM_EVENT_M5_LAYER_FEATURE_NAMES),
    ("smc_local_event_layer", SMC_LOCAL_EVENT_LAYER_FEATURE_NAMES),
    (
        "volatility_squeeze_local_layer",
        VOLATILITY_SQUEEZE_LOCAL_LAYER_FEATURE_NAMES,
    ),
)

# This is the code-owned full-stack retention contract.  A feature-selection
# artifact may rank additional evidence, but it may never rank away one of
# these registered causal layer outputs.  Keep the flattened order stable: it
# becomes part of the immutable model-native signal identity.
MODEL_NATIVE_MANDATORY_FAMILY_FEATURES = MODEL_NATIVE_SPECIALIST_LAYER_FEATURES
MODEL_NATIVE_MANDATORY_SELECTED_FIELDS = tuple(
    feature
    for _family, features in MODEL_NATIVE_MANDATORY_FAMILY_FEATURES
    for feature in features
)
# Both counts are DERIVED from the declared registry (rule 13: a repeated
# literal in a consumer is not ownership proof).  V29 Phase A stage 2 grew
# the pre-V29 registry by the event families above.
MODEL_NATIVE_MANDATORY_FAMILY_COUNT = len(MODEL_NATIVE_MANDATORY_FAMILY_FEATURES)
MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT = len(
    MODEL_NATIVE_MANDATORY_SELECTED_FIELDS
)

# The FULL emitted surface of every registered family, in the same label order.
#
# All remaining specialist layers emit exactly their mandatory raw/event
# surface.  The retired chart and candlestick scorebooks had separate
# candidate-only emissions; no such hidden surface remains.
_SPECIALIST_LAYER_FULL_EMISSION_OVERRIDES: dict[str, tuple[str, ...]] = {}
MODEL_NATIVE_SPECIALIST_LAYER_EMITTED_FEATURES: tuple[
    tuple[str, tuple[str, ...]], ...
] = tuple(
    (label, _SPECIALIST_LAYER_FULL_EMISSION_OVERRIDES.get(label, tuple(features)))
    for label, features in MODEL_NATIVE_SPECIALIST_LAYER_FEATURES
)
_emitted_labels = {label for label, _features in MODEL_NATIVE_SPECIALIST_LAYER_FEATURES}
if set(_SPECIALIST_LAYER_FULL_EMISSION_OVERRIDES) - _emitted_labels:
    raise RuntimeError(
        "MODEL_NATIVE_SPECIALIST_LAYER_EMISSION_OVERRIDE_UNKNOWN_LABEL: "
        f"{sorted(set(_SPECIALIST_LAYER_FULL_EMISSION_OVERRIDES) - _emitted_labels)}"
    )
for (
    (_mandatory_label, _mandatory_features),
    (_emitted_label, _emitted_features),
) in zip(
    MODEL_NATIVE_SPECIALIST_LAYER_FEATURES,
    MODEL_NATIVE_SPECIALIST_LAYER_EMITTED_FEATURES,
):
    if _mandatory_label != _emitted_label:
        raise RuntimeError(
            "MODEL_NATIVE_SPECIALIST_LAYER_EMISSION_ORDER_MISMATCH: "
            f"{_mandatory_label} != {_emitted_label}"
        )
    if not set(_mandatory_features).issubset(set(_emitted_features)):
        raise RuntimeError(
            "MODEL_NATIVE_SPECIALIST_LAYER_MANDATORY_NOT_EMITTED: "
            f"{_mandatory_label} "
            f"{sorted(set(_mandatory_features) - set(_emitted_features))[:10]}"
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


# The chart dispatcher now contains only exact foundation event ages and
# their observed-event masks.  The
# retired chart scorebook has no replacement layer: genuine level/trendline
# registry objects are already separate mandatory owners.
CHART_LAYER_SOURCE_FIELDS = _ordered_unique(
    name
    for name in FOUNDATION_STRUCTURE_SOURCE_FIELDS
    if not name.startswith("candle.")
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


def add_chart_feature(
    arrays: list[np.ndarray],
    names: list[str],
    name: str,
    arr: np.ndarray,
) -> None:
    clean = np.asarray(arr, dtype=np.float32)
    if not np.isfinite(clean).all():
        raise RuntimeError("MODEL_NATIVE_GENERATED_FEATURE_NONFINITE")
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
    if price_field not in available:
        raise RuntimeError(f"{context}_SOURCE_PRICE_MISSING: required={price_field}")
    missing_ohlc = [
        field for field in PRICE_DERIVED_SOURCE_OHLC_FIELDS if field not in available
    ]
    if missing_ohlc:
        raise RuntimeError(f"{context}_SOURCE_OHLC_MISSING: {missing_ohlc}")
    src = pd.read_parquet(
        source,
        columns=["time", *PRICE_DERIVED_SOURCE_OHLC_FIELDS],
        engine="pyarrow",
    )
    src = _normalize_source_times(src, context=context)
    ohlc_values = {
        field: _require_finite_positive_column(src, field, context=context)
        for field in PRICE_DERIVED_SOURCE_OHLC_FIELDS
    }

    source_index = pd.DatetimeIndex(src["time"])
    missing_times = sample_times.difference(source_index)
    if len(missing_times):
        raise RuntimeError(
            f"{context}_SOURCE_ROW_GAP: missing={len(missing_times)} first={missing_times[0]}"
        )

    high = pd.Series(ohlc_values["high"], index=source_index, dtype=np.float64)
    low = pd.Series(ohlc_values["low"], index=source_index, dtype=np.float64)
    close = pd.Series(ohlc_values[price_field], index=source_index, dtype=np.float64)
    ema_spread_block = ema50_200_spread_atr_block(high, low, close)
    ema50 = ema_spread_block["ema50"]
    ema200 = ema_spread_block["ema200"]
    spread = ema_spread_block["spread"]
    denom = close.abs()

    spread_bps = spread / denom * 1e4
    spread_atr = ema_spread_block["spread_atr"]
    price_vs_ema50 = (close - ema50) / denom * 1e4
    price_vs_ema200 = (close - ema200) / denom * 1e4
    ema50_slope = ema50.diff() / denom * 1e4
    ema200_slope = ema200.diff() / denom * 1e4

    def causal_delta(values: pd.Series) -> pd.Series:
        return values.diff()

    spread_delta = causal_delta(spread_bps)
    spread_accel = causal_delta(spread_delta)

    # V30 Kaufman efficiency ratio, window 30 (see the name-tuple comment):
    # the exact ER of basic_v1.kama_np — |net 30-bar change| over the summed
    # |1-bar changes| of the same window, algebraically in [0, 1] by the
    # triangle inequality. Zero realized volatility leaves the ratio
    # unavailable; it is never replaced by an epsilon or fabricated neutral
    # value. The rolling min_periods=30 warmup stays an honest NaN prefix
    # inside the layer's existing 201-row floor.
    kama_change_30 = (close - close.shift(30)).abs()
    kama_volatility_30 = (
        close.diff().abs().rolling(30, min_periods=30).sum()
    )
    kama_efficiency_30 = kama_change_30 / kama_volatility_30.where(
        kama_volatility_30 > 0.0
    )

    # V30 GAP-2/GAP-3 local durations (see the name-tuple comment).  Exact
    # per-TF construction: mask the state to NaN wherever its EMA source is
    # still inside the causal warmup (the shared classic EMA owner emits NaN
    # here), then count the raw uncapped run with the one state-age owner — a NaN warmup, never a
    # "0 bars since the state began" that reads as a fresh flip (rule 2e).
    # First finite row per field: classic EMA200 seed -> index 199 for the
    # 50/200 cross age and the ema200 side age, ema50 -> index 49; all three
    # are inside the layer's existing 201-row floor, which is therefore
    # unchanged (it is still set by ema50_200_spread_accel at index 201).
    bull_state = (spread > 0).astype(np.float64).where(spread.notna())
    ema50_200_state_age_bars = raw_state_age_bars(
        bull_state.to_numpy(dtype=np.float64),
        bull_state.notna().to_numpy(dtype=bool),
    )
    price_state_age_bars = {}
    # V30 package 3 (2026-08-13): the four price-vs-EMA cross events of the
    # same GAP-3 block, from the SAME ``price_gap`` series the age fields use
    # and the SAME htf event owner the per-TF lane calls.  ``_cross_up_event``
    # emits NaN wherever the series or its previous bar is still inside the
    # causal warmup, so the ema200 pair's first finite row is source index 200
    # (classic EMA200 first finite gap at 199, plus one bar for the
    # shift) and the ema50 pair's is 50 — both inside the layer's existing
    # 201-row floor, which is therefore unchanged (still set by
    # ema50_200_spread_accel at index 201; re-verified below on the full
    # 19-column layer).
    price_x_cross = {}
    for ema_span, ema_line in ((50, ema50), (200, ema200)):
        price_gap = close - ema_line
        side_state = (price_gap > 0).astype(np.float64).where(price_gap.notna())
        price_state_age_bars[ema_span] = raw_state_age_bars(
            side_state.to_numpy(dtype=np.float64),
            side_state.notna().to_numpy(dtype=bool),
        )
        price_x_cross[(ema_span, "up")] = _htf_cross_up_event_v4(price_gap)
        price_x_cross[(ema_span, "down")] = _htf_cross_down_event_v4(price_gap)

    raw = pd.DataFrame(
        {
            "ema50_200_spread_bps": spread_bps,
            "ema50_200_spread_atr": spread_atr,
            "ema50_200_bull_state": bull_state,
            "ema50_200_cross_up": _htf_cross_up_event_v4(spread),
            "ema50_200_cross_down": _htf_cross_down_event_v4(spread),
            "price_vs_ema50_bps": price_vs_ema50,
            "price_vs_ema200_bps": price_vs_ema200,
            "ema50_slope_bps": ema50_slope,
            "ema200_slope_bps": ema200_slope,
            "ema50_200_spread_delta": spread_delta,
            "ema50_200_spread_accel": spread_accel,
            "kama_efficiency_30": kama_efficiency_30,
            "ema50_200_state_age_bars": ema50_200_state_age_bars,
            "price_vs_ema50_state_age_bars": price_state_age_bars[50],
            "price_vs_ema200_state_age_bars": price_state_age_bars[200],
            "price_x_ema50_cross_up": price_x_cross[(50, "up")],
            "price_x_ema50_cross_down": price_x_cross[(50, "down")],
            "price_x_ema200_cross_up": price_x_cross[(200, "up")],
            "price_x_ema200_cross_down": price_x_cross[(200, "down")],
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
    for column in aligned.columns:
        add_chart_feature(
            arrays,
            names,
            f"local_{column}",
            aligned[column].to_numpy(dtype=np.float32),
        )
    out = np.column_stack(arrays).astype(np.float32, copy=False)
    if tuple(names) != PRICE_DERIVED_FEATURE_NAMES:
        raise RuntimeError("PRICE_DERIVED_FEATURE_ORDER_INVALID")
    return out, names


def build_candle_primitive_derived_layer(
    sample_df: pd.DataFrame,
    source_parquet: Path,
) -> tuple[np.ndarray, list[str]]:
    """Build raw closed-bar candle geometry with exact timestamp/OHLC proof."""

    context = "CANDLE_PRIMITIVE_DERIVED"
    sample_times = _require_sample_times(sample_df, context=context)
    source = Path(source_parquet).expanduser().resolve()
    available = _read_source_schema(source, context=context)
    missing = [name for name in CANDLE_PRIMITIVE_SOURCE_FIELDS if name not in available]
    if missing:
        raise RuntimeError(f"{context}_SOURCE_FIELDS_MISSING: {missing}")
    src = pd.read_parquet(source, columns=list(CANDLE_PRIMITIVE_SOURCE_FIELDS), engine="pyarrow")
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

    candle_x, candle_names = build_entry_candle_primitive_layer(src)
    candle_x = np.asarray(candle_x, dtype=np.float32)
    if candle_x.ndim != 2 or candle_x.shape[0] != len(src) or candle_x.shape[1] != len(candle_names):
        raise RuntimeError(
            f"{context}_OUTPUT_SHAPE_INVALID: shape={candle_x.shape} names={len(candle_names)}"
        )
    if not candle_names:
        raise RuntimeError(f"{context}_OUTPUT_EMPTY")
    if np.isinf(candle_x).any():
        raise RuntimeError(f"{context}_OUTPUT_INFINITY")
    complete = np.isfinite(candle_x).all(axis=1)
    if not complete.any():
        raise RuntimeError(f"{context}_OUTPUT_UNAVAILABLE")
    first_complete = int(np.argmax(complete))
    if first_complete != 1 or not complete[first_complete:].all():
        raise RuntimeError(f"{context}_OUTPUT_AVAILABILITY_INVALID")
    candle_df = pd.DataFrame(candle_x, columns=candle_names, index=source_index)
    aligned = candle_df.loc[sample_times]
    if not np.isfinite(aligned.to_numpy(dtype=np.float64)).all():
        raise RuntimeError(f"{context}_ALIGNED_OUTPUT_NONFINITE")
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


def build_volatility_squeeze_local_layer(
    sample_df: pd.DataFrame | None,
    source_parquet: Path,
    *,
    timeframe: str,
    artifact_set: VolatilitySqueezeArtifactSet,
    raw_frame: bool = False,
) -> tuple[np.ndarray, list[str]] | tuple[pd.DataFrame, list[str]]:
    """Build native M5 Entry or native M1 Exit squeeze state from one owner.

    The whole exact closed-OHLCV prefix is evaluated before row alignment, so
    bounded offline chunks and the owner's explicit live carry have identical
    first-row/event memory.  The six-clock manifest is mandatory; accepting a
    loose params payload here would create an unbound TRAIN/live split.
    """

    context = "VOLATILITY_SQUEEZE_LOCAL_LAYER"
    if timeframe not in {"M1", "M5"}:
        raise RuntimeError(f"{context}_TIMEFRAME_INVALID")
    frozen = require_volatility_squeeze_artifact_set(artifact_set)
    src = _read_v29_price_source(
        source_parquet,
        context=context,
        columns=("time", "open", "high", "low", "close", "volume"),
    )
    source_index = pd.DatetimeIndex(src.pop("time"))
    source = src.set_axis(source_index)
    raw, _carry = compute_volatility_squeeze_state(
        source,
        timeframe=timeframe,
        params=frozen.require_params(timeframe),
    )
    if tuple(raw.columns) != VOLATILITY_SQUEEZE_LOCAL_LAYER_FEATURE_NAMES:
        raise RuntimeError(f"{context}_FEATURE_ORDER_INVALID")
    if raw_frame:
        return raw, list(VOLATILITY_SQUEEZE_LOCAL_LAYER_FEATURE_NAMES)
    if sample_df is None:
        raise RuntimeError(f"{context}_SAMPLE_FRAME_REQUIRED")
    sample_times = _require_sample_times(sample_df, context=context)
    return _align_v29_layer_frame(
        raw,
        sample_times,
        VOLATILITY_SQUEEZE_LOCAL_LAYER_FEATURE_NAMES,
        context=context,
    )


def build_level_registry_m5_layer(
    sample_df: pd.DataFrame,
    source_parquet: Path,
    *,
    recurrence_threshold_atr: float,
    max_evidence_age_bars: int,
    decision_clock: str,
    decision_bar_seconds: int,
    raw_frame: bool = False,
) -> tuple[np.ndarray, list[str]] | tuple[pd.DataFrame, list[str]]:
    """Native M5/M1 level-registry block (design doc §1.2).

    ``recurrence_threshold_atr`` is the TRAIN-fitted frozen nearest-same-side
    pivot recurrence threshold
    (``fit_level_registry_hyperparameters_v1``); it has no default (rule 2a).
    ``raw_frame=True`` returns the unaligned full-history frame so the
    caller can measure the layer's warmup floor before choosing sample rows.
    """

    expected_seconds = {"M1": 60, "M5": 300}
    if (
        decision_clock not in expected_seconds
        or isinstance(decision_bar_seconds, bool)
        or decision_bar_seconds != expected_seconds.get(decision_clock)
    ):
        raise RuntimeError("LEVEL_REGISTRY_LOCAL_DECISION_CLOCK_INVALID")
    context = f"LEVEL_REGISTRY_{decision_clock}_LAYER"
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
        recurrence_threshold_atr=recurrence_threshold_atr,
        max_evidence_age_bars=max_evidence_age_bars,
        decision_clock=decision_clock.lower(),
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
    timeframe: str,
    band_atr: float,
    identity_expiry_bars: int,
    seq_len: int,
    raw_frame: bool = False,
) -> tuple[np.ndarray, list[str]] | tuple[pd.DataFrame, list[str]]:
    """Native M5/M1 trendline/channel block (design doc §2/§4.1 block E).

    ``band_atr`` is the TRAIN-fitted frozen band for the explicit native
    ``timeframe`` and candidate window ``seq_len``.  No clock, band, or window
    has a default (rule 2a). ``raw_frame=True`` returns the unaligned
    full-history frame for warmup-floor measurement.
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
        timeframe=timeframe,
        seq_len=seq_len,
        band_atr=band_atr,
        identity_expiry_bars=identity_expiry_bars,
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
    """Native M5/M1 structure_swing event block (G1/G2/G4 + the V30
    package-8A emission-only additions; the name tuple is the owner).

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
    """Native M5/M1 momentum event and continuous block (design §4.1 block E).

    One formula owner:
    ``htf_features.compute_v29_momentum_event_block_from_ohlc`` — the same
    function backing the per-TF V4 lane, run here on the caller's native M5
    Entry or M1 Exit clock.
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
    raw = compute_v29_momentum_event_block_from_ohlc(
        ohlc,
        include_v30_primitives=True,
    ).astype(np.float64)
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


def build_smc_local_event_layer(
    sample_df: pd.DataFrame,
    source_parquet: Path,
    *,
    raw_frame: bool = False,
) -> tuple[np.ndarray, list[str]] | tuple[pd.DataFrame, list[str]]:
    """Native M5/M1 SMC displacement, sided sweep-depth and event block."""

    context = "SMC_LOCAL_EVENT_LAYER"
    sample_times = (
        None if raw_frame else _require_sample_times(sample_df, context=context)
    )
    src = _read_v29_price_source(
        source_parquet,
        context=context,
        columns=("time", "high", "low", "close", "atr"),
    )
    source_index = pd.DatetimeIndex(src["time"])
    smc_source = pd.DataFrame(
        {
            "high": _require_finite_positive_column(src, "high", context=context),
            "low": _require_finite_positive_column(src, "low", context=context),
            "close": _require_finite_positive_column(src, "close", context=context),
            "atr": _require_finite_positive_column(src, "atr", context=context),
        },
        index=source_index,
    )
    computed = compute_smc_features(
        smc_source,
        include_v30_additions=True,
    )
    raw = computed.loc[:, list(SMC_LOCAL_EVENT_LAYER_FEATURE_NAMES)].astype(
        np.float64
    )
    if raw_frame:
        return raw, list(SMC_LOCAL_EVENT_LAYER_FEATURE_NAMES)
    return _align_v29_layer_frame(
        raw,
        sample_times,
        SMC_LOCAL_EVENT_LAYER_FEATURE_NAMES,
        context=context,
    )


def build_chart_layer(x: np.ndarray, feature_names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Emit raw local BOS/CHOCH ages after the full-history event floor."""

    x, _idx = _require_matrix_contract(
        x,
        feature_names,
        CHART_LAYER_SOURCE_FIELDS,
        context="CHART_LAYER",
    )
    foundation_x, foundation_names = build_entry_foundation_structure_layer(x, feature_names)
    if foundation_x.shape != (x.shape[0], len(foundation_names)) or not foundation_names:
        raise RuntimeError(
            f"CHART_LAYER_FOUNDATION_OUTPUT_INVALID: shape={foundation_x.shape} names={len(foundation_names)}"
        )
    for column in range(foundation_x.shape[1]):
        finite = np.isfinite(foundation_x[:, column])
        if finite.any() and not finite[int(np.argmax(finite)) :].all():
            raise RuntimeError("CHART_LAYER_CHILD_AVAILABILITY_INVALID")
    out = np.asarray(foundation_x, dtype=np.float32)
    names = list(foundation_names)
    if out.shape[1] != len(names) or len(names) != len(set(names)):
        raise RuntimeError(f"CHART_LAYER_OUTPUT_CONTRACT_INVALID: shape={out.shape} names={len(names)}")
    return out, names
