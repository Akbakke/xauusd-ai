"""V4-only causal multi-timeframe feature and immutable-cache owner.

The sole cache source is exact native M5 OHLCV. It emits the fixed ordered
``MULTI_TF_FEATURE_COUNT_V4``-field surface for M5/M15/H1/H4/D1 (the pre-V29
surface + the V29 Phase-A per-TF event fields + the V29 level/trendline
registry blocks; the count is derived from the declared name tuples, never
restated), routes Entry on M15/H1/H4/D1 and Exit on M5/M15/H1/H4/D1, and fails
closed on any schema, byte, chronology, warmup, or feature-order mismatch. No
historical cache contract or computed-feature fallback is exposed.

The V29 registry blocks carry TRAIN-fitted per-TF constants (level recurrence
threshold, trendline band). Those constants have no legitimate default (rule
2a): every surface computation requires an explicit
``v29_registry_constants`` payload produced by
:func:`fit_v29_registry_constants_from_m5` on the declared TRAIN window and
frozen in the immutable cache manifest.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import stat
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

# The session owner is the SSoT for the UTC trading-session boundary and its
# causal day ids.  Do not derive a second clock from calendar midnight here.
from gx1.time.session_detector import (
    TRADING_SESSION_BOUNDARY_OFFSET,
    TRADING_SESSION_BOUNDARY_UTC_HOUR,
    TRADING_SESSION_CLOCK_SCHEMA_VERSION,
    trading_session_id_vectorized,
    trading_session_label,
)
from gx1.features.technical_indicators_v1 import (
    classic_ema,
    ema50_200_spread_atr_block,
    technical_indicator_contract_metadata,
    wilder_atr,
    wilder_atr14_positive,
    wilder_rsi,
)
from gx1.features.event_age_v1 import (
    raw_event_age_bars,
    raw_state_age_bars,
)
from gx1.features.volume_features import (
    VOLUME_FEATURE_NAMES,
    compute_volume_features,
)
from gx1.features.volatility_squeeze_state_v1 import (
    VOLATILITY_SQUEEZE_FEATURE_NAMES,
    VolatilitySqueezeArtifactSet,
    compute_volatility_squeeze_state,
    require_volatility_squeeze_artifact_binding,
    require_volatility_squeeze_artifact_set,
)

# Retained shared warmup floors used by the active context owner.
D1_EMA200_MIN_BARS = 220
H1_ATR100_MIN_BARS = 120
M15_ATR100_MIN_BARS = 200
H4_EMA50_MIN_BARS = 80
D1_PCTL252_MIN_BARS = 270
# V30 (2026-08-13): warmup gate for the H4 sibling of the H1/M15 range
# compression ratio.  Same atr14/atr100 formula, so the same 100-bar ATR
# warmup floor applies; the value is inherited from the H1 sibling gate
# (derived assignment, not a new magnitude — rule 2b).
H4_ATR100_MIN_BARS = H1_ATR100_MIN_BARS

def _last_valid(series: pd.Series) -> float:
    s = series.dropna()
    return float(s.iloc[-1]) if len(s) else float("nan")


def _ema(series: pd.Series, span: int) -> pd.Series:
    """Compatibility route to the canonical classic SMA-seeded EMA."""

    return classic_ema(series, span)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    """Compatibility route to the one canonical Wilder ATR owner."""

    return wilder_atr(high, low, close, n)



def _resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Aggregate exact observed OHLCV for the V4 feature owner.

    Keyed on the declared TIMEFRAME, not a bare rule string, so the bin origin
    travels with the cadence (V30 package 3: the D1 bin opens on the trading
    day, see MULTI_TF_RESAMPLE_ORIGIN_OFFSET).
    """
    required = ("open", "high", "low", "close", "volume")
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise RuntimeError(
            f"HTF_V4_VOLUME_SOURCE_MISSING: exact OHLCV source required; missing={missing}"
        )
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": _last_valid,
        "volume": "sum",
    }
    return (
        multi_tf_resample(df.loc[:, list(required)], timeframe)
        .agg(agg)
        .dropna(how="all")
    )



def _validate_m5_input(
    m5_candles: pd.DataFrame,
    *,
    require_volume: bool = False,
    bar_duration: pd.Timedelta = pd.Timedelta(minutes=5),
) -> None:
    if not isinstance(m5_candles, pd.DataFrame):
        raise TypeError(
            f"HTF_INPUT_FAIL: m5_candles must be DataFrame, got {type(m5_candles).__name__}"
        )
    if m5_candles.empty:
        raise RuntimeError("HTF_INPUT_FAIL: m5_candles must be non-empty")
    if not isinstance(bar_duration, pd.Timedelta) or bar_duration <= pd.Timedelta(0):
        raise RuntimeError("HTF_INPUT_FAIL: bar_duration must be positive")
    required_cols = ["open", "high", "low", "close"]
    if require_volume:
        required_cols.append("volume")
    missing = [c for c in required_cols if c not in m5_candles.columns]
    if missing:
        raise RuntimeError(
            f"HTF_INPUT_FAIL: m5_candles missing required columns: {missing}"
        )
    if not isinstance(m5_candles.index, pd.DatetimeIndex):
        raise RuntimeError(
            "HTF_INPUT_FAIL: m5_candles index must be DatetimeIndex"
        )
    if m5_candles.index.tz is None:
        raise RuntimeError("HTF_INPUT_FAIL: m5_candles index must be timezone-aware UTC")
    if any(pd.Timestamp(ts).utcoffset() != pd.Timedelta(0) for ts in m5_candles.index[:1]):
        raise RuntimeError("HTF_INPUT_FAIL: m5_candles index must be UTC")
    if (
        m5_candles.index.hasnans
        or not m5_candles.index.is_unique
        or not m5_candles.index.is_monotonic_increasing
    ):
        raise RuntimeError(
            "HTF_INPUT_FAIL: timestamps must be finite, unique and chronological"
        )
    if np.any(m5_candles.index.asi8 % int(bar_duration.value) != 0):
        raise RuntimeError("HTF_INPUT_FAIL: timestamps are off the declared base grid")
    numeric = m5_candles.loc[:, required_cols].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("HTF_INPUT_FAIL: exact OHLCV sources must be finite")
    open_values = numeric["open"].to_numpy(dtype=np.float64)
    high_values = numeric["high"].to_numpy(dtype=np.float64)
    low_values = numeric["low"].to_numpy(dtype=np.float64)
    close_values = numeric["close"].to_numpy(dtype=np.float64)
    if (
        np.any(open_values <= 0.0)
        or np.any(high_values <= 0.0)
        or np.any(low_values <= 0.0)
        or np.any(close_values <= 0.0)
        or np.any(high_values < close_values)
        or np.any(low_values > close_values)
        or np.any(high_values < low_values)
        or (require_volume and np.any(high_values < open_values))
        or (require_volume and np.any(low_values > open_values))
    ):
        raise RuntimeError("HTF_INPUT_FAIL: OHLC geometry is invalid")
    if require_volume:
        if m5_candles["volume"].map(
            lambda value: isinstance(value, (bool, np.bool_))
        ).any():
            raise RuntimeError(
                "HTF_V4_VOLUME_SOURCE_INVALID: observed volume must be a "
                "positive integer tick count"
            )
        volume_values = numeric["volume"].to_numpy(dtype=np.float64)
        if np.any(volume_values <= 0.0) or not np.equal(
            volume_values, np.floor(volume_values)
        ).all():
            raise RuntimeError(
                "HTF_V4_VOLUME_SOURCE_INVALID: observed volume must be a "
                "positive integer tick count"
            )


# ---------------------------------------------------------------------------
# Sole V4 per-bar multi-timeframe surface.
# ---------------------------------------------------------------------------

# Exact ordered V4 feature contract. Persistent field names ending in _v2 are
# model fields and remain unchanged; they are not compatibility APIs.
from gx1.features.smc_v1 import (  # noqa: E402
    SMC_CAUSAL_REPLAY_SCHEMA_VERSION,
    SMC_MTF_FEATURE_NAMES_V1,
    SMC_MTF_GEOMETRY_FEATURE_NAMES_V1,
    SWING_LOOKBACK,
)
from gx1.features.level_registry_v1 import (  # noqa: E402
    LEVEL_REGISTRY_MTF_FEATURE_NAMES,
    compute_level_registry_mtf_block_v1,
    fit_level_registry_hyperparameters_v1,
)
from gx1.features.trendline_registry_v1 import (  # noqa: E402
    TRENDLINE_REGISTRY_FEATURE_NAMES_V1,
    compute_trendline_registry_features_v1,
    fit_trendline_registry_hyperparameters_v1,
)
from gx1.contracts.registry_hyperparameter_fit_v1 import (  # noqa: E402
    REGISTRY_HYPERPARAMETER_FIT_SCHEMA_V1,
    require_registry_hyperparameter_payload_v1,
)
from gx1.features.swing_structure_v1 import (  # noqa: E402
    SWING_FEATURE_NAMES_V1,
    SWING_V29_ADDITION_NAMES_V1,
    compute_swing_structure_features,
    swing_structure_contract_metadata,
)


def _candle_primitive_feature_names_v4() -> tuple[str, ...]:
    from gx1.features.entry_candle_primitives_v1 import (
        CANDLE_PRIMITIVE_FEATURE_NAMES,
    )

    return tuple(
        f"mtf_{name.replace('.', '_')}"
        for name in CANDLE_PRIMITIVE_FEATURE_NAMES
    )


MULTI_TF_V4_GROUP_A_BASE_FEATURES = (
    "atr_bps_14",
    "rsi14_centered",
    # V30 emission win (2026-08-13): raw Wilder RSI k-bar velocity
    # rsi14[t] - rsi14[t-5].  k=5 adopts this file's existing EMA-slope
    # lookback convention (ema20/50/200_slope_atr use shift(5)); the value is
    # algebraically bounded in [-100, 100] by the RSI domain, so no clip
    # constant is introduced.
    "rsi14_delta_5",
    "mom_5_atr",
    "mom_20_atr",
    "close_open_atr",
    "body_pct",
    "ema20_dist_atr",
    "ema50_dist_atr",
    "ema100_dist_atr",
    "ema200_dist_atr",
    "ema20_slope_atr",
    "ema50_slope_atr",
    "ema200_slope_atr",
    "ema_stack_aligned_v2",
    "vwap_local_cycle_dist_atr",
    "vwap20_dist_atr",
    "vwap96_dist_atr",
    "vwap_rolling5_slope_atr",
    "bb_position",
    "bb_width_atr",
    "adx14",
    # V30 emission win (2026-08-13): the signed normalized DI spread
    # (plus_di - minus_di)/(plus_di + minus_di) computed and previously
    # discarded inside the same _adx14 producer as adx14 (see its
    # docstring for the warmup/zero-denominator convention).
    "di_spread_signed",
    "trend_state_age_bars",
)
MULTI_TF_V4_CANDLE_PRIMITIVE_FEATURES = _candle_primitive_feature_names_v4()
# The exact swing-v2 local owner is projected independently on every native TF:
# no bar-zero pseudo-level, no partial ATR, no age/run cap and no normalized
# alias. Prefix-unavailable pivot measurements remain NaN until the shared
# causal-history trim; event ages remain NaN until their first genuine event.
#
# Per-TF spelling: base fields keep their historical ``swing_``
# prefix (their source names — ``bars_since_swing_high`` … — are ambiguous
# without it); the V29/V30 addition names are emitted VERBATIM, because they already
# carry their own ``swing_``/``bars_since_swing_``/``consecutive_`` identity
# and re-prefixing would spell ``swing_swing_high_break_event``.  The
# per-TF-name -> producer-field mapping is therefore declared explicitly here
# instead of being recovered by a prefix strip at the emission site.
MULTI_TF_V4_SWING_SOURCE_FIELD_BY_FEATURE = {
    **{
        (name if name.startswith("swing_") else f"swing_{name}"): name
        for name in SWING_FEATURE_NAMES_V1
    },
    **{name: name for name in SWING_V29_ADDITION_NAMES_V1},
}
MULTI_TF_V4_SWING_FEATURES = tuple(MULTI_TF_V4_SWING_SOURCE_FIELD_BY_FEATURE)
if set(MULTI_TF_V4_SWING_SOURCE_FIELD_BY_FEATURE.values()) != set(
    SWING_FEATURE_NAMES_V1 + SWING_V29_ADDITION_NAMES_V1
) or len(MULTI_TF_V4_SWING_SOURCE_FIELD_BY_FEATURE) != len(
    SWING_FEATURE_NAMES_V1 + SWING_V29_ADDITION_NAMES_V1
):
    raise RuntimeError("HTF_V4_SWING_SOURCE_FIELD_MAP_INVALID")

# ---------------------------------------------------------------------------
# V29 Phase A per-TF EVENT additions
# (docs/V29_EVENT_SURFACE_DESIGN_20260811.md §3; source reports trend_ema
# GAP-1/2/3 and momentum_flow G1/G2 of 2026-08-11).
#
# Constant origins (rule 2a, one sentence each):
# - RSI 30/70 bands are Wilder's published constants (1978, "New Concepts in
#   Technical Trading Systems"); 50 is the RSI midline and the affine center
#   this file already uses for rsi14_centered.  RSI_EXTREME_BAND_WIDTH is
#   derived arithmetic over those published constants (50 - 30), not a new
#   magnitude.
# - EMA event warmups come from the shared classic SMA-seeded EMA owner
#   (technical_indicators_v1.classic_ema).
# - Every age uses the shared raw observed-native-row age owner. Event ages
#   stay NaN until the first genuine event; state durations start at the first
#   valid state. Neither path is capped or log transformed.
# - The divergence pivot source is smc_v1's confirmed-pivot machinery with
#   its named SWING_LOOKBACK = 3 (the one pivot truth; no second detector).
#
# Routing and disk-cache widths derive from the declared feature tuples; no
# consumer may restate the historical surface widths.
RSI_WILDER_OVERSOLD = 30.0
RSI_WILDER_OVERBOUGHT = 70.0
RSI_WILDER_MIDLINE = 50.0
RSI_EXTREME_BAND_WIDTH = RSI_WILDER_MIDLINE - RSI_WILDER_OVERSOLD

MULTI_TF_V4_TREND_EVENT_FEATURES = (
    "ema50_200_spread_atr",
    "ema50_200_bull_state",
    "ema50_200_cross_up",
    "ema50_200_cross_down",
    "ema50_200_state_age_bars",
    "price_x_ema50_cross_up",
    "price_x_ema50_cross_down",
    "price_x_ema200_cross_up",
    "price_x_ema200_cross_down",
    "price_vs_ema50_state_age_bars",
    "price_vs_ema200_state_age_bars",
)
MULTI_TF_V4_MOMENTUM_EVENT_FEATURES = (
    "rsi_cross_up_30",
    "rsi_cross_down_70",
    "rsi_cross_up_50",
    "rsi_cross_down_50",
    "rsi_extreme_event_age_bars",
    "mom20_sign_flip_up",
    "mom20_sign_flip_down",
    "bear_divergence_event",
    "bull_divergence_event",
    # V30 emission win (2026-08-13): the divergence STRENGTH the design doc
    # declared for G1 (§3 momentum row: "event/strength/age") and the Phase-A
    # build dropped.  Value = (RSI delta between the pivot pair / 50) x
    # (price delta at the same pivots / ATR at the newer pivot's own bar);
    # /50 is RSI_WILDER_MIDLINE, the existing affine-map constant the design
    # row names, and the pivot-bar-ATR convention is the trendline registry's
    # "deviation measured at the pivot's own bar with that bar's ATR".  Both
    # fields are gated on their own event (0 off-event, flag-disambiguated
    # per design B.5) and NaN over the same undefined prefix as the events.
    "bear_divergence_strength",
    "bull_divergence_strength",
    "divergence_event_age_bars",
)

# Native-clock continuous momentum evidence. These are appended to the local
# M5 Entry / M1 Exit event block so each decision sequence sees its own RSI
# level, RSI velocity and short/medium momentum.
LOCAL_MOMENTUM_V30_PRIMITIVE_FEATURES = (
    "rsi14_centered",
    "rsi14_delta_5",
    "mom_5_atr",
    "mom_20_atr",
)

# Persistent model inputs that historically came from three separate HTF
# implementations.  They now have one owner: the native-M5 V4 lane.  The
# fixed per-bar V4 matrices remain unchanged; this compact scalar surface is
# computed from the same closed OHLCV bars and projected onto either local
# decision clock.  Names are persistent model fields, not compatibility APIs.
# V30 Phase-A completion (2026-08-13): momentum G3 raw-RSI ctx scalars.  The
# design doc §3 momentum row G3 declared `m5_rsi14`, `h1_rsi14_raw`,
# `h4_rsi14_raw` ("z-fields kept ... one `_rsi` producer, one unit") and the
# Phase-A build never wired them, leaving the RSI term structure with raw
# Wilder levels on M15/D1 (`m15_rsi14_canon_v2`, `d1_rsi14_canon_v2`) and only
# 48-bar z-scores on M5/H1/H4.  The three new fields are the VERBATIM siblings
# of the M15/D1 canon fields: same `_rsi(close, 14)` owner, same raw 0-100
# unit, same native-TF clock, same last-closed projection — so the spelling
# follows the existing canon convention (`<tf>_rsi14_canon_v2`) rather than the
# design doc's pre-implementation `*_raw` sketch (the code tuples are the
# authority, design doc STAGE-2 CORRECTION).
#
# KEEP-REASON for the five `*_rsi14_canon_v2` scalars (2026-08-18, V30 wave 2).
# All five were proposed for retirement as "the affine image
# 50*rsi14_centered + 50 of the MTF lane column"; all five were REFUSED, for
# two independent reasons, and the reasons are written here so the next
# session inherits them instead of re-deriving a wrong lemma.
#
# 1. THE AFFINE LEMMA DOES NOT REACH A LANE. The retirement argument was that
#    the input normalizer's `asinh((x - median)/IQR)` is affine-equivariant,
#    so a positive affine image of an already-present column is redundant.
#    Equivariance requires the SAME fitted (median, IQR) pair. Normalization
#    is fitted PER SURFACE: the bundle carries separate `input_norm_{surface}_*`
#    statistics and separate `mtf_m5..mtf_d1` field-name sets, fitted on
#    different row populations -- M5 decision rows, where a D1 value is a step
#    function held across a whole day, against each lane's own window
#    (`PRODUCTION_MTF_PER_TF_WINDOW_BARS` owns those window lengths; they are
#    not restated here). Two different fitted pairs are two different maps,
#    and the algebra never runs.
# 2. POOLING. The lane column `rsi14_centered` reaches the model only through
#    `encoded.mean(dim=1)` over that lane's window, while the ctx scalar enters
#    `_build_family_context_tokens` -> `z_v3`, which is computed BEFORE the MTF
#    route and gates it. The ctx scalar is the only un-pooled current-bar
#    reading of that timeframe's Wilder RSI level. M15 additionally has no
#    `_v1m15_rsi14_z` at all, so there the ctx scalar is the lane's only
#    momentum oscillator.
# 3. `m5_rsi14_canon_v2` HAS A THIRD, DECISIVE REASON: the ctx_cont tuple is
#    SHARED by the native-M5 Entry surface and the native-M1 Exit surface
#    (build_entry_exit_m1_enriched_frame_v1). `MODEL_NATIVE_MTF_SCALAR_ROUTES_V4`
#    routes "M5" on the 1-minute clock too, where this scalar is the LAST
#    CLOSED M5 bar, while that surface's own `rsi14_centered` is native M1
#    (the entry/exit feature-owner resolution contract declares
#    `independent_native_resolution_computation_required` and forbids
#    cross-resolution value copying). On the Exit clock the identity
#    `= 50*rsi14_centered + 50` is simply FALSE. Partial retirement is not
#    expressible either: `project_model_native_mtf_scalars_v4` fails closed
#    unless the projected set equals the output tuple exactly.
MODEL_NATIVE_MTF_SCALAR_FIELDS_BY_TIMEFRAME_V4 = {
    # The M5 slot was empty, not forbidden: the compact scalar surface is
    # "computed from the same closed OHLCV bars and projected onto either local
    # decision clock" (block comment above).  On the M5 decision clock the
    # projection of the M5 frame is the identity — cutoff = t + 5min -
    # MULTI_TF_SHIFT["M5"] = t, and the bar labelled t closes exactly when the
    # decision bar closes, the same closed-bar rule every other TF uses here.
    "M5": ("m5_rsi14_canon_v2",),
    "M15": (
        "m15_rsi14_canon_v2",
        "m15_range_z_20_canon_v2",
        "m15_ema5_20_spread_atr_canon_v2",
    ),
    "H1": (
        "_v1h1_ema_diff",
        "_v1h1_atr_bps",
        "_v1h1_rsi14_z",
        "_v1h1_slope3",
        "_v1h1_slope5",
        "h1_rsi14_canon_v2",
    ),
    "H4": (
        "h4_mid_ema50_dist_atr_canon_v2",
        "_v1h4_ema_diff",
        "_v1h4_atr_bps",
        "_v1h4_rsi14_z",
        "_v1h4_slope3",
        "_v1h4_slope5",
        "h4_rsi14_canon_v2",
    ),
    "D1": (
        "D1_dist_from_ema200_atr",
        "d1_atr14_bps_canon_v2",
        "d1_rsi14_canon_v2",
        "d1_ema_slope_20_canon_v2",
        "d1_range_z_20_canon_v2",
        "d1_close_pct_in_20day_range_canon_v2",
        "d1_change_5_bps_canon_v2",
        "d1_dist_change_1bar_atr_v4",
    ),
}
MODEL_NATIVE_MTF_SCALAR_OUTPUT_FIELDS_V4 = (
    "D1_dist_from_ema200_atr",
    "_v1h1_ema_diff",
    "_v1h1_atr_bps",
    "_v1h1_rsi14_z",
    "_v1h1_slope3",
    "_v1h1_slope5",
    "_v1h4_ema_diff",
    "_v1h4_atr_bps",
    "_v1h4_rsi14_z",
    "_v1h4_slope3",
    "_v1h4_slope5",
    "d1_atr14_bps_canon_v2",
    "d1_rsi14_canon_v2",
    "d1_ema_slope_20_canon_v2",
    "d1_range_z_20_canon_v2",
    "d1_close_pct_in_20day_range_canon_v2",
    "d1_change_5_bps_canon_v2",
    "m15_rsi14_canon_v2",
    "m15_range_z_20_canon_v2",
    "m15_ema5_20_spread_atr_canon_v2",
    # V30 (2026-08-13): the three momentum-G3 raw-RSI siblings, ordered M5 ->
    # H1 -> H4 as the design row lists them; the ctx_cont contract appends the
    # same three names in the same order (the single-owner test requires the
    # ctx_cont intersection to preserve this tuple's order).
    "m5_rsi14_canon_v2",
    "h1_rsi14_canon_v2",
    "h4_rsi14_canon_v2",
    "h4_mid_ema50_dist_atr_canon_v2",
    "d1_dist_change_1bar_atr_v4",
)
# v8 (2026-08-18, V30 wave 2): four names in the tuple above change, no value
# moves. `_v1h{1,4}_atr` -> `_v1h{1,4}_atr_bps` and `d1_atr14_canon_v2` ->
# `d1_atr14_bps_canon_v2` because the 2026-08-09 era-proxy repair converted
# those three to bps and left `_atr` in the name, which means "in ATR units"
# everywhere else here; `d1_pct_change_5_canon_v2` ->
# `d1_change_5_bps_canon_v2` because pandas' pct_change is a FRACTION and the
# producer multiplies by 1e4. Positions are preserved, so
# `field_order_sha256` changes but the projection order does not. All four are
# atomic: they sit in the same two hash-bound tuples, so landing three and
# deferring one invalidates every canonical artifact anyway, on BOTH clocks
# (MODEL_NATIVE_MTF_SCALAR_ROUTES_V4 is keyed on decision_bar_duration).
MODEL_NATIVE_MTF_SCALAR_CONTRACT_V4 = (
    "model_native_mtf_scalar_owner_native_m5_v8"
)
# V30 (2026-08-13): the scalar-projection route per decision clock, previously
# repeated as two identical literals inside the projection function and the
# owner marker.  Both now read this one owner, so the marker can never describe
# a route the projection did not take.  The M5 entry (5-minute clock) is new:
# it carries the momentum-G3 `m5_rsi14_canon_v2` scalar, whose projection onto
# the M5 clock is the identity (see the field-tuple comment).  This route is the
# SCALAR surface only; the per-TF windowed sequence route stays
# ENTRY_MTF_CONTEXT_TIMEFRAMES = M15/H1/H4/D1 (entry_exit_feature_base_v1) and
# is untouched.
MODEL_NATIVE_MTF_SCALAR_ROUTES_V4 = {
    pd.Timedelta(minutes=5): ("M5", "M15", "H1", "H4", "D1"),
    pd.Timedelta(minutes=1): ("M5", "M15", "H1", "H4", "D1"),
}
# V29 Phase A per-TF REGISTRY blocks (stage 2 wiring, design doc §1.3/§2):
# the immutable pivot-anchor block and the trendline/channel block run
# independently on every TF clock next to
# compute_smc_mtf_primitives_v1.  Their exact ordered names are owned by the
# two registry modules; this owner only sequences them.  Both blocks carry
# TRAIN-fitted constants (level recurrence threshold / trendline band) that must
# arrive through an explicit ``v29_registry_constants`` payload — no default
# exists here (rule 2a).
MULTI_TF_V4_LEVEL_REGISTRY_FEATURES = tuple(LEVEL_REGISTRY_MTF_FEATURE_NAMES)
MULTI_TF_V4_TRENDLINE_REGISTRY_FEATURES = tuple(
    TRENDLINE_REGISTRY_FEATURE_NAMES_V1
)
# Raw tick-count activity primitives.  The exact same owner computes these on
# every already-resampled closed OHLCV frame; computed feature values are never
# resampled between clocks.
MULTI_TF_V4_VOLUME_FEATURES = tuple(VOLUME_FEATURE_NAMES)
MULTI_TF_V4_VOLATILITY_SQUEEZE_FEATURES = tuple(
    VOLATILITY_SQUEEZE_FEATURE_NAMES
)

MULTI_TF_PER_BAR_FEATURES_V4 = (
    MULTI_TF_V4_GROUP_A_BASE_FEATURES
    + MULTI_TF_V4_VOLUME_FEATURES
    + MULTI_TF_V4_VOLATILITY_SQUEEZE_FEATURES
    + MULTI_TF_V4_CANDLE_PRIMITIVE_FEATURES
    + MULTI_TF_V4_SWING_FEATURES
    + SMC_MTF_FEATURE_NAMES_V1
    + SMC_MTF_GEOMETRY_FEATURE_NAMES_V1
    + MULTI_TF_V4_TREND_EVENT_FEATURES
    + MULTI_TF_V4_MOMENTUM_EVENT_FEATURES
    + MULTI_TF_V4_LEVEL_REGISTRY_FEATURES
    + MULTI_TF_V4_TRENDLINE_REGISTRY_FEATURES
)
MULTI_TF_FEATURE_COUNT_V4 = len(MULTI_TF_PER_BAR_FEATURES_V4)
MULTI_TF_FEATURE_NAMES_SHA256_V4 = hashlib.sha256(
    json.dumps(
        list(MULTI_TF_PER_BAR_FEATURES_V4),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()
# V14 (2026-08-15): the trendline-registry presence masks
# geomline_{above,below}_active leave the per-TF surface (they were the ">= 1"
# indicator of the counts beside them), and body_pct adopts the sibling candle
# owner's zero-range share convention instead of a mid-series NaN.
# V15 (2026-08-15): mtf_candle_raw_zero_range_flag leaves the per-TF surface.
# It is constant 0.0 post-warmup on H4 and D1 (gold prints no zero-range
# 4-hour or daily bar), which build_multi_tf_v4_liveness_contract fails closed
# on, and a declared-constant exemption would only move the failure to
# [ENTRY_INPUT_NORMALIZATION_UNSCALEABLE]. Its two inputs, high and low,
# remain model inputs on every lane. See the CANDLE_PRIMITIVE_FEATURE_VERSION
# note in gx1.features.entry_candle_primitives_v1 for the measurements.
# V16 (2026-08-15): the trendline registry's ACTIVE staleness bound returns to
# the declared receptive field, so 29 previously constant/all-NaN geomline_*
# columns now carry real values under unchanged names. A V15 matrix holds
# different numbers under the same feature-name hash and must not be read as
# current.
# V17 (2026-08-18, V30 wave 2): the per-lane surface NARROWS. The candle owner
# retired 4 columns (close_location, range_change, the two rejection depths),
# the SMC geometry owner retired 3 (both *_break_displacement_atr ReLUs and
# nearest_level_abs), and the level registry retired 3
# (both *_completed_reaction_count and the above-side presence mask); the two
# per-bar sweep CONDITIONS are renamed to mtf_smc_sweep_{up,down}_state. Every
# retired column was an exact continuous function of columns that remain in
# the same specialist family, so nothing left the learned path -- but a V16
# matrix is 10 columns per lane wider and MUST NOT be read as current.
# V18 (2026-08-18, V30 wave 2, contract step): the per-lane WIDTH is unchanged
# from V17, but one column's VALUES change and its name with them.
# `vwap_local_cycle_slope_atr` -> `vwap_rolling5_slope_atr`: the old field
# differenced a CUMULATIVE session VWAP across a fixed row count, so on H4 a
# measured 83.47% of its defined rows straddled a session reset and the
# subtrahend belonged to a different accumulation. The repaired field
# differences the rolling 5-bar VWAP the D1 branch already used. A V17 matrix
# holds the broken numbers under a name that no longer exists.
HTF_V4_MATRIX_CONTRACT = "HTF_V4_EIGHT_FAMILY_CAUSAL_MATRIX_V18"
# v5: the manifest additionally binds the immutable v29_registry_constants
# payload (TRAIN-fitted level/trendline registry constants + provenance).
# v6 (V30 package 3, 2026-08-13): the manifest additionally binds the declared
# resample ORIGIN contract.  The D1 bin now opens on the trading day
# (MULTI_TF_RESAMPLE_ORIGIN_OFFSET), so a cache written on the midnight-UTC D1
# axis holds different bars under the same feature names — a new manifest key
# plus a new schema version reject it before a single array is loaded rather
# than letting it masquerade as current.
# v7 adds the shared per-TF volume primitives. v8 preserves those fields and
# phases H4/D1 from the one UTC trading-session owner. v9 binds the shared SMC
# causal replay and one-shot level-identity sweeps. v18 replaces capped age and
# premium aliases with raw event ages and raw pivot-envelope position. v19
# binds the swing-v2 formula owner and its wider raw current-state surface.
# v23 retires the two trendline presence masks and rewrites body_pct on
# zero-range bars; a v22 cache holds different columns under the same names.
# v24 retires mtf_candle_raw_zero_range_flag; a v23 cache is one column wider
# under the same feature-name hash key and must not be read as current.
# v25 (2026-08-15) carries the repaired trendline staleness bound: the manifest
# embeds the v29_registry_constants payload, whose key set just lost
# trendline_expiry_bars, and the emitted geomline_* values change.
# v26 (2026-08-18) carries the V30 wave-2 narrowing: a v25 cache is 10 columns
# per lane wider and carries two per-TF sweep columns under their pre-rename
# names.
HTF_V4_CACHE_SCHEMA_VERSION = "htf_v4_disk_cache_manifest_v27"
HTF_V4_CACHE_BUILDER_VERSION = (
    "prebuild_multi_tf_cache_v4_raw_continuous_scalar_fidelity_20260814"
)
# v12 re-proves support for raw uncapped event ages without persistent global
# ever-seen masks; earlier payloads cannot describe this width.
# v15 describes the surface without the two trendline presence masks.
# v16 describes the surface without mtf_candle_raw_zero_range_flag — the only
# per-TF field that could not reach a liveness verdict on H4/D1 at all.
# v17 (2026-08-18) describes the V30 wave-2 surface: 10 fewer columns per lane
# and the two renamed sweep-state columns. A v16 artifact answers liveness for
# names this surface no longer emits.
HTF_V4_FULL_INPUT_LIVENESS_SCHEMA_VERSION = "htf_v4_full_input_liveness_v18"
# Deliberate bit-identical aliases inside the fixed per-bar V4 model surface,
# exempted from the duplicate-column failure in
# :func:`build_multi_tf_v4_liveness_contract`.  Each entry is the exact ordered
# pair ``(first_emitted_name, later_emitted_name)`` in
# MULTI_TF_PER_BAR_FEATURES_V4 order — that is the order the duplicate check
# builds the pair in.
#
# 2026-08-15: emptied.  The three pairs declared here had gone stale — five of
# their six names had already left MULTI_TF_PER_BAR_FEATURES_V4
# (`mtf_pattern_*_share` never existed on this surface; `upper_wick_pct` and
# `lower_wick_pct` were retired in favour of the causal candle owner's
# `mtf_candle_raw_{upper,lower}_wick_share`), so no emitted pair could ever
# match and the exemption silently protected nothing.  The surface currently
# declares no alias at all; every duplicated column is therefore a hard
# failure.  ``test_declared_alias_pairs_name_fields_the_surface_emits`` in
# tests/test_htf_v4_per_bar_contract.py keeps a future entry from going stale
# the same way.
HTF_V4_DECLARED_ALIAS_PAIRS: frozenset[tuple[str, str]] = frozenset()

MULTI_TF_RESAMPLE_RULES = {
    # Resample cadence only. Entry window lengths are explicit recipe inputs
    # and must form a strictly increasing wall-clock coverage pyramid.
    "M5": "5min",
    "M15": "15min",
    "H1": "1h",
    "H4": "4h",
    "D1": "1D",
}

# V30 package 3 (2026-08-13) — ONE daily clock, anchored to the trading day.
#
# The cadence above is a duration; it does not say WHERE the bin starts.
# pandas defaults the "1D" origin to midnight UTC, which cut the gold tape's
# trading day in half: the 22:00-24:00 UTC Sunday reopen became its own "D1"
# bar and was fed to ATR-14 / EMA200 / RSI / the 252-bar percentile as one
# complete observation, and Monday's D1 features read it as the previous day.
#
# MEASURED on the complete declared native M5 tape
# (XAU_M5_NATIVE_2019_20260804_V4, 537,861 rows, 2019-01-01..2026-08-04),
# coverage = bars present / 288 possible M5 bars per D1 bin:
#   midnight-UTC origin : N=2,360 bins, 401 bins (16.99%) at <=10% coverage,
#                         Sunday-bin median coverage 8.33%
#   22:00-UTC origin    : N=1,960 bins,   1 bin  ( 0.05%) at <=10% coverage,
#                         Sunday-bin median coverage 95.83%
# (rule 2f: the population is the complete declared tape, not a sample, so the
# comparison carries no sampling error; the log is
# GX1_DATA/logs/v30_package3_20260813/d1_coverage_audit.log.)
#
# The magnitude is inherited from the session owner, never restated here.
# Every TF derives its phase as ``boundary % duration``.  M1/M5/M15/H1 remain
# on their UTC grids; H4 becomes 22/02/06/10/14/18 so no H4 observation spans
# two trading-session ids; D1 opens on the boundary itself.
MULTI_TF_D1_TRADING_DAY_ORIGIN_HOUR = TRADING_SESSION_BOUNDARY_UTC_HOUR
MULTI_TF_RESAMPLE_ORIGIN_OFFSET = {
    timeframe: TRADING_SESSION_BOUNDARY_OFFSET % pd.Timedelta(rule)
    for timeframe, rule in MULTI_TF_RESAMPLE_RULES.items()
}
if tuple(MULTI_TF_RESAMPLE_ORIGIN_OFFSET) != tuple(MULTI_TF_RESAMPLE_RULES):
    raise RuntimeError(
        "HTF_V4_RESAMPLE_ORIGIN_CONTRACT_INVALID: exact ordered "
        "M5/M15/H1/H4/D1 origin offsets required"
    )
for _tf_name, _offset in MULTI_TF_RESAMPLE_ORIGIN_OFFSET.items():
    _rule_duration = pd.Timedelta(MULTI_TF_RESAMPLE_RULES[_tf_name])
    if (
        not isinstance(_offset, pd.Timedelta)
        or _offset < pd.Timedelta(0)
        or _offset >= _rule_duration
        or _offset % pd.Timedelta(MULTI_TF_RESAMPLE_RULES["M5"]) != pd.Timedelta(0)
    ):
        raise RuntimeError(
            f"HTF_V4_RESAMPLE_ORIGIN_CONTRACT_INVALID: {_tf_name} offset={_offset!r}"
        )
# The exact declared origin contract, recorded in the cache manifest so a cache
# built on another daily clock can never load against this owner.
MULTI_TF_RESAMPLE_ORIGIN_CONTRACT = {
    _tf: str(_off) for _tf, _off in MULTI_TF_RESAMPLE_ORIGIN_OFFSET.items()
}

MULTI_TF_TIMEFRAMES = tuple(MULTI_TF_RESAMPLE_RULES)
MULTI_TF_TIMEFRAMES_LOWER = tuple(
    timeframe.lower() for timeframe in MULTI_TF_TIMEFRAMES
)
MULTI_TF_TIMEFRAMES_LOWER_M5_LAST = (
    *MULTI_TF_TIMEFRAMES_LOWER[1:],
    MULTI_TF_TIMEFRAMES_LOWER[0],
)

# Pandas-Timedelta shift per TF: ensures we use only CLOSED bars at-or-before t
MULTI_TF_SHIFT = {
    "M5": pd.Timedelta(minutes=5),
    "M15": pd.Timedelta(minutes=15),
    "H1": pd.Timedelta(hours=1),
    "H4": pd.Timedelta(hours=4),
    "D1": pd.Timedelta(days=1),
}
MULTI_TF_PYRAMID_SCHEMA_VERSION = "entry_multi_tf_causal_resolution_pyramid_v1"


def multi_tf_bar_label(values, timeframe: str):
    """Floor UTC timestamps onto one timeframe's declared bar-opening grid.

    THE flooring owner for the V4 multi-TF axis.  ``pandas`` ``floor`` has no
    origin argument, so the declared origin offset is applied by shifting into
    the midnight-anchored grid, flooring, and shifting back —
    ``(t - offset).floor(rule) + offset``.  Verified equal to
    ``resample(rule, offset=offset)``'s bin-left edges on pandas 2.3.3 for the
    D1 offset (the audit log named at MULTI_TF_RESAMPLE_ORIGIN_OFFSET), so the
    two mechanisms cannot drift.

    Accepts anything with a ``floor`` method on the pandas datetime interface
    (``Timestamp``, ``DatetimeIndex``, ``Series.dt`` output).
    """
    if timeframe not in MULTI_TF_RESAMPLE_RULES:
        raise RuntimeError(f"HTF_V4_TIMEFRAME_INVALID: {timeframe!r}")
    if timeframe == "D1":
        return trading_session_label(values, context="HTF_V4_D1_BAR_LABEL")
    offset = MULTI_TF_RESAMPLE_ORIGIN_OFFSET[timeframe]
    floored = (values - offset).floor(MULTI_TF_RESAMPLE_RULES[timeframe])
    return floored + offset


def multi_tf_resample(frame, timeframe: str):
    """Return one timeframe's resampler on the declared cadence AND origin.

    THE resampling owner for the V4 multi-TF axis; pairs with
    :func:`multi_tf_bar_label` so a bin label produced by one is always the
    bin label produced by the other.
    """
    if timeframe not in MULTI_TF_RESAMPLE_RULES:
        raise RuntimeError(f"HTF_V4_TIMEFRAME_INVALID: {timeframe!r}")
    return frame.resample(
        MULTI_TF_RESAMPLE_RULES[timeframe],
        offset=MULTI_TF_RESAMPLE_ORIGIN_OFFSET[timeframe],
    )


def multi_tf_last_closed_label(
    decision_bar_start: pd.Timestamp | str,
    timeframe: str,
    *,
    base_bar_duration: pd.Timedelta = pd.Timedelta(minutes=5),
) -> pd.Timestamp:
    """Return the exact opening label of the last closed bar for one TF.

    ``decision_bar_start`` is the opening timestamp of an observed M5 candle.
    Its information becomes available five minutes later.  HTF resample labels
    are bar-opening timestamps, so the availability cutoff must be shifted by
    the full HTF duration and then floored to that timeframe's declared grid.

    No-lookahead proof, unchanged by the V30 D1 trading-day origin: a bar
    labelled ``L`` on a grid of duration ``d`` covers ``[L, L + d)`` and is
    therefore closed at ``L + d``.  This returns the largest grid point
    ``L <= t + base_bar_duration - d``, so ``L + d <= t + base_bar_duration``
    — the bar is closed no later than the moment the decision bar's own
    information becomes available.  The argument uses only "the grid points
    are spaced ``d`` apart", never where the grid starts, so shifting the D1
    grid's origin from 00:00 to 22:00 UTC leaves it intact.
    """
    if timeframe not in MULTI_TF_RESAMPLE_RULES:
        raise RuntimeError(
            f"HTF_V4_TIMEFRAME_INVALID: {timeframe!r}"
        )
    if not isinstance(base_bar_duration, pd.Timedelta) or base_bar_duration <= pd.Timedelta(0):
        raise RuntimeError("HTF_V4_BASE_BAR_DURATION_INVALID")
    timestamp = pd.Timestamp(decision_bar_start)
    if timestamp.tz is None or timestamp.utcoffset() != pd.Timedelta(0):
        raise RuntimeError(
            "HTF_V4_DECISION_TIMESTAMP_INVALID: timezone-aware UTC required"
        )
    return multi_tf_bar_label(
        timestamp + base_bar_duration - MULTI_TF_SHIFT[timeframe],
        timeframe,
    )


def build_multi_tf_v4_closed_timestamp_indices(
    m5_index: pd.DatetimeIndex,
) -> dict[str, pd.DatetimeIndex]:
    """Derive the sole V4 cache axis from an exact native-M5 source."""
    base_bar_duration = pd.Timedelta(minutes=5)
    if not isinstance(m5_index, pd.DatetimeIndex) or len(m5_index) == 0:
        raise RuntimeError(
            "HTF_V4_SOURCE_TIMESTAMP_GEOMETRY_INVALID: non-empty "
            "DatetimeIndex required"
        )
    m5_index = m5_index.as_unit("ns")
    if (
        m5_index.tz is None
        or m5_index.hasnans
        or not m5_index.is_unique
        or not m5_index.is_monotonic_increasing
        or m5_index[0].utcoffset() != pd.Timedelta(0)
    ):
        raise RuntimeError(
            "HTF_V4_SOURCE_TIMESTAMP_GEOMETRY_INVALID: exact chronological "
            "unique UTC timestamps required"
        )
    if not m5_index.floor(base_bar_duration).equals(m5_index):
        raise RuntimeError(
            "HTF_V4_SOURCE_TIMESTAMP_GEOMETRY_INVALID: source timestamps "
            "must lie on the exact M5 UTC grid"
        )

    expected: dict[str, pd.DatetimeIndex] = {}
    for timeframe in MULTI_TF_RESAMPLE_RULES:
        labels = multi_tf_bar_label(m5_index, timeframe).drop_duplicates()
        last_closed = multi_tf_last_closed_label(
            m5_index[-1],
            timeframe,
            base_bar_duration=base_bar_duration,
        )
        labels = labels[labels <= last_closed]
        if len(labels) and m5_index[0] > labels[0]:
            labels = labels[1:]
        if len(labels) == 0:
            raise RuntimeError(
                f"HTF_V4_NO_COMPLETE_RESAMPLED_BARS: {timeframe}"
            )
        expected[timeframe] = labels
    return expected


def require_multi_tf_resolution_pyramid(
    per_tf_seq_lens: dict[str, int],
) -> dict[str, object]:
    """Validate explicit windows as strictly increasing wall-clock coverage."""
    expected_tfs = tuple(MULTI_TF_RESAMPLE_RULES)
    if not isinstance(per_tf_seq_lens, dict) or tuple(per_tf_seq_lens) != expected_tfs:
        raise RuntimeError(
            "MULTI_TF_RESOLUTION_PYRAMID_ORDER_INVALID: exact "
            "M5/M15/H1/H4/D1 declaration required"
        )
    lengths: dict[str, int] = {}
    coverage_seconds: dict[str, int] = {}
    for tf in expected_tfs:
        value = per_tf_seq_lens[tf]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RuntimeError(
                f"MULTI_TF_RESOLUTION_PYRAMID_LENGTH_INVALID: {tf}={value!r}"
            )
        lengths[tf] = int(value)
        coverage_seconds[tf] = int(
            value * MULTI_TF_SHIFT[tf].total_seconds()
        )
    spans = tuple(coverage_seconds.values())
    if any(left >= right for left, right in zip(spans, spans[1:])):
        raise RuntimeError(
            "MULTI_TF_RESOLUTION_PYRAMID_COVERAGE_INVALID: progressively "
            f"coarser timeframes must cover strictly older history; {coverage_seconds}"
        )
    payload: dict[str, object] = {
        "schema_version": MULTI_TF_PYRAMID_SCHEMA_VERSION,
        "timeframe_order": list(expected_tfs),
        "per_tf_seq_lens": lengths,
        "coverage_seconds": coverage_seconds,
        "strictly_increasing_wall_clock_coverage": True,
    }
    payload["contract_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return payload


# ---------------------------------------------------------------------------
# V29 registry constants — the one payload carrying the TRAIN-fitted level
# recurrence threshold and trendline band per TF (rule 18: fitted once on the declared
# TRAIN window, frozen, never refitted downstream) plus the entry-M5-lane
# trendline fit (candidate window = the Entry model sequence length).  The
# Both operators are selected by immutable chronological inner-TRAIN
# competing-risk artifacts. No q/reaction/retest recipe input exists.
# ---------------------------------------------------------------------------
# v8 / provenance v7 (2026-08-15): trendline_expiry_bars leaves both the
# per-clock payload and the entry_m5 block. The trendline registry stopped
# consuming a fitted identity lifetime (it measures bars-to-projection-break,
# not bars since a promoted line was touched, and every fitted value was
# <= SWING_LOOKBACK, which deleted each line on its own promotion bar).
V29_REGISTRY_CONSTANTS_SCHEMA_VERSION = "htf_v4_v29_registry_constants_v8"
V29_REGISTRY_CONSTANTS_PROVENANCE_SCHEMA_VERSION = (
    "htf_v4_v29_registry_constants_provenance_v7"
)
_V29_REGISTRY_CONSTANTS_KEYS = frozenset(
    {
        "schema_version",
        "declared_train_window_start",
        "declared_train_window_end",
        "declared_inner_fit_window_end",
        "level_recurrence_threshold_atr",
        "level_expiry_bars",
        "trendline_band_atr",
        "per_tf_seq_lens",
        "entry_m5",
        "provenance",
        "contract_sha256",
    }
)
_V29_REGISTRY_ENTRY_M5_KEYS = frozenset({"seq_len", "trendline_band_atr"})
_V29_REGISTRY_CONSTANTS_PROVENANCE_KEYS = frozenset(
    {
        "schema_version",
        "lane",
        "module",
        "payload_schema_version",
        "fit_owner",
        "declared_train_window_start",
        "declared_train_window_end",
        "declared_inner_fit_window_end",
        "n_train_m5_rows",
        "inner_fit_end_exclusive_by_clock",
        "level_recurrence_threshold",
        "trendline_band",
        "entry_m5_trendline_band",
    }
)


def _contract_sha256_without_field(value: Mapping) -> str:
    payload = dict(value)
    payload.pop("contract_sha256", None)
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _sha256_file_exact(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_contract_sha256(value: Mapping, *, context: str) -> None:
    observed = value.get("contract_sha256")
    expected = _contract_sha256_without_field(value)
    if (
        not isinstance(observed, str)
        or len(observed) != 64
        or any(character not in "0123456789abcdef" for character in observed)
        or observed != expected
    ):
        raise RuntimeError(f"{context}: contract_sha256")


def _require_nonnegative_int(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise RuntimeError(context)
    observed = int(value)
    if observed < 0:
        raise RuntimeError(context)
    return observed


def _require_finite_number(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise RuntimeError(context)
    observed = float(value)
    if not math.isfinite(observed):
        raise RuntimeError(context)
    return observed


def _require_canonical_utc_timestamp_label(
    value: object,
    *,
    context: str,
) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(context)
    try:
        timestamp = pd.Timestamp(value)
    except Exception as exc:
        raise RuntimeError(context) from exc
    if (
        pd.isna(timestamp)
        or timestamp.tzinfo is None
        or timestamp.utcoffset() != pd.Timedelta(0)
        or timestamp.isoformat() != value
    ):
        raise RuntimeError(context)
    return value


def _require_registry_hyperfit_provenance(
    value: object,
    *,
    registry_kind: str,
    timeframe: str,
    selected_threshold_atr: float,
    # ``None`` means the caller's payload carries no second copy of the fitted
    # lifetime to cross-check against (the trendline lanes since 2026-08-15:
    # the registry stopped consuming it, so the constants payload stopped
    # declaring it).  Keyword-only WITHOUT a default on purpose — a default
    # would let a lane that does own a copy drop the binding silently.
    learned_expiry_bars: int | None,
    source_row_count: int,
    inner_fit_end_exclusive: int,
    window_start: str,
    window_end: str,
    population_configuration: Mapping[str, object] | None,
    context: str,
) -> None:
    try:
        payload = require_registry_hyperparameter_payload_v1(
            value,
            registry_kind=registry_kind,
            clock=timeframe.upper(),
        )
    except RuntimeError as exc:
        raise RuntimeError(f"{context}: registry hyperfit provenance") from exc
    source = payload["source_provenance"]
    observed_population = payload["population_configuration"]
    if registry_kind == "horizontal_level" and learned_expiry_bars is None:
        raise RuntimeError(f"{context}: level lifetime binding required")
    if registry_kind == "horizontal_level":
        expected_level_keys = {
            "owner",
            "runtime_owner",
            "identity_control",
            "threshold_selection",
            "lifetime_selection",
            "swing_lookback",
            "canonical_lifecycle_sha256",
            "canonical_episode_sha256",
            "canonical_identity_state_sha256",
            "canonical_recurrence_observation_count",
            "canonical_outcome_stream_sha256",
            "canonical_event_tape_sha256",
            "fit_phase",
            "population_decomposition_contract",
            "selected_threshold_atr",
            "selected_expiry_bars",
            "selected_runtime_recurrence_observation_count",
            "final_fit_recurrence_population_sha256",
            "final_serve_recurrence_population_sha256",
            "canonical_fit_outcome_population_sha256",
            "selected_derived_outcome_population_sha256",
            "selected_derived_lifecycle_sha256",
            "selected_derived_episode_sha256",
            "selected_derived_state_sha256",
            "selected_derived_emission_sha256",
            "selected_eligibility_expiry_count",
        }
        if (
            not isinstance(observed_population, Mapping)
            or set(observed_population) != expected_level_keys
            or observed_population["owner"]
            != "level_immutable_anchor_recurrence_decomposition_v2"
            or observed_population["runtime_owner"]
            != "gx1.features.level_registry_v1._run_level_registry"
            or observed_population["identity_control"]
            != (
                "immutable_per_confirmed_pivot_anchor_exact_center_break_touch_"
                "and_parameter_independent_identity_retention"
            )
            or observed_population["threshold_selection"]
            != "recurrence_confirmed_on_nearest_same_side_distance"
            or observed_population["lifetime_selection"]
            != (
                "learned_expiry_of_event_and_slot_eligibility_without_"
                "identity_deletion"
            )
            or observed_population["swing_lookback"] != int(SWING_LOOKBACK)
            or observed_population["fit_phase"]
            != "immutable_recurrence_canonical_outcome_decomposition"
            or observed_population["population_decomposition_contract"]
            != (
                "serve_recurrence_uses_all_prior_immutable_same_birth_side_"
                "anchors_via_price_index;train_outcomes_use_parameter_"
                "independent_exact_cross_lifecycle;selected_expiry_gates_"
                "only_event_slot_eligibility"
            )
            or float(observed_population["selected_threshold_atr"])
            != float(selected_threshold_atr)
            or int(observed_population["selected_expiry_bars"])
            != int(learned_expiry_bars)
            or not isinstance(
                observed_population["canonical_recurrence_observation_count"], int
            )
            or observed_population["canonical_recurrence_observation_count"] <= 0
            or observed_population["selected_runtime_recurrence_observation_count"]
            != observed_population["canonical_recurrence_observation_count"]
            or observed_population["final_fit_recurrence_population_sha256"]
            != observed_population["final_serve_recurrence_population_sha256"]
            or observed_population["canonical_fit_outcome_population_sha256"]
            != payload["outcome_stream_sha256"]
            or not isinstance(
                observed_population["selected_eligibility_expiry_count"], int
            )
            or observed_population["selected_eligibility_expiry_count"] < 0
        ):
            raise RuntimeError(f"{context}: canonical level tape binding")
        for hash_key in (
            "canonical_lifecycle_sha256",
            "canonical_episode_sha256",
            "canonical_identity_state_sha256",
            "canonical_outcome_stream_sha256",
            "canonical_event_tape_sha256",
            "final_fit_recurrence_population_sha256",
            "final_serve_recurrence_population_sha256",
            "canonical_fit_outcome_population_sha256",
            "selected_derived_outcome_population_sha256",
            "selected_derived_lifecycle_sha256",
            "selected_derived_episode_sha256",
            "selected_derived_state_sha256",
            "selected_derived_emission_sha256",
        ):
            value_hash = observed_population[hash_key]
            if (
                not isinstance(value_hash, str)
                or len(value_hash) != 64
                or any(char not in "0123456789abcdef" for char in value_hash)
            ):
                raise RuntimeError(f"{context}: canonical level tape hash")
    elif observed_population != dict(population_configuration or {}):
        raise RuntimeError(f"{context}: registry hyperfit binding")
    if (
        payload["schema_version"] != REGISTRY_HYPERPARAMETER_FIT_SCHEMA_V1
        or float(payload["selected_threshold_atr"])
        != float(selected_threshold_atr)
        or (
            learned_expiry_bars is not None
            and int(payload["learned_expiry_bars"]) != int(learned_expiry_bars)
        )
        or int(payload["outer_train_rows"]) != int(source_row_count)
        or int(payload["inner_fit_end_exclusive"])
        != int(inner_fit_end_exclusive)
        or source["declared_train_window_start"] != window_start
        or source["declared_train_window_end"] != window_end
        or payload["future_outcomes_usage"]
        != "TRAIN_hyperparameter_fit_only_not_apply_features"
    ):
        raise RuntimeError(f"{context}: registry hyperfit binding")


def _require_positive_finite_float(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
        raise RuntimeError(f"HTF_V4_V29_REGISTRY_CONSTANTS_INVALID: {label}={value!r}")
    out = float(value)
    if not math.isfinite(out) or out <= 0.0:
        raise RuntimeError(f"HTF_V4_V29_REGISTRY_CONSTANTS_INVALID: {label}={value!r}")
    return out


def require_v29_registry_constants(value: object) -> dict:
    """Validate the exact TRAIN-fitted V29 registry constants payload."""

    if not isinstance(value, Mapping) or not value:
        raise RuntimeError(
            "HTF_V4_V29_REGISTRY_CONSTANTS_MISSING: the V29 registry blocks "
            "require the TRAIN-fitted constants payload (no default exists)"
        )
    observed = dict(value)
    if set(observed) != _V29_REGISTRY_CONSTANTS_KEYS:
        raise RuntimeError(
            "HTF_V4_V29_REGISTRY_CONSTANTS_INVALID: exact keys differ "
            f"missing={sorted(_V29_REGISTRY_CONSTANTS_KEYS - set(observed))} "
            f"unexpected={sorted(set(observed) - _V29_REGISTRY_CONSTANTS_KEYS)}"
        )
    if observed["schema_version"] != V29_REGISTRY_CONSTANTS_SCHEMA_VERSION:
        raise RuntimeError(
            "HTF_V4_V29_REGISTRY_CONSTANTS_INVALID: schema_version="
            f"{observed['schema_version']!r}"
        )
    window_start = _require_canonical_utc_timestamp_label(
        observed["declared_train_window_start"],
        context=(
            "HTF_V4_V29_REGISTRY_CONSTANTS_INVALID: "
            "declared_train_window_start"
        ),
    )
    window_end = _require_canonical_utc_timestamp_label(
        observed["declared_train_window_end"],
        context=(
            "HTF_V4_V29_REGISTRY_CONSTANTS_INVALID: "
            "declared_train_window_end"
        ),
    )
    inner_window_end = _require_canonical_utc_timestamp_label(
        observed["declared_inner_fit_window_end"],
        context=(
            "HTF_V4_V29_REGISTRY_CONSTANTS_INVALID: "
            "declared_inner_fit_window_end"
        ),
    )
    if not (
        pd.Timestamp(window_start)
        < pd.Timestamp(inner_window_end)
        < pd.Timestamp(window_end)
    ):
        raise RuntimeError(
            "HTF_V4_V29_REGISTRY_CONSTANTS_INVALID: inner split must lie "
            "strictly inside the declared TRAIN window"
        )
    expected_tfs = tuple(MULTI_TF_RESAMPLE_RULES)
    # Exact key SET, canonical iteration order for the value checks. Key
    # insertion order is not semantic here: the payload legitimately transits
    # the repo's canonical sort_keys=True JSON serialization (e.g. inside the
    # hash-bound split manifests), which reorders mapping keys — demanding
    # insertion order made the first V29 smoke launch fail on a payload whose
    # content was exact (2026-08-12).
    for mapping_name in ("level_recurrence_threshold_atr", "trendline_band_atr"):
        mapping = observed[mapping_name]
        if not isinstance(mapping, Mapping) or set(mapping) != set(expected_tfs):
            raise RuntimeError(
                f"HTF_V4_V29_REGISTRY_CONSTANTS_INVALID: {mapping_name} must "
                f"declare exactly {expected_tfs}"
            )
        for tf_name in expected_tfs:
            _require_positive_finite_float(
                mapping[tf_name], label=f"{mapping_name}.{tf_name}"
            )
    for mapping_name in ("level_expiry_bars",):
        mapping = observed[mapping_name]
        if not isinstance(mapping, Mapping) or set(mapping) != set(expected_tfs):
            raise RuntimeError(
                f"HTF_V4_V29_REGISTRY_CONSTANTS_INVALID: {mapping_name} keys"
            )
        for tf_name in expected_tfs:
            if (
                isinstance(mapping[tf_name], bool)
                or not isinstance(mapping[tf_name], (int, np.integer))
                or int(mapping[tf_name]) <= 0
            ):
                raise RuntimeError(
                    f"HTF_V4_V29_REGISTRY_CONSTANTS_INVALID: {mapping_name}.{tf_name}"
                )
    seq_lens = observed["per_tf_seq_lens"]
    if not isinstance(seq_lens, Mapping) or set(seq_lens) != set(expected_tfs):
        raise RuntimeError(
            "HTF_V4_V29_REGISTRY_CONSTANTS_INVALID: per_tf_seq_lens must "
            f"declare exactly {expected_tfs}"
        )
    require_multi_tf_resolution_pyramid({tf: seq_lens[tf] for tf in expected_tfs})
    entry_m5 = observed["entry_m5"]
    if not isinstance(entry_m5, Mapping) or set(entry_m5) != _V29_REGISTRY_ENTRY_M5_KEYS:
        raise RuntimeError(
            "HTF_V4_V29_REGISTRY_CONSTANTS_INVALID: entry_m5 exact keys required"
        )
    entry_seq_len = entry_m5["seq_len"]
    if (
        isinstance(entry_seq_len, bool)
        or not isinstance(entry_seq_len, (int, np.integer))
        or int(entry_seq_len) <= 0
    ):
        raise RuntimeError(
            f"HTF_V4_V29_REGISTRY_CONSTANTS_INVALID: entry_m5.seq_len={entry_seq_len!r}"
        )
    _require_positive_finite_float(
        entry_m5["trendline_band_atr"], label="entry_m5.trendline_band_atr"
    )
    provenance = observed["provenance"]
    if (
        not isinstance(provenance, Mapping)
        or set(provenance) != _V29_REGISTRY_CONSTANTS_PROVENANCE_KEYS
    ):
        raise RuntimeError("HTF_V4_V29_REGISTRY_CONSTANTS_INVALID: provenance")
    n_train_rows = _require_nonnegative_int(
        provenance["n_train_m5_rows"],
        context="HTF_V4_V29_REGISTRY_CONSTANTS_INVALID: provenance binding",
    )
    if (
        provenance["schema_version"]
        != V29_REGISTRY_CONSTANTS_PROVENANCE_SCHEMA_VERSION
        or provenance["lane"] != "M5"
        or provenance["module"] != "gx1.features.htf_features"
        or provenance["payload_schema_version"]
        != V29_REGISTRY_CONSTANTS_SCHEMA_VERSION
        or provenance["fit_owner"]
        != "gx1.features.htf_features.fit_v29_registry_constants_from_m5"
        or provenance["declared_train_window_start"] != window_start
        or provenance["declared_train_window_end"] != window_end
        or provenance["declared_inner_fit_window_end"] != inner_window_end
        or n_train_rows <= 0
        or not isinstance(provenance["inner_fit_end_exclusive_by_clock"], Mapping)
        or set(provenance["inner_fit_end_exclusive_by_clock"]) != set(expected_tfs)
        or not isinstance(provenance["level_recurrence_threshold"], Mapping)
        or set(provenance["level_recurrence_threshold"]) != set(expected_tfs)
        or not isinstance(provenance["trendline_band"], Mapping)
        or set(provenance["trendline_band"]) != set(expected_tfs)
    ):
        raise RuntimeError(
            "HTF_V4_V29_REGISTRY_CONSTANTS_INVALID: provenance binding"
        )
    for tf_name in expected_tfs:
        inner_end = provenance["inner_fit_end_exclusive_by_clock"][tf_name]
        if (
            isinstance(inner_end, bool)
            or not isinstance(inner_end, (int, np.integer))
            or not 1 < int(inner_end) < int(
                provenance["level_recurrence_threshold"][tf_name].get(
                    "outer_train_rows", 0
                )
            ) - 1
        ):
            raise RuntimeError(
                "HTF_V4_V29_REGISTRY_CONSTANTS_INVALID: inner split binding"
            )
        _require_registry_hyperfit_provenance(
            provenance["level_recurrence_threshold"][tf_name],
            registry_kind="horizontal_level",
            timeframe=tf_name,
            window_start=window_start,
            window_end=window_end,
            selected_threshold_atr=float(
                observed["level_recurrence_threshold_atr"][tf_name]
            ),
            learned_expiry_bars=int(observed["level_expiry_bars"][tf_name]),
            source_row_count=int(
                provenance["level_recurrence_threshold"][tf_name][
                    "outer_train_rows"
                ]
            ),
            inner_fit_end_exclusive=int(inner_end),
            population_configuration=None,
            context="HTF_V4_V29_REGISTRY_CONSTANTS_INVALID",
        )
        _require_registry_hyperfit_provenance(
            provenance["trendline_band"][tf_name],
            registry_kind="trendline",
            timeframe=tf_name,
            window_start=window_start,
            window_end=window_end,
            selected_threshold_atr=float(observed["trendline_band_atr"][tf_name]),
            learned_expiry_bars=None,
            source_row_count=int(provenance["trendline_band"][tf_name]["outer_train_rows"]),
            inner_fit_end_exclusive=int(inner_end),
            population_configuration={
                "owner": "trendline_exact_runtime_candidate_population_v1",
                "seq_len": int(seq_lens[tf_name]),
                "swing_lookback": int(SWING_LOOKBACK),
            },
            context="HTF_V4_V29_REGISTRY_CONSTANTS_INVALID",
        )
    _require_registry_hyperfit_provenance(
        provenance["entry_m5_trendline_band"],
        registry_kind="trendline",
        timeframe="M5",
        window_start=window_start,
        window_end=window_end,
        selected_threshold_atr=float(entry_m5["trendline_band_atr"]),
        learned_expiry_bars=None,
        source_row_count=int(provenance["entry_m5_trendline_band"]["outer_train_rows"]),
        inner_fit_end_exclusive=int(
            provenance["inner_fit_end_exclusive_by_clock"]["M5"]
        ),
        population_configuration={
            "owner": "trendline_exact_runtime_candidate_population_v1",
            "seq_len": int(entry_seq_len),
            "swing_lookback": int(SWING_LOOKBACK),
        },
        context="HTF_V4_V29_REGISTRY_CONSTANTS_INVALID",
    )
    _require_contract_sha256(
        observed,
        context="HTF_V4_V29_REGISTRY_CONSTANTS_INVALID",
    )
    return observed


def load_v29_registry_constants_manifest(path) -> dict:
    """Load frozen V29 registry constants from an explicit JSON artifact.

    Accepts only a V4 cache ``manifest.json`` whose constants live under its
    ``v29_registry_constants`` key. Bare payloads and synthetic test evidence
    are forbidden at this production artifact boundary.
    """

    artifact = Path(path).expanduser()
    if artifact.is_symlink() or not artifact.is_file():
        raise RuntimeError(
            f"HTF_V4_V29_REGISTRY_CONSTANTS_ARTIFACT_MISSING: {artifact}"
        )
    artifact = artifact.resolve(strict=True)
    if artifact != artifact.parent / "manifest.json":
        raise RuntimeError(
            "HTF_V4_V29_REGISTRY_CONSTANTS_CONTAINER_REQUIRED"
        )
    cache = load_multi_tf_v4_cache(artifact.parent)
    return dict(cache.v29_registry_constants)


def load_volatility_squeeze_artifacts_from_cache_manifest(
    path,
) -> VolatilitySqueezeArtifactSet:
    """Load the exact squeeze artifact set bound by one current V4 cache."""

    artifact = Path(path).expanduser()
    if artifact.is_symlink() or not artifact.is_file():
        raise RuntimeError("HTF_V4_VOLATILITY_SQUEEZE_CACHE_MANIFEST_MISSING")
    try:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "HTF_V4_VOLATILITY_SQUEEZE_CACHE_MANIFEST_INVALID"
        ) from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != HTF_V4_CACHE_SCHEMA_VERSION
        or set(payload) != _HTF_V4_CACHE_MANIFEST_KEYS
    ):
        raise RuntimeError(
            "HTF_V4_VOLATILITY_SQUEEZE_CACHE_MANIFEST_INVALID"
        )
    return require_volatility_squeeze_artifact_binding(
        payload.get("volatility_squeeze_artifact_set")
    )


def fit_v29_registry_constants_from_m5(
    m5_df: pd.DataFrame,
    *,
    declared_train_window_start,
    declared_train_window_end,
    declared_inner_fit_window_end,
    source_provenance_by_clock: Mapping[str, Mapping[str, object]],
    per_tf_seq_lens: dict[str, int],
    entry_m5_seq_len: int,
) -> dict:
    """Fit the V29 registry constants once on the declared TRAIN window.

    ``m5_df`` is the exact native-M5 OHLCV source; only rows inside the closed
    interval ``[declared_train_window_start, declared_train_window_end]``
    participate (rule 18: fit on the physical TRAIN population, freeze, never
    refit).  Both bounds are required and are the only TRAIN-population
    authority here — the lower bound was missing until 2026-08-15, which
    silently fitted the four decision-bearing constants per clock on the whole
    source history instead of the declared TRAIN rows (rule 2g).  Every TF is
    resampled through the same closed-bar geometry the surface computation
    uses, so the fitted population equals the admitted population (rule 2g).
    Sample sizes and sampling bounds are recorded in the provenance payload
    (rule 2f).
    """

    _validate_m5_input(m5_df, require_volume=True)
    window_start = pd.Timestamp(declared_train_window_start)
    if window_start.tzinfo is None or window_start.utcoffset() != pd.Timedelta(0):
        raise RuntimeError(
            "HTF_V4_V29_REGISTRY_FIT_WINDOW_INVALID: "
            "declared_train_window_start must be timezone-aware UTC"
        )
    window_end = pd.Timestamp(declared_train_window_end)
    if window_end.tzinfo is None or window_end.utcoffset() != pd.Timedelta(0):
        raise RuntimeError(
            "HTF_V4_V29_REGISTRY_FIT_WINDOW_INVALID: declared_train_window_end "
            "must be timezone-aware UTC"
        )
    if window_start >= window_end:
        raise RuntimeError(
            "HTF_V4_V29_REGISTRY_FIT_WINDOW_INVALID: "
            "declared_train_window_start must precede declared_train_window_end"
        )
    inner_window_end = pd.Timestamp(declared_inner_fit_window_end)
    if (
        inner_window_end.tzinfo is None
        or inner_window_end.utcoffset() != pd.Timedelta(0)
        or not window_start < inner_window_end < window_end
    ):
        raise RuntimeError(
            "HTF_V4_V29_REGISTRY_FIT_INNER_WINDOW_INVALID"
        )
    if (
        isinstance(entry_m5_seq_len, bool)
        or not isinstance(entry_m5_seq_len, (int, np.integer))
        or int(entry_m5_seq_len) <= 0
    ):
        raise RuntimeError(
            f"HTF_V4_V29_REGISTRY_FIT_ENTRY_SEQ_LEN_INVALID: {entry_m5_seq_len!r}"
        )
    pyramid = require_multi_tf_resolution_pyramid(dict(per_tf_seq_lens))
    lengths = dict(pyramid["per_tf_seq_lens"])
    if (
        not isinstance(source_provenance_by_clock, Mapping)
        or set(source_provenance_by_clock) != set(MULTI_TF_RESAMPLE_RULES)
    ):
        raise RuntimeError("HTF_V4_V29_REGISTRY_FIT_SOURCE_PROVENANCE_INVALID")

    source = m5_df.copy(deep=False)
    source.index = source.index.as_unit("ns")
    train_source = source[
        (source.index >= window_start) & (source.index <= window_end)
    ]
    if train_source.empty:
        raise RuntimeError(
            "HTF_V4_V29_REGISTRY_FIT_WINDOW_EMPTY: no source rows inside the "
            f"declared TRAIN window [{window_start.isoformat()}, "
            f"{window_end.isoformat()}]"
        )
    expected_indices = build_multi_tf_v4_closed_timestamp_indices(
        train_source.index
    )
    window_start_label = str(window_start.isoformat())
    window_label = str(window_end.isoformat())
    inner_window_label = str(inner_window_end.isoformat())
    level_recurrence_threshold_atr: dict[str, float] = {}
    level_expiry_bars: dict[str, int] = {}
    trendline_band_atr: dict[str, float] = {}
    provenance: dict[str, object] = {
        "schema_version": V29_REGISTRY_CONSTANTS_PROVENANCE_SCHEMA_VERSION,
        "lane": "M5",
        "module": "gx1.features.htf_features",
        "payload_schema_version": V29_REGISTRY_CONSTANTS_SCHEMA_VERSION,
        "fit_owner": "gx1.features.htf_features.fit_v29_registry_constants_from_m5",
        "declared_train_window_start": window_start_label,
        "declared_train_window_end": window_label,
        "declared_inner_fit_window_end": inner_window_label,
        "n_train_m5_rows": int(len(train_source)),
        "inner_fit_end_exclusive_by_clock": {},
        "level_recurrence_threshold": {},
        "trendline_band": {},
    }
    entry_band: float | None = None
    entry_provenance: dict | None = None
    for tf_name in MULTI_TF_RESAMPLE_RULES:
        resampled = _resample_ohlcv(train_source, tf_name)
        resampled = resampled.dropna(subset=["open", "high", "low", "close"])
        resampled = resampled.loc[expected_indices[tf_name]]
        fit_frame = resampled[["high", "low", "close"]].copy()
        fit_frame["atr"] = _atr(
            resampled["high"], resampled["low"], resampled["close"], 14
        )
        inner_fit_end_exclusive = int(
            fit_frame.index.searchsorted(inner_window_end, side="right")
        )
        if not 1 < inner_fit_end_exclusive < len(fit_frame) - 1:
            raise RuntimeError(
                f"HTF_V4_V29_REGISTRY_FIT_INNER_SUPPORT_INVALID: {tf_name}"
            )
        provenance["inner_fit_end_exclusive_by_clock"][tf_name] = (
            inner_fit_end_exclusive
        )
        tol_provenance = fit_level_registry_hyperparameters_v1(
            fit_frame,
            tf=tf_name.lower(),
            inner_fit_end_exclusive=inner_fit_end_exclusive,
            source_provenance=source_provenance_by_clock[tf_name],
        )
        level_recurrence_threshold_atr[tf_name] = float(
            tol_provenance["selected_threshold_atr"]
        )
        level_expiry_bars[tf_name] = int(
            tol_provenance["learned_expiry_bars"]
        )
        provenance["level_recurrence_threshold"][tf_name] = tol_provenance
        band_payload = fit_trendline_registry_hyperparameters_v1(
            fit_frame,
            timeframe=tf_name,
            seq_len=int(lengths[tf_name]),
            inner_fit_end_exclusive=inner_fit_end_exclusive,
            source_provenance=source_provenance_by_clock[tf_name],
        )
        trendline_band_atr[tf_name] = float(
            band_payload["selected_threshold_atr"]
        )
        provenance["trendline_band"][tf_name] = band_payload
        if tf_name == "M5":
            entry_payload = fit_trendline_registry_hyperparameters_v1(
                fit_frame,
                timeframe="M5",
                seq_len=int(entry_m5_seq_len),
                inner_fit_end_exclusive=inner_fit_end_exclusive,
                source_provenance=source_provenance_by_clock[tf_name],
            )
            entry_band = float(entry_payload["selected_threshold_atr"])
            entry_provenance = entry_payload
    if entry_band is None or entry_provenance is None:
        raise RuntimeError("HTF_V4_V29_REGISTRY_FIT_ENTRY_M5_MISSING")
    provenance["entry_m5_trendline_band"] = entry_provenance
    constants = {
        "schema_version": V29_REGISTRY_CONSTANTS_SCHEMA_VERSION,
        "declared_train_window_start": window_start_label,
        "declared_train_window_end": window_label,
        "declared_inner_fit_window_end": inner_window_label,
        "level_recurrence_threshold_atr": level_recurrence_threshold_atr,
        "level_expiry_bars": level_expiry_bars,
        "trendline_band_atr": trendline_band_atr,
        "per_tf_seq_lens": {tf: int(lengths[tf]) for tf in MULTI_TF_RESAMPLE_RULES},
        "entry_m5": {
            "seq_len": int(entry_m5_seq_len),
            "trendline_band_atr": entry_band,
        },
        "provenance": provenance,
    }
    constants["contract_sha256"] = _contract_sha256_without_field(constants)
    return require_v29_registry_constants(constants)


# ---------------------------------------------------------------------------
# V29 M1-lane registry params — the Exit local M1 lane runs the same
# level/trendline local-layer blocks on the native M1 clock (the shared local
# layer in entry_model_native_feature_layers_v1), so it needs its own TRAIN
# fit on that clock (rule 2g: measure where the decision is made).  Same fit
# competing-risk hyperparameter owners, same
# rule-18/2f pattern as the M5 constants above.  The payload is frozen into
# the M1-enriched frame manifest (the M1-side hash-bound artifact) and
# consumed fail-closed by the M1 materializer.  No default exists anywhere.
# ---------------------------------------------------------------------------
# v8 / provenance v7 (2026-08-15): the exit_m1 block loses
# trendline_expiry_bars, mirroring the M5 constants payload above.
V29_REGISTRY_M1_LANE_PARAMS_SCHEMA_VERSION = (
    "htf_v4_v29_registry_m1_lane_params_v8"
)
V29_REGISTRY_M1_LANE_PROVENANCE_SCHEMA_VERSION = (
    "htf_v4_v29_registry_m1_lane_provenance_v7"
)
V29_REGISTRY_M1_LANE_MANIFEST_KEY = "v29_registry_m1_lane_params"
_V29_REGISTRY_M1_LANE_PARAMS_KEYS = frozenset(
    {
        "schema_version",
        "declared_train_window_start",
        "declared_train_window_end",
        "declared_inner_fit_window_end",
        "level_recurrence_threshold_atr",
        "level_expiry_bars",
        "exit_m1",
        "provenance",
        "contract_sha256",
    }
)
_V29_REGISTRY_EXIT_M1_KEYS = frozenset({"seq_len", "trendline_band_atr"})
_V29_REGISTRY_M1_LANE_PROVENANCE_KEYS = frozenset(
    {
        "schema_version",
        "lane",
        "module",
        "payload_schema_version",
        "fit_owner",
        "declared_train_window_start",
        "declared_train_window_end",
        "declared_inner_fit_window_end",
        "n_train_m1_rows",
        "inner_fit_end_exclusive",
        "level_recurrence_threshold",
        "trendline_band",
    }
)


def require_v29_registry_m1_lane_params(value: object) -> dict:
    """Validate the exact TRAIN-fitted V29 M1-lane registry params payload."""

    if not isinstance(value, Mapping) or not value:
        raise RuntimeError(
            "HTF_V4_V29_REGISTRY_M1_LANE_PARAMS_MISSING: the Exit M1 lane's "
            "registry layers require the TRAIN-fitted M1-lane params payload "
            "(no default exists)"
        )
    observed = dict(value)
    if set(observed) != _V29_REGISTRY_M1_LANE_PARAMS_KEYS:
        raise RuntimeError(
            "HTF_V4_V29_REGISTRY_M1_LANE_PARAMS_INVALID: exact keys differ "
            f"missing={sorted(_V29_REGISTRY_M1_LANE_PARAMS_KEYS - set(observed))} "
            f"unexpected={sorted(set(observed) - _V29_REGISTRY_M1_LANE_PARAMS_KEYS)}"
        )
    if observed["schema_version"] != V29_REGISTRY_M1_LANE_PARAMS_SCHEMA_VERSION:
        raise RuntimeError(
            "HTF_V4_V29_REGISTRY_M1_LANE_PARAMS_INVALID: schema_version="
            f"{observed['schema_version']!r}"
        )
    window_start = _require_canonical_utc_timestamp_label(
        observed["declared_train_window_start"],
        context=(
            "HTF_V4_V29_REGISTRY_M1_LANE_PARAMS_INVALID: "
            "declared_train_window_start"
        ),
    )
    window_end = _require_canonical_utc_timestamp_label(
        observed["declared_train_window_end"],
        context=(
            "HTF_V4_V29_REGISTRY_M1_LANE_PARAMS_INVALID: "
            "declared_train_window_end"
        ),
    )
    inner_window_end = _require_canonical_utc_timestamp_label(
        observed["declared_inner_fit_window_end"],
        context=(
            "HTF_V4_V29_REGISTRY_M1_LANE_PARAMS_INVALID: "
            "declared_inner_fit_window_end"
        ),
    )
    if not (
        pd.Timestamp(window_start)
        < pd.Timestamp(inner_window_end)
        < pd.Timestamp(window_end)
    ):
        raise RuntimeError(
            "HTF_V4_V29_REGISTRY_M1_LANE_PARAMS_INVALID: inner split"
        )
    _require_positive_finite_float(
        observed["level_recurrence_threshold_atr"],
        label="level_recurrence_threshold_atr",
    )
    if (
        isinstance(observed["level_expiry_bars"], bool)
        or not isinstance(observed["level_expiry_bars"], (int, np.integer))
        or int(observed["level_expiry_bars"]) <= 0
    ):
        raise RuntimeError(
            "HTF_V4_V29_REGISTRY_M1_LANE_PARAMS_INVALID: level_expiry_bars"
        )
    exit_m1 = observed["exit_m1"]
    if (
        not isinstance(exit_m1, Mapping)
        or set(exit_m1) != _V29_REGISTRY_EXIT_M1_KEYS
    ):
        raise RuntimeError(
            "HTF_V4_V29_REGISTRY_M1_LANE_PARAMS_INVALID: exit_m1 exact keys "
            "required"
        )
    exit_seq_len = exit_m1["seq_len"]
    if (
        isinstance(exit_seq_len, bool)
        or not isinstance(exit_seq_len, (int, np.integer))
        or int(exit_seq_len) <= 0
    ):
        raise RuntimeError(
            "HTF_V4_V29_REGISTRY_M1_LANE_PARAMS_INVALID: "
            f"exit_m1.seq_len={exit_seq_len!r}"
        )
    _require_positive_finite_float(
        exit_m1["trendline_band_atr"], label="exit_m1.trendline_band_atr"
    )
    provenance = observed["provenance"]
    if (
        not isinstance(provenance, Mapping)
        or set(provenance) != _V29_REGISTRY_M1_LANE_PROVENANCE_KEYS
    ):
        raise RuntimeError(
            "HTF_V4_V29_REGISTRY_M1_LANE_PARAMS_INVALID: provenance"
        )
    n_train_rows = _require_nonnegative_int(
        provenance["n_train_m1_rows"],
        context=(
            "HTF_V4_V29_REGISTRY_M1_LANE_PARAMS_INVALID: provenance binding"
        ),
    )
    if (
        provenance["schema_version"]
        != V29_REGISTRY_M1_LANE_PROVENANCE_SCHEMA_VERSION
        or provenance["lane"] != "M1"
        or provenance["module"] != "gx1.features.htf_features"
        or provenance["payload_schema_version"]
        != V29_REGISTRY_M1_LANE_PARAMS_SCHEMA_VERSION
        or provenance["fit_owner"]
        != "gx1.features.htf_features.fit_v29_registry_m1_lane_params_from_m1"
        or provenance["declared_train_window_start"] != window_start
        or provenance["declared_train_window_end"] != window_end
        or provenance["declared_inner_fit_window_end"] != inner_window_end
        or n_train_rows <= 0
    ):
        raise RuntimeError(
            "HTF_V4_V29_REGISTRY_M1_LANE_PARAMS_INVALID: provenance binding"
        )
    inner_end = provenance["inner_fit_end_exclusive"]
    if (
        isinstance(inner_end, bool)
        or not isinstance(inner_end, (int, np.integer))
        or not 1 < int(inner_end) < n_train_rows - 1
    ):
        raise RuntimeError(
            "HTF_V4_V29_REGISTRY_M1_LANE_PARAMS_INVALID: inner split binding"
        )
    _require_registry_hyperfit_provenance(
        provenance["level_recurrence_threshold"],
        registry_kind="horizontal_level",
        timeframe="M1",
        window_start=window_start,
        window_end=window_end,
        selected_threshold_atr=float(observed["level_recurrence_threshold_atr"]),
        learned_expiry_bars=int(observed["level_expiry_bars"]),
        source_row_count=n_train_rows,
        inner_fit_end_exclusive=int(inner_end),
        population_configuration=None,
        context="HTF_V4_V29_REGISTRY_M1_LANE_PARAMS_INVALID",
    )
    _require_registry_hyperfit_provenance(
        provenance["trendline_band"],
        registry_kind="trendline",
        timeframe="M1",
        window_start=window_start,
        window_end=window_end,
        selected_threshold_atr=float(exit_m1["trendline_band_atr"]),
        learned_expiry_bars=None,
        source_row_count=n_train_rows,
        inner_fit_end_exclusive=int(inner_end),
        population_configuration={
            "owner": "trendline_exact_runtime_candidate_population_v1",
            "seq_len": int(exit_seq_len),
            "swing_lookback": int(SWING_LOOKBACK),
        },
        context="HTF_V4_V29_REGISTRY_M1_LANE_PARAMS_INVALID",
    )
    _require_contract_sha256(
        observed,
        context="HTF_V4_V29_REGISTRY_M1_LANE_PARAMS_INVALID",
    )
    return observed


def load_v29_registry_m1_lane_params_manifest(path) -> dict:
    """Load frozen V29 M1-lane registry params from an explicit JSON artifact.

    Accepts only the M1-enriched frame ``manifest.json`` carrying the params
    under ``v29_registry_m1_lane_params``. Bare payloads and synthetic test
    evidence are forbidden at this production artifact boundary.
    """

    artifact = Path(path).expanduser()
    if artifact.is_symlink() or not artifact.is_file():
        raise RuntimeError(
            f"HTF_V4_V29_REGISTRY_M1_LANE_PARAMS_ARTIFACT_MISSING: {artifact}"
        )
    artifact = artifact.resolve(strict=True)
    try:
        payload = json.loads(
            artifact.read_text(encoding="utf-8"),
            object_pairs_hook=_json_object_without_duplicate_keys,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise RuntimeError(
            f"HTF_V4_V29_REGISTRY_M1_LANE_PARAMS_ARTIFACT_INVALID: {artifact}"
        ) from exc
    if (
        not isinstance(payload, dict)
        or V29_REGISTRY_M1_LANE_MANIFEST_KEY not in payload
    ):
        raise RuntimeError(
            "HTF_V4_V29_REGISTRY_M1_LANE_PARAMS_CONTAINER_REQUIRED"
        )
    from gx1.contracts.entry_exit_feature_base_v1 import (
        ENTRY_EXIT_ENRICHED_CAUSAL_FRAME_SCHEMA_VERSION,
        EXIT_DECISION_BAR_SECONDS,
        require_entry_exit_shared_feature_base_contract,
    )

    manifest_without_hash = dict(payload)
    declared_manifest_sha256 = manifest_without_hash.pop(
        "manifest_sha256", None
    )
    output_raw = payload.get("output_parquet")
    output_sha256 = payload.get("output_parquet_sha256")
    if not isinstance(output_raw, str):
        raise RuntimeError(
            "HTF_V4_V29_REGISTRY_M1_LANE_PARAMS_CONTAINER_REQUIRED"
        )
    output = Path(output_raw).expanduser()
    if (
        not output.is_absolute()
        or output.is_symlink()
        or not output.is_file()
        or output.resolve(strict=True) != output
        or artifact != Path(f"{output}.manifest.json")
        or payload.get("schema_version")
        != ENTRY_EXIT_ENRICHED_CAUSAL_FRAME_SCHEMA_VERSION
        or payload.get("decision") != "PASS"
        or payload.get("timeframe") != "M1"
        or payload.get("base_bar_seconds") != EXIT_DECISION_BAR_SECONDS
        or not isinstance(output_sha256, str)
        or len(output_sha256) != 64
        or any(character not in "0123456789abcdef" for character in output_sha256)
        or _sha256_file_exact(output) != output_sha256
        or declared_manifest_sha256
        != hashlib.sha256(
            json.dumps(
                manifest_without_hash,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
    ):
        raise RuntimeError(
            "HTF_V4_V29_REGISTRY_M1_LANE_PARAMS_CONTAINER_IDENTITY_INVALID"
        )
    require_entry_exit_shared_feature_base_contract(
        payload.get("shared_feature_base_contract"),
        context="HTF_V4_V29_REGISTRY_M1_LANE_PARAMS_CONTAINER",
    )
    params = require_v29_registry_m1_lane_params(
        payload[V29_REGISTRY_M1_LANE_MANIFEST_KEY]
    )
    return params


def fit_v29_registry_m1_lane_params_from_m1(
    m1_df: pd.DataFrame,
    *,
    declared_train_window_start,
    declared_train_window_end,
    declared_inner_fit_window_end,
    source_provenance: Mapping[str, object],
    exit_m1_seq_len: int,
) -> dict:
    """Fit the Exit M1-lane registry params once on the declared TRAIN window.

    ``m1_df`` is the exact native-M1 OHLCV source; only rows inside the closed
    interval ``[declared_train_window_start, declared_train_window_end]``
    participate (rule 18: fit on the physical TRAIN population, freeze, never
    refit).  Both bounds are required and are the only TRAIN-population
    authority here — the lower bound was missing until 2026-08-15, which
    silently fitted the decision-bearing constants on the whole source history
    instead of the declared TRAIN rows (rule 2g).  The fit population is the native
    M1 clock itself — the same clock, ATR convention (``_atr``, 14) and pivot
    admission the shared local layer uses at serve (rule 2g).  The trendline
    candidate window is the Exit model sequence length (named contract
    constant, mirroring the entry-M5 lane's use of the Entry sequence
    length).  Sample sizes and sampling bounds are recorded per fit owner
    (rule 2f).
    """

    _validate_m5_input(
        m1_df,
        require_volume=False,
        bar_duration=pd.Timedelta(minutes=1),
    )
    window_start = pd.Timestamp(declared_train_window_start)
    if window_start.tzinfo is None or window_start.utcoffset() != pd.Timedelta(0):
        raise RuntimeError(
            "HTF_V4_V29_REGISTRY_M1_FIT_WINDOW_INVALID: "
            "declared_train_window_start must be timezone-aware UTC"
        )
    window_end = pd.Timestamp(declared_train_window_end)
    if window_end.tzinfo is None or window_end.utcoffset() != pd.Timedelta(0):
        raise RuntimeError(
            "HTF_V4_V29_REGISTRY_M1_FIT_WINDOW_INVALID: "
            "declared_train_window_end must be timezone-aware UTC"
        )
    if window_start >= window_end:
        raise RuntimeError(
            "HTF_V4_V29_REGISTRY_M1_FIT_WINDOW_INVALID: "
            "declared_train_window_start must precede declared_train_window_end"
        )
    inner_window_end = pd.Timestamp(declared_inner_fit_window_end)
    if (
        inner_window_end.tzinfo is None
        or inner_window_end.utcoffset() != pd.Timedelta(0)
        or not window_start < inner_window_end < window_end
    ):
        raise RuntimeError("HTF_V4_V29_REGISTRY_M1_FIT_INNER_WINDOW_INVALID")
    if (
        isinstance(exit_m1_seq_len, bool)
        or not isinstance(exit_m1_seq_len, (int, np.integer))
        or int(exit_m1_seq_len) <= 0
    ):
        raise RuntimeError(
            "HTF_V4_V29_REGISTRY_M1_FIT_EXIT_SEQ_LEN_INVALID: "
            f"{exit_m1_seq_len!r}"
        )
    source = m1_df.copy(deep=False)
    source.index = source.index.as_unit("ns")
    train_source = source[
        (source.index >= window_start) & (source.index <= window_end)
    ]
    if train_source.empty:
        raise RuntimeError(
            "HTF_V4_V29_REGISTRY_M1_FIT_WINDOW_EMPTY: no source rows inside "
            f"the declared TRAIN window [{window_start.isoformat()}, "
            f"{window_end.isoformat()}]"
        )
    window_start_label = str(window_start.isoformat())
    window_label = str(window_end.isoformat())
    inner_window_label = str(inner_window_end.isoformat())
    fit_frame = train_source[["high", "low", "close"]].copy()
    fit_frame["atr"] = _atr(
        train_source["high"], train_source["low"], train_source["close"], 14
    )
    inner_fit_end_exclusive = int(
        fit_frame.index.searchsorted(inner_window_end, side="right")
    )
    if not 1 < inner_fit_end_exclusive < len(fit_frame) - 1:
        raise RuntimeError("HTF_V4_V29_REGISTRY_M1_FIT_INNER_SUPPORT_INVALID")
    tol_provenance = fit_level_registry_hyperparameters_v1(
        fit_frame,
        tf="m1",
        inner_fit_end_exclusive=inner_fit_end_exclusive,
        source_provenance=source_provenance,
    )
    band_payload = fit_trendline_registry_hyperparameters_v1(
        fit_frame,
        timeframe="M1",
        seq_len=int(exit_m1_seq_len),
        inner_fit_end_exclusive=inner_fit_end_exclusive,
        source_provenance=source_provenance,
    )
    params = {
        "schema_version": V29_REGISTRY_M1_LANE_PARAMS_SCHEMA_VERSION,
        "declared_train_window_start": window_start_label,
        "declared_train_window_end": window_label,
        "declared_inner_fit_window_end": inner_window_label,
        "level_recurrence_threshold_atr": float(
            tol_provenance["selected_threshold_atr"]
        ),
        "level_expiry_bars": int(tol_provenance["learned_expiry_bars"]),
        "exit_m1": {
            "seq_len": int(exit_m1_seq_len),
            "trendline_band_atr": float(
                band_payload["selected_threshold_atr"]
            ),
        },
        "provenance": {
            "schema_version": V29_REGISTRY_M1_LANE_PROVENANCE_SCHEMA_VERSION,
            "lane": "M1",
            "module": "gx1.features.htf_features",
            "payload_schema_version": V29_REGISTRY_M1_LANE_PARAMS_SCHEMA_VERSION,
            "fit_owner": (
                "gx1.features.htf_features."
                "fit_v29_registry_m1_lane_params_from_m1"
            ),
            "declared_train_window_start": window_start_label,
            "declared_train_window_end": window_label,
            "declared_inner_fit_window_end": inner_window_label,
            "n_train_m1_rows": int(len(train_source)),
            "inner_fit_end_exclusive": inner_fit_end_exclusive,
            "level_recurrence_threshold": tol_provenance,
            "trendline_band": band_payload,
        },
    }
    params["contract_sha256"] = _contract_sha256_without_field(params)
    return require_v29_registry_m1_lane_params(params)


def build_multi_tf_v4_liveness_contract(
    features: dict[str, pd.DataFrame],
) -> dict[str, object]:
    """Prove every V4 field is finite, variable and non-duplicated on every TF."""

    if tuple(features) != tuple(MULTI_TF_RESAMPLE_RULES):
        raise RuntimeError(
            "HTF_V4_LIVENESS_TIMEFRAME_ORDER_INVALID: exact "
            "M5/M15/H1/H4/D1 required"
        )
    failures: list[str] = []
    timeframe_rows: dict[str, object] = {}
    for tf_name in MULTI_TF_RESAMPLE_RULES:
        frame = features[tf_name]
        if (
            not isinstance(frame, pd.DataFrame)
            or tuple(frame.columns) != MULTI_TF_PER_BAR_FEATURES_V4
            or frame.attrs.get("htf_feature_contract")
            != HTF_V4_MATRIX_CONTRACT
        ):
            raise RuntimeError(
                f"HTF_V4_LIVENESS_SURFACE_INVALID: {tf_name}"
            )
        values = np.asarray(frame.attrs.get("feats_np"))
        warmup_rows = frame.attrs.get("causal_warmup_rows")
        if (
            values.dtype != np.dtype(np.float32)
            or values.shape
            != (len(frame), MULTI_TF_FEATURE_COUNT_V4)
            or isinstance(warmup_rows, bool)
            or not isinstance(warmup_rows, (int, np.integer))
            or not 0 <= int(warmup_rows) < len(frame)
        ):
            raise RuntimeError(
                f"HTF_V4_LIVENESS_ARRAY_INVALID: {tf_name}"
            )
        warmup = int(warmup_rows)
        live = values[warmup:].astype(np.float64, copy=False)
        if not np.isfinite(live).all():
            failures.append(f"{tf_name}:nonfinite_post_warmup")
        field_stats: dict[str, object] = {}
        column_hash_owner: dict[str, str] = {}
        duplicate_pairs: list[list[str]] = []
        constant_candidates: list[str] = []
        for index, feature_name in enumerate(MULTI_TF_PER_BAR_FEATURES_V4):
            column = live[:, index]
            unique_count = int(np.unique(column).size)
            standard_deviation = float(np.std(column, dtype=np.float64))
            nonzero_fraction = float(np.mean(np.abs(column) > 1e-12))
            digest = hashlib.sha256(
                np.ascontiguousarray(column).view(np.uint8)
            ).hexdigest()
            if unique_count <= 1 or standard_deviation <= 0.0:
                constant_candidates.append(feature_name)
            prior = column_hash_owner.get(digest)
            if prior is not None:
                pair = (prior, feature_name)
                if pair not in HTF_V4_DECLARED_ALIAS_PAIRS:
                    duplicate_pairs.append([prior, feature_name])
            else:
                column_hash_owner[digest] = feature_name
            field_stats[feature_name] = {
                "unique_count": unique_count,
                "mean": float(np.mean(column, dtype=np.float64)),
                "std": standard_deviation,
                "minimum": float(np.min(column)),
                "maximum": float(np.max(column)),
                "nonzero_fraction": nonzero_fraction,
                "values_sha256": digest,
            }
        constant_fields = list(constant_candidates)
        if constant_fields:
            failures.append(
                f"{tf_name}:constant_fields={constant_fields}"
            )
        if duplicate_pairs:
            failures.append(
                f"{tf_name}:exact_duplicate_fields={duplicate_pairs}"
            )
        timeframe_rows[tf_name] = {
            "rows": int(len(frame)),
            "warmup_rows": warmup,
            "live_rows": int(len(live)),
            "constant_fields": constant_fields,
            "exact_duplicate_pairs": duplicate_pairs,
            "fields": field_stats,
        }
    payload: dict[str, object] = {
        "schema_version": HTF_V4_FULL_INPUT_LIVENESS_SCHEMA_VERSION,
        "matrix_contract": HTF_V4_MATRIX_CONTRACT,
        "feature_names_sha256": MULTI_TF_FEATURE_NAMES_SHA256_V4,
        "timeframe_order": list(MULTI_TF_RESAMPLE_RULES),
        "decision": "PASS" if not failures else "FAIL",
        "failures": failures,
        "timeframes": timeframe_rows,
    }
    payload["contract_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return payload


def require_multi_tf_v4_liveness_contract(
    value: object,
) -> dict[str, object]:
    """Validate the exact immutable V4 per-field/per-timeframe proof."""

    if not isinstance(value, dict):
        raise RuntimeError("HTF_V4_LIVENESS_CONTRACT_MISSING")
    expected_keys = {
        "schema_version",
        "matrix_contract",
        "feature_names_sha256",
        "timeframe_order",
        "decision",
        "failures",
        "timeframes",
        "contract_sha256",
    }
    if set(value) != expected_keys:
        raise RuntimeError("HTF_V4_LIVENESS_CONTRACT_KEYS_INVALID")
    identity_payload = {
        key: item for key, item in value.items() if key != "contract_sha256"
    }
    expected_sha = hashlib.sha256(
        json.dumps(
            identity_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if value.get("contract_sha256") != expected_sha:
        raise RuntimeError("HTF_V4_LIVENESS_CONTRACT_IDENTITY_INVALID")
    if (
        value.get("schema_version") != HTF_V4_FULL_INPUT_LIVENESS_SCHEMA_VERSION
        or value.get("matrix_contract") != HTF_V4_MATRIX_CONTRACT
        or value.get("feature_names_sha256")
        != MULTI_TF_FEATURE_NAMES_SHA256_V4
        or value.get("timeframe_order") != list(MULTI_TF_RESAMPLE_RULES)
        or value.get("decision") != "PASS"
        or value.get("failures") != []
    ):
        raise RuntimeError("HTF_V4_LIVENESS_CONTRACT_DECISION_INVALID")
    timeframes = value.get("timeframes")
    if not isinstance(timeframes, dict) or set(timeframes) != set(
        MULTI_TF_RESAMPLE_RULES
    ):
        raise RuntimeError("HTF_V4_LIVENESS_TIMEFRAME_ORDER_INVALID")
    expected_tf_keys = {
        "rows",
        "warmup_rows",
        "live_rows",
        "constant_fields",
        "exact_duplicate_pairs",
        "fields",
    }
    expected_stat_keys = {
        "unique_count",
        "mean",
        "std",
        "minimum",
        "maximum",
        "nonzero_fraction",
        "values_sha256",
    }
    for tf_name, row in timeframes.items():
        if not isinstance(row, dict) or set(row) != expected_tf_keys:
            raise RuntimeError(f"HTF_V4_LIVENESS_TF_KEYS_INVALID: {tf_name}")
        rows = row.get("rows")
        warmup = row.get("warmup_rows")
        live_rows = row.get("live_rows")
        if (
            isinstance(rows, bool)
            or not isinstance(rows, int)
            or rows <= 0
            or isinstance(warmup, bool)
            or not isinstance(warmup, int)
            or not 0 <= warmup < rows
            or live_rows != rows - warmup
            or row.get("constant_fields") != []
            or row.get("exact_duplicate_pairs") != []
        ):
            raise RuntimeError(f"HTF_V4_LIVENESS_TF_DECISION_INVALID: {tf_name}")
        fields = row.get("fields")
        if not isinstance(fields, dict) or set(fields) != set(
            MULTI_TF_PER_BAR_FEATURES_V4
        ):
            raise RuntimeError(f"HTF_V4_LIVENESS_FIELDS_INVALID: {tf_name}")
        for field_name, stats in fields.items():
            if not isinstance(stats, dict) or set(stats) != expected_stat_keys:
                raise RuntimeError(
                    f"HTF_V4_LIVENESS_STATS_KEYS_INVALID: {tf_name}:{field_name}"
                )
            unique_count = stats.get("unique_count")
            numeric = [
                stats.get(name)
                for name in (
                    "mean",
                    "std",
                    "minimum",
                    "maximum",
                    "nonzero_fraction",
                )
            ]
            if (
                isinstance(unique_count, bool)
                or not isinstance(unique_count, int)
                or unique_count <= 1
                or any(
                    isinstance(item, bool)
                    or not isinstance(item, (int, float))
                    or not math.isfinite(float(item))
                    for item in numeric
                )
                or float(stats["std"]) <= 0.0
                or not 0.0 < float(stats["nonzero_fraction"]) <= 1.0
                or float(stats["minimum"]) > float(stats["maximum"])
                or not isinstance(stats.get("values_sha256"), str)
                or len(stats["values_sha256"]) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in stats["values_sha256"]
                )
            ):
                raise RuntimeError(
                    f"HTF_V4_LIVENESS_STATS_INVALID: {tf_name}:{field_name}"
                )
    return value


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Compatibility route to the one canonical Wilder RSI owner."""

    try:
        return wilder_rsi(close, n)
    except RuntimeError as exc:
        message = str(exc)
        if "WILDER_RSI_PERIOD_INVALID" in message:
            raise RuntimeError("HTF_RSI_PERIOD_INVALID") from exc
        if "WILDER_RSI_SOURCE_NONFINITE" in message:
            raise RuntimeError("HTF_RSI_SOURCE_NONFINITE") from exc
        raise


def _rolling_vwap(close: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    """Rolling N-bar VWAP from observed volume only."""
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        raise RuntimeError("HTF_V4_ROLLING_VWAP_WINDOW_INVALID")
    if not (
        isinstance(close, pd.Series)
        and isinstance(volume, pd.Series)
        and close.index.equals(volume.index)
    ):
        raise RuntimeError("HTF_V4_ROLLING_VWAP_SOURCE_INVALID")
    close_values = close.to_numpy(dtype=np.float64)
    if not np.isfinite(close_values).all():
        raise RuntimeError("HTF_V4_ROLLING_VWAP_SOURCE_INVALID")
    if volume.isna().any() or (~np.isfinite(volume.to_numpy(dtype=np.float64))).any():
        raise RuntimeError("HTF_V4_VOLUME_SOURCE_INVALID: rolling VWAP volume is non-finite")
    if (volume <= 0.0).any():
        raise RuntimeError("HTF_V4_VOLUME_SOURCE_INVALID: rolling VWAP volume must be positive")
    pv = close * volume
    pv_sum = pv.rolling(window, min_periods=window).sum()
    v_sum = volume.rolling(window, min_periods=window).sum()
    return pv_sum / v_sum


_SESSION_VWAP_STATE_SCHEMA_VERSION = "htf_v4_session_vwap_state_v1"
_SESSION_VWAP_STATE_KEYS = frozenset(
    {
        "schema_version",
        "clock_schema_version",
        "bar_duration_ns",
        "last_index_ns",
        "last_session_id",
        "price_volume_sum",
        "volume_sum",
    }
)


def _session_vwap(
    close: pd.Series,
    volume: pd.Series,
    *,
    bar_duration: pd.Timedelta,
    state: Mapping | None = None,
    return_state: bool = False,
):
    """Causal VWAP on the shared UTC trading-session clock.

    The recurrence is identical for one-shot, prefix, and chunked execution.
    Missing weekend/holiday rows advance the session id without synthesizing
    observations; the first observed row in a new id resets the accumulators.
    """

    if not isinstance(close, pd.Series) or not isinstance(volume, pd.Series):
        raise RuntimeError("HTF_V4_SESSION_VWAP_SOURCE_INVALID")
    if close.empty or volume.empty or not close.index.equals(volume.index):
        raise RuntimeError("HTF_V4_SESSION_VWAP_SOURCE_GEOMETRY_INVALID")
    if not isinstance(close.index, pd.DatetimeIndex):
        raise RuntimeError("HTF_V4_SESSION_VWAP_CLOCK_INVALID")
    if not isinstance(bar_duration, pd.Timedelta) or bar_duration <= pd.Timedelta(0):
        raise RuntimeError("HTF_V4_SESSION_VWAP_BAR_DURATION_INVALID")
    close_values = close.to_numpy(dtype=np.float64)
    volume_values = volume.to_numpy(dtype=np.float64)
    if not np.isfinite(close_values).all():
        raise RuntimeError("HTF_V4_SESSION_VWAP_SOURCE_INVALID: close is non-finite")
    if not np.isfinite(volume_values).all():
        raise RuntimeError(
            "HTF_V4_VOLUME_SOURCE_INVALID: session VWAP volume is non-finite"
        )
    if np.any(volume_values <= 0.0):
        raise RuntimeError(
            "HTF_V4_VOLUME_SOURCE_INVALID: session VWAP volume must be positive"
        )
    session_ids = trading_session_id_vectorized(
        close.index,
        context="HTF_V4_SESSION_VWAP",
    )
    bar_duration_ns = int(bar_duration.value)
    grid_offset_ns = int((TRADING_SESSION_BOUNDARY_OFFSET % bar_duration).value)
    if np.any((close.index.asi8 - grid_offset_ns) % bar_duration_ns != 0):
        raise RuntimeError("HTF_V4_SESSION_VWAP_BAR_OFF_GRID")

    if state is None:
        previous_index_ns: int | None = None
        previous_session_id: int | None = None
        price_volume_sum = 0.0
        volume_sum = 0.0
    else:
        if not isinstance(state, Mapping) or set(state) != _SESSION_VWAP_STATE_KEYS:
            raise RuntimeError("HTF_V4_SESSION_VWAP_STATE_INVALID")
        if (
            state.get("schema_version") != _SESSION_VWAP_STATE_SCHEMA_VERSION
            or state.get("clock_schema_version")
            != TRADING_SESSION_CLOCK_SCHEMA_VERSION
            or state.get("bar_duration_ns") != bar_duration_ns
        ):
            raise RuntimeError("HTF_V4_SESSION_VWAP_STATE_CONTRACT_MISMATCH")
        previous_index_ns = state.get("last_index_ns")
        previous_session_id = state.get("last_session_id")
        price_volume_sum = state.get("price_volume_sum")
        volume_sum = state.get("volume_sum")
        integer_state = (previous_index_ns, previous_session_id)
        numeric_state = (price_volume_sum, volume_sum)
        if (
            any(isinstance(value, (bool, np.bool_)) for value in integer_state)
            or not all(isinstance(value, (int, np.integer)) for value in integer_state)
            or any(isinstance(value, (bool, np.bool_)) for value in numeric_state)
            or not all(
                isinstance(value, (int, float, np.integer, np.floating))
                for value in numeric_state
            )
            or not np.isfinite(np.asarray(numeric_state, dtype=np.float64)).all()
            or float(volume_sum) <= 0.0
            or int(close.index.asi8[0]) <= int(previous_index_ns)
        ):
            raise RuntimeError("HTF_V4_SESSION_VWAP_STATE_INVALID")
        previous_index_ns = int(previous_index_ns)
        previous_session_id = int(previous_session_id)
        price_volume_sum = float(price_volume_sum)
        volume_sum = float(volume_sum)

    observed = np.empty(len(close_values), dtype=np.float64)
    for row, (session_id, price, row_volume) in enumerate(
        zip(session_ids, close_values, volume_values, strict=True)
    ):
        current_session_id = int(session_id)
        if previous_session_id != current_session_id:
            price_volume_sum = 0.0
            volume_sum = 0.0
        price_volume_sum += float(price) * float(row_volume)
        volume_sum += float(row_volume)
        observed[row] = price_volume_sum / volume_sum
        previous_session_id = current_session_id

    result = pd.Series(observed, index=close.index, name=close.name)
    next_state = {
        "schema_version": _SESSION_VWAP_STATE_SCHEMA_VERSION,
        "clock_schema_version": TRADING_SESSION_CLOCK_SCHEMA_VERSION,
        "bar_duration_ns": bar_duration_ns,
        "last_index_ns": int(close.index.asi8[-1]),
        "last_session_id": int(previous_session_id),
        "price_volume_sum": float(price_volume_sum),
        "volume_sum": float(volume_sum),
    }
    return (result, next_state) if return_state else result


def _adx14(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    n: int = 14,
) -> tuple[pd.Series, pd.Series]:
    """Classic Wilder ADX and signed DI spread with SMA/Wilder seeds.

    Returns ``(adx, di_spread_signed)``.  V30 emission win (2026-08-13):
    ``plus_di``/``minus_di`` were computed here and discarded — the |...| in
    the DX numerator folded the trend DIRECTION out of the surface.  The
    signed normalized spread ``(plus_di - minus_di) / (plus_di + minus_di)``
    (Wilder's own DX quantity without the absolute value, algebraically
    bounded in [-1, 1]) is now emitted next to raw ``adx14``. A zero
    denominator (no directional movement observed yet) is honest NaN. The
    first DI/DX observation is row ``n`` after smoothing rows 1..n.
    The first ADX is the arithmetic mean of DX rows n..2n-1 and is emitted at
    row ``2n-1``; subsequent values use Wilder's recurrence. Zero TR or DI
    sums are unavailable, never parked on an epsilon.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise RuntimeError("HTF_ADX_PERIOD_INVALID")
    if not (
        isinstance(high, pd.Series)
        and isinstance(low, pd.Series)
        and isinstance(close, pd.Series)
        and high.index.equals(low.index)
        and high.index.equals(close.index)
    ):
        raise RuntimeError("HTF_ADX_SOURCE_INVALID")
    highs = high.to_numpy(dtype=np.float64)
    lows = low.to_numpy(dtype=np.float64)
    closes = close.to_numpy(dtype=np.float64)
    if (
        not np.isfinite(highs).all()
        or not np.isfinite(lows).all()
        or not np.isfinite(closes).all()
        or np.any(highs < lows)
        or np.any(highs < closes)
        or np.any(lows > closes)
    ):
        raise RuntimeError("HTF_ADX_SOURCE_INVALID")
    rows = len(closes)
    tr = np.full(rows, np.nan, dtype=np.float64)
    plus_dm = np.zeros(rows, dtype=np.float64)
    minus_dm = np.zeros(rows, dtype=np.float64)
    if rows:
        tr[0] = highs[0] - lows[0]
    for row in range(1, rows):
        tr[row] = max(
            highs[row] - lows[row],
            abs(highs[row] - closes[row - 1]),
            abs(lows[row] - closes[row - 1]),
        )
        up = highs[row] - highs[row - 1]
        down = lows[row - 1] - lows[row]
        plus_dm[row] = up if up > down and up > 0.0 else 0.0
        minus_dm[row] = down if down > up and down > 0.0 else 0.0

    smoothed_tr = np.full(rows, np.nan, dtype=np.float64)
    smoothed_plus = np.full(rows, np.nan, dtype=np.float64)
    smoothed_minus = np.full(rows, np.nan, dtype=np.float64)
    if rows > n:
        smoothed_tr[n] = float(np.sum(tr[1 : n + 1], dtype=np.float64))
        smoothed_plus[n] = float(
            np.sum(plus_dm[1 : n + 1], dtype=np.float64)
        )
        smoothed_minus[n] = float(
            np.sum(minus_dm[1 : n + 1], dtype=np.float64)
        )
        for row in range(n + 1, rows):
            smoothed_tr[row] = (
                smoothed_tr[row - 1]
                - smoothed_tr[row - 1] / n
                + tr[row]
            )
            smoothed_plus[row] = (
                smoothed_plus[row - 1]
                - smoothed_plus[row - 1] / n
                + plus_dm[row]
            )
            smoothed_minus[row] = (
                smoothed_minus[row - 1]
                - smoothed_minus[row - 1] / n
                + minus_dm[row]
            )

    positive_tr = np.where(smoothed_tr > 0.0, smoothed_tr, np.nan)
    plus_di = 100.0 * smoothed_plus / positive_tr
    minus_di = 100.0 * smoothed_minus / positive_tr
    di_sum = plus_di + minus_di
    positive_di_sum = np.where(di_sum > 0.0, di_sum, np.nan)
    di_spread = (plus_di - minus_di) / positive_di_sum
    dx = 100.0 * np.abs(plus_di - minus_di) / positive_di_sum
    adx = np.full(rows, np.nan, dtype=np.float64)
    first_adx = 2 * n - 1
    if rows > first_adx and np.isfinite(dx[n : first_adx + 1]).all():
        adx[first_adx] = float(
            np.mean(dx[n : first_adx + 1], dtype=np.float64)
        )
        for row in range(first_adx + 1, rows):
            if not np.isfinite(dx[row]):
                break
            adx[row] = ((n - 1) * adx[row - 1] + dx[row]) / n
    return (
        pd.Series(adx, index=close.index, dtype=np.float64),
        pd.Series(di_spread, index=close.index, dtype=np.float64),
    )


def _cross_up_event(series: pd.Series) -> pd.Series:
    """Edge-triggered upward zero-cross on closed bars.

    Exact formula of the local layer's ``ema50_200_cross_up``
    (entry_model_native_feature_layers_v1.build_price_derived_layer:
    ``(spread > 0) & (spread.shift(1) <= 0)``), emitted as NaN wherever the
    series or its previous bar is still inside the causal warmup (rule 2e:
    an unknown crossing must not read as "no crossing").
    """
    previous = series.shift(1)
    event = ((series > 0) & (previous <= 0)).astype(np.float64)
    return event.where(series.notna() & previous.notna())


def _cross_down_event(series: pd.Series) -> pd.Series:
    """Mirror of :func:`_cross_up_event` (the local ``ema50_200_cross_down``)."""
    previous = series.shift(1)
    event = ((series < 0) & (previous >= 0)).astype(np.float64)
    return event.where(series.notna() & previous.notna())


def _compute_v29_momentum_event_frame(
    *,
    high: pd.Series,
    low: pd.Series,
    rsi: pd.Series,
    mom_20_atr: pd.Series,
    atr_positive: pd.Series,
) -> pd.DataFrame:
    """One formula owner for the V29 momentum G1/G2 event fields.

    ``rsi`` is the raw Wilder 0-100 series with its 14-row warmup already
    masked; ``mom_20_atr`` is the raw 20-bar ATR-normalized momentum this
    file emits; ``atr_positive`` is the caller's positive-only Wilder ATR,
    consumed at the pivot bars by the V30 divergence-strength
    fields.  Called by :func:`compute_per_bar_features_v4` (per-TF lane) and
    :func:`compute_v29_momentum_event_block_from_ohlc` (native M5/M1 lane) so
    the two lanes cannot drift.
    """
    from gx1.features.smc_v1 import (
        SWING_LOOKBACK,
        _detect_swing_pivots,
        _track_recent_swings,
    )

    frame = pd.DataFrame(index=rsi.index, dtype=np.float64)
    # Momentum G2: RSI threshold events on the raw Wilder 0-100 series (the
    # exact masked `rsi`, BEFORE the centered affine map).  Thresholds are
    # Wilder's published 30/70 bands and the 50 midline (named module
    # constants); a threshold cross is the zero-cross of (rsi - level).
    frame["rsi_cross_up_30"] = _cross_up_event(rsi - RSI_WILDER_OVERSOLD)
    frame["rsi_cross_down_70"] = _cross_down_event(rsi - RSI_WILDER_OVERBOUGHT)
    frame["rsi_cross_up_50"] = _cross_up_event(rsi - RSI_WILDER_MIDLINE)
    frame["rsi_cross_down_50"] = _cross_down_event(rsi - RSI_WILDER_MIDLINE)
    rsi_np = rsi.to_numpy(dtype=np.float64)
    rsi_valid = np.isfinite(rsi_np)
    rsi_extreme = rsi_valid & (
        np.abs(rsi_np - RSI_WILDER_MIDLINE) >= RSI_EXTREME_BAND_WIDTH
    )
    frame["rsi_extreme_event_age_bars"] = raw_event_age_bars(
        rsi_extreme.astype(np.bool_), rsi_valid
    )

    # Momentum G2: mom_20_atr zero-line sign flips.  Zero is the natural
    # named constant of a signed difference, so the crossing series is the
    # emitted raw field itself.
    frame["mom20_sign_flip_up"] = _cross_up_event(mom_20_atr)
    frame["mom20_sign_flip_down"] = _cross_down_event(mom_20_atr)

    # Momentum G1: RSI divergence on confirmed price pivots.  One pivot
    # truth: smc_v1's _detect_swing_pivots/_track_recent_swings with its
    # named SWING_LOOKBACK (= 3).  A pivot at bar j becomes visible only from
    # its confirmation bar j + SWING_LOOKBACK, and the RSI value read AT bar
    # j uses only data <= j, so the event is causal by the same argument
    # already proven for the swing features.  Bearish: price higher-high
    # pivot pair with RSI lower-high; bullish mirrored on the low pivots.
    high_np = high.to_numpy(dtype=np.float64)
    low_np = low.to_numpy(dtype=np.float64)
    n_rows = len(rsi_np)
    pivot_high_mask, pivot_low_mask = _detect_swing_pivots(
        high_np, low_np, SWING_LOOKBACK
    )
    last_sh, prev_sh, last_sl, prev_sl = _track_recent_swings(
        pivot_high_mask, pivot_low_mask, SWING_LOOKBACK
    )
    clip_last_sh = np.clip(last_sh, 0, n_rows - 1)
    clip_prev_sh = np.clip(prev_sh, 0, n_rows - 1)
    clip_last_sl = np.clip(last_sl, 0, n_rows - 1)
    clip_prev_sl = np.clip(prev_sl, 0, n_rows - 1)
    new_high_pair = last_sh != np.roll(last_sh, 1)
    new_high_pair[0] = False
    new_low_pair = last_sl != np.roll(last_sl, 1)
    new_low_pair[0] = False
    # Defined once the OLDER pivot of the pair has a valid RSI.  Pivot
    # indices are non-decreasing and RSI validity is one suffix, so each
    # mask is one honest NaN warmup prefix, never a mid-series hole.
    bear_defined = (prev_sh >= 0) & rsi_valid[clip_prev_sh]
    bull_defined = (prev_sl >= 0) & rsi_valid[clip_prev_sl]
    bear_event = (
        new_high_pair
        & bear_defined
        & (high_np[clip_last_sh] > high_np[clip_prev_sh])
        & (rsi_np[clip_last_sh] < rsi_np[clip_prev_sh])
    )
    bull_event = (
        new_low_pair
        & bull_defined
        & (low_np[clip_last_sl] < low_np[clip_prev_sl])
        & (rsi_np[clip_last_sl] > rsi_np[clip_prev_sl])
    )
    frame["bear_divergence_event"] = np.where(
        bear_defined, bear_event.astype(np.float64), np.nan
    )
    frame["bull_divergence_event"] = np.where(
        bull_defined, bull_event.astype(np.float64), np.nan
    )
    # V30 divergence STRENGTH (design doc §3 momentum row "event/strength/
    # age"; the Phase-A build kept event+age only).  Formula: (RSI delta
    # between the pivot pair / RSI_WILDER_MIDLINE) x (price delta at the same
    # pivots / positive ATR at the newer pivot's own bar).  /50 is the design
    # row's named "/50 strength" constant (the existing rsi14_centered affine
    # divisor); the pivot-bar ATR is the trendline-registry touch convention
    # ("measured at the pivot's own bar with that bar's ATR"). Each side emits
    # a non-negative magnitude; direction lives in the field identity exactly
    # as in the sibling 0/1 events. Off-event
    # bars are flag-disambiguated 0 (design B.5); the undefined prefix is the
    # events' own NaN prefix.  ``atr_positive`` at the newer pivot is finite
    # wherever the pair is defined: bear/bull_defined requires a valid RSI at
    # the OLDER pivot (RSI warmup 14 rows) and the ATR warmup is 13 rows, so
    # no mid-series NaN can enter through the ATR read.
    atr_positive_np = atr_positive.to_numpy(dtype=np.float64)
    bear_strength = (
        (rsi_np[clip_prev_sh] - rsi_np[clip_last_sh]) / RSI_WILDER_MIDLINE
    ) * (
        (high_np[clip_last_sh] - high_np[clip_prev_sh])
        / atr_positive_np[clip_last_sh]
    )
    bull_strength = (
        (rsi_np[clip_last_sl] - rsi_np[clip_prev_sl]) / RSI_WILDER_MIDLINE
    ) * (
        (low_np[clip_prev_sl] - low_np[clip_last_sl])
        / atr_positive_np[clip_last_sl]
    )
    frame["bear_divergence_strength"] = np.where(
        bear_defined, np.where(bear_event, bear_strength, 0.0), np.nan
    )
    frame["bull_divergence_strength"] = np.where(
        bull_defined, np.where(bull_event, bull_strength, 0.0), np.nan
    )
    divergence_defined = bear_defined & bull_defined
    frame["divergence_event_age_bars"] = raw_event_age_bars(
        (bear_event | bull_event).astype(np.bool_), divergence_defined
    )
    return frame.loc[:, list(MULTI_TF_V4_MOMENTUM_EVENT_FEATURES)]


def compute_v29_momentum_event_block_from_ohlc(
    ohlc: pd.DataFrame,
    *,
    include_v30_primitives: bool = False,
) -> pd.DataFrame:
    """Native M5/M1 momentum block (design doc §5.1 block E).

    Computes the exact ``MULTI_TF_V4_MOMENTUM_EVENT_FEATURES`` fields on the
    provided closed-bar OHLC clock with this owner's own input conventions
    (Wilder RSI with the 14-row mask and raw ``mom_20_atr`` over the
    positive-only Wilder ATR — the identical expressions
    :func:`compute_per_bar_features_v4` uses, in this same file). When
    ``include_v30_primitives`` is true, the four native-clock continuous
    primitives in ``LOCAL_MOMENTUM_V30_PRIMITIVE_FEATURES`` are appended.
    """
    if not isinstance(include_v30_primitives, bool):
        raise RuntimeError("HTF_V4_LOCAL_MOMENTUM_PRIMITIVE_FLAG_INVALID")
    required = ("high", "low", "close")
    missing = [name for name in required if name not in ohlc.columns]
    if missing:
        raise RuntimeError(
            f"HTF_V4_V29_MOMENTUM_EVENT_SOURCE_MISSING: {missing}"
        )
    high = ohlc["high"].astype(np.float64)
    low = ohlc["low"].astype(np.float64)
    close = ohlc["close"].astype(np.float64)
    _atr14, atr_positive = wilder_atr14_positive(high, low, close)
    rsi = _rsi(close, 14)
    rsi.iloc[:14] = np.nan
    mom_5_atr = (close - close.shift(5)) / atr_positive
    mom_20_atr = (close - close.shift(20)) / atr_positive
    events = _compute_v29_momentum_event_frame(
        high=high,
        low=low,
        rsi=rsi,
        mom_20_atr=mom_20_atr,
        atr_positive=atr_positive,
    )
    if not include_v30_primitives:
        return events
    primitives = pd.DataFrame(
        {
            "rsi14_centered": (
                (rsi - RSI_WILDER_MIDLINE) / RSI_WILDER_MIDLINE
            ),
            "rsi14_delta_5": rsi - rsi.shift(5),
            "mom_5_atr": mom_5_atr,
            "mom_20_atr": mom_20_atr,
        },
        index=ohlc.index,
    )
    out = pd.concat([events, primitives], axis=1)
    expected = (
        *MULTI_TF_V4_MOMENTUM_EVENT_FEATURES,
        *LOCAL_MOMENTUM_V30_PRIMITIVE_FEATURES,
    )
    if tuple(out.columns) != expected:
        raise RuntimeError("HTF_V4_LOCAL_MOMENTUM_OUTPUT_ORDER_INVALID")
    return out


def validate_causal_feature_matrix(
    values,
    *,
    expected_width: int,
    context: str,
) -> int:
    """Validate an exact feature matrix and return its warmup-prefix length.

    A model feature may be unavailable only in one chronological prefix. Once a
    complete row exists, every later row must be finite. Numeric sentinels are
    deliberately not introduced here.
    """
    arr = np.asarray(values)
    if arr.ndim != 2 or arr.shape[1] != int(expected_width) or arr.shape[0] == 0:
        raise RuntimeError(
            f"[{context}] feature matrix must have non-zero shape (N, {expected_width}); "
            f"observed={arr.shape}"
        )
    try:
        numeric = arr.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"[{context}] feature matrix must be numeric") from exc
    if np.isinf(numeric).any():
        raise RuntimeError(f"[{context}] feature matrix contains infinity")
    complete = np.isfinite(numeric).all(axis=1)
    if not complete.any():
        return int(len(numeric))
    first_complete = int(np.argmax(complete))
    if not complete[first_complete:].all():
        raise RuntimeError(
            f"[{context}] non-finite feature rows are not one causal warmup prefix"
        )
    return first_complete


def compute_per_bar_features_v4(
    ohlcv: pd.DataFrame,
    *,
    timeframe: str,
    v29_registry_constants,
    volatility_squeeze_artifacts: VolatilitySqueezeArtifactSet,
) -> pd.DataFrame:
    """Compute the exact fixed-width V4 surface directly from one OHLCV TF.

    ``timeframe`` is the declared MULTI_TF_RESAMPLE_RULES key of ``ohlcv``. It
    selects the local-cycle VWAP owner (D1 → rolling 5-bar, intraday TFs →
    shared trading-session VWAP); the retired median-bar-spacing inference is
    gone.  ``v29_registry_constants`` is the TRAIN-fitted registry-constants
    payload (:func:`require_v29_registry_constants`); the level/trendline
    registry blocks cannot be computed without it and no default exists.
    """

    if timeframe not in MULTI_TF_RESAMPLE_RULES:
        raise RuntimeError(
            f"HTF_V4_TIMEFRAME_INVALID: {timeframe!r}"
        )
    registry_constants = require_v29_registry_constants(v29_registry_constants)
    squeeze_artifacts = require_volatility_squeeze_artifact_set(
        volatility_squeeze_artifacts
    )

    from gx1.features.entry_candle_primitives_v1 import (
        build_entry_candle_primitive_layer,
    )
    from gx1.features.smc_v1 import compute_smc_mtf_primitives_v1

    _validate_m5_input(ohlcv, require_volume=True)
    df = ohlcv[["open", "high", "low", "close", "volume"]].astype(
        np.float64
    ).copy()
    out = pd.DataFrame(index=df.index, dtype=np.float64)

    close = df["close"]
    open_ = df["open"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    ema_spread_block = ema50_200_spread_atr_block(high, low, close)
    atr14 = ema_spread_block["atr14"]
    atr_positive = ema_spread_block["atr14_positive"]

    out["atr_bps_14"] = atr14 / close * 1e4
    rsi = _rsi(close, 14)
    rsi.iloc[:14] = np.nan
    out["rsi14_centered"] = (rsi - 50.0) / 50.0
    # V30 (2026-08-13): raw Wilder RSI velocity on the masked series.  k=5 is
    # this file's existing EMA-slope lookback convention (the shift(5) used by
    # ema20/50/200_slope_atr below); the RSI domain bounds the delta in
    # [-100, 100] algebraically, so no clip constant is introduced.  The
    # masked 14-row RSI warmup plus the 5-bar shift form one honest NaN
    # prefix.
    out["rsi14_delta_5"] = rsi - rsi.shift(5)
    for lag in (5, 20):
        out[f"mom_{lag}_atr"] = (
            (close - close.shift(lag)) / atr_positive
        )
    out["close_open_atr"] = (close - open_) / atr_positive

    # Zero-range convention (2026-08-15) — ADOPTED VERBATIM from the sibling
    # candle owner in this same repository,
    # gx1.features.entry_candle_primitives_v1.
    # compute_entry_candle_primitive_chunk: on a ``high == low`` bar every
    # range SHARE is mathematically undefined, and that owner emits share 0.0
    # there.  No magnitude is invented: ``body_pct`` is the unsigned twin of
    # ``mtf_candle_raw_body_signed_range``, which the sibling already sets to
    # 0.0 on exactly these rows.
    #
    # WHAT CARRIES THE DISTINCTION NOW (2026-08-15).  This comment used to
    # name ``candle.raw_zero_range_flag`` / ``mtf_candle_raw_zero_range_flag``
    # as the field that separated the storage zero from a real observation.
    # That flag has been RETIRED — it was constant 0.0 post-warmup on H4 and
    # D1 (liveness RED) and unscaleable as a declared constant; the argument
    # is in the CANDLE_PRIMITIVE_FEATURE_VERSION note of the sibling owner.
    # The distinction did not leave with it.  PROVEN FROM ALGEBRA there and
    # true on this surface for the same reason: the three range shares
    # partition the range exactly, so on any lane
    #     mtf_candle_raw_body_signed_range == 0
    #     and mtf_candle_raw_upper_wick_share == 0
    #     and mtf_candle_raw_lower_wick_share == 0
    # holds if and only if ``high == low`` on that bar — those three are
    # emitted by the sibling owner from this very frame, beside body_pct.  A
    # real-range doji has a zero body share but nonzero wick shares.
    # MEASURED, so a future reader can re-judge the trade rather than
    # re-measure it: on the complete declared native M5 tape
    # XAU_M5_NATIVE_2019_20260804_V4 (537,861 rows, 2019-01-01..2026-08-04),
    # resampled by this owner and counted after the >=199-row causal warmup,
    # zero-range bars number 215 on M5 (0.040%), 24 on M15 (0.013%), 14 on H1
    # (0.031%), 0 on H4 and 0 on D1 — 253 rows of the whole tape.
    #
    # The retired ``(high - low).where((high - low) > 0.0)`` mask made
    # ``body_pct`` NaN on a zero-range bar, i.e. a hole in the MIDDLE of the
    # series, which validate_causal_feature_matrix rejects as "not one causal
    # warmup prefix" — a single such bar aborted the entire per-timeframe
    # build.  Only an exact 0.0 range takes the convention: a NaN or negative
    # (invalid-geometry) range still propagates NaN and still fails closed.
    bar_range = high - low
    body = (close - open_).abs()
    out["body_pct"] = body.div(bar_range.where(bar_range > 0.0)).mask(
        bar_range.eq(0.0), 0.0
    )

    ema20 = _ema(close, 20)
    ema50 = ema_spread_block["ema50"]
    ema100 = _ema(close, 100)
    ema200 = ema_spread_block["ema200"]
    out["ema20_dist_atr"] = (close - ema20) / atr_positive
    out["ema50_dist_atr"] = (close - ema50) / atr_positive
    out["ema100_dist_atr"] = (close - ema100) / atr_positive
    out["ema200_dist_atr"] = (close - ema200) / atr_positive
    out["ema20_slope_atr"] = (
        (ema20 - ema20.shift(5)) / atr_positive
    )
    out["ema50_slope_atr"] = (
        (ema50 - ema50.shift(5)) / atr_positive
    )
    out["ema200_slope_atr"] = (
        (ema200 - ema200.shift(5)) / atr_positive
    )

    bull = (ema20 > ema50) & (ema50 > ema100) & (ema100 > ema200)
    bear = (ema20 < ema50) & (ema50 < ema100) & (ema100 < ema200)
    stack_defined = ema20.notna() & ema50.notna() & ema100.notna() & ema200.notna()
    stack = pd.Series(np.nan, index=close.index, dtype=np.float64)
    stack.loc[stack_defined] = 0.0
    stack[bull] = 1
    stack[bear] = -1
    out["ema_stack_aligned_v2"] = stack

    # Declared-TF selection (2026-08-09): value-identical to the retired
    # >=23h median-spacing inference for the five declared timeframes, without
    # inferring the clock from data or swallowing NaT spacing.
    vwap_rolling5 = _rolling_vwap(close, volume, 5)
    local_cycle_vwap = (
        vwap_rolling5
        if timeframe == "D1"
        else _session_vwap(
            close,
            volume,
            bar_duration=pd.Timedelta(MULTI_TF_RESAMPLE_RULES[timeframe]),
        )
    )
    out["vwap_local_cycle_dist_atr"] = (
        (close - local_cycle_vwap) / atr_positive
    )
    vwap20 = _rolling_vwap(close, volume, 20)
    out["vwap20_dist_atr"] = (close - vwap20) / atr_positive
    vwap96 = _rolling_vwap(close, volume, 96)
    out["vwap96_dist_atr"] = (close - vwap96) / atr_positive
    # 2026-08-18 (V30 wave 2) REPAIR + RENAME. Until now this emitted
    # `vwap_local_cycle_slope_atr` = the 5-bar difference of
    # `local_cycle_vwap`, which is a CUMULATIVE session VWAP on every intraday
    # lane (reset at each session boundary) and a ROLLING 5-bar VWAP only on
    # D1. Differencing a cumulative series over a fixed row count is not a
    # slope of anything when the window straddles a reset: the subtrahend
    # belongs to a different accumulation. [M, INHERITED] share of defined rows
    # whose shift(5) crosses a session-id change: H4 83.47%, H1 21.83%,
    # M15 5.46%, M5 1.82%.
    #
    # Masking those rows fails closed on H4 (most of the lane would be NaN),
    # and branching per lane is forbidden (one owner, one formula). The only
    # union that invents nothing is to adopt the D1 branch's OWN operand
    # everywhere: `_rolling_vwap(close, volume, 5)` is already this function's
    # local-cycle VWAP for D1, and `shift(5)` is already this file's EMA-slope
    # lookback (ema20/50/200_slope_atr). No new magnitude is chosen.
    #
    # The rename is mandatory, not cosmetic: keeping `vwap_local_cycle_slope_atr`
    # over a rolling operand would trade a hollow field for a lying one. On D1
    # the emitted float is unchanged (the operand is the same series there);
    # on M5/M15/H1/H4 the value changes, which is the repair.
    out["vwap_rolling5_slope_atr"] = (
        (vwap_rolling5 - vwap_rolling5.shift(5)) / atr_positive
    )

    sma20 = close.rolling(20, min_periods=20).mean()
    std20 = close.rolling(20, min_periods=20).std()
    bb_upper = sma20 + 2.0 * std20
    bb_lower = sma20 - 2.0 * std20
    bb_width = bb_upper - bb_lower
    out["bb_position"] = (close - bb_lower) / bb_width.where(bb_width > 0.0)
    out["bb_width_atr"] = bb_width / atr_positive
    adx, di_spread_signed = _adx14(high, low, close, 14)
    out["adx14"] = adx
    # V30 (2026-08-13): the signed DI spread from the same _adx14 producer
    # (see its docstring); already normalized to [-1, 1] by construction.
    out["di_spread_signed"] = di_spread_signed
    out["trend_state_age_bars"] = raw_state_age_bars(
        stack.to_numpy(dtype=np.float64),
        stack.notna().to_numpy(dtype=bool),
    )

    # ------------------------------------------------------------------
    # V29 Phase A per-TF EVENT additions (trend_ema GAP-1/2/3 + momentum
    # G1/G2, 2026-08-11 reports; design doc §3).  Every field is computed on
    # this one TF clock from series this owner already produces; every event
    # is a closed-bar edge trigger; every age uses this file's log1p/500
    # convention; every warmup is one honest NaN prefix.
    # ------------------------------------------------------------------
    v29 = pd.DataFrame(index=df.index, dtype=np.float64)

    # GAP-1: local and per-TF routes consume the exact same float64 EMA50/200
    # + positive-only Wilder ATR block, then cast once at storage boundary.
    spread_50_200 = ema_spread_block["spread"]
    v29["ema50_200_spread_atr"] = ema_spread_block["spread_atr"]
    bull_state_50_200 = (spread_50_200 > 0).astype(np.float64).where(
        spread_50_200.notna()
    )
    v29["ema50_200_bull_state"] = bull_state_50_200
    v29["ema50_200_cross_up"] = _cross_up_event(spread_50_200)
    v29["ema50_200_cross_down"] = _cross_down_event(spread_50_200)
    # GAP-2: raw duration of the current 50/200 state on this native clock.
    v29["ema50_200_state_age_bars"] = raw_state_age_bars(
        bull_state_50_200.to_numpy(dtype=np.float64),
        bull_state_50_200.notna().to_numpy(dtype=bool),
    )

    # GAP-3: price-vs-EMA cross events + side age (same sign-flip construction
    # as GAP-1, same age convention as GAP-2).  The side's sign is already
    # carried by ema50_dist_atr/ema200_dist_atr above, so the age is emitted
    # unsigned in [0, 1] (rule 2e: no synthetic signed-zero packing).
    for ema_span, ema_line in ((50, ema50), (200, ema200)):
        price_gap = close - ema_line
        v29[f"price_x_ema{ema_span}_cross_up"] = _cross_up_event(price_gap)
        v29[f"price_x_ema{ema_span}_cross_down"] = _cross_down_event(price_gap)
        side_state = (price_gap > 0).astype(np.float64).where(price_gap.notna())
        v29[f"price_vs_ema{ema_span}_state_age_bars"] = raw_state_age_bars(
            side_state.to_numpy(dtype=np.float64),
            side_state.notna().to_numpy(dtype=bool),
        )

    # Momentum G1/G2 events: computed by the one owner function below, from
    # the exact same masked-RSI and raw mom_20_atr series this owner just
    # produced (bit-identical inputs, one formula owner for both the per-TF
    # lane and the native M5/M1 lane).
    momentum_events = _compute_v29_momentum_event_frame(
        high=high,
        low=low,
        rsi=rsi,
        mom_20_atr=out["mom_20_atr"],
        atr_positive=atr_positive,
    )
    for name in MULTI_TF_V4_MOMENTUM_EVENT_FEATURES:
        v29[name] = momentum_events[name]

    v29_names = (
        MULTI_TF_V4_TREND_EVENT_FEATURES + MULTI_TF_V4_MOMENTUM_EVENT_FEATURES
    )
    if set(v29.columns) != set(v29_names):
        raise RuntimeError("HTF_V4_V29_EVENT_FIELDS_INVALID")
    v29 = v29.loc[:, list(v29_names)]

    out = out.loc[:, list(MULTI_TF_V4_GROUP_A_BASE_FEATURES)]
    volume_features = compute_volume_features(df[["volume"]])
    if tuple(volume_features) != MULTI_TF_V4_VOLUME_FEATURES:
        raise RuntimeError("HTF_V4_VOLUME_FEATURE_CONTRACT_INVALID")
    for name in MULTI_TF_V4_VOLUME_FEATURES:
        out[name] = volume_features[name]

    squeeze_frame, _squeeze_carry = compute_volatility_squeeze_state(
        df[["open", "high", "low", "close", "volume"]],
        timeframe=timeframe,
        params=squeeze_artifacts.require_params(timeframe),
    )
    if (
        tuple(squeeze_frame.columns)
        != MULTI_TF_V4_VOLATILITY_SQUEEZE_FEATURES
        or not squeeze_frame.index.equals(out.index)
    ):
        raise RuntimeError("HTF_V4_VOLATILITY_SQUEEZE_CONTRACT_INVALID")
    for name in MULTI_TF_V4_VOLATILITY_SQUEEZE_FEATURES:
        out[name] = squeeze_frame[name]

    candle_source = df[["open", "high", "low", "close"]].copy()
    candle_source.index.name = "time"
    candle_values, candle_names = build_entry_candle_primitive_layer(
        candle_source.reset_index()
    )
    observed_candle_names = tuple(
        f"mtf_{name.replace('.', '_')}"
        for name in candle_names
    )
    candle_values = np.asarray(candle_values, dtype=np.float64)
    if (
        candle_values.shape
        != (len(out), len(MULTI_TF_V4_CANDLE_PRIMITIVE_FEATURES))
        or observed_candle_names != MULTI_TF_V4_CANDLE_PRIMITIVE_FEATURES
    ):
        raise RuntimeError("HTF_V4_CANDLE_PRIMITIVE_CONTRACT_INVALID")
    for name, values in zip(
        MULTI_TF_V4_CANDLE_PRIMITIVE_FEATURES,
        candle_values.T,
        strict=True,
    ):
        out[name] = values

    # V30 (2026-08-13): the per-TF lane now consumes the V29 additions too (see
    # MULTI_TF_V4_SWING_SOURCE_FIELD_BY_FEATURE).  This is the call-site
    # contract switch the producer's docstring reserved for the stage-2 wiring,
    # taken at the V30 rebuild boundary — never an environment gate.
    swing = compute_swing_structure_features(
        high.to_numpy(dtype=np.float64),
        low.to_numpy(dtype=np.float64),
        close.to_numpy(dtype=np.float64),
        include_v29_additions=True,
    )
    missing_swing = [
        MULTI_TF_V4_SWING_SOURCE_FIELD_BY_FEATURE[name]
        for name in MULTI_TF_V4_SWING_FEATURES
        if MULTI_TF_V4_SWING_SOURCE_FIELD_BY_FEATURE[name] not in swing
    ]
    if missing_swing:
        raise RuntimeError(f"HTF_V4_SWING_FIELD_MISSING: {missing_swing}")
    # Built as one block and concatenated below (the full
    # MULTI_TF_V4_SWING_FEATURES width, not the pre-V30 five): a per-column
    # insert into the already-wide `out` frame fragments its BlockManager past
    # pandas' 100-block warning threshold.
    swing_frame = pd.DataFrame(
        {
            name: np.asarray(
                swing[MULTI_TF_V4_SWING_SOURCE_FIELD_BY_FEATURE[name]],
                dtype=np.float64,
            )
            for name in MULTI_TF_V4_SWING_FEATURES
        },
        index=df.index,
        columns=list(MULTI_TF_V4_SWING_FEATURES),
    )

    smc_source = df[["high", "low", "close"]].copy()
    smc_source["atr"] = atr14
    primitives = compute_smc_mtf_primitives_v1(smc_source)
    if not primitives.index.equals(out.index):
        raise RuntimeError("HTF_V4_SMC_ROW_AXIS_MISMATCH")

    # V29 registry blocks (design doc §1.3/§2): both run on this TF clock from
    # the same high/low/close/atr14 source as the smc primitives, with that
    # TF's TRAIN-fitted frozen constants.  Emission names/order are owned by
    # the registry modules.
    level_matrix, level_names = compute_level_registry_mtf_block_v1(
        smc_source,
        tf=timeframe.lower(),
        recurrence_threshold_atr=registry_constants[
            "level_recurrence_threshold_atr"
        ][timeframe],
        max_evidence_age_bars=int(
            registry_constants["level_expiry_bars"][timeframe]
        ),
    )
    if tuple(level_names) != MULTI_TF_V4_LEVEL_REGISTRY_FEATURES:
        raise RuntimeError("HTF_V4_LEVEL_REGISTRY_NAME_DRIFT")
    level_frame = pd.DataFrame(
        np.asarray(level_matrix, dtype=np.float64),
        index=df.index,
        columns=list(level_names),
    )
    trendline_frame, _trendline_state = compute_trendline_registry_features_v1(
        smc_source,
        timeframe=timeframe,
        seq_len=int(registry_constants["per_tf_seq_lens"][timeframe]),
        band_atr=registry_constants["trendline_band_atr"][timeframe],
    )
    if tuple(trendline_frame.columns) != MULTI_TF_V4_TRENDLINE_REGISTRY_FEATURES:
        raise RuntimeError("HTF_V4_TRENDLINE_REGISTRY_NAME_DRIFT")
    if not level_frame.index.equals(out.index) or not trendline_frame.index.equals(
        out.index
    ):
        raise RuntimeError("HTF_V4_V29_REGISTRY_ROW_AXIS_MISMATCH")

    out = pd.concat(
        (
            out,
            swing_frame,
            primitives,
            v29,
            level_frame,
            trendline_frame.astype(np.float64),
        ),
        axis=1,
    )
    if tuple(out.columns) != MULTI_TF_PER_BAR_FEATURES_V4:
        raise RuntimeError(
            "HTF_V4_COLUMN_ORDER_INVALID: "
            f"observed={tuple(out.columns)}"
        )
    validate_causal_feature_matrix(
        out.to_numpy(dtype=np.float64, copy=False),
        expected_width=MULTI_TF_FEATURE_COUNT_V4,
        context="HTF_V4_CAUSAL_FEATURES",
    )
    return out.astype(np.float32)


def _model_native_rsi_z48_v4(close: pd.Series) -> pd.Series:
    rsi = _rsi(close, 14)
    mean = rsi.rolling(48, min_periods=48).mean()
    std = rsi.rolling(48, min_periods=48).std(ddof=0).replace(0.0, np.nan)
    return (rsi - mean) / std


def _model_native_htf_slope_v4(
    values: pd.Series,
    *,
    order: int,
) -> pd.Series:
    """Compute the ``order``-bar change on the native HTF clock.

    2026-08-09 repair: the previous ``np.diff(source, n=order, prepend=...)``
    was the ORDER-th finite difference (coefficients 1,-5,10,-10,5,-1 for
    order 5) — a noise amplifier, not a slope. The intended quantity is the
    k-bar change ``x[i] - x[i-order]``. Closed-bar availability is owned by
    the MTF projection, so the formula itself carries no second publish lag.
    The first ``order`` outputs are an honest NaN warmup prefix: the
    scalar-frame consumer asserts it through
    ``validate_causal_feature_matrix`` (one contiguous warmup prefix, hard
    failure on any later NaN) instead of masking with ``nan_to_num``.
    """

    source = values.to_numpy(dtype=np.float64)
    if source.ndim != 1 or order not in {3, 5} or len(source) < order:
        raise RuntimeError("HTF_V4_MODEL_NATIVE_HTF_SLOPE_INPUT_INVALID")
    slope = source - np.concatenate(
        [np.full(order, np.nan), source[:-order]]
    )
    return pd.Series(
        slope,
        index=values.index,
        dtype=np.float64,
    )


def _compute_model_native_mtf_scalar_frame_v4(
    ohlcv: pd.DataFrame,
    *,
    timeframe: str,
) -> pd.DataFrame:
    """Compute the compact persistent scalar surface on one closed TF clock."""

    if timeframe not in MODEL_NATIVE_MTF_SCALAR_FIELDS_BY_TIMEFRAME_V4:
        raise RuntimeError(
            f"HTF_V4_MODEL_NATIVE_SCALAR_TIMEFRAME_INVALID: {timeframe!r}"
        )
    expected_fields = MODEL_NATIVE_MTF_SCALAR_FIELDS_BY_TIMEFRAME_V4[timeframe]
    _validate_m5_input(ohlcv, require_volume=True)
    source = ohlcv.loc[:, ["open", "high", "low", "close", "volume"]].astype(
        np.float64
    )
    high = source["high"]
    low = source["low"]
    close = source["close"]
    atr14, atr14_positive = wilder_atr14_positive(high, low, close)
    out = pd.DataFrame(index=source.index)

    if timeframe in {"H1", "H4"}:
        prefix = "_v1h1" if timeframe == "H1" else "_v1h4"
        ema12 = _ema(close, 12)
        ema26 = _ema(close, 26)
        if timeframe == "H4":
            mid = (high + low) * 0.5
            ema50 = _ema(mid, 50)
            distance = (mid - ema50) / atr14_positive
            distance.iloc[: H4_EMA50_MIN_BARS - 1] = np.nan
            out["h4_mid_ema50_dist_atr_canon_v2"] = distance
        # 2026-08-09 era-proxy repair: these fields were raw USD magnitudes and
        # tracked the multi-year gold price level instead of market state.
        # Units now follow this file's own conventions:
        # - ema_diff in ATR-multiples (the D1_dist_from_ema200_atr convention);
        #   the slope3/slope5 fields below inherit the normalized series so
        #   they are ATR-multiple changes, not USD changes.
        # - atr in bps of price (the atr_bps_14 convention). Normalization
        #   statistics are refitted at the next dataset rebuild.
        # 2026-08-18 (V30 wave 2): the NAME now carries that unit too. The
        # 2026-08-09 repair changed the value and left the name saying `_atr`,
        # which reads as "in ATR units" everywhere else in this file
        # (ema50_slope_atr, mom_5_atr, ...) -- the exact opposite of a bps
        # ratio. `_bps` is copied verbatim from the sibling owner `atr_bps_14`
        # in this same file; the emitted float is bit-identical.
        ema_diff = (ema12 - ema26) / atr14_positive
        out[f"{prefix}_ema_diff"] = ema_diff
        out[f"{prefix}_atr_bps"] = atr14 / close * 1e4
        out[f"{prefix}_rsi14_z"] = _model_native_rsi_z48_v4(close)
        out[f"{prefix}_slope3"] = _model_native_htf_slope_v4(
            ema_diff,
            order=3,
        )
        out[f"{prefix}_slope5"] = _model_native_htf_slope_v4(
            ema_diff,
            order=5,
        )
        # V30 momentum G3: the raw Wilder level from the same one `_rsi` owner
        # the M15/D1 canon fields call, on this TF's own closed bars.  The
        # `_v1h{1,4}_rsi14_z` 48-bar z-score above is kept (design row: "z-fields
        # kept"); a z-score of a bounded oscillator hides WHERE in the 0-100
        # domain the market is, which is exactly the M15/D1 evidence M5/H1/H4
        # lacked.
        out[f"{timeframe.lower()}_rsi14_canon_v2"] = _rsi(close, 14)
    elif timeframe == "M5":
        # V30 momentum G3 (see the field-tuple comment): the M5 sibling of
        # `m15_rsi14_canon_v2` — same producer, same unit, this TF's clock.
        out["m5_rsi14_canon_v2"] = _rsi(close, 14)
    elif timeframe == "M15":
        out["m15_rsi14_canon_v2"] = _rsi(close, 14)
        bar_range = high - low
        range_mean = bar_range.rolling(20, min_periods=20).mean()
        range_std = bar_range.rolling(20, min_periods=20).std(ddof=0).replace(
            0.0, np.nan
        )
        out["m15_range_z_20_canon_v2"] = (
            bar_range - range_mean
        ) / range_std
        out["m15_ema5_20_spread_atr_canon_v2"] = (
            _ema(close, 5) - _ema(close, 20)
        ) / atr14_positive
    elif timeframe == "D1":
        mid = (high + low) * 0.5
        ema200 = _ema(mid, 200)
        distance = (mid - ema200) / atr14_positive
        distance.iloc[: D1_EMA200_MIN_BARS - 1] = np.nan
        out["D1_dist_from_ema200_atr"] = distance
        # 2026-08-09 era-proxy repair: d1_atr14_bps_canon_v2 in bps of price
        # (atr_bps_14 convention); d1_ema_slope_20_canon_v2 in ATR-multiples
        # (D1_dist_from_ema200_atr convention, positive ATR only).
        # 2026-08-18 (V30 wave 2) rename, no value change: the ATR field was
        # `d1_atr14_canon_v2` while carrying bps, and `d1_pct_change_5_canon_v2`
        # multiplies pandas' FRACTIONAL pct_change by 1e4 -- a 100x gap between
        # the name and the number. `_bps` is inserted BEFORE the `_canon_v2`
        # generation marker so that marker keeps its position. The `_change_`
        # token is load-bearing and must survive any future rename: it is the
        # only token in entry_full_input_liveness_v1.FIELD_SEMANTIC_SIGNED_PATTERN
        # that this field matches, so a spelling like `d1_ret_5_bps` would
        # silently drop it out of the signed-semantics gate.
        out["d1_atr14_bps_canon_v2"] = atr14 / close * 1e4
        out["d1_rsi14_canon_v2"] = _rsi(close, 14)
        ema20 = _ema(close, 20)
        out["d1_ema_slope_20_canon_v2"] = (
            ema20 - ema20.shift(5)
        ) / atr14_positive
        bar_range = high - low
        range_mean = bar_range.rolling(20, min_periods=20).mean()
        range_std = bar_range.rolling(20, min_periods=20).std(ddof=0).replace(
            0.0, np.nan
        )
        out["d1_range_z_20_canon_v2"] = (
            bar_range - range_mean
        ) / range_std
        high20 = high.rolling(20, min_periods=20).max()
        low20 = low.rolling(20, min_periods=20).min()
        out["d1_close_pct_in_20day_range_canon_v2"] = (
            close - low20
        ) / (high20 - low20).replace(0.0, np.nan)
        out["d1_change_5_bps_canon_v2"] = close.pct_change(5) * 10000.0
        out["d1_dist_change_1bar_atr_v4"] = distance.diff()

    if tuple(out.columns) != expected_fields:
        raise RuntimeError(
            "HTF_V4_MODEL_NATIVE_SCALAR_ORDER_INVALID: "
            f"timeframe={timeframe} observed={tuple(out.columns)} "
            f"expected={expected_fields}"
        )
    values = out.to_numpy(dtype=np.float64, copy=False)
    validate_causal_feature_matrix(
        values,
        expected_width=len(expected_fields),
        context=f"HTF_V4_MODEL_NATIVE_SCALARS_{timeframe}",
    )
    return out.astype(np.float32)


def _attach_model_native_mtf_scalar_frame_v4(
    frame: pd.DataFrame,
    scalar_frame: pd.DataFrame,
    *,
    timeframe: str,
) -> None:
    expected_fields = MODEL_NATIVE_MTF_SCALAR_FIELDS_BY_TIMEFRAME_V4[timeframe]
    if not frame.index.equals(scalar_frame.index):
        raise RuntimeError(
            f"HTF_V4_MODEL_NATIVE_SCALAR_TIMESTAMP_MISMATCH: {timeframe}"
        )
    values = np.ascontiguousarray(
        scalar_frame.to_numpy(dtype=np.float32, copy=False)
    )
    if expected_fields:
        warmup_rows = validate_causal_feature_matrix(
            values,
            expected_width=len(expected_fields),
            context=f"HTF_V4_MODEL_NATIVE_SCALARS_{timeframe}",
        )
    else:
        if values.shape != (len(frame), 0):
            raise RuntimeError(
                "HTF_V4_MODEL_NATIVE_SCALAR_EMPTY_MATRIX_INVALID"
            )
        warmup_rows = 0
    frame.attrs["model_native_mtf_scalar_fields_v4"] = expected_fields
    frame.attrs["model_native_mtf_scalars_np_v4"] = values
    frame.attrs["model_native_mtf_scalar_warmup_rows_v4"] = warmup_rows
    frame.attrs["model_native_mtf_scalar_contract_v4"] = (
        MODEL_NATIVE_MTF_SCALAR_CONTRACT_V4
    )


def build_multi_tf_per_bar_features_v4(
    m5_df: pd.DataFrame,
    *,
    v29_registry_constants,
    volatility_squeeze_artifacts: VolatilitySqueezeArtifactSet,
) -> dict:
    """Build all eight causal specialist families from native M5 only.

    ``v29_registry_constants`` is the frozen TRAIN-fitted registry-constants
    payload; it is validated once here and passed to every per-TF surface
    computation.  There is no default (rule 2a).
    """
    registry_constants = require_v29_registry_constants(v29_registry_constants)
    squeeze_artifacts = require_volatility_squeeze_artifact_set(
        volatility_squeeze_artifacts
    )
    base_bar_duration = pd.Timedelta(minutes=5)
    _validate_m5_input(
        m5_df,
        require_volume=True,
        bar_duration=base_bar_duration,
    )
    source = m5_df.copy(deep=False)
    source.index = source.index.as_unit("ns")
    expected_indices = build_multi_tf_v4_closed_timestamp_indices(
        source.index,
    )
    result = {}
    for tf_name in MULTI_TF_RESAMPLE_RULES:
        resampled = _resample_ohlcv(source, tf_name)
        resampled = resampled.dropna(subset=["open", "high", "low", "close"])
        expected_index = expected_indices[tf_name]
        if not resampled.index.is_unique or not expected_index.isin(
            resampled.index
        ).all():
            raise RuntimeError(
                f"HTF_V4_RESAMPLED_TIMESTAMP_GEOMETRY_INVALID: {tf_name}"
            )
        resampled = resampled.loc[expected_index]
        if not resampled.index.equals(expected_index):
            raise RuntimeError(
                f"HTF_V4_RESAMPLED_TIMESTAMP_GEOMETRY_INVALID: {tf_name}"
            )
        computed = compute_per_bar_features_v4(
            resampled,
            timeframe=tf_name,
            v29_registry_constants=registry_constants,
            volatility_squeeze_artifacts=squeeze_artifacts,
        )
        # Retain the exact float32 matrix used to construct the DataFrame so
        # every in-memory V4 consumer sees the same verified bytes as attrs.
        # A fragmented pandas result may otherwise allocate a fresh matrix on
        # each ``to_numpy`` call and violate the one-cache P0 contract.
        feats_np = computed.to_numpy(dtype=np.float32, copy=False)
        feats = pd.DataFrame(
            feats_np,
            index=computed.index,
            columns=MULTI_TF_PER_BAR_FEATURES_V4,
            copy=False,
        )
        ts_int64 = feats.index.asi8.astype(np.int64, copy=True)
        # V4 is the active Entry/Exit owner surface. Keep one shared float32
        # matrix instead of duplicating it in attrs.
        warmup_rows = validate_causal_feature_matrix(
            feats_np,
            expected_width=MULTI_TF_FEATURE_COUNT_V4,
            context=f"HTF_V4_{tf_name}",
        )
        feats.attrs["ts_int64"] = ts_int64
        feats.attrs["feats_np"] = feats_np
        feats.attrs["causal_warmup_rows"] = warmup_rows
        feats.attrs["htf_feature_contract"] = HTF_V4_MATRIX_CONTRACT
        scalar_frame = _compute_model_native_mtf_scalar_frame_v4(
            resampled,
            timeframe=tf_name,
        )
        _attach_model_native_mtf_scalar_frame_v4(
            feats,
            scalar_frame,
            timeframe=tf_name,
        )
        result[tf_name] = feats
    return result



# Records, per verified cache frame object, that its full-matrix validation has
# passed. The frames are immutable for a run, so the O(frame) equality and
# finiteness checks below need to run once per frame, not once per sample. The
# token binds the frame's exact identity (the two cache-array data pointers,
# length and width); any replacement or in-place change misses the token and the
# full validation runs again. The checks themselves are unchanged.
# 2026-08-09 soundness fix: values store ``(frame, token)``. Keying on
# ``id(frame)`` alone was unsound — a freed frame's id can be reused by a new,
# never-validated object (demonstrated). Storing the frame pins it so its id
# stays stable for the memo's lifetime, and a hit additionally requires the
# stored object to be the same one (`is`) with an equal token.
_HTF_FRAMES_VALIDATED: dict = {}


def require_multi_tf_v4_frames(
    features: Mapping[str, pd.DataFrame],
) -> Mapping[str, pd.DataFrame]:
    """Validate the exact ordered fixed-width V4 cache matrices and views."""

    expected_tfs = tuple(MULTI_TF_RESAMPLE_RULES)
    if not isinstance(features, Mapping) or tuple(features) != expected_tfs:
        raise RuntimeError(
            "HTF_V4_CACHE_SET_INVALID: exact ordered M5/M15/H1/H4/D1 required"
        )
    for timeframe in expected_tfs:
        frame = features[timeframe]
        if (
            not isinstance(frame, pd.DataFrame)
            or frame.empty
            or tuple(frame.columns) != MULTI_TF_PER_BAR_FEATURES_V4
            or frame.attrs.get("htf_feature_contract") != HTF_V4_MATRIX_CONTRACT
        ):
            raise RuntimeError(
                f"HTF_V4_CACHE_FRAME_CONTRACT_INVALID: {timeframe}"
            )
        timestamps = np.asarray(frame.attrs.get("ts_int64"))
        verified = np.asarray(frame.attrs.get("feats_np"))
        _frame_token = (
            verified.__array_interface__["data"][0]
            if verified.dtype == np.dtype(np.float32) else None,
            timestamps.__array_interface__["data"][0]
            if timestamps.dtype == np.dtype(np.int64) else None,
            len(frame),
            int(verified.shape[1]) if verified.ndim == 2 else -1,
        )
        _memo_hit = _HTF_FRAMES_VALIDATED.get(id(frame))
        if (
            _memo_hit is not None
            and _memo_hit[0] is frame
            and _memo_hit[1] == _frame_token
        ):
            continue
        frame_values = frame.to_numpy(dtype=np.float32, copy=False)
        if (
            timestamps.dtype != np.dtype(np.int64)
            or timestamps.shape != (len(frame),)
            or np.any(np.diff(timestamps) <= 0)
            or not np.array_equal(frame.index.asi8, timestamps)
            or verified.dtype != np.dtype(np.float32)
            or verified.shape != (len(frame), MULTI_TF_FEATURE_COUNT_V4)
            or not np.shares_memory(frame_values, verified)
            or not np.array_equal(frame_values, verified, equal_nan=True)
        ):
            raise RuntimeError(
                f"HTF_V4_CACHE_VERIFIED_MATRIX_INVALID: {timeframe}"
            )
        warmup_rows = validate_causal_feature_matrix(
            verified,
            expected_width=MULTI_TF_FEATURE_COUNT_V4,
            context=f"HTF_V4_CACHE_{timeframe}",
        )
        if (
            warmup_rows == len(frame)
            or frame.attrs.get("causal_warmup_rows") != warmup_rows
        ):
            raise RuntimeError(
                f"HTF_V4_CACHE_WARMUP_INVALID: {timeframe}"
            )
        _HTF_FRAMES_VALIDATED[id(frame)] = (frame, _frame_token)
    return features


def require_model_native_mtf_scalar_owner_v4(
    features: Mapping[str, pd.DataFrame],
) -> Mapping[str, pd.DataFrame]:
    """Require the exact compact scalar surface on every verified V4 frame."""

    require_multi_tf_v4_frames(features)
    for timeframe, expected_fields in (
        MODEL_NATIVE_MTF_SCALAR_FIELDS_BY_TIMEFRAME_V4.items()
    ):
        frame = features[timeframe]
        fields = frame.attrs.get("model_native_mtf_scalar_fields_v4")
        values = np.asarray(
            frame.attrs.get("model_native_mtf_scalars_np_v4")
        )
        if (
            fields != expected_fields
            or values.dtype != np.dtype(np.float32)
            or values.shape != (len(frame), len(expected_fields))
            or frame.attrs.get("model_native_mtf_scalar_contract_v4")
            != MODEL_NATIVE_MTF_SCALAR_CONTRACT_V4
        ):
            raise RuntimeError(
                f"HTF_V4_MODEL_NATIVE_SCALAR_CONTRACT_INVALID: {timeframe}"
            )
        if expected_fields:
            warmup_rows = validate_causal_feature_matrix(
                values,
                expected_width=len(expected_fields),
                context=f"HTF_V4_MODEL_NATIVE_SCALARS_{timeframe}",
            )
        else:
            warmup_rows = 0
        if frame.attrs.get("model_native_mtf_scalar_warmup_rows_v4") != warmup_rows:
            raise RuntimeError(
                f"HTF_V4_MODEL_NATIVE_SCALAR_WARMUP_INVALID: {timeframe}"
            )
    return features


def bind_model_native_mtf_scalar_owner_v4(
    features: Mapping[str, pd.DataFrame],
    native_m5_ohlcv: pd.DataFrame,
) -> Mapping[str, pd.DataFrame]:
    """Bind deterministic scalar views to a verified cache from its exact M5 source."""

    require_multi_tf_v4_frames(features)
    _validate_m5_input(
        native_m5_ohlcv,
        require_volume=True,
        bar_duration=pd.Timedelta(minutes=5),
    )
    source = native_m5_ohlcv.copy(deep=False)
    source.index = source.index.as_unit("ns")
    expected_indices = build_multi_tf_v4_closed_timestamp_indices(source.index)
    for timeframe in MULTI_TF_RESAMPLE_RULES:
        if not features[timeframe].index.equals(expected_indices[timeframe]):
            raise RuntimeError(
                "HTF_V4_MODEL_NATIVE_SCALAR_SOURCE_GEOMETRY_MISMATCH: "
                f"{timeframe}"
            )
        resampled = _resample_ohlcv(source, timeframe).dropna(
            subset=["open", "high", "low", "close"]
        )
        if not expected_indices[timeframe].isin(resampled.index).all():
            raise RuntimeError(
                "HTF_V4_MODEL_NATIVE_SCALAR_SOURCE_GEOMETRY_MISMATCH: "
                f"{timeframe}"
            )
        resampled = resampled.loc[expected_indices[timeframe]]
        scalar_frame = _compute_model_native_mtf_scalar_frame_v4(
            resampled,
            timeframe=timeframe,
        )
        _attach_model_native_mtf_scalar_frame_v4(
            features[timeframe],
            scalar_frame,
            timeframe=timeframe,
        )
    return require_model_native_mtf_scalar_owner_v4(features)


def project_model_native_mtf_scalars_v4(
    features: Mapping[str, pd.DataFrame],
    target_ts_ns,
    *,
    decision_bar_duration: pd.Timedelta,
) -> dict[str, np.ndarray]:
    """Project the one native-M5 scalar owner onto local M5 or local M1."""

    require_model_native_mtf_scalar_owner_v4(features)
    routes = MODEL_NATIVE_MTF_SCALAR_ROUTES_V4
    if decision_bar_duration not in routes:
        raise RuntimeError(
            "HTF_V4_MODEL_NATIVE_PROJECTION_CLOCK_INVALID: exact M1 or M5 required"
        )
    target = np.asarray(target_ts_ns, dtype=np.int64)
    if (
        target.ndim != 1
        or len(target) < 5
        or np.any(np.diff(target) <= 0)
        or np.any(target % int(decision_bar_duration.value) != 0)
    ):
        raise RuntimeError("HTF_V4_MODEL_NATIVE_PROJECTION_TARGET_INVALID")

    projected: dict[str, np.ndarray] = {}
    for timeframe in routes[decision_bar_duration]:
        fields = MODEL_NATIVE_MTF_SCALAR_FIELDS_BY_TIMEFRAME_V4[timeframe]
        if not fields:
            continue
        frame = features[timeframe]
        timestamps = np.asarray(frame.attrs["ts_int64"], dtype=np.int64)
        values = np.asarray(frame.attrs["model_native_mtf_scalars_np_v4"])
        cutoff = (
            target
            + int(decision_bar_duration.value)
            - int(MULTI_TF_SHIFT[timeframe].value)
        )
        right = np.searchsorted(timestamps, cutoff, side="right") - 1
        valid = right >= 0
        safe = np.clip(right, 0, len(timestamps) - 1)
        for column, name in enumerate(fields):
            if name in projected:
                raise RuntimeError(
                    f"HTF_V4_MODEL_NATIVE_PROJECTION_DUPLICATE_FIELD: {name}"
                )
            aligned = np.full(len(target), np.nan, dtype=np.float64)
            aligned[valid] = values[safe[valid], column]
            projected[name] = aligned

    if set(projected) != set(MODEL_NATIVE_MTF_SCALAR_OUTPUT_FIELDS_V4):
        raise RuntimeError(
            "HTF_V4_MODEL_NATIVE_PROJECTION_FIELDS_INVALID: "
            f"missing={sorted(set(MODEL_NATIVE_MTF_SCALAR_OUTPUT_FIELDS_V4) - set(projected))} "
            f"unexpected={sorted(set(projected) - set(MODEL_NATIVE_MTF_SCALAR_OUTPUT_FIELDS_V4))}"
        )
    ordered = {
        name: projected[name]
        for name in MODEL_NATIVE_MTF_SCALAR_OUTPUT_FIELDS_V4
    }
    validate_causal_feature_matrix(
        np.column_stack(list(ordered.values())),
        expected_width=len(MODEL_NATIVE_MTF_SCALAR_OUTPUT_FIELDS_V4),
        context="HTF_V4_MODEL_NATIVE_PROJECTION",
    )
    return ordered


def model_native_mtf_owner_marker_v4(
    *,
    decision_bar_duration: pd.Timedelta,
) -> dict[str, object]:
    routes = MODEL_NATIVE_MTF_SCALAR_ROUTES_V4
    if decision_bar_duration not in routes:
        raise RuntimeError("HTF_V4_MODEL_NATIVE_OWNER_MARKER_CLOCK_INVALID")
    fields = list(MODEL_NATIVE_MTF_SCALAR_OUTPUT_FIELDS_V4)
    return {
        "schema_version": MODEL_NATIVE_MTF_SCALAR_CONTRACT_V4,
        "source": "exact_native_m5_closed_ohlcv",
        "decision_bar_seconds": int(decision_bar_duration.total_seconds()),
        "route_timeframes": list(routes[decision_bar_duration]),
        "field_order": fields,
        "field_order_sha256": hashlib.sha256(
            json.dumps(
                fields,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
    }


def attach_model_native_mtf_scalars_v4(
    frame: pd.DataFrame,
    *,
    multi_tf: Mapping[str, pd.DataFrame],
    decision_bar_duration: pd.Timedelta,
) -> pd.DataFrame:
    """Attach all persistent MTF scalars once; existing fields are an owner conflict."""

    _validate_m5_input(
        frame,
        require_volume=True,
        bar_duration=decision_bar_duration,
    )
    conflicts = sorted(
        set(MODEL_NATIVE_MTF_SCALAR_OUTPUT_FIELDS_V4) & set(frame.columns)
    )
    if conflicts:
        raise RuntimeError(
            f"HTF_V4_MODEL_NATIVE_DUPLICATE_MTF_OWNER: {conflicts}"
        )
    projected = project_model_native_mtf_scalars_v4(
        multi_tf,
        frame.index.asi8.astype(np.int64, copy=False),
        decision_bar_duration=decision_bar_duration,
    )
    for name in MODEL_NATIVE_MTF_SCALAR_OUTPUT_FIELDS_V4:
        frame[name] = projected[name]
    frame.attrs["model_native_mtf_owner_v4"] = model_native_mtf_owner_marker_v4(
        decision_bar_duration=decision_bar_duration
    )
    return frame


def require_model_native_mtf_owner_marker_v4(
    frame: pd.DataFrame,
    *,
    decision_bar_duration: pd.Timedelta,
) -> dict[str, object]:
    expected = model_native_mtf_owner_marker_v4(
        decision_bar_duration=decision_bar_duration
    )
    if frame.attrs.get("model_native_mtf_owner_v4") != expected:
        raise RuntimeError("HTF_V4_MODEL_NATIVE_OWNER_MARKER_MISSING")
    observed = tuple(
        name
        for name in MODEL_NATIVE_MTF_SCALAR_OUTPUT_FIELDS_V4
        if name in frame.columns
    )
    if observed != MODEL_NATIVE_MTF_SCALAR_OUTPUT_FIELDS_V4:
        raise RuntimeError("HTF_V4_MODEL_NATIVE_OWNER_FIELDS_MISSING")
    return expected


def project_multi_tf_v4_scalars(
    multi_tf: Mapping[str, pd.DataFrame],
    target_ts_ns,
    per_tf_map,
    tfs=("m15", "h1", "h4", "d1"),
    skip=frozenset(),
    *,
    decision_bar_duration: pd.Timedelta,
) -> dict[str, np.ndarray]:
    """Project persistent scalar fields from explicit verified V4 cache bytes."""

    require_multi_tf_v4_frames(multi_tf)
    if decision_bar_duration not in (
        pd.Timedelta(minutes=1),
        pd.Timedelta(minutes=5),
    ):
        raise RuntimeError(
            "HTF_V4_PROJECTION_DECISION_CLOCK_INVALID: exact M1 or M5 required"
        )
    target_ts_ns = np.asarray(target_ts_ns, dtype=np.int64)
    if (
        target_ts_ns.ndim != 1
        or len(target_ts_ns) == 0
        or np.any(np.diff(target_ts_ns) <= 0)
        or np.any(target_ts_ns % int(decision_bar_duration.value) != 0)
    ):
        raise RuntimeError(
            "HTF_V4_PROJECTION_TARGET_INVALID: exact chronological local grid required"
        )
    requested_tfs = tuple(str(name).lower() for name in tfs)
    if (
        any(name.upper() not in MULTI_TF_SHIFT for name in requested_tfs)
        or len(set(requested_tfs)) != len(requested_tfs)
    ):
        raise RuntimeError(
            f"HTF_V4_PROJECTION_TF_INVALID: tfs={requested_tfs}"
        )
    projection = tuple(
        (str(output_name), str(source_name))
        for output_name, source_name in per_tf_map
    )
    if not projection or len(set(projection)) != len(projection):
        raise RuntimeError(
            "HTF_V4_PROJECTION_MAP_INVALID: non-empty unique map required"
        )

    out: dict[str, np.ndarray] = {}
    for tf_lower in requested_tfs:
        tf_key = tf_lower.upper()
        frame = multi_tf[tf_key]
        timestamps = np.asarray(frame.attrs["ts_int64"], dtype=np.int64)
        verified = np.asarray(frame.attrs["feats_np"])
        positions = {
            str(name): index for index, name in enumerate(frame.columns)
        }
        decision_close_ns = target_ts_ns + int(decision_bar_duration.value)
        cutoffs = decision_close_ns - int(MULTI_TF_SHIFT[tf_key].value)
        right = np.searchsorted(timestamps, cutoffs, side="right") - 1
        valid = right >= 0
        safe = np.clip(right, 0, len(timestamps) - 1)
        for output_name, source_name in projection:
            if (tf_lower, output_name) in skip:
                continue
            if source_name not in positions:
                raise RuntimeError(
                    f"HTF_V4_PROJECTION_SOURCE_MISSING: {tf_key}.{source_name}"
                )
            projected = np.full(len(target_ts_ns), np.nan, dtype=np.float64)
            projected[valid] = verified[
                safe[valid],
                positions[source_name],
            ]
            out[f"{tf_lower}_{output_name}_v2"] = projected
    if not out:
        raise RuntimeError("HTF_V4_PROJECTION_EMPTY")
    validate_causal_feature_matrix(
        np.column_stack(list(out.values())),
        expected_width=len(out),
        context="HTF_V4_PROJECTION",
    )
    return out


# Raw continuous MTF context projection. Output names ending in _v2 are
# persistent model fields; no categorical regime is derived here.
MODEL_NATIVE_CONTEXT_MTF_PROJECTION = (
    ("ema20_slope_atr", "ema20_slope_atr"),
    ("ema_stack_aligned", "ema_stack_aligned_v2"),
    ("trend_state_age_bars", "trend_state_age_bars"),
    ("mom_5_atr", "mom_5_atr"),
    ("mom_20_atr", "mom_20_atr"),
    ("rsi14_centered", "rsi14_centered"),
    ("atr_bps_14", "atr_bps_14"),
)
MODEL_NATIVE_CONTEXT_MTF_TIMEFRAMES = MULTI_TF_TIMEFRAMES_LOWER_M5_LAST
# ``lower_wick_pct`` left this tuple on 2026-08-18: its source column was
# retired from MULTI_TF_PER_BAR_FEATURES_V4 in favour of
# ``mtf_candle_raw_{upper,lower}_wick_share``, so the projection's own
# HTF_V4_PROJECTION_SOURCE_MISSING guard would have raised on m5/m15/h1/h4 --
# the D1 skip below was the only lane exempted. The sole consumer is a
# frozen-scope live route, which is why nothing noticed.
MODEL_NATIVE_CONTEXT_MTF_SKIP = frozenset()


_HTF_V4_CACHE_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "cache_identity_sha256",
        "feature_count",
        "feature_names",
        "shift_contract",
        # V30 package 3 (2026-08-13): the declared bin ORIGIN per timeframe.
        # Cadence alone does not identify a bar; the D1 trading-day origin does.
        "resample_origin_contract",
        "builder_version",
        "smc_causal_replay_schema_version",
        "technical_indicator_owner",
        "swing_structure_owner",
        "m5_prebuilt_source",
        "m5_prebuilt_source_sha256",
        "v29_registry_constants",
        "volatility_squeeze_artifact_set",
        "full_input_liveness",
        "tfs",
    }
)
_HTF_V4_CACHE_TF_KEYS = frozenset(
    {
        "n_bars",
        "feature_count",
        "feats_npy",
        "feats_npy_sha256",
        "feats_npy_size_bytes",
        "ts_npy",
        "ts_npy_sha256",
        "ts_npy_size_bytes",
        "first_ts_ns",
        "last_ts_ns",
        "causal_warmup_rows",
    }
)


class MultiTFV4DiskCache(dict):
    """Verified TF mapping with one content-bound disk-cache identity."""

    def __init__(
        self,
        *,
        cache_identity_sha256: str,
        manifest_sha256: str,
        m5_prebuilt_source: str,
        m5_prebuilt_source_sha256: str,
        v29_registry_constants: dict,
        volatility_squeeze_artifacts: VolatilitySqueezeArtifactSet,
    ) -> None:
        super().__init__()
        self.cache_identity_sha256 = cache_identity_sha256
        self.manifest_sha256 = manifest_sha256
        self.m5_prebuilt_source = m5_prebuilt_source
        self.m5_prebuilt_source_sha256 = m5_prebuilt_source_sha256
        self.v29_registry_constants = v29_registry_constants
        self.volatility_squeeze_artifacts = volatility_squeeze_artifacts


def compute_htf_v4_cache_identity(manifest: dict) -> str:
    """Return the canonical identity for a manifest and all declared arrays."""

    identity_payload = dict(manifest)
    identity_payload.pop("cache_identity_sha256", None)
    try:
        encoded = json.dumps(
            identity_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RuntimeError("HTF_V4_CACHE_MANIFEST_INVALID: non-canonical value") from exc
    return hashlib.sha256(encoded).hexdigest()


def _json_object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _cache_path_has_symlink_component(path: Path) -> bool:
    absolute = path if path.is_absolute() else Path.cwd() / path
    return any(component.is_symlink() for component in (absolute, *absolute.parents))


def _read_cache_file_bytes(
    directory_fd: int,
    name: str,
    *,
    expected_sha256: str | None,
    expected_size_bytes: int | None,
    label: str,
) -> bytes:
    """Read one regular cache file once and verify those exact bytes.

    ``dir_fd`` pins the already-opened cache directory. ``O_NOFOLLOW`` prevents
    a manifest-named symlink from being resolved between inventory validation
    and open. The returned bytes are also the bytes passed to ``numpy.load``.
    """

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(name, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise RuntimeError(f"HTF_V4_CACHE_FILE_INVALID: {label}") from exc
    try:
        file_stat = os.fstat(fd)
        if not stat.S_ISREG(file_stat.st_mode):
            raise RuntimeError(f"HTF_V4_CACHE_FILE_INVALID: {label} is not regular")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        observed_size = 0
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
            observed_size += len(chunk)
    finally:
        os.close(fd)
    if expected_size_bytes is not None and observed_size != expected_size_bytes:
        raise RuntimeError(
            f"HTF_V4_CACHE_SIZE_MISMATCH: {label} "
            f"observed={observed_size} expected={expected_size_bytes}"
        )
    observed_sha256 = digest.hexdigest()
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise RuntimeError(
            f"HTF_V4_CACHE_SHA256_MISMATCH: {label} "
            f"observed={observed_sha256} expected={expected_sha256}"
        )
    return b"".join(chunks)


def _exact_cache_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise RuntimeError(
            f"HTF_V4_CACHE_CONTRACT_MISMATCH: {label} must be an exact SHA-256"
        )
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise RuntimeError(
            f"HTF_V4_CACHE_CONTRACT_MISMATCH: {label} must be an exact SHA-256"
        )
    return value


def _exact_cache_int(
    value: object,
    *,
    label: str,
    minimum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise RuntimeError(
            f"HTF_V4_CACHE_CONTRACT_MISMATCH: {label} must be an exact integer"
        )
    observed = int(value)
    if observed < minimum:
        raise RuntimeError(
            f"HTF_V4_CACHE_CONTRACT_MISMATCH: {label}={observed} < {minimum}"
        )
    return observed


def _load_verified_cache_npy(
    directory_fd: int,
    name: str,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
    label: str,
) -> np.ndarray:
    payload = _read_cache_file_bytes(
        directory_fd,
        name,
        expected_sha256=expected_sha256,
        expected_size_bytes=expected_size_bytes,
        label=label,
    )
    try:
        loaded = np.load(io.BytesIO(payload), allow_pickle=False)
    except Exception as exc:
        raise RuntimeError(f"HTF_V4_CACHE_NPY_INVALID: {label}") from exc
    if not isinstance(loaded, np.ndarray):
        raise RuntimeError(f"HTF_V4_CACHE_NPY_INVALID: {label} is not an ndarray")
    return loaded


def load_multi_tf_v4_cache(cache_dir) -> MultiTFV4DiskCache:
    """Load the sole immutable V4 cache after byte and contract verification."""
    supplied = Path(cache_dir).expanduser()
    absolute = supplied if supplied.is_absolute() else Path.cwd() / supplied
    if _cache_path_has_symlink_component(absolute):
        raise RuntimeError(
            f"HTF_V4_CACHE_PATH_INVALID: cache path traverses a symlink: {absolute}"
        )
    try:
        resolved_cache_dir = absolute.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"HTF_V4_CACHE_PATH_INVALID: {absolute}") from exc
    if not resolved_cache_dir.is_dir():
        raise RuntimeError(f"HTF_V4_CACHE_PATH_INVALID: {resolved_cache_dir}")

    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    try:
        directory_fd = os.open(resolved_cache_dir, directory_flags)
    except OSError as exc:
        raise RuntimeError(
            f"HTF_V4_CACHE_PATH_INVALID: {resolved_cache_dir}"
        ) from exc
    try:
        initial_inventory = set(os.listdir(directory_fd))
        if "manifest.json" not in initial_inventory:
            raise RuntimeError(
                f"HTF_V4_CACHE_MANIFEST_MISSING: {resolved_cache_dir / 'manifest.json'}"
            )
        manifest_bytes = _read_cache_file_bytes(
            directory_fd,
            "manifest.json",
            expected_sha256=None,
            expected_size_bytes=None,
            label="manifest.json",
        )
        try:
            manifest = json.loads(
                manifest_bytes.decode("utf-8"),
                object_pairs_hook=_json_object_without_duplicate_keys,
            )
        except (UnicodeError, ValueError) as exc:
            raise RuntimeError(
                f"HTF_V4_CACHE_MANIFEST_INVALID: {resolved_cache_dir / 'manifest.json'}"
            ) from exc
        if not isinstance(manifest, dict):
            raise RuntimeError("HTF_V4_CACHE_MANIFEST_INVALID: root must be an object")
        schema_version = manifest.get("schema_version")
        if schema_version != HTF_V4_CACHE_SCHEMA_VERSION:
            raise RuntimeError(
                "HTF_V4_CACHE_CONTRACT_REQUIRED: reject legacy manifest "
                f"before array load; observed={schema_version!r} "
                f"expected={HTF_V4_CACHE_SCHEMA_VERSION!r}"
            )
        expected_manifest_keys = _HTF_V4_CACHE_MANIFEST_KEYS
        if set(manifest) != expected_manifest_keys:
            raise RuntimeError(
                "HTF_V4_CACHE_CONTRACT_MISMATCH: manifest exact keys differ "
                f"missing={sorted(expected_manifest_keys - set(manifest))} "
                f"unexpected={sorted(set(manifest) - expected_manifest_keys)}"
            )
        expected_shift = {
            tf: str(shift) for tf, shift in MULTI_TF_SHIFT.items()
        }
        matrix_contract = HTF_V4_MATRIX_CONTRACT
        feature_width = MULTI_TF_FEATURE_COUNT_V4
        feature_names = MULTI_TF_PER_BAR_FEATURES_V4
        builder_version = HTF_V4_CACHE_BUILDER_VERSION
        contracts = {
            "schema_version": schema_version,
            "builder_version": builder_version,
            "smc_causal_replay_schema_version": (
                SMC_CAUSAL_REPLAY_SCHEMA_VERSION
            ),
            "technical_indicator_owner": technical_indicator_contract_metadata(),
            "swing_structure_owner": swing_structure_contract_metadata(),
            "feature_count": feature_width,
            "feature_names": list(feature_names),
            "shift_contract": expected_shift,
            "resample_origin_contract": dict(MULTI_TF_RESAMPLE_ORIGIN_CONTRACT),
        }
        for name, expected in contracts.items():
            if manifest.get(name) != expected:
                raise RuntimeError(
                    f"HTF_V4_CACHE_CONTRACT_MISMATCH: {name} observed={manifest.get(name)!r} "
                    f"expected={expected!r}"
                )
        source_path = Path(str(manifest.get("m5_prebuilt_source") or "")).expanduser()
        if not source_path.is_absolute():
            raise RuntimeError(
                "HTF_V4_CACHE_CONTRACT_MISMATCH: m5_prebuilt_source must be absolute"
            )
        m5_prebuilt_source_sha256 = _exact_cache_sha256(
            manifest["m5_prebuilt_source_sha256"],
            label="m5_prebuilt_source_sha256",
        )
        if (
            source_path.is_symlink()
            or not source_path.is_file()
            or source_path.resolve(strict=True) != source_path
            or _sha256_file_exact(source_path) != m5_prebuilt_source_sha256
        ):
            raise RuntimeError(
                "HTF_V4_CACHE_SOURCE_IDENTITY_MISMATCH: the declared M5 "
                "source bytes do not match the cache manifest"
            )
        cache_identity_sha256 = _exact_cache_sha256(
            manifest["cache_identity_sha256"],
            label="cache_identity_sha256",
        )
        computed_cache_identity = compute_htf_v4_cache_identity(manifest)
        if cache_identity_sha256 != computed_cache_identity:
            raise RuntimeError(
                "HTF_V4_CACHE_IDENTITY_MISMATCH: "
                f"observed={cache_identity_sha256} expected={computed_cache_identity}"
            )
        tf_manifest = manifest.get("tfs")
        if not isinstance(tf_manifest, dict) or tuple(tf_manifest) != tuple(
            MULTI_TF_RESAMPLE_RULES
        ):
            raise RuntimeError(
                "HTF_V4_CACHE_CONTRACT_MISMATCH: ordered exact "
                "M5/M15/H1/H4/D1 entries required"
            )
        declared_inventory = {"manifest.json"}
        for tf_name in MULTI_TF_RESAMPLE_RULES:
            info = tf_manifest[tf_name]
            if not isinstance(info, dict) or set(info) != _HTF_V4_CACHE_TF_KEYS:
                observed_keys = set(info) if isinstance(info, dict) else set()
                raise RuntimeError(
                    f"HTF_V4_CACHE_CONTRACT_MISMATCH: {tf_name} exact keys differ "
                    f"missing={sorted(_HTF_V4_CACHE_TF_KEYS - observed_keys)} "
                    f"unexpected={sorted(observed_keys - _HTF_V4_CACHE_TF_KEYS)}"
                )
            feats_name = str(info["feats_npy"])
            ts_name = str(info["ts_npy"])
            expected_names = (f"{tf_name}_feats.npy", f"{tf_name}_ts.npy")
            if (feats_name, ts_name) != expected_names:
                raise RuntimeError(
                    f"HTF_V4_CACHE_CONTRACT_MISMATCH: {tf_name} filenames "
                    f"observed={(feats_name, ts_name)!r} expected={expected_names!r}"
                )
            declared_inventory.update((feats_name, ts_name))
        if initial_inventory != declared_inventory:
            raise RuntimeError(
                "HTF_V4_CACHE_INVENTORY_MISMATCH: "
                f"missing={sorted(declared_inventory - initial_inventory)} "
                f"unexpected={sorted(initial_inventory - declared_inventory)}"
            )

        try:
            manifest_registry_constants = require_v29_registry_constants(
                manifest.get("v29_registry_constants")
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "HTF_V4_CACHE_V29_REGISTRY_CONSTANTS_INVALID"
            ) from exc
        try:
            manifest_squeeze_artifacts = require_volatility_squeeze_artifact_binding(
                manifest.get("volatility_squeeze_artifact_set")
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "HTF_V4_CACHE_VOLATILITY_SQUEEZE_ARTIFACT_SET_INVALID"
            ) from exc
        out = MultiTFV4DiskCache(
            cache_identity_sha256=cache_identity_sha256,
            manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
            m5_prebuilt_source=str(source_path),
            m5_prebuilt_source_sha256=m5_prebuilt_source_sha256,
            v29_registry_constants=manifest_registry_constants,
            volatility_squeeze_artifacts=manifest_squeeze_artifacts,
        )
        for tf_name in MULTI_TF_RESAMPLE_RULES:
            info = tf_manifest[tf_name]
            n_bars = _exact_cache_int(
                info["n_bars"], label=f"{tf_name}.n_bars", minimum=1
            )
            feature_count = _exact_cache_int(
                info["feature_count"],
                label=f"{tf_name}.feature_count",
                minimum=1,
            )
            if feature_count != feature_width:
                raise RuntimeError(
                    f"HTF_V4_CACHE_CONTRACT_MISMATCH: {tf_name}.feature_count "
                    f"observed={feature_count} expected={feature_width}"
                )
            feats_size = _exact_cache_int(
                info["feats_npy_size_bytes"],
                label=f"{tf_name}.feats_npy_size_bytes",
                minimum=1,
            )
            ts_size = _exact_cache_int(
                info["ts_npy_size_bytes"],
                label=f"{tf_name}.ts_npy_size_bytes",
                minimum=1,
            )
            feats_np = _load_verified_cache_npy(
                directory_fd,
                str(info["feats_npy"]),
                expected_sha256=_exact_cache_sha256(
                    info["feats_npy_sha256"],
                    label=f"{tf_name}.feats_npy_sha256",
                ),
                expected_size_bytes=feats_size,
                label=f"{tf_name}.feats_npy",
            )
            ts_int64 = _load_verified_cache_npy(
                directory_fd,
                str(info["ts_npy"]),
                expected_sha256=_exact_cache_sha256(
                    info["ts_npy_sha256"],
                    label=f"{tf_name}.ts_npy_sha256",
                ),
                expected_size_bytes=ts_size,
                label=f"{tf_name}.ts_npy",
            )
            if (
                feats_np.dtype != np.dtype(np.float32)
                or ts_int64.dtype != np.dtype(np.int64)
            ):
                raise RuntimeError(
                    f"HTF_V4_CACHE_CONTRACT_MISMATCH: {tf_name} requires "
                    "float32 features/int64 timestamps"
                )
            if feats_np.shape != (n_bars, feature_width):
                raise RuntimeError(
                    f"HTF_V4_CACHE_CONTRACT_MISMATCH: {tf_name} feature shape "
                    f"observed={feats_np.shape} "
                    f"expected={(n_bars, feature_width)}"
                )
            if ts_int64.shape != (n_bars,) or np.any(np.diff(ts_int64) <= 0):
                raise RuntimeError(
                    f"HTF_V4_CACHE_CONTRACT_MISMATCH: {tf_name} timestamps invalid"
                )
            warmup_rows = validate_causal_feature_matrix(
                feats_np,
                expected_width=feature_width,
                context=f"HTF_V4_CACHE_{tf_name}",
            )
            if warmup_rows == len(feats_np):
                raise RuntimeError(
                    f"HTF_V4_CACHE_WARMUP_INCOMPLETE: {tf_name} has no complete row"
                )
            expected_meta = {
                "n_bars": n_bars,
                "feature_count": feature_width,
                "first_ts_ns": int(ts_int64[0]),
                "last_ts_ns": int(ts_int64[-1]),
                "causal_warmup_rows": warmup_rows,
            }
            for name, expected in expected_meta.items():
                observed = _exact_cache_int(
                    info[name],
                    label=f"{tf_name}.{name}",
                    minimum=0,
                )
                if observed != expected:
                    raise RuntimeError(
                        f"HTF_V4_CACHE_CONTRACT_MISMATCH: {tf_name}.{name} "
                        f"observed={observed!r} expected={expected!r}"
                    )
            # Keep one verified feature matrix. DataFrame columns and the
            # fast-path attrs must be two views of those same bytes; a separate
            # placeholder matrix would let consumers read unverified values and
            # would double the cache's resident memory.
            idx = pd.DatetimeIndex(ts_int64.astype("datetime64[ns]"), tz="UTC")
            verified_feats = np.ascontiguousarray(feats_np)
            df = pd.DataFrame(
                verified_feats,
                index=idx,
                columns=feature_names,
                copy=False,
            )
            frame_values = df.to_numpy(dtype=np.float32, copy=False)
            if (
                not np.shares_memory(frame_values, verified_feats)
                or not np.array_equal(frame_values, verified_feats, equal_nan=True)
            ):
                raise RuntimeError(
                    f"HTF_V4_CACHE_MATRIX_VIEW_INVALID: {tf_name}"
                )
            df.attrs["ts_int64"] = np.ascontiguousarray(ts_int64)
            df.attrs["feats_np"] = frame_values
            df.attrs["causal_warmup_rows"] = warmup_rows
            df.attrs["htf_feature_contract"] = matrix_contract
            out[tf_name] = df
        try:
            require_multi_tf_v4_liveness_contract(
                manifest.get("full_input_liveness")
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "HTF_V4_CACHE_FULL_INPUT_LIVENESS_INVALID"
            ) from exc
        observed_liveness = build_multi_tf_v4_liveness_contract(out)
        if (
            observed_liveness.get("decision") != "PASS"
            or manifest.get("full_input_liveness") != observed_liveness
        ):
            raise RuntimeError(
                "HTF_V4_CACHE_FULL_INPUT_LIVENESS_INVALID"
            )
        final_inventory = set(os.listdir(directory_fd))
        if final_inventory != declared_inventory:
            raise RuntimeError(
                "HTF_V4_CACHE_INVENTORY_CHANGED_DURING_LOAD: "
                f"missing={sorted(declared_inventory - final_inventory)} "
                f"unexpected={sorted(final_inventory - declared_inventory)}"
            )
        return out
    finally:
        os.close(directory_fd)



# Per-frame validation memo for slice_multi_tf_v4_window. Keyed by the frame's
# id and bound to its exact cache-array identities, so a reused immutable frame
# is validated once instead of on every window slice. Bounded to the handful of
# multi-TF frames a run holds.
# 2026-08-09 soundness fix: values store ``(frame, token)`` — see
# _HTF_FRAMES_VALIDATED. Pinning the frame keeps its id stable; a hit requires
# identity (`is`) plus an equal token, so a recycled id can never inherit a
# freed frame's validation.
_HTF_WINDOW_VALIDATED: dict = {}


def slice_multi_tf_v4_window(
    feats: pd.DataFrame, target_ts: pd.Timestamp, n: int, tf_shift: pd.Timedelta,
) -> np.ndarray:
    """Slice the last `n` per-bar feature rows whose close-time is <= (target_ts - tf_shift).

    Returns an exact finite ``(n, n_features)`` float32 array. Missing history,
    indicator warmup, malformed cache metadata, and non-finite evidence are hard
    errors; this owner never pads or substitutes a neutral value.

    `tf_shift` enforces the "only closed bars" invariant: e.g. for H1, target=12:35
    means we use H1 bars closing at-or-before 11:35 (the 11:00 H1 bar, since
    12:00 H1 bar hasn't closed yet at 12:35).

    Verified V4 fast path: when `feats.attrs["ts_int64"]` and `feats.attrs["feats_np"]`
    are present (set by build_multi_tf_per_bar_features), we use numpy
    searchsorted on int64 timestamps — ~100× faster than pandas .loc.
    """
    if not isinstance(feats, pd.DataFrame) or feats.empty:
        raise RuntimeError("HTF_WINDOW_SOURCE_MISSING: exact non-empty feature table required")
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or int(n) <= 0:
        raise RuntimeError(f"HTF_WINDOW_LENGTH_INVALID: n={n!r}")
    n = int(n)
    if not isinstance(tf_shift, pd.Timedelta) or tf_shift <= pd.Timedelta(0):
        raise RuntimeError(f"HTF_WINDOW_SHIFT_INVALID: tf_shift={tf_shift!r}")
    target = pd.Timestamp(target_ts)
    if target.tzinfo is None or target.utcoffset() != pd.Timedelta(0):
        raise RuntimeError("HTF_WINDOW_TARGET_INVALID: target_ts must be timezone-aware UTC")
    declared_contract = feats.attrs.get("htf_feature_contract")
    if (
        declared_contract != HTF_V4_MATRIX_CONTRACT
        or tuple(feats.columns) != MULTI_TF_PER_BAR_FEATURES_V4
    ):
        raise RuntimeError(
            "HTF_V4_WINDOW_SOURCE_CONTRACT_INVALID: exact fixed-width V4 required"
        )

    ts_int64 = np.asarray(feats.attrs.get("ts_int64"))
    feats_np = np.asarray(feats.attrs.get("feats_np"))
    width = int(feats.shape[1])
    # The cache-array validation compares the entire per-timeframe frame
    # (e.g. 476k x MULTI_TF_FEATURE_COUNT_V4 for M5) with np.array_equal on
    # every window slice. The
    # frame is immutable during a run, so the full check is run once per frame
    # object and memoised: a token bound to this frame's exact identity
    # (id, shape, and the two cache-array identities) records that it passed.
    # The check itself is unchanged; only its per-window repetition is removed.
    _seen = _HTF_WINDOW_VALIDATED.get(id(feats))
    _token = (feats_np.__array_interface__["data"][0], ts_int64.__array_interface__["data"][0], len(feats), width)
    if _seen is None or _seen[0] is not feats or _seen[1] != _token:
        if (
            ts_int64.dtype != np.dtype(np.int64)
            or ts_int64.shape != (len(feats),)
            or feats_np.dtype != np.dtype(np.float32)
            or feats_np.shape != (len(feats), width)
            or not np.shares_memory(
                feats.to_numpy(dtype=np.float32, copy=False),
                feats_np,
            )
            or not np.array_equal(
                feats.to_numpy(dtype=np.float32, copy=False),
                feats_np,
                equal_nan=True,
            )
        ):
            raise RuntimeError("HTF_WINDOW_SOURCE_INVALID: malformed exact cache arrays")
        _HTF_WINDOW_VALIDATED[id(feats)] = (feats, _token)
    warmup_rows = feats.attrs.get("causal_warmup_rows")
    if (
        isinstance(warmup_rows, bool)
        or not isinstance(warmup_rows, (int, np.integer))
        or not 0 <= int(warmup_rows) <= len(feats)
    ):
        raise RuntimeError("HTF_WINDOW_SOURCE_INVALID: causal warmup metadata missing")

    cutoff_ns = int(target.value) - int(tf_shift.value)
    right = int(np.searchsorted(ts_int64, cutoff_ns, side="right"))
    if right < n:
        raise RuntimeError(
            f"HTF_WINDOW_HISTORY_INSUFFICIENT: need={n} closed_rows={right} target={target.isoformat()}"
        )
    left = right - n
    if left < int(warmup_rows):
        raise RuntimeError(
            f"HTF_WINDOW_WARMUP_INCOMPLETE: first_row={left} warmup_rows={int(warmup_rows)}"
        )
    tail = np.asarray(feats_np[left:right], dtype=np.float32)
    if tail.shape != (n, width) or not np.isfinite(tail).all():
        raise RuntimeError("HTF_WINDOW_SOURCE_INVALID: selected feature evidence is non-finite")
    return np.ascontiguousarray(tail)


def get_model_native_multi_tf_route_windows(
    features: dict[str, pd.DataFrame],
    *,
    decision_bar_start: pd.Timestamp,
    per_tf_seq_lens: dict[str, int],
    route_timeframes: tuple[str, ...],
    base_bar_duration: pd.Timedelta,
) -> dict[str, np.ndarray]:
    """Slice one canonical Entry or Exit MTF route from the shared V4 cache.

    Entry and Exit deliberately use this same owner.  Their only differences
    are the local decision clock and the exact route declared by the shared
    feature-base contract.  The cache remains M5/M15/H1/H4/D1; no route copies,
    padding, neutral values or computed-feature resampling are permitted.
    """

    from gx1.contracts.entry_exit_feature_base_v1 import (
        ENTRY_DECISION_BAR_SECONDS,
        ENTRY_MTF_CONTEXT_TIMEFRAMES,
        EXIT_DECISION_BAR_SECONDS,
        EXIT_MTF_CONTEXT_TIMEFRAMES,
    )

    expected_cache_tfs = tuple(MULTI_TF_RESAMPLE_RULES)
    require_multi_tf_v4_frames(features)
    route = tuple(route_timeframes)
    canonical_routes = {
        tuple(ENTRY_MTF_CONTEXT_TIMEFRAMES): pd.Timedelta(
            seconds=ENTRY_DECISION_BAR_SECONDS
        ),
        tuple(EXIT_MTF_CONTEXT_TIMEFRAMES): pd.Timedelta(
            seconds=EXIT_DECISION_BAR_SECONDS
        ),
    }
    if route not in canonical_routes:
        raise RuntimeError(
            f"MODEL_NATIVE_MTF_ROUTE_INVALID: observed={route!r}"
        )
    if base_bar_duration != canonical_routes[route]:
        raise RuntimeError(
            "MODEL_NATIVE_MTF_LOCAL_CLOCK_INVALID: "
            f"route={route!r} observed={base_bar_duration} "
            f"expected={canonical_routes[route]}"
        )
    if (
        not isinstance(per_tf_seq_lens, dict)
        or tuple(per_tf_seq_lens) != expected_cache_tfs
        or any(
            isinstance(per_tf_seq_lens[tf], bool)
            or not isinstance(per_tf_seq_lens[tf], (int, np.integer))
            or int(per_tf_seq_lens[tf]) <= 0
            for tf in expected_cache_tfs
        )
    ):
        raise RuntimeError(
            "MODEL_NATIVE_MTF_SEQUENCE_LENGTHS_INVALID: exact ordered positive "
            "M5/M15/H1/H4/D1 mapping required"
        )
    target = pd.Timestamp(decision_bar_start)
    if target.tz is None or target.utcoffset() != pd.Timedelta(0):
        raise RuntimeError(
            "MODEL_NATIVE_MTF_DECISION_TIMESTAMP_INVALID: timezone-aware UTC required"
        )
    availability = target + base_bar_duration
    return {
        tf: slice_multi_tf_v4_window(
            features[tf],
            availability,
            n=int(per_tf_seq_lens[tf]),
            tf_shift=MULTI_TF_SHIFT[tf],
        )
        for tf in route
    }


def require_multi_tf_decision_window_coverage(
    features: dict[str, pd.DataFrame],
    *,
    per_tf_seq_lens: dict[str, int],
    decision_times_by_route_split: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Prove Entry +5m and Exit +1m TRAIN/VAL routes on one V4 cache."""

    from gx1.contracts.entry_exit_feature_base_v1 import (
        ENTRY_DECISION_BAR_SECONDS,
        ENTRY_MTF_CONTEXT_TIMEFRAMES,
        EXIT_DECISION_BAR_SECONDS,
        EXIT_MTF_CONTEXT_TIMEFRAMES,
    )

    pyramid = require_multi_tf_resolution_pyramid(per_tf_seq_lens)
    expected_tfs = tuple(MULTI_TF_RESAMPLE_RULES)
    try:
        require_multi_tf_v4_frames(features)
    except RuntimeError as exc:
        raise RuntimeError(
            "MULTI_TF_DECISION_COVERAGE_FEATURE_SET_INVALID: exact ordered "
            "fixed-width V4 M5/M15/H1/H4/D1 cache required"
        ) from exc
    if (
        not isinstance(decision_times_by_route_split, dict)
        or tuple(decision_times_by_route_split) != ("entry", "exit")
        or any(
            not isinstance(route_splits, dict)
            or tuple(route_splits) != ("train", "val")
            for route_splits in decision_times_by_route_split.values()
        )
    ):
        raise RuntimeError(
            "MULTI_TF_DECISION_COVERAGE_ROUTE_SPLIT_SET_INVALID: exact ordered "
            "entry/exit and train/val decision times required"
        )

    route_specs = {
        "entry": {
            "timeframes": tuple(ENTRY_MTF_CONTEXT_TIMEFRAMES),
            "base_bar_duration": pd.Timedelta(
                seconds=ENTRY_DECISION_BAR_SECONDS
            ),
        },
        "exit": {
            "timeframes": tuple(EXIT_MTF_CONTEXT_TIMEFRAMES),
            "base_bar_duration": pd.Timedelta(
                seconds=EXIT_DECISION_BAR_SECONDS
            ),
        },
    }
    route_rows: dict[str, dict[str, object]] = {}
    route_windows: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}
    for route, spec in route_specs.items():
        split_bounds: dict[str, dict[str, object]] = {}
        for split, raw_times in decision_times_by_route_split[route].items():
            try:
                times = pd.DatetimeIndex(
                    pd.to_datetime(raw_times, utc=True, errors="raise")
                )
            except Exception as exc:
                raise RuntimeError(
                    f"MULTI_TF_DECISION_COVERAGE_TIME_INVALID: {route}.{split}"
                ) from exc
            if (
                times.empty
                or times.hasnans
                or not times.is_monotonic_increasing
                or not times.is_unique
            ):
                raise RuntimeError(
                    "MULTI_TF_DECISION_COVERAGE_TIME_INVALID: "
                    f"{route}.{split} must be non-empty, unique and chronological"
                )
            first = pd.Timestamp(times[0])
            last = pd.Timestamp(times[-1])
            split_bounds[split] = {
                "rows": int(len(times)),
                "first_utc": first.isoformat(),
                "last_utc": last.isoformat(),
            }
            for edge, target in (("first", first), ("last", last)):
                try:
                    route_windows[(route, split, edge)] = (
                        get_model_native_multi_tf_route_windows(
                            features,
                            decision_bar_start=target,
                            per_tf_seq_lens=per_tf_seq_lens,
                            route_timeframes=spec["timeframes"],
                            base_bar_duration=spec["base_bar_duration"],
                        )
                    )
                except RuntimeError as exc:
                    raise RuntimeError(
                        "MULTI_TF_DECISION_COVERAGE_UNAVAILABLE: "
                        f"{route}.{split}.{edge} target={target.isoformat()}: {exc}"
                    ) from exc
        route_rows[route] = {
            "timeframes": list(spec["timeframes"]),
            "target_availability_shift_seconds": int(
                spec["base_bar_duration"].total_seconds()
            ),
            "split_bounds": split_bounds,
        }

    per_tf: dict[str, object] = {}
    for tf in expected_tfs:
        frame = features[tf]
        n = int(per_tf_seq_lens[tf])
        route_metadata: dict[str, object] = {}
        for route, spec in route_specs.items():
            enabled = tf in spec["timeframes"]
            boundary_rows: dict[str, object] = {}
            if enabled:
                for split in ("train", "val"):
                    bounds = route_rows[route]["split_bounds"][split]
                    for edge in ("first", "last"):
                        window = route_windows[(route, split, edge)][tf]
                        boundary_rows[f"{split}_{edge}"] = {
                            "target_utc": bounds[f"{edge}_utc"],
                            "window_sha256": hashlib.sha256(
                                np.ascontiguousarray(
                                    window,
                                    dtype="<f4",
                                ).tobytes()
                            ).hexdigest(),
                        }
            route_metadata[route] = {
                "enabled": enabled,
                "boundaries": boundary_rows,
            }
        per_tf[tf] = {
            "seq_len": n,
            "coverage_seconds": pyramid["coverage_seconds"][tf],
            "causal_warmup_rows": int(frame.attrs["causal_warmup_rows"]),
            "routes": route_metadata,
        }

    payload: dict[str, object] = {
        "schema_version": "entry_exit_multi_tf_decision_window_coverage_v2",
        "cache_contract": HTF_V4_MATRIX_CONTRACT,
        "routes": route_rows,
        "resolution_pyramid": pyramid,
        "per_tf": per_tf,
        "all_route_split_boundaries_sliceable": True,
    }
    payload["contract_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return require_multi_tf_decision_window_coverage_metadata(
        payload,
        per_tf_seq_lens=per_tf_seq_lens,
    )


def require_multi_tf_decision_window_coverage_metadata(
    value: Mapping[str, object],
    *,
    per_tf_seq_lens: dict[str, int],
) -> dict[str, object]:
    """Strictly validate the immutable split-boundary coverage proof."""

    expected_keys = {
        "schema_version",
        "cache_contract",
        "routes",
        "resolution_pyramid",
        "per_tf",
        "all_route_split_boundaries_sliceable",
        "contract_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise RuntimeError("MULTI_TF_DECISION_COVERAGE_METADATA_KEYS_INVALID")
    payload = dict(value)
    pyramid = require_multi_tf_resolution_pyramid(per_tf_seq_lens)
    if (
        payload["schema_version"]
        != "entry_exit_multi_tf_decision_window_coverage_v2"
        or payload["cache_contract"] != HTF_V4_MATRIX_CONTRACT
        or payload["resolution_pyramid"] != pyramid
        or payload["all_route_split_boundaries_sliceable"] is not True
    ):
        raise RuntimeError("MULTI_TF_DECISION_COVERAGE_METADATA_INVALID")
    from gx1.contracts.entry_exit_feature_base_v1 import (
        ENTRY_DECISION_BAR_SECONDS,
        ENTRY_MTF_CONTEXT_TIMEFRAMES,
        EXIT_DECISION_BAR_SECONDS,
        EXIT_MTF_CONTEXT_TIMEFRAMES,
    )
    expected_routes = {
        "entry": (
            tuple(ENTRY_MTF_CONTEXT_TIMEFRAMES),
            ENTRY_DECISION_BAR_SECONDS,
        ),
        "exit": (
            tuple(EXIT_MTF_CONTEXT_TIMEFRAMES),
            EXIT_DECISION_BAR_SECONDS,
        ),
    }
    routes = payload["routes"]
    if not isinstance(routes, dict) or tuple(routes) != tuple(expected_routes):
        raise RuntimeError("MULTI_TF_DECISION_COVERAGE_ROUTE_METADATA_INVALID")
    parsed_route_bounds: dict[
        str,
        dict[str, tuple[pd.Timestamp, pd.Timestamp]],
    ] = {}
    for route, (timeframes, availability_seconds) in expected_routes.items():
        raw_route = routes[route]
        if not isinstance(raw_route, dict) or set(raw_route) != {
            "timeframes",
            "target_availability_shift_seconds",
            "split_bounds",
        }:
            raise RuntimeError(
                "MULTI_TF_DECISION_COVERAGE_ROUTE_METADATA_INVALID"
            )
        if (
            raw_route["timeframes"] != list(timeframes)
            or raw_route["target_availability_shift_seconds"]
            != availability_seconds
            or not isinstance(raw_route["split_bounds"], dict)
            or tuple(raw_route["split_bounds"]) != ("train", "val")
        ):
            raise RuntimeError(
                "MULTI_TF_DECISION_COVERAGE_ROUTE_METADATA_INVALID"
            )
        parsed_route_bounds[route] = {}
        for split, raw in raw_route["split_bounds"].items():
            if not isinstance(raw, dict) or set(raw) != {
                "rows",
                "first_utc",
                "last_utc",
            }:
                raise RuntimeError(
                    "MULTI_TF_DECISION_COVERAGE_SPLIT_METADATA_INVALID"
                )
            rows = raw["rows"]
            first = pd.Timestamp(raw["first_utc"])
            last = pd.Timestamp(raw["last_utc"])
            if (
                isinstance(rows, bool)
                or not isinstance(rows, int)
                or rows <= 0
                or first.tzinfo is None
                or last.tzinfo is None
                or first.utcoffset() != pd.Timedelta(0)
                or last.utcoffset() != pd.Timedelta(0)
                or first > last
            ):
                raise RuntimeError(
                    "MULTI_TF_DECISION_COVERAGE_SPLIT_METADATA_INVALID"
                )
            parsed_route_bounds[route][split] = (first, last)
    per_tf = payload["per_tf"]
    if not isinstance(per_tf, dict) or tuple(per_tf) != tuple(
        MULTI_TF_RESAMPLE_RULES
    ):
        raise RuntimeError("MULTI_TF_DECISION_COVERAGE_TF_METADATA_INVALID")
    expected_boundaries = tuple(
        f"{split}_{edge}"
        for split in ("train", "val")
        for edge in ("first", "last")
    )
    for tf, raw in per_tf.items():
        if not isinstance(raw, dict) or set(raw) != {
            "seq_len",
            "coverage_seconds",
            "causal_warmup_rows",
            "routes",
        }:
            raise RuntimeError(
                "MULTI_TF_DECISION_COVERAGE_TF_METADATA_INVALID"
            )
        warmup = raw["causal_warmup_rows"]
        if (
            raw["seq_len"] != per_tf_seq_lens[tf]
            or raw["coverage_seconds"] != pyramid["coverage_seconds"][tf]
            or isinstance(warmup, bool)
            or not isinstance(warmup, int)
            or warmup < 0
        ):
            raise RuntimeError(
                "MULTI_TF_DECISION_COVERAGE_TF_METADATA_INVALID"
            )
        tf_routes = raw["routes"]
        if not isinstance(tf_routes, dict) or tuple(tf_routes) != tuple(
            expected_routes
        ):
            raise RuntimeError(
                "MULTI_TF_DECISION_COVERAGE_TF_ROUTE_METADATA_INVALID"
            )
        for route, (route_tfs, _availability) in expected_routes.items():
            route_row = tf_routes[route]
            enabled = tf in route_tfs
            if (
                not isinstance(route_row, dict)
                or set(route_row) != {"enabled", "boundaries"}
                or route_row["enabled"] is not enabled
                or not isinstance(route_row["boundaries"], dict)
                or tuple(route_row["boundaries"])
                != (expected_boundaries if enabled else ())
            ):
                raise RuntimeError(
                    "MULTI_TF_DECISION_COVERAGE_TF_ROUTE_METADATA_INVALID"
                )
            for boundary, row in route_row["boundaries"].items():
                if not isinstance(row, dict) or set(row) != {
                    "target_utc",
                    "window_sha256",
                }:
                    raise RuntimeError(
                        "MULTI_TF_DECISION_COVERAGE_BOUNDARY_METADATA_INVALID"
                    )
                split, edge = boundary.rsplit("_", 1)
                expected_target = parsed_route_bounds[route][split][
                    0 if edge == "first" else 1
                ]
                if (
                    pd.Timestamp(row["target_utc"]) != expected_target
                    or not isinstance(row["window_sha256"], str)
                    or len(row["window_sha256"]) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in row["window_sha256"]
                    )
                ):
                    raise RuntimeError(
                        "MULTI_TF_DECISION_COVERAGE_BOUNDARY_METADATA_INVALID"
                    )
    observed_hash = payload.pop("contract_sha256")
    expected_hash = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if observed_hash != expected_hash:
        raise RuntimeError("MULTI_TF_DECISION_COVERAGE_HASH_INVALID")
    payload["contract_sha256"] = observed_hash
    return payload
