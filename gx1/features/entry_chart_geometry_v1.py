"""Canonical Entry chart-geometry features.

This layer computes multi-timeframe EMA trend state, sided support/resistance
proximity, failed-breakout reversal pressure, compression-release breakout
pressure and late-trend reversal risk from already-materialized snap/ctx_cont
inputs.

V30 package 7 (2026-08-13), operator-authorized: the layer no longer claims to
draw what a discretionary chart trader draws by hand.  Its input tuple is
pre-reduced scalars — no price, no pivot coordinate, no bar index, no slope, no
anchor — so every field it used to emit under a ``trendline`` / ``rail`` /
``channel`` / ``triangle`` / ``flag`` / ``apex`` / ``fib`` name was NAME-ONLY by
algebra (docs/INDICATOR_FIDELITY_AUDIT_20260813.md §1a).  Those 43 columns are
removed here.  The real implementations already exist and are wired on every
timeframe: ``level_registry_v1`` (touch counts, break, retest hold/fail) and
``trendline_registry_v1`` (fitted 2-anchor lines, slope, touch counts,
ACTIVE/BROKEN, retest window, apex solve), both shipped 2026-08-11.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from gx1.features.entry_volatility_semantics_v1 import (
    atr_ratio_compression_pressure,
    atr_ratio_expansion_pressure,
    bollinger_expansion_pressure,
    bollinger_squeeze_pressure,
)


CHART_GEOMETRY_FEATURE_VERSION = (
    "entry_chart_geometry_v5_20260813_name_only_composite_removal"
)
CHART_GEOMETRY_FEATURE_PREFIX = "chart.geometry_"

CHART_GEOMETRY_SOURCE_FIELDS = (
    "snap._v1_ema_diff",
    "snap.ema20_slope",
    "snap.pos_vs_ema200",
    "ctx_cont._v1h1_ema_diff",
    "ctx_cont._v1h1_slope5",
    "ctx_cont._v1h4_ema_diff",
    "ctx_cont._v1h4_slope5",
    "ctx_cont.d1_ema_slope_20_canon_v2",
    "ctx_cont.m15_trend_sign_canon_v2",
    "ctx_cont.regime_stack_sum_v3",
    "ctx_cont.regime_tf_agreement_v3",
    "ctx_cont.regime_divergence_flag_v3",
    "ctx_cont.dist_last_swing_high_atr",
    "ctx_cont.dist_last_swing_low_atr",
    "ctx_cont.bars_since_swing_high",
    "ctx_cont.bars_since_swing_low",
    "ctx_cont.sr_support_proximity_exp",
    "ctx_cont.sr_resistance_proximity_exp",
    "ctx_cont.sr_support_minus_resistance_prox",
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
    # V30 package 7 (2026-08-13): `snap.smc_premium_discount`,
    # `ctx_cont.retracement_from_last_impulse` and
    # `ctx_cont.d1_close_pct_in_20day_range_canon_v2` were declared here only to
    # feed the removed Fibonacci block, and `snap.smc_sweep_size_atr` /
    # `ctx_cont.smc_sweep_size_recent_tau12` only the removed
    # `line_pattern_tail_risk`.  A declared source this layer never reads is the
    # same name-only defect class the removal repairs, so they are dropped from
    # the required tuple (rule 10).  All five remain model inputs elsewhere:
    # the two sweep-size fields and `snap.smc_premium_discount` are declared
    # sources of entry_smc_liquidity_quality_v1 / entry_foundation_structure_v1
    # / entry_support_resistance_memory_v1, and both ctx fields are members of
    # MODEL_NATIVE_CTX_CONT_FIELDS (`d1_close_pct_in_20day_range_canon_v2` is
    # additionally routed to chart_geometry_encoder by
    # entry_specialist_feature_groups_v1).
    "snap.smc_bos_up",
    "snap.smc_bos_down",
    "ctx_cont.smc_bos_pressure_last12",
    "ctx_cont.smc_bos_pressure_last48",
    "snap.smc_choch",
    "ctx_cont.smc_choch_recent_tau12",
    "ctx_cont.smc_choch_recent_tau24",
    "snap.smc_sweep_up",
    "snap.smc_sweep_down",
    "ctx_cont.smc_sweep_bull_pressure_last12",
    "ctx_cont.smc_sweep_bull_pressure_last48",
    "ctx_cont.smc_sweep_recency_tau24",
    "snap.body_pct",
    "ctx_cont.H1_range_compression_ratio",
    "ctx_cont.M15_range_compression_ratio",
    "snap._v1_bb_squeeze_20_2",
    "snap.atr_z",
    "snap.rvol_20",
    "snap.vol_ratio_5_20",
    "ctx_cont.h1_trend_age_bars_norm_v2",
    "ctx_cont.h4_trend_age_bars_norm_v2",
    "ctx_cont.D1_atr_percentile_252",
)


def _name_index(names: Iterable[str]) -> dict[str, int]:
    values = list(names)
    invalid = [name for name in values if not isinstance(name, str) or not name]
    if invalid:
        raise RuntimeError(f"CHART_GEOMETRY_FEATURE_NAMES_INVALID: {invalid[:10]}")
    duplicates = sorted({name for name in values if values.count(name) > 1})
    if duplicates:
        raise RuntimeError(f"CHART_GEOMETRY_FEATURE_NAMES_DUPLICATE: {duplicates[:10]}")
    return {name: i for i, name in enumerate(values)}


def missing_chart_geometry_source_fields(feature_names: Iterable[str]) -> list[str]:
    available = set(feature_names)
    return [name for name in CHART_GEOMETRY_SOURCE_FIELDS if name not in available]


def _require_source_matrix(
    x: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, dict[str, int]]:
    try:
        matrix = np.asarray(x, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("CHART_GEOMETRY_INPUT_NOT_NUMERIC") from exc
    if matrix.ndim != 2:
        raise RuntimeError(f"CHART_GEOMETRY_INPUT_NOT_2D: shape={matrix.shape}")
    if matrix.shape[0] == 0:
        raise RuntimeError("CHART_GEOMETRY_INPUT_EMPTY")
    if len(feature_names) != matrix.shape[1]:
        raise RuntimeError(
            "CHART_GEOMETRY_FEATURE_NAME_COUNT_MISMATCH: "
            f"names={len(feature_names)} columns={matrix.shape[1]}"
        )
    index = _name_index(feature_names)
    missing = missing_chart_geometry_source_fields(feature_names)
    if missing:
        raise RuntimeError(
            "CHART_GEOMETRY_SOURCE_FIELDS_MISSING: "
            f"{missing[:30]} total={len(missing)}"
        )
    if not np.isfinite(matrix).all():
        bad = np.argwhere(~np.isfinite(matrix))[0]
        row, column = int(bad[0]), int(bad[1])
        raise RuntimeError(
            "CHART_GEOMETRY_SOURCE_NONFINITE: "
            f"row={row} field={feature_names[column]}"
        )
    return matrix, index


def _col(x: np.ndarray, index: dict[str, int], name: str) -> np.ndarray:
    try:
        column = index[name]
    except KeyError as exc:
        raise RuntimeError(f"CHART_GEOMETRY_SOURCE_FIELD_MISSING: {name}") from exc
    arr = np.asarray(x[:, column], dtype=np.float32)
    if arr.ndim != 1 or not np.isfinite(arr).all():
        raise RuntimeError(f"CHART_GEOMETRY_SOURCE_FIELD_INVALID: {name}")
    return arr


def _clip(arr: np.ndarray, lo: float = -25.0, hi: float = 25.0) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float32)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise RuntimeError(f"CHART_GEOMETRY_DERIVED_VALUE_INVALID: shape={values.shape}")
    return np.clip(values, lo, hi).astype(np.float32, copy=False)


def _clip01(arr: np.ndarray) -> np.ndarray:
    return _clip(arr, 0.0, 1.0)


def _pos(arr: np.ndarray) -> np.ndarray:
    return np.maximum(arr, 0.0).astype(np.float32, copy=False)


def _neg(arr: np.ndarray) -> np.ndarray:
    return np.maximum(-arr, 0.0).astype(np.float32, copy=False)


def _prox_abs(arr: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.abs(arr))).astype(np.float32, copy=False)


def _prox_valid_side(arr: np.ndarray, *, positive_is_valid: bool) -> np.ndarray:
    """One-sided level proximity: 1/(1+|dist|) on the level's intact side, else 0.

    1/(1+|dist|) on a signed distance discards which side of the level price
    is on and reads a broken/swept level as "near" — a broken support is not
    support. Each field's intact side follows its producer's exact sign
    convention (augment_forward_outcome_v2 pivot/liquidity emitters).
    """
    values = np.asarray(arr, dtype=np.float32)
    if positive_is_valid:
        return np.where(values >= 0.0, 1.0 / (1.0 + values), 0.0).astype(np.float32)
    return np.where(values <= 0.0, 1.0 / (1.0 - values), 0.0).astype(np.float32)


def _recency(arr: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.maximum(arr, 0.0))).astype(np.float32, copy=False)


def _tanh(arr: np.ndarray, scale: float = 1.0) -> np.ndarray:
    return np.tanh(arr / max(float(scale), 1e-6)).astype(np.float32, copy=False)


def _lag1(arr: np.ndarray) -> np.ndarray:
    out = np.empty_like(arr, dtype=np.float32)
    if arr.size:
        out[0] = 0.0
        out[1:] = arr[:-1]
    return out


def _delta(arr: np.ndarray) -> np.ndarray:
    return _clip(arr - _lag1(arr), -5.0, 5.0)


def _cross_up(arr: np.ndarray) -> np.ndarray:
    prev = _lag1(arr)
    return ((arr > 0.0) & (prev <= 0.0)).astype(np.float32)


def _cross_down(arr: np.ndarray) -> np.ndarray:
    prev = _lag1(arr)
    return ((arr < 0.0) & (prev >= 0.0)).astype(np.float32)


def _add(arrays: list[np.ndarray], names: list[str], name: str, arr: np.ndarray, *, lo: float = -25.0, hi: float = 25.0) -> None:
    clean = _clip(np.asarray(arr, dtype=np.float32), lo, hi)
    if clean.ndim != 1:
        raise RuntimeError(f"chart geometry feature {name} is not 1D: {clean.shape}")
    if not np.isfinite(clean).all():
        raise RuntimeError(f"chart geometry feature {name} contains non-finite values")
    arrays.append(clean)
    names.append(f"{CHART_GEOMETRY_FEATURE_PREFIX}{name}")


def build_entry_chart_geometry_layer(
    x: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Build deterministic chart-geometry features from exact canonical sources."""
    x, idx = _require_source_matrix(x, feature_names)
    arrays: list[np.ndarray] = []
    names: list[str] = []

    def c(name: str) -> np.ndarray:
        return _col(x, idx, name)

    # Unit assumptions (owners), audited per producer after the upstream
    # USD->ATR-multiple conversion wave (GAP-6 comment repair 2026-08-11):
    # - snap._v1_ema_diff: emitted in ATR-multiples by basic_v1
    #   (unit-conversion owner); tanh scale=1.0 is dimensionally sane.
    # - snap.ema20_slope, snap.pos_vs_ema200: emitted in BPS OF PRICE by
    #   materialize_build_canonical_features_v1 (delta / price * 1e4), NOT
    #   ATR-multiples; tanh(1.0x) saturates on routine multi-bps moves.  A
    #   producer-side normalization is decided only by the pre-registered
    #   GAP-6 saturation measurement (design doc §6.4/§7.2) — the comment is
    #   repaired here, the scale is not moved without that measurement.
    # - ctx_cont._v1h*_ema_diff, ctx_cont.d1_ema_slope_20_canon_v2: emitted in
    #   ATR-multiples by htf_features (unit-conversion owner); scale=1.0 sane.
    # - ctx_cont._v1h*_slope5: 5th-order difference of the same H1/H4 ema_diff
    #   series (htf_features._model_native_htf_slope_v4); differencing
    #   preserves units, so ATR-multiples after the conversion; scale=1.0 sane.
    # - ctx_cont.m15_trend_sign_canon_v2: dimensionless sign in {-1,0,1}
    #   (htf_features); scale=1.0 sane.
    m5_ema = _tanh(c("snap._v1_ema_diff"))
    m5_slope = _tanh(c("snap.ema20_slope"))
    m5_pos_ema200 = _tanh(c("snap.pos_vs_ema200"))
    h1_ema = _tanh(c("ctx_cont._v1h1_ema_diff"))
    h1_slope = _tanh(c("ctx_cont._v1h1_slope5"))
    h4_ema = _tanh(c("ctx_cont._v1h4_ema_diff"))
    h4_slope = _tanh(c("ctx_cont._v1h4_slope5"))
    d1_slope = _tanh(c("ctx_cont.d1_ema_slope_20_canon_v2"))
    m15_trend = _tanh(c("ctx_cont.m15_trend_sign_canon_v2"))
    regime_stack = _tanh(c("ctx_cont.regime_stack_sum_v3"), scale=3.0)
    regime_agreement = _clip01(c("ctx_cont.regime_tf_agreement_v3"))
    regime_divergence = _clip01(c("ctx_cont.regime_divergence_flag_v3"))

    mtf_trend = _clip(
        0.16 * m5_ema
        + 0.10 * m5_slope
        + 0.10 * m5_pos_ema200
        + 0.18 * h1_ema
        + 0.08 * h1_slope
        + 0.18 * h4_ema
        + 0.08 * h4_slope
        + 0.08 * d1_slope
        + 0.04 * m15_trend
    )
    trend_up = _pos(mtf_trend)
    trend_down = _neg(mtf_trend)
    trend_delta = _delta(mtf_trend)
    sign_stack = np.vstack([m5_ema, h1_ema, h4_ema, d1_slope])
    sign_agree_up = (sign_stack > 0.0).mean(axis=0).astype(np.float32)
    sign_agree_down = (sign_stack < 0.0).mean(axis=0).astype(np.float32)
    agreement_pressure = _clip01(np.maximum(sign_agree_up, sign_agree_down) * (0.50 + regime_agreement))
    divergence_pressure = _clip01(regime_divergence + np.abs(m5_ema - h4_ema) * 0.25 + np.abs(h1_ema - d1_slope) * 0.20)
    ema_bull_cross_cluster = _clip01(
        0.40 * _cross_up(m5_ema)
        + 0.25 * _cross_up(h1_ema)
        + 0.20 * _cross_up(h4_ema)
        + 0.15 * _cross_up(d1_slope)
    )
    ema_bear_cross_cluster = _clip01(
        0.40 * _cross_down(m5_ema)
        + 0.25 * _cross_down(h1_ema)
        + 0.20 * _cross_down(h4_ema)
        + 0.15 * _cross_down(d1_slope)
    )
    ema_bull_follow = _clip01(
        0.20 * _pos(_delta(m5_ema))
        + 0.25 * _pos(_delta(h1_ema))
        + 0.25 * _pos(_delta(h4_ema))
        + 0.15 * _pos(_delta(d1_slope))
        + 0.15 * trend_up
    )
    ema_bear_follow = _clip01(
        0.20 * _neg(_delta(m5_ema))
        + 0.25 * _neg(_delta(h1_ema))
        + 0.25 * _neg(_delta(h4_ema))
        + 0.15 * _neg(_delta(d1_slope))
        + 0.15 * trend_down
    )
    ema_bull_confirmation = _clip01(
        (ema_bull_cross_cluster + ema_bull_follow) * (0.50 + agreement_pressure) * (0.75 + sign_agree_up)
    )
    ema_bear_confirmation = _clip01(
        (ema_bear_cross_cluster + ema_bear_follow) * (0.50 + agreement_pressure) * (0.75 + sign_agree_down)
    )
    ema_cross_up_pressure = (
        _cross_up(mtf_trend) * (1.0 + agreement_pressure)
        + _pos(trend_delta) * (0.65 + agreement_pressure)
        + ema_bull_confirmation
    )
    ema_cross_down_pressure = (
        _cross_down(mtf_trend) * (1.0 + agreement_pressure)
        + _neg(trend_delta) * (0.65 + agreement_pressure)
        + ema_bear_confirmation
    )
    # V30 package 7 (2026-08-13), operator-authorized removals:
    # - `h8_proxy_trend_score`: there is no H8 timeframe in
    #   MULTI_TF_RESAMPLE_RULES.  It was a re-weighted duplicate of
    #   mtf_trend_score's own H1/H4/D1 terms wearing a fabricated timeframe
    #   label.  Every input it consumed (`ctx_cont._v1h1_ema_diff`,
    #   `ctx_cont._v1h4_ema_diff`, `ctx_cont.d1_ema_slope_20_canon_v2`) is a
    #   ctx_cont contract field AND is still read by mtf_trend_score below.
    # - `ema_stack_bull/bear_pressure`: hand-written products of two fields
    #   this layer still emits (`mtf_trend_score` through _pos/_neg, and
    #   `mtf_trend_agreement_pressure`).  They had zero consumers anywhere in
    #   the tree; the trend specialist's own `trend.ema_stack_bull/bear_pressure`
    #   is a different owner and is untouched.
    _add(arrays, names, "mtf_trend_score", mtf_trend, lo=-2.0, hi=2.0)
    _add(arrays, names, "mtf_trend_agreement_pressure", agreement_pressure, lo=0.0, hi=1.0)
    _add(arrays, names, "mtf_trend_divergence_pressure", divergence_pressure, lo=0.0, hi=1.0)
    _add(
        arrays,
        names,
        "ema_cross_up_pressure",
        ema_cross_up_pressure,
        lo=0.0,
        hi=3.0,
    )
    _add(
        arrays,
        names,
        "ema_cross_down_pressure",
        ema_cross_down_pressure,
        lo=0.0,
        hi=3.0,
    )

    near_swing_high = _prox_abs(c("ctx_cont.dist_last_swing_high_atr"))
    near_swing_low = _prox_abs(c("ctx_cont.dist_last_swing_low_atr"))
    recent_swing_high = _recency(c("ctx_cont.bars_since_swing_high"))
    recent_swing_low = _recency(c("ctx_cont.bars_since_swing_low"))
    swing_high_line = _clip01(near_swing_high * (0.50 + recent_swing_high))
    swing_low_line = _clip01(near_swing_low * (0.50 + recent_swing_low))
    support_sources = np.vstack(
        [
            _clip01(c("ctx_cont.sr_support_proximity_exp")),
            # Pivot supports: producer emits (price-S)/ATR, so the support is
            # intact below price on dist>=0; dist<0 means price broke below it.
            _prox_valid_side(c("ctx_cont.dist_to_S1_atr"), positive_is_valid=True),
            _prox_valid_side(c("ctx_cont.dist_to_S2_atr"), positive_is_valid=True),
            # TF lows: producer emits (price-nearest_lo)/ATR where an unswept
            # low below price gives dist>=0; dist<0 means the low was swept.
            _prox_valid_side(c("ctx_cont.dist_to_h1_lo_atr"), positive_is_valid=True),
            _prox_valid_side(c("ctx_cont.dist_to_h4_lo_atr"), positive_is_valid=True),
            _prox_valid_side(c("ctx_cont.dist_to_d1_lo_atr"), positive_is_valid=True),
            swing_low_line,
        ]
    )
    resistance_sources = np.vstack(
        [
            _clip01(c("ctx_cont.sr_resistance_proximity_exp")),
            # Pivot resistances: producer emits (price-R)/ATR, so the
            # resistance is intact above price on dist<=0; dist>0 means price
            # broke above it.
            _prox_valid_side(c("ctx_cont.dist_to_R1_atr"), positive_is_valid=False),
            _prox_valid_side(c("ctx_cont.dist_to_R2_atr"), positive_is_valid=False),
            # TF highs: producer emits (nearest_hi-price)/ATR where an unswept
            # high above price gives dist>=0; dist<0 means the high was swept.
            _prox_valid_side(c("ctx_cont.dist_to_h1_hi_atr"), positive_is_valid=True),
            _prox_valid_side(c("ctx_cont.dist_to_h4_hi_atr"), positive_is_valid=True),
            _prox_valid_side(c("ctx_cont.dist_to_d1_hi_atr"), positive_is_valid=True),
            swing_high_line,
        ]
    )
    support_stack = support_sources.max(axis=0).astype(np.float32)
    resistance_stack = resistance_sources.max(axis=0).astype(np.float32)
    level_mean = ((support_sources.mean(axis=0) + resistance_sources.mean(axis=0)) * 0.5).astype(np.float32)
    level_max = np.maximum(support_stack, resistance_stack).astype(np.float32)
    # V30 package 7 (2026-08-13), operator-authorized removals — the five
    # exact-affine duplicates of the two emitted stacks
    # (docs/FEATURE_VALUE_REVIEW_20260813.md A.4).  `support_minus_resistance`,
    # `channel_position`, `channel_center`, `channel_edge` and `level_max` are
    # each a closed-form function of `support_stack`, `resistance_stack` and the
    # single ctx field `ctx_cont.sr_support_minus_resistance_prox`; all three of
    # those remain model inputs (the two stacks are still emitted below and stay
    # in CHART_GEOMETRY_MODEL_NATIVE_FEATURE_NAMES, and the ctx field is a
    # MODEL_NATIVE_CTX_CONT_FIELDS member), so the model can form any of them.
    # They survive here as LOCALS only, because `channel_edge` is a term of the
    # retained failed-breakout pair and removing that term would change a kept
    # field's semantics without any measurement to justify it.
    support_minus_resistance = _clip(c("ctx_cont.sr_support_minus_resistance_prox") + support_stack - resistance_stack, -2.0, 2.0)
    # Low-to-high channel position: support/lower rail -> 0, resistance/upper rail -> 1.
    # The stack balance is support-minus-resistance, so it must be inverted here.
    channel_position = _clip01(0.5 - 0.5 * support_minus_resistance)
    channel_center = _clip01(1.0 - 2.0 * np.abs(channel_position - 0.5))
    channel_edge = _clip01(level_max * (1.0 - channel_center))
    _add(arrays, names, "support_line_proximity_stack", support_stack, lo=0.0, hi=1.0)
    _add(arrays, names, "resistance_line_proximity_stack", resistance_stack, lo=0.0, hi=1.0)
    _add(arrays, names, "major_level_proximity_mean", level_mean, lo=0.0, hi=1.0)

    h1_ratio = c("ctx_cont.H1_range_compression_ratio")
    m15_ratio = c("ctx_cont.M15_range_compression_ratio")
    bb_relative_width = c("snap._v1_bb_squeeze_20_2")
    h1_compression = atr_ratio_compression_pressure(h1_ratio)
    m15_compression = atr_ratio_compression_pressure(m15_ratio)
    squeeze = bollinger_squeeze_pressure(bb_relative_width)
    compression = _clip01(0.40 * h1_compression + 0.35 * m15_compression + 0.25 * squeeze)
    vol_impulse = _clip01(
        0.34 * _pos(_tanh(c("snap.atr_z"), scale=2.0))
        + 0.33 * _pos(_tanh(c("snap.rvol_20"), scale=2.0))
        + 0.33 * _pos(_tanh(c("snap.vol_ratio_5_20"), scale=2.0))
    )
    expansion_impulse = _clip01(
        0.30 * atr_ratio_expansion_pressure(h1_ratio)
        + 0.25 * atr_ratio_expansion_pressure(m15_ratio)
        + 0.20 * bollinger_expansion_pressure(bb_relative_width)
        + 0.25 * vol_impulse
    )
    release = _clip01(_lag1(compression) * _pos(_delta(expansion_impulse)))
    trend_age = _clip01(
        0.5 * c("ctx_cont.h1_trend_age_bars_norm_v2") + 0.5 * c("ctx_cont.h4_trend_age_bars_norm_v2")
    )
    d1_atr_pct = _clip01(c("ctx_cont.D1_atr_percentile_252"))

    bos_up = _clip01(c("snap.smc_bos_up") + 0.5 * _pos(c("ctx_cont.smc_bos_pressure_last12")) + 0.25 * _pos(c("ctx_cont.smc_bos_pressure_last48")))
    bos_down = _clip01(c("snap.smc_bos_down") + 0.5 * _neg(c("ctx_cont.smc_bos_pressure_last12")) + 0.25 * _neg(c("ctx_cont.smc_bos_pressure_last48")))
    choch = _clip01(c("snap.smc_choch") + c("ctx_cont.smc_choch_recent_tau12") + 0.5 * c("ctx_cont.smc_choch_recent_tau24"))
    sweep_up = _clip01(c("snap.smc_sweep_up") + 0.5 * _neg(c("ctx_cont.smc_sweep_bull_pressure_last12")) + 0.25 * _neg(c("ctx_cont.smc_sweep_bull_pressure_last48")))
    sweep_down = _clip01(c("snap.smc_sweep_down") + 0.5 * _pos(c("ctx_cont.smc_sweep_bull_pressure_last12")) + 0.25 * _pos(c("ctx_cont.smc_sweep_bull_pressure_last48")))
    sweep_recent = _clip01(c("ctx_cont.smc_sweep_recency_tau24"))
    wick_level = _clip01(1.0 - c("snap.body_pct"))
    breakout_up_seed = _clip01(
        0.50 * bos_up + 0.20 * ema_bull_confirmation + 0.20 * release + 0.10 * _pos(regime_stack)
    )
    breakout_down_seed = _clip01(
        0.50 * bos_down + 0.20 * ema_bear_confirmation + 0.20 * release + 0.10 * _neg(regime_stack)
    )
    # V30 package 7 (2026-08-13), operator-authorized removals:
    # `support_bounce_long_pressure` / `resistance_reject_short_pressure` were
    # hand-written products of already-emitted evidence, and
    # `trendline_break_up/down_pressure` contain NO LINE — the "break" term is
    # `smc_bos_*`, a horizontal swing break, and the sided-proximity factor
    # returns 0 once price passes the level, so the field peaked BEFORE the
    # break and decayed AT it (audit §1a).  Every input all four consumed is
    # still a model input: `support_stack`/`resistance_stack` are emitted above,
    # `trend_up`/`trend_down` are _pos/_neg of the emitted `mtf_trend_score`,
    # `sweep_up/down` and `bos_up/down` come from the ctx/snap SMC fields that
    # remain declared sources of this layer and members of the ctx contract,
    # `wick_level` is 1 - `snap.body_pct`, `agreement_pressure` is emitted, and
    # `compression`/`release` still drive the retained compression-breakout
    # pair.  The real sloped-line evidence is the per-TF trendline registry.
    _add(
        arrays,
        names,
        "failed_breakout_high_reversal_pressure",
        sweep_up
        * sweep_recent
        * resistance_stack
        * wick_level
        * (1.0 + trend_down + choch + divergence_pressure + 0.5 * channel_edge),
        lo=0.0,
        hi=5.0,
    )
    _add(
        arrays,
        names,
        "failed_breakout_low_reversal_pressure",
        sweep_down
        * sweep_recent
        * support_stack
        * wick_level
        * (1.0 + trend_up + choch + divergence_pressure + 0.5 * channel_edge),
        lo=0.0,
        hi=5.0,
    )

    # V30 package 7 (2026-08-13), operator-authorized removal of the ENTIRE
    # Fibonacci block (15 columns) and the chart-pattern composites that
    # depended on it, per docs/INDICATOR_FIDELITY_AUDIT_20260813.md §1a:
    # - the three `fib_extension_*` fields were ALGEBRAICALLY IMPOSSIBLE:
    #   `fib_position` was `_clip01`-ed and an extension is by definition
    #   > 100%, so `_pos(fib_position - 0.786)` and `_pos(0.236 - fib_position)`
    #   could only fire inside the retracement band the name excludes;
    # - `fib_golden_zone_proximity` was mislabelled (0.500-0.618, not the
    #   classic 0.618-0.786);
    # - the five `fib_retracement_*_proximity` columns carried ONE degree of
    #   freedom (deterministic exp(-12*|p-k|) shifts of a single scalar);
    # - `fib_position_proxy` itself was a hand-weighted 0.55/0.30/0.15 blend
    #   (unsourced weights) of a REAL retracement with a premium proxy and a
    #   20-DAY range position - three quantities on different ranges and
    #   different clocks;
    # - `ascending/descending_triangle_pressure` were identical formulas up to
    #   the EMA sign, testing no flat resistance, no rising lows, no touch count
    #   and no convergence; `bull/bear_flag_pullback_pressure` and
    #   `flag_breakout_readiness_pressure` are the same construction over the
    #   mislabelled golden zone; `triangle_apex_compression_pressure` solves no
    #   apex.
    # RULE-4 CHECK: `ctx_cont.retracement_from_last_impulse` is the REAL
    # retracement and REMAINS A MODEL INPUT - it is a MODEL_NATIVE_CTX_CONT_FIELDS
    # member (so it reaches the model on every row regardless of TRAIN ranking)
    # and is still a declared, executed source of entry_trend_ema_v1 and
    # entry_foundation_structure_v1.  The other two blend inputs also remain
    # model inputs: `snap.smc_premium_discount` is a declared source of
    # entry_smc_liquidity_quality_v1 / entry_support_resistance_memory_v1, and
    # `ctx_cont.d1_close_pct_in_20day_range_canon_v2` is a ctx_cont contract
    # field explicitly routed to chart_geometry_encoder.  `compression`,
    # `support_stack`, `resistance_stack`, `trend_up/down` and the EMA
    # follow-through terms all remain emitted or directly derived from emitted
    # fields.  The genuine converging-line/apex evidence is the per-TF
    # trendline registry (`geomline_*`), which is already mandatory.

    _add(
        arrays,
        names,
        "compression_breakout_up_pressure",
        release * trend_up * (0.5 + breakout_up_seed + resistance_stack + ema_bull_confirmation),
        lo=0.0,
        hi=5.0,
    )
    _add(
        arrays,
        names,
        "compression_breakout_down_pressure",
        release * trend_down * (0.5 + breakout_down_seed + support_stack + ema_bear_confirmation),
        lo=0.0,
        hi=5.0,
    )
    _add(arrays, names, "late_trend_reversal_risk", trend_age * choch * (0.5 + divergence_pressure + d1_atr_pct), lo=0.0, hi=5.0)
    _add(arrays, names, "ema_cross_mtf_bull_confirmation", ema_bull_confirmation, lo=0.0, hi=1.0)
    _add(arrays, names, "ema_cross_mtf_bear_confirmation", ema_bear_confirmation, lo=0.0, hi=1.0)
    # V30 package 7 (2026-08-13), operator-authorized removal of the remaining
    # NAME-ONLY block (audit §1a).  All of these names promised a mechanism this
    # layer cannot compute, because its input tuple is pre-reduced scalars - one
    # distance per side, no price, no pivot coordinate, no bar index, no slope,
    # no anchor, and two points are required to define a line:
    # - the four `*_rail_*` fields contained no rail and no slope; "rising" was
    #   the sign of a 9-term EMA blend, and the "short trap" / "long trap"
    #   variants contained no failure and no breakdown - each was a strictly
    #   more directional rescaling of its own twin;
    # - `mtf_channel_retest_*_quality` performed no retest: the "window" was
    #   `_lag1` (one bar) and nothing tested that it was the SAME level that
    #   broke; `mtf_channel_breakout_*_quality` and
    #   `trendline_channel_confluence_pressure` / `channel_edge_rejection_pressure`
    #   are functions of the same two proximity stacks and the breakout seed;
    # - `fib_extension_exhaustion_risk` inherited the algebraically impossible
    #   extension terms; `line_pattern_tail_risk` multiplied a wick/sweep sum by
    #   the removed channel-edge duplicate.
    # RULE-4 CHECK: every input these composites consumed is still a model
    # input - `support_line_proximity_stack`, `resistance_line_proximity_stack`,
    # `major_level_proximity_mean`, `mtf_trend_score`,
    # `mtf_trend_agreement_pressure`, `mtf_trend_divergence_pressure`,
    # `ema_cross_up/down_pressure` and `ema_cross_mtf_bull/bear_confirmation`
    # are all still emitted by this layer; the SMC sweep/BOS/CHoCH and the
    # compression/vol inputs remain declared sources here and members of the ctx
    # contract.  The genuine line/channel/retest evidence is the per-TF
    # `geomline_*` trendline registry and the `level_*` level registry, both
    # mandatory on all five timeframes since 2026-08-11.

    out = np.column_stack(arrays).astype(np.float32, copy=False) if arrays else np.empty((x.shape[0], 0), dtype=np.float32)
    if not np.isfinite(out).all():
        raise RuntimeError("chart geometry layer contains non-finite values")
    if len(set(names)) != len(names):
        dupes = sorted({name for name in names if names.count(name) > 1})
        raise RuntimeError(f"chart geometry layer has duplicate names: {dupes[:10]}")
    return out, names


_CHART_GEOMETRY_NAME_PROBE = np.zeros(
    (1, len(CHART_GEOMETRY_SOURCE_FIELDS)), dtype=np.float32
)
for _ratio_field in (
    "ctx_cont.H1_range_compression_ratio",
    "ctx_cont.M15_range_compression_ratio",
):
    _CHART_GEOMETRY_NAME_PROBE[
        0, CHART_GEOMETRY_SOURCE_FIELDS.index(_ratio_field)
    ] = 1.0
CHART_GEOMETRY_FEATURE_NAMES = tuple(
    build_entry_chart_geometry_layer(
        _CHART_GEOMETRY_NAME_PROBE,
        list(CHART_GEOMETRY_SOURCE_FIELDS),
    )[1]
)
del _CHART_GEOMETRY_NAME_PROBE, _ratio_field

# The model-native geometry surface retains the exact current-bar inputs used
# by structural auxiliary supervision and the pretrain polarity proof.  A TRAIN
# ranking may never remove either dependency.
#
# V30 package 7 (2026-08-13): 16 of the previous 18 pins were the NAME-ONLY
# rail / channel / retest / Fibonacci composites removed above, so the pinned
# set is now exactly the two sided proximity stacks that the aux-label and
# polarity contracts still bind.  The layer's remaining 13 fields stay in the
# TRAIN-ranked candidate pool, the same status the other 40 non-pinned
# chart-geometry fields already had.  The `CHART_GEOMETRY_SMART2_*` marker/suffix
# pair is gone with them: its anchor field
# (`chart.geometry_trendline_channel_confluence_pressure`) was removed, and the
# suffix had no consumer outside its own test (rule 10).
CHART_GEOMETRY_MODEL_NATIVE_FEATURE_NAMES = (
    f"{CHART_GEOMETRY_FEATURE_PREFIX}support_line_proximity_stack",
    f"{CHART_GEOMETRY_FEATURE_PREFIX}resistance_line_proximity_stack",
)
_missing_model_native = tuple(
    name
    for name in CHART_GEOMETRY_MODEL_NATIVE_FEATURE_NAMES
    if name not in CHART_GEOMETRY_FEATURE_NAMES
)
if _missing_model_native:
    raise RuntimeError(
        f"CHART_GEOMETRY_MODEL_NATIVE_FEATURE_NAMES_NOT_EMITTED: {_missing_model_native}"
    )
del _missing_model_native
