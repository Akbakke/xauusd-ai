"""Strict closed-bar SMC/liquidity quality features for model-native Entry."""
from __future__ import annotations

from typing import Iterable

import numpy as np


SMC_LIQUIDITY_QUALITY_FEATURE_VERSION = (
    "entry_smc_liquidity_quality_v1_20260722_distinct_liquidity_pool_stack_fail_closed"
)
SMC_LIQUIDITY_QUALITY_FEATURE_PREFIX = "chart.smc_liquidity_"

SMC_LIQUIDITY_QUALITY_FEATURE_SUFFIXES = (
    "sweep_low_quality_long",
    "sweep_high_quality_short",
    "reclaim_confirmation_long",
    "reclaim_confirmation_short",
    "false_break_reversal_pressure_long",
    "false_break_reversal_pressure_short",
    "sweep_reclaim_strength_long",
    "sweep_reclaim_strength_short",
    "false_breakout_quality_long",
    "false_breakout_quality_short",
    "inducement_trap_risk_long",
    "inducement_trap_risk_short",
    "liquidity_pool_proximity_low",
    "liquidity_pool_proximity_high",
    "liquidity_pool_proximity_balance_low_minus_high",
    "two_sided_liquidity_pool_pressure",
    "wick_rejection_strength_long",
    "wick_rejection_strength_short",
    "premium_discount_trend_alignment_long",
    "premium_discount_trend_alignment_short",
    "premium_discount_reclaim_confluence_long",
    "premium_discount_reclaim_confluence_short",
    "continuation_pressure_long",
    "continuation_pressure_short",
)
SMC_LIQUIDITY_QUALITY_FEATURE_NAMES = tuple(
    f"{SMC_LIQUIDITY_QUALITY_FEATURE_PREFIX}{name}" for name in SMC_LIQUIDITY_QUALITY_FEATURE_SUFFIXES
)

SMC_LIQUIDITY_QUALITY_SOURCE_FIELDS = (
    "snap.smc_sweep_up",
    "snap.smc_sweep_down",
    "snap.smc_sweep_size_atr",
    "snap.smc_bars_since_sweep",
    "snap.smc_premium_discount",
    "snap.smc_choch",
    "snap.body_pct",
    "snap.wick_asym",
    "snap._v1_body_share_1",
    "snap._v1_clv",
    "ctx_cont.smc_sweep_bull_pressure_last12",
    "ctx_cont.smc_sweep_bull_pressure_last48",
    "ctx_cont.smc_sweep_size_recent_tau12",
    "ctx_cont.smc_sweep_recency_tau24",
    "ctx_cont.smc_choch_recent_tau12",
    "ctx_cont.smc_choch_recent_tau24",
    "ctx_cont.sr_nearest_pivot_abs_atr",
    "ctx_cont.sr_support_proximity_exp",
    "ctx_cont.sr_resistance_proximity_exp",
    "ctx_cont.liquidity_hi_nearest_abs_atr",
    "ctx_cont.liquidity_lo_nearest_abs_atr",
    "ctx_cont.dist_to_R1_atr",
    "ctx_cont.dist_to_R2_atr",
    "ctx_cont.dist_to_S1_atr",
    "ctx_cont.dist_to_S2_atr",
    "ctx_cont.dist_to_m5_hi_atr",
    "ctx_cont.dist_to_m5_lo_atr",
    "ctx_cont.dist_to_m15_hi_atr",
    "ctx_cont.dist_to_m15_lo_atr",
    "ctx_cont.dist_to_h1_hi_atr",
    "ctx_cont.dist_to_h1_lo_atr",
    "ctx_cont.dist_to_h4_hi_atr",
    "ctx_cont.dist_to_h4_lo_atr",
    "ctx_cont.dist_to_d1_hi_atr",
    "ctx_cont.dist_to_d1_lo_atr",
    "ctx_cont.dist_last_swing_high_atr",
    "ctx_cont.dist_last_swing_low_atr",
    "ctx_cont.bars_since_swing_high",
    "ctx_cont.bars_since_swing_low",
    "ctx_cont.wick_ratio",
    "ctx_cont._v1h1_ema_diff",
    "ctx_cont._v1h4_ema_diff",
    "ctx_cont.d1_ema_slope_20_canon_v2",
    "ctx_cont.m15_trend_sign_canon_v2",
    "ctx_cont.regime_stack_sum_v3",
    "chart.foundation_sweep_low_reclaim_up_proxy",
    "chart.foundation_sweep_high_reclaim_down_proxy",
    "chart.foundation_false_breakout_high_followthrough_down_proxy",
    "chart.foundation_false_breakout_low_followthrough_up_proxy",
    "chart.geometry_support_line_proximity_stack",
    "chart.geometry_resistance_line_proximity_stack",
    "chart.geometry_failed_breakout_high_reversal_pressure",
    "chart.geometry_failed_breakout_low_reversal_pressure",
    "candle.pattern_upper_wick_share",
    "candle.pattern_lower_wick_share",
    "candle.pattern_close_location",
    "candle.pattern_body_direction",
    "candle.pattern_body_share",
)


def _name_index(names: Iterable[str]) -> dict[str, int]:
    return {name: i for i, name in enumerate(names)}


def missing_smc_liquidity_quality_source_fields(feature_names: Iterable[str]) -> list[str]:
    available = set(feature_names)
    return [name for name in SMC_LIQUIDITY_QUALITY_SOURCE_FIELDS if name not in available]


def _col(x: np.ndarray, index: dict[str, int], name: str) -> np.ndarray:
    if name not in index:
        raise RuntimeError(f"SMC_LIQUIDITY_QUALITY_SOURCE_FIELD_MISSING: {name}")
    arr = np.asarray(x[:, index[name]], dtype=np.float32)
    if not np.isfinite(arr).all():
        raise RuntimeError(f"SMC_LIQUIDITY_QUALITY_SOURCE_NONFINITE: {name}")
    return arr


def _clip(arr: np.ndarray, lo: float = -25.0, hi: float = 25.0) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float32)
    if not np.isfinite(values).all():
        raise RuntimeError("SMC_LIQUIDITY_QUALITY_DERIVATION_NONFINITE")
    return np.clip(values, lo, hi).astype(np.float32, copy=False)


def _clip01(arr: np.ndarray) -> np.ndarray:
    return _clip(arr, 0.0, 1.0)


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


def _count_proxy(sources: np.ndarray, threshold: float = 0.55) -> np.ndarray:
    return (sources > float(threshold)).mean(axis=0).astype(np.float32, copy=False)


def _add(arrays: list[np.ndarray], names: list[str], name: str, arr: np.ndarray, *, lo: float = -25.0, hi: float = 25.0) -> None:
    clean = _clip(np.asarray(arr, dtype=np.float32), lo, hi)
    if clean.ndim != 1:
        raise RuntimeError(f"SMC/liquidity quality feature {name} is not 1D: {clean.shape}")
    if not np.isfinite(clean).all():
        raise RuntimeError(f"SMC/liquidity quality feature {name} contains non-finite values")
    arrays.append(clean)
    names.append(f"{SMC_LIQUIDITY_QUALITY_FEATURE_PREFIX}{name}")


def build_entry_smc_liquidity_quality_layer(
    x: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Build deterministic SMC/liquidity quality candidates from closed-bar inputs."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise RuntimeError(f"SMC_LIQUIDITY_QUALITY_INPUT_NOT_2D: shape={x.shape}")
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise RuntimeError(f"SMC_LIQUIDITY_QUALITY_INPUT_EMPTY: shape={x.shape}")
    if x.shape[1] != len(feature_names):
        raise RuntimeError(
            "SMC_LIQUIDITY_QUALITY_FEATURE_NAME_DIM_MISMATCH: "
            f"cols={x.shape[1]} names={len(feature_names)}"
        )
    if any(not isinstance(name, str) or not name for name in feature_names):
        raise RuntimeError("SMC_LIQUIDITY_QUALITY_FEATURE_NAME_INVALID")
    if len(feature_names) != len(set(feature_names)):
        seen: set[str] = set()
        duplicates = sorted(
            {name for name in feature_names if name in seen or seen.add(name)}
        )
        raise RuntimeError(
            f"SMC_LIQUIDITY_QUALITY_DUPLICATE_FEATURE_NAMES: {duplicates[:20]}"
        )
    idx = _name_index(feature_names)
    missing = missing_smc_liquidity_quality_source_fields(feature_names)
    if missing:
        raise RuntimeError(f"SMC_LIQUIDITY_QUALITY_SOURCE_FIELDS_MISSING: {missing[:20]} total={len(missing)}")
    if not np.isfinite(x).all():
        bad_rows, bad_cols = np.where(~np.isfinite(x))
        examples = [
            {"row": int(row), "feature": feature_names[int(col)]}
            for row, col in zip(bad_rows[:10], bad_cols[:10])
        ]
        raise RuntimeError(f"SMC_LIQUIDITY_QUALITY_SOURCE_NONFINITE: {examples}")
    arrays: list[np.ndarray] = []
    names: list[str] = []

    def c(name: str) -> np.ndarray:
        return _col(x, idx, name)

    h1_trend = _tanh(c("ctx_cont._v1h1_ema_diff"))
    h4_trend = _tanh(c("ctx_cont._v1h4_ema_diff"))
    d1_slope = _tanh(c("ctx_cont.d1_ema_slope_20_canon_v2"))
    m15_trend = _tanh(c("ctx_cont.m15_trend_sign_canon_v2"))
    regime_stack = _tanh(c("ctx_cont.regime_stack_sum_v3"), scale=3.0)
    trend = _clip(0.32 * h1_trend + 0.30 * h4_trend + 0.20 * d1_slope + 0.12 * m15_trend + 0.06 * regime_stack)
    trend_up = _pos(trend)
    trend_down = _neg(trend)

    sweep_up = _clip01(
        c("snap.smc_sweep_up")
        + 0.50 * _neg(c("ctx_cont.smc_sweep_bull_pressure_last12"))
        + 0.25 * _neg(c("ctx_cont.smc_sweep_bull_pressure_last48"))
    )
    sweep_down = _clip01(
        c("snap.smc_sweep_down")
        + 0.50 * _pos(c("ctx_cont.smc_sweep_bull_pressure_last12"))
        + 0.25 * _pos(c("ctx_cont.smc_sweep_bull_pressure_last48"))
    )
    sweep_recent = _clip01(_recency(c("snap.smc_bars_since_sweep")) + c("ctx_cont.smc_sweep_recency_tau24"))
    sweep_size = _clip01(c("snap.smc_sweep_size_atr") + c("ctx_cont.smc_sweep_size_recent_tau12"))
    choch_recent = _clip01(c("snap.smc_choch") + c("ctx_cont.smc_choch_recent_tau12") + 0.5 * c("ctx_cont.smc_choch_recent_tau24"))

    premium = _clip01(c("snap.smc_premium_discount"))
    discount = _clip01(1.0 - premium)
    clv_unit = _clip01(0.5 + 0.5 * _clip(c("snap._v1_clv"), -1.0, 1.0))
    body_direction = _clip(c("snap.body_pct") + c("candle.pattern_body_direction"), -1.0, 1.0)
    body_share = _clip01(np.abs(c("snap.body_pct")) + c("snap._v1_body_share_1") + c("candle.pattern_body_share"))
    body_bull = _clip01(_pos(body_direction) + _pos(c("candle.pattern_body_direction")))
    body_bear = _clip01(_neg(body_direction) + _neg(c("candle.pattern_body_direction")))

    wick_asym = _clip(c("snap.wick_asym"), -1.0, 1.0)
    wick_ratio = _clip01(c("ctx_cont.wick_ratio"))
    upper_wick = _clip01(_pos(wick_asym) + 0.50 * wick_ratio + c("candle.pattern_upper_wick_share"))
    lower_wick = _clip01(_neg(wick_asym) + 0.50 * (1.0 - wick_ratio) + c("candle.pattern_lower_wick_share"))
    close_near_high = _clip01(0.55 * (1.0 - wick_ratio) + 0.30 * clv_unit + 0.15 * c("candle.pattern_close_location"))
    close_near_low = _clip01(0.55 * wick_ratio + 0.30 * (1.0 - clv_unit) + 0.15 * (1.0 - c("candle.pattern_close_location")))
    wick_reject_long = _clip01(lower_wick * (0.50 + close_near_high + 0.25 * body_bull) * (0.75 + 0.25 * body_share))
    wick_reject_short = _clip01(upper_wick * (0.50 + close_near_low + 0.25 * body_bear) * (0.75 + 0.25 * body_share))

    recent_swing_low = _recency(c("ctx_cont.bars_since_swing_low"))
    recent_swing_high = _recency(c("ctx_cont.bars_since_swing_high"))
    near_swing_low = _prox_abs(c("ctx_cont.dist_last_swing_low_atr"))
    near_swing_high = _prox_abs(c("ctx_cont.dist_last_swing_high_atr"))
    support_sources = np.vstack(
        [
            _clip01(c("ctx_cont.sr_support_proximity_exp")),
            _prox_abs(c("ctx_cont.sr_nearest_pivot_abs_atr")),
            _prox_abs(c("ctx_cont.dist_to_S1_atr")),
            _prox_abs(c("ctx_cont.dist_to_S2_atr")),
            _prox_abs(c("ctx_cont.dist_to_m5_lo_atr")),
            _prox_abs(c("ctx_cont.dist_to_m15_lo_atr")),
            _prox_abs(c("ctx_cont.dist_to_h1_lo_atr")),
            _prox_abs(c("ctx_cont.dist_to_h4_lo_atr")),
            _prox_abs(c("ctx_cont.dist_to_d1_lo_atr")),
            _prox_abs(c("ctx_cont.liquidity_lo_nearest_abs_atr")),
            near_swing_low * (0.50 + recent_swing_low),
            _clip01(c("chart.geometry_support_line_proximity_stack")),
        ]
    )
    resistance_sources = np.vstack(
        [
            _clip01(c("ctx_cont.sr_resistance_proximity_exp")),
            _prox_abs(c("ctx_cont.sr_nearest_pivot_abs_atr")),
            _prox_abs(c("ctx_cont.dist_to_R1_atr")),
            _prox_abs(c("ctx_cont.dist_to_R2_atr")),
            _prox_abs(c("ctx_cont.dist_to_m5_hi_atr")),
            _prox_abs(c("ctx_cont.dist_to_m15_hi_atr")),
            _prox_abs(c("ctx_cont.dist_to_h1_hi_atr")),
            _prox_abs(c("ctx_cont.dist_to_h4_hi_atr")),
            _prox_abs(c("ctx_cont.dist_to_d1_hi_atr")),
            _prox_abs(c("ctx_cont.liquidity_hi_nearest_abs_atr")),
            near_swing_high * (0.50 + recent_swing_high),
            _clip01(c("chart.geometry_resistance_line_proximity_stack")),
        ]
    )
    support_stack = support_sources.max(axis=0).astype(np.float32)
    resistance_stack = resistance_sources.max(axis=0).astype(np.float32)
    support_touch_count = _count_proxy(support_sources)
    resistance_touch_count = _count_proxy(resistance_sources)

    # A liquidity pool is not the same object as generic support/resistance.
    # The previous max(support_stack, liquidity_proximity) expression was
    # algebraically identical to support_stack because liquidity_proximity was
    # already one of the max inputs above.  That silently spent two model
    # slots on exact S/R duplicates.  Keep the S/R stack for confluence, while
    # the pool surface now measures a causal blend of the dedicated liquidity
    # locator, the most recent swing, and clustered lower/higher TF extremes.
    # This preserves the intended cross-family cooperation without allowing
    # either family to masquerade as the other.
    low_liquidity_proximity = _prox_abs(c("ctx_cont.liquidity_lo_nearest_abs_atr"))
    high_liquidity_proximity = _prox_abs(c("ctx_cont.liquidity_hi_nearest_abs_atr"))
    low_swing_proximity = _clip01(near_swing_low * (0.50 + recent_swing_low))
    high_swing_proximity = _clip01(near_swing_high * (0.50 + recent_swing_high))
    low_tf_cluster = _clip01(
        0.30 * _prox_abs(c("ctx_cont.dist_to_m5_lo_atr"))
        + 0.25 * _prox_abs(c("ctx_cont.dist_to_m15_lo_atr"))
        + 0.20 * _prox_abs(c("ctx_cont.dist_to_h1_lo_atr"))
        + 0.15 * _prox_abs(c("ctx_cont.dist_to_h4_lo_atr"))
        + 0.10 * _prox_abs(c("ctx_cont.dist_to_d1_lo_atr"))
    )
    high_tf_cluster = _clip01(
        0.30 * _prox_abs(c("ctx_cont.dist_to_m5_hi_atr"))
        + 0.25 * _prox_abs(c("ctx_cont.dist_to_m15_hi_atr"))
        + 0.20 * _prox_abs(c("ctx_cont.dist_to_h1_hi_atr"))
        + 0.15 * _prox_abs(c("ctx_cont.dist_to_h4_hi_atr"))
        + 0.10 * _prox_abs(c("ctx_cont.dist_to_d1_hi_atr"))
    )
    low_pool = _clip01(
        0.55 * low_liquidity_proximity
        + 0.25 * low_swing_proximity
        + 0.20 * low_tf_cluster
    )
    high_pool = _clip01(
        0.55 * high_liquidity_proximity
        + 0.25 * high_swing_proximity
        + 0.20 * high_tf_cluster
    )

    foundation_reclaim_long = _clip01(c("chart.foundation_sweep_low_reclaim_up_proxy") / 5.0)
    foundation_reclaim_short = _clip01(c("chart.foundation_sweep_high_reclaim_down_proxy") / 5.0)
    foundation_false_low = _clip01(c("chart.foundation_false_breakout_low_followthrough_up_proxy") / 5.0)
    foundation_false_high = _clip01(c("chart.foundation_false_breakout_high_followthrough_down_proxy") / 5.0)
    geometry_false_low = _clip01(c("chart.geometry_failed_breakout_low_reversal_pressure") / 5.0)
    geometry_false_high = _clip01(c("chart.geometry_failed_breakout_high_reversal_pressure") / 5.0)

    sweep_low_quality = _clip(
        sweep_down
        * sweep_recent
        * (0.50 + sweep_size)
        * (0.50 + low_pool + support_stack)
        * (0.50 + wick_reject_long)
        * (0.50 + discount + trend_up),
        0.0,
        5.0,
    )
    sweep_high_quality = _clip(
        sweep_up
        * sweep_recent
        * (0.50 + sweep_size)
        * (0.50 + high_pool + resistance_stack)
        * (0.50 + wick_reject_short)
        * (0.50 + premium + trend_down),
        0.0,
        5.0,
    )
    reclaim_long = _clip01(0.45 * foundation_reclaim_long + 0.35 * (sweep_low_quality / 5.0) + 0.20 * close_near_high)
    reclaim_short = _clip01(0.45 * foundation_reclaim_short + 0.35 * (sweep_high_quality / 5.0) + 0.20 * close_near_low)
    false_low_context = _clip01(0.40 * foundation_false_low + 0.20 * geometry_false_low + 0.40 * (sweep_low_quality / 5.0 + choch_recent * trend_up))
    false_high_context = _clip01(0.40 * foundation_false_high + 0.20 * geometry_false_high + 0.40 * (sweep_high_quality / 5.0 + choch_recent * trend_down))
    long_pd_alignment = _clip01(discount * (0.50 + trend_up) * (0.50 + support_stack + low_pool))
    short_pd_alignment = _clip01(premium * (0.50 + trend_down) * (0.50 + resistance_stack + high_pool))
    inducement_risk_long = _clip(
        high_pool
        * resistance_stack
        * premium
        * (0.50 + sweep_up + wick_reject_short + false_high_context)
        * (1.0 - reclaim_long),
        0.0,
        5.0,
    )
    inducement_risk_short = _clip(
        low_pool
        * support_stack
        * discount
        * (0.50 + sweep_down + wick_reject_long + false_low_context)
        * (1.0 - reclaim_short),
        0.0,
        5.0,
    )
    sweep_reclaim_strength_long = _clip01(
        0.30 * (sweep_low_quality / 5.0)
        + 0.25 * reclaim_long
        + 0.20 * wick_reject_long
        + 0.15 * close_near_high
        + 0.10 * foundation_reclaim_long
    )
    sweep_reclaim_strength_short = _clip01(
        0.30 * (sweep_high_quality / 5.0)
        + 0.25 * reclaim_short
        + 0.20 * wick_reject_short
        + 0.15 * close_near_low
        + 0.10 * foundation_reclaim_short
    )
    false_break_reversal_pressure_long = _clip01(
        (0.30 * false_low_context + 0.25 * wick_reject_long + 0.20 * reclaim_long + 0.15 * discount + 0.10 * low_pool)
        * (0.75 + 0.25 * choch_recent)
    )
    false_break_reversal_pressure_short = _clip01(
        (0.30 * false_high_context + 0.25 * wick_reject_short + 0.20 * reclaim_short + 0.15 * premium + 0.10 * high_pool)
        * (0.75 + 0.25 * choch_recent)
    )
    false_breakout_quality_long = _clip01(
        false_low_context
        * (0.45 + 0.35 * reclaim_long + 0.20 * wick_reject_long)
        * (0.70 + 0.30 * low_pool)
    )
    false_breakout_quality_short = _clip01(
        false_high_context
        * (0.45 + 0.35 * reclaim_short + 0.20 * wick_reject_short)
        * (0.70 + 0.30 * high_pool)
    )
    liquidity_pool_balance = _clip(low_pool - high_pool, -1.0, 1.0)
    two_sided_liquidity_pressure = _clip01(
        np.minimum(low_pool, high_pool)
        * (0.50 + 0.25 * support_touch_count + 0.25 * resistance_touch_count)
        * (0.75 + 0.25 * sweep_recent)
    )
    premium_discount_reclaim_confluence_long = _clip01(
        discount
        * reclaim_long
        * (0.45 + 0.25 * support_stack + 0.20 * low_pool + 0.10 * wick_reject_long)
    )
    premium_discount_reclaim_confluence_short = _clip01(
        premium
        * reclaim_short
        * (0.45 + 0.25 * resistance_stack + 0.20 * high_pool + 0.10 * wick_reject_short)
    )
    continuation_pressure_long = _clip01(
        (
            0.35 * trend_up
            + 0.25 * long_pd_alignment
            + 0.20 * sweep_reclaim_strength_long
            + 0.20 * reclaim_long
        )
        * (1.0 - 0.35 * (inducement_risk_long / 5.0))
    )
    continuation_pressure_short = _clip01(
        (
            0.35 * trend_down
            + 0.25 * short_pd_alignment
            + 0.20 * sweep_reclaim_strength_short
            + 0.20 * reclaim_short
        )
        * (1.0 - 0.35 * (inducement_risk_short / 5.0))
    )

    _add(arrays, names, "sweep_low_quality_long", sweep_low_quality, lo=0.0, hi=5.0)
    _add(arrays, names, "sweep_high_quality_short", sweep_high_quality, lo=0.0, hi=5.0)
    _add(arrays, names, "reclaim_confirmation_long", reclaim_long, lo=0.0, hi=1.0)
    _add(arrays, names, "reclaim_confirmation_short", reclaim_short, lo=0.0, hi=1.0)
    _add(arrays, names, "false_break_reversal_pressure_long", false_break_reversal_pressure_long, lo=0.0, hi=1.0)
    _add(arrays, names, "false_break_reversal_pressure_short", false_break_reversal_pressure_short, lo=0.0, hi=1.0)
    _add(arrays, names, "sweep_reclaim_strength_long", sweep_reclaim_strength_long, lo=0.0, hi=1.0)
    _add(arrays, names, "sweep_reclaim_strength_short", sweep_reclaim_strength_short, lo=0.0, hi=1.0)
    _add(arrays, names, "false_breakout_quality_long", false_breakout_quality_long, lo=0.0, hi=1.0)
    _add(arrays, names, "false_breakout_quality_short", false_breakout_quality_short, lo=0.0, hi=1.0)
    _add(arrays, names, "inducement_trap_risk_long", inducement_risk_long, lo=0.0, hi=5.0)
    _add(arrays, names, "inducement_trap_risk_short", inducement_risk_short, lo=0.0, hi=5.0)
    _add(arrays, names, "liquidity_pool_proximity_low", low_pool, lo=0.0, hi=1.0)
    _add(arrays, names, "liquidity_pool_proximity_high", high_pool, lo=0.0, hi=1.0)
    _add(arrays, names, "liquidity_pool_proximity_balance_low_minus_high", liquidity_pool_balance, lo=-1.0, hi=1.0)
    _add(arrays, names, "two_sided_liquidity_pool_pressure", two_sided_liquidity_pressure, lo=0.0, hi=1.0)
    _add(arrays, names, "wick_rejection_strength_long", wick_reject_long, lo=0.0, hi=1.0)
    _add(arrays, names, "wick_rejection_strength_short", wick_reject_short, lo=0.0, hi=1.0)
    _add(arrays, names, "premium_discount_trend_alignment_long", long_pd_alignment, lo=0.0, hi=1.0)
    _add(arrays, names, "premium_discount_trend_alignment_short", short_pd_alignment, lo=0.0, hi=1.0)
    _add(
        arrays,
        names,
        "premium_discount_reclaim_confluence_long",
        premium_discount_reclaim_confluence_long,
        lo=0.0,
        hi=1.0,
    )
    _add(
        arrays,
        names,
        "premium_discount_reclaim_confluence_short",
        premium_discount_reclaim_confluence_short,
        lo=0.0,
        hi=1.0,
    )
    _add(arrays, names, "continuation_pressure_long", continuation_pressure_long, lo=0.0, hi=1.0)
    _add(arrays, names, "continuation_pressure_short", continuation_pressure_short, lo=0.0, hi=1.0)

    out = np.column_stack(arrays).astype(np.float32, copy=False) if arrays else np.empty((x.shape[0], 0), dtype=np.float32)
    if not np.isfinite(out).all():
        raise RuntimeError("SMC/liquidity quality layer contains non-finite values")
    if len(set(names)) != len(names):
        dupes = sorted({name for name in names if names.count(name) > 1})
        raise RuntimeError(f"SMC/liquidity quality layer has duplicate names: {dupes[:10]}")
    if tuple(names) != SMC_LIQUIDITY_QUALITY_FEATURE_NAMES:
        raise RuntimeError("SMC_LIQUIDITY_QUALITY_FEATURE_NAME_DRIFT")
    return out, names
