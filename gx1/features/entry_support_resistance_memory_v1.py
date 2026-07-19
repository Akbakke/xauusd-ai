"""Canonical support/resistance level-memory features for Entry specialists.

This smart layer derives causal numeric memory around already-materialized
support/resistance, pivot, liquidity and SMC fields for the model-native
train/serve feature path.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


SUPPORT_RESISTANCE_MEMORY_FEATURE_VERSION = (
    "entry_support_resistance_memory_v1_20260717_closed_bar_level_memory_failclosed"
)
SUPPORT_RESISTANCE_MEMORY_FEATURE_PREFIX = "chart.sr_memory_"

SUPPORT_RESISTANCE_MEMORY_SOURCE_FIELDS = (
    "ctx_cont.sr_nearest_pivot_abs_atr",
    "ctx_cont.sr_support_proximity_exp",
    "ctx_cont.sr_resistance_proximity_exp",
    "ctx_cont.sr_support_minus_resistance_prox",
    "ctx_cont.liquidity_hi_nearest_abs_atr",
    "ctx_cont.liquidity_lo_nearest_abs_atr",
    "ctx_cont.liquidity_lo_minus_hi_prox",
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
    "snap.smc_sweep_up",
    "snap.smc_sweep_down",
    "snap.smc_sweep_size_atr",
    "snap.smc_bars_since_sweep",
    "ctx_cont.smc_sweep_bull_pressure_last12",
    "ctx_cont.smc_sweep_bull_pressure_last48",
    "ctx_cont.smc_sweep_size_recent_tau12",
    "ctx_cont.smc_sweep_recency_tau24",
    "snap.smc_bos_up",
    "snap.smc_bos_down",
    "ctx_cont.smc_bos_pressure_last12",
    "ctx_cont.smc_bos_pressure_last48",
    "snap.smc_choch",
    "ctx_cont.smc_choch_recent_tau12",
    "ctx_cont.smc_choch_recent_tau24",
    "snap.smc_premium_discount",
    "snap.body_pct",
    "snap.wick_asym",
    "snap._v1_body_share_1",
    "snap._v1_clv",
    "ctx_cont.wick_ratio",
    "ctx_cont._v1h1_ema_diff",
    "ctx_cont._v1h4_ema_diff",
    "ctx_cont.d1_ema_slope_20_canon_v2",
    "ctx_cont.m15_trend_sign_canon_v2",
    "ctx_cont.regime_stack_sum_v3",
    "chart.geometry_support_line_proximity_stack",
    "chart.geometry_resistance_line_proximity_stack",
    "chart.geometry_support_minus_resistance_stack",
    "chart.geometry_major_level_proximity_max",
    "chart.geometry_support_bounce_long_pressure",
    "chart.geometry_resistance_reject_short_pressure",
    "chart.geometry_trendline_break_up_pressure",
    "chart.geometry_trendline_break_down_pressure",
    "chart.smc_liquidity_reclaim_confirmation_long",
    "chart.smc_liquidity_reclaim_confirmation_short",
    "chart.smc_liquidity_false_breakout_quality_long",
    "chart.smc_liquidity_false_breakout_quality_short",
)


def _name_index(names: Iterable[str]) -> dict[str, int]:
    values = list(names)
    invalid = [name for name in values if not isinstance(name, str) or not name]
    if invalid:
        raise RuntimeError(f"SUPPORT_RESISTANCE_MEMORY_FEATURE_NAMES_INVALID: {invalid[:10]}")
    duplicates = sorted({name for name in values if values.count(name) > 1})
    if duplicates:
        raise RuntimeError(f"SUPPORT_RESISTANCE_MEMORY_FEATURE_NAMES_DUPLICATE: {duplicates[:10]}")
    return {name: i for i, name in enumerate(values)}


def missing_support_resistance_memory_source_fields(feature_names: Iterable[str]) -> list[str]:
    available = set(feature_names)
    return [name for name in SUPPORT_RESISTANCE_MEMORY_SOURCE_FIELDS if name not in available]


def _require_source_matrix(
    x: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, dict[str, int]]:
    try:
        matrix = np.asarray(x, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("SUPPORT_RESISTANCE_MEMORY_INPUT_NOT_NUMERIC") from exc
    if matrix.ndim != 2:
        raise RuntimeError(f"SUPPORT_RESISTANCE_MEMORY_INPUT_NOT_2D: shape={matrix.shape}")
    if matrix.shape[0] == 0:
        raise RuntimeError("SUPPORT_RESISTANCE_MEMORY_INPUT_EMPTY")
    if len(feature_names) != matrix.shape[1]:
        raise RuntimeError(
            "SUPPORT_RESISTANCE_MEMORY_FEATURE_NAME_COUNT_MISMATCH: "
            f"names={len(feature_names)} columns={matrix.shape[1]}"
        )
    index = _name_index(feature_names)
    missing = missing_support_resistance_memory_source_fields(feature_names)
    if missing:
        raise RuntimeError(
            "SUPPORT_RESISTANCE_MEMORY_SOURCE_FIELDS_MISSING: "
            f"{missing[:30]} total={len(missing)}"
        )
    if not np.isfinite(matrix).all():
        bad = np.argwhere(~np.isfinite(matrix))[0]
        row, column = int(bad[0]), int(bad[1])
        raise RuntimeError(
            "SUPPORT_RESISTANCE_MEMORY_SOURCE_NONFINITE: "
            f"row={row} field={feature_names[column]}"
        )
    return matrix, index


def _col(x: np.ndarray, index: dict[str, int], name: str) -> np.ndarray:
    try:
        column = index[name]
    except KeyError as exc:
        raise RuntimeError(f"SUPPORT_RESISTANCE_MEMORY_SOURCE_FIELD_MISSING: {name}") from exc
    arr = np.asarray(x[:, column], dtype=np.float32)
    if arr.ndim != 1 or not np.isfinite(arr).all():
        raise RuntimeError(f"SUPPORT_RESISTANCE_MEMORY_SOURCE_FIELD_INVALID: {name}")
    return arr


def _clip(arr: np.ndarray, lo: float = -25.0, hi: float = 25.0) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float32)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise RuntimeError(
            f"SUPPORT_RESISTANCE_MEMORY_DERIVED_VALUE_INVALID: shape={values.shape}"
        )
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


def _decayed_memory(arr: np.ndarray, decay: float) -> np.ndarray:
    values = _clip01(arr)
    out = np.empty_like(values, dtype=np.float32)
    carry = np.float32(0.0)
    alpha = np.float32(1.0 - float(decay))
    decay32 = np.float32(decay)
    for i, value in enumerate(values):
        carry = decay32 * carry + alpha * np.float32(value)
        out[i] = carry
    return _clip01(out)


def _add(
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
        raise RuntimeError(f"support/resistance memory feature {name} is not 1D: {clean.shape}")
    if not np.isfinite(clean).all():
        raise RuntimeError(f"support/resistance memory feature {name} contains non-finite values")
    arrays.append(clean)
    names.append(f"{SUPPORT_RESISTANCE_MEMORY_FEATURE_PREFIX}{name}")


def build_entry_support_resistance_memory_layer(
    x: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Build deterministic support/resistance memory from exact sources."""
    x, idx = _require_source_matrix(x, feature_names)
    arrays: list[np.ndarray] = []
    names: list[str] = []

    def c(name: str) -> np.ndarray:
        return _col(x, idx, name)

    h1_trend = _tanh(c("ctx_cont._v1h1_ema_diff"))
    h4_trend = _tanh(c("ctx_cont._v1h4_ema_diff"))
    d1_slope = _tanh(c("ctx_cont.d1_ema_slope_20_canon_v2"))
    m15_trend = _tanh(c("ctx_cont.m15_trend_sign_canon_v2"))
    regime_stack = _tanh(c("ctx_cont.regime_stack_sum_v3"), scale=3.0)
    trend = _clip(0.30 * h1_trend + 0.30 * h4_trend + 0.20 * d1_slope + 0.12 * m15_trend + 0.08 * regime_stack)
    trend_up = _pos(trend)
    trend_down = _neg(trend)

    pivot_proximity = _prox_abs(c("ctx_cont.sr_nearest_pivot_abs_atr"))
    s1 = _prox_abs(c("ctx_cont.dist_to_S1_atr"))
    s2 = _prox_abs(c("ctx_cont.dist_to_S2_atr"))
    r1 = _prox_abs(c("ctx_cont.dist_to_R1_atr"))
    r2 = _prox_abs(c("ctx_cont.dist_to_R2_atr"))
    m5_lo = _prox_abs(c("ctx_cont.dist_to_m5_lo_atr"))
    m5_hi = _prox_abs(c("ctx_cont.dist_to_m5_hi_atr"))
    m15_lo = _prox_abs(c("ctx_cont.dist_to_m15_lo_atr"))
    m15_hi = _prox_abs(c("ctx_cont.dist_to_m15_hi_atr"))
    h1_lo = _prox_abs(c("ctx_cont.dist_to_h1_lo_atr"))
    h1_hi = _prox_abs(c("ctx_cont.dist_to_h1_hi_atr"))
    h4_lo = _prox_abs(c("ctx_cont.dist_to_h4_lo_atr"))
    h4_hi = _prox_abs(c("ctx_cont.dist_to_h4_hi_atr"))
    d1_lo = _prox_abs(c("ctx_cont.dist_to_d1_lo_atr"))
    d1_hi = _prox_abs(c("ctx_cont.dist_to_d1_hi_atr"))
    swing_low = _prox_abs(c("ctx_cont.dist_last_swing_low_atr")) * (0.50 + _recency(c("ctx_cont.bars_since_swing_low")))
    swing_high = _prox_abs(c("ctx_cont.dist_last_swing_high_atr")) * (0.50 + _recency(c("ctx_cont.bars_since_swing_high")))

    support_sources = np.vstack(
        [
            _clip01(c("ctx_cont.sr_support_proximity_exp")),
            pivot_proximity,
            s1,
            s2,
            m5_lo,
            m15_lo,
            h1_lo,
            h4_lo,
            d1_lo,
            _prox_abs(c("ctx_cont.liquidity_lo_nearest_abs_atr")),
            _clip01(swing_low),
            _clip01(c("chart.geometry_support_line_proximity_stack")),
        ]
    )
    resistance_sources = np.vstack(
        [
            _clip01(c("ctx_cont.sr_resistance_proximity_exp")),
            pivot_proximity,
            r1,
            r2,
            m5_hi,
            m15_hi,
            h1_hi,
            h4_hi,
            d1_hi,
            _prox_abs(c("ctx_cont.liquidity_hi_nearest_abs_atr")),
            _clip01(swing_high),
            _clip01(c("chart.geometry_resistance_line_proximity_stack")),
        ]
    )
    support_stack = support_sources.max(axis=0).astype(np.float32)
    resistance_stack = resistance_sources.max(axis=0).astype(np.float32)
    nearest_level = np.maximum(support_stack, resistance_stack).astype(np.float32)
    support_minus_resistance = _clip(
        c("ctx_cont.sr_support_minus_resistance_prox") + c("chart.geometry_support_minus_resistance_stack") + support_stack - resistance_stack,
        -2.0,
        2.0,
    )

    support_mtf = _clip01(0.34 * h1_lo + 0.33 * h4_lo + 0.33 * d1_lo)
    resistance_mtf = _clip01(0.34 * h1_hi + 0.33 * h4_hi + 0.33 * d1_hi)
    mtf_level_confluence = _clip01(np.maximum(support_mtf, resistance_mtf) * (0.70 + 0.30 * nearest_level))
    mtf_balance = _clip(support_mtf - resistance_mtf, -1.0, 1.0)

    low_liquidity = np.maximum(_prox_abs(c("ctx_cont.liquidity_lo_nearest_abs_atr")), support_stack).astype(np.float32)
    high_liquidity = np.maximum(_prox_abs(c("ctx_cont.liquidity_hi_nearest_abs_atr")), resistance_stack).astype(np.float32)
    liquidity_balance = _clip(c("ctx_cont.liquidity_lo_minus_hi_prox") + low_liquidity - high_liquidity, -2.0, 2.0)
    two_sided_liquidity = _clip01(np.minimum(low_liquidity, high_liquidity) * (0.50 + 0.50 * nearest_level))

    support_touch_event = _clip01(0.45 * support_stack + 0.25 * support_mtf + 0.20 * low_liquidity + 0.10 * _pos(support_minus_resistance))
    resistance_touch_event = _clip01(0.45 * resistance_stack + 0.25 * resistance_mtf + 0.20 * high_liquidity + 0.10 * _neg(support_minus_resistance))
    support_fast = _decayed_memory(support_touch_event, 0.82)
    support_slow = _decayed_memory(support_touch_event, 0.96)
    resistance_fast = _decayed_memory(resistance_touch_event, 0.82)
    resistance_slow = _decayed_memory(resistance_touch_event, 0.96)
    support_repeated = _clip01((0.62 * support_fast + 0.38 * support_slow) * (0.50 + support_stack))
    resistance_repeated = _clip01((0.62 * resistance_fast + 0.38 * resistance_slow) * (0.50 + resistance_stack))
    repeated_balance = _clip(support_repeated - resistance_repeated, -1.0, 1.0)

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
    bos_up = _clip01(c("snap.smc_bos_up") + 0.50 * _pos(c("ctx_cont.smc_bos_pressure_last12")) + 0.25 * _pos(c("ctx_cont.smc_bos_pressure_last48")))
    bos_down = _clip01(c("snap.smc_bos_down") + 0.50 * _neg(c("ctx_cont.smc_bos_pressure_last12")) + 0.25 * _neg(c("ctx_cont.smc_bos_pressure_last48")))
    choch = _clip01(c("snap.smc_choch") + c("ctx_cont.smc_choch_recent_tau12") + 0.50 * c("ctx_cont.smc_choch_recent_tau24"))

    premium = _clip01(c("snap.smc_premium_discount"))
    discount = _clip01(1.0 - premium)
    clv_unit = _clip01(0.5 + 0.5 * _clip(c("snap._v1_clv"), -1.0, 1.0))
    body_direction = _clip(c("snap.body_pct"), -1.0, 1.0)
    body_share = _clip01(np.abs(body_direction) + c("snap._v1_body_share_1"))
    body_bull = _clip01(_pos(body_direction) + clv_unit)
    body_bear = _clip01(_neg(body_direction) + (1.0 - clv_unit))
    wick_asym = _clip(c("snap.wick_asym"), -1.0, 1.0)
    wick_ratio = _clip01(c("ctx_cont.wick_ratio"))
    lower_wick = _clip01(_neg(wick_asym) + 0.50 * (1.0 - wick_ratio))
    upper_wick = _clip01(_pos(wick_asym) + 0.50 * wick_ratio)
    close_near_high = _clip01(0.55 * clv_unit + 0.45 * (1.0 - wick_ratio))
    close_near_low = _clip01(0.55 * (1.0 - clv_unit) + 0.45 * wick_ratio)
    wick_reject_long = _clip01(lower_wick * (0.55 + close_near_high + 0.25 * body_bull) * (0.75 + 0.25 * body_share))
    wick_reject_short = _clip01(upper_wick * (0.55 + close_near_low + 0.25 * body_bear) * (0.75 + 0.25 * body_share))

    support_respect_long = _clip01(
        support_stack
        * (0.45 + 0.25 * support_repeated + 0.20 * support_mtf + 0.10 * discount)
        * (0.45 + 0.35 * wick_reject_long + 0.20 * body_bull)
    )
    resistance_respect_short = _clip01(
        resistance_stack
        * (0.45 + 0.25 * resistance_repeated + 0.20 * resistance_mtf + 0.10 * premium)
        * (0.45 + 0.35 * wick_reject_short + 0.20 * body_bear)
    )
    support_break_down = _clip(
        support_stack
        * (0.45 + support_repeated + support_mtf)
        * (0.45 + bos_down + trend_down + body_bear)
        * (1.0 - 0.50 * wick_reject_long),
        0.0,
        5.0,
    )
    resistance_break_up = _clip(
        resistance_stack
        * (0.45 + resistance_repeated + resistance_mtf)
        * (0.45 + bos_up + trend_up + body_bull)
        * (1.0 - 0.50 * wick_reject_short),
        0.0,
        5.0,
    )
    support_reclaim_long = _clip01(
        support_stack
        * low_liquidity
        * sweep_down
        * sweep_recent
        * (0.45 + sweep_size + wick_reject_long + close_near_high)
        * (0.70 + 0.30 * discount)
        + 0.35 * c("chart.smc_liquidity_reclaim_confirmation_long")
    )
    resistance_reclaim_short = _clip01(
        resistance_stack
        * high_liquidity
        * sweep_up
        * sweep_recent
        * (0.45 + sweep_size + wick_reject_short + close_near_low)
        * (0.70 + 0.30 * premium)
        + 0.35 * c("chart.smc_liquidity_reclaim_confirmation_short")
    )
    respect_minus_break = _clip(support_respect_long + resistance_respect_short - 0.50 * (support_break_down + resistance_break_up), -2.0, 2.0)
    reclaim_minus_break = _clip(support_reclaim_long + resistance_reclaim_short - 0.50 * (support_break_down + resistance_break_up), -2.0, 2.0)

    low_level_rejection_long = _clip01(
        low_liquidity
        * support_stack
        * (0.40 + sweep_down + sweep_recent)
        * (0.35 + wick_reject_long + choch)
        * (0.80 + 0.20 * discount)
        + 0.30 * c("chart.smc_liquidity_false_breakout_quality_long")
        + 0.15 * c("chart.geometry_support_bounce_long_pressure")
    )
    high_level_rejection_short = _clip01(
        high_liquidity
        * resistance_stack
        * (0.40 + sweep_up + sweep_recent)
        * (0.35 + wick_reject_short + choch)
        * (0.80 + 0.20 * premium)
        + 0.30 * c("chart.smc_liquidity_false_breakout_quality_short")
        + 0.15 * c("chart.geometry_resistance_reject_short_pressure")
    )
    rejection_balance = _clip(low_level_rejection_long - high_level_rejection_short, -1.0, 1.0)
    resistance_continuation_long = _clip(
        high_liquidity
        * resistance_stack
        * (0.45 + resistance_repeated + resistance_mtf)
        * (0.45 + bos_up + trend_up + body_bull)
        * (1.0 - 0.40 * wick_reject_short)
        + 0.20 * c("chart.geometry_trendline_break_up_pressure"),
        0.0,
        5.0,
    )
    support_continuation_short = _clip(
        low_liquidity
        * support_stack
        * (0.45 + support_repeated + support_mtf)
        * (0.45 + bos_down + trend_down + body_bear)
        * (1.0 - 0.40 * wick_reject_long)
        + 0.20 * c("chart.geometry_trendline_break_down_pressure"),
        0.0,
        5.0,
    )
    continuation_balance = _clip(resistance_continuation_long - support_continuation_short, -2.0, 2.0)
    exhaustion_risk = _clip(
        two_sided_liquidity
        * nearest_level
        * (0.40 + support_repeated + resistance_repeated)
        * (0.40 + sweep_size + choch)
        * (0.80 + 0.20 * np.abs(liquidity_balance)),
        0.0,
        5.0,
    )
    trap_risk_long = _clip(high_liquidity * resistance_stack * premium * (0.50 + sweep_up + wick_reject_short) * (1.0 - support_reclaim_long), 0.0, 5.0)
    trap_risk_short = _clip(low_liquidity * support_stack * discount * (0.50 + sweep_down + wick_reject_long) * (1.0 - resistance_reclaim_short), 0.0, 5.0)

    _add(arrays, names, "nearest_pivot_proximity", pivot_proximity, lo=0.0, hi=1.0)
    _add(arrays, names, "nearest_level_proximity", nearest_level, lo=0.0, hi=1.0)
    _add(arrays, names, "support_level_proximity_stack", support_stack, lo=0.0, hi=1.0)
    _add(arrays, names, "resistance_level_proximity_stack", resistance_stack, lo=0.0, hi=1.0)
    _add(arrays, names, "support_minus_resistance_level_balance", support_minus_resistance, lo=-2.0, hi=2.0)
    _add(arrays, names, "support_repeated_test_fast", support_fast, lo=0.0, hi=1.0)
    _add(arrays, names, "resistance_repeated_test_fast", resistance_fast, lo=0.0, hi=1.0)
    _add(arrays, names, "support_repeated_test_slow", support_slow, lo=0.0, hi=1.0)
    _add(arrays, names, "resistance_repeated_test_slow", resistance_slow, lo=0.0, hi=1.0)
    _add(arrays, names, "support_repeated_test_pressure", support_repeated, lo=0.0, hi=1.0)
    _add(arrays, names, "resistance_repeated_test_pressure", resistance_repeated, lo=0.0, hi=1.0)
    _add(arrays, names, "repeated_test_balance", repeated_balance, lo=-1.0, hi=1.0)
    _add(arrays, names, "support_respect_pressure_long", support_respect_long, lo=0.0, hi=1.0)
    _add(arrays, names, "resistance_respect_pressure_short", resistance_respect_short, lo=0.0, hi=1.0)
    _add(arrays, names, "support_break_pressure_down", support_break_down, lo=0.0, hi=5.0)
    _add(arrays, names, "resistance_break_pressure_up", resistance_break_up, lo=0.0, hi=5.0)
    _add(arrays, names, "support_reclaim_pressure_long", support_reclaim_long, lo=0.0, hi=1.0)
    _add(arrays, names, "resistance_reclaim_pressure_short", resistance_reclaim_short, lo=0.0, hi=1.0)
    _add(arrays, names, "respect_minus_break_balance", respect_minus_break, lo=-2.0, hi=2.0)
    _add(arrays, names, "reclaim_minus_break_balance", reclaim_minus_break, lo=-2.0, hi=2.0)
    _add(arrays, names, "mtf_h1_h4_d1_support_confluence", support_mtf, lo=0.0, hi=1.0)
    _add(arrays, names, "mtf_h1_h4_d1_resistance_confluence", resistance_mtf, lo=0.0, hi=1.0)
    _add(arrays, names, "mtf_h1_h4_d1_level_confluence", mtf_level_confluence, lo=0.0, hi=1.0)
    _add(arrays, names, "mtf_h1_h4_d1_support_minus_resistance", mtf_balance, lo=-1.0, hi=1.0)
    _add(arrays, names, "liquidity_low_level_rejection_long", low_level_rejection_long, lo=0.0, hi=1.0)
    _add(arrays, names, "liquidity_high_level_rejection_short", high_level_rejection_short, lo=0.0, hi=1.0)
    _add(arrays, names, "liquidity_level_rejection_balance", rejection_balance, lo=-1.0, hi=1.0)
    _add(arrays, names, "liquidity_resistance_break_continuation_long", resistance_continuation_long, lo=0.0, hi=5.0)
    _add(arrays, names, "liquidity_support_break_continuation_short", support_continuation_short, lo=0.0, hi=5.0)
    _add(arrays, names, "liquidity_level_continuation_balance", continuation_balance, lo=-2.0, hi=2.0)
    _add(arrays, names, "two_sided_liquidity_level_pressure", two_sided_liquidity, lo=0.0, hi=1.0)
    _add(arrays, names, "level_memory_exhaustion_risk", exhaustion_risk, lo=0.0, hi=5.0)
    _add(arrays, names, "major_level_trap_risk_long", trap_risk_long, lo=0.0, hi=5.0)
    _add(arrays, names, "major_level_trap_risk_short", trap_risk_short, lo=0.0, hi=5.0)

    out = np.column_stack(arrays).astype(np.float32, copy=False) if arrays else np.empty((x.shape[0], 0), dtype=np.float32)
    if not np.isfinite(out).all():
        raise RuntimeError("support/resistance memory layer contains non-finite values")
    if len(set(names)) != len(names):
        dupes = sorted({name for name in names if names.count(name) > 1})
        raise RuntimeError(f"support/resistance memory layer has duplicate names: {dupes[:10]}")
    return out, names


SUPPORT_RESISTANCE_MEMORY_FEATURE_NAMES = tuple(
    name
    for name in build_entry_support_resistance_memory_layer(
        np.zeros((1, len(SUPPORT_RESISTANCE_MEMORY_SOURCE_FIELDS)), dtype=np.float32),
        list(SUPPORT_RESISTANCE_MEMORY_SOURCE_FIELDS),
    )[1]
)
