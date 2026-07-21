"""Canonical Entry multi-timeframe confluence features.

This model-native layer binds closed-bar Entry families across M5/M15/H1/H4/D1:
trend/EMA, structure/BOS, SMC sweep/reclaim, Fibonacci/SR proximity and
session/regime agreement with the same exact source contract in train and serve.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


MTF_CONFLUENCE_FEATURE_VERSION = (
    "entry_mtf_confluence_v1_20260717_causal_family_agreement_failclosed"
)
MTF_CONFLUENCE_TIMEFRAMES = ("m5", "m15", "h1", "h4", "d1")

MTF_CONFLUENCE_SOURCE_FIELDS = (
    "snap._v1_ema_diff",
    "snap.ema20_slope",
    "snap.pos_vs_ema200",
    "ctx_cont.m15_trend_sign_canon_v2",
    "ctx_cont._v1h1_ema_diff",
    "ctx_cont._v1h1_slope5",
    "ctx_cont._v1h4_ema_diff",
    "ctx_cont._v1h4_slope5",
    "ctx_cont.d1_ema_slope_20_canon_v2",
    "ctx_cont.d1_pct_change_5_canon_v2",
    "ctx_cont.regime_tf_agreement_v3",
    "ctx_cont.regime_stack_sum_v3",
    "ctx_cont.regime_divergence_flag_v3",
    "chart.foundation_hh_state",
    "chart.foundation_hl_state",
    "chart.foundation_lh_state",
    "chart.foundation_ll_state",
    "chart.foundation_structure_up_minus_down",
    "chart.foundation_bos_up_age_bars",
    "chart.foundation_bos_down_age_bars",
    "chart.foundation_bos_up_recent_tau24",
    "chart.foundation_bos_down_recent_tau24",
    "chart.foundation_bos_recent_balance",
    "chart.foundation_choch_recent_tau24",
    "chart.foundation_pullback_phase_up",
    "chart.foundation_pullback_phase_down",
    "chart.foundation_pullback_depth_norm",
    *tuple(f"ctx_cont.struct_continuation_up_{tf}_v3" for tf in MTF_CONFLUENCE_TIMEFRAMES),
    *tuple(f"ctx_cont.struct_continuation_down_{tf}_v3" for tf in MTF_CONFLUENCE_TIMEFRAMES),
    "chart.foundation_sweep_low_reclaim_up_proxy",
    "chart.foundation_sweep_high_reclaim_down_proxy",
    "chart.foundation_false_breakout_low_followthrough_up_proxy",
    "chart.foundation_false_breakout_high_followthrough_down_proxy",
    "chart.foundation_sweep_reclaim_balance_proxy",
    "snap.smc_sweep_up",
    "snap.smc_sweep_down",
    "snap.smc_sweep_size_atr",
    "snap.smc_bars_since_sweep",
    "snap.smc_premium_discount",
    "ctx_cont.smc_sweep_recency_tau24",
    "ctx_cont.smc_sweep_size_recent_tau12",
    "ctx_cont.sr_nearest_pivot_abs_atr",
    "ctx_cont.sr_support_proximity_exp",
    "ctx_cont.sr_resistance_proximity_exp",
    "ctx_cont.sr_support_minus_resistance_prox",
    "ctx_cont.dist_to_S1_atr",
    "ctx_cont.dist_to_S2_atr",
    "ctx_cont.dist_to_R1_atr",
    "ctx_cont.dist_to_R2_atr",
    *tuple(f"ctx_cont.dist_to_{tf}_lo_atr" for tf in MTF_CONFLUENCE_TIMEFRAMES),
    *tuple(f"ctx_cont.dist_to_{tf}_hi_atr" for tf in MTF_CONFLUENCE_TIMEFRAMES),
    "ctx_cont.liquidity_lo_nearest_abs_atr",
    "ctx_cont.liquidity_hi_nearest_abs_atr",
    "ctx_cont.retracement_from_last_impulse",
    "ctx_cont.d1_close_pct_in_20day_range_canon_v2",
    "ctx_cont.minutes_since_session_open",
    "ctx_cont.minutes_to_next_session_boundary",
    "ctx_cont.session_change_flag",
    "ctx_cont.session_tradable",
    "ctx_cont.is_ASIA",
    "ctx_cont.is_asia_eu_overlap",
    "ctx_cont.is_eu_us_overlap",
    "ctx_cont.is_eu_only",
    "ctx_cont.is_us_only",
    "ctx_cont.spread_bps",
    "ctx_cont.atr_bps",
    "ctx_cat.spread_bucket",
    "ctx_cat.vol_regime_id",
    "ctx_cont.vol_pct_m5_1yr",
    "ctx_cont.vol_pct_h1_1yr",
    "ctx_cont.D1_atr_percentile_252",
    "ctx_cont.d1_regime_changed_flag_v3",
    "ctx_cont.bars_since_d1_regime_change_v3",
    "trend.ema_mtf_score",
    "trend.ema_mtf_agreement_pressure",
    "trend.ema_mtf_divergence_pressure",
    "chart.structure_swing_mtf_structure_agreement",
    "chart.structure_swing_mtf_structure_divergence",
    "chart.smc_liquidity_sweep_reclaim_strength_long",
    "chart.smc_liquidity_sweep_reclaim_strength_short",
    "chart.smc_liquidity_false_breakout_quality_long",
    "chart.smc_liquidity_false_breakout_quality_short",
    "chart.geometry_support_line_proximity_stack",
    "chart.geometry_resistance_line_proximity_stack",
    "chart.geometry_fib_golden_zone_proximity",
    "session_regime.regime_persistence_score",
    "session_regime.mtf_regime_divergence_pressure",
    "session_regime.spread_cost_pressure",
)

MTF_CONFLUENCE_FEATURE_NAMES = (
    "trend.mtf_confluence_trend_direction_score",
    "trend.mtf_confluence_trend_tf_agreement",
    "trend.mtf_confluence_trend_tf_conflict",
    "trend.mtf_confluence_trend_m5_m15_h1_h4_d1_alignment",
    "trend.mtf_confluence_long_trend_bias",
    "trend.mtf_confluence_short_trend_bias",
    "chart.structure_swing_mtf_confluence_structure_direction_score",
    "chart.structure_swing_mtf_confluence_bos_alignment_up",
    "chart.structure_swing_mtf_confluence_bos_alignment_down",
    "chart.structure_swing_mtf_confluence_structure_conflict",
    "chart.structure_swing_mtf_confluence_pullback_abstain_pressure",
    "chart.smc_liquidity_mtf_confluence_sweep_reclaim_long",
    "chart.smc_liquidity_mtf_confluence_sweep_reclaim_short",
    "chart.smc_liquidity_mtf_confluence_false_breakout_long",
    "chart.smc_liquidity_mtf_confluence_false_breakout_short",
    "chart.smc_liquidity_mtf_confluence_liquidity_conflict",
    "chart.smc_liquidity_mtf_confluence_premium_discount_alignment",
    "chart.geometry_mtf_confluence_fib_sr_long_proximity",
    "chart.geometry_mtf_confluence_fib_sr_short_proximity",
    "chart.geometry_mtf_confluence_sr_balance",
    "chart.geometry_mtf_confluence_major_level_density",
    "session_regime.mtf_confluence_session_permission",
    "session_regime.mtf_confluence_regime_agreement",
    "session_regime.mtf_confluence_regime_conflict",
    "session_regime.mtf_confluence_spread_vol_abstain",
    "session_regime.mtf_confluence_session_regime_tradable_long",
    "session_regime.mtf_confluence_session_regime_tradable_short",
    "session_regime.mtf_confluence_long_agreement_score",
    "session_regime.mtf_confluence_short_agreement_score",
    "session_regime.mtf_confluence_direction_balance",
    "session_regime.mtf_confluence_conflict_score",
    "session_regime.mtf_confluence_abstain_score",
)


def _name_index(names: Iterable[str]) -> dict[str, int]:
    values = list(names)
    invalid = [name for name in values if not isinstance(name, str) or not name]
    if invalid:
        raise RuntimeError(f"MTF_CONFLUENCE_FEATURE_NAMES_INVALID: {invalid[:10]}")
    duplicates = sorted({name for name in values if values.count(name) > 1})
    if duplicates:
        raise RuntimeError(f"MTF_CONFLUENCE_FEATURE_NAMES_DUPLICATE: {duplicates[:10]}")
    return {name: i for i, name in enumerate(values)}


def missing_mtf_confluence_source_fields(feature_names: Iterable[str]) -> list[str]:
    """Return exact canonical source fields absent from ``feature_names``."""
    available = set(feature_names)
    return [name for name in MTF_CONFLUENCE_SOURCE_FIELDS if name not in available]


def _require_source_matrix(
    x: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, dict[str, int]]:
    try:
        matrix = np.asarray(x, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("MTF_CONFLUENCE_INPUT_NOT_NUMERIC") from exc
    if matrix.ndim != 2:
        raise RuntimeError(f"MTF_CONFLUENCE_INPUT_NOT_2D: shape={matrix.shape}")
    if matrix.shape[0] == 0:
        raise RuntimeError("MTF_CONFLUENCE_INPUT_EMPTY")
    if len(feature_names) != matrix.shape[1]:
        raise RuntimeError(
            "MTF_CONFLUENCE_FEATURE_NAME_COUNT_MISMATCH: "
            f"names={len(feature_names)} columns={matrix.shape[1]}"
        )
    index = _name_index(feature_names)
    missing = missing_mtf_confluence_source_fields(feature_names)
    if missing:
        raise RuntimeError(
            "MTF_CONFLUENCE_SOURCE_FIELDS_MISSING: "
            f"{missing[:30]} total={len(missing)}"
        )
    if not np.isfinite(matrix).all():
        bad = np.argwhere(~np.isfinite(matrix))[0]
        row, column = int(bad[0]), int(bad[1])
        raise RuntimeError(
            "MTF_CONFLUENCE_SOURCE_NONFINITE: "
            f"row={row} field={feature_names[column]}"
        )
    atr_bps = matrix[:, index["ctx_cont.atr_bps"]]
    spread_bps = matrix[:, index["ctx_cont.spread_bps"]]
    if np.any(atr_bps <= 0.0):
        raise RuntimeError("MTF_CONFLUENCE_ATR_BPS_NOT_POSITIVE")
    if np.any(spread_bps < 0.0):
        raise RuntimeError("MTF_CONFLUENCE_SPREAD_BPS_NEGATIVE")
    return matrix, index


def _col(x: np.ndarray, index: dict[str, int], name: str) -> np.ndarray:
    try:
        column = index[name]
    except KeyError as exc:
        raise RuntimeError(f"MTF_CONFLUENCE_SOURCE_FIELD_MISSING: {name}") from exc
    arr = np.asarray(x[:, column], dtype=np.float32)
    if arr.ndim != 1 or not np.isfinite(arr).all():
        raise RuntimeError(f"MTF_CONFLUENCE_SOURCE_FIELD_INVALID: {name}")
    return arr


def _clip(arr: np.ndarray, lo: float = -25.0, hi: float = 25.0) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float32)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise RuntimeError(f"MTF_CONFLUENCE_DERIVED_VALUE_INVALID: shape={values.shape}")
    return np.clip(values, lo, hi).astype(np.float32, copy=False)


def _clip01(arr: np.ndarray) -> np.ndarray:
    return _clip(arr, 0.0, 1.0)


def _pos(arr: np.ndarray) -> np.ndarray:
    return np.maximum(arr, 0.0).astype(np.float32, copy=False)


def _neg(arr: np.ndarray) -> np.ndarray:
    return np.maximum(-arr, 0.0).astype(np.float32, copy=False)


def _tanh(arr: np.ndarray, scale: float = 1.0) -> np.ndarray:
    return np.tanh(np.asarray(arr, dtype=np.float32) / max(float(scale), 1e-6)).astype(np.float32, copy=False)


def _prox_abs(arr: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.abs(np.asarray(arr, dtype=np.float32)))).astype(np.float32, copy=False)


def _recency(age: np.ndarray, tau: float = 24.0) -> np.ndarray:
    return np.exp(-np.maximum(np.asarray(age, dtype=np.float32), 0.0) / max(float(tau), 1e-6)).astype(np.float32)


def _safe_ratio(num: np.ndarray, denom: np.ndarray, *, denom_floor: float = 1e-3) -> np.ndarray:
    return (np.asarray(num, dtype=np.float32) / np.maximum(np.abs(np.asarray(denom, dtype=np.float32)), denom_floor)).astype(
        np.float32,
        copy=False,
    )


def _fib_proximity(position: np.ndarray, level: float) -> np.ndarray:
    return np.exp(-np.abs(_clip01(position) - float(level)) * 12.0).astype(np.float32)


def _mean(parts: Iterable[np.ndarray]) -> np.ndarray:
    stack = np.vstack([np.asarray(part, dtype=np.float32) for part in parts])
    return stack.mean(axis=0).astype(np.float32, copy=False)


def _sign_shares(parts: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stack = np.vstack([np.asarray(part, dtype=np.float32) for part in parts])
    active = np.abs(stack) > 1e-6
    active_count = np.maximum(active.sum(axis=0).astype(np.float32), 1.0)
    bull = ((stack > 0.0) & active).sum(axis=0).astype(np.float32) / active_count
    bear = ((stack < 0.0) & active).sum(axis=0).astype(np.float32) / active_count
    neutral = 1.0 - _clip01(active_count / float(stack.shape[0]))
    return _clip01(bull), _clip01(bear), _clip01(neutral)


def _add(arrays: list[np.ndarray], names: list[str], name: str, arr: np.ndarray, *, lo: float = -25.0, hi: float = 25.0) -> None:
    clean = _clip(np.asarray(arr, dtype=np.float32), lo, hi)
    if clean.ndim != 1:
        raise RuntimeError(f"MTF confluence feature {name} is not 1D: {clean.shape}")
    if not np.isfinite(clean).all():
        raise RuntimeError(f"MTF confluence feature {name} contains non-finite values")
    arrays.append(clean)
    names.append(name)


def build_entry_mtf_confluence_layer(
    x: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Build deterministic, closed-bar MTF confluence features.

    The implementation uses only the current as-of row and already-materialized
    closed-bar feature columns. It never references future rows.
    """
    x, idx = _require_source_matrix(x, feature_names)

    def c(name: str) -> np.ndarray:
        return _col(x, idx, name)

    m5_trend = _clip(
        0.45 * _tanh(c("snap._v1_ema_diff"))
        + 0.35 * _tanh(c("snap.ema20_slope"))
        + 0.20 * _tanh(c("snap.pos_vs_ema200")),
        -1.0,
        1.0,
    )
    m15_trend = _tanh(c("ctx_cont.m15_trend_sign_canon_v2"))
    h1_trend = _clip(0.62 * _tanh(c("ctx_cont._v1h1_ema_diff")) + 0.38 * _tanh(c("ctx_cont._v1h1_slope5")), -1.0, 1.0)
    h4_trend = _clip(0.62 * _tanh(c("ctx_cont._v1h4_ema_diff")) + 0.38 * _tanh(c("ctx_cont._v1h4_slope5")), -1.0, 1.0)
    d1_trend = _clip(
        0.70 * _tanh(c("ctx_cont.d1_ema_slope_20_canon_v2"))
        + 0.30 * _tanh(c("ctx_cont.d1_pct_change_5_canon_v2"), scale=25.0),
        -1.0,
        1.0,
    )
    trend_smart = _tanh(c("trend.ema_mtf_score"))
    trend_parts = [m5_trend, m15_trend, h1_trend, h4_trend, d1_trend]
    trend_direction_raw = _mean(trend_parts)
    trend_direction = _clip(0.85 * trend_direction_raw + 0.15 * trend_smart, -1.0, 1.0)
    trend_bull_share, trend_bear_share, trend_neutral_share = _sign_shares(trend_parts)
    regime_agreement_raw = _clip01(c("ctx_cont.regime_tf_agreement_v3"))
    regime_divergence_raw = _clip01(c("ctx_cont.regime_divergence_flag_v3"))
    trend_agreement = _clip01(
        0.64 * np.maximum(trend_bull_share, trend_bear_share)
        + 0.22 * regime_agreement_raw
        + 0.14 * c("trend.ema_mtf_agreement_pressure")
    )
    trend_conflict = _clip01(
        0.46 * np.minimum(trend_bull_share, trend_bear_share) * 2.0
        + 0.22 * regime_divergence_raw
        + 0.16 * np.abs(m5_trend - d1_trend)
        + 0.10 * c("trend.ema_mtf_divergence_pressure")
        + 0.06 * trend_neutral_share
    )
    trend_alignment = _clip(trend_direction * trend_agreement, -1.0, 1.0)
    long_trend_support = _clip01(_pos(trend_direction) * trend_agreement * (1.0 - 0.35 * trend_conflict))
    short_trend_support = _clip01(_neg(trend_direction) * trend_agreement * (1.0 - 0.35 * trend_conflict))

    hh = _clip01(c("chart.foundation_hh_state"))
    hl = _clip01(c("chart.foundation_hl_state"))
    lh = _clip01(c("chart.foundation_lh_state"))
    ll = _clip01(c("chart.foundation_ll_state"))
    structure_balance = _clip(c("chart.foundation_structure_up_minus_down"), -2.0, 2.0)
    structure_up_base = _clip01(0.35 * hh + 0.35 * hl + 0.30 * _pos(structure_balance * 0.5))
    structure_down_base = _clip01(0.35 * lh + 0.35 * ll + 0.30 * _neg(structure_balance * 0.5))
    struct_up_tf = _mean(_clip01(c(f"ctx_cont.struct_continuation_up_{tf}_v3")) for tf in MTF_CONFLUENCE_TIMEFRAMES)
    struct_down_tf = _mean(_clip01(c(f"ctx_cont.struct_continuation_down_{tf}_v3")) for tf in MTF_CONFLUENCE_TIMEFRAMES)
    bos_up_recent = _clip01(
        c("chart.foundation_bos_up_recent_tau24")
        + 0.20 * _recency(c("chart.foundation_bos_up_age_bars"))
        + 0.15 * _pos(c("chart.foundation_bos_recent_balance"))
    )
    bos_down_recent = _clip01(
        c("chart.foundation_bos_down_recent_tau24")
        + 0.20 * _recency(c("chart.foundation_bos_down_age_bars"))
        + 0.15 * _neg(c("chart.foundation_bos_recent_balance"))
    )
    choch_recent = _clip01(c("chart.foundation_choch_recent_tau24"))
    structure_direction = _clip(
        0.36 * structure_balance * 0.5
        + 0.24 * (struct_up_tf - struct_down_tf)
        + 0.22 * (bos_up_recent - bos_down_recent)
        + 0.18 * ((hh + hl) - (lh + ll)) * 0.5,
        -1.0,
        1.0,
    )
    structure_agreement = _clip01(
        np.maximum(struct_up_tf, struct_down_tf) * 0.58
        + np.maximum(structure_up_base, structure_down_base) * 0.27
        + c("chart.structure_swing_mtf_structure_agreement") * 0.15
    )
    trend_structure_disagreement = _clip01(np.abs(np.sign(trend_direction) - np.sign(structure_direction)) * 0.5)
    structure_conflict = _clip01(
        0.28 * np.minimum(struct_up_tf, struct_down_tf) * 2.0
        + 0.24 * np.minimum(structure_up_base, structure_down_base) * 2.0
        + 0.20 * choch_recent
        + 0.16 * trend_structure_disagreement
        + 0.12 * c("chart.structure_swing_mtf_structure_divergence")
    )
    bos_alignment_up = _clip01(
        (0.42 * bos_up_recent + 0.30 * struct_up_tf + 0.28 * structure_up_base)
        * (0.72 + 0.28 * long_trend_support)
        * (1.0 - 0.40 * choch_recent)
    )
    bos_alignment_down = _clip01(
        (0.42 * bos_down_recent + 0.30 * struct_down_tf + 0.28 * structure_down_base)
        * (0.72 + 0.28 * short_trend_support)
        * (1.0 - 0.40 * choch_recent)
    )
    pullback_abstain = _clip01(
        c("chart.foundation_pullback_depth_norm")
        * (0.45 * (_clip01(c("chart.foundation_pullback_phase_up")) + _clip01(c("chart.foundation_pullback_phase_down"))) + 0.20)
        + 0.24 * structure_conflict
        + 0.20 * choch_recent
        + 0.11 * (1.0 - structure_agreement)
    )

    support_sources = np.vstack(
        [
            _clip01(c("ctx_cont.sr_support_proximity_exp")),
            _prox_abs(c("ctx_cont.sr_nearest_pivot_abs_atr")),
            _prox_abs(c("ctx_cont.dist_to_S1_atr")),
            _prox_abs(c("ctx_cont.dist_to_S2_atr")),
            *[_prox_abs(c(f"ctx_cont.dist_to_{tf}_lo_atr")) for tf in MTF_CONFLUENCE_TIMEFRAMES],
            _prox_abs(c("ctx_cont.liquidity_lo_nearest_abs_atr")),
            _clip01(c("chart.geometry_support_line_proximity_stack")),
        ]
    )
    resistance_sources = np.vstack(
        [
            _clip01(c("ctx_cont.sr_resistance_proximity_exp")),
            _prox_abs(c("ctx_cont.sr_nearest_pivot_abs_atr")),
            _prox_abs(c("ctx_cont.dist_to_R1_atr")),
            _prox_abs(c("ctx_cont.dist_to_R2_atr")),
            *[_prox_abs(c(f"ctx_cont.dist_to_{tf}_hi_atr")) for tf in MTF_CONFLUENCE_TIMEFRAMES],
            _prox_abs(c("ctx_cont.liquidity_hi_nearest_abs_atr")),
            _clip01(c("chart.geometry_resistance_line_proximity_stack")),
        ]
    )
    support_stack = support_sources.max(axis=0).astype(np.float32)
    resistance_stack = resistance_sources.max(axis=0).astype(np.float32)
    level_density = _clip01(0.70 * np.maximum(support_stack, resistance_stack) + 0.30 * _mean([support_sources.mean(axis=0), resistance_sources.mean(axis=0)]))
    sr_balance = _clip(c("ctx_cont.sr_support_minus_resistance_prox") * 0.25 + support_stack - resistance_stack, -1.0, 1.0)
    premium = _clip01(c("snap.smc_premium_discount"))
    discount = _clip01(1.0 - premium)
    fib_position = _clip01(
        0.46 * c("ctx_cont.retracement_from_last_impulse")
        + 0.34 * premium
        + 0.20 * c("ctx_cont.d1_close_pct_in_20day_range_canon_v2")
    )
    fib_golden = np.maximum.reduce(
        [
            _fib_proximity(fib_position, 0.500),
            _fib_proximity(fib_position, 0.618),
            _clip01(c("chart.geometry_fib_golden_zone_proximity")),
        ]
    ).astype(np.float32)
    fib_sr_long = _clip01(fib_golden * (0.50 * support_stack + 0.26 * discount + 0.24 * long_trend_support))
    fib_sr_short = _clip01(fib_golden * (0.50 * resistance_stack + 0.26 * premium + 0.24 * short_trend_support))

    sweep_low_reclaim = _clip01(c("chart.foundation_sweep_low_reclaim_up_proxy") / 5.0)
    sweep_high_reclaim = _clip01(c("chart.foundation_sweep_high_reclaim_down_proxy") / 5.0)
    false_low = _clip01(c("chart.foundation_false_breakout_low_followthrough_up_proxy") / 5.0)
    false_high = _clip01(c("chart.foundation_false_breakout_high_followthrough_down_proxy") / 5.0)
    sweep_recent = _clip01(c("ctx_cont.smc_sweep_recency_tau24") + _recency(c("snap.smc_bars_since_sweep")))
    sweep_size = _clip01(c("snap.smc_sweep_size_atr") + c("ctx_cont.smc_sweep_size_recent_tau12"))
    sweep_down_context = _clip01(c("snap.smc_sweep_down") * (0.55 + 0.25 * sweep_recent + 0.20 * sweep_size))
    sweep_up_context = _clip01(c("snap.smc_sweep_up") * (0.55 + 0.25 * sweep_recent + 0.20 * sweep_size))
    sweep_reclaim_long = _clip01(
        0.33 * sweep_low_reclaim
        + 0.22 * sweep_down_context
        + 0.18 * support_stack
        + 0.12 * discount
        + 0.15 * c("chart.smc_liquidity_sweep_reclaim_strength_long")
    )
    sweep_reclaim_short = _clip01(
        0.33 * sweep_high_reclaim
        + 0.22 * sweep_up_context
        + 0.18 * resistance_stack
        + 0.12 * premium
        + 0.15 * c("chart.smc_liquidity_sweep_reclaim_strength_short")
    )
    false_breakout_long = _clip01(
        (0.42 * false_low + 0.25 * sweep_reclaim_long + 0.20 * support_stack + 0.13 * long_trend_support)
        + 0.12 * c("chart.smc_liquidity_false_breakout_quality_long")
    )
    false_breakout_short = _clip01(
        (0.42 * false_high + 0.25 * sweep_reclaim_short + 0.20 * resistance_stack + 0.13 * short_trend_support)
        + 0.12 * c("chart.smc_liquidity_false_breakout_quality_short")
    )
    liquidity_conflict = _clip01(
        0.36 * np.minimum(sweep_reclaim_long, sweep_reclaim_short) * 2.0
        + 0.28 * np.minimum(support_stack, resistance_stack) * 2.0
        + 0.20 * np.abs(c("chart.foundation_sweep_reclaim_balance_proxy")) / 5.0
        + 0.16 * np.minimum(fib_sr_long, fib_sr_short) * 2.0
    )
    premium_discount_alignment = _clip(
        discount * (0.55 * sweep_reclaim_long + 0.45 * fib_sr_long)
        - premium * (0.55 * sweep_reclaim_short + 0.45 * fib_sr_short),
        -1.0,
        1.0,
    )

    spread_ratio = _clip(_safe_ratio(c("ctx_cont.spread_bps"), c("ctx_cont.atr_bps")), 0.0, 5.0)
    spread_pressure = _clip01(np.maximum(_tanh(spread_ratio, scale=0.15), c("session_regime.spread_cost_pressure")))
    spread_bucket_high = (np.rint(c("ctx_cat.spread_bucket")) >= 2).astype(np.float32)
    vol_regime_high = (np.rint(c("ctx_cat.vol_regime_id")) >= 2).astype(np.float32)
    vol_pressure = _clip01(
        0.35 * c("ctx_cont.vol_pct_m5_1yr")
        + 0.30 * c("ctx_cont.vol_pct_h1_1yr")
        + 0.20 * c("ctx_cont.D1_atr_percentile_252")
        + 0.15 * vol_regime_high
    )
    spread_vol_abstain = _clip01(0.46 * spread_pressure + 0.24 * spread_bucket_high + 0.30 * vol_pressure)
    session_change = _clip01(c("ctx_cont.session_change_flag"))
    opening_risk = _clip01(np.exp(-_clip(c("ctx_cont.minutes_since_session_open"), 0.0, 1440.0) / 30.0) + session_change)
    boundary_risk = _clip01(
        np.exp(-_clip(c("ctx_cont.minutes_to_next_session_boundary"), 0.0, 1440.0) / 30.0)
        + session_change
    )
    asia = _clip01(c("ctx_cont.is_ASIA"))
    asia_eu = _clip01(c("ctx_cont.is_asia_eu_overlap"))
    eu_us = _clip01(c("ctx_cont.is_eu_us_overlap"))
    eu = _clip01(c("ctx_cont.is_eu_only") + asia_eu + eu_us)
    us = _clip01(c("ctx_cont.is_us_only") + eu_us)
    active_session = _clip01(asia + eu + us)
    session_permission = _clip01(
        active_session
        * _clip01(c("ctx_cont.session_tradable"))
        * (1.0 - 0.44 * spread_pressure)
        * (1.0 - 0.28 * np.maximum(opening_risk, boundary_risk))
    )
    d1_regime_changed = _clip01(c("ctx_cont.d1_regime_changed_flag_v3"))
    bars_since_change = _clip01(c("ctx_cont.bars_since_d1_regime_change_v3"))
    regime_stack = _tanh(c("ctx_cont.regime_stack_sum_v3"), scale=3.0)
    regime_agreement = _clip01(
        0.31 * regime_agreement_raw
        + 0.22 * trend_agreement
        + 0.18 * structure_agreement
        + 0.14 * c("session_regime.regime_persistence_score")
        + 0.10 * (1.0 - d1_regime_changed)
        + 0.05 * np.abs(regime_stack)
    )
    regime_conflict = _clip01(
        0.26 * regime_divergence_raw
        + 0.23 * trend_conflict
        + 0.21 * structure_conflict
        + 0.12 * d1_regime_changed
        + 0.08 * (1.0 - bars_since_change)
        + 0.10 * c("session_regime.mtf_regime_divergence_pressure")
    )

    long_raw = _clip01(
        0.20 * long_trend_support
        + 0.19 * bos_alignment_up
        + 0.17 * sweep_reclaim_long
        + 0.13 * false_breakout_long
        + 0.14 * fib_sr_long
        + 0.10 * regime_agreement
        + 0.07 * session_permission
    )
    short_raw = _clip01(
        0.20 * short_trend_support
        + 0.19 * bos_alignment_down
        + 0.17 * sweep_reclaim_short
        + 0.13 * false_breakout_short
        + 0.14 * fib_sr_short
        + 0.10 * regime_agreement
        + 0.07 * session_permission
    )
    directional_opposition = _clip01(np.minimum(long_raw, short_raw) * 2.0)
    conflict_score = _clip01(
        0.22 * trend_conflict
        + 0.20 * structure_conflict
        + 0.20 * liquidity_conflict
        + 0.18 * regime_conflict
        + 0.12 * directional_opposition
        + 0.08 * pullback_abstain
    )
    long_agreement = _clip01(long_raw * (1.0 - 0.36 * conflict_score) * (1.0 - 0.20 * spread_vol_abstain))
    short_agreement = _clip01(short_raw * (1.0 - 0.36 * conflict_score) * (1.0 - 0.20 * spread_vol_abstain))
    session_regime_long = _clip01(long_agreement * session_permission * (1.0 - 0.28 * spread_vol_abstain))
    session_regime_short = _clip01(short_agreement * session_permission * (1.0 - 0.28 * spread_vol_abstain))
    abstain_score = _clip01(
        0.34 * conflict_score
        + 0.22 * (1.0 - np.maximum(long_raw, short_raw))
        + 0.20 * (1.0 - session_permission)
        + 0.16 * spread_vol_abstain
        + 0.08 * pullback_abstain
    )

    arrays: list[np.ndarray] = []
    names: list[str] = []
    _add(arrays, names, "trend.mtf_confluence_trend_direction_score", trend_direction, lo=-1.0, hi=1.0)
    _add(arrays, names, "trend.mtf_confluence_trend_tf_agreement", trend_agreement, lo=0.0, hi=1.0)
    _add(arrays, names, "trend.mtf_confluence_trend_tf_conflict", trend_conflict, lo=0.0, hi=1.0)
    _add(arrays, names, "trend.mtf_confluence_trend_m5_m15_h1_h4_d1_alignment", trend_alignment, lo=-1.0, hi=1.0)
    _add(arrays, names, "trend.mtf_confluence_long_trend_bias", long_trend_support, lo=0.0, hi=1.0)
    _add(arrays, names, "trend.mtf_confluence_short_trend_bias", short_trend_support, lo=0.0, hi=1.0)
    _add(arrays, names, "chart.structure_swing_mtf_confluence_structure_direction_score", structure_direction, lo=-1.0, hi=1.0)
    _add(arrays, names, "chart.structure_swing_mtf_confluence_bos_alignment_up", bos_alignment_up, lo=0.0, hi=1.0)
    _add(arrays, names, "chart.structure_swing_mtf_confluence_bos_alignment_down", bos_alignment_down, lo=0.0, hi=1.0)
    _add(arrays, names, "chart.structure_swing_mtf_confluence_structure_conflict", structure_conflict, lo=0.0, hi=1.0)
    _add(arrays, names, "chart.structure_swing_mtf_confluence_pullback_abstain_pressure", pullback_abstain, lo=0.0, hi=1.0)
    _add(arrays, names, "chart.smc_liquidity_mtf_confluence_sweep_reclaim_long", sweep_reclaim_long, lo=0.0, hi=1.0)
    _add(arrays, names, "chart.smc_liquidity_mtf_confluence_sweep_reclaim_short", sweep_reclaim_short, lo=0.0, hi=1.0)
    _add(arrays, names, "chart.smc_liquidity_mtf_confluence_false_breakout_long", false_breakout_long, lo=0.0, hi=1.0)
    _add(arrays, names, "chart.smc_liquidity_mtf_confluence_false_breakout_short", false_breakout_short, lo=0.0, hi=1.0)
    _add(arrays, names, "chart.smc_liquidity_mtf_confluence_liquidity_conflict", liquidity_conflict, lo=0.0, hi=1.0)
    _add(arrays, names, "chart.smc_liquidity_mtf_confluence_premium_discount_alignment", premium_discount_alignment, lo=-1.0, hi=1.0)
    _add(arrays, names, "chart.geometry_mtf_confluence_fib_sr_long_proximity", fib_sr_long, lo=0.0, hi=1.0)
    _add(arrays, names, "chart.geometry_mtf_confluence_fib_sr_short_proximity", fib_sr_short, lo=0.0, hi=1.0)
    _add(arrays, names, "chart.geometry_mtf_confluence_sr_balance", sr_balance, lo=-1.0, hi=1.0)
    _add(arrays, names, "chart.geometry_mtf_confluence_major_level_density", level_density, lo=0.0, hi=1.0)
    _add(arrays, names, "session_regime.mtf_confluence_session_permission", session_permission, lo=0.0, hi=1.0)
    _add(arrays, names, "session_regime.mtf_confluence_regime_agreement", regime_agreement, lo=0.0, hi=1.0)
    _add(arrays, names, "session_regime.mtf_confluence_regime_conflict", regime_conflict, lo=0.0, hi=1.0)
    _add(arrays, names, "session_regime.mtf_confluence_spread_vol_abstain", spread_vol_abstain, lo=0.0, hi=1.0)
    _add(arrays, names, "session_regime.mtf_confluence_session_regime_tradable_long", session_regime_long, lo=0.0, hi=1.0)
    _add(arrays, names, "session_regime.mtf_confluence_session_regime_tradable_short", session_regime_short, lo=0.0, hi=1.0)
    _add(arrays, names, "session_regime.mtf_confluence_long_agreement_score", long_agreement, lo=0.0, hi=1.0)
    _add(arrays, names, "session_regime.mtf_confluence_short_agreement_score", short_agreement, lo=0.0, hi=1.0)
    _add(arrays, names, "session_regime.mtf_confluence_direction_balance", long_agreement - short_agreement, lo=-1.0, hi=1.0)
    _add(arrays, names, "session_regime.mtf_confluence_conflict_score", conflict_score, lo=0.0, hi=1.0)
    _add(arrays, names, "session_regime.mtf_confluence_abstain_score", abstain_score, lo=0.0, hi=1.0)

    out = np.column_stack(arrays).astype(np.float32, copy=False) if arrays else np.empty((x.shape[0], 0), dtype=np.float32)
    if tuple(names) != MTF_CONFLUENCE_FEATURE_NAMES:
        raise RuntimeError("MTF confluence feature order drifted")
    if not np.isfinite(out).all():
        raise RuntimeError("MTF confluence layer contains non-finite values")
    return out, names
