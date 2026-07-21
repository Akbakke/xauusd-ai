"""Canonical Entry chart-geometry features.

This layer is a causal numeric proxy for what a discretionary chart trader
draws by hand: trend lines, support/resistance zones, Fibonacci pullback zones,
EMA cross pressure, compression breakouts, and simple continuation/reversal
patterns. It uses only already-materialized snap/ctx_cont inputs.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


CHART_GEOMETRY_FEATURE_VERSION = (
    "entry_chart_geometry_v1_20260717_numeric_lines_fib_patterns_failclosed"
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
    "snap.smc_premium_discount",
    "ctx_cont.retracement_from_last_impulse",
    "ctx_cont.d1_close_pct_in_20day_range_canon_v2",
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
    "snap.smc_sweep_size_atr",
    "ctx_cont.smc_sweep_size_recent_tau12",
    "ctx_cont.smc_sweep_recency_tau24",
    "ctx_cont.wick_ratio",
    "snap.wick_asym",
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


def _fib_proximity(position: np.ndarray, level: float) -> np.ndarray:
    return np.exp(-np.abs(_clip01(position) - float(level)) * 12.0).astype(np.float32)


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
    h8_proxy = _clip(0.45 * h1_ema + 0.40 * h4_ema + 0.15 * d1_slope)
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
    _add(arrays, names, "mtf_trend_score", mtf_trend, lo=-2.0, hi=2.0)
    _add(arrays, names, "h8_proxy_trend_score", h8_proxy, lo=-2.0, hi=2.0)
    _add(arrays, names, "mtf_trend_agreement_pressure", agreement_pressure, lo=0.0, hi=1.0)
    _add(arrays, names, "mtf_trend_divergence_pressure", divergence_pressure, lo=0.0, hi=1.0)
    _add(arrays, names, "ema_stack_bull_pressure", trend_up * agreement_pressure, lo=0.0, hi=2.0)
    _add(arrays, names, "ema_stack_bear_pressure", trend_down * agreement_pressure, lo=0.0, hi=2.0)
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
            _prox_abs(c("ctx_cont.dist_to_S1_atr")),
            _prox_abs(c("ctx_cont.dist_to_S2_atr")),
            _prox_abs(c("ctx_cont.dist_to_h1_lo_atr")),
            _prox_abs(c("ctx_cont.dist_to_h4_lo_atr")),
            _prox_abs(c("ctx_cont.dist_to_d1_lo_atr")),
            swing_low_line,
        ]
    )
    resistance_sources = np.vstack(
        [
            _clip01(c("ctx_cont.sr_resistance_proximity_exp")),
            _prox_abs(c("ctx_cont.dist_to_R1_atr")),
            _prox_abs(c("ctx_cont.dist_to_R2_atr")),
            _prox_abs(c("ctx_cont.dist_to_h1_hi_atr")),
            _prox_abs(c("ctx_cont.dist_to_h4_hi_atr")),
            _prox_abs(c("ctx_cont.dist_to_d1_hi_atr")),
            swing_high_line,
        ]
    )
    support_stack = support_sources.max(axis=0).astype(np.float32)
    resistance_stack = resistance_sources.max(axis=0).astype(np.float32)
    level_mean = ((support_sources.mean(axis=0) + resistance_sources.mean(axis=0)) * 0.5).astype(np.float32)
    level_max = np.maximum(support_stack, resistance_stack).astype(np.float32)
    support_minus_resistance = _clip(c("ctx_cont.sr_support_minus_resistance_prox") + support_stack - resistance_stack, -2.0, 2.0)
    # Low-to-high channel position: support/lower rail -> 0, resistance/upper rail -> 1.
    # The stack balance is support-minus-resistance, so it must be inverted here.
    channel_position = _clip01(0.5 - 0.5 * support_minus_resistance)
    channel_center = _clip01(1.0 - 2.0 * np.abs(channel_position - 0.5))
    channel_edge = _clip01(level_max * (1.0 - channel_center))
    sr_balance = _clip01(1.0 - np.abs(support_stack - resistance_stack))
    channel_confluence = _clip01(
        (0.50 * level_max + 0.30 * level_mean + 0.20 * sr_balance)
        * (0.65 + 0.35 * agreement_pressure)
        * (1.0 - 0.25 * divergence_pressure)
    )
    channel_edge_retest = _clip01(channel_edge * (0.50 + sr_balance) * (0.50 + agreement_pressure))
    _add(arrays, names, "support_line_proximity_stack", support_stack, lo=0.0, hi=1.0)
    _add(arrays, names, "resistance_line_proximity_stack", resistance_stack, lo=0.0, hi=1.0)
    _add(arrays, names, "support_minus_resistance_stack", support_minus_resistance, lo=-2.0, hi=2.0)
    _add(arrays, names, "major_level_proximity_max", level_max, lo=0.0, hi=1.0)
    _add(arrays, names, "major_level_proximity_mean", level_mean, lo=0.0, hi=1.0)
    _add(arrays, names, "channel_position_low_to_high", channel_position, lo=0.0, hi=1.0)
    _add(arrays, names, "channel_center_bias", channel_center, lo=0.0, hi=1.0)
    _add(arrays, names, "channel_edge_pressure", channel_edge, lo=0.0, hi=1.0)

    h1_compression = _clip01(c("ctx_cont.H1_range_compression_ratio"))
    m15_compression = _clip01(c("ctx_cont.M15_range_compression_ratio"))
    squeeze = _clip01(c("snap._v1_bb_squeeze_20_2"))
    compression = _clip01(0.40 * h1_compression + 0.35 * m15_compression + 0.25 * squeeze)
    vol_impulse = _clip01(
        0.34 * _pos(_tanh(c("snap.atr_z"), scale=2.0))
        + 0.33 * _pos(_tanh(c("snap.rvol_20"), scale=2.0))
        + 0.33 * _pos(_tanh(c("snap.vol_ratio_5_20"), scale=2.0))
    )
    release = _clip01(compression * _pos(_delta(vol_impulse)))
    trend_age = _clip01(
        0.5 * c("ctx_cont.h1_trend_age_bars_norm_v2") + 0.5 * c("ctx_cont.h4_trend_age_bars_norm_v2")
    )
    d1_atr_pct = _clip01(c("ctx_cont.D1_atr_percentile_252"))
    compression_apex = _clip01(compression * sr_balance * level_mean)

    bos_up = _clip01(c("snap.smc_bos_up") + 0.5 * _pos(c("ctx_cont.smc_bos_pressure_last12")) + 0.25 * _pos(c("ctx_cont.smc_bos_pressure_last48")))
    bos_down = _clip01(c("snap.smc_bos_down") + 0.5 * _neg(c("ctx_cont.smc_bos_pressure_last12")) + 0.25 * _neg(c("ctx_cont.smc_bos_pressure_last48")))
    choch = _clip01(c("snap.smc_choch") + c("ctx_cont.smc_choch_recent_tau12") + 0.5 * c("ctx_cont.smc_choch_recent_tau24"))
    sweep_up = _clip01(c("snap.smc_sweep_up") + 0.5 * _neg(c("ctx_cont.smc_sweep_bull_pressure_last12")) + 0.25 * _neg(c("ctx_cont.smc_sweep_bull_pressure_last48")))
    sweep_down = _clip01(c("snap.smc_sweep_down") + 0.5 * _pos(c("ctx_cont.smc_sweep_bull_pressure_last12")) + 0.25 * _pos(c("ctx_cont.smc_sweep_bull_pressure_last48")))
    sweep_recent = _clip01(c("ctx_cont.smc_sweep_recency_tau24"))
    sweep_size = _clip01(c("snap.smc_sweep_size_atr") + c("ctx_cont.smc_sweep_size_recent_tau12"))
    wick_level = _clip01(c("ctx_cont.wick_ratio") + np.abs(c("snap.wick_asym")))
    breakout_up_seed = _clip01(
        0.50 * bos_up + 0.20 * ema_bull_confirmation + 0.20 * release + 0.10 * _pos(regime_stack)
    )
    breakout_down_seed = _clip01(
        0.50 * bos_down + 0.20 * ema_bear_confirmation + 0.20 * release + 0.10 * _neg(regime_stack)
    )
    _add(
        arrays,
        names,
        "support_bounce_long_pressure",
        support_stack * trend_up * (1.0 + sweep_down + wick_level + 0.5 * channel_confluence),
        lo=0.0,
        hi=5.0,
    )
    _add(
        arrays,
        names,
        "resistance_reject_short_pressure",
        resistance_stack * trend_down * (1.0 + sweep_up + wick_level + 0.5 * channel_confluence),
        lo=0.0,
        hi=5.0,
    )
    _add(
        arrays,
        names,
        "trendline_break_up_pressure",
        resistance_stack * breakout_up_seed * (1.0 + trend_up + agreement_pressure + 0.5 * compression),
        lo=0.0,
        hi=5.0,
    )
    _add(
        arrays,
        names,
        "trendline_break_down_pressure",
        support_stack * breakout_down_seed * (1.0 + trend_down + agreement_pressure + 0.5 * compression),
        lo=0.0,
        hi=5.0,
    )
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

    premium = _clip01(c("snap.smc_premium_discount"))
    retracement = _clip01(c("ctx_cont.retracement_from_last_impulse"))
    d1_loc = _clip01(c("ctx_cont.d1_close_pct_in_20day_range_canon_v2"))
    fib_position = _clip01(0.55 * retracement + 0.30 * premium + 0.15 * d1_loc)
    fib236 = _fib_proximity(fib_position, 0.236)
    fib382 = _fib_proximity(fib_position, 0.382)
    fib500 = _fib_proximity(fib_position, 0.500)
    fib618 = _fib_proximity(fib_position, 0.618)
    fib786 = _fib_proximity(fib_position, 0.786)
    golden = np.maximum(fib500, fib618).astype(np.float32)
    extension_up = _clip01(_pos(fib_position - 0.786) * 5.0)
    extension_down = _clip01(_pos(0.236 - fib_position) * 5.0)
    fib_support_confluence = _clip01(
        golden * support_stack * (0.50 + channel_confluence) * (0.50 + agreement_pressure)
    )
    fib_resistance_confluence = _clip01(
        golden * resistance_stack * (0.50 + channel_confluence) * (0.50 + agreement_pressure)
    )
    _add(arrays, names, "fib_position_proxy", fib_position, lo=0.0, hi=1.0)
    _add(arrays, names, "fib_retracement_236_proximity", fib236, lo=0.0, hi=1.0)
    _add(arrays, names, "fib_retracement_382_proximity", fib382, lo=0.0, hi=1.0)
    _add(arrays, names, "fib_retracement_500_proximity", fib500, lo=0.0, hi=1.0)
    _add(arrays, names, "fib_retracement_618_proximity", fib618, lo=0.0, hi=1.0)
    _add(arrays, names, "fib_retracement_786_proximity", fib786, lo=0.0, hi=1.0)
    _add(arrays, names, "fib_golden_zone_proximity", golden, lo=0.0, hi=1.0)
    _add(
        arrays,
        names,
        "fib_pullback_long_pressure",
        trend_up
        * golden
        * (0.50 + retracement)
        * (0.50 + support_stack)
        * (0.75 + ema_bull_follow),
        lo=0.0,
        hi=3.0,
    )
    _add(
        arrays,
        names,
        "fib_pullback_short_pressure",
        trend_down
        * golden
        * (0.50 + retracement)
        * (0.50 + resistance_stack)
        * (0.75 + ema_bear_follow),
        lo=0.0,
        hi=3.0,
    )
    _add(
        arrays,
        names,
        "fib_extension_breakout_up_pressure",
        extension_up * trend_up * breakout_up_seed * (0.75 + resistance_stack),
        lo=0.0,
        hi=3.0,
    )
    _add(
        arrays,
        names,
        "fib_extension_breakout_down_pressure",
        extension_down * trend_down * breakout_down_seed * (0.75 + support_stack),
        lo=0.0,
        hi=3.0,
    )

    _add(
        arrays,
        names,
        "ascending_triangle_pressure",
        support_stack
        * resistance_stack
        * compression
        * trend_up
        * (0.75 + _pos(_delta(support_stack)) + 0.25 * ema_bull_follow),
        lo=0.0,
        hi=3.0,
    )
    _add(
        arrays,
        names,
        "descending_triangle_pressure",
        support_stack
        * resistance_stack
        * compression
        * trend_down
        * (0.75 + _pos(_delta(resistance_stack)) + 0.25 * ema_bear_follow),
        lo=0.0,
        hi=3.0,
    )
    _add(
        arrays,
        names,
        "bull_flag_pullback_pressure",
        trend_up * golden * compression * channel_center * (0.75 + support_stack + 0.25 * ema_bull_follow),
        lo=0.0,
        hi=3.0,
    )
    _add(
        arrays,
        names,
        "bear_flag_pullback_pressure",
        trend_down * golden * compression * channel_center * (0.75 + resistance_stack + 0.25 * ema_bear_follow),
        lo=0.0,
        hi=3.0,
    )
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
    _add(arrays, names, "line_pattern_tail_risk", (sweep_size + wick_level + divergence_pressure + d1_atr_pct) * channel_edge, lo=0.0, hi=5.0)

    prior_breakout_up = _lag1(_clip01(breakout_up_seed + extension_up * trend_up + release))
    prior_breakout_down = _lag1(_clip01(breakout_down_seed + extension_down * trend_down + release))
    mtf_breakout_up_quality = _clip01(
        resistance_stack * breakout_up_seed * (0.50 + trend_up) * (0.50 + agreement_pressure + release)
    )
    mtf_breakout_down_quality = _clip01(
        support_stack * breakout_down_seed * (0.50 + trend_down) * (0.50 + agreement_pressure + release)
    )
    mtf_retest_long_quality = _clip01(
        support_stack * prior_breakout_up * trend_up * (0.50 + agreement_pressure) * (0.50 + compression + golden)
    )
    mtf_retest_short_quality = _clip01(
        resistance_stack * prior_breakout_down * trend_down * (0.50 + agreement_pressure) * (0.50 + compression + golden)
    )
    fib_extension_exhaustion = _clip01(
        (extension_up * trend_up * resistance_stack + extension_down * trend_down * support_stack)
        * (0.50 + d1_atr_pct + divergence_pressure)
    )
    rising_support_rail_touch = _clip01(
        support_stack
        * trend_up
        * (0.50 + channel_confluence)
        * (0.50 + agreement_pressure)
        * (1.0 - 0.35 * divergence_pressure)
    )
    falling_resistance_rail_touch = _clip01(
        resistance_stack
        * trend_down
        * (0.50 + channel_confluence)
        * (0.50 + agreement_pressure)
        * (1.0 - 0.35 * divergence_pressure)
    )
    _add(
        arrays,
        names,
        "trendline_channel_confluence_pressure",
        channel_confluence * (0.50 + compression) * (0.50 + agreement_pressure),
        lo=0.0,
        hi=3.0,
    )
    _add(
        arrays,
        names,
        "channel_edge_rejection_pressure",
        channel_edge_retest * (sweep_up + sweep_down + wick_level) * (0.50 + divergence_pressure + choch),
        lo=0.0,
        hi=3.0,
    )
    _add(
        arrays,
        names,
        "rising_support_rail_long_pressure",
        rising_support_rail_touch
        * (0.75 + ema_bull_follow + fib_support_confluence + mtf_retest_long_quality)
        * (1.0 - 0.35 * breakout_down_seed),
        lo=0.0,
        hi=3.0,
    )
    _add(
        arrays,
        names,
        "rising_support_rail_short_trap_pressure",
        rising_support_rail_touch
        * (0.75 + ema_bull_confirmation + support_stack + fib_support_confluence)
        * (1.0 - 0.50 * breakout_down_seed),
        lo=0.0,
        hi=3.0,
    )
    _add(
        arrays,
        names,
        "falling_resistance_rail_short_pressure",
        falling_resistance_rail_touch
        * (0.75 + ema_bear_follow + fib_resistance_confluence + mtf_retest_short_quality)
        * (1.0 - 0.35 * breakout_up_seed),
        lo=0.0,
        hi=3.0,
    )
    _add(
        arrays,
        names,
        "falling_resistance_rail_long_trap_pressure",
        falling_resistance_rail_touch
        * (0.75 + ema_bear_confirmation + resistance_stack + fib_resistance_confluence)
        * (1.0 - 0.50 * breakout_up_seed),
        lo=0.0,
        hi=3.0,
    )
    _add(
        arrays,
        names,
        "fib_support_confluence_long_pressure",
        fib_support_confluence * trend_up * (0.75 + ema_bull_follow),
        lo=0.0,
        hi=3.0,
    )
    _add(
        arrays,
        names,
        "fib_resistance_confluence_short_pressure",
        fib_resistance_confluence * trend_down * (0.75 + ema_bear_follow),
        lo=0.0,
        hi=3.0,
    )
    _add(arrays, names, "fib_extension_exhaustion_risk", fib_extension_exhaustion, lo=0.0, hi=1.0)
    _add(arrays, names, "ema_cross_mtf_bull_confirmation", ema_bull_confirmation, lo=0.0, hi=1.0)
    _add(arrays, names, "ema_cross_mtf_bear_confirmation", ema_bear_confirmation, lo=0.0, hi=1.0)
    _add(arrays, names, "triangle_apex_compression_pressure", compression_apex, lo=0.0, hi=1.0)
    _add(
        arrays,
        names,
        "flag_breakout_readiness_pressure",
        golden
        * compression
        * channel_confluence
        * (0.50 + np.maximum(trend_up, trend_down))
        * (0.50 + release + 0.25 * vol_impulse),
        lo=0.0,
        hi=3.0,
    )
    _add(arrays, names, "mtf_channel_breakout_up_quality", mtf_breakout_up_quality, lo=0.0, hi=1.0)
    _add(arrays, names, "mtf_channel_breakout_down_quality", mtf_breakout_down_quality, lo=0.0, hi=1.0)
    _add(arrays, names, "mtf_channel_retest_long_quality", mtf_retest_long_quality, lo=0.0, hi=1.0)
    _add(arrays, names, "mtf_channel_retest_short_quality", mtf_retest_short_quality, lo=0.0, hi=1.0)

    out = np.column_stack(arrays).astype(np.float32, copy=False) if arrays else np.empty((x.shape[0], 0), dtype=np.float32)
    if not np.isfinite(out).all():
        raise RuntimeError("chart geometry layer contains non-finite values")
    if len(set(names)) != len(names):
        dupes = sorted({name for name in names if names.count(name) > 1})
        raise RuntimeError(f"chart geometry layer has duplicate names: {dupes[:10]}")
    return out, names


CHART_GEOMETRY_FEATURE_NAMES = tuple(
    name
    for name in build_entry_chart_geometry_layer(
        np.zeros((1, len(CHART_GEOMETRY_SOURCE_FIELDS)), dtype=np.float32),
        list(CHART_GEOMETRY_SOURCE_FIELDS),
    )[1]
)

CHART_GEOMETRY_SMART2_FEATURE_NAMES = CHART_GEOMETRY_FEATURE_NAMES[41:]

# The model-native geometry surface retains this exact 13-field causal
# rail/trap subset. Its name and ownership are independent of retired tensor
# widths; the complete seq513 identity is bound by the signal manifest.
CHART_GEOMETRY_MODEL_NATIVE_FEATURE_NAMES = (
    f"{CHART_GEOMETRY_FEATURE_PREFIX}trendline_channel_confluence_pressure",
    f"{CHART_GEOMETRY_FEATURE_PREFIX}channel_edge_rejection_pressure",
    f"{CHART_GEOMETRY_FEATURE_PREFIX}rising_support_rail_long_pressure",
    f"{CHART_GEOMETRY_FEATURE_PREFIX}rising_support_rail_short_trap_pressure",
    f"{CHART_GEOMETRY_FEATURE_PREFIX}falling_resistance_rail_short_pressure",
    f"{CHART_GEOMETRY_FEATURE_PREFIX}falling_resistance_rail_long_trap_pressure",
    f"{CHART_GEOMETRY_FEATURE_PREFIX}fib_support_confluence_long_pressure",
    f"{CHART_GEOMETRY_FEATURE_PREFIX}fib_resistance_confluence_short_pressure",
    f"{CHART_GEOMETRY_FEATURE_PREFIX}fib_extension_exhaustion_risk",
    f"{CHART_GEOMETRY_FEATURE_PREFIX}mtf_channel_breakout_up_quality",
    f"{CHART_GEOMETRY_FEATURE_PREFIX}mtf_channel_breakout_down_quality",
    f"{CHART_GEOMETRY_FEATURE_PREFIX}mtf_channel_retest_long_quality",
    f"{CHART_GEOMETRY_FEATURE_PREFIX}mtf_channel_retest_short_quality",
)
