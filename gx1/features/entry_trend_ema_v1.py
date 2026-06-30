"""Entry trend/EMA specialist extension features.

This layer is deliberately pure and report-only until a later dataset rebuild
chooses to include it. It derives causal trend/EMA mechanism features from the
already materialized Entry snap/context fields.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


TREND_EMA_FEATURE_VERSION = "entry_trend_ema_v1_20260630_mtf_stack_cross_age_pressure_failclosed"
TREND_EMA_FEATURE_PREFIX = "trend.ema_"

TREND_EMA_SOURCE_FIELDS = (
    "snap._v1_ema_diff",
    "snap.ema20_slope",
    "snap.pos_vs_ema200",
    "snap._v1_close_ema_slope_3",
    "snap._v1_kama_slope_30",
    "snap._v1_tema_slope_20",
    "ctx_cont.distance_ema_fast",
    "ctx_cont._v1h1_ema_diff",
    "ctx_cont._v1h1_slope3",
    "ctx_cont._v1h1_slope5",
    "ctx_cont._v1h4_ema_diff",
    "ctx_cont._v1h4_slope3",
    "ctx_cont._v1h4_slope5",
    "ctx_cont.d1_ema_slope_20_canon_v2",
    "ctx_cont.d1_pct_change_5_canon_v2",
    "ctx_cont.m15_trend_sign_canon_v2",
    "ctx_cont.m5_trend_age_bars_norm_v2",
    "ctx_cont.m15_trend_age_bars_norm_v2",
    "ctx_cont.h1_trend_age_bars_norm_v2",
    "ctx_cont.h4_trend_age_bars_norm_v2",
    "ctx_cont.d1_trend_age_bars_norm_v2",
    "ctx_cont.regime_tf_agreement_v3",
    "ctx_cont.regime_stack_sum_v3",
    "ctx_cont.regime_divergence_flag_v3",
    "ctx_cont.d1_dist_roc_288_v3",
    "ctx_cont.d1_trend_age_mature_flag_v3",
    "ctx_cont.retracement_from_last_impulse",
)


TREND_EMA_FEATURE_DESCRIPTIONS = {
    "mtf_score": "Signed M5/M15/H1/H4/D1 EMA-stack and trend-slope composite.",
    "stack_alignment_score": "Signed EMA-stack alignment weighted by slope agreement and regime stack.",
    "stack_bull_pressure": "Bullish EMA-stack pressure gated by MTF stack agreement and slope coherence.",
    "stack_bear_pressure": "Bearish EMA-stack pressure gated by MTF stack agreement and slope coherence.",
    "inflect_up_pressure": "Causal bullish EMA cross/turning pressure gated by current alignment context.",
    "inflect_down_pressure": "Causal bearish EMA cross/turning pressure gated by current alignment context.",
    "slope_score": "Signed slope composite across M5/H1/H4/D1 slope proxies.",
    "slope_curvature": "Causal one-bar change in the slope composite.",
    "slope_curvature_abs": "Absolute slope curvature pressure for trend acceleration/deceleration.",
    "mtf_agreement_pressure": "Agreement pressure across EMA-stack signs, slope coherence and existing regime agreement.",
    "mtf_divergence_pressure": "Divergence pressure across M5/H1/H4/D1, EMA-vs-slope direction and existing regime divergence.",
    "distance_fast_abs": "Absolute distance-to-fast-EMA proxy, squashed for stability.",
    "distance_stack_abs": "Mean absolute distance of M5/H1/H4/D1 EMA proxies.",
    "distance_stretch_pressure": "Directional extension from the EMA stack plus counter-trend stress.",
    "retrace_to_fast_long_pressure": "Uptrend pullback/retrace-to-fast-EMA pressure.",
    "retrace_to_fast_short_pressure": "Downtrend pullback/retrace-to-fast-EMA pressure.",
    "trend_age_mean": "Mean normalized trend age across M5/M15/H1/H4/D1.",
    "trend_age_exhaustion_pressure": "Mature trend, EMA extension and fading slope-support pressure.",
    "late_reversal_risk": "Mature trend, divergence and direction-aware adverse curvature turning pressure.",
    "d1_flip_pressure": "D1/MTF trend-age turning pressure from age, divergence and D1 distance change.",
}

TREND_EMA_FEATURE_SUFFIXES = (
    "mtf_score",
    "stack_alignment_score",
    "stack_bull_pressure",
    "stack_bear_pressure",
    "inflect_up_pressure",
    "inflect_down_pressure",
    "slope_score",
    "slope_curvature",
    "slope_curvature_abs",
    "mtf_agreement_pressure",
    "mtf_divergence_pressure",
    "distance_fast_abs",
    "distance_stack_abs",
    "distance_stretch_pressure",
    "retrace_to_fast_long_pressure",
    "retrace_to_fast_short_pressure",
    "trend_age_mean",
    "trend_age_exhaustion_pressure",
    "late_reversal_risk",
    "d1_flip_pressure",
)

TREND_EMA_FEATURE_NAMES = tuple(
    f"{TREND_EMA_FEATURE_PREFIX}{suffix}" for suffix in TREND_EMA_FEATURE_SUFFIXES
)


def _name_index(names: Iterable[str]) -> dict[str, int]:
    return {str(name): i for i, name in enumerate(names)}


def missing_trend_ema_source_fields(feature_names: Iterable[str]) -> list[str]:
    available = {str(name) for name in feature_names}
    return [name for name in TREND_EMA_SOURCE_FIELDS if name not in available]


def _col(x: np.ndarray, index: dict[str, int], name: str, default: float = 0.0) -> np.ndarray:
    if name not in index:
        return np.full(x.shape[0], float(default), dtype=np.float32)
    arr = np.asarray(x[:, index[name]], dtype=np.float32)
    return np.nan_to_num(arr, nan=float(default), posinf=float(default), neginf=float(default))


def _clip(arr: np.ndarray, lo: float = -25.0, hi: float = 25.0) -> np.ndarray:
    return np.clip(np.nan_to_num(arr, nan=0.0, posinf=hi, neginf=lo), lo, hi).astype(np.float32, copy=False)


def _clip01(arr: np.ndarray) -> np.ndarray:
    return _clip(arr, 0.0, 1.0)


def _pos(arr: np.ndarray) -> np.ndarray:
    return np.maximum(arr, 0.0).astype(np.float32, copy=False)


def _neg(arr: np.ndarray) -> np.ndarray:
    return np.maximum(-arr, 0.0).astype(np.float32, copy=False)


def _tanh(arr: np.ndarray, scale: float = 1.0) -> np.ndarray:
    return np.tanh(arr / max(float(scale), 1e-6)).astype(np.float32, copy=False)


def _prox_abs(arr: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.abs(arr))).astype(np.float32, copy=False)


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


def _mean_abs_deviation(parts: list[np.ndarray]) -> np.ndarray:
    stack = np.vstack(parts)
    center = stack.mean(axis=0)
    return np.abs(stack - center).mean(axis=0).astype(np.float32)


def _add(arrays: list[np.ndarray], names: list[str], name: str, arr: np.ndarray, *, lo: float = -25.0, hi: float = 25.0) -> None:
    clean = _clip(np.asarray(arr, dtype=np.float32), lo, hi)
    if clean.ndim != 1:
        raise RuntimeError(f"trend/EMA feature {name} is not 1D: {clean.shape}")
    if not np.isfinite(clean).all():
        raise RuntimeError(f"trend/EMA feature {name} contains non-finite values")
    arrays.append(clean)
    names.append(f"{TREND_EMA_FEATURE_PREFIX}{name}")


def build_entry_trend_ema_layer(
    x: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Build deterministic trend/EMA specialist features.

    The input rows must be in causal chronological order when cross/curvature
    pressure is used. The implementation only references the current row and
    `_lag1`, never future rows.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise RuntimeError(f"trend/EMA input matrix must be 2D, got {x.shape}")
    if x.shape[1] != len(feature_names):
        raise RuntimeError(
            f"trend/EMA feature name count {len(feature_names)} does not match matrix width {x.shape[1]}"
        )
    missing = missing_trend_ema_source_fields(feature_names)
    if missing:
        raise RuntimeError(f"trend/EMA required source fields missing: {missing}")

    idx = _name_index(feature_names)
    arrays: list[np.ndarray] = []
    names: list[str] = []

    def c(name: str, default: float = 0.0) -> np.ndarray:
        return _col(x, idx, name, default=default)

    m5_ema = _tanh(c("snap._v1_ema_diff"))
    m5_slope = _tanh(c("snap.ema20_slope"))
    m5_slope3 = _tanh(c("snap._v1_close_ema_slope_3"))
    m5_kama_slope = _tanh(c("snap._v1_kama_slope_30"))
    m5_tema_slope = _tanh(c("snap._v1_tema_slope_20"))
    m5_pos_ema200 = _tanh(c("snap.pos_vs_ema200"))
    fast_dist = _tanh(c("ctx_cont.distance_ema_fast"), scale=2.0)
    h1_ema = _tanh(c("ctx_cont._v1h1_ema_diff"))
    h1_slope3 = _tanh(c("ctx_cont._v1h1_slope3"))
    h1_slope5 = _tanh(c("ctx_cont._v1h1_slope5"))
    h4_ema = _tanh(c("ctx_cont._v1h4_ema_diff"))
    h4_slope3 = _tanh(c("ctx_cont._v1h4_slope3"))
    h4_slope5 = _tanh(c("ctx_cont._v1h4_slope5"))
    d1_slope = _tanh(c("ctx_cont.d1_ema_slope_20_canon_v2"))
    d1_change = _tanh(c("ctx_cont.d1_pct_change_5_canon_v2"))
    m15_trend = _tanh(c("ctx_cont.m15_trend_sign_canon_v2"))
    d1_dist_roc = _tanh(c("ctx_cont.d1_dist_roc_288_v3"))

    regime_agreement = _clip01(c("ctx_cont.regime_tf_agreement_v3"))
    regime_stack = _tanh(c("ctx_cont.regime_stack_sum_v3"), scale=3.0)
    regime_divergence = _clip01(c("ctx_cont.regime_divergence_flag_v3"))

    slope_stack = np.vstack(
        [
            m5_slope,
            m5_slope3,
            m5_kama_slope,
            m5_tema_slope,
            h1_slope3,
            h1_slope5,
            h4_slope3,
            h4_slope5,
            d1_slope,
            d1_change,
        ]
    )
    slope_bull_share = (slope_stack > 0.0).mean(axis=0).astype(np.float32)
    slope_bear_share = (slope_stack < 0.0).mean(axis=0).astype(np.float32)
    slope_direction = _clip(slope_bull_share - slope_bear_share, -1.0, 1.0)
    slope_coherence = np.maximum(slope_bull_share, slope_bear_share).astype(np.float32)
    slope_dispersion = _clip01(0.5 * _mean_abs_deviation(list(slope_stack)))

    slope_score = _clip(
        0.20 * m5_slope
        + 0.12 * m5_slope3
        + 0.08 * m5_kama_slope
        + 0.08 * m5_tema_slope
        + 0.14 * h1_slope3
        + 0.10 * h1_slope5
        + 0.14 * h4_slope3
        + 0.10 * h4_slope5
        + 0.04 * d1_slope,
        -2.0,
        2.0,
    )

    sign_stack = np.vstack([m5_ema, m5_pos_ema200, h1_ema, h4_ema, d1_slope, m15_trend])
    bull_share = (sign_stack > 0.0).mean(axis=0).astype(np.float32)
    bear_share = (sign_stack < 0.0).mean(axis=0).astype(np.float32)
    ema_direction = _clip(bull_share - bear_share, -1.0, 1.0)
    ema_stack_dispersion = _clip01(0.5 * _mean_abs_deviation(list(sign_stack)))
    stack_consensus = _clip01(
        0.45 * np.maximum(bull_share, bear_share)
        + 0.25 * (1.0 - ema_stack_dispersion)
        + 0.20 * regime_agreement
        + 0.10 * (1.0 - slope_dispersion)
    )
    ema_slope_coherence = _clip01(1.0 - 0.50 * np.abs(ema_direction - slope_direction))
    slope_agreement = _clip01(0.55 * slope_coherence + 0.25 * ema_slope_coherence + 0.20 * regime_agreement)
    tf_direction_consensus = _clip(
        (0.62 * ema_direction + 0.38 * slope_direction) * (0.55 + 0.45 * stack_consensus),
        -1.0,
        1.0,
    )

    mtf_score = _clip(
        0.14 * m5_ema
        + 0.10 * m5_pos_ema200
        + 0.10 * fast_dist
        + 0.18 * h1_ema
        + 0.18 * h4_ema
        + 0.10 * d1_slope
        + 0.06 * d1_change
        + 0.06 * m15_trend
        + 0.04 * slope_score
        + 0.04 * slope_direction,
        -2.0,
        2.0,
    )
    stack_alignment = _clip(
        tf_direction_consensus * (0.65 + 0.35 * slope_agreement) + 0.20 * regime_stack,
        -1.5,
        1.5,
    )
    alignment_coherence = _clip01(
        0.45 * stack_consensus
        + 0.25 * slope_agreement
        + 0.20 * regime_agreement
        + 0.10 * ema_slope_coherence
    )
    agreement_pressure = _clip01(alignment_coherence * (0.60 + 0.40 * ema_slope_coherence))
    divergence_pressure = _clip01(
        regime_divergence
        + np.abs(ema_direction - slope_direction) * 0.25
        + np.abs(m5_ema - h4_ema) * 0.20
        + np.abs(h1_ema - d1_slope) * 0.20
        + np.abs(m5_pos_ema200 - h1_ema) * 0.10
    )

    trend_up = _pos(mtf_score)
    trend_down = _neg(mtf_score)
    slope_delta = _delta(slope_score)
    mtf_delta = _delta(mtf_score)
    fast_delta = _delta(fast_dist)
    bull_context = _clip01(
        0.28 * bull_share
        + 0.22 * slope_bull_share
        + 0.18 * stack_consensus
        + 0.20 * regime_agreement
        + 0.07 * _pos(mtf_score)
        + 0.05 * ema_slope_coherence
    )
    bear_context = _clip01(
        0.28 * bear_share
        + 0.22 * slope_bear_share
        + 0.18 * stack_consensus
        + 0.20 * regime_agreement
        + 0.07 * _neg(mtf_score)
        + 0.05 * ema_slope_coherence
    )
    cross_quality = _clip01(0.75 + 0.25 * ema_slope_coherence - 0.25 * regime_divergence)
    m5_cross_base = _clip(0.55 * m5_ema + 0.25 * fast_dist + 0.20 * m5_slope, -2.0, 2.0)
    turn_up_impulse = _clip01(0.45 * _pos(mtf_delta) + 0.35 * _pos(slope_delta) + 0.20 * _pos(fast_delta))
    turn_down_impulse = _clip01(0.45 * _neg(mtf_delta) + 0.35 * _neg(slope_delta) + 0.20 * _neg(fast_delta))
    inflect_up = cross_quality * (
        _cross_up(mtf_score) * (0.55 + bull_context)
        + _cross_up(m5_cross_base) * (0.35 + bull_context)
        + _cross_up(slope_score) * (0.25 + 0.50 * bull_context)
        + turn_up_impulse * (0.20 + 0.55 * bull_context)
    )
    inflect_down = cross_quality * (
        _cross_down(mtf_score) * (0.55 + bear_context)
        + _cross_down(m5_cross_base) * (0.35 + bear_context)
        + _cross_down(slope_score) * (0.25 + 0.50 * bear_context)
        + turn_down_impulse * (0.20 + 0.55 * bear_context)
    )

    fast_distance_abs = np.abs(fast_dist).astype(np.float32)
    distance_stack_abs = np.vstack(
        [np.abs(m5_ema), np.abs(m5_pos_ema200), np.abs(h1_ema), np.abs(h4_ema), np.abs(d1_slope)]
    ).mean(axis=0).astype(np.float32)
    directional_extension = _clip01(
        np.abs(tf_direction_consensus)
        * (0.45 * distance_stack_abs + 0.25 * fast_distance_abs + 0.20 * np.abs(mtf_score) + 0.10 * np.abs(d1_dist_roc))
    )
    countertrend_stress = _clip01(
        0.45 * (trend_up * _neg(fast_dist) + trend_down * _pos(fast_dist))
        + 0.35 * divergence_pressure
        + 0.20 * (1.0 - ema_slope_coherence)
    )
    distance_stretch = _clip01(0.72 * directional_extension + 0.28 * countertrend_stress)
    near_fast_ema = _prox_abs(c("ctx_cont.distance_ema_fast"))
    retracement = _clip01(c("ctx_cont.retracement_from_last_impulse"))

    age_stack = np.vstack(
        [
            _clip01(c("ctx_cont.m5_trend_age_bars_norm_v2")),
            _clip01(c("ctx_cont.m15_trend_age_bars_norm_v2")),
            _clip01(c("ctx_cont.h1_trend_age_bars_norm_v2")),
            _clip01(c("ctx_cont.h4_trend_age_bars_norm_v2")),
            _clip01(c("ctx_cont.d1_trend_age_bars_norm_v2")),
        ]
    )
    trend_age_mean = age_stack.mean(axis=0).astype(np.float32)
    short_age = age_stack[:2].mean(axis=0).astype(np.float32)
    long_age = age_stack[2:].mean(axis=0).astype(np.float32)
    age_stack_pressure = _clip01(0.65 * trend_age_mean + 0.20 * long_age + 0.15 * np.maximum(short_age, long_age))
    mature_d1 = _clip01(c("ctx_cont.d1_trend_age_mature_flag_v3"))
    age_pressure = _clip01(0.62 * age_stack_pressure + 0.38 * mature_d1)
    trend_support = _clip01(
        0.34 * stack_consensus
        + 0.28 * slope_agreement
        + 0.20 * regime_agreement
        + 0.15 * ema_slope_coherence
        + 0.03 * (1.0 - ema_stack_dispersion)
    )
    slope_fade = _clip01(1.0 - trend_support + 0.35 * divergence_pressure)
    exhaustion = _clip01(
        age_pressure
        * (0.42 + 0.40 * distance_stretch + 0.18 * np.abs(mtf_score))
        * (0.75 + 0.25 * slope_fade)
    )
    adverse_curvature_up = _neg(slope_delta) + _neg(mtf_delta)
    adverse_curvature_down = _pos(slope_delta) + _pos(mtf_delta)
    adverse_curvature = np.where(mtf_score >= 0.0, adverse_curvature_up, adverse_curvature_down).astype(np.float32)
    late_reversal = _clip01(
        exhaustion
        * (0.40 + 0.45 * divergence_pressure + 0.15 * countertrend_stress)
        * (0.45 + adverse_curvature + 0.30 * slope_fade)
    )
    d1_turn_against_trend = np.where(mtf_score >= 0.0, _neg(d1_dist_roc), _pos(d1_dist_roc)).astype(np.float32)
    regime_flip = _clip01(
        (0.45 + 0.55 * age_pressure)
        * (
            0.35 * regime_divergence
            + 0.25 * d1_turn_against_trend
            + 0.20 * np.abs(ema_direction - slope_direction)
            + 0.20 * adverse_curvature
        )
    )

    _add(arrays, names, "mtf_score", mtf_score, lo=-2.0, hi=2.0)
    _add(arrays, names, "stack_alignment_score", stack_alignment, lo=-1.5, hi=1.5)
    _add(arrays, names, "stack_bull_pressure", trend_up * agreement_pressure, lo=0.0, hi=2.0)
    _add(arrays, names, "stack_bear_pressure", trend_down * agreement_pressure, lo=0.0, hi=2.0)
    _add(arrays, names, "inflect_up_pressure", inflect_up, lo=0.0, hi=2.0)
    _add(arrays, names, "inflect_down_pressure", inflect_down, lo=0.0, hi=2.0)
    _add(arrays, names, "slope_score", slope_score, lo=-2.0, hi=2.0)
    _add(arrays, names, "slope_curvature", slope_delta, lo=-2.0, hi=2.0)
    _add(arrays, names, "slope_curvature_abs", np.abs(slope_delta), lo=0.0, hi=2.0)
    _add(arrays, names, "mtf_agreement_pressure", agreement_pressure, lo=0.0, hi=1.0)
    _add(arrays, names, "mtf_divergence_pressure", divergence_pressure, lo=0.0, hi=1.0)
    _add(arrays, names, "distance_fast_abs", fast_distance_abs, lo=0.0, hi=1.0)
    _add(arrays, names, "distance_stack_abs", distance_stack_abs, lo=0.0, hi=1.0)
    _add(arrays, names, "distance_stretch_pressure", distance_stretch, lo=0.0, hi=1.0)
    _add(arrays, names, "retrace_to_fast_long_pressure", trend_up * near_fast_ema * (0.50 + retracement), lo=0.0, hi=3.0)
    _add(arrays, names, "retrace_to_fast_short_pressure", trend_down * near_fast_ema * (0.50 + retracement), lo=0.0, hi=3.0)
    _add(arrays, names, "trend_age_mean", trend_age_mean, lo=0.0, hi=1.0)
    _add(arrays, names, "trend_age_exhaustion_pressure", exhaustion, lo=0.0, hi=1.0)
    _add(arrays, names, "late_reversal_risk", late_reversal, lo=0.0, hi=1.0)
    _add(arrays, names, "d1_flip_pressure", regime_flip, lo=0.0, hi=1.0)

    out = np.column_stack(arrays).astype(np.float32, copy=False) if arrays else np.empty((x.shape[0], 0), dtype=np.float32)
    if not np.isfinite(out).all():
        raise RuntimeError("trend/EMA layer contains non-finite values")
    if len(set(names)) != len(names):
        dupes = sorted({name for name in names if names.count(name) > 1})
        raise RuntimeError(f"trend/EMA layer has duplicate names: {dupes[:10]}")
    if tuple(names) != TREND_EMA_FEATURE_NAMES:
        raise RuntimeError("trend/EMA layer emitted non-deterministic feature names")
    return out, names
