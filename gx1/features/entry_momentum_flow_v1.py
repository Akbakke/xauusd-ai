"""Entry momentum/flow challenger features.

This layer is intentionally inactive by default. It builds deterministic,
closed-bar momentum derivatives from already-emitted snap/ctx/candle/chart
inputs so they can be audited before any seq215 dataset rebuild or training.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


MOMENTUM_FLOW_FEATURE_VERSION = "entry_momentum_flow_v1_20260630_causal_closed_bar_derivatives"
MOMENTUM_FLOW_FEATURE_PREFIX = "momentum.flow_"

MOMENTUM_FLOW_REQUIRED_SOURCE_FIELDS = (
    "ret_1",
    "ret_5",
    "ret_20",
    "_v1_clv",
    "micro_momentum_3",
    "micro_momentum_5",
    "micro_acceleration",
)

MOMENTUM_FLOW_OPTIONAL_SOURCE_FIELDS = (
    "atr_bps",
    "atr_z",
    "_v1_range_z",
    "_v1h1_slope5",
    "_v1h4_slope5",
    "d1_pct_change_5_canon_v2",
    "d1_ema_slope_20_canon_v2",
    "regime_tf_agreement_v3",
    "regime_divergence_flag_v3",
    "dip_confirmed_m5_v3",
    "dip_confirmed_m15_v3",
    "dip_confirmed_h1_v3",
    "dip_confirmed_h4_v3",
    "dip_confirmed_d1_v3",
    "dip_confirmed_mean_5tf",
    "dip_confirmed_max_5tf",
    "dip_proximity_h1_v3",
    "dip_proximity_h4_v3",
    "dip_proximity_d1_v3",
    "dip_proximity_mean_h1h4d1",
    "chart.foundation_impulse_direction",
    "chart.foundation_impulse_pullback_alignment",
    "chart.foundation_compression_release_up",
    "chart.foundation_compression_release_down",
    "candle.pattern_bull_continuation_pressure",
    "candle.pattern_bear_continuation_pressure",
    "candle.pattern_bull_reversal_pressure",
    "candle.pattern_bear_reversal_pressure",
    "wick_asym",
    "wick_ratio",
    "candle.pattern_upper_wick_share",
    "candle.pattern_lower_wick_share",
)

_SOURCE_ALIASES: dict[str, tuple[str, ...]] = {
    "ret_1": ("snap.ret_1", "ret_1"),
    "ret_5": ("snap.ret_5", "ret_5"),
    "ret_20": ("snap.ret_20", "ret_20"),
    "_v1_clv": ("snap._v1_clv", "_v1_clv", "candle.pattern_close_location"),
    "micro_momentum_3": ("ctx_cont.micro_momentum_3", "micro_momentum_3"),
    "micro_momentum_5": ("ctx_cont.micro_momentum_5", "micro_momentum_5"),
    "micro_acceleration": ("ctx_cont.micro_acceleration", "micro_acceleration"),
    "atr_bps": ("ctx_cont.atr_bps", "atr_bps"),
    "atr_z": ("snap.atr_z", "ctx_cont.atr_z", "atr_z"),
    "_v1_range_z": ("snap._v1_range_z", "ctx_cont._v1_range_z", "_v1_range_z"),
    "_v1h1_slope5": ("ctx_cont._v1h1_slope5", "_v1h1_slope5"),
    "_v1h4_slope5": ("ctx_cont._v1h4_slope5", "_v1h4_slope5"),
    "d1_pct_change_5_canon_v2": ("ctx_cont.d1_pct_change_5_canon_v2", "d1_pct_change_5_canon_v2"),
    "d1_ema_slope_20_canon_v2": ("ctx_cont.d1_ema_slope_20_canon_v2", "d1_ema_slope_20_canon_v2"),
    "regime_tf_agreement_v3": ("ctx_cont.regime_tf_agreement_v3", "regime_tf_agreement_v3"),
    "regime_divergence_flag_v3": ("ctx_cont.regime_divergence_flag_v3", "regime_divergence_flag_v3"),
    "dip_confirmed_m5_v3": ("ctx_cont.dip_confirmed_m5_v3", "dip_confirmed_m5_v3"),
    "dip_confirmed_m15_v3": ("ctx_cont.dip_confirmed_m15_v3", "dip_confirmed_m15_v3"),
    "dip_confirmed_h1_v3": ("ctx_cont.dip_confirmed_h1_v3", "dip_confirmed_h1_v3"),
    "dip_confirmed_h4_v3": ("ctx_cont.dip_confirmed_h4_v3", "dip_confirmed_h4_v3"),
    "dip_confirmed_d1_v3": ("ctx_cont.dip_confirmed_d1_v3", "dip_confirmed_d1_v3"),
    "dip_confirmed_mean_5tf": ("ctx_cont.dip_confirmed_mean_5tf", "dip_confirmed_mean_5tf"),
    "dip_confirmed_max_5tf": ("ctx_cont.dip_confirmed_max_5tf", "dip_confirmed_max_5tf"),
    "dip_proximity_h1_v3": ("ctx_cont.dip_proximity_h1_v3", "dip_proximity_h1_v3"),
    "dip_proximity_h4_v3": ("ctx_cont.dip_proximity_h4_v3", "dip_proximity_h4_v3"),
    "dip_proximity_d1_v3": ("ctx_cont.dip_proximity_d1_v3", "dip_proximity_d1_v3"),
    "dip_proximity_mean_h1h4d1": ("ctx_cont.dip_proximity_mean_h1h4d1", "dip_proximity_mean_h1h4d1"),
    "chart.foundation_impulse_direction": ("chart.foundation_impulse_direction",),
    "chart.foundation_impulse_pullback_alignment": ("chart.foundation_impulse_pullback_alignment",),
    "chart.foundation_compression_release_up": ("chart.foundation_compression_release_up",),
    "chart.foundation_compression_release_down": ("chart.foundation_compression_release_down",),
    "candle.pattern_bull_continuation_pressure": ("candle.pattern_bull_continuation_pressure",),
    "candle.pattern_bear_continuation_pressure": ("candle.pattern_bear_continuation_pressure",),
    "candle.pattern_bull_reversal_pressure": ("candle.pattern_bull_reversal_pressure",),
    "candle.pattern_bear_reversal_pressure": ("candle.pattern_bear_reversal_pressure",),
    "wick_asym": ("snap.wick_asym", "wick_asym"),
    "wick_ratio": ("ctx_cont.wick_ratio", "wick_ratio"),
    "candle.pattern_upper_wick_share": ("candle.pattern_upper_wick_share",),
    "candle.pattern_lower_wick_share": ("candle.pattern_lower_wick_share",),
}

MOMENTUM_FLOW_FEATURE_SUFFIXES = (
    "vol_scale_bps_proxy",
    "ret1_vol_norm",
    "ret5_vol_norm",
    "ret20_vol_norm",
    "micro_impulse_norm",
    "acceleration_norm",
    "deceleration_pressure",
    "impulse_direction_score",
    "bull_followthrough_score",
    "bear_followthrough_score",
    "followthrough_balance",
    "mtf_confirmation_score",
    "mtf_conflict_score",
    "mtf_bull_confirmation",
    "mtf_bear_confirmation",
    "candle_bull_pressure",
    "candle_bear_pressure",
    "bull_exhaustion_pressure",
    "bear_exhaustion_pressure",
    "dip_continuation_long_input",
    "dip_continuation_short_input",
    "dip_reversal_long_risk_input",
    "dip_reversal_short_risk_input",
    "impulse_pullback_alignment_score",
    "compression_release_followthrough_score",
    "clean_edge_score",
)

MOMENTUM_FLOW_FEATURE_NAMES = tuple(
    f"{MOMENTUM_FLOW_FEATURE_PREFIX}{name}" for name in MOMENTUM_FLOW_FEATURE_SUFFIXES
)


def _name_index(names: Iterable[str]) -> dict[str, int]:
    return {str(name): i for i, name in enumerate(names)}


def _aliases(name: str) -> tuple[str, ...]:
    return _SOURCE_ALIASES.get(name, (name,))


def _has_source(index: dict[str, int], canonical: str) -> bool:
    return any(alias in index for alias in _aliases(canonical))


def missing_entry_momentum_flow_source_fields(feature_names: Iterable[str]) -> list[str]:
    """Return required source fields not represented by any accepted alias."""
    available = {str(name) for name in feature_names}
    missing: list[str] = []
    for canonical in MOMENTUM_FLOW_REQUIRED_SOURCE_FIELDS:
        if not any(alias in available for alias in _aliases(canonical)):
            missing.append(canonical)
    return missing


def _col(x: np.ndarray, index: dict[str, int], canonical: str, default: float = 0.0) -> np.ndarray:
    for name in _aliases(canonical):
        if name in index:
            arr = np.asarray(x[:, index[name]], dtype=np.float32)
            return np.nan_to_num(arr, nan=float(default), posinf=float(default), neginf=float(default))
    return np.full(x.shape[0], float(default), dtype=np.float32)


def _clip(arr: np.ndarray, lo: float = -25.0, hi: float = 25.0) -> np.ndarray:
    return np.clip(np.nan_to_num(arr, nan=0.0, posinf=hi, neginf=lo), lo, hi).astype(np.float32, copy=False)


def _clip01(arr: np.ndarray) -> np.ndarray:
    return _clip(arr, 0.0, 1.0)


def _pos(arr: np.ndarray) -> np.ndarray:
    return np.maximum(arr, 0.0).astype(np.float32, copy=False)


def _neg(arr: np.ndarray) -> np.ndarray:
    return np.maximum(-arr, 0.0).astype(np.float32, copy=False)


def _safe_scale(*parts: np.ndarray, floor: float = 1.0) -> np.ndarray:
    stack = np.vstack([np.abs(np.asarray(part, dtype=np.float32)) for part in parts])
    return np.maximum(np.nanmax(stack, axis=0), float(floor)).astype(np.float32, copy=False)


def _tanh(arr: np.ndarray, scale: float | np.ndarray = 1.0) -> np.ndarray:
    den = np.maximum(np.asarray(scale, dtype=np.float32), 1e-6)
    return np.tanh(np.asarray(arr, dtype=np.float32) / den).astype(np.float32, copy=False)


def _mean01(parts: list[np.ndarray]) -> np.ndarray:
    if not parts:
        raise RuntimeError("momentum flow mean requested with no parts")
    return _clip01(np.vstack(parts).mean(axis=0))


def _sign_agreement(parts: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    stack = np.vstack([np.asarray(part, dtype=np.float32) for part in parts])
    valid = np.abs(stack) > 1e-6
    valid_count = np.maximum(valid.sum(axis=0).astype(np.float32), 1.0)
    bull = ((stack > 0.0) & valid).sum(axis=0).astype(np.float32) / valid_count
    bear = ((stack < 0.0) & valid).sum(axis=0).astype(np.float32) / valid_count
    confirmation = np.maximum(bull, bear).astype(np.float32)
    conflict = np.minimum(bull, bear).astype(np.float32)
    return _clip01(confirmation), _clip01(conflict), _clip01(bull), _clip01(bear)


def _add(arrays: list[np.ndarray], names: list[str], suffix: str, arr: np.ndarray, *, lo: float = -25.0, hi: float = 25.0) -> None:
    clean = _clip(np.asarray(arr, dtype=np.float32), lo, hi)
    if clean.ndim != 1:
        raise RuntimeError(f"momentum flow feature {suffix} is not 1D: {clean.shape}")
    if not np.isfinite(clean).all():
        raise RuntimeError(f"momentum flow feature {suffix} contains non-finite values")
    arrays.append(clean)
    names.append(f"{MOMENTUM_FLOW_FEATURE_PREFIX}{suffix}")


def build_entry_momentum_flow_layer(
    x: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Build deterministic candidate momentum/flow features from as-of inputs."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise RuntimeError(f"momentum flow input matrix must be 2D, got {x.shape}")
    idx = _name_index(feature_names)
    missing = missing_entry_momentum_flow_source_fields(feature_names)
    if missing:
        raise RuntimeError(f"momentum flow required source fields missing: {missing}")

    def c(name: str, default: float = 0.0) -> np.ndarray:
        return _col(x, idx, name, default=default)

    ret1 = c("ret_1")
    ret5 = c("ret_5")
    ret20 = c("ret_20")
    clv = _tanh(c("_v1_clv"), scale=2.0)
    mom3 = c("micro_momentum_3")
    mom5 = c("micro_momentum_5")
    micro_accel = c("micro_acceleration")
    atr_bps = c("atr_bps")
    atr_z = _tanh(c("atr_z"), scale=3.0)
    range_z = _tanh(c("_v1_range_z"), scale=3.0)
    wick_signal_available = any(
        _has_source(idx, name)
        for name in (
            "wick_asym",
            "wick_ratio",
            "candle.pattern_upper_wick_share",
            "candle.pattern_lower_wick_share",
        )
    )
    wick_weight = 1.0 if wick_signal_available else 0.0
    wick_asym = _clip(c("wick_asym"), -1.0, 1.0)
    wick_ratio = _clip01(c("wick_ratio", default=0.5))
    upper_wick_share = _clip01(c("candle.pattern_upper_wick_share"))
    lower_wick_share = _clip01(c("candle.pattern_lower_wick_share"))
    upper_wick_flow = _clip01(wick_weight * (_pos(wick_asym) + 0.50 * wick_ratio + 0.50 * upper_wick_share))
    lower_wick_flow = _clip01(
        wick_weight * (_neg(wick_asym) + 0.50 * (1.0 - wick_ratio) + 0.50 * lower_wick_share)
    )

    vol_scale = _safe_scale(
        ret1,
        ret5 / np.sqrt(5.0),
        ret20 / np.sqrt(20.0),
        atr_bps,
        floor=1.0,
    )
    ret1_norm = _tanh(ret1, vol_scale)
    ret5_norm = _tanh(ret5, vol_scale * np.sqrt(5.0))
    ret20_norm = _tanh(ret20, vol_scale * np.sqrt(20.0))

    micro_scale = _safe_scale(mom3, mom5 / np.sqrt(5.0), floor=0.5)
    micro_impulse = _tanh(0.65 * mom3 + 0.35 * mom5, micro_scale)
    micro_accel_norm = _tanh(micro_accel, micro_scale)
    return_acceleration = _clip(
        0.65 * (ret1_norm - ret5_norm) + 0.35 * (ret5_norm - ret20_norm),
        -2.0,
        2.0,
    )
    acceleration = _clip(
        0.65 * return_acceleration + 0.35 * micro_accel_norm,
        -2.0,
        2.0,
    )
    deceleration = _clip01(
        0.55 * np.abs(ret20_norm)
        + 0.35 * np.abs(ret5_norm)
        - 0.45 * np.abs(ret1_norm)
        - 0.15 * np.abs(acceleration)
    )
    impulse_direction = _clip(
        0.35 * ret1_norm + 0.35 * ret5_norm + 0.20 * ret20_norm + 0.10 * micro_impulse,
        -2.0,
        2.0,
    )

    h1_flow = _tanh(c("_v1h1_slope5"), scale=2.0)
    h4_flow = _tanh(c("_v1h4_slope5"), scale=2.0)
    d1_ret = _tanh(c("d1_pct_change_5_canon_v2"), scale=25.0)
    d1_slope = _tanh(c("d1_ema_slope_20_canon_v2"), scale=2.0)
    mtf_confirmation, mtf_conflict, mtf_bull, mtf_bear = _sign_agreement(
        [ret5_norm, ret20_norm, h1_flow, h4_flow, d1_ret + d1_slope]
    )
    regime_agreement = _clip01(c("regime_tf_agreement_v3"))
    regime_divergence = _clip01(c("regime_divergence_flag_v3"))
    mtf_confirmation = _clip01(0.75 * mtf_confirmation + 0.25 * regime_agreement)
    mtf_conflict = _clip01(0.80 * mtf_conflict + 0.20 * regime_divergence)
    bull_persistence = _clip01(
        0.25 * _pos(ret1_norm)
        + 0.30 * _pos(ret5_norm)
        + 0.20 * _pos(ret20_norm)
        + 0.15 * _pos(micro_impulse)
        + 0.10 * mtf_bull
    )
    bear_persistence = _clip01(
        0.25 * _neg(ret1_norm)
        + 0.30 * _neg(ret5_norm)
        + 0.20 * _neg(ret20_norm)
        + 0.15 * _neg(micro_impulse)
        + 0.10 * mtf_bear
    )
    persistence_balance = _clip(bull_persistence - bear_persistence, -1.0, 1.0)

    bull_cont = _clip01(c("candle.pattern_bull_continuation_pressure"))
    bear_cont = _clip01(c("candle.pattern_bear_continuation_pressure"))
    bull_reversal = _clip01(c("candle.pattern_bull_reversal_pressure"))
    bear_reversal = _clip01(c("candle.pattern_bear_reversal_pressure"))
    clv_wick_bull_flow = _clip01(0.50 * _pos(clv) + 0.35 * lower_wick_flow + 0.15 * _pos(return_acceleration))
    clv_wick_bear_flow = _clip01(0.50 * _neg(clv) + 0.35 * upper_wick_flow + 0.15 * _neg(return_acceleration))
    candle_bull = _clip01(
        0.30 * _pos(clv)
        + 0.25 * bull_cont
        + 0.15 * bull_reversal
        + 0.10 * _pos(micro_impulse)
        + 0.20 * clv_wick_bull_flow
    )
    candle_bear = _clip01(
        0.30 * _neg(clv)
        + 0.25 * bear_cont
        + 0.15 * bear_reversal
        + 0.10 * _neg(micro_impulse)
        + 0.20 * clv_wick_bear_flow
    )

    vol_follow_quality = _clip01(
        1.0
        - 0.30 * np.maximum(np.abs(atr_z), np.abs(range_z))
        + 0.20 * mtf_confirmation
        - 0.20 * mtf_conflict
    )
    bull_follow_raw = _clip01(
        0.25 * _pos(ret1_norm)
        + 0.30 * _pos(ret5_norm)
        + 0.15 * _pos(ret20_norm)
        + 0.15 * _pos(micro_impulse)
        + 0.15 * candle_bull
    )
    bear_follow_raw = _clip01(
        0.25 * _neg(ret1_norm)
        + 0.30 * _neg(ret5_norm)
        + 0.15 * _neg(ret20_norm)
        + 0.15 * _neg(micro_impulse)
        + 0.15 * candle_bear
    )
    bull_follow = _clip01((0.90 * bull_follow_raw + 0.10 * bull_persistence) * (0.70 + 0.30 * vol_follow_quality))
    bear_follow = _clip01((0.90 * bear_follow_raw + 0.10 * bear_persistence) * (0.70 + 0.30 * vol_follow_quality))
    follow_balance = _clip(bull_follow - bear_follow, -1.0, 1.0)

    high_vol_stress = _clip01(
        0.35 * np.abs(atr_z)
        + 0.35 * np.abs(range_z)
        + 0.30 * np.maximum(_pos(ret20_norm), _neg(ret20_norm))
    )
    bearish_reversal_pressure = _clip01(
        _pos(impulse_direction)
        * (
            0.22 * deceleration
            + 0.20 * _neg(return_acceleration)
            + 0.18 * high_vol_stress
            + 0.18 * candle_bear
            + 0.12 * upper_wick_flow
            + 0.10 * _neg(clv)
        )
    )
    bullish_reversal_pressure = _clip01(
        _neg(impulse_direction)
        * (
            0.22 * deceleration
            + 0.20 * _pos(return_acceleration)
            + 0.18 * high_vol_stress
            + 0.18 * candle_bull
            + 0.12 * lower_wick_flow
            + 0.10 * _pos(clv)
        )
    )
    bull_exhaustion = _clip01(
        0.70
        * _clip01(
            _pos(impulse_direction)
            * (
                0.30
                + 0.25 * deceleration
                + 0.20 * high_vol_stress
                + 0.15 * candle_bear
                + 0.10 * _neg(clv)
            )
        )
        + 0.30 * bearish_reversal_pressure
    )
    bear_exhaustion = _clip01(
        0.70
        * _clip01(
            _neg(impulse_direction)
            * (
                0.30
                + 0.25 * deceleration
                + 0.20 * high_vol_stress
                + 0.15 * candle_bull
                + 0.10 * _pos(clv)
            )
        )
        + 0.30 * bullish_reversal_pressure
    )

    dip_confirmed = _mean01(
        [
            c("dip_confirmed_m5_v3"),
            c("dip_confirmed_m15_v3"),
            c("dip_confirmed_h1_v3"),
            c("dip_confirmed_h4_v3"),
            c("dip_confirmed_d1_v3"),
            c("dip_confirmed_mean_5tf"),
            c("dip_confirmed_max_5tf"),
        ]
    )
    dip_proximity = _mean01(
        [
            c("dip_proximity_h1_v3"),
            c("dip_proximity_h4_v3"),
            c("dip_proximity_d1_v3"),
            c("dip_proximity_mean_h1h4d1"),
        ]
    )
    dip_setup = _clip01(0.55 * dip_confirmed + 0.45 * dip_proximity)
    dip_cont_long = _clip01(
        dip_setup * (0.45 * bull_follow + 0.30 * mtf_bull + 0.25 * (1.0 - bull_exhaustion))
    )
    dip_cont_short = _clip01(
        dip_setup * (0.45 * bear_follow + 0.30 * mtf_bear + 0.25 * (1.0 - bear_exhaustion))
    )
    dip_reversal_long = _clip01(
        dip_setup * (0.40 * bear_follow + 0.25 * mtf_conflict + 0.20 * bull_exhaustion + 0.15 * candle_bear)
    )
    dip_reversal_short = _clip01(
        dip_setup * (0.40 * bull_follow + 0.25 * mtf_conflict + 0.20 * bear_exhaustion + 0.15 * candle_bull)
    )

    impulse_pullback_alignment = _clip(
        0.40 * _tanh(c("chart.foundation_impulse_direction"), scale=1.0)
        + 0.30 * _tanh(c("chart.foundation_impulse_pullback_alignment"), scale=1.0)
        + 0.20 * follow_balance
        + 0.10 * persistence_balance,
        -1.0,
        1.0,
    )
    compression_release_follow = _clip(
        c("chart.foundation_compression_release_up") * bull_follow
        - c("chart.foundation_compression_release_down") * bear_follow,
        -1.0,
        1.0,
    )
    clean_edge = _clip01(
        np.abs(follow_balance)
        * mtf_confirmation
        * (1.0 - mtf_conflict)
        * (1.0 - np.maximum(bull_exhaustion, bear_exhaustion))
        * (0.80 + 0.20 * vol_follow_quality)
        * (0.75 + 0.25 * np.maximum(bull_persistence, bear_persistence))
    )

    arrays: list[np.ndarray] = []
    names: list[str] = []
    _add(arrays, names, "vol_scale_bps_proxy", vol_scale, lo=0.0, hi=500.0)
    _add(arrays, names, "ret1_vol_norm", ret1_norm, lo=-1.0, hi=1.0)
    _add(arrays, names, "ret5_vol_norm", ret5_norm, lo=-1.0, hi=1.0)
    _add(arrays, names, "ret20_vol_norm", ret20_norm, lo=-1.0, hi=1.0)
    _add(arrays, names, "micro_impulse_norm", micro_impulse, lo=-1.0, hi=1.0)
    _add(arrays, names, "acceleration_norm", acceleration, lo=-2.0, hi=2.0)
    _add(arrays, names, "deceleration_pressure", deceleration, lo=0.0, hi=1.0)
    _add(arrays, names, "impulse_direction_score", impulse_direction, lo=-2.0, hi=2.0)
    _add(arrays, names, "bull_followthrough_score", bull_follow, lo=0.0, hi=1.0)
    _add(arrays, names, "bear_followthrough_score", bear_follow, lo=0.0, hi=1.0)
    _add(arrays, names, "followthrough_balance", follow_balance, lo=-1.0, hi=1.0)
    _add(arrays, names, "mtf_confirmation_score", mtf_confirmation, lo=0.0, hi=1.0)
    _add(arrays, names, "mtf_conflict_score", mtf_conflict, lo=0.0, hi=1.0)
    _add(arrays, names, "mtf_bull_confirmation", mtf_bull, lo=0.0, hi=1.0)
    _add(arrays, names, "mtf_bear_confirmation", mtf_bear, lo=0.0, hi=1.0)
    _add(arrays, names, "candle_bull_pressure", candle_bull, lo=0.0, hi=1.0)
    _add(arrays, names, "candle_bear_pressure", candle_bear, lo=0.0, hi=1.0)
    _add(arrays, names, "bull_exhaustion_pressure", bull_exhaustion, lo=0.0, hi=1.0)
    _add(arrays, names, "bear_exhaustion_pressure", bear_exhaustion, lo=0.0, hi=1.0)
    _add(arrays, names, "dip_continuation_long_input", dip_cont_long, lo=0.0, hi=1.0)
    _add(arrays, names, "dip_continuation_short_input", dip_cont_short, lo=0.0, hi=1.0)
    _add(arrays, names, "dip_reversal_long_risk_input", dip_reversal_long, lo=0.0, hi=1.0)
    _add(arrays, names, "dip_reversal_short_risk_input", dip_reversal_short, lo=0.0, hi=1.0)
    _add(arrays, names, "impulse_pullback_alignment_score", impulse_pullback_alignment, lo=-1.0, hi=1.0)
    _add(arrays, names, "compression_release_followthrough_score", compression_release_follow, lo=-1.0, hi=1.0)
    _add(arrays, names, "clean_edge_score", clean_edge, lo=0.0, hi=1.0)

    out = np.column_stack(arrays).astype(np.float32, copy=False) if arrays else np.empty((x.shape[0], 0), dtype=np.float32)
    if tuple(names) != MOMENTUM_FLOW_FEATURE_NAMES:
        raise RuntimeError("momentum flow feature order drifted")
    if not np.isfinite(out).all():
        raise RuntimeError("momentum flow layer contains non-finite values")
    return out, names
