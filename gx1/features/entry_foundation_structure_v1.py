"""Explicit Entry foundation structure features.

This layer turns the existing causal ``snap.*`` and ``ctx_cont.*`` fields into
first-class model-native sequence evidence. It does not
read future prices and it intentionally keeps proxy names explicit where the
current contract does not expose the exact raw event.
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


FOUNDATION_STRUCTURE_FEATURE_VERSION = (
    "entry_foundation_structure_v2_20260722_exact_compression_semantics_failclosed"
)
FOUNDATION_STRUCTURE_FEATURE_PREFIX = "chart.foundation_"
FOUNDATION_STRUCTURE_REQUIRED_FAMILIES = (
    "hh_hl_lh_ll_state",
    "bos_choch_age",
    "sweep_reclaim_false_breakout",
    "compression_expansion",
    "impulse_pullback_phase",
    "session_x_structure",
)
FOUNDATION_STRUCTURE_SOURCE_FIELDS = (
    "ctx_cont._v1h1_ema_diff",
    "ctx_cont._v1h4_ema_diff",
    "ctx_cont.d1_ema_slope_20_canon_v2",
    "ctx_cont.m15_trend_sign_canon_v2",
    "ctx_cont.regime_stack_sum_v3",
    "ctx_cont.dist_last_swing_high_atr",
    "ctx_cont.dist_last_swing_low_atr",
    "ctx_cont.bars_since_swing_high",
    "ctx_cont.bars_since_swing_low",
    "snap.smc_swing_state",
    "snap.smc_bos_up",
    "snap.smc_bos_down",
    "ctx_cont.smc_bos_pressure_last12",
    "ctx_cont.smc_bos_pressure_last48",
    "snap.smc_choch",
    "ctx_cont.smc_choch_recent_tau12",
    "ctx_cont.smc_choch_recent_tau24",
    "ctx_cont.struct_pullback_depth_h1_v3",
    "ctx_cont.struct_pullback_depth_h4_v3",
    "ctx_cont.retracement_from_last_impulse",
    "snap.smc_sweep_up",
    "snap.smc_sweep_down",
    "snap.smc_bars_since_sweep",
    "ctx_cont.smc_sweep_bull_pressure_last12",
    "ctx_cont.smc_sweep_bull_pressure_last48",
    "snap.smc_sweep_size_atr",
    "ctx_cont.smc_sweep_size_recent_tau12",
    "ctx_cont.smc_sweep_recency_tau24",
    "ctx_cont.wick_ratio",
    "snap.wick_asym",
    "ctx_cont.sr_support_proximity_exp",
    "ctx_cont.sr_resistance_proximity_exp",
    "ctx_cont.H1_range_compression_ratio",
    "ctx_cont.M15_range_compression_ratio",
    "snap._v1_bb_squeeze_20_2",
    "snap.atr_z",
    "snap.rvol_20",
    "snap.vol_ratio_5_20",
    "ctx_cont.m15_trend_age_bars_norm_v2",
    "ctx_cont.h1_trend_age_bars_norm_v2",
    "ctx_cont.h4_trend_age_bars_norm_v2",
    "ctx_cont.is_ASIA",
    "ctx_cont.is_asia_eu_overlap",
    "ctx_cont.is_eu_us_overlap",
    "ctx_cont.is_eu_only",
    "ctx_cont.is_us_only",
)


def _name_index(names: Iterable[str]) -> dict[str, int]:
    values = list(names)
    invalid = [name for name in values if not isinstance(name, str) or not name]
    if invalid:
        raise RuntimeError(f"FOUNDATION_STRUCTURE_FEATURE_NAMES_INVALID: {invalid[:10]}")
    duplicates = sorted({name for name in values if values.count(name) > 1})
    if duplicates:
        raise RuntimeError(f"FOUNDATION_STRUCTURE_FEATURE_NAMES_DUPLICATE: {duplicates[:10]}")
    return {name: i for i, name in enumerate(values)}


def missing_foundation_structure_source_fields(feature_names: Iterable[str]) -> list[str]:
    available = set(feature_names)
    return [name for name in FOUNDATION_STRUCTURE_SOURCE_FIELDS if name not in available]


def _require_source_matrix(
    x: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, dict[str, int]]:
    try:
        matrix = np.asarray(x, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("FOUNDATION_STRUCTURE_INPUT_NOT_NUMERIC") from exc
    if matrix.ndim != 2:
        raise RuntimeError(f"FOUNDATION_STRUCTURE_INPUT_NOT_2D: shape={matrix.shape}")
    if matrix.shape[0] == 0:
        raise RuntimeError("FOUNDATION_STRUCTURE_INPUT_EMPTY")
    if len(feature_names) != matrix.shape[1]:
        raise RuntimeError(
            "FOUNDATION_STRUCTURE_FEATURE_NAME_COUNT_MISMATCH: "
            f"names={len(feature_names)} columns={matrix.shape[1]}"
        )
    index = _name_index(feature_names)
    missing = missing_foundation_structure_source_fields(feature_names)
    if missing:
        raise RuntimeError(
            "FOUNDATION_STRUCTURE_SOURCE_FIELDS_MISSING: "
            f"{missing[:30]} total={len(missing)}"
        )
    if not np.isfinite(matrix).all():
        bad = np.argwhere(~np.isfinite(matrix))[0]
        row, column = int(bad[0]), int(bad[1])
        raise RuntimeError(
            "FOUNDATION_STRUCTURE_SOURCE_NONFINITE: "
            f"row={row} field={feature_names[column]}"
        )
    return matrix, index


def _col(x: np.ndarray, index: dict[str, int], name: str) -> np.ndarray:
    try:
        column = index[name]
    except KeyError as exc:
        raise RuntimeError(f"FOUNDATION_STRUCTURE_SOURCE_FIELD_MISSING: {name}") from exc
    arr = np.asarray(x[:, column], dtype=np.float32)
    if arr.ndim != 1 or not np.isfinite(arr).all():
        raise RuntimeError(f"FOUNDATION_STRUCTURE_SOURCE_FIELD_INVALID: {name}")
    return arr


def _clip(arr: np.ndarray, lo: float = -25.0, hi: float = 25.0) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float32)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise RuntimeError(f"FOUNDATION_STRUCTURE_DERIVED_VALUE_INVALID: shape={values.shape}")
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


def _bars_since_event(event: np.ndarray, *, cap: int = 96) -> np.ndarray:
    flags = np.asarray(event, dtype=bool)
    out = np.empty(flags.shape[0], dtype=np.float32)
    last = -1
    for i, hit in enumerate(flags):
        if bool(hit):
            last = i
        out[i] = float(cap if last < 0 else min(i - last, cap))
    return out


def _state_eq(state: np.ndarray, value: int) -> np.ndarray:
    return (np.rint(state).astype(np.int16, copy=False) == int(value)).astype(np.float32)


def _add(arrays: list[np.ndarray], names: list[str], name: str, arr: np.ndarray, *, lo: float = -25.0, hi: float = 25.0) -> None:
    clean = _clip(np.asarray(arr, dtype=np.float32), lo, hi)
    if clean.ndim != 1:
        raise RuntimeError(f"foundation feature {name} is not 1D: {clean.shape}")
    if not np.isfinite(clean).all():
        raise RuntimeError(f"foundation feature {name} contains non-finite values")
    arrays.append(clean)
    names.append(f"{FOUNDATION_STRUCTURE_FEATURE_PREFIX}{name}")


def build_entry_foundation_structure_layer(
    x: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Build explicit foundation structure evidence from exact contract fields."""
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
    trend = _clip(0.35 * h1_trend + 0.30 * h4_trend + 0.20 * d1_slope + 0.10 * m15_trend + 0.05 * regime_stack)
    trend_up = _pos(trend)
    trend_down = _neg(trend)

    near_high = _prox_abs(c("ctx_cont.dist_last_swing_high_atr"))
    near_low = _prox_abs(c("ctx_cont.dist_last_swing_low_atr"))
    recent_high = _recency(c("ctx_cont.bars_since_swing_high"))
    recent_low = _recency(c("ctx_cont.bars_since_swing_low"))
    high_context = _clip01(near_high * (0.50 + recent_high))
    low_context = _clip01(near_low * (0.50 + recent_low))

    swing_state = c("snap.smc_swing_state")
    clean_up = _state_eq(swing_state, 0)
    up_bias = _state_eq(swing_state, 1)
    down_bias = _state_eq(swing_state, 2)
    clean_down = _state_eq(swing_state, 3)

    bos_up_event = _clip01(c("snap.smc_bos_up"))
    bos_down_event = _clip01(c("snap.smc_bos_down"))
    bos_pressure12 = c("ctx_cont.smc_bos_pressure_last12")
    bos_pressure48 = c("ctx_cont.smc_bos_pressure_last48")
    bos_up_pressure = _clip01(bos_up_event + 0.5 * _pos(bos_pressure12) + 0.25 * _pos(bos_pressure48))
    bos_down_pressure = _clip01(bos_down_event + 0.5 * _neg(bos_pressure12) + 0.25 * _neg(bos_pressure48))
    choch_event = _clip01(c("snap.smc_choch"))
    choch_recent_contract = _clip01(c("ctx_cont.smc_choch_recent_tau12") + 0.5 * c("ctx_cont.smc_choch_recent_tau24"))
    pullback = _clip01(0.60 * c("ctx_cont.struct_pullback_depth_h1_v3") + 0.40 * c("ctx_cont.struct_pullback_depth_h4_v3"))
    retracement = _clip01(c("ctx_cont.retracement_from_last_impulse"))

    hh_state = _clip01(0.85 * clean_up + 0.45 * up_bias + 0.50 * high_context * trend_up + 0.35 * bos_up_pressure)
    hl_state = _clip01(0.75 * clean_up + 0.55 * up_bias + 0.60 * low_context * trend_up * (1.0 + pullback + retracement))
    lh_state = _clip01(0.55 * down_bias + 0.45 * clean_down + 0.60 * high_context * trend_down * (1.0 + pullback + retracement))
    ll_state = _clip01(0.85 * clean_down + 0.45 * down_bias + 0.50 * low_context * trend_down + 0.35 * bos_down_pressure)
    _add(arrays, names, "hh_state", hh_state, lo=0.0, hi=1.0)
    _add(arrays, names, "hl_state", hl_state, lo=0.0, hi=1.0)
    _add(arrays, names, "lh_state", lh_state, lo=0.0, hi=1.0)
    _add(arrays, names, "ll_state", ll_state, lo=0.0, hi=1.0)
    _add(arrays, names, "structure_up_minus_down", (hh_state + hl_state) - (lh_state + ll_state), lo=-2.0, hi=2.0)

    bos_up_age = _bars_since_event(bos_up_event > 0.5, cap=96)
    bos_down_age = _bars_since_event(bos_down_event > 0.5, cap=96)
    choch_age = _bars_since_event(choch_event > 0.5, cap=96)
    bos_up_recent = np.exp(-bos_up_age / 24.0).astype(np.float32) * (0.5 + 0.5 * bos_up_pressure)
    bos_down_recent = np.exp(-bos_down_age / 24.0).astype(np.float32) * (0.5 + 0.5 * bos_down_pressure)
    choch_recent = _clip01(np.exp(-choch_age / 24.0).astype(np.float32) * (0.5 + 0.5 * choch_recent_contract))
    _add(arrays, names, "bos_up_age_bars", bos_up_age, lo=0.0, hi=96.0)
    _add(arrays, names, "bos_down_age_bars", bos_down_age, lo=0.0, hi=96.0)
    _add(arrays, names, "bos_up_recent_tau24", bos_up_recent, lo=0.0, hi=1.0)
    _add(arrays, names, "bos_down_recent_tau24", bos_down_recent, lo=0.0, hi=1.0)
    _add(arrays, names, "bos_recent_balance", bos_up_recent - bos_down_recent, lo=-1.0, hi=1.0)
    _add(arrays, names, "choch_age_bars", choch_age, lo=0.0, hi=96.0)
    _add(arrays, names, "choch_recent_tau24", choch_recent, lo=0.0, hi=1.0)
    _add(arrays, names, "bars_since_structure_break_min", np.minimum(np.minimum(bos_up_age, bos_down_age), choch_age), lo=0.0, hi=96.0)

    sweep_bull_pressure12 = c("ctx_cont.smc_sweep_bull_pressure_last12")
    sweep_bull_pressure48 = c("ctx_cont.smc_sweep_bull_pressure_last48")
    sweep_up = _clip01(c("snap.smc_sweep_up") + 0.5 * _neg(sweep_bull_pressure12) + 0.25 * _neg(sweep_bull_pressure48))
    sweep_down = _clip01(c("snap.smc_sweep_down") + 0.5 * _pos(sweep_bull_pressure12) + 0.25 * _pos(sweep_bull_pressure48))
    sweep_recent = _clip01(_recency(c("snap.smc_bars_since_sweep")) + c("ctx_cont.smc_sweep_recency_tau24"))
    sweep_size = _clip01(c("snap.smc_sweep_size_atr") + c("ctx_cont.smc_sweep_size_recent_tau12"))
    wick_level = _clip01(c("ctx_cont.wick_ratio") + np.abs(c("snap.wick_asym")))
    support_prox = _clip01(c("ctx_cont.sr_support_proximity_exp"))
    resistance_prox = _clip01(c("ctx_cont.sr_resistance_proximity_exp"))
    sweep_low_reclaim_up = _clip(sweep_down * sweep_recent * (0.50 + sweep_size) * (0.50 + wick_level) * (0.50 + support_prox + low_context + trend_up), 0.0, 5.0)
    sweep_high_reclaim_down = _clip(sweep_up * sweep_recent * (0.50 + sweep_size) * (0.50 + wick_level) * (0.50 + resistance_prox + high_context + trend_down), 0.0, 5.0)
    false_high_follow_down = _clip(sweep_up * resistance_prox * wick_level * (0.50 + trend_down + choch_recent), 0.0, 5.0)
    false_low_follow_up = _clip(sweep_down * support_prox * wick_level * (0.50 + trend_up + choch_recent), 0.0, 5.0)
    _add(arrays, names, "sweep_low_reclaim_up_proxy", sweep_low_reclaim_up, lo=0.0, hi=5.0)
    _add(arrays, names, "sweep_high_reclaim_down_proxy", sweep_high_reclaim_down, lo=0.0, hi=5.0)
    _add(arrays, names, "false_breakout_high_followthrough_down_proxy", false_high_follow_down, lo=0.0, hi=5.0)
    _add(arrays, names, "false_breakout_low_followthrough_up_proxy", false_low_follow_up, lo=0.0, hi=5.0)
    _add(arrays, names, "sweep_reclaim_balance_proxy", sweep_low_reclaim_up - sweep_high_reclaim_down, lo=-5.0, hi=5.0)

    h1_ratio = c("ctx_cont.H1_range_compression_ratio")
    m15_ratio = c("ctx_cont.M15_range_compression_ratio")
    bb_relative_width = c("snap._v1_bb_squeeze_20_2")
    h1_compression = atr_ratio_compression_pressure(h1_ratio)
    m15_compression = atr_ratio_compression_pressure(m15_ratio)
    squeeze = bollinger_squeeze_pressure(bb_relative_width)
    h1_expansion = atr_ratio_expansion_pressure(h1_ratio)
    m15_expansion = atr_ratio_expansion_pressure(m15_ratio)
    bb_expansion = bollinger_expansion_pressure(bb_relative_width)
    atr_z = _pos(_tanh(c("snap.atr_z"), scale=2.0))
    rvol = _pos(_tanh(c("snap.rvol_20"), scale=2.0))
    vol_ratio = _pos(_tanh(c("snap.vol_ratio_5_20"), scale=2.0))
    compression = _clip01(0.45 * h1_compression + 0.35 * m15_compression + 0.20 * squeeze)
    expansion = _clip01(
        0.20 * h1_expansion
        + 0.15 * m15_expansion
        + 0.15 * bb_expansion
        + 0.20 * atr_z
        + 0.15 * rvol
        + 0.15 * vol_ratio
    )
    expansion_delta = _pos(expansion - _lag1(expansion))
    release_trigger = _clip(_lag1(compression) * expansion_delta, 0.0, 5.0)
    _add(arrays, names, "compression_state", compression, lo=0.0, hi=1.0)
    _add(arrays, names, "expansion_state", expansion, lo=0.0, hi=1.0)
    _add(arrays, names, "compression_release_trigger", release_trigger, lo=0.0, hi=5.0)
    _add(arrays, names, "compression_release_up", release_trigger * trend_up, lo=0.0, hi=5.0)
    _add(arrays, names, "compression_release_down", release_trigger * trend_down, lo=0.0, hi=5.0)

    trend_delta = _clip(trend - _lag1(trend), lo=-2.0, hi=2.0)
    trend_age_m15 = _clip01(c("ctx_cont.m15_trend_age_bars_norm_v2"))
    trend_age_h1 = _clip01(c("ctx_cont.h1_trend_age_bars_norm_v2"))
    trend_age_h4 = _clip01(c("ctx_cont.h4_trend_age_bars_norm_v2"))
    impulse_direction = _clip(trend + 0.35 * (bos_up_recent - bos_down_recent) + 0.20 * trend_delta, lo=-2.0, hi=2.0)
    impulse_age_proxy = _clip01((trend_age_m15 + trend_age_h1 + trend_age_h4) / 3.0)
    pullback_phase_up = _clip01(trend_up * (0.50 * retracement + 0.50 * pullback) * (1.0 - 0.50 * choch_recent))
    pullback_phase_down = _clip01(trend_down * (0.50 * retracement + 0.50 * pullback) * (1.0 - 0.50 * choch_recent))
    _add(arrays, names, "impulse_direction", impulse_direction, lo=-2.0, hi=2.0)
    _add(arrays, names, "impulse_age_proxy", impulse_age_proxy, lo=0.0, hi=1.0)
    _add(arrays, names, "pullback_phase_up", pullback_phase_up, lo=0.0, hi=1.0)
    _add(arrays, names, "pullback_phase_down", pullback_phase_down, lo=0.0, hi=1.0)
    _add(arrays, names, "pullback_depth_norm", _clip01(0.50 * retracement + 0.50 * pullback), lo=0.0, hi=1.0)
    _add(arrays, names, "impulse_pullback_alignment", impulse_direction * (0.50 * retracement + 0.50 * pullback), lo=-2.0, hi=2.0)

    asia = _clip01(c("ctx_cont.is_ASIA"))
    asia_eu = _clip01(c("ctx_cont.is_asia_eu_overlap"))
    eu_us = _clip01(c("ctx_cont.is_eu_us_overlap"))
    eu = _clip01(c("ctx_cont.is_eu_only") + asia_eu + eu_us)
    us = _clip01(c("ctx_cont.is_us_only") + eu_us)
    overlap = _clip01(asia_eu + eu_us)
    session_groups = (
        ("asia", asia),
        ("eu", eu),
        ("us", us),
        ("overlap", overlap),
    )
    structure_signals = (
        ("hh_state", hh_state),
        ("hl_state", hl_state),
        ("lh_state", lh_state),
        ("ll_state", ll_state),
        ("bos_balance", bos_up_recent - bos_down_recent),
        ("choch_recent", choch_recent),
        ("sweep_reclaim_balance", sweep_low_reclaim_up - sweep_high_reclaim_down),
    )
    for session_name, session_signal in session_groups:
        for signal_name, signal in structure_signals:
            _add(arrays, names, f"{session_name}_x_{signal_name}", session_signal * signal, lo=-5.0, hi=5.0)

    out = np.column_stack(arrays).astype(np.float32, copy=False) if arrays else np.empty((x.shape[0], 0), dtype=np.float32)
    if not np.isfinite(out).all():
        raise RuntimeError("foundation structure layer contains non-finite values")
    if len(set(names)) != len(names):
        dupes = sorted({name for name in names if names.count(name) > 1})
        raise RuntimeError(f"foundation structure layer has duplicate names: {dupes[:10]}")
    return out, names


_FOUNDATION_NAME_PROBE = np.zeros(
    (1, len(FOUNDATION_STRUCTURE_SOURCE_FIELDS)), dtype=np.float32
)
for _ratio_field in (
    "ctx_cont.H1_range_compression_ratio",
    "ctx_cont.M15_range_compression_ratio",
):
    _FOUNDATION_NAME_PROBE[0, FOUNDATION_STRUCTURE_SOURCE_FIELDS.index(_ratio_field)] = 1.0
FOUNDATION_STRUCTURE_FEATURE_NAMES = tuple(
    build_entry_foundation_structure_layer(
        _FOUNDATION_NAME_PROBE,
        list(FOUNDATION_STRUCTURE_SOURCE_FIELDS),
    )[1]
)
del _FOUNDATION_NAME_PROBE, _ratio_field
