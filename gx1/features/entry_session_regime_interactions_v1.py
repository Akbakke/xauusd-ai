"""Entry session/regime interaction features.

This layer derives causal session, spread-cost and regime-state interaction
signals from already-materialized closed-bar ``snap.*`` and ``ctx_cont.*``
fields. It is intentionally dormant until a later manifest/rebuild gate wires
the features into a challenger dataset.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


SESSION_REGIME_INTERACTION_FEATURE_VERSION = "entry_session_regime_interactions_v1_20260630"
SESSION_REGIME_INTERACTION_FEATURE_PREFIX = "session_regime."

SESSION_REGIME_INTERACTION_SOURCE_FIELDS = (
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
    "ctx_cont.vol_pct_m5_1yr",
    "ctx_cont.vol_pct_h1_1yr",
    "ctx_cont.D1_atr_percentile_252",
    "ctx_cont.atr_ratio_m5_m15",
    "ctx_cont.atr_ratio_m15_d1",
    "ctx_cont.atr_ratio_h1_d1",
    "ctx_cont.regime_tf_agreement_v3",
    "ctx_cont.regime_stack_sum_v3",
    "ctx_cont.regime_divergence_flag_v3",
    "ctx_cont.d1_dist_to_boundary_v3",
    "ctx_cont.d1_regime_changed_flag_v3",
    "ctx_cont.bars_since_d1_regime_change_v3",
    "ctx_cont.m5_regime_class_id_v2",
    "ctx_cont.m15_regime_class_id_v2",
    "ctx_cont.h1_regime_class_id_v2",
    "ctx_cont.h4_regime_class_id_v2",
    "ctx_cont.d1_regime_class_id_v2",
    "chart.foundation_structure_up_minus_down",
    "chart.foundation_bos_recent_balance",
    "chart.foundation_choch_recent_tau24",
    "chart.foundation_sweep_reclaim_balance_proxy",
    "chart.foundation_compression_release_trigger",
    "chart.foundation_impulse_direction",
    "chart.foundation_pullback_depth_norm",
)


def _name_index(names: Iterable[str]) -> dict[str, int]:
    return {str(name): i for i, name in enumerate(names)}


def missing_session_regime_interaction_source_fields(feature_names: Iterable[str]) -> list[str]:
    available = {str(name) for name in feature_names}
    return [name for name in SESSION_REGIME_INTERACTION_SOURCE_FIELDS if name not in available]


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


def _tanh(arr: np.ndarray, scale: float = 1.0) -> np.ndarray:
    return np.tanh(arr / max(float(scale), 1e-6)).astype(np.float32, copy=False)


def _safe_ratio(num: np.ndarray, denom: np.ndarray, *, denom_floor: float = 1e-3) -> np.ndarray:
    return (num / np.maximum(np.abs(denom), float(denom_floor))).astype(np.float32, copy=False)


def _lag1(arr: np.ndarray) -> np.ndarray:
    out = np.empty_like(arr, dtype=np.float32)
    if arr.size:
        out[0] = 0.0
        out[1:] = arr[:-1]
    return out


def _class_vote_agreement(
    x: np.ndarray,
    idx: dict[str, int],
    names: tuple[str, str, str, str, str],
) -> tuple[np.ndarray, np.ndarray]:
    if not all(name in idx for name in names):
        zeros = np.zeros(x.shape[0], dtype=np.float32)
        return zeros, zeros
    m5, m15, h1, h4, d1 = (np.rint(_col(x, idx, name)).astype(np.int16, copy=False) for name in names)
    stack = np.vstack([m5, m15, h1, h4])
    vote_agreement = (stack == d1).mean(axis=0).astype(np.float32)
    short_long_mismatch = ((m5 != d1) | (m15 != d1) | (h1 != h4)).astype(np.float32)
    return vote_agreement, short_long_mismatch


def _add(arrays: list[np.ndarray], names: list[str], name: str, arr: np.ndarray, *, lo: float = -25.0, hi: float = 25.0) -> None:
    clean = _clip(np.asarray(arr, dtype=np.float32), lo, hi)
    if clean.ndim != 1:
        raise RuntimeError(f"session/regime feature {name} is not 1D: {clean.shape}")
    if not np.isfinite(clean).all():
        raise RuntimeError(f"session/regime feature {name} contains non-finite values")
    arrays.append(clean)
    names.append(f"{SESSION_REGIME_INTERACTION_FEATURE_PREFIX}{name}")


def build_entry_session_regime_interaction_layer(
    x: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Build causal session/regime interaction candidates from current fields."""
    x = np.asarray(x, dtype=np.float32)
    idx = _name_index(feature_names)
    arrays: list[np.ndarray] = []
    names: list[str] = []

    def c(name: str, default: float = 0.0) -> np.ndarray:
        return _col(x, idx, name, default=default)

    minutes_since_open = _clip(c("ctx_cont.minutes_since_session_open", default=1440.0), 0.0, 1440.0)
    minutes_to_boundary = _clip(c("ctx_cont.minutes_to_next_session_boundary", default=1440.0), 0.0, 1440.0)
    session_change = _clip01(c("ctx_cont.session_change_flag"))
    session_tradable = _clip01(c("ctx_cont.session_tradable"))
    open_risk = _clip01(np.exp(-minutes_since_open / 30.0).astype(np.float32) + session_change)
    boundary_risk = _clip01(np.exp(-minutes_to_boundary / 30.0).astype(np.float32) + session_change)
    boundary_transition = np.maximum(open_risk, boundary_risk).astype(np.float32)
    mid_session_stability = _clip01(session_tradable * (1.0 - open_risk) * (1.0 - boundary_risk))

    asia = _clip01(c("ctx_cont.is_ASIA"))
    asia_eu = _clip01(c("ctx_cont.is_asia_eu_overlap"))
    eu_us = _clip01(c("ctx_cont.is_eu_us_overlap"))
    eu = _clip01(c("ctx_cont.is_eu_only") + asia_eu + eu_us)
    us = _clip01(c("ctx_cont.is_us_only") + eu_us)
    overlap = _clip01(asia_eu + eu_us)
    active_session = _clip01(asia + eu + us)

    spread_ratio = _clip(_safe_ratio(c("ctx_cont.spread_bps"), c("ctx_cont.atr_bps", default=1.0)), 0.0, 5.0)
    spread_pressure = _clip01(_tanh(spread_ratio, scale=0.15))
    low_spread_permission = _clip01(1.0 - spread_pressure)

    vol_m5 = _clip01(c("ctx_cont.vol_pct_m5_1yr"))
    vol_h1 = _clip01(c("ctx_cont.vol_pct_h1_1yr"))
    d1_atr_pct = _clip01(c("ctx_cont.D1_atr_percentile_252"))
    vol_pressure = _clip01(0.45 * vol_m5 + 0.35 * vol_h1 + 0.20 * d1_atr_pct)
    atr_ratio_pressure = _clip01(
        0.34 * _tanh(c("ctx_cont.atr_ratio_m5_m15"), scale=2.0)
        + 0.33 * _tanh(c("ctx_cont.atr_ratio_m15_d1"), scale=2.0)
        + 0.33 * _tanh(c("ctx_cont.atr_ratio_h1_d1"), scale=2.0)
    )
    vol_expansion_pressure = _clip01(0.70 * vol_pressure + 0.30 * _pos(vol_pressure - _lag1(vol_pressure)) + 0.20 * atr_ratio_pressure)

    regime_agreement = _clip01(c("ctx_cont.regime_tf_agreement_v3"))
    regime_stack_signed = _tanh(c("ctx_cont.regime_stack_sum_v3"), scale=3.0)
    regime_divergence = _clip01(c("ctx_cont.regime_divergence_flag_v3"))
    regime_boundary_pressure = _clip01(1.0 - c("ctx_cont.d1_dist_to_boundary_v3", default=1.0))
    d1_regime_changed = _clip01(c("ctx_cont.d1_regime_changed_flag_v3"))
    bars_since_change_norm = _clip01(c("ctx_cont.bars_since_d1_regime_change_v3", default=1.0))
    regime_change_pressure = _clip01(
        d1_regime_changed
        + 0.45 * (1.0 - bars_since_change_norm)
        + 0.35 * regime_boundary_pressure
        + 0.20 * regime_divergence
    )
    regime_persistence = _clip01(bars_since_change_norm * regime_agreement * (1.0 - d1_regime_changed) * (1.0 - 0.50 * regime_boundary_pressure))
    class_vote_agreement, short_long_mismatch = _class_vote_agreement(
        x,
        idx,
        (
            "ctx_cont.m5_regime_class_id_v2",
            "ctx_cont.m15_regime_class_id_v2",
            "ctx_cont.h1_regime_class_id_v2",
            "ctx_cont.h4_regime_class_id_v2",
            "ctx_cont.d1_regime_class_id_v2",
        ),
    )
    mtf_agreement_pressure = _clip01(0.70 * regime_agreement + 0.30 * class_vote_agreement)
    mtf_divergence_pressure = _clip01(regime_divergence + 0.50 * (1.0 - regime_agreement) + 0.35 * short_long_mismatch)

    structure_bias = _clip(c("chart.foundation_structure_up_minus_down") / 2.0, -1.0, 1.0)
    bos_balance = _clip(c("chart.foundation_bos_recent_balance"), -1.0, 1.0)
    choch_recent = _clip01(c("chart.foundation_choch_recent_tau24"))
    sweep_balance = _clip(c("chart.foundation_sweep_reclaim_balance_proxy") / 5.0, -1.0, 1.0)
    compression_release = _clip01(c("chart.foundation_compression_release_trigger") / 5.0)
    impulse_direction = _clip(c("chart.foundation_impulse_direction") / 2.0, -1.0, 1.0)
    pullback_depth = _clip01(c("chart.foundation_pullback_depth_norm"))
    structure_alignment = _clip(structure_bias * regime_stack_signed * mtf_agreement_pressure, -1.0, 1.0)
    structure_conflict = _clip01(np.abs(structure_bias) * mtf_divergence_pressure)

    _add(arrays, names, "session_opening_risk", open_risk, lo=0.0, hi=1.0)
    _add(arrays, names, "session_boundary_risk", boundary_risk, lo=0.0, hi=1.0)
    _add(arrays, names, "session_mid_age_stability", mid_session_stability, lo=0.0, hi=1.0)
    _add(arrays, names, "asia_eu_overlap_transition_pressure", asia_eu * (0.50 + vol_pressure) * (0.50 + boundary_transition), lo=0.0, hi=2.0)
    _add(arrays, names, "eu_us_overlap_momentum_continuation", eu_us * low_spread_permission * mtf_agreement_pressure * np.abs(impulse_direction), lo=0.0, hi=1.0)
    _add(arrays, names, "eu_us_overlap_divergence_risk", eu_us * mtf_divergence_pressure * (0.50 + vol_pressure), lo=0.0, hi=2.0)

    _add(arrays, names, "spread_cost_ratio", spread_ratio, lo=0.0, hi=5.0)
    _add(arrays, names, "spread_cost_pressure", spread_pressure, lo=0.0, hi=1.0)
    _add(arrays, names, "spread_cost_x_boundary_risk", spread_pressure * boundary_transition, lo=0.0, hi=1.0)
    _add(arrays, names, "spread_cost_x_vol_expansion", spread_pressure * vol_expansion_pressure, lo=0.0, hi=1.0)
    _add(arrays, names, "spread_cost_x_overlap_risk", spread_pressure * overlap * (0.50 + vol_pressure), lo=0.0, hi=2.0)

    _add(arrays, names, "regime_persistence_score", regime_persistence, lo=0.0, hi=1.0)
    _add(arrays, names, "regime_change_pressure", regime_change_pressure, lo=0.0, hi=1.0)
    _add(arrays, names, "regime_change_x_session_boundary", regime_change_pressure * boundary_transition, lo=0.0, hi=1.0)
    _add(arrays, names, "mtf_regime_agreement_pressure", mtf_agreement_pressure, lo=0.0, hi=1.0)
    _add(arrays, names, "mtf_regime_divergence_pressure", mtf_divergence_pressure, lo=0.0, hi=1.0)
    _add(arrays, names, "mtf_regime_class_vote_agreement", class_vote_agreement, lo=0.0, hi=1.0)
    _add(arrays, names, "mtf_regime_short_long_mismatch", short_long_mismatch, lo=0.0, hi=1.0)

    _add(arrays, names, "asia_range_liquidity_reversal_pressure", asia * np.abs(sweep_balance) * (1.0 - np.abs(impulse_direction)) * (1.0 - mtf_agreement_pressure), lo=0.0, hi=1.0)
    _add(arrays, names, "eu_structure_breakout_readiness", eu * low_spread_permission * compression_release * mtf_agreement_pressure * np.abs(bos_balance), lo=0.0, hi=1.0)
    _add(arrays, names, "us_momentum_followthrough_pressure", us * low_spread_permission * mtf_agreement_pressure * impulse_direction, lo=-1.0, hi=1.0)
    _add(arrays, names, "overlap_liquidity_sweep_risk", overlap * np.abs(sweep_balance) * (0.50 + choch_recent) * (0.50 + mtf_divergence_pressure), lo=0.0, hi=2.0)
    _add(arrays, names, "session_vol_spread_breakout_readiness", active_session * compression_release * vol_expansion_pressure * low_spread_permission * (1.0 - boundary_transition), lo=0.0, hi=1.0)
    _add(arrays, names, "session_vol_spread_tail_risk", active_session * compression_release * vol_expansion_pressure * _clip01(spread_pressure + boundary_transition + mtf_divergence_pressure), lo=0.0, hi=1.0)
    _add(arrays, names, "session_structure_regime_alignment", active_session * mid_session_stability * structure_alignment, lo=-1.0, hi=1.0)
    _add(arrays, names, "session_structure_regime_conflict", active_session * structure_conflict * (0.50 + pullback_depth), lo=0.0, hi=2.0)

    out = np.column_stack(arrays).astype(np.float32, copy=False) if arrays else np.empty((x.shape[0], 0), dtype=np.float32)
    if not np.isfinite(out).all():
        raise RuntimeError("session/regime interaction layer contains non-finite values")
    if len(set(names)) != len(names):
        dupes = sorted({name for name in names if names.count(name) > 1})
        raise RuntimeError(f"session/regime interaction layer has duplicate names: {dupes[:10]}")
    return out, names


SESSION_REGIME_INTERACTION_FEATURE_NAMES = tuple(
    name for name in build_entry_session_regime_interaction_layer(np.zeros((1, 0), dtype=np.float32), [])[1]
)
