"""Entry session/regime interaction features.

This layer derives causal session, spread-cost and regime-state interaction
signals from already-materialized closed-bar ``snap.*`` and ``ctx_cont.*``
fields. It is intentionally dormant until a later manifest/rebuild gate wires
the features into a challenger dataset.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


SESSION_REGIME_INTERACTION_FEATURE_VERSION = "entry_session_regime_interactions_v1_20260630_h4d1_transition_risk"
SESSION_REGIME_INTERACTION_FEATURE_PREFIX = "session_regime."

SESSION_REGIME_INTERACTION_SOURCE_FIELDS = (
    "ret_1",
    "ret_5",
    "ret_20",
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
    "ctx_cont.micro_momentum_3",
    "ctx_cont.micro_momentum_5",
    "ctx_cont.micro_acceleration",
    "ctx_cont.vol_pct_m5_1yr",
    "ctx_cont.vol_pct_h1_1yr",
    "ctx_cont.D1_atr_percentile_252",
    "ctx_cat.vol_regime_id",
    "ctx_cat.spread_bucket",
    "ctx_cat.atr_bucket",
    "ctx_cat.H4_trend_sign_cat",
    "ctx_cont.atr_ratio_m5_m15",
    "ctx_cont.atr_ratio_m15_d1",
    "ctx_cont.atr_ratio_h1_d1",
    "ctx_cont._v1h4_ema_diff",
    "ctx_cont._v1h4_slope5",
    "ctx_cont.d1_ema_slope_20_canon_v2",
    "ctx_cont.regime_tf_agreement_v3",
    "ctx_cont.regime_stack_sum_v3",
    "ctx_cont.regime_divergence_flag_v3",
    "ctx_cont.d1_dist_to_boundary_v3",
    "ctx_cont.d1_dist_roc_288_v3",
    "ctx_cont.d1_regime_changed_flag_v3",
    "ctx_cont.bars_since_d1_regime_change_v3",
    "ctx_cont.d1_trend_age_mature_flag_v3",
    "ctx_cont.m5_regime_class_id_v2",
    "ctx_cont.m15_regime_class_id_v2",
    "ctx_cont.h1_regime_class_id_v2",
    "ctx_cont.h4_regime_class_id_v2",
    "ctx_cont.d1_regime_class_id_v2",
    "ctx_cont.h4_trend_age_bars_norm_v2",
    "ctx_cont.d1_trend_age_bars_norm_v2",
    "chart.foundation_structure_up_minus_down",
    "chart.foundation_bos_recent_balance",
    "chart.foundation_choch_recent_tau24",
    "chart.foundation_sweep_reclaim_balance_proxy",
    "chart.foundation_compression_release_trigger",
    "chart.foundation_impulse_direction",
    "chart.foundation_pullback_depth_norm",
)

_SOURCE_ALIASES: dict[str, tuple[str, ...]] = {
    "ret_1": ("snap.ret_1", "ret_1"),
    "ret_5": ("snap.ret_5", "ret_5"),
    "ret_20": ("snap.ret_20", "ret_20"),
    "ctx_cont.minutes_since_session_open": ("ctx_cont.minutes_since_session_open", "minutes_since_session_open"),
    "ctx_cont.minutes_to_next_session_boundary": (
        "ctx_cont.minutes_to_next_session_boundary",
        "minutes_to_next_session_boundary",
    ),
    "ctx_cont.session_change_flag": ("ctx_cont.session_change_flag", "session_change_flag"),
    "ctx_cont.session_tradable": ("ctx_cont.session_tradable", "session_tradable"),
    "ctx_cont.is_ASIA": ("ctx_cont.is_ASIA", "is_ASIA"),
    "ctx_cont.is_asia_eu_overlap": ("ctx_cont.is_asia_eu_overlap", "is_asia_eu_overlap"),
    "ctx_cont.is_eu_us_overlap": ("ctx_cont.is_eu_us_overlap", "is_eu_us_overlap"),
    "ctx_cont.is_eu_only": ("ctx_cont.is_eu_only", "is_eu_only"),
    "ctx_cont.is_us_only": ("ctx_cont.is_us_only", "is_us_only"),
    "ctx_cont.spread_bps": ("ctx_cont.spread_bps", "spread_bps"),
    "ctx_cont.atr_bps": ("ctx_cont.atr_bps", "atr_bps"),
    "ctx_cont.micro_momentum_3": ("ctx_cont.micro_momentum_3", "micro_momentum_3"),
    "ctx_cont.micro_momentum_5": ("ctx_cont.micro_momentum_5", "micro_momentum_5"),
    "ctx_cont.micro_acceleration": ("ctx_cont.micro_acceleration", "micro_acceleration"),
    "ctx_cont.vol_pct_m5_1yr": ("ctx_cont.vol_pct_m5_1yr", "vol_pct_m5_1yr"),
    "ctx_cont.vol_pct_h1_1yr": ("ctx_cont.vol_pct_h1_1yr", "vol_pct_h1_1yr"),
    "ctx_cont.D1_atr_percentile_252": ("ctx_cont.D1_atr_percentile_252", "D1_atr_percentile_252"),
    "ctx_cat.vol_regime_id": ("ctx_cat.vol_regime_id", "vol_regime_id"),
    "ctx_cat.spread_bucket": ("ctx_cat.spread_bucket", "spread_bucket"),
    "ctx_cat.atr_bucket": ("ctx_cat.atr_bucket", "atr_bucket"),
    "ctx_cat.H4_trend_sign_cat": ("ctx_cat.H4_trend_sign_cat", "H4_trend_sign_cat", "h4_trend_sign_cat"),
    "ctx_cont.atr_ratio_m5_m15": ("ctx_cont.atr_ratio_m5_m15", "atr_ratio_m5_m15"),
    "ctx_cont.atr_ratio_m15_d1": ("ctx_cont.atr_ratio_m15_d1", "atr_ratio_m15_d1"),
    "ctx_cont.atr_ratio_h1_d1": ("ctx_cont.atr_ratio_h1_d1", "atr_ratio_h1_d1"),
    "ctx_cont._v1h4_ema_diff": ("ctx_cont._v1h4_ema_diff", "_v1h4_ema_diff"),
    "ctx_cont._v1h4_slope5": ("ctx_cont._v1h4_slope5", "_v1h4_slope5"),
    "ctx_cont.d1_ema_slope_20_canon_v2": (
        "ctx_cont.d1_ema_slope_20_canon_v2",
        "d1_ema_slope_20_canon_v2",
    ),
    "ctx_cont.regime_tf_agreement_v3": ("ctx_cont.regime_tf_agreement_v3", "regime_tf_agreement_v3"),
    "ctx_cont.regime_stack_sum_v3": ("ctx_cont.regime_stack_sum_v3", "regime_stack_sum_v3"),
    "ctx_cont.regime_divergence_flag_v3": (
        "ctx_cont.regime_divergence_flag_v3",
        "regime_divergence_flag_v3",
    ),
    "ctx_cont.d1_dist_to_boundary_v3": ("ctx_cont.d1_dist_to_boundary_v3", "d1_dist_to_boundary_v3"),
    "ctx_cont.d1_dist_roc_288_v3": ("ctx_cont.d1_dist_roc_288_v3", "d1_dist_roc_288_v3"),
    "ctx_cont.d1_regime_changed_flag_v3": (
        "ctx_cont.d1_regime_changed_flag_v3",
        "d1_regime_changed_flag_v3",
    ),
    "ctx_cont.bars_since_d1_regime_change_v3": (
        "ctx_cont.bars_since_d1_regime_change_v3",
        "bars_since_d1_regime_change_v3",
    ),
    "ctx_cont.d1_trend_age_mature_flag_v3": (
        "ctx_cont.d1_trend_age_mature_flag_v3",
        "d1_trend_age_mature_flag_v3",
    ),
    "ctx_cont.m5_regime_class_id_v2": ("ctx_cont.m5_regime_class_id_v2", "m5_regime_class_id_v2"),
    "ctx_cont.m15_regime_class_id_v2": ("ctx_cont.m15_regime_class_id_v2", "m15_regime_class_id_v2"),
    "ctx_cont.h1_regime_class_id_v2": ("ctx_cont.h1_regime_class_id_v2", "h1_regime_class_id_v2"),
    "ctx_cont.h4_regime_class_id_v2": ("ctx_cont.h4_regime_class_id_v2", "h4_regime_class_id_v2"),
    "ctx_cont.d1_regime_class_id_v2": ("ctx_cont.d1_regime_class_id_v2", "d1_regime_class_id_v2"),
    "ctx_cont.h4_trend_age_bars_norm_v2": (
        "ctx_cont.h4_trend_age_bars_norm_v2",
        "h4_trend_age_bars_norm_v2",
    ),
    "ctx_cont.d1_trend_age_bars_norm_v2": (
        "ctx_cont.d1_trend_age_bars_norm_v2",
        "d1_trend_age_bars_norm_v2",
    ),
}


def _name_index(names: Iterable[str]) -> dict[str, int]:
    return {str(name): i for i, name in enumerate(names)}


def _aliases(name: str) -> tuple[str, ...]:
    return _SOURCE_ALIASES.get(name, (name,))


def _has(index: dict[str, int], name: str) -> bool:
    return any(alias in index for alias in _aliases(name))


def missing_session_regime_interaction_source_fields(feature_names: Iterable[str]) -> list[str]:
    available = {str(name) for name in feature_names}
    return [
        name
        for name in SESSION_REGIME_INTERACTION_SOURCE_FIELDS
        if not any(alias in available for alias in _aliases(name))
    ]


def _col(x: np.ndarray, index: dict[str, int], name: str, default: float = 0.0) -> np.ndarray:
    for alias in _aliases(name):
        if alias in index:
            arr = np.asarray(x[:, index[alias]], dtype=np.float32)
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


def _bucket_ge(arr: np.ndarray, threshold: int) -> np.ndarray:
    return (np.rint(np.nan_to_num(arr, nan=0.0)) >= int(threshold)).astype(np.float32)


def _bucket_le(arr: np.ndarray, threshold: int) -> np.ndarray:
    return (np.rint(np.nan_to_num(arr, nan=0.0)) <= int(threshold)).astype(np.float32)


def _regime_class_sign(arr: np.ndarray) -> np.ndarray:
    ids = np.rint(np.nan_to_num(arr, nan=0.0)).astype(np.int16, copy=False)
    return np.where(np.isin(ids, (1, 2)), 1.0, np.where(np.isin(ids, (3, 4)), -1.0, 0.0)).astype(np.float32)


def _h4_trend_cat_sign(arr: np.ndarray) -> np.ndarray:
    ids = np.rint(np.nan_to_num(arr, nan=1.0)).astype(np.int16, copy=False)
    return np.where(ids >= 2, 1.0, np.where(ids <= 0, -1.0, 0.0)).astype(np.float32)


def _class_vote_agreement(
    x: np.ndarray,
    idx: dict[str, int],
    names: tuple[str, str, str, str, str],
) -> tuple[np.ndarray, np.ndarray]:
    if not all(_has(idx, name) for name in names):
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
    *,
    fail_on_missing_sources: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Build causal session/regime interaction candidates from current fields."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise RuntimeError(f"session/regime input matrix must be 2D, got {x.shape}")
    idx = _name_index(feature_names)
    missing = missing_session_regime_interaction_source_fields(feature_names)
    if fail_on_missing_sources and missing:
        raise RuntimeError(
            "ENTRY_SESSION_REGIME_INTERACTION_MISSING_SOURCE_FIELDS: "
            f"{missing[:30]} total={len(missing)}"
        )
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
    session_age_progress = _clip01(minutes_since_open / 360.0)

    asia = _clip01(c("ctx_cont.is_ASIA"))
    asia_eu = _clip01(c("ctx_cont.is_asia_eu_overlap"))
    eu_us = _clip01(c("ctx_cont.is_eu_us_overlap"))
    eu = _clip01(c("ctx_cont.is_eu_only") + asia_eu + eu_us)
    us = _clip01(c("ctx_cont.is_us_only") + eu_us)
    overlap = _clip01(asia_eu + eu_us)
    active_session = _clip01(asia + eu + us)

    spread_ratio = _clip(_safe_ratio(c("ctx_cont.spread_bps"), c("ctx_cont.atr_bps", default=1.0)), 0.0, 5.0)
    spread_pressure = _clip01(_tanh(spread_ratio, scale=0.15))
    spread_bucket = c("ctx_cat.spread_bucket")
    spread_bucket_high = _bucket_ge(spread_bucket, 2)
    spread_bucket_medium_or_high = _bucket_ge(spread_bucket, 1)
    spread_bucket_pressure = _clip01(np.maximum(spread_pressure, 0.50 * spread_bucket_medium_or_high + 0.50 * spread_bucket_high))
    low_spread_permission = _clip01(1.0 - spread_pressure)
    bucket_low_spread_permission = _clip01((1.0 - spread_bucket_pressure) * low_spread_permission)
    atr_bucket = c("ctx_cat.atr_bucket")
    atr_bucket_low = _bucket_le(atr_bucket, 1)
    atr_bucket_mid_or_high = _bucket_ge(atr_bucket, 2)
    atr_bucket_high = _bucket_ge(atr_bucket, 3)
    atr_bucket_extreme = _bucket_ge(atr_bucket, 4)
    atr_bucket_pressure = _clip01(0.30 * atr_bucket_mid_or_high + 0.35 * atr_bucket_high + 0.35 * atr_bucket_extreme)
    spread_atr_bucket_pressure = _clip01(np.maximum(spread_bucket_pressure, spread_pressure) * np.maximum(atr_bucket_pressure, c("ctx_cont.D1_atr_percentile_252")))
    low_cost_mid_atr_permission = _clip01(bucket_low_spread_permission * (1.0 - atr_bucket_extreme) * (0.50 + 0.50 * atr_bucket_low))

    vol_m5 = _clip01(c("ctx_cont.vol_pct_m5_1yr"))
    vol_h1 = _clip01(c("ctx_cont.vol_pct_h1_1yr"))
    d1_atr_pct = _clip01(c("ctx_cont.D1_atr_percentile_252"))
    vol_regime = c("ctx_cat.vol_regime_id", default=1.0)
    vol_regime_high = _bucket_ge(vol_regime, 2)
    vol_regime_extreme = _bucket_ge(vol_regime, 3)
    vol_pressure = _clip01(0.45 * vol_m5 + 0.35 * vol_h1 + 0.20 * d1_atr_pct)
    vol_regime_pressure = _clip01(np.maximum(vol_pressure, 0.65 * vol_regime_high + 0.35 * vol_regime_extreme))
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
    d1_dist_roc = _tanh(c("ctx_cont.d1_dist_roc_288_v3"), scale=2.0)
    d1_regime_changed = _clip01(c("ctx_cont.d1_regime_changed_flag_v3"))
    bars_since_change_norm = _clip01(c("ctx_cont.bars_since_d1_regime_change_v3", default=1.0))
    d1_trend_mature = _clip01(c("ctx_cont.d1_trend_age_mature_flag_v3"))
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
    regime_stack_abs = _clip01(np.abs(regime_stack_signed))
    regime_stack_bull = _pos(regime_stack_signed) * mtf_agreement_pressure
    regime_stack_bear = _neg(regime_stack_signed) * mtf_agreement_pressure
    regime_stack_chop = _clip01((1.0 - regime_stack_abs) * (0.50 + mtf_divergence_pressure))

    h4_regime_sign = _regime_class_sign(c("ctx_cont.h4_regime_class_id_v2"))
    d1_regime_sign = _regime_class_sign(c("ctx_cont.d1_regime_class_id_v2"))
    h4_trend_cat_sign = _h4_trend_cat_sign(c("ctx_cat.H4_trend_sign_cat", default=1.0))
    h4_ema = _tanh(c("ctx_cont._v1h4_ema_diff"), scale=2.0)
    h4_slope = _tanh(c("ctx_cont._v1h4_slope5"), scale=2.0)
    d1_slope = _tanh(c("ctx_cont.d1_ema_slope_20_canon_v2"), scale=2.0)
    h4_age = _clip01(c("ctx_cont.h4_trend_age_bars_norm_v2"))
    d1_age = _clip01(c("ctx_cont.d1_trend_age_bars_norm_v2"))
    h4_d1_regime_agreement = ((h4_regime_sign == d1_regime_sign) & (np.abs(d1_regime_sign) > 0.0)).astype(np.float32)
    h4_d1_regime_mismatch = (h4_regime_sign * d1_regime_sign < 0.0).astype(np.float32)
    h4_d1_slope_coherence = _clip01(1.0 - 0.50 * np.abs(h4_slope - d1_slope))
    h4_d1_stack_score = _clip(
        0.22 * h4_regime_sign
        + 0.18 * h4_trend_cat_sign
        + 0.22 * d1_regime_sign
        + 0.18 * h4_ema
        + 0.10 * h4_slope
        + 0.10 * d1_slope,
        -1.0,
        1.0,
    )
    h4_d1_stack_coherence = _clip01(
        0.35 * h4_d1_regime_agreement
        + 0.25 * h4_d1_slope_coherence
        + 0.20 * mtf_agreement_pressure
        + 0.20 * (1.0 - h4_d1_regime_mismatch)
    )
    h4_d1_stack_bull = _pos(h4_d1_stack_score) * h4_d1_stack_coherence
    h4_d1_stack_bear = _neg(h4_d1_stack_score) * h4_d1_stack_coherence
    h4_d1_age_exhaustion = _clip01((0.45 * h4_age + 0.40 * d1_age + 0.15 * d1_trend_mature) * np.abs(h4_d1_stack_score))
    h4_d1_roc_reversal = _clip01(_neg(d1_dist_roc * h4_d1_stack_score) + 0.35 * h4_d1_regime_mismatch)
    regime_transition_abstain = _clip01(
        0.30 * regime_change_pressure
        + 0.25 * mtf_divergence_pressure
        + 0.20 * h4_d1_regime_mismatch
        + 0.15 * regime_boundary_pressure
        + 0.10 * h4_d1_roc_reversal
    )
    regime_transition_tail = _clip01(
        regime_transition_abstain
        * (0.35 + 0.25 * spread_bucket_pressure + 0.25 * atr_bucket_pressure + 0.15 * vol_regime_pressure)
    )

    structure_bias = _clip(c("chart.foundation_structure_up_minus_down") / 2.0, -1.0, 1.0)
    bos_balance = _clip(c("chart.foundation_bos_recent_balance"), -1.0, 1.0)
    choch_recent = _clip01(c("chart.foundation_choch_recent_tau24"))
    sweep_balance = _clip(c("chart.foundation_sweep_reclaim_balance_proxy") / 5.0, -1.0, 1.0)
    compression_release = _clip01(c("chart.foundation_compression_release_trigger") / 5.0)
    impulse_direction = _clip(c("chart.foundation_impulse_direction") / 2.0, -1.0, 1.0)
    pullback_depth = _clip01(c("chart.foundation_pullback_depth_norm"))
    structure_alignment = _clip(structure_bias * regime_stack_signed * mtf_agreement_pressure, -1.0, 1.0)
    structure_conflict = _clip01(np.abs(structure_bias) * mtf_divergence_pressure)
    ret1 = _tanh(c("ret_1"), scale=10.0)
    ret5 = _tanh(c("ret_5"), scale=20.0)
    ret20 = _tanh(c("ret_20"), scale=40.0)
    micro3 = _tanh(c("ctx_cont.micro_momentum_3"), scale=1.0)
    micro5 = _tanh(c("ctx_cont.micro_momentum_5"), scale=1.0)
    micro_accel = _tanh(c("ctx_cont.micro_acceleration"), scale=1.0)
    momentum_score = _clip(
        0.25 * ret1 + 0.30 * ret5 + 0.20 * ret20 + 0.15 * micro3 + 0.10 * micro5,
        -1.0,
        1.0,
    )
    momentum_abs = _clip01(np.abs(momentum_score))
    momentum_accel_alignment = _clip01(_pos(momentum_score * micro_accel))

    _add(arrays, names, "session_opening_risk", open_risk, lo=0.0, hi=1.0)
    _add(arrays, names, "session_boundary_risk", boundary_risk, lo=0.0, hi=1.0)
    _add(arrays, names, "session_mid_age_stability", mid_session_stability, lo=0.0, hi=1.0)
    _add(arrays, names, "session_age_progress_norm", session_age_progress, lo=0.0, hi=1.0)
    _add(arrays, names, "session_boundary_transition_pressure", boundary_transition, lo=0.0, hi=1.0)
    _add(arrays, names, "asia_eu_overlap_transition_pressure", asia_eu * (0.50 + vol_pressure) * (0.50 + boundary_transition), lo=0.0, hi=2.0)
    _add(arrays, names, "eu_us_overlap_momentum_continuation", eu_us * low_spread_permission * mtf_agreement_pressure * np.abs(impulse_direction), lo=0.0, hi=1.0)
    _add(arrays, names, "eu_us_overlap_divergence_risk", eu_us * mtf_divergence_pressure * (0.50 + vol_pressure), lo=0.0, hi=2.0)
    _add(arrays, names, "asia_open_boundary_fade_risk", asia * boundary_transition * (0.50 + spread_bucket_pressure + vol_regime_pressure), lo=0.0, hi=2.0)
    _add(arrays, names, "asia_mid_session_low_cost_permission", asia * mid_session_stability * low_cost_mid_atr_permission * (1.0 - vol_regime_pressure), lo=0.0, hi=1.0)
    _add(arrays, names, "eu_opening_structure_break_risk", eu * open_risk * np.abs(bos_balance) * (0.50 + spread_pressure + mtf_divergence_pressure), lo=0.0, hi=2.0)
    _add(arrays, names, "us_close_boundary_spread_tail_risk", us * boundary_risk * spread_atr_bucket_pressure * (0.50 + vol_regime_pressure), lo=0.0, hi=2.0)

    _add(arrays, names, "spread_cost_ratio", spread_ratio, lo=0.0, hi=5.0)
    _add(arrays, names, "spread_cost_pressure", spread_pressure, lo=0.0, hi=1.0)
    _add(arrays, names, "spread_cost_x_boundary_risk", spread_pressure * boundary_transition, lo=0.0, hi=1.0)
    _add(arrays, names, "spread_cost_x_vol_expansion", spread_pressure * vol_expansion_pressure, lo=0.0, hi=1.0)
    _add(arrays, names, "spread_cost_x_overlap_risk", spread_pressure * overlap * (0.50 + vol_pressure), lo=0.0, hi=2.0)
    _add(arrays, names, "spread_bucket_high_pressure", spread_bucket_pressure, lo=0.0, hi=1.0)
    _add(arrays, names, "spread_bucket_low_cost_permission", bucket_low_spread_permission, lo=0.0, hi=1.0)
    _add(arrays, names, "atr_bucket_high_pressure", atr_bucket_pressure, lo=0.0, hi=1.0)
    _add(arrays, names, "spread_atr_bucket_joint_pressure", spread_atr_bucket_pressure, lo=0.0, hi=1.0)
    _add(arrays, names, "spread_atr_boundary_abstain_score", spread_atr_bucket_pressure * boundary_transition, lo=0.0, hi=1.0)
    _add(arrays, names, "low_spread_mid_atr_execution_permission", low_cost_mid_atr_permission, lo=0.0, hi=1.0)
    _add(arrays, names, "vol_regime_high_pressure", vol_regime_pressure, lo=0.0, hi=1.0)
    _add(
        arrays,
        names,
        "vol_regime_extreme_tail_pressure",
        vol_regime_extreme * _clip01(0.40 * vol_expansion_pressure + 0.30 * spread_pressure + 0.30 * boundary_transition),
        lo=0.0,
        hi=1.0,
    )

    _add(arrays, names, "regime_persistence_score", regime_persistence, lo=0.0, hi=1.0)
    _add(arrays, names, "regime_change_pressure", regime_change_pressure, lo=0.0, hi=1.0)
    _add(arrays, names, "regime_change_x_session_boundary", regime_change_pressure * boundary_transition, lo=0.0, hi=1.0)
    _add(arrays, names, "mtf_regime_agreement_pressure", mtf_agreement_pressure, lo=0.0, hi=1.0)
    _add(arrays, names, "mtf_regime_divergence_pressure", mtf_divergence_pressure, lo=0.0, hi=1.0)
    _add(arrays, names, "mtf_regime_class_vote_agreement", class_vote_agreement, lo=0.0, hi=1.0)
    _add(arrays, names, "mtf_regime_short_long_mismatch", short_long_mismatch, lo=0.0, hi=1.0)
    _add(arrays, names, "regime_stack_bull_trend_pressure", regime_stack_bull, lo=0.0, hi=1.0)
    _add(arrays, names, "regime_stack_bear_trend_pressure", regime_stack_bear, lo=0.0, hi=1.0)
    _add(arrays, names, "regime_stack_chop_pressure", regime_stack_chop, lo=0.0, hi=1.0)
    _add(arrays, names, "h4_d1_regime_sign_agreement", h4_d1_regime_agreement, lo=0.0, hi=1.0)
    _add(arrays, names, "h4_d1_regime_sign_mismatch", h4_d1_regime_mismatch, lo=0.0, hi=1.0)
    _add(arrays, names, "h4_d1_trend_stack_score", h4_d1_stack_score, lo=-1.0, hi=1.0)
    _add(arrays, names, "h4_d1_stack_bull_pressure", h4_d1_stack_bull, lo=0.0, hi=1.0)
    _add(arrays, names, "h4_d1_stack_bear_pressure", h4_d1_stack_bear, lo=0.0, hi=1.0)
    _add(arrays, names, "h4_d1_trend_age_exhaustion_pressure", h4_d1_age_exhaustion, lo=0.0, hi=1.0)
    _add(arrays, names, "h4_d1_d1_roc_reversal_pressure", h4_d1_roc_reversal, lo=0.0, hi=1.0)
    _add(arrays, names, "regime_transition_abstain_score", regime_transition_abstain, lo=0.0, hi=1.0)
    _add(arrays, names, "regime_transition_tail_risk", regime_transition_tail, lo=0.0, hi=1.0)
    _add(arrays, names, "regime_transition_x_overlap_risk", overlap * regime_transition_tail * (0.50 + vol_pressure), lo=0.0, hi=2.0)

    _add(arrays, names, "asia_range_liquidity_reversal_pressure", asia * np.abs(sweep_balance) * (1.0 - np.abs(impulse_direction)) * (1.0 - mtf_agreement_pressure), lo=0.0, hi=1.0)
    _add(arrays, names, "eu_structure_breakout_readiness", eu * low_spread_permission * compression_release * mtf_agreement_pressure * np.abs(bos_balance), lo=0.0, hi=1.0)
    _add(arrays, names, "us_momentum_followthrough_pressure", us * low_spread_permission * mtf_agreement_pressure * impulse_direction, lo=-1.0, hi=1.0)
    _add(arrays, names, "overlap_liquidity_sweep_risk", overlap * np.abs(sweep_balance) * (0.50 + choch_recent) * (0.50 + mtf_divergence_pressure), lo=0.0, hi=2.0)
    _add(arrays, names, "session_vol_spread_breakout_readiness", active_session * compression_release * vol_expansion_pressure * low_spread_permission * (1.0 - boundary_transition), lo=0.0, hi=1.0)
    _add(arrays, names, "session_vol_spread_tail_risk", active_session * compression_release * vol_expansion_pressure * _clip01(spread_pressure + boundary_transition + mtf_divergence_pressure), lo=0.0, hi=1.0)
    _add(arrays, names, "session_structure_regime_alignment", active_session * mid_session_stability * structure_alignment, lo=-1.0, hi=1.0)
    _add(arrays, names, "session_structure_regime_conflict", active_session * structure_conflict * (0.50 + pullback_depth), lo=0.0, hi=2.0)
    _add(arrays, names, "asia_momentum_fade_pressure", asia * momentum_abs * (1.0 - mtf_agreement_pressure) * (0.50 + np.abs(sweep_balance)), lo=0.0, hi=2.0)
    _add(arrays, names, "eu_signed_momentum_followthrough", eu * momentum_score * low_spread_permission * mtf_agreement_pressure * (0.50 + vol_regime_pressure), lo=-1.0, hi=1.0)
    _add(arrays, names, "us_signed_momentum_followthrough", us * momentum_score * low_spread_permission * mtf_agreement_pressure * (0.50 + momentum_accel_alignment), lo=-1.0, hi=1.0)
    _add(arrays, names, "overlap_momentum_vol_spread_risk", overlap * momentum_abs * vol_expansion_pressure * _clip01(spread_pressure + boundary_transition), lo=0.0, hi=1.0)
    _add(arrays, names, "asia_structure_mean_reversion_pressure", asia * np.abs(structure_bias) * (1.0 - mtf_agreement_pressure) * (0.50 + np.abs(sweep_balance)), lo=0.0, hi=2.0)
    _add(arrays, names, "eu_structure_regime_continuation_bias", eu * structure_alignment * bucket_low_spread_permission, lo=-1.0, hi=1.0)
    _add(arrays, names, "us_late_session_structure_chase_risk", us * momentum_abs * np.abs(structure_bias) * _clip01(boundary_risk + spread_bucket_pressure + vol_regime_pressure), lo=0.0, hi=1.0)
    _add(arrays, names, "session_momentum_structure_alignment_score", active_session * momentum_score * structure_bias * mtf_agreement_pressure, lo=-1.0, hi=1.0)
    _add(arrays, names, "asia_h4_d1_liquidity_reversal_risk", asia * np.abs(sweep_balance) * h4_d1_roc_reversal * (0.50 + regime_transition_abstain), lo=0.0, hi=2.0)
    _add(arrays, names, "eu_h4_d1_structure_continuation_long", eu * h4_d1_stack_bull * _pos(structure_bias) * bucket_low_spread_permission, lo=0.0, hi=1.0)
    _add(arrays, names, "eu_h4_d1_structure_continuation_short", eu * h4_d1_stack_bear * _neg(structure_bias) * bucket_low_spread_permission, lo=0.0, hi=1.0)
    _add(arrays, names, "us_h4_d1_momentum_followthrough", us * momentum_score * h4_d1_stack_score * low_spread_permission * (1.0 - regime_transition_abstain), lo=-1.0, hi=1.0)
    _add(arrays, names, "overlap_h4_d1_liquidity_tail_risk", overlap * np.abs(sweep_balance) * regime_transition_tail * (0.50 + spread_atr_bucket_pressure), lo=0.0, hi=2.0)
    _add(arrays, names, "session_trend_structure_liquidity_long_score", active_session * h4_d1_stack_bull * _pos(structure_bias) * _pos(sweep_balance) * low_cost_mid_atr_permission, lo=0.0, hi=1.0)
    _add(arrays, names, "session_trend_structure_liquidity_short_score", active_session * h4_d1_stack_bear * _neg(structure_bias) * _neg(sweep_balance) * low_cost_mid_atr_permission, lo=0.0, hi=1.0)

    out = np.column_stack(arrays).astype(np.float32, copy=False) if arrays else np.empty((x.shape[0], 0), dtype=np.float32)
    if not np.isfinite(out).all():
        raise RuntimeError("session/regime interaction layer contains non-finite values")
    if len(set(names)) != len(names):
        dupes = sorted({name for name in names if names.count(name) > 1})
        raise RuntimeError(f"session/regime interaction layer has duplicate names: {dupes[:10]}")
    return out, names


SESSION_REGIME_INTERACTION_FEATURE_NAMES = tuple(
    name
    for name in build_entry_session_regime_interaction_layer(
        np.zeros((1, 0), dtype=np.float32),
        [],
        fail_on_missing_sources=False,
    )[1]
)
