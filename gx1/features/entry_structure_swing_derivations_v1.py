"""Causal structure/swing quality derivations for Entry research gates.

This module is intentionally not wired into the active seq146/seq215 manifests.
It derives higher-order structure/swing signals from already materialized
closed-bar ``chart.*`` and ``ctx_cont.*`` fields so the next foundation rebuild
can add them behind the normal feature, specialist and readiness audits.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


STRUCTURE_SWING_DERIVATION_FEATURE_VERSION = "entry_structure_swing_derivations_v1_20260630_causal_quality_state_v2"
STRUCTURE_SWING_DERIVATION_FEATURE_PREFIX = "chart.structure_swing_"

STRUCTURE_TFS = ("m5", "m15", "h1", "h4", "d1")

STRUCTURE_SWING_DERIVATION_SOURCE_FIELDS = (
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
    "chart.foundation_choch_age_bars",
    "chart.foundation_choch_recent_tau24",
    "chart.foundation_impulse_direction",
    "chart.foundation_impulse_age_proxy",
    "chart.foundation_pullback_phase_up",
    "chart.foundation_pullback_phase_down",
    "chart.foundation_pullback_depth_norm",
    "chart.foundation_impulse_pullback_alignment",
    "chart.foundation_compression_state",
    "chart.foundation_expansion_state",
    "chart.foundation_compression_release_trigger",
    "ctx_cont.struct_tf_agree_count_v3",
    *tuple(f"ctx_cont.struct_continuation_up_{tf}_v3" for tf in STRUCTURE_TFS),
    *tuple(f"ctx_cont.struct_pullback_in_uptrend_{tf}_v3" for tf in STRUCTURE_TFS),
    *tuple(f"ctx_cont.struct_continuation_down_{tf}_v3" for tf in STRUCTURE_TFS),
    *tuple(f"ctx_cont.struct_bounce_in_downtrend_{tf}_v3" for tf in STRUCTURE_TFS),
    *tuple(f"ctx_cont.struct_pullback_depth_{tf}_v3" for tf in STRUCTURE_TFS),
)


def _name_index(names: Iterable[str]) -> dict[str, int]:
    return {str(name): i for i, name in enumerate(names)}


def missing_structure_swing_derivation_source_fields(feature_names: Iterable[str]) -> list[str]:
    available = {str(name) for name in feature_names}
    return [name for name in STRUCTURE_SWING_DERIVATION_SOURCE_FIELDS if name not in available]


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


def _lag1(arr: np.ndarray) -> np.ndarray:
    out = np.empty_like(arr, dtype=np.float32)
    if arr.size:
        out[0] = 0.0
        out[1:] = arr[:-1]
    return out


def _age_recency(age: np.ndarray, tau: float = 24.0) -> np.ndarray:
    return np.exp(-np.maximum(age, 0.0) / max(float(tau), 1e-6)).astype(np.float32)


def _depth_quality(depth: np.ndarray) -> np.ndarray:
    return np.exp(-np.abs(_clip01(depth) - 0.50) * 4.0).astype(np.float32)


def _mean_fields(c, names: Iterable[str]) -> np.ndarray:
    fields = [_clip01(c(name)) for name in names]
    if not fields:
        return np.zeros(0, dtype=np.float32)
    return np.vstack(fields).mean(axis=0).astype(np.float32)


def _add(arrays: list[np.ndarray], names: list[str], name: str, arr: np.ndarray, *, lo: float = -25.0, hi: float = 25.0) -> None:
    clean = _clip(np.asarray(arr, dtype=np.float32), lo, hi)
    if clean.ndim != 1:
        raise RuntimeError(f"structure/swing derivation {name} is not 1D: {clean.shape}")
    if not np.isfinite(clean).all():
        raise RuntimeError(f"structure/swing derivation {name} contains non-finite values")
    arrays.append(clean)
    names.append(f"{STRUCTURE_SWING_DERIVATION_FEATURE_PREFIX}{name}")


def build_entry_structure_swing_derivation_layer(
    x: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Build deterministic, closed-bar structure/swing quality features."""
    x = np.asarray(x, dtype=np.float32)
    idx = _name_index(feature_names)
    arrays: list[np.ndarray] = []
    names: list[str] = []

    def c(name: str, default: float = 0.0) -> np.ndarray:
        return _col(x, idx, name, default=default)

    hh = _clip01(c("chart.foundation_hh_state"))
    hl = _clip01(c("chart.foundation_hl_state"))
    lh = _clip01(c("chart.foundation_lh_state"))
    ll = _clip01(c("chart.foundation_ll_state"))
    structure_balance_raw = _clip(c("chart.foundation_structure_up_minus_down"), -2.0, 2.0)
    structure_balance = _clip(0.50 * structure_balance_raw + 0.50 * ((hh + hl) - (lh + ll)), -2.0, 2.0)
    up_structure = _clip01(0.35 * hh + 0.35 * hl + 0.30 * _pos(structure_balance * 0.5))
    down_structure = _clip01(0.35 * lh + 0.35 * ll + 0.30 * _neg(structure_balance * 0.5))
    hh_hl_consistency_up = _clip01(
        np.minimum(hh, hl)
        * (1.0 - 0.35 * np.minimum(lh, ll))
        * (0.75 + 0.25 * _clip01(_pos(structure_balance) * 0.5))
    )
    lh_ll_consistency_down = _clip01(
        np.minimum(lh, ll)
        * (1.0 - 0.35 * np.minimum(hh, hl))
        * (0.75 + 0.25 * _clip01(_neg(structure_balance) * 0.5))
    )
    hh_hl_lh_ll_conflict = _clip01(
        0.65 * np.minimum(up_structure, down_structure)
        + 0.35 * np.minimum(np.minimum(hh, hl), np.minimum(lh, ll))
    )

    bos_up_recent = _clip01(
        c("chart.foundation_bos_up_recent_tau24")
        + 0.25 * _age_recency(c("chart.foundation_bos_up_age_bars", default=96.0))
    )
    bos_down_recent = _clip01(
        c("chart.foundation_bos_down_recent_tau24")
        + 0.25 * _age_recency(c("chart.foundation_bos_down_age_bars", default=96.0))
    )
    bos_balance = _clip(c("chart.foundation_bos_recent_balance") + bos_up_recent - bos_down_recent, -2.0, 2.0)
    choch_recent = _clip01(
        c("chart.foundation_choch_recent_tau24")
        + 0.25 * _age_recency(c("chart.foundation_choch_age_bars", default=96.0))
    )

    impulse_direction = _clip(c("chart.foundation_impulse_direction"), -2.0, 2.0)
    impulse_age = _clip01(c("chart.foundation_impulse_age_proxy"))
    pullback_phase_up = _clip01(c("chart.foundation_pullback_phase_up"))
    pullback_phase_down = _clip01(c("chart.foundation_pullback_phase_down"))
    pullback_depth = _clip01(c("chart.foundation_pullback_depth_norm"))
    impulse_pullback_alignment = _clip(c("chart.foundation_impulse_pullback_alignment"), -2.0, 2.0)
    compression = _clip01(c("chart.foundation_compression_state"))
    expansion = _clip01(c("chart.foundation_expansion_state"))
    release = _clip01(c("chart.foundation_compression_release_trigger"))

    mtf_up_cont = _mean_fields(c, (f"ctx_cont.struct_continuation_up_{tf}_v3" for tf in STRUCTURE_TFS))
    mtf_up_pullback = _mean_fields(c, (f"ctx_cont.struct_pullback_in_uptrend_{tf}_v3" for tf in STRUCTURE_TFS))
    mtf_down_cont = _mean_fields(c, (f"ctx_cont.struct_continuation_down_{tf}_v3" for tf in STRUCTURE_TFS))
    mtf_down_pullback = _mean_fields(c, (f"ctx_cont.struct_bounce_in_downtrend_{tf}_v3" for tf in STRUCTURE_TFS))
    mtf_depth = _mean_fields(c, (f"ctx_cont.struct_pullback_depth_{tf}_v3" for tf in STRUCTURE_TFS))
    mtf_up = _clip01(0.65 * mtf_up_cont + 0.35 * mtf_up_pullback)
    mtf_down = _clip01(0.65 * mtf_down_cont + 0.35 * mtf_down_pullback)
    tf_agree_count = _clip01(c("ctx_cont.struct_tf_agree_count_v3") / 5.0)
    mtf_agreement = _clip01(0.60 * np.maximum(mtf_up, mtf_down) + 0.40 * tf_agree_count)
    mtf_divergence = _clip01((2.0 * np.minimum(mtf_up, mtf_down)) * 0.65 + (1.0 - tf_agree_count) * np.maximum(mtf_up, mtf_down) * 0.35)

    combined_depth = _clip01(0.55 * pullback_depth + 0.45 * mtf_depth)
    depth_quality = _depth_quality(combined_depth)
    structure_delta = _clip(structure_balance - _lag1(structure_balance), -2.0, 2.0)
    expansion_quality = _clip01(0.55 * expansion + 0.45 * release)
    bos_choch_recency_alignment_up = _clip01(
        bos_up_recent
        * (1.0 - 0.45 * choch_recent)
        * (0.50 + 0.30 * hh_hl_consistency_up + 0.20 * mtf_up)
    )
    bos_choch_recency_alignment_down = _clip01(
        bos_down_recent
        * (1.0 - 0.45 * choch_recent)
        * (0.50 + 0.30 * lh_ll_consistency_down + 0.20 * mtf_down)
    )
    bos_choch_recency_conflict = _clip01(
        choch_recent
        * (
            0.35 * np.maximum(bos_up_recent, bos_down_recent)
            + 0.30 * hh_hl_lh_ll_conflict
            + 0.20 * mtf_divergence
            + 0.15 * (1.0 - mtf_agreement)
        )
    )
    pullback_depth_phase_alignment_up = _clip01(
        pullback_phase_up
        * depth_quality
        * (0.50 * hh_hl_consistency_up + 0.30 * mtf_up + 0.20 * _pos(impulse_pullback_alignment))
    )
    pullback_depth_phase_alignment_down = _clip01(
        pullback_phase_down
        * depth_quality
        * (0.50 * lh_ll_consistency_down + 0.30 * mtf_down + 0.20 * _neg(impulse_pullback_alignment))
    )
    break_confirmation_up = _clip01(
        bos_up_recent
        * expansion_quality
        * (
            0.35 * hh_hl_consistency_up
            + 0.25 * mtf_up
            + 0.20 * _pos(structure_delta)
            + 0.20 * _pos(impulse_direction)
        )
        * (1.0 - 0.35 * choch_recent)
    )
    break_confirmation_down = _clip01(
        bos_down_recent
        * expansion_quality
        * (
            0.35 * lh_ll_consistency_down
            + 0.25 * mtf_down
            + 0.20 * _neg(structure_delta)
            + 0.20 * _neg(impulse_direction)
        )
        * (1.0 - 0.35 * choch_recent)
    )

    swing_leg_quality_up = _clip01(
        up_structure
        * (0.25 + 0.25 * bos_up_recent + 0.25 * mtf_up + 0.15 * impulse_age + 0.10 * depth_quality)
        * (1.0 - 0.35 * choch_recent)
    )
    swing_leg_quality_down = _clip01(
        down_structure
        * (0.25 + 0.25 * bos_down_recent + 0.25 * mtf_down + 0.15 * impulse_age + 0.10 * depth_quality)
        * (1.0 - 0.35 * choch_recent)
    )
    bos_followthrough_up = _clip01(
        bos_up_recent
        * (0.45 * up_structure + 0.30 * mtf_up + 0.25 * _pos(structure_delta))
        * (0.70 + 0.30 * expansion_quality)
        * (1.0 - 0.35 * choch_recent)
    )
    bos_followthrough_down = _clip01(
        bos_down_recent
        * (0.45 * down_structure + 0.30 * mtf_down + 0.25 * _neg(structure_delta))
        * (0.70 + 0.30 * expansion_quality)
        * (1.0 - 0.35 * choch_recent)
    )
    choch_failure_up = _clip01(
        choch_recent
        * (0.45 * up_structure + 0.30 * mtf_up + 0.15 * bos_up_recent + 0.10 * _pos(impulse_direction))
        * (1.0 - _clip01(0.50 * down_structure + 0.30 * bos_down_recent + 0.20 * mtf_down))
    )
    choch_failure_down = _clip01(
        choch_recent
        * (0.45 * down_structure + 0.30 * mtf_down + 0.15 * bos_down_recent + 0.10 * _neg(impulse_direction))
        * (1.0 - _clip01(0.50 * up_structure + 0.30 * bos_up_recent + 0.20 * mtf_up))
    )
    pullback_continuation_up = _clip01(
        pullback_phase_up
        * depth_quality
        * (0.45 * up_structure + 0.35 * mtf_up + 0.20 * _pos(impulse_pullback_alignment))
        * (1.0 - 0.25 * choch_failure_down)
    )
    pullback_continuation_down = _clip01(
        pullback_phase_down
        * depth_quality
        * (0.45 * down_structure + 0.35 * mtf_down + 0.20 * _neg(impulse_pullback_alignment))
        * (1.0 - 0.25 * choch_failure_up)
    )
    regime_state = _clip(
        0.45 * structure_balance
        + 0.25 * impulse_direction
        + 0.20 * (mtf_up - mtf_down)
        + 0.10 * bos_balance,
        -2.0,
        2.0,
    )
    regime_confidence = _clip01(
        (np.abs(regime_state) / 2.0)
        * (0.50 + 0.50 * mtf_agreement)
        * (1.0 - 0.35 * mtf_divergence)
    )
    neutral_structure = 1.0 - _clip01(np.abs(regime_state) / 2.0)
    no_recent_break = 1.0 - _clip01(np.maximum(bos_up_recent, bos_down_recent))
    compression_pressure = _clip01(
        (0.35 * compression + 0.25 * neutral_structure + 0.20 * mtf_divergence + 0.20 * depth_quality)
        * (0.50 + 0.50 * no_recent_break)
        * (1.0 - 0.25 * release)
    )
    swing_compression_setup = _clip01(
        (
            0.35 * compression
            + 0.20 * neutral_structure
            + 0.20 * hh_hl_lh_ll_conflict
            + 0.15 * mtf_divergence
            + 0.10 * depth_quality
        )
        * (0.55 + 0.45 * no_recent_break)
        * (1.0 - 0.30 * expansion_quality)
    )

    _add(arrays, names, "hh_hl_consistency_up", hh_hl_consistency_up, lo=0.0, hi=1.0)
    _add(arrays, names, "lh_ll_consistency_down", lh_ll_consistency_down, lo=0.0, hi=1.0)
    _add(arrays, names, "hh_hl_lh_ll_conflict", hh_hl_lh_ll_conflict, lo=0.0, hi=1.0)
    _add(arrays, names, "swing_leg_quality_up", swing_leg_quality_up, lo=0.0, hi=1.0)
    _add(arrays, names, "swing_leg_quality_down", swing_leg_quality_down, lo=0.0, hi=1.0)
    _add(arrays, names, "swing_leg_quality_balance", swing_leg_quality_up - swing_leg_quality_down, lo=-1.0, hi=1.0)
    _add(arrays, names, "bos_choch_recency_alignment_up", bos_choch_recency_alignment_up, lo=0.0, hi=1.0)
    _add(arrays, names, "bos_choch_recency_alignment_down", bos_choch_recency_alignment_down, lo=0.0, hi=1.0)
    _add(arrays, names, "bos_choch_recency_conflict", bos_choch_recency_conflict, lo=0.0, hi=1.0)
    _add(arrays, names, "bos_followthrough_up_quality", bos_followthrough_up, lo=0.0, hi=1.0)
    _add(arrays, names, "bos_followthrough_down_quality", bos_followthrough_down, lo=0.0, hi=1.0)
    _add(arrays, names, "bos_followthrough_balance", bos_followthrough_up - bos_followthrough_down, lo=-1.0, hi=1.0)
    _add(arrays, names, "break_confirmation_up", break_confirmation_up, lo=0.0, hi=1.0)
    _add(arrays, names, "break_confirmation_down", break_confirmation_down, lo=0.0, hi=1.0)
    _add(arrays, names, "break_confirmation_balance", break_confirmation_up - break_confirmation_down, lo=-1.0, hi=1.0)
    _add(arrays, names, "choch_failure_up_risk", choch_failure_up, lo=0.0, hi=1.0)
    _add(arrays, names, "choch_failure_down_risk", choch_failure_down, lo=0.0, hi=1.0)
    _add(arrays, names, "pullback_depth_quality", depth_quality, lo=0.0, hi=1.0)
    _add(arrays, names, "pullback_depth_phase_alignment_up", pullback_depth_phase_alignment_up, lo=0.0, hi=1.0)
    _add(arrays, names, "pullback_depth_phase_alignment_down", pullback_depth_phase_alignment_down, lo=0.0, hi=1.0)
    _add(arrays, names, "pullback_phase_continuation_up", pullback_continuation_up, lo=0.0, hi=1.0)
    _add(arrays, names, "pullback_phase_continuation_down", pullback_continuation_down, lo=0.0, hi=1.0)
    _add(arrays, names, "market_structure_regime_state", regime_state, lo=-2.0, hi=2.0)
    _add(arrays, names, "market_structure_regime_confidence", regime_confidence, lo=0.0, hi=1.0)
    _add(arrays, names, "structure_compression_pressure", compression_pressure, lo=0.0, hi=1.0)
    _add(arrays, names, "swing_compression_setup", swing_compression_setup, lo=0.0, hi=1.0)
    _add(arrays, names, "mtf_structure_agreement", mtf_agreement, lo=0.0, hi=1.0)
    _add(arrays, names, "mtf_structure_divergence", mtf_divergence, lo=0.0, hi=1.0)

    out = np.column_stack(arrays).astype(np.float32, copy=False) if arrays else np.empty((x.shape[0], 0), dtype=np.float32)
    if not np.isfinite(out).all():
        raise RuntimeError("structure/swing derivation layer contains non-finite values")
    if len(set(names)) != len(names):
        dupes = sorted({name for name in names if names.count(name) > 1})
        raise RuntimeError(f"structure/swing derivation layer has duplicate names: {dupes[:10]}")
    return out, names


STRUCTURE_SWING_DERIVATION_FEATURE_NAMES = tuple(
    name for name in build_entry_structure_swing_derivation_layer(np.zeros((1, 0), dtype=np.float32), [])[1]
)
