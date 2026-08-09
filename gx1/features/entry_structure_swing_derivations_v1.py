"""Strict causal structure/swing derivations for model-native Entry."""
from __future__ import annotations

from typing import Iterable

import numpy as np

from gx1.features.htf_features import MULTI_TF_TIMEFRAMES_LOWER

STRUCTURE_SWING_DERIVATION_FEATURE_VERSION = (
    "entry_structure_swing_derivations_v2_20260809_producer_domain_recency_raw_depth_failclosed"
)
STRUCTURE_SWING_DERIVATION_FEATURE_PREFIX = "chart.structure_swing_"

STRUCTURE_TFS = MULTI_TF_TIMEFRAMES_LOWER

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
    "ctx_cont.dist_last_swing_high_atr",
    "ctx_cont.dist_last_swing_low_atr",
    "ctx_cont.bars_since_swing_high",
    "ctx_cont.bars_since_swing_low",
    "ctx_cont.struct_tf_agree_count_v3",
    *tuple(f"ctx_cont.struct_continuation_up_{tf}_v3" for tf in STRUCTURE_TFS),
    *tuple(f"ctx_cont.struct_pullback_in_uptrend_{tf}_v3" for tf in STRUCTURE_TFS),
    *tuple(f"ctx_cont.struct_continuation_down_{tf}_v3" for tf in STRUCTURE_TFS),
    *tuple(f"ctx_cont.struct_bounce_in_downtrend_{tf}_v3" for tf in STRUCTURE_TFS),
    *tuple(f"ctx_cont.struct_pullback_depth_{tf}_v3" for tf in STRUCTURE_TFS),
)

STRUCTURE_SWING_DERIVATION_FEATURE_SUFFIXES = (
    "hh_hl_consistency_up",
    "lh_ll_consistency_down",
    "hh_hl_lh_ll_conflict",
    "swing_leg_quality_up",
    "swing_leg_quality_down",
    "swing_leg_quality_balance",
    "bos_choch_recency_alignment_up",
    "bos_choch_recency_alignment_down",
    "bos_choch_recency_conflict",
    "bos_followthrough_up_quality",
    "bos_followthrough_down_quality",
    "bos_followthrough_balance",
    "break_confirmation_up",
    "break_confirmation_down",
    "break_confirmation_balance",
    "choch_failure_up_risk",
    "choch_failure_down_risk",
    "pullback_depth_quality",
    "pullback_depth_phase_alignment_up",
    "pullback_depth_phase_alignment_down",
    "pullback_phase_continuation_up",
    "pullback_phase_continuation_down",
    "market_structure_regime_state",
    "market_structure_regime_confidence",
    "structure_compression_pressure",
    "swing_compression_setup",
    "mtf_structure_agreement",
    "mtf_structure_divergence",
)

STRUCTURE_SWING_DERIVATION_FEATURE_NAMES = tuple(
    f"{STRUCTURE_SWING_DERIVATION_FEATURE_PREFIX}{suffix}"
    for suffix in STRUCTURE_SWING_DERIVATION_FEATURE_SUFFIXES
)


def _name_index(names: Iterable[str]) -> dict[str, int]:
    return {name: i for i, name in enumerate(names)}


def missing_structure_swing_derivation_source_fields(feature_names: Iterable[str]) -> list[str]:
    available = set(feature_names)
    return [name for name in STRUCTURE_SWING_DERIVATION_SOURCE_FIELDS if name not in available]


def _col(x: np.ndarray, index: dict[str, int], name: str) -> np.ndarray:
    if name not in index:
        raise RuntimeError(f"structure/swing derivation required source field missing: {name}")
    arr = np.asarray(x[:, index[name]], dtype=np.float32)
    if not np.isfinite(arr).all():
        raise RuntimeError(f"STRUCTURE_SWING_DERIVATION_SOURCE_NONFINITE: {name}")
    return arr


def _clip(arr: np.ndarray, lo: float = -25.0, hi: float = 25.0) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float32)
    if not np.isfinite(values).all():
        raise RuntimeError("STRUCTURE_SWING_DERIVATION_DERIVATION_NONFINITE")
    return np.clip(values, lo, hi).astype(np.float32, copy=False)


def _clip01(arr: np.ndarray) -> np.ndarray:
    return _clip(arr, 0.0, 1.0)


def _pos(arr: np.ndarray) -> np.ndarray:
    return np.maximum(arr, 0.0).astype(np.float32, copy=False)


def _neg(arr: np.ndarray) -> np.ndarray:
    return np.maximum(-arr, 0.0).astype(np.float32, copy=False)


def _prox_abs(arr: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.abs(np.asarray(arr, dtype=np.float32)))).astype(np.float32, copy=False)


def _lag1(arr: np.ndarray) -> np.ndarray:
    out = np.empty_like(arr, dtype=np.float32)
    if arr.size:
        out[0] = 0.0
        out[1:] = arr[:-1]
    return out


def _age_recency(age: np.ndarray, tau: float = 24.0) -> np.ndarray:
    return np.exp(-np.maximum(age, 0.0) / max(float(tau), 1e-6)).astype(np.float32)


def _recency(arr: np.ndarray) -> np.ndarray:
    # One recency convention per field: this is the sibling owner's exact
    # transform (entry_foundation_structure_v1._recency, 1/(1+bars)) already
    # applied to the same bars_since_swing_* sources there.
    return (1.0 / (1.0 + np.maximum(arr, 0.0))).astype(np.float32, copy=False)


def _mean_fields(c, names: Iterable[str]) -> np.ndarray:
    fields = [_clip01(c(name)) for name in names]
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
    if x.ndim != 2:
        raise RuntimeError(f"STRUCTURE_SWING_DERIVATION_INPUT_NOT_2D: shape={x.shape}")
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise RuntimeError(f"STRUCTURE_SWING_DERIVATION_INPUT_EMPTY: shape={x.shape}")
    if x.shape[1] != len(feature_names):
        raise RuntimeError(
            "STRUCTURE_SWING_DERIVATION_FEATURE_NAME_DIM_MISMATCH: "
            f"cols={x.shape[1]} names={len(feature_names)}"
        )
    if any(not isinstance(name, str) or not name for name in feature_names):
        raise RuntimeError("STRUCTURE_SWING_DERIVATION_FEATURE_NAME_INVALID")
    if len(feature_names) != len(set(feature_names)):
        seen: set[str] = set()
        duplicates = sorted(
            {name for name in feature_names if name in seen or seen.add(name)}
        )
        raise RuntimeError(
            f"STRUCTURE_SWING_DERIVATION_DUPLICATE_FEATURE_NAMES: {duplicates[:20]}"
        )
    idx = _name_index(feature_names)
    missing = missing_structure_swing_derivation_source_fields(feature_names)
    if missing:
        raise RuntimeError(f"structure/swing derivation required source fields missing: {missing}")
    if not np.isfinite(x).all():
        bad_rows, bad_cols = np.where(~np.isfinite(x))
        examples = [
            {"row": int(row), "feature": feature_names[int(col)]}
            for row, col in zip(bad_rows[:10], bad_cols[:10])
        ]
        raise RuntimeError(f"STRUCTURE_SWING_DERIVATION_SOURCE_NONFINITE: {examples}")
    arrays: list[np.ndarray] = []
    names: list[str] = []

    def c(name: str) -> np.ndarray:
        return _col(x, idx, name)

    hh = _clip01(c("chart.foundation_hh_state"))
    hl = _clip01(c("chart.foundation_hl_state"))
    lh = _clip01(c("chart.foundation_lh_state"))
    ll = _clip01(c("chart.foundation_ll_state"))
    # Single clipped field: the former 0.5/0.5 blend was a no-op because its
    # second term recomputed the producer expression of
    # chart.foundation_structure_up_minus_down ((hh+hl)-(lh+ll)) bit-identically
    # from the same float32 columns. Output is bit-identical to the blend.
    structure_balance = _clip(c("chart.foundation_structure_up_minus_down"), -2.0, 2.0)
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
        + 0.25 * _age_recency(c("chart.foundation_bos_up_age_bars"))
    )
    bos_down_recent = _clip01(
        c("chart.foundation_bos_down_recent_tau24")
        + 0.25 * _age_recency(c("chart.foundation_bos_down_age_bars"))
    )
    bos_balance = _clip(c("chart.foundation_bos_recent_balance") + bos_up_recent - bos_down_recent, -2.0, 2.0)
    choch_recent = _clip01(
        c("chart.foundation_choch_recent_tau24")
        + 0.25 * _age_recency(c("chart.foundation_choch_age_bars"))
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
    # Producer domain: augment_forward_outcome_v2 emits struct_tf_agree_count_v3
    # as the mean of five per-TF binary pullback flags, already in [0,1]. The
    # former /5.0 double-divided a mean as if it were a count, collapsing this
    # gate to [0,0.2] and mtf_divergence's (1-count) gate to [0.8,1.0].
    tf_agree_count = _clip01(c("ctx_cont.struct_tf_agree_count_v3"))
    mtf_agreement = _clip01(0.60 * np.maximum(mtf_up, mtf_down) + 0.40 * tf_agree_count)
    mtf_divergence = _clip01((2.0 * np.minimum(mtf_up, mtf_down)) * 0.65 + (1.0 - tf_agree_count) * np.maximum(mtf_up, mtf_down) * 0.35)

    dist_high = _clip(c("ctx_cont.dist_last_swing_high_atr"), -8.0, 8.0)
    dist_low = _clip(c("ctx_cont.dist_last_swing_low_atr"), -8.0, 8.0)
    swing_high_proximity = _clip01(_prox_abs(dist_high))
    swing_low_proximity = _clip01(_prox_abs(dist_low))
    # Sibling-owner recency convention (1/(1+bars)): with SWING_LOOKBACK_V1=2
    # the typical bars_since_swing_* is 2-6 bars, so the former tau=48
    # exponential squashed these into [0.78,0.96]; 1/(1+bars) spans ~[0.14,0.33]
    # over the same typical ages and matches foundation's transform on the
    # exact same fields.
    swing_high_recent = _clip01(_recency(c("ctx_cont.bars_since_swing_high")))
    swing_low_recent = _clip01(_recency(c("ctx_cont.bars_since_swing_low")))
    swing_break_up = _clip01(_pos(dist_high) / 2.0)
    swing_break_down = _clip01(_neg(dist_low) / 2.0)
    swing_room_up = _clip01(_neg(dist_high) / 4.0)
    swing_room_down = _clip01(_pos(dist_low) / 4.0)
    # The 0.25-weighted recency term now contributes ~[0.035,0.083] for the
    # typical 2-6-bar swing ages (was a near-constant ~[0.20,0.24] under tau=48).
    swing_boundary_pressure = _clip01(
        0.45 * np.maximum(swing_high_proximity, swing_low_proximity)
        + 0.25 * np.maximum(swing_high_recent, swing_low_recent)
        + 0.30 * np.maximum(swing_break_up, swing_break_down)
    )
    # The 0.20-weighted proximity*recency products below now vary with swing
    # age instead of the recency factor sitting near 1 (tau=48 saturation).
    swing_distance_confirmation_up = _clip01(
        0.40 * swing_break_up
        + 0.25 * swing_room_up
        + 0.20 * swing_low_proximity * swing_low_recent
        + 0.15 * (1.0 - swing_high_proximity)
    )
    swing_distance_confirmation_down = _clip01(
        0.40 * swing_break_down
        + 0.25 * swing_room_down
        + 0.20 * swing_high_proximity * swing_high_recent
        + 0.15 * (1.0 - swing_low_proximity)
    )

    combined_depth = _clip01(0.55 * pullback_depth + 0.45 * mtf_depth)
    # Raw clipped depth passthrough. The former exp(-|depth-0.50|*4.0) shape
    # hand-authored a Fibonacci 50%-retracement prior with an unsourced 4.0
    # sharpness; Fib patterns are OOT-refuted in the project record. The model
    # learns which depth is good (rule-4 precedent: remove the hand-written
    # vote, keep the underlying evidence). Column name/count unchanged.
    depth_quality = combined_depth
    structure_delta = _clip(structure_balance - _lag1(structure_balance), -2.0, 2.0)
    expansion_quality = _clip01(0.55 * expansion + 0.45 * release)
    continuation_pressure_up = _clip01(
        0.28 * bos_up_recent
        + 0.24 * mtf_up
        + 0.20 * hh_hl_consistency_up
        + 0.16 * swing_distance_confirmation_up
        + 0.12 * _pos(impulse_direction)
    )
    continuation_pressure_down = _clip01(
        0.28 * bos_down_recent
        + 0.24 * mtf_down
        + 0.20 * lh_ll_consistency_down
        + 0.16 * swing_distance_confirmation_down
        + 0.12 * _neg(impulse_direction)
    )
    long_invalidation_pressure = _clip01(
        0.34 * bos_down_recent
        + 0.24 * choch_recent
        + 0.18 * mtf_down
        + 0.14 * swing_break_down
        + 0.10 * _neg(structure_delta)
    )
    short_invalidation_pressure = _clip01(
        0.34 * bos_up_recent
        + 0.24 * choch_recent
        + 0.18 * mtf_up
        + 0.14 * swing_break_up
        + 0.10 * _pos(structure_delta)
    )
    bos_choch_recency_alignment_up = _clip01(
        bos_up_recent
        * (1.0 - 0.45 * choch_recent)
        * (0.48 + 0.25 * hh_hl_consistency_up + 0.20 * mtf_up + 0.10 * swing_distance_confirmation_up)
        * (1.0 - 0.25 * long_invalidation_pressure)
    )
    bos_choch_recency_alignment_down = _clip01(
        bos_down_recent
        * (1.0 - 0.45 * choch_recent)
        * (0.48 + 0.25 * lh_ll_consistency_down + 0.20 * mtf_down + 0.10 * swing_distance_confirmation_down)
        * (1.0 - 0.25 * short_invalidation_pressure)
    )
    bos_choch_recency_conflict = _clip01(
        0.75
        * choch_recent
        * (
            0.30 * np.maximum(bos_up_recent, bos_down_recent)
            + 0.25 * hh_hl_lh_ll_conflict
            + 0.18 * mtf_divergence
            + 0.12 * (1.0 - mtf_agreement)
            + 0.15 * np.maximum(long_invalidation_pressure, short_invalidation_pressure)
        )
        + 0.25 * np.minimum(continuation_pressure_up, continuation_pressure_down)
    )
    pullback_depth_phase_alignment_up = _clip01(
        pullback_phase_up
        * depth_quality
        * (
            0.44 * hh_hl_consistency_up
            + 0.26 * mtf_up
            + 0.16 * _pos(impulse_pullback_alignment)
            + 0.14 * swing_distance_confirmation_up
        )
        * (1.0 - 0.20 * long_invalidation_pressure)
    )
    pullback_depth_phase_alignment_down = _clip01(
        pullback_phase_down
        * depth_quality
        * (
            0.44 * lh_ll_consistency_down
            + 0.26 * mtf_down
            + 0.16 * _neg(impulse_pullback_alignment)
            + 0.14 * swing_distance_confirmation_down
        )
        * (1.0 - 0.20 * short_invalidation_pressure)
    )
    break_confirmation_up = _clip01(
        bos_up_recent
        * expansion_quality
        * (
            0.30 * hh_hl_consistency_up
            + 0.22 * mtf_up
            + 0.18 * _pos(structure_delta)
            + 0.18 * _pos(impulse_direction)
            + 0.12 * swing_distance_confirmation_up
        )
        * (1.0 - 0.35 * choch_recent)
        * (1.0 - 0.25 * long_invalidation_pressure)
    )
    break_confirmation_down = _clip01(
        bos_down_recent
        * expansion_quality
        * (
            0.30 * lh_ll_consistency_down
            + 0.22 * mtf_down
            + 0.18 * _neg(structure_delta)
            + 0.18 * _neg(impulse_direction)
            + 0.12 * swing_distance_confirmation_down
        )
        * (1.0 - 0.35 * choch_recent)
        * (1.0 - 0.25 * short_invalidation_pressure)
    )

    swing_leg_quality_up = _clip01(
        up_structure
        * (
            0.20
            + 0.22 * bos_up_recent
            + 0.22 * mtf_up
            + 0.13 * impulse_age
            + 0.10 * depth_quality
            + 0.13 * swing_distance_confirmation_up
        )
        * (1.0 - 0.35 * choch_recent)
        * (1.0 - 0.25 * long_invalidation_pressure)
    )
    swing_leg_quality_down = _clip01(
        down_structure
        * (
            0.20
            + 0.22 * bos_down_recent
            + 0.22 * mtf_down
            + 0.13 * impulse_age
            + 0.10 * depth_quality
            + 0.13 * swing_distance_confirmation_down
        )
        * (1.0 - 0.35 * choch_recent)
        * (1.0 - 0.25 * short_invalidation_pressure)
    )
    bos_followthrough_up = _clip01(
        bos_up_recent
        * (
            0.38 * up_structure
            + 0.25 * mtf_up
            + 0.20 * _pos(structure_delta)
            + 0.17 * swing_distance_confirmation_up
        )
        * (0.70 + 0.30 * expansion_quality)
        * (1.0 - 0.35 * choch_recent)
        * (1.0 - 0.25 * long_invalidation_pressure)
    )
    bos_followthrough_down = _clip01(
        bos_down_recent
        * (
            0.38 * down_structure
            + 0.25 * mtf_down
            + 0.20 * _neg(structure_delta)
            + 0.17 * swing_distance_confirmation_down
        )
        * (0.70 + 0.30 * expansion_quality)
        * (1.0 - 0.35 * choch_recent)
        * (1.0 - 0.25 * short_invalidation_pressure)
    )
    choch_failure_up = _clip01(
        choch_recent
        * (
            0.38 * up_structure
            + 0.25 * mtf_up
            + 0.14 * bos_up_recent
            + 0.10 * _pos(impulse_direction)
            + 0.13 * swing_distance_confirmation_up
        )
        * (1.0 - _clip01(0.50 * down_structure + 0.30 * bos_down_recent + 0.20 * mtf_down))
    )
    choch_failure_down = _clip01(
        choch_recent
        * (
            0.38 * down_structure
            + 0.25 * mtf_down
            + 0.14 * bos_down_recent
            + 0.10 * _neg(impulse_direction)
            + 0.13 * swing_distance_confirmation_down
        )
        * (1.0 - _clip01(0.50 * up_structure + 0.30 * bos_up_recent + 0.20 * mtf_up))
    )
    pullback_continuation_up = _clip01(
        pullback_phase_up
        * depth_quality
        * (
            0.38 * up_structure
            + 0.29 * mtf_up
            + 0.17 * _pos(impulse_pullback_alignment)
            + 0.16 * swing_distance_confirmation_up
        )
        * (1.0 - 0.25 * choch_failure_down)
        * (1.0 - 0.20 * long_invalidation_pressure)
    )
    pullback_continuation_down = _clip01(
        pullback_phase_down
        * depth_quality
        * (
            0.38 * down_structure
            + 0.29 * mtf_down
            + 0.17 * _neg(impulse_pullback_alignment)
            + 0.16 * swing_distance_confirmation_down
        )
        * (1.0 - 0.25 * choch_failure_up)
        * (1.0 - 0.20 * short_invalidation_pressure)
    )
    regime_state = _clip(
        0.42 * structure_balance
        + 0.24 * impulse_direction
        + 0.18 * (mtf_up - mtf_down)
        + 0.08 * bos_balance
        + 0.08 * (continuation_pressure_up - continuation_pressure_down),
        -2.0,
        2.0,
    )
    dominant_invalidation = np.where(regime_state >= 0.0, long_invalidation_pressure, short_invalidation_pressure)
    regime_confidence = _clip01(
        (np.abs(regime_state) / 2.0)
        * (0.50 + 0.50 * mtf_agreement)
        * (1.0 - 0.35 * mtf_divergence)
        * (1.0 - 0.25 * dominant_invalidation)
    )
    neutral_structure = 1.0 - _clip01(np.abs(regime_state) / 2.0)
    no_recent_break = 1.0 - _clip01(np.maximum(bos_up_recent, bos_down_recent))
    compression_pressure = _clip01(
        (
            0.30 * compression
            + 0.22 * neutral_structure
            + 0.18 * mtf_divergence
            + 0.16 * depth_quality
            + 0.14 * swing_boundary_pressure
        )
        * (0.50 + 0.50 * no_recent_break)
        * (1.0 - 0.25 * release)
    )
    swing_compression_setup = _clip01(
        (
            0.30 * compression
            + 0.17 * neutral_structure
            + 0.18 * hh_hl_lh_ll_conflict
            + 0.13 * mtf_divergence
            + 0.10 * depth_quality
            + 0.12 * swing_boundary_pressure
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
    if tuple(names) != STRUCTURE_SWING_DERIVATION_FEATURE_NAMES:
        raise RuntimeError("structure/swing derivation feature order drifted")
    if not np.isfinite(out).all():
        raise RuntimeError("structure/swing derivation layer contains non-finite values")
    if len(set(names)) != len(names):
        dupes = sorted({name for name in names if names.count(name) > 1})
        raise RuntimeError(f"structure/swing derivation layer has duplicate names: {dupes[:10]}")
    return out, names
