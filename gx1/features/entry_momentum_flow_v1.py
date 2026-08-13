"""Canonical Entry momentum/flow specialist features.

This layer builds deterministic, closed-bar momentum derivatives from exact
snap/ctx/candle/chart inputs used by the model-native feature stack.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


MOMENTUM_FLOW_FEATURE_VERSION = (
    "entry_momentum_flow_v3_20260813_removed_candle_vote_inputs"
)
MOMENTUM_FLOW_FEATURE_PREFIX = "momentum.flow_"

MOMENTUM_FLOW_SOURCE_FIELDS = (
    "snap.ret_1",
    "snap.ret_5",
    "snap.ret_20",
    "snap._v1_clv",
    "ctx_cont.micro_momentum_3",
    "ctx_cont.micro_momentum_5",
    "ctx_cont.micro_acceleration",
    "ctx_cont.atr_bps",
    "snap.atr_z",
    "snap._v1_range_z",
    "ctx_cont._v1h1_slope5",
    "ctx_cont._v1h4_slope5",
    "ctx_cont.d1_pct_change_5_canon_v2",
    "ctx_cont.d1_ema_slope_20_canon_v2",
    "ctx_cont.regime_tf_agreement_v3",
    "ctx_cont.regime_divergence_flag_v3",
    "ctx_cont.dip_confirmed_m5_v3",
    "ctx_cont.dip_confirmed_m15_v3",
    "ctx_cont.dip_confirmed_h1_v3",
    "ctx_cont.dip_confirmed_h4_v3",
    "ctx_cont.dip_confirmed_d1_v3",
    "ctx_cont.dip_confirmed_mean_5tf",
    "ctx_cont.dip_confirmed_max_5tf",
    "ctx_cont.dip_proximity_h1_v3",
    "ctx_cont.dip_proximity_h4_v3",
    "ctx_cont.dip_proximity_d1_v3",
    "ctx_cont.dip_proximity_mean_h1h4d1",
    "chart.foundation_impulse_direction",
    "chart.foundation_impulse_pullback_alignment",
    "chart.foundation_compression_release_up",
    "chart.foundation_compression_release_down",
    # V30 package 7 (2026-08-13): the four aggregate candle VOTES
    # (`candle.pattern_bull/bear_continuation_pressure`,
    # `candle.pattern_bull/bear_reversal_pressure`) were removed from their
    # producer as hand-written OR-sums of already-emitted candle columns
    # (rule 4, the mtf_confluence class).  They are therefore no longer
    # declared here.
    "candle.pattern_body_share",
    "candle.pattern_upper_wick_share",
    "candle.pattern_lower_wick_share",
)

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
    "flow_divergence_pressure",
    "mtf_bull_confirmation",
    "mtf_bear_confirmation",
    "bodyflow_bull_pressure",
    "bodyflow_bear_pressure",
    "bull_exhaustion_pressure",
    "bear_exhaustion_pressure",
    "dip_continuation_long_input",
    "dip_continuation_short_input",
    "bad_path_long_initial_pressure",
    "bad_path_short_initial_pressure",
    "impulse_pullback_alignment_score",
    "compression_release_followthrough_score",
    "clean_edge_score",
)

MOMENTUM_FLOW_FEATURE_NAMES = tuple(
    f"{MOMENTUM_FLOW_FEATURE_PREFIX}{name}" for name in MOMENTUM_FLOW_FEATURE_SUFFIXES
)


def _name_index(names: Iterable[str]) -> dict[str, int]:
    return {str(name): i for i, name in enumerate(names)}


def missing_entry_momentum_flow_source_fields(feature_names: Iterable[str]) -> list[str]:
    """Return exact canonical source fields absent from ``feature_names``."""
    available = {str(name) for name in feature_names}
    return [name for name in MOMENTUM_FLOW_SOURCE_FIELDS if name not in available]


def _col(x: np.ndarray, index: dict[str, int], name: str) -> np.ndarray:
    if name not in index:
        raise RuntimeError(f"momentum flow required source field missing: {name}")
    return np.asarray(x[:, index[name]], dtype=np.float32)


def _clip(arr: np.ndarray, lo: float = -25.0, hi: float = 25.0) -> np.ndarray:
    return np.clip(arr, lo, hi).astype(np.float32, copy=False)


def _clip01(arr: np.ndarray) -> np.ndarray:
    return _clip(arr, 0.0, 1.0)


def _pos(arr: np.ndarray) -> np.ndarray:
    return np.maximum(arr, 0.0).astype(np.float32, copy=False)


def _neg(arr: np.ndarray) -> np.ndarray:
    return np.maximum(-arr, 0.0).astype(np.float32, copy=False)


def _safe_scale(*parts: np.ndarray, floor: float = 1.0) -> np.ndarray:
    stack = np.vstack([np.abs(np.asarray(part, dtype=np.float32)) for part in parts])
    return np.maximum(np.max(stack, axis=0), float(floor)).astype(np.float32, copy=False)


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


def _add(
    arrays: list[np.ndarray],
    names: list[str],
    suffix: str,
    arr: np.ndarray,
    *,
    lo: float = -25.0,
    hi: float = 25.0,
) -> None:
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
    if x.shape[0] == 0:
        raise RuntimeError("momentum flow input matrix must contain at least one row")
    if x.shape[1] != len(feature_names):
        raise RuntimeError(
            f"momentum flow feature name count {len(feature_names)} does not match matrix width {x.shape[1]}"
        )
    if len(feature_names) != len(set(feature_names)):
        duplicates = sorted({name for name in feature_names if feature_names.count(name) > 1})
        raise RuntimeError(f"momentum flow duplicate feature names: {duplicates[:10]}")
    missing = missing_entry_momentum_flow_source_fields(feature_names)
    if missing:
        raise RuntimeError(f"momentum flow required source fields missing: {missing}")
    if not np.isfinite(x).all():
        bad_rows, bad_cols = np.where(~np.isfinite(x))
        examples = [
            {"row": int(row), "feature": feature_names[int(col)]}
            for row, col in zip(bad_rows[:10], bad_cols[:10])
        ]
        raise RuntimeError(f"momentum flow input matrix contains non-finite values: {examples}")
    idx = _name_index(feature_names)

    def c(name: str) -> np.ndarray:
        return _col(x, idx, name)

    ret1 = c("snap.ret_1")
    ret5 = c("snap.ret_5")
    ret20 = c("snap.ret_20")
    # basic_v1 emits _v1_clv in [0, 1] with declared neutral 0.5 (close at the
    # bar low -> 0.0, at the bar high -> 1.0, degenerate range -> 0.5).
    # Recenter to a signed [-1, 1] conviction so _pos/_neg split bull/bear
    # location instead of treating every row as bullish.
    clv = _tanh(2.0 * (c("snap._v1_clv") - 0.5), scale=1.0)
    mom3 = c("ctx_cont.micro_momentum_3")
    mom5 = c("ctx_cont.micro_momentum_5")
    micro_accel = c("ctx_cont.micro_acceleration")
    atr_bps = c("ctx_cont.atr_bps")
    atr_z = _tanh(c("snap.atr_z"), scale=3.0)
    range_z = _tanh(c("snap._v1_range_z"), scale=3.0)
    upper_wick_share = _clip01(c("candle.pattern_upper_wick_share"))
    lower_wick_share = _clip01(c("candle.pattern_lower_wick_share"))
    upper_wick_flow = upper_wick_share
    lower_wick_flow = lower_wick_share

    vol_scale = _safe_scale(
        ret1,
        ret5 / np.sqrt(5.0),
        ret20 / np.sqrt(20.0),
        atr_bps,
        floor=1.0,
    )
    # Normalize returns by the ATR term only: including |ret| itself in the
    # scale bounds |ret_norm| by tanh(1) and flattens the tail. atr_bps is the
    # family's declared bps-volatility owner; sqrt-horizon scaling and the
    # 1.0 bps floor follow the vol_scale convention above.
    atr_scale = np.maximum(atr_bps, 1.0)
    ret1_norm = _tanh(ret1, atr_scale)
    ret5_norm = _tanh(ret5, atr_scale * np.sqrt(5.0))
    ret20_norm = _tanh(ret20, atr_scale * np.sqrt(20.0))

    # micro_momentum_3/5 are bps (micro_structure_v1: close.diff(k)/close*1e4),
    # so the floor adopts the same 1.0 bps floor as atr_scale above.
    micro_scale = _safe_scale(mom3, mom5 / np.sqrt(5.0), floor=1.0)
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

    # Unit owner: htf_features.py. _v1h1_slope5/_v1h4_slope5 are true 5-bar
    # slopes and d1_ema_slope_20_canon_v2 is an EMA slope, all in ATR-multiple
    # units (O(1)), so scale=2.0 is dimensionally sane for them.
    h1_flow = _tanh(c("ctx_cont._v1h1_slope5"), scale=2.0)
    h4_flow = _tanh(c("ctx_cont._v1h4_slope5"), scale=2.0)
    # d1_pct_change_5_canon_v2 is bps; scale=25.0 saturates near ~100 bps and
    # is a candidate for a TRAIN-fitted scale in the next recipe decision.
    d1_ret = _tanh(c("ctx_cont.d1_pct_change_5_canon_v2"), scale=25.0)
    d1_slope = _tanh(c("ctx_cont.d1_ema_slope_20_canon_v2"), scale=2.0)
    mtf_confirmation, mtf_conflict, mtf_bull, mtf_bear = _sign_agreement(
        [ret5_norm, ret20_norm, h1_flow, h4_flow, d1_ret + d1_slope]
    )
    regime_agreement = _clip01(c("ctx_cont.regime_tf_agreement_v3"))
    regime_divergence = _clip01(c("ctx_cont.regime_divergence_flag_v3"))
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

    body_share = _clip01(c("candle.pattern_body_share"))
    body_impulse = _clip01(0.55 * body_share + 0.45 * np.abs(clv))
    clv_body_bull_flow = _clip01(
        0.40 * _pos(clv) * (0.50 + 0.50 * body_impulse)
        + 0.24 * lower_wick_flow
        + 0.18 * _pos(return_acceleration)
        + 0.18 * _pos(micro_impulse)
    )
    clv_body_bear_flow = _clip01(
        0.40 * _neg(clv) * (0.50 + 0.50 * body_impulse)
        + 0.24 * upper_wick_flow
        + 0.18 * _neg(return_acceleration)
        + 0.18 * _neg(micro_impulse)
    )
    # V30 package 7 (2026-08-13): the two pre-fused candle-vote terms
    # (0.24 x `*_continuation_pressure` + 0.16 x `*_reversal_pressure`) are
    # gone with their producer.  The surviving weights are kept EXACTLY as they
    # were: renormalizing them to restore a 1.0 ceiling would invent a
    # magnitude with no origin (rule 2b), while dropping a term is removing an
    # exception.  What remains is what the field's name claims — body/CLV flow
    # plus the 1- and 5-bar vol-normalized returns.  RULE-4: none of the candle
    # evidence is lost to the model; every column the two votes summed
    # (`bullish/bearish_engulfing_quality`, `tweezer_bottom/top_quality`,
    # `morning_star_bull/evening_star_bear_quality`,
    # `three_bar_bull/bear_reversal_quality`, `bull/bear_rejection_continuation_score`,
    # `bullish/bearish_engulfing_followthrough_quality`,
    # `three_bar_bull/bear_continuation_quality`, `outside_bar_quality`,
    # `bull_lower/bear_upper_wick_rejection_quality`) is emitted by the
    # candlestick layer inside its MANDATORY smart3 suffix, so the model sees
    # the same evidence unfused on every row.
    body_flow_bull = _clip01(
        0.30 * clv_body_bull_flow
        + 0.16 * _pos(ret1_norm)
        + 0.14 * _pos(ret5_norm)
    )
    body_flow_bear = _clip01(
        0.30 * clv_body_bear_flow
        + 0.16 * _neg(ret1_norm)
        + 0.14 * _neg(ret5_norm)
    )
    short_long_opposition = _clip01(_pos(ret1_norm) * _neg(ret20_norm) + _neg(ret1_norm) * _pos(ret20_norm))
    impulse_clv_conflict = _clip01(_pos(impulse_direction) * _neg(clv) + _neg(impulse_direction) * _pos(clv))
    accel_impulse_conflict = _clip01(
        _pos(impulse_direction) * _neg(acceleration) + _neg(impulse_direction) * _pos(acceleration)
    )
    flow_divergence = _clip01(
        0.34 * mtf_conflict
        + 0.20 * regime_divergence
        + 0.18 * short_long_opposition
        + 0.16 * impulse_clv_conflict
        + 0.12 * accel_impulse_conflict
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
        + 0.15 * body_flow_bull
    )
    bear_follow_raw = _clip01(
        0.25 * _neg(ret1_norm)
        + 0.30 * _neg(ret5_norm)
        + 0.15 * _neg(ret20_norm)
        + 0.15 * _neg(micro_impulse)
        + 0.15 * body_flow_bear
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
            + 0.18 * body_flow_bear
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
            + 0.18 * body_flow_bull
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
                + 0.15 * body_flow_bear
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
                + 0.15 * body_flow_bull
                + 0.10 * _pos(clv)
            )
        )
        + 0.30 * bullish_reversal_pressure
    )

    dip_confirmed = _mean01(
        [
            c("ctx_cont.dip_confirmed_m5_v3"),
            c("ctx_cont.dip_confirmed_m15_v3"),
            c("ctx_cont.dip_confirmed_h1_v3"),
            c("ctx_cont.dip_confirmed_h4_v3"),
            c("ctx_cont.dip_confirmed_d1_v3"),
            c("ctx_cont.dip_confirmed_mean_5tf"),
            c("ctx_cont.dip_confirmed_max_5tf"),
        ]
    )
    dip_proximity = _mean01(
        [
            c("ctx_cont.dip_proximity_h1_v3"),
            c("ctx_cont.dip_proximity_h4_v3"),
            c("ctx_cont.dip_proximity_d1_v3"),
            c("ctx_cont.dip_proximity_mean_h1h4d1"),
        ]
    )
    dip_setup = _clip01(0.55 * dip_confirmed + 0.45 * dip_proximity)
    dip_cont_long = _clip01(
        dip_setup * (0.45 * bull_follow + 0.30 * mtf_bull + 0.25 * (1.0 - bull_exhaustion))
    )
    dip_cont_short = _clip01(
        dip_setup * (0.45 * bear_follow + 0.30 * mtf_bear + 0.25 * (1.0 - bear_exhaustion))
    )
    bad_path_long = _clip01(
        0.22 * bear_follow
        + 0.18 * body_flow_bear
        + 0.17 * bull_exhaustion
        + 0.16 * flow_divergence
        + 0.12 * high_vol_stress
        + 0.09 * _neg(return_acceleration)
        + 0.06 * upper_wick_flow
    )
    bad_path_short = _clip01(
        0.22 * bull_follow
        + 0.18 * body_flow_bull
        + 0.17 * bear_exhaustion
        + 0.16 * flow_divergence
        + 0.12 * high_vol_stress
        + 0.09 * _pos(return_acceleration)
        + 0.06 * lower_wick_flow
    )
    bad_path_long = _clip01(
        0.70 * bad_path_long
        + 0.30
        * dip_setup
        * (0.40 * bear_follow + 0.25 * flow_divergence + 0.20 * bull_exhaustion + 0.15 * body_flow_bear)
    )
    bad_path_short = _clip01(
        0.70 * bad_path_short
        + 0.30
        * dip_setup
        * (0.40 * bull_follow + 0.25 * flow_divergence + 0.20 * bear_exhaustion + 0.15 * body_flow_bull)
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
        * (1.0 - flow_divergence)
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
    _add(arrays, names, "flow_divergence_pressure", flow_divergence, lo=0.0, hi=1.0)
    _add(arrays, names, "mtf_bull_confirmation", mtf_bull, lo=0.0, hi=1.0)
    _add(arrays, names, "mtf_bear_confirmation", mtf_bear, lo=0.0, hi=1.0)
    _add(arrays, names, "bodyflow_bull_pressure", body_flow_bull, lo=0.0, hi=1.0)
    _add(arrays, names, "bodyflow_bear_pressure", body_flow_bear, lo=0.0, hi=1.0)
    _add(arrays, names, "bull_exhaustion_pressure", bull_exhaustion, lo=0.0, hi=1.0)
    _add(arrays, names, "bear_exhaustion_pressure", bear_exhaustion, lo=0.0, hi=1.0)
    _add(arrays, names, "dip_continuation_long_input", dip_cont_long, lo=0.0, hi=1.0)
    _add(arrays, names, "dip_continuation_short_input", dip_cont_short, lo=0.0, hi=1.0)
    _add(arrays, names, "bad_path_long_initial_pressure", bad_path_long, lo=0.0, hi=1.0)
    _add(arrays, names, "bad_path_short_initial_pressure", bad_path_short, lo=0.0, hi=1.0)
    _add(arrays, names, "impulse_pullback_alignment_score", impulse_pullback_alignment, lo=-1.0, hi=1.0)
    _add(arrays, names, "compression_release_followthrough_score", compression_release_follow, lo=-1.0, hi=1.0)
    _add(arrays, names, "clean_edge_score", clean_edge, lo=0.0, hi=1.0)

    out = (
        np.column_stack(arrays).astype(np.float32, copy=False)
        if arrays
        else np.empty((x.shape[0], 0), dtype=np.float32)
    )
    if tuple(names) != MOMENTUM_FLOW_FEATURE_NAMES:
        raise RuntimeError("momentum flow feature order drifted")
    if not np.isfinite(out).all():
        raise RuntimeError("momentum flow layer contains non-finite values")
    return out, names
