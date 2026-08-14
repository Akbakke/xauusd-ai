"""Canonical Entry volatility/compression specialist features.

This layer derives deterministic, closed-bar volatility-regime features from
exact, already materialized Entry snap/context/chart fields.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from gx1.features.entry_volatility_semantics_v1 import (
    atr_ratio_compression_pressure,
    bollinger_squeeze_pressure,
    center_cross_tf_atr_ratio,
    pk_sigma20_per_bar_bps,
    rvol_20_per_bar_bps,
    vol_per_bar_bps_pressure,
)


VOL_COMPRESSION_FEATURE_VERSION = "entry_vol_compression_v4_20260813_cross_tf_scaling_law_center_and_volatility_unit_repair"
VOL_COMPRESSION_FEATURE_PREFIX = "vol_compression."

VOL_COMPRESSION_SOURCE_FIELDS = (
    "snap.atr_z",
    "snap.rvol_20",
    "snap._v1_range_z",
    "snap._v1_bb_squeeze_20_2",
    "snap.vol_ratio_5_20",
    "snap.signed_vol_z_20",
    "snap._v1_pk_sigma20",
    "snap._v1_bb_bandwidth_delta_10",
    "snap._v1_kurt_r",
    "snap.vol_z_20",
    "snap.vol_pct_96",
    "snap.ret_1",
    "snap.ret_5",
    "snap.ret_20",
    "ctx_cont.H1_range_compression_ratio",
    "ctx_cont.M15_range_compression_ratio",
    "ctx_cont.D1_atr_percentile_252",
    "ctx_cont.d1_range_z_20_canon_v2",
    "ctx_cont.m15_range_z_20_canon_v2",
    "ctx_cont.atr_ratio_m5_m15",
    "ctx_cont.atr_ratio_m5_h4",
    "ctx_cont.atr_ratio_m15_d1",
    "ctx_cont.atr_ratio_h1_d1",
    "ctx_cont.vol_pct_m5_1yr",
    "ctx_cont.vol_pct_h1_1yr",
    "ctx_cat.atr_bucket",
    "ctx_cat.vol_regime_id",
    "ctx_cont._v1h1_ema_diff",
    "ctx_cont._v1h4_ema_diff",
    "ctx_cont.d1_ema_slope_20_canon_v2",
    "ctx_cont.m15_trend_sign_canon_v2",
    "ctx_cont.regime_tf_agreement_v3",
    "ctx_cont.regime_divergence_flag_v3",
    "chart.foundation_compression_state",
    "chart.foundation_expansion_state",
    "chart.foundation_compression_release_trigger",
    "chart.foundation_compression_release_up",
    "chart.foundation_compression_release_down",
    "chart.foundation_impulse_direction",
)

VOL_COMPRESSION_FEATURE_SUFFIXES = (
    "atr_percentile_blend",
    "atr_percentile_m5_minus_h1",
    "low_atr_percentile_pressure",
    "high_atr_percentile_pressure",
    "squeeze_state_score",
    "range_compression_stack",
    "compression_agreement_score",
    "compression_divergence_score",
    "compression_depth_score",
    "compression_persistence_score",
    "compression_release_impulse",
    "compression_release_up_pressure",
    "compression_release_down_pressure",
    "expansion_state_score",
    "expansion_acceleration_score",
    "expansion_direction_score",
    "expansion_direction_confidence",
    "high_vol_tail_risk",
    "high_vol_chop_tail_risk",
    "low_vol_breakout_setup",
    "low_vol_false_breakout_risk",
    "mtf_vol_agreement_score",
    "mtf_vol_divergence_score",
    "short_tf_vol_expansion_pressure",
    "higher_tf_vol_expansion_pressure",
    "squeeze_breakout_asym_score",
    "vol_term_structure_slope",
    "vol_forecast_confidence_score",
)

VOL_COMPRESSION_FEATURE_NAMES = tuple(
    f"{VOL_COMPRESSION_FEATURE_PREFIX}{suffix}" for suffix in VOL_COMPRESSION_FEATURE_SUFFIXES
)


def _name_index(names: Iterable[str]) -> dict[str, int]:
    return {str(name): i for i, name in enumerate(names)}


def missing_entry_vol_compression_source_fields(feature_names: Iterable[str]) -> list[str]:
    """Return exact canonical source fields absent from ``feature_names``."""
    available = {str(name) for name in feature_names}
    return [name for name in VOL_COMPRESSION_SOURCE_FIELDS if name not in available]


def _col(x: np.ndarray, index: dict[str, int], name: str) -> np.ndarray:
    if name not in index:
        raise RuntimeError(f"vol/compression required source field missing: {name}")
    return np.asarray(x[:, index[name]], dtype=np.float32)


def _clip(arr: np.ndarray, lo: float = -25.0, hi: float = 25.0) -> np.ndarray:
    return np.clip(arr, lo, hi).astype(np.float32, copy=False)


def _clip01(arr: np.ndarray) -> np.ndarray:
    return _clip(arr, 0.0, 1.0)


def _pos(arr: np.ndarray) -> np.ndarray:
    return np.maximum(arr, 0.0).astype(np.float32, copy=False)


def _neg(arr: np.ndarray) -> np.ndarray:
    return np.maximum(-arr, 0.0).astype(np.float32, copy=False)


def _tanh(arr: np.ndarray, scale: float = 1.0) -> np.ndarray:
    return np.tanh(np.asarray(arr, dtype=np.float32) / max(float(scale), 1e-6)).astype(np.float32, copy=False)


def _unit_from_z(arr: np.ndarray, scale: float = 2.5) -> np.ndarray:
    return _clip01(0.5 + 0.5 * _tanh(arr, scale=scale))


def _lag1(arr: np.ndarray) -> np.ndarray:
    out = np.empty_like(arr, dtype=np.float32)
    if arr.size:
        out[0] = 0.0
        out[1:] = arr[:-1]
    return out


def _ctx_cat_domain_max(field: str) -> float:
    """Return the max categorical code for ``field`` from the domain owner.

    Deferred import: ``gx1.contracts.entry_model_native_signal_v1`` imports the
    feature-layers registry, which imports this module for its name tuple, so a
    module-level import here would be circular.  By the time the builder runs
    the contract module is fully initialized.
    """

    from gx1.contracts.entry_model_native_signal_v1 import (
        MODEL_NATIVE_CTX_CAT_DOMAINS,
    )

    domain = MODEL_NATIVE_CTX_CAT_DOMAINS[field]
    # ``len(domain) - 1`` equals the max code only for contiguous zero-based
    # domains; verify that instead of silently mis-scaling if the owner drifts.
    if tuple(domain) != tuple(range(len(domain))):
        raise RuntimeError(
            f"vol/compression ctx_cat domain not contiguous from zero: {field}"
        )
    return float(len(domain) - 1)


def _dispersion01(parts: list[np.ndarray]) -> np.ndarray:
    stack = np.vstack([_clip01(part) for part in parts])
    center = stack.mean(axis=0)
    return _clip01(np.abs(stack - center).mean(axis=0) * 2.0)


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
        raise RuntimeError(f"vol/compression feature {suffix} is not 1D: {clean.shape}")
    if not np.isfinite(clean).all():
        raise RuntimeError(f"vol/compression feature {suffix} contains non-finite values")
    arrays.append(clean)
    names.append(f"{VOL_COMPRESSION_FEATURE_PREFIX}{suffix}")


def build_entry_vol_compression_layer(
    x: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Build deterministic volatility/compression specialist features.

    Rows must be ordered by as-of time when release acceleration and compression
    persistence are used. The implementation references only the current row and
    lagged rows, never future rows or target/outcome fields.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise RuntimeError(f"vol/compression input matrix must be 2D, got {x.shape}")
    if x.shape[0] == 0:
        raise RuntimeError("vol/compression input matrix must contain at least one row")
    if x.shape[1] != len(feature_names):
        raise RuntimeError(
            f"vol/compression feature name count {len(feature_names)} does not match matrix width {x.shape[1]}"
        )
    if len(feature_names) != len(set(feature_names)):
        duplicates = sorted({name for name in feature_names if feature_names.count(name) > 1})
        raise RuntimeError(f"vol/compression duplicate feature names: {duplicates[:10]}")
    missing = missing_entry_vol_compression_source_fields(feature_names)
    if missing:
        raise RuntimeError(f"vol/compression required source fields missing: {missing}")
    if not np.isfinite(x).all():
        bad_rows, bad_cols = np.where(~np.isfinite(x))
        examples = [
            {"row": int(row), "feature": feature_names[int(col)]}
            for row, col in zip(bad_rows[:10], bad_cols[:10])
        ]
        raise RuntimeError(f"vol/compression input matrix contains non-finite values: {examples}")

    idx = _name_index(feature_names)
    arrays: list[np.ndarray] = []
    names: list[str] = []

    def c(name: str) -> np.ndarray:
        return _col(x, idx, name)

    atr_unit = _unit_from_z(c("snap.atr_z"), scale=2.5)
    range_unit = _unit_from_z(c("snap._v1_range_z"), scale=2.5)
    # ``rvol_20`` is NOT a z-score: it is a LEVEL in bps*sqrt(20) (TRAIN p50
    # 18.44).  Feeding it to the z-score scale 2.5 made
    # ``tanh(rvol_20 / 2.5)`` exactly 1.0 on 29.90% of the 369,303 TRAIN rows
    # and >= 0.999999 on 51.51%, i.e. this term was a constant over half the
    # population, while ``_v1_pk_sigma20`` -- the SAME physical quantity as a
    # dimensionless fraction -- went through the SAME 2.5 and stayed pinned at
    # ~1.6e-4, a measured 45,004x magnitude mismatch inside the single 7-term
    # ``high_vol_tail_risk`` sum below.  Both legs are now put on ONE declared
    # unit (per-bar bps) by the semantics owner and share ONE TRAIN-fitted p90
    # scale.  See entry_volatility_semantics_v1 for the population and log.
    rvol_unit = vol_per_bar_bps_pressure(rvol_20_per_bar_bps(c("snap.rvol_20")))
    vol_z_unit = _unit_from_z(c("snap.vol_z_20"), scale=2.5)
    vol_pct_96 = _clip01(c("snap.vol_pct_96"))
    m5_vol_pct = _clip01(c("ctx_cont.vol_pct_m5_1yr"))
    h1_vol_pct = _clip01(c("ctx_cont.vol_pct_h1_1yr"))
    d1_atr_pct = _clip01(c("ctx_cont.D1_atr_percentile_252"))
    # Divisors derived from the categorical domain owner
    # (MODEL_NATIVE_CTX_CAT_DOMAINS), not repeated literals: len(domain) - 1
    # per field, verified contiguous zero-based in _ctx_cat_domain_max.
    atr_bucket = _clip01(c("ctx_cat.atr_bucket") / _ctx_cat_domain_max("atr_bucket"))
    vol_regime = _clip01(c("ctx_cat.vol_regime_id") / _ctx_cat_domain_max("vol_regime_id"))
    # CROSS-TF ATR ratios, centred by the ONE owner transform
    # (entry_volatility_semantics_v1.center_cross_tf_atr_ratio) on each pair's
    # own sqrt(bar-duration) scaling expectation, so 0 means "short-dated vol
    # exactly at its scaling expectation" and the sign is the term-structure
    # statement.  Two separate wrong centres were used here before (2026-08-13,
    # V30 package 8B):
    #   * ``_tanh(ratio - 1.0)`` (the three lines this block replaces) assumed
    #     centre 1, so with measured TRAIN
    #     means 0.5697/0.1064/0.2182 it returned -0.406/-0.891/-0.782 on almost
    #     every row -- a sign that could never flip;
    #   * ``center_atr_ratio`` (the SAME-TF transform) further down left
    #     ``_pos(...)`` alive on 0 of 369,303 rows for (m15,d1) and (h1,d1),
    #     0.007% for (m5,h4) and 1.205% for (m5,m15), which is the measured 75%
    #     dead weight in ``higher_tf_vol_expansion_pressure``.
    # One centred value per pair now serves both consumers; the duplicate
    # transform is gone rather than re-weighted (no magnitude invented).
    atr_ratio_m5_m15 = center_cross_tf_atr_ratio(
        c("ctx_cont.atr_ratio_m5_m15"), short_timeframe="M5", long_timeframe="M15"
    )
    atr_ratio_m5_h4 = center_cross_tf_atr_ratio(
        c("ctx_cont.atr_ratio_m5_h4"), short_timeframe="M5", long_timeframe="H4"
    )
    atr_ratio_m15_d1 = center_cross_tf_atr_ratio(
        c("ctx_cont.atr_ratio_m15_d1"), short_timeframe="M15", long_timeframe="D1"
    )
    atr_ratio_h1_d1 = center_cross_tf_atr_ratio(
        c("ctx_cont.atr_ratio_h1_d1"), short_timeframe="H1", long_timeframe="D1"
    )

    atr_percentile = _clip01(
        0.20 * atr_unit
        + 0.15 * vol_z_unit
        + 0.15 * vol_pct_96
        + 0.15 * m5_vol_pct
        + 0.15 * h1_vol_pct
        + 0.15 * d1_atr_pct
        + 0.05 * np.maximum(atr_bucket, vol_regime)
    )
    # Preserve the continuous percentile blend as its own feature and express
    # low/high *tail* pressure only outside the central half.  The previous
    # high-pressure field was byte-identical to ``atr_percentile_blend`` and
    # therefore occupied a mandatory slot without adding information.
    low_atr_pressure = _clip01((0.5 - atr_percentile) * 2.0)
    high_atr_pressure = _clip01((atr_percentile - 0.5) * 2.0)
    atr_m5_h1_spread = _clip(
        0.55 * (m5_vol_pct - h1_vol_pct)
        + 0.25 * atr_ratio_m5_m15
        - 0.10 * atr_ratio_m15_d1
        - 0.10 * atr_ratio_h1_d1,
        -1.0,
        1.0,
    )

    h1_compression = atr_ratio_compression_pressure(
        c("ctx_cont.H1_range_compression_ratio")
    )
    m15_compression = atr_ratio_compression_pressure(
        c("ctx_cont.M15_range_compression_ratio")
    )
    squeeze = bollinger_squeeze_pressure(c("snap._v1_bb_squeeze_20_2"))
    bb_delta = _tanh(c("snap._v1_bb_bandwidth_delta_10"), scale=1.0)
    bandwidth_narrowing = _neg(bb_delta)
    bandwidth_expanding = _pos(bb_delta)
    foundation_compression = _clip01(c("chart.foundation_compression_state"))
    foundation_expansion = _clip01(c("chart.foundation_expansion_state"))
    # Producer bound is [0, 1] per entry_foundation_structure_v1 (2026-08-09
    # wave: the declared [0, 5] clip there was never reachable — the release
    # trigger is a product of two [0, 1] quantities), so no rescale is needed.
    foundation_release = _clip01(c("chart.foundation_compression_release_trigger"))
    foundation_release_up = _clip01(c("chart.foundation_compression_release_up"))
    foundation_release_down = _clip01(c("chart.foundation_compression_release_down"))

    range_compression_stack = _clip01(
        0.32 * h1_compression
        + 0.28 * m15_compression
        + 0.22 * squeeze
        + 0.10 * bandwidth_narrowing
        + 0.08 * foundation_compression
    )
    compression_dispersion = _dispersion01([h1_compression, m15_compression, squeeze, foundation_compression])
    compression_agreement = _clip01(range_compression_stack * (1.0 - 0.65 * compression_dispersion))
    compression_depth = _clip01(
        0.45 * range_compression_stack
        + 0.25 * low_atr_pressure
        + 0.15 * (1.0 - range_unit)
        + 0.15 * (1.0 - rvol_unit)
    )
    compression_persistence = _clip01(compression_depth * (0.55 + 0.45 * _lag1(compression_depth)))

    # The four centred cross-TF ratios are computed once above.
    short_tf_expansion = _clip01(
        0.35 * _pos(atr_ratio_m5_m15)
        + 0.35 * _pos(atr_ratio_m5_h4)
        + 0.20 * _pos(_tanh(c("snap.vol_ratio_5_20"), scale=2.0))
        + 0.10 * bandwidth_expanding
    )
    higher_tf_expansion = _clip01(
        0.40 * _pos(atr_ratio_m15_d1)
        + 0.35 * _pos(atr_ratio_h1_d1)
        + 0.15 * _pos(_tanh(c("ctx_cont.d1_range_z_20_canon_v2"), scale=2.5))
        + 0.10 * _pos(_tanh(c("ctx_cont.m15_range_z_20_canon_v2"), scale=2.5))
    )
    term_structure_slope = _clip(
        0.30 * atr_m5_h1_spread
        + 0.20 * (h1_vol_pct - d1_atr_pct)
        + 0.20 * atr_ratio_m5_m15
        + 0.15 * atr_ratio_m5_h4
        + 0.15 * atr_ratio_m15_d1,
        -1.0,
        1.0,
    )
    vol_dispersion = _dispersion01([m5_vol_pct, h1_vol_pct, d1_atr_pct, vol_pct_96, atr_unit])
    ratio_dispersion = _clip01(
        (np.abs(atr_ratio_m5_m15) + np.abs(atr_ratio_m5_h4) + np.abs(atr_ratio_m15_d1) + np.abs(atr_ratio_h1_d1))
        / 4.0
    )
    mtf_agreement = _clip01(
        1.0
        - 0.70 * vol_dispersion
        - 0.25 * ratio_dispersion
        + 0.15 * _clip01(c("ctx_cont.regime_tf_agreement_v3"))
    )
    # Normalized by the sum's unclipped algebraic max so the additive terms
    # keep resolution instead of clip-saturating at 1.  Derived max:
    # mtf_agreement = clip01(1 - 0.70*vol_dispersion - 0.25*ratio_dispersion
    # + 0.15*agreement) with both dispersions <= 1, so its unclipped minimum is
    # 1 - 0.70 - 0.25 = 0.05 (the clip floor never binds) and
    # (1 - mtf_agreement) <= 0.95; ratio_dispersion <= 1,
    # |term_structure_slope| <= 1 (clipped), divergence flag <= 1.
    # max = 0.95 + 0.25 + 0.20 + 0.15 = 1.55
    mtf_divergence = _clip01(
        (
            1.0
            - mtf_agreement
            + 0.25 * ratio_dispersion
            + 0.20 * np.abs(term_structure_slope)
            + 0.15 * _clip01(c("ctx_cont.regime_divergence_flag_v3"))
        )
        / 1.55
    )
    compression_divergence = _clip01(
        0.40 * compression_dispersion
        + 0.25 * mtf_divergence
        + 0.20 * compression_depth * high_atr_pressure
        + 0.15 * bandwidth_expanding
    )

    range_expansion = _clip01(
        0.28 * high_atr_pressure
        + 0.22 * range_unit
        + 0.20 * rvol_unit
        + 0.15 * short_tf_expansion
        + 0.10 * higher_tf_expansion
        + 0.05 * bandwidth_expanding
    )
    expansion_state = _clip01(0.45 * foundation_expansion + 0.40 * range_expansion + 0.15 * bandwidth_expanding)
    expansion_acceleration = _clip01(
        _pos(expansion_state - _lag1(expansion_state))
        + 0.30 * short_tf_expansion
        + 0.20 * bandwidth_expanding
    )
    compression_release_impulse = _clip01(
        0.40 * foundation_release
        + 0.35 * _lag1(compression_depth) * expansion_acceleration
        + 0.25 * compression_depth * bandwidth_expanding
    )

    ret1 = _tanh(c("snap.ret_1"), scale=15.0)
    ret5 = _tanh(c("snap.ret_5"), scale=30.0)
    ret20 = _tanh(c("snap.ret_20"), scale=60.0)
    signed_vol = _tanh(c("snap.signed_vol_z_20"), scale=3.0)
    # Unit note (2026-08-09 upstream ATR-normalization wave): `_v1h1_ema_diff`,
    # `_v1h4_ema_diff` and `d1_ema_slope_20_canon_v2` are ATR-multiple units
    # (O(1) per bar), so tanh scale 2.0 is dimensionally sane here: +-2 ATR of
    # displacement maps to +-tanh(1) ~= 0.76 before saturation.
    trend_proxy = _clip(
        0.30 * _tanh(c("ctx_cont._v1h1_ema_diff"), scale=2.0)
        + 0.25 * _tanh(c("ctx_cont._v1h4_ema_diff"), scale=2.0)
        + 0.20 * _tanh(c("ctx_cont.d1_ema_slope_20_canon_v2"), scale=2.0)
        + 0.15 * _tanh(c("ctx_cont.m15_trend_sign_canon_v2"), scale=2.0)
        + 0.10 * _tanh(c("chart.foundation_impulse_direction"), scale=2.0),
        -1.0,
        1.0,
    )
    foundation_direction = _clip(foundation_release_up - foundation_release_down, -1.0, 1.0)
    direction_raw = _clip(
        0.24 * signed_vol
        + 0.20 * ret5
        + 0.14 * ret1
        + 0.14 * ret20
        + 0.18 * trend_proxy
        + 0.10 * foundation_direction,
        -1.0,
        1.0,
    )
    expansion_direction = _clip(
        direction_raw * (0.55 + 0.45 * expansion_state)
        + foundation_direction * compression_release_impulse,
        -1.0,
        1.0,
    )
    direction_confidence_mag = _clip01(
        np.abs(direction_raw)
        * (0.45 + 0.30 * expansion_state + 0.25 * compression_release_impulse)
        * (0.65 + 0.35 * mtf_agreement)
    )
    # Signed emission: the field name promises direction, but np.abs above
    # discards the only sign this composite consumes.  Sign =
    # np.sign(direction_raw); an exact direction_raw == 0 yields 0 (honest
    # neutral).  The chop/false-breakout/forecast consumers below keep using
    # the unsigned magnitude, where (1 - confidence) means uncertainty.
    direction_confidence = (direction_confidence_mag * np.sign(direction_raw)).astype(np.float32)
    release_up = _clip01(
        foundation_release_up
        + compression_release_impulse * (0.55 * _pos(direction_raw) + 0.45 * _pos(expansion_direction))
    )
    release_down = _clip01(
        foundation_release_down
        + compression_release_impulse * (0.55 * _neg(direction_raw) + 0.45 * _neg(expansion_direction))
    )
    squeeze_breakout_asym = _clip(
        compression_depth * direction_raw * (0.50 + 0.50 * mtf_agreement),
        -1.0,
        1.0,
    )

    kurt_pressure = _clip01(np.abs(_tanh(c("snap._v1_kurt_r"), scale=4.0)))
    # Same unit repair as ``rvol_unit``: the Parkinson sigma is a dimensionless
    # fraction (TRAIN p50 4.097e-4), so the shared 2.5 scale pinned it at
    # ~1.6e-4 -- five orders of magnitude below the ``rvol_unit`` term it is
    # summed with.  ``np.abs`` is dropped because the owner fails closed on a
    # negative volatility magnitude instead of silently folding it.
    sigma_pressure = vol_per_bar_bps_pressure(
        pk_sigma20_per_bar_bps(c("snap._v1_pk_sigma20"))
    )
    high_vol_tail_risk = _clip01(
        0.26 * high_atr_pressure
        + 0.18 * range_unit
        + 0.16 * rvol_unit
        + 0.14 * expansion_state
        + 0.10 * mtf_divergence
        + 0.08 * kurt_pressure
        + 0.08 * sigma_pressure
    )
    high_vol_chop_tail = _clip01(
        high_vol_tail_risk
        * (0.45 + 0.30 * mtf_divergence + 0.15 * compression_divergence + 0.10 * (1.0 - direction_confidence_mag))
    )
    low_vol_breakout_setup = _clip01(
        low_atr_pressure
        * compression_agreement
        * (0.60 + 0.40 * compression_persistence)
        * (0.70 + 0.30 * mtf_agreement)
        * (1.0 - 0.45 * high_vol_tail_risk)
    )
    low_vol_false_breakout_risk = _clip01(
        low_vol_breakout_setup
        * (0.45 * mtf_divergence + 0.30 * compression_divergence + 0.25 * (1.0 - direction_confidence_mag))
    )
    vol_forecast_confidence = _clip01(
        0.30 * mtf_agreement
        + 0.20 * direction_confidence_mag
        + 0.18 * np.maximum(compression_release_impulse, compression_agreement)
        + 0.17 * np.maximum(expansion_state, low_vol_breakout_setup)
        + 0.15 * (1.0 - high_vol_chop_tail)
    )

    _add(arrays, names, "atr_percentile_blend", atr_percentile, lo=0.0, hi=1.0)
    _add(arrays, names, "atr_percentile_m5_minus_h1", atr_m5_h1_spread, lo=-1.0, hi=1.0)
    _add(arrays, names, "low_atr_percentile_pressure", low_atr_pressure, lo=0.0, hi=1.0)
    _add(arrays, names, "high_atr_percentile_pressure", high_atr_pressure, lo=0.0, hi=1.0)
    _add(arrays, names, "squeeze_state_score", squeeze, lo=0.0, hi=1.0)
    _add(arrays, names, "range_compression_stack", range_compression_stack, lo=0.0, hi=1.0)
    _add(arrays, names, "compression_agreement_score", compression_agreement, lo=0.0, hi=1.0)
    _add(arrays, names, "compression_divergence_score", compression_divergence, lo=0.0, hi=1.0)
    _add(arrays, names, "compression_depth_score", compression_depth, lo=0.0, hi=1.0)
    _add(arrays, names, "compression_persistence_score", compression_persistence, lo=0.0, hi=1.0)
    _add(arrays, names, "compression_release_impulse", compression_release_impulse, lo=0.0, hi=1.0)
    _add(arrays, names, "compression_release_up_pressure", release_up, lo=0.0, hi=1.0)
    _add(arrays, names, "compression_release_down_pressure", release_down, lo=0.0, hi=1.0)
    _add(arrays, names, "expansion_state_score", expansion_state, lo=0.0, hi=1.0)
    _add(arrays, names, "expansion_acceleration_score", expansion_acceleration, lo=0.0, hi=1.0)
    _add(arrays, names, "expansion_direction_score", expansion_direction, lo=-1.0, hi=1.0)
    _add(arrays, names, "expansion_direction_confidence", direction_confidence, lo=-1.0, hi=1.0)
    _add(arrays, names, "high_vol_tail_risk", high_vol_tail_risk, lo=0.0, hi=1.0)
    _add(arrays, names, "high_vol_chop_tail_risk", high_vol_chop_tail, lo=0.0, hi=1.0)
    _add(arrays, names, "low_vol_breakout_setup", low_vol_breakout_setup, lo=0.0, hi=1.0)
    _add(arrays, names, "low_vol_false_breakout_risk", low_vol_false_breakout_risk, lo=0.0, hi=1.0)
    _add(arrays, names, "mtf_vol_agreement_score", mtf_agreement, lo=0.0, hi=1.0)
    _add(arrays, names, "mtf_vol_divergence_score", mtf_divergence, lo=0.0, hi=1.0)
    _add(arrays, names, "short_tf_vol_expansion_pressure", short_tf_expansion, lo=0.0, hi=1.0)
    _add(arrays, names, "higher_tf_vol_expansion_pressure", higher_tf_expansion, lo=0.0, hi=1.0)
    _add(arrays, names, "squeeze_breakout_asym_score", squeeze_breakout_asym, lo=-1.0, hi=1.0)
    _add(arrays, names, "vol_term_structure_slope", term_structure_slope, lo=-1.0, hi=1.0)
    _add(arrays, names, "vol_forecast_confidence_score", vol_forecast_confidence, lo=0.0, hi=1.0)

    out = np.column_stack(arrays).astype(np.float32, copy=False) if arrays else np.empty((x.shape[0], 0), dtype=np.float32)
    if tuple(names) != VOL_COMPRESSION_FEATURE_NAMES:
        raise RuntimeError("vol/compression feature order drifted")
    if not np.isfinite(out).all():
        raise RuntimeError("vol/compression layer contains non-finite values")
    return out, names
