"""One-truth semantics for ATR-ratio compression and Bollinger squeeze.

The canonical H1/M15 fields are ``ATR14 / ATR100``: values below one mean
compression and values above one mean expansion.  The canonical Bollinger
field is ``bandwidth / rolling_mean(bandwidth) - 1``: negative values mean a
squeeze and positive values mean expansion.  Keeping these transforms here
prevents feature families from silently assigning the opposite meaning.

There are two ATR-ratio families and they do NOT share a centre:

* SAME-TF (``ATR14 / ATR100`` on one timeframe) is centred at 1, so
  :func:`center_atr_ratio` is the whole transform.
* CROSS-TF (``ATR(short) / ATR(long)``) is centred at
  ``sqrt(bars_short / bars_long)`` because ATR scales as the square root of
  bar duration; feeding it to :func:`center_atr_ratio` makes ``log2`` almost
  always negative and any ``max(x, 0)`` consumer identically dead.  Use
  :func:`center_cross_tf_atr_ratio`, which re-centres on that scaling-law
  expectation first.

MEASURED (V30 package 8B, 2026-08-13) on the complete declared TRAIN
population XAU_ENTRY_EXIT_M15_20260811_V29J (369,303 rows -- the complete
population, so rule 2f carries no sampling error), for the four cross-TF
ratios (m5,m15) / (m5,h4) / (m15,d1) / (h1,d1):

    mean   0.5697 / 0.1435 / 0.1064 / 0.2182
    p50    0.5471 / 0.1256 / 0.0936 / 0.2072
    sqrt(bars_short/bars_long)
           0.5774 / 0.1443 / 0.1021 / 0.2041

The observed centres ARE the scaling law (the means agree to 1-4%; the medians
sit below the means because the ratios are right-skewed).  That is exactly why
the misapplied same-TF transform left ``_pos(center_atr_ratio(cross_tf))``
alive on 1.205% of rows for (m5,m15), 0.007% for (m5,h4), and on exactly ZERO
of the 369,303 rows for (m15,d1) and (h1,d1).
"""
from __future__ import annotations

import numpy as np


def _finite_vector(values: np.ndarray, *, field: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 1 or not np.isfinite(arr).all():
        raise RuntimeError(f"ENTRY_VOLATILITY_SEMANTICS_INVALID: {field}")
    return arr


def center_atr_ratio(
    values: np.ndarray, *, field: str = "atr14_over_atr100"
) -> np.ndarray:
    """Return the signed centered SAME-TF ATR-ratio ``tanh(log2(ratio))``.

    This is the one-truth transform for a ``ATR_a / ATR_b`` ratio whose two
    legs are measured on the SAME timeframe -- ``ATR14 / ATR100`` and the
    ``*_range_compression_ratio`` fields built from it.  Such a ratio is
    centred at 1, so the output is 0 at ratio 1, ``-tanh(1)`` at ratio 0.5 and
    ``+tanh(1)`` at ratio 2, in [-1, 1].

    It is NOT the transform for a cross-timeframe ratio: ATR scales as the
    square root of bar duration, so ``ATR(short)/ATR(long)`` is centred at
    ``sqrt(bars_short/bars_long)`` << 1 and this function would return an
    almost always negative value, silently killing every ``max(x, 0)``
    consumer.  Use :func:`center_cross_tf_atr_ratio` there.

    Fails closed on non-finite or non-positive input; consumers must call this
    owner instead of re-deriving it with a soft floor that would read corrupt
    input as maximal compression.
    """

    ratio = _finite_vector(values, field=field)
    if (ratio <= 0.0).any():
        raise RuntimeError("ENTRY_VOLATILITY_SEMANTICS_ATR_RATIO_NOT_POSITIVE")
    centered = np.tanh(np.log(ratio) / np.log(2.0))
    return np.clip(centered, -1.0, 1.0).astype(np.float32, copy=False)


def cross_tf_atr_ratio_scaling_expectation(
    short_timeframe: str, long_timeframe: str
) -> float:
    """Return the scaling-law centre of ``ATR(short) / ATR(long)``.

    ATR is a mean absolute range over one bar, and a random walk's absolute
    range over a bar of duration ``T`` scales as ``sqrt(T)``.  The declared
    per-timeframe bar durations come from the ONE resample owner
    (``htf_features.MULTI_TF_BARS_IN_M5``, itself derived from
    ``MULTI_TF_RESAMPLE_RULES``), so the expectation is
    ``sqrt(bars_short / bars_long)`` and no magnitude is introduced here.

    Deferred import: ``gx1.features.htf_features`` imports
    ``swing_structure_v1`` -> ``entry_foundation_structure_v1`` -> this module,
    so a module-level import would be circular.  By the time any feature layer
    calls this, the resample owner is fully initialized.
    """

    from gx1.features.htf_features import MULTI_TF_BARS_IN_M5

    short_key = str(short_timeframe).lower()
    long_key = str(long_timeframe).lower()
    if short_key not in MULTI_TF_BARS_IN_M5 or long_key not in MULTI_TF_BARS_IN_M5:
        raise RuntimeError(
            "ENTRY_VOLATILITY_SEMANTICS_UNKNOWN_TIMEFRAME: "
            f"short={short_timeframe!r} long={long_timeframe!r} "
            f"declared={sorted(MULTI_TF_BARS_IN_M5)}"
        )
    bars_short = int(MULTI_TF_BARS_IN_M5[short_key])
    bars_long = int(MULTI_TF_BARS_IN_M5[long_key])
    if bars_short <= 0 or bars_long <= bars_short:
        raise RuntimeError(
            "ENTRY_VOLATILITY_SEMANTICS_CROSS_TF_ORDER_INVALID: "
            f"short={short_key}({bars_short}) long={long_key}({bars_long})"
        )
    return float(np.sqrt(bars_short / bars_long))


def center_cross_tf_atr_ratio(
    values: np.ndarray, *, short_timeframe: str, long_timeframe: str
) -> np.ndarray:
    """Return the signed centered CROSS-TF ATR-ratio in [-1, 1].

    ``tanh(log2(ratio / sqrt(bars_short / bars_long)))``: 0 means the
    short-dated ATR sits exactly at its square-root-of-duration scaling
    expectation, positive means short-dated volatility is ABOVE that
    expectation, negative below.  The sign is therefore the term-structure
    statement the field names promise, and ``max(x, 0)`` consumers become live.

    Same fail-closed contract as :func:`center_atr_ratio`.
    """

    expectation = cross_tf_atr_ratio_scaling_expectation(
        short_timeframe, long_timeframe
    )
    field = f"atr_{str(short_timeframe).lower()}_over_atr_{str(long_timeframe).lower()}"
    ratio = _finite_vector(values, field=field)
    return center_atr_ratio(ratio / np.float32(expectation), field=field)


def atr_ratio_compression_pressure(values: np.ndarray) -> np.ndarray:
    """Return [0,1] pressure where lower ATR14/ATR100 means more compression."""

    return np.maximum(-center_atr_ratio(values), 0.0).astype(
        np.float32, copy=False
    )


def atr_ratio_expansion_pressure(values: np.ndarray) -> np.ndarray:
    """Return [0,1] pressure where higher ATR14/ATR100 means more expansion."""

    return np.maximum(center_atr_ratio(values), 0.0).astype(
        np.float32, copy=False
    )


# ---------------------------------------------------------------------------
# Unit owner for the two per-bar realized-volatility measures on the Entry snap
# surface.  They are the SAME physical quantity in two different units, which is
# how a single ``tanh(x / 2.5)`` scale came to be applied to both:
#
#   snap.rvol_20        = std(pct_change) * 1e4 * sqrt(20)   -> bps * sqrt(20)
#                         (gx1/scripts/materialize_build_canonical_features_v1
#                          .rvol_window; the sqrt(window) is the producer's own
#                          declared factor and 20 is the window in the field
#                          name)
#   snap._v1_pk_sigma20 = Parkinson sigma from log(high/low)  -> dimensionless
#                         (gx1/features/basic_v1._parkinson_sigma)
#
# MEASURED (V30 package 8B, 2026-08-13) on the complete declared TRAIN
# population XAU_ENTRY_EXIT_M15_20260811_V29J (369,303 rows; complete
# population, so rule 2f carries no sampling error).  Raw log:
# GX1_DATA/logs/v30_package8b_20260813/train_vol_unit_scale_fit.json.
#
#   raw p50: rvol_20 = 18.4375, _v1_pk_sigma20 = 4.0968e-4
#            -> magnitude ratio 45,004 inside one 7-term sum, and
#               tanh(rvol_20 / 2.5) is exactly 1.0 on 29.90% of rows and
#               >= 0.999999 on 51.51%, while tanh(sigma / 2.5) is pinned at
#               ~1.6e-4.
#   after the unit repair below, both in per-bar bps:
#            rvol_20/sqrt(20)      p50 4.1227  p90 9.1725
#            _v1_pk_sigma20*1e4    p50 4.0968  p90 9.0751
#
# The two estimators agree to 0.6% at p50 and 1.1% at p90 once the units match,
# which is the proof that they are one quantity and may share one scale.
RVOL_20_WINDOW_BARS = 20
BPS_PER_UNIT_FRACTION = 1.0e4

# Fitted statistic, same role and naming as
# entry_session_regime_interactions_v1.SPREAD_RATIO_TANH_SCALE_TRAIN_P90: the
# p90 of the close-to-close per-bar realized volatility (rvol_20 / sqrt(20), in
# bps) on the complete declared TRAIN population named above, so that
# tanh(x / scale) reads tanh(1) ~= 0.762 at the p90 instead of saturating.  The
# Parkinson estimator's p90 on the same population is 9.0751 (1.06% away), so
# this one scale is honest for both legs of the sum.
VOL_PER_BAR_BPS_TANH_SCALE_TRAIN_P90 = 9.172530


def rvol_20_per_bar_bps(values: np.ndarray) -> np.ndarray:
    """Return ``snap.rvol_20`` as per-bar bps (undo the producer's sqrt(20))."""

    arr = _finite_vector(values, field="rvol_20")
    return (arr / np.float32(np.sqrt(float(RVOL_20_WINDOW_BARS)))).astype(
        np.float32, copy=False
    )


def pk_sigma20_per_bar_bps(values: np.ndarray) -> np.ndarray:
    """Return ``snap._v1_pk_sigma20`` as per-bar bps (dimensionless -> bps)."""

    arr = _finite_vector(values, field="_v1_pk_sigma20")
    return (arr * np.float32(BPS_PER_UNIT_FRACTION)).astype(np.float32, copy=False)


def vol_per_bar_bps_pressure(values: np.ndarray) -> np.ndarray:
    """Return [0,1] volatility pressure from a NON-NEGATIVE per-bar bps input.

    ``tanh(bps / VOL_PER_BAR_BPS_TANH_SCALE_TRAIN_P90)``.  Fails closed on a
    negative input: a realized-volatility magnitude cannot be negative, and
    silently folding one to zero would hide a producer defect.
    """

    arr = _finite_vector(values, field="vol_per_bar_bps")
    if (arr < 0.0).any():
        raise RuntimeError("ENTRY_VOLATILITY_SEMANTICS_VOL_BPS_NEGATIVE")
    pressure = np.tanh(arr / np.float32(VOL_PER_BAR_BPS_TANH_SCALE_TRAIN_P90))
    return np.clip(pressure, 0.0, 1.0).astype(np.float32, copy=False)


def bollinger_squeeze_pressure(values: np.ndarray) -> np.ndarray:
    """Return [0,1] squeeze from ``bandwidth/mean_bandwidth - 1``."""

    relative_delta = _finite_vector(values, field="bb_width_over_mean_minus_one")
    return np.clip(-relative_delta, 0.0, 1.0).astype(np.float32, copy=False)


def bollinger_expansion_pressure(values: np.ndarray) -> np.ndarray:
    """Return [0,1] expansion from ``bandwidth/mean_bandwidth - 1``."""

    relative_delta = _finite_vector(values, field="bb_width_over_mean_minus_one")
    return np.clip(relative_delta, 0.0, 1.0).astype(np.float32, copy=False)


__all__ = [
    "BPS_PER_UNIT_FRACTION",
    "RVOL_20_WINDOW_BARS",
    "VOL_PER_BAR_BPS_TANH_SCALE_TRAIN_P90",
    "atr_ratio_compression_pressure",
    "atr_ratio_expansion_pressure",
    "bollinger_expansion_pressure",
    "bollinger_squeeze_pressure",
    "center_atr_ratio",
    "center_cross_tf_atr_ratio",
    "cross_tf_atr_ratio_scaling_expectation",
    "pk_sigma20_per_bar_bps",
    "rvol_20_per_bar_bps",
    "vol_per_bar_bps_pressure",
]
