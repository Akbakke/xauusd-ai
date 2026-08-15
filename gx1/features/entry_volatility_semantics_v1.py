"""One-truth physical units for native-clock volatility estimators.

Handwritten compression/expansion pressure transforms formerly kept here
were test-only remnants of the retired volatility scorebook.  The model now
consumes raw ATR evidence and the TRAIN-fitted causal squeeze state; this file
only owns the unit conversion shared by the raw realized-volatility measures.
"""
from __future__ import annotations

import numpy as np


def _finite_vector(values: np.ndarray, *, field: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 1 or not np.isfinite(arr).all():
        raise RuntimeError(f"ENTRY_VOLATILITY_SEMANTICS_INVALID: {field}")
    return arr


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
# The two estimators agree to 0.6% at p50 and 1.1% at p90 once the units match.
# That proves the conversion, not an absolute serve scale: the active v14 model
# consumes the raw primitives separately and its hash-bound TRAIN-only input
# normalization owns their magnitude on the physical M1+M5 TRAIN union.
RVOL_20_WINDOW_BARS = 20
BPS_PER_UNIT_FRACTION = 1.0e4
LOCAL_VOLATILITY_DECISION_CLOCKS = ("m1", "m5")


def _require_local_volatility_clock(decision_clock: str) -> str:
    """Require the exact native clock whose closed bars produced the input.

    The unit conversion is identical on M1 and M5, but the observations are
    not interchangeable: ``rvol_20`` means twenty native bars.  Requiring the
    clock at the boundary prevents an M5-fitted absolute magnitude from being
    silently reused on Exit's M1 values.  The model instead receives each raw
    native primitive and fits one hash-bound TRAIN-only normalization contract
    over the physical M1+M5 TRAIN union.
    """

    if decision_clock not in LOCAL_VOLATILITY_DECISION_CLOCKS:
        raise RuntimeError(
            "ENTRY_VOLATILITY_SEMANTICS_LOCAL_CLOCK_INVALID: "
            f"decision_clock={decision_clock!r} "
            f"expected={LOCAL_VOLATILITY_DECISION_CLOCKS}"
        )
    return decision_clock


def rvol_20_per_bar_bps(
    values: np.ndarray,
    *,
    decision_clock: str,
) -> np.ndarray:
    """Return native-clock ``rvol_20`` as per-bar bps.

    The producer stores ``std(native close returns) * 1e4 * sqrt(20)``.
    Dividing by that declared window factor restores the per-native-bar unit;
    it does not convert M1 observations into M5 observations or vice versa.
    """

    _require_local_volatility_clock(decision_clock)
    arr = _finite_vector(values, field="rvol_20")
    if (arr < 0.0).any():
        raise RuntimeError("ENTRY_VOLATILITY_SEMANTICS_RVOL_NEGATIVE")
    return (arr / np.float32(np.sqrt(float(RVOL_20_WINDOW_BARS)))).astype(
        np.float32, copy=False
    )


def pk_sigma20_per_bar_bps(
    values: np.ndarray,
    *,
    decision_clock: str,
) -> np.ndarray:
    """Return native-clock Parkinson sigma as per-bar bps."""

    _require_local_volatility_clock(decision_clock)
    arr = _finite_vector(values, field="_v1_pk_sigma20")
    if (arr < 0.0).any():
        raise RuntimeError("ENTRY_VOLATILITY_SEMANTICS_PARKINSON_NEGATIVE")
    return (arr * np.float32(BPS_PER_UNIT_FRACTION)).astype(np.float32, copy=False)


__all__ = [
    "BPS_PER_UNIT_FRACTION",
    "LOCAL_VOLATILITY_DECISION_CLOCKS",
    "RVOL_20_WINDOW_BARS",
    "pk_sigma20_per_bar_bps",
    "rvol_20_per_bar_bps",
]
