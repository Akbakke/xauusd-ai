#!/usr/bin/env python3
"""canonical_v3 augmentation — produces canonical_v3 from canonical_v2 by:

  1. Pruning 11 redundant features (5 exact duplicates + 6 near-duplicates @ |corr|>0.95)
  2. Adding 4 cyclic time features (hour_sin, hour_cos, dow_sin, dow_cos)
  3. Adding 1 SMC × swing-state interaction (smc_premium_state)
  4. Adding 1 cross-TF momentum feature (m5h1_momentum)
  5. (Future) V10 outputs as cross-bridge link — requires V10 inference pass; deferred to a
     follow-up step that joins this augmented parquet with V10 v2 inference.

Output: `canonical_v3.parquet` in same dir as canonical_v2.

Per audit findings (project_gx1_audit_findings_2026q2.md):
    - 5 exact duplicates: _v1_r5↔_v1_int_r5_atr, _v1h4_slope5↔_v1_int_slope_h4_atr,
      _v1_clv↔_v1_int_clv_atr, ret_20↔roc20, _v1_body_tr↔_v1_body_share_1
    - 7 near-duplicates (|corr|>0.95): atr↔_v1_atr14, std50↔rvol_60, etc.

Net feature change: -5 columns = 11 removed + 6 added. The additions are
cyclic time (4), smc_premium_state (1), and m5h1_momentum (1).

Notes:
  - This is NOT lookahead-unsafe — all derivations come from existing canonical_v2 features
    or from the timestamp itself.
  - The pruned features remain in canonical_v2; this script does not modify v2.
  - V10 v3 / V3 v6 contracts that target canonical_v3 must be updated separately.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from gx1.time.session_detector import m5_decision_availability

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Pairs to prune: keep the FIRST element, drop the SECOND. Choice is principled —
# keep the more general / canonical-naming variant, drop the alias.
PAIRS_TO_PRUNE = [
    # Exact duplicates (corr=1.000)
    ("_v1_r5", "_v1_int_r5_atr"),
    ("_v1h4_slope5", "_v1_int_slope_h4_atr"),
    ("_v1_clv", "_v1_int_clv_atr"),
    ("ret_20", "roc20"),
    ("_v1_body_share_1", "_v1_body_tr"),
    # Near-duplicates (|corr|>0.95)
    ("_v1_atr14", "atr"),                              # _v1 family wins (used in V3 io)
    ("rvol_60", "std50"),                              # rvol_60 is more interpretable
    ("ema20_slope", "m15_ema_slope_5_canon_v2"),       # both useful but corr 0.962 → keep M5
    ("_v1_ema_diff", "_v1_vwap_drift48"),              # _v1_ema_diff is the canonical
    ("atr50", "m15_atr14_canon_v2"),                   # atr50 is more "current TF"
]
DROP_COLUMNS = [b for _, b in PAIRS_TO_PRUNE]


def add_cyclic_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclic features for the M5 row's observable decision time.

    Canonical rows are labelled by M5 bar start.  Their contents become known
    five minutes later, so hour/day boundaries must use ``label + 5min``.
    """
    if isinstance(df.index, pd.DatetimeIndex):
        ts = df.index
    elif "time" in df.columns:
        # 2026-06-11 fix: pd.to_datetime(Series) returns a Series (no .hour) — wrap in
        # DatetimeIndex so the column-branch behaves like the index-branch (latent bug;
        # the daemon path always used the index branch).
        ts = pd.DatetimeIndex(pd.to_datetime(df["time"], utc=True))
    else:
        raise RuntimeError("[canonical_v3] no DatetimeIndex or 'time' column found")
    decision_ts = m5_decision_availability(ts)
    hour = decision_ts.hour.to_numpy(dtype=np.float32)
    dow = decision_ts.dayofweek.to_numpy(dtype=np.float32)
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24).astype(np.float32)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24).astype(np.float32)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7).astype(np.float32)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7).astype(np.float32)
    return df

def add_smc_premium_state_interaction(df: pd.DataFrame) -> pd.DataFrame:
    """smc_premium_state = smc_premium_discount × indicator(smc_swing_state == 0)

    This is a conditional coordinate for the learned model: it exposes premium
    position while structure is clean HH+HL. It does not prescribe direction.
    """
    if "smc_premium_discount" not in df.columns or "smc_swing_state" not in df.columns:
        raise RuntimeError(
            "[canonical_v3] smc_premium_discount and smc_swing_state are required"
        )
    df = df.copy()
    pd_score = pd.to_numeric(
        df["smc_premium_discount"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    state_raw = pd.to_numeric(
        df["smc_swing_state"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    if (
        not np.isfinite(pd_score).all()
        or np.any(pd_score < 0.0)
        or np.any(pd_score > 1.0)
    ):
        raise RuntimeError(
            "[canonical_v3] smc_premium_discount must be finite and within [0,1]"
        )
    if (
        not np.isfinite(state_raw).all()
        or np.any(state_raw != np.floor(state_raw))
        or np.any(state_raw < 0.0)
        or np.any(state_raw > 4.0)
    ):
        raise RuntimeError(
            "[canonical_v3] smc_swing_state must use the exact finite enum 0..4"
        )
    state = state_raw.astype(np.int8)
    df["smc_premium_state"] = (pd_score * (state == 0).astype(np.float32)).astype(np.float32)
    return df


def _atr_normalized_h1_momentum(
    close: np.ndarray,
    h1_atr: np.ndarray,
    *,
    horizon_rows: int,
) -> np.ndarray:
    """Return one-hour price change scaled by aligned completed-H1 ATR.

    Causal HTF construction keeps the historical warmup prefix as NaN (it is
    not a neutral zero), so exactly one leading non-finite H1-ATR prefix is
    carried through as NaN for the downstream causal trim owner. Any
    non-finite value after the first finite observation is a data gap and
    fails closed. A finite zero H1 ATR is an availability state, not tiny
    volatility: such rows are exactly neutral rather than divided by an
    epsilon that would fabricate million-scale momentum.
    """

    if (
        close.ndim != 1
        or h1_atr.ndim != 1
        or close.shape != h1_atr.shape
        or isinstance(horizon_rows, bool)
        or not isinstance(horizon_rows, int)
        or horizon_rows <= 0
        or len(close) <= horizon_rows
    ):
        raise RuntimeError("[canonical_v3] close/H1 ATR shape mismatch")
    if not np.isfinite(close).all():
        raise RuntimeError("[canonical_v3] close must be finite")
    finite = np.isfinite(h1_atr)
    if not finite.any():
        raise RuntimeError("[canonical_v3] H1 ATR has no finite observations")
    first_finite = int(np.flatnonzero(finite)[0])
    if not finite[first_finite:].all():
        raise RuntimeError(
            "[canonical_v3] H1 ATR has non-finite values after causal warmup"
        )
    if np.any(h1_atr[first_finite:] < 0.0):
        raise RuntimeError("[canonical_v3] H1 ATR must be non-negative")
    # 2026-08-09 unit repair: _v1h1_atr changed from raw USD to bps in
    # htf_features (era-proxy repair). The numerator must match: convert the
    # close change to bps of price (the repo's ret_* convention,
    # materialize_build_canonical_features_v1 pct_change*1e4) so the ratio
    # stays a dimensionless ATR-multiple of the one-hour move.
    delta = close - np.roll(close, horizon_rows)
    delta[:horizon_rows] = 0.0
    delta_bps = delta / np.maximum(close, 1e-9) * 10000.0
    result = np.zeros_like(delta_bps, dtype=np.float64)
    np.divide(delta_bps, h1_atr, out=result, where=finite & (h1_atr > 1e-6))
    result[:first_finite] = np.nan
    return result


def add_cross_tf_momentum(
    df: pd.DataFrame,
    *,
    decision_bar_duration: pd.Timedelta,
) -> pd.DataFrame:
    """Add one-hour local momentum only after the sole V4 H1 projection."""

    from gx1.features.htf_features import require_model_native_mtf_owner_marker_v4

    require_model_native_mtf_owner_marker_v4(
        df,
        decision_bar_duration=decision_bar_duration,
    )
    if "m5h1_momentum" in df.columns:
        raise RuntimeError("[canonical_v3] duplicate m5h1_momentum owner")
    if "close" not in df.columns or "_v1h1_atr" not in df.columns:
        raise RuntimeError(
            "[canonical_v3] exact close and V4-projected _v1h1_atr are required"
        )
    if decision_bar_duration not in {
        pd.Timedelta(minutes=1),
        pd.Timedelta(minutes=5),
    }:
        raise RuntimeError("[canonical_v3] exact M1 or M5 decision clock required")
    horizon_rows = int(pd.Timedelta(hours=1) / decision_bar_duration)
    # Shallow copy: this owner only INSERTS m5h1_momentum, guarded against
    # duplication above, and never mutates an existing column, so the result is
    # byte-identical while the existing column buffers stay shared. A deep copy
    # here duplicates the whole frame - about 6.6 GiB on the native M1 clock -
    # and is what put the enriched stage over the 10G producer ceiling.
    out = df.copy(deep=False)
    close = pd.to_numeric(out["close"], errors="coerce").to_numpy(np.float64)
    h1_atr = pd.to_numeric(
        out["_v1h1_atr"], errors="coerce"
    ).to_numpy(np.float64)
    out["m5h1_momentum"] = _atr_normalized_h1_momentum(
        close,
        h1_atr,
        horizon_rows=horizon_rows,
    ).astype(np.float32)
    out.attrs["m5h1_momentum_owner"] = {
        "owner": "native_m5_v4_projection",
        "decision_bar_seconds": int(decision_bar_duration.total_seconds()),
        "wall_clock_horizon_seconds": 3600,
        "horizon_rows": horizon_rows,
    }
    return out
