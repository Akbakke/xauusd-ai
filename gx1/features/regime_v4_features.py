"""Multi-TF regime CONDITIONING + regime-CHANGE-DETECTION features (REGIME_V4).

2026-06-03. ONE-TRUTH: both the build-side (add_ctx_cont_columns_to_prebuilt.py) and the
live-side (v12_ctx_augment_live.py) call `add_regime_v4_features` so the computation cannot
drift (train/serve skew = silent death). Reuse-first: the per-TF regime classes
(`{tf}_regime_class_id_v2`), trend-age (`{tf}_trend_age_bars_norm_v2`), and ema-stack
(`{tf}_ema_stack_aligned_v2`) are ALREADY computed in the canonical pipeline (htf_features.py)
but were never wired into the entry/exit models — this module wires them + derives the
"regime is shifting" change-detection signals on top.

Design: GX1_DATA/DESIGN_MULTI_TF_REGIME_CONDITIONING_CHANGEDETECT_20260603.md.

Gated by env GX1_REGIME_V4 at the call sites (default OFF = bit-parity with cement). When a
caller opts in, this is FAIL-CLOSED: missing source columns raise (no silent degenerate).

The feature list (REGIME_V4_FEATURE_NAMES) is what gets appended to the entry ctx_cont
contract (signal_bridge_v3.ORDERED_CTX_CONT_NAMES_V3) and the EXIT_IO_V8 tail. R1/R2 are
passthrough (already present); F* are derived here.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

_TFS = ("m15", "h1", "h4", "d1")
# M5-bar cadence per TF (one bar of TF = N M5 bars), for transition look-backs.
_TF_BARS = {"m15": 3, "h1": 12, "h4": 48, "d1": 288}

# Source columns this module REUSES (must exist on the frame when GX1_REGIME_V4 is enabled).
REGIME_V4_SOURCE_COLS: List[str] = (
    [f"{tf}_regime_class_id_v2" for tf in _TFS]        # R1
    + [f"{tf}_trend_age_bars_norm_v2" for tf in _TFS]  # R2
    + [f"{tf}_ema_stack_aligned_v2" for tf in _TFS]    # R3 (drives F2)
    + ["D1_dist_from_ema200_atr"]                       # drives F4/F6
)

# Derived "regime is shifting" + cross-TF state features this module CREATES.
REGIME_V4_DERIVED_COLS: List[str] = [
    "regime_tf_agreement_v3",          # F1  C  cross-TF agreement w/ D1 sign [0,1]
    "regime_stack_sum_v3",             # F2  C  mean ema-stack [-1,1]
    "regime_divergence_flag_v3",       # F3  C->T  TFs disagree
    "d1_dist_roc_288_v3",              # F4  T  D1-dist rate-of-change (momentum)
    "d1_dist_to_boundary_v3",          # F6  T  |D1-dist| small = near sign-flip
    "d1_regime_changed_flag_v3",       # F8  T  regime class changed vs prev bar
    "bars_since_d1_regime_change_v3",  # F9  T  recency of last D1 regime change [0,1]
    "d1_trend_age_mature_flag_v3",     # F10 C->T  trend exhaustion proxy
]

# Full ctx_cont extension block (R1 + R2 reuse + derived). R3 ema-stack is NOT appended to
# ctx_cont (it only feeds F2); add it separately if a model wants the raw per-TF stack.
REGIME_V4_FEATURE_NAMES: List[str] = (
    [f"{tf}_regime_class_id_v2" for tf in _TFS]
    + [f"{tf}_trend_age_bars_norm_v2" for tf in _TFS]
    + REGIME_V4_DERIVED_COLS
)


def _sign_from_class(class_id: np.ndarray) -> np.ndarray:
    """Per-TF regime sign from the 5-class regime id (htf_features._regime_class enum):
    classes {1,2} = up, {3,4} = down, 0 = neutral/none."""
    c = np.asarray(class_id, dtype=np.int64)
    return np.where(np.isin(c, (1, 2)), 1, np.where(np.isin(c, (3, 4)), -1, 0)).astype(np.float64)


def add_regime_v4_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add REGIME_V4 derived features in place (and validate the reuse sources exist).

    The frame MUST be time-sorted ascending (shift/run-length depend on it). Build-side passes
    the full-history prebuilt; live-side passes the rolling cv3 window (must carry >=288 bars of
    D1-dist history for F4 to be exact, else early rows clip to 0 — acceptable, fail-soft on
    history depth only, fail-CLOSED on missing columns).
    """
    missing = [c for c in REGIME_V4_SOURCE_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"[REGIME_V4] required source columns missing (GX1_REGIME_V4 enabled but pipeline "
            f"did not provide them): {missing}"
        )

    signs = {tf: _sign_from_class(df[f"{tf}_regime_class_id_v2"].to_numpy()) for tf in _TFS}
    d1_sign = signs["d1"]

    # F1: fraction of TFs whose regime sign agrees with D1 (cross-TF agreement) -> [0,1]
    agree = np.mean([(signs[tf] == d1_sign).astype(np.float64) for tf in _TFS], axis=0)
    df["regime_tf_agreement_v3"] = agree.astype(np.float32)

    # F2: mean ema-stack alignment across TFs -> [-1,1]
    stack = np.mean(
        [np.nan_to_num(df[f"{tf}_ema_stack_aligned_v2"].to_numpy(dtype=np.float64), nan=0.0) for tf in _TFS],
        axis=0,
    )
    df["regime_stack_sum_v3"] = stack.astype(np.float32)

    # F3: TFs disagree (divergence) -> transition onset
    df["regime_divergence_flag_v3"] = (agree <= 0.5).astype(np.float32)

    # F4: D1-dist rate-of-change over ~1 D1 bar (288 M5 bars). Clip MANDATORY (corrupt tails).
    d1d = np.nan_to_num(df["D1_dist_from_ema200_atr"].to_numpy(dtype=np.float64), nan=0.0)
    roc = d1d - pd.Series(d1d).shift(_TF_BARS["d1"]).fillna(0.0).to_numpy()
    df["d1_dist_roc_288_v3"] = np.clip(roc, -5.0, 5.0).astype(np.float32)

    # F6: |D1-dist| small = near the sign-flip boundary = instability
    df["d1_dist_to_boundary_v3"] = np.clip(np.abs(d1d), 0.0, 5.0).astype(np.float32)

    # F8: D1 regime class changed vs previous bar
    d1c = df["d1_regime_class_id_v2"].to_numpy(dtype=np.int64)
    changed = np.zeros(len(d1c), dtype=np.float64)
    if len(d1c) > 1:
        changed[1:] = (d1c[1:] != d1c[:-1]).astype(np.float64)
    df["d1_regime_changed_flag_v3"] = changed.astype(np.float32)

    # F9: bars-since-last-D1-regime-change, normalized log1p/log1p(500) -> [0,1] (recency).
    #     Same construction as htf_features._trend_age_bars but keyed on the regime CLASS
    #     (catches 1<->2 / 3<->4 sub-flips the ema-stack misses).
    s = pd.Series(d1c)
    grp = (s != s.shift(1)).cumsum()
    age = s.groupby(grp).cumcount().to_numpy(dtype=np.float64)
    age = np.clip(age, 0.0, 500.0)
    df["bars_since_d1_regime_change_v3"] = (np.log1p(age) / np.log1p(500.0)).astype(np.float32)

    # F10: D1 trend exhaustion proxy (reuses R2 trend-age)
    d1_age = np.nan_to_num(df["d1_trend_age_bars_norm_v2"].to_numpy(dtype=np.float64), nan=0.0)
    df["d1_trend_age_mature_flag_v3"] = (d1_age > 0.8).astype(np.float32)

    return df
