"""Multi-TF regime CONDITIONING + regime-CHANGE-DETECTION features (REGIME_V4).

2026-06-03. ONE-TRUTH: both the build-side (add_ctx_cont_columns_to_prebuilt.py) and the
live-side (v12_ctx_augment_live.py) call `add_regime_v4_features` so the computation cannot
drift (train/serve skew = silent death). Reuse-first: the per-TF regime classes
(`{tf}_regime_class_id_v2`), trend-age (`{tf}_trend_age_bars_norm_v2`), and ema-stack
(`{tf}_ema_stack_aligned_v2`) are ALREADY computed in the canonical pipeline (htf_features.py)
but were never wired into the entry/exit models — this module wires them + derives the
"regime is shifting" change-detection signals on top.

The full surface is unconditional in model-native Entry. Missing source columns
raise; no environment-selected subset or neutral substitute exists.

The feature list (REGIME_V4_FEATURE_NAMES) is the tail of the model-native
Entry ctx_cont contract and the EXIT_IO_V8 tail. R1/R2 are passthrough
(already present); F* are derived here.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from gx1.features.htf_features import (
    MULTI_TF_BARS_IN_M5,
    REGIME_V4_MTF_TIMEFRAMES,
    MULTI_TF_SHIFT,
    validate_causal_feature_matrix,
)

# 2026-06-05 (user vedtak): M5 ADDED — the immediate-flow TF is the MOST important for the entry's
# direction ("trade with the trend her og nå"). M5 now participates in the regime-class cross-TF
# agreement/divergence (regime_tf_agreement_v3 etc.), not just the M5 seq. Appended (not prepended) so the
# m15/h1/h4/d1 features keep their relative order; everything is by-name so order is non-load-bearing.
# Grows REGIME_V4_FEATURE_NAMES 16->18 -> ctx_cont 121->123, EXIT_IO_V8 171->173 (contracts import dynamically).
_TFS = REGIME_V4_MTF_TIMEFRAMES
# M5-bar cadence per TF (one bar of TF = N M5 bars), for transition look-backs.
_TF_BARS = {
    timeframe: MULTI_TF_BARS_IN_M5[timeframe]
    for timeframe in _TFS
}

# Source columns this module reuses; every one is mandatory.
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
    raw = np.asarray(class_id, dtype=np.float64)
    if (
        raw.ndim != 1
        or not np.isfinite(raw).all()
        or not np.equal(raw, np.rint(raw)).all()
        or np.any((raw < 0.0) | (raw > 4.0))
    ):
        raise RuntimeError("[REGIME_V4] regime classes must use exact finite enum values 0..4")
    c = raw.astype(np.int64)
    return np.where(np.isin(c, (1, 2)), 1, np.where(np.isin(c, (3, 4)), -1, 0)).astype(np.float64)


def add_regime_v4_features(
    df: pd.DataFrame,
    *,
    base_bar_duration: pd.Timedelta = pd.Timedelta(minutes=5),
) -> pd.DataFrame:
    """Add REGIME_V4 derived features in place (and validate the reuse sources exist).

    The frame MUST be time-sorted ascending (shift/run-length depend on it). Build-side passes
    the full-history prebuilt; live-side passes the rolling cv3 window (must carry >=288 bars of
    D1-dist history for F4 to be exact. The causal prefix remains NaN and is trimmed
    by the shared warmup contract; missing columns fail closed.
    """
    if not isinstance(base_bar_duration, pd.Timedelta) or base_bar_duration <= pd.Timedelta(0):
        raise RuntimeError("[REGIME_V4] base bar duration must be positive")
    if any(
        int(shift.value) % int(base_bar_duration.value) != 0
        for shift in MULTI_TF_SHIFT.values()
    ):
        raise RuntimeError("[REGIME_V4] base bar duration must divide every MTF duration")
    tf_bars = {
        timeframe: int(
            MULTI_TF_SHIFT[timeframe.upper()].value
            // int(base_bar_duration.value)
        )
        for timeframe in _TFS
    }
    missing = [c for c in REGIME_V4_SOURCE_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"[REGIME_V4] required source columns missing: {missing}"
        )
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
        raise RuntimeError("[REGIME_V4] frame requires a timezone-aware UTC DatetimeIndex")
    if any(pd.Timestamp(ts).utcoffset() != pd.Timedelta(0) for ts in df.index[:1]):
        raise RuntimeError("[REGIME_V4] frame index must be UTC")
    if df.empty or df.index.hasnans or not df.index.is_unique or not df.index.is_monotonic_increasing:
        raise RuntimeError("[REGIME_V4] frame must be non-empty, unique and chronological")
    # Build the numeric view in one pass instead of .loc-copy then .apply-copy.
    # Same values, same column order, one fewer full-frame duplicate - which on
    # the native M1 clock is 341 MB per avoided copy.
    source = pd.DataFrame(
        {name: pd.to_numeric(df[name], errors="coerce") for name in REGIME_V4_SOURCE_COLS},
        index=df.index,
        copy=False,
    )
    source_values = source.to_numpy(dtype=np.float64)
    source_warmups = [
        validate_causal_feature_matrix(
            source_values[:, column : column + 1],
            expected_width=1,
            context=f"REGIME_V4_SOURCE_{REGIME_V4_SOURCE_COLS[column]}",
        )
        for column in range(source_values.shape[1])
    ]
    source_start = max(source_warmups)
    n_rows = len(df)
    derived = {
        name: np.full(n_rows, np.nan, dtype=np.float64)
        for name in REGIME_V4_DERIVED_COLS
    }
    if source_start == n_rows:
        for name, values in derived.items():
            df[name] = values.astype(np.float32)
        df.attrs["causal_regime_v4_warmup_rows"] = n_rows
        return df

    suffix = source.iloc[source_start:]
    for tf in _TFS:
        classes = suffix[f"{tf}_regime_class_id_v2"].to_numpy(dtype=np.float64)
        _sign_from_class(classes)
        age = suffix[f"{tf}_trend_age_bars_norm_v2"].to_numpy(dtype=np.float64)
        stack_values = suffix[f"{tf}_ema_stack_aligned_v2"].to_numpy(dtype=np.float64)
        if np.any((age < 0.0) | (age > 1.0)):
            raise RuntimeError(f"[REGIME_V4] {tf} trend age must be within [0, 1]")
        if not np.isin(stack_values, (-1.0, 0.0, 1.0)).all():
            raise RuntimeError(f"[REGIME_V4] {tf} EMA stack must use exact enum -1/0/1")

    signs = {
        tf: _sign_from_class(suffix[f"{tf}_regime_class_id_v2"].to_numpy(dtype=np.float64))
        for tf in _TFS
    }
    d1_sign = signs["d1"]

    # F1: fraction of TFs whose regime sign agrees with D1 (cross-TF agreement) -> [0,1]
    agree = np.mean([(signs[tf] == d1_sign).astype(np.float64) for tf in _TFS], axis=0)
    derived["regime_tf_agreement_v3"][source_start:] = agree

    # F2: mean ema-stack alignment across TFs -> [-1,1]
    stack = np.mean(
        [suffix[f"{tf}_ema_stack_aligned_v2"].to_numpy(dtype=np.float64) for tf in _TFS],
        axis=0,
    )
    derived["regime_stack_sum_v3"][source_start:] = stack

    # F3: TFs disagree (divergence) -> transition onset
    derived["regime_divergence_flag_v3"][source_start:] = (agree <= 0.5).astype(np.float64)

    # F4: D1-dist rate-of-change over ~1 D1 bar (288 M5 bars). Clip MANDATORY (corrupt tails).
    d1d = suffix["D1_dist_from_ema200_atr"].to_numpy(dtype=np.float64)
    roc_lookback = tf_bars["d1"]
    if len(d1d) > roc_lookback:
        roc = np.clip(d1d[roc_lookback:] - d1d[:-roc_lookback], -5.0, 5.0)
        derived["d1_dist_roc_288_v3"][source_start + roc_lookback:] = roc

    # F6: |D1-dist| small = near the sign-flip boundary = instability
    derived["d1_dist_to_boundary_v3"][source_start:] = np.clip(np.abs(d1d), 0.0, 5.0)

    # F8: D1 regime class changed vs previous bar
    d1c = suffix["d1_regime_class_id_v2"].to_numpy(dtype=np.int64)
    changed = np.full(len(d1c), np.nan, dtype=np.float64)
    if len(d1c) > 1:
        changed[1:] = (d1c[1:] != d1c[:-1]).astype(np.float64)
    derived["d1_regime_changed_flag_v3"][source_start:] = changed

    # F9: bars-since-last-D1-regime-change, normalized log1p/log1p(500) -> [0,1] (recency).
    #     Same construction as htf_features._trend_age_bars but keyed on the regime CLASS
    #     (catches 1<->2 / 3<->4 sub-flips the ema-stack misses).
    transitions = np.flatnonzero(d1c[1:] != d1c[:-1]) + 1
    if len(transitions):
        first_transition = int(transitions[0])
        age = np.empty(len(d1c) - first_transition, dtype=np.float64)
        last_change = first_transition
        for offset, row in enumerate(range(first_transition, len(d1c))):
            if row > first_transition and d1c[row] != d1c[row - 1]:
                last_change = row
            age[offset] = min(float(row - last_change), 500.0)
        derived["bars_since_d1_regime_change_v3"][source_start + first_transition:] = (
            np.log1p(age) / np.log1p(500.0)
        )

    # F10: D1 trend exhaustion proxy (reuses R2 trend-age)
    d1_age = suffix["d1_trend_age_bars_norm_v2"].to_numpy(dtype=np.float64)
    derived["d1_trend_age_mature_flag_v3"][source_start:] = (d1_age > 0.8).astype(np.float64)

    # Release each float64 buffer as it is cast, at the memory peak.
    for name in list(derived):
        df[name] = derived.pop(name).astype(np.float32)
    derived_values = df.loc[:, REGIME_V4_DERIVED_COLS].to_numpy(dtype=np.float64)
    derived_start = validate_causal_feature_matrix(
        derived_values,
        expected_width=len(REGIME_V4_DERIVED_COLS),
        context="REGIME_V4_DERIVED",
    )
    df.attrs["causal_regime_v4_warmup_rows"] = derived_start

    return df
