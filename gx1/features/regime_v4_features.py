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

# ── V29 Phase A additions (session_regime G2) ────────────────────────────────
# docs/V29_EVENT_SURFACE_DESIGN_20260811.md §3 +
# GX1_DATA/logs/event_gap_review_20260811/session_regime.md check 3 / G2.
# Per-TF regime-flip EVENTS for the four TFs whose flips are invisible today:
# the D1 flip exists below as F8/F9; the only per-TF proxy,
# `{tf}_trend_age_bars_norm_v2`, resets on EMA-stack sign change only and
# misses the 1<->2 / 3<->4 sub-flips (stated in the F9 comment). Construction
# mirrors F8/F9 verbatim (origin: d1_regime_changed_flag_v3 /
# bars_since_d1_regime_change_v3 in this file), keyed on the exact 5-class id,
# aged on each TF's OWN bar clock via `tf_bars` (the F9 unit repair precedent:
# divide base rows by tf_bars) — zero new numbers.
# DECLARED SEPARATELY from REGIME_V4_FEATURE_NAMES: the accepted ctx_cont /
# EXIT_IO tails bind the pre-V29 surface. The stage-2 V29 wiring wave adopts
# these names into the contracts together with the V29 rebuild, so train==serve
# moves at exactly one boundary (rule 6). NO cross-TF flip-alignment or
# flip-agreement aggregate is built (rule 4 / mtf_confluence precedent —
# excluded by the design doc §4); the fusion learns the coincidence from the
# per-TF flags + ages.
REGIME_V4_V29_FLIP_TFS = ("m5", "m15", "h1", "h4")
REGIME_V4_V29_ADDITION_COLS: List[str] = (
    [f"{tf}_regime_changed_flag_v3" for tf in REGIME_V4_V29_FLIP_TFS]
    + [f"{tf}_regime_flip_age_norm" for tf in REGIME_V4_V29_FLIP_TFS]
)
if len(set(REGIME_V4_V29_ADDITION_COLS)) != len(REGIME_V4_V29_ADDITION_COLS) or (
    set(REGIME_V4_V29_ADDITION_COLS)
    & set(REGIME_V4_FEATURE_NAMES + REGIME_V4_SOURCE_COLS)
):
    raise RuntimeError(
        "[REGIME_V4] V29 addition names must be unique and disjoint from the "
        "bound pre-V29 surface"
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


def _class_flip_flag_and_age(
    class_ids: np.ndarray,
    *,
    tf_bars_per_row: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Edge-triggered regime-class flip flag + normalized flip age for ONE TF.

    Vectorized mirror of the F8/F9 construction below (the D1 originals are the
    origin; a drift-guard test asserts this helper reproduces them exactly on
    the D1 lane):

      flag[t]     = class[t] != class[t-1]      (first row unknown -> NaN)  [F8]
      age_norm[t] = log1p(min((t - last_flip_row) / tf_bars_per_row, 500))
                    / log1p(500)                                             [F9]

    Keyed on the exact 5-class regime id, so 1<->2 / 3<->4 sub-flips fire (the
    trend-age proxy resets on EMA-stack sign only and misses them). On the base
    clock a {tf} class value repeats tf_bars_per_row times, so comparing
    consecutive base rows fires exactly once per own-TF step change, and the
    age is measured in that TF's OWN bars (F9 unit-repair precedent). Rows
    before the first observed flip stay NaN — the age since an unobserved flip
    is unknown (same honest warmup as F9).
    """
    ids = np.asarray(class_ids)
    if ids.ndim != 1 or ids.dtype != np.int64:
        raise RuntimeError("[REGIME_V4] flip helper requires a 1-D int64 class array")
    if not np.isfinite(float(tf_bars_per_row)) or tf_bars_per_row < 1.0:
        raise RuntimeError("[REGIME_V4] tf_bars_per_row must be >= 1")
    n = len(ids)
    flag = np.full(n, np.nan, dtype=np.float64)
    age_norm = np.full(n, np.nan, dtype=np.float64)
    if n > 1:
        changed = ids[1:] != ids[:-1]
        flag[1:] = changed.astype(np.float64)
        idx = np.arange(n, dtype=np.int64)
        change_row = np.where(
            np.concatenate((np.zeros(1, dtype=bool), changed)), idx, np.int64(-1)
        )
        last_change = np.maximum.accumulate(change_row)
        valid = last_change >= 0
        if valid.any():
            age = np.minimum(
                (idx[valid] - last_change[valid]) / float(tf_bars_per_row), 500.0
            )
            age_norm[valid] = np.log1p(age) / np.log1p(500.0)
    return flag, age_norm


def compute_regime_v29_flip_frame(
    class_frame: pd.DataFrame,
    *,
    base_bar_duration: pd.Timedelta = pd.Timedelta(minutes=5),
) -> pd.DataFrame:
    """Compute the 8 V29 per-TF regime-flip fields from the exact class ids.

    Canonical producer of ``REGIME_V4_V29_ADDITION_COLS`` for the seq513
    causal lane (V29 stage 2).  ``class_frame`` carries a chronological
    tz-aware UTC index plus the four ``{tf}_regime_class_id_v2`` columns on
    the base clock; each column may open with one honest NaN warmup prefix
    (the shared causal-prefix contract) and the flip state starts at the
    common finite suffix of the four columns.  The formula owner is
    :func:`_class_flip_flag_and_age` — the same helper the ctx-path
    ``add_regime_v4_features`` V29 branch uses, so the two call surfaces
    cannot drift.
    """
    if not isinstance(base_bar_duration, pd.Timedelta) or base_bar_duration <= pd.Timedelta(0):
        raise RuntimeError("[REGIME_V4] base bar duration must be positive")
    if any(
        int(shift.value) % int(base_bar_duration.value) != 0
        for shift in MULTI_TF_SHIFT.values()
    ):
        raise RuntimeError("[REGIME_V4] base bar duration must divide every MTF duration")
    if not isinstance(class_frame, pd.DataFrame) or class_frame.empty:
        raise RuntimeError("[REGIME_V4] flip frame requires a non-empty DataFrame")
    if not isinstance(class_frame.index, pd.DatetimeIndex) or class_frame.index.tz is None:
        raise RuntimeError("[REGIME_V4] flip frame requires a tz-aware UTC DatetimeIndex")
    if (
        class_frame.index.hasnans
        or not class_frame.index.is_unique
        or not class_frame.index.is_monotonic_increasing
    ):
        raise RuntimeError("[REGIME_V4] flip frame index must be unique and chronological")
    required = [f"{tf}_regime_class_id_v2" for tf in REGIME_V4_V29_FLIP_TFS]
    missing = [name for name in required if name not in class_frame.columns]
    if missing:
        raise RuntimeError(f"[REGIME_V4] flip source columns missing: {missing}")
    source = pd.DataFrame(
        {
            name: pd.to_numeric(class_frame[name], errors="coerce")
            for name in required
        },
        index=class_frame.index,
        copy=False,
    )
    source_values = source.to_numpy(dtype=np.float64)
    source_start = max(
        validate_causal_feature_matrix(
            source_values[:, column : column + 1],
            expected_width=1,
            context=f"REGIME_V4_V29_FLIP_SOURCE_{required[column]}",
        )
        for column in range(source_values.shape[1])
    )
    tf_bars = {
        timeframe: int(
            MULTI_TF_SHIFT[timeframe.upper()].value
            // int(base_bar_duration.value)
        )
        for timeframe in REGIME_V4_V29_FLIP_TFS
    }
    n_rows = len(class_frame)
    out = pd.DataFrame(index=class_frame.index, dtype=np.float64)
    for tf in REGIME_V4_V29_FLIP_TFS:
        column = np.full(n_rows, np.nan, dtype=np.float64)
        age_column = np.full(n_rows, np.nan, dtype=np.float64)
        suffix = source[f"{tf}_regime_class_id_v2"].to_numpy(dtype=np.float64)[
            source_start:
        ]
        # Exact-enum proof before the int64 cast (same check as the ctx path).
        _sign_from_class(suffix)
        flag, age_norm = _class_flip_flag_and_age(
            suffix.astype(np.int64),
            tf_bars_per_row=float(tf_bars[tf]),
        )
        column[source_start:] = flag
        age_column[source_start:] = age_norm
        out[f"{tf}_regime_changed_flag_v3"] = column
        out[f"{tf}_regime_flip_age_norm"] = age_column
    out = out.loc[:, list(REGIME_V4_V29_ADDITION_COLS)]
    if list(out.columns) != list(REGIME_V4_V29_ADDITION_COLS):
        raise RuntimeError("[REGIME_V4] flip frame output order invalid")
    return out


def add_regime_v4_features(
    df: pd.DataFrame,
    *,
    base_bar_duration: pd.Timedelta = pd.Timedelta(minutes=5),
    include_v29_additions: bool = False,
) -> pd.DataFrame:
    """Add REGIME_V4 derived features in place (and validate the reuse sources exist).

    The frame MUST be time-sorted ascending (shift/run-length depend on it). Build-side passes
    the full-history prebuilt; live-side passes the rolling cv3 window (must carry >=288 bars of
    D1-dist history for F4 to be exact. The causal prefix remains NaN and is trimmed
    by the shared warmup contract; missing columns fail closed.

    ``include_v29_additions`` additionally emits REGIME_V4_V29_ADDITION_COLS
    (per-TF flip flag + own-TF-bar flip age; see the tuple's comment). Default
    False == the accepted pre-V29 contract surface, byte-identical; the stage-2
    V29 wiring flips the canonical call sites explicitly together with the
    contract/dimension updates and the V29 rebuild — it is a call-site contract
    switch, never an environment gate.
    """
    if not isinstance(include_v29_additions, bool):
        raise RuntimeError("[REGIME_V4] include_v29_additions must be a bool")
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
    emitted_cols = list(REGIME_V4_DERIVED_COLS) + (
        list(REGIME_V4_V29_ADDITION_COLS) if include_v29_additions else []
    )
    derived = {
        name: np.full(n_rows, np.nan, dtype=np.float64)
        for name in emitted_cols
    }
    if source_start == n_rows:
        for name, values in derived.items():
            df[name] = values.astype(np.float32)
        df.attrs["causal_regime_v4_warmup_rows"] = n_rows
        return df

    suffix = source.iloc[source_start:]
    # Regime-class enum validation is NOT repeated here: the `signs` mapping
    # below calls _sign_from_class on the identical per-TF class arrays before
    # any derived output is emitted, so the exact 0..4 enum check still runs
    # fail-closed for every TF.
    for tf in _TFS:
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

    # F1: fraction of the four non-D1 TFs whose regime sign STRICTLY agrees
    # with a signed D1 regime: (sign == d1_sign) & (|d1_sign| > 0) -> [0,1].
    # Convention: identical to h4_d1_regime_sign_agreement in
    # entry_session_regime_interactions_v1. The old count let D1 vote on
    # itself (a guaranteed 1/5 floor) and counted neutral==neutral (0==0)
    # as agreement.
    non_d1_tfs = [tf for tf in _TFS if tf != "d1"]
    d1_is_signed = np.abs(d1_sign) > 0.0
    agree = np.mean(
        [
            ((signs[tf] == d1_sign) & d1_is_signed).astype(np.float64)
            for tf in non_d1_tfs
        ],
        axis=0,
    )
    derived["regime_tf_agreement_v3"][source_start:] = agree

    # F2: mean ema-stack alignment across TFs -> [-1,1]
    stack = np.mean(
        [suffix[f"{tf}_ema_stack_aligned_v2"].to_numpy(dtype=np.float64) for tf in _TFS],
        axis=0,
    )
    derived["regime_stack_sum_v3"][source_start:] = stack

    # F3: TFs disagree (divergence) -> transition onset. The <= 0.5 threshold
    # is unchanged: agree remains in [0,1] under the new count (values
    # {0, .25, .5, .75, 1} over 4 voters instead of {.2..1} over 5), and 0.5
    # is the exact half-agreement point of both ranges, so the flag keeps the
    # meaning "no strict majority of voting TFs agrees with D1" with no
    # rescaling algebraically required.
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
    #     _trend_age_bars counts bars OF ITS OWN TF, so the D1 age is measured
    #     in D1 bars: (row - last_change) / tf_bars["d1"] base rows per D1 bar,
    #     capped at 500 D1 bars. The old code counted base-clock rows against
    #     the same 500 cap (500 M5 rows = ~41h < 2 D1 bars), which saturated
    #     the normalization within two days of any change.
    transitions = np.flatnonzero(d1c[1:] != d1c[:-1]) + 1
    if len(transitions):
        first_transition = int(transitions[0])
        d1_bars_per_row = float(tf_bars["d1"])
        age = np.empty(len(d1c) - first_transition, dtype=np.float64)
        last_change = first_transition
        for offset, row in enumerate(range(first_transition, len(d1c))):
            if row > first_transition and d1c[row] != d1c[row - 1]:
                last_change = row
            age[offset] = min(float(row - last_change) / d1_bars_per_row, 500.0)
        derived["bars_since_d1_regime_change_v3"][source_start + first_transition:] = (
            np.log1p(age) / np.log1p(500.0)
        )

    # F10: D1 trend exhaustion proxy (reuses R2 trend-age)
    d1_age = suffix["d1_trend_age_bars_norm_v2"].to_numpy(dtype=np.float64)
    derived["d1_trend_age_mature_flag_v3"][source_start:] = (d1_age > 0.8).astype(np.float64)

    if include_v29_additions:
        # V29 (session_regime G2): per-TF flip flag + own-TF-bar flip age for
        # the four TFs without one (D1 exists as F8/F9 above — the origin the
        # helper mirrors). The class arrays were already enum-validated for
        # every TF by the `signs` mapping above; the int64 cast is exact.
        for tf in REGIME_V4_V29_FLIP_TFS:
            tf_classes = suffix[f"{tf}_regime_class_id_v2"].to_numpy(dtype=np.int64)
            flag, age_norm = _class_flip_flag_and_age(
                tf_classes,
                tf_bars_per_row=float(tf_bars[tf]),
            )
            derived[f"{tf}_regime_changed_flag_v3"][source_start:] = flag
            derived[f"{tf}_regime_flip_age_norm"][source_start:] = age_norm

    # Release each float64 buffer as it is cast, at the memory peak.
    for name in list(derived):
        df[name] = derived.pop(name).astype(np.float32)
    derived_values = df.loc[:, emitted_cols].to_numpy(dtype=np.float64)
    derived_start = validate_causal_feature_matrix(
        derived_values,
        expected_width=len(emitted_cols),
        context="REGIME_V4_DERIVED",
    )
    df.attrs["causal_regime_v4_warmup_rows"] = derived_start

    return df
