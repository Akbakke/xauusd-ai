"""ONE-TRUTH swing-structure ctx features for the V10 entry contract.

Computes the 5 swing-structure features the V10 ctx_cont contract carries:
  dist_last_swing_high_atr, dist_last_swing_low_atr,
  bars_since_swing_high, bars_since_swing_low, retracement_from_last_impulse.

LOOKAHEAD-SAFE: a swing pivot at bar j (high[j] strictly exceeds its `lookback`
neighbours on BOTH sides) is only REFLECTED into the features from bar j+lookback
— never AT bar j — so the value at bar i uses only pivots confirmed by bar i. The
live decision bar (last row) is therefore causal, and train == serve bit-for-bit.

WHY THIS FILE (rule 7): until 2026-06-24 this math lived in TWO copies — the live
augmenter (v12_ctx_augment_live._add_swing_features) and the V10 training-dataset
builder (build_entry_v10_ctx_training_dataset_v3) — and one of them reflected the
pivot AT bar j (a 2-bar look-ahead). Both now delegate here so the computation can
only ever exist once. smc_v1.py was considered but it owns the smc_* family with a
different lookback (3); these are the entry ctx_cont swing features (lookback 2).
Do NOT re-implement this elsewhere — import it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gx1.features.entry_foundation_structure_v1 import FOUNDATION_EVENT_AGE_CAP

SWING_FEATURE_NAMES_V1 = (
    "dist_last_swing_high_atr",
    "dist_last_swing_low_atr",
    "bars_since_swing_high",
    "bars_since_swing_low",
    "retracement_from_last_impulse",
)
SWING_LOOKBACK_V1 = 2
SWING_ATR_PERIOD_V1 = 14

# ── V29 Phase A additions (structure_swing G1/G2/G4) ─────────────────────────
# docs/V29_EVENT_SURFACE_DESIGN_20260811.md §3 +
# GX1_DATA/logs/event_gap_review_20260811/structure_swing.md G1/G2/G4.
# The broken level is THIS file's own last CONFIRMED lookback-2 entry swing
# (last_high/last_low in the fill loop below) — the local pivot truth. It is a
# DIFFERENT level definition from smc_v1's lookback-3 swing that smc_bos_*
# breaks (structure report §5: no ownership collision); nothing is imported
# from smc_v1. G3 retest events are explicitly Phase B (design doc §3) and are
# NOT built here.
#   G1  swing_high/low_break_event: first closed-bar close strictly through the
#       last confirmed level, armed once per level (edge semantics per the
#       repaired-BOS idiom, smc_v1 `cond & ~prev_cond`: fires only on the first
#       crossing, never while price stays beyond), + one signed
#       swing_break_displacement_atr at the event bar (0 off-event — the
#       smc_bos_* event encoding).
#   G2  bars_since_swing_{high,low}_break: the entry_foundation_structure_v1
#       `_bars_since_event` convention (cap = FOUNDATION_EVENT_AGE_CAP, age
#       initialized at the cap, 0 on the event bar, +1 capped after).
#   G4  swing_{high,low}_sequence_delta_atr + consecutive_higher_lows_count /
#       consecutive_lower_highs_count: real pivot-price arithmetic retaining
#       ONE prior confirmed pivot per side (report spec); counts capped at
#       FOUNDATION_EVENT_AGE_CAP and normalized log1p(x)/log1p(cap) (the
#       htf_features `trend_age_bars_norm` convention).
# Constant origins (rule 2a): SWING_LOOKBACK_V1=2, SWING_ATR_PERIOD_V1=14,
# FOUNDATION_EVENT_AGE_CAP=96 (imported — one truth), log1p(x)/log1p(cap)
# convention. Zero new numbers (design doc §3, structure_swing row).
# DECLARED SEPARATELY from SWING_FEATURE_NAMES_V1 because the pre-V30 accepted
# contracts bound the 5-name V1 surface only.  That declaration is PERFORMED as
# of V30 package 2 (2026-08-13): the promised adoption — "the stage-2 V29
# wiring adopts these names into the ctx/111-surface contracts together with
# the V29 rebuild (rule 6: train==serve moves at one boundary)" — is now done
# at the V30 rebuild boundary.  ``MODEL_NATIVE_CTX_CONT_SWING_FIELDS``
# (entry_model_native_signal_v1) is ``SWING_FEATURE_NAMES_V1 +
# SWING_V29_ADDITION_NAMES_V1``, and ``MULTI_TF_V4_SWING_FEATURES``
# (htf_features) carries the same nine names per TF.  The two tuples stay
# separate because the V1 five keep the historical ``swing_``-prefixed per-TF
# spelling and the seq513 lane routes the additions through their own
# ``swing_structure_event_layer``.
# V30 package 8A (2026-08-13) appends six EMISSION-ONLY additions — every one
# of them is a quantity this function already computes (or already holds in a
# loop variable) and then throws away; no new mechanism, no new magnitude
# (docs/INDICATOR_FIDELITY_AUDIT_20260813.md §5):
#   consecutive_higher_highs_count / consecutive_lower_lows_count: the two
#     MISSING members of the four-run-counter set. Identical arithmetic to the
#     two G4 counters above with the opposite strict comparison, the same
#     FOUNDATION_EVENT_AGE_CAP and the same log1p(min(run,cap))/log1p(cap)
#     normalization. Without them "uptrend structure unbroken for N pivots" is
#     not formable from any emitted field.
#   swing_high_level_intact / swing_low_level_intact: the G1 ``armed_high`` /
#     ``armed_low`` loop state, which IS exactly "the last confirmed swing
#     high/low has not been closed through" — structure intact. Set on
#     adoption, cleared on the break bar, previously never emitted.
#   bars_since_swing_high_norm / bars_since_swing_low_norm: the ONE age
#     convention (log1p(min(age, 500))/log1p(500)) applied to the two raw
#     unbounded V1 ages, using the existing owner helper
#     ``htf_features._event_age_norm`` (imported, never re-derived). The raw
#     V1 fields stay byte-identical.
SWING_V29_ADDITION_NAMES_V1 = (
    "swing_high_break_event",
    "swing_low_break_event",
    "swing_break_displacement_atr",
    "bars_since_swing_high_break",
    "bars_since_swing_low_break",
    "swing_high_sequence_delta_atr",
    "swing_low_sequence_delta_atr",
    "consecutive_higher_lows_count",
    "consecutive_lower_highs_count",
    "consecutive_higher_highs_count",
    "consecutive_lower_lows_count",
    "swing_high_level_intact",
    "swing_low_level_intact",
    "bars_since_swing_high_norm",
    "bars_since_swing_low_norm",
)
# Columns whose honest warmup is a leading NaN prefix (no delta exists until a
# SECOND pivot on that side is confirmed; emitting 0 would fabricate "equal
# pivots", rule 2e). The shared HTF matrix owner trims a single chronological
# prefix, so NaN is legal only as that prefix — enforced in the output guard.
#
# The two V30 intact flags join the set for the same reason: before the FIRST
# confirmed pivot on that side there is no level, so "intact" has no truth
# value and a 0 would read as "the level has been broken" (rule 2e). Their
# prefix is a strict SUBSET of the sequence-delta prefix on the same side (the
# delta needs a SECOND pivot, the intact flag only the first), so the shared
# HTF matrix owner's single-prefix trim length is unchanged by their addition.
_SWING_V29_NAN_PREFIX_NAMES = (
    "swing_high_sequence_delta_atr",
    "swing_low_sequence_delta_atr",
    "swing_high_level_intact",
    "swing_low_level_intact",
)
if len(set(SWING_V29_ADDITION_NAMES_V1)) != len(SWING_V29_ADDITION_NAMES_V1) or (
    set(SWING_V29_ADDITION_NAMES_V1) & set(SWING_FEATURE_NAMES_V1)
):
    raise RuntimeError(
        "SWING_STRUCTURE_V29_NAMES_INVALID: addition names must be unique and "
        "disjoint from the bound V1 surface"
    )
if not set(_SWING_V29_NAN_PREFIX_NAMES) <= set(SWING_V29_ADDITION_NAMES_V1):
    raise RuntimeError("SWING_STRUCTURE_V29_NAN_PREFIX_NAMES_INVALID")


def _bars_since_event_capped(events: np.ndarray, *, cap: int) -> np.ndarray:
    """Age-in-bars since the last event, initialized AT the cap.

    Identical algorithm to entry_foundation_structure_v1._bars_since_event
    (the cited convention owner): age starts at ``cap`` ("no event within the
    cap window"), resets to 0 on an event bar, else increments capped. No
    carry scope is needed here: both canonical callers (the V4 per-TF lane and
    the ctx builders) run on full contiguous history; a windowed live path
    adopting these fields must add carry semantics first (stage-2 concern).
    """
    flags = np.asarray(events, dtype=bool)
    if flags.ndim != 1:
        raise RuntimeError("SWING_STRUCTURE_V29_EVENT_FLAGS_INVALID")
    if isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0:
        raise RuntimeError("SWING_STRUCTURE_V29_AGE_CAP_INVALID")
    out = np.empty(flags.shape[0], dtype=np.float64)
    age = cap
    for i, hit in enumerate(flags):
        if bool(hit):
            age = 0
        else:
            age = min(age + 1, cap)
        out[i] = float(age)
    return out


def compute_swing_structure_features(
    high,
    low,
    close,
    *,
    lookback: int = SWING_LOOKBACK_V1,
    atr_period: int = SWING_ATR_PERIOD_V1,
    eps: float = 1e-9,
    include_v29_additions: bool = False,
) -> dict[str, np.ndarray]:
    """Return {feature_name: float32 ndarray} for the 5 swing-structure features.

    high/low/close: equal-length, chronologically-ordered 1-D array-likes (np or pd).
    `lookback` is BOTH the pivot half-window and the confirmation lag (a pivot at j is
    reflected from bar j+lookback). The first/last `lookback` bars can never be pivots.
    Bit-identical to the pre-2026-06-24 live `_add_swing_features` (causal-fixed variant).

    ``include_v29_additions`` appends SWING_V29_ADDITION_NAMES_V1 (break events,
    displacement, break ages, pivot-sequence deltas, run counts — see the
    tuple's comment). Default False == the accepted pre-V29 contract surface,
    byte-identical. V30 package 2 (2026-08-13) flipped every canonical call
    site to True together with the contract/dimension updates (a call-site
    contract switch, never an environment gate): the seq513 swing event layer,
    the per-TF V4 lane, and the three ctx producers (offline dataset builder,
    live ctx augmenter, model-native state frame). The default stays False so
    the pre-V30 surface remains reproducible byte-for-byte from this owner.
    """
    if not isinstance(include_v29_additions, bool):
        raise RuntimeError("SWING_STRUCTURE_V29_FLAG_INVALID")
    h = np.asarray(high, dtype=np.float64)
    low_values = np.asarray(low, dtype=np.float64)
    c = np.asarray(close, dtype=np.float64)
    if h.ndim != 1 or low_values.ndim != 1 or c.ndim != 1:
        raise RuntimeError(
            "SWING_STRUCTURE_SOURCE_NOT_1D: "
            f"high={h.shape} low={low_values.shape} close={c.shape}"
        )
    if not (len(h) == len(low_values) == len(c)) or len(c) == 0:
        raise RuntimeError(
            "SWING_STRUCTURE_SOURCE_LENGTH_INVALID: "
            f"high={len(h)} low={len(low_values)} close={len(c)}"
        )
    if (
        not np.isfinite(h).all()
        or not np.isfinite(low_values).all()
        or not np.isfinite(c).all()
    ):
        raise RuntimeError("SWING_STRUCTURE_SOURCE_NONFINITE")
    if np.any(h <= 0.0) or np.any(low_values <= 0.0) or np.any(c <= 0.0):
        raise RuntimeError("SWING_STRUCTURE_SOURCE_NONPOSITIVE")
    if np.any(h < low_values) or np.any(h < c) or np.any(low_values > c):
        raise RuntimeError("SWING_STRUCTURE_SOURCE_GEOMETRY_INVALID")
    if isinstance(lookback, bool) or not isinstance(lookback, int) or lookback < 1:
        raise RuntimeError(f"SWING_STRUCTURE_LOOKBACK_INVALID: {lookback!r}")
    if (
        isinstance(atr_period, bool)
        or not isinstance(atr_period, int)
        or atr_period < 1
    ):
        raise RuntimeError(f"SWING_STRUCTURE_ATR_PERIOD_INVALID: {atr_period!r}")
    if not np.isfinite(float(eps)) or float(eps) <= 0.0:
        raise RuntimeError(f"SWING_STRUCTURE_EPS_INVALID: {eps!r}")
    n = len(c)

    # ATR = TR rolling-mean (pandas rolling, matching the live/train convention).
    prev_close = np.empty(n, dtype=np.float64)
    if n:
        prev_close[0] = c[0]
        prev_close[1:] = c[:-1]
    tr = np.maximum(
        np.abs(h - low_values),
        np.maximum(np.abs(h - prev_close), np.abs(low_values - prev_close)),
    )
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=1).mean().to_numpy()
    atr_safe = np.clip(atr, eps, None)

    # Pivot detection: strict, full ±lookback window (so the first/last `lookback` bars
    # are never pivots — same edge convention as the live decision bar).
    pivot_high = np.zeros(n, dtype=bool)
    pivot_low = np.zeros(n, dtype=bool)
    for i in range(lookback, n - lookback):
        if (
            h[i] > h[i - lookback : i].max()
            and h[i] > h[i + 1 : i + lookback + 1].max()
        ):
            pivot_high[i] = True
        if (
            low_values[i] < low_values[i - lookback : i].min()
            and low_values[i] < low_values[i + 1 : i + lookback + 1].min()
        ):
            pivot_low[i] = True

    # Confirmation-lag forward fill: reflect a pivot at bar j only from bar j+lookback.
    last_high_vals = np.empty(n, dtype=np.float64)
    last_low_vals = np.empty(n, dtype=np.float64)
    last_high_idx = np.empty(n, dtype=np.int64)
    last_low_idx = np.empty(n, dtype=np.int64)
    last_high = float(h[0]) if n else 0.0
    last_low = float(low_values[0]) if n else 0.0
    last_hi_i = 0
    last_lo_i = 0
    # V29 state (G1/G4). The bar-0 backfill values above are NOT confirmed
    # pivots: arming, prev-pivot retention and run counts start only at the
    # first ADOPTION (pivot confirmed at j+lookback) — levels are usable only
    # post-confirmation.
    armed_high = False
    armed_low = False
    high_adoptions = 0
    low_adoptions = 0
    prev_high_val = np.nan  # previous confirmed swing-high price (one prior pivot, G4)
    prev_low_val = np.nan
    last_high_adopt_i = -1
    last_low_adopt_i = -1
    lh_run = 0  # consecutive lower highs (updates on high-pivot adoption)
    hl_run = 0  # consecutive higher lows (updates on low-pivot adoption)
    # V30 package 8A: the two MISSING members of the run-counter set, same
    # arithmetic, opposite strict comparison.
    hh_run = 0  # consecutive higher highs (updates on high-pivot adoption)
    ll_run = 0  # consecutive lower lows (updates on low-pivot adoption)
    high_break = np.zeros(n, dtype=np.float64)
    low_break = np.zeros(n, dtype=np.float64)
    displacement = np.zeros(n, dtype=np.float64)
    prev_high_vals = np.full(n, np.nan, dtype=np.float64)
    prev_low_vals = np.full(n, np.nan, dtype=np.float64)
    lh_run_vals = np.zeros(n, dtype=np.float64)
    hl_run_vals = np.zeros(n, dtype=np.float64)
    hh_run_vals = np.zeros(n, dtype=np.float64)
    ll_run_vals = np.zeros(n, dtype=np.float64)
    # V30 package 8A: the post-update ``armed_*`` state per bar (NaN before the
    # first confirmed pivot on that side — no level exists, so "intact" has no
    # truth value).
    high_intact_vals = np.full(n, np.nan, dtype=np.float64)
    low_intact_vals = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        j = i - lookback
        if j >= 0 and pivot_high[j]:
            new_high = float(h[j])
            if high_adoptions > 0:
                # G4: strict pivot-vs-pivot comparison (strictness matches the
                # strict pivot detector above; an equal high is not a lower high).
                lh_run = lh_run + 1 if new_high < last_high else 0
                hh_run = hh_run + 1 if new_high > last_high else 0
                prev_high_val = last_high
            last_high = new_high
            last_hi_i = j
            high_adoptions += 1
            armed_high = True  # armed once per level (G1)
            last_high_adopt_i = i
        if j >= 0 and pivot_low[j]:
            new_low = float(low_values[j])
            if low_adoptions > 0:
                hl_run = hl_run + 1 if new_low > last_low else 0
                ll_run = ll_run + 1 if new_low < last_low else 0
                prev_low_val = last_low
            last_low = new_low
            last_lo_i = j
            low_adoptions += 1
            armed_low = True
            last_low_adopt_i = i
        last_high_vals[i] = last_high
        last_low_vals[i] = last_low
        last_high_idx[i] = last_hi_i
        last_low_idx[i] = last_lo_i
        prev_high_vals[i] = prev_high_val
        prev_low_vals[i] = prev_low_val
        lh_run_vals[i] = float(lh_run)
        hl_run_vals[i] = float(hl_run)
        hh_run_vals[i] = float(hh_run)
        ll_run_vals[i] = float(ll_run)
        # G1 break checks: strictly causal — the armed level is a confirmed
        # pivot and the trigger is THIS closed bar's close (strict crossing,
        # the smc_bos "close first crosses" convention). At the adoption bar
        # itself a break can never fire: the strict pivot window guarantees
        # close[i] < a newly adopted high level and > a newly adopted low
        # level, so arming and checking in one pass is exact.
        fired_high = armed_high and c[i] > last_high
        fired_low = armed_low and c[i] < last_low
        if fired_high:
            high_break[i] = 1.0
            armed_high = False
        if fired_low:
            low_break[i] = 1.0
            armed_low = False
        if fired_high and fired_low:
            # Two genuine first-crossings on one bar (possible when the armed
            # low level sits above an older armed high level). One signed
            # displacement field (G1 is 3 fields): carry the break of the MORE
            # RECENTLY adopted level — the current structural boundary. A
            # same-bar double adoption (outside-bar pivot at the same j) ties
            # on the high side: a fixed documented order, not a data guess.
            if last_low_adopt_i > last_high_adopt_i:
                displacement[i] = (c[i] - last_low) / atr_safe[i]
            else:
                displacement[i] = (c[i] - last_high) / atr_safe[i]
        elif fired_high:
            displacement[i] = (c[i] - last_high) / atr_safe[i]
        elif fired_low:
            displacement[i] = (c[i] - last_low) / atr_safe[i]
        # V30 package 8A: emit the post-update G1 arming state — "the last
        # confirmed swing high/low has not been closed through" = structure
        # intact. It is read AFTER the break check on this bar, so the bar
        # whose close breaks the level already reads 0 (that close IS the
        # break). Before the first adoption on that side no level exists, so
        # the value stays NaN (one leading prefix, rule 2e).
        if high_adoptions > 0:
            high_intact_vals[i] = 1.0 if armed_high else 0.0
        if low_adoptions > 0:
            low_intact_vals[i] = 1.0 if armed_low else 0.0

    idx = np.arange(n, dtype=np.int64)
    denom = np.maximum(last_high_vals - last_low_vals, eps)
    retracement = np.zeros(n, dtype=np.float64)
    up_mask = last_high_idx > last_low_idx
    down_mask = last_low_idx > last_high_idx
    retracement[up_mask] = (last_high_vals[up_mask] - c[up_mask]) / denom[up_mask]
    retracement[down_mask] = (c[down_mask] - last_low_vals[down_mask]) / denom[
        down_mask
    ]
    retracement = np.clip(retracement, 0.0, 1.0)

    result = {
        "dist_last_swing_high_atr": ((c - last_high_vals) / atr_safe).astype(
            np.float32
        ),
        "dist_last_swing_low_atr": ((c - last_low_vals) / atr_safe).astype(np.float32),
        "bars_since_swing_high": (idx - last_high_idx).astype(np.float32),
        "bars_since_swing_low": (idx - last_low_idx).astype(np.float32),
        "retracement_from_last_impulse": retracement.astype(np.float32),
    }
    expected_names = SWING_FEATURE_NAMES_V1
    if include_v29_additions:
        cap = FOUNDATION_EVENT_AGE_CAP
        log_cap = np.log1p(float(cap))
        result["swing_high_break_event"] = high_break.astype(np.float32)
        result["swing_low_break_event"] = low_break.astype(np.float32)
        result["swing_break_displacement_atr"] = displacement.astype(np.float32)
        # G2: _bars_since_event convention (entry_foundation_structure_v1),
        # computed in the same owner as the events so they cannot diverge.
        result["bars_since_swing_high_break"] = _bars_since_event_capped(
            high_break > 0.0, cap=cap
        ).astype(np.float32)
        result["bars_since_swing_low_break"] = _bars_since_event_capped(
            low_break > 0.0, cap=cap
        ).astype(np.float32)
        # G4 deltas: per-bar ATR denominator, the dist_last_swing_*_atr
        # convention of this file; NaN until a second pivot per side exists.
        result["swing_high_sequence_delta_atr"] = np.where(
            np.isfinite(prev_high_vals),
            (last_high_vals - prev_high_vals) / atr_safe,
            np.nan,
        ).astype(np.float32)
        result["swing_low_sequence_delta_atr"] = np.where(
            np.isfinite(prev_low_vals),
            (last_low_vals - prev_low_vals) / atr_safe,
            np.nan,
        ).astype(np.float32)
        # G4 run counts: capped at FOUNDATION_EVENT_AGE_CAP, normalized
        # log1p(x)/log1p(cap) (the htf trend_age_bars_norm convention). A raw
        # zero is real arithmetic: zero observed consecutive pairs so far.
        result["consecutive_higher_lows_count"] = (
            np.log1p(np.minimum(hl_run_vals, float(cap))) / log_cap
        ).astype(np.float32)
        result["consecutive_lower_highs_count"] = (
            np.log1p(np.minimum(lh_run_vals, float(cap))) / log_cap
        ).astype(np.float32)
        # V30 package 8A: the two MISSING run counters, byte-for-byte the same
        # arithmetic/cap/normalization as the two above (rule 2b: no new
        # magnitude — the cap and the log1p convention are the imported ones).
        result["consecutive_higher_highs_count"] = (
            np.log1p(np.minimum(hh_run_vals, float(cap))) / log_cap
        ).astype(np.float32)
        result["consecutive_lower_lows_count"] = (
            np.log1p(np.minimum(ll_run_vals, float(cap))) / log_cap
        ).astype(np.float32)
        result["swing_high_level_intact"] = high_intact_vals.astype(np.float32)
        result["swing_low_level_intact"] = low_intact_vals.astype(np.float32)
        # V30 package 8A — ONE age convention. The two raw V1 ages above are
        # unbounded bar counts; the system convention for an event age is
        # htf_features._event_age_norm = log1p(min(age, 500))/log1p(500) (the
        # trend_age_bars_norm convention, also used by regime_v4_features).
        # The helper is IMPORTED from its owner, never re-derived (rule 13).
        # The import is function-local because htf_features imports this
        # module at module scope (SWING_V29_ADDITION_NAMES_V1); a module-level
        # import here would be circular. The raw fields are untouched.
        from gx1.features.htf_features import _event_age_norm

        result["bars_since_swing_high_norm"] = _event_age_norm(
            result["bars_since_swing_high"].astype(np.float64)
        ).astype(np.float32)
        result["bars_since_swing_low_norm"] = _event_age_norm(
            result["bars_since_swing_low"].astype(np.float64)
        ).astype(np.float32)
        expected_names = SWING_FEATURE_NAMES_V1 + SWING_V29_ADDITION_NAMES_V1
    if tuple(result) != expected_names or any(
        values.shape != (n,) for values in result.values()
    ):
        raise RuntimeError("SWING_STRUCTURE_OUTPUT_INVALID")
    for name, values in result.items():
        finite = np.isfinite(values)
        if name in _SWING_V29_NAN_PREFIX_NAMES:
            # NaN legal ONLY as one leading warmup prefix (rule 2e honest
            # warmup + the shared HTF matrix owner's single-prefix contract).
            if finite.any() and not finite[int(np.argmax(finite)):].all():
                raise RuntimeError("SWING_STRUCTURE_OUTPUT_INVALID")
        elif not finite.all():
            raise RuntimeError("SWING_STRUCTURE_OUTPUT_INVALID")
    return result
