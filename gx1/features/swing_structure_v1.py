"""One causal swing-structure owner for every local and MTF clock.

LOOKAHEAD-SAFE: a swing pivot at bar j (high[j] strictly exceeds its `lookback`
neighbours on BOTH sides) is only REFLECTED into the features from bar j+lookback
— never AT bar j — so the value at bar i uses only pivots confirmed by bar i. The
decision bar is therefore causal, and train == serve bit-for-bit.

WHY THIS FILE (rule 7): until 2026-06-24 this math lived in TWO copies — the live
augmenter (v12_ctx_augment_live._add_swing_features) and the V10 training-dataset
builder (build_entry_v10_ctx_training_dataset_v3) — and one of them reflected the
pivot AT bar j (a 2-bar look-ahead). Both now delegate here so the computation can
only ever exist once. smc_v1.py was considered but it owns the smc_* family with a
different lookback (3); these are the lookback-2 structure features.  No bar is
used as a pseudo-pivot, no ATR warmup is filled, and no age/run is capped or
log-normalized.  Unavailable numeric values are stored only together with an
explicit ``present``/``seen`` mask.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from gx1.features.technical_indicators_v1 import wilder_atr
from gx1.features.event_age_v1 import raw_event_age_bars

SWING_FEATURE_NAMES_V1 = (
    "dist_last_swing_high_atr",
    "dist_last_swing_low_atr",
    "bars_since_swing_high",
    "bars_since_swing_low",
    "retracement_from_last_impulse",
    "swing_impulse_present",
)
SWING_LOOKBACK_V1 = 2
SWING_ATR_PERIOD_V1 = 14
SWING_STRUCTURE_FEATURE_VERSION = (
    "swing_structure_v3_raw_uncapped_honest_event_prefix_20260814"
)

# Stateful structure additions.
# The broken level is THIS file's own last CONFIRMED lookback-2 entry swing
# (last_high/last_low in the fill loop below) — the local pivot truth. It is a
# DIFFERENT level definition from smc_v1's lookback-3 swing that smc_bos_*
# breaks (structure report §5: no ownership collision); nothing is imported
# from smc_v1. A break fires once per confirmed level. High/low displacement,
# age and availability are separate, so a double-sided bar loses neither side.
# Pivot sequence deltas and four strict run counters retain raw units.
SWING_V29_ADDITION_NAMES_V1 = (
    "swing_high_break_event",
    "swing_low_break_event",
    "swing_high_break_displacement_atr",
    "swing_low_break_displacement_atr",
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
)
if len(set(SWING_V29_ADDITION_NAMES_V1)) != len(SWING_V29_ADDITION_NAMES_V1) or (
    set(SWING_V29_ADDITION_NAMES_V1) & set(SWING_FEATURE_NAMES_V1)
):
    raise RuntimeError(
        "SWING_STRUCTURE_V29_NAMES_INVALID: addition names must be unique and "
        "disjoint from the bound V1 surface"
    )

SWING_STRUCTURE_FORMULA_CONTRACT = (
    "pivot=strict_symmetric_lookback2_confirmed_after_two_closed_bars",
    "initial_state=no_pseudo_pivot_causal_nan_prefix",
    "atr=shared_classic_wilder14_positive_only_no_epsilon",
    "distance_and_sequence_delta=raw_atr_units_no_clip_or_floor",
    "retracement=raw_position_over_absolute_distinct_row_last_pivot_impulse_no_clip",
    "break=one_shot_per_confirmed_level_sided_raw_displacement_with_atr_nan_prefix",
    "age_and_run=raw_uncapped_observed_bar_or_pivot_counts",
    "event_age_unobserved_storage=nan_until_first_genuine_break",
    "impulse_unavailable_storage=zero_with_explicit_present_mask",
)
SWING_STRUCTURE_FORMULA_SHA256 = hashlib.sha256(
    "\n".join(SWING_STRUCTURE_FORMULA_CONTRACT).encode("utf-8")
).hexdigest()
SWING_STRUCTURE_FEATURE_NAMES_SHA256 = hashlib.sha256(
    "\n".join((*SWING_FEATURE_NAMES_V1, *SWING_V29_ADDITION_NAMES_V1)).encode(
        "utf-8"
    )
).hexdigest()


def swing_structure_contract_metadata() -> dict[str, object]:
    names = (*SWING_FEATURE_NAMES_V1, *SWING_V29_ADDITION_NAMES_V1)
    return {
        "owner": "gx1.features.swing_structure_v1",
        "feature_version": SWING_STRUCTURE_FEATURE_VERSION,
        "formula_sha256": SWING_STRUCTURE_FORMULA_SHA256,
        "formula_contract": list(SWING_STRUCTURE_FORMULA_CONTRACT),
        "lookback": SWING_LOOKBACK_V1,
        "atr_period": SWING_ATR_PERIOD_V1,
        "feature_count": len(names),
        "ordered_feature_names_sha256": SWING_STRUCTURE_FEATURE_NAMES_SHA256,
        "ordered_feature_names": list(names),
    }


def compute_swing_structure_features(
    high,
    low,
    close,
    *,
    lookback: int = SWING_LOOKBACK_V1,
    atr_period: int = SWING_ATR_PERIOD_V1,
    include_v29_additions: bool = False,
) -> dict[str, np.ndarray]:
    """Return the exact ordered float32 swing-structure surface.

    high/low/close: equal-length, chronologically-ordered 1-D array-likes (np or pd).
    `lookback` is BOTH the pivot half-window and the confirmation lag (a pivot at j is
    reflected from bar j+lookback). The first/last `lookback` bars can never be pivots.
    ``include_v29_additions`` appends break events, sided displacement,
    uncapped ages, sequence deltas, exact run counts and their masks.  Every
    canonical current-contract caller passes ``True`` explicitly.
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
    n = len(c)

    # One shared ATR owner: classic Wilder SMA seed, recursive RMA, and a
    # genuinely unavailable denominator when ATR is zero.  No partial-window
    # ATR and no epsilon substitution are permitted.
    index = pd.RangeIndex(n)
    atr = wilder_atr(
        pd.Series(h, index=index, dtype=np.float64),
        pd.Series(low_values, index=index, dtype=np.float64),
        pd.Series(c, index=index, dtype=np.float64),
        atr_period,
    ).to_numpy(dtype=np.float64)
    atr_positive = np.where(atr > 0.0, atr, np.nan)

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
    # No bar-zero pseudo-pivot.  A level exists only after its causal
    # confirmation bar; explicit presence masks disambiguate every stored zero
    # before that point.
    last_high = np.nan
    last_low = np.nan
    last_hi_i = -1
    last_lo_i = -1
    # V29 state (G1/G4).
    armed_high = False
    armed_low = False
    high_adoptions = 0
    low_adoptions = 0
    prev_high_val = np.nan  # previous confirmed swing-high price (one prior pivot, G4)
    prev_low_val = np.nan
    lh_run = 0  # consecutive lower highs (updates on high-pivot adoption)
    hl_run = 0  # consecutive higher lows (updates on low-pivot adoption)
    # V30 package 8A: the two MISSING members of the run-counter set, same
    # arithmetic, opposite strict comparison.
    hh_run = 0  # consecutive higher highs (updates on high-pivot adoption)
    ll_run = 0  # consecutive lower lows (updates on low-pivot adoption)
    high_break = np.zeros(n, dtype=np.float64)
    low_break = np.zeros(n, dtype=np.float64)
    # Event-gated displacement is unavailable until the shared Wilder ATR
    # denominator exists.  A pre-seed break must never be parked at zero,
    # because zero means "no break on this ATR-observable bar" after warmup.
    atr_observable = np.isfinite(atr_positive)
    high_displacement = np.where(atr_observable, 0.0, np.nan)
    low_displacement = np.where(atr_observable, 0.0, np.nan)
    prev_high_vals = np.full(n, np.nan, dtype=np.float64)
    prev_low_vals = np.full(n, np.nan, dtype=np.float64)
    lh_run_vals = np.zeros(n, dtype=np.float64)
    hl_run_vals = np.zeros(n, dtype=np.float64)
    hh_run_vals = np.zeros(n, dtype=np.float64)
    ll_run_vals = np.zeros(n, dtype=np.float64)
    # Prefix-unavailable level state stays NaN and is removed by the shared
    # causal-history trim; it is never parked on a numeric sentinel.
    high_present_vals = np.zeros(n, dtype=np.float64)
    low_present_vals = np.zeros(n, dtype=np.float64)
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
        if fired_high and np.isfinite(atr_positive[i]):
            high_displacement[i] = (c[i] - last_high) / atr_positive[i]
        if fired_low and np.isfinite(atr_positive[i]):
            low_displacement[i] = (last_low - c[i]) / atr_positive[i]
        # V30 package 8A: emit the post-update G1 arming state — "the last
        # confirmed swing high/low has not been closed through" = structure
        # intact. It is read AFTER the break check on this bar, so the bar
        # whose close breaks the level already reads 0 (that close IS the
        # break). Before the first adoption on that side no level exists, so
        # the value stays NaN (one leading prefix, rule 2e).
        if high_adoptions > 0:
            high_present_vals[i] = 1.0
            high_intact_vals[i] = 1.0 if armed_high else 0.0
        if low_adoptions > 0:
            low_present_vals[i] = 1.0
            low_intact_vals[i] = 1.0 if armed_low else 0.0

    row_index = np.arange(n, dtype=np.int64)
    high_present = high_present_vals > 0.0
    low_present = low_present_vals > 0.0
    impulse_width = np.abs(last_high_vals - last_low_vals)
    impulse_present = (
        high_present
        & low_present
        & np.isfinite(impulse_width)
        & (impulse_width > 0.0)
        & (last_high_idx != last_low_idx)
    )
    retracement = np.zeros(n, dtype=np.float64)
    up_mask = impulse_present & (last_high_idx > last_low_idx)
    down_mask = impulse_present & (last_low_idx > last_high_idx)
    retracement[up_mask] = (
        last_high_vals[up_mask] - c[up_mask]
    ) / impulse_width[up_mask]
    retracement[down_mask] = (
        c[down_mask] - last_low_vals[down_mask]
    ) / impulse_width[
        down_mask
    ]

    high_distance = np.full(n, np.nan, dtype=np.float64)
    low_distance = np.full(n, np.nan, dtype=np.float64)
    high_age = np.full(n, np.nan, dtype=np.float64)
    low_age = np.full(n, np.nan, dtype=np.float64)
    high_distance_defined = high_present & np.isfinite(atr_positive)
    low_distance_defined = low_present & np.isfinite(atr_positive)
    high_distance[high_distance_defined] = (
        c[high_distance_defined] - last_high_vals[high_distance_defined]
    ) / atr_positive[high_distance_defined]
    low_distance[low_distance_defined] = (
        c[low_distance_defined] - last_low_vals[low_distance_defined]
    ) / atr_positive[low_distance_defined]
    high_age[high_present] = (
        row_index[high_present] - last_high_idx[high_present]
    ).astype(np.float64)
    low_age[low_present] = (
        row_index[low_present] - last_low_idx[low_present]
    ).astype(np.float64)

    result = {
        "dist_last_swing_high_atr": high_distance.astype(np.float32),
        "dist_last_swing_low_atr": low_distance.astype(np.float32),
        "bars_since_swing_high": high_age.astype(np.float32),
        "bars_since_swing_low": low_age.astype(np.float32),
        "retracement_from_last_impulse": retracement.astype(np.float32),
        "swing_impulse_present": impulse_present.astype(np.float32),
    }
    expected_names = SWING_FEATURE_NAMES_V1
    if include_v29_additions:
        result["swing_high_break_event"] = high_break.astype(np.float32)
        result["swing_low_break_event"] = low_break.astype(np.float32)
        result["swing_high_break_displacement_atr"] = high_displacement.astype(
            np.float32
        )
        result["swing_low_break_displacement_atr"] = low_displacement.astype(
            np.float32
        )
        high_break_age = raw_event_age_bars((high_break > 0.0).astype(np.bool_))
        low_break_age = raw_event_age_bars((low_break > 0.0).astype(np.bool_))
        result["bars_since_swing_high_break"] = high_break_age.astype(np.float32)
        result["bars_since_swing_low_break"] = low_break_age.astype(np.float32)

        high_delta_present = np.isfinite(prev_high_vals) & np.isfinite(atr_positive)
        low_delta_present = np.isfinite(prev_low_vals) & np.isfinite(atr_positive)
        high_delta = np.full(n, np.nan, dtype=np.float64)
        low_delta = np.full(n, np.nan, dtype=np.float64)
        high_delta[high_delta_present] = (
            last_high_vals[high_delta_present] - prev_high_vals[high_delta_present]
        ) / atr_positive[high_delta_present]
        low_delta[low_delta_present] = (
            last_low_vals[low_delta_present] - prev_low_vals[low_delta_present]
        ) / atr_positive[low_delta_present]
        result["swing_high_sequence_delta_atr"] = high_delta.astype(np.float32)
        result["swing_low_sequence_delta_atr"] = low_delta.astype(np.float32)

        # Exact uncapped integer run lengths.  The names say ``count`` and the
        # bytes now carry counts; no static logarithm or 96-pivot saturation.
        result["consecutive_higher_lows_count"] = hl_run_vals.astype(np.float32)
        result["consecutive_lower_highs_count"] = lh_run_vals.astype(np.float32)
        result["consecutive_higher_highs_count"] = hh_run_vals.astype(np.float32)
        result["consecutive_lower_lows_count"] = ll_run_vals.astype(np.float32)
        result["swing_high_level_intact"] = high_intact_vals.astype(np.float32)
        result["swing_low_level_intact"] = low_intact_vals.astype(np.float32)
        expected_names = SWING_FEATURE_NAMES_V1 + SWING_V29_ADDITION_NAMES_V1
    if tuple(result) != expected_names or any(
        values.shape != (n,) for values in result.values()
    ):
        raise RuntimeError("SWING_STRUCTURE_OUTPUT_INVALID")
    nan_prefix_names = {
        "dist_last_swing_high_atr",
        "dist_last_swing_low_atr",
        "bars_since_swing_high",
        "bars_since_swing_low",
        "swing_high_break_displacement_atr",
        "swing_low_break_displacement_atr",
        "bars_since_swing_high_break",
        "bars_since_swing_low_break",
        "swing_high_sequence_delta_atr",
        "swing_low_sequence_delta_atr",
        "swing_high_level_intact",
        "swing_low_level_intact",
    }
    for name, values in result.items():
        finite = np.isfinite(values)
        if name in nan_prefix_names:
            if finite.any() and not finite[int(np.argmax(finite)):].all():
                raise RuntimeError("SWING_STRUCTURE_OUTPUT_INVALID")
        elif not finite.all():
            raise RuntimeError("SWING_STRUCTURE_OUTPUT_INVALID")
    return result
