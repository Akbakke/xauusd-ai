"""
Smart Money Concept (SMC) features — V1 implementation.

For each M5 bar, compute 9 SMC features that describe market structure,
liquidity events, and premium/discount position. Designed to feed the
canonical_v3 feature parquet and downstream model-native consumers.

Features (all per-bar, lookahead-safe):
  smc_swing_state        int8     0=HH+HL (clean up), 1=HH+LL (two-sided expansion), 2=LH+HL (contraction/inside), 3=LH+LL (clean down), 4=exact ties or warmup
  smc_bos_up             float32  1.0 only on the bar where close first crosses above the last confirmed swing high
  smc_bos_down           float32  1.0 only on the bar where close first crosses below the last confirmed swing low
  smc_choch              float32  1.0 only on the bar where the non-zero structure sign flips up↔down
  smc_sweep_up           float32  1.0 if high > last swing high but close <= it (false breakout / liquidity hunt)
  smc_sweep_down         float32  1.0 if low  < last swing low  but close >= it
  smc_sweep_size_atr     float32  magnitude of the wick beyond the swept level, ATR-normalized
  smc_bars_since_sweep   float32  bars elapsed since most recent sweep (clipped 999)
  smc_premium_discount   float32  close position inside the causal 4-pivot envelope
                                  [min(last_sl, prev_sl), max(last_sh, prev_sh)], in [0, 1]

Lookahead safety: a swing pivot at bar j is only considered "confirmed" once
j + SWING_LOOKBACK bars have elapsed. So features at bar i only use swings
confirmed up to bar (i - SWING_LOOKBACK), no future leakage.

2026-08-13 (V30 package 8A): the M5 owner gained six OPTIONAL owner-parity
emissions behind ``include_v30_additions`` (``SMC_V30_ADDITION_NAMES_V1``) and
the MTF owner gained three mandatory ones (signed BOS displacement and the two
de-duplicated sweep events).  Both are emission-only: every value is a quantity
the owner already computed and discarded, or the sided/de-duplicated form its
sibling already emitted.  No accepted field changed.

2026-08-09 backport: four defects in this M5 owner were repaired from the
mechanisms already proven in :func:`compute_smc_mtf_primitives_v1` below —
CHOCH now compares the last observed non-zero structure sign instead of
adjacent rows, BOS is a crossing event instead of a persistent state, the
premium/discount range is the 4-pivot envelope instead of the last-high/
last-low pair, and the swing-state predicates for states 1/2 (previously
self-contradictory for distinct prices) form a true partition of the generic
two-sided comparisons.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


SWING_LOOKBACK = 3  # bars look-around for swing pivot detection (3 → 7-bar window centered)

def _detect_swing_pivots(high: np.ndarray, low: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (swing_high_mask, swing_low_mask) — bool arrays, True at pivot bars.

    Swing high at bar i if high[i] is the maximum over [i-n, i+n].
    Swing low at bar i if low[i] is the minimum over [i-n, i+n].
    Edges (first n + last n bars) cannot be pivots.
    """
    nb = len(high)
    sh = np.zeros(nb, dtype=bool)
    sl = np.zeros(nb, dtype=bool)
    for i in range(n, nb - n):
        wh = high[i - n : i + n + 1]
        wl = low[i - n : i + n + 1]
        if high[i] >= wh.max() - 1e-12:
            sh[i] = True
        if low[i] <= wl.min() + 1e-12:
            sl[i] = True
    return sh, sl


def _track_recent_swings(
    swing_high_mask: np.ndarray,
    swing_low_mask: np.ndarray,
    n_lookback: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """For each bar i, return (last_sh_idx, prev_sh_idx, last_sl_idx, prev_sl_idx).

    A swing at bar j is "confirmed" by bar j + n_lookback. So at bar i, the
    most recent confirmed swing is at most (i - n_lookback). Returns -1 if no
    confirmed swing exists yet.
    """
    nb = len(swing_high_mask)
    last_sh = np.full(nb, -1, dtype=np.int64)
    prev_sh = np.full(nb, -1, dtype=np.int64)
    last_sl = np.full(nb, -1, dtype=np.int64)
    prev_sl = np.full(nb, -1, dtype=np.int64)

    cur_last_sh, cur_prev_sh = -1, -1
    cur_last_sl, cur_prev_sl = -1, -1
    for i in range(nb):
        confirm_idx = i - n_lookback
        if confirm_idx >= 0:
            if swing_high_mask[confirm_idx]:
                cur_prev_sh = cur_last_sh
                cur_last_sh = confirm_idx
            if swing_low_mask[confirm_idx]:
                cur_prev_sl = cur_last_sl
                cur_last_sl = confirm_idx
        last_sh[i] = cur_last_sh
        prev_sh[i] = cur_prev_sh
        last_sl[i] = cur_last_sl
        prev_sl[i] = cur_prev_sl
    return last_sh, prev_sh, last_sl, prev_sl


def compute_smc_features(
    df: pd.DataFrame,
    *,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    atr_col: str = "atr",
    swing_lookback: int = SWING_LOOKBACK,
    include_v30_additions: bool = False,
) -> pd.DataFrame:
    """Compute SMC features. Returns the 9 ``SMC_FEATURE_NAMES`` columns.

    Required columns on df: high, low, close and atr. All are exact observed or
    causally computed inputs; no ATR sentinel is permitted.

    ``include_v30_additions`` appends ``SMC_V30_ADDITION_NAMES_V1`` — the six
    2026-08-13 owner-parity emissions (see that tuple's comment). Default False
    == the accepted canonical surface, byte-identical; the flag is a call-site
    CONTRACT switch (the ``swing_structure_v1.include_v29_additions``
    precedent), never an environment gate. No call site flips it yet: the
    canonical M5 frame is bound to per-artifact column-count contracts
    (``audit_seq513_source_cascade_v1``, ``materialize_cv3_modelrange_v1``)
    and a raw canonical column has no route into the model surface anyway —
    ``materialize_entry_model_native_train_feature_ranker_v1._candidate_universe``
    scans only specialist-layer owners, and
    ``build_entry_v10_ctx_training_dataset_v3._build_inline_seq_structure_extension``
    exposes only the frozen base block plus specialist-layer outputs. The three
    quantities themselves DO reach the model today on every timeframe through
    :func:`compute_smc_mtf_primitives_v1`, which carries their per-TF siblings.
    """
    if not isinstance(include_v30_additions, bool):
        raise RuntimeError("[SMC_V30_ADDITION_FLAG_INVALID]")
    nb = len(df)
    if nb == 0:
        raise RuntimeError("[SMC_SOURCE_EMPTY] cannot produce SMC features from zero rows")
    required = (high_col, low_col, close_col, atr_col)
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise RuntimeError(f"[SMC_SOURCE_MISSING] required columns missing: {missing}")
    high = pd.to_numeric(df[high_col], errors="coerce").to_numpy(dtype=np.float64)
    low = pd.to_numeric(df[low_col], errors="coerce").to_numpy(dtype=np.float64)
    close = pd.to_numeric(df[close_col], errors="coerce").to_numpy(dtype=np.float64)
    atr = pd.to_numeric(df[atr_col], errors="coerce").to_numpy(dtype=np.float64)
    invalid_numeric = (
        (~np.isfinite(high))
        | (~np.isfinite(low))
        | (~np.isfinite(close))
        | (~np.isfinite(atr))
    )
    if invalid_numeric.any():
        raise RuntimeError(
            "[SMC_SOURCE_NONFINITE] OHLC/ATR contains unavailable values: "
            f"count={int(np.count_nonzero(invalid_numeric))}"
        )
    invalid_geometry = (
        (high < low)
        | (high < close)
        | (low > close)
        | (atr <= 0.0)
    )
    if invalid_geometry.any():
        raise RuntimeError(
            "[SMC_SOURCE_INVALID] OHLC geometry must be valid and ATR strictly "
            f"positive: count={int(np.count_nonzero(invalid_geometry))}"
        )

    # 1. Detect swing pivots (lookahead-safe via confirmation lag)
    sh_mask, sl_mask = _detect_swing_pivots(high, low, swing_lookback)
    last_sh, prev_sh, last_sl, prev_sl = _track_recent_swings(sh_mask, sl_mask, swing_lookback)

    # Look up swing prices at each bar's most-recent confirmed swing
    last_sh_price = np.where(last_sh >= 0, high[np.clip(last_sh, 0, nb - 1)], np.nan)
    prev_sh_price = np.where(prev_sh >= 0, high[np.clip(prev_sh, 0, nb - 1)], np.nan)
    last_sl_price = np.where(last_sl >= 0, low[np.clip(last_sl, 0, nb - 1)], np.nan)
    prev_sl_price = np.where(prev_sl >= 0, low[np.clip(prev_sl, 0, nb - 1)], np.nan)

    # 2. swing_state: HH/HL/LH/LL pattern at bar i.  The four generic
    # two-sided cases form a true partition (2026-08-09 repair: the retired
    # predicates for states 1/2 were self-contradictory for distinct prices,
    # so those states were reachable only on exact float ties).
    higher_high = last_sh_price > prev_sh_price
    lower_high = last_sh_price < prev_sh_price
    higher_low = last_sl_price > prev_sl_price
    lower_low = last_sl_price < prev_sl_price
    # Default = 4: warmup (a missing pivot compares False via NaN) or an
    # exact price tie on either side.
    swing_state = np.full(nb, 4, dtype=np.int8)
    swing_state[higher_high & higher_low] = 0  # clean up
    swing_state[higher_high & lower_low] = 1   # two-sided expansion
    swing_state[lower_high & higher_low] = 2   # contraction / inside structure
    swing_state[lower_high & lower_low] = 3    # clean down

    # 3. BOS up/down — a causal crossing event, not a persistent "price
    # remains outside the last swing" state (backported 2026-08-09 from
    # compute_smc_mtf_primitives_v1: the persistent form turned one break
    # into many identical event observations, while the downstream
    # _bars_since_event ages and rolling bos-pressure means were designed
    # for events).  Fires only where the break condition holds now and did
    # not hold on the previous bar against the previous bar's level.
    has_sh = last_sh >= 0
    has_sl = last_sl >= 0
    # NaN swing prices (no confirmed pivot) compare False; has_sh/has_sl gate
    # them explicitly as well.
    cond_up = has_sh & (close > last_sh_price)
    cond_down = has_sl & (close < last_sl_price)
    prev_cond_up = np.roll(cond_up, 1)
    prev_cond_down = np.roll(cond_down, 1)
    # Guard the first bar: np.roll wraps the last element around.
    prev_cond_up[0] = False
    prev_cond_down[0] = False
    bos_up = (cond_up & ~prev_cond_up).astype(np.float32)
    bos_down = (cond_down & ~prev_cond_down).astype(np.float32)

    # 3b. BOS displacement — V30 package 8A (2026-08-13), EMISSION ONLY.
    # Signed (close - broken level)/atr AT the firing bar, 0 off-event: the
    # flag-disambiguated-zero convention the sibling break-displacement
    # geometry already uses in compute_smc_mtf_primitives_v1 below
    # ((close - last_high)/atr, (last_low - close)/atr), reused rather than
    # re-derived. The sign is not a second direction rule: a BOS up closes
    # ABOVE its level (positive) and a BOS down BELOW its level (negative) by
    # construction of cond_up/cond_down. When both fire on one bar (possible
    # when the armed low level sits above the armed high level) the more
    # recently CONFIRMED level is carried, tie to the high side — the exact
    # documented rule swing_structure_v1's G1 displacement already uses.
    # The tie-break is taken on the EVENTS, not on the raw conditions: a bar
    # where the up-event fires while the down CONDITION merely persists is a
    # one-sided event and must not be routed to the low level (which would
    # leave a firing flag with a 0 displacement).
    fired_up = bos_up > 0.0
    fired_down = bos_down > 0.0
    fire_high = fired_up & (~fired_down | (last_sh >= last_sl))
    fire_low = fired_down & ~fire_high
    bos_displacement_atr = np.zeros(nb, dtype=np.float64)
    bos_displacement_atr[fire_high] = (
        close[fire_high] - last_sh_price[fire_high]
    ) / atr[fire_high]
    bos_displacement_atr[fire_low] = (
        close[fire_low] - last_sl_price[fire_low]
    ) / atr[fire_low]
    bos_displacement_atr = bos_displacement_atr.astype(np.float32)

    # 4. CHOCH — structure flip.  A high and a low pivot normally confirm on
    # different bars, so every up↔down transition passes through a mixed
    # state (1/2/4); comparing only adjacent rows therefore made CHOCH
    # structurally near-dead.  Backported 2026-08-09 from
    # compute_smc_mtf_primitives_v1: compare the current non-zero structure
    # sign (+1 = state 0 clean up, -1 = state 3 clean down) with the last
    # observed non-zero sign, maintained causally.
    choch = np.zeros(nb, dtype=np.float32)
    struct_sign = np.zeros(nb, dtype=np.float64)
    struct_sign[swing_state == 0] = 1.0
    struct_sign[swing_state == 3] = -1.0
    prior_nonzero_sign = 0.0
    for i in range(nb):
        current_sign = struct_sign[i]
        if current_sign == 0.0:
            continue
        if prior_nonzero_sign != 0.0 and current_sign != prior_nonzero_sign:
            choch[i] = 1.0
        prior_nonzero_sign = current_sign

    # 5. Liquidity sweep — wick beyond swept level, close back inside
    sweep_up = np.zeros(nb, dtype=np.float32)
    sweep_down = np.zeros(nb, dtype=np.float32)
    sweep_size_atr = np.zeros(nb, dtype=np.float32)
    # V30 package 8A (2026-08-13), EMISSION ONLY: the two SIDED depths the
    # combined field already computes and then collapses with max().  A bar
    # that sweeps BOTH sides emits one magnitude today and drops which side it
    # belongs to; the MTF sibling below has emitted the sided pair since its
    # first version (mtf_smc_sweep_up_depth_atr / _down_depth_atr) — this is
    # the same construction, not a new one.  ``smc_sweep_size_atr`` itself is
    # untouched and stays byte-identical (build on, never remove).
    sweep_up_depth_atr = np.zeros(nb, dtype=np.float32)
    sweep_down_depth_atr = np.zeros(nb, dtype=np.float32)
    last_sweep_at = -1
    bars_since_sweep = np.full(nb, 999, dtype=np.float32)
    for i in range(nb):
        a = atr[i]
        any_sweep = False
        if has_sh[i]:
            sh_price = last_sh_price[i]
            if high[i] > sh_price and close[i] <= sh_price:
                sweep_up[i] = 1.0
                sweep_size_atr[i] = float((high[i] - sh_price) / a)
                sweep_up_depth_atr[i] = float((high[i] - sh_price) / a)
                any_sweep = True
        if has_sl[i]:
            sl_price = last_sl_price[i]
            if low[i] < sl_price and close[i] >= sl_price:
                sweep_down[i] = 1.0
                sd = float((sl_price - low[i]) / a)
                sweep_down_depth_atr[i] = sd
                if sd > sweep_size_atr[i]:
                    sweep_size_atr[i] = sd
                any_sweep = True
        if any_sweep:
            last_sweep_at = i
        bars_since_sweep[i] = float(i - last_sweep_at) if last_sweep_at >= 0 else 999.0
    bars_since_sweep = np.clip(bars_since_sweep, 0, 999).astype(np.float32)

    # 5b. Sweep EVENT de-duplication — V30 package 8A, EMISSION ONLY.  The
    # flags above are per-bar CONDITIONS: the same unchanged swing level poked
    # on five consecutive bars raises the flag five times and resets
    # bars_since_sweep each time, exactly the defect BOS was repaired for on
    # 2026-08-09.  The repair idiom is reused verbatim from the BOS block
    # above (`cond & ~prev_cond` with the first bar guarded against np.roll's
    # wrap), and — like BOS — it fires only on the first bar of a run, so a
    # level that CHANGES while the condition stays true does not refire.  The
    # de-duplicated events are emitted as NEW fields; the repeating flags stay
    # untouched so no existing consumer changes silently.
    cond_sweep_up = sweep_up > 0.0
    cond_sweep_down = sweep_down > 0.0
    prev_cond_sweep_up = np.roll(cond_sweep_up, 1)
    prev_cond_sweep_down = np.roll(cond_sweep_down, 1)
    prev_cond_sweep_up[0] = False
    prev_cond_sweep_down[0] = False
    sweep_up_event = (cond_sweep_up & ~prev_cond_sweep_up).astype(np.float32)
    sweep_down_event = (cond_sweep_down & ~prev_cond_sweep_down).astype(np.float32)

    # 6. Premium/discount score — close position inside the causal 4-pivot
    # envelope (backported 2026-08-09 from compute_smc_mtf_primitives_v1:
    # requiring last_sh_price > last_sl_price fabricated 0.5 through normal
    # trend geometry, e.g. a new swing low confirmed above the previous
    # swing high).  Valid once both sides have BOTH a current and a previous
    # confirmed pivot and the envelope has positive width.
    # Deviation from the MTF variant: this M5 owner's contract is finite
    # per-row (no mid-series NaN), so the warmup prefix before both-side
    # pivots exist — and the degenerate zero-width envelope of equal-price
    # pivots — keep the pre-existing 0.5 midpoint instead of emitting NaN.
    env_high = np.maximum(last_sh_price, prev_sh_price)
    env_low = np.minimum(last_sl_price, prev_sl_price)
    valid_pd = (
        (last_sh >= 0)
        & (prev_sh >= 0)
        & (last_sl >= 0)
        & (prev_sl >= 0)
        & (env_high > env_low)
    )
    pd_score = np.full(nb, 0.5, dtype=np.float32)
    rng = np.where(valid_pd, env_high - env_low, 1.0)
    pd_score[valid_pd] = ((close[valid_pd] - env_low[valid_pd]) / rng[valid_pd]).astype(np.float32)
    pd_score = np.clip(pd_score, 0.0, 1.0)

    out_cols = {
        "smc_swing_state": swing_state,
        "smc_bos_up": bos_up,
        "smc_bos_down": bos_down,
        "smc_choch": choch,
        "smc_sweep_up": sweep_up,
        "smc_sweep_down": sweep_down,
        "smc_sweep_size_atr": sweep_size_atr,
        "smc_bars_since_sweep": bars_since_sweep,
        "smc_premium_discount": pd_score,
    }
    if include_v30_additions:
        # ONE age convention: log1p(min(age, 500))/log1p(500), the
        # trend_age_bars_norm convention owned by htf_features._event_age_norm
        # (imported, never re-derived — rule 13).  The import is function-local
        # because htf_features imports this module at module scope
        # (SMC_MTF_FEATURE_NAMES_V1); a module-level import would be circular.
        # The 999 "no sweep yet" sentinel maps to the cap, i.e. 1.0 = maximally
        # stale, which is exactly what it means.  The raw field is untouched.
        from gx1.features.htf_features import _event_age_norm

        out_cols["smc_bos_displacement_atr"] = bos_displacement_atr
        out_cols["smc_sweep_up_depth_atr"] = sweep_up_depth_atr
        out_cols["smc_sweep_down_depth_atr"] = sweep_down_depth_atr
        out_cols["smc_sweep_up_event"] = sweep_up_event
        out_cols["smc_sweep_down_event"] = sweep_down_event
        out_cols["smc_bars_since_sweep_norm"] = _event_age_norm(
            bars_since_sweep.astype(np.float64)
        ).astype(np.float32)
    expected_columns = tuple(SMC_FEATURE_NAMES) + (
        SMC_V30_ADDITION_NAMES_V1 if include_v30_additions else ()
    )
    if tuple(out_cols) != expected_columns:
        raise RuntimeError("[SMC_OUTPUT_ORDER_INVALID]")
    out = pd.DataFrame(out_cols, index=df.index)
    return out


SMC_FEATURE_NAMES = [
    "smc_swing_state",
    "smc_bos_up",
    "smc_bos_down",
    "smc_choch",
    "smc_sweep_up",
    "smc_sweep_down",
    "smc_sweep_size_atr",
    "smc_bars_since_sweep",
    "smc_premium_discount",
]
# V30 package 8A (2026-08-13) — owner-parity emissions for the M5 SMC block,
# declared SEPARATELY from SMC_FEATURE_NAMES because that 9-name list is the
# accepted canonical-M5 column contract and several artifact column-count
# audits are bound to it.  Each name is a quantity this owner already computes
# and discards, or the sided/de-duplicated form its own MTF sibling already
# emits (docs/INDICATOR_FIDELITY_AUDIT_20260813.md §4b: "a fix landed in one
# owner and not its sibling"):
#   smc_bos_displacement_atr   signed (close - broken level)/atr at the firing
#                              bar, 0 off-event (flag-disambiguated zero)
#   smc_sweep_up/down_depth_atr  the sided depths behind the max()-collapsed
#                              smc_sweep_size_atr
#   smc_sweep_up/down_event    the BOS-style de-duplicated first-bar events
#   smc_bars_since_sweep_norm  the one age convention applied to the raw 999
#                              sentinel count
SMC_V30_ADDITION_NAMES_V1 = (
    "smc_bos_displacement_atr",
    "smc_sweep_up_depth_atr",
    "smc_sweep_down_depth_atr",
    "smc_sweep_up_event",
    "smc_sweep_down_event",
    "smc_bars_since_sweep_norm",
)
if set(SMC_V30_ADDITION_NAMES_V1) & set(SMC_FEATURE_NAMES) or len(
    set(SMC_V30_ADDITION_NAMES_V1)
) != len(SMC_V30_ADDITION_NAMES_V1):
    raise RuntimeError("[SMC_V30_ADDITION_NAMES_INVALID]")
# Exact fixed-width primitives for the multi-resolution Entry surface.  Unlike
# the historical M5 contract above, this contract is independent of ambient
# environment flags and never emits numeric unknown/sentinel values.  Rows are
# NaN until two highs and two lows have been causally confirmed; the shared HTF
# matrix owner trims that one chronological warmup prefix before a row can
# reach training or serving.
SMC_MTF_FEATURE_NAMES_V1 = (
    "mtf_smc_structure_bias",
    "mtf_smc_bos_up",
    "mtf_smc_bos_down",
    "mtf_smc_choch_up",
    "mtf_smc_choch_down",
    "mtf_smc_sweep_up",
    "mtf_smc_sweep_down",
    "mtf_smc_sweep_up_depth_atr",
    "mtf_smc_sweep_down_depth_atr",
    "mtf_smc_premium_discount",
    "mtf_smc_range_width_atr",
    # V30 package 8A (2026-08-13), EMISSION ONLY — appended so the pre-existing
    # per-TF column order is byte-stable ahead of them:
    #   mtf_smc_bos_displacement_atr  signed (close - broken level)/atr at the
    #     BOS bar, 0 off-event.  Same construction as the
    #     mtf_geometry_*_break_displacement_atr siblings below, event-gated;
    #     the BOS flags already tell 0-the-value from 0-the-non-event.
    #   mtf_smc_sweep_up/down_event  the sweep flags above are per-bar
    #     CONDITIONS, so one unchanged level poked on five consecutive bars
    #     emits five sweeps.  These are the de-duplicated first-bar events,
    #     the exact `cond & ~prev_cond` idiom BOS already uses in this file.
    #     The repeating flags stay untouched (rule 25b sweeps the defect class
    #     across the sibling owner without changing an accepted field).
    "mtf_smc_bos_displacement_atr",
    "mtf_smc_sweep_up_event",
    "mtf_smc_sweep_down_event",
)

SMC_MTF_GEOMETRY_FEATURE_NAMES_V1 = (
    "mtf_geometry_support_dist_atr",
    "mtf_geometry_resistance_dist_atr",
    "mtf_geometry_support_age_bars",
    "mtf_geometry_resistance_age_bars",
    "mtf_geometry_support_rail_slope_atr_per_bar",
    "mtf_geometry_resistance_rail_slope_atr_per_bar",
    "mtf_geometry_support_break_displacement_atr",
    "mtf_geometry_resistance_break_displacement_atr",
    "mtf_geometry_nearest_level_abs_atr",
    "mtf_geometry_range_mid_dist_atr",
)


def compute_smc_mtf_primitives_v1(
    df: pd.DataFrame,
    *,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    atr_col: str = "atr",
    swing_lookback: int = SWING_LOOKBACK,
) -> pd.DataFrame:
    """Return fixed, causal SMC and S/R geometry roots for one resolution.

    The same observed-bar calculation runs independently on M5, M15, H1, H4
    and D1.  It contains no direction decision, cross-timeframe proxy, ambient
    feature flag or resolution-specific weight.
    """
    if (
        isinstance(swing_lookback, bool)
        or not isinstance(swing_lookback, int)
        or swing_lookback < 1
    ):
        raise RuntimeError(
            f"[SMC_MTF_SWING_LOOKBACK_INVALID] {swing_lookback!r}"
        )
    required = (high_col, low_col, close_col, atr_col)
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise RuntimeError(f"[SMC_MTF_SOURCE_MISSING] {missing}")
    if len(df) == 0:
        raise RuntimeError("[SMC_MTF_SOURCE_EMPTY]")

    high = pd.to_numeric(df[high_col], errors="coerce").to_numpy(dtype=np.float64)
    low = pd.to_numeric(df[low_col], errors="coerce").to_numpy(dtype=np.float64)
    close = pd.to_numeric(df[close_col], errors="coerce").to_numpy(dtype=np.float64)
    atr = pd.to_numeric(df[atr_col], errors="coerce").to_numpy(dtype=np.float64)
    if (
        not np.isfinite(high).all()
        or not np.isfinite(low).all()
        or not np.isfinite(close).all()
        or np.isinf(atr).any()
    ):
        raise RuntimeError("[SMC_MTF_SOURCE_NONFINITE]")
    atr_available = np.isfinite(atr)
    if atr_available.any():
        first_atr = int(np.argmax(atr_available))
        if not atr_available[first_atr:].all():
            raise RuntimeError("[SMC_MTF_ATR_AVAILABILITY_INVALID]")
    if (
        np.any(high <= 0.0)
        or np.any(low <= 0.0)
        or np.any(close <= 0.0)
        or np.any(atr[atr_available] <= 0.0)
        or np.any(high < low)
        or np.any(high < close)
        or np.any(low > close)
    ):
        raise RuntimeError("[SMC_MTF_SOURCE_GEOMETRY_INVALID]")

    n_rows = len(df)
    swing_high_mask, swing_low_mask = _detect_swing_pivots(
        high,
        low,
        swing_lookback,
    )
    last_high_idx, prev_high_idx, last_low_idx, prev_low_idx = (
        _track_recent_swings(
            swing_high_mask,
            swing_low_mask,
            swing_lookback,
        )
    )
    clipped_last_high = np.clip(last_high_idx, 0, n_rows - 1)
    clipped_prev_high = np.clip(prev_high_idx, 0, n_rows - 1)
    clipped_last_low = np.clip(last_low_idx, 0, n_rows - 1)
    clipped_prev_low = np.clip(prev_low_idx, 0, n_rows - 1)
    last_high = high[clipped_last_high]
    prev_high = high[clipped_prev_high]
    last_low = low[clipped_last_low]
    prev_low = low[clipped_prev_low]

    # The structural range is the causal envelope of both most-recent
    # confirmed high pivots and both most-recent confirmed low pivots.  Using
    # only the latest high/low made a perfectly valid equal-pivot transition
    # collapse to zero width on real XAU M15 data (2025-08-18).  The previous
    # confirmed pivots are already required below and are known at the same
    # decision time, so the envelope defines the geometry without an epsilon,
    # sentinel, future observation, or dropped interior row.
    pivot_stack = np.vstack((last_high, prev_high, last_low, prev_low))
    range_low = np.min(pivot_stack, axis=0)
    range_high = np.max(pivot_stack, axis=0)
    channel_width = range_high - range_low
    available = (
        (last_high_idx >= 0)
        & (prev_high_idx >= 0)
        & (last_low_idx >= 0)
        & (prev_low_idx >= 0)
        & (channel_width > 0.0)
        & atr_available
    )
    if not available.any():
        return pd.DataFrame(
            np.full(
                (
                    n_rows,
                    len(SMC_MTF_FEATURE_NAMES_V1)
                    + len(SMC_MTF_GEOMETRY_FEATURE_NAMES_V1),
                ),
                np.nan,
                dtype=np.float32,
            ),
            index=df.index,
            columns=(
                SMC_MTF_FEATURE_NAMES_V1
                + SMC_MTF_GEOMETRY_FEATURE_NAMES_V1
            ),
        )

    # The mean of the independently observed high- and low-structure signs:
    # +1=HH+HL, -1=LH+LL, and 0=mixed.  This is evidence, not a direction rule.
    high_sign = np.sign(last_high - prev_high)
    low_sign = np.sign(last_low - prev_low)
    structure_bias = (high_sign + low_sign) / 2.0

    # BOS is a causal crossing event, not a persistent "price remains outside
    # the last swing" state.  The latter is already represented continuously
    # by the support/resistance break-displacement geometry fields; keeping it
    # here as well would duplicate evidence and turn one break into many
    # identical event observations.
    previous_close = np.roll(close, 1)
    previous_last_high = np.roll(last_high, 1)
    previous_last_low = np.roll(last_low, 1)
    previous_available = np.roll(available, 1)
    previous_available[0] = False
    bos_up = (
        available
        & (
            (~previous_available)
            | (previous_close <= previous_last_high)
            | (last_high != previous_last_high)
        )
        & (close > last_high)
    ).astype(np.float64)
    bos_down = (
        available
        & (
            (~previous_available)
            | (previous_close >= previous_last_low)
            | (last_low != previous_last_low)
        )
        & (close < last_low)
    ).astype(np.float64)
    choch_up = np.zeros(n_rows, dtype=np.float64)
    choch_down = np.zeros(n_rows, dtype=np.float64)
    # A high and a low pivot normally confirm on different bars, so structure
    # transitions pass through a mixed/zero state. Compare with the last
    # observed non-zero structure sign; comparing only adjacent rows made both
    # CHOCH fields effectively dead.
    prior_nonzero_sign = 0.0
    for row in range(n_rows):
        if not available[row]:
            continue
        current_sign = float(np.sign(structure_bias[row]))
        if current_sign == 0.0:
            continue
        if prior_nonzero_sign < 0.0 and current_sign > 0.0:
            choch_up[row] = 1.0
        elif prior_nonzero_sign > 0.0 and current_sign < 0.0:
            choch_down[row] = 1.0
        prior_nonzero_sign = current_sign

    # V30 package 8A: signed BOS displacement at the firing bar.  Both sides
    # can fire on one bar (the causal envelope allows last_low > last_high);
    # the more recently CONFIRMED level is carried, tie to the high side — the
    # documented rule swing_structure_v1's G1 displacement already uses.
    use_high_level = (bos_up > 0.0) & (
        (bos_down <= 0.0) | (last_high_idx >= last_low_idx)
    )
    use_low_level = (bos_down > 0.0) & ~use_high_level
    bos_displacement = np.zeros(n_rows, dtype=np.float64)
    np.divide(close - last_high, atr, out=bos_displacement, where=use_high_level)
    np.divide(close - last_low, atr, out=bos_displacement, where=use_low_level)

    sweep_up = (high > last_high) & (close <= last_high)
    sweep_down = (low < last_low) & (close >= last_low)
    sweep_up_depth = np.where(sweep_up, (high - last_high) / atr, 0.0)
    sweep_down_depth = np.where(sweep_down, (last_low - low) / atr, 0.0)
    # V30 package 8A: de-duplicated first-bar sweep events (the BOS
    # `cond & ~prev_cond` idiom).  Gated by `available` on both bars so an
    # unavailable warmup predecessor cannot manufacture a rising edge.
    sweep_up_cond = sweep_up & available
    sweep_down_cond = sweep_down & available
    prev_sweep_up_cond = np.roll(sweep_up_cond, 1)
    prev_sweep_down_cond = np.roll(sweep_down_cond, 1)
    prev_sweep_up_cond[0] = False
    prev_sweep_down_cond[0] = False
    sweep_up_event = sweep_up_cond & ~prev_sweep_up_cond
    sweep_down_event = sweep_down_cond & ~prev_sweep_down_cond
    premium_discount = np.zeros(n_rows, dtype=np.float64)
    np.divide(
        close - range_low,
        channel_width,
        out=premium_discount,
        where=available,
    )
    premium_discount = np.clip(premium_discount, 0.0, 1.0)

    row_index = np.arange(n_rows, dtype=np.int64)
    support_dist = (close - last_low) / atr
    resistance_dist = (last_high - close) / atr
    support_age = row_index - last_low_idx
    resistance_age = row_index - last_high_idx
    support_rail_span = last_low_idx - prev_low_idx
    resistance_rail_span = last_high_idx - prev_high_idx
    support_slope = np.full(n_rows, np.nan, dtype=np.float64)
    resistance_slope = np.full(n_rows, np.nan, dtype=np.float64)
    np.divide(
        last_low - prev_low,
        support_rail_span.astype(np.float64) * atr,
        out=support_slope,
        where=available & (support_rail_span > 0),
    )
    np.divide(
        last_high - prev_high,
        resistance_rail_span.astype(np.float64) * atr,
        out=resistance_slope,
        where=available & (resistance_rail_span > 0),
    )
    support_break = np.maximum(last_low - close, 0.0) / atr
    resistance_break = np.maximum(close - last_high, 0.0) / atr
    nearest_level = np.minimum(np.abs(support_dist), np.abs(resistance_dist))
    range_mid_dist = (close - ((range_high + range_low) / 2.0)) / atr

    values = {
        "mtf_smc_structure_bias": structure_bias,
        "mtf_smc_bos_up": bos_up,
        "mtf_smc_bos_down": bos_down,
        "mtf_smc_choch_up": choch_up,
        "mtf_smc_choch_down": choch_down,
        "mtf_smc_sweep_up": sweep_up.astype(np.float64),
        "mtf_smc_sweep_down": sweep_down.astype(np.float64),
        "mtf_smc_sweep_up_depth_atr": sweep_up_depth,
        "mtf_smc_sweep_down_depth_atr": sweep_down_depth,
        "mtf_smc_premium_discount": premium_discount,
        "mtf_smc_range_width_atr": channel_width / atr,
        "mtf_smc_bos_displacement_atr": bos_displacement,
        "mtf_smc_sweep_up_event": sweep_up_event.astype(np.float64),
        "mtf_smc_sweep_down_event": sweep_down_event.astype(np.float64),
        "mtf_geometry_support_dist_atr": support_dist,
        "mtf_geometry_resistance_dist_atr": resistance_dist,
        "mtf_geometry_support_age_bars": support_age,
        "mtf_geometry_resistance_age_bars": resistance_age,
        "mtf_geometry_support_rail_slope_atr_per_bar": support_slope,
        "mtf_geometry_resistance_rail_slope_atr_per_bar": resistance_slope,
        "mtf_geometry_support_break_displacement_atr": support_break,
        "mtf_geometry_resistance_break_displacement_atr": resistance_break,
        "mtf_geometry_nearest_level_abs_atr": nearest_level,
        "mtf_geometry_range_mid_dist_atr": range_mid_dist,
    }
    expected_names = (
        SMC_MTF_FEATURE_NAMES_V1 + SMC_MTF_GEOMETRY_FEATURE_NAMES_V1
    )
    if tuple(values) != expected_names:
        raise RuntimeError("[SMC_MTF_OUTPUT_ORDER_INVALID]")

    out = pd.DataFrame(index=df.index, columns=expected_names, dtype=np.float64)
    for name, raw in values.items():
        column = np.asarray(raw, dtype=np.float64)
        column[~available] = np.nan
        out[name] = column
    numeric = out.to_numpy(dtype=np.float64, copy=False)
    complete = np.isfinite(numeric).all(axis=1)
    first_complete = int(np.argmax(complete))
    if (
        not complete.any()
        or not complete[first_complete:].all()
        or np.isinf(numeric).any()
    ):
        raise RuntimeError("[SMC_MTF_OUTPUT_AVAILABILITY_INVALID]")
    return out.astype(np.float32)
