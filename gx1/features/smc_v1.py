"""
Smart Money Concept (SMC) features — V1 implementation.

For each M5 bar, compute 9 SMC features that describe market structure,
liquidity events, and premium/discount position. Designed to feed the
canonical_v3 feature parquet and downstream model-native consumers.

Features (all per-bar, lookahead-safe):
  smc_swing_state        int8     0=HH+HL (clean up), 1=up-bias, 2=down-bias, 3=LH+LL (clean down), 4=mixed
  smc_bos_up             float32  1.0 if close > last confirmed swing high (break of structure up)
  smc_bos_down           float32  1.0 if close < last confirmed swing low (break of structure down)
  smc_choch              float32  1.0 only on the bar where structure flipped up↔down
  smc_sweep_up           float32  1.0 if high > last swing high but close <= it (false breakout / liquidity hunt)
  smc_sweep_down         float32  1.0 if low  < last swing low  but close >= it
  smc_sweep_size_atr     float32  magnitude of the wick beyond the swept level, ATR-normalized
  smc_bars_since_sweep   float32  bars elapsed since most recent sweep (clipped 999)
  smc_premium_discount   float32  (close - last_swing_low) / (last_swing_high - last_swing_low), in [0, 1]

Lookahead safety: a swing pivot at bar j is only considered "confirmed" once
j + SWING_LOOKBACK bars have elapsed. So features at bar i only use swings
confirmed up to bar (i - SWING_LOOKBACK), no future leakage.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd


SWING_LOOKBACK = 3  # bars look-around for swing pivot detection (3 → 7-bar window centered)

# sweep-THEN-reclaim (2026-06-11, env-gated GX1_SMC_SWEEP_RECLAIM, default OFF = contract byte-unchanged).
# A sweep that RECLAIMS the swept level = stop-hunt reversal (recoverable dip); a sweep with no reclaim =
# falling knife. CONTINUOUS displacement × decay (NOT a sparse binary → clears feature_liveness --strict).
_SWEEP_RECLAIM_ON = os.environ.get("GX1_SMC_SWEEP_RECLAIM", "0") == "1"
RECLAIM_WINDOW = 8      # bars after a sweep within which a reclaim still counts
DECAY_TAU = 12.0        # bars; exp-decay of the displacement after the reclaim (~1h on M5)
SWEEP_CAP = 5.0         # clip sweep-depth (ATR) so a data spike can't blow the value
DISP_CAP = 5.0          # clip reclaim displacement (ATR)


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
) -> pd.DataFrame:
    """Compute SMC features. Returns DataFrame with 9 new columns indexed same as input.

    Required columns on df: high, low, close and atr. All are exact observed or
    causally computed inputs; no ATR sentinel is permitted.
    """
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

    # 2. swing_state: HH/HL/LH/LL pattern at bar i
    higher_high = last_sh_price > prev_sh_price
    lower_high = last_sh_price < prev_sh_price
    higher_low = last_sl_price > prev_sl_price
    lower_low = last_sl_price < prev_sl_price
    # Default = 4 (mixed/unknown)
    swing_state = np.full(nb, 4, dtype=np.int8)
    swing_state[higher_high & higher_low] = 0          # clean up
    swing_state[(higher_high | higher_low) & ~(higher_high & higher_low) & ~lower_high & ~lower_low] = 1  # up-bias
    swing_state[(lower_high | lower_low) & ~(lower_high & lower_low) & ~higher_high & ~higher_low] = 2    # down-bias
    swing_state[lower_high & lower_low] = 3            # clean down

    # 3. BOS up/down — close breaks beyond last confirmed swing
    bos_up = np.zeros(nb, dtype=np.float32)
    bos_down = np.zeros(nb, dtype=np.float32)
    has_sh = last_sh >= 0
    has_sl = last_sl >= 0
    bos_up[has_sh] = (close[has_sh] > last_sh_price[has_sh]).astype(np.float32)
    bos_down[has_sl] = (close[has_sl] < last_sl_price[has_sl]).astype(np.float32)

    # 4. CHOCH — bar where state flipped up↔down
    choch = np.zeros(nb, dtype=np.float32)
    prev_state = -1
    for i in range(nb):
        cur = int(swing_state[i])
        if prev_state >= 0:
            was_up = prev_state in (0, 1)
            was_down = prev_state in (2, 3)
            now_up = cur in (0, 1)
            now_down = cur in (2, 3)
            if (was_up and now_down) or (was_down and now_up):
                choch[i] = 1.0
        prev_state = cur

    # 5. Liquidity sweep — wick beyond swept level, close back inside
    sweep_up = np.zeros(nb, dtype=np.float32)
    sweep_down = np.zeros(nb, dtype=np.float32)
    sweep_size_atr = np.zeros(nb, dtype=np.float32)
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
                any_sweep = True
        if has_sl[i]:
            sl_price = last_sl_price[i]
            if low[i] < sl_price and close[i] >= sl_price:
                sweep_down[i] = 1.0
                sd = float((sl_price - low[i]) / a)
                if sd > sweep_size_atr[i]:
                    sweep_size_atr[i] = sd
                any_sweep = True
        if any_sweep:
            last_sweep_at = i
        bars_since_sweep[i] = float(i - last_sweep_at) if last_sweep_at >= 0 else 999.0
    bars_since_sweep = np.clip(bars_since_sweep, 0, 999).astype(np.float32)

    # 5b. Sweep-THEN-reclaim (env-gated). Bullish = a DOWN-sweep (grab below support) then close back
    # ABOVE the swept low on an up bar; bearish = an UP-sweep then close back BELOW the swept high.
    # Magnitude = sweep-depth × (1 + reclaim-displacement), both ATR-normalized + clipped, then exp-decayed.
    sweep_reclaim_up = np.zeros(nb, dtype=np.float32)    # bullish: down-sweep → up-reclaim
    sweep_reclaim_down = np.zeros(nb, dtype=np.float32)  # bearish: up-sweep → down-reclaim
    if _SWEEP_RECLAIM_ON:
        # 2026-06-11 FAIL-LOUD: the old fallback open_=close made the reclaim conditions
        # (close>open / close<open) permanently FALSE → silently all-zero reclaim features
        # when the caller's frame lacks 'open'. Rule 9: a dead feature must fail, not ship.
        if "open" not in df.columns:
            raise ValueError(
                "[SMC_SWEEP_RECLAIM] GX1_SMC_SWEEP_RECLAIM=1 but the input frame has no 'open' "
                "column — the reclaim features would be constant-zero. Pass OHLC including 'open'."
            )
        open_ = df["open"].to_numpy(dtype=np.float64)
        pend_dn = (-1, np.nan, 0.0)   # DOWN-sweep awaiting an UP-reclaim (bullish): (sweep_bar, swept_lvl, wick_atr)
        pend_up = (-1, np.nan, 0.0)   # UP-sweep awaiting a DOWN-reclaim (bearish)
        act_up = (-1, 0.0)            # active bullish reclaim: (reclaim_bar, strength)
        act_dn = (-1, 0.0)
        for i in range(nb):
            a = atr[i]
            if sweep_down[i] > 0.0 and has_sl[i]:
                pend_dn = (i, float(last_sl_price[i]), min(float(sweep_size_atr[i]), SWEEP_CAP))
            if sweep_up[i] > 0.0 and has_sh[i]:
                pend_up = (i, float(last_sh_price[i]), min(float(sweep_size_atr[i]), SWEEP_CAP))
            sb, lvl, wick = pend_dn
            if sb >= 0 and (i - sb) <= RECLAIM_WINDOW and close[i] > lvl and close[i] > open_[i]:
                disp = min((close[i] - lvl) / a, DISP_CAP)
                act_up = (i, wick * (1.0 + max(disp, 0.0)))
                pend_dn = (-1, np.nan, 0.0)
                act_dn = (-1, 0.0)  # consumed + opposite invalidated
            sb2, lvl2, wick2 = pend_up
            if sb2 >= 0 and (i - sb2) <= RECLAIM_WINDOW and close[i] < lvl2 and close[i] < open_[i]:
                disp = min((lvl2 - close[i]) / a, DISP_CAP)
                act_dn = (i, wick2 * (1.0 + max(disp, 0.0)))
                pend_up = (-1, np.nan, 0.0)
                act_up = (-1, 0.0)
            rb, st = act_up
            if rb >= 0:
                v = st * float(np.exp(-(i - rb) / DECAY_TAU))
                sweep_reclaim_up[i] = v if v > 1e-3 else 0.0
            rb, st = act_dn
            if rb >= 0:
                v = st * float(np.exp(-(i - rb) / DECAY_TAU))
                sweep_reclaim_down[i] = v if v > 1e-3 else 0.0

    # 6. Premium/discount score — close position in [last_sl, last_sh] range
    pd_score = np.full(nb, 0.5, dtype=np.float32)
    valid_pd = (last_sh >= 0) & (last_sl >= 0) & (last_sh_price > last_sl_price)
    rng = np.where(valid_pd, last_sh_price - last_sl_price, 1.0)
    pd_score[valid_pd] = ((close[valid_pd] - last_sl_price[valid_pd]) / rng[valid_pd]).astype(np.float32)
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
    if _SWEEP_RECLAIM_ON:  # appended ONLY under the flag → default-OFF keeps the 9-col contract byte-identical
        out_cols["smc_sweep_reclaim_up_displacement_atr"] = sweep_reclaim_up
        out_cols["smc_sweep_reclaim_down_displacement_atr"] = sweep_reclaim_down
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
if _SWEEP_RECLAIM_ON:  # contract grows ONLY under the flag (GX1_SMC_SWEEP_RECLAIM=1)
    SMC_FEATURE_NAMES = SMC_FEATURE_NAMES + [
        "smc_sweep_reclaim_up_displacement_atr",
        "smc_sweep_reclaim_down_displacement_atr",
    ]


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

    sweep_up = (high > last_high) & (close <= last_high)
    sweep_down = (low < last_low) & (close >= last_low)
    sweep_up_depth = np.where(sweep_up, (high - last_high) / atr, 0.0)
    sweep_down_depth = np.where(sweep_down, (last_low - low) / atr, 0.0)
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
