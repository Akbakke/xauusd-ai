"""
Smart Money Concept (SMC) features — V1 implementation.

For each M5 bar, compute 9 SMC features that describe market structure,
liquidity events, and premium/discount position. Designed to feed the
canonical_v3 feature parquet (and downstream XGB / V10 / V3 / IQL).

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
) -> pd.DataFrame:
    """Compute SMC features. Returns DataFrame with 9 new columns indexed same as input.

    Required columns on df: high, low, close (and atr if present — falls back to 1.0).
    """
    nb = len(df)
    high = df[high_col].to_numpy(dtype=np.float64)
    low = df[low_col].to_numpy(dtype=np.float64)
    close = df[close_col].to_numpy(dtype=np.float64)
    atr = (
        df[atr_col].fillna(method="ffill").fillna(0.0).to_numpy(dtype=np.float64)
        if atr_col in df.columns
        else np.ones(nb, dtype=np.float64)
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
        a = atr[i] if atr[i] > 1e-9 else 1e-6
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

    # 6. Premium/discount score — close position in [last_sl, last_sh] range
    pd_score = np.full(nb, 0.5, dtype=np.float32)
    valid_pd = (last_sh >= 0) & (last_sl >= 0) & (last_sh_price > last_sl_price)
    rng = np.where(valid_pd, last_sh_price - last_sl_price, 1.0)
    pd_score[valid_pd] = ((close[valid_pd] - last_sl_price[valid_pd]) / rng[valid_pd]).astype(np.float32)
    pd_score = np.clip(pd_score, 0.0, 1.0)

    out = pd.DataFrame(
        {
            "smc_swing_state": swing_state,
            "smc_bos_up": bos_up,
            "smc_bos_down": bos_down,
            "smc_choch": choch,
            "smc_sweep_up": sweep_up,
            "smc_sweep_down": sweep_down,
            "smc_sweep_size_atr": sweep_size_atr,
            "smc_bars_since_sweep": bars_since_sweep,
            "smc_premium_discount": pd_score,
        },
        index=df.index,
    )
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
