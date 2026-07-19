#!/usr/bin/env python3
"""gx1.features.trade_overlay — ONE-TRUTH V3 trade-state overlay (19 cols).

The V3 exit transformer's in-trade window has 19 trade-state slots that are
OVERLAID onto the canonical feature window (0 pre-trade). This module is the
single source of truth for how those 19 slots are computed. The live serve path
(`gx1.execution.v12_trade_state.TradeState.build_v3_overlay`) calls it directly;
any future V3 training builder must call this same function and prove parity.

Basis: INTRABAR favorable/adverse excursion. The caller supplies, per in-trade
bar:
  peak[t]    favorable excursion bps  (train: peak_high; long bid_high-entry,
                                        short entry-ask_low; serve: spread-side
                                        bid_high/ask_low — same basis)
  trough[t]  adverse excursion bps    (train: trough_low)
  cur_pnl[t] close-mark pnl bps       (long bid_close, short ask_close)
  atr_bps[t] per-bar (ask_high-bid_low)/mid*1e4
plus entry_snap (5 constants broadcast across the window).

Column order MUST equal v12_v3_live.TRADE_STATE_FEATURE_NAMES (the consumer
maps by name->index). DO NOT reorder. Both train + serve call THIS function, so
they are bit-identical by construction (the retrain's contract value = whatever
this emits). float32 output (the contract dtype); float64 intermediates.
"""
from __future__ import annotations

import numpy as np

# 19 overlay column names, IN ORDER. Must match
# gx1/execution/v12_v3_live.py:TRADE_STATE_FEATURE_NAMES exactly.
OVERLAY_COL_NAMES = [
    "p_long_entry", "p_hat_entry", "uncertainty_entry", "entropy_entry", "margin_entry",
    "pnl_bps_now", "mfe_bps", "mae_bps", "dd_from_mfe_bps",
    "distance_from_peak_mfe_bps", "bars_held", "time_since_mfe_bars",
    "mfe_decay_rate", "pnl_velocity", "pnl_acceleration",
    "rolling_slope_since_entry", "atr_bps_now",
    "giveback_ratio", "giveback_acceleration",
]
N_OVERLAY_COLS = 19
assert len(OVERLAY_COL_NAMES) == N_OVERLAY_COLS


def compute_trade_overlay(
    peak: np.ndarray,
    trough: np.ndarray,
    cur_pnl: np.ndarray,
    atr_bps: np.ndarray,
    entry_snap: dict,
) -> np.ndarray:
    """Return the (n, 19) float32 trade-state overlay.

    This function now owns the exact math. cols 0-4 = entry-snapshot constants,
    cols 5-18 = per-bar trade-state. A dataset implementation may consume this
    helper, but may not duplicate or reinterpret it.
    """
    peak = np.asarray(peak, dtype=np.float64)
    trough = np.asarray(trough, dtype=np.float64)
    cur_pnl = np.asarray(cur_pnl, dtype=np.float64)
    atr_bps = np.asarray(atr_bps, dtype=np.float64)
    n = len(cur_pnl)
    overlay = np.zeros((n, N_OVERLAY_COLS), dtype=np.float32)
    if n == 0:
        return overlay

    # cum MFE/MAE from INTRABAR peak/trough
    cum_peak = np.maximum.accumulate(peak)
    cum_trough = np.minimum.accumulate(trough)

    # arg_peak: index of the running max of peak[0..t]
    arg_peak = np.zeros(n, dtype=np.int32)
    running_max = -np.inf
    running_max_idx = 0
    for i in range(n):
        if peak[i] >= running_max:
            running_max = float(peak[i])
            running_max_idx = i
        arg_peak[i] = running_max_idx

    # pnl velocity / acceleration
    pnl_vel = np.zeros(n, dtype=np.float64)
    pnl_acc = np.zeros(n, dtype=np.float64)
    if n >= 2:
        pnl_vel[1:] = cur_pnl[1:] - cur_pnl[:-1]
    if n >= 3:
        pnl_acc[2:] = pnl_vel[2:] - pnl_vel[1:-1]

    # mfe_decay: cum_peak[t] - cum_peak[t-4]  (train v2.py:475-477)
    mfe_decay = np.zeros(n, dtype=np.float64)
    if n > 4:
        mfe_decay[4:] = cum_peak[4:] - cum_peak[:-4]

    # giveback: 1 - cur_pnl/max(cum_peak,1e-6), clip[-10,10] (train v2.py:478-479)
    pos_peak = np.maximum(cum_peak, 1e-6)
    giveback = np.clip((1.0 - cur_pnl / pos_peak), -10.0, 10.0)

    # giveback acceleration: 2nd diff of giveback (train v2.py:480-484)
    giveback_acc = np.zeros(n, dtype=np.float64)
    if n >= 3:
        gv_vel = np.zeros(n, dtype=np.float64)
        gv_vel[1:] = giveback[1:] - giveback[:-1]
        giveback_acc[2:] = gv_vel[2:] - gv_vel[1:-1]

    # rolling_slope: expanding closed-form OLS of cur_pnl vs bar_idx over [0..i]
    # (train v2.py:489-501)
    rolling_slope = np.zeros(n, dtype=np.float64)
    if n >= 3:
        _idx = np.arange(n, dtype=np.float64)
        _cum_y = np.cumsum(cur_pnl)
        _cum_xy = np.cumsum(_idx * cur_pnl)
        for i in range(2, n):
            m = i + 1
            sum_x = i * m / 2.0
            sum_x2 = i * m * (2 * i + 1) / 6.0
            denom = m * sum_x2 - sum_x * sum_x
            if abs(denom) > 1e-9:
                rolling_slope[i] = (m * _cum_xy[i] - sum_x * _cum_y[i]) / denom

    # cols 0-4 entry-snapshot (broadcast), order per train v2.py:518-522
    overlay[:, 0] = float(entry_snap.get("p_long_entry", 0.0))
    overlay[:, 1] = float(entry_snap.get("p_hat_entry", 0.0))
    overlay[:, 2] = float(entry_snap.get("uncertainty_entry", 0.0))
    overlay[:, 3] = float(entry_snap.get("entropy_entry", 0.0))
    overlay[:, 4] = float(entry_snap.get("margin_entry", 0.0))
    # cols 5-18 per-bar trade-state, order per train v2.py:523-536
    overlay[:, 5] = cur_pnl
    overlay[:, 6] = cum_peak
    overlay[:, 7] = cum_trough
    overlay[:, 8] = cum_peak - cur_pnl                         # dd_from_mfe_bps
    overlay[:, 9] = cum_peak - cur_pnl                         # distance_from_peak (synonym)
    overlay[:, 10] = np.arange(n, dtype=np.float64)            # bars_held (0-based!)
    overlay[:, 11] = np.arange(n, dtype=np.float64) - arg_peak.astype(np.float64)  # time_since_mfe
    overlay[:, 12] = mfe_decay
    overlay[:, 13] = pnl_vel
    overlay[:, 14] = pnl_acc
    overlay[:, 15] = rolling_slope
    overlay[:, 16] = atr_bps
    overlay[:, 17] = giveback
    overlay[:, 18] = giveback_acc
    return overlay
