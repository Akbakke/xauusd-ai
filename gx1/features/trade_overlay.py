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


def compute_m1_micro_feature_arrays(
    close_mid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the one-truth M1 log-return/RMS-volatility feature arrays.

    The five outputs are aligned to ``close_mid`` and correspond to the
    5/15/60-bar log returns plus the 15/60-bar root-mean-square log returns,
    all in bps. Structural warm-up values are zero in both train and serve.
    """

    close = np.asarray(close_mid, dtype=np.float64)
    if close.ndim != 1:
        raise ValueError("M1 micro-feature close input must be one-dimensional")
    if not np.isfinite(close).all() or np.any(close <= 0.0):
        raise ValueError("M1 micro-feature close input must be finite and positive")
    n = len(close)
    if n == 0:
        empty = np.empty(0, dtype=np.float32)
        return empty, empty, empty, empty, empty

    log_close = np.log(close)

    def _rolling_return_bps(lag: int) -> np.ndarray:
        out = np.zeros(n, dtype=np.float32)
        if n > lag:
            out[lag:] = (
                (log_close[lag:] - log_close[:-lag]) * 10_000.0
            ).astype(np.float32)
        return out

    bar_log_ret_bps = np.zeros(n, dtype=np.float64)
    if n > 1:
        bar_log_ret_bps[1:] = np.diff(log_close) * 10_000.0

    def _rolling_rms_bps(window: int) -> np.ndarray:
        out = np.zeros(n, dtype=np.float32)
        if n < 2:
            return out
        squared = bar_log_ret_bps * bar_log_ret_bps
        cumulative = np.concatenate(([0.0], np.cumsum(squared)))
        for index in range(n):
            left = max(0, index - window + 1)
            count = index - left + 1
            if count >= 2:
                mean_square = (
                    cumulative[index + 1] - cumulative[left]
                ) / count
                out[index] = float(np.sqrt(max(mean_square, 0.0)))
        return out

    return (
        _rolling_return_bps(5),
        _rolling_return_bps(15),
        _rolling_return_bps(60),
        _rolling_rms_bps(15),
        _rolling_rms_bps(60),
    )


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
    if any(array.ndim != 1 for array in (peak, trough, cur_pnl, atr_bps)):
        raise ValueError("trade-overlay inputs must be one-dimensional")
    n = len(cur_pnl)
    if any(len(array) != n for array in (peak, trough, atr_bps)):
        raise ValueError("trade-overlay input lengths must match")
    if (
        not all(
            np.isfinite(array).all()
            for array in (peak, trough, cur_pnl, atr_bps)
        )
        or np.any(atr_bps <= 0.0)
        or np.any(peak < trough)
        or np.any(cur_pnl > peak)
        or np.any(cur_pnl < trough)
    ):
        raise ValueError("trade-overlay input values are invalid")
    required_entry_fields = {
        "p_long_entry",
        "p_hat_entry",
        "uncertainty_entry",
        "entropy_entry",
        "margin_entry",
    }
    if not isinstance(entry_snap, dict) or set(entry_snap) != required_entry_fields:
        raise ValueError("trade-overlay entry snapshot exact schema mismatch")
    entry_values = {
        name: float(entry_snap[name]) for name in required_entry_fields
    }
    if not all(np.isfinite(value) for value in entry_values.values()):
        raise ValueError("trade-overlay entry snapshot must be finite")
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
    overlay[:, 0] = entry_values["p_long_entry"]
    overlay[:, 1] = entry_values["p_hat_entry"]
    overlay[:, 2] = entry_values["uncertainty_entry"]
    overlay[:, 3] = entry_values["entropy_entry"]
    overlay[:, 4] = entry_values["margin_entry"]
    # cols 5-18 per-bar trade-state, order per train v2.py:523-536
    overlay[:, 5] = cur_pnl
    overlay[:, 6] = cum_peak
    overlay[:, 7] = cum_trough
    overlay[:, 8] = cum_peak - cur_pnl                         # dd_from_mfe_bps
    overlay[:, 9] = cum_peak - cur_pnl                         # distance_from_peak (synonym)
    overlay[:, 10] = np.arange(1, n + 1, dtype=np.float64)     # bars_held (one-based)
    overlay[:, 11] = np.arange(n, dtype=np.float64) - arg_peak.astype(np.float64)  # time_since_mfe
    overlay[:, 12] = mfe_decay
    overlay[:, 13] = pnl_vel
    overlay[:, 14] = pnl_acc
    overlay[:, 15] = rolling_slope
    overlay[:, 16] = atr_bps
    overlay[:, 17] = giveback
    overlay[:, 18] = giveback_acc
    return overlay
