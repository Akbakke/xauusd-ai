#!/usr/bin/env python3
"""V12 trade-state tracker — maintains per-trade running state across M1 bars
for the Exit-IQL V12.1 per-bar exit-decision loop.

Each open trade has a TradeState that records:
  - entry timestamp + side + entry prices (bid+ask snapshot)
  - V10 snapshot at entry (frozen — used as 'v10_*_at_entry_v1' features)
  - Per-bar PnL trajectory (used for MFE/MAE/cum_peak/drawdown features)
  - Recent M1 return window (used for vol/return-since-entry features)

The Exit-IQL state vector at any M1 bar combines:
  - TradeState-derived running stats (~15 trade-state features)
  - V3 v8 outputs at this bar (4 features) — currently stubbed to 0
  - V10 entry-snapshot (10 features)
  - canonical_v3 + augmented features at current bar (~170 features)

Usage:
    trade = TradeState.open(
        entry_ts=current_minute,
        side="long",
        entry_bid=4685.0, entry_ask=4685.5,
        v10_snapshot=v10_out,
    )
    # Each M1 bar after entry:
    trade.update_bar(now_ts, bid=4686.0, ask=4686.5, m1_close=4686.0)
    state_features = trade.build_exit_state_features(
        canonical_v3_row=augmented_cv3.loc[now_ts],
        v3_v8_out=None,   # if None, V3-tracking features default to 0
    )
"""
from __future__ import annotations

import json
import logging
import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.features.trade_overlay import OVERLAY_COL_NAMES, compute_trade_overlay

LOG = logging.getLogger("v12_trade_state")

SIDE_LONG = "long"
SIDE_SHORT = "short"
SIDES = (SIDE_LONG, SIDE_SHORT)

# Default V3-tracking values when V3 v8 inference is not wired
DEFAULT_V3_FEATURES = {
    "v3_v8_should_exit_prob": 0.0,
    "v3_v8_profit_protect_prob": 0.0,
    "v3_v8_family_argmax": 0.0,
    "v3_v8_family_logit_max": 0.0,
}


@dataclass
class TradeState:
    """Per-trade running state.

    Long entry: ask_open at entry → exit at bid_close (spread cost on both sides).
    Short entry: bid_open at entry → exit at ask_close.
    """
    entry_ts: pd.Timestamp
    side: str                            # "long" or "short"
    entry_bid: float                     # bid at entry minute
    entry_ask: float                     # ask at entry minute
    entry_spread_bps: float
    v10_snapshot: dict[str, Any]         # frozen V10 outputs at entry

    # Identity (set after OANDA fill — used as state-file name + close-trade id)
    trade_id: str | None = None
    units: int = 1

    # Running state (updated per M1 bar)
    bars_in_trade: int = 0
    current_bid: float = 0.0
    current_ask: float = 0.0
    current_pnl_bps: float = 0.0         # bid/ask asymmetric
    cum_mfe_bps: float = 0.0             # running max favorable PnL
    cum_mae_bps: float = 0.0             # running min (most-adverse) PnL
    bars_since_mfe_peak: int = 0

    # Recent M1 return window (for vol/momentum features)
    m1_returns_window: deque = field(default_factory=lambda: deque(maxlen=120))

    # V3 v8 running stats (updated per bar via update_v3)
    v3_last_prob: float = 0.0
    v3_max_prob_in_trade: float = 0.0
    v3_consecutive_exits: int = 0
    v3_max_consecutive_exits: int = 0
    v3_total_exit_decisions: int = 0
    v3_signal_acceleration: float = 0.0      # latest delta-prob bar-to-bar

    # Per-bar trajectory (used for V3 overlay + trade-state metrics)
    pnl_history: deque = field(default_factory=lambda: deque(maxlen=2000))   # bps per bar
    mfe_at_bar: float = 0.0                  # mfe_bps at last bar
    time_since_mfe_peak_bars: int = 0
    last_atr_bps: float = 0.0

    # V4 (R13) intrabar excursion history — feeds the ONE-TRUTH V3 overlay helper
    # (gx1.features.trade_overlay.compute_trade_overlay), the SAME function the
    # train builder calls, so build_v3_overlay is bit-identical to training.
    # peak/trough = INTRABAR favorable/adverse excursion bps (spread-side: long
    # bid_high/bid_low, short ask_low/ask_high); atr = per-bar (ask_high-bid_low)/
    # mid*1e4. One value appended per CLOSED M1 bar, in lock-step with pnl_history.
    peak_history: deque = field(default_factory=lambda: deque(maxlen=2000))
    trough_history: deque = field(default_factory=lambda: deque(maxlen=2000))
    atr_bps_history: deque = field(default_factory=lambda: deque(maxlen=2000))

    @classmethod
    def open(
        cls,
        entry_ts: pd.Timestamp,
        side: str,
        entry_bid: float,
        entry_ask: float,
        v10_snapshot: dict[str, Any] | None = None,
        trade_id: str | None = None,
        units: int = 1,
    ) -> "TradeState":
        if side not in SIDES:
            raise ValueError(f"side must be {SIDES}, got {side!r}")
        if entry_bid <= 0 or entry_ask <= 0 or entry_ask <= entry_bid:
            raise ValueError(f"invalid prices: bid={entry_bid} ask={entry_ask}")
        spread_bps = (entry_ask - entry_bid) / entry_bid * 10000.0
        return cls(
            entry_ts=pd.Timestamp(entry_ts),
            side=str(side),
            entry_bid=float(entry_bid),
            entry_ask=float(entry_ask),
            entry_spread_bps=float(spread_bps),
            v10_snapshot=dict(v10_snapshot or {}),
            trade_id=trade_id,
            units=int(units),
            current_bid=float(entry_bid),
            current_ask=float(entry_ask),
        )

    def _pnl_bps(self, bid: float, ask: float) -> float:
        """Bid/ask asymmetric unrealized PnL in bps."""
        if self.side == SIDE_LONG:
            # entry @ ask, mark @ bid
            return (bid - self.entry_ask) / self.entry_ask * 10000.0
        else:
            # entry @ bid, mark @ ask
            return (self.entry_bid - ask) / self.entry_bid * 10000.0

    def _intrabar_excursion(
        self, bid_high: float, bid_low: float, ask_high: float, ask_low: float,
        bid_close: float, ask_close: float,
    ) -> tuple[float, float, float]:
        """Per-bar INTRABAR favorable/adverse excursion + atr in bps, IDENTICAL to
        the train builder (materialize_build_exit_iql_per_bar_dataset_v1.compute_per_bar_signals).

          long  (entry@ask): peak=(bid_high-entry_ask)/entry_ask, trough=(bid_low-entry_ask)/entry_ask
          short (entry@bid): peak=(entry_bid-ask_low)/entry_bid,  trough=(entry_bid-ask_high)/entry_bid
          atr = (ask_high-bid_low)/mid*1e4,  mid=(ask_close+bid_close)/2
        """
        if self.side == SIDE_LONG:
            peak = (bid_high - self.entry_ask) / self.entry_ask * 10000.0
            trough = (bid_low - self.entry_ask) / self.entry_ask * 10000.0
        else:
            peak = (self.entry_bid - ask_low) / self.entry_bid * 10000.0
            trough = (self.entry_bid - ask_high) / self.entry_bid * 10000.0
        mid = (ask_close + bid_close) / 2.0
        atr = (ask_high - bid_low) / mid * 10000.0 if mid > 0 else 0.0
        return float(peak), float(trough), float(atr)

    def update_bar(
        self, bid: float, ask: float, m1_close: float,
        bid_high: float | None = None, bid_low: float | None = None,
        ask_high: float | None = None, ask_low: float | None = None,
    ) -> None:
        """Advance trade state by one CLOSED M1 bar.

        bid/ask = this bar's CLOSE bid/ask (the mark prices). m1_close = mid close
        (drives the return window). bid_high/bid_low/ask_high/ask_low = this bar's
        intrabar range — REQUIRED for the V4 one-truth V3 overlay (intrabar MFE/MAE
        basis, matches the train builder). When absent (tests / non-live callers
        without OHLC) the bar's range collapses to its close (peak=trough=close-mark
        pnl, atr=spread) — a degraded close-only fallback that re-introduces the
        pre-V4 overlay skew, so the LIVE pipeline MUST pass them
        (v12_pipeline.make_exit_decision threads them from the collector parquet).
        """
        self.bars_in_trade += 1
        prev_close = m1_close if not self.m1_returns_window else self.m1_returns_window[-1]
        ret_bps = (m1_close - prev_close) / prev_close * 10000.0 if prev_close > 0 else 0.0
        self.m1_returns_window.append(m1_close)

        self.current_bid = float(bid)
        self.current_ask = float(ask)
        self.current_pnl_bps = float(self._pnl_bps(bid, ask))
        prev_peak = self.cum_mfe_bps
        self.cum_mfe_bps = max(self.cum_mfe_bps, self.current_pnl_bps)
        self.cum_mae_bps = min(self.cum_mae_bps, self.current_pnl_bps)
        if self.cum_mfe_bps > prev_peak:
            self.bars_since_mfe_peak = 0
            self.time_since_mfe_peak_bars = 0
        else:
            self.bars_since_mfe_peak += 1
            self.time_since_mfe_peak_bars += 1
        self.mfe_at_bar = self.cum_mfe_bps
        self.pnl_history.append(self.current_pnl_bps)

        # V4 (R13): record INTRABAR excursion + per-bar atr for the one-truth overlay.
        if None not in (bid_high, bid_low, ask_high, ask_low):
            peak, trough, atr = self._intrabar_excursion(
                float(bid_high), float(bid_low), float(ask_high), float(ask_low), bid, ask,
            )
        else:
            # close-only degrade: no intrabar range available for this caller.
            peak = trough = self.current_pnl_bps
            denom = (ask + bid) / 2.0
            atr = (ask - bid) / denom * 10000.0 if denom > 0 else 0.0
        self.peak_history.append(float(peak))
        self.trough_history.append(float(trough))
        self.atr_bps_history.append(float(atr))

    def update_v3(self, v3_v8_out: dict | None) -> None:
        """Update V3 v8 running stats from latest V3 inference output."""
        if not v3_v8_out:
            return
        prob = float(v3_v8_out.get("v3_v8_should_exit_prob", 0.0))
        self.v3_signal_acceleration = prob - self.v3_last_prob
        self.v3_last_prob = prob
        if prob > self.v3_max_prob_in_trade:
            self.v3_max_prob_in_trade = prob
        if prob > 0.5:
            self.v3_consecutive_exits += 1
            self.v3_total_exit_decisions += 1
            if self.v3_consecutive_exits > self.v3_max_consecutive_exits:
                self.v3_max_consecutive_exits = self.v3_consecutive_exits
        else:
            self.v3_consecutive_exits = 0

    # ── feature construction ────────────────────────────────────────

    def _rolling_return_bps(self, n: int) -> float:
        """M1 close return over EXACTLY n bars (bps).

        Returns 0.0 if fewer than (n+1) bars are available — matches training
        convention where partial-window features were 0-filled at trade start.
        (Audit H2 2026-05-19: previously returned a SHORTER-lookback value
        silently, e.g. ``m1_last_60bar_return_bps_v1`` was actually a 2-bar
        return early in trade, diverging from training data.)
        """
        if len(self.m1_returns_window) < n + 1:
            return 0.0
        prev = self.m1_returns_window[-(n + 1)]
        cur = self.m1_returns_window[-1]
        if prev <= 0:
            return 0.0
        return (cur - prev) / prev * 10000.0

    def _rolling_vol_bps(self, n: int) -> float:
        """Std of M1 close-to-close returns (bps) over last n bars."""
        if len(self.m1_returns_window) < 3:
            return 0.0
        arr = np.array(list(self.m1_returns_window)[-(n + 1):], dtype=np.float64)
        if len(arr) < 3:
            return 0.0
        rets = np.diff(arr) / arr[:-1] * 10000.0
        return float(rets.std())

    def build_trade_state_features(self) -> dict[str, float]:
        """The ~13 per-bar trade-state features Exit-IQL V12.1 expects."""
        drawdown_from_peak = self.cum_mfe_bps - self.current_pnl_bps
        # bar_return_bps_v1: return of THIS bar (last - prev close)
        if len(self.m1_returns_window) >= 2:
            bar_return = (self.m1_returns_window[-1] - self.m1_returns_window[-2]) / self.m1_returns_window[-2] * 10000.0
        else:
            bar_return = 0.0

        # Trade-state derivatives — formulas MUST match training EXACTLY
        # (`materialize_build_exit_iql_per_bar_dataset_v1.py:283-322`). Live had
        # subtly wrong formulas in earlier fix (audit 3 C-1/C-2 2026-05-20):
        #   - mfe_decay_rate: training is cum_peak[t] - cum_peak[t-4]
        #     (forward growth ≥ 0), live had used (max(h[-4:]) - h[-1])/4
        #   - giveback_ratio: training clips to [-10, 10] with 1e-6 epsilon,
        #     live had clipped to [0, 2] with 1.0 epsilon (different scale).
        #   - giveback_acceleration: training is SECOND diff of giveback,
        #     live had used FIRST diff (velocity, not acceleration).
        #   - rolling_slope: training is closed-form OLS slope of pnl vs bar_idx
        #     over [0..t], live had used rolling-return divided by bars.
        h = np.asarray(list(self.pnl_history), dtype=np.float64)
        n = len(h)
        if n >= 1:
            cum_peak = np.maximum.accumulate(h)
        # pnl_velocity / pnl_acceleration: first / second discrete differences
        pnl_velocity = float(h[-1] - h[-2]) if n >= 2 else 0.0
        pnl_acceleration = float(h[-1] - 2.0 * h[-2] + h[-3]) if n >= 3 else 0.0
        # mfe_decay_rate: cum_peak[t] - cum_peak[t-4]  (monotone ≥ 0 increase)
        mfe_decay_rate = float(cum_peak[-1] - cum_peak[-5]) if n >= 5 else 0.0
        # giveback_ratio: 1 - cur_pnl/max(cum_peak, 1e-6), clipped to [-10, 10]
        if n >= 1:
            pos_peak = max(float(cum_peak[-1]), 1e-6)
            giveback_ratio = 1.0 - h[-1] / pos_peak
            giveback_ratio = float(np.clip(giveback_ratio, -10.0, 10.0))
        else:
            giveback_ratio = 0.0
        # giveback_acceleration: second diff of giveback timeseries
        if n >= 3:
            gv_t = 1.0 - h[-1] / max(float(cum_peak[-1]), 1e-6)
            gv_t1 = 1.0 - h[-2] / max(float(cum_peak[-2]), 1e-6)
            gv_t2 = 1.0 - h[-3] / max(float(cum_peak[-3]), 1e-6)
            giveback_acceleration = float(np.clip(gv_t - 2.0 * gv_t1 + gv_t2, -10.0, 10.0))
        else:
            giveback_acceleration = 0.0

        # rolling_slope_since_entry_v1: closed-form OLS slope of pnl vs bar_idx
        # over [0..n-1]. slope = (n·Σxy − Σx·Σy) / (n·Σx² − (Σx)²).
        if n >= 3:
            idx = np.arange(n, dtype=np.float64)
            sum_x = idx.sum()
            sum_x2 = (idx * idx).sum()
            sum_y = h.sum()
            sum_xy = (idx * h).sum()
            denom = n * sum_x2 - sum_x * sum_x
            rolling_slope = float((n * sum_xy - sum_x * sum_y) / denom) if abs(denom) > 1e-9 else 0.0
        else:
            rolling_slope = 0.0

        return {
            "bars_in_trade_v1": float(self.bars_in_trade),
            "current_unrealized_pnl_bps_v1": float(self.current_pnl_bps),
            "current_mfe_bps_v1": float(self.cum_mfe_bps),
            "current_mae_bps_v1": float(self.cum_mae_bps),
            "bars_since_mfe_peak_v1": float(self.bars_since_mfe_peak),
            "pnl_drawdown_from_peak_v1": float(drawdown_from_peak),
            "bar_return_bps_v1": float(bar_return),
            "m1_last_5bar_return_bps_v1": float(self._rolling_return_bps(5)),
            "m1_last_15bar_return_bps_v1": float(self._rolling_return_bps(15)),
            "m1_last_60bar_return_bps_v1": float(self._rolling_return_bps(60)),
            "m1_realized_vol_15bar_bps_v1": float(self._rolling_vol_bps(15)),
            "m1_realized_vol_60bar_bps_v1": float(self._rolling_vol_bps(60)),
            # 6 trade-state derivatives — match training builder formulas exactly.
            "pnl_velocity_v1": float(pnl_velocity),
            "pnl_acceleration_v1": float(pnl_acceleration),
            "mfe_decay_rate_v1": float(mfe_decay_rate),
            "giveback_ratio_v1": float(giveback_ratio),
            "giveback_acceleration_v1": float(giveback_acceleration),
            "rolling_slope_since_entry_v1": float(rolling_slope),
        }

    def build_v10_entry_snapshot_features(self) -> dict[str, float]:
        """V10 outputs frozen at trade entry, exposed as exit-IQL features."""
        s = self.v10_snapshot
        dp = s.get("direction_probs", [0.0, 0.0, 0.0])
        p_long_e = float(dp[0] if hasattr(dp, "__len__") else 0.0)
        p_short_e = float(dp[1] if hasattr(dp, "__len__") and len(dp) > 1 else 0.0)
        return {
            "v10_p_long_at_entry_v1": p_long_e,
            "v10_p_short_at_entry_v1": p_short_e,
            "v10_path_quality_at_entry_v1": float(s.get("path_quality", 0.0)),
            "v10_mfe_pred_at_entry_v1": float(s.get("mfe_first_n", 0.0)),
            "v10_tradable_at_entry_v1": float(s.get("tradable_prob", 0.0)),
            "v10_bad_path_at_entry_v1": float(s.get("bad_path_prob", 0.0)),
            # V10 v3+ aux head outputs frozen at entry — Exit-IQL was retrained
            # 2026-05-19 to consume these 4 features (208-dim state vector).
            # Without them they're silently 0-filled and Exit-IQL Q-values
            # drift from what training saw.
            "v10_tf_agreement_at_entry_v1": float(s.get("tf_agreement_pred", 0.0)),
            "v10_path_quality_std_at_entry_v1": float(s.get("path_quality_std", 0.0)),
            "v10_position_size_at_entry_v1": float(s.get("position_size_pred", 0.0)),
            "v10_hold_horizon_at_entry_v1": float(s.get("hold_horizon_pred", 0.0)),
            # V3 v8 frozen at entry (would be from V3 inference at entry bar)
            "p_long_entry_v1": p_long_e,
            "p_hat_entry_v1": float(max(p_long_e, p_short_e, 1.0 - p_long_e - p_short_e)),
            "uncertainty_entry_v1": float(1.0 - max(p_long_e, p_short_e, 1.0 - p_long_e - p_short_e)),
            # margin = top1-top2 gap of the 3-class probs — matches the candidate-gen
            # (sorted[-1]-sorted[-2]) and the Exit-IQL state builder
            # (materialize_build_exit_iql_per_bar_dataset_v2_m1.py:247 candidate_row["margin"]).
            # NOT abs(p_long-p_short) (wrong when p_flat is top1).
            "margin_entry_v1": float(
                sorted((p_long_e, p_short_e, max(0.0, 1.0 - p_long_e - p_short_e)))[-1]
                - sorted((p_long_e, p_short_e, max(0.0, 1.0 - p_long_e - p_short_e)))[-2]),
            # entropy
            "entropy_entry_v1": float(self.v10_snapshot.get("entropy_at_entry",
                _shannon_entropy([p_long_e, p_short_e, max(0.0, 1.0 - p_long_e - p_short_e)]))),
            # rolling_slope_since_entry_v1 is now produced by build_trade_state_features()
            # using closed-form OLS slope over pnl_history (matches training).
        }

    def build_side_one_hot(self) -> dict[str, float]:
        return {
            "side_v1_long": 1.0 if self.side == SIDE_LONG else 0.0,
            "side_v1_short": 1.0 if self.side == SIDE_SHORT else 0.0,
        }

    def build_v3_tracking_features(self) -> dict[str, float]:
        """The 7 V3-tracking features Exit-IQL V12.1.1 expects in its state vector.

        Computed as running stats over v3_v8_should_exit_prob across the trade's
        bars (per V12 Phase 4 spec):

          v3_should_exit_decision_v1  = (latest prob > 0.5)
          v3_decision_confidence_v1   = |latest prob - 0.5|
          v3_max_prob_in_trade_v1     = running max of prob since entry
          v3_consecutive_exits_v1     = current consecutive bars with prob > 0.5
          v3_signal_acceleration_v1   = Δ prob bar-to-bar
          v3_total_exit_decisions_v1  = count of bars with prob > 0.5
          v3_max_consecutive_exits_v1 = max run anywhere in trade
        """
        prob = self.v3_last_prob
        return {
            "v3_should_exit_decision_v1": 1.0 if prob > 0.5 else 0.0,
            "v3_decision_confidence_v1": float(abs(prob - 0.5)),
            "v3_max_prob_in_trade_v1": float(self.v3_max_prob_in_trade),
            "v3_consecutive_exits_v1": float(self.v3_consecutive_exits),
            "v3_signal_acceleration_v1": float(self.v3_signal_acceleration),
            "v3_total_exit_decisions_v1": float(self.v3_total_exit_decisions),
            "v3_max_consecutive_exits_v1": float(self.v3_max_consecutive_exits),
        }

    def build_v3_overlay(self) -> dict[str, np.ndarray]:
        """Build the 19-feature trade-state overlay for V3's in-trade portion of
        the 512-bar window, via the ONE-TRUTH helper
        (gx1.features.trade_overlay.compute_trade_overlay) — the SAME function the
        train builder (materialize_build_v3_training_dataset_v2.py) calls, so
        build==serve is bit-identical (V4/R13). Replaces the pre-V4 "MVP" overlay
        that approximated ~10 of the 19 slots (close-mark MFE instead of intrabar,
        1-based bars_held, wrong mfe_decay lag/giveback formula, slope=pnl/bars).

        Arrays have length = number of recorded in-trade bars (one per CLOSED M1
        bar, peak/trough/atr in lock-step with pnl_history). The consumer
        (v12_v3_live) RIGHT-ALIGNS them onto the END of the 512-bar window and
        uses the last min(len, 512) values.
        """
        n = len(self.pnl_history)
        if n == 0:
            # No closed bar yet — emit a single zero row (consumer right-aligns).
            return {name: np.zeros(1, dtype=np.float32) for name in OVERLAY_COL_NAMES}

        # peak/trough/atr are appended in lock-step with pnl_history; from_dict
        # backfills them for trades persisted before V4 so lengths always match.
        peak = np.fromiter(self.peak_history, dtype=np.float64, count=n)
        trough = np.fromiter(self.trough_history, dtype=np.float64, count=n)
        cur_pnl = np.fromiter(self.pnl_history, dtype=np.float64, count=n)
        atr = np.fromiter(self.atr_bps_history, dtype=np.float64, count=n)

        # Entry-snapshot (V10 direction softmax @entry, frozen). margin = top1-top2
        # gap of the 3-class probs — matches the candidate-gen
        # (materialize_inference_batch_candidates_v3_v1.py:403 sorted[-1]-sorted[-2])
        # and the train overlay entry-snap. NOT abs(p_long-p_short).
        s = self.v10_snapshot or {}
        dp = s.get("direction_probs", [0.0, 0.0, 0.0])
        p_long_e = float(dp[0] if hasattr(dp, "__len__") else 0.0)
        p_short_e = float(dp[1] if hasattr(dp, "__len__") and len(dp) > 1 else 0.0)
        p_flat_e = float(dp[2] if hasattr(dp, "__len__") and len(dp) > 2 else max(0.0, 1.0 - p_long_e - p_short_e))
        p_hat_e = max(p_long_e, p_short_e, p_flat_e)
        _sorted = sorted((p_long_e, p_short_e, p_flat_e))
        margin_e = _sorted[-1] - _sorted[-2]
        uncertainty_e = 1.0 - p_hat_e
        entropy_e = _shannon_entropy([p_long_e, p_short_e, p_flat_e])

        overlay = compute_trade_overlay(peak, trough, cur_pnl, atr, {
            "p_long_entry": p_long_e,
            "p_hat_entry": p_hat_e,
            "uncertainty_entry": uncertainty_e,
            "entropy_entry": entropy_e,
            "margin_entry": margin_e,
        })
        # Consumer maps by name → return one (n,) array per overlay column.
        return {name: overlay[:, i] for i, name in enumerate(OVERLAY_COL_NAMES)}

    # ── persistence ──────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize TradeState to a JSON-safe dict."""
        return {
            "entry_ts": self.entry_ts.isoformat(),
            "side": self.side,
            "entry_bid": self.entry_bid,
            "entry_ask": self.entry_ask,
            "entry_spread_bps": self.entry_spread_bps,
            "v10_snapshot": _jsonable(self.v10_snapshot),
            "trade_id": self.trade_id,
            "units": self.units,
            "bars_in_trade": self.bars_in_trade,
            "current_bid": self.current_bid,
            "current_ask": self.current_ask,
            "current_pnl_bps": self.current_pnl_bps,
            "cum_mfe_bps": self.cum_mfe_bps,
            "cum_mae_bps": self.cum_mae_bps,
            "bars_since_mfe_peak": self.bars_since_mfe_peak,
            "m1_returns_window": list(self.m1_returns_window),
            # V3 tracking stats
            "v3_last_prob": self.v3_last_prob,
            "v3_max_prob_in_trade": self.v3_max_prob_in_trade,
            "v3_consecutive_exits": self.v3_consecutive_exits,
            "v3_max_consecutive_exits": self.v3_max_consecutive_exits,
            "v3_total_exit_decisions": self.v3_total_exit_decisions,
            "v3_signal_acceleration": self.v3_signal_acceleration,
            # Per-bar trajectory
            "pnl_history": list(self.pnl_history),
            "mfe_at_bar": self.mfe_at_bar,
            "time_since_mfe_peak_bars": self.time_since_mfe_peak_bars,
            "last_atr_bps": self.last_atr_bps,
            # V4 (R13) intrabar excursion history — in lock-step with pnl_history.
            "peak_history": list(self.peak_history),
            "trough_history": list(self.trough_history),
            "atr_bps_history": list(self.atr_bps_history),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TradeState":
        """Rehydrate TradeState from a serialized dict (e.g. after restart)."""
        t = cls(
            entry_ts=pd.Timestamp(d["entry_ts"]),
            side=str(d["side"]),
            entry_bid=float(d["entry_bid"]),
            entry_ask=float(d["entry_ask"]),
            entry_spread_bps=float(d["entry_spread_bps"]),
            v10_snapshot=dict(d.get("v10_snapshot") or {}),
            trade_id=d.get("trade_id"),
            units=int(d.get("units", 1)),
            bars_in_trade=int(d.get("bars_in_trade", 0)),
            current_bid=float(d.get("current_bid", 0.0)),
            current_ask=float(d.get("current_ask", 0.0)),
            current_pnl_bps=float(d.get("current_pnl_bps", 0.0)),
            cum_mfe_bps=float(d.get("cum_mfe_bps", 0.0)),
            cum_mae_bps=float(d.get("cum_mae_bps", 0.0)),
            bars_since_mfe_peak=int(d.get("bars_since_mfe_peak", 0)),
            v3_last_prob=float(d.get("v3_last_prob", 0.0)),
            v3_max_prob_in_trade=float(d.get("v3_max_prob_in_trade", 0.0)),
            v3_consecutive_exits=int(d.get("v3_consecutive_exits", 0)),
            v3_max_consecutive_exits=int(d.get("v3_max_consecutive_exits", 0)),
            v3_total_exit_decisions=int(d.get("v3_total_exit_decisions", 0)),
            v3_signal_acceleration=float(d.get("v3_signal_acceleration", 0.0)),
            mfe_at_bar=float(d.get("mfe_at_bar", 0.0)),
            time_since_mfe_peak_bars=int(d.get("time_since_mfe_peak_bars", 0)),
            last_atr_bps=float(d.get("last_atr_bps", 0.0)),
        )
        for v in d.get("m1_returns_window") or []:
            t.m1_returns_window.append(float(v))
        for v in d.get("pnl_history") or []:
            t.pnl_history.append(float(v))
        # V4 (R13): intrabar excursion history. Trades persisted BEFORE V4 lack
        # these — backfill with a close-only degrade (peak=trough=pnl, atr=last_atr)
        # so build_v3_overlay stays length-aligned with pnl_history. Subsequent
        # update_bar calls then carry real intrabar values for the newest bars.
        peak_h = d.get("peak_history")
        trough_h = d.get("trough_history")
        atr_h = d.get("atr_bps_history")
        if peak_h is not None and trough_h is not None and atr_h is not None:
            for v in peak_h:
                t.peak_history.append(float(v))
            for v in trough_h:
                t.trough_history.append(float(v))
            for v in atr_h:
                t.atr_bps_history.append(float(v))
        else:
            for v in t.pnl_history:
                t.peak_history.append(float(v))
                t.trough_history.append(float(v))
                t.atr_bps_history.append(float(t.last_atr_bps))
        return t

    def save(self, path: Path) -> None:
        """Atomically write trade state to disk (so an interrupted write can't corrupt).

        `path` may be either a file path or a directory (when using multi-trade
        persistence). If a directory is given, the filename is derived from
        `trade_id` via `state_filename()`.
        """
        if path.is_dir():
            path = path / self.state_filename()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), default=str, indent=2))
        os.replace(tmp, path)

    def state_filename(self) -> str:
        """Filename for this trade's state file (one file per trade)."""
        tid = self.trade_id or f"virtual_{self.entry_ts.strftime('%Y%m%dT%H%M%S')}"
        return f"open_trade_{tid}.json"

    def delete_state_file(self, directory: Path) -> None:
        """Remove this trade's persisted state file from the directory."""
        (directory / self.state_filename()).unlink(missing_ok=True)

    @classmethod
    def load(cls, path: Path) -> "TradeState | None":
        """Load a saved trade state, or None if no file present."""
        if not path.is_file():
            return None
        try:
            return cls.from_dict(json.loads(path.read_text()))
        except Exception as exc:
            LOG.warning(f"failed to load trade state from {path}: {exc}")
            return None

    @classmethod
    def load_all(cls, directory: Path,
                 legacy_single_file: Path | None = None) -> list["TradeState"]:
        """Load all persisted open trades from `directory`.

        If `legacy_single_file` is given and exists, it's migrated into the
        directory (preserves a trade opened before the multi-trade refactor)
        and removed from its old location.

        Returns trades sorted by entry_ts ascending.
        """
        directory.mkdir(parents=True, exist_ok=True)
        trades: list[TradeState] = []

        if legacy_single_file is not None and legacy_single_file.is_file():
            t = cls.load(legacy_single_file)
            if t is not None:
                t.save(directory)
                legacy_single_file.unlink(missing_ok=True)
                LOG.info(f"migrated legacy trade state {legacy_single_file.name} "
                          f"→ {directory}/{t.state_filename()}")

        for p in sorted(directory.glob("open_trade_*.json")):
            t = cls.load(p)
            if t is not None:
                trades.append(t)
        trades.sort(key=lambda x: x.entry_ts)
        return trades


def _jsonable(o):
    """Recursively convert numpy/pandas/etc. to JSON-safe types."""
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return [_jsonable(x) for x in o.tolist()]
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(x) for x in o]
    return o


def _shannon_entropy(probs):
    s = 0.0
    for p in probs:
        if p > 1e-12:
            s -= p * float(np.log(p))
    return s
