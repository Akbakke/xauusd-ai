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

    @classmethod
    def open(
        cls,
        entry_ts: pd.Timestamp,
        side: str,
        entry_bid: float,
        entry_ask: float,
        v10_snapshot: dict[str, Any] | None = None,
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

    def update_bar(self, bid: float, ask: float, m1_close: float) -> None:
        """Advance trade state by one M1 bar."""
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
        """M1 close return over the last n bars (in bps), or 0 if not enough data."""
        if len(self.m1_returns_window) < 2:
            return 0.0
        lookback = min(n, len(self.m1_returns_window) - 1)
        prev = self.m1_returns_window[-(lookback + 1)] if lookback + 1 <= len(self.m1_returns_window) else self.m1_returns_window[0]
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
            # V3 v8 frozen at entry (would be from V3 inference at entry bar)
            "p_long_entry_v1": p_long_e,
            "p_hat_entry_v1": float(max(p_long_e, p_short_e, 1.0 - p_long_e - p_short_e)),
            "uncertainty_entry_v1": float(1.0 - max(p_long_e, p_short_e, 1.0 - p_long_e - p_short_e)),
            "margin_entry_v1": float(abs(p_long_e - p_short_e)),
            # entropy
            "entropy_entry_v1": float(self.v10_snapshot.get("entropy_at_entry",
                _shannon_entropy([p_long_e, p_short_e, max(0.0, 1.0 - p_long_e - p_short_e)]))),
            # rolling slope since entry (approximated via simple regression on m1 returns)
            "rolling_slope_since_entry_v1": float(self._rolling_return_bps(self.bars_in_trade) / max(1, self.bars_in_trade)),
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
        """Build the 19-feature trade-state overlay matrix for V3 v8's in-trade
        portion of the 512-bar window.

        Returned arrays have length = bars_in_trade. They map to the END of V3's
        512-bar window (the most recent bars are in-trade).

        Note: this is the simplified MVP overlay — fills each in-trade bar with
        approximations rather than the precise per-bar history that training
        used. For full per-bar trajectory we'd need to record every bar's stats
        (pnl_history captures pnl but not derivatives). Most of the 19 features
        are slow-moving so the approximation is reasonable.
        """
        n = max(1, self.bars_in_trade)
        # Trade-bar arrays — most of these are "current state" broadcast back to all
        # in-trade bars. mfe_decay_rate, pnl_velocity, pnl_acceleration computed
        # from pnl_history when available.
        pnl_arr = np.array(list(self.pnl_history), dtype=np.float32) if self.pnl_history else np.zeros(n, dtype=np.float32)
        if len(pnl_arr) < n:
            pnl_arr = np.pad(pnl_arr, (n - len(pnl_arr), 0))
        pnl_arr = pnl_arr[-n:]

        pnl_velocity = np.zeros(n, dtype=np.float32)
        pnl_velocity[1:] = np.diff(pnl_arr)
        pnl_acceleration = np.zeros(n, dtype=np.float32)
        pnl_acceleration[1:] = np.diff(pnl_velocity)

        # Cumulative MFE per bar (approximation: monotone running max)
        mfe_arr = np.maximum.accumulate(pnl_arr) if len(pnl_arr) > 0 else np.zeros(n, dtype=np.float32)
        mae_arr = np.minimum.accumulate(pnl_arr) if len(pnl_arr) > 0 else np.zeros(n, dtype=np.float32)
        dd_from_mfe = mfe_arr - pnl_arr
        time_since_mfe = np.arange(n, dtype=np.float32)   # simplified — bar index
        bars_held = np.arange(1, n + 1, dtype=np.float32)
        atr_arr = np.full(n, self.last_atr_bps, dtype=np.float32)
        # giveback ratio: drawdown_from_peak / peak (clipped)
        with np.errstate(divide="ignore", invalid="ignore"):
            giveback_ratio = np.where(mfe_arr > 0, dd_from_mfe / np.maximum(mfe_arr, 1e-6), 0.0)
            giveback_ratio = np.clip(giveback_ratio, 0.0, 2.0)
        giveback_accel = np.zeros(n, dtype=np.float32)
        giveback_accel[1:] = np.diff(giveback_ratio)
        # mfe_decay_rate: rate of mfe decay (negative slope of mfe over recent bars)
        mfe_decay = np.zeros(n, dtype=np.float32)
        if n >= 5:
            mfe_decay[5:] = (mfe_arr[5:] - mfe_arr[:-5]) / 5.0
        # rolling_slope_since_entry (approximation: pnl_arr[-1]/bars_held)
        rolling_slope = pnl_arr / np.maximum(bars_held, 1.0)

        # Entry-snapshot features are CONSTANT across all in-trade bars
        s = self.v10_snapshot or {}
        dp = s.get("direction_probs", [0.0, 0.0, 0.0])
        p_long_e = float(dp[0] if hasattr(dp, "__len__") else 0.0)
        p_short_e = float(dp[1] if hasattr(dp, "__len__") and len(dp) > 1 else 0.0)
        p_flat_e = float(dp[2] if hasattr(dp, "__len__") and len(dp) > 2 else max(0.0, 1.0 - p_long_e - p_short_e))
        p_hat_e = max(p_long_e, p_short_e, p_flat_e)
        margin_e = abs(p_long_e - p_short_e)
        uncertainty_e = 1.0 - p_hat_e
        entropy_e = _shannon_entropy([p_long_e, p_short_e, p_flat_e])

        return {
            # Entry-snapshot (constant across in-trade window)
            "p_long_entry": np.full(n, p_long_e, dtype=np.float32),
            "p_hat_entry": np.full(n, p_hat_e, dtype=np.float32),
            "uncertainty_entry": np.full(n, uncertainty_e, dtype=np.float32),
            "entropy_entry": np.full(n, entropy_e, dtype=np.float32),
            "margin_entry": np.full(n, margin_e, dtype=np.float32),
            # Per-bar trade-state
            "pnl_bps_now": pnl_arr,
            "mfe_bps": mfe_arr,
            "mae_bps": mae_arr,
            "dd_from_mfe_bps": dd_from_mfe.astype(np.float32),
            "distance_from_peak_mfe_bps": dd_from_mfe.astype(np.float32),  # synonymous
            "bars_held": bars_held,
            "time_since_mfe_bars": time_since_mfe,
            "mfe_decay_rate": mfe_decay,
            "pnl_velocity": pnl_velocity,
            "pnl_acceleration": pnl_acceleration,
            "rolling_slope_since_entry": rolling_slope.astype(np.float32),
            "atr_bps_now": atr_arr,
            "giveback_ratio": giveback_ratio.astype(np.float32),
            "giveback_acceleration": giveback_accel,
        }

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
        return t

    def save(self, path: Path) -> None:
        """Atomically write trade state to disk (so an interrupted write can't corrupt)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), default=str, indent=2))
        os.replace(tmp, path)

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
