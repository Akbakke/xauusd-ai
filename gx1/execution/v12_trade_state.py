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
        else:
            self.bars_since_mfe_peak += 1

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
        )
        for v in d.get("m1_returns_window") or []:
            t.m1_returns_window.append(float(v))
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
