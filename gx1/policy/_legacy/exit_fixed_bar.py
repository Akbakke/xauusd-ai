"\"\"\"Simple fixed/random bar exit policy for sanity checks.\"\"\""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from gx1.utils.pnl import compute_pnl_bps

log = logging.getLogger(__name__)


@dataclass
class FixedBarDecision:
    exit_price: float
    reason: str
    bars_held: int
    pnl_bps: float


class FixedBarExitPolicy:
    """Close trades after a fixed number of bars or a random range."""

    def __init__(self, mode: str = "fixed", bars: int = 3, min_bars: int = 1, max_bars: int = 6, seed: Optional[int] = None) -> None:
        self.mode = mode.lower()
        self.bars = max(1, bars)
        self.min_bars = max(1, min_bars)
        self.max_bars = max(self.min_bars, max_bars)
        self._rng = random.Random(seed)
        self._states: dict[str, dict[str, float | int]] = {}

    def reset_on_entry(self, entry_bid: float, entry_ask: float, trade_id: str, side: str = "long") -> None:
        target = self.bars if self.mode != "random" else self._rng.randint(self.min_bars, self.max_bars)
        self._states[trade_id] = {
            "entry_bid": float(entry_bid),
            "entry_ask": float(entry_ask),
            "target_bars": target,
            "bars_held": 0,
            "side": side.lower(),
        }
        log.debug("[EXIT_FIXED_BAR] trade_id=%s target_bars=%d", trade_id, target)

    def has_state(self, trade_id: str) -> bool:
        return trade_id in self._states

    def on_bar(self, trade_id: str, price_bid: float, price_ask: float, side: str = "long", ts: Optional[pd.Timestamp] = None) -> Optional[FixedBarDecision]:
        """
        Process a new bar and check if exit should trigger.
        
        Args:
            trade_id: Trade identifier
            price_bid: Current bar bid close price
            price_ask: Current bar ask close price
            side: Trade side ("long" or "short")
            ts: Current bar timestamp (optional, for logging/debugging)
        
        Returns:
            FixedBarDecision if exit triggered, None otherwise
        """
        state = self._states.get(trade_id)
        if state is None:
            raise RuntimeError(f"reset_on_entry must be called before on_bar for {trade_id}")
        state["bars_held"] += 1
        if state["bars_held"] >= state["target_bars"]:
            state_side = state.get("side", side).lower()
            # Calculate PnL using bid/ask prices
            pnl_bps = compute_pnl_bps(
                state["entry_bid"],
                state["entry_ask"],
                float(price_bid),
                float(price_ask),
                state_side,
            )
            # Exit price: use bid for LONG (we sell at bid), ask for SHORT (we buy back at ask)
            exit_price = float(price_bid if state_side == "long" else price_ask)
            self._states.pop(trade_id, None)
            log.debug(
                "[EXIT_FIXED_BAR] Exit triggered for trade %s: bars_held=%d target=%d pnl=%.2f bps "
                "(entry_bid=%.5f entry_ask=%.5f exit_bid=%.5f exit_ask=%.5f)",
                trade_id, state["bars_held"], state["target_bars"], pnl_bps,
                state["entry_bid"], state["entry_ask"], price_bid, price_ask
            )
            return FixedBarDecision(
                exit_price=exit_price,
                reason=f"FIXED_BARS_{int(state['target_bars'])}",
                bars_held=int(state["bars_held"]),
                pnl_bps=pnl_bps,
            )
        return None
