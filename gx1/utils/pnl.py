"""
PnL utility helpers shared across execution and policy modules.
"""

from __future__ import annotations

from typing import Literal


def compute_pnl_bps(
    entry_bid: float,
    entry_ask: float,
    exit_bid: float,
    exit_ask: float,
    side: Literal["long", "short"],
) -> float:
    """
    Compute trade PnL in basis points using realistic bid/ask fill prices.
    
    NO FALLBACK TO MID - bid/ask are REQUIRED.
    If bid/ask are missing, this function will raise ValueError.
    
    Fill logic:
    - LONG: entry = ask (buy at ask), exit = bid (sell at bid)
    - SHORT: entry = bid (sell at bid), exit = ask (buy at ask)

    Args:
        entry_bid: Entry bid price (REQUIRED).
        entry_ask: Entry ask price (REQUIRED).
        exit_bid: Exit bid price (REQUIRED).
        exit_ask: Exit ask price (REQUIRED).
        side: "long" or "short".

    Returns:
        Signed PnL in basis points.

    Raises:
        ValueError: If any bid/ask price is missing or invalid.
    """
    # Validate all prices are provided and positive
    if entry_bid <= 0:
        raise ValueError(f"entry_bid must be > 0, got {entry_bid}")
    if entry_ask <= 0:
        raise ValueError(f"entry_ask must be > 0, got {entry_ask}")
    if exit_bid <= 0:
        raise ValueError(f"exit_bid must be > 0, got {exit_bid}")
    if exit_ask <= 0:
        raise ValueError(f"exit_ask must be > 0, got {exit_ask}")
    
    # Validate ask >= bid (spread should be non-negative)
    if entry_ask < entry_bid:
        raise ValueError(f"entry_ask ({entry_ask}) must be >= entry_bid ({entry_bid})")
    if exit_ask < exit_bid:
        raise ValueError(f"exit_ask ({exit_ask}) must be >= exit_bid ({exit_bid})")
    
    # Spread-aware calculation
    if side.lower() == "long":
        # LONG: buy at ask, sell at bid
        actual_entry = entry_ask
        actual_exit = exit_bid
    elif side.lower() == "short":
        # SHORT: sell at bid, buy at ask
        actual_entry = entry_bid
        actual_exit = exit_ask
    else:
        raise ValueError(f"side must be 'long' or 'short', got {side}")
    
    pnl_bps = (actual_exit - actual_entry) / actual_entry * 10000.0
    return float(pnl_bps)
