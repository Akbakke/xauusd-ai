"""
Excursion metrics from OHLC for trade_outcomes TRUTH SSoT.

Computes MAE, MFE, time-to-MAE/MFE, close-to-MFE, exit efficiency,
and post-exit 12-bar MFE/MAE from candles between entry and exit.
Long-only for now.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def compute_excursion_metrics(
    entry_time: pd.Timestamp,
    exit_time: pd.Timestamp,
    entry_price: float,
    exit_price: float,
    candles: pd.DataFrame,
    side: str = "long",
    post_exit_bars: int = 12,
) -> Dict[str, Optional[float]]:
    """
    Compute excursion metrics from OHLC between entry and exit.

    Args:
        entry_time: Trade entry timestamp
        exit_time: Trade exit timestamp
        entry_price: Entry price (bid for long, ask for short)
        exit_price: Exit price (bid for long, ask for short)
        candles: DataFrame with DatetimeIndex and OHLC columns
        side: "long" or "short" (long-only fully implemented)
        post_exit_bars: Number of bars after exit for post_exit_* metrics

    Returns:
        Dict with: mae_bps, mfe_bps, time_to_mae_bars, time_to_mfe_bars,
        close_to_mfe_bps, exit_efficiency, post_exit_mfe_12b_bps, post_exit_mae_12b_bps
    """
    # Use 0.0/0 for defaults to avoid NaN in parquet (TRUTH hash requires no NaNs)
    out: Dict[str, Optional[float]] = {
        "mae_bps": 0.0,
        "mfe_bps": 0.0,
        "time_to_mae_bars": 0,
        "time_to_mfe_bars": 0,
        "close_to_mfe_bps": 0.0,
        "exit_efficiency": 1.0,
        "post_exit_mfe_12b_bps": 0.0,
        "post_exit_mae_12b_bps": 0.0,
    }
    if candles is None or len(candles) == 0:
        return out

    # Ensure index is timezone-aware for comparison
    idx = candles.index
    if hasattr(idx, "tz") and idx.tz is None and entry_time.tzinfo:
        try:
            candles = candles.copy()
            candles.index = pd.to_datetime(candles.index, utc=True)
        except Exception:
            pass

    trade_bars = candles[(candles.index >= entry_time) & (candles.index <= exit_time)]
    if len(trade_bars) == 0:
        return out  # Already has 0.0 defaults

    side_lower = (side or "long").lower()
    ep = float(entry_price)
    xp = float(exit_price)

    # Resolve OHLC columns (support bid_high/bid_low for long, ask_high/ask_low for short)
    if side_lower == "long":
        high_col = "bid_high" if "bid_high" in trade_bars.columns else "high"
        low_col = "bid_low" if "bid_low" in trade_bars.columns else "low"
        close_col = "bid_close" if "bid_close" in trade_bars.columns else "close"
    else:
        high_col = "ask_high" if "ask_high" in trade_bars.columns else "high"
        low_col = "ask_low" if "ask_low" in trade_bars.columns else "low"
        close_col = "ask_close" if "ask_close" in trade_bars.columns else "close"

    if high_col not in trade_bars.columns or low_col not in trade_bars.columns:
        return out

    highs = np.asarray(trade_bars[high_col].values, dtype=float)
    lows = np.asarray(trade_bars[low_col].values, dtype=float)

    # PnL in bps: (price - entry) / entry * 10000 for long
    if side_lower == "long":
        bar_highs_pnl = ((highs - ep) / ep) * 10000.0
        bar_lows_pnl = ((lows - ep) / ep) * 10000.0
    else:
        bar_highs_pnl = ((ep - highs) / ep) * 10000.0
        bar_lows_pnl = ((ep - lows) / ep) * 10000.0

    mfe_bps = float(np.max(bar_highs_pnl))
    mae_raw = float(np.min(bar_lows_pnl))
    mae_bps = abs(mae_raw)  # Positive magnitude

    # Time to first MAE/MFE (bar index, 0-based)
    time_to_mfe_bars = int(np.argmax(bar_highs_pnl))
    time_to_mae_bars = int(np.argmin(bar_lows_pnl))

    out["mae_bps"] = mae_bps
    out["mfe_bps"] = mfe_bps
    out["time_to_mae_bars"] = time_to_mae_bars
    out["time_to_mfe_bars"] = time_to_mfe_bars

    # Realized PnL at exit (bps)
    if side_lower == "long":
        pnl_bps = ((xp - ep) / ep) * 10000.0
    else:
        pnl_bps = ((ep - xp) / ep) * 10000.0

    # close_to_mfe_bps: amount "given back" from peak (mfe - pnl when pnl < mfe)
    if mfe_bps > 0 and pnl_bps < mfe_bps:
        out["close_to_mfe_bps"] = float(mfe_bps - pnl_bps)
    else:
        out["close_to_mfe_bps"] = 0.0

    # exit_efficiency: pnl / mfe (capture ratio). If mfe<=0, use 1.0
    if mfe_bps > 0:
        out["exit_efficiency"] = float(pnl_bps / mfe_bps)
    else:
        out["exit_efficiency"] = 1.0

    # Post-exit 12 bars: candles AFTER exit_time (use 0.0 when no bars to avoid NaN in parquet/hash)
    post_start = exit_time
    post_candles = candles[candles.index > post_start].head(post_exit_bars)
    if len(post_candles) > 0 and close_col in post_candles.columns:
        # Use exit price as reference for post-exit PnL (long: how much did price move after we exited)
        closes = np.asarray(post_candles[close_col].values, dtype=float)
        if side_lower == "long":
            post_pnl = ((closes - xp) / xp) * 10000.0
        else:
            post_pnl = ((xp - closes) / xp) * 10000.0
        out["post_exit_mfe_12b_bps"] = float(np.max(post_pnl))
        out["post_exit_mae_12b_bps"] = float(np.min(post_pnl))
    else:
        out["post_exit_mfe_12b_bps"] = 0.0
        out["post_exit_mae_12b_bps"] = 0.0

    return out
