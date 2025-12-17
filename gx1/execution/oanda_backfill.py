"""
Robust backfill for M5 candles with target-driven paging.

This module provides deterministic backfill that pages backward until
target_bars are reached or no more data is available.
"""
import logging
import pandas as pd
from typing import Optional, Tuple
from datetime import datetime, timezone

log = logging.getLogger(__name__)


def backfill_m5_candles_until_target(
    oanda_client,
    instrument: str,
    granularity: str,
    target_bars: int,
    *,
    price: str = "M",  # "M" for mid, "BA" for bid/ask
    max_batch: int = 5000,
    max_iters: int = 10,
    min_new_per_iter: int = 5,
    now_utc: Optional[pd.Timestamp] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Fetch candles in batches, paging backward using `to=<earliest_time>` until
    we have target_bars or no more data is available.
    
    Parameters
    ----------
    oanda_client : OandaClient
        OANDA API client instance
    instrument : str
        Instrument symbol (e.g., "XAU_USD")
    granularity : str
        Candle granularity (e.g., "M5")
    target_bars : int
        Target number of bars to fetch
    price : str, default "M"
        Price type: "M" for mid OHLC, "BA" for bid/ask OHLC
    max_batch : int, default 5000
        Maximum candles per API call
    max_iters : int, default 10
        Maximum number of paging iterations
    min_new_per_iter : int, default 5
        Minimum new candles per iteration to continue paging
    now_utc : pd.Timestamp, optional
        Current UTC time (defaults to now)
    logger : logging.Logger, optional
        Logger instance (defaults to module logger)
    
    Returns
    -------
    Tuple[pd.DataFrame, dict]
        DataFrame with candles (sorted ascending by time, no duplicates),
        and metadata dict with:
        - total_bars: int
        - iterations: int
        - stop_reason: str
        - earliest_time: pd.Timestamp
        - latest_time: pd.Timestamp
    """
    if logger is None:
        logger = log
    
    if now_utc is None:
        now_utc = pd.Timestamp.now(tz="UTC")
    
    all_candles = []
    earliest_time = None
    latest_time = None
    iterations = 0
    stop_reason = "max_iters"
    
    logger.info(
        "[BACKFILL] Starting target-driven backfill: target=%d bars, price=%s, max_batch=%d, max_iters=%d",
        target_bars, price, max_batch, max_iters
    )
    
    # First iteration: fetch latest candles (without `to` parameter)
    try:
        logger.info("[BACKFILL] Iteration 1: fetching latest %d candles", max_batch)
        candles_df = oanda_client.get_candles(
            instrument=instrument,
            granularity=granularity,
            count=max_batch,
            include_mid=(price == "M"),
        )
        
        if candles_df.empty:
            logger.warning("[BACKFILL] No candles returned in first iteration")
            return pd.DataFrame(), {
                "total_bars": 0,
                "iterations": 0,
                "stop_reason": "no_data_first_iter",
                "earliest_time": None,
                "latest_time": None,
            }
        
        all_candles.append(candles_df)
        earliest_time = candles_df.index.min()
        latest_time = candles_df.index.max()
        iterations = 1
        
        logger.info(
            "[BACKFILL] Iteration 1: fetched %d candles, earliest=%s, latest=%s, total=%d",
            len(candles_df), earliest_time.isoformat(), latest_time.isoformat(), len(candles_df)
        )
        
        # Check if we already have enough
        if len(candles_df) >= target_bars:
            logger.info("[BACKFILL] Target reached in first iteration: %d >= %d", len(candles_df), target_bars)
            stop_reason = "target_reached"
            combined_df = candles_df.sort_index().drop_duplicates(keep="last")
            return combined_df, {
                "total_bars": len(combined_df),
                "iterations": iterations,
                "stop_reason": stop_reason,
                "earliest_time": earliest_time,
                "latest_time": latest_time,
            }
    
    except Exception as e:
        logger.error("[BACKFILL] Failed in first iteration: %s", e, exc_info=True)
        return pd.DataFrame(), {
            "total_bars": 0,
            "iterations": 0,
            "stop_reason": "error_first_iter",
            "earliest_time": None,
            "latest_time": None,
        }
    
    # Subsequent iterations: page backward
    for iter_num in range(2, max_iters + 1):
        if len(all_candles) > 0:
            # Combine all candles so far to find earliest time
            combined_df = pd.concat(all_candles).sort_index().drop_duplicates(keep="last")
            earliest_time = combined_df.index.min()
            current_total = len(combined_df)
            
            if current_total >= target_bars:
                logger.info(
                    "[BACKFILL] Target reached after iteration %d: %d >= %d",
                    iter_num - 1, current_total, target_bars
                )
                stop_reason = "target_reached"
                return combined_df, {
                    "total_bars": current_total,
                    "iterations": iter_num - 1,
                    "stop_reason": stop_reason,
                    "earliest_time": earliest_time,
                    "latest_time": latest_time,
                }
        
        # Calculate `to` time: earliest_time minus 1 second to avoid overlap
        to_time = earliest_time - pd.Timedelta(seconds=1)
        
        try:
            logger.info(
                "[BACKFILL] Iteration %d: fetching up to %s (to=%s), count=%d",
                iter_num, earliest_time.isoformat(), to_time.isoformat(), max_batch
            )
            
            # Calculate from_ts: go back max_batch bars from to_time
            from_ts_calc = to_time - pd.Timedelta(minutes=5 * max_batch)
            
            candles_df = oanda_client.get_candles(
                instrument=instrument,
                granularity=granularity,
                from_ts=from_ts_calc,
                to_ts=to_time,
                include_mid=(price == "M"),
            )
            
            if candles_df.empty:
                logger.info("[BACKFILL] Iteration %d: no more candles available", iter_num)
                stop_reason = "no_more_data"
                break
            
            # Check how many new candles we got
            if len(all_candles) > 0:
                combined_before = pd.concat(all_candles).sort_index().drop_duplicates(keep="last")
                combined_after = pd.concat(all_candles + [candles_df]).sort_index().drop_duplicates(keep="last")
                new_rows = len(combined_after) - len(combined_before)
            else:
                new_rows = len(candles_df)
            
            logger.info(
                "[BACKFILL] Iteration %d: fetched %d candles, %d new, earliest=%s, total=%d",
                iter_num, len(candles_df), new_rows,
                candles_df.index.min().isoformat() if not candles_df.empty else "N/A",
                len(combined_after) if len(all_candles) > 0 else len(candles_df)
            )
            
            # Check if we got enough new data
            if new_rows < min_new_per_iter:
                logger.info(
                    "[BACKFILL] Iteration %d: insufficient new candles (%d < %d), stopping",
                    iter_num, new_rows, min_new_per_iter
                )
                stop_reason = "insufficient_new_data"
                break
            
            all_candles.append(candles_df)
            earliest_time = candles_df.index.min()
            iterations = iter_num
            
            # Check if we have enough total
            combined_df = pd.concat(all_candles).sort_index().drop_duplicates(keep="last")
            if len(combined_df) >= target_bars:
                logger.info(
                    "[BACKFILL] Target reached after iteration %d: %d >= %d",
                    iter_num, len(combined_df), target_bars
                )
                stop_reason = "target_reached"
                latest_time = combined_df.index.max()
                return combined_df, {
                    "total_bars": len(combined_df),
                    "iterations": iterations,
                    "stop_reason": stop_reason,
                    "earliest_time": earliest_time,
                    "latest_time": latest_time,
                }
        
        except Exception as e:
            logger.warning("[BACKFILL] Iteration %d failed: %s", iter_num, e)
            stop_reason = f"error_iter_{iter_num}"
            break
    
    # Combine all candles
    if len(all_candles) == 0:
        return pd.DataFrame(), {
            "total_bars": 0,
            "iterations": 0,
            "stop_reason": stop_reason,
            "earliest_time": None,
            "latest_time": None,
        }
    
    combined_df = pd.concat(all_candles).sort_index().drop_duplicates(keep="last")
    latest_time = combined_df.index.max()
    
    logger.info(
        "[BACKFILL] Backfill complete: total=%d bars, iterations=%d, stop_reason=%s, earliest=%s, latest=%s",
        len(combined_df), iterations, stop_reason,
        earliest_time.isoformat() if earliest_time is not None else "N/A",
        latest_time.isoformat() if latest_time is not None else "N/A"
    )
    
    return combined_df, {
        "total_bars": len(combined_df),
        "iterations": iterations,
        "stop_reason": stop_reason,
        "earliest_time": earliest_time,
        "latest_time": latest_time,
    }

