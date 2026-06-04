"""
Price-scale glitch guard (data-quality, offline build/eval).

Why this exists
---------------
2026-06-03 diagnosis: the canonical_v3 M5 build had a x10 price-DEFLATION glitch
confined to 2026-04-02 00:01 .. 2026-04-19 23:55 (3156 bars, close ~457-488 instead
of ~4700), bracketed by a x0.1 entry snap and a x10 exit snap. Four longs that entered
at the deflated ~475 and exited after the x10 snap-back realized ~90,000 bps each —
inflating the Phase 6 OOT headline from an honest ~28 bps/take to a fake +136 bps/take
(those 4 takes were ~80% of total backtest PnL). The live OANDA tape was unaffected;
this was purely an offline historical-data corruption that no guard caught.

This module is a single, dependency-light, fail-closed check to run wherever an offline
price series is loaded for forward-outcome / PnL computation. A legitimate gold M5 bar
never moves >300% bar-to-bar (even weekend gaps are ~1-2%), so a consecutive-bar ratio
outside [1/max_ratio, max_ratio] (default 3.0) is a feed/scale glitch, not a real move.

Contract
--------
- No file I/O. Pure function over an in-memory frame.
- `assert_no_price_scale_glitch` raises RuntimeError (fail-closed) by default.
- `detect_price_scale_glitch` is the no-throw core returning the offending rows.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Bars whose consecutive close-ratio falls outside [1/MAX, MAX] are scale glitches.
# 3.0 = 300% single-bar move; gold M5 never does this legitimately (flash-crashes are
# <10%, weekend gaps <2%). Kept well below the observed x10 glitch for margin.
DEFAULT_MAX_BAR_RATIO = 3.0

# Default price columns to validate on a bid/ask M5 tape. Close-side prices drive PnL.
DEFAULT_PRICE_COLS = ("bid_close", "ask_close")


def detect_price_scale_glitch(
    df: pd.DataFrame,
    price_cols: Sequence[str] = DEFAULT_PRICE_COLS,
    max_bar_ratio: float = DEFAULT_MAX_BAR_RATIO,
    time_col: str = "time",
) -> pd.DataFrame:
    """Return the rows whose price jumps >max_bar_ratio (or <1/max_bar_ratio) vs the
    previous bar in ANY of `price_cols`. Empty frame => clean. No exceptions."""
    cols = [c for c in price_cols if c in df.columns]
    if not cols or len(df) < 2:
        return df.iloc[0:0]
    lo, hi = 1.0 / max_bar_ratio, max_bar_ratio
    flagged = np.zeros(len(df), dtype=bool)
    worst_ratio = np.ones(len(df), dtype=np.float64)
    worst_col = np.array([""] * len(df), dtype=object)
    for c in cols:
        p = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64)
        prev = np.empty_like(p)
        prev[0] = np.nan
        prev[1:] = p[:-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = p / prev
        bad = np.isfinite(ratio) & (prev > 0) & ((ratio > hi) | (ratio < lo))
        # track the most extreme deviation per row for reporting
        dev = np.where(np.isfinite(ratio), np.abs(np.log(np.where(ratio > 0, ratio, np.nan))), 0.0)
        take = bad & (dev > np.abs(np.log(np.where(worst_ratio > 0, worst_ratio, 1.0))))
        worst_ratio = np.where(take, ratio, worst_ratio)
        worst_col = np.where(take, c, worst_col)
        flagged |= bad
    if not flagged.any():
        return df.iloc[0:0]
    out_cols = ([time_col] if time_col in df.columns else []) + cols
    out = df.loc[flagged, out_cols].copy()
    out["_glitch_bar_ratio"] = worst_ratio[flagged]
    out["_glitch_col"] = worst_col[flagged]
    return out


def assert_no_price_scale_glitch(
    df: pd.DataFrame,
    price_cols: Sequence[str] = DEFAULT_PRICE_COLS,
    max_bar_ratio: float = DEFAULT_MAX_BAR_RATIO,
    time_col: str = "time",
    context: str = "",
    on_violation: str = "raise",
) -> pd.DataFrame:
    """Fail-closed price-scale glitch guard for OFFLINE build/eval data loads.

    on_violation="raise" (default): RuntimeError listing the offending window —
        use in offline candidate-gen / forward-outcome / Phase-6 builders so a
        corrupt historical segment can NEVER silently inflate PnL again.
    on_violation="warn": log only and return the offending rows — use on the LIVE
        incremental daemon, where a hard raise must never kill data collection.

    Returns the offending rows (empty frame if clean).
    """
    bad = detect_price_scale_glitch(df, price_cols, max_bar_ratio, time_col)
    if len(bad) == 0:
        return bad
    span = ""
    if time_col in bad.columns:
        span = f" spanning {bad[time_col].min()} .. {bad[time_col].max()}"
    head = bad.head(8).to_string(index=False)
    msg = (
        f"[PRICE_SCALE_GLITCH]{(' ' + context) if context else ''}: "
        f"{len(bad)} bar(s) jump >{max_bar_ratio}x (or <{1/max_bar_ratio:.3f}x) vs the "
        f"previous bar{span} — feed/scale corruption, not a real move. "
        f"First rows:\n{head}"
    )
    if on_violation == "warn":
        log.warning(msg)
        return bad
    raise RuntimeError(msg)


if __name__ == "__main__":  # tiny self-test
    n = 200
    t = pd.date_range("2026-04-01", periods=n, freq="5min", tz="UTC")
    close = pd.Series(4700.0 + np.arange(n) * 0.1)
    df_ok = pd.DataFrame({"time": t, "bid_close": close, "ask_close": close + 0.3})
    assert len(detect_price_scale_glitch(df_ok)) == 0, "clean series flagged"
    # inject a x10 deflation window (entry x0.1 snap, exit x10 snap) like the real glitch
    bad = df_ok.copy()
    bad.loc[50:69, ["bid_close", "ask_close"]] /= 10.0
    flags = detect_price_scale_glitch(bad)
    assert len(flags) == 2, f"expected 2 boundary glitches, got {len(flags)}"
    try:
        assert_no_price_scale_glitch(bad, context="selftest")
        raise SystemExit("FAIL: did not raise on glitch")
    except RuntimeError:
        pass
    print("price_glitch_guard self-test OK")
