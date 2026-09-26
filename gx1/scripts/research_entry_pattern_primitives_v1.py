#!/usr/bin/env python3
"""Tape-based chart / SMC pattern event primitives for the Entry direction research instrument.

Research only, no authority.  Operator question 2026-09-23: *"mange får jo til
å lese momentum, chart pattern, bullish flag, fair value gap, support and
resistance — hvorfor gjør ikke vi det?"*  This module materialises the
primitives the production surface does NOT carry (indicator fidelity audit
2026-08-13: fair value gaps, order blocks, equal-high/low pools ABSENT;
session-anchored levels designed but never admitted) plus flag and N-bar
range breakouts, on every closed timeframe, so the walk-forward instrument
and the setup-edge evaluator can measure them out of sample.

Why a new module (rule 21): no production owner exists for these concepts,
and the level/trendline registries are production contracts with
TRAIN-fitted tolerances that a research surface must not silently extend.
Every threshold below is an explicit CLI input recorded in the manifest
(rule 2a origin class 3); the swing lookback is the SMC owner's named
constant; ATR and EMA are the owners' functions; the multi-timeframe bars use
the owner's resampler and origin offsets; last-closed sampling follows
``htf_features.multi_tf_last_closed_label``.

Causality: every value at a decision row uses only bars closed by the
decision time (row start + 5 min).  Zones are one-shot: a fair value gap or
order block is removed at its first touch (retest or failure); an equal-high
pool is removed at its sweep or break.  Distances are in ATR units and are
capped at ``DISTANCE_CAP_ATR``; ages are capped at ``--age-cap-bars``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from gx1.features.htf_features import (
    MULTI_TF_RESAMPLE_RULES,
    MULTI_TF_SHIFT,
    _resample_ohlcv,
)
from gx1.features.smc_v1 import SWING_LOOKBACK, _detect_swing_pivots
from gx1.features.technical_indicators_v1 import classic_ema, wilder_atr
from gx1.time.session_detector import (
    SESSION_BOUNDARIES,
    TRADING_SESSION_BOUNDARY_OFFSET,
    TRADING_SESSION_DURATION,
    trading_session_id_vectorized,
)

SCHEMA_VERSION = "entry_pattern_primitives_research_v1"
AUTHORITY = {"research_only": True, "candidate": False, "promotion": False, "test": False, "paper": False, "live": False}
TIMEFRAMES = ("M5", "H1", "H4", "D1")
M5_BAR = pd.Timedelta(minutes=5)
ATR_PERIOD = 14
DISTANCE_CAP_ATR = 20.0
EMA_SPANS = (20, 50, 200)
EMA_SLOW_SLOPE_LOOKBACK_BARS = 20  # the repository's existing slow-span slope convention (2026-09-21 wave)
BPS = 1e4


@dataclass(frozen=True)
class Params:
    swing_lookback: int
    zone_lookback_bars: int
    fvg_min_gap_atr: float
    ob_displacement_atr: float
    ob_displacement_bars: int
    ob_search_bars: int
    eq_tolerance_atr: float
    flag_impulse_bars: int
    flag_impulse_atr: float
    flag_consolidation_min_bars: int
    flag_consolidation_max_bars: int
    flag_consolidation_atr: float
    range_breakout_bars: int
    age_cap_bars: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_tape_ohlcv(root: Path, *, truncate_before: pd.Timestamp) -> pd.DataFrame:
    truncate_before = pd.Timestamp(truncate_before)
    if truncate_before.tzinfo is None:
        raise RuntimeError("PATTERN_TAPE_BOUNDARY_NOT_UTC_AWARE")
    truncate_before = truncate_before.tz_convert("UTC")
    files = sorted(p for p in root.glob("year=*/*.parquet") if int(p.parent.name.split("=", 1)[1]) <= truncate_before.year)
    if not files or not (root / "MANIFEST.json").is_file():
        raise RuntimeError("PATTERN_TAPE_MISSING")
    cols = ["time", "open", "high", "low", "close", "volume", "bid_close", "ask_close"]
    frame = pd.concat([
        pq.read_table(str(f), columns=cols, filters=[("time", "<", truncate_before.to_pydatetime())]).to_pandas()
        for f in files
    ], ignore_index=True)
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    frame = frame.sort_values("time", kind="mergesort").reset_index(drop=True)
    if frame.empty or not (frame["time"] < truncate_before).all():
        raise RuntimeError("PATTERN_TAPE_READ_BOUNDARY_INVALID")
    if frame["time"].duplicated().any():
        raise RuntimeError("PATTERN_TAPE_DUPLICATE_TIME")
    for name in ("open", "high", "low", "close"):
        if not np.isfinite(frame[name].to_numpy(np.float64)).all():
            raise RuntimeError(f"PATTERN_TAPE_NONFINITE: {name}")
    return frame


def closed_bars(tape: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Closed OHLCV bars on the owner's grid (M5 = the tape itself)."""
    if timeframe == "M5":
        return tape.set_index("time")[["open", "high", "low", "close", "volume"]].copy()
    if timeframe not in MULTI_TF_RESAMPLE_RULES:
        raise RuntimeError(f"PATTERN_TIMEFRAME_INVALID: {timeframe}")
    bars = _resample_ohlcv(tape.set_index("time"), timeframe)
    bars = bars.dropna(subset=["open", "high", "low", "close"])
    return bars


def _cap_age(age: np.ndarray, cap: int) -> np.ndarray:
    out = np.asarray(age, dtype=np.float64)
    out[out < 0] = float(cap)
    return np.minimum(out, float(cap))


def _bars_since(event: np.ndarray, cap: int) -> np.ndarray:
    out = np.full(len(event), -1, dtype=np.int64)
    last = -1
    for i in range(len(event)):
        if event[i]:
            last = i
        out[i] = -1 if last < 0 else i - last
    return _cap_age(out, cap)


def track_zones(
    *, bottom_new: np.ndarray, top_new: np.ndarray, event: np.ndarray, low: np.ndarray, high: np.ndarray,
    close: np.ndarray, atr: np.ndarray, side: str, lookback: int, age_cap: int,
) -> dict[str, np.ndarray]:
    """One-shot zone tracker for bullish (below price) or bearish (above price) zones.

    A bullish zone is touched when ``low <= top``; the touch is a retest hold if
    ``close >= bottom`` and a failure otherwise.  Bearish zones mirror this.
    Zones expire after ``lookback`` bars.  Returns active count, signed
    distance to the nearest active zone edge (ATR, capped; +cap when none),
    retest / failure events and bars since the last retest hold.
    """
    n = len(close)
    active: list[tuple[float, float, int]] = []
    active_count = np.zeros(n, dtype=np.float64)
    nearest = np.full(n, DISTANCE_CAP_ATR, dtype=np.float64)
    retest = np.zeros(n, dtype=bool)
    failure = np.zeros(n, dtype=bool)
    for j in range(n):
        keep: list[tuple[float, float, int]] = []
        for bottom, top, birth in active:
            if j - birth > lookback:
                continue
            if side == "bull":
                touched = low[j] <= top
                held = close[j] >= bottom
            else:
                touched = high[j] >= bottom
                held = close[j] <= top
            if touched:
                if held:
                    retest[j] = True
                else:
                    failure[j] = True
                continue
            keep.append((bottom, top, birth))
        active = keep
        if event[j]:
            active.append((float(bottom_new[j]), float(top_new[j]), j))
        active_count[j] = float(len(active))
        if active and np.isfinite(atr[j]) and atr[j] > 0:
            if side == "bull":
                dist = min((close[j] - top) / atr[j] for _, top, _ in active if top <= close[j]) if any(top <= close[j] for _, top, _ in active) else DISTANCE_CAP_ATR
            else:
                dist = min((bottom - close[j]) / atr[j] for bottom, _, _ in active if bottom >= close[j]) if any(bottom >= close[j] for bottom, _, _ in active) else DISTANCE_CAP_ATR
            nearest[j] = float(min(max(dist, -DISTANCE_CAP_ATR), DISTANCE_CAP_ATR))
    return {
        "active_count": active_count,
        "nearest_dist_atr": nearest,
        "retest_event": retest.astype(np.float64),
        "failure_event": failure.astype(np.float64),
        "bars_since_retest": _bars_since(retest, age_cap),
    }


def fair_value_gaps(high: np.ndarray, low: np.ndarray, atr: np.ndarray, *, min_gap_atr: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Three-candle imbalance: bullish when low[i] > high[i-2], bearish when high[i] < low[i-2]."""
    n = len(high)
    bull = np.zeros(n, dtype=bool)
    bear = np.zeros(n, dtype=bool)
    bull_bottom = np.full(n, np.nan)
    bull_top = np.full(n, np.nan)
    bear_bottom = np.full(n, np.nan)
    bear_top = np.full(n, np.nan)
    for i in range(2, n):
        if not np.isfinite(atr[i]) or atr[i] <= 0:
            continue
        gap_up = low[i] - high[i - 2]
        gap_down = low[i - 2] - high[i]
        if gap_up > 0 and gap_up >= min_gap_atr * atr[i]:
            bull[i] = True
            bull_bottom[i], bull_top[i] = high[i - 2], low[i]
        if gap_down > 0 and gap_down >= min_gap_atr * atr[i]:
            bear[i] = True
            bear_bottom[i], bear_top[i] = high[i], low[i - 2]
    return bull, bull_bottom, bull_top, bear, bear_bottom, bear_top


def order_blocks(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, atr: np.ndarray,
    *, displacement_atr: float, displacement_bars: int, search_bars: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Last opposite-colour candle before a displacement of ``displacement_atr`` ATR over ``displacement_bars`` bars.

    Edge-triggered on the displacement condition so one move yields one block.
    """
    n = len(close)
    bull = np.zeros(n, dtype=bool)
    bear = np.zeros(n, dtype=bool)
    bull_bottom = np.full(n, np.nan)
    bull_top = np.full(n, np.nan)
    bear_bottom = np.full(n, np.nan)
    bear_top = np.full(n, np.nan)
    k = int(displacement_bars)
    prev_up = prev_down = False
    for i in range(k, n):
        if not np.isfinite(atr[i]) or atr[i] <= 0:
            prev_up = prev_down = False
            continue
        move = close[i] - close[i - k]
        up = move >= displacement_atr * atr[i]
        down = -move >= displacement_atr * atr[i]
        if up and not prev_up:
            start = i - k
            for c in range(start, max(start - search_bars, -1), -1):
                if close[c] < open_[c]:
                    bull[i] = True
                    bull_bottom[i], bull_top[i] = low[c], high[c]
                    break
        if down and not prev_down:
            start = i - k
            for c in range(start, max(start - search_bars, -1), -1):
                if close[c] > open_[c]:
                    bear[i] = True
                    bear_bottom[i], bear_top[i] = low[c], high[c]
                    break
        prev_up, prev_down = up, down
    return bull, bull_bottom, bull_top, bear, bear_bottom, bear_top


def equal_pools(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, atr: np.ndarray,
    *, swing_lookback: int, tolerance_atr: float, lookback: int, age_cap: int,
) -> dict[str, np.ndarray]:
    """Equal-high / equal-low liquidity pools from confirmed swing pivots (one-shot on sweep or break)."""
    n = len(close)
    sh, sl = _detect_swing_pivots(high, low, swing_lookback)
    out: dict[str, np.ndarray] = {}
    for kind, mask, price in (("eqh", sh, high), ("eql", sl, low)):
        recent: list[tuple[float, int]] = []  # unpaired confirmed swings (level, bar)
        pools: list[tuple[float, int]] = []
        active_count = np.zeros(n)
        nearest = np.full(n, DISTANCE_CAP_ATR)
        form = np.zeros(n, dtype=bool)
        sweep = np.zeros(n, dtype=bool)
        brk = np.zeros(n, dtype=bool)
        for j in range(n):
            # sweep / break checks on existing pools
            keep: list[tuple[float, int]] = []
            for level, birth in pools:
                if j - birth > lookback:
                    continue
                if kind == "eqh":
                    if high[j] > level and close[j] <= level:
                        sweep[j] = True
                        continue
                    if close[j] > level:
                        brk[j] = True
                        continue
                else:
                    if low[j] < level and close[j] >= level:
                        sweep[j] = True
                        continue
                    if close[j] < level:
                        brk[j] = True
                        continue
                keep.append((level, birth))
            pools = keep
            # a pivot at p is confirmed at p + swing_lookback
            p = j - swing_lookback
            if p >= 0 and mask[p] and np.isfinite(atr[j]) and atr[j] > 0:
                level = float(price[p])
                recent = [(lv, b) for lv, b in recent if j - b <= lookback]
                paired = None
                for idx, (lv, b) in enumerate(recent):
                    if abs(lv - level) <= tolerance_atr * atr[j]:
                        paired = idx
                        break
                if paired is not None:
                    lv, _ = recent.pop(paired)
                    pool_level = max(lv, level) if kind == "eqh" else min(lv, level)
                    pools.append((pool_level, j))
                    form[j] = True
                else:
                    recent.append((level, j))
            active_count[j] = float(len(pools))
            if pools and np.isfinite(atr[j]) and atr[j] > 0:
                if kind == "eqh":
                    above = [lv for lv, _ in pools if lv >= close[j]]
                    dist = (min(above) - close[j]) / atr[j] if above else DISTANCE_CAP_ATR
                else:
                    below = [lv for lv, _ in pools if lv <= close[j]]
                    dist = (close[j] - max(below)) / atr[j] if below else DISTANCE_CAP_ATR
                nearest[j] = float(min(dist, DISTANCE_CAP_ATR))
        out[f"{kind}_form_event"] = form.astype(np.float64)
        out[f"{kind}_active_count"] = active_count
        out[f"{kind}_nearest_dist_atr"] = nearest
        out[f"{kind}_sweep_event"] = sweep.astype(np.float64)
        out[f"{kind}_break_event"] = brk.astype(np.float64)
        out[f"{kind}_bars_since_sweep"] = _bars_since(sweep, age_cap)
    return out


def flags(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, atr: np.ndarray,
    *, impulse_bars: int, impulse_atr: float, cons_min: int, cons_max: int, cons_atr: float, age_cap: int,
) -> dict[str, np.ndarray]:
    """Impulse leg followed by a tight consolidation; breakout in the impulse direction, breakdown against it.

    The leg's end is the last bar that extended the leg by more than the
    consolidation scale (``cons_atr`` ATR); smaller new extremes inside the
    consolidation do not restart it.  Tightness is judged on the segment
    strictly before the current bar, so the breakout bar cannot widen the range
    it breaks out of.
    """
    n = len(close)
    out: dict[str, np.ndarray] = {}
    for kind, sign in (("bull_flag", 1.0), ("bear_flag", -1.0)):
        active = np.zeros(n)
        breakout = np.zeros(n, dtype=bool)
        failed = np.zeros(n, dtype=bool)
        watch_start = -1
        leg = float("nan")
        for j in range(impulse_bars, n):
            if not np.isfinite(atr[j]) or atr[j] <= 0:
                watch_start = -1
                continue
            impulse = sign * (close[j] - close[j - impulse_bars]) >= impulse_atr * atr[j]
            if impulse and (watch_start < 0 or sign * (close[j] - leg) > cons_atr * atr[j]):
                watch_start = j
                leg = float(close[j])
                continue
            if watch_start < 0:
                continue
            if j - watch_start > cons_max:
                watch_start = -1
                continue
            span = j - 1 - watch_start
            if span < cons_min:
                continue
            seg_high = float(high[watch_start + 1 : j].max())
            seg_low = float(low[watch_start + 1 : j].min())
            if (seg_high - seg_low) > cons_atr * atr[j]:
                watch_start = -1
                continue
            active[j] = 1.0
            if sign > 0 and close[j] > seg_high:
                breakout[j] = True
                watch_start = -1
            elif sign > 0 and close[j] < seg_low:
                failed[j] = True
                watch_start = -1
            elif sign < 0 and close[j] < seg_low:
                breakout[j] = True
                watch_start = -1
            elif sign < 0 and close[j] > seg_high:
                failed[j] = True
                watch_start = -1
        out[f"{kind}_active"] = active
        out[f"{kind}_breakout_event"] = breakout.astype(np.float64)
        out[f"{kind}_failed_event"] = failed.astype(np.float64)
        out[f"{kind}_bars_since_breakout"] = _bars_since(breakout, age_cap)
    return out


def range_breakouts(high: np.ndarray, low: np.ndarray, close: np.ndarray, *, bars: int, age_cap: int) -> dict[str, np.ndarray]:
    n = len(close)
    up = np.zeros(n, dtype=bool)
    down = np.zeros(n, dtype=bool)
    for j in range(bars, n):
        up[j] = close[j] > high[j - bars : j].max()
        down[j] = close[j] < low[j - bars : j].min()
    up_event = up & ~np.concatenate([[False], up[:-1]])
    down_event = down & ~np.concatenate([[False], down[:-1]])
    return {
        "range_break_up_event": up_event.astype(np.float64),
        "range_break_down_event": down_event.astype(np.float64),
        "range_break_up_state": up.astype(np.float64),
        "range_break_down_state": down.astype(np.float64),
        "bars_since_range_break_up": _bars_since(up_event, age_cap),
        "bars_since_range_break_down": _bars_since(down_event, age_cap),
    }


def trend_context(close: np.ndarray, atr: np.ndarray) -> dict[str, np.ndarray]:
    series = pd.Series(close)
    emas = {span: classic_ema(series, span).to_numpy(np.float64) for span in EMA_SPANS}
    e20, e50, e200 = emas[20], emas[50], emas[200]
    stack = np.zeros(len(close))
    bull = (close > e20) & (e20 > e50) & (e50 > e200)
    bear = (close < e20) & (e20 < e50) & (e50 < e200)
    stack[bull] = 1.0
    stack[bear] = -1.0
    slope = np.full(len(close), np.nan)
    k = EMA_SLOW_SLOPE_LOOKBACK_BARS
    with np.errstate(invalid="ignore", divide="ignore"):
        slope[k:] = (e200[k:] - e200[:-k]) / atr[k:]
    dist200 = np.full(len(close), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        dist200 = (close - e200) / atr
    return {
        "ema_stack": stack,
        "ema200_slope_atr": np.clip(np.nan_to_num(slope, nan=0.0), -DISTANCE_CAP_ATR, DISTANCE_CAP_ATR),
        "ema200_dist_atr": np.clip(np.nan_to_num(dist200, nan=0.0), -DISTANCE_CAP_ATR, DISTANCE_CAP_ATR),
        "ema_warm": np.isfinite(e200).astype(np.float64),
    }


def compute_timeframe_primitives(bars: pd.DataFrame, params: Params) -> pd.DataFrame:
    open_ = bars["open"].to_numpy(np.float64)
    high = bars["high"].to_numpy(np.float64)
    low = bars["low"].to_numpy(np.float64)
    close = bars["close"].to_numpy(np.float64)
    atr = wilder_atr(bars["high"], bars["low"], bars["close"], ATR_PERIOD).to_numpy(np.float64)
    cols: dict[str, np.ndarray] = {"atr": np.nan_to_num(atr, nan=0.0)}
    bull, bb, bt, bear, sb, st = fair_value_gaps(high, low, atr, min_gap_atr=params.fvg_min_gap_atr)
    cols["fvg_bull_event"] = bull.astype(np.float64)
    cols["fvg_bear_event"] = bear.astype(np.float64)
    for name, value in track_zones(bottom_new=bb, top_new=bt, event=bull, low=low, high=high, close=close, atr=atr, side="bull", lookback=params.zone_lookback_bars, age_cap=params.age_cap_bars).items():
        cols[f"fvg_bull_{name}"] = value
    for name, value in track_zones(bottom_new=sb, top_new=st, event=bear, low=low, high=high, close=close, atr=atr, side="bear", lookback=params.zone_lookback_bars, age_cap=params.age_cap_bars).items():
        cols[f"fvg_bear_{name}"] = value
    obull, obb, obt, obear, osb, ost = order_blocks(open_, high, low, close, atr, displacement_atr=params.ob_displacement_atr, displacement_bars=params.ob_displacement_bars, search_bars=params.ob_search_bars)
    cols["ob_bull_event"] = obull.astype(np.float64)
    cols["ob_bear_event"] = obear.astype(np.float64)
    for name, value in track_zones(bottom_new=obb, top_new=obt, event=obull, low=low, high=high, close=close, atr=atr, side="bull", lookback=params.zone_lookback_bars, age_cap=params.age_cap_bars).items():
        cols[f"ob_bull_{name}"] = value
    for name, value in track_zones(bottom_new=osb, top_new=ost, event=obear, low=low, high=high, close=close, atr=atr, side="bear", lookback=params.zone_lookback_bars, age_cap=params.age_cap_bars).items():
        cols[f"ob_bear_{name}"] = value
    cols.update(equal_pools(high, low, close, atr, swing_lookback=params.swing_lookback, tolerance_atr=params.eq_tolerance_atr, lookback=params.zone_lookback_bars, age_cap=params.age_cap_bars))
    cols.update(flags(high, low, close, atr, impulse_bars=params.flag_impulse_bars, impulse_atr=params.flag_impulse_atr, cons_min=params.flag_consolidation_min_bars, cons_max=params.flag_consolidation_max_bars, cons_atr=params.flag_consolidation_atr, age_cap=params.age_cap_bars))
    cols.update(range_breakouts(high, low, close, bars=params.range_breakout_bars, age_cap=params.age_cap_bars))
    cols.update(trend_context(close, atr))
    frame = pd.DataFrame(cols, index=bars.index)
    return frame


def session_anchored_levels(tape: pd.DataFrame, atr: np.ndarray, *, age_cap: int) -> pd.DataFrame:
    """PDH/PDL/PDC, completed Asia range and week open on the owner's 22:00 UTC trading-day clock (M5 rows)."""
    time = pd.DatetimeIndex(tape["time"])
    high = tape["high"].to_numpy(np.float64)
    low = tape["low"].to_numpy(np.float64)
    close = tape["close"].to_numpy(np.float64)
    open_ = tape["open"].to_numpy(np.float64)
    day = trading_session_id_vectorized(time, context="PATTERN_SESSION")
    hour = time.hour.to_numpy()
    asia_start, asia_end = SESSION_BOUNDARIES["ASIA"]
    in_asia = (hour >= asia_start) | (hour < asia_end)
    n = len(time)
    pdh = np.full(n, np.nan)
    pdl = np.full(n, np.nan)
    pdc = np.full(n, np.nan)
    asia_hi = np.full(n, np.nan)
    asia_lo = np.full(n, np.nan)
    asia_current = np.zeros(n)
    week_open = np.full(n, np.nan)
    pdh_break = np.zeros(n, dtype=bool)
    pdl_break = np.zeros(n, dtype=bool)
    asia_hi_break = np.zeros(n, dtype=bool)
    asia_lo_break = np.zeros(n, dtype=bool)
    prev_day = None
    cur_day = None
    cur_high = cur_low = cur_close = np.nan
    prev_high = prev_low = prev_close = np.nan
    cur_asia_hi = cur_asia_lo = np.nan
    done_asia_hi = done_asia_lo = np.nan
    asia_done_day = None
    pdh_broken = pdl_broken = asia_hi_broken = asia_lo_broken = False
    labels = pd.DatetimeIndex(day * int(TRADING_SESSION_DURATION.value) + int(TRADING_SESSION_BOUNDARY_OFFSET.value), tz="UTC")
    week_start = labels - pd.to_timedelta(((labels.weekday + 1) % 7), unit="D")
    cur_week = None
    cur_week_open = np.nan
    for i in range(n):
        d = int(day[i])
        if d != cur_day:
            if cur_day is not None:
                prev_high, prev_low, prev_close = cur_high, cur_low, cur_close
                prev_day = cur_day
            cur_day = d
            cur_high, cur_low, cur_close = high[i], low[i], close[i]
            cur_asia_hi = cur_asia_lo = np.nan
            pdh_broken = pdl_broken = asia_hi_broken = asia_lo_broken = False
        else:
            cur_high = max(cur_high, high[i])
            cur_low = min(cur_low, low[i])
            cur_close = close[i]
        if in_asia[i]:
            cur_asia_hi = high[i] if np.isnan(cur_asia_hi) else max(cur_asia_hi, high[i])
            cur_asia_lo = low[i] if np.isnan(cur_asia_lo) else min(cur_asia_lo, low[i])
        elif not np.isnan(cur_asia_hi) and asia_done_day != cur_day:
            done_asia_hi, done_asia_lo, asia_done_day = cur_asia_hi, cur_asia_lo, cur_day
        ws = week_start[i]
        if ws != cur_week:
            cur_week = ws
            cur_week_open = open_[i]
        week_open[i] = cur_week_open
        if prev_day is not None:
            pdh[i], pdl[i], pdc[i] = prev_high, prev_low, prev_close
            if not pdh_broken and close[i] > prev_high:
                pdh_break[i] = True
                pdh_broken = True
            if not pdl_broken and close[i] < prev_low:
                pdl_break[i] = True
                pdl_broken = True
        if asia_done_day is not None:
            asia_hi[i], asia_lo[i] = done_asia_hi, done_asia_lo
            asia_current[i] = 1.0 if asia_done_day == cur_day else 0.0
            if asia_done_day == cur_day:
                if not asia_hi_broken and close[i] > done_asia_hi:
                    asia_hi_break[i] = True
                    asia_hi_broken = True
                if not asia_lo_broken and close[i] < done_asia_lo:
                    asia_lo_break[i] = True
                    asia_lo_broken = True
    safe_atr = np.where(np.isfinite(atr) & (atr > 0), atr, np.nan)

    def dist(level: np.ndarray) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            value = (close - level) / safe_atr
        return np.clip(np.nan_to_num(value, nan=0.0), -DISTANCE_CAP_ATR, DISTANCE_CAP_ATR)

    return pd.DataFrame(
        {
            "pdh_dist_atr": dist(pdh), "pdl_dist_atr": dist(pdl), "pdc_dist_atr": dist(pdc),
            "pdh_present": np.isfinite(pdh).astype(np.float64),
            "pdh_break_event": pdh_break.astype(np.float64), "pdl_break_event": pdl_break.astype(np.float64),
            "bars_since_pdh_break": _bars_since(pdh_break, age_cap), "bars_since_pdl_break": _bars_since(pdl_break, age_cap),
            "asia_hi_dist_atr": dist(asia_hi), "asia_lo_dist_atr": dist(asia_lo), "asia_range_current": asia_current,
            "asia_range_present": np.isfinite(asia_hi).astype(np.float64),
            "asia_hi_break_event": asia_hi_break.astype(np.float64), "asia_lo_break_event": asia_lo_break.astype(np.float64),
            "week_open_dist_atr": dist(week_open),
        },
        index=time,
    )


def sample_last_closed(bar_labels: pd.DatetimeIndex, values: pd.DataFrame, decision_time: pd.DatetimeIndex, timeframe: str) -> pd.DataFrame:
    """Last bar closed by decision time: label <= t + 5 min - TF duration (owner rule)."""
    if timeframe == "M5":
        index = bar_labels.searchsorted(decision_time, side="left")
        if np.any(index >= len(bar_labels)) or not np.array_equal(bar_labels.values[index], decision_time.values):
            raise RuntimeError("PATTERN_M5_DECISION_ROW_MISSING")
    else:
        cutoff = decision_time + M5_BAR - MULTI_TF_SHIFT[timeframe]
        index = bar_labels.searchsorted(cutoff, side="right") - 1
        if np.any(index < 0):
            raise RuntimeError(f"PATTERN_NO_CLOSED_BAR: {timeframe}")
    sampled = values.iloc[index].reset_index(drop=True)
    sampled.columns = [f"{timeframe}:{c}" for c in sampled.columns]
    return sampled


def build(tape: pd.DataFrame, decision_time: pd.DatetimeIndex, params: Params) -> tuple[pd.DataFrame, dict[str, Any]]:
    blocks: list[pd.DataFrame] = []
    stats: dict[str, Any] = {}
    for tf in TIMEFRAMES:
        bars = closed_bars(tape, tf)
        frame = compute_timeframe_primitives(bars, params)
        if tf == "M5":
            frame = pd.concat([frame, session_anchored_levels(tape, frame["atr"].to_numpy(np.float64), age_cap=params.age_cap_bars)], axis=1)
        sampled = sample_last_closed(pd.DatetimeIndex(bars.index), frame, decision_time, tf)
        blocks.append(sampled)
        stats[tf] = {"bars": int(len(bars)), "fields": int(frame.shape[1])}
        for name in ("fvg_bull_event", "fvg_bear_event", "ob_bull_event", "ob_bear_event", "eqh_form_event", "eql_form_event", "bull_flag_breakout_event", "bear_flag_breakout_event", "range_break_up_event"):
            stats[tf][f"rate_{name}"] = float(np.mean(frame[name].to_numpy()))
    out = pd.concat([pd.DataFrame({"time": decision_time})] + blocks, axis=1)
    if not np.isfinite(out.drop(columns=["time"]).to_numpy(np.float64)).all():
        raise RuntimeError("PATTERN_OUTPUT_NONFINITE")
    return out, stats


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--native-m5-root", required=True)
    p.add_argument("--dataset-dir", required=True, help="decision rows = the TRAIN (+VAL) split times; tape is truncated at the VAL split end")
    p.add_argument("--include-val", action="store_true", help="also emit rows for the VAL split (confirmation stage)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--swing-lookback", type=int, default=SWING_LOOKBACK)
    p.add_argument("--zone-lookback-bars", type=int, required=True)
    p.add_argument("--fvg-min-gap-atr", type=float, required=True)
    p.add_argument("--ob-displacement-atr", type=float, required=True)
    p.add_argument("--ob-displacement-bars", type=int, required=True)
    p.add_argument("--ob-search-bars", type=int, required=True)
    p.add_argument("--eq-tolerance-atr", type=float, required=True)
    p.add_argument("--flag-impulse-bars", type=int, required=True)
    p.add_argument("--flag-impulse-atr", type=float, required=True)
    p.add_argument("--flag-consolidation-min-bars", type=int, required=True)
    p.add_argument("--flag-consolidation-max-bars", type=int, required=True)
    p.add_argument("--flag-consolidation-atr", type=float, required=True)
    p.add_argument("--range-breakout-bars", type=int, required=True)
    p.add_argument("--age-cap-bars", type=int, required=True)
    return p


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    if out_dir.exists():
        raise RuntimeError("PATTERN_OUT_DIR_EXISTS")
    out_dir.mkdir(parents=True)
    dataset_dir = Path(args.dataset_dir)
    manifest = json.loads((dataset_dir / "entry_dataset__ENTRY_FITTED_Q_train.manifest.json").read_text(encoding="utf-8"))
    val_start = pd.Timestamp(manifest["splits"]["val"]["start"])
    val_end = pd.Timestamp(manifest["splits"]["val"]["end"])
    times = [pd.DatetimeIndex(pd.to_datetime(pq.read_table(str(dataset_dir / "entry_dataset__ENTRY_FITTED_Q_train.parquet"), columns=["time"]).column("time").to_pandas(), utc=True))]
    if args.include_val:
        times.append(pd.DatetimeIndex(pd.to_datetime(pq.read_table(str(dataset_dir / "entry_dataset__ENTRY_FITTED_Q_val.parquet"), columns=["time"]).column("time").to_pandas(), utc=True)))
    decision_time = pd.DatetimeIndex(np.concatenate([t.values for t in times])).tz_localize("UTC") if times[0].tz is None else pd.DatetimeIndex(np.concatenate([t.asi8 for t in times]), tz="UTC")
    if not decision_time.is_monotonic_increasing or decision_time.has_duplicates:
        raise RuntimeError("PATTERN_DECISION_TIME_INVALID")
    tape = load_tape_ohlcv(Path(args.native_m5_root), truncate_before=(val_end if args.include_val else val_start))
    params = Params(
        swing_lookback=args.swing_lookback, zone_lookback_bars=args.zone_lookback_bars, fvg_min_gap_atr=args.fvg_min_gap_atr,
        ob_displacement_atr=args.ob_displacement_atr, ob_displacement_bars=args.ob_displacement_bars, ob_search_bars=args.ob_search_bars,
        eq_tolerance_atr=args.eq_tolerance_atr, flag_impulse_bars=args.flag_impulse_bars, flag_impulse_atr=args.flag_impulse_atr,
        flag_consolidation_min_bars=args.flag_consolidation_min_bars, flag_consolidation_max_bars=args.flag_consolidation_max_bars,
        flag_consolidation_atr=args.flag_consolidation_atr, range_breakout_bars=args.range_breakout_bars, age_cap_bars=args.age_cap_bars,
    )
    frame, stats = build(tape, decision_time, params)
    parquet_path = out_dir / "pattern_primitives.parquet"
    frame.to_parquet(parquet_path, index=False)
    report = {
        "schema_version": SCHEMA_VERSION,
        "authority": AUTHORITY,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "instrument_source_sha256": _sha256_file(Path(__file__)),
        "inputs": {
            "native_m5_root": str(args.native_m5_root),
            "native_m5_manifest_sha256": _sha256_file(Path(args.native_m5_root) / "MANIFEST.json"),
            "dataset_dir": str(dataset_dir),
            "decision_rows": int(len(decision_time)),
            "include_val": bool(args.include_val),
            "tape_truncated_before": str(val_end if args.include_val else val_start),
        },
        "params": params.__dict__,
        "conventions": {
            "distance_cap_atr": DISTANCE_CAP_ATR, "atr_period": ATR_PERIOD, "ema_spans": list(EMA_SPANS),
            "ema_slow_slope_lookback_bars": EMA_SLOW_SLOPE_LOOKBACK_BARS, "timeframes": list(TIMEFRAMES),
            "last_closed_rule": "label <= t + 5min - TF duration (htf_features.multi_tf_last_closed_label)",
        },
        "columns": [c for c in frame.columns if c != "time"],
        "stats": stats,
        "output": {"parquet": str(parquet_path), "sha256": _sha256_file(parquet_path), "rows": int(len(frame)), "columns": int(frame.shape[1] - 1)},
    }
    (out_dir / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    report = run(build_parser().parse_args())
    print(json.dumps({"rows": report["output"]["rows"], "columns": report["output"]["columns"], "stats": report["stats"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
