#!/usr/bin/env python3
"""Model-free XAUUSD baselines: scalp rules on M5, trend rules on D1.

Preregistered in docs/MODEL_FREE_BASELINES_PREREG_20260927.md; every cell, window and decision
threshold below is that registration. Research evidence only: one pre-TEST native M5 tape is read
up to the declared end, fills and costs go through the walk-forward instrument's owners
(``load_tape``, ``load_cost_policy``, ``apply_cost_policy``), and per-cell statistics are written.
The legacy CLI retains that registration and reads no VAL or TEST row.
The 2026-09-29 research helpers below provide continuous portfolio accounting,
historical benchmark financing and causal risk sizing for separately registered
A/B/C callers. They do not change native costs or authorize a market run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import pandas as pd

from gx1.scripts.research_entry_direction_walkforward_v1 import (
    BPS,
    Tape,
    apply_cost_policy,
    load_cost_policy,
    load_tape,
)

MOM_LOOKBACK_BARS = (1, 6, 12)
MOM_HOLD_BARS = (6, 12, 19)
ORB_SESSIONS = {"london": (7, 16), "new_york": (13, 20)}  # UTC open hour -> exit hour
ORB_RANGE_BARS = 12
SESSION_WINDOWS = {"asia": (22, 7), "london": (7, 13), "new_york": (13, 20)}  # UTC start -> end hour
TSMOM_LOOKBACK_D1 = (21, 63, 126, 252)
SMA_D1 = 200
SWING_HORIZONS_D1 = (5, 10, 20)
TRADING_DAY_START_HOUR_UTC = 22
GO_T = 3.15  # one-sided Bonferroni 0.05 / 62 cells
LOVENDE_T = 2.0
YEAR_SHARE = 0.60
FULL_YEARS = tuple(range(2012, 2025))
BEAR_FOLD = (pd.Timestamp("2011-09-01", tz="UTC"), pd.Timestamp("2016-01-01", tz="UTC"))
BAR = pd.Timedelta(minutes=5)



@dataclass(frozen=True)
class ResearchFinancingCurve:
    """Piecewise annual benchmark plus explicit markup; a research proxy, not broker history.

    Positive values are costs: LONG = benchmark + markup; SHORT = markup - benchmark.
    Rates apply from effective_at through the next change, including closed-market time.
    The caller must bind source bytes, rate-date semantics and coverage_end in its prereg.
    """

    effective_at: pd.DatetimeIndex
    benchmark_annual_rate: np.ndarray
    coverage_end: pd.Timestamp
    broker_markup: float
    seconds_per_year: float

    def __post_init__(self) -> None:
        times = pd.DatetimeIndex(self.effective_at)
        end = pd.Timestamp(self.coverage_end)
        rates = np.array(self.benchmark_annual_rate, dtype=np.float64, copy=True)
        if (len(times) == 0 or times.tz is None or times.hasnans
                or times.has_duplicates or not times.is_monotonic_increasing
                or rates.shape != (len(times),) or not np.isfinite(rates).all()
                or pd.isna(end) or end.tzinfo is None or end <= times[-1]
                or not np.isfinite(self.broker_markup) or self.broker_markup < 0
                or not np.isfinite(self.seconds_per_year) or self.seconds_per_year <= 0):
            raise RuntimeError("BASELINE_FINANCING_CURVE_INVALID")
        rates.setflags(write=False)
        object.__setattr__(self, "effective_at", times.tz_convert("UTC"))
        object.__setattr__(self, "coverage_end", end.tz_convert("UTC"))
        object.__setattr__(self, "benchmark_annual_rate", rates)

    def integrated_cost_rates(
        self, starts: pd.DatetimeIndex, ends: pd.DatetimeIndex,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Integral of each annual cost rate over [start, end), as fractions of opening notional."""
        starts, ends = pd.DatetimeIndex(starts), pd.DatetimeIndex(ends)
        if (starts.tz is None or ends.tz is None or starts.hasnans or ends.hasnans
                or len(starts) != len(ends) or np.any(ends.asi8 < starts.asi8)
                or np.any(starts.asi8 < self.effective_at[0].value)
                or np.any(ends.asi8 > self.coverage_end.value)):
            raise RuntimeError("BASELINE_FINANCING_COVERAGE_INVALID")
        changes = self.effective_at.asi8
        prefix = np.concatenate([[0.0], np.cumsum(
            np.diff(changes) / 1e9 * self.benchmark_annual_rate[:-1]
        )])

        def primitive(times: pd.DatetimeIndex) -> np.ndarray:
            index = np.searchsorted(changes, times.asi8, side="right") - 1
            return prefix[index] + (times.asi8 - changes[index]) / 1e9 * self.benchmark_annual_rate[index]

        benchmark = (primitive(ends) - primitive(starts)) / self.seconds_per_year
        markup = (ends.asi8 - starts.asi8) / 1e9 / self.seconds_per_year * self.broker_markup
        return benchmark + markup, markup - benchmark


def portfolio_path(
    tape: Tape, held_units: np.ndarray, *,
    initial_equity: float, slippage_bps_per_execution: float,
    commission_bps_per_execution: float,
    financing_curve: ResearchFinancingCurve | None,
    liquidate_at_end: bool,
) -> pd.DataFrame:
    """Cash ledger with q[i] held from quote i to quote i+1; no periodic forced round trips.

    Quotes must carry their actual availability times (the caller owns the bar clock).
    Spread is paid on quantity changes only. Slippage/commission are charged on
    executed BID/ASK notional. Financing uses opening BID/ASK notional, with weighted
    opening basis on adds and proportional release on reductions, matching the
    existing open-notional financing convention. None explicitly means zero financing.
    Remaining units are marked at executable liquidation value including exit costs.
    This supplies both continuous buy-and-hold and the same ledger for model positions.
    """
    n = len(tape.time)
    units = np.asarray(held_units, dtype=np.float64)
    prices = [np.asarray(v, dtype=np.float64) for v in (tape.mid, tape.bid, tape.ask)]
    mid, bid, ask = prices
    if (n < 2 or tape.time.tz is None or tape.time.hasnans or tape.time.has_duplicates
            or not tape.time.is_monotonic_increasing or units.shape != (n - 1,)
            or not np.isfinite(units).all() or any(v.shape != (n,) for v in prices)
            or not all(np.isfinite(v).all() and (v > 0).all() for v in prices)
            or np.any(bid > mid) or np.any(mid > ask)
            or not np.isfinite(initial_equity) or initial_equity <= 0
            or not np.isfinite(slippage_bps_per_execution) or slippage_bps_per_execution < 0
            or not np.isfinite(commission_bps_per_execution) or commission_bps_per_execution < 0
            or not isinstance(liquidate_at_end, bool)):
        raise RuntimeError("BASELINE_PORTFOLIO_INPUT_INVALID")
    after = np.concatenate([units, [0.0 if liquidate_at_end else units[-1]]])
    before = np.concatenate([[0.0], after[:-1]])
    traded = after - before
    execution_price = np.where(traded > 0, ask, bid)
    executed_notional = np.abs(traded) * execution_price
    spread_cost = np.where(traded > 0, traded * (ask - mid), -traded * (mid - bid))
    slippage_cost = executed_notional * slippage_bps_per_execution / BPS
    commission_cost = executed_notional * commission_bps_per_execution / BPS
    mid_pnl = np.concatenate([[0.0], units * np.diff(mid)])
    basis_after = np.zeros(n, dtype=np.float64)
    for i, q in enumerate(after):
        old = before[i]
        if q == 0:
            continue
        if old * q > 0:
            if abs(q) <= abs(old):
                basis_after[i] = basis_after[i - 1] * abs(q / old)
            else:
                basis_after[i] = basis_after[i - 1] + abs(q - old) * execution_price[i]
        else:
            basis_after[i] = abs(q) * execution_price[i]
    financing = np.zeros(n, dtype=np.float64)
    if financing_curve is not None:
        long_cost, short_cost = financing_curve.integrated_cost_rates(tape.time[:-1], tape.time[1:])
        financing[1:] = basis_after[:-1] * np.where(units > 0, long_cost, np.where(units < 0, short_cost, 0.0))
    net_cash_pnl = mid_pnl - spread_cost - slippage_cost - commission_cost - financing
    equity_mid = initial_equity + np.cumsum(net_cash_pnl)
    close_price = np.where(after > 0, bid, ask)
    exit_spread = np.abs(after) * np.abs(mid - close_price)
    exit_fees = np.abs(after) * close_price * (slippage_bps_per_execution + commission_bps_per_execution) / BPS
    liquidation_reserve = exit_spread + exit_fees
    frame = pd.DataFrame({
        "time": tape.time, "held_units_after": after, "traded_units": traded,
        "opening_notional_after": basis_after, "mid_pnl": mid_pnl,
        "spread_cost": spread_cost, "slippage_cost": slippage_cost,
        "commission_cost": commission_cost, "financing_cost": financing,
        "equity_mid": equity_mid, "liquidation_reserve": liquidation_reserve,
        "equity_liquidation": equity_mid - liquidation_reserve,
    })
    frame.attrs.update(
        initial_equity=float(initial_equity),
        financing="historical_benchmark_plus_markup_proxy" if financing_curve is not None else "zero",
        historical_broker_cost_truth=False, liquidated_at_end=liquidate_at_end,
        financing_notional="weighted_opening_bid_ask_notional",
    )
    return frame


def portfolio_period_returns(frame: pd.DataFrame) -> np.ndarray:
    """Return one value per interval, including initial execution costs in the first."""
    initial = float(frame.attrs["initial_equity"])
    equity = frame["equity_liquidation"].to_numpy(dtype=np.float64)
    if len(equity) < 2 or not np.isfinite(equity).all() or np.any(equity <= 0):
        raise RuntimeError("BASELINE_PORTFOLIO_RETURNS_INVALID")
    previous = np.concatenate([[initial], equity[1:-1]])
    return equity[1:] / previous - 1.0


def portfolio_summary(frame: pd.DataFrame, *, periods_per_year: float) -> dict[str, Any]:
    """Daily/declared-clock portfolio statistics; include entry costs in the first interval.

    Sharpe is relative to zero cash return. An insolvent path retains cash PnL and
    drawdown but does not receive a compounded return/Sharpe interpretation.
    """
    initial = float(frame.attrs["initial_equity"])
    equity = frame["equity_liquidation"].to_numpy(dtype=np.float64)
    if len(equity) < 2 or not np.isfinite(periods_per_year) or periods_per_year <= 0:
        raise RuntimeError("BASELINE_PORTFOLIO_SUMMARY_INVALID")
    # First interval starts from initial capital, not the already cost-debited first quote.
    peak = np.maximum.accumulate(np.concatenate([[initial], equity]))
    drawdown = 1.0 - np.concatenate([[initial], equity]) / peak
    out: dict[str, Any] = {
        "intervals": len(equity) - 1,
        "total_net_bps": float((equity[-1] / initial - 1) * BPS),
        "total_mid_pnl_bps": float(frame["mid_pnl"].sum() / initial * BPS),
        "max_drawdown": float(drawdown.max()),
        "insolvent": bool(np.any(equity <= 0)),
        "open_units": float(frame["held_units_after"].iloc[-1]),
        "execution_count": int((frame["traded_units"] != 0).sum()),
        "periods_per_year": float(periods_per_year),
    }
    for name in ("spread_cost", "slippage_cost", "commission_cost", "financing_cost", "liquidation_reserve"):
        value = frame[name].iloc[-1] if name == "liquidation_reserve" else frame[name].sum()
        out[name + "_bps"] = float(value / initial * BPS)
    if not out["insolvent"]:
        returns = portfolio_period_returns(frame)
        out["mean_period_net_bps"] = float(returns.mean() * BPS)
        if len(returns) > 1:
            sd = float(returns.std(ddof=1))
            out["realized_annual_vol"] = sd * math.sqrt(periods_per_year)
            if sd > 0:
                out["sharpe_zero_cash"] = float(returns.mean() / sd * math.sqrt(periods_per_year))
    return out


def causal_risk_units(
    mid: np.ndarray, side: np.ndarray, *, lookback: int, periods_per_year: float,
    target_annual_vol: float, max_gross_leverage: float, initial_equity: float,
) -> np.ndarray:
    """Same past-return volatility scale for each model and always-LONG.

    q[i] uses returns ending at i and applies only after quote i. Capital is the
    fixed declared initial budget, not future realized strategy risk. Warmup and
    zero-variance windows remain NaN; callers must use a common valid population.
    """
    mid, side = np.asarray(mid, dtype=np.float64), np.asarray(side, dtype=np.float64)
    parameters = (periods_per_year, target_annual_vol, max_gross_leverage, initial_equity)
    if (mid.ndim != 1 or side.shape != mid.shape or not np.isfinite(mid).all()
            or np.any(mid <= 0) or not np.isin(side[~np.isnan(side)], [-1, 0, 1]).all()
            or not isinstance(lookback, int) or isinstance(lookback, bool) or lookback < 2
            or not all(np.isfinite(x) and x > 0 for x in parameters)):
        raise RuntimeError("BASELINE_RISK_INPUT_INVALID")
    returns = pd.Series(mid).pct_change(fill_method=None)
    sigma = returns.rolling(lookback, min_periods=lookback).std(ddof=1).to_numpy() * math.sqrt(periods_per_year)
    units = np.full(len(mid), np.nan)
    valid = np.isfinite(sigma) & (sigma > 0) & np.isfinite(side)
    leverage = np.minimum(target_annual_vol / sigma[valid], max_gross_leverage)
    units[valid] = side[valid] * leverage * initial_equity / mid[valid]
    return units


def trade_outcomes(
    tape: Tape,
    entry: np.ndarray,
    exit_: np.ndarray,
    side: np.ndarray,
    policy: dict[str, Any],
    *,
    financing: bool,
    financing_curve: ResearchFinancingCurve | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Gross and net bps per trade; side +1 long, -1 short, 0 flat (flat costs nothing)."""
    entry = np.asarray(entry, dtype=np.int64)
    exit_ = np.asarray(exit_, dtype=np.int64)
    side = np.asarray(side, dtype=np.int64)
    if len(entry) == 0:
        return np.zeros(0), np.zeros(0)
    if np.any(exit_ <= entry):
        raise RuntimeError("BASELINE_EXIT_NOT_AFTER_ENTRY")
    long_gross = (tape.bid[exit_] / tape.ask[entry] - 1.0) * BPS
    short_gross = (1.0 - tape.ask[exit_] / tape.bid[entry]) * BPS
    elapsed = np.asarray((tape.time[exit_] - tape.time[entry]).total_seconds(), dtype=np.float64)
    scenario = dict(policy)
    if not financing or financing_curve is not None:
        scenario["long_annual_cost_rate"] = 0.0
        scenario["short_annual_cost_rate"] = 0.0
    long_net, short_net = apply_cost_policy(long_gross, short_gross, elapsed, scenario)
    if financing and financing_curve is not None:
        long_cost, short_cost = financing_curve.integrated_cost_rates(tape.time[entry], tape.time[exit_])
        long_net -= long_cost * BPS
        short_net -= short_cost * BPS
    gross = np.where(side > 0, long_gross, np.where(side < 0, short_gross, 0.0))
    net = np.where(side > 0, long_net, np.where(side < 0, short_net, 0.0))
    return gross, net


def contiguity_breaks(time: pd.DatetimeIndex) -> np.ndarray:
    """breaks[i] = number of non-5-minute steps among bars 0..i; [a, b] is gap-free iff equal."""
    steps = np.diff(time.asi8) != BAR.value
    return np.concatenate([[0], np.cumsum(steps)])


def momentum_trades(
    tape: Tape, *, lookback: int, hold: int, reverse: bool, start: pd.Timestamp
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedy chronological non-overlapping MOM/REV trades on gap-free windows."""
    mid = tape.mid
    breaks = contiguity_breaks(tape.time)
    first = int(tape.time.searchsorted(start, side="left"))
    entries, exits, sides = [], [], []
    t = max(first, lookback)
    last = len(mid) - 1
    while t + hold <= last:
        if breaks[t - lookback] != breaks[t + hold]:
            t += 1
            continue
        signal = np.sign(mid[t] / mid[t - lookback] - 1.0)
        if signal == 0:
            t += 1
            continue
        entries.append(t)
        exits.append(t + hold)
        sides.append(int(-signal if reverse else signal))
        t += hold
    return np.asarray(entries, dtype=np.int64), np.asarray(exits, dtype=np.int64), np.asarray(sides, dtype=np.int64)


def _day_groups(time: pd.DatetimeIndex) -> dict[pd.Timestamp, np.ndarray]:
    """Contiguous index ranges per UTC calendar day of a sorted tape."""
    days = time.floor("D")
    cuts = np.flatnonzero(days[1:] != days[:-1]) + 1
    bounds = np.concatenate([[0], cuts, [len(time)]])
    return {days[a]: np.arange(a, b) for a, b in zip(bounds[:-1], bounds[1:])}


def orb_trades(tape: Tape, *, open_hour: int, exit_hour: int, start: pd.Timestamp) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One opening-range breakout per UTC day and session."""
    entries, exits, sides = [], [], []
    minutes = tape.time.hour * 60 + tape.time.minute
    for day, idx in _day_groups(tape.time).items():
        if day < start.floor("D"):
            continue
        m = minutes[idx]
        range_idx = idx[(m >= open_hour * 60) & (m < open_hour * 60 + ORB_RANGE_BARS * 5)]
        if len(range_idx) != ORB_RANGE_BARS or tape.time[range_idx[0]] < start:
            continue
        window = idx[(m >= open_hour * 60 + ORB_RANGE_BARS * 5) & (m < exit_hour * 60)]
        if len(window) < 2:
            continue
        high = tape.mid[range_idx].max()
        low = tape.mid[range_idx].min()
        exit_index = int(window[-1])
        for j in window[:-1]:
            side = 1 if tape.mid[j] > high else -1 if tape.mid[j] < low else 0
            if side:
                entries.append(int(j))
                exits.append(exit_index)
                sides.append(side)
                break
    return np.asarray(entries, dtype=np.int64), np.asarray(exits, dtype=np.int64), np.asarray(sides, dtype=np.int64)


def session_trades(
    tape: Tape, *, start_hour: int, end_hour: int, side: int, start: pd.Timestamp
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hold from the first bar at/after the window start to the last bar before its end."""
    entries, exits = [], []
    time = tape.time
    days = pd.date_range(start.floor("D"), time[-1].floor("D"), freq="D", tz="UTC")
    for day in days:
        begin = day + pd.Timedelta(hours=start_hour)
        end = day + pd.Timedelta(hours=end_hour)
        if end <= begin:  # window crosses midnight: starts the previous day
            begin -= pd.Timedelta(days=1)
        if begin < start:
            continue
        first = int(time.searchsorted(begin, side="left"))
        last = int(time.searchsorted(end, side="left")) - 1
        if first >= len(time) or last <= first or time[first] >= end:
            continue
        entries.append(first)
        exits.append(last)
    n = len(entries)
    return np.asarray(entries, dtype=np.int64), np.asarray(exits, dtype=np.int64), np.full(n, side, dtype=np.int64)


def d1_closes(tape: Tape) -> np.ndarray:
    """Index of the last M5 bar of each trading day starting 22:00 UTC."""
    label = (tape.time + pd.Timedelta(hours=24 - TRADING_DAY_START_HOUR_UTC)).floor("D")
    change = np.flatnonzero(label[1:] != label[:-1])
    return np.concatenate([change, [len(tape.time) - 1]]).astype(np.int64)


def swing_signals(close: np.ndarray) -> dict[str, np.ndarray]:
    """Signal per D1 close; NaN where the lookback is not yet available."""
    n = len(close)
    out: dict[str, np.ndarray] = {}
    signs = []
    for lookback in TSMOM_LOOKBACK_D1:
        s = np.full(n, np.nan)
        s[lookback:] = np.sign(close[lookback:] / close[:-lookback] - 1.0)
        out[f"tsmom_{lookback}"] = s
        signs.append(s)
    out["combo"] = np.sign(np.mean(np.vstack(signs), axis=0))
    sma = np.full(n, np.nan)
    csum = np.cumsum(np.concatenate([[0.0], close]))
    sma[SMA_D1 - 1:] = (csum[SMA_D1:] - csum[:-SMA_D1]) / SMA_D1
    out["sma200"] = np.sign(close - sma)
    return out



def stationary_bootstrap_indices(
    observations: int, *, draws: int, mean_block_length: float, seed: int,
) -> Iterator[np.ndarray]:
    """Politis-Romano circular stationary bootstrap with geometric block lengths.

    A caller uses each yielded index ONCE across every strategy, benchmark,
    normalizer and endpoint in the declared family; never resample sides separately.
    The block length and seed must be bound before outcome inspection.
    """
    if (not isinstance(observations, int) or observations < 2
            or not isinstance(draws, int) or draws < 2
            or not np.isfinite(mean_block_length) or not 1 <= mean_block_length <= observations
            or not isinstance(seed, int) or seed < 0):
        raise RuntimeError("BASELINE_BOOTSTRAP_INPUT_INVALID")
    rng = np.random.default_rng(seed)
    positions = np.arange(observations)
    for _ in range(draws):
        starts = rng.integers(observations, size=observations)
        restart = rng.random(observations) < 1.0 / mean_block_length
        restart[0] = True
        latest_start = np.maximum.accumulate(np.where(restart, positions, 0))
        yield (starts[latest_start] + positions - latest_start) % observations


def max_t_inference(
    estimates: np.ndarray, bootstrap_estimates: np.ndarray, *, names: list[str],
    alpha: float, desired_power: float, effect_sizes: np.ndarray,
    minimum_relevant_effect: np.ndarray,
) -> dict[str, Any]:
    """Single-step paired bootstrap max-|t| family inference, not an IID t-test.

    Rows of bootstrap_estimates MUST come from shared stationary-bootstrap draws
    of the complete declared family. SE is the bootstrap SD, held fixed for
    centered bootstrap roots (not a nested/bootstrap-t variance estimate).
    Two-sided simultaneous bands give direction-aware economic-effect decisions.
    Power/MDE use a plug-in location-shift model and the joint max-|t| cutoff;
    they are conditional precision diagnostics, not observed trading edge.
    """
    point = np.asarray(estimates, dtype=np.float64)
    boot = np.asarray(bootstrap_estimates, dtype=np.float64)
    effects = np.asarray(effect_sizes, dtype=np.float64)
    minimum = np.asarray(minimum_relevant_effect, dtype=np.float64)
    k = len(names)
    if (k == 0 or len(set(names)) != k or any(not isinstance(n, str) or not n for n in names)
            or point.shape != (k,) or boot.ndim != 2 or boot.shape[1] != k or len(boot) < 2
            or effects.shape != (k, 3) or minimum.shape != (k,)
            or not all(np.isfinite(x).all() for x in (point, boot, effects, minimum))
            or np.any(effects <= 0) or np.any(np.diff(effects, axis=1) <= 0)
            or np.any(minimum <= 0) or not 0 < alpha < 1 or not 0.5 < desired_power < 1):
        raise RuntimeError("BASELINE_MAX_T_INPUT_INVALID")
    se = boot.std(axis=0, ddof=1)
    if np.any(se <= 0):
        raise RuntimeError("BASELINE_MAX_T_DEGENERATE_ENDPOINT")
    draws = len(boot)
    errors = boot - point
    roots = errors / se
    maximum = np.max(np.abs(roots), axis=1)
    # Match the (exceedances + 1)/(draws + 1) p-value resolution. A plain
    # empirical percentile can yield a GO although the adjusted p exceeds alpha.
    critical_rank = math.ceil((draws + 1) * (1 - alpha))
    if critical_rank > draws:
        raise RuntimeError("BASELINE_MAX_T_DRAWS_TOO_FEW_FOR_ALPHA")
    critical = float(np.sort(maximum)[critical_rank - 1])
    lower, upper = point - critical * se, point + critical * se
    rows = []
    for j, name in enumerate(names):
        adjusted_p = (1 + int(np.sum(maximum >= abs(point[j] / se[j])))) / (draws + 1)
        # The positive-direction test rejects at estimate > critical * SE.
        powers = [float(np.mean(effect + errors[:, j] > critical * se[j])) for effect in effects[j]]
        mde = max(0.0, float(critical * se[j] - np.quantile(errors[:, j], 1 - desired_power, method="lower")))
        verdict = ("GO" if lower[j] > minimum[j] else
                   "NO_GO" if upper[j] < minimum[j] else "INKONKLUSIV")
        rows.append({
            "name": name, "estimate": float(point[j]), "bootstrap_se": float(se[j]),
            "simultaneous_lower": float(lower[j]), "simultaneous_upper": float(upper[j]),
            "max_t_adjusted_two_sided_p": adjusted_p,
            "minimum_relevant_effect": float(minimum[j]), "effect_verdict": verdict,
            "power_at_declared_effects": [
                {"effect": float(effect), "power": power,
                 "conditional_monte_carlo_se": math.sqrt(power * (1 - power) / draws)}
                for effect, power in zip(effects[j], powers)
            ],
            "mde_against_zero": mde,
            "plug_in_power_at_mde": float(np.mean(mde + errors[:, j] > critical * se[j])),
        })
    return {
        "method": "single_step_paired_stationary_bootstrap_max_abs_t_fixed_bootstrap_se",
        "family": list(names), "family_size": k, "bootstrap_draws": draws,
        "alpha": float(alpha), "desired_power": float(desired_power),
        "simultaneous_critical_value": critical,
        "minimum_resolvable_p": 1 / (draws + 1),
        "power_model": "location_shift_of_centered_bootstrap_errors_under_joint_max_abs_t_cutoff",
        "scope": "effect inference only; economic and data-admissibility gates remain separate",
        "endpoints": rows,
    }


def summarize(values: np.ndarray, times: pd.DatetimeIndex, full_years: tuple[int, ...] = FULL_YEARS) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    n = int(len(values))
    out: dict[str, Any] = {"n": n}
    if n < 2:
        return out
    sd = float(values.std(ddof=1))
    mean = float(values.mean())
    out.update(mean_bps=mean, sd_bps=sd)
    if sd > 0:
        out["t"] = mean / (sd / math.sqrt(n))
    years = times.year
    per_year = {int(y): float(values[years == y].mean()) for y in full_years if (years == y).any()}
    out["per_year_mean_bps"] = per_year
    out["positive_full_year_share"] = sum(1 for v in per_year.values() if v > 0) / len(full_years)
    return out


def verdict(stats: dict[str, Any], *, scenario_b_mean: float | None = None) -> str:
    t = stats.get("t")
    if t is None or stats.get("positive_full_year_share", 0.0) < YEAR_SHARE:
        return "NO_GO"
    if scenario_b_mean is not None and not scenario_b_mean > 0:
        return "NO_GO"
    if t >= GO_T:
        return "GO"
    if t >= LOVENDE_T:
        return "LOVENDE"
    return "NO_GO"


def _trade_record(tape: Tape, entries, exits, sides, policy) -> dict[str, Any]:
    gross, net = trade_outcomes(tape, entries, exits, sides, policy, financing=True)
    times = tape.time[entries] if len(entries) else pd.DatetimeIndex([], tz="UTC")
    stats = summarize(net, times)
    if len(gross):
        stats["gross_mean_bps"] = float(gross.mean())
        stats["gross_hit_rate"] = float((gross > 0).mean())
    stats["verdict"] = verdict(stats)
    return stats


def scalp_arm(tape: Tape, policy: dict[str, Any], start: pd.Timestamp) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    for lookback in MOM_LOOKBACK_BARS:
        for hold in MOM_HOLD_BARS:
            for reverse in (False, True):
                name = f"{'rev' if reverse else 'mom'}_k{lookback}_h{hold}"
                cells[name] = _trade_record(tape, *momentum_trades(tape, lookback=lookback, hold=hold, reverse=reverse, start=start), policy)
    for session, (open_hour, exit_hour) in ORB_SESSIONS.items():
        cells[f"orb_{session}"] = _trade_record(tape, *orb_trades(tape, open_hour=open_hour, exit_hour=exit_hour, start=start), policy)
    for window, (start_hour, end_hour) in SESSION_WINDOWS.items():
        for side, label in ((1, "long"), (-1, "short")):
            cells[f"session_{window}_{label}"] = _trade_record(
                tape, *session_trades(tape, start_hour=start_hour, end_hour=end_hour, side=side, start=start), policy
            )
    return cells


def swing_arm(tape: Tape, policy: dict[str, Any], start: pd.Timestamp) -> dict[str, Any]:
    closes = d1_closes(tape)
    mid_close = tape.mid[closes]
    signals = swing_signals(mid_close)
    first_valid = int(np.searchsorted(tape.time[closes].asi8, start.value, side="left"))
    first_valid = max(first_valid, max(TSMOM_LOOKBACK_D1), SMA_D1 - 1)
    cells: dict[str, Any] = {}
    for horizon in SWING_HORIZONS_D1:
        offsets: dict[str, list[float]] = {}
        for offset in range(horizon):
            days = np.arange(first_valid + offset, len(closes) - horizon, horizon)
            entries, exits = closes[days], closes[days + horizon]
            times = tape.time[entries]
            always = np.ones(len(days), dtype=np.int64)
            ref = {f: trade_outcomes(tape, entries, exits, always, policy, financing=f)[1] for f in (True, False)}
            for name, signal in signals.items():
                s = signal[days]
                if np.isnan(s).any():
                    raise RuntimeError("BASELINE_SIGNAL_UNAVAILABLE_AT_DECISION")
                for variant, sides in (("ls", s.astype(np.int64)), ("lf", (s > 0).astype(np.int64))):
                    cell = f"{name}_{variant}_h{horizon}"
                    net_a = trade_outcomes(tape, entries, exits, sides, policy, financing=True)[1]
                    net_b = trade_outcomes(tape, entries, exits, sides, policy, financing=False)[1]
                    delta_a, delta_b = net_a - ref[True], net_b - ref[False]
                    offsets.setdefault(cell, []).append(float(delta_a.mean()))
                    if offset != 0:
                        continue
                    stats = summarize(delta_a, times)
                    stats["strategy_mean_bps_A"] = float(net_a.mean())
                    stats["always_long_mean_bps_A"] = float(ref[True].mean())
                    stats["delta_mean_bps_B"] = float(delta_b.mean())
                    stats["always_long_mean_bps_B"] = float(ref[False].mean())
                    bear = (times >= BEAR_FOLD[0]) & (times < BEAR_FOLD[1])
                    stats["bear_fold_delta_mean_bps_A"] = float(delta_a[bear].mean()) if bear.any() else None
                    stats["bear_fold_n"] = int(bear.sum())
                    stats["verdict"] = verdict(stats, scenario_b_mean=stats["delta_mean_bps_B"])
                    cells[cell] = stats
        for cell, values in offsets.items():
            cells[cell]["offset_robustness_delta_A"] = {"mean": float(np.mean(values)), "min": float(np.min(values))}
    return cells


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True,
                          cwd=Path(__file__).resolve().parents[2]).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-m5-root", type=Path, required=True)
    parser.add_argument("--cost-policy", type=Path, required=True)
    parser.add_argument("--cost-policy-sha256", required=True)
    parser.add_argument("--eval-start", required=True, help="first entry time (UTC)")
    parser.add_argument("--read-end-exclusive", required=True, help="tape is truncated before this time (UTC)")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise RuntimeError("BASELINE_OUTPUT_EXISTS")
    start = pd.Timestamp(args.eval_start)
    end = pd.Timestamp(args.read_end_exclusive)
    if start.tzinfo is None or end.tzinfo is None:
        raise RuntimeError("BASELINE_TIMES_MUST_BE_UTC_AWARE")
    tape = load_tape(args.native_m5_root, truncate_before=end)
    policy = load_cost_policy(args.cost_policy, args.cost_policy_sha256)
    scalp = scalp_arm(tape, policy, start)
    swing = swing_arm(tape, policy, start)
    verdicts = {name: cell["verdict"] for name, cell in {**scalp, **swing}.items()}
    result = {
        "schema_version": "gx1_model_free_baselines_v1",
        "preregistration": "docs/MODEL_FREE_BASELINES_PREREG_20260927.md",
        "source_commit": _git("rev-parse", "HEAD"),
        "source_clean": _git("status", "--porcelain") == "",
        "tape": {"root": str(args.native_m5_root), "manifest_sha256": tape.manifest_sha256,
                 "first_bar": str(tape.time[0]), "last_bar": str(tape.time[-1]), "bars": len(tape.time)},
        "cost_policy": {k: v for k, v in policy.items() if k != "decision"},
        "eval_start": str(start),
        "read_end_exclusive": str(end),
        "cells": {"scalp": scalp, "swing": swing},
        "decision": {
            "scalp_go": sorted(n for n, v in verdicts.items() if v == "GO" and n in scalp),
            "swing_go": sorted(n for n, v in verdicts.items() if v == "GO" and n in swing),
            "lovende": sorted(n for n, v in verdicts.items() if v == "LOVENDE"),
        },
    }
    args.out_dir.mkdir(parents=True)
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False, default=float)
    (args.out_dir / "results.json").write_text(payload + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "results_sha256": hashlib.sha256(payload.encode()).hexdigest()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
