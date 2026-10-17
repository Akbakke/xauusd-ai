"""M5 intraday recheck on the NATIVE_D1_003 generation (operator request 15.10.2026; preregistration
configs/research/M5_INTRADAY_RECHECK_20261015_001.json).

A bounded offline research arm, not a native authority: decisions after the closed M5 bar, horizons of 30/120/360
minutes, a ridge and a GBDT reference on the same rows, targets and walk-forward periods, feature set A (the
existing 254-field surface) against B (A + the 61 D1 macro/positioning/panel fields), and a one-position trade
simulation on executable M1 bid/ask with declared latency, slippage and financing. Everything that decides is read
from the bound spec; normalization, ridge alpha, GBDT round count and thresholds come only from each train window
and its inner validation. The sealed TEST split is never opened.

Why a new module (rule 21): the existing walk-forward instrument (research_entry_direction_walkforward_v1) is a
closed one-off study on M5 close prices; this arm needs M1 bid/ask execution with latency, rollover financing, a
one-position trade simulator and a GBDT reference, none of which that owner has. It reuses no decision-bearing code.

Run: scripts/gx1_capped_run.sh --class producer --mem 12G --swap 512M -- .venv/bin/python -m
     gx1.scripts.research_m5_intraday_recheck_v1 --spec configs/research/M5_INTRADAY_RECHECK_20261015_001.json
     --spec-sha256 <sha> --out-dir <run root>/RESULT_001
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time as _time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

SCHEMA_VERSION = "gx1_m5_intraday_recheck_v1"
NS = 10**9
MINUTE_NS = 60 * NS
DECISION_OFFSET_NS = 300 * NS
ENTRY_TOLERANCE_NS = 300 * NS
EXIT_TOLERANCE_NS = 4200 * NS
WEEKLY_GAP_NS = 6 * 3600 * NS
SPREAD_STALENESS_NS = 15 * MINUTE_NS


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def require_binding(binding: dict[str, str]) -> Path:
    path = Path(binding["path"])
    if not path.is_file() or sha256_file(path) != binding["sha256"]:
        raise RuntimeError(f"M5_RECHECK_BINDING_MISMATCH: {path}")
    return path


def load_spec(path: Path, expected_sha256: str) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        raise RuntimeError("M5_RECHECK_SPEC_HASH_MISMATCH")
    spec = json.loads(path.read_text())
    if spec.get("schema_version") != SCHEMA_VERSION or spec["data"]["test_split_read"] is not False:
        raise RuntimeError("M5_RECHECK_SPEC_INVALID")
    if spec["execution"]["max_open_positions"] != 1 or spec["execution"]["no_mfe_or_optimal_exit"] is not True:
        raise RuntimeError("M5_RECHECK_SPEC_EXECUTION_INVALID")
    return spec


# --------------------------------------------------------------------------------------------- data
@dataclass
class Decisions:
    row_time_ns: np.ndarray      # M5 bar open
    decision_ns: np.ndarray      # bar close (open + 300 s)
    snap: np.ndarray             # float32 (n, 315)
    session: np.ndarray          # int


def load_decisions(spec: dict[str, Any]) -> Decisions:
    times, snaps, sessions = [], [], []
    width = None
    for split in spec["data"]["splits_used"]:
        binding = spec["data"]["datasets"][split]
        require_binding(binding["manifest"])
        path = require_binding(binding["parquet"])
        f = pq.ParquetFile(path)
        for rg in range(f.metadata.num_row_groups):
            t = f.read_row_group(rg, columns=["time", "snap", "ctx_cat"])
            times.append(t.column("time").to_numpy().astype("datetime64[ns]").astype(np.int64))
            flat = pc.list_flatten(t.column("snap")).to_numpy(zero_copy_only=False).astype(np.float32)
            w = flat.size // t.num_rows
            if width is None:
                width = w
            if w != width or flat.size != w * t.num_rows:
                raise RuntimeError("M5_RECHECK_SNAP_WIDTH")
            snaps.append(flat.reshape(t.num_rows, w))
            cat = pc.list_flatten(t.column("ctx_cat")).to_numpy(zero_copy_only=False).astype(np.int64)
            if cat.size != t.num_rows:
                raise RuntimeError("M5_RECHECK_SESSION_WIDTH")
            sessions.append(cat)
    row_time = np.concatenate(times)
    order = np.argsort(row_time, kind="stable")
    row_time = row_time[order]
    if np.any(np.diff(row_time) <= 0):
        raise RuntimeError("M5_RECHECK_DECISION_CLOCK_NOT_STRICT")
    end = pd.Timestamp(spec["data"]["development_period_utc"][1]).value
    if row_time[-1] + DECISION_OFFSET_NS > end:
        raise RuntimeError("M5_RECHECK_DECISION_BEYOND_DEVELOPMENT_PERIOD")
    snap = np.vstack(snaps)[order]
    session = np.concatenate(sessions)[order]
    return Decisions(row_time, row_time + DECISION_OFFSET_NS, snap, session)


@dataclass
class Tape:
    time_ns: np.ndarray
    bid_open: np.ndarray
    ask_open: np.ndarray
    bid_close: np.ndarray
    ask_close: np.ndarray


def load_tape(spec: dict[str, Any], start_year: int) -> Tape:
    tape = spec["data"]["execution_tape"]
    require_binding(tape["manifest"])
    root = Path(tape["root"])
    files = sorted(p for p in root.glob("year=*/*.parquet") if int(p.parent.name.split("=", 1)[1]) >= start_year)
    if not files:
        raise RuntimeError("M5_RECHECK_TAPE_EMPTY")
    cols = tape["columns"]
    frames = [pq.read_table(str(p), columns=cols).to_pandas() for p in files]
    df = pd.concat(frames, ignore_index=True)
    t = pd.to_datetime(df["time"], utc=True).astype("int64").to_numpy()
    order = np.argsort(t, kind="stable")
    t = t[order]
    if np.any(np.diff(t) <= 0):
        raise RuntimeError("M5_RECHECK_TAPE_NOT_STRICT")
    if t[-1] >= pd.Timestamp(tape["end_utc_exclusive"]).value:
        raise RuntimeError("M5_RECHECK_TAPE_BEYOND_TEST_SEAL")
    arr = {c: df[c].to_numpy(np.float64)[order] for c in cols if c != "time"}
    if not all(np.isfinite(v).all() and (v > 0).all() for v in arr.values()):
        raise RuntimeError("M5_RECHECK_TAPE_PRICES_INVALID")
    if np.any(arr["ask_open"] < arr["bid_open"]) or np.any(arr["ask_close"] < arr["bid_close"]):
        raise RuntimeError("M5_RECHECK_TAPE_CROSSED")
    return Tape(t, arr["bid_open"], arr["ask_open"], arr["bid_close"], arr["ask_close"])


# ---------------------------------------------------------------------------------------- execution
@dataclass
class Execution:
    valid: np.ndarray            # bool
    entry_ns: np.ndarray
    exit_ns: np.ndarray
    y_mid_bps: np.ndarray        # learning target (cost-free mid return)
    long_gross_bps: np.ndarray   # executable, spread included, before slippage/financing
    short_gross_bps: np.ndarray
    rollovers: np.ndarray        # 17:00 New York rollovers crossed


def ny_rollovers_crossed(entry_ns: np.ndarray, exit_ns: np.ndarray) -> np.ndarray:
    """Number of 17:00 America/New_York instants in (entry, exit]."""
    def day_index(ns):
        local = pd.DatetimeIndex(ns, tz="UTC").tz_convert("America/New_York").tz_localize(None)
        shifted = local - pd.Timedelta(hours=17)
        return shifted.normalize().asi8 // (86400 * NS)
    return (day_index(exit_ns) - day_index(entry_ns)).astype(np.int64)


def execution_for_horizon(decision_ns: np.ndarray, tape: Tape, *, horizon_minutes: int, latency_seconds: int) -> Execution:
    t = tape.time_ns
    target_entry = decision_ns + latency_seconds * NS
    ei = np.searchsorted(t, target_entry, side="left")
    ok = ei < len(t)
    ei_c = np.minimum(ei, len(t) - 1)
    ok &= (t[ei_c] - target_entry) <= ENTRY_TOLERANCE_NS
    entry_ns = t[ei_c]
    target_exit = entry_ns + horizon_minutes * MINUTE_NS
    xi = np.searchsorted(t, target_exit, side="left")
    ok &= xi < len(t)
    xi_c = np.minimum(xi, len(t) - 1)
    ok &= (t[xi_c] - target_exit) <= EXIT_TOLERANCE_NS
    big_gap = np.r_[0, (np.diff(t) > WEEKLY_GAP_NS).astype(np.int64)]
    cum = np.cumsum(big_gap)
    ok &= (cum[xi_c] - cum[ei_c]) == 0
    exit_ns = t[xi_c]
    mid_in = (tape.bid_open[ei_c] + tape.ask_open[ei_c]) / 2
    mid_out = (tape.bid_open[xi_c] + tape.ask_open[xi_c]) / 2
    y = 1e4 * np.log(mid_out / mid_in)
    long_gross = 1e4 * (tape.bid_open[xi_c] / tape.ask_open[ei_c] - 1.0)
    short_gross = 1e4 * (tape.bid_open[ei_c] / tape.ask_open[xi_c] - 1.0)
    roll = ny_rollovers_crossed(entry_ns, exit_ns)
    for a in (y, long_gross, short_gross):
        a[~ok] = np.nan
    return Execution(ok, entry_ns, exit_ns, y, long_gross, short_gross, roll)


# Explicit research successor. Historical v1 callers/results remain unchanged.
EXECUTION_ACCOUNTING_V2 = "gx1_m5_entry_notional_accounting_v2"

def execution_for_horizon_v2(decision_ns: np.ndarray, tape: Tape, *, horizon_minutes: int,
                             latency_seconds: int) -> Execution:
    """Same v1 timestamps/masks/mid-label; both cash sides use entry notional.

    This deliberately does not repair the old route or silently alter its
    300/4200-second timing tolerances. A future P0 recipe must bind this name.
    """
    from gx1.contracts.entry_causal_m1_outcomes_v1 import _entry_notional_pnl_bps
    ex = execution_for_horizon(decision_ns, tape, horizon_minutes=horizon_minutes,
                               latency_seconds=latency_seconds)
    entry = np.searchsorted(tape.time_ns, ex.entry_ns)
    end = np.searchsorted(tape.time_ns, ex.exit_ns)
    long, short = _entry_notional_pnl_bps(
        entry_bid=tape.bid_open[entry], entry_ask=tape.ask_open[entry],
        exit_bid=tape.bid_open[end], exit_ask=tape.ask_open[end])
    long[~ex.valid] = np.nan; short[~ex.valid] = np.nan
    return Execution(ex.valid.copy(), ex.entry_ns.copy(), ex.exit_ns.copy(),
                     ex.y_mid_bps.copy(), long, short, ex.rollovers.copy())


def net_trade_bps_v2(side: int, long_gross: float, short_gross: float, rollovers: int, *,
                     slippage_bps: float, financing: dict[str, float]) -> float:
    """Version-2 rollover comparator; v3 owns the prepared P0 cash projection.

    Only with v2 gross; spread is already in bid/ask.

    The declared rollover model is reused, not asserted to be historical swap
    truth. Commission remains zero in the bound research recipe.
    """
    if side not in (-1, 1) or not np.isfinite(slippage_bps) or slippage_bps < 0 or rollovers < 0:
        raise ValueError('M5_V2_EXECUTION_COST_INPUT')
    return net_trade_bps(side, long_gross, short_gross, rollovers,
                         slippage_bps=slippage_bps, financing=financing)


ECONOMICS_PROJECTION_V3 = "gx1_m5_entry_notional_current_terms_projection_v3"

def net_cash_values_v3(execution: Execution, *, economics: dict) -> np.ndarray:
    """Separate research projection through the existing native cash-cost owner.

    LONG/SHORT gross must come from execution_for_horizon_v2. Wall-clock
    financing, clipped favorable credits and execution fees follow the frozen
    parameter authority. This is NOT a historical broker-billing reconstruction.
    Invalid rows remain NaN, including FLAT; future outcomes are never features.
    """
    import torch
    from gx1.contracts.entry_observed_market_v1 import build_entry_observed_market_targets
    valid=np.asarray(execution.valid,bool)
    out=np.full((len(valid),3),np.nan,dtype=np.float64)
    if valid.any():
        elapsed=(execution.exit_ns[valid]-execution.entry_ns[valid])//NS
        gross=np.column_stack((execution.long_gross_bps[valid],execution.short_gross_bps[valid]))
        result=build_entry_observed_market_targets(
            gross_return_bps=torch.tensor(gross,dtype=torch.float64,device='cpu'),
            elapsed_wall_clock_seconds=torch.tensor(elapsed,dtype=torch.int64,device='cpu'),economics=economics)
        out[valid]=result['target_bps'].numpy()
    return out




def exact_intraday_mask_v3(decision_ns, tape, execution, *, horizon_minutes, latency_seconds=60):
    """Research-only exact endpoints and fully observed M1 path, no calendar guess.

    Rejects scheduled closures and unclassified gaps alike. This eligibility
    variant does not change old v1/v2 execution, any target or a trading hold rule.
    """
    t=tape.time_ns
    ei=np.minimum(np.searchsorted(t,execution.entry_ns),len(t)-1)
    xi=np.minimum(np.searchsorted(t,execution.exit_ns),len(t)-1)
    gaps=np.r_[0,np.cumsum(np.diff(t)!=MINUTE_NS)]
    interior=gaps[xi]-gaps[ei]
    entry_delay=execution.entry_ns-np.asarray(decision_ns)-latency_seconds*NS
    terminal_delay=execution.exit_ns-execution.entry_ns-horizon_minutes*MINUTE_NS
    valid=execution.valid&(entry_delay==0)&(terminal_delay==0)&(interior==0)
    return {'valid':valid,'entry_delay_ns':entry_delay,'terminal_delay_ns':terminal_delay,
            'interior_gap_intervals':interior}


def spread_at_decision(decision_ns: np.ndarray, tape: Tape) -> np.ndarray:
    """Spread (bps of mid) of the last M1 bar closed at the decision: bar open <= decision - 60 s."""
    t = tape.time_ns
    si = np.searchsorted(t, decision_ns - MINUTE_NS, side="right") - 1
    ok = si >= 0
    si_c = np.maximum(si, 0)
    ok &= (decision_ns - t[si_c]) <= SPREAD_STALENESS_NS
    mid = (tape.bid_close[si_c] + tape.ask_close[si_c]) / 2
    out = 1e4 * (tape.ask_close[si_c] - tape.bid_close[si_c]) / mid
    out[~ok] = np.nan
    return out


def net_trade_bps(side: int, long_gross: float, short_gross: float, rollovers: int, *, slippage_bps: float,
                  financing: dict[str, float]) -> float:
    """Executable gross (spread already inside bid/ask) minus slippage on both executions plus financing."""
    if side > 0:
        return long_gross - 2 * slippage_bps + rollovers * financing["long_bps_per_rollover"]
    return short_gross - 2 * slippage_bps + rollovers * financing["short_bps_per_rollover"]


# ------------------------------------------------------------------------------------------- folds
@dataclass
class Fold:
    name: str
    eval_start_ns: int
    eval_end_ns: int
    train: np.ndarray
    inner_train: np.ndarray
    inner_val: np.ndarray
    eval: np.ndarray


def build_fold(name: str, eval_start: str, eval_end: str, decision_ns: np.ndarray, exit_ns: np.ndarray,
               valid: np.ndarray, *, train_years: int, inner_fraction: float = 0.2) -> Fold:
    s = pd.Timestamp(eval_start, tz="UTC").value
    e = pd.Timestamp(eval_end, tz="UTC").value
    w0 = (pd.Timestamp(eval_start, tz="UTC") - pd.DateOffset(years=train_years)).value
    embargo = s - 86400 * NS
    train = np.flatnonzero(valid & (decision_ns >= w0) & (decision_ns < embargo) & (exit_ns < s))
    if len(train) < 1000:
        raise RuntimeError(f"M5_RECHECK_FOLD_TOO_SMALL: {name}")
    cut = train[int(len(train) * (1 - inner_fraction))]
    inner_val = train[decision_ns[train] >= decision_ns[cut]]
    iv_start = decision_ns[inner_val[0]]
    inner_train = train[(decision_ns[train] < iv_start) & (exit_ns[train] < iv_start)]
    ev = np.flatnonzero((decision_ns >= s) & (decision_ns < e))
    return Fold(name, s, e, train, inner_train, inner_val, ev)


def period_name(start: str, end: str) -> str:
    """'2016' for a calendar year, '2025-01_2026-07' otherwise."""
    one_year = start[5:] == "01-01" and end[5:] == "01-01" and int(end[:4]) == int(start[:4]) + 1
    return start[:4] if one_year else f"{start[:7]}_{end[:7]}"


# ------------------------------------------------------------------------------------------ models
def design_matrix(snap: np.ndarray, session: np.ndarray, width: int, session_values: np.ndarray) -> np.ndarray:
    onehot = (session[:, None] == session_values[None, :]).astype(np.float32)
    return np.hstack([snap[:, :width], onehot])


def standardizer(x: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sub = x[rows].astype(np.float64)
    mean = np.nanmean(sub, axis=0)
    std = np.nanstd(sub, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > 0), std, 1.0)
    return mean, std


def standardize(x: np.ndarray, rows: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    z = (x[rows].astype(np.float64) - mean) / std
    z[~np.isfinite(z)] = 0.0
    return z


def ridge_fit(z: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    intercept = float(np.mean(y))
    zc = z - z.mean(axis=0)
    gram = zc.T @ zc
    beta = np.linalg.solve(gram + alpha * np.eye(gram.shape[0]), zc.T @ (y - intercept))
    return beta, intercept - float(z.mean(axis=0) @ beta)


def ridge_select_and_fit(x: np.ndarray, y: np.ndarray, fold: Fold, grid: list[float], *, return_state: bool = False) -> dict[str, Any]:
    m, s = standardizer(x, fold.inner_train)
    zi, zv = standardize(x, fold.inner_train, m, s), standardize(x, fold.inner_val, m, s)
    scores = {}
    for a in grid:
        b, c = ridge_fit(zi, y[fold.inner_train], a)
        scores[a] = float(np.mean((zv @ b + c - y[fold.inner_val]) ** 2))
    best = min(grid, key=lambda a: (scores[a], a))
    b_in, c_in = ridge_fit(zi, y[fold.inner_train], best)
    inner_pred = zv @ b_in + c_in
    m, s = standardizer(x, fold.train)
    b, c = ridge_fit(standardize(x, fold.train, m, s), y[fold.train], best)
    pred = standardize(x, fold.eval, m, s) @ b + c
    result = {"alpha": best, "inner_mse": scores, "pred": pred, "inner_pred": inner_pred}
    if return_state:
        result.update(beta=b, intercept=c, mean=m, std=s,
                      train_pred=standardize(x, fold.train, m, s) @ b+c)
    return result


def gbdt_select_and_fit(x: np.ndarray, y: np.ndarray, fold: Fold, cfg: dict[str, Any]) -> dict[str, Any]:
    import lightgbm as lgb
    params = dict(cfg["params"])
    dtr = lgb.Dataset(x[fold.inner_train], y[fold.inner_train], free_raw_data=True)
    dva = lgb.Dataset(x[fold.inner_val], y[fold.inner_val], reference=dtr, free_raw_data=True)
    booster = lgb.train(params, dtr, num_boost_round=cfg["max_rounds"], valid_sets=[dva],
                        callbacks=[lgb.early_stopping(cfg["early_stopping_rounds"], verbose=False)])
    rounds = max(int(booster.best_iteration or 0), 1)
    inner_pred = booster.predict(x[fold.inner_val], num_iteration=rounds)
    full = lgb.train(params, lgb.Dataset(x[fold.train], y[fold.train], free_raw_data=True), num_boost_round=rounds)
    return {"rounds": rounds, "pred": full.predict(x[fold.eval]), "inner_pred": inner_pred}


# ----------------------------------------------------------------------------------------- trading
def simulate(order_rows: np.ndarray, side_of_row: np.ndarray, ex: Execution, decision_ns: np.ndarray,
             spec: dict[str, Any]) -> list[dict[str, Any]]:
    """One open position; a row is considered only when flat at its decision time and its execution is valid."""
    fin = spec["execution"]["financing"]
    slips = spec["execution"]["slippage_bps_per_execution"]["sensitivity"]
    trades, busy_until = [], -1
    for r, side in zip(order_rows, side_of_row):
        if side == 0 or not ex.valid[r]:
            continue
        if decision_ns[r] < busy_until:
            continue
        rec = {"row": int(r), "entry_ns": int(ex.entry_ns[r]), "exit_ns": int(ex.exit_ns[r]), "side": int(side),
               "gross_bps": float(ex.long_gross_bps[r] if side > 0 else ex.short_gross_bps[r]),
               "rollovers": int(ex.rollovers[r])}
        for s in slips:
            rec[f"net_bps_slip{s:g}"] = net_trade_bps(side, ex.long_gross_bps[r], ex.short_gross_bps[r], ex.rollovers[r],
                                                     slippage_bps=s, financing=fin)
        trades.append(rec)
        busy_until = ex.exit_ns[r]
    return trades


def trade_summary(trades: list[dict[str, Any]], period_ns: int, slips: list[float]) -> dict[str, Any]:
    out: dict[str, Any] = {"trades": len(trades)}
    if not trades:
        return out
    held = sum(t["exit_ns"] - t["entry_ns"] for t in trades)
    out["exposure"] = held / period_ns
    out["long_share"] = float(np.mean([t["side"] > 0 for t in trades]))
    for s in slips:
        net = np.array([t[f"net_bps_slip{s:g}"] for t in trades])
        cum = np.cumsum(net)
        peak = np.maximum.accumulate(np.r_[0.0, cum])[1:]
        out[f"slip{s:g}"] = {"net_total_bps": float(net.sum()), "net_per_trade_bps": float(net.mean()),
                              "win_rate": float(np.mean(net > 0)), "max_drawdown_bps": float(np.max(peak - cum))}
    return out


def weekly_block_bootstrap(trades: list[dict[str, Any]], key: str, *, reps: int, seed: int, levels: list[float]) -> dict[str, Any]:
    if len(trades) < 2:
        return {}
    weeks = pd.DatetimeIndex([t["entry_ns"] for t in trades], tz="UTC").to_period("W").astype(str)
    df = pd.DataFrame({"w": weeks, "v": [t[key] for t in trades]})
    g = df.groupby("w")["v"].agg(["sum", "count"])
    sums, counts = g["sum"].to_numpy(), g["count"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(reps, len(sums)))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    out = {"weeks": int(len(sums)), "point": float(sums.sum() / counts.sum())}
    for lv in levels:
        a = (1 - lv) / 2
        out[f"ci{lv:.4f}"] = [float(np.quantile(means, a)), float(np.quantile(means, 1 - a))]
    return out


def classify(cell: dict[str, Any], spec: dict[str, Any]) -> str:
    pooled = cell["pooled"]
    n = pooled.get("trades", 0)
    if n < 30:
        return "INCONCLUSIVE"
    boot = cell["bootstrap_central"]
    fam = boot[f"ci{1 - 0.10 / 12:.4f}"]
    raw = boot["ci0.9000"]
    periods = [p for p in cell["periods"].values() if p["trading"].get("trades", 0) >= 5]
    pos_share = np.mean([p["trading"]["slip2"]["net_per_trade_bps"] > 0 for p in periods]) if periods else 0.0
    if raw[1] < 0:
        return "NEGATIVE"
    if fam[0] > 0 and pos_share >= 0.6 and pooled["slip3"]["net_per_trade_bps"] > 0:
        return "PROMISING"
    return "INCONCLUSIVE"


# -------------------------------------------------------------------------------------------- main
def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


def run(spec: dict[str, Any], spec_sha: str, out_dir: Path) -> dict[str, Any]:
    t0 = _time.monotonic()

    def log(m: str) -> None:
        print(json.dumps({"t": round(_time.monotonic() - t0, 1), "event": m}), flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "cache"
    cache.mkdir(exist_ok=True)
    dec = load_decisions(spec)
    log(f"decisions {len(dec.decision_ns)} snap {dec.snap.shape}")
    tape = load_tape(spec, start_year=pd.Timestamp(dec.decision_ns[0]).year)
    log(f"tape {len(tape.time_ns)}")
    spread = spread_at_decision(dec.decision_ns, tape)
    session_values = np.unique(dec.session)
    sets = {k: v["snap_slice"][1] for k, v in spec["feature_sets"].items() if "snap_slice" in v}
    slips = spec["execution"]["slippage_bps_per_execution"]["sensitivity"]
    central = spec["execution"]["slippage_bps_per_execution"]["central"]
    wf = spec["walk_forward"]
    result: dict[str, Any] = {"schema_version": SCHEMA_VERSION + "_result", "spec_sha256": spec_sha, "cells": {}, "baselines": {},
                              "test_data_used": False, "decisions": int(len(dec.decision_ns)),
                              "spread_bps_median": float(np.nanmedian(spread))}
    for h in spec["horizons_minutes"]:
        ex = execution_for_horizon(dec.decision_ns, tape, horizon_minutes=h, latency_seconds=spec["execution"]["latency_seconds"])
        log(f"h{h}: valid {int(ex.valid.sum())} of {len(ex.valid)}")
        folds = [build_fold(period_name(a, b), a, b, dec.decision_ns, ex.exit_ns, ex.valid, train_years=wf["train_window_years"])
                 for a, b in wf["evaluation_periods_utc"]]
        # drift baselines with the same sequencing and costs
        for bname, side in (("always_long", 1), ("always_short", -1)):
            per = {}
            all_tr = []
            for fold in folds:
                tr = simulate(fold.eval, np.full(len(fold.eval), side), ex, dec.decision_ns, spec)
                all_tr += tr
                per[fold.name] = trade_summary(tr, fold.eval_end_ns - fold.eval_start_ns, slips)
            result["baselines"][f"{bname}_h{h}"] = {"periods": per, "pooled": trade_summary(all_tr, sum(f.eval_end_ns - f.eval_start_ns for f in folds), slips)}
        for set_name, width in sets.items():
            x = design_matrix(dec.snap, dec.session, width, session_values)
            for model in ("ridge", "gbdt"):
                key = f"{model}_{set_name}_h{h}"
                cell = {"periods": {}, "model": model, "feature_set": set_name, "horizon_minutes": h}
                trades_all, trades_sec = [], []
                for fold in folds:
                    cpath = cache / f"{key}_{fold.name}.npz"
                    if cpath.exists():
                        z = np.load(cpath, allow_pickle=False)
                        if str(z["spec_sha256"]) != spec_sha:
                            raise RuntimeError("M5_RECHECK_CACHE_FROM_OTHER_SPEC")
                        pred, inner_pred, meta = z["pred"], z["inner_pred"], json.loads(str(z["meta"]))
                    else:
                        y = ex.y_mid_bps
                        if model == "ridge":
                            r = ridge_select_and_fit(x, y, fold, spec["models"]["ridge"]["alpha_grid"])
                            meta = {"alpha": r["alpha"], "inner_mse": {str(k): v for k, v in r["inner_mse"].items()}}
                        else:
                            r = gbdt_select_and_fit(x, y, fold, spec["models"]["gbdt"])
                            meta = {"rounds": r["rounds"]}
                        pred, inner_pred = np.asarray(r["pred"], float), np.asarray(r["inner_pred"], float)
                        np.savez(cpath, pred=pred, inner_pred=inner_pred, meta=json.dumps(meta), spec_sha256=spec_sha)
                    ev = fold.eval
                    ok = ex.valid[ev]
                    yv, pv = ex.y_mid_bps[ev][ok], pred[ok]
                    pq_ = {"rows": int(ok.sum()), "rank_ic": spearman(pv, yv), "pearson": float(np.corrcoef(pv, yv)[0, 1]),
                           "sign_hit": float(np.mean(np.sign(pv[yv != 0]) == np.sign(yv[yv != 0]))),
                           "r2_vs_zero": float(1 - np.sum((yv - pv) ** 2) / np.sum(yv ** 2))}
                    c = spread[ev] + 2 * central
                    side = np.where(pred > c, 1, np.where(pred < -c, -1, 0))
                    side[~np.isfinite(c)] = 0
                    tr = simulate(ev, side, ex, dec.decision_ns, spec)
                    thr = float(np.quantile(np.abs(inner_pred), 0.9))
                    side2 = np.where(np.abs(pred) > thr, np.sign(pred), 0).astype(int)
                    tr2 = simulate(ev, side2, ex, dec.decision_ns, spec)
                    trades_all += tr
                    trades_sec += tr2
                    cell["periods"][fold.name] = {"prediction": pq_, "model_meta": meta, "train_rows": int(len(fold.train)),
                                                  "inner_val_rows": int(len(fold.inner_val)),
                                                  "trading": trade_summary(tr, fold.eval_end_ns - fold.eval_start_ns, slips),
                                                  "secondary": trade_summary(tr2, fold.eval_end_ns - fold.eval_start_ns, slips),
                                                  "secondary_threshold_bps": thr}
                    log(f"{key} {fold.name}: ic {pq_['rank_ic']:+.4f} trades {len(tr)} meta {meta if model == 'gbdt' else meta['alpha']}")
                total_ns = sum(f.eval_end_ns - f.eval_start_ns for f in folds)
                cell["pooled"] = trade_summary(trades_all, total_ns, slips)
                cell["secondary_pooled"] = trade_summary(trades_sec, total_ns, slips)
                ics = np.array([p["prediction"]["rank_ic"] for p in cell["periods"].values()])
                cell["rank_ic_across_periods"] = {"mean": float(ics.mean()), "sd": float(ics.std(ddof=1)),
                                                  "t": float(ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics)))), "positive_periods": int((ics > 0).sum()),
                                                  "periods": int(len(ics))}
                levels = [0.90, 1 - 0.10 / 12]
                cell["bootstrap_central"] = weekly_block_bootstrap(trades_all, f"net_bps_slip{central:g}", reps=4000, seed=20261015, levels=levels)
                cell["classification"] = classify(cell, spec)
                pd.DataFrame(trades_all).to_parquet(out_dir / f"TRADES_{key}.parquet", index=False)
                result["cells"][key] = cell
                log(f"{key}: {cell['classification']} trades {cell['pooled'].get('trades')} ic_mean {cell['rank_ic_across_periods']['mean']:+.4f}")
    # A versus B
    comp = {}
    for model in ("ridge", "gbdt"):
        for h in spec["horizons_minutes"]:
            a, b = result["cells"].get(f"{model}_A_h{h}"), result["cells"].get(f"{model}_B_h{h}")
            if not a or not b:
                continue
            d = [b["periods"][k]["prediction"]["rank_ic"] - a["periods"][k]["prediction"]["rank_ic"] for k in a["periods"]]
            comp[f"{model}_h{h}"] = {"rank_ic_diff_mean": float(np.mean(d)), "rank_ic_diff_t": float(np.mean(d) / (np.std(d, ddof=1) / np.sqrt(len(d)))),
                                     "b_better_periods": int(np.sum(np.array(d) > 0)), "periods": len(d),
                                     "net_per_trade_A": a["pooled"].get("slip2", {}).get("net_per_trade_bps"),
                                     "net_per_trade_B": b["pooled"].get("slip2", {}).get("net_per_trade_bps")}
    result["a_versus_b"] = comp
    result["elapsed_seconds"] = round(_time.monotonic() - t0, 1)
    tmp = out_dir / ".RESULT.json.tmp"
    tmp.write_text(json.dumps(result, indent=1))
    tmp.replace(out_dir / "RESULT.json")
    return result


# This bounded mode extends the existing M5 ridge/clock/label/uncertainty
# owner. It never runs the original one-off study or its economic simulation.
GC_PROBE_ID = 'GC_M5_TRADE_FLOW_PROBE_20261017_007'
GC_PROBE_CODE = (
    'gx1/scripts/research_m5_intraday_recheck_v1.py', 'tests/test_gc_m5_trade_probe_v1.py',
    'gx1/scripts/gc_source_disposition_v1.py', 'gx1/contracts/gc_source_disposition_v1.py',
    'gx1/contracts/gc_order_flow_source_v1.py', 'gx1/scripts/research_ta_campaign_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py', 'scripts/gx1_capped_run.sh',
    'gx1/contracts/entry_model_native_bundle_commit_v1.py',
)


def gc_input_matrices(snap, context, session, fields, ctx_names, gc, candidates, common_masks, initial_train):
    """Freeze input-only cohort/fields before reading a single outcome.

    All 315 base fields stay. Each context alias is checked on every cohort
    row and consumes its existing snap column, never a second scaler owner.
    New GC constants/unobserved fields are excluded using only initial TRAIN.
    """
    if not np.isfinite(snap).all() or snap.shape[1] != len(fields):
        raise RuntimeError('GC_PROBE_NATIVE_SNAP_INVALID')
    for j,name in enumerate(ctx_names):
        k=fields.index('ctx_cont.'+name)
        if not np.array_equal(snap[:,k],context[:,j]):
            raise RuntimeError('GC_PROBE_CONTEXT_ALIAS_MISMATCH:'+name)
    categories=np.unique(session[initial_train])
    if not np.isin(session,categories).all():
        raise RuntimeError('GC_PROBE_SESSION_DOMAIN_NOT_IN_INITIAL_TRAIN')
    x0=design_matrix(snap,session,len(fields),categories).astype(np.float64)
    base_names=list(fields)+[f'session_id={int(v)}' for v in categories]
    common=np.column_stack([gc[n] for n in common_masks]).astype(np.float64)
    if not np.isfinite(common).all():
        raise RuntimeError('GC_PROBE_COMMON_AVAILABILITY_INVALID')
    x0=np.column_stack((x0,common))
    names0=base_names+common_masks
    matrices={'A':x0}; names={'A':names0}; excluded={}
    for arm,previous_arm in (('B','A'),('C','B')):
        arrays=[]; added=[]
        for name in candidates[arm]:
            values=gc[name].astype(np.float64)
            if np.isinf(values).any():raise RuntimeError('GC_PROBE_INFINITE_FEATURE:'+name)
            observed=values[initial_train]
            observed=observed[np.isfinite(observed)]
            if len(np.unique(observed))<2:
                excluded[name]='NO_VARIATION_OR_OBSERVATION_IN_INITIAL_TRAIN_INPUTS'
                continue
            arrays.extend((values,(~np.isfinite(values)).astype(np.float64)))
            added.extend((name,name+'__missing'))
        if not arrays:
            raise RuntimeError('GC_PROBE_EMPTY_ARM_INCREMENT:'+arm)
        matrices[arm]=np.column_stack((matrices[previous_arm],*arrays))
        names[arm]=names[previous_arm]+added
    return matrices,names,excluded


def gc_probe_runtime(binding_path):
    import os,subprocess
    from gx1.scripts import gc_source_disposition_v1 as io
    from gx1.contracts.gx1_capped_execution_v1 import require_capped_cpu_audit_execution,require_capped_lock_ancestry
    root=Path('/home/andre2/GX1_RUNS/GC_ORDER_FLOW_RESEARCH_001')/GC_PROBE_ID
    if binding_path != root/'EXECUTION_BINDING.json':
        raise RuntimeError('GC_PROBE_BINDING_PATH_INVALID')
    binding=io.read_json(binding_path)
    plan_path=io.REPO/f'configs/research/{GC_PROBE_ID}.json'
    io.verify_digest(plan_path,binding['plan_sha256'],'PROBE_PLAN')
    spec=io.read_json(plan_path)
    io.verify_digest(root/'OPERATOR_APPROVAL.json',binding['approval_sha256'],'PROBE_APPROVAL')
    approval=io.read_json(root/'OPERATOR_APPROVAL.json')
    io.verify_digest(io.REPO/'NEXT_RUN_POLICY.json',binding['policy_sha256'],'PROBE_POLICY')
    policy=io.read_json(io.REPO/'NEXT_RUN_POLICY.json')
    gate=policy['current_work']['gc_m5_trade_probe_007']
    if (spec['study_id']!=GC_PROBE_ID or binding['source_head']!=io.git_head()
            or policy['training_enabled'] is not False or gate['status']!='EXECUTION_CLAIMED'
            or gate['claimed_invocations']!=1 or gate['completed_invocations']!=0
            or gate['execution_authorized'] is not False or gate['native_training_authorized'] is not False
            or approval['approved_plan_sha256']!=binding['plan_sha256']
            or approval['single_use'] is not True or approval['max_invocations']!=1
            or approval['status']!='APPROVED_SINGLE_USE_PENDING_EXECUTION'):
        raise RuntimeError('GC_PROBE_SINGLE_USE_SCOPE_INVALID')
    if set(binding['code_sha256'])!=set(GC_PROBE_CODE):
        raise RuntimeError('GC_PROBE_CODE_INVENTORY_INVALID')
    for name,digest in binding['code_sha256'].items():io.verify_digest(io.REPO/name,digest,'PROBE_CODE')
    if (subprocess.check_output(['git','status','--porcelain'],cwd=io.REPO,text=True)
            or subprocess.check_output(['git','branch','--show-current'],cwd=io.REPO,text=True).strip()!='work/gx1-current'):
        raise RuntimeError('GC_PROBE_SOURCE_IDENTITY_INVALID')
    proof=require_capped_cpu_audit_execution();proof['lock']=require_capped_lock_ancestry()
    if proof['cpu_affinity']!=list(range(8)) or (root/'RESULTS').exists() or (root/'.RESULTS.staging').exists():
        raise RuntimeError('GC_PROBE_EXECUTION_OR_OUTPUT_INVALID')
    io.write_json(root/'EXECUTION_CLAIM.json',{'status':'CLAIMED','invocation':1,
        'binding_sha256':io.sha256_file(binding_path),'source_head':binding['source_head']})
    return root,spec,binding,proof


def load_gc_native_rows(spec, table):
    """Column-pruned immutable sources, exact native split/physical row/time."""
    from gx1.scripts import gc_source_disposition_v1 as io
    signal=io.read_json(require_binding(spec['signal_manifest']))['model_native_signal_contract']
    fields,ctx_names=signal['fields'],signal['ctx_cont_names']
    times=table['time'].to_numpy().astype('datetime64[ns]').astype(np.int64)
    indices=table['native_row_index'].to_numpy(); splits=np.array(table['native_split'].to_pylist())
    snap=np.empty((len(times),len(fields)),dtype=np.float64)
    context=np.empty((len(times),len(ctx_names)),dtype=np.float64)
    session=np.empty(len(times),dtype=np.int64)
    for split,item in spec['native_datasets'].items():
        require_binding(item['manifest']); path=require_binding(item['parquet'])
        f=pq.ParquetFile(path); positions=np.flatnonzero(splits==split); wanted=indices[positions]
        if np.any(np.diff(wanted)<=0):raise RuntimeError('GC_PROBE_NATIVE_ROW_ORDER_INVALID')
        offset=found=0
        for rg in range(f.metadata.num_row_groups):
            n=f.metadata.row_group(rg).num_rows
            lo,hi=np.searchsorted(wanted,[offset,offset+n])
            if hi>lo:
                raw=f.read_row_group(rg,columns=['time','snap','ctx_cont','ctx_cat'],use_threads=False)
                local=wanted[lo:hi]-offset; dest=positions[lo:hi]
                t=raw['time'].to_numpy().astype('datetime64[ns]').astype(np.int64)
                if not np.array_equal(t[local],times[dest]):raise RuntimeError('GC_PROBE_NATIVE_ROW_TIME_MISMATCH')
                for name,width,out in (('snap',len(fields),snap),('ctx_cont',len(ctx_names),context)):
                    flat=pc.list_flatten(raw[name]).to_numpy(zero_copy_only=False)
                    if flat.size!=n*width:raise RuntimeError('GC_PROBE_NATIVE_WIDTH_MISMATCH')
                    out[dest]=flat.reshape(n,width)[local]
                flat=pc.list_flatten(raw['ctx_cat']).to_numpy(zero_copy_only=False)
                if flat.size!=n:raise RuntimeError('GC_PROBE_NATIVE_SESSION_WIDTH_MISMATCH')
                session[dest]=flat[local];found+=hi-lo
            offset+=n
        if found!=len(positions):raise RuntimeError('GC_PROBE_NATIVE_ROWS_MISSING')
    return times,snap,context,session,fields,ctx_names


def gc_prediction_metrics(y,p):
    error=float(np.mean((y-p)**2))
    corr=float(pd.Series(y).corr(pd.Series(p),method='spearman')) if np.ptp(p)>0 else float('nan')
    return {'rows':len(y),'mse_bps2':error,'rank_ic':corr if np.isfinite(corr) else None}


def run_gc_trade_probe(binding_path):
    import os
    from gx1.scripts import gc_source_disposition_v1 as io
    from gx1.contracts.entry_model_native_bundle_commit_v1 import publish_bundle_directory_noreplace
    root,spec,binding,proof=gc_probe_runtime(binding_path)
    stage=root/'.RESULTS.staging';(stage/'MODELS').mkdir(parents=True,exist_ok=False)
    started=_time.monotonic()
    try:
        bindings=list(spec['feature_inputs'].values())+[spec['signal_manifest']]+spec['execution_tape_partitions']+[spec['reference']['data']['execution_tape']['manifest']]
        bindings += [b for item in spec['native_datasets'].values() for b in item.values()]
        for item in bindings:io.require_regular_file(Path(item['path']),'PROBE_CANONICAL_INPUT')
        for item in spec['feature_inputs'].values():require_binding(item)
        admitted=io.read_json(Path(spec['feature_inputs']['result']['path']))
        terminal=io.read_json(Path(spec['feature_inputs']['terminal']['path']))
        if (admitted['status']!='FEATURE_BUILD_COMPLETE_NOT_TRAINING_ADMISSION' or terminal['status']!='COMPLETE'
                or terminal['artifacts']['M5_FEATURES.parquet']['sha256']!=spec['feature_inputs']['parquet']['sha256']):
            raise RuntimeError('GC_PROBE_PUBLISHED_FEATURE_ADMISSION_INVALID')
        table=pq.read_table(spec['feature_inputs']['parquet']['path'])
        full_population=table.num_rows
        table=table.filter(table['gc_m5_abc_input_valid'])
        if table.num_rows!=admitted['abc_gc_input_valid_rows']:raise RuntimeError('GC_PROBE_COHORT_OWNER_PARITY_FAILED')
        times,snap,context,session,fields,ctx_names=load_gc_native_rows(spec,table)
        decisions=times+DECISION_OFFSET_NS
        cutoff=pd.Timestamp(spec['folds'][0]['validation_start_utc_inclusive']).value-86400*NS
        initial=np.flatnonzero(decisions<cutoff)
        if len(initial)<spec['minimum_train_rows']:raise RuntimeError('GC_PROBE_INITIAL_TRAIN_TOO_SMALL')
        gc={n:table[n].to_numpy(zero_copy_only=False) for n in spec['common_masks']+spec['candidates']['B']+spec['candidates']['C']}
        matrices,names,excluded=gc_input_matrices(snap,context,session,fields,ctx_names,gc,
            spec['candidates'],spec['common_masks'],initial)
        # This file is created and fsynced before any M1 price/target is read.
        lock={'cohort_rows':len(times),'native_population_rows':full_population,
            'cohort_time_sha256':hashlib.sha256(times.tobytes()).hexdigest(),
            'native_coordinate_sha256':hashlib.sha256(table.select(['time','native_split','native_row_index']).to_pandas().to_json().encode()).hexdigest(),
            'field_order':names,'excluded_new_gc_fields':excluded,'initial_train_rows':len(initial),
            'native_context_aliases_verified':len(ctx_names),'labels_read_at_lock':False}
        io.write_json(stage/'INPUT_LOCK.json',lock)
        print('GC_PROBE_INPUTS_LOCKED '+json.dumps({'rows':len(times),'widths':{k:len(v) for k,v in names.items()}}),flush=True)
        for item in spec['execution_tape_partitions']:require_binding(item)
        actual={str(p) for p in Path(spec['reference']['data']['execution_tape']['root']).glob('year=*/*.parquet') if int(p.parent.name.split('=')[1])>=spec['start_year']}
        if actual!={i['path'] for i in spec['execution_tape_partitions']}:raise RuntimeError('GC_PROBE_UNBOUND_EXECUTION_PARTITION')
        tape=load_tape(spec['reference'],spec['start_year'])
        prediction_frames=[]; cells={}; models=[]
        for horizon in spec['horizons_minutes']:
            ex=execution_for_horizon(decisions,tape,horizon_minutes=horizon,latency_seconds=spec['latency_seconds'])
            for declared in spec['folds']:
                name=declared['fold_id'];start=declared['validation_start_utc_inclusive'][:10];end=declared['validation_end_utc_exclusive'][:10]
                fold=build_fold(name,start,end,decisions,ex.exit_ns,ex.valid,
                    train_years=spec['train_window_years'],inner_fraction=spec['inner_fraction'])
                ev=fold.eval[ex.valid[fold.eval] & (ex.exit_ns[fold.eval]<fold.eval_end_ns)]
                fold=Fold(fold.name,fold.eval_start_ns,fold.eval_end_ns,fold.train,fold.inner_train,fold.inner_val,ev)
                if len(ev)==0 or np.any(ex.exit_ns[fold.train]>=fold.eval_start_ns):
                    raise RuntimeError('GC_PROBE_FOLD_SUPPORT_OR_PURGE_INVALID')
                frame=pd.DataFrame({'time_ns':times[ev],'decision_ns':decisions[ev],'exit_ns':ex.exit_ns[ev],
                    'fold':name,'horizon_minutes':horizon,'y_mid_bps':ex.y_mid_bps[ev],
                    'train_constant_bps':float(ex.y_mid_bps[fold.train].mean())})
                for arm in ('A','B','C'):
                    x=matrices[arm]
                    for train_rows in (fold.inner_train,fold.train):
                        if not np.all(np.isfinite(x[train_rows]).any(axis=0)):
                            raise RuntimeError('GC_PROBE_NO_TRAIN_OWNED_IMPUTATION:'+arm)
                    fit=ridge_select_and_fit(x,ex.y_mid_bps,fold,spec['alpha_grid'],return_state=True)
                    model_name=f'{name}_{arm}_h{horizon}.npz';path=stage/'MODELS'/model_name
                    np.savez(path,beta=fit['beta'],intercept=fit['intercept'],mean=fit['mean'],std=fit['std'])
                    with np.load(path,allow_pickle=False) as loaded:
                        replay=standardize(x,ev,loaded['mean'],loaded['std']) @ loaded['beta']+float(loaded['intercept'])
                    if not np.array_equal(replay,fit['pred']):raise RuntimeError('GC_PROBE_MODEL_READBACK_PARITY_FAILED')
                    frame['pred_'+arm]=fit['pred']
                    key=f'{arm}_h{horizon}';cell=cells.setdefault(key,{'periods':{}})
                    cell['periods'][name]={'train_rows':len(fold.train),'inner_train_rows':len(fold.inner_train),
                        'inner_validation_rows':len(fold.inner_val),'alpha':fit['alpha'],'inner_mse':fit['inner_mse'],
                        'train':gc_prediction_metrics(ex.y_mid_bps[fold.train],fit['train_pred']),
                        'development':gc_prediction_metrics(ex.y_mid_bps[ev],fit['pred'])}
                    models.append({'file':'MODELS/'+model_name,'arm':arm,'horizon_minutes':horizon,'fold':name,
                        'field_order':names[arm],'train_rows_sha256':hashlib.sha256(times[fold.train].tobytes()).hexdigest()})
                    print(f'GC_PROBE_FIT_COMPLETE {key} {name} train={len(fold.train)} development={len(ev)} alpha={fit["alpha"]}',flush=True)
                prediction_frames.append(frame)
        if len(models)!=spec['max_final_models']:raise RuntimeError('GC_PROBE_FINAL_MODEL_INVENTORY_INVALID')
        predictions=pd.concat(prediction_frames,ignore_index=True)
        comparisons={}
        for horizon in spec['horizons_minutes']:
            frame=predictions[predictions.horizon_minutes==horizon]
            for arm in ('A','B','C'):
                cells[f'{arm}_h{horizon}']['pooled_development']=gc_prediction_metrics(frame.y_mid_bps.to_numpy(),frame['pred_'+arm].to_numpy())
            for better,baseline in (('C','B'),('B','A')):
                loss=(frame.y_mid_bps-frame['pred_'+baseline])**2-(frame.y_mid_bps-frame['pred_'+better])**2
                rows=[{'entry_ns':int(t),'improvement_bps2':float(v)} for t,v in zip(frame.decision_ns,loss)]
                boot=weekly_block_bootstrap(rows,'improvement_bps2',reps=spec['bootstrap_reps'],seed=spec['bootstrap_seed'],levels=spec['confidence_levels'])
                family_key=f'ci{spec["confidence_levels"][-1]:.4f}'
                interval=boot.get(family_key)
                decision='INCONCLUSIVE'
                if interval is not None and boot['weeks']>=2:
                    if interval[0]>0:decision='POSITIVE_PAIRED_PREDICTION_CONTRAST_ONLY'
                    elif interval[1]<0:decision='NEGATIVE_PAIRED_PREDICTION_CONTRAST'
                comparisons[f'{better}_minus_{baseline}_h{horizon}']={'bootstrap':boot,'decision':decision,
                    'positive_development_folds':int(sum(float(g.mean())>0 for _,g in loss.groupby(frame.fold)))}
        pred_table=pa_table_from_frame(predictions)
        pq.write_table(pred_table,stage/'PREDICTIONS.parquet',compression='zstd')
        if not pred_table.equals(pq.read_table(stage/'PREDICTIONS.parquet')):raise RuntimeError('GC_PROBE_PREDICTION_READBACK_FAILED')
        # Original immutable inputs, code, policy, approval and head must still match.
        for item in list(spec['feature_inputs'].values())+[spec['signal_manifest']]+spec['execution_tape_partitions']:
            require_binding(item)
        require_binding(spec['reference']['data']['execution_tape']['manifest'])
        for item in spec['native_datasets'].values():
            require_binding(item['manifest']);require_binding(item['parquet'])
        for name,digest in binding['code_sha256'].items():io.verify_digest(io.REPO/name,digest,'PROBE_CODE_POST')
        io.verify_digest(io.REPO/'NEXT_RUN_POLICY.json',binding['policy_sha256'],'PROBE_POLICY_POST')
        io.verify_digest(io.REPO/f'configs/research/{GC_PROBE_ID}.json',binding['plan_sha256'],'PROBE_PLAN_POST')
        io.verify_digest(root/'OPERATOR_APPROVAL.json',binding['approval_sha256'],'PROBE_APPROVAL_POST')
        io.verify_digest(binding_path,io.read_json(root/'EXECUTION_CLAIM.json')['binding_sha256'],'PROBE_BINDING_POST')
        if io.git_head()!=binding['source_head']:raise RuntimeError('GC_PROBE_SOURCE_HEAD_CHANGED')
        result={'status':'MATCHED_TRADE_ONLY_PREDICTION_PROBE_COMPLETE','study_id':GC_PROBE_ID,
            'source_head':binding['source_head'],'plan_sha256':binding['plan_sha256'],'input_lock':lock,
            'cells':cells,'comparisons':comparisons,'models':models,'model_count':len(models),'limits':proof,
            'elapsed_seconds':_time.monotonic()-started,'test_accessed':False,'economics_evaluated':False,
            'native_training_started':False,'research_fit_started':True,'new_spending':False,
            'interpretation':spec['interpretation']}
        io.write_json(stage/'RESULT.json',result)
        inventory={}
        for path in sorted(stage.rglob('*')):
            if path.is_file():
                with path.open('rb') as f:os.fsync(f.fileno())
                inventory[str(path.relative_to(stage))]={'sha256':io.sha256_file(path),'bytes':path.stat().st_size}
        io.write_json(stage/'TERMINAL.json',{'status':'COMPLETE','exit_code':0,'invocation':1,
            'artifacts':inventory,'research_fit_completed':True,'native_training_started':False,
            'test_accessed':False,'finished_utc':pd.Timestamp.now(tz='UTC').isoformat()})
        if {str(p.relative_to(stage)) for p in stage.rglob('*') if p.is_file()}!=set(inventory)|{'TERMINAL.json'}:raise RuntimeError('GC_PROBE_PUBLICATION_INVENTORY_INVALID')
        for name,item in inventory.items():io.verify_digest(stage/name,item['sha256'],'PROBE_ARTIFACT')
        for path in stage.glob('*.json'):io.read_json(path)
        for directory in (stage/'MODELS',stage):
            fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(fd)
            finally:os.close(fd)
        publish_bundle_directory_noreplace(stage,root/'RESULTS')
        print('GC_PROBE_COMPLETE '+json.dumps(comparisons,sort_keys=True),flush=True)
    except Exception as exc:
        if not (root/'FAILURE.json').exists():io.write_json(root/'FAILURE.json',{'status':'FAILED_NO_ADMISSION','error':str(exc),'exception':type(exc).__name__,'test_accessed':False})
        raise


def pa_table_from_frame(frame):
    import pyarrow as pa
    table=pa.Table.from_pandas(frame,preserve_index=False)
    if table.num_rows==0:raise RuntimeError('GC_PROBE_EMPTY_PREDICTIONS')
    return table


# Bounded P0 successor; original one-off studies/functions remain unchanged.
P0_ID = 'GC_P0_FIRST_DEVELOPMENT_20261017_021'
P0_ARMS = ('CORE','F','L','V','FL','FV','LV','FLV')


def p0_field_contract(preparation, native_fields, macro_fields):
    macro=['ctx_cont.'+n for n in macro_fields]
    core=[n for n in preparation['core_native_fields'] if n not in macro]
    v=preparation['layers']['V']['existing_native_fields']
    if len(macro)!=61 or not set(macro)<=set(preparation['core_native_fields']):
        raise RuntimeError('P0_MACRO_EXCLUSION_NOT_EXACT')
    if set(core)&set(v) or set(core)&set(preparation['pattern_pause_fields']):
        raise RuntimeError('P0_DUPLICATED_OR_PAUSED_FIELD')
    if set(native_fields)!=set(core+v+macro+preparation['pattern_pause_fields']):
        raise RuntimeError('P0_NATIVE_PARTITION_INCOMPLETE')
    names={}
    for arm in P0_ARMS:
        names[arm]=list(core)
        for layer in ('F','L','V'):
            if arm!='CORE' and layer in arm:
                names[arm]+=v if layer=='V' else preparation['layers'][layer]['fields']
        if len(names[arm])!=len(set(names[arm])):raise RuntimeError('P0_DUPLICATE_FIELD')
    return names,macro


def p0_rule_predictions(snap, fields, flow, levels, train, evaluation, arm, y, recipe):
    """A fixed sign-vote rule, TRAIN conditional means; no parameter search.

    Core: EMA slope direction; F: sign OFI; L: sign VWAP displacement
    (continuation hypothesis). Votes sum, no requirement for unanimity. V
    conditions on TRAIN RV terciles. Unobserved TRAIN states emit no evidence.
    """
    score=np.sign(snap[:,fields.index(recipe['core_direction_field'])]).astype(int)
    if arm!='CORE' and 'F' in arm:score+=np.sign(flow[recipe['F_direction_field']]).astype(int)
    if arm!='CORE' and 'L' in arm:score+=np.sign(levels[recipe['L_direction_field']]).astype(int)
    regime=np.zeros(len(score),int);cuts=[]
    if arm!='CORE' and 'V' in arm:
        rv=snap[:,fields.index(recipe['V_state_field'])]
        cuts=np.quantile(rv[train],recipe['vol_quantiles']).tolist()
        regime=np.searchsorted(cuts,rv,side='right')
    states=np.column_stack((score,regime));lookup={}
    for state in np.unique(states[train],axis=0):
        mask=np.all(states[train]==state,axis=1)
        lookup[tuple(state)]=float(np.mean(y[train[mask]]))
    out=np.array([lookup.get(tuple(state),np.nan) for state in states[evaluation]])
    return out,{'vol_cuts_train_only':cuts,'states':[{'score':int(k[0]),'vol':int(k[1]),'mean_bps':v} for k,v in sorted(lookup.items())]}


def p0_prediction_quality(y,p):
    ok=np.isfinite(y)&np.isfinite(p);y=np.asarray(y)[ok];p=np.asarray(p)[ok]
    if not len(y):return {'rows':0}
    out=gc_prediction_metrics(y,p)
    out.update(mae_bps=float(np.mean(np.abs(y-p))),sign_hit=float(np.mean(np.sign(y)==np.sign(p))),
               mean_prediction_bps=float(p.mean()),mean_outcome_bps=float(y.mean()))
    if np.ptp(p)>0:
        slope=float(np.sum((p-p.mean())*(y-y.mean()))/np.sum((p-p.mean())**2))
        out.update(calibration_slope=slope,calibration_intercept_bps=float(y.mean()-slope*p.mean()))
    return out


def p0_actions(pred, spread, economics, horizon_minutes):
    """Decision-known expected cash approximation; exact realized v3 accounting.

    Linear mid-bps proxy minus last CLOSED XAU spread, round-trip fees,
    projected horizon funding; no future spread or outcome in the action.
    """
    pred=np.asarray(pred);spread=np.asarray(spread)
    cost=economics['round_trip_execution_cost_bps']
    fin=np.asarray(economics['annual_financing_cost_bps'])*horizon_minutes*60/economics['seconds_per_year']
    q=np.column_stack((pred-spread-cost-fin[0],-pred-spread-cost-fin[1],np.zeros(len(pred))))
    good=np.isfinite(pred)&np.isfinite(spread)
    action=np.argmax(np.where(np.isfinite(q),q,-np.inf),axis=1)
    unique=(q==q.max(axis=1)[:,None]).sum(axis=1)==1
    return np.where(good&unique,np.array([1,-1,0])[action],0).astype(np.int8)


def p0_equity_summary(net, times, entries, exited, held_seconds, turnover, costs, funding):
    dates=pd.DatetimeIndex(times,tz='UTC').normalize()
    last=np.r_[np.flatnonzero(dates[1:]!=dates[:-1]),len(times)-1]
    daily=pd.Series(net[last],index=dates[last])
    # Calendar-day zero returns during closures are retained explicitly.
    days=pd.date_range(daily.index[0],daily.index[-1],freq='D')
    daily=daily.reindex(days).ffill();changes=np.diff(np.r_[0.,daily.to_numpy()])
    sd=float(np.std(changes,ddof=1)) if len(changes)>1 else 0.
    dd=float(np.max(np.maximum.accumulate(np.r_[0.,net])[1:]-net))
    return {'net_total_bps':float(net[-1]),'sharpe_calendar_daily_365':float(np.mean(changes)/sd*np.sqrt(365.25)) if sd>0 else None,
            'max_drawdown_observed_M1_bps':dd,'trades_opened':entries,'trades_closed':exited,
            'open_positions_at_end':entries-exited,'exposure_wall_clock':held_seconds/((times[-1]-times[0])/NS),
            'turnover_notional_units':turnover,'execution_cost_and_exit_reserve_bps':float(costs),
            'financing_bps':float(funding),'days':len(changes)},daily


def p0_position_ledger(decisions, sides, tape, evaluation_start, evaluation_end, scenarios, seconds_per_year):
    """One constant initial-equity notional. Change only on a new qualified signal.

    No stop, time exit or forced boundary close. Retain position across missing
    features/quotes and model-fold changes. Final open position is executable
    mark-to-market with reserved exit fee. Financing accrues across closures.
    All marks are observed M1 OPEN quotes; intraminute paths are unobserved.
    """
    scope=(tape.time_ns>=evaluation_start)&(tape.time_ns<evaluation_end)
    tt=tape.time_ns[scope];bid=tape.bid_open[scope];ask=tape.ask_open[scope]
    if len(tt)<2:raise RuntimeError('P0_LEDGER_EMPTY')
    target=np.asarray(decisions)+MINUTE_NS;idx=np.searchsorted(tt,target)
    good=(idx<len(tt));good &= tt[np.minimum(idx,len(tt)-1)]==target
    if np.any(np.diff(decisions)<=0):raise RuntimeError('P0_LEDGER_DECISION_ORDER')
    orders=[];side=0
    for i,wanted in zip(idx[good],np.asarray(sides)[good]):
        if wanted!=side:orders.append((int(i),int(wanted)));side=int(wanted)
    gross=np.zeros(len(tt));long_time=np.zeros(len(tt));short_time=np.zeros(len(tt));opened=np.zeros(len(tt),np.int32)
    realized=lt=st=0.;side=0;start=0;price=0.;count=closed=0;held=0.;turnover=0.;trades=[]
    for stop,wanted in orders+[(len(tt),0)]:
        if stop<start:raise RuntimeError('P0_LEDGER_CLOCK')
        sl=slice(start,stop)
        gross[sl]=realized;long_time[sl]=lt;short_time[sl]=st;opened[sl]=count
        if side:
            px=bid[sl] if side>0 else ask[sl]
            gross[sl]+=side*1e4*(px-price)/price
            elapsed=(tt[sl]-tt[start])/NS
            if side>0:long_time[sl]+=elapsed
            else:short_time[sl]+=elapsed
            mark_index=min(stop,len(tt)-1)
            duration=(tt[mark_index]-tt[start])/NS;held+=duration
            terminal=stop==len(tt)
            px_end=bid[mark_index] if side>0 else ask[mark_index]
            pnl=float(side*1e4*(px_end-price)/price)
            trades.append({'entry_ns':int(tt[start]),'mark_or_exit_ns':int(tt[mark_index]),'side':side,
                'gross_bps':pnl,'elapsed_seconds':float(duration),'terminal_open':terminal,
                'entry_price':float(price),'exit_or_mark_price':float(px_end)})
            if not terminal:
                realized+=pnl;closed+=1;turnover+=float(px_end/price)
                if side>0:lt+=duration
                else:st+=duration
        if stop==len(tt):break
        side=wanted;start=stop
        if side:
            price=ask[start] if side>0 else bid[start];count+=1;turnover+=1.
    summaries={};daily={}
    for name,cfg in scenarios.items():
        fee=opened*cfg['round_trip_execution_cost_bps']
        rates=cfg['annual_financing_cost_bps']
        funding=(long_time*rates[0]+short_time*rates[1])/seconds_per_year
        net=gross-fee-funding
        summaries[name],daily[name]=p0_equity_summary(net,tt,count,closed,held,turnover,fee[-1],funding[-1])
    return summaries,daily,trades,{'rejected_missing_exact_execution_quote':int((~good).sum()),'executed_signal_changes':len(orders),
        'final_mark_ns':int(tt[-1]),'final_open_position':side,'mark_resolution':'observed M1 open; stale through absent quotes',
        'fee_treatment':'full round trip reserved at entry including terminal open exit; no spread added twice'}


def p0_representativeness(decisions,snap,session,fields,common,initial,recipe):
    rv=snap[:,fields.index(recipe['V_state_field'])]
    cuts=np.quantile(rv[initial],recipe['vol_quantiles'])
    trend=np.sign(snap[:,fields.index(recipe['core_direction_field'])]).astype(int)
    dates=pd.DatetimeIndex(decisions,tz='UTC')
    strata={'year':dates.year.astype(str),'month':dates.strftime('%Y-%m'),'native_session_id':session.astype(str),
            'past_RV_train_tercile':np.searchsorted(cuts,rv).astype(str),'EMA_trend_sign':trend.astype(str)}
    result={'population':len(common),'matched':int(common.sum()),'coverage':float(common.mean()),
            'RV_tercile_cuts_initial_train_only':cuts.tolist(),'strata':{}}
    for name,values in strata.items():
        rows=[]
        for value in np.unique(values):
            mask=values==value;n=int(mask.sum());k=int((mask&common).sum())
            rows.append({'stratum':str(value),'population':n,'matched':k,'coverage':k/n,
                         'population_share':n/len(common),'matched_share':k/int(common.sum())})
        result['strata'][name]=rows
    return result


def run_p0_development(binding_path):
    import os,subprocess,lightgbm as lgb
    from gx1.scripts import gc_source_disposition_v1 as io
    from gx1.features.d1_macro_context_v1 import D1_MACRO_CONTEXT_FIELDS
    from gx1.contracts.gx1_capped_execution_v1 import require_capped_cpu_audit_execution,require_capped_lock_ancestry
    from gx1.contracts.entry_model_native_bundle_commit_v1 import publish_bundle_directory_noreplace
    root=Path('/home/andre2/GX1_RUNS/GC_ORDER_FLOW_RESEARCH_001')/P0_ID
    if binding_path!=root/'EXECUTION_BINDING.json':raise RuntimeError('P0_BINDING_PATH')
    binding=io.read_json(binding_path);plan_path=io.REPO/f'configs/research/{P0_ID}.json'
    io.verify_digest(plan_path,binding['plan_sha256'],'P0_PLAN');spec=io.read_json(plan_path)
    io.verify_digest(io.REPO/'NEXT_RUN_POLICY.json',binding['policy_sha256'],'P0_POLICY')
    io.verify_digest(root/'OPERATOR_APPROVAL.json',binding['approval_sha256'],'P0_APPROVAL')
    policy=io.read_json(io.REPO/'NEXT_RUN_POLICY.json');gate=policy['current_work']['gc_p0_development_021'];approval=io.read_json(root/'OPERATOR_APPROVAL.json')
    if (policy['training_enabled'] is not False or gate['status']!='EXECUTION_CLAIMED' or gate['claimed_invocations']!=1
        or gate['completed_invocations']!=0 or gate['execution_authorized'] is not False or gate['native_training_authorized'] is not False
        or approval['approved_plan_sha256']!=binding['plan_sha256'] or approval['single_use'] is not True
        or approval['max_invocations']!=1 or approval['status']!='APPROVED_SINGLE_USE_PENDING_EXECUTION'):
        raise RuntimeError('P0_SINGLE_USE_SCOPE')
    if io.git_head()!=binding['source_head'] or subprocess.check_output(['git','status','--porcelain'],text=True,cwd=io.REPO):
        raise RuntimeError('P0_SOURCE_NOT_CLEAN_BOUND')
    if subprocess.check_output(['git','branch','--show-current'],text=True,cwd=io.REPO).strip()!='work/gx1-current':raise RuntimeError('P0_BRANCH')
    if set(binding['code_sha256'])!=set(spec['code_paths']):raise RuntimeError('P0_CODE_INVENTORY')
    for p,h in binding['code_sha256'].items():io.verify_digest(io.REPO/p,h,'P0_CODE')
    proof=require_capped_cpu_audit_execution();proof['lock']=require_capped_lock_ancestry()
    if proof['cpu_affinity']!=list(range(8)) or (root/'RESULTS').exists() or (root/'.RESULTS.staging').exists():raise RuntimeError('P0_OUTPUT_OR_CAPS')
    io.write_json(root/'EXECUTION_CLAIM.json',{'invocation':1,'binding_sha256':io.sha256_file(binding_path)})
    stage=root/'.RESULTS.staging';(stage/'MODELS').mkdir(parents=True);started=_time.monotonic()
    try:
        for b in spec['inputs'].values():require_binding(b)
        preparation=io.read_json(Path(spec['inputs']['preparation']['path']))
        source=io.read_json(Path(spec['inputs']['native_source_plan']['path']))
        all_bindings=[source['signal_manifest'],*source['execution_tape_partitions'],source['reference']['data']['execution_tape']['manifest']]
        all_bindings += [b for split in source['native_datasets'].values() for b in split.values()]
        # load_gc_native_rows verifies native manifests and physical row identity.
        base=pq.read_table(source['feature_inputs']['parquet']['path'])
        require_binding(source['feature_inputs']['parquet'])
        times,snap,context,session,fields,ctx_names=load_gc_native_rows(source,base)
        if not np.isfinite(snap).all():raise RuntimeError('P0_NONFINITE_NATIVE')
        for j,n in enumerate(ctx_names):
            if not np.array_equal(snap[:,fields.index('ctx_cont.'+n)],context[:,j]):raise RuntimeError('P0_CONTEXT_ALIAS')
        decisions=times+DECISION_OFFSET_NS
        if decisions.max()>=pd.Timestamp(spec['TEST_seal_utc']).value:raise RuntimeError('P0_TEST_SEAL')
        ft=pq.read_table(spec['inputs']['F']['path']);lev=pq.read_table(spec['inputs']['L']['path'])
        for table in (ft,lev):
            if not np.array_equal(table['decision_ns'].to_numpy(),decisions):raise RuntimeError('P0_FEATURE_CLOCK')
        if not np.array_equal(ft['contract_index'].to_numpy(),lev['contract_index'].to_numpy()):raise RuntimeError('P0_FEATURE_CONTRACT_SELECTION')
        quality=io.read_json(Path(spec['inputs']['quality_result']['path']))
        terminal=io.read_json(Path(spec['inputs']['quality_terminal']['path']))
        if terminal['status']!='COMPLETE' or terminal['result_sha256']!=spec['inputs']['quality_result']['sha256']:raise RuntimeError('P0_QUALITY_TERMINAL')
        names,excluded=p0_field_contract(preparation,fields,D1_MACRO_CONTEXT_FIELDS)
        if names!=spec['feature_order'] or {a:len(v) for a,v in names.items()}!=spec['feature_counts'] or excluded!=spec['excluded_macro_fields']:raise RuntimeError('P0_RECIPE_FIELD_PARITY')
        if lgb.__version__!='4.6.0':raise RuntimeError('P0_TREE_LIBRARY_VERSION')
        F={n:ft[n].to_numpy() for n in preparation['layers']['F']['fields']};L={n:lev[n].to_numpy() for n in preparation['layers']['L']['fields']}
        common=ft['ofi_valid'].to_numpy()&ft['bbo_valid'].to_numpy()&lev['rolling15_valid'].to_numpy()
        common &= np.isfinite(np.column_stack(list(F.values())+list(L.values()))).all(axis=1)
        if int(common.sum())!=spec['expected_common_rows']:raise RuntimeError('P0_COHORT_DRIFT')
        cutoff=pd.Timestamp(spec['folds'][0]['validation_start_utc_inclusive']).value-86400*NS
        representation=p0_representativeness(decisions,snap,session,fields,common,np.flatnonzero(decisions<cutoff),spec['rule'])
        io.write_json(stage/'REPRESENTATIVENESS.json',representation)
        rows=np.flatnonzero(common);d=decisions[rows];ss=snap[rows];F={n:v[rows] for n,v in F.items()};L={n:v[rows] for n,v in L.items()}
        matrices={arm:np.column_stack([ss[:,fields.index(n)] if n in fields else F[n] if n in F else L[n] for n in ns]) for arm,ns in names.items()}
        lock={'rows':len(rows),'decision_sha256':hashlib.sha256(d.tobytes()).hexdigest(),'fields':names,'excluded_macro_fields':excluded,
              'labels_read_at_lock':False,'macro_excision_does_not_exclude_nov_dec_2025':True,'common_rows_sha256':hashlib.sha256(rows.tobytes()).hexdigest()}
        io.write_json(stage/'INPUT_LOCK.json',lock);print('P0_INPUT_LOCKED '+json.dumps({a:len(n) for a,n in names.items()}),flush=True)
        for b in source['execution_tape_partitions']:require_binding(b)
        actual={str(p) for p in Path(source['reference']['data']['execution_tape']['root']).glob('year=*/*.parquet') if int(p.parent.name.split('=')[1])>=2024}
        if actual!={b['path'] for b in source['execution_tape_partitions']}:raise RuntimeError('P0_UNBOUND_TAPE')
        tape=load_tape(source['reference'],2024)
        ex=execution_for_horizon_v2(d,tape,horizon_minutes=spec['primary_horizon_minutes'],latency_seconds=60)
        valid=exact_intraday_mask_v3(d,tape,ex,horizon_minutes=spec['primary_horizon_minutes'])['valid']
        net_values=net_cash_values_v3(ex,economics=spec['economics'])
        if not np.isfinite(net_values[valid]).all() or not np.all(net_values[valid,2]==0):raise RuntimeError('P0_V3_VALUES')
        if valid.sum()!=len(rows):raise RuntimeError('P0_PRIMARY_EXACT_PATH_CHANGED')
        spread=spread_at_decision(d,tape);frames=[];fit_receipts=[]
        for declared in spec['folds']:
            fold=build_fold(declared['fold_id'],declared['validation_start_utc_inclusive'][:10],declared['validation_end_utc_exclusive'][:10],d,ex.exit_ns,valid,train_years=3)
            if np.any(ex.exit_ns[fold.train]>=fold.eval_start_ns) or np.any(d[fold.train]>=fold.eval_start_ns-86400*NS):raise RuntimeError('P0_PURGE')
            ev=fold.eval;frame=pd.DataFrame({'decision_ns':d[ev],'fold':fold.name,'y30_mid_bps':ex.y_mid_bps[ev],
                'outcome_within_fold':ex.exit_ns[ev]<fold.eval_end_ns,'spread_at_decision_bps':spread[ev],
                'train_mean_bps':float(ex.y_mid_bps[fold.train].mean())})
            for arm in P0_ARMS:
                pred,state=p0_rule_predictions(ss,fields,F,L,fold.train,ev,arm,ex.y_mid_bps,spec['rule'])
                frame['rule_'+arm]=pred;io.write_json(stage/'MODELS'/f'{fold.name}_{arm}_RULE.json',state)
                model=lgb.train(spec['tree']['params'],lgb.Dataset(matrices[arm][fold.train],ex.y_mid_bps[fold.train]),num_boost_round=spec['tree']['rounds'])
                path=stage/'MODELS'/f'{fold.name}_{arm}_TREE.txt';model.save_model(str(path))
                predicted=model.predict(matrices[arm][ev]);replay=lgb.Booster(model_file=str(path)).predict(matrices[arm][ev])
                if not np.array_equal(predicted,replay):raise RuntimeError('P0_TREE_READBACK')
                frame['tree_'+arm]=predicted
                fit_receipts.append({'fold':fold.name,'arm':arm,'train_rows':len(fold.train),'eval_rows':len(ev),
                    'train_decision_sha256':hashlib.sha256(d[fold.train].tobytes()).hexdigest(),
                    'training_end_outcome_ns':int(ex.exit_ns[fold.train].max()),'evaluation_start_ns':fold.eval_start_ns})
            frames.append(frame);print('P0_FOLD_COMPLETE '+fold.name,flush=True)
        if len(fit_receipts)!=spec['max_tree_fits']:raise RuntimeError('P0_MODEL_COUNT')
        predictions=pd.concat(frames,ignore_index=True)
        if np.any(np.diff(predictions.decision_ns)<=0):raise RuntimeError('P0_PREDICTION_ORDER')
        evaluation_start=pd.Timestamp(spec['folds'][0]['validation_start_utc_inclusive']).value
        evaluation_end=pd.Timestamp(spec['TEST_seal_utc']).value
        pred_d=predictions.decision_ns.to_numpy();positions=np.searchsorted(d,pred_d)
        secondary={}
        for horizon in spec['secondary_horizons_minutes']:
            eh=execution_for_horizon_v2(pred_d,tape,horizon_minutes=horizon,latency_seconds=60)
            ok=exact_intraday_mask_v3(pred_d,tape,eh,horizon_minutes=horizon)['valid']
            predictions[f'y{horizon}_mid_bps']=eh.y_mid_bps
            predictions[f'h{horizon}_exact_valid']=ok
        cells={};daily_vectors={};all_trades=[];all_daily=[]
        keys=[f'{model}_{arm}' for model in ('rule','tree') for arm in P0_ARMS]+['always_LONG','always_FLAT']
        for key in keys:
            if key.startswith('always_'):side=np.full(len(predictions),1 if key=='always_LONG' else 0,np.int8)
            else:side=p0_actions(predictions[key].to_numpy(),predictions.spread_at_decision_bps.to_numpy(),spec['economics'],spec['primary_horizon_minutes'])
            predictions['action_'+key]=side
            sums,daily,trades,account=p0_position_ledger(pred_d,side,tape,evaluation_start,evaluation_end,spec['cost_scenarios'],spec['economics']['seconds_per_year'])
            prediction={};diagnostics={};fold_metrics={}
            if not key.startswith('always_'):
                good=predictions.outcome_within_fold.to_numpy()
                prediction=p0_prediction_quality(predictions.y30_mid_bps.to_numpy()[good],predictions[key].to_numpy()[good])
                prediction['train_mean_reference_mse_bps2']=float(np.mean((predictions.y30_mid_bps.to_numpy()[good]-predictions.train_mean_bps.to_numpy()[good])**2))
                for fold,part in predictions[good].groupby('fold'):
                    fold_metrics[fold]=p0_prediction_quality(part.y30_mid_bps.to_numpy(),part[key].to_numpy())
                for h in spec['secondary_horizons_minutes']:
                    ok=predictions[f'h{h}_exact_valid'].to_numpy()
                    diagnostics[str(h)]=p0_prediction_quality(predictions[f'y{h}_mid_bps'].to_numpy()[ok],predictions[key].to_numpy()[ok])
            cells[key]={'prediction_h30':prediction,'prediction_folds':fold_metrics,'secondary_same_h30_forecast_no_refit':diagnostics,
                        'economics':sums,'ledger':account,'missing_rule_evidence':int(predictions[key].isna().sum()) if key in predictions else 0}
            daily_vectors[key]=daily[spec['central_scenario']]
            for scenario,series in daily.items():
                part=pd.DataFrame({'day':series.index,'equity_bps':series.to_numpy(),'strategy':key,'scenario':scenario});all_daily.append(part)
            all_trades.extend([dict(t,strategy=key) for t in trades])
        comparisons={}
        for model in ('rule','tree'):
            baseline=model+'_CORE';frame=predictions[predictions.outcome_within_fold]
            for arm in P0_ARMS[1:]:
                key=model+'_'+arm;ok=np.isfinite(frame[baseline])&np.isfinite(frame[key])
                loss=(frame.y30_mid_bps[ok]-frame[baseline][ok])**2-(frame.y30_mid_bps[ok]-frame[key][ok])**2
                rec=[{'entry_ns':int(t),'v':float(v)} for t,v in zip(frame.decision_ns[ok],loss)]
                pb=weekly_block_bootstrap(rec,'v',reps=spec['bootstrap_reps'],seed=spec['bootstrap_seed'],levels=spec['confidence_levels'])
                delta=daily_vectors[key]-daily_vectors[baseline];changes=delta.diff();changes.iloc[0]=delta.iloc[0]
                rec=[{'entry_ns':int(t.value),'v':float(v)} for t,v in changes.items()]
                eb=weekly_block_bootstrap(rec,'v',reps=spec['bootstrap_reps'],seed=spec['bootstrap_seed'],levels=spec['confidence_levels'])
                comparisons[key+'_minus_CORE']={'paired_prediction_MSE_improvement_bps2':pb,'paired_calendar_daily_net_improvement_bps':eb,
                    'net_total_difference_bps':float(delta.iloc[-1]),'all_26_months_development':True}
        for key in keys:
            series=daily_vectors[key];change=series.diff();change.iloc[0]=series.iloc[0]
            cells[key]['monthly_net_bps']={str(k):float(v) for k,v in change.groupby(change.index.strftime('%Y-%m')).sum().items()}
        predictions.to_parquet(stage/'PREDICTIONS.parquet',index=False)
        pd.DataFrame(all_trades).to_parquet(stage/'TRADES_AND_OPEN_MARKS.parquet',index=False)
        pd.concat(all_daily,ignore_index=True).to_parquet(stage/'DAILY_EQUITY_ALL_SCENARIOS.parquet',index=False)
        for b in list(spec['inputs'].values())+all_bindings+[source['feature_inputs']['parquet']]:require_binding(b)
        for p,h in binding['code_sha256'].items():io.verify_digest(io.REPO/p,h,'P0_CODE_POST')
        io.verify_digest(io.REPO/'NEXT_RUN_POLICY.json',binding['policy_sha256'],'P0_POLICY_POST')
        io.verify_digest(plan_path,binding['plan_sha256'],'P0_PLAN_POST');io.verify_digest(root/'OPERATOR_APPROVAL.json',binding['approval_sha256'],'P0_APPROVAL_POST')
        io.verify_digest(binding_path,io.read_json(root/'EXECUTION_CLAIM.json')['binding_sha256'],'P0_BINDING_POST')
        if io.git_head()!=binding['source_head']:raise RuntimeError('P0_HEAD_CHANGED')
        result={'status':'P0_FIRST_DEVELOPMENT_COMPLETE_NOT_CONFIRMATION','source_head':binding['source_head'],'plan_sha256':binding['plan_sha256'],
                'input_lock':lock,'representation':representation,'fit_receipts':fit_receipts,'tree_fits':len(fit_receipts),
                'development_prediction_rows':len(predictions),'cells':cells,'contrasts':comparisons,'limits':proof,
                'elapsed_seconds':_time.monotonic()-started,'TEST_accessed':False,'native_training_started':False,
                'source_unchanged':True,'historical_funding_truth':False,'untouched_confirmation':False}
        io.write_json(stage/'RESULT.json',result)
        inventory={}
        for path in sorted(stage.rglob('*')):
            if path.is_file():
                with path.open('rb') as f:os.fsync(f.fileno())
                inventory[str(path.relative_to(stage))]={'sha256':io.sha256_file(path),'bytes':path.stat().st_size}
        io.write_json(stage/'TERMINAL.json',{'status':'COMPLETE','exit_code':0,'source_unchanged':True,'test_data_used':False,'TEST_accessed':False,'research_fit_completed':True,
            'result_sha256':io.sha256_file(stage/'RESULT.json'),'artifacts':inventory,'finished_utc':pd.Timestamp.now(tz='UTC').isoformat()})
        for name,b in inventory.items():io.verify_digest(stage/name,b['sha256'],'P0_ARTIFACT')
        for path in stage.glob('*.parquet'):
            if pq.ParquetFile(path).metadata.num_rows<1:raise RuntimeError('P0_EMPTY_ARTIFACT')
        for directory in (stage/'MODELS',stage):
            fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(fd)
            finally:os.close(fd)
        publish_bundle_directory_noreplace(stage,root/'RESULTS');print('P0_COMPLETE',flush=True)
    except Exception as exc:
        if not (root/'FAILURE.json').exists():io.write_json(root/'FAILURE.json',{'status':'FAILED_NO_ADMISSION','error':str(exc),'exception':type(exc).__name__})
        raise


def main(argv: list[str] | None = None) -> int:
    import sys
    actual = sys.argv[1:] if argv is None else argv
    if len(actual) == 2 and actual[0] == '--p0-binding':
        run_p0_development(Path(actual[1]))
        return 0
    if len(actual) == 2 and actual[0] == '--gc-binding':
        run_gc_trade_probe(Path(actual[1]))
        return 0
    p = argparse.ArgumentParser()
    p.add_argument("--spec", required=True, type=Path)
    p.add_argument("--spec-sha256", required=True)
    p.add_argument("--out-dir", required=True, type=Path)
    a = p.parse_args(argv)
    spec = load_spec(a.spec, a.spec_sha256)
    run(spec, a.spec_sha256, a.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
