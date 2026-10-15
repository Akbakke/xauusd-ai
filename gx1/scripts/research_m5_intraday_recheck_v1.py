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


def ridge_select_and_fit(x: np.ndarray, y: np.ndarray, fold: Fold, grid: list[float]) -> dict[str, Any]:
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
    return {"alpha": best, "inner_mse": scores, "pred": pred, "inner_pred": inner_pred}


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


def main(argv: list[str] | None = None) -> int:
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
