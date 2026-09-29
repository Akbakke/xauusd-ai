#!/usr/bin/env python3
"""Bounded A/B/C research orchestration. Currently implements the preregistered A arm.

Reuses indicator, clock, learner, portfolio and inference owners. No native
training, broker access, TEST, automatic parameter search or result promotion.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from gx1.features.htf_features import _resample_ohlc_for_model_native_scalars
from gx1.features.technical_indicators_v1 import classic_ema, wilder_atr
from gx1.time.session_detector import M5_BAR_DURATION, TRADING_SESSION_DURATION
from gx1.scripts.research_entry_direction_walkforward_v1 import RidgeGram, Tape, fit_hgb
from gx1.scripts.research_model_free_baselines_v1 import (
    ResearchFinancingCurve, causal_risk_units, max_t_inference, portfolio_path,
    portfolio_period_returns, portfolio_summary, stationary_bootstrap_indices,
)

ROOT = Path(__file__).resolve().parents[2]
FEATURES = [*(f"momentum_{n}_atr14" for n in (21, 63, 126, 252)),
            "range_position_252", "atr14_over_atr252", "ema200_distance_atr14"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def write_json(path: Path, obj: dict) -> None:
    if path.exists():
        raise RuntimeError(f"TA_OUTPUT_EXISTS: {path}")
    temp = path.with_suffix(path.suffix + ".part")
    with temp.open("x") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.rename(temp, path)


def checked_spec(path: Path, digest: str) -> dict:
    if sha(path) != digest:
        raise RuntimeError("TA_SPEC_HASH")
    if git("branch", "--show-current") != "work/gx1-current" or git("status", "--porcelain"):
        raise RuntimeError("TA_REQUIRES_CANONICAL_CLEAN_SOURCE")
    relative = path.resolve().relative_to(ROOT).as_posix()
    git("ls-files", "--error-unmatch", relative)
    spec = json.loads(path.read_text())
    for name, expected in spec["source_files"].items():
        if sha(ROOT / name) != expected:
            raise RuntimeError(f"TA_SOURCE_HASH: {name}")
    return spec


def fetch_funding(spec: dict, spec_path: Path) -> dict:
    out = Path(spec["output_directory"])
    out.mkdir(parents=True, exist_ok=False)
    request = Request(spec["url"], headers={"User-Agent": "GX1 offline research"})
    try:
        with urlopen(request, timeout=spec.get("timeout_seconds", 30)) as response:
            raw = response.read(spec["maximum_bytes"] + 1)
            final_url = response.url
        if len(raw) > spec["maximum_bytes"]:
            raise RuntimeError("TA_FUNDING_RESPONSE_TOO_LARGE")
        raw_path = out / "DFF.csv"
        raw_path.write_bytes(raw)
        frame = pd.read_csv(io.BytesIO(raw))
        if list(frame.columns) != ["observation_date", "DFF"]:
            raise RuntimeError(f"TA_FUNDING_SCHEMA: {list(frame.columns)}")
        dates = pd.DatetimeIndex(pd.to_datetime(frame.observation_date, utc=True))
        expected = pd.date_range(spec["start_date"], spec["end_date"], freq="D", tz="UTC")
        rates = pd.to_numeric(frame.DFF, errors="raise").to_numpy(float) / 100.0
        if not dates.equals(expected) or not np.isfinite(rates).all():
            raise RuntimeError("TA_FUNDING_COVERAGE")
        receipt = {"status": "COMPLETE", "fetched_utc": datetime.now(timezone.utc).isoformat(),
                   "manifest": str(spec_path), "manifest_sha256": sha(spec_path),
                   "raw_path": str(raw_path), "raw_sha256": sha(raw_path), "rows": len(frame),
                   "first_date": str(dates[0]), "last_date": str(dates[-1]),
                   "final_url": final_url, "purpose": "cost_proxy_only_not_predictor"}
        write_json(out / "RECEIPT.json", receipt)
        return receipt
    except Exception as exc:
        write_json(out / "FAILED.json", {"error": str(exc), "status": "FAILED"})
        raise


def load_funding(spec: dict) -> ResearchFinancingCurve:
    fetch_manifest = Path(spec["funding"]["manifest"])
    if sha(fetch_manifest) != spec["funding"]["manifest_sha256"]:
        raise RuntimeError("TA_FUNDING_MANIFEST_HASH")
    fetch = json.loads(fetch_manifest.read_text())
    receipt = json.loads((Path(fetch["output_directory"]) / "RECEIPT.json").read_text())
    if receipt["status"] != "COMPLETE" or receipt["manifest_sha256"] != sha(fetch_manifest):
        raise RuntimeError("TA_FUNDING_RECEIPT")
    raw_path = Path(receipt["raw_path"])
    if sha(raw_path) != receipt["raw_sha256"]:
        raise RuntimeError("TA_FUNDING_BYTES_HASH")
    frame = pd.read_csv(raw_path)
    return ResearchFinancingCurve(
        effective_at=pd.DatetimeIndex(pd.to_datetime(frame.observation_date, utc=True)),
        benchmark_annual_rate=frame.DFF.to_numpy(float) / 100,
        coverage_end=pd.Timestamp(fetch["end_date"], tz="UTC") + pd.Timedelta(days=1),
        broker_markup=spec["funding"]["markup"], seconds_per_year=spec["funding"]["seconds_per_year"],
    )


def load_market(spec: dict) -> tuple[pd.DataFrame, dict]:
    root = Path(spec["tape"]["root"])
    if sha(root / "MANIFEST.json") != spec["tape"]["manifest_sha256"]:
        raise RuntimeError("TA_TAPE_MANIFEST_HASH")
    manifest = json.loads((root / "MANIFEST.json").read_text())
    if (manifest["timestamp_semantics"] != "bar_start_utc"
            or manifest["decision_available_offset_seconds"] != int(M5_BAR_DURATION.total_seconds())
            or manifest["market_closure_contract"] != "oanda_complete_true_source_absence_no_synthesis_v1"):
        raise RuntimeError("TA_TAPE_CLOCK")
    start, end = pd.Timestamp(spec["read_start"]), pd.Timestamp(spec["read_end_exclusive"])
    frames, bindings = [], []
    # Construct only authorized year paths; do not enumerate or stat TEST-year files.
    for year in range(start.year, (end - pd.Timedelta(nanoseconds=1)).year + 1):
        path = root / f"year={year}" / "part-000.parquet"
        digest = sha(path)
        if digest != manifest["year_sha256"][f"year={year}"]:
            raise RuntimeError(f"TA_TAPE_YEAR_HASH: {year}")
        frame = pq.read_table(
            path, columns=["time", "open", "high", "low", "close", "bid_open", "ask_open"],
            filters=[("time", ">=", start.to_pydatetime()), ("time", "<", end.to_pydatetime())],
        ).to_pandas()
        frames.append(frame)
        bindings.append({"path": str(path), "sha256": digest, "rows": len(frame)})
    frame = pd.concat(frames, ignore_index=True).set_index("time").sort_index()
    if frame.index.has_duplicates or not (frame.index < end).all() or not np.isfinite(frame.to_numpy()).all():
        raise RuntimeError("TA_MARKET_ROWS_INVALID")
    return frame, {"parts": bindings, "test_accessed": False}


def daily_panel(frame: pd.DataFrame, end: pd.Timestamp) -> pd.DataFrame:
    daily = _resample_ohlc_for_model_native_scalars(frame, "D1")
    closed = daily.index + TRADING_SESSION_DURATION
    fill_index = frame.index.searchsorted(closed, side="left")
    valid = (closed < end) & (fill_index < len(frame))
    daily = daily.loc[valid].copy()
    fill_index = fill_index[valid]
    daily["decision_time"] = closed[valid]
    daily["fill_time"] = frame.index[fill_index]
    for dest, origin in [("fill_mid", "open"), ("fill_bid", "bid_open"), ("fill_ask", "ask_open")]:
        daily[dest] = frame[origin].to_numpy()[fill_index]
    if daily.fill_time.duplicated().any() or np.any(daily.fill_time < daily.decision_time):
        raise RuntimeError("TA_DAILY_FILL_CLOCK")
    a14 = wilder_atr(daily.high, daily.low, daily.close, 14)
    a252 = wilder_atr(daily.high, daily.low, daily.close, 252)
    positive = a14.where(a14 > 0)
    for n in (21, 63, 126, 252):
        daily[f"momentum_{n}_atr14"] = (daily.close - daily.close.shift(n)) / positive
    low, high = daily.low.rolling(252).min(), daily.high.rolling(252).max()
    daily["range_position_252"] = (daily.close - low) / (high - low).where(high > low)
    daily["atr14_over_atr252"] = a14 / a252.where(a252 > 0)
    daily["ema200_distance_atr14"] = (daily.close - classic_ema(daily.close, 200)) / positive
    daily["atr14"] = a14
    return daily.reset_index(names="session_open")


def fit_predictions(panel: pd.DataFrame, spec: dict) -> tuple[dict, list[dict]]:
    X = panel[FEATURES].to_numpy(float)
    mid, atr = panel.fill_mid.to_numpy(), panel.atr14.to_numpy()
    times = pd.DatetimeIndex(panel.fill_time)
    n, max_h = len(panel), max(spec["horizons"])
    positions = np.arange(n)
    eligible = np.isfinite(X).all(axis=1) & (positions + max_h < n)
    forecasts, fits = {}, []
    for horizon in spec["horizons"]:
        target = np.full(n, np.nan)
        target[:-horizon] = (mid[horizon:] - mid[:-horizon]) / atr[:-horizon]
        forecasts[horizon] = {name: np.full(n, np.nan) for name in ["ridge", "hgb", "constant"]}
        for year in spec["fold_years"]:
            start = pd.Timestamp(f"{year}-01-01", tz="UTC")
            end = pd.Timestamp(f"{year + 1}-01-01", tz="UTC")
            past = np.flatnonzero(eligible & (times < start))
            fit = past[times[past + max_h] < start]
            hold = np.flatnonzero(eligible & (times >= start) & (times < end))
            if len(fit) < spec["min_fit_rows"] or not len(hold):
                fits.append({"year": year, "horizon": horizon, "status": "INSUFFICIENT_CAUSAL_ROWS",
                             "fit_rows": len(fit), "hold_rows": len(hold)})
                continue
            gram = RidgeGram(X[fit], X[hold], inner_fraction=spec["inner_fraction"],
                             fit_positions=fit, purge_bars=max_h, min_inner_rows=spec["min_inner_rows"],
                             constant_alternative=False)
            ridge, ri = gram.fit_predict(target[fit])
            hgb, hi = fit_hgb(
                X[fit], target[fit], X[hold], inner_fraction=spec["inner_fraction"],
                fit_positions=fit, purge_bars=max_h, min_inner_rows=spec["min_inner_rows"],
                max_iter=spec["hgb"]["max_iter"], seed=spec["seed"],
                learning_rate=spec["hgb"]["learning_rate"], min_samples_leaf=spec["hgb"]["min_samples_leaf"],
            )
            for name, pred in [("ridge", ridge), ("hgb", hgb),
                               ("constant", np.full(len(hold), target[fit].mean()))]:
                forecasts[horizon][name][hold] = pred
            fits.append({"year": year, "horizon": horizon, "status": "FIT",
                         "fit_rows": len(fit), "hold_rows": len(hold),
                         "last_fit_outcome_time": str(times[fit[-1] + max_h]),
                         "first_hold_time": str(times[hold[0]]), "ridge": ri, "hgb": hi})
            print(f"[TA-A] fitted year={year} horizon={horizon} rows={len(fit)}/{len(hold)}", flush=True)
    return forecasts, fits


def statistics(values: dict[str, np.ndarray], comparisons: list[dict],
               normalizer: np.ndarray, annualization: float, index: np.ndarray) -> np.ndarray:
    out = []
    for c in comparisons:
        a, b = values[c["model"]][index], values[c["baseline"]][index]
        delta = a - b
        if c["metric"] == "mean_delta_bps":
            value = delta.mean()
        elif c["metric"] == "normalized_delta":
            value = (delta / normalizer[index]).mean()
        elif c["metric"] == "sharpe_delta":
            sa, sb = a.std(ddof=1), b.std(ddof=1)
            value = (a.mean() / sa - b.mean() / sb) * np.sqrt(annualization) if sa > 0 and sb > 0 else np.nan
        else:
            raise RuntimeError("TA_METRIC")
        out.append(value)
    return np.asarray(out)


def evaluate(panel: pd.DataFrame, forecasts: dict, curve: ResearchFinancingCurve, spec: dict, out: Path) -> dict:
    common = np.logical_and.reduce([np.isfinite(v) for arm in forecasts.values() for v in arm.values()])
    rows = np.flatnonzero(common)
    if len(rows) < 2 or np.any(np.diff(rows) != 1):
        raise RuntimeError("TA_COMMON_EVALUATION_POPULATION_INVALID")
    sample = panel.iloc[rows]
    time = pd.DatetimeIndex(sample.fill_time)
    tape = Tape(time, sample.fill_mid.to_numpy(), sample.fill_bid.to_numpy(),
                sample.fill_ask.to_numpy(), spec["tape"]["manifest_sha256"], spec["tape"]["root"])
    full_mid = panel.fill_mid.to_numpy()
    sigma = pd.Series(full_mid).pct_change(fill_method=None).rolling(spec["risk"]["lookback"]).std(ddof=1).to_numpy()
    normalizer = sigma[rows[:-1]] * 1e4
    if not np.isfinite(normalizer).all() or np.any(normalizer <= 0):
        raise RuntimeError("TA_NORMALIZER")
    values, summaries = {}, {}
    for horizon, arm in forecasts.items():
        signals = {name: np.sign(pred) for name, pred in arm.items()}
        signals["trend"] = np.sign(np.sign(panel[FEATURES[:4]].to_numpy()).mean(axis=1))
        signals["long"] = np.ones(len(panel))
        signals["buy_hold"] = np.ones(len(panel))
        for name, sides in signals.items():
            if name == "buy_hold":
                units = np.full(len(rows) - 1, spec["initial_equity"] / tape.mid[0])
            else:
                units = causal_risk_units(
                    full_mid, sides, lookback=spec["risk"]["lookback"],
                    periods_per_year=spec["periods_per_year"], target_annual_vol=spec["risk"]["target_annual_vol"],
                    max_gross_leverage=spec["risk"]["max_gross_leverage"],
                    initial_equity=spec["initial_equity"],
                )[rows[:-1]]
            for financing in ["historical_proxy", "zero"]:
                key = f"h{horizon}:{name}:{financing}"
                path = portfolio_path(
                    tape, units, initial_equity=spec["initial_equity"],
                    slippage_bps_per_execution=spec["slippage_bps_per_execution"],
                    commission_bps_per_execution=spec["commission_bps_per_execution"],
                    financing_curve=curve if financing == "historical_proxy" else None, liquidate_at_end=True,
                )
                path.to_parquet(out / (key.replace(":", "_") + ".parquet"), index=False)
                summaries[key] = portfolio_summary(path, periods_per_year=spec["periods_per_year"])
                values[key] = portfolio_period_returns(path) * 1e4
    comparisons = []
    for horizon in spec["horizons"]:
        for learner in ["ridge", "hgb"]:
            for funding in ["historical_proxy", "zero"]:
                for baseline in ["long", "constant", "trend", "buy_hold"]:
                    for metric in spec["effects"]:
                        comparisons.append({
                            "name": f"h{horizon}:{learner}:{funding}:vs_{baseline}:{metric}",
                            "model": f"h{horizon}:{learner}:{funding}",
                            "baseline": f"h{horizon}:{baseline}:{funding}", "metric": metric,
                        })
    n = len(normalizer)
    point = statistics(values, comparisons, normalizer, spec["periods_per_year"], np.arange(n))
    boot = np.empty((spec["bootstrap_draws"], len(comparisons)))
    for b, index in enumerate(stationary_bootstrap_indices(
        n, draws=spec["bootstrap_draws"], mean_block_length=spec["mean_block_length"], seed=spec["seed"],
    )):
        boot[b] = statistics(values, comparisons, normalizer, spec["periods_per_year"], index)
    # Undefined Sharpe/zero bootstrap variation remains an explicit untestable member.
    eligible = np.isfinite(point) & np.isfinite(boot).all(axis=0) & (boot.std(axis=0, ddof=1) > 0)
    inference = {}
    if eligible.any():
        chosen = [c for c, ok in zip(comparisons, eligible) if ok]
        inference = max_t_inference(
            point[eligible], boot[:, eligible], names=[c["name"] for c in chosen],
            alpha=spec["alpha"], desired_power=spec["desired_power"],
            effect_sizes=np.array([spec["effects"][c["metric"]] for c in chosen]),
            minimum_relevant_effect=np.array([spec["effects"][c["metric"]][0] for c in chosen]),
        )
    endpoints = {row["name"]: row for row in inference.get("endpoints", [])}
    untestable = []
    for c, ok in zip(comparisons, eligible):
        if not ok:
            untestable.append(c["name"])
            endpoints[c["name"]] = {"name": c["name"], "effect_verdict": "INKONKLUSIV",
                                    "reason": "undefined_statistic_or_zero_bootstrap_variation"}
    decisions = {}
    for learner in ["ridge", "hgb"]:
        required = [c["name"] for c in comparisons if
                    c["name"].startswith(f"h{spec['primary_horizon']}:{learner}:")
                    and ":vs_buy_hold:" not in c["name"]]
        verdicts = [endpoints[name]["effect_verdict"] for name in required]
        positive_net = all(summaries[f"h{spec['primary_horizon']}:{learner}:{f}"]["total_net_bps"] > 0
                           for f in ["historical_proxy", "zero"])
        decision = ("GO" if all(v == "GO" for v in verdicts) and positive_net else
                    "NO_GO" if "NO_GO" in verdicts else "INKONKLUSIV")
        decisions[learner] = {"decision": decision, "required_endpoints": required,
                             "positive_net_both_funding_scenarios": positive_net}
    yearly = {}
    for year in sorted(set(time[1:].year)):
        index = np.flatnonzero(time[1:].year == year)
        yearly[str(year)] = {key: {"intervals": len(index), "mean_net_bps": float(v[index].mean())}
                            for key, v in values.items()}
    pd.DataFrame({"time": time[1:], "gold_sigma_bps": normalizer, **values}).to_parquet(out / "PAIRED_RETURNS.parquet", index=False)
    np.savez_compressed(out / "BOOTSTRAP_STATISTICS.npz", estimates=point, bootstrap_estimates=boot)
    return {"evaluation": {"first_fill": str(time[0]), "last_fill": str(time[-1]), "intervals": n,
                           "reused_development_history": True},
            "portfolios": summaries, "per_year": yearly, "inference": inference,
            "declared_family": [c["name"] for c in comparisons], "untestable_endpoints": untestable,
            "endpoints": list(endpoints.values()), "decisions": decisions}


def run_a(spec: dict, spec_path: Path) -> dict:
    if spec["horizons"] != [20, 5] or spec["feature_names"] != FEATURES:
        raise RuntimeError("TA_PREREG_SCOPE")
    out = Path(spec["output_directory"])
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "STARTED.json", {"git_head": git("rev-parse", "HEAD"),
                                     "preregistration": str(spec_path), "preregistration_sha256": sha(spec_path)})
    try:
        curve = load_funding(spec)
        market, binding = load_market(spec)
        panel = daily_panel(market, pd.Timestamp(spec["read_end_exclusive"]))
        panel.to_parquet(out / "DAILY_PANEL.parquet", index=False)
        forecasts, fits = fit_predictions(panel, spec)
        write_json(out / "FITS.json", {"fits": fits})
        pd.DataFrame({"fill_time": panel.fill_time, **{
            f"h{h}:{name}": pred for h, arm in forecasts.items() for name, pred in arm.items()
        }}).to_parquet(out / "PREDICTIONS.parquet", index=False)
        result = evaluate(panel, forecasts, curve, spec, out)
        result["artifacts"] = {
            path.name: {"sha256": sha(path), "bytes": path.stat().st_size}
            for path in sorted(out.iterdir()) if path.is_file()
        }
        result.update(schema="gx1_ta_measurement_a_v1", git_head=git("rev-parse", "HEAD"),
                      preregistration_sha256=sha(spec_path), inputs=binding,
                      primary_horizon=spec["primary_horizon"], test_accessed=False,
                      native_training=False, evidence_class="measured_reused_development_walkforward")
        write_json(out / "RESULT.json", result)
        write_json(out / "TERMINAL.json", {"status": "COMPLETE", "result_sha256": sha(out / "RESULT.json"),
                                         "finished_utc": datetime.now(timezone.utc).isoformat()})
        return {"status": "COMPLETE", "out": str(out), "decisions": result["decisions"]}
    except Exception as exc:
        write_json(out / "TERMINAL.json", {"status": "FAILED", "error": str(exc),
                                         "finished_utc": datetime.now(timezone.utc).isoformat()})
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["fetch-funding", "run-a"])
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--spec-sha256", required=True)
    args = parser.parse_args()
    spec = checked_spec(args.spec, args.spec_sha256)
    result = fetch_funding(spec, args.spec) if args.mode == "fetch-funding" else run_a(spec, args.spec)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
