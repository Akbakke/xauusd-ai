#!/usr/bin/env python3
"""
Offline SNIPER performance analysis by regime class (no replay).

This script:
- Finds SNIPER OBS run_dirs for 2025 (Q1–Q4, baseline/guarded) or uses --run-dirs.
- Reads trade_journal_index.csv from each run.
- Classifies each trade into a regime class using `regime_classifier.classify_regime`.
- Aggregates performance metrics per (quarter × variant × regime_class):
  - trades
  - EV/trade (mean pnl_bps)
  - win_rate
  - avg_win, avg_loss, payoff (avg_win / |avg_loss|)
  - p90 loss (90th percentile of loss magnitude in bps)
  - median duration (minutes), if entry_time/exit_time available
- Writes a markdown report under reports/:
  SNIPER_2025_REGIME_SPLIT_REPORT__YYYYMMDD_HHMMSS.md

This is pure analysis; it does not change any trading or replay logic.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .regime_classifier import RegimeClass, classify_regime


SNIPER_BASE = Path("gx1/wf_runs")


@dataclass
class RunInfo:
    run_dir: Path
    quarter: str
    variant: str


def _parse_run_info(run_dir: Path) -> RunInfo:
    """
    Parse quarter and variant from a SNIPER OBS run directory name.
    Expected pattern: SNIPER_OBS_Qx_2025_<variant>_YYYYMMDD_HHMMSS
    """
    name = run_dir.name
    parts = name.split("_")
    # Fallbacks if pattern deviates
    quarter = "UNKNOWN"
    variant = "unknown"
    if len(parts) >= 4 and parts[0] == "SNIPER" and parts[1] == "OBS":
        # SNIPER_OBS_Q1_2025_baseline_...
        quarter = parts[2]
        if len(parts) >= 5:
            variant = parts[4]
    elif len(parts) >= 5 and parts[0] == "SNIPER" and parts[1] == "OBS":
        quarter = parts[2]
        variant = parts[3]
    return RunInfo(run_dir=run_dir, quarter=quarter, variant=variant)


def _autodetect_run_dirs() -> List[RunInfo]:
    """
    Autodetect latest SNIPER OBS run per (quarter, variant) from gx1/wf_runs.
    """
    quarters = ["Q1", "Q2", "Q3", "Q4"]
    variants = ["baseline", "guarded"]
    result: List[RunInfo] = []

    for q in quarters:
        for v in variants:
            pattern = f"SNIPER_OBS_{q}_2025_{v}_*"
            matches = sorted(SNIPER_BASE.glob(pattern))
            if matches:
                result.append(_parse_run_info(matches[-1]))
    return result


def load_trades_for_run(run: RunInfo) -> pd.DataFrame:
    """
    Load trade_journal_index.csv for a given run and annotate with quarter/variant.
    """
    index_path = run.run_dir / "trade_journal" / "trade_journal_index.csv"
    if not index_path.exists():
        raise FileNotFoundError(f"trade_journal_index.csv not found: {index_path}")

    df = pd.read_csv(index_path)
    df["quarter"] = run.quarter
    df["variant"] = run.variant

    # Parse datetimes if present for duration computation
    if "entry_time" in df.columns:
        df["entry_time_parsed"] = pd.to_datetime(
            df["entry_time"], utc=True, errors="coerce"
        )
    else:
        df["entry_time_parsed"] = pd.NaT
    if "exit_time" in df.columns:
        df["exit_time_parsed"] = pd.to_datetime(
            df["exit_time"], utc=True, errors="coerce"
        )
    else:
    ##        df["exit_time_parsed"] = pd.NaT
        df["exit_time_parsed"] = pd.NaT

    # Compute duration (minutes) when possible
    df["duration_min"] = (
        (df["exit_time_parsed"] - df["entry_time_parsed"])
        .dt.total_seconds()
        .div(60.0)
    )

    return df


def classify_trades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add regime_class and regime_reason columns to the trade DataFrame.
    """
    regime_classes: List[str] = []
    regime_reasons: List[str] = []
    for _, row in df.iterrows():
        r_class, reason = classify_regime(row)
        regime_classes.append(r_class)
        regime_reasons.append(reason)
    df = df.copy()
    df["regime_class"] = regime_classes
    df["regime_reason"] = regime_reasons
    return df


def aggregate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate metrics per (quarter, variant, regime_class).
    """
    groups = df.groupby(["quarter", "variant", "regime_class"], dropna=False)
    records: List[Dict[str, object]] = []

    for (q, v, rc), g in groups:
        if g.empty:
            continue
        pnl = pd.to_numeric(g["pnl_bps"], errors="coerce")
        pnl = pnl.dropna()
        if pnl.empty:
            continue

        trades = len(pnl)
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        win_rate = float((pnl > 0).mean()) if trades > 0 else 0.0
        ev = float(pnl.mean()) if trades > 0 else 0.0

        avg_win = float(wins.mean()) if not wins.empty else float("nan")
        avg_loss = float(losses.mean()) if not losses.empty else float("nan")
        if not wins.empty and not losses.empty and avg_loss != 0:
            payoff = float(avg_win / abs(avg_loss))
        else:
            payoff = float("nan")

        # p90 loss: 90th percentile of absolute negative pnl
        if not losses.empty:
            loss_mag = losses.abs().astype(float)
            p90_loss = float(np.percentile(loss_mag.values, 90))
        else:
            p90_loss = 0.0

        # median duration in minutes
        if "duration_min" in g.columns:
            dur = g["duration_min"].dropna()
            median_dur = float(dur.median()) if not dur.empty else float("nan")
        else:
            median_dur = float("nan")

        records.append(
            {
                "quarter": q,
                "variant": v,
                "regime_class": rc,
                "trades": trades,
                "ev_per_trade_bps": ev,
                "win_rate": win_rate,
                "avg_win_bps": avg_win,
                "avg_loss_bps": avg_loss,
                "payoff": payoff,
                "p90_loss_bps": p90_loss,
                "duration_median_min": median_dur,
            }
        )

    return pd.DataFrame.from_records(records)


def build_report(df_metrics: pd.DataFrame, df_all: pd.DataFrame, ts: str) -> str:
    """
    Build markdown report for SNIPER 2025 regime split.
    """
    lines: List[str] = []
    lines.append(f"# SNIPER 2025 Regime Split Report ({ts})")
    lines.append("")
    lines.append(
        "This report analyzes existing SNIPER 2025 trade journals by coarse "
        "regime classes (A_TREND, B_MIXED, C_CHOP) without re-running any replay."
    )
    lines.append("")

    # Executive summary: which regime contributes most to EV, and which explains Q4 weakness
    lines.append("## Executive Summary")
    lines.append("")

    if not df_metrics.empty:
        overall = (
            df_metrics.groupby("regime_class")["ev_per_trade_bps"]
            .mean()
            .sort_values(ascending=False)
        )
        top_regime = overall.index[0]
        top_ev = overall.iloc[0]

        # Q4 weakness: compare regimes in Q4 / baseline
        q4_base = df_metrics[
            (df_metrics["quarter"] == "Q4") & (df_metrics["variant"] == "baseline")
        ]
        if not q4_base.empty:
            q4_by_regime = (
                q4_base.set_index("regime_class")["ev_per_trade_bps"]
                .sort_values(ascending=False)
            )
            weakest_regime = q4_by_regime.index[-1]
            weakest_ev = q4_by_regime.iloc[-1]
        else:
            weakest_regime = "N/A"
            weakest_ev = float("nan")

        lines.append(
            f"- **Best contributing regime (EV)**: `{top_regime}` "
            f"(overall mean EV ≈ {top_ev:.1f} bps per trade)."
        )
        lines.append(
            f"- **Regime most associated with Q4 baseline weakness**: "
            f"`{weakest_regime}` (Q4 baseline EV ≈ {weakest_ev:.1f} bps)."
        )
    else:
        lines.append("- No metrics computed; see sanity checks below.")
    lines.append("")

    # Per quarter / variant tables
    lines.append("## Per-quarter regime metrics")
    lines.append("")
    for q in sorted(df_metrics["quarter"].unique()):
        lines.append(f"### Quarter {q}")
        for variant in sorted(df_metrics["variant"].unique()):
            sub = df_metrics[
                (df_metrics["quarter"] == q) & (df_metrics["variant"] == variant)
            ]
            if sub.empty:
                continue
            lines.append(f"#### Variant: `{variant}`")
            lines.append("")
            lines.append(
                "| Regime | Trades | EV (bps) | Win% | Payoff | p90 loss (bps) | Median dur (min) |"
            )
            lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
            for _, row in (
                sub.sort_values("regime_class")[
                    [
                        "regime_class",
                        "trades",
                        "ev_per_trade_bps",
                        "win_rate",
                        "payoff",
                        "p90_loss_bps",
                        "duration_median_min",
                    ]
                ].iterrows()
            ):
                lines.append(
                    f"| {row['regime_class']} "
                    f"| {int(row['trades'])} "
                    f"| {row['ev_per_trade_bps']:.1f} "
                    f"| {row['win_rate']*100:.1f}% "
                    f"| {row['payoff']:.2f} "
                    f"| {row['p90_loss_bps']:.1f} "
                    f"| {row['duration_median_min']:.1f} |"
                )
            lines.append("")

    # Full-year aggregate per regime and variant
    lines.append("## Full-year regime metrics (Q1–Q4 combined)")
    lines.append("")
    if not df_metrics.empty:
        full_year = (
            df_metrics.groupby(["variant", "regime_class"])
            .agg(
                trades=("trades", "sum"),
                ev_per_trade_bps=("ev_per_trade_bps", "mean"),
                win_rate=("win_rate", "mean"),
                payoff=("payoff", "mean"),
                p90_loss_bps=("p90_loss_bps", "mean"),
                duration_median_min=("duration_median_min", "mean"),
            )
            .reset_index()
        )
        for variant in sorted(full_year["variant"].unique()):
            sub = full_year[full_year["variant"] == variant]
            lines.append(f"### Variant: `{variant}`")
            lines.append("")
            lines.append(
                "| Regime | Trades | EV (bps) | Win% | Payoff | p90 loss (bps) | Median dur (min) |"
            )
            lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
            for _, row in (
                sub.sort_values("regime_class")[
                    [
                        "regime_class",
                        "trades",
                        "ev_per_trade_bps",
                        "win_rate",
                        "payoff",
                        "p90_loss_bps",
                        "duration_median_min",
                    ]
                ].iterrows()
            ):
                lines.append(
                    f"| {row['regime_class']} "
                    f"| {int(row['trades'])} "
                    f"| {row['ev_per_trade_bps']:.1f} "
                    f"| {row['win_rate']*100:.1f}% "
                    f"| {row['payoff']:.2f} "
                    f"| {row['p90_loss_bps']:.1f} "
                    f"| {row['duration_median_min']:.1f} |"
                )
            lines.append("")
    else:
        lines.append("- No metrics available.")
        lines.append("")

    # Sanity checks
    lines.append("## Sanity checks")
    lines.append("")
    if not df_all.empty and not df_metrics.empty:
        # Check that trades per quarter/variant match sum over regimes
        base_counts = (
            df_all.groupby(["quarter", "variant"]).size().rename("trades_total")
        )
        regime_counts = (
            df_metrics.groupby(["quarter", "variant"])["trades"]
            .sum()
            .rename("trades_by_regime")
        )
        joined = pd.concat([base_counts, regime_counts], axis=1)
        joined["diff"] = joined["trades_total"] - joined["trades_by_regime"]

        bad = joined[joined["diff"] != 0]
        if bad.empty:
            lines.append(
                "- ✅ Sum of trades over A/B/C regimes matches total trades for each (quarter, variant)."
            )
        else:
            lines.append(
                "- ⚠️ Mismatch between total trades and regime-summed trades for some (quarter, variant):"
            )
            for idx, row in bad.reset_index().iterrows():
                lines.append(
                    f"  - {row['quarter']} / {row['variant']}: "
                    f"total={row['trades_total']}, by_regime={row['trades_by_regime']}, diff={row['diff']}"
                )
    else:
        lines.append("- No trades loaded; cannot perform trade-count sanity check.")
    lines.append("")

    # Coverage of regime classification
    if not df_all.empty:
        if "regime_class" in df_all.columns:
            missing = df_all["regime_class"].isna().sum()
            total = len(df_all)
            lines.append(
                f"- Regime classification coverage: {total - missing}/{total} trades "
                f"({(1 - missing / max(1, total)) * 100:.1f}% with a regime_class)."
            )
        else:
            lines.append("- ⚠️ Column `regime_class` missing in combined DataFrame.")
    lines.append("")

    # Next step (policy, no retrain)
    lines.append("## Next step (policy, no retrain)")
    lines.append("")
    lines.append(
        "- Consider using this regime split as a **gate** in policy space, without "
        "changing the learned models:"
    )
    lines.append(
        "  - For example, if `C_CHOP` regimes consistently show low or negative EV and "
        "unfavorable payoff, you may choose to **turn SNIPER off or reduce size** in "
        "those regimes, while keeping full size in `A_TREND` and `B_MIXED`."
    )
    lines.append(
        "- Any such gating should be evaluated with out-of-sample backtests, but the "
        "logic itself can remain a light‑weight overlay on top of existing models."
    )
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze SNIPER 2025 performance by regime class (offline)."
    )
    parser.add_argument(
        "--run-dirs",
        type=Path,
        nargs="*",
        default=None,
        help=(
            "Explicit SNIPER run directories (gx1/wf_runs/...). "
            "If omitted, the latest Q1–Q4 baseline/guarded runs are used."
        ),
    )
    args = parser.parse_args()

    run_infos: List[RunInfo]
    if args.run_dirs:
        run_infos = [_parse_run_info(rd) for rd in args.run_dirs]
    else:
        run_infos = _autodetect_run_dirs()

    if not run_infos:
        print("No SNIPER run_dirs found; nothing to analyze.")
        return 0

    all_dfs: List[pd.DataFrame] = []
    for ri in run_infos:
        try:
            df = load_trades_for_run(ri)
        except FileNotFoundError as e:
            print(f"Skipping {ri.run_dir}: {e}")
            continue
        df = classify_trades(df)
        all_dfs.append(df)

    if not all_dfs:
        print("No trades loaded; nothing to analyze.")
        return 0

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_metrics = aggregate_metrics(df_all)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_text = build_report(df_metrics, df_all, ts)

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"SNIPER_2025_REGIME_SPLIT_REPORT__{ts}.md"
    report_path.write_text(report_text, encoding="utf-8")

    print(f"Wrote regime split report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


