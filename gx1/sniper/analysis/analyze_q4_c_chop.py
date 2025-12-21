#!/usr/bin/env python3
"""
Analyze Q4 × C_CHOP performance for SNIPER (baseline and guarded) using
existing JSON trade journals only (no replay, no policy changes).

Goals:
- Quantify C_CHOP edge in Q4 for baseline and guarded.
- Explore hypothetical size multipliers for C_CHOP trades:
  [1.0, 0.75, 0.50, 0.30]
- Provide tail-risk vs EV tradeoff to inform future policy overlays.

This script:
- Autodetects latest Q4 2025 baseline and guarded run_dirs.
- Loads trades from trade_journal/trades/*.json (JSON-first).
- Classifies regimes with classify_regime(...).
- Filters to C_CHOP trades only.
- Computes core metrics and size-sensitivity tables.
- Optionally breaks out by session where there are ≥200 trades.

Output:
- reports/SNIPER_Q4_C_CHOP_ANALYSIS__YYYYMMDD_HHMMSS.md
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from gx1.sniper.analysis.regime_classifier import classify_regime


SNIPER_BASE = Path("gx1/wf_runs")
SIZE_MULTIPLIERS = [1.0, 0.75, 0.50, 0.30]


@dataclass
class Q4Run:
    name: str  # "baseline" or "guarded"
    path: Optional[Path]


def _latest_run(pattern: str) -> Optional[Path]:
    matches = sorted(SNIPER_BASE.glob(pattern))
    return matches[-1] if matches else None


def autodetect_q4_runs() -> List[Q4Run]:
    return [
        Q4Run("baseline", _latest_run("SNIPER_OBS_Q4_2025_baseline_*")),
        Q4Run("guarded", _latest_run("SNIPER_OBS_Q4_2025_guarded_*")),
    ]


def _first_non_null(*candidates):
    for v in candidates:
        if v is not None:
            return v
    return None


def trade_to_row(trade: Dict[str, Any], trade_file: str) -> Dict[str, Any]:
    """
    Extract fields needed for C_CHOP analysis from a trade JSON.
    """
    entry = trade.get("entry_snapshot") or {}
    feature_ctx = entry.get("feature_context") or {}
    exit_summary = trade.get("exit_summary") or entry.get("exit_summary") or {}

    # pnl_bps
    pnl = exit_summary.get("realized_pnl_bps")
    if pnl is None:
        pnl = exit_summary.get("pnl_bps")
    if pnl is None:
        pnl = trade.get("pnl_bps")

    # Regime inputs
    trend_regime = _first_non_null(
        entry.get("trend_regime"),
        feature_ctx.get("trend_regime"),
        trade.get("trend_regime"),
    )
    vol_regime = _first_non_null(
        entry.get("vol_regime"),
        feature_ctx.get("vol_regime"),
        trade.get("vol_regime"),
    )
    atr_bps = _first_non_null(
        entry.get("atr_bps"),
        feature_ctx.get("atr_bps"),
        trade.get("atr_bps"),
    )
    spread_bps = _first_non_null(
        entry.get("spread_bps"),
        feature_ctx.get("spread_bps"),
        trade.get("spread_bps"),
    )
    session = _first_non_null(
        entry.get("session"),
        feature_ctx.get("session"),
        trade.get("session"),
    )

    # Duration (if available): prefer explicit minutes, else derive from times
    duration_minutes = exit_summary.get("duration_minutes")
    if duration_minutes is None:
        try:
            entry_time = _first_non_null(
                entry.get("entry_time"), trade.get("entry_time")
            )
            exit_time = _first_non_null(
                exit_summary.get("exit_time"), trade.get("exit_time")
            )
            if entry_time and exit_time:
                t_entry = pd.to_datetime(entry_time, utc=True, errors="coerce")
                t_exit = pd.to_datetime(exit_time, utc=True, errors="coerce")
                if pd.notna(t_entry) and pd.notna(t_exit):
                    duration_minutes = (t_exit - t_entry).total_seconds() / 60.0
        except Exception:
            duration_minutes = None

    return {
        "trade_id": trade.get("trade_id"),
        "trade_file": trade_file,
        "pnl_bps": pnl,
        "trend_regime": trend_regime,
        "vol_regime": vol_regime,
        "atr_bps": atr_bps,
        "spread_bps": spread_bps,
        "session": session,
        "duration_minutes": duration_minutes,
    }


def load_trades_json(run_dir: Path) -> pd.DataFrame:
    trades_dir = run_dir / "trade_journal" / "trades"
    if not trades_dir.exists():
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for json_file in sorted(trades_dir.glob("*.json")):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                trade = json.load(f)
        except Exception:
            continue
        if not isinstance(trade, dict):
            continue
        rows.append(trade_to_row(trade, json_file.name))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["pnl_bps"] = pd.to_numeric(df["pnl_bps"], errors="coerce")
    df["atr_bps"] = pd.to_numeric(df["atr_bps"], errors="coerce")
    df["spread_bps"] = pd.to_numeric(df["spread_bps"], errors="coerce")
    df["duration_minutes"] = pd.to_numeric(df["duration_minutes"], errors="coerce")
    return df


def attach_regime(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        df_local = df.copy()
        df_local["regime_class"] = None
        df_local["regime_reason"] = None
        return df_local
    classes: List[Optional[str]] = []
    reasons: List[Optional[str]] = []
    for _, row in df.iterrows():
        try:
            c, r = classify_regime(row.to_dict())
        except Exception as exc:
            c, r = None, f"classify_error:{type(exc).__name__}"
        classes.append(c)
        reasons.append(r)
    df_local = df.copy()
    df_local["regime_class"] = classes
    df_local["regime_reason"] = reasons
    return df_local


def compute_core_metrics(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty or "pnl_bps" not in df.columns:
        return {
            "trades": 0,
            "ev_per_trade_bps": 0.0,
            "win_rate": 0.0,
            "payoff": float("nan"),
            "p90_loss_bps": 0.0,
            "median_duration_min": float("nan"),
            "p90_duration_min": float("nan"),
        }
    pnl = pd.to_numeric(df["pnl_bps"], errors="coerce").dropna()
    trades = len(pnl)
    if trades == 0:
        return {
            "trades": 0,
            "ev_per_trade_bps": 0.0,
            "win_rate": 0.0,
            "payoff": float("nan"),
            "p90_loss_bps": 0.0,
            "median_duration_min": float("nan"),
            "p90_duration_min": float("nan"),
        }
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    ev = float(pnl.mean())
    win_rate = float((pnl > 0).mean())
    avg_win = float(wins.mean()) if not wins.empty else float("nan")
    avg_loss = float(losses.mean()) if not losses.empty else float("nan")
    payoff = (
        float(avg_win / abs(avg_loss))
        if not wins.empty and not losses.empty and avg_loss != 0
        else float("nan")
    )
    if not losses.empty:
        loss_mag = losses.abs().astype(float)
        p90_loss = float(np.percentile(loss_mag.values, 90))
    else:
        p90_loss = 0.0

    dur = pd.to_numeric(df.get("duration_minutes"), errors="coerce").dropna()
    if not dur.empty:
        median_dur = float(np.median(dur.values))
        p90_dur = float(np.percentile(dur.values, 90))
    else:
        median_dur = float("nan")
        p90_dur = float("nan")

    return {
        "trades": trades,
        "ev_per_trade_bps": ev,
        "win_rate": win_rate,
        "payoff": payoff,
        "p90_loss_bps": p90_loss,
        "median_duration_min": median_dur,
        "p90_duration_min": p90_dur,
    }


def compute_size_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Offline size sensitivity for C_CHOP trades.
    """
    results: List[Dict[str, Any]] = []
    if df.empty or "pnl_bps" not in df.columns:
        return pd.DataFrame()

    base_pnl = pd.to_numeric(df["pnl_bps"], errors="coerce").dropna()
    if base_pnl.empty:
        return pd.DataFrame()

    for m in SIZE_MULTIPLIERS:
        pnl_scaled = base_pnl * m
        wins = pnl_scaled[pnl_scaled > 0]
        losses = pnl_scaled[pnl_scaled < 0]
        ev = float(pnl_scaled.mean())
        win_rate = float((pnl_scaled > 0).mean())
        avg_win = float(wins.mean()) if not wins.empty else float("nan")
        avg_loss = float(losses.mean()) if not losses.empty else float("nan")
        payoff = (
            float(avg_win / abs(avg_loss))
            if not wins.empty and not losses.empty and avg_loss != 0
            else float("nan")
        )
        if not losses.empty:
            loss_mag = losses.abs().astype(float)
            p90_loss = float(np.percentile(loss_mag.values, 90))
        else:
            p90_loss = 0.0

        results.append(
            {
                "multiplier": m,
                "ev_per_trade_bps": ev,
                "win_rate": win_rate,
                "payoff": payoff,
                "p90_loss_bps": p90_loss,
            }
        )
    return pd.DataFrame(results)


def _metrics_table(name: str, metrics: Dict[str, float]) -> List[str]:
    lines: List[str] = []
    lines.append(f"### Q4 × C_CHOP core metrics – {name}")
    lines.append("")
    lines.append("| trades | EV (bps) | Win% | Payoff | p90 loss (bps) | median dur (min) | p90 dur (min) |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    lines.append(
        f"| {metrics['trades']} | {metrics['ev_per_trade_bps']:.2f} | "
        f"{metrics['win_rate']*100:.1f}% | {metrics['payoff']:.2f} | "
        f"{metrics['p90_loss_bps']:.1f} | "
        f"{metrics['median_duration_min']:.1f} | {metrics['p90_duration_min']:.1f} |"
    )
    lines.append("")
    return lines


def _size_sensitivity_table(name: str, df_sens: pd.DataFrame) -> List[str]:
    lines: List[str] = []
    lines.append(f"### Q4 × C_CHOP size sensitivity – {name}")
    lines.append("")
    if df_sens.empty:
        lines.append("_No C_CHOP trades; size sensitivity not available._")
        lines.append("")
        return lines
    lines.append("| multiplier | EV (bps) | Win% | Payoff | p90 loss (bps) |")
    lines.append("| ---: | ---: | ---: | ---: | ---: |")
    for _, row in df_sens.iterrows():
        lines.append(
            f"| {row['multiplier']:.2f} | {row['ev_per_trade_bps']:.2f} | "
            f"{row['win_rate']*100:.1f}% | {row['payoff']:.2f} | "
            f"{row['p90_loss_bps']:.1f} |"
        )
    lines.append("")
    return lines


def _session_blocks(name: str, df_c: pd.DataFrame) -> List[str]:
    lines: List[str] = []
    if "session" not in df_c.columns or df_c.empty:
        return lines
    for session, df_s in df_c.groupby("session"):
        if len(df_s) < 200:
            continue
        metrics = compute_core_metrics(df_s)
        sens = compute_size_sensitivity(df_s)
        lines.append(f"#### Session: {session} – {name}")
        lines.append("")
        lines.extend(_metrics_table("session", metrics))
        lines.extend(_size_sensitivity_table("session", sens))
    return lines


def build_report(runs: List[Q4Run], ts: str) -> str:
    lines: List[str] = []
    lines.append(f"# SNIPER Q4 C_CHOP Analysis ({ts})")
    lines.append("")
    lines.append(
        "Offline analysis of Q4 2025 C_CHOP trades for baseline and guarded. "
        "No replay; JSON trade journals only."
    )
    lines.append("")

    results: Dict[str, Dict[str, Any]] = {}

    for run in runs:
        if run.path is None:
            continue
        df = load_trades_json(run.path)
        df = attach_regime(df)
        df_c = df[df["regime_class"] == "C_CHOP"].copy()
        core = compute_core_metrics(df_c)
        sens = compute_size_sensitivity(df_c)
        results[run.name] = {
            "df_all": df,
            "df_c": df_c,
            "core": core,
            "sens": sens,
        }

    # Executive summary
    lines.append("## Executive summary")
    lines.append("")
    for name in ["baseline", "guarded"]:
        if name not in results:
            continue
        core = results[name]["core"]
        lines.append(
            f"- {name}: Q4 × C_CHOP trades={core['trades']}, "
            f"EV={core['ev_per_trade_bps']:.2f} bps, "
            f"p90 loss={core['p90_loss_bps']:.1f} bps"
        )
    lines.append("")

    # Per-run sections
    for name in ["baseline", "guarded"]:
        if name not in results:
            continue
        core = results[name]["core"]
        sens = results[name]["sens"]
        df_c = results[name]["df_c"]

        lines.append(f"## Variant: {name}")
        lines.append("")
        lines.append(f"- Total C_CHOP trades: {core['trades']}")
        lines.append("")
        lines.extend(_metrics_table(name, core))
        lines.extend(_size_sensitivity_table(name, sens))
        lines.extend(_session_blocks(name, df_c))

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze Q4 × C_CHOP performance and size sensitivity (JSON-only)."
    )
    _ = parser.parse_args()

    runs = autodetect_q4_runs()
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_text = build_report(runs, ts)

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"SNIPER_Q4_C_CHOP_ANALYSIS__{ts}.md"
    report_path.write_text(report_text, encoding="utf-8")

    print(f"Wrote Q4 C_CHOP analysis report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


