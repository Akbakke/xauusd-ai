#!/usr/bin/env python3
"""
Analyze Q1 Depth Ladder attribution to explain PnL loss.

This script reads existing Q1 artifacts and explains where PnL disappeared.
No new replays are run - this is purely truth-driven analysis.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

def load_trade_outcomes(trade_file: Path) -> pd.DataFrame:
    """Load trade outcomes parquet file."""
    if not trade_file.exists():
        raise FileNotFoundError(f"Trade outcomes file not found: {trade_file}")
    return pd.read_parquet(trade_file)

def load_metrics(metrics_file: Path) -> Dict[str, Any]:
    """Load metrics JSON file."""
    if not metrics_file.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")
    with open(metrics_file, "r") as f:
        return json.load(f)

def analyze_session_pnl(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Analyze PnL by session (EU, OVERLAP, US)."""
    if "session" not in df.columns:
        # Try to infer from timestamp or use default
        df["session"] = "UNKNOWN"
    
    session_stats = {}
    for session in ["EU", "OVERLAP", "US"]:
        session_df = df[df["session"] == session]
        if len(session_df) > 0:
            session_stats[session] = {
                "total_pnl_bps": float(session_df["pnl_bps"].sum()),
                "mean_pnl_bps": float(session_df["pnl_bps"].mean()),
                "median_pnl_bps": float(session_df["pnl_bps"].median()),
                "n_trades": len(session_df),
                "n_winners": int((session_df["pnl_bps"] > 0).sum()),
                "n_losers": int((session_df["pnl_bps"] < 0).sum()),
            }
        else:
            session_stats[session] = {
                "total_pnl_bps": 0.0,
                "mean_pnl_bps": 0.0,
                "median_pnl_bps": 0.0,
                "n_trades": 0,
                "n_winners": 0,
                "n_losers": 0,
            }
    
    return session_stats

def analyze_tail_losses(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze tail losses (P5, P50, P95) and bottom 5% losses."""
    pnl_sorted = df["pnl_bps"].sort_values()
    
    # Percentiles
    p5 = float(np.percentile(pnl_sorted, 5))
    p50 = float(np.median(pnl_sorted))
    p95 = float(np.percentile(pnl_sorted, 95))
    
    # Bottom 5% losses
    bottom_5pct_threshold = np.percentile(pnl_sorted, 5)
    bottom_5pct = pnl_sorted[pnl_sorted <= bottom_5pct_threshold]
    
    tail_analysis = {
        "p5_loss_bps": p5,
        "p50_bps": p50,
        "p95_bps": p95,
        "bottom_5pct_count": len(bottom_5pct),
        "bottom_5pct_avg_loss_bps": float(bottom_5pct.mean()) if len(bottom_5pct) > 0 else 0.0,
        "bottom_5pct_total_loss_bps": float(bottom_5pct.sum()) if len(bottom_5pct) > 0 else 0.0,
    }
    
    return tail_analysis

def generate_attribution_report(
    baseline_trades: pd.DataFrame,
    lplus1_trades: pd.DataFrame,
    baseline_metrics: Dict[str, Any],
    lplus1_metrics: Dict[str, Any],
    output_path: Path
) -> None:
    """Generate attribution analysis report."""
    
    # Session analysis
    baseline_sessions = analyze_session_pnl(baseline_trades)
    lplus1_sessions = analyze_session_pnl(lplus1_trades)
    
    # Tail analysis
    baseline_tail = analyze_tail_losses(baseline_trades)
    lplus1_tail = analyze_tail_losses(lplus1_trades)
    
    # Overall stats
    baseline_total_pnl = float(baseline_trades["pnl_bps"].sum())
    lplus1_total_pnl = float(lplus1_trades["pnl_bps"].sum())
    pnl_delta = lplus1_total_pnl - baseline_total_pnl
    
    baseline_winners = int((baseline_trades["pnl_bps"] > 0).sum())
    lplus1_winners = int((lplus1_trades["pnl_bps"] > 0).sum())
    winners_delta = lplus1_winners - baseline_winners
    
    baseline_losers = int((baseline_trades["pnl_bps"] < 0).sum())
    lplus1_losers = int((lplus1_trades["pnl_bps"] < 0).sum())
    losers_delta = lplus1_losers - baseline_losers
    
    baseline_avg_win = float(baseline_trades[baseline_trades["pnl_bps"] > 0]["pnl_bps"].mean()) if baseline_winners > 0 else 0.0
    lplus1_avg_win = float(lplus1_trades[lplus1_trades["pnl_bps"] > 0]["pnl_bps"].mean()) if lplus1_winners > 0 else 0.0
    
    baseline_avg_loss = float(baseline_trades[baseline_trades["pnl_bps"] < 0]["pnl_bps"].mean()) if baseline_losers > 0 else 0.0
    lplus1_avg_loss = float(lplus1_trades[lplus1_trades["pnl_bps"] < 0]["pnl_bps"].mean()) if lplus1_losers > 0 else 0.0
    
    # Generate conclusion
    conclusion = []
    if winners_delta < 0:
        conclusion.append(f"PnL-tap skyldes færre winners ({baseline_winners} → {lplus1_winners}, delta: {winners_delta})")
    if abs(lplus1_avg_loss) > abs(baseline_avg_loss) * 1.1:
        conclusion.append(f"PnL-tap skyldes større losses (avg loss: {baseline_avg_loss:.2f} → {lplus1_avg_loss:.2f} bps)")
    
    # Check for session-specific edge loss
    session_edge_loss = False
    for session in ["EU", "OVERLAP", "US"]:
        baseline_session_pnl = baseline_sessions[session]["total_pnl_bps"]
        lplus1_session_pnl = lplus1_sessions[session]["total_pnl_bps"]
        if baseline_session_pnl > 0 and lplus1_session_pnl < baseline_session_pnl * 0.5:
            session_edge_loss = True
            conclusion.append(f"Session-spesifikk edge forsvinner i {session} (baseline: {baseline_session_pnl:.2f} bps → L+1: {lplus1_session_pnl:.2f} bps)")
    
    if not conclusion:
        conclusion.append("PnL-tap er jevnt fordelt over alle dimensjoner")
    
    # Generate markdown report
    report = f"""# Depth Ladder Q1 Attribution Analysis

**Date:** {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary

| Metric | Baseline | L+1 | Delta |
|--------|----------|-----|-------|
| Total PnL (bps) | {baseline_total_pnl:.2f} | {lplus1_total_pnl:.2f} | {pnl_delta:+.2f} |
| Winners | {baseline_winners} | {lplus1_winners} | {winners_delta:+d} |
| Losers | {baseline_losers} | {lplus1_losers} | {losers_delta:+d} |
| Avg Win (bps) | {baseline_avg_win:.2f} | {lplus1_avg_win:.2f} | {lplus1_avg_win - baseline_avg_win:+.2f} |
| Avg Loss (bps) | {baseline_avg_loss:.2f} | {lplus1_avg_loss:.2f} | {lplus1_avg_loss - baseline_avg_loss:+.2f} |

---

## Session PnL Analysis

### Baseline

| Session | Total PnL (bps) | Trades | Winners | Losers | Mean PnL (bps) |
|---------|-----------------|--------|---------|--------|----------------|
"""
    
    for session in ["EU", "OVERLAP", "US"]:
        stats = baseline_sessions[session]
        report += f"| {session} | {stats['total_pnl_bps']:.2f} | {stats['n_trades']} | {stats['n_winners']} | {stats['n_losers']} | {stats['mean_pnl_bps']:.2f} |\n"
    
    report += "\n### L+1\n\n| Session | Total PnL (bps) | Trades | Winners | Losers | Mean PnL (bps) |\n"
    report += "|---------|-----------------|--------|---------|--------|----------------|\n"
    
    for session in ["EU", "OVERLAP", "US"]:
        stats = lplus1_sessions[session]
        report += f"| {session} | {stats['total_pnl_bps']:.2f} | {stats['n_trades']} | {stats['n_winners']} | {stats['n_losers']} | {stats['mean_pnl_bps']:.2f} |\n"
    
    report += "\n### Delta\n\n| Session | PnL Delta (bps) | Trades Delta | Winners Delta | Losers Delta |\n"
    report += "|---------|-----------------|--------------|---------------|--------------|\n"
    
    for session in ["EU", "OVERLAP", "US"]:
        baseline_stats = baseline_sessions[session]
        lplus1_stats = lplus1_sessions[session]
        report += f"| {session} | {lplus1_stats['total_pnl_bps'] - baseline_stats['total_pnl_bps']:+.2f} | {lplus1_stats['n_trades'] - baseline_stats['n_trades']:+d} | {lplus1_stats['n_winners'] - baseline_stats['n_winners']:+d} | {lplus1_stats['n_losers'] - baseline_stats['n_losers']:+d} |\n"
    
    report += "\n---\n\n## Tail Analysis\n\n"
    report += "### Baseline\n\n"
    report += f"| Metric | Value |\n"
    report += f"|--------|-------|\n"
    report += f"| P5 Loss (bps) | {baseline_tail['p5_loss_bps']:.2f} |\n"
    report += f"| P50 (bps) | {baseline_tail['p50_bps']:.2f} |\n"
    report += f"| P95 (bps) | {baseline_tail['p95_bps']:.2f} |\n"
    report += f"| Bottom 5% Count | {baseline_tail['bottom_5pct_count']} |\n"
    report += f"| Bottom 5% Avg Loss (bps) | {baseline_tail['bottom_5pct_avg_loss_bps']:.2f} |\n"
    report += f"| Bottom 5% Total Loss (bps) | {baseline_tail['bottom_5pct_total_loss_bps']:.2f} |\n"
    
    report += "\n### L+1\n\n"
    report += f"| Metric | Value |\n"
    report += f"|--------|-------|\n"
    report += f"| P5 Loss (bps) | {lplus1_tail['p5_loss_bps']:.2f} |\n"
    report += f"| P50 (bps) | {lplus1_tail['p50_bps']:.2f} |\n"
    report += f"| P95 (bps) | {lplus1_tail['p95_bps']:.2f} |\n"
    report += f"| Bottom 5% Count | {lplus1_tail['bottom_5pct_count']} |\n"
    report += f"| Bottom 5% Avg Loss (bps) | {lplus1_tail['bottom_5pct_avg_loss_bps']:.2f} |\n"
    report += f"| Bottom 5% Total Loss (bps) | {lplus1_tail['bottom_5pct_total_loss_bps']:.2f} |\n"
    
    report += "\n### Delta\n\n"
    report += f"| Metric | Delta |\n"
    report += f"|--------|-------|\n"
    report += f"| P5 Loss (bps) | {lplus1_tail['p5_loss_bps'] - baseline_tail['p5_loss_bps']:+.2f} |\n"
    report += f"| P50 (bps) | {lplus1_tail['p50_bps'] - baseline_tail['p50_bps']:+.2f} |\n"
    report += f"| P95 (bps) | {lplus1_tail['p95_bps'] - baseline_tail['p95_bps']:+.2f} |\n"
    report += f"| Bottom 5% Count | {lplus1_tail['bottom_5pct_count'] - baseline_tail['bottom_5pct_count']:+d} |\n"
    report += f"| Bottom 5% Avg Loss (bps) | {lplus1_tail['bottom_5pct_avg_loss_bps'] - baseline_tail['bottom_5pct_avg_loss_bps']:+.2f} |\n"
    report += f"| Bottom 5% Total Loss (bps) | {lplus1_tail['bottom_5pct_total_loss_bps'] - baseline_tail['bottom_5pct_total_loss_bps']:+.2f} |\n"
    
    report += "\n---\n\n## Conclusion\n\n"
    for line in conclusion:
        report += f"- {line}\n"
    
    # Write report
    with open(output_path, "w") as f:
        f.write(report)
    
    print(f"✅ Attribution report written: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze Q1 Depth Ladder attribution")
    parser.add_argument("--baseline-root", type=Path, required=True, help="Baseline Q1 root directory")
    parser.add_argument("--lplus1-root", type=Path, required=True, help="L+1 Q1 root directory")
    parser.add_argument("--out-root", type=Path, required=True, help="Output root directory")
    
    args = parser.parse_args()
    
    # Find trade outcomes files
    baseline_trade_files = list(args.baseline_root.glob("**/trade_outcomes_*_MERGED.parquet"))
    lplus1_trade_files = list(args.lplus1_root.glob("**/trade_outcomes_*_MERGED.parquet"))
    
    if not baseline_trade_files:
        raise FileNotFoundError(f"No trade outcomes file found in {args.baseline_root}")
    if not lplus1_trade_files:
        raise FileNotFoundError(f"No trade outcomes file found in {args.lplus1_root}")
    
    # Find metrics files
    baseline_metrics_files = list(args.baseline_root.glob("**/metrics_*_MERGED.json"))
    lplus1_metrics_files = list(args.lplus1_root.glob("**/metrics_*_MERGED.json"))
    
    if not baseline_metrics_files:
        raise FileNotFoundError(f"No metrics file found in {args.baseline_root}")
    if not lplus1_metrics_files:
        raise FileNotFoundError(f"No metrics file found in {args.lplus1_root}")
    
    # Load data
    baseline_trades = load_trade_outcomes(baseline_trade_files[0])
    lplus1_trades = load_trade_outcomes(lplus1_trade_files[0])
    baseline_metrics = load_metrics(baseline_metrics_files[0])
    lplus1_metrics = load_metrics(lplus1_metrics_files[0])
    
    # Generate report
    output_path = args.out_root / "DEPTH_LADDER_Q1_ATTRIBUTION.md"
    generate_attribution_report(
        baseline_trades,
        lplus1_trades,
        baseline_metrics,
        lplus1_metrics,
        output_path
    )

if __name__ == "__main__":
    main()
