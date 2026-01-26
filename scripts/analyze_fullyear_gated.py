#!/usr/bin/env python3
"""
Analyze FULLYEAR PREBUILT GATED replay results.

Generates comprehensive analysis report with:
- High-level metrics (PnL, winrate, DD, tails)
- Session/regime breakdown
- Eligibility blocks stats
- Top winners/losers
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import Counter
from datetime import datetime
import pandas as pd
import numpy as np

def load_trades_from_chunks(run_dir: Path) -> List[Dict[str, Any]]:
    """Load all trades from chunk trade journals."""
    trades = []
    trade_journal_dirs = list(run_dir.glob("chunk_*/trade_journal/trades/*.json"))
    
    for json_file in trade_journal_dirs:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                trade = json.load(f)
            trades.append(trade)
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}", file=sys.stderr)
    
    return trades

def extract_trade_metrics(trade: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract metrics from a single trade."""
    exit_summary = trade.get("exit_summary") or {}
    entry_snapshot = trade.get("entry_snapshot") or {}
    feature_context = trade.get("feature_context") or {}
    
    # PnL
    pnl_bps = exit_summary.get("realized_pnl_bps") or exit_summary.get("pnl_bps") or trade.get("pnl_bps")
    if pnl_bps is None:
        return None
    
    # Timestamps
    entry_time = entry_snapshot.get("entry_time") or trade.get("entry_time")
    exit_time = exit_summary.get("exit_time") or trade.get("exit_time")
    
    # Regime fields
    session = entry_snapshot.get("session") or feature_context.get("session") or trade.get("session")
    trend_regime = entry_snapshot.get("trend_regime") or feature_context.get("trend_regime") or trade.get("trend_regime")
    vol_regime = entry_snapshot.get("vol_regime") or feature_context.get("vol_regime") or trade.get("vol_regime")
    
    # Exit reason
    exit_reason = exit_summary.get("exit_reason") or trade.get("exit_reason")
    
    return {
        "trade_id": trade.get("trade_id"),
        "pnl_bps": float(pnl_bps),
        "entry_time": entry_time,
        "exit_time": exit_time,
        "session": session,
        "trend_regime": trend_regime,
        "vol_regime": vol_regime,
        "exit_reason": exit_reason,
    }

def compute_metrics(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate metrics from trades."""
    if not trades:
        return {
            "n_trades": 0,
            "pnl_total": 0.0,
            "pnl_mean": 0.0,
            "winrate": 0.0,
            "max_dd": 0.0,
            "p5": 0.0,
            "p95": 0.0,
            "worst_trade": None,
            "best_trade": None,
        }
    
    pnls = [t["pnl_bps"] for t in trades]
    wins = [p for p in pnls if p > 0]
    
    # Drawdown
    equity = np.cumsum(pnls)
    running_max = np.maximum.accumulate(equity)
    drawdown = equity - running_max
    max_dd = float(np.min(drawdown)) if len(drawdown) > 0 else 0.0
    
    # Percentiles
    p5 = float(np.percentile(pnls, 5))
    p95 = float(np.percentile(pnls, 95))
    
    # Worst/best
    worst_idx = np.argmin(pnls)
    best_idx = np.argmax(pnls)
    
    return {
        "n_trades": len(trades),
        "pnl_total": float(np.sum(pnls)),
        "pnl_mean": float(np.mean(pnls)),
        "winrate": len(wins) / len(pnls) if pnls else 0.0,
        "max_dd": max_dd,
        "p5": p5,
        "p95": p95,
        "worst_trade": trades[worst_idx] if worst_idx < len(trades) else None,
        "best_trade": trades[best_idx] if best_idx < len(trades) else None,
    }

def load_footer_stats(run_dir: Path) -> Dict[str, Any]:
    """Load eligibility stats from chunk footers."""
    footers = list(run_dir.glob("chunk_*/chunk_footer.json"))
    
    total_lookup_attempts = 0
    total_lookup_hits = 0
    total_eligibility_blocks = 0
    total_bars_processed = 0
    
    for footer_path in footers:
        try:
            with open(footer_path, "r") as f:
                footer = json.load(f)
            total_lookup_attempts += footer.get("lookup_attempts", 0)
            total_lookup_hits += footer.get("lookup_hits", 0)
            total_eligibility_blocks += footer.get("lookup_misses", 0)
            total_bars_processed += footer.get("bars_processed", 0)
        except Exception:
            pass
    
    return {
        "lookup_attempts": total_lookup_attempts,
        "lookup_hits": total_lookup_hits,
        "eligibility_blocks": total_eligibility_blocks,
        "bars_processed": total_bars_processed,
        "eligibility_block_rate": total_eligibility_blocks / total_lookup_attempts if total_lookup_attempts > 0 else 0.0,
    }

def generate_report(run_dir: Path, output_path: Path) -> None:
    """Generate analysis report."""
    print(f"Loading trades from: {run_dir}")
    all_trades = load_trades_from_chunks(run_dir)
    print(f"Loaded {len(all_trades)} trades")
    
    # Extract metrics
    trade_rows = []
    for trade in all_trades:
        row = extract_trade_metrics(trade)
        if row:
            trade_rows.append(row)
    
    if not trade_rows:
        report = f"# FULLYEAR PREBUILT GATED Analysis\n\n**Run**: `{run_dir.name}`\n\n**ERROR**: No valid trades found.\n"
        with open(output_path, "w") as f:
            f.write(report)
        return
    
    df = pd.DataFrame(trade_rows)
    
    # Overall metrics
    overall = compute_metrics(trade_rows)
    
    # Footer stats (eligibility)
    footer_stats = load_footer_stats(run_dir)
    
    # Session breakdown
    session_stats = {}
    for session in ["ASIA", "EU", "OVERLAP", "US"]:
        session_trades = [t for t in trade_rows if t.get("session") == session]
        if session_trades:
            session_stats[session] = compute_metrics(session_trades)
    
    # Regime breakdown (if available)
    regime_stats = {}
    for regime in ["A_TREND", "B_MIXED", "C_CHOP", "TREND_UP", "TREND_DOWN", "TREND_NEUTRAL"]:
        regime_trades = [t for t in trade_rows if t.get("trend_regime") == regime or t.get("vol_regime") == regime]
        if regime_trades:
            regime_stats[regime] = compute_metrics(regime_trades)
    
    # Top winners/losers
    sorted_trades = sorted(trade_rows, key=lambda x: x["pnl_bps"])
    top_losers = sorted_trades[:20]
    top_winners = sorted_trades[-20:][::-1]
    
    # Run context
    run_id = run_dir.name
    perf_json = run_dir / f"perf_{run_id}.json"
    run_ctx = {}
    if perf_json.exists():
        try:
            with open(perf_json, "r") as f:
                perf = json.load(f)
            run_ctx = {
                "head": perf.get("git_head"),
                "policy": perf.get("policy_path"),
                "dataset": perf.get("data_path"),
                "prebuilt_sha": perf.get("features_file_sha256"),
            }
        except Exception:
            pass
    
    # Generate report
    report_lines = [
        "# FULLYEAR PREBUILT GATED Analysis",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Run ID:** `{run_id}`",
        f"**Source:** `{run_dir}`",
        "",
        "## RUN_CTX",
        "",
        f"- **root:** {run_dir.parent.parent}",
        f"- **head:** {run_ctx.get('head', 'unknown')}",
        f"- **run-id:** {run_id}",
        f"- **policy:** {run_ctx.get('policy', 'unknown')}",
        f"- **dataset:** {run_ctx.get('dataset', 'unknown')}",
        f"- **prebuilt sha:** {run_ctx.get('prebuilt_sha', 'unknown')}",
        "",
        "## High-Level Metrics",
        "",
        f"- **n_trades:** {overall['n_trades']:,}",
        f"- **pnl_total:** {overall['pnl_total']:.2f} bps",
        f"- **avg_trade:** {overall['pnl_mean']:.2f} bps",
        f"- **winrate:** {overall['winrate']:.1%}",
        f"- **maxDD:** {overall['max_dd']:.2f} bps",
        f"- **p5/p95:** {overall['p5']:.2f} / {overall['p95']:.2f} bps",
        f"- **worst_trade:** {overall['worst_trade']['pnl_bps']:.2f} bps" if overall['worst_trade'] else "-",
        f"- **best_trade:** {overall['best_trade']['pnl_bps']:.2f} bps" if overall['best_trade'] else "-",
        "",
        "## Eligibility Stats",
        "",
        f"- **lookup_attempts:** {footer_stats['lookup_attempts']:,}",
        f"- **lookup_hits:** {footer_stats['lookup_hits']:,}",
        f"- **eligibility_blocks:** {footer_stats['eligibility_blocks']:,}",
        f"- **eligibility_block_rate:** {footer_stats['eligibility_block_rate']:.1%}",
        f"- **bars_processed:** {footer_stats['bars_processed']:,}",
        "",
    ]
    
    # Session breakdown
    if session_stats:
        report_lines.extend([
            "## Session Breakdown",
            "",
            "| Session | Trades | EV/Trade | Winrate | P5 | P95 |",
            "|---------|--------|----------|---------|----|-----|",
        ])
        for session in ["ASIA", "EU", "OVERLAP", "US"]:
            if session in session_stats:
                s = session_stats[session]
                report_lines.append(
                    f"| {session} | {s['n_trades']} | {s['pnl_mean']:.2f} | {s['winrate']:.1%} | {s['p5']:.2f} | {s['p95']:.2f} |"
                )
        report_lines.append("")
    
    # Regime breakdown
    if regime_stats:
        report_lines.extend([
            "## Regime Breakdown",
            "",
            "| Regime | Trades | EV/Trade | Winrate | P5 | P95 |",
            "|--------|--------|----------|---------|----|-----|",
        ])
        for regime in sorted(regime_stats.keys()):
            r = regime_stats[regime]
            report_lines.append(
                f"| {regime} | {r['n_trades']} | {r['pnl_mean']:.2f} | {r['winrate']:.1%} | {r['p5']:.2f} | {r['p95']:.2f} |"
            )
        report_lines.append("")
    
    # Top losers
    report_lines.extend([
        "## Top 20 Losers",
        "",
        "| Trade ID | PnL (bps) | Session | Exit Reason |",
        "|----------|-----------|---------|-------------|",
    ])
    for trade in top_losers:
        report_lines.append(
            f"| {trade['trade_id']} | {trade['pnl_bps']:.2f} | {trade.get('session', 'N/A')} | {trade.get('exit_reason', 'N/A')} |"
        )
    report_lines.append("")
    
    # Top winners
    report_lines.extend([
        "## Top 20 Winners",
        "",
        "| Trade ID | PnL (bps) | Session | Exit Reason |",
        "|----------|-----------|---------|-------------|",
    ])
    for trade in top_winners:
        report_lines.append(
            f"| {trade['trade_id']} | {trade['pnl_bps']:.2f} | {trade.get('session', 'N/A')} | {trade.get('exit_reason', 'N/A')} |"
        )
    report_lines.append("")
    
    # Write report
    report = "\n".join(report_lines)
    with open(output_path, "w") as f:
        f.write(report)
    
    print(f"✅ Report written to: {output_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: analyze_fullyear_gated.py <run_dir> [output_path]")
        sys.exit(1)
    
    run_dir = Path(sys.argv[1])
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        sys.exit(1)
    
    if len(sys.argv) >= 3:
        output_path = Path(sys.argv[2])
    else:
        output_path = run_dir.parent / f"FULLYEAR_GATED_ANALYSIS_{run_dir.name}.md"
    
    generate_report(run_dir, output_path)

if __name__ == "__main__":
    main()
