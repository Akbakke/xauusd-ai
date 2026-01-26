#!/usr/bin/env python3
"""
Generate threshold analysis report for a single threshold run.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np

def load_perf_json(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Load perf JSON."""
    perf_files = list(run_dir.glob("perf_*.json"))
    if not perf_files:
        return None
    
    try:
        with open(perf_files[0], "r") as f:
            return json.load(f)
    except Exception:
        return None

def load_trade_journals(run_dir: Path) -> List[Dict[str, Any]]:
    """Load all trade journals from run."""
    trades = []
    trade_dirs = list(run_dir.glob("chunk_*/trades/*.json"))
    
    for trade_path in trade_dirs:
        try:
            with open(trade_path, "r") as f:
                trade = json.load(f)
                trades.append(trade)
        except Exception:
            pass
    
    return trades

def load_chunk_footers(run_dir: Path) -> List[Dict[str, Any]]:
    """Load all chunk footers."""
    footers = []
    footer_files = list(run_dir.glob("chunk_*/chunk_footer.json"))
    
    for footer_path in footer_files:
        try:
            with open(footer_path, "r") as f:
                footer = json.load(f)
                footers.append(footer)
        except Exception:
            pass
    
    return footers

def generate_report(run_dir: Path, threshold: float, output_path: Path) -> None:
    """Generate threshold analysis report."""
    print(f"Generating threshold analysis for: {run_dir}")
    
    # Load data
    perf = load_perf_json(run_dir)
    trades = load_trade_journals(run_dir)
    footers = load_chunk_footers(run_dir)
    
    # Extract metrics
    n_trades = len(trades)
    
    if n_trades == 0:
        report = f"""# FULLYEAR THRESHOLD ANALYSIS

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Run ID:** {run_dir.name}
**Threshold:** {threshold:.2f}
**Source:** {run_dir}

## RUN_CTX

- **root:** {run_dir.parent.parent}
- **head:** {perf.get('git_head', 'unknown') if perf else 'unknown'}
- **run-id:** {perf.get('run_id', run_dir.name) if perf else run_dir.name}
- **policy:** {perf.get('policy_path', 'unknown') if perf else 'unknown'}
- **dataset:** {perf.get('data_path', 'unknown') if perf else 'unknown'}

## Results

- **Threshold:** {threshold:.2f}
- **n_trades:** 0
- **Total PnL:** 0.0
- **Avg Trade PnL:** N/A
- **Winrate:** N/A
- **Max DD:** 0.0
- **P5 Trade PnL:** N/A
- **P95 Trade PnL:** N/A

## Notes

No trades generated with threshold {threshold:.2f}.
"""
        with open(output_path, "w") as f:
            f.write(report)
        return
    
    # Calculate trade metrics
    pnl_values = [float(t.get("pnl_bps", 0.0)) for t in trades]
    total_pnl = sum(pnl_values)
    avg_pnl = total_pnl / n_trades if n_trades > 0 else 0.0
    
    wins = [p for p in pnl_values if p > 0]
    winrate = len(wins) / n_trades if n_trades > 0 else 0.0
    
    # Calculate max DD (cumulative)
    cumulative_pnl = []
    running_sum = 0.0
    peak = 0.0
    max_dd = 0.0
    for pnl in pnl_values:
        running_sum += pnl
        if running_sum > peak:
            peak = running_sum
        dd = peak - running_sum
        if dd > max_dd:
            max_dd = dd
        cumulative_pnl.append(running_sum)
    
    p5_pnl = np.percentile(pnl_values, 5) if pnl_values else 0.0
    p95_pnl = np.percentile(pnl_values, 95) if pnl_values else 0.0
    
    # Per session breakdown
    session_pnl = {}
    session_counts = {}
    for trade in trades:
        session = trade.get("session", "UNKNOWN")
        pnl = float(trade.get("pnl_bps", 0.0))
        if session not in session_pnl:
            session_pnl[session] = []
            session_counts[session] = 0
        session_pnl[session].append(pnl)
        session_counts[session] += 1
    
    # Eligibility blocks
    total_eligibility_blocks = sum(f.get("tripwire_eligibility_blocks", 0) for f in footers)
    total_lookup_attempts = sum(f.get("lookup_attempts", 0) for f in footers)
    eligibility_rate = total_eligibility_blocks / total_lookup_attempts if total_lookup_attempts > 0 else 0.0
    
    # Generate report
    report_lines = [
        "# FULLYEAR THRESHOLD ANALYSIS",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Run ID:** {run_dir.name}",
        f"**Threshold:** {threshold:.2f}",
        f"**Source:** {run_dir}",
        "",
        "## RUN_CTX",
        "",
        f"- **root:** {run_dir.parent.parent}",
        f"- **head:** {perf.get('git_head', 'unknown') if perf else 'unknown'}",
        f"- **run-id:** {perf.get('run_id', run_dir.name) if perf else run_dir.name}",
        f"- **policy:** {perf.get('policy_path', 'unknown') if perf else 'unknown'}",
        f"- **dataset:** {perf.get('data_path', 'unknown') if perf else 'unknown'}",
        "",
        "## Results",
        "",
        f"- **Threshold:** {threshold:.2f}",
        f"- **n_trades:** {n_trades:,}",
        f"- **Total PnL:** {total_pnl:.2f} bps",
        f"- **Avg Trade PnL:** {avg_pnl:.2f} bps",
        f"- **Winrate:** {winrate:.1%}",
        f"- **Max DD:** {max_dd:.2f} bps",
        f"- **P5 Trade PnL:** {p5_pnl:.2f} bps",
        f"- **P95 Trade PnL:** {p95_pnl:.2f} bps",
        "",
        "## Per Session Breakdown",
        "",
        "| Session | Trades | Total PnL | Avg PnL | Winrate |",
        "|---------|--------|-----------|---------|--------|",
    ]
    
    for session in sorted(session_pnl.keys()):
        session_trades = session_pnl[session]
        session_total = sum(session_trades)
        session_avg = session_total / len(session_trades) if session_trades else 0.0
        session_wins = len([p for p in session_trades if p > 0])
        session_winrate = session_wins / len(session_trades) if session_trades else 0.0
        report_lines.append(
            f"| {session} | {len(session_trades)} | {session_total:.2f} | {session_avg:.2f} | {session_winrate:.1%} |"
        )
    
    report_lines.extend([
        "",
        "## Eligibility Stats",
        "",
        f"- **Total lookup attempts:** {total_lookup_attempts:,}",
        f"- **Eligibility blocks:** {total_eligibility_blocks:,}",
        f"- **Eligibility block rate:** {eligibility_rate:.1%}",
        "",
    ])
    
    # Write report
    report = "\n".join(report_lines)
    with open(output_path, "w") as f:
        f.write(report)
    
    print(f"✅ Report written to: {output_path}")

def main():
    if len(sys.argv) < 3:
        print("Usage: generate_threshold_analysis.py <run_dir> <threshold> [output_path]")
        sys.exit(1)
    
    run_dir = Path(sys.argv[1])
    threshold = float(sys.argv[2])
    
    if len(sys.argv) >= 4:
        output_path = Path(sys.argv[3])
    else:
        output_path = run_dir.parent / f"FULLYEAR_THRESHOLD_ANALYSIS_{threshold:.2f}.md"
    
    generate_report(run_dir, threshold, output_path)

if __name__ == "__main__":
    main()
