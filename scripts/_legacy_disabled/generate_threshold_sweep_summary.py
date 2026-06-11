#!/usr/bin/env python3
"""
Generate summary report for threshold sweep.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import re

def extract_threshold_from_dir(dir_name: str) -> Optional[float]:
    """Extract threshold from directory name."""
    match = re.search(r'threshold_([0-9.]+)_', dir_name)
    if match:
        return float(match.group(1))
    return None

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
    """Load all trade journals."""
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

def calculate_metrics(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate trade metrics."""
    if not trades:
        return {
            "n_trades": 0,
            "total_pnl": 0.0,
            "avg_pnl": 0.0,
            "winrate": 0.0,
            "max_dd": 0.0,
            "p5_pnl": 0.0,
            "p95_pnl": 0.0,
        }
    
    import numpy as np
    
    pnl_values = [float(t.get("pnl_bps", 0.0)) for t in trades]
    total_pnl = sum(pnl_values)
    avg_pnl = total_pnl / len(pnl_values)
    
    wins = [p for p in pnl_values if p > 0]
    winrate = len(wins) / len(pnl_values) if pnl_values else 0.0
    
    # Calculate max DD
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
    
    p5_pnl = np.percentile(pnl_values, 5) if pnl_values else 0.0
    p95_pnl = np.percentile(pnl_values, 95) if pnl_values else 0.0
    
    return {
        "n_trades": len(trades),
        "total_pnl": total_pnl,
        "avg_pnl": avg_pnl,
        "winrate": winrate,
        "max_dd": max_dd,
        "p5_pnl": p5_pnl,
        "p95_pnl": p95_pnl,
    }

def generate_summary(sweep_dir: Path, output_path: Path) -> None:
    """Generate sweep summary report."""
    print(f"Generating sweep summary for: {sweep_dir}")
    
    # Find all threshold run directories
    run_dirs = [d for d in sweep_dir.iterdir() if d.is_dir() and d.name.startswith("threshold_")]
    
    if not run_dirs:
        print("No threshold runs found")
        return
    
    # Load data for each threshold
    results = []
    for run_dir in sorted(run_dirs):
        threshold = extract_threshold_from_dir(run_dir.name)
        if threshold is None:
            continue
        
        perf = load_perf_json(run_dir)
        trades = load_trade_journals(run_dir)
        metrics = calculate_metrics(trades)
        
        results.append({
            "threshold": threshold,
            "run_dir": run_dir,
            "perf": perf,
            "metrics": metrics,
        })
    
    # Sort by threshold
    results.sort(key=lambda x: x["threshold"])
    
    # Generate report
    report_lines = [
        "# FULLYEAR THRESHOLD SWEEP SUMMARY",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Source:** {sweep_dir}",
        "",
        "## Summary Table",
        "",
        "| Threshold | Trades | Total PnL | Avg PnL | Winrate | Max DD | P5 PnL | P95 PnL | Notes |",
        "|-----------|--------|-----------|---------|---------|--------|--------|---------|-------|",
    ]
    
    for r in results:
        threshold = r["threshold"]
        metrics = r["metrics"]
        
        # Determine notes
        notes = []
        if metrics["n_trades"] == 0:
            notes.append("No trades")
        elif metrics["total_pnl"] < 0:
            notes.append("Negative PnL")
        elif metrics["max_dd"] > abs(metrics["total_pnl"]) * 2:
            notes.append("High DD")
        
        notes_str = "; ".join(notes) if notes else "-"
        
        report_lines.append(
            f"| {threshold:.2f} | {metrics['n_trades']:,} | {metrics['total_pnl']:.2f} | "
            f"{metrics['avg_pnl']:.2f} | {metrics['winrate']:.1%} | {metrics['max_dd']:.2f} | "
            f"{metrics['p5_pnl']:.2f} | {metrics['p95_pnl']:.2f} | {notes_str} |"
        )
    
    report_lines.extend([
        "",
        "## Notes",
        "",
        "- Threshold override applied via GX1_ANALYSIS_MODE=1 and GX1_ENTRY_THRESHOLD_OVERRIDE",
        "- All runs use same exit logic and risk management",
        "- Eligibility blocks unchanged across thresholds",
        "",
    ])
    
    # Write report
    report = "\n".join(report_lines)
    with open(output_path, "w") as f:
        f.write(report)
    
    print(f"✅ Summary written to: {output_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: generate_threshold_sweep_summary.py <sweep_dir> [output_path]")
        sys.exit(1)
    
    sweep_dir = Path(sys.argv[1])
    
    if len(sys.argv) >= 3:
        output_path = Path(sys.argv[2])
    else:
        output_path = sweep_dir.parent / "FULLYEAR_THRESHOLD_SWEEP_SUMMARY.md"
    
    generate_summary(sweep_dir, output_path)

if __name__ == "__main__":
    main()
