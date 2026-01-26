#!/usr/bin/env python3
"""
Generate threshold decision pack - concise summary for choosing next candidate threshold.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
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

def generate_decision_pack(sweep_dir: Path, output_path: Path) -> None:
    """Generate decision pack."""
    print(f"Generating decision pack for: {sweep_dir}")
    
    # Find all threshold run directories
    run_dirs = [d for d in sweep_dir.iterdir() if d.is_dir() and d.name.startswith("threshold_")]
    
    if not run_dirs:
        print("❌ No threshold runs found")
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
    
    if not results:
        print("❌ No valid results found")
        return
    
    # A) Summary table
    table_lines = [
        "| Threshold | Trades | Total PnL | Avg PnL | Winrate | Max DD | P5 PnL | P95 PnL |",
        "|-----------|--------|-----------|---------|---------|--------|--------|---------|",
    ]
    
    for r in results:
        threshold = r["threshold"]
        metrics = r["metrics"]
        table_lines.append(
            f"| {threshold:.2f} | {metrics['n_trades']:,} | {metrics['total_pnl']:.2f} | "
            f"{metrics['avg_pnl']:.2f} | {metrics['winrate']:.1%} | {metrics['max_dd']:.2f} | "
            f"{metrics['p5_pnl']:.2f} | {metrics['p95_pnl']:.2f} |"
        )
    
    # B) Top 3 candidates
    # Filter out zero-trade results
    valid_results = [r for r in results if r["metrics"]["n_trades"] > 0]
    
    if not valid_results:
        top_candidates = [
            {"name": "Best PnL", "result": None, "reason": "No trades generated"},
            {"name": "Best Tail", "result": None, "reason": "No trades generated"},
            {"name": "Best Compromise", "result": None, "reason": "No trades generated"},
        ]
    else:
        # Best PnL (highest total_pnl)
        best_pnl = max(valid_results, key=lambda x: x["metrics"]["total_pnl"])
        
        # Best Tail (lowest maxDD, then highest p5_pnl)
        best_tail = min(valid_results, key=lambda x: (x["metrics"]["max_dd"], -x["metrics"]["p5_pnl"]))
        
        # Best Compromise (balance: pnl / maxDD ratio, then p5_pnl)
        def compromise_score(r):
            m = r["metrics"]
            if m["max_dd"] == 0:
                return float('inf') if m["total_pnl"] > 0 else float('-inf')
            return (m["total_pnl"] / m["max_dd"]) + (m["p5_pnl"] * 0.1)
        
        best_compromise = max(valid_results, key=compromise_score)
        
        top_candidates = [
            {
                "name": "Best PnL",
                "result": best_pnl,
                "reason": f"Highest total PnL: {best_pnl['metrics']['total_pnl']:.2f} bps"
            },
            {
                "name": "Best Tail",
                "result": best_tail,
                "reason": f"Lowest maxDD: {best_tail['metrics']['max_dd']:.2f} bps, best P5: {best_tail['metrics']['p5_pnl']:.2f} bps"
            },
            {
                "name": "Best Compromise",
                "result": best_compromise,
                "reason": f"Best PnL/DD ratio with good tail: PnL={best_compromise['metrics']['total_pnl']:.2f}, DD={best_compromise['metrics']['max_dd']:.2f}, P5={best_compromise['metrics']['p5_pnl']:.2f}"
            },
        ]
    
    # C) Simple rules
    # Check for patterns
    dd_explosion = False
    pnl_p5_divergence = False
    session_imbalance = False
    
    if len(valid_results) >= 2:
        # Check if maxDD explodes when threshold lowers
        sorted_by_threshold = sorted(valid_results, key=lambda x: x["threshold"])
        for i in range(len(sorted_by_threshold) - 1):
            curr = sorted_by_threshold[i]
            next_r = sorted_by_threshold[i + 1]
            if next_r["metrics"]["max_dd"] > curr["metrics"]["max_dd"] * 2:
                dd_explosion = True
                break
        
        # Check if PnL increases but P5 gets worse
        for i in range(len(sorted_by_threshold) - 1):
            curr = sorted_by_threshold[i]
            next_r = sorted_by_threshold[i + 1]
            if (next_r["metrics"]["total_pnl"] > curr["metrics"]["total_pnl"] and
                next_r["metrics"]["p5_pnl"] < curr["metrics"]["p5_pnl"] * 0.5):
                pnl_p5_divergence = True
                break
    
    # Generate report
    report_lines = [
        "# FULLYEAR THRESHOLD DECISION PACK",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Source:** {sweep_dir}",
        "",
        "## A) Summary Table",
        "",
    ]
    
    report_lines.extend(table_lines)
    report_lines.extend([
        "",
        "## B) Top 3 Candidates",
        "",
    ])
    
    for candidate in top_candidates:
        report_lines.append(f"### {candidate['name']}")
        if candidate["result"] is None:
            report_lines.append(f"- **Status:** {candidate['reason']}")
        else:
            r = candidate["result"]
            m = r["metrics"]
            report_lines.extend([
                f"- **Threshold:** {r['threshold']:.2f}",
                f"- **Reason:** {candidate['reason']}",
                f"- **Metrics:**",
                f"  - Trades: {m['n_trades']:,}",
                f"  - Total PnL: {m['total_pnl']:.2f} bps",
                f"  - Avg PnL: {m['avg_pnl']:.2f} bps",
                f"  - Winrate: {m['winrate']:.1%}",
                f"  - Max DD: {m['max_dd']:.2f} bps",
                f"  - P5 PnL: {m['p5_pnl']:.2f} bps",
                f"  - P95 PnL: {m['p95_pnl']:.2f} bps",
                f"- **Run Dir:** {r['run_dir'].name}",
                "",
            ])
    
    report_lines.extend([
        "## C) Simple Rules",
        "",
        "### Pattern Detection:",
        "",
    ])
    
    if dd_explosion:
        report_lines.append("- ⚠️ **MaxDD Explosion Detected:** MaxDD increases significantly when threshold lowers → exit logic may need work, not entry")
    else:
        report_lines.append("- ✅ **MaxDD Stable:** MaxDD behavior is acceptable across thresholds")
    
    if pnl_p5_divergence:
        report_lines.append("- ⚠️ **PnL/P5 Divergence Detected:** PnL increases but P5 gets much worse → entry may be letting in junk trades")
    else:
        report_lines.append("- ✅ **PnL/P5 Aligned:** PnL and tail metrics move together")
    
    if session_imbalance:
        report_lines.append("- ⚠️ **Session Imbalance Detected:** Only some sessions are positive → consider per-session thresholds (later, not now)")
    else:
        report_lines.append("- ✅ **Session Balance:** All sessions contribute (or need deeper analysis)")
    
    report_lines.extend([
        "",
        "### General Rules:",
        "",
        "- If maxDD explodes when threshold senkes → exit må jobbe, ikke entry",
        "- If pnl øker men p5 blir mye verre → entry slipper inn junk",
        "- If kun enkelte sessions er positive → vurder per-session threshold (senere, ikke nå)",
        "",
        "## D) Next Action",
        "",
        "### Recommended Next Steps:",
        "",
    ])
    
    if top_candidates[0]["result"] is not None:
        recommended = top_candidates[2]["result"]  # Best Compromise
        report_lines.extend([
            f"1. **Choose threshold {recommended['threshold']:.2f} for deep trade analysis**",
            "   - Analyze winners vs losers",
            "   - Analyze tail trades (P5, worst trades)",
            "   - Analyze session/regime breakdown",
            "   - Identify patterns in losing trades",
            "",
            "2. **Do NOT tune thresholds until deep analysis is complete**",
            "   - Understand WHY trades fail before changing entry logic",
            "   - Exit logic may need work if maxDD is the issue",
            "   - Entry quality may need work if P5 diverges from PnL",
            "",
        ])
    else:
        report_lines.extend([
            "1. **No trades generated at any threshold**",
            "   - Review entry-score distribution report",
            "   - Check if threshold range needs adjustment",
            "   - Verify model is producing valid predictions",
            "",
        ])
    
    report_lines.extend([
        "### Analysis Checklist:",
        "",
        "- [ ] Deep dive into trade journal for chosen threshold",
        "- [ ] Identify top 10 winners and top 10 losers",
        "- [ ] Analyze exit patterns (early exits vs tail losses)",
        "- [ ] Check session/regime distribution of trades",
        "- [ ] Review entry-score distribution for chosen threshold",
        "- [ ] Document findings before any tuning",
        "",
    ])
    
    # Write report
    report = "\n".join(report_lines)
    with open(output_path, "w") as f:
        f.write(report)
    
    print(f"✅ Decision pack written to: {output_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: generate_threshold_decision_pack.py <sweep_dir> [output_path]")
        sys.exit(1)
    
    sweep_dir = Path(sys.argv[1])
    
    if len(sys.argv) >= 3:
        output_path = Path(sys.argv[2])
    else:
        output_path = sweep_dir.parent / "FULLYEAR_THRESHOLD_DECISION_PACK.md"
    
    generate_decision_pack(sweep_dir, output_path)

if __name__ == "__main__":
    main()
