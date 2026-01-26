#!/usr/bin/env python3
"""
Threshold sweep for entry quality analysis.

Sweeps p_long threshold from 0.16 to 0.30 and reports:
- Trade count
- goes_against_us rate
- Tail proxy (e.g., 95th percentile MAE)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import numpy as np

def load_trades_from_jsonl(jsonl_path: Path) -> List[Dict]:
    """Load all trades from merged_trade_index.jsonl."""
    trades = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                trades.append(json.loads(line))
    return trades

def load_trade_json(trade_json_path: Path) -> Dict:
    """Load individual trade JSON file."""
    with open(trade_json_path, 'r') as f:
        return json.load(f)

def calculate_mae_mfe_from_exit_summary(trade_json: Dict) -> tuple:
    """Extract MAE/MFE from exit_summary."""
    exit_summary = trade_json.get("exit_summary")
    if not exit_summary:
        return None, None
    
    mae_bps = exit_summary.get("max_mae_bps")
    mfe_bps = exit_summary.get("max_mfe_bps")
    
    if mae_bps is not None and mfe_bps is not None:
        return float(mae_bps), float(mfe_bps)
    return None, None

def threshold_sweep(
    run_dir: Path,
    threshold_min: float = 0.16,
    threshold_max: float = 0.30,
    threshold_step: float = 0.01,
    mae_threshold_bps: float = 5.0,
    tail_percentile: float = 95.0,
) -> Dict[str, Any]:
    """
    Sweep p_long threshold and calculate metrics.
    
    Args:
        run_dir: Run directory with trade_journal/merged_trade_index.jsonl
        threshold_min: Minimum threshold
        threshold_max: Maximum threshold
        threshold_step: Step size
        mae_threshold_bps: MAE threshold for goes_against_us
        tail_percentile: Percentile for tail proxy
    
    Returns:
        Dict with sweep results
    """
    jsonl_path = run_dir / "trade_journal" / "merged_trade_index.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL index not found: {jsonl_path}")
    
    # Load trades from JSONL
    trades = load_trades_from_jsonl(jsonl_path)
    print(f"Loaded {len(trades)} trades from JSONL")
    
    # Find trade JSON files
    chunks_dir = run_dir / "parallel_chunks"
    trade_json_files = {}
    
    if chunks_dir.exists():
        for chunk_dir in chunks_dir.glob("chunk_*"):
            trades_dir = chunk_dir / "trade_journal" / "trades"
            if trades_dir.exists():
                for json_file in trades_dir.glob("*.json"):
                    trade_json = load_trade_json(json_file)
                    trade_uid = trade_json.get("trade_uid")
                    if trade_uid:
                        trade_json_files[trade_uid] = json_file
    
    print(f"Found {len(trade_json_files)} trade JSON files")
    
    # Load all trade data
    all_trades = []
    missing_json_count = 0
    
    for trade in trades:
        trade_uid = trade.get("trade_uid")
        if not trade_uid:
            continue
        
        trade_json_path = trade_json_files.get(trade_uid)
        if not trade_json_path:
            missing_json_count += 1
            continue
        
        trade_json = load_trade_json(trade_json_path)
        if not trade_json:
            continue
        
        # Extract p_long
        entry_snapshot = trade_json.get("entry_snapshot") or {}
        entry_score = entry_snapshot.get("entry_score") or {}
        p_long = entry_score.get("p_long")
        
        if p_long is None:
            continue
        
        # Calculate MAE/MFE
        mae_bps, mfe_bps = calculate_mae_mfe_from_exit_summary(trade_json)
        if mae_bps is None or mfe_bps is None:
            continue
        
        # Fix sign convention: MAE should be positive magnitude (>=0)
        if mae_bps < 0:
            mae_bps = abs(mae_bps)  # Convert to positive magnitude
        if mfe_bps < 0:
            mfe_bps = abs(mfe_bps)  # Ensure MFE is also positive
        
        all_trades.append({
            "trade_uid": trade_uid,
            "p_long": p_long,
            "mae_bps": mae_bps,
            "mfe_bps": mfe_bps,
            "goes_against_us": mae_bps > mae_threshold_bps,
        })
    
    print(f"Analyzed {len(all_trades)} trades ({missing_json_count} missing JSON files)")
    
    # Sweep thresholds
    thresholds = np.arange(threshold_min, threshold_max + threshold_step, threshold_step)
    results = []
    
    for threshold in thresholds:
        # Filter trades above threshold
        filtered = [t for t in all_trades if t["p_long"] >= threshold]
        
        if len(filtered) == 0:
            continue
        
        mae_values = [t["mae_bps"] for t in filtered]
        goes_against_us_count = sum(1 for t in filtered if t["goes_against_us"])
        goes_against_us_rate = goes_against_us_count / len(filtered)
        
        # Tail proxy: percentile of MAE
        tail_proxy = np.percentile(mae_values, tail_percentile) if mae_values else None
        
        results.append({
            "threshold": float(threshold),
            "trade_count": len(filtered),
            "goes_against_us_count": goes_against_us_count,
            "goes_against_us_rate": float(goes_against_us_rate),
            "tail_proxy_p95": float(tail_proxy) if tail_proxy is not None else None,
            "avg_mae_bps": float(np.mean(mae_values)),
            "avg_mfe_bps": float(np.mean([t["mfe_bps"] for t in filtered])),
        })
    
    return {
        "sweep_params": {
            "threshold_min": threshold_min,
            "threshold_max": threshold_max,
            "threshold_step": threshold_step,
            "mae_threshold_bps": mae_threshold_bps,
            "tail_percentile": tail_percentile,
        },
        "total_trades": len(all_trades),
        "results": results,
    }

def write_markdown_report(summary: Dict[str, Any], output_path: Path) -> None:
    """Write markdown summary report."""
    with open(output_path, 'w') as f:
        f.write("# Entry Quality Threshold Sweep\n\n")
        f.write(f"**Total trades analyzed:** {summary['total_trades']}\n")
        f.write(f"**Threshold range:** {summary['sweep_params']['threshold_min']:.2f} - {summary['sweep_params']['threshold_max']:.2f}\n")
        f.write(f"**Step size:** {summary['sweep_params']['threshold_step']:.2f}\n")
        f.write(f"**MAE threshold:** {summary['sweep_params']['mae_threshold_bps']} bps\n")
        f.write(f"**Tail percentile:** {summary['sweep_params']['tail_percentile']}%\n\n")
        
        f.write("## Results\n\n")
        f.write("| Threshold | Trade Count | Goes Against Us | Rate | Avg MAE | Avg MFE | Tail Proxy (P95) |\n")
        f.write("|-----------|-------------|-----------------|------|---------|---------|-------------------|\n")
        
        for r in summary['results']:
            tail_proxy_str = f"{r['tail_proxy_p95']:.2f}" if r['tail_proxy_p95'] is not None else "N/A"
            f.write(
                f"| {r['threshold']:.2f} | {r['trade_count']} | {r['goes_against_us_count']} | "
                f"{r['goes_against_us_rate']*100:.1f}% | {r['avg_mae_bps']:.2f} | {r['avg_mfe_bps']:.2f} | "
                f"{tail_proxy_str} |\n"
            )
        f.write("\n")
        
        # Find optimal threshold (minimize goes_against_us_rate while maintaining trade count)
        if summary['results']:
            # Sort by goes_against_us_rate, then by trade_count (descending)
            sorted_results = sorted(
                summary['results'],
                key=lambda x: (x['goes_against_us_rate'], -x['trade_count'])
            )
            optimal = sorted_results[0]
            
            f.write("## Optimal Threshold (Lowest goes_against_us_rate)\n\n")
            f.write(f"**Threshold:** {optimal['threshold']:.2f}\n")
            f.write(f"**Trade count:** {optimal['trade_count']}\n")
            f.write(f"**Goes against us rate:** {optimal['goes_against_us_rate']*100:.1f}%\n")
            tail_proxy_optimal = f"{optimal['tail_proxy_p95']:.2f}" if optimal['tail_proxy_p95'] is not None else "N/A"
            f.write(f"**Tail proxy (P95):** {tail_proxy_optimal} bps\n")
            f.write(f"**Avg MAE:** {optimal['avg_mae_bps']:.2f} bps\n")
            f.write(f"**Avg MFE:** {optimal['avg_mfe_bps']:.2f} bps\n\n")

def main():
    parser = argparse.ArgumentParser(description="Threshold sweep for entry quality")
    parser.add_argument("run_dir", type=Path, help="Run directory with trade_journal/")
    parser.add_argument("--threshold-min", type=float, default=0.16, help="Minimum threshold (default: 0.16)")
    parser.add_argument("--threshold-max", type=float, default=0.30, help="Maximum threshold (default: 0.30)")
    parser.add_argument("--threshold-step", type=float, default=0.01, help="Step size (default: 0.01)")
    parser.add_argument("--mae-threshold", type=float, default=5.0, help="MAE threshold for goes_against_us (default: 5.0 bps)")
    parser.add_argument("--tail-percentile", type=float, default=95.0, help="Tail percentile (default: 95.0)")
    parser.add_argument("--output", type=Path, help="Output directory (default: run_dir/threshold_sweep)")
    
    args = parser.parse_args()
    
    if not args.run_dir.exists():
        print(f"ERROR: Run directory not found: {args.run_dir}")
        sys.exit(1)
    
    output_dir = args.output or (args.run_dir / "threshold_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("THRESHOLD SWEEP - ENTRY QUALITY")
    print("=" * 80)
    print(f"Run directory: {args.run_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Sweep
    summary = threshold_sweep(
        args.run_dir,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        threshold_step=args.threshold_step,
        mae_threshold_bps=args.mae_threshold,
        tail_percentile=args.tail_percentile,
    )
    
    # Write JSON
    json_path = output_dir / "threshold_sweep_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Wrote JSON: {json_path}")
    
    # Write Markdown
    md_path = output_dir / "threshold_sweep_report.md"
    write_markdown_report(summary, md_path)
    print(f"✅ Wrote Markdown: {md_path}")
    
    print()
    print("=" * 80)
    print("SWEEP COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()

