#!/usr/bin/env python3
"""
Analyze entry quality metrics from merged trade index JSONL and trade JSONs.

Calculates:
- MAE_bps (max adverse excursion) in first N bars
- MFE_bps (max favorable excursion) in first N bars
- goes_against_us flag: MAE_bps > threshold

Aggregates by:
- session (EU/US/OVERLAP)
- regime buckets (trend/vol)
- p_long bins

Output: markdown summary + JSON
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
pd.set_option('mode.chained_assignment', None)  # Suppress SettingWithCopyWarning

def load_trades_from_jsonl(jsonl_path: Path) -> List[Dict]:
    """Load all trades from merged_trade_index.jsonl."""
    trades = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                trades.append(json.loads(line))
    return trades

def load_trade_json(trade_json_path: Path) -> Optional[Dict]:
    """Load individual trade JSON file."""
    try:
        with open(trade_json_path, 'r') as f:
            return json.load(f)
    except Exception:
        return None

def calculate_mae_mfe_first_n_bars(
    trade_json: Dict,
    n_bars: int = 6,
    entry_time: Optional[str] = None,
) -> Tuple[Optional[float], Optional[float], bool]:
    """
    Calculate MAE and MFE in first N bars after entry.
    
    Returns:
        (mae_bps, mfe_bps, is_first_n_bars) or (None, None, False) if insufficient data
        is_first_n_bars: True if calculated from first N bars, False if using full-trade lifetime
    """
    exit_summary = trade_json.get("exit_summary")
    if not exit_summary:
        return None, None, False
    
    # Try to calculate from exit events (if available)
    exit_events = trade_json.get("exit_events", [])
    if exit_events and entry_time:
        try:
            entry_ts = pd.to_datetime(entry_time, utc=True)
            first_n_events = []
            for event in exit_events:
                event_ts = pd.to_datetime(event.get("timestamp"), utc=True)
                bars_held = event.get("bars_held", 0)
                if bars_held <= n_bars:
                    first_n_events.append(event)
                elif event_ts > entry_ts + pd.Timedelta(minutes=5 * n_bars):  # Approximate: 5 min per bar
                    break
            
            if first_n_events:
                pnl_values = [e.get("pnl_bps") for e in first_n_events if e.get("pnl_bps") is not None]
                if pnl_values:
                    mae_bps = float(min(pnl_values))
                    mfe_bps = float(max(pnl_values))
                    return mae_bps, mfe_bps, True
        except Exception:
            pass  # Fall back to exit_summary
    
    # Fallback: Use exit_summary (full-trade lifetime MAE/MFE)
    mae_bps = exit_summary.get("max_mae_bps")
    mfe_bps = exit_summary.get("max_mfe_bps")
    
    if mae_bps is not None and mfe_bps is not None:
        # NOTE: This is full-trade lifetime, not first N bars
        # Price trace is removed after metrics calculation to prevent bloat
        return float(mae_bps), float(mfe_bps), False
    
    return None, None, False

def bin_p_long(p_long: Optional[float], bin_size: float = 0.02) -> str:
    """Bin p_long value into range string."""
    if p_long is None:
        return "unknown"
    
    # Round down to nearest bin
    bin_start = int(p_long / bin_size) * bin_size
    bin_end = bin_start + bin_size
    return f"{bin_start:.2f}-{bin_end:.2f}"

def analyze_entry_quality(
    run_dir: Path,
    n_bars: int = 6,
    mae_threshold_bps: float = 5.0,
    p_long_bin_size: float = 0.02,
) -> Dict[str, Any]:
    """
    Analyze entry quality metrics.
    
    Args:
        run_dir: Run directory with trade_journal/merged_trade_index.jsonl
        n_bars: Number of bars to analyze after entry
        mae_threshold_bps: MAE threshold for goes_against_us flag
        p_long_bin_size: Bin size for p_long aggregation
    
    Returns:
        Dict with analysis results
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
                    if trade_json:
                        trade_uid = trade_json.get("trade_uid")
                        if trade_uid:
                            trade_json_files[trade_uid] = json_file
    
    print(f"Found {len(trade_json_files)} trade JSON files")
    
    # Analyze each trade
    results = []
    missing_json_count = 0
    
    for trade in trades:
        trade_uid = trade.get("trade_uid")
        if not trade_uid:
            continue
        
        # Load trade JSON
        trade_json_path = trade_json_files.get(trade_uid)
        if not trade_json_path:
            missing_json_count += 1
            continue
        
        trade_json = load_trade_json(trade_json_path)
        if not trade_json:
            continue
        
        # Calculate MAE/MFE
        mae_bps, mfe_bps, is_first_n_bars = calculate_mae_mfe_first_n_bars(
            trade_json,
            n_bars=n_bars,
            entry_time=trade.get("entry_time"),
        )
        
        if mae_bps is None or mfe_bps is None:
            continue
        
        # Fix sign convention: MAE should be positive magnitude (>=0)
        # Current convention: MAE is negative (adverse move), MFE is positive (favorable move)
        # We want: MAE = abs(mae_bps) if mae_bps < 0, else mae_bps
        if mae_bps < 0:
            mae_bps = abs(mae_bps)  # Convert to positive magnitude
        if mfe_bps < 0:
            mfe_bps = abs(mfe_bps)  # Ensure MFE is also positive
        
        # Extract metadata
        entry_snapshot = trade_json.get("entry_snapshot") or {}
        session = trade.get("session") or entry_snapshot.get("session") or "UNKNOWN"
        vol_regime = trade.get("vol_regime") or entry_snapshot.get("vol_regime") or "UNKNOWN"
        trend_regime = trade.get("trend_regime") or entry_snapshot.get("trend_regime") or "UNKNOWN"
        
        entry_score = entry_snapshot.get("entry_score") or {}
        p_long = entry_score.get("p_long")
        
        # Determine goes_against_us (MAE is now positive magnitude)
        goes_against_us = mae_bps > mae_threshold_bps
        
        # Bin p_long
        p_long_bin = bin_p_long(p_long, p_long_bin_size)
        
        results.append({
            "trade_uid": trade_uid,
            "trade_id": trade.get("trade_id"),
            "session": session,
            "vol_regime": vol_regime,
            "trend_regime": trend_regime,
            "p_long": p_long,
            "p_long_bin": p_long_bin,
            "mae_bps": mae_bps,
            "mfe_bps": mfe_bps,
            "is_first_n_bars": is_first_n_bars,  # True if calculated from first N bars, False if full-trade lifetime
            "goes_against_us": goes_against_us,
            "entry_time": trade.get("entry_time"),
            "side": trade.get("side"),
        })
    
    print(f"Analyzed {len(results)} trades ({missing_json_count} missing JSON files)")
    
    # Aggregate results
    by_session = defaultdict(lambda: {"total": 0, "goes_against_us": 0, "mae_sum": 0.0, "mfe_sum": 0.0})
    by_vol_regime = defaultdict(lambda: {"total": 0, "goes_against_us": 0, "mae_sum": 0.0, "mfe_sum": 0.0})
    by_trend_regime = defaultdict(lambda: {"total": 0, "goes_against_us": 0, "mae_sum": 0.0, "mfe_sum": 0.0})
    by_p_long_bin = defaultdict(lambda: {"total": 0, "goes_against_us": 0, "mae_sum": 0.0, "mfe_sum": 0.0})
    
    for r in results:
        # Session
        by_session[r["session"]]["total"] += 1
        by_session[r["session"]]["mae_sum"] += r["mae_bps"]
        by_session[r["session"]]["mfe_sum"] += r["mfe_bps"]
        if r["goes_against_us"]:
            by_session[r["session"]]["goes_against_us"] += 1
        
        # Vol regime
        by_vol_regime[r["vol_regime"]]["total"] += 1
        by_vol_regime[r["vol_regime"]]["mae_sum"] += r["mae_bps"]
        by_vol_regime[r["vol_regime"]]["mfe_sum"] += r["mfe_bps"]
        if r["goes_against_us"]:
            by_vol_regime[r["vol_regime"]]["goes_against_us"] += 1
        
        # Trend regime
        by_trend_regime[r["trend_regime"]]["total"] += 1
        by_trend_regime[r["trend_regime"]]["mae_sum"] += r["mae_bps"]
        by_trend_regime[r["trend_regime"]]["mfe_sum"] += r["mfe_bps"]
        if r["goes_against_us"]:
            by_trend_regime[r["trend_regime"]]["goes_against_us"] += 1
        
        # p_long bin
        by_p_long_bin[r["p_long_bin"]]["total"] += 1
        by_p_long_bin[r["p_long_bin"]]["mae_sum"] += r["mae_bps"]
        by_p_long_bin[r["p_long_bin"]]["mfe_sum"] += r["mfe_bps"]
        if r["goes_against_us"]:
            by_p_long_bin[r["p_long_bin"]]["goes_against_us"] += 1
    
    # Calculate rates and averages
    def calc_metrics(d: Dict) -> Dict:
        total = d["total"]
        if total == 0:
            return {
                "total": 0,
                "goes_against_us_count": 0,
                "goes_against_us_rate": 0.0,
                "avg_mae_bps": 0.0,
                "avg_mfe_bps": 0.0,
            }
        return {
            "total": total,
            "goes_against_us_count": d["goes_against_us"],
            "goes_against_us_rate": d["goes_against_us"] / total,
            "avg_mae_bps": d["mae_sum"] / total,
            "avg_mfe_bps": d["mfe_sum"] / total,
        }
    
    summary = {
        "total_trades": len(results),
        "n_bars_analyzed": n_bars,
        "mae_threshold_bps": mae_threshold_bps,
        "by_session": {k: calc_metrics(v) for k, v in sorted(by_session.items())},
        "by_vol_regime": {k: calc_metrics(v) for k, v in sorted(by_vol_regime.items())},
        "by_trend_regime": {k: calc_metrics(v) for k, v in sorted(by_trend_regime.items())},
        "by_p_long_bin": {k: calc_metrics(v) for k, v in sorted(by_p_long_bin.items())},
        "overall": calc_metrics({
            "total": len(results),
            "goes_against_us": sum(1 for r in results if r["goes_against_us"]),
            "mae_sum": sum(r["mae_bps"] for r in results),
            "mfe_sum": sum(r["mfe_bps"] for r in results),
        }),
    }
    
    return summary

def write_markdown_report(summary: Dict[str, Any], output_path: Path) -> None:
    """Write markdown summary report."""
    with open(output_path, 'w') as f:
        f.write("# Entry Quality Analysis\n\n")
        f.write(f"**Total trades analyzed:** {summary['total_trades']}\n")
        f.write(f"**Bars analyzed:** {summary['n_bars_analyzed']}\n")
        f.write(f"**MAE threshold:** {summary['mae_threshold_bps']} bps\n\n")
        
        f.write("## Overall Metrics\n\n")
        overall = summary['overall']
        f.write(f"- **Total trades:** {overall['total']}\n")
        f.write(f"- **Goes against us:** {overall['goes_against_us_count']} ({overall['goes_against_us_rate']*100:.1f}%)\n")
        f.write(f"- **Avg MAE:** {overall['avg_mae_bps']:.2f} bps\n")
        f.write(f"- **Avg MFE:** {overall['avg_mfe_bps']:.2f} bps\n\n")
        
        f.write("## By Session\n\n")
        f.write("| Session | Total | Goes Against Us | Rate | Avg MAE | Avg MFE |\n")
        f.write("|---------|-------|-----------------|------|---------|----------|\n")
        for session, metrics in summary['by_session'].items():
            f.write(f"| {session} | {metrics['total']} | {metrics['goes_against_us_count']} | {metrics['goes_against_us_rate']*100:.1f}% | {metrics['avg_mae_bps']:.2f} | {metrics['avg_mfe_bps']:.2f} |\n")
        f.write("\n")
        
        f.write("## By Vol Regime\n\n")
        f.write("| Regime | Total | Goes Against Us | Rate | Avg MAE | Avg MFE |\n")
        f.write("|--------|-------|-----------------|------|---------|----------|\n")
        for regime, metrics in summary['by_vol_regime'].items():
            f.write(f"| {regime} | {metrics['total']} | {metrics['goes_against_us_count']} | {metrics['goes_against_us_rate']*100:.1f}% | {metrics['avg_mae_bps']:.2f} | {metrics['avg_mfe_bps']:.2f} |\n")
        f.write("\n")
        
        f.write("## By Trend Regime\n\n")
        f.write("| Regime | Total | Goes Against Us | Rate | Avg MAE | Avg MFE |\n")
        f.write("|--------|-------|-----------------|------|---------|----------|\n")
        for regime, metrics in summary['by_trend_regime'].items():
            f.write(f"| {regime} | {metrics['total']} | {metrics['goes_against_us_count']} | {metrics['goes_against_us_rate']*100:.1f}% | {metrics['avg_mae_bps']:.2f} | {metrics['avg_mfe_bps']:.2f} |\n")
        f.write("\n")
        
        f.write("## By p_long Bin\n\n")
        f.write("| p_long Range | Total | Goes Against Us | Rate | Avg MAE | Avg MFE |\n")
        f.write("|--------------|-------|-----------------|------|---------|----------|\n")
        for bin_range, metrics in summary['by_p_long_bin'].items():
            f.write(f"| {bin_range} | {metrics['total']} | {metrics['goes_against_us_count']} | {metrics['goes_against_us_rate']*100:.1f}% | {metrics['avg_mae_bps']:.2f} | {metrics['avg_mfe_bps']:.2f} |\n")
        f.write("\n")

def main():
    parser = argparse.ArgumentParser(description="Analyze entry quality metrics")
    parser.add_argument("run_dir", type=Path, help="Run directory with trade_journal/")
    parser.add_argument("--n-bars", type=int, default=6, help="Number of bars to analyze (default: 6)")
    parser.add_argument("--mae-threshold", type=float, default=5.0, help="MAE threshold for goes_against_us (default: 5.0 bps)")
    parser.add_argument("--p-long-bin-size", type=float, default=0.02, help="p_long bin size (default: 0.02)")
    parser.add_argument("--output", type=Path, help="Output directory (default: run_dir/entry_quality_analysis)")
    
    args = parser.parse_args()
    
    if not args.run_dir.exists():
        print(f"ERROR: Run directory not found: {args.run_dir}")
        sys.exit(1)
    
    output_dir = args.output or (args.run_dir / "entry_quality_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("ENTRY QUALITY ANALYSIS")
    print("=" * 80)
    print(f"Run directory: {args.run_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Analyze
    summary = analyze_entry_quality(
        args.run_dir,
        n_bars=args.n_bars,
        mae_threshold_bps=args.mae_threshold,
        p_long_bin_size=args.p_long_bin_size,
    )
    
    # Write JSON
    json_path = output_dir / "entry_quality_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Wrote JSON: {json_path}")
    
    # Write Markdown
    md_path = output_dir / "entry_quality_report.md"
    write_markdown_report(summary, md_path)
    print(f"✅ Wrote Markdown: {md_path}")
    
    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()

