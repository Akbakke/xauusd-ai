#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Report Truth Decomposition: Payoff Shapes (Delayed Edge vs Instant Fail)

DEL 4: Analyze payoff patterns per session and ATR bucket.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def analyze_payoff_shapes(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze payoff shapes per session and ATR bucket.
    
    Returns nested dict: results[session][atr_bucket] = metrics
    """
    # Create ATR buckets
    if "atr_bps" in df.columns and df["atr_bps"].notna().sum() > 0:
        atr_quantiles = df["atr_bps"].quantile([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        df["atr_bucket"] = pd.cut(
            df["atr_bps"],
            bins=atr_quantiles.values,
            labels=["Q0-Q20", "Q20-Q40", "Q40-Q60", "Q60-Q80", "Q80-Q100"],
            include_lowest=True,
        )
    else:
        df["atr_bucket"] = "ALL"
    
    results = {}
    sessions = ["ASIA", "EU", "OVERLAP", "US"]
    
    for session in sessions:
        df_session = df[df["entry_session"] == session].copy()
        
        if len(df_session) == 0:
            results[session] = {}
            continue
        
        session_results = {}
        
        for atr_bucket in df_session["atr_bucket"].unique():
            df_bucket = df_session[df_session["atr_bucket"] == atr_bucket].copy()
            
            winners = df_bucket[df_bucket["pnl_bps"] > 0]
            losers = df_bucket[df_bucket["pnl_bps"] <= 0]
            
            if len(winners) == 0 and len(losers) == 0:
                continue
            
            # Metrics
            winner_bars_held = winners["bars_held"].values if len(winners) > 0 else np.array([])
            loser_bars_held = losers["bars_held"].values if len(losers) > 0 else np.array([])
            
            # Quick-fail rate: % losers with holding_bars <= 3
            quick_fail_rate = (loser_bars_held <= 3).mean() if len(loser_bars_held) > 0 else 0.0
            
            # Delayed-payoff rate: % winners with holding_bars >= 10
            delayed_payoff_rate = (winner_bars_held >= 10).mean() if len(winner_bars_held) > 0 else 0.0
            
            session_results[str(atr_bucket)] = {
                "n_trades": len(df_bucket),
                "n_winners": len(winners),
                "n_losers": len(losers),
                "median_winner_bars_held": float(np.median(winner_bars_held)) if len(winner_bars_held) > 0 else 0.0,
                "median_loser_bars_held": float(np.median(loser_bars_held)) if len(loser_bars_held) > 0 else 0.0,
                "quick_fail_rate": float(quick_fail_rate),
                "delayed_payoff_rate": float(delayed_payoff_rate),
            }
        
        results[session] = session_results
    
    return results


def generate_report(
    results: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Generate DEL 4 markdown report."""
    md_path = output_dir / "TRUTH_DECOMP_PAYOFF_SHAPES.md"
    
    with open(md_path, "w") as f:
        f.write("# Truth Decomposition: Payoff Shapes (Delayed Edge vs Instant Fail)\n\n")
        f.write(f"**Generated:** {pd.Timestamp.now().isoformat()}\n\n")
        
        for session in ["ASIA", "EU", "OVERLAP", "US"]:
            session_results = results.get(session, {})
            
            if not session_results:
                continue
            
            f.write(f"## {session}\n\n")
            
            f.write("| ATR Bucket | Trades | Winners | Losers | Winner Median Bars | Loser Median Bars | Quick-Fail % | Delayed-Payoff % |\n")
            f.write("|------------|--------|---------|--------|-------------------|------------------|--------------|------------------|\n")
            
            for atr_bucket, metrics in sorted(session_results.items()):
                f.write(
                    f"| {atr_bucket} | {metrics['n_trades']:,} | {metrics['n_winners']:,} | "
                    f"{metrics['n_losers']:,} | {metrics['median_winner_bars_held']:.1f} | "
                    f"{metrics['median_loser_bars_held']:.1f} | {metrics['quick_fail_rate']:.1%} | "
                    f"{metrics['delayed_payoff_rate']:.1%} |\n"
                )
            f.write("\n")
        
        # Special focus on OVERLAP
        f.write("## OVERLAP Pattern Confirmation\n\n")
        overlap_results = results.get("OVERLAP", {})
        if overlap_results:
            f.write("**Hypothesis:** Losers die fast, winners need time\n\n")
            
            all_winners = []
            all_losers = []
            for metrics in overlap_results.values():
                if metrics["n_winners"] > 0:
                    all_winners.append(metrics["median_winner_bars_held"])
                if metrics["n_losers"] > 0:
                    all_losers.append(metrics["median_loser_bars_held"])
            
            if all_winners and all_losers:
                overall_winner_median = np.median(all_winners)
                overall_loser_median = np.median(all_losers)
                
                f.write(f"- **Overall Winner Median Bars:** {overall_winner_median:.1f}\n")
                f.write(f"- **Overall Loser Median Bars:** {overall_loser_median:.1f}\n")
                f.write(f"- **Ratio:** {overall_winner_median / overall_loser_median:.2f}x\n\n")
                
                if overall_winner_median > overall_loser_median * 1.5:
                    f.write("✅ **CONFIRMED:** Winners take significantly longer than losers\n\n")
                else:
                    f.write("⚠️ **PARTIAL:** Pattern exists but not as strong as expected\n\n")
    
    log.info(f"✅ Wrote payoff shapes report: {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Report Truth Decomposition: Payoff Shapes")
    parser.add_argument(
        "--trade-table",
        type=Path,
        required=True,
        help="Path to canonical trade table parquet file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for reports",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Path to append to JSON (optional)",
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    workspace_root = Path(__file__).parent.parent.parent
    if not args.trade_table.is_absolute():
        args.trade_table = workspace_root / args.trade_table
    if not args.output_dir.is_absolute():
        args.output_dir = workspace_root / args.output_dir
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    log.info("=" * 60)
    log.info("PAYOFF SHAPES (DELAYED EDGE vs INSTANT FAIL)")
    log.info("=" * 60)
    log.info(f"Trade table: {args.trade_table}")
    log.info(f"Output dir: {args.output_dir}")
    log.info("")
    
    # Load trade table
    log.info("Loading trade table...")
    df = pd.read_parquet(args.trade_table)
    log.info(f"Loaded {len(df):,} trades")
    
    # Analyze payoff shapes
    log.info("Analyzing payoff shapes...")
    results = analyze_payoff_shapes(df)
    
    # Generate report
    generate_report(results, args.output_dir)
    
    # Append to JSON if requested
    if args.json_output:
        json_path = workspace_root / args.json_output if not args.json_output.is_absolute() else args.json_output
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
        else:
            data = {}
        
        data["payoff_shapes"] = results
        
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        
        log.info(f"✅ Appended to JSON: {json_path}")
    
    log.info("✅ Payoff shapes analysis complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
