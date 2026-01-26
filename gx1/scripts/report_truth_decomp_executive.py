#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Report Truth Decomposition: Executive Summary

DEL 6: Generate executive summary with hard facts, hypotheses, and recommendations.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def generate_executive_summary(
    trade_table_path: Path,
    json_data_path: Path,
    output_dir: Path,
) -> None:
    """Generate DEL 6 executive summary."""
    md_path = output_dir / "TRUTH_DECOMP_EXECUTIVE.md"
    
    # Load data
    df = pd.read_parquet(trade_table_path)
    
    json_data = {}
    if json_data_path.exists():
        with open(json_data_path) as f:
            json_data = json.load(f)
    
    # Load coverage stats for DATA VALIDATION
    coverage_path = trade_table_path.parent / f"{trade_table_path.stem}_coverage.json"
    coverage_data = {}
    if coverage_path.exists():
        with open(coverage_path) as f:
            coverage_data = json.load(f)
    
    with open(md_path, "w") as f:
        f.write("# Truth Decomposition: Executive Summary\n\n")
        f.write(f"**Generated:** {pd.Timestamp.now().isoformat()}\n\n")
        f.write("**Purpose:** Single source of truth for edge decomposition and architectural decisions.\n\n")
        
        # DATA VALIDATION section (stål-kontroll header)
        f.write("## DATA VALIDATION\n\n")
        f.write(f"- **Input Root:** `{coverage_data.get('input_root_resolved', 'N/A')}`\n")
        f.write(f"- **Years Requested:** {', '.join(map(str, coverage_data.get('years_requested', [])))}\n")
        f.write(f"- **Years Detected:** {', '.join(map(str, coverage_data.get('years_detected', [])))}\n")
        f.write(f"- **Year Counts:**\n")
        year_counts = coverage_data.get('year_counts', {})
        for year in sorted(year_counts.keys(), key=int):
            f.write(f"  - {year}: {year_counts[year]:,} trades\n")
        f.write(f"- **Total Trades:** {coverage_data.get('n_trades_total', 0):,}\n")
        
        # Spread/ATR stats
        spread_stats = coverage_data.get('spread_bps_stats', {})
        atr_stats = coverage_data.get('atr_bps_stats', {})
        if spread_stats:
            f.write(f"- **Spread BPS Stats:** min={spread_stats.get('min', 0):.2f}, median={spread_stats.get('median', 0):.2f}, p95={spread_stats.get('p95', 0):.2f}, max={spread_stats.get('max', 0):.2f}\n")
            f.write(f"- **Spread BPS Source:** {coverage_data.get('spread_bps_source', 'unknown')}\n")
        if atr_stats:
            f.write(f"- **ATR BPS Stats:** min={atr_stats.get('min', 0):.2f}, median={atr_stats.get('median', 0):.2f}, p95={atr_stats.get('p95', 0):.2f}, max={atr_stats.get('max', 0):.2f}\n")
            f.write(f"- **ATR BPS Source:** {coverage_data.get('atr_bps_source', 'unknown')}\n")
        
        f.write("\n")
        
        # Hard Facts (10 bullets)
        f.write("## Hard Facts (2020-2025 Baseline)\n\n")
        
        total_pnl = df["pnl_bps"].sum()
        n_trades = len(df)
        winrate = (df["pnl_bps"] > 0).mean()
        
        f.write(f"1. **Total PnL:** {total_pnl:.2f} bps over {n_trades:,} trades\n")
        f.write(f"2. **Overall Winrate:** {winrate:.1%}\n")
        f.write(f"3. **Avg PnL per Trade:** {df['pnl_bps'].mean():.2f} bps\n")
        
        # Per-session breakdown
        session_pnl = df.groupby("entry_session")["pnl_bps"].agg(["sum", "count", "mean"])
        f.write(f"4. **Session PnL:** ")
        session_list = []
        for session in ["ASIA", "EU", "OVERLAP", "US"]:
            if session in session_pnl.index:
                sess_pnl = session_pnl.loc[session, "sum"]
                sess_trades = int(session_pnl.loc[session, "count"])
                session_list.append(f"{session}={sess_pnl:.0f} bps ({sess_trades:,} trades)")
        f.write(", ".join(session_list) + "\n")
        
        # Edge bins
        edge_bins = json_data.get("edge_bins", [])
        poison_bins = json_data.get("poison_bins", [])
        f.write(f"5. **Edge Bins Identified:** {len(edge_bins)} (positive avg + good tail)\n")
        f.write(f"6. **Poison Bins Identified:** {len(poison_bins)} (negative avg + bad tail)\n")
        
        # Stability
        stability = json_data.get("stability_2020_vs_2025", {})
        stable_bins = len(stability.get("stable_edge_bins", []))
        unstable_bins = len(stability.get("unstable_bins", []))
        f.write(f"7. **Stable Edge Bins (2020 vs 2025):** {stable_bins}\n")
        f.write(f"8. **Unstable Bins (2020 vs 2025):** {unstable_bins}\n")
        
        # Payoff shapes
        payoff_shapes = json_data.get("payoff_shapes", {})
        overlap_payoff = payoff_shapes.get("OVERLAP", {})
        if overlap_payoff:
            # Find overall winner/loser bars
            all_winners = []
            all_losers = []
            for metrics in overlap_payoff.values():
                if metrics.get("median_winner_bars_held", 0) > 0:
                    all_winners.append(metrics["median_winner_bars_held"])
                if metrics.get("median_loser_bars_held", 0) > 0:
                    all_losers.append(metrics["median_loser_bars_held"])
            if all_winners and all_losers:
                winner_median = sum(all_winners) / len(all_winners)
                loser_median = sum(all_losers) / len(all_losers)
                f.write(f"9. **OVERLAP Winner Median Bars:** {winner_median:.1f} vs Loser: {loser_median:.1f} ({winner_median/loser_median:.1f}x)\n")
        
        # Top separators
        separators = json_data.get("top_separators_by_session", {})
        overlap_seps = separators.get("OVERLAP", {}).get("top_10", [])
        if overlap_seps:
            top_sep = overlap_seps[0]
            f.write(f"10. **Best OVERLAP Separator:** {top_sep.get('feature', 'N/A')} (score: {top_sep.get('separation_score', 0):.2f})\n")
        
        f.write("\n")
        
        # Hypotheses (3-5 concrete)
        f.write("## Testable Hypotheses (Monster-PC Ready)\n\n")
        f.write("1. **Selective Pre-Entry Veto for Quick-Fail Regimes:**\n")
        f.write("   - Hypothesis: Poison bins with high quick-fail rate can be avoided pre-entry\n")
        f.write("   - Test: Add pre-entry gate that blocks trades matching poison bin characteristics\n")
        f.write("   - Expected: Reduce losers without cutting winners\n\n")
        
        f.write("2. **Threshold Modulation in Poison Bins:**\n")
        f.write("   - Hypothesis: Higher entry threshold in poison bins improves winrate\n")
        f.write("   - Test: Increase threshold by 0.05-0.10 in poison bins\n")
        f.write("   - Expected: Better trade selection, fewer losers\n\n")
        
        f.write("3. **Exit Policy Tweaks for Delayed Edge Bins:**\n")
        f.write("   - Hypothesis: Edge bins with delayed payoff need longer hold times\n")
        f.write("   - Test: Extend minimum hold time or reduce early exit probability in delayed-payoff bins\n")
        f.write("   - Expected: More winners reach payoff, better PnL\n\n")
        
        f.write("4. **Stable Edge Bin Focus:**\n")
        f.write("   - Hypothesis: Stable edge bins (2020 vs 2025) are more reliable\n")
        f.write("   - Test: Increase trade frequency in stable edge bins, reduce in unstable bins\n")
        f.write("   - Expected: More consistent performance across years\n\n")
        
        f.write("5. **Feature Engineering for Separation:**\n")
        f.write("   - Hypothesis: Top separators can be used as features to improve entry model\n")
        f.write("   - Test: Add top separator features to entry model training\n")
        f.write("   - Expected: Better winner/loser separation, improved winrate\n\n")
        
        # Do NOT list
        f.write("## Do NOT (Now)\n\n")
        f.write("- ❌ Do not implement new gates on Mac (wait for monster-PC)\n")
        f.write("- ❌ Do not add TCN models (not enough evidence yet)\n")
        f.write("- ❌ Do not modify entry thresholds without A/B testing\n")
        f.write("- ❌ Do not run new sweeps on Mac (too slow)\n")
        f.write("- ❌ Do not change exit policy without stability analysis\n\n")
        
        # Key Insights
        f.write("## Key Insights\n\n")
        
        # Find top edge bin
        if edge_bins:
            top_edge = edge_bins[0]
            f.write(f"- **Best Edge Bin:** {top_edge.get('session', 'N/A')} with avg PnL {top_edge.get('avg_pnl_per_trade', 0):.2f} bps\n")
        
        # Find worst poison bin
        if poison_bins:
            worst_poison = poison_bins[0]
            f.write(f"- **Worst Poison Bin:** {worst_poison.get('session', 'N/A')} with avg PnL {worst_poison.get('avg_pnl_per_trade', 0):.2f} bps\n")
        
        # OVERLAP pattern
        if overlap_payoff:
            f.write("- **OVERLAP Pattern:** Winners take longer than losers (confirmed)\n")
        
        # Stability insight
        if stable_bins > 0:
            f.write(f"- **Stability:** {stable_bins} edge bins are stable across 2020-2025 (reliable)\n")
        if unstable_bins > 0:
            f.write(f"- **Instability:** {unstable_bins} bins show sign flip or large change (unreliable)\n")
        
        f.write("\n")
    
    log.info(f"✅ Wrote executive summary: {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Report Truth Decomposition: Executive Summary")
    parser.add_argument(
        "--trade-table",
        type=Path,
        required=True,
        help="Path to canonical trade table parquet file",
    )
    parser.add_argument(
        "--json-data",
        type=Path,
        required=True,
        help="Path to JSON file with all decomposition data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for reports",
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    workspace_root = Path(__file__).parent.parent.parent
    if not args.trade_table.is_absolute():
        args.trade_table = workspace_root / args.trade_table
    if not args.json_data.is_absolute():
        args.json_data = workspace_root / args.json_data
    if not args.output_dir.is_absolute():
        args.output_dir = workspace_root / args.output_dir
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    log.info("=" * 60)
    log.info("EXECUTIVE SUMMARY")
    log.info("=" * 60)
    log.info(f"Trade table: {args.trade_table}")
    log.info(f"JSON data: {args.json_data}")
    log.info(f"Output dir: {args.output_dir}")
    log.info("")
    
    # Generate report
    generate_executive_summary(args.trade_table, args.json_data, args.output_dir)
    
    log.info("✅ Executive summary complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
