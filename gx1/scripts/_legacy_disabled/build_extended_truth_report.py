#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extended Truth Report Builder - Deep Edge Analysis

Utvider Truth Report med:
- DEL 1: Transition Truth (Session → Session matrix)
- DEL 2: Time-to-Payoff Analyse (OVERLAP winners)
- DEL 3: OVERLAP Loss Attribution (Top 5%)
- DEL 4: Executive Truth Summary
- DEL 5: Future-Ready Design Notes
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd

# Add workspace root to path
import sys
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

from gx1.scripts.build_truth_report import (
    load_trades_from_journal,
    compute_transition_matrix,
    compute_time_to_payoff_overlap,
    compute_overlap_loss_attribution,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def generate_transition_truth_report(
    baseline_df: pd.DataFrame,
    w1_df: pd.DataFrame,
    year: int,
    output_dir: Path,
) -> None:
    """Generate DEL 1: Transition Truth report."""
    log.info(f"Generating transition truth report for {year}")
    
    baseline_matrix = compute_transition_matrix(baseline_df)
    w1_matrix = compute_transition_matrix(w1_df)
    
    sessions = ["ASIA", "EU", "OVERLAP", "US"]
    
    # Write Markdown report
    md_path = output_dir / f"TRANSITION_TRUTH_{year}.md"
    with open(md_path, "w") as f:
        f.write(f"# Transition Truth Report: {year}\n\n")
        f.write(f"**Generated:** {pd.Timestamp.now().isoformat()}\n\n")
        
        # BASELINE matrix
        f.write("## BASELINE Transition Matrix\n\n")
        f.write("| Entry → Exit | Trades | Total PnL (bps) | Avg PnL | Winrate |\n")
        f.write("|--------------|--------|-----------------|---------|----------|\n")
        for entry_sess in sessions:
            for exit_sess in sessions:
                data = baseline_matrix[entry_sess][exit_sess]
                if data["trade_count"] > 0:
                    f.write(
                        f"| {entry_sess} → {exit_sess} | {data['trade_count']:,} | "
                        f"{data['total_pnl']:.2f} | {data['avg_pnl']:.2f} | {data['winrate']:.1%} |\n"
                    )
        f.write("\n")
        
        # W1 matrix
        f.write("## OVERLAP_WINDOW_TIGHT Transition Matrix\n\n")
        f.write("| Entry → Exit | Trades | Total PnL (bps) | Avg PnL | Winrate |\n")
        f.write("|--------------|--------|-----------------|---------|----------|\n")
        for entry_sess in sessions:
            for exit_sess in sessions:
                data = w1_matrix[entry_sess][exit_sess]
                if data["trade_count"] > 0:
                    f.write(
                        f"| {entry_sess} → {exit_sess} | {data['trade_count']:,} | "
                        f"{data['total_pnl']:.2f} | {data['avg_pnl']:.2f} | {data['winrate']:.1%} |\n"
                    )
        f.write("\n")
        
        # Delta matrix
        f.write("## Delta Matrix (W1 - BASELINE)\n\n")
        f.write("| Entry → Exit | Δ Trades | Δ Total PnL (bps) | Δ Avg PnL | Δ Winrate |\n")
        f.write("|--------------|----------|-------------------|-----------|-----------|\n")
        for entry_sess in sessions:
            for exit_sess in sessions:
                baseline_data = baseline_matrix[entry_sess][exit_sess]
                w1_data = w1_matrix[entry_sess][exit_sess]
                f.write(
                    f"| {entry_sess} → {exit_sess} | "
                    f"{w1_data['trade_count'] - baseline_data['trade_count']:+d} | "
                    f"{w1_data['total_pnl'] - baseline_data['total_pnl']:+.2f} | "
                    f"{w1_data['avg_pnl'] - baseline_data['avg_pnl']:+.2f} | "
                    f"{w1_data['winrate'] - baseline_data['winrate']:+.1%} |\n"
                )
        f.write("\n")
    
    # Write JSON
    json_path = output_dir / f"transition_truth_{year}.json"
    with open(json_path, "w") as f:
        json.dump({
            "year": year,
            "baseline_matrix": baseline_matrix,
            "w1_matrix": w1_matrix,
        }, f, indent=2, default=str)
    
    log.info(f"✅ Wrote transition truth report: {md_path}")


def generate_time_to_payoff_report(
    baseline_df: pd.DataFrame,
    w1_df: pd.DataFrame,
    year: int,
    output_dir: Path,
) -> None:
    """Generate DEL 2: Time-to-Payoff report for OVERLAP."""
    log.info(f"Generating time-to-payoff report for {year}")
    
    baseline_payoff = compute_time_to_payoff_overlap(baseline_df)
    w1_payoff = compute_time_to_payoff_overlap(w1_df)
    
    # Write Markdown report
    md_path = output_dir / f"TIME_TO_PAYOFF_OVERLAP_{year}.md"
    with open(md_path, "w") as f:
        f.write(f"# Time-to-Payoff Analysis: OVERLAP Winners {year}\n\n")
        f.write(f"**Generated:** {pd.Timestamp.now().isoformat()}\n\n")
        
        f.write("## BASELINE OVERLAP Winners\n\n")
        f.write(f"- **Total Winners:** {baseline_payoff['n_trades']:,}\n")
        f.write(f"- **Median Bars Held:** {baseline_payoff['median_bars_held']:.0f}\n")
        f.write(f"- **P75 Bars Held:** {baseline_payoff['p75_bars_held']:.0f}\n")
        f.write(f"- **P90 Bars Held:** {baseline_payoff['p90_bars_held']:.0f}\n")
        f.write(f"- **Median Bars to MFE (est):** {baseline_payoff['median_bars_to_mfe_est']:.0f}\n")
        f.write(f"- **P75 Bars to MFE (est):** {baseline_payoff['p75_bars_to_mfe_est']:.0f}\n")
        f.write(f"- **P90 Bars to MFE (est):** {baseline_payoff['p90_bars_to_mfe_est']:.0f}\n\n")
        
        f.write("## OVERLAP_WINDOW_TIGHT OVERLAP Winners\n\n")
        f.write(f"- **Total Winners:** {w1_payoff['n_trades']:,}\n")
        f.write(f"- **Median Bars Held:** {w1_payoff['median_bars_held']:.0f}\n")
        f.write(f"- **P75 Bars Held:** {w1_payoff['p75_bars_held']:.0f}\n")
        f.write(f"- **P90 Bars Held:** {w1_payoff['p90_bars_held']:.0f}\n")
        f.write(f"- **Median Bars to MFE (est):** {w1_payoff['median_bars_to_mfe_est']:.0f}\n")
        f.write(f"- **P75 Bars to MFE (est):** {w1_payoff['p75_bars_to_mfe_est']:.0f}\n")
        f.write(f"- **P90 Bars to MFE (est):** {w1_payoff['p90_bars_to_mfe_est']:.0f}\n\n")
        
        # Comparison
        f.write("## Comparison\n\n")
        delta_winners = w1_payoff['n_trades'] - baseline_payoff['n_trades']
        f.write(f"- **Δ Winners:** {delta_winners:+d} ({delta_winners/baseline_payoff['n_trades']*100:+.1f}%)\n")
        f.write(f"- **Δ Median Bars Held:** {w1_payoff['median_bars_held'] - baseline_payoff['median_bars_held']:+.0f}\n")
        f.write(f"- **Δ Median Bars to MFE:** {w1_payoff['median_bars_to_mfe_est'] - baseline_payoff['median_bars_to_mfe_est']:+.0f}\n\n")
        
        # Conclusion
        f.write("## Conclusion\n\n")
        if delta_winners < 0:
            f.write(f"⚠️ **W1 cuts {abs(delta_winners)} winners before payoff**\n\n")
        if w1_payoff['median_bars_to_mfe_est'] < baseline_payoff['median_bars_to_mfe_est']:
            f.write("⚠️ **W1 winners reach MFE faster** (may indicate early exits)\n\n")
        elif w1_payoff['median_bars_to_mfe_est'] > baseline_payoff['median_bars_to_mfe_est']:
            f.write("✅ **W1 winners take longer to reach MFE** (may indicate better timing)\n\n")
        else:
            f.write("➡️ **Timing appears similar**\n\n")
    
    # Write JSON
    json_path = output_dir / f"time_to_payoff_overlap_{year}.json"
    with open(json_path, "w") as f:
        json.dump({
            "year": year,
            "baseline": baseline_payoff,
            "w1": w1_payoff,
        }, f, indent=2, default=str)
    
    log.info(f"✅ Wrote time-to-payoff report: {md_path}")


def generate_overlap_loss_attribution_report(
    baseline_df: pd.DataFrame,
    w1_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Generate DEL 3: OVERLAP Loss Attribution report."""
    log.info("Generating OVERLAP loss attribution report")
    
    baseline_losses = compute_overlap_loss_attribution(baseline_df, top_pct=5.0)
    w1_losses = compute_overlap_loss_attribution(w1_df, top_pct=5.0)
    
    # Write Markdown report
    md_path = output_dir / "OVERLAP_LOSS_ATTRIBUTION.md"
    with open(md_path, "w") as f:
        f.write("# OVERLAP Loss Attribution (Top 5%)\n\n")
        f.write(f"**Generated:** {pd.Timestamp.now().isoformat()}\n\n")
        
        f.write("## BASELINE OVERLAP Top Losses\n\n")
        f.write(f"- **Total OVERLAP Trades:** {baseline_losses['n_overlap_trades']:,}\n")
        f.write(f"- **Top 5% Losses:** {baseline_losses['n_top_losses']}\n\n")
        
        if baseline_losses['holding_time_stats']:
            f.write("### Holding Time Stats\n\n")
            f.write(f"- **Median:** {baseline_losses['holding_time_stats']['median']:.0f} bars\n")
            f.write(f"- **P75:** {baseline_losses['holding_time_stats']['p75']:.0f} bars\n")
            f.write(f"- **P90:** {baseline_losses['holding_time_stats']['p90']:.0f} bars\n\n")
        
        if baseline_losses['adverse_excursion_stats']:
            f.write("### Adverse Excursion (MAE) Stats\n\n")
            f.write(f"- **Median MAE:** {baseline_losses['adverse_excursion_stats']['median']:.2f} bps\n")
            f.write(f"- **P75 MAE:** {baseline_losses['adverse_excursion_stats']['p75']:.2f} bps\n")
            f.write(f"- **P90 MAE:** {baseline_losses['adverse_excursion_stats']['p90']:.2f} bps\n\n")
        
        f.write("### Exit Session Distribution\n\n")
        f.write("| Exit Session | Count |\n")
        f.write("|--------------|-------|\n")
        for sess, count in sorted(baseline_losses['exit_session_distribution'].items(), key=lambda x: -x[1]):
            f.write(f"| {sess} | {count} |\n")
        f.write("\n")
        
        f.write("## OVERLAP_WINDOW_TIGHT OVERLAP Top Losses\n\n")
        f.write(f"- **Total OVERLAP Trades:** {w1_losses['n_overlap_trades']:,}\n")
        f.write(f"- **Top 5% Losses:** {w1_losses['n_top_losses']}\n\n")
        
        if w1_losses['holding_time_stats']:
            f.write("### Holding Time Stats\n\n")
            f.write(f"- **Median:** {w1_losses['holding_time_stats']['median']:.0f} bars\n")
            f.write(f"- **P75:** {w1_losses['holding_time_stats']['p75']:.0f} bars\n")
            f.write(f"- **P90:** {w1_losses['holding_time_stats']['p90']:.0f} bars\n\n")
        
        if w1_losses['adverse_excursion_stats']:
            f.write("### Adverse Excursion (MAE) Stats\n\n")
            f.write(f"- **Median MAE:** {w1_losses['adverse_excursion_stats']['median']:.2f} bps\n")
            f.write(f"- **P75 MAE:** {w1_losses['adverse_excursion_stats']['p75']:.2f} bps\n")
            f.write(f"- **P90 MAE:** {w1_losses['adverse_excursion_stats']['p90']:.2f} bps\n\n")
        
        f.write("### Exit Session Distribution\n\n")
        f.write("| Exit Session | Count |\n")
        f.write("|--------------|-------|\n")
        for sess, count in sorted(w1_losses['exit_session_distribution'].items(), key=lambda x: -x[1]):
            f.write(f"| {sess} | {count} |\n")
        f.write("\n")
        
        # Analysis
        f.write("## Analysis\n\n")
        delta_trades = w1_losses['n_overlap_trades'] - baseline_losses['n_overlap_trades']
        f.write(f"- **Δ OVERLAP Trades:** {delta_trades:+d}\n")
        f.write(f"- **Δ Top Losses:** {w1_losses['n_top_losses'] - baseline_losses['n_top_losses']:+d}\n\n")
        
        if baseline_losses['holding_time_stats'] and w1_losses['holding_time_stats']:
            median_delta = w1_losses['holding_time_stats']['median'] - baseline_losses['holding_time_stats']['median']
            if median_delta < 0:
                f.write("⚠️ **W1 losses are held shorter** (may indicate fake-breaks or early exits)\n\n")
            elif median_delta > 0:
                f.write("⚠️ **W1 losses are held longer** (may indicate grind-tap or carry-feil)\n\n")
            else:
                f.write("➡️ **Holding time similar**\n\n")
    
    # Write JSON
    json_path = output_dir / "overlap_loss_attribution.json"
    with open(json_path, "w") as f:
        json.dump({
            "baseline": baseline_losses,
            "w1": w1_losses,
        }, f, indent=2, default=str)
    
    log.info(f"✅ Wrote OVERLAP loss attribution report: {md_path}")


def generate_executive_summary(
    all_metrics: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Generate DEL 4: Executive Truth Summary."""
    log.info("Generating executive truth summary")
    
    md_path = output_dir / "GX1_TRUTH_SUMMARY_EXECUTIVE.md"
    with open(md_path, "w") as f:
        f.write("# GX1 Truth Summary - Executive Decision Document\n\n")
        f.write(f"**Generated:** {pd.Timestamp.now().isoformat()}\n\n")
        f.write("**Purpose:** Single source of truth for architectural decisions.\n\n")
        
        # Extract key findings
        baseline_combined = all_metrics.get(("BASELINE", "2024_2025"))
        w1_combined = all_metrics.get(("OVERLAP_WINDOW_TIGHT", "2024_2025"))
        
        if baseline_combined and w1_combined:
            delta_pnl = w1_combined['total_pnl_bps'] - baseline_combined['total_pnl_bps']
            delta_trades = w1_combined['n_trades'] - baseline_combined['n_trades']
            
            f.write("## Hard Facts (2024-2025)\n\n")
            f.write(f"1. **BASELINE Performance:** {baseline_combined['total_pnl_bps']:.2f} bps, {baseline_combined['n_trades']:,} trades\n")
            f.write(f"2. **W1 Performance:** {w1_combined['total_pnl_bps']:.2f} bps, {w1_combined['n_trades']:,} trades\n")
            f.write(f"3. **W1 Delta:** {delta_pnl:+.2f} bps ({delta_pnl/baseline_combined['total_pnl_bps']*100:+.1f}%), {delta_trades:+d} trades\n\n")
            
            # Session breakdown
            baseline_sessions = baseline_combined.get('session_breakdown', {})
            w1_sessions = w1_combined.get('session_breakdown', {})
            
            f.write("## Where Edge Comes From\n\n")
            for session in ["ASIA", "EU", "OVERLAP", "US"]:
                baseline_sess = baseline_sessions.get(session, {})
                w1_sess = w1_sessions.get(session, {})
                f.write(f"- **{session}:** BASELINE={baseline_sess.get('total_pnl_bps', 0):.2f} bps, W1={w1_sess.get('total_pnl_bps', 0):.2f} bps\n")
            f.write("\n")
            
            f.write("## Why W1 Fails\n\n")
            overlap_delta = w1_sessions.get('OVERLAP', {}).get('total_pnl_bps', 0) - baseline_sessions.get('OVERLAP', {}).get('total_pnl_bps', 0)
            f.write(f"1. **OVERLAP is the primary edge source** in BASELINE\n")
            f.write(f"2. **W1 cuts {abs(delta_trades):,} trades** (primarily from OVERLAP)\n")
            f.write(f"3. **OVERLAP PnL drops by {abs(overlap_delta):.2f} bps** in W1\n")
            f.write(f"4. **W1 removes both winners and losers**, but loses more winners than it saves\n\n")
            
            f.write("## Recommendations\n\n")
            f.write("### ❌ Do NOT (Now)\n\n")
            f.write("- Do not implement new gates\n")
            f.write("- Do not add TCN models\n")
            f.write("- Do not modify entry thresholds\n")
            f.write("- Do not run new sweeps on Mac\n\n")
            
            f.write("### ✅ Do (When Monster-PC Ready)\n\n")
            f.write("1. **Deep OVERLAP Analysis:**\n")
            f.write("   - Analyze why OVERLAP winners take time to pay off\n")
            f.write("   - Identify if W1 cuts winners too early\n")
            f.write("   - Consider temporal features (not gates) for OVERLAP\n\n")
            f.write("2. **Transition Analysis:**\n")
            f.write("   - Study EU→OVERLAP and OVERLAP→US transitions\n")
            f.write("   - Understand cross-session carry dynamics\n\n")
            f.write("3. **Feature Engineering:**\n")
            f.write("   - Build features that capture temporal patterns\n")
            f.write("   - Consider multi-timeframe signals\n\n")
            f.write("4. **Model Architecture:**\n")
            f.write("   - Evaluate if TCN is needed (based on time-to-payoff findings)\n")
            f.write("   - Consider ensemble approaches\n\n")
    
    log.info(f"✅ Wrote executive summary: {md_path}")


def generate_future_design_notes(
    all_metrics: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Generate DEL 5: Future-Ready Design Notes (no implementation)."""
    log.info("Generating future design notes")
    
    md_path = output_dir / "FUTURE_DESIGN_NOTES.md"
    with open(md_path, "w") as f:
        f.write("# Future Design Notes - TCN & Temporal Signals\n\n")
        f.write(f"**Generated:** {pd.Timestamp.now().isoformat()}\n\n")
        f.write("**⚠️ DESIGN NOTES ONLY - NO IMPLEMENTATION**\n\n")
        
        # Extract time-to-payoff data
        f.write("## Temporal Signal Analysis\n\n")
        f.write("Based on time-to-payoff analysis:\n\n")
        f.write("### Relevant Time Scales\n\n")
        f.write("- **M5 bars:** Primary timeframe\n")
        f.write("- **Session transitions:** EU→OVERLAP, OVERLAP→US\n")
        f.write("- **Payoff windows:** See TIME_TO_PAYOFF_OVERLAP reports\n\n")
        
        f.write("### Potential TCN Applications\n\n")
        f.write("1. **Side-Channel:**\n")
        f.write("   - Use TCN to predict time-to-payoff\n")
        f.write("   - Filter trades that need longer hold times\n")
        f.write("   - Risk: May add complexity without clear benefit\n\n")
        f.write("2. **Expert:**\n")
        f.write("   - TCN as separate model for temporal patterns\n")
        f.write("   - Combine with entry model via ensemble\n")
        f.write("   - Risk: Requires significant training data\n\n")
        f.write("3. **Veto:**\n")
        f.write("   - Use TCN to veto trades that look like slow losers\n")
        f.write("   - Lower risk, but may cut winners\n")
        f.write("   - Risk: Same problem as W1 gate\n\n")
        f.write("4. **Never:**\n")
        f.write("   - If time-to-payoff is too variable\n")
        f.write("   - If edge comes from other factors\n")
        f.write("   - If complexity cost > benefit\n\n")
        
        f.write("## Decision Framework\n\n")
        f.write("Consider TCN if:\n")
        f.write("- Time-to-payoff shows clear patterns\n")
        f.write("- Temporal features improve prediction\n")
        f.write("- Edge is primarily temporal (not regime-based)\n\n")
        f.write("Avoid TCN if:\n")
        f.write("- Edge comes from session/regime selection\n")
        f.write("- Time-to-payoff is too noisy\n")
        f.write("- Simpler solutions (features, not models) suffice\n\n")
    
    log.info(f"✅ Wrote future design notes: {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Build Extended Truth Report")
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Root directory containing profile/year subdirectories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        # DEL 4A: Use GX1_DATA env vars for default paths
        default=Path(os.getenv("GX1_REPORTS_ROOT", "../GX1_DATA/reports")) / "truth",
        help="Output directory for reports",
    )
    parser.add_argument(
        "--years",
        type=str,
        default="2024,2025",
        help="Comma-separated list of years to analyze",
    )
    
    args = parser.parse_args()
    
    years = [int(y.strip()) for y in args.years.split(",")]
    
    # Load all data
    all_data = {}
    for profile in ["BASELINE", "OVERLAP_WINDOW_TIGHT"]:
        for year in years:
            year_dir = args.input_root / profile / f"YEAR_{year}" / "chunk_0"
            if not year_dir.exists():
                log.warning(f"Directory not found: {year_dir}")
                continue
            
            try:
                df = load_trades_from_journal(year_dir)
                all_data[(profile, year)] = df
                log.info(f"✅ Loaded {len(df)} trades for {profile} {year}")
            except Exception as e:
                log.error(f"Failed to load {profile} {year}: {e}", exc_info=True)
    
    # Generate DEL 1: Transition Truth (per year)
    for year in years:
        baseline_df = all_data.get(("BASELINE", year))
        w1_df = all_data.get(("OVERLAP_WINDOW_TIGHT", year))
        if baseline_df is not None and w1_df is not None:
            generate_transition_truth_report(baseline_df, w1_df, year, args.output_dir)
    
    # Generate DEL 1: Transition Truth (combined)
    if len(years) > 1:
        baseline_combined = pd.concat([all_data.get(("BASELINE", y)) for y in years if all_data.get(("BASELINE", y)) is not None], ignore_index=True)
        w1_combined = pd.concat([all_data.get(("OVERLAP_WINDOW_TIGHT", y)) for y in years if all_data.get(("OVERLAP_WINDOW_TIGHT", y)) is not None], ignore_index=True)
        if len(baseline_combined) > 0 and len(w1_combined) > 0:
            generate_transition_truth_report(baseline_combined, w1_combined, "COMBINED", args.output_dir)
    
    # Generate DEL 2: Time-to-Payoff (per year)
    for year in years:
        baseline_df = all_data.get(("BASELINE", year))
        w1_df = all_data.get(("OVERLAP_WINDOW_TIGHT", year))
        if baseline_df is not None and w1_df is not None:
            generate_time_to_payoff_report(baseline_df, w1_df, year, args.output_dir)
    
    # Generate DEL 3: OVERLAP Loss Attribution (combined)
    if len(years) > 1:
        baseline_combined = pd.concat([all_data.get(("BASELINE", y)) for y in years if all_data.get(("BASELINE", y)) is not None], ignore_index=True)
        w1_combined = pd.concat([all_data.get(("OVERLAP_WINDOW_TIGHT", y)) for y in years if all_data.get(("OVERLAP_WINDOW_TIGHT", y)) is not None], ignore_index=True)
        if len(baseline_combined) > 0 and len(w1_combined) > 0:
            generate_overlap_loss_attribution_report(baseline_combined, w1_combined, args.output_dir)
    
    # Generate DEL 4: Executive Summary (need to load metrics from truth reports)
    # For now, compute inline
    all_metrics = {}
    for (profile, year), df in all_data.items():
        # Compute basic metrics
        all_metrics[(profile, year)] = {
            "total_pnl_bps": float(df["pnl_bps"].sum()),
            "n_trades": len(df),
        }
    
    # Generate DEL 4 & 5
    generate_executive_summary(all_metrics, args.output_dir)
    generate_future_design_notes(all_metrics, args.output_dir)
    
    log.info("✅ Extended truth report generation complete")


if __name__ == "__main__":
    main()
