#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SNIPER P4.1 Analysis Script

Analyzes P4.1 replay data and compares with baseline and P4.

Usage:
    python3 gx1/scripts/analyze_sniper_p4_1.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def find_csv(pattern: str) -> Optional[Path]:
    """Find CSV file matching pattern."""
    from glob import glob
    matches = glob(pattern, recursive=True)
    if matches:
        return Path(matches[0])
    return None


def load_trades(csv_path: Path) -> pd.DataFrame:
    """Load trades from CSV."""
    log.info(f"Loading trades from {csv_path}")
    df = pd.read_csv(csv_path, on_bad_lines='skip', engine='python')
    
    # Ensure numeric columns
    numeric_cols = ['pnl_bps', 'bars_held', 'entry_p_long', 'atr_bps']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    log.info(f"Loaded {len(df)} trades")
    return df


def calculate_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate standard metrics for a trade subset."""
    if len(df) == 0:
        return {
            'count': 0,
            'winrate': 0.0,
            'mean_pnl': 0.0,
            'median_pnl': 0.0,
            'p01': 0.0,
            'p05': 0.0,
            'p95': 0.0,
            'sl_rate': 0.0,
            'avg_bars_held': 0.0,
        }
    
    wins = (df['pnl_bps'] > 0).sum()
    losses = (df['pnl_bps'] < 0).sum()
    sl_trades = df['exit_reason'].str.contains('SL|STOP', case=False, na=False).sum()
    pnl = df['pnl_bps'].dropna()
    
    return {
        'count': len(df),
        'winrate': wins / len(df) if len(df) > 0 else 0.0,
        'mean_pnl': pnl.mean() if len(pnl) > 0 else 0.0,
        'median_pnl': pnl.median() if len(pnl) > 0 else 0.0,
        'p01': pnl.quantile(0.01) if len(pnl) > 0 else 0.0,
        'p05': pnl.quantile(0.05) if len(pnl) > 0 else 0.0,
        'p95': pnl.quantile(0.95) if len(pnl) > 0 else 0.0,
        'sl_rate': sl_trades / len(df) if len(df) > 0 else 0.0,
        'avg_bars_held': df['bars_held'].mean() if 'bars_held' in df.columns else 0.0,
    }


def regime_performance(df: pd.DataFrame) -> str:
    """Generate regime performance table."""
    lines = []
    lines.append("## Regime Performance (Trend × Vol)")
    lines.append("")
    lines.append("| Trend | Vol | Count | Win Rate | Mean PnL | Median PnL | P05 | P95 | SL Rate | Avg Bars |")
    lines.append("|-------|-----|-------|----------|----------|------------|-----|-----|---------|----------|")
    
    regime_cols = ['trend_regime', 'vol_regime']
    if not all(col in df.columns for col in regime_cols):
        regime_cols = ['trend_regime', 'vol_regime_entry']
    
    for (trend, vol), group in df.groupby(regime_cols):
        metrics = calculate_metrics(group)
        lines.append(
            f"| {trend or 'N/A'} | {vol or 'N/A'} | {metrics['count']} | "
            f"{metrics['winrate']:.1%} | {metrics['mean_pnl']:.2f} | "
            f"{metrics['median_pnl']:.2f} | {metrics['p05']:.2f} | "
            f"{metrics['p95']:.2f} | {metrics['sl_rate']:.1%} | "
            f"{metrics['avg_bars_held']:.1f} |"
        )
    
    lines.append("")
    return "\n".join(lines)


def session_breakdown(df: pd.DataFrame) -> str:
    """Generate session breakdown table."""
    lines = []
    lines.append("## Session Breakdown")
    lines.append("")
    lines.append("| Session | Count | Win Rate | Mean PnL | Median PnL | P05 | P95 | SL Rate | Avg Bars |")
    lines.append("|---------|-------|----------|----------|------------|-----|-----|---------|----------|")
    
    session_col = 'session_entry' if 'session_entry' in df.columns else 'session'
    
    for session, group in df.groupby(session_col):
        metrics = calculate_metrics(group)
        lines.append(
            f"| {session or 'N/A'} | {metrics['count']} | "
            f"{metrics['winrate']:.1%} | {metrics['mean_pnl']:.2f} | "
            f"{metrics['median_pnl']:.2f} | {metrics['p05']:.2f} | "
            f"{metrics['p95']:.2f} | {metrics['sl_rate']:.1%} | "
            f"{metrics['avg_bars_held']:.1f} |"
        )
    
    lines.append("")
    return "\n".join(lines)


def tail_risk_analysis(df: pd.DataFrame) -> str:
    """Generate tail risk analysis."""
    lines = []
    lines.append("## Tail Risk Analysis")
    lines.append("")
    
    pnl = df['pnl_bps'].dropna()
    if len(pnl) == 0:
        lines.append("*No PnL data available*")
        lines.append("")
        return "\n".join(lines)
    
    p01 = pnl.quantile(0.01)
    p05 = pnl.quantile(0.05)
    p95 = pnl.quantile(0.95)
    p99 = pnl.quantile(0.99)
    
    lines.append(f"**Percentiles:**")
    lines.append(f"- P01: {p01:.2f} bps")
    lines.append(f"- P05: {p05:.2f} bps")
    lines.append(f"- P95: {p95:.2f} bps")
    lines.append(f"- P99: {p99:.2f} bps")
    lines.append("")
    
    # Worst 50 trades
    worst_trades = df.nsmallest(50, 'pnl_bps')[['trade_id', 'pnl_bps', 'exit_reason', 'bars_held']]
    lines.append("**Worst 50 Trades:**")
    lines.append("")
    lines.append("| Trade ID | PnL (bps) | Exit Reason | Bars Held |")
    lines.append("|----------|-----------|-------------|-----------|")
    for _, trade in worst_trades.iterrows():
        lines.append(
            f"| {trade['trade_id']} | {trade['pnl_bps']:.2f} | "
            f"{trade.get('exit_reason', 'N/A')} | {trade.get('bars_held', 'N/A')} |"
        )
    lines.append("")
    
    # Histogram summary
    lines.append("**PnL Distribution Summary:**")
    bins = [-np.inf, -50, -20, -10, 0, 10, 20, 50, np.inf]
    labels = ['<-50', '-50 to -20', '-20 to -10', '-10 to 0', '0 to 10', '10 to 20', '20 to 50', '>50']
    df['pnl_bucket'] = pd.cut(pnl, bins=bins, labels=labels)
    bucket_counts = df['pnl_bucket'].value_counts().sort_index()
    for bucket, count in bucket_counts.items():
        pct = count / len(df) * 100
        lines.append(f"- {bucket} bps: {count} trades ({pct:.1f}%)")
    lines.append("")
    
    return "\n".join(lines)


def comparison_table(baseline_metrics: Optional[Dict], p4_metrics: Optional[Dict], p41_metrics: Dict) -> str:
    """Generate comparison table."""
    lines = []
    lines.append("## Comparison: Baseline vs P4 vs P4.1")
    lines.append("")
    lines.append("| Metric | Baseline | P4 | P4.1 | Change vs P4 |")
    lines.append("|--------|----------|----|----|---------------|")
    
    metrics_to_compare = [
        ('Trades', 'count', lambda x: f"{int(x)}"),
        ('Win Rate', 'winrate', lambda x: f"{x:.1%}"),
        ('Mean PnL (bps)', 'mean_pnl', lambda x: f"{x:.2f}"),
        ('Median PnL (bps)', 'median_pnl', lambda x: f"{x:.2f}"),
        ('P05 (bps)', 'p05', lambda x: f"{x:.2f}"),
        ('P95 (bps)', 'p95', lambda x: f"{x:.2f}"),
        ('SL Rate', 'sl_rate', lambda x: f"{x:.1%}"),
        ('Avg Bars Held', 'avg_bars_held', lambda x: f"{x:.1f}"),
    ]
    
    for metric_name, metric_key, formatter in metrics_to_compare:
        baseline_val = baseline_metrics.get(metric_key) if baseline_metrics else None
        p4_val = p4_metrics.get(metric_key) if p4_metrics else None
        p41_val = p41_metrics.get(metric_key)
        
        baseline_str = formatter(baseline_val) if baseline_val is not None else "N/A"
        p4_str = formatter(p4_val) if p4_val is not None else "N/A"
        p41_str = formatter(p41_val)
        
        if p4_val is not None and p41_val is not None:
            change = p41_val - p4_val
            if 'Rate' in metric_name or 'Win' in metric_name:
                change_str = f"{change:+.1%}"
            else:
                change_str = f"{change:+.2f}"
        else:
            change_str = "N/A"
        
        lines.append(f"| {metric_name} | {baseline_str} | {p4_str} | {p41_str} | {change_str} |")
    
    lines.append("")
    return "\n".join(lines)


def generate_recommendation(p4_metrics: Optional[Dict], p41_metrics: Dict) -> str:
    """Generate recommendation based on comparison."""
    lines = []
    lines.append("## Recommendation")
    lines.append("")
    
    if p4_metrics is None:
        lines.append("*P4 metrics not available for comparison*")
        lines.append("")
        return "\n".join(lines)
    
    p41_mean = p41_metrics.get('mean_pnl', 0.0)
    p4_mean = p4_metrics.get('mean_pnl', 0.0)
    p41_p05 = p41_metrics.get('p05', 0.0)
    p4_p05 = p4_metrics.get('p05', 0.0)
    
    # Decision logic
    mean_improved = p41_mean > p4_mean
    tail_improved = p41_p05 > p4_p05
    
    if mean_improved and tail_improved:
        recommendation = "YES – bedre edge og tryggere tail. God kandidat for live."
        lines.append(f"**Vurdering:** {recommendation}")
        lines.append("")
        lines.append("**Rationale:**")
        lines.append(f"- Mean PnL forbedret: {p41_mean:.2f} bps (vs P4: {p4_mean:.2f} bps)")
        lines.append(f"- Tail risk forbedret: P05 = {p41_p05:.2f} bps (vs P4: {p4_p05:.2f} bps)")
        lines.append(f"- Total trades: {p41_metrics.get('count', 0)}")
    elif mean_improved or tail_improved:
        recommendation = "MODERATE – delvis forbedring, vurder videre tuning."
        lines.append(f"**Vurdering:** {recommendation}")
        lines.append("")
        lines.append("**Rationale:**")
        if mean_improved:
            lines.append(f"- Mean PnL forbedret: {p41_mean:.2f} bps (vs P4: {p4_mean:.2f} bps)")
        else:
            lines.append(f"- Mean PnL ikke forbedret: {p41_mean:.2f} bps (vs P4: {p4_mean:.2f} bps)")
        if tail_improved:
            lines.append(f"- Tail risk forbedret: P05 = {p41_p05:.2f} bps (vs P4: {p4_p05:.2f} bps)")
        else:
            lines.append(f"- Tail risk ikke forbedret: P05 = {p41_p05:.2f} bps (vs P4: {p4_p05:.2f} bps)")
    else:
        recommendation = "NO – ikke nok forbedring, trenger tuning."
        lines.append(f"**Vurdering:** {recommendation}")
        lines.append("")
        lines.append("**Rationale:**")
        lines.append(f"- Mean PnL ikke forbedret: {p41_mean:.2f} bps (vs P4: {p4_mean:.2f} bps)")
        lines.append(f"- Tail risk ikke forbedret: P05 = {p41_p05:.2f} bps (vs P4: {p4_p05:.2f} bps)")
    
    lines.append("")
    return "\n".join(lines)


def main():
    """Generate P4.1 analysis report."""
    log.info("Starting P4.1 analysis report generation")
    
    # Find P4.1 CSV
    p41_csv = find_csv("runs/replay_shadow/SNIPER_P4_1/**/trade_log*_merged.csv")
    if not p41_csv:
        log.error("Could not find P4.1 CSV file")
        return
    
    # Load P4.1 data
    p41_df = load_trades(p41_csv)
    if len(p41_df) == 0:
        log.error("No trades found in P4.1 CSV")
        return
    
    # Try to load P4 for comparison
    p4_csv = find_csv("runs/replay_shadow/SNIPER_P4_COMBINED/**/trade_log*_merged.csv")
    p4_df = None
    if p4_csv:
        try:
            p4_df = load_trades(p4_csv)
            log.info(f"Loaded P4 data for comparison: {len(p4_df)} trades")
        except Exception as e:
            log.warning(f"Failed to load P4 data: {e}")
    
    # Try to load baseline for comparison
    baseline_csv = find_csv("runs/replay_shadow/FULLYEAR_2025_RECAP/**/trade_log*_merged.csv")
    baseline_df = None
    if baseline_csv:
        try:
            baseline_df = load_trades(baseline_csv)
            log.info(f"Loaded baseline data for comparison: {len(baseline_df)} trades")
        except Exception as e:
            log.warning(f"Failed to load baseline data: {e}")
    
    # Calculate metrics
    p41_metrics = calculate_metrics(p41_df)
    p4_metrics = calculate_metrics(p4_df) if p4_df is not None else None
    baseline_metrics = calculate_metrics(baseline_df) if baseline_df is not None else None
    
    # Generate report sections
    report_sections = []
    
    report_sections.append("# SNIPER P4.1 CANARY Analysis Report")
    report_sections.append("")
    report_sections.append(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_sections.append(f"**Source:** {p41_csv}")
    report_sections.append(f"**Total Trades:** {len(p41_df)}")
    report_sections.append("")
    
    report_sections.append(regime_performance(p41_df))
    report_sections.append(session_breakdown(p41_df))
    report_sections.append(tail_risk_analysis(p41_df))
    report_sections.append(comparison_table(baseline_metrics, p4_metrics, p41_metrics))
    report_sections.append(generate_recommendation(p4_metrics, p41_metrics))
    
    # Write report
    output_path = Path("reports/sniper/SNIPER_P4_1_ANALYSIS.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_sections))
    
    log.info(f"Report written to {output_path}")
    
    # Print summary to terminal
    print("\n=== P4.1 COMPLETE ===")
    print(f"Trades: {p41_metrics['count']}")
    p4_mean_str = f"{p4_metrics['mean_pnl']:.2f}" if p4_metrics else "N/A"
    print(f"Mean bps: {p41_metrics['mean_pnl']:.2f}  (vs P4={p4_mean_str})")
    print(f"Median bps: {p41_metrics['median_pnl']:.2f}")
    p4_p05_str = f"{p4_metrics['p05']:.2f}" if p4_metrics else "N/A"
    print(f"P05: {p41_metrics['p05']:.2f}       (vs P4={p4_p05_str})")
    
    # Generate recommendation
    if p4_metrics:
        p41_mean = p41_metrics['mean_pnl']
        p4_mean = p4_metrics['mean_pnl']
        p41_p05 = p41_metrics['p05']
        p4_p05 = p4_metrics['p05']
        
        if p41_mean > p4_mean and p41_p05 > p4_p05:
            recommendation = "YES – bedre edge og tryggere tail. God kandidat for live."
        elif p41_mean > p4_mean or p41_p05 > p4_p05:
            recommendation = "MODERATE – delvis forbedring, vurder videre tuning."
        else:
            recommendation = "NO – ikke nok forbedring, trenger tuning."
    else:
        recommendation = "N/A – P4 metrics ikke tilgjengelig for sammenligning"
    
    print(f"Recommended action: {recommendation}")
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()

