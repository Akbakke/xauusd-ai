#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SNIPER P4 CANARY Full Analysis Report Generator

Analyzes P4 replay data and generates comprehensive statistics report.

Usage:
    python3 gx1/scripts/analyze_sniper_p4_full_report.py
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


def find_p4_csv() -> Path:
    """Find P4 trade log CSV file."""
    base_dir = Path("runs/replay_shadow/SNIPER_P4_COMBINED")
    
    # Try multiple patterns
    patterns = [
        "**/trade_log*_merged.csv",
        "trade_log*_merged.csv",
        "**/*trade_log*.csv",
    ]
    
    for pattern in patterns:
        matches = list(base_dir.glob(pattern))
        if matches:
            csv_path = matches[0]
            log.info(f"Found P4 CSV: {csv_path}")
            return csv_path
    
    raise FileNotFoundError(f"Could not find P4 trade log CSV in {base_dir}")


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
            'p05': 0.0,
            'p95': 0.0,
            'sl_rate': 0.0,
            'avg_bars_held': 0.0,
        }
    
    wins = (df['pnl_bps'] > 0).sum()
    losses = (df['pnl_bps'] < 0).sum()
    sl_trades = df['exit_reason'].str.contains('SL|STOP', case=False, na=False).sum()
    
    return {
        'count': len(df),
        'winrate': wins / len(df) if len(df) > 0 else 0.0,
        'mean_pnl': df['pnl_bps'].mean(),
        'median_pnl': df['pnl_bps'].median(),
        'p05': df['pnl_bps'].quantile(0.05),
        'p95': df['pnl_bps'].quantile(0.95),
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
    
    # Group by trend and vol regime
    regime_cols = ['trend_regime', 'vol_regime']
    if not all(col in df.columns for col in regime_cols):
        # Fallback to entry regimes
        regime_cols = ['trend_regime', 'vol_regime_entry']
        if not all(col in df.columns for col in regime_cols):
            regime_cols = ['trend_regime', 'vol_regime']
    
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
    
    # Use session_entry if available, otherwise session
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


def session_timing_performance(df: pd.DataFrame) -> str:
    """Generate session timing performance (Early/Mid/Late)."""
    lines = []
    lines.append("## Session Timing Performance")
    lines.append("")
    lines.append("Analysis: Split entries by time-of-day within each session.")
    lines.append("")
    
    # Parse entry_time to get hour
    if 'entry_time' not in df.columns:
        lines.append("*Note: entry_time column not found, skipping timing analysis*")
        lines.append("")
        return "\n".join(lines)
    
    df = df.copy()
    df['entry_time_parsed'] = pd.to_datetime(df['entry_time'], errors='coerce')
    df['entry_hour'] = df['entry_time_parsed'].dt.hour
    
    # Use session_entry if available
    session_col = 'session_entry' if 'session_entry' in df.columns else 'session'
    
    lines.append("| Session | Timing | Count | Win Rate | Mean PnL | Median PnL | P05 | P95 | SL Rate | Avg Bars |")
    lines.append("|---------|--------|-------|----------|----------|------------|-----|-----|---------|----------|")
    
    for session, session_group in df.groupby(session_col):
        if len(session_group) == 0:
            continue
        
        # Get hour range for this session
        hours = session_group['entry_hour'].dropna()
        if len(hours) == 0:
            continue
        
        hour_min = hours.min()
        hour_max = hours.max()
        hour_range = hour_max - hour_min + 1
        
        # Split into thirds
        third = hour_range / 3
        early_end = hour_min + third
        mid_end = hour_min + 2 * third
        
        def get_timing(hour):
            if pd.isna(hour):
                return "Unknown"
            if hour <= early_end:
                return "Early"
            elif hour <= mid_end:
                return "Mid"
            else:
                return "Late"
        
        session_group = session_group.copy()
        session_group['timing'] = session_group['entry_hour'].apply(get_timing)
        
        for timing, group in session_group.groupby('timing'):
            metrics = calculate_metrics(group)
            lines.append(
                f"| {session or 'N/A'} | {timing} | {metrics['count']} | "
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


def trade_quality_distributions(df: pd.DataFrame) -> str:
    """Generate trade quality distribution summaries."""
    lines = []
    lines.append("## Trade Quality Distributions")
    lines.append("")
    
    # PnL distribution
    pnl = df['pnl_bps'].dropna()
    if len(pnl) > 0:
        lines.append("### PnL Distribution")
        lines.append(f"- Min: {pnl.min():.2f} bps")
        lines.append(f"- P05: {pnl.quantile(0.05):.2f} bps")
        lines.append(f"- P25: {pnl.quantile(0.25):.2f} bps")
        lines.append(f"- Median: {pnl.median():.2f} bps")
        lines.append(f"- P75: {pnl.quantile(0.75):.2f} bps")
        lines.append(f"- P95: {pnl.quantile(0.95):.2f} bps")
        lines.append(f"- Max: {pnl.max():.2f} bps")
        lines.append(f"- Mean: {pnl.mean():.2f} bps")
        lines.append(f"- Std: {pnl.std():.2f} bps")
        lines.append("")
    
    # Bars held distribution
    if 'bars_held' in df.columns:
        bars = df['bars_held'].dropna()
        if len(bars) > 0:
            lines.append("### Holding Time Distribution (bars)")
            lines.append(f"- Min: {bars.min():.0f}")
            lines.append(f"- P25: {bars.quantile(0.25):.0f}")
            lines.append(f"- Median: {bars.median():.0f}")
            lines.append(f"- P75: {bars.quantile(0.75):.0f}")
            lines.append(f"- Max: {bars.max():.0f}")
            lines.append(f"- Mean: {bars.mean():.1f}")
            lines.append("")
    
    # Try to extract MAE/MFE from extra JSON if available
    if 'extra' in df.columns:
        mae_values = []
        mfe_values = []
        for extra_str in df['extra'].dropna():
            try:
                if isinstance(extra_str, str):
                    extra = json.loads(extra_str)
                else:
                    extra = extra_str
                
                # Look for MAE/MFE in various possible locations
                if isinstance(extra, dict):
                    # Check common locations
                    for key in ['max_mae_bps', 'mae_bps', 'max_adverse_excursion']:
                        if key in extra:
                            val = float(extra[key])
                            if not np.isnan(val):
                                mae_values.append(val)
                    
                    for key in ['max_mfe_bps', 'mfe_bps', 'max_favorable_excursion']:
                        if key in extra:
                            val = float(extra[key])
                            if not np.isnan(val):
                                mfe_values.append(val)
            except:
                pass
        
        if mae_values:
            mae_arr = np.array(mae_values)
            lines.append("### Max Adverse Excursion (MAE) Distribution")
            lines.append(f"- Min: {mae_arr.min():.2f} bps")
            lines.append(f"- P25: {mae_arr.quantile(0.25):.2f} bps")
            lines.append(f"- Median: {np.median(mae_arr):.2f} bps")
            lines.append(f"- P75: {np.quantile(mae_arr, 0.75):.2f} bps")
            lines.append(f"- Max: {mae_arr.max():.2f} bps")
            lines.append(f"- Mean: {mae_arr.mean():.2f} bps")
            lines.append("")
        
        if mfe_values:
            mfe_arr = np.array(mfe_values)
            lines.append("### Max Favorable Excursion (MFE) Distribution")
            lines.append(f"- Min: {mfe_arr.min():.2f} bps")
            lines.append(f"- P25: {mfe_arr.quantile(0.25):.2f} bps")
            lines.append(f"- Median: {np.median(mfe_arr):.2f} bps")
            lines.append(f"- P75: {np.quantile(mfe_arr, 0.75):.2f} bps")
            lines.append(f"- Max: {mfe_arr.max():.2f} bps")
            lines.append(f"- Mean: {mfe_arr.mean():.2f} bps")
            lines.append("")
    
    return "\n".join(lines)


def generate_conclusion(df: pd.DataFrame) -> str:
    """Generate conclusion block with key insights."""
    lines = []
    lines.append("## Conclusion")
    lines.append("")
    
    # Best/worst regime
    regime_cols = ['trend_regime', 'vol_regime']
    if not all(col in df.columns for col in regime_cols):
        regime_cols = ['trend_regime', 'vol_regime_entry']
    
    regime_metrics = []
    for (trend, vol), group in df.groupby(regime_cols):
        if len(group) >= 5:  # Minimum sample size
            metrics = calculate_metrics(group)
            regime_metrics.append({
                'trend': trend,
                'vol': vol,
                'count': metrics['count'],
                'winrate': metrics['winrate'],
                'mean_pnl': metrics['mean_pnl'],
            })
    
    if regime_metrics:
        best_regime = max(regime_metrics, key=lambda x: x['mean_pnl'])
        worst_regime = min(regime_metrics, key=lambda x: x['mean_pnl'])
        
        lines.append(f"**Best Performing Regime:** {best_regime['trend']} × {best_regime['vol']}")
        lines.append(f"- {best_regime['count']} trades, {best_regime['winrate']:.1%} win rate, {best_regime['mean_pnl']:.2f} bps mean PnL")
        lines.append("")
        
        lines.append(f"**Worst Performing Regime:** {worst_regime['trend']} × {worst_regime['vol']}")
        lines.append(f"- {worst_regime['count']} trades, {worst_regime['winrate']:.1%} win rate, {worst_regime['mean_pnl']:.2f} bps mean PnL")
        lines.append("")
    
    # Best/worst session
    session_col = 'session_entry' if 'session_entry' in df.columns else 'session'
    session_metrics = []
    for session, group in df.groupby(session_col):
        if len(group) >= 5:
            metrics = calculate_metrics(group)
            session_metrics.append({
                'session': session,
                'count': metrics['count'],
                'winrate': metrics['winrate'],
                'mean_pnl': metrics['mean_pnl'],
            })
    
    if session_metrics:
        best_session = max(session_metrics, key=lambda x: x['mean_pnl'])
        worst_session = min(session_metrics, key=lambda x: x['mean_pnl'])
        
        lines.append(f"**Best Performing Session:** {best_session['session']}")
        lines.append(f"- {best_session['count']} trades, {best_session['winrate']:.1%} win rate, {best_session['mean_pnl']:.2f} bps mean PnL")
        lines.append("")
        
        lines.append(f"**Worst Performing Session:** {worst_session['session']}")
        lines.append(f"- {worst_session['count']} trades, {worst_session['winrate']:.1%} win rate, {worst_session['mean_pnl']:.2f} bps mean PnL")
        lines.append("")
    
    # Tail risk assessment
    pnl = df['pnl_bps'].dropna()
    if len(pnl) > 0:
        p05 = pnl.quantile(0.05)
        p01 = pnl.quantile(0.01)
        
        lines.append(f"**Tail Risk Assessment:**")
        lines.append(f"- P05 drawdown: {p05:.2f} bps")
        lines.append(f"- P01 drawdown: {p01:.2f} bps")
        
        if p05 < -20:
            lines.append("- ⚠️ High tail risk: P05 below -20 bps")
        elif p05 < -10:
            lines.append("- ⚠️ Moderate tail risk: P05 below -10 bps")
        else:
            lines.append("- ✅ Low tail risk: P05 above -10 bps")
        lines.append("")
    
    # Overall performance summary
    overall_metrics = calculate_metrics(df)
    lines.append(f"**Overall Performance:**")
    lines.append(f"- Total trades: {overall_metrics['count']}")
    lines.append(f"- Win rate: {overall_metrics['winrate']:.1%}")
    lines.append(f"- Mean PnL: {overall_metrics['mean_pnl']:.2f} bps")
    lines.append(f"- Median PnL: {overall_metrics['median_pnl']:.2f} bps")
    lines.append(f"- SL rate: {overall_metrics['sl_rate']:.1%}")
    lines.append("")
    
    return "\n".join(lines)


def main():
    """Generate full P4 analysis report."""
    log.info("Starting P4 full analysis report generation")
    
    # Find CSV
    csv_path = find_p4_csv()
    
    # Load data
    df = load_trades(csv_path)
    
    if len(df) == 0:
        log.error("No trades found in CSV")
        return
    
    # Generate report sections
    report_sections = []
    
    report_sections.append("# SNIPER P4 CANARY Full Analysis Report")
    report_sections.append("")
    report_sections.append(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_sections.append(f"**Source:** {csv_path}")
    report_sections.append(f"**Total Trades:** {len(df)}")
    report_sections.append("")
    
    report_sections.append(regime_performance(df))
    report_sections.append(session_breakdown(df))
    report_sections.append(session_timing_performance(df))
    report_sections.append(tail_risk_analysis(df))
    report_sections.append(trade_quality_distributions(df))
    report_sections.append(generate_conclusion(df))
    
    # Write report
    output_path = Path("reports/sniper/SNIPER_P4_FULL_ANALYSIS.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_sections))
    
    log.info(f"Report written to {output_path}")


if __name__ == "__main__":
    main()

