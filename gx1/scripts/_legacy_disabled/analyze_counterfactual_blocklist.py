#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Counterfactual Analysis: Estimate Impact of Blocking Trades

Analyzes baseline trades to estimate the impact of blocking specific trade patterns
without running new replays. Used for hypothesis validation before implementation.

Input:
- trades parquet: reports/truth_decomp/trades_baseline_2020_2025.parquet
- blocklist spec (json) with rules for matching trades
- stable edge bins (json) for exclusion logic

Output:
- Counterfactual report (markdown)
- Counterfactual metrics (json)
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


def load_trades(parquet_path: Path) -> pd.DataFrame:
    """Load trades parquet and create bin columns."""
    log.info(f"Loading trades from {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    # Create bin columns
    df['atr_bucket'] = pd.qcut(
        df['atr_bps'].rank(method='first'),
        q=5,
        labels=['Q0-Q20', 'Q20-Q40', 'Q40-Q60', 'Q60-Q80', 'Q80-Q100'],
        duplicates='drop'
    )
    df['spread_bucket'] = pd.qcut(
        df['spread_bps'].rank(method='first'),
        q=[0, 0.5, 0.8, 0.95, 1.0],
        labels=['Q0-Q50', 'Q50-Q80', 'Q80-Q95', 'Q95-Q100'],
        duplicates='drop'
    )
    
    # Create bin key
    df['bin_key'] = (
        df['entry_session'].astype(str) + '|' +
        'ATR:' + df['atr_bucket'].astype(str) + '|' +
        'SPREAD:' + df['spread_bucket'].astype(str) + '|' +
        'TREND:' + df['trend_regime'].astype(str) + '|' +
        'VOL:' + df['vol_regime'].astype(str)
    )
    
    log.info(f"Loaded {len(df):,} trades")
    return df


def load_stable_edge_bins(json_path: Path) -> set:
    """Load stable edge bin keys."""
    log.info(f"Loading stable edge bins from {json_path}")
    with open(json_path) as f:
        data = json.load(f)
    stable_bin_keys = {b['bin_key'] for b in data.get('stable_edge_bins', [])}
    log.info(f"Loaded {len(stable_bin_keys)} stable edge bins")
    return stable_bin_keys


def match_trades(df: pd.DataFrame, rule: Dict[str, Any], stable_bin_keys: Optional[set] = None) -> pd.Series:
    """
    Match trades against a blocklist rule.
    
    Rule can specify:
    - session, atr_bucket, spread_bucket, trend_regime, vol_regime
    - bars_held_max (optional)
    - exclude_stable_edge (optional, requires stable_bin_keys)
    """
    mask = pd.Series(True, index=df.index)
    
    if 'session' in rule:
        mask &= (df['entry_session'] == rule['session'])
    
    if 'atr_bucket' in rule:
        mask &= (df['atr_bucket'].astype(str) == rule['atr_bucket'])
    
    if 'spread_bucket' in rule:
        mask &= (df['spread_bucket'].astype(str) == rule['spread_bucket'])
    
    if 'trend_regime' in rule:
        mask &= (df['trend_regime'] == rule['trend_regime'])
    
    if 'vol_regime' in rule:
        mask &= (df['vol_regime'] == rule['vol_regime'])
    
    if 'bars_held_max' in rule:
        mask &= (df['bars_held'] <= rule['bars_held_max'])
    
    if rule.get('exclude_stable_edge', False):
        if stable_bin_keys is None:
            raise ValueError("exclude_stable_edge requires stable_bin_keys")
        mask &= (~df['bin_key'].isin(stable_bin_keys))
    
    return mask


def compute_tail_metrics(series: pd.Series) -> Dict[str, float]:
    """Compute tail metrics (P1, P5, P50, P95, worst 1%, worst 20)."""
    if len(series) == 0:
        return {
            'p1': 0.0,
            'p5': 0.0,
            'p50': 0.0,
            'p95': 0.0,
            'worst_1pct_avg': 0.0,
            'worst_20_sum': 0.0,
        }
    
    sorted_vals = np.sort(series.values)
    n = len(sorted_vals)
    
    p1_idx = max(0, int(n * 0.01))
    p5_idx = max(0, int(n * 0.05))
    p50_idx = max(0, int(n * 0.50))
    p95_idx = min(n - 1, int(n * 0.95))
    
    worst_1pct = sorted_vals[:max(1, n // 100)]
    worst_20 = sorted_vals[:min(20, n)]
    
    return {
        'p1': float(sorted_vals[p1_idx]),
        'p5': float(sorted_vals[p5_idx]),
        'p50': float(sorted_vals[p50_idx]),
        'p95': float(sorted_vals[p95_idx]),
        'worst_1pct_avg': float(np.mean(worst_1pct)),
        'worst_20_sum': float(np.sum(worst_20)),
    }


def analyze_counterfactual(
    df: pd.DataFrame,
    blocklist_spec: Dict[str, Any],
    stable_bin_keys: Optional[set] = None,
) -> Dict[str, Any]:
    """
    Analyze counterfactual impact of blocking trades.
    
    Returns metrics for baseline and counterfactual scenarios.
    """
    log.info(f"Analyzing blocklist: {blocklist_spec.get('name', 'unknown')}")
    
    # Match trades to block
    would_block = pd.Series(False, index=df.index)
    
    for rule in blocklist_spec.get('rules', []):
        rule_mask = match_trades(df, rule, stable_bin_keys)
        would_block |= rule_mask
    
    blocked_trades = df[would_block]
    remaining_trades = df[~would_block]
    
    log.info(f"Would block {len(blocked_trades):,} trades ({len(blocked_trades)/len(df)*100:.2f}%)")
    log.info(f"Remaining: {len(remaining_trades):,} trades")
    
    # ============================================================================
    # SURGICAL CONFIDENCE CHECKS
    # ============================================================================
    trades_total = len(df)
    target_bin_trades = len(blocked_trades)
    
    if trades_total < 10_000:
        log.warning(f"⚠️  WARNING: Only {trades_total:,} trades (< 10,000). Results may be unreliable.")
    
    if target_bin_trades < 200 and target_bin_trades > 0:
        log.warning(f"⚠️  WARNING: Only {target_bin_trades:,} trades in target bin (< 200). Results may be unreliable.")
    elif target_bin_trades == 0:
        log.error(f"❌ FATAL: No trades match blocklist rules. Check blocklist spec.")
        raise ValueError("No trades match blocklist rules")
    
    # Unit sanity (use stats from coverage if available, otherwise from df)
    spread_median = df['spread_bps'].median()
    atr_p95 = df['atr_bps'].quantile(0.95)
    
    if spread_median > 150:
        log.warning(f"⚠️  WARNING: spread_bps median = {spread_median:.2f} (> 150). Units may be incorrect.")
    
    if atr_p95 > 50:
        log.warning(f"⚠️  WARNING: atr_bps p95 = {atr_p95:.2f} (> 50). Units may be incorrect.")
    
    # Baseline metrics
    baseline_metrics = {
        'total_trades': len(df),
        'total_pnl_bps': float(df['pnl_bps'].sum()),
        'avg_pnl_per_trade': float(df['pnl_bps'].mean()),
        'winrate': float((df['pnl_bps'] > 0).mean()),
        'winners': int((df['pnl_bps'] > 0).sum()),
        'losers': int((df['pnl_bps'] <= 0).sum()),
        'tail_metrics': compute_tail_metrics(df['pnl_bps']),
    }
    
    # Counterfactual metrics (without blocked trades)
    counterfactual_metrics = {
        'total_trades': len(remaining_trades),
        'total_pnl_bps': float(remaining_trades['pnl_bps'].sum()),
        'avg_pnl_per_trade': float(remaining_trades['pnl_bps'].mean()),
        'winrate': float((remaining_trades['pnl_bps'] > 0).mean()),
        'winners': int((remaining_trades['pnl_bps'] > 0).sum()),
        'losers': int((remaining_trades['pnl_bps'] <= 0).sum()),
        'tail_metrics': compute_tail_metrics(remaining_trades['pnl_bps']),
    }
    
    # Blocked trades breakdown
    blocked_metrics = {
        'total_trades': len(blocked_trades),
        'total_pnl_bps': float(blocked_trades['pnl_bps'].sum()),
        'avg_pnl_per_trade': float(blocked_trades['pnl_bps'].mean()),
        'winrate': float((blocked_trades['pnl_bps'] > 0).mean()),
        'winners': int((blocked_trades['pnl_bps'] > 0).sum()),
        'losers': int((blocked_trades['pnl_bps'] <= 0).sum()),
        'winners_pnl_bps': float(blocked_trades[blocked_trades['pnl_bps'] > 0]['pnl_bps'].sum()),
        'losers_pnl_bps': float(blocked_trades[blocked_trades['pnl_bps'] <= 0]['pnl_bps'].sum()),
        'tail_metrics': compute_tail_metrics(blocked_trades['pnl_bps']),
    }
    
    # Delta
    delta_metrics = {
        'total_trades': counterfactual_metrics['total_trades'] - baseline_metrics['total_trades'],
        'total_pnl_bps': counterfactual_metrics['total_pnl_bps'] - baseline_metrics['total_pnl_bps'],
        'avg_pnl_per_trade': counterfactual_metrics['avg_pnl_per_trade'] - baseline_metrics['avg_pnl_per_trade'],
        'winrate': counterfactual_metrics['winrate'] - baseline_metrics['winrate'],
        'tail_metrics': {
            k: counterfactual_metrics['tail_metrics'][k] - baseline_metrics['tail_metrics'][k]
            for k in baseline_metrics['tail_metrics']
        },
    }
    
    # Per-session breakdown
    session_breakdown = {}
    for session in df['entry_session'].unique():
        session_df = df[df['entry_session'] == session]
        session_blocked = blocked_trades[blocked_trades['entry_session'] == session]
        session_remaining = remaining_trades[remaining_trades['entry_session'] == session]
        
        session_breakdown[session] = {
            'baseline': {
                'trades': len(session_df),
                'pnl_bps': float(session_df['pnl_bps'].sum()),
                'avg_pnl': float(session_df['pnl_bps'].mean()),
            },
            'blocked': {
                'trades': len(session_blocked),
                'pnl_bps': float(session_blocked['pnl_bps'].sum()),
                'avg_pnl': float(session_blocked['pnl_bps'].mean()),
            },
            'counterfactual': {
                'trades': len(session_remaining),
                'pnl_bps': float(session_remaining['pnl_bps'].sum()),
                'avg_pnl': float(session_remaining['pnl_bps'].mean()),
            },
        }
    
    # Per-year breakdown (use 'year' column if available, otherwise extract from entry_time)
    year_breakdown = {}
    if 'year' in df.columns:
        year_col = 'year'
    elif 'entry_time' in df.columns:
        df = df.copy()
        df['year'] = pd.to_datetime(df['entry_time']).dt.year
        blocked_trades = blocked_trades.copy()
        blocked_trades['year'] = pd.to_datetime(blocked_trades['entry_time']).dt.year
        remaining_trades = remaining_trades.copy()
        remaining_trades['year'] = pd.to_datetime(remaining_trades['entry_time']).dt.year
        year_col = 'year'
    else:
        year_col = None
    
    if year_col:
        for year in sorted(df[year_col].unique()):
            year_df = df[df[year_col] == year]
            year_blocked = blocked_trades[blocked_trades[year_col] == year]
            year_remaining = remaining_trades[remaining_trades[year_col] == year]
        
        year_breakdown[int(year)] = {
            'baseline': {
                'trades': len(year_df),
                'pnl_bps': float(year_df['pnl_bps'].sum()),
                'avg_pnl': float(year_df['pnl_bps'].mean()),
            },
            'blocked': {
                'trades': len(year_blocked),
                'pnl_bps': float(year_blocked['pnl_bps'].sum()),
                'avg_pnl': float(year_blocked['pnl_bps'].mean()),
            },
            'counterfactual': {
                'trades': len(year_remaining),
                'pnl_bps': float(year_remaining['pnl_bps'].sum()),
                'avg_pnl': float(year_remaining['pnl_bps'].mean()),
            },
        }
    
    # Stable edge bins impact
    stable_edge_impact = {}
    if stable_bin_keys:
        stable_df = df[df['bin_key'].isin(stable_bin_keys)]
        stable_blocked = blocked_trades[blocked_trades['bin_key'].isin(stable_bin_keys)]
        stable_remaining = remaining_trades[remaining_trades['bin_key'].isin(stable_bin_keys)]
        
        stable_edge_impact = {
            'baseline': {
                'trades': len(stable_df),
                'pnl_bps': float(stable_df['pnl_bps'].sum()),
                'avg_pnl': float(stable_df['pnl_bps'].mean()),
            },
            'blocked': {
                'trades': len(stable_blocked),
                'pnl_bps': float(stable_blocked['pnl_bps'].sum()),
                'avg_pnl': float(stable_blocked['pnl_bps'].mean()),
            },
            'counterfactual': {
                'trades': len(stable_remaining),
                'pnl_bps': float(stable_remaining['pnl_bps'].sum()),
                'avg_pnl': float(stable_remaining['pnl_bps'].mean()),
            },
            'delta_pnl_bps': float(stable_remaining['pnl_bps'].sum()) - float(stable_df['pnl_bps'].sum()),
            'delta_pct': (float(stable_remaining['pnl_bps'].sum()) - float(stable_df['pnl_bps'].sum())) / float(stable_df['pnl_bps'].sum()) * 100 if float(stable_df['pnl_bps'].sum()) != 0 else 0.0,
        }
    
    return {
        'blocklist_name': blocklist_spec.get('name', 'unknown'),
        'blocklist_description': blocklist_spec.get('description', ''),
        'baseline': baseline_metrics,
        'counterfactual': counterfactual_metrics,
        'blocked': blocked_metrics,
        'delta': delta_metrics,
        'session_breakdown': session_breakdown,
        'year_breakdown': year_breakdown,
        'stable_edge_impact': stable_edge_impact,
        'confidence_checks': {
            'trades_total': trades_total,
            'target_bin_trades': target_bin_trades,
            'spread_median': float(spread_median),
            'atr_p95': float(atr_p95),
        },
    }


def generate_report(results: Dict[str, Any], out_path: Path) -> None:
    """Generate markdown report from counterfactual analysis results."""
    log.info(f"Generating report: {out_path}")
    
    bl_name = results['blocklist_name']
    bl_desc = results['blocklist_description']
    
    baseline = results['baseline']
    counterfactual = results['counterfactual']
    blocked = results['blocked']
    delta = results['delta']
    stable_edge = results.get('stable_edge_impact', {})
    confidence = results['confidence_checks']
    
    lines = [
        f"# Counterfactual Analysis: {bl_name}",
        "",
        f"**Description:** {bl_desc}",
        f"**Generated:** {pd.Timestamp.now().isoformat()}",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        f"- **Blocked Trades:** {blocked['total_trades']:,} ({blocked['total_trades']/baseline['total_trades']*100:.2f}%)",
        f"- **Δ Total PnL:** {delta['total_pnl_bps']:+.2f} bps ({delta['total_pnl_bps']/baseline['total_pnl_bps']*100:+.2f}%)",
        f"- **Δ Avg PnL/Trade:** {delta['avg_pnl_per_trade']:+.2f} bps",
        f"- **Δ Winrate:** {delta['winrate']:+.2%}",
        "",
        "### Tail Metrics",
        "",
        "| Metric | Baseline | Counterfactual | Δ |",
        "|--------|----------|----------------|---|",
        f"| P1 | {baseline['tail_metrics']['p1']:.2f} | {counterfactual['tail_metrics']['p1']:.2f} | {delta['tail_metrics']['p1']:+.2f} |",
        f"| P5 | {baseline['tail_metrics']['p5']:.2f} | {counterfactual['tail_metrics']['p5']:.2f} | {delta['tail_metrics']['p5']:+.2f} |",
        f"| P50 | {baseline['tail_metrics']['p50']:.2f} | {counterfactual['tail_metrics']['p50']:.2f} | {delta['tail_metrics']['p50']:+.2f} |",
        f"| P95 | {baseline['tail_metrics']['p95']:.2f} | {counterfactual['tail_metrics']['p95']:.2f} | {delta['tail_metrics']['p95']:+.2f} |",
        f"| Worst 1% Avg | {baseline['tail_metrics']['worst_1pct_avg']:.2f} | {counterfactual['tail_metrics']['worst_1pct_avg']:.2f} | {delta['tail_metrics']['worst_1pct_avg']:+.2f} |",
        f"| Worst 20 Sum | {baseline['tail_metrics']['worst_20_sum']:.2f} | {counterfactual['tail_metrics']['worst_20_sum']:.2f} | {delta['tail_metrics']['worst_20_sum']:+.2f} |",
        "",
        "---",
        "",
        "## Blocked Trades Breakdown",
        "",
        f"- **Total Blocked:** {blocked['total_trades']:,}",
        f"- **Winners Blocked:** {blocked['winners']:,} ({blocked['winners']/blocked['total_trades']*100:.1f}%)",
        f"- **Losers Blocked:** {blocked['losers']:,} ({blocked['losers']/blocked['total_trades']*100:.1f}%)",
        f"- **Blocked Winners PnL:** {blocked['winners_pnl_bps']:+.2f} bps",
        f"- **Blocked Losers PnL:** {blocked['losers_pnl_bps']:.2f} bps",
        f"- **Net Blocked PnL:** {blocked['total_pnl_bps']:+.2f} bps",
        "",
        "---",
        "",
        "## Stable Edge Bins Impact",
        "",
    ]
    
    if stable_edge:
        lines.extend([
            f"- **Baseline Stable Edge PnL:** {stable_edge['baseline']['pnl_bps']:+.2f} bps ({stable_edge['baseline']['trades']:,} trades)",
            f"- **Blocked from Stable Edge:** {stable_edge['blocked']['trades']:,} trades ({stable_edge['blocked']['pnl_bps']:+.2f} bps)",
            f"- **Counterfactual Stable Edge PnL:** {stable_edge['counterfactual']['pnl_bps']:+.2f} bps ({stable_edge['counterfactual']['trades']:,} trades)",
            f"- **Δ Stable Edge PnL:** {stable_edge['delta_pnl_bps']:+.2f} bps ({stable_edge['delta_pct']:+.2f}%)",
            "",
        ])
    else:
        lines.append("- No stable edge bins data available", "")
    
    lines.extend([
        "---",
        "",
        "## Per-Session Breakdown",
        "",
        "| Session | Baseline Trades | Baseline PnL | Blocked Trades | Blocked PnL | Counterfactual Trades | Counterfactual PnL |",
        "|---------|-----------------|--------------|----------------|-------------|----------------------|-------------------|",
    ])
    
    for session in sorted(results['session_breakdown'].keys()):
        sess = results['session_breakdown'][session]
        lines.append(
            f"| {session} | {sess['baseline']['trades']:,} | {sess['baseline']['pnl_bps']:+.2f} | "
            f"{sess['blocked']['trades']:,} | {sess['blocked']['pnl_bps']:+.2f} | "
            f"{sess['counterfactual']['trades']:,} | {sess['counterfactual']['pnl_bps']:+.2f} |"
        )
    
    lines.extend([
        "",
        "---",
        "",
        "## Per-Year Breakdown",
        "",
        "| Year | Baseline Trades | Baseline PnL | Blocked Trades | Blocked PnL | Counterfactual Trades | Counterfactual PnL |",
        "|------|-----------------|--------------|----------------|-------------|----------------------|-------------------|",
    ])
    
    for year in sorted(results['year_breakdown'].keys()):
        yr = results['year_breakdown'][year]
        lines.append(
            f"| {year} | {yr['baseline']['trades']:,} | {yr['baseline']['pnl_bps']:+.2f} | "
            f"{yr['blocked']['trades']:,} | {yr['blocked']['pnl_bps']:+.2f} | "
            f"{yr['counterfactual']['trades']:,} | {yr['counterfactual']['pnl_bps']:+.2f} |"
        )
    
    lines.extend([
        "",
        "---",
        "",
        "## Confidence Checks",
        "",
        f"- **Total Trades:** {confidence['trades_total']:,}",
        f"- **Target Bin Trades:** {confidence['target_bin_trades']:,}",
        f"- **Spread Median:** {confidence['spread_median']:.2f} bps",
        f"- **ATR P95:** {confidence['atr_p95']:.2f} bps",
        "",
    ])
    
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    
    log.info(f"✅ Report written: {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze counterfactual impact of blocking trades')
    parser.add_argument('--trades-parquet', type=Path, required=True,
                        help='Path to trades parquet file')
    parser.add_argument('--stable-edge-json', type=Path, required=True,
                        help='Path to stable edge bins JSON')
    parser.add_argument('--blocklist-json', type=Path, required=True,
                        help='Path to blocklist spec JSON')
    parser.add_argument('--out-root', type=Path, required=True,
                        help='Output root directory')
    
    args = parser.parse_args()
    
    # Load data
    df = load_trades(args.trades_parquet)
    stable_bin_keys = load_stable_edge_bins(args.stable_edge_json)
    
    # Load blocklist specs
    with open(args.blocklist_json) as f:
        blocklist_data = json.load(f)
    
    # Analyze each blocklist variant
    all_results = {}
    
    for variant_name, blocklist_spec in blocklist_data.items():
        log.info(f"\n{'='*80}")
        log.info(f"Analyzing variant: {variant_name}")
        log.info(f"{'='*80}\n")
        
        try:
            results = analyze_counterfactual(df, blocklist_spec, stable_bin_keys)
            all_results[variant_name] = results
            
            # Generate report
            report_path = args.out_root / f"HYPOTHESIS_001_COUNTERFACTUAL_{variant_name.upper()}.md"
            generate_report(results, report_path)
            
        except Exception as e:
            log.error(f"❌ Error analyzing {variant_name}: {e}")
            raise
    
    # Save combined JSON
    json_path = args.out_root / "hypothesis_001_counterfactual.json"
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    log.info(f"\n✅ Counterfactual analysis complete")
    log.info(f"   Reports: {args.out_root}")
    log.info(f"   JSON: {json_path}")


if __name__ == '__main__':
    main()
