#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Counterfactual Exit-Delay Analysis: Estimate Impact of Early Exit

Analyzes baseline trades to estimate the impact of early exit for low-MFE trades
without running new replays. Used for hypothesis validation before implementation.

⚠️  DO NOT IMPLEMENT AT RUNTIME BEFORE MONSTER-PC VALIDATION

Input:
- trades parquet: reports/truth_decomp/trades_baseline_2020_2025.parquet
- stable edge bins (json) for exclusion logic
- Parameter grid: N (bars) ∈ {3, 5, 7}, X (MFE threshold) ∈ {0.0, 0.5, 1.0}

Output:
- Counterfactual report (markdown)
- Counterfactual metrics (json)
"""

import argparse
import json
import itertools
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def load_trades(parquet_path: Path) -> pd.DataFrame:
    """Load trades parquet."""
    log.info(f"Loading trades from {parquet_path}")
    df = pd.read_parquet(parquet_path)
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


def create_bin_key(row: pd.Series) -> str:
    """Create bin key from row."""
    # Create bin columns if not exists
    if 'atr_bucket' not in row.index:
        return None
    
    return (
        str(row['entry_session']) + '|' +
        'ATR:' + str(row['atr_bucket']) + '|' +
        'SPREAD:' + str(row['spread_bucket']) + '|' +
        'TREND:' + str(row['trend_regime']) + '|' +
        'VOL:' + str(row['vol_regime'])
    )


def simulate_early_exit(
    df: pd.DataFrame,
    N: int,
    X: float,
    session: str = 'OVERLAP',
) -> pd.DataFrame:
    """
    Simulate early exit for trades matching criteria.
    
    Criteria:
    - entry_session == session
    - bars_held >= N
    - max_mfe_bps <= X
    
    Early exit simulation:
    - exit_price = entry_price + (exit_price - entry_price) * (N / bars_held)
    - exit_time = entry_time + N bars (approximated)
    - bars_held = N (simulated)
    - pnl_bps = recalculated based on exit_price
    """
    df = df.copy()
    
    # Match criteria
    mask = (
        (df['entry_session'] == session) &
        (df['bars_held'] >= N) &
        (df['max_mfe_bps'] <= X)
    )
    
    affected_trades = df[mask].copy()
    
    if len(affected_trades) == 0:
        return df, affected_trades
    
    # Simulate early exit price (linear interpolation)
    # price_at_N = entry_price + (exit_price - entry_price) * (N / bars_held)
    affected_trades['exit_price_early'] = (
        affected_trades['entry_price'] +
        (affected_trades['exit_price'] - affected_trades['entry_price']) *
        (N / affected_trades['bars_held'])
    )
    
    # Recalculate PnL for early exit
    # For long: pnl = (exit_price - entry_price) / entry_price * 10000
    # For short: pnl = (entry_price - exit_price) / entry_price * 10000
    # Assume all trades are long (baseline is long-only)
    affected_trades['pnl_bps_early'] = (
        (affected_trades['exit_price_early'] - affected_trades['entry_price']) /
        affected_trades['entry_price'] * 10000
    )
    
    # Update bars_held
    affected_trades['bars_held_early'] = N
    
    # Update main dataframe
    df.loc[mask, 'exit_price'] = affected_trades['exit_price_early'].values
    df.loc[mask, 'pnl_bps'] = affected_trades['pnl_bps_early'].values
    df.loc[mask, 'bars_held'] = affected_trades['bars_held_early'].values
    
    return df, affected_trades


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


def analyze_counterfactual_exit_delay(
    df: pd.DataFrame,
    N: int,
    X: float,
    session: str,
    stable_bin_keys: Optional[set] = None,
) -> Dict[str, Any]:
    """
    Analyze counterfactual impact of early exit.
    
    Returns metrics for baseline and counterfactual scenarios.
    """
    log.info(f"Analyzing: N={N}, X={X}, session={session}")
    
    # Create bin keys if needed
    if 'bin_key' not in df.columns:
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
        df['bin_key'] = df.apply(create_bin_key, axis=1)
    
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
    
    # Session-specific baseline
    session_df = df[df['entry_session'] == session]
    session_baseline = {
        'total_trades': len(session_df),
        'total_pnl_bps': float(session_df['pnl_bps'].sum()),
        'avg_pnl_per_trade': float(session_df['pnl_bps'].mean()),
        'winrate': float((session_df['pnl_bps'] > 0).mean()),
    }
    
    # Simulate early exit
    counterfactual_df, affected_trades = simulate_early_exit(df, N, X, session)
    
    # ============================================================================
    # SURGICAL CONFIDENCE CHECKS
    # ============================================================================
    trades_total = len(df)
    affected_count = len(affected_trades)
    
    if trades_total < 10_000:
        log.warning(f"⚠️  WARNING: Only {trades_total:,} trades (< 10,000). Results may be unreliable.")
    
    if affected_count < 50:
        log.warning(f"⚠️  WARNING: Only {affected_count:,} affected trades (< 50). Results may be unreliable.")
    elif affected_count == 0:
        log.warning(f"⚠️  WARNING: No trades match criteria (N={N}, X={X}, session={session})")
    
    # Check stable edge bins
    if stable_bin_keys and 'bin_key' in affected_trades.columns:
        stable_affected = affected_trades[affected_trades['bin_key'].isin(stable_bin_keys)]
        stable_count = len(stable_affected)
        if stable_count > 0:
            stable_pct = stable_count / len(affected_trades) * 100
            if stable_pct > 5:
                log.error(f"❌ FATAL: {stable_count} stable edge trades affected ({stable_pct:.1f}%). Kill criteria violated.")
                raise ValueError(f"Stable edge bins blocked > 5%: {stable_pct:.1f}%")
            else:
                log.warning(f"⚠️  WARNING: {stable_count} stable edge trades affected ({stable_pct:.1f}%)")
    
    # Unit sanity
    spread_median = df['spread_bps'].median()
    atr_p95 = df['atr_bps'].quantile(0.95)
    
    if spread_median > 150:
        log.warning(f"⚠️  WARNING: spread_bps median = {spread_median:.2f} (> 150). Units may be incorrect.")
    
    if atr_p95 > 50:
        log.warning(f"⚠️  WARNING: atr_bps p95 = {atr_p95:.2f} (> 50). Units may be incorrect.")
    
    # Counterfactual metrics
    counterfactual_metrics = {
        'total_trades': len(counterfactual_df),
        'total_pnl_bps': float(counterfactual_df['pnl_bps'].sum()),
        'avg_pnl_per_trade': float(counterfactual_df['pnl_bps'].mean()),
        'winrate': float((counterfactual_df['pnl_bps'] > 0).mean()),
        'winners': int((counterfactual_df['pnl_bps'] > 0).sum()),
        'losers': int((counterfactual_df['pnl_bps'] <= 0).sum()),
        'tail_metrics': compute_tail_metrics(counterfactual_df['pnl_bps']),
    }
    
    # Session-specific counterfactual
    session_counterfactual_df = counterfactual_df[counterfactual_df['entry_session'] == session]
    session_counterfactual = {
        'total_trades': len(session_counterfactual_df),
        'total_pnl_bps': float(session_counterfactual_df['pnl_bps'].sum()),
        'avg_pnl_per_trade': float(session_counterfactual_df['pnl_bps'].mean()),
        'winrate': float((session_counterfactual_df['pnl_bps'] > 0).mean()),
    }
    
    # Affected trades breakdown
    affected_metrics = {
        'total_trades': len(affected_trades),
        'total_pnl_bps_baseline': float(affected_trades['pnl_bps'].sum()) if 'pnl_bps' in affected_trades.columns else 0.0,
        'total_pnl_bps_early': float(affected_trades['pnl_bps_early'].sum()) if 'pnl_bps_early' in affected_trades.columns else 0.0,
        'winners_baseline': int((affected_trades['pnl_bps'] > 0).sum()) if 'pnl_bps' in affected_trades.columns else 0,
        'losers_baseline': int((affected_trades['pnl_bps'] <= 0).sum()) if 'pnl_bps' in affected_trades.columns else 0,
        'winners_early': int((affected_trades['pnl_bps_early'] > 0).sum()) if 'pnl_bps_early' in affected_trades.columns else 0,
        'losers_early': int((affected_trades['pnl_bps_early'] <= 0).sum()) if 'pnl_bps_early' in affected_trades.columns else 0,
    }
    
    # Delta
    delta_metrics = {
        'total_trades': counterfactual_metrics['total_trades'] - baseline_metrics['total_trades'],
        'total_pnl_bps': counterfactual_metrics['total_pnl_bps'] - baseline_metrics['total_pnl_bps'],
        'total_pnl_pct': (counterfactual_metrics['total_pnl_bps'] - baseline_metrics['total_pnl_bps']) / baseline_metrics['total_pnl_bps'] * 100 if baseline_metrics['total_pnl_bps'] != 0 else 0.0,
        'avg_pnl_per_trade': counterfactual_metrics['avg_pnl_per_trade'] - baseline_metrics['avg_pnl_per_trade'],
        'winrate': counterfactual_metrics['winrate'] - baseline_metrics['winrate'],
        'tail_metrics': {
            k: counterfactual_metrics['tail_metrics'][k] - baseline_metrics['tail_metrics'][k]
            for k in baseline_metrics['tail_metrics']
        },
    }
    
    # Session delta
    session_delta = {
        'total_pnl_bps': session_counterfactual['total_pnl_bps'] - session_baseline['total_pnl_bps'],
        'total_pnl_pct': (session_counterfactual['total_pnl_bps'] - session_baseline['total_pnl_bps']) / session_baseline['total_pnl_bps'] * 100 if session_baseline['total_pnl_bps'] != 0 else 0.0,
        'avg_pnl_per_trade': session_counterfactual['avg_pnl_per_trade'] - session_baseline['avg_pnl_per_trade'],
    }
    
    # Stable edge bins impact
    stable_edge_impact = {}
    if stable_bin_keys and 'bin_key' in df.columns:
        stable_df = df[df['bin_key'].isin(stable_bin_keys)]
        stable_counterfactual = counterfactual_df[counterfactual_df['bin_key'].isin(stable_bin_keys)]
        stable_affected = affected_trades[affected_trades['bin_key'].isin(stable_bin_keys)] if 'bin_key' in affected_trades.columns else pd.DataFrame()
        
        stable_edge_impact = {
            'baseline': {
                'trades': len(stable_df),
                'pnl_bps': float(stable_df['pnl_bps'].sum()),
                'avg_pnl': float(stable_df['pnl_bps'].mean()),
            },
            'affected': {
                'trades': len(stable_affected),
                'pnl_bps_baseline': float(stable_affected['pnl_bps'].sum()) if len(stable_affected) > 0 and 'pnl_bps' in stable_affected.columns else 0.0,
                'pnl_bps_early': float(stable_affected['pnl_bps_early'].sum()) if len(stable_affected) > 0 and 'pnl_bps_early' in stable_affected.columns else 0.0,
            },
            'counterfactual': {
                'trades': len(stable_counterfactual),
                'pnl_bps': float(stable_counterfactual['pnl_bps'].sum()),
                'avg_pnl': float(stable_counterfactual['pnl_bps'].mean()),
            },
            'delta_pnl_bps': float(stable_counterfactual['pnl_bps'].sum()) - float(stable_df['pnl_bps'].sum()),
            'delta_pct': (float(stable_counterfactual['pnl_bps'].sum()) - float(stable_df['pnl_bps'].sum())) / float(stable_df['pnl_bps'].sum()) * 100 if float(stable_df['pnl_bps'].sum()) != 0 else 0.0,
        }
    
    return {
        'params': {'N': N, 'X': X, 'session': session},
        'baseline': baseline_metrics,
        'counterfactual': counterfactual_metrics,
        'session_baseline': session_baseline,
        'session_counterfactual': session_counterfactual,
        'session_delta': session_delta,
        'affected': affected_metrics,
        'delta': delta_metrics,
        'stable_edge_impact': stable_edge_impact,
        'confidence_checks': {
            'trades_total': trades_total,
            'affected_trades': affected_count,
            'spread_median': float(spread_median),
            'atr_p95': float(atr_p95),
        },
    }


def generate_results_report(all_results: Dict[str, Any], out_path: Path) -> None:
    """Generate markdown report from all counterfactual analyses."""
    log.info(f"Generating report: {out_path}")
    
    # Sort by worst-20 improvement (descending)
    sorted_results = sorted(
        all_results.items(),
        key=lambda x: x[1]['delta']['tail_metrics']['worst_20_sum'],
        reverse=True
    )
    
    lines = [
        "# Hypothesis #2: Early Exit for Low-MFE Trades - Results",
        "",
        f"**Generated:** {pd.Timestamp.now().isoformat()}",
        "",
        "⚠️  **DO NOT IMPLEMENT AT RUNTIME BEFORE MONSTER-PC VALIDATION**",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        f"**Total Variants Tested:** {len(all_results)}",
        "",
        "### Top 5 Best Variants (by Worst-20 Improvement)",
        "",
        "| Rank | N | X | Session | Δ Total PnL | Δ Worst-20 | Δ Worst 1% | Affected Trades |",
        "|------|---|---|---------|--------------|-------------|-------------|-----------------|",
    ]
    
    for rank, (variant_key, result) in enumerate(sorted_results[:5], 1):
        params = result['params']
        delta = result['delta']
        affected = result['affected']
        lines.append(
            f"| {rank} | {params['N']} | {params['X']:.1f} | {params['session']} | "
            f"{delta['total_pnl_bps']:+.2f} ({delta['total_pnl_pct']:+.2f}%) | "
            f"{delta['tail_metrics']['worst_20_sum']:+.2f} | "
            f"{delta['tail_metrics']['worst_1pct_avg']:+.2f} | "
            f"{affected['total_trades']:,} |"
        )
    
    lines.extend([
        "",
        "### Worst Variants (for Learning)",
        "",
        "| Rank | N | X | Session | Δ Total PnL | Δ Worst-20 | Affected Trades |",
        "|------|---|---|---------|--------------|-------------|-----------------|",
    ])
    
    for rank, (variant_key, result) in enumerate(sorted_results[-3:], 1):
        params = result['params']
        delta = result['delta']
        affected = result['affected']
        lines.append(
            f"| {rank} | {params['N']} | {params['X']:.1f} | {params['session']} | "
            f"{delta['total_pnl_bps']:+.2f} ({delta['total_pnl_pct']:+.2f}%) | "
            f"{delta['tail_metrics']['worst_20_sum']:+.2f} | "
            f"{affected['total_trades']:,} |"
        )
    
    lines.extend([
        "",
        "---",
        "",
        "## GO / NO-GO Recommendation",
        "",
    ])
    
    # Find best variant that meets kill criteria
    best_variant = None
    for variant_key, result in sorted_results:
        delta = result['delta']
        stable_edge = result.get('stable_edge_impact', {})
        
        # Check kill criteria
        if (delta['total_pnl_pct'] > -10 and  # Total PnL reduction < 10%
            stable_edge.get('delta_pct', 0) > -5 and  # Stable edge reduction < 5%
            delta['tail_metrics']['worst_20_sum'] > 0):  # Tail improvement
            best_variant = (variant_key, result)
            break
    
    if best_variant:
        variant_key, result = best_variant
        params = result['params']
        lines.extend([
            f"**✅ GO:** Variant N={params['N']}, X={params['X']}, session={params['session']}",
            "",
            f"- Δ Total PnL: {result['delta']['total_pnl_pct']:+.2f}% (within -10% limit)",
            f"- Δ Worst-20: {result['delta']['tail_metrics']['worst_20_sum']:+.2f} bps (improvement)",
            f"- Stable Edge Impact: {stable_edge.get('delta_pct', 0):+.2f}% (within -5% limit)",
            f"- Affected Trades: {result['affected']['total_trades']:,}",
            "",
            "**Next Steps:**",
            "1. Validate on monster-PC with A/B test",
            "2. Implement with `GX1_HYPOTHESIS_002_EARLY_EXIT` flag",
            "3. Monitor stable edge bins closely",
        ])
    else:
        lines.extend([
            "**❌ NO-GO:** No variant meets all kill criteria",
            "",
            "**Issues:**",
            "- All variants either reduce total PnL > 10% OR",
            "- Affect stable edge bins > 5% OR",
            "- Do not improve tail metrics",
            "",
            "**Recommendation:**",
            "- Refine hypothesis (adjust parameters or criteria)",
            "- Consider alternative approaches",
        ])
    
    lines.extend([
        "",
        "---",
        "",
        "## Comparison with Hypothesis #1 (Entry-Side)",
        "",
        "| Aspect | Hypothesis #1 (Entry Veto) | Hypothesis #2 (Exit Delay) |",
        "|--------|----------------------------|----------------------------|",
        "| Approach | Block trades before entry | Exit trades earlier |",
        "| Trade Count | Reduced | Preserved |",
        "| Edge Preservation | High (blocks before entry) | Medium (exits after entry) |",
        "| Tail Improvement | Moderate | High (if successful) |",
        "| Implementation | Pre-entry gate | Exit policy modification |",
        "",
        "**Key Insight:**",
        "- Entry-side reduces noise before it enters",
        "- Exit-side preserves opportunities but cuts losers earlier",
        "- Both approaches can complement each other",
        "",
        "---",
        "",
        "## Detailed Results (All Variants)",
        "",
    ])
    
    for variant_key, result in sorted_results:
        params = result['params']
        delta = result['delta']
        affected = result['affected']
        stable_edge = result.get('stable_edge_impact', {})
        
        lines.extend([
            f"### Variant: N={params['N']}, X={params['X']}, session={params['session']}",
            "",
            f"- **Δ Total PnL:** {delta['total_pnl_bps']:+.2f} bps ({delta['total_pnl_pct']:+.2f}%)",
            f"- **Δ Worst-20:** {delta['tail_metrics']['worst_20_sum']:+.2f} bps",
            f"- **Δ Worst 1%:** {delta['tail_metrics']['worst_1pct_avg']:+.2f} bps",
            f"- **Affected Trades:** {affected['total_trades']:,}",
            f"- **Stable Edge Impact:** {stable_edge.get('delta_pct', 0):+.2f}%",
            "",
        ])
    
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    
    log.info(f"✅ Report written: {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze counterfactual impact of early exit')
    parser.add_argument('--trades-parquet', type=Path, required=True,
                        help='Path to trades parquet file')
    parser.add_argument('--stable-edge-json', type=Path, required=True,
                        help='Path to stable edge bins JSON')
    parser.add_argument('--out-root', type=Path, required=True,
                        help='Output root directory')
    parser.add_argument('--N-values', type=str, default='3,5,7',
                        help='Comma-separated N values (bars threshold)')
    parser.add_argument('--X-values', type=str, default='2.0,3.0,5.0,7.0',
                        help='Comma-separated X values (MFE threshold in bps)')
    parser.add_argument('--session', type=str, default='OVERLAP',
                        help='Target session (default: OVERLAP)')
    
    args = parser.parse_args()
    
    # Parse parameter grid
    N_values = [int(n.strip()) for n in args.N_values.split(',')]
    X_values = [float(x.strip()) for x in args.X_values.split(',')]
    
    log.info(f"Parameter grid: N ∈ {N_values}, X ∈ {X_values}, session={args.session}")
    log.info(f"Total combinations: {len(N_values) * len(X_values)}")
    
    # Load data
    df = load_trades(args.trades_parquet)
    stable_bin_keys = load_stable_edge_bins(args.stable_edge_json)
    
    # Analyze all combinations
    all_results = {}
    
    for N, X in itertools.product(N_values, X_values):
        variant_key = f"N{N}_X{X:.1f}_{args.session}"
        log.info(f"\n{'='*80}")
        log.info(f"Analyzing variant: {variant_key}")
        log.info(f"{'='*80}\n")
        
        try:
            result = analyze_counterfactual_exit_delay(
                df.copy(),  # Copy to avoid modifying original
                N, X, args.session, stable_bin_keys
            )
            all_results[variant_key] = result
            
        except Exception as e:
            log.error(f"❌ Error analyzing {variant_key}: {e}")
            # Continue with other variants
            continue
    
    # Generate report
    report_path = args.out_root / "HYPOTHESIS_002_RESULTS.md"
    generate_results_report(all_results, report_path)
    
    # Save JSON
    json_path = args.out_root / "hypothesis_002_results.json"
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    log.info(f"\n✅ Counterfactual exit-delay analysis complete")
    log.info(f"   Report: {report_path}")
    log.info(f"   JSON: {json_path}")


if __name__ == '__main__':
    main()
