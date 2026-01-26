#!/usr/bin/env python3
"""
SNIPER 2025 Signal Access Analysis

Read-only analysis script that processes SNIPER 2025 chunk-level trade journals
to understand signal distribution, near-miss rates, and regime-mix patterns.

Output:
- reports/sniper/diagnostics/SNIPER_SIGNAL_ACCESS_2025.md
- CSV metrics for further comparison
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Thresholds
MIN_PROB_LONG = 0.67  # SNIPER threshold
NEAR_MISS_MIN = 0.60
NEAR_MISS_MAX = 0.67


def find_sniper_trade_journals(base_path: Path) -> List[Path]:
    """Find all SNIPER trade journal directories from 2025."""
    trade_dirs = []
    
    # Search in wf_runs for SNIPER runs
    wf_runs_path = base_path / "gx1" / "wf_runs"
    if wf_runs_path.exists():
        # Look for SNIPER runs (Q1-Q4 2025)
        for run_dir in wf_runs_path.iterdir():
            if not run_dir.is_dir():
                continue
            
            run_name = run_dir.name
            if not ('SNIPER' in run_name.upper() and '2025' in run_name):
                continue
            
            # Priority 1: parallel_chunks (chunk-level journals)
            parallel_chunks = run_dir / "parallel_chunks"
            if parallel_chunks.exists():
                for chunk_dir in parallel_chunks.iterdir():
                    if not chunk_dir.is_dir() or not chunk_dir.name.startswith("chunk_"):
                        continue
                    
                    trade_journal = chunk_dir / "trade_journal"
                    if trade_journal.exists() and (trade_journal / "trades").exists():
                        json_files = list((trade_journal / "trades").glob("*.json"))
                        if json_files:
                            trade_dirs.append(trade_journal)
                            log.info(f"Found SNIPER chunk journal: {trade_journal} ({len(json_files)} trades)")
            
            # Priority 2: direct trade_journal (fallback)
            trade_journal = run_dir / "trade_journal"
            if trade_journal.exists() and (trade_journal / "trades").exists():
                json_files = list((trade_journal / "trades").glob("*.json"))
                if json_files:
                    # Check if SNIPER-related
                    try:
                        with open(json_files[0], 'r') as f:
                            data = json.load(f)
                            policy = data.get('entry_snapshot', {}).get('policy_name', '')
                            if 'SNIPER' in policy.upper() or 'SNIPER' in run_name.upper():
                                trade_dirs.append(trade_journal)
                                log.info(f"Found SNIPER trade journal: {trade_journal} ({len(json_files)} trades)")
                    except Exception as e:
                        log.debug(f"Error checking {json_files[0]}: {e}")
    
    return trade_dirs


def load_trade_data(trade_dir: Path) -> List[Dict]:
    """Load all trades from a trade journal directory."""
    trades = []
    trades_path = trade_dir / "trades"
    
    if not trades_path.exists():
        return trades
    
    for trade_file in trades_path.glob("*.json"):
        try:
            with open(trade_file, 'r') as f:
                trade = json.load(f)
                trades.append(trade)
        except Exception as e:
            log.warning(f"Error loading {trade_file}: {e}")
    
    return trades


def extract_signal_data(trades: List[Dict]) -> pd.DataFrame:
    """Extract signal data from trades."""
    rows = []
    
    for trade in trades:
        entry_snapshot = trade.get('entry_snapshot')
        if entry_snapshot is None:
            # Try feature_context as fallback
            feature_context = trade.get('feature_context', {})
            if not feature_context:
                continue  # Skip trades without entry data
            
            entry_snapshot = feature_context
        
        if not isinstance(entry_snapshot, dict):
            continue
        
        # Extract key fields
        row = {
            'trade_id': trade.get('trade_id'),
            'entry_time': entry_snapshot.get('entry_time'),
            'session': entry_snapshot.get('session'),
            'side': entry_snapshot.get('side'),
            'p_long': entry_snapshot.get('p_long'),
            'p_short': entry_snapshot.get('p_short'),
            'n_signals': entry_snapshot.get('n_signals'),
            'trend_regime': entry_snapshot.get('trend_regime') or entry_snapshot.get('brain_trend_regime'),
            'vol_regime': entry_snapshot.get('vol_regime') or entry_snapshot.get('brain_vol_regime'),
            'atr_bps': entry_snapshot.get('atr_bps'),
            'spread_bps': entry_snapshot.get('spread_bps'),
            'base_units': entry_snapshot.get('base_units'),
            'entry_price': entry_snapshot.get('entry_price'),
        }
        
        # Extract from policy_state if available
        policy_state = entry_snapshot.get('policy_state', {})
        if isinstance(policy_state, dict):
            if not row['trend_regime']:
                row['trend_regime'] = policy_state.get('brain_trend_regime')
            if not row['vol_regime']:
                row['vol_regime'] = policy_state.get('brain_vol_regime')
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def analyze_p_long_distribution(df: pd.DataFrame) -> Dict:
    """Analyze p_long distribution per session."""
    results = {}
    
    for session in ['EU', 'OVERLAP', 'US', 'ASIA']:
        session_df = df[df['session'] == session]
        if len(session_df) == 0:
            continue
        
        p_long_vals = session_df['p_long'].dropna()
        if len(p_long_vals) == 0:
            continue
        
        results[session] = {
            'count': len(p_long_vals),
            'mean': float(p_long_vals.mean()),
            'median': float(p_long_vals.median()),
            'std': float(p_long_vals.std()),
            'min': float(p_long_vals.min()),
            'max': float(p_long_vals.max()),
            'percentiles': {
                'p10': float(p_long_vals.quantile(0.10)),
                'p25': float(p_long_vals.quantile(0.25)),
                'p50': float(p_long_vals.quantile(0.50)),
                'p75': float(p_long_vals.quantile(0.75)),
                'p90': float(p_long_vals.quantile(0.90)),
                'p95': float(p_long_vals.quantile(0.95)),
                'p99': float(p_long_vals.quantile(0.99)),
            },
            'above_threshold': int((p_long_vals >= MIN_PROB_LONG).sum()),
            'near_miss': int(((p_long_vals >= NEAR_MISS_MIN) & (p_long_vals < NEAR_MISS_MAX)).sum()),
        }
    
    return results


def analyze_near_miss_rate(df: pd.DataFrame) -> Dict:
    """Analyze near-miss rate (p_long in [0.60, 0.67))."""
    p_long_vals = df['p_long'].dropna()
    
    if len(p_long_vals) == 0:
        return {}
    
    total = len(p_long_vals)
    near_miss = ((p_long_vals >= NEAR_MISS_MIN) & (p_long_vals < NEAR_MISS_MAX)).sum()
    above_threshold = (p_long_vals >= MIN_PROB_LONG).sum()
    
    return {
        'total_cycles': total,
        'near_miss_count': int(near_miss),
        'near_miss_rate': float(near_miss / total) if total > 0 else 0.0,
        'above_threshold_count': int(above_threshold),
        'above_threshold_rate': float(above_threshold / total) if total > 0 else 0.0,
    }


def analyze_regime_mix(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze regime-mix (trend × vol) with p_long."""
    regime_df = df[['trend_regime', 'vol_regime', 'p_long', 'session']].copy()
    regime_df = regime_df.dropna(subset=['trend_regime', 'vol_regime', 'p_long'])
    
    if len(regime_df) == 0:
        return pd.DataFrame()
    
    # Create cross-tabulation
    regime_stats = regime_df.groupby(['trend_regime', 'vol_regime']).agg({
        'p_long': ['count', 'mean', 'std', 'min', 'max'],
    }).reset_index()
    
    regime_stats.columns = ['trend_regime', 'vol_regime', 'count', 'p_long_mean', 'p_long_std', 'p_long_min', 'p_long_max']
    
    return regime_stats


def generate_report(
    p_long_dist: Dict,
    near_miss: Dict,
    regime_mix: pd.DataFrame,
    total_trades: int,
    output_path: Path
) -> None:
    """Generate markdown report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("# SNIPER 2025 Signal Access Analysis\n\n")
        f.write(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n")
        f.write("---\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- **Total Trades Analyzed:** {total_trades}\n")
        f.write(f"- **Near-Miss Rate:** {near_miss.get('near_miss_rate', 0.0):.2%}\n")
        f.write(f"- **Above Threshold Rate:** {near_miss.get('above_threshold_rate', 0.0):.2%}\n\n")
        
        f.write("## p_long Distribution by Session\n\n")
        for session, stats in p_long_dist.items():
            f.write(f"### {session}\n\n")
            f.write(f"- **Count:** {stats['count']}\n")
            f.write(f"- **Mean:** {stats['mean']:.4f}\n")
            f.write(f"- **Median:** {stats['median']:.4f}\n")
            f.write(f"- **Std:** {stats['std']:.4f}\n")
            f.write(f"- **Range:** [{stats['min']:.4f}, {stats['max']:.4f}]\n")
            f.write(f"- **Above Threshold (≥0.67):** {stats['above_threshold']} ({stats['above_threshold']/stats['count']*100:.1f}%)\n")
            f.write(f"- **Near-Miss [0.60, 0.67):** {stats['near_miss']} ({stats['near_miss']/stats['count']*100:.1f}%)\n\n")
            f.write("Percentiles:\n")
            for p_name, p_val in stats['percentiles'].items():
                f.write(f"- {p_name}: {p_val:.4f}\n")
            f.write("\n")
        
        f.write("## Regime-Mix Analysis\n\n")
        if len(regime_mix) > 0:
            f.write("| Trend | Vol | Count | p_long Mean | p_long Std | p_long Min | p_long Max |\n")
            f.write("|-------|-----|-------|-------------|------------|------------|------------|\n")
            for _, row in regime_mix.iterrows():
                f.write(f"| {row['trend_regime']} | {row['vol_regime']} | {row['count']} | "
                       f"{row['p_long_mean']:.4f} | {row['p_long_std']:.4f} | "
                       f"{row['p_long_min']:.4f} | {row['p_long_max']:.4f} |\n")
        else:
            f.write("No regime data available.\n")
        
        f.write("\n## Near-Miss Analysis\n\n")
        f.write(f"- **Total Cycles:** {near_miss.get('total_cycles', 0)}\n")
        f.write(f"- **Near-Miss Count:** {near_miss.get('near_miss_count', 0)}\n")
        f.write(f"- **Near-Miss Rate:** {near_miss.get('near_miss_rate', 0.0):.2%}\n")
        f.write(f"- **Above Threshold Count:** {near_miss.get('above_threshold_count', 0)}\n")
        f.write(f"- **Above Threshold Rate:** {near_miss.get('above_threshold_rate', 0.0):.2%}\n")


def main():
    """Main analysis function."""
    base_path = Path(".")
    
    log.info("Finding SNIPER trade journals...")
    trade_dirs = find_sniper_trade_journals(base_path)
    
    if not trade_dirs:
        log.warning("No SNIPER trade journals found")
        return
    
    log.info(f"Found {len(trade_dirs)} SNIPER trade journal directories")
    
    # Load all trades
    all_trades = []
    for trade_dir in trade_dirs:
        trades = load_trade_data(trade_dir)
        all_trades.extend(trades)
        log.info(f"Loaded {len(trades)} trades from {trade_dir}")
    
    if not all_trades:
        log.warning("No trades found")
        return
    
    log.info(f"Total trades: {len(all_trades)}")
    
    # Extract signal data
    df = extract_signal_data(all_trades)
    log.info(f"Extracted {len(df)} trade records")
    
    # Analyze
    p_long_dist = analyze_p_long_distribution(df)
    near_miss = analyze_near_miss_rate(df)
    regime_mix = analyze_regime_mix(df)
    
    # Generate report
    output_path = base_path / "reports" / "sniper" / "diagnostics" / "SNIPER_SIGNAL_ACCESS_2025.md"
    generate_report(p_long_dist, near_miss, regime_mix, len(all_trades), output_path)
    log.info(f"Report generated: {output_path}")
    
    # Save CSV
    csv_path = base_path / "reports" / "sniper" / "diagnostics" / "SNIPER_SIGNAL_ACCESS_2025.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    log.info(f"CSV saved: {csv_path}")
    
    # Save regime mix CSV
    if len(regime_mix) > 0:
        regime_csv = base_path / "reports" / "sniper" / "diagnostics" / "SNIPER_REGIME_MIX_2025.csv"
        regime_mix.to_csv(regime_csv, index=False)
        log.info(f"Regime mix CSV saved: {regime_csv}")


if __name__ == "__main__":
    main()

