#!/usr/bin/env python3
"""
SNIPER Shadow Threshold Analysis

Analyzes shadow threshold hits from shadow_journal jsonl files to determine
which threshold would give desired trade frequency (1-5 trades/day).

Output:
- reports/sniper/diagnostics/SNIPER_SHADOW_THRESHOLDS_<timestamp>.md
- reports/sniper/diagnostics/SNIPER_SHADOW_THRESHOLDS_<timestamp>.csv
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Cycles per day (assuming ~5 min cycles)
CYCLES_PER_DAY = 288  # 24 hours * 60 min / 5 min


def load_shadow_journal(journal_path: Path) -> pd.DataFrame:
    """Load shadow journal jsonl file."""
    records = []
    with open(journal_path, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError as e:
                    log.warning(f"Skipping invalid JSON line: {e}")
    
    if not records:
        return pd.DataFrame()
    
    # Expand shadow_hits dict into columns
    df = pd.DataFrame(records)
    if 'shadow_hits' in df.columns:
        shadow_cols = {}
        for idx, row in df.iterrows():
            shadow_hits = row.get('shadow_hits', {})
            for thr_str, hit in shadow_hits.items():
                thr = float(thr_str)
                col_name = f"shadow_hit_{thr:.2f}"
                if col_name not in shadow_cols:
                    shadow_cols[col_name] = {}
                shadow_cols[col_name][idx] = hit
        
        for col_name, values in shadow_cols.items():
            df[col_name] = pd.Series(values)
    
    return df


def analyze_shadow_thresholds(df: pd.DataFrame) -> Dict:
    """Analyze shadow threshold hit rates."""
    if len(df) == 0:
        return {}
    
    results = {}
    
    # Get shadow threshold columns
    shadow_cols = [col for col in df.columns if col.startswith('shadow_hit_')]
    thresholds = [float(col.replace('shadow_hit_', '')) for col in shadow_cols]
    thresholds = sorted(thresholds, reverse=True)
    
    total_cycles = len(df)
    
    for thr in thresholds:
        col_name = f"shadow_hit_{thr:.2f}"
        if col_name not in df.columns:
            continue
        
        hits = df[col_name].fillna(False).astype(bool)
        hit_count = int(hits.sum())
        hit_rate = float(hit_count / total_cycles) if total_cycles > 0 else 0.0
        
        # Estimate trades per day
        trades_per_day = hit_rate * CYCLES_PER_DAY
        
        results[thr] = {
            'threshold': thr,
            'hit_count': hit_count,
            'hit_rate': hit_rate,
            'hit_rate_pct': hit_rate * 100.0,
            'trades_per_day_est': trades_per_day,
        }
    
    # Analyze by session
    if 'session' in df.columns:
        session_results = {}
        for session in df['session'].unique():
            session_df = df[df['session'] == session]
            session_total = len(session_df)
            
            session_hits = {}
            for thr in thresholds:
                col_name = f"shadow_hit_{thr:.2f}"
                if col_name not in session_df.columns:
                    continue
                hits = session_df[col_name].fillna(False).astype(bool)
                hit_count = int(hits.sum())
                hit_rate = float(hit_count / session_total) if session_total > 0 else 0.0
                session_hits[thr] = {
                    'hit_count': hit_count,
                    'hit_rate': hit_rate,
                    'hit_rate_pct': hit_rate * 100.0,
                }
            session_results[session] = {
                'total_cycles': session_total,
                'hits_by_threshold': session_hits,
            }
        results['by_session'] = session_results
    
    # Analyze by regime (if available)
    if 'trend_regime' in df.columns and 'vol_regime' in df.columns:
        regime_results = {}
        for trend in df['trend_regime'].dropna().unique():
            for vol in df['vol_regime'].dropna().unique():
                regime_df = df[(df['trend_regime'] == trend) & (df['vol_regime'] == vol)]
                regime_total = len(regime_df)
                if regime_total == 0:
                    continue
                
                regime_key = f"{trend}_{vol}"
                regime_hits = {}
                for thr in thresholds:
                    col_name = f"shadow_hit_{thr:.2f}"
                    if col_name not in regime_df.columns:
                        continue
                    hits = regime_df[col_name].fillna(False).astype(bool)
                    hit_count = int(hits.sum())
                    hit_rate = float(hit_count / regime_total) if regime_total > 0 else 0.0
                    regime_hits[thr] = {
                        'hit_count': hit_count,
                        'hit_rate': hit_rate,
                        'hit_rate_pct': hit_rate * 100.0,
                    }
                regime_results[regime_key] = {
                    'trend_regime': trend,
                    'vol_regime': vol,
                    'total_cycles': regime_total,
                    'hits_by_threshold': regime_hits,
                }
        results['by_regime'] = regime_results
    
    # p_long distribution for near-misses
    if 'p_long' in df.columns:
        p_long_vals = df['p_long'].dropna()
        results['p_long_stats'] = {
            'count': len(p_long_vals),
            'mean': float(p_long_vals.mean()),
            'median': float(p_long_vals.median()),
            'min': float(p_long_vals.min()),
            'max': float(p_long_vals.max()),
            'p10': float(p_long_vals.quantile(0.10)),
            'p25': float(p_long_vals.quantile(0.25)),
            'p50': float(p_long_vals.quantile(0.50)),
            'p75': float(p_long_vals.quantile(0.75)),
            'p90': float(p_long_vals.quantile(0.90)),
        }
    
    results['total_cycles'] = total_cycles
    
    return results


def generate_report(
    results: Dict,
    journal_path: Path,
    output_path: Path,
) -> None:
    """Generate comprehensive shadow threshold analysis report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
    
    with open(output_path, 'w') as f:
        f.write("# SNIPER Shadow Threshold Analysis\n\n")
        f.write(f"**Generated:** {timestamp}\n")
        f.write(f"**Source Journal:** `{journal_path}`\n\n")
        f.write("---\n\n")
        
        f.write("## Executive Summary\n\n")
        total_cycles = results.get('total_cycles', 0)
        f.write(f"- **Total Cycles Analyzed:** {total_cycles}\n")
        f.write(f"- **Cycles per Day (estimate):** {CYCLES_PER_DAY}\n\n")
        
        # Threshold summary table
        f.write("## Shadow Threshold Hit Rates\n\n")
        f.write("| Threshold | Hit Count | Hit Rate (%) | Est. Trades/Day | Recommendation |\n")
        f.write("|-----------|-----------|--------------|-----------------|----------------|\n")
        
        thresholds = sorted([k for k in results.keys() if isinstance(k, float)], reverse=True)
        recommended_threshold: Optional[float] = None
        
        for thr in thresholds:
            thr_data = results[thr]
            hit_rate_pct = thr_data['hit_rate_pct']
            trades_per_day = thr_data['trades_per_day_est']
            
            # Recommendation logic
            if 1.0 <= trades_per_day <= 5.0:
                rec = "✅ **RECOMMENDED** (1-5 trades/day)"
                if recommended_threshold is None:
                    recommended_threshold = thr
            elif trades_per_day > 20.0:
                rec = "⚠️ **TOO HIGH** (>20 trades/day)"
            elif trades_per_day > 5.0:
                rec = "⚠️ Moderate (5-20 trades/day)"
            elif trades_per_day > 0.1:
                rec = "Low (<1 trade/day)"
            else:
                rec = "Very Low (<0.1 trades/day)"
            
            f.write(f"| {thr:.2f} | {thr_data['hit_count']} | {hit_rate_pct:.2f}% | {trades_per_day:.2f} | {rec} |\n")
        
        f.write("\n")
        
        # Session breakdown
        if 'by_session' in results:
            f.write("## Hit Rates by Session\n\n")
            for session, session_data in results['by_session'].items():
                f.write(f"### {session}\n\n")
                f.write(f"Total Cycles: {session_data['total_cycles']}\n\n")
                f.write("| Threshold | Hit Count | Hit Rate (%) |\n")
                f.write("|-----------|-----------|--------------|\n")
                for thr in sorted(session_data['hits_by_threshold'].keys(), reverse=True):
                    hit_data = session_data['hits_by_threshold'][thr]
                    f.write(f"| {thr:.2f} | {hit_data['hit_count']} | {hit_data['hit_rate_pct']:.2f}% |\n")
                f.write("\n")
        
        # Regime breakdown
        if 'by_regime' in results:
            f.write("## Hit Rates by Regime\n\n")
            for regime_key, regime_data in results['by_regime'].items():
                f.write(f"### {regime_data['trend_regime']} × {regime_data['vol_regime']}\n\n")
                f.write(f"Total Cycles: {regime_data['total_cycles']}\n\n")
                f.write("| Threshold | Hit Count | Hit Rate (%) |\n")
                f.write("|-----------|-----------|--------------|\n")
                for thr in sorted(regime_data['hits_by_threshold'].keys(), reverse=True):
                    hit_data = regime_data['hits_by_threshold'][thr]
                    f.write(f"| {thr:.2f} | {hit_data['hit_count']} | {hit_data['hit_rate_pct']:.2f}% |\n")
                f.write("\n")
        
        # p_long distribution
        if 'p_long_stats' in results:
            f.write("## p_long Distribution (Near-Misses)\n\n")
            stats = results['p_long_stats']
            f.write(f"- **Count:** {stats['count']}\n")
            f.write(f"- **Mean:** {stats['mean']:.4f}\n")
            f.write(f"- **Median:** {stats['median']:.4f}\n")
            f.write(f"- **Min:** {stats['min']:.4f}\n")
            f.write(f"- **Max:** {stats['max']:.4f}\n")
            f.write(f"- **p10:** {stats['p10']:.4f}\n")
            f.write(f"- **p25:** {stats['p25']:.4f}\n")
            f.write(f"- **p50:** {stats['p50']:.4f}\n")
            f.write(f"- **p75:** {stats['p75']:.4f}\n")
            f.write(f"- **p90:** {stats['p90']:.4f}\n\n")
        
        # Recommendation
        f.write("## Recommendation\n\n")
        if recommended_threshold is not None:
            f.write(f"**Recommended Threshold:** `{recommended_threshold:.2f}`\n\n")
            f.write(f"This threshold is estimated to produce **{results[recommended_threshold]['trades_per_day_est']:.2f} trades/day**,\n")
            f.write("which falls within the target range of 1-5 trades/day.\n\n")
            f.write("**Next Steps:**\n")
            f.write("1. Continue monitoring shadow hits for additional cycles (6-12 hours)\n")
            f.write("2. Validate recommendation with historical data if available\n")
            f.write("3. Consider A/B testing in replay mode before live deployment\n")
            f.write("4. **Do not change production threshold** until sufficient validation\n\n")
        else:
            f.write("**No threshold found in target range (1-5 trades/day).**\n\n")
            f.write("**Options:**\n")
            f.write("1. Continue collecting data (may need more cycles)\n")
            f.write("2. Consider lower thresholds if current range is too conservative\n")
            f.write("3. Review model calibration if hit rates are consistently low\n\n")
        
        f.write("---\n\n")
        f.write("**Note:** This analysis is based on shadow threshold hits only.\n")
        f.write("Actual trade frequency may vary based on market conditions, spreads, and other factors.\n")


def main():
    """Main analysis function."""
    import sys
    
    if len(sys.argv) > 1:
        journal_path = Path(sys.argv[1])
    else:
        # Find newest shadow journal
        base_path = Path(".")
        shadow_journals = list((base_path / "runs" / "live_demo").glob("SNIPER_*/shadow/shadow_hits.jsonl"))
        if not shadow_journals:
            log.error("No shadow journals found. Expected: runs/live_demo/SNIPER_*/shadow/shadow_hits.jsonl")
            return
        journal_path = max(shadow_journals, key=lambda p: p.stat().st_mtime)
    
    log.info(f"Analyzing shadow journal: {journal_path}")
    
    # Load and analyze
    df = load_shadow_journal(journal_path)
    if len(df) == 0:
        log.warning("No shadow records found in journal")
        return
    
    log.info(f"Loaded {len(df)} shadow records")
    
    results = analyze_shadow_thresholds(df)
    
    # Generate report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = Path("reports/sniper/diagnostics") / f"SNIPER_SHADOW_THRESHOLDS_{timestamp}.md"
    generate_report(results, journal_path, output_path)
    log.info(f"Report generated: {output_path}")
    
    # Save CSV
    csv_path = Path("reports/sniper/diagnostics") / f"SNIPER_SHADOW_THRESHOLDS_{timestamp}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    log.info(f"CSV saved: {csv_path}")


if __name__ == "__main__":
    main()

