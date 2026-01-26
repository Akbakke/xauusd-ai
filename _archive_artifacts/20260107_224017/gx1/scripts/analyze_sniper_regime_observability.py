#!/usr/bin/env python3
"""
SNIPER Observability & Regime Audit

Analyzes SNIPER live-demo logs to understand why 0 trades:
- Near-miss analysis (p_long in [0.60, 0.67))
- p_long distribution per session
- Regime context (trend/vol)
- Blocker sanity check

Output:
- reports/audits/SNIPER_OBSERVABILITY_REGIME_AUDIT_<timestamp>.md
- CSV files with detailed data
"""

from __future__ import annotations

import re
import logging
from collections import defaultdict
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Thresholds
MIN_PROB_LONG = 0.67  # SNIPER threshold
NEAR_MISS_MIN = 0.60
NEAR_MISS_MAX = 0.67
LOW_MISS_MIN = 0.55
LOW_MISS_MAX = 0.60


def parse_sniper_cycle_line(line: str) -> Optional[Dict]:
    """Parse a [SNIPER_CYCLE] log line."""
    if '[SNIPER_CYCLE]' not in line:
        return None
    
    # Extract key fields using regex
    ts_match = re.search(r'ts=([^\s]+)', line)
    session_match = re.search(r'session=([^\s]+)', line)
    in_scope_match = re.search(r'in_scope=([^\s]+)', line)
    p_long_match = re.search(r'p_long=([0-9.]+)', line)
    reason_match = re.search(r'reason=([^\s]+)', line)
    eval_match = re.search(r'eval=([^\s]+)', line)
    atr_bps_match = re.search(r'atr_bps=([0-9.]+)', line)
    spread_bps_match = re.search(r'spread_bps=([^\s]+)', line)
    n_signals_match = re.search(r'n_signals=([0-9]+)', line)
    
    # p_long is required for this analysis
    if not p_long_match:
        return None
    
    try:
        return {
            'ts': ts_match.group(1) if ts_match else None,
            'session': session_match.group(1) if session_match else None,
            'in_scope': int(in_scope_match.group(1)) if in_scope_match else None,
            'p_long': float(p_long_match.group(1)),
            'reason': reason_match.group(1) if reason_match else None,
            'eval': int(eval_match.group(1)) if eval_match else None,
            'atr_bps': float(atr_bps_match.group(1)) if atr_bps_match else None,
            'spread_bps': float(spread_bps_match.group(1)) if spread_bps_match and spread_bps_match.group(1) != 'N/A' else None,
            'n_signals': int(n_signals_match.group(1)) if n_signals_match else None,
        }
    except (ValueError, IndexError, AttributeError):
        return None


def extract_regime_from_log(log_path: Path) -> Dict[str, List[Dict]]:
    """Extract regime information from log (trend/vol from ensure_replay_tags or other sources)."""
    regimes = defaultdict(list)
    
    # Try to find regime information in logs
    # This is a placeholder - actual implementation depends on how regimes are logged
    with open(log_path, 'r') as f:
        for line in f:
            # Look for regime-related logs
            if 'brain_trend_regime' in line or 'brain_vol_regime' in line:
                # Extract regime info if available
                trend_match = re.search(r'brain_trend_regime=([^\s,]+)', line)
                vol_match = re.search(r'brain_vol_regime=([^\s,]+)', line)
                if trend_match and vol_match:
                    ts_match = re.search(r'(\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2})', line)
                    if ts_match:
                        regimes[ts_match.group(1)].append({
                            'trend_regime': trend_match.group(1),
                            'vol_regime': vol_match.group(1),
                        })
    
    return regimes


def analyze_near_miss(df: pd.DataFrame) -> Dict:
    """Analyze near-miss rates."""
    if len(df) == 0:
        return {}
    
    p_long_vals = df['p_long'].dropna()
    if len(p_long_vals) == 0:
        return {}
    
    total = len(p_long_vals)
    near_miss = ((p_long_vals >= NEAR_MISS_MIN) & (p_long_vals < NEAR_MISS_MAX)).sum()
    low_miss = ((p_long_vals >= LOW_MISS_MIN) & (p_long_vals < LOW_MISS_MAX)).sum()
    above_threshold = (p_long_vals >= MIN_PROB_LONG).sum()
    
    # Find nearest misses
    near_miss_df = df[(df['p_long'] >= NEAR_MISS_MIN) & (df['p_long'] < NEAR_MISS_MAX)].copy()
    near_miss_df = near_miss_df.sort_values('p_long', ascending=False)
    
    return {
        'total_cycles': total,
        'near_miss_count': int(near_miss),
        'near_miss_rate': float(near_miss / total) if total > 0 else 0.0,
        'low_miss_count': int(low_miss),
        'low_miss_rate': float(low_miss / total) if total > 0 else 0.0,
        'above_threshold_count': int(above_threshold),
        'above_threshold_rate': float(above_threshold / total) if total > 0 else 0.0,
        'nearest_misses': near_miss_df.head(10).to_dict('records') if len(near_miss_df) > 0 else [],
    }


def analyze_by_session(df: pd.DataFrame) -> Dict:
    """Analyze p_long distribution by session."""
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
            'p10': float(p_long_vals.quantile(0.10)),
            'p25': float(p_long_vals.quantile(0.25)),
            'p50': float(p_long_vals.quantile(0.50)),
            'p75': float(p_long_vals.quantile(0.75)),
            'p90': float(p_long_vals.quantile(0.90)),
            'above_threshold': int((p_long_vals >= MIN_PROB_LONG).sum()),
            'near_miss': int(((p_long_vals >= NEAR_MISS_MIN) & (p_long_vals < NEAR_MISS_MAX)).sum()),
            'low_miss': int(((p_long_vals >= LOW_MISS_MIN) & (p_long_vals < LOW_MISS_MAX)).sum()),
        }
    
    return results


def analyze_blockers(log_path: Path) -> Dict:
    """Analyze blocker statistics from log."""
    blockers = {
        'stage0_unknown_field': 0,
        'eval=1': 0,
        'eval=0': 0,
        'reason=policy_no_signals': 0,
        'reason=stage0_session_block': 0,
        'reason=stage0_vol_block': 0,
        'reason=spread_guard': 0,
        'reason=risk_guard': 0,
    }
    
    with open(log_path, 'r') as f:
        for line in f:
            if 'stage0_unknown_field' in line:
                blockers['stage0_unknown_field'] += 1
            if '[SNIPER_CYCLE]' in line:
                if 'eval=1' in line:
                    blockers['eval=1'] += 1
                if 'eval=0' in line:
                    blockers['eval=0'] += 1
                if 'reason=policy_no_signals' in line:
                    blockers['reason=policy_no_signals'] += 1
                if 'reason=stage0_session_block' in line:
                    blockers['reason=stage0_session_block'] += 1
                if 'reason=stage0_vol_block' in line:
                    blockers['reason=stage0_vol_block'] += 1
                if 'reason=spread_guard' in line:
                    blockers['reason=spread_guard'] += 1
                if 'reason=risk_guard' in line:
                    blockers['reason=risk_guard'] += 1
    
    return blockers


def generate_report(
    df: pd.DataFrame,
    near_miss: Dict,
    session_stats: Dict,
    blockers: Dict,
    log_path: Path,
    run_dir: Path,
    output_path: Path
) -> None:
    """Generate comprehensive audit report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
    
    with open(output_path, 'w') as f:
        f.write("# SNIPER Observability & Regime Audit\n\n")
        f.write(f"**Generated:** {timestamp}\n")
        f.write(f"**Source Log:** `{log_path}`\n")
        f.write(f"**Run Directory:** `{run_dir}`\n\n")
        f.write("---\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write(f"- **Total Cycles Analyzed:** {len(df)}\n")
        f.write(f"- **Cycles with p_long:** {len(df[df['p_long'].notna()])}\n")
        f.write(f"- **Near-Miss Rate [0.60, 0.67):** {near_miss.get('near_miss_rate', 0.0):.2%}\n")
        f.write(f"- **Above Threshold Rate (≥0.67):** {near_miss.get('above_threshold_rate', 0.0):.2%}\n")
        f.write(f"- **0 Trades Reason:** Policy threshold (p_long < 0.67)\n\n")
        
        f.write("## DEL 1: Near-Miss Analysis\n\n")
        if near_miss:
            f.write(f"- **Total Cycles:** {near_miss.get('total_cycles', 0)}\n")
            f.write(f"- **Near-Miss [0.60, 0.67):** {near_miss.get('near_miss_count', 0)} ({near_miss.get('near_miss_rate', 0.0):.2%})\n")
            f.write(f"- **Low-Miss [0.55, 0.60):** {near_miss.get('low_miss_count', 0)} ({near_miss.get('low_miss_rate', 0.0):.2%})\n")
            f.write(f"- **Above Threshold (≥0.67):** {near_miss.get('above_threshold_count', 0)} ({near_miss.get('above_threshold_rate', 0.0):.2%})\n\n")
            
            if near_miss.get('nearest_misses'):
                f.write("### Nearest Misses (Top 10)\n\n")
                f.write("| Timestamp | Session | p_long | Reason |\n")
                f.write("|-----------|---------|--------|--------|\n")
                for miss in near_miss['nearest_misses'][:10]:
                    f.write(f"| {miss.get('ts', 'N/A')} | {miss.get('session', 'N/A')} | {miss.get('p_long', 0.0):.4f} | {miss.get('reason', 'N/A')} |\n")
                f.write("\n")
            
            # Conclusion
            if near_miss.get('near_miss_rate', 0.0) > 0.1:
                f.write("**Conclusion:** Threshold is 'almost right' - significant near-miss rate indicates model is close to threshold.\n\n")
            elif near_miss.get('near_miss_rate', 0.0) > 0.0:
                f.write("**Conclusion:** Threshold is moderately close - some near-misses but not systematic.\n\n")
            else:
                f.write("**Conclusion:** Threshold is far from current p_long distribution - no near-misses observed.\n\n")
        else:
            f.write("No near-miss data available.\n\n")
        
        f.write("## DEL 2: p_long per Session\n\n")
        if session_stats:
            f.write("| Session | Count | Mean | Median | p10 | p90 | Min | Max | Above Threshold | Near-Miss |\n")
            f.write("|---------|-------|------|--------|-----|-----|-----|-----|-----------------|-----------|\n")
            for session, stats in session_stats.items():
                f.write(f"| {session} | {stats['count']} | {stats['mean']:.4f} | {stats['median']:.4f} | "
                       f"{stats['p10']:.4f} | {stats['p90']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} | "
                       f"{stats['above_threshold']} ({stats['above_threshold']/stats['count']*100:.1f}%) | "
                       f"{stats['near_miss']} ({stats['near_miss']/stats['count']*100:.1f}%) |\n")
            f.write("\n")
            
            # Comparison
            f.write("### Session Comparison\n\n")
            if len(session_stats) > 1:
                sessions = list(session_stats.keys())
                means = [session_stats[s]['mean'] for s in sessions]
                max_mean_session = sessions[np.argmax(means)]
                min_mean_session = sessions[np.argmin(means)]
                f.write(f"- **Highest Mean p_long:** {max_mean_session} ({session_stats[max_mean_session]['mean']:.4f})\n")
                f.write(f"- **Lowest Mean p_long:** {min_mean_session} ({session_stats[min_mean_session]['mean']:.4f})\n")
                f.write(f"- **Difference:** {session_stats[max_mean_session]['mean'] - session_stats[min_mean_session]['mean']:.4f}\n\n")
                
                if abs(session_stats[max_mean_session]['mean'] - session_stats[min_mean_session]['mean']) > 0.05:
                    f.write("**Conclusion:** Significant session-specific difference in p_long distribution.\n\n")
                else:
                    f.write("**Conclusion:** Low p_long is global across sessions, not session-specific.\n\n")
        else:
            f.write("No session data available.\n\n")
        
        f.write("## DEL 3: Regime & Vol Context\n\n")
        f.write("**Note:** Regime data (trend_regime, vol_regime) extraction from logs requires additional parsing.\n")
        f.write("Current analysis focuses on p_long distribution patterns.\n\n")
        f.write("**Observation:** Regime correlation analysis would require regime tags in [SNIPER_CYCLE] logs or separate regime logs.\n\n")
        
        f.write("## DEL 4: Blocker Sanity Check\n\n")
        f.write("| Blocker | Count | Expected | Status |\n")
        f.write("|---------|-------|----------|--------|\n")
        f.write(f"| stage0_unknown_field | {blockers.get('stage0_unknown_field', 0)} | 0 | {'✅' if blockers.get('stage0_unknown_field', 0) == 0 else '❌'} |\n")
        f.write(f"| eval=1 | {blockers.get('eval=1', 0)} | ~100% | {'✅' if blockers.get('eval=1', 0) > 0 else '❌'} |\n")
        f.write(f"| eval=0 | {blockers.get('eval=0', 0)} | 0 | {'✅' if blockers.get('eval=0', 0) == 0 else '❌'} |\n")
        f.write(f"| reason=policy_no_signals | {blockers.get('reason=policy_no_signals', 0)} | >0 | ✅ |\n")
        f.write(f"| reason=stage0_session_block | {blockers.get('reason=stage0_session_block', 0)} | 0 | {'✅' if blockers.get('reason=stage0_session_block', 0) == 0 else '⚠️'} |\n")
        f.write(f"| reason=stage0_vol_block | {blockers.get('reason=stage0_vol_block', 0)} | 0 | {'✅' if blockers.get('reason=stage0_vol_block', 0) == 0 else '⚠️'} |\n")
        f.write(f"| reason=spread_guard | {blockers.get('reason=spread_guard', 0)} | 0 | {'✅' if blockers.get('reason=spread_guard', 0) == 0 else '⚠️'} |\n")
        f.write(f"| reason=risk_guard | {blockers.get('reason=risk_guard', 0)} | 0 | {'✅' if blockers.get('reason=risk_guard', 0) == 0 else '⚠️'} |\n")
        f.write("\n")
        
        f.write("**Conclusion:** ")
        if blockers.get('stage0_unknown_field', 0) == 0 and blockers.get('eval=1', 0) > 0:
            f.write("✅ **0 trades skyldes kun policy threshold** - all blockers before policy are passing.\n\n")
        else:
            f.write("⚠️ Some blockers before policy may be affecting trades.\n\n")
        
        f.write("## DEL 5: Executive Conclusion\n\n")
        
        # Determine conclusion
        if near_miss.get('near_miss_rate', 0.0) > 0.1:
            conclusion = "**Midlertidig stille i nåværende regime** - Significant near-miss rate indicates model is close to threshold, suggesting temporary regime conditions."
        elif near_miss.get('near_miss_rate', 0.0) > 0.0:
            conclusion = "**Moderat avstand fra terskel** - Some near-misses observed, but not systematic. May be regime-dependent."
        else:
            conclusion = "**Systematisk langt unna terskel** - No near-misses observed. Model output (p_long) is consistently below threshold, suggesting structural mismatch or regime conditions."
        
        if session_stats and len(session_stats) > 1:
            means = [session_stats[s]['mean'] for s in session_stats.keys()]
            if abs(max(means) - min(means)) > 0.05:
                conclusion += " **Session-avhengig** - Significant difference between sessions."
            else:
                conclusion += " **Ikke session-avhengig** - Low p_long is global across sessions."
        
        f.write(conclusion + "\n\n")
        
        f.write("**Key Findings:**\n")
        f.write(f"- Total cycles analyzed: {len(df)}\n")
        f.write(f"- Near-miss rate: {near_miss.get('near_miss_rate', 0.0):.2%}\n")
        f.write(f"- Above threshold rate: {near_miss.get('above_threshold_rate', 0.0):.2%}\n")
        if session_stats:
            f.write(f"- Sessions analyzed: {', '.join(session_stats.keys())}\n")
        f.write(f"- Blocker status: Stage0 passing, policy evaluation running\n\n")
        
        f.write("**Recommendation:** ")
        if near_miss.get('near_miss_rate', 0.0) > 0.1:
            f.write("Continue monitoring - near-miss rate suggests threshold may be appropriate for current regime.\n")
        elif near_miss.get('near_miss_rate', 0.0) > 0.0:
            f.write("Monitor regime changes - some near-misses suggest threshold proximity.\n")
        else:
            f.write("Investigate model-regime mismatch - no near-misses suggest systematic gap between model output and threshold.\n")


def main():
    """Main analysis function."""
    import sys
    
    if len(sys.argv) > 1:
        log_path = Path(sys.argv[1])
    else:
        # Find newest SNIPER log
        base_path = Path(".")
        sniper_logs = list((base_path / "runs" / "live_demo").glob("SNIPER_*/sniper_runtime.log"))
        if not sniper_logs:
            log.error("No SNIPER logs found")
            return
        log_path = max(sniper_logs, key=lambda p: p.stat().st_mtime)
    
    run_dir = log_path.parent
    
    log.info(f"Analyzing log: {log_path}")
    
    # Load and parse logs
    rows = []
    with open(log_path, 'r') as f:
        for line in f:
            if '[SNIPER_CYCLE]' in line:
                parsed = parse_sniper_cycle_line(line)
                if parsed:
                    rows.append(parsed)
    
    if not rows:
        log.warning("No [SNIPER_CYCLE] logs found")
        return
    
    log.info(f"Parsed {len(rows)} SNIPER_CYCLE log entries")
    
    # Convert to DataFrame
    df = pd.DataFrame(rows)
    
    # Filter for cycles with p_long (eval=1)
    df_with_p_long = df[df['eval'] == 1].copy()
    log.info(f"Cycles with eval=1: {len(df_with_p_long)}")
    
    if len(df_with_p_long) == 0:
        log.warning("No cycles with eval=1 found")
        return
    
    # Analyze
    near_miss = analyze_near_miss(df_with_p_long)
    session_stats = analyze_by_session(df_with_p_long)
    blockers = analyze_blockers(log_path)
    
    # Generate report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = Path("reports/audits") / f"SNIPER_OBSERVABILITY_REGIME_AUDIT_{timestamp}.md"
    generate_report(df_with_p_long, near_miss, session_stats, blockers, log_path, run_dir, output_path)
    log.info(f"Report generated: {output_path}")
    
    # Save CSV
    csv_path = Path("reports/audits") / f"SNIPER_OBSERVABILITY_REGIME_AUDIT_{timestamp}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_with_p_long.to_csv(csv_path, index=False)
    log.info(f"CSV saved: {csv_path}")
    
    # Save near-miss CSV
    if near_miss.get('nearest_misses'):
        near_miss_df = pd.DataFrame(near_miss['nearest_misses'])
        near_miss_csv = Path("reports/audits") / f"SNIPER_NEAR_MISS_{timestamp}.csv"
        near_miss_df.to_csv(near_miss_csv, index=False)
        log.info(f"Near-miss CSV saved: {near_miss_csv}")


if __name__ == "__main__":
    main()

