#!/usr/bin/env python3
"""
Analyze Phase-C.2 tuning results and compare with baseline + Phase-C variants.

Extracts metrics from Q2 2025 replays for variants D, E, F and compares
with PROD baseline and existing Phase-C variants (A, B, C).
"""

import sys
from pathlib import Path

# Add project root to PYTHONPATH
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import json
from typing import Dict, Any, Optional

def load_trade_log(trade_log_path: Path) -> pd.DataFrame:
    """Load trade log CSV."""
    if not trade_log_path.exists():
        raise FileNotFoundError(f"Trade log not found: {trade_log_path}")
    
    df = pd.read_csv(trade_log_path, on_bad_lines='skip', engine='python')
    return df

def compute_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute trading metrics from trade log."""
    # Filter closed trades
    closed = df[
        df['exit_time'].notna() & 
        (df['exit_time'] != '') &
        df['pnl_bps'].notna() &
        (df['pnl_bps'] != '')
    ].copy()
    
    if len(closed) == 0:
        return {
            "n_trades": 0,
            "trades_per_day": 0.0,
            "win_rate": 0.0,
            "ev_per_trade_bps": 0.0,
            "ev_per_day_bps": 0.0,
            "max_drawdown_bps": 0.0,
            "missing_exit_profile": 0,
        }
    
    # Convert types
    closed['pnl_bps'] = pd.to_numeric(closed['pnl_bps'], errors='coerce')
    closed = closed[closed['pnl_bps'].notna()]
    
    if len(closed) == 0:
        return {
            "n_trades": 0,
            "trades_per_day": 0.0,
            "win_rate": 0.0,
            "ev_per_trade_bps": 0.0,
            "ev_per_day_bps": 0.0,
            "max_drawdown_bps": 0.0,
            "missing_exit_profile": 0,
        }
    
    # Period
    closed['entry_time'] = pd.to_datetime(closed['entry_time'], utc=True, errors='coerce')
    closed['exit_time'] = pd.to_datetime(closed['exit_time'], utc=True, errors='coerce')
    period_start = closed['entry_time'].min()
    period_end = closed['exit_time'].max()
    period_days = (period_end - period_start).days if pd.notna(period_end) and pd.notna(period_start) else 0
    
    # Metrics
    n_trades = len(closed)
    pnls = closed['pnl_bps'].values
    wins = pnls[pnls > 0]
    
    ev_per_trade = np.mean(pnls)
    ev_per_day = ev_per_trade * (n_trades / max(period_days, 1)) if period_days > 0 else 0
    winrate = len(wins) / n_trades if n_trades > 0 else 0
    
    # Max drawdown
    equity = np.cumsum(pnls)
    running_max = np.maximum.accumulate(equity)
    drawdown = equity - running_max
    max_dd = np.min(drawdown) if len(drawdown) > 0 else 0
    
    # Missing exit_profile
    missing_exit_profile = df['exit_profile'].isna().sum() if 'exit_profile' in df.columns else 0
    
    return {
        "n_trades": n_trades,
        "trades_per_day": n_trades / max(period_days, 1) if period_days > 0 else 0.0,
        "win_rate": winrate * 100.0,
        "ev_per_trade_bps": ev_per_trade,
        "ev_per_day_bps": ev_per_day,
        "max_drawdown_bps": max_dd,
        "missing_exit_profile": int(missing_exit_profile),
    }

def load_results_json(results_path: Path) -> Optional[Dict[str, Any]]:
    """Load results.json if it exists."""
    if not results_path.exists():
        return None
    
    try:
        with open(results_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load {results_path}: {e}")
        return None

def main():
    """Main analysis function."""
    base_dir = Path("gx1/wf_runs")
    
    # Define variants
    variants = {
        "Baseline PROD": {
            "trade_log": base_dir / "FARM_V2B_EXIT_A_PROD_2025" / "trade_log_entry_v9_exit_v3_adaptive_3SEG_merged.csv",
            "results": base_dir / "FARM_V2B_EXIT_A_PROD_2025" / "results.json",
        },
        "Variant A (mp67)": {
            "trade_log": base_dir / "EXIT_A_PHASEC_mp67" / "trade_log_entry_v9_exit_v3_adaptive_3SEG_merged.csv",
            "results": base_dir / "EXIT_A_PHASEC_mp67" / "results.json",
            "note": "Existing Phase-C results",
        },
        "Variant B (mp69)": {
            "trade_log": base_dir / "EXIT_A_PHASEC_mp69" / "trade_log_entry_v9_exit_v3_adaptive_3SEG_merged.csv",
            "results": base_dir / "EXIT_A_PHASEC_mp69" / "results.json",
            "note": "Existing Phase-C results",
        },
        "Variant C (max6)": {
            "trade_log": base_dir / "EXIT_A_PHASEC_max6" / "trade_log_entry_v9_exit_v3_adaptive_3SEG_merged.csv",
            "results": base_dir / "EXIT_A_PHASEC_max6" / "results.json",
            "note": "Existing Phase-C results",
        },
        "Variant D (max4)": {
            "trade_log": base_dir / "GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_A_PHASEC2_max4" / "trade_log_entry_v9_exit_v3_adaptive_3SEG_merged.csv",
            "results": base_dir / "GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_A_PHASEC2_max4" / "results.json",
            "note": "Phase-C.2: max_open_trades=4",
        },
        "Variant E (rule6)": {
            "trade_log": base_dir / "GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_A_PHASEC2_rule6" / "trade_log_entry_v9_exit_v3_adaptive_3SEG_merged.csv",
            "results": base_dir / "GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_A_PHASEC2_rule6" / "results.json",
            "note": "Phase-C.2: rule_a_profit_min_bps=6",
        },
        "Variant F (trail3)": {
            "trade_log": base_dir / "GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_A_PHASEC2_trail3" / "trade_log_entry_v9_exit_v3_adaptive_3SEG_merged.csv",
            "results": base_dir / "GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_A_PHASEC2_trail3" / "results.json",
            "note": "Phase-C.2: rule_a_trailing_stop_bps=3",
        },
    }
    
    results = {}
    
    for variant_name, paths in variants.items():
        print(f"\n{'='*80}")
        print(f"Analyzing: {variant_name}")
        print(f"{'='*80}")
        
        trade_log_path = paths["trade_log"]
        results_path = paths["results"]
        
        if not trade_log_path.exists():
            print(f"⚠️  Trade log not found: {trade_log_path}")
            print(f"   Skipping {variant_name}")
            continue
        
        try:
            df = load_trade_log(trade_log_path)
            metrics = compute_metrics(df)
            results_json = load_results_json(results_path)
            
            if results_json:
                # Merge with results.json if available
                metrics.update({
                    "period_days": results_json.get("period_days", 0),
                    "total_bars": results_json.get("total_bars", 0),
                })
            
            metrics["variant_name"] = variant_name
            metrics["note"] = paths.get("note", "")
            results[variant_name] = metrics
            
            print(f"✅ Metrics computed:")
            print(f"   Trades: {metrics['n_trades']}")
            print(f"   Trades/day: {metrics['trades_per_day']:.2f}")
            print(f"   Win rate: {metrics['win_rate']:.1f}%")
            print(f"   EV/trade: {metrics['ev_per_trade_bps']:.2f} bps")
            print(f"   EV/day: {metrics['ev_per_day_bps']:.2f} bps")
            print(f"   Max DD: {metrics['max_drawdown_bps']:.2f} bps")
            print(f"   Missing exit_profile: {metrics['missing_exit_profile']}")
            
        except Exception as e:
            print(f"❌ Error analyzing {variant_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results to CSV
    output_dir = Path("gx1/tuning")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_df = pd.DataFrame.from_dict(results, orient='index')
    csv_path = output_dir / "exit_a_long_phaseC2_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n✅ Results saved to: {csv_path}")
    
    # Save results to JSON
    json_path = output_dir / "exit_a_long_phaseC2_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Results saved to: {json_path}")
    
    # Print comparison table
    print(f"\n{'='*80}")
    print("COMPARISON TABLE")
    print(f"{'='*80}")
    print(results_df.to_string(index=False))
    
    return results

if __name__ == "__main__":
    main()

