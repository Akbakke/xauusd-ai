#!/usr/bin/env python3
"""
Analyze hybrid exit Q2 replay results.
Compares RULE5 vs RULE6A performance within the same replay.
"""
import pandas as pd
import sys
from pathlib import Path
from typing import Dict, Any

def calculate_max_drawdown(pnl_series: pd.Series) -> float:
    """Calculate maximum drawdown in bps."""
    cumulative = pnl_series.cumsum()
    running_max = cumulative.expanding().max()
    drawdown = cumulative - running_max
    return drawdown.min()

def analyze_group(df: pd.DataFrame, group_name: str, days: float) -> Dict[str, Any]:
    """Analyze a group of trades."""
    if len(df) == 0:
        return {
            'group': group_name,
            'trades': 0,
            'trades_per_day': 0.0,
            'win_rate': 0.0,
            'ev_per_trade': 0.0,
            'ev_per_day': 0.0,
            'max_drawdown': 0.0,
        }
    
    n_trades = len(df)
    trades_per_day = n_trades / days
    win_rate = (df['pnl_bps'] > 0).sum() / n_trades * 100
    ev_per_trade = df['pnl_bps'].mean()
    ev_per_day = ev_per_trade * trades_per_day
    max_drawdown = calculate_max_drawdown(df['pnl_bps'])
    
    return {
        'group': group_name,
        'trades': n_trades,
        'trades_per_day': trades_per_day,
        'win_rate': win_rate,
        'ev_per_trade': ev_per_trade,
        'ev_per_day': ev_per_day,
        'max_drawdown': max_drawdown,
    }

def main():
    # Path to trade log (check both possible locations)
    possible_paths = [
        Path("gx1/wf_runs/FARM_V2B_EXIT_HYBRID_Q2/trade_log_entry_v9_exit_v3_adaptive_3SEG_merged.csv"),
        Path("gx1/wf_runs/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_Q2/trade_log_entry_v9_exit_v3_adaptive_3SEG_merged.csv"),
    ]
    
    trade_log_path = None
    for path in possible_paths:
        if path.exists():
            trade_log_path = path
            break
    
    if trade_log_path is None:
        # Try to find any merged CSV in HYBRID_Q2 directories
        for base_dir in ["gx1/wf_runs/FARM_V2B_EXIT_HYBRID_Q2", "gx1/wf_runs/GX1_V11_OANDA_DEMO_V9_FARM_V2B_EXIT_HYBRID_Q2"]:
            base_path = Path(base_dir)
            if base_path.exists():
                csv_files = list(base_path.glob("*merged.csv"))
                if csv_files:
                    trade_log_path = csv_files[0]
                    break
    
    if not trade_log_path.exists():
        print(f"ERROR: Trade log not found: {trade_log_path}")
        print("Please run the hybrid Q2 replay first.")
        sys.exit(1)
    
    print("=" * 80)
    print("HYBRID EXIT Q2 ANALYSIS")
    print("=" * 80)
    print()
    print(f"Loading trade log: {trade_log_path}")
    
    # Load trade log
    df = pd.read_csv(trade_log_path)
    
    # Filter to closed trades only (exclude REPLAY_END)
    df_closed = df[df['exit_reason'] != 'REPLAY_END'].copy()
    
    # Filter to long-only
    df_long = df_closed[df_closed['side'] == 'long'].copy()
    
    if len(df_long) == 0:
        print("ERROR: No long trades found in trade log")
        sys.exit(1)
    
    print(f"✅ Loaded {len(df_long)} closed long trades")
    print()
    
    # Calculate days from date range
    if 'exit_time' in df_long.columns:
        df_long['exit_time'] = pd.to_datetime(df_long['exit_time'])
        date_range = df_long['exit_time'].max() - df_long['exit_time'].min()
        days = date_range.days + 1  # Inclusive
    elif 'entry_time' in df_long.columns:
        df_long['entry_time'] = pd.to_datetime(df_long['entry_time'])
        date_range = df_long['entry_time'].max() - df_long['entry_time'].min()
        days = date_range.days + 1
    else:
        # Q2 is approximately 91 days (2025-04-01 to 2025-06-30)
        days = 91
        print(f"⚠️  No date columns found, using default Q2 days: {days}")
    
    print(f"📅 Period: {days} days")
    print()
    
    # Check exit_profile column
    if 'exit_profile' not in df_long.columns:
        print("ERROR: 'exit_profile' column not found in trade log")
        print(f"Available columns: {list(df_long.columns)}")
        sys.exit(1)
    
    # Check unique exit profiles
    unique_profiles = df_long['exit_profile'].unique()
    print(f"📊 Exit profiles found: {list(unique_profiles)}")
    print()
    
    # Split into groups
    rule5_df = df_long[df_long['exit_profile'] == 'FARM_EXIT_V2_RULES_A'].copy()
    rule6a_df = df_long[df_long['exit_profile'] == 'FARM_EXIT_V2_RULES_ADAPTIVE_v1'].copy()
    
    # Also check for variations
    if len(rule5_df) == 0:
        # Try alternative names
        rule5_df = df_long[df_long['exit_profile'].str.contains('RULE5', case=False, na=False)].copy()
    if len(rule6a_df) == 0:
        rule6a_df = df_long[df_long['exit_profile'].str.contains('ADAPTIVE', case=False, na=False)].copy()
    
    print(f"📈 RULE5 trades: {len(rule5_df)}")
    print(f"📈 RULE6A trades: {len(rule6a_df)}")
    print(f"📈 Other/Unknown: {len(df_long) - len(rule5_df) - len(rule6a_df)}")
    print()
    
    # Analyze groups
    total_stats = analyze_group(df_long, "TOTAL", days)
    rule5_stats = analyze_group(rule5_df, "RULE5_ONLY", days)
    rule6a_stats = analyze_group(rule6a_df, "RULE6A_ONLY", days)
    
    # Print table
    print("=" * 80)
    print("RESULTS TABLE")
    print("=" * 80)
    print()
    print(f"{'Group':<15} {'Trades':<8} {'Trades/day':<12} {'Win%':<8} {'EV/trade':<12} {'EV/day':<12} {'MaxDD':<10}")
    print("-" * 80)
    
    for stats in [total_stats, rule5_stats, rule6a_stats]:
        print(f"{stats['group']:<15} "
              f"{stats['trades']:<8} "
              f"{stats['trades_per_day']:<12.2f} "
              f"{stats['win_rate']:<8.1f} "
              f"{stats['ev_per_trade']:<12.2f} "
              f"{stats['ev_per_day']:<12.2f} "
              f"{stats['max_drawdown']:<10.2f}")
    
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    
    # Find best EV/day
    groups_with_trades = [s for s in [rule5_stats, rule6a_stats] if s['trades'] > 0]
    if groups_with_trades:
        best_ev_day = max(groups_with_trades, key=lambda x: x['ev_per_day'])
        worst_ev_day = min(groups_with_trades, key=lambda x: x['ev_per_day'])
        
        print(f"🏆 Highest EV/day: {best_ev_day['group']} ({best_ev_day['ev_per_day']:.2f} bps)")
        print(f"   - Trades: {best_ev_day['trades']} ({best_ev_day['trades_per_day']:.2f}/day)")
        print(f"   - Win rate: {best_ev_day['win_rate']:.1f}%")
        print(f"   - EV/trade: {best_ev_day['ev_per_trade']:.2f} bps")
        print()
        
        print(f"📉 Lowest EV/day: {worst_ev_day['group']} ({worst_ev_day['ev_per_day']:.2f} bps)")
        print(f"   - Trades: {worst_ev_day['trades']} ({worst_ev_day['trades_per_day']:.2f}/day)")
        print(f"   - Win rate: {worst_ev_day['win_rate']:.1f}%")
        print(f"   - EV/trade: {worst_ev_day['ev_per_trade']:.2f} bps")
        print()
    
    # Find lowest drawdown
    groups_with_trades = [s for s in [rule5_stats, rule6a_stats] if s['trades'] > 0]
    if groups_with_trades:
        best_dd = min(groups_with_trades, key=lambda x: x['max_drawdown'])
        worst_dd = max(groups_with_trades, key=lambda x: x['max_drawdown'])
        
        print(f"🛡️  Lowest Max Drawdown: {best_dd['group']} ({best_dd['max_drawdown']:.2f} bps)")
        print(f"   - Max DD: {best_dd['max_drawdown']:.2f} bps")
        print()
        
        if best_dd['group'] != worst_dd['group']:
            print(f"⚠️  Highest Max Drawdown: {worst_dd['group']} ({worst_dd['max_drawdown']:.2f} bps)")
            print()
    
    # Compare total vs individual groups
    print("=" * 80)
    print("HYBRID vs INDIVIDUAL STRATEGIES")
    print("=" * 80)
    print()
    
    if total_stats['trades'] > 0:
        print(f"📊 Total (Hybrid):")
        print(f"   EV/day: {total_stats['ev_per_day']:.2f} bps")
        print(f"   Trades/day: {total_stats['trades_per_day']:.2f}")
        print(f"   Max DD: {total_stats['max_drawdown']:.2f} bps")
        print()
        
        if len(groups_with_trades) >= 2:
            rule5_ev = rule5_stats['ev_per_day']
            rule6a_ev = rule6a_stats['ev_per_day']
            total_ev = total_stats['ev_per_day']
            
            print(f"Comparison:")
            print(f"   RULE5_ONLY: {rule5_ev:.2f} bps/day")
            print(f"   RULE6A_ONLY: {rule6a_ev:.2f} bps/day")
            print(f"   HYBRID (Total): {total_ev:.2f} bps/day")
            print()
            
            if total_ev > max(rule5_ev, rule6a_ev):
                print("✅ Hybrid performs BETTER than both individual strategies!")
            elif total_ev > min(rule5_ev, rule6a_ev):
                print("⚠️  Hybrid performs between individual strategies")
            else:
                print("❌ Hybrid performs WORSE than both individual strategies")
            print()
    
    # Exit profile distribution
    print("=" * 80)
    print("EXIT PROFILE DISTRIBUTION")
    print("=" * 80)
    print()
    
    profile_counts = df_long['exit_profile'].value_counts()
    for profile, count in profile_counts.items():
        pct = count / len(df_long) * 100
        print(f"  {profile}: {count} ({pct:.1f}%)")
    
    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()

