#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline replay script for testing entry + exit together.

Runs a full replay with both entry and exit strategies enabled.
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from gx1.execution.oanda_demo_runner import GX1DemoRunner, load_yaml_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_replay_policy(
    entry_config_path: Optional[str] = None,
    exit_config_path: Optional[str] = None,
    base_policy_path: str = "gx1/configs/policies/GX1_V11_OANDA_DEMO.yaml",
) -> Dict[str, Any]:
    """
    Create a policy dict for replay mode.
    
    Args:
        entry_config_path: Path to entry config (optional, uses default if None)
        exit_config_path: Path to exit config (optional, uses default if None)
        base_policy_path: Path to base policy YAML
    
    Returns:
        Policy dict for replay
    """
    # Load base policy
    base_policy = load_yaml_config(Path(base_policy_path))
    
    # Override entry config if provided
    if entry_config_path:
        base_policy["entry_config"] = entry_config_path
    
    # Override exit config if provided
    if exit_config_path:
        base_policy["exit_config"] = exit_config_path
    
    # Ensure replay-friendly settings
    base_policy["risk"]["dry_run"] = True
    base_policy["execution"]["dry_run"] = True
    
    # Disable tick_exit for bar-based exits only
    if "tick_exit" in base_policy:
        base_policy["tick_exit"]["enabled"] = False
    
    # Configure exit_control to allow EXIT_V2_DRIFT exits
    if "exit_control" not in base_policy:
        base_policy["exit_control"] = {}
    # Allow EXIT_V2_DRIFT exits (including SL)
    base_policy["exit_control"]["allowed_loss_closers"] = [
        "BROKER_SL",
        "SOFT_STOP_TICK",
        "EXIT_V2_DRIFT_SL",
        "EXIT_V2_DRIFT_SL_BREAKEVEN",
        "EXIT_V2_DRIFT_TP2",
        "EXIT_V2_DRIFT_TIMEOUT",
    ]
    
    # Enable ENTRY_V9 if not already configured
    if "entry_models" not in base_policy:
        base_policy["entry_models"] = {}
    if "v9" not in base_policy["entry_models"]:
        base_policy["entry_models"]["v9"] = {
            "enabled": True,
            "model_dir": "gx1/models/entry_v9/nextgen_2020_2025_clean",
        }
    else:
        # Ensure enabled
        base_policy["entry_models"]["v9"]["enabled"] = True
    
    return base_policy


def compute_summary_stats(trades_df: pd.DataFrame, period_days: float) -> Dict[str, Any]:
    """
    Compute summary statistics from trades DataFrame.
    
    Args:
        trades_df: DataFrame with trade results
        period_days: Number of days in the test period
    
    Returns:
        Dict with summary statistics
    """
    if len(trades_df) == 0:
        return {
            "n_trades": 0,
            "mean_pnl_bps": 0.0,
            "median_pnl_bps": 0.0,
            "ev_per_trade": 0.0,
            "trades_per_day": 0.0,
            "ev_per_day": 0.0,
            "tp2_rate": 0.0,
            "sl_rate": 0.0,
            "timeout_rate": 0.0,
            "sl_breakeven_rate": 0.0,
            "median_bars_held": 0.0,
            "session_breakdown": {},
            "volatility_breakdown": {},
        }
    
    pnl_values = trades_df["pnl_bps"].values if "pnl_bps" in trades_df.columns else trades_df["realized_pnl_bps"].values
    
    # Exit reason fractions
    exit_reason_col = "exit_reason" if "exit_reason" in trades_df.columns else "reason"
    if exit_reason_col in trades_df.columns:
        tp2_rate = (trades_df[exit_reason_col].str.contains("TP2", case=False, na=False)).mean()
        sl_rate = (trades_df[exit_reason_col].str.contains("SL", case=False, na=False) & 
                  ~trades_df[exit_reason_col].str.contains("BREAKEVEN", case=False, na=False)).mean()
        timeout_rate = (trades_df[exit_reason_col].str.contains("TIMEOUT", case=False, na=False)).mean()
        sl_breakeven_rate = (trades_df[exit_reason_col].str.contains("SL_BREAKEVEN", case=False, na=False) |
                            trades_df[exit_reason_col].str.contains("BREAKEVEN", case=False, na=False)).mean()
    else:
        tp2_rate = sl_rate = timeout_rate = sl_breakeven_rate = 0.0
    
    # Bars held
    bars_held_col = "bars_held" if "bars_held" in trades_df.columns else "bars_in_trade"
    median_bars_held = float(trades_df[bars_held_col].median()) if bars_held_col in trades_df.columns else 0.0
    
    # Session breakdown
    session_breakdown = {}
    if "session" in trades_df.columns:
        for session in ["ASIA", "EU", "US", "OVERLAP"]:
            session_trades = trades_df[trades_df["session"].str.contains(session, case=False, na=False)]
            if len(session_trades) > 0:
                session_breakdown[session] = {
                    "n_trades": len(session_trades),
                    "mean_pnl_bps": float(session_trades["pnl_bps"].mean() if "pnl_bps" in session_trades.columns else session_trades["realized_pnl_bps"].mean()),
                }
    
    # Volatility breakdown
    volatility_breakdown = {}
    vol_col = "vol_regime" if "vol_regime" in trades_df.columns else "atr_regime"
    if vol_col in trades_df.columns:
        for regime in ["LOW", "MEDIUM", "HIGH", "EXTREME"]:
            regime_trades = trades_df[trades_df[vol_col].str.contains(regime, case=False, na=False)]
            if len(regime_trades) > 0:
                volatility_breakdown[regime] = {
                    "n_trades": len(regime_trades),
                    "mean_pnl_bps": float(regime_trades["pnl_bps"].mean() if "pnl_bps" in regime_trades.columns else regime_trades["realized_pnl_bps"].mean()),
                }
    
    trades_per_day = len(trades_df) / period_days if period_days > 0 else 0.0
    mean_pnl = float(np.mean(pnl_values))
    ev_per_day = mean_pnl * trades_per_day
    
    return {
        "n_trades": len(trades_df),
        "mean_pnl_bps": mean_pnl,
        "median_pnl_bps": float(np.median(pnl_values)),
        "ev_per_trade": mean_pnl,
        "trades_per_day": trades_per_day,
        "ev_per_day": ev_per_day,
        "tp2_rate": float(tp2_rate),
        "sl_rate": float(sl_rate),
        "timeout_rate": float(timeout_rate),
        "sl_breakeven_rate": float(sl_breakeven_rate),
        "median_bars_held": median_bars_held,
        "session_breakdown": session_breakdown,
        "volatility_breakdown": volatility_breakdown,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline replay with entry + exit")
    parser.add_argument(
        "--price-data",
        type=str,
        required=True,
        help="Path to price data parquet/CSV",
    )
    parser.add_argument(
        "--entry-model",
        type=str,
        default=None,
        help="Path to entry model config (optional)",
    )
    parser.add_argument(
        "--exit-config",
        type=str,
        default=None,
        help="Path to exit config (e.g., gx1/configs/exits/EXIT_V2_DRIFT.yaml)",
    )
    parser.add_argument(
        "--limit-bars",
        type=int,
        default=None,
        help="Limit to last N bars (e.g., 60000)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/replay_entry_exit.json",
        help="Output path for results JSON",
    )
    parser.add_argument(
        "--base-policy",
        type=str,
        default="gx1/configs/policies/GX1_V11_OANDA_DEMO.yaml",
        help="Base policy YAML path",
    )
    args = parser.parse_args()
    
    print("=" * 80)
    print("OFFLINE REPLAY: ENTRY + EXIT")
    print("=" * 80)
    print()
    
    # Create replay policy
    print("[1/5] Creating replay policy...")
    policy_dict = create_replay_policy(
        entry_config_path=args.entry_model,
        exit_config_path=args.exit_config,
        base_policy_path=args.base_policy,
    )
    
    if args.exit_config:
        print(f"   Exit config: {args.exit_config}")
    if args.entry_model:
        print(f"   Entry model: {args.entry_model}")
    print()
    
    # Load price data
    print(f"[2/5] Loading price data: {args.price_data}")
    price_path = Path(args.price_data)
    
    if price_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(price_path)
    else:
        df = pd.read_csv(price_path)
    
    # Normalize column names
    df.columns = [c.lower() for c in df.columns]
    
    # Set up time index
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time").sort_index()
    elif "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df = df.set_index("ts").sort_index()
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    
    # Limit to last N bars if specified
    if args.limit_bars and len(df) > args.limit_bars:
        df = df.iloc[-args.limit_bars:].copy()
        print(f"   Limited to last {args.limit_bars:,} bars")
    
    print(f"✅ Loaded {len(df):,} bars")
    print(f"   Period: {df.index.min()} to {df.index.max()}")
    
    # Calculate period in days
    period_days = (df.index.max() - df.index.min()).total_seconds() / (24 * 3600)
    print(f"   Period: {period_days:.1f} days")
    print()
    
    # Create runner
    print("[3/5] Initializing runner...")
    
    # Save policy to temp file and pass path
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp_file:
        import yaml
        yaml.dump(policy_dict, tmp_file)
        tmp_policy_path = Path(tmp_file.name)
    
    runner = GX1DemoRunner(
        tmp_policy_path,
        dry_run_override=True,
        replay_mode=True,
        fast_replay=True,  # Skip heavy reporting
    )
    
    # Set max bars if specified
    if args.limit_bars:
        runner._max_bars = args.limit_bars
    
    print("✅ Runner initialized")
    print()
    
    # Run replay with progress tracking
    print("[4/5] Running replay...")
    print(f"   Processing {len(df):,} bars...")
    print("   Progress will be logged every 1000 bars")
    print()
    
    # Monkey-patch run_replay to add progress tracking
    original_run_replay = runner.run_replay
    
    def run_replay_with_progress(csv_path):
        # Call original but we'll track progress in the loop
        # We need to patch the inner loop instead
        return original_run_replay(csv_path)
    
    # Actually, we need to patch the inner processing loop
    # Let's add a progress callback to the runner
    import time
    start_time = time.time()
    last_log_time = start_time
    
    # Store original method
    original_process_bar = None
    
    # Try to add progress tracking by patching the replay loop
    # We'll use a simpler approach: monitor the trade log file
    runner.run_replay(price_path)
    
    # Check if replay completed by looking for summary dump
    elapsed = time.time() - start_time
    print(f"✅ Replay complete (took {elapsed/60:.1f} minutes)")
    print()
    
    # Extract trade results
    print("[5/5] Extracting results...")
    
    # Read trade log
    trade_log_path = runner.trade_log_path
    if not trade_log_path.exists():
        print(f"❌ Trade log not found: {trade_log_path}")
        return 1
    
    trades_df = pd.read_csv(trade_log_path, on_bad_lines='skip', engine='python')
    
    # Filter closed trades
    closed_trades = trades_df[
        trades_df["exit_time"].notna() &
        trades_df["pnl_bps"].notna() &
        (trades_df["pnl_bps"] != "")
    ].copy()
    
    if len(closed_trades) == 0:
        print("❌ No closed trades found!")
        return 1
    
    # Convert pnl_bps to float
    closed_trades["pnl_bps"] = pd.to_numeric(closed_trades["pnl_bps"], errors='coerce')
    closed_trades = closed_trades[closed_trades["pnl_bps"].notna()]
    
    # Add exit reason if available
    if "exit_reason" not in closed_trades.columns and "notes" in closed_trades.columns:
        # Try to extract from notes
        closed_trades["exit_reason"] = closed_trades["notes"].str.extract(r'(EXIT_V2_DRIFT_\w+)', expand=False)
    
    # Compute summary stats
    summary = compute_summary_stats(closed_trades, period_days)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"✅ Results saved to {output_path}")
    print()
    
    # Print summary
    print("=" * 80)
    print("REPLAY SUMMARY")
    print("=" * 80)
    print()
    print(f"n_trades: {summary['n_trades']:,}")
    print(f"EV/trade (mean_pnl_bps): {summary['mean_pnl_bps']:.2f} bps")
    print(f"trades_per_day: {summary['trades_per_day']:.2f}")
    print(f"EV/day: {summary['ev_per_day']:.2f} bps")
    print()
    print("Exit Rates:")
    print(f"  TP2-rate: {summary['tp2_rate']:.4f} ({summary['tp2_rate']*100:.2f}%)")
    print(f"  SL-rate: {summary['sl_rate']:.4f} ({summary['sl_rate']*100:.2f}%)")
    print(f"  TIMEOUT-rate: {summary['timeout_rate']:.4f} ({summary['timeout_rate']*100:.2f}%)")
    print(f"  SL_BREAKEVEN-rate: {summary['sl_breakeven_rate']:.4f} ({summary['sl_breakeven_rate']*100:.2f}%)")
    print()
    print(f"median_bars_held: {summary['median_bars_held']:.1f}")
    print()
    
    if summary['session_breakdown']:
        print("Session Breakdown:")
        for session, stats in summary['session_breakdown'].items():
            print(f"  {session}: {stats['n_trades']:,} trades, mean_pnl={stats['mean_pnl_bps']:.2f} bps")
        print()
    
    if summary['volatility_breakdown']:
        print("Volatility Breakdown:")
        for regime, stats in summary['volatility_breakdown'].items():
            print(f"  {regime}: {stats['n_trades']:,} trades, mean_pnl={stats['mean_pnl_bps']:.2f} bps")
        print()
    
    return 0


if __name__ == "__main__":
    exit(main())

