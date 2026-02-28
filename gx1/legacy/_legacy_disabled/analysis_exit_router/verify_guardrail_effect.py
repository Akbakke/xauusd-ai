#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify guardrail effect with high precision.

DEL 1: Calculate EV/trade and winrate directly from trade list (high precision)
DEL 2: Trade-diff on overridden trades (RULE6A → RULE5)
DEL 3: Check for caching / incorrect use of baseline-summary
DEL 4: Clean report text (inconsistencies)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import pandas as pd
import numpy as np


def extract_range_edge_dist_atr(extra_str: str) -> Optional[float]:
    """Extract range_edge_dist_atr from extra JSON string."""
    if pd.isna(extra_str) or not extra_str:
        return None
    try:
        if isinstance(extra_str, str):
            extra = json.loads(extra_str)
        else:
            extra = extra_str
        reda = extra.get("range_edge_dist_atr")
        if reda is not None:
            return float(reda)
    except (json.JSONDecodeError, (TypeError, ValueError)):
        pass
    return None


def get_trade_key(row: pd.Series) -> str:
    """Generate stable trade key for matching."""
    # Use entry_time + entry_price as primary key (trade_id varies between runs)
    entry_time = str(row.get("entry_time", ""))
    entry_price = str(row.get("entry_price", ""))
    
    # Also include side/direction if available for extra uniqueness
    side = str(row.get("side", ""))
    direction = str(row.get("direction", ""))
    
    return f"{entry_time}_{entry_price}_{side}_{direction}"


def calculate_metrics(df: pd.DataFrame, label: str) -> Dict[str, Any]:
    """
    DEL 1: Calculate EV/trade and winrate directly from trade list with high precision.
    
    Returns:
        dict with: n_trades, sum_pnl, mean_pnl, winrate, min_pnl, max_pnl, median_pnl
    """
    pnl = pd.to_numeric(df["pnl_bps"], errors='coerce').dropna()
    
    if len(pnl) == 0:
        return {
            "n_trades": 0,
            "sum_pnl": 0.0,
            "mean_pnl": 0.0,
            "winrate": 0.0,
            "min_pnl": 0.0,
            "max_pnl": 0.0,
            "median_pnl": 0.0,
        }
    
    n_trades = len(pnl)
    sum_pnl = float(pnl.sum())
    mean_pnl = float(pnl.mean())
    winrate = float((pnl > 0).mean())
    min_pnl = float(pnl.min())
    max_pnl = float(pnl.max())
    median_pnl = float(pnl.median())
    
    return {
        "n_trades": n_trades,
        "sum_pnl": sum_pnl,
        "mean_pnl": mean_pnl,
        "winrate": winrate,
        "min_pnl": min_pnl,
        "max_pnl": max_pnl,
        "median_pnl": median_pnl,
    }


def find_overridden_trades(
    df_baseline: pd.DataFrame,
    df_guardrail: pd.DataFrame,
) -> pd.DataFrame:
    """
    DEL 2: Find trades that were overridden by guardrail (RULE6A → RULE5).
    
    Returns DataFrame with matched trades and their differences.
    """
    # Filter to RULE6A in baseline
    baseline_rule6a = df_baseline[
        df_baseline["exit_profile"] == "FARM_EXIT_V2_RULES_ADAPTIVE_v1"
    ].copy()
    
    # Filter to RULE5 in guardrail
    guardrail_rule5 = df_guardrail[
        df_guardrail["exit_profile"] == "FARM_EXIT_V2_RULES_A"
    ].copy()
    
    # Generate trade keys
    baseline_rule6a["trade_key"] = baseline_rule6a.apply(get_trade_key, axis=1)
    guardrail_rule5["trade_key"] = guardrail_rule5.apply(get_trade_key, axis=1)
    
    # Match trades
    baseline_keys = set(baseline_rule6a["trade_key"].values)
    guardrail_keys = set(guardrail_rule5["trade_key"].values)
    matched_keys = baseline_keys & guardrail_keys
    
    if len(matched_keys) == 0:
        return pd.DataFrame()
    
    # Build comparison DataFrame
    results = []
    for key in matched_keys:
        baseline_row = baseline_rule6a[baseline_rule6a["trade_key"] == key].iloc[0]
        guardrail_row = guardrail_rule5[guardrail_rule5["trade_key"] == key].iloc[0]
        
        baseline_pnl = float(pd.to_numeric(baseline_row["pnl_bps"], errors='coerce'))
        guardrail_pnl = float(pd.to_numeric(guardrail_row["pnl_bps"], errors='coerce'))
        delta_pnl = guardrail_pnl - baseline_pnl
        
        baseline_reda = extract_range_edge_dist_atr(baseline_row.get("extra"))
        guardrail_reda = extract_range_edge_dist_atr(guardrail_row.get("extra"))
        reda = baseline_reda if baseline_reda is not None else guardrail_reda
        
        results.append({
            "trade_key": key,
            "trade_id": baseline_row.get("trade_id", ""),
            "baseline_exit_profile": baseline_row["exit_profile"],
            "guardrail_exit_profile": guardrail_row["exit_profile"],
            "baseline_exit_time": baseline_row.get("exit_time", ""),
            "guardrail_exit_time": guardrail_row.get("exit_time", ""),
            "baseline_bars_held": float(pd.to_numeric(baseline_row.get("bars_held"), errors='coerce')) if pd.notna(baseline_row.get("bars_held")) else None,
            "guardrail_bars_held": float(pd.to_numeric(guardrail_row.get("bars_held"), errors='coerce')) if pd.notna(guardrail_row.get("bars_held")) else None,
            "baseline_pnl_bps": baseline_pnl,
            "guardrail_pnl_bps": guardrail_pnl,
            "delta_pnl_bps": delta_pnl,
            "range_edge_dist_atr": reda,
        })
    
    return pd.DataFrame(results)


def verify_no_caching(
    baseline_metrics: Dict[str, Any],
    guardrail_metrics: Dict[str, Any],
) -> None:
    """
    DEL 3: Check for caching / incorrect use of baseline-summary.
    
    Assert that if exit_profile distribution changes but PnL is identical,
    we print a debug warning.
    """
    # Check if sum and mean are identical to high precision
    sum_diff = abs(baseline_metrics["sum_pnl"] - guardrail_metrics["sum_pnl"])
    mean_diff = abs(baseline_metrics["mean_pnl"] - guardrail_metrics["mean_pnl"])
    
    if sum_diff < 1e-9 and mean_diff < 1e-9:
        print()
        print("⚠️  DEBUG WARNING: Sum and mean PnL are identical to high precision!")
        print(f"   Baseline sum: {baseline_metrics['sum_pnl']:.15f}")
        print(f"   Guardrail sum: {guardrail_metrics['sum_pnl']:.15f}")
        print(f"   Difference: {sum_diff:.15e}")
        print(f"   Baseline mean: {baseline_metrics['mean_pnl']:.15f}")
        print(f"   Guardrail mean: {guardrail_metrics['mean_pnl']:.15f}")
        print(f"   Difference: {mean_diff:.15e}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Verify guardrail effect with high precision"
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Path to baseline trade log CSV",
    )
    parser.add_argument(
        "--guardrail",
        type=Path,
        required=True,
        help="Path to guardrail trade log CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("gx1/analysis/guardrail_verification"),
        help="Output directory for CSV and report",
    )
    
    args = parser.parse_args()
    
    # Load trade logs
    print("Loading trade logs...")
    df_baseline = pd.read_csv(args.baseline)
    df_guardrail = pd.read_csv(args.guardrail)
    
    print(f"Baseline trades: {len(df_baseline)}")
    print(f"Guardrail trades: {len(df_guardrail)}")
    print()
    
    # DEL 1: Calculate metrics with high precision
    print("=" * 100)
    print("DEL 1: HIGH PRECISION METRICS")
    print("=" * 100)
    print()
    
    baseline_metrics = calculate_metrics(df_baseline, "baseline")
    guardrail_metrics = calculate_metrics(df_guardrail, "guardrail")
    
    print("BASELINE:")
    print(f"  N: {baseline_metrics['n_trades']}")
    print(f"  sum(pnl_bps): {baseline_metrics['sum_pnl']:.6f}")
    print(f"  mean(pnl_bps): {baseline_metrics['mean_pnl']:.6f}")
    print(f"  winrate: {baseline_metrics['winrate']:.6f}")
    print(f"  min: {baseline_metrics['min_pnl']:.6f}")
    print(f"  max: {baseline_metrics['max_pnl']:.6f}")
    print(f"  median: {baseline_metrics['median_pnl']:.6f}")
    print()
    
    print("GUARDRAIL:")
    print(f"  N: {guardrail_metrics['n_trades']}")
    print(f"  sum(pnl_bps): {guardrail_metrics['sum_pnl']:.6f}")
    print(f"  mean(pnl_bps): {guardrail_metrics['mean_pnl']:.6f}")
    print(f"  winrate: {guardrail_metrics['winrate']:.6f}")
    print(f"  min: {guardrail_metrics['min_pnl']:.6f}")
    print(f"  max: {guardrail_metrics['max_pnl']:.6f}")
    print(f"  median: {guardrail_metrics['median_pnl']:.6f}")
    print()
    
    # Differences (high precision, not rounded)
    sum_diff = guardrail_metrics["sum_pnl"] - baseline_metrics["sum_pnl"]
    mean_diff = guardrail_metrics["mean_pnl"] - baseline_metrics["mean_pnl"]
    winrate_diff = guardrail_metrics["winrate"] - baseline_metrics["winrate"]
    
    print("DIFFERENCES (baseline → guardrail):")
    print(f"  sum(pnl_bps): {sum_diff:.6f}")
    print(f"  mean(pnl_bps): {mean_diff:.6f}")
    print(f"  winrate: {winrate_diff:.6f}")
    print()
    
    # DEL 3: Verify no caching
    verify_no_caching(baseline_metrics, guardrail_metrics)
    
    # DEL 2: Find overridden trades
    print("=" * 100)
    print("DEL 2: OVERRIDDEN TRADES ANALYSIS")
    print("=" * 100)
    print()
    
    overridden_df = find_overridden_trades(df_baseline, df_guardrail)
    
    if len(overridden_df) == 0:
        print("⚠️  No overridden trades found (no RULE6A → RULE5 conversions)")
        print()
    else:
        print(f"Found {len(overridden_df)} overridden trades (RULE6A → RULE5)")
        print()
        
        # Aggregates
        delta_pnl = overridden_df["delta_pnl_bps"]
        print("AGGREGATES FOR OVERRIDDEN TRADES:")
        print(f"  Count: {len(overridden_df)}")
        print(f"  mean(delta_pnl_bps): {delta_pnl.mean():.6f}")
        print(f"  median(delta_pnl_bps): {delta_pnl.median():.6f}")
        print(f"  min(delta_pnl_bps): {delta_pnl.min():.6f}")
        print(f"  max(delta_pnl_bps): {delta_pnl.max():.6f}")
        print(f"  delta == 0 (exact): {(delta_pnl == 0.0).sum()}")
        print(f"  delta == 0 (within 1e-6): {(abs(delta_pnl) < 1e-6).sum()}")
        print()
        
        # Check if exit actually changed
        bars_changed = (overridden_df["baseline_bars_held"] != overridden_df["guardrail_bars_held"]).sum()
        exit_time_changed = (overridden_df["baseline_exit_time"] != overridden_df["guardrail_exit_time"]).sum()
        
        print("EXIT CHANGES:")
        print(f"  bars_held changed: {bars_changed}/{len(overridden_df)}")
        print(f"  exit_time changed: {exit_time_changed}/{len(overridden_df)}")
        print()
        
        if (abs(delta_pnl) < 1e-6).sum() > len(overridden_df) * 0.8:
            print("💡 INSIGHT:")
            print("   Most overridden trades have delta_pnl ≈ 0.")
            print("   This suggests RULE5 and RULE6A gave the same exit in practice,")
            print("   or the exit logic didn't actually change for these cases.")
            print()
        
        # Save CSV
        args.output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = args.output_dir / "overridden_trades.csv"
        overridden_df.to_csv(csv_path, index=False)
        print(f"✅ Saved overridden trades to: {csv_path}")
        print()
    
    # DEL 4: Clean report with consistency checks
    print("=" * 100)
    print("DEL 4: CONSISTENCY VERIFICATION")
    print("=" * 100)
    print()
    
    # Count exit profiles
    baseline_rule5 = (df_baseline["exit_profile"] == "FARM_EXIT_V2_RULES_A").sum()
    baseline_rule6a = (df_baseline["exit_profile"] == "FARM_EXIT_V2_RULES_ADAPTIVE_v1").sum()
    guardrail_rule5 = (df_guardrail["exit_profile"] == "FARM_EXIT_V2_RULES_A").sum()
    guardrail_rule6a = (df_guardrail["exit_profile"] == "FARM_EXIT_V2_RULES_ADAPTIVE_v1").sum()
    
    print("EXIT PROFILE COUNTS:")
    print(f"  Baseline:")
    print(f"    RULE5: {baseline_rule5}")
    print(f"    RULE6A: {baseline_rule6a}")
    print(f"    Total: {baseline_rule5 + baseline_rule6a} (expected: {len(df_baseline)})")
    print(f"  Guardrail:")
    print(f"    RULE5: {guardrail_rule5}")
    print(f"    RULE6A: {guardrail_rule6a}")
    print(f"    Total: {guardrail_rule5 + guardrail_rule6a} (expected: {len(df_guardrail)})")
    print()
    
    # Verify counts sum correctly
    baseline_total_check = baseline_rule5 + baseline_rule6a == len(df_baseline)
    guardrail_total_check = guardrail_rule5 + guardrail_rule6a == len(df_guardrail)
    
    if not baseline_total_check:
        print(f"⚠️  Baseline counts don't sum: {baseline_rule5 + baseline_rule6a} != {len(df_baseline)}")
    if not guardrail_total_check:
        print(f"⚠️  Guardrail counts don't sum: {guardrail_rule5 + guardrail_rule6a} != {len(df_guardrail)}")
    
    if baseline_total_check and guardrail_total_check:
        print("✅ All counts sum correctly")
        print()
    
    # Overridden trades count check
    if len(overridden_df) > 0:
        blocked_count = len(overridden_df)
        remaining_rule6a = guardrail_rule6a
        baseline_rule6a_check = blocked_count + remaining_rule6a == baseline_rule6a
        
        print("OVERRIDDEN TRADES COUNT CHECK:")
        print(f"  Blocked (RULE6A → RULE5): {blocked_count}")
        print(f"  Remaining RULE6A: {remaining_rule6a}")
        print(f"  Baseline RULE6A: {baseline_rule6a}")
        print(f"  Check: {blocked_count} + {remaining_rule6a} = {baseline_rule6a} {'✅' if baseline_rule6a_check else '⚠️'}")
        print()
    
    # Generate markdown summary
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "verification_summary.md"
    
    with open(summary_path, 'w') as f:
        f.write("# Guardrail Verification Summary\n\n")
        f.write("## DEL 1: High Precision Metrics\n\n")
        f.write("### Baseline\n")
        f.write(f"- N: {baseline_metrics['n_trades']}\n")
        f.write(f"- sum(pnl_bps): {baseline_metrics['sum_pnl']:.6f}\n")
        f.write(f"- mean(pnl_bps): {baseline_metrics['mean_pnl']:.6f}\n")
        f.write(f"- winrate: {baseline_metrics['winrate']:.6f}\n\n")
        f.write("### Guardrail\n")
        f.write(f"- N: {guardrail_metrics['n_trades']}\n")
        f.write(f"- sum(pnl_bps): {guardrail_metrics['sum_pnl']:.6f}\n")
        f.write(f"- mean(pnl_bps): {guardrail_metrics['mean_pnl']:.6f}\n")
        f.write(f"- winrate: {guardrail_metrics['winrate']:.6f}\n\n")
        f.write("### Differences\n")
        f.write(f"- sum(pnl_bps): {sum_diff:.6f}\n")
        f.write(f"- mean(pnl_bps): {mean_diff:.6f}\n")
        f.write(f"- winrate: {winrate_diff:.6f}\n\n")
        
        if len(overridden_df) > 0:
            f.write("## DEL 2: Overridden Trades\n\n")
            f.write(f"- Count: {len(overridden_df)}\n")
            f.write(f"- mean(delta_pnl_bps): {delta_pnl.mean():.6f}\n")
            f.write(f"- median(delta_pnl_bps): {delta_pnl.median():.6f}\n")
            f.write(f"- delta == 0 (exact): {(delta_pnl == 0.0).sum()}\n")
            f.write(f"- delta == 0 (within 1e-6): {(abs(delta_pnl) < 1e-6).sum()}\n\n")
    
    print(f"✅ Saved markdown summary to: {summary_path}")
    print()
    print("=" * 100)
    print("✅ VERIFICATION COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()

