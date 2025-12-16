#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare multiple FARM exit-router runs side by side.

Loads trade logs from multiple wf_run directories and computes key metrics
for comparison (EV per trade, EV per day, win rate, RULE6A share, etc.).
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta

import pandas as pd
import numpy as np


def map_exit_profile_to_policy(exit_profile: str) -> str:
    """
    Map exit profile name to policy label.
    
    Returns "RULE5", "RULE6A", or "OTHER".
    """
    if not exit_profile or pd.isna(exit_profile):
        return "OTHER"
    if not isinstance(exit_profile, str):
        return "OTHER"
    
    ep_upper = exit_profile.upper()
    if "RULES_ADAPTIVE" in ep_upper or "RULE6A" in ep_upper:
        return "RULE6A"
    if "RULES_A" in ep_upper or "RULE5" in ep_upper:
        return "RULE5"
    return "OTHER"


def load_trades_from_run(run_dir: Path) -> pd.DataFrame:
    """
    Load trade logs from a run directory.
    
    Looks for:
    1. Merged CSV files (trade_log*.csv or trades_merged*.csv)
    2. Falls back to JSON files in trades/ subdirectory
    
    Returns a DataFrame with trade data.
    """
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    
    # First, look for merged CSV files
    csv_files = list(run_dir.rglob("*.csv"))
    merged_csv = None
    
    # Prefer files with "trade" and "merged" in name
    for f in csv_files:
        name_lower = f.name.lower()
        if "trade" in name_lower and "merged" in name_lower:
            merged_csv = f
            break
    
    # If no merged CSV, look for any CSV with "trade" in name
    if merged_csv is None:
        for f in csv_files:
            name_lower = f.name.lower()
            if "trade" in name_lower:
                merged_csv = f
                break
    
    # If found, read CSV
    if merged_csv:
        try:
            df = pd.read_csv(merged_csv)
            return df
        except Exception as e:
            print(f"⚠️  Warning: Could not read CSV {merged_csv}: {e}")
    
    # Fall back to JSON files in trades/ subdirectory
    trades_dir = run_dir / "trades"
    if trades_dir.exists():
        json_files = list(trades_dir.glob("*.json"))
        if json_files:
            try:
                trades = []
                for json_file in json_files:
                    with open(json_file, 'r') as f:
                        trade_data = json.load(f)
                        trades.append(trade_data)
                
                # Convert to DataFrame
                df = pd.DataFrame(trades)
                return df
            except Exception as e:
                print(f"⚠️  Warning: Could not read JSON files from {trades_dir}: {e}")
    
    # Also check for chunk CSV files in parallel_chunks/
    chunks_dir = run_dir / "parallel_chunks"
    if chunks_dir.exists():
        chunk_csv_files = list(chunks_dir.glob("trade_log_chunk_*.csv"))
        if chunk_csv_files:
            try:
                dfs = []
                for chunk_file in sorted(chunk_csv_files):
                    df_chunk = pd.read_csv(chunk_file)
                    dfs.append(df_chunk)
                df = pd.concat(dfs, ignore_index=True)
                return df
            except Exception as e:
                print(f"⚠️  Warning: Could not read chunk CSV files: {e}")
    
    raise FileNotFoundError(f"No trade logs found in {run_dir}")


def compute_metrics_for_run(tag: str, run_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Compute metrics for a single run.
    
    Returns a dictionary with metrics, or None if the run is invalid.
    """
    try:
        df = load_trades_from_run(run_dir)
    except FileNotFoundError as e:
        print(f"⚠️  Warning: Skipping {tag} ({run_dir}): {e}")
        return None
    except Exception as e:
        print(f"⚠️  Warning: Skipping {tag} ({run_dir}): {e}")
        return None
    
    # Filter to closed trades only (if status column exists)
    if "status" in df.columns:
        df = df[df["status"] == "CLOSED"].copy()
    
    if len(df) == 0:
        print(f"⚠️  Warning: No trades found in {tag}")
        return None
    
    if len(df) == 0:
        print(f"⚠️  Warning: No closed trades found in {tag}")
        return None
    
    # Extract PnL
    pnl_col = None
    for col in ["pnl_bps", "metrics.pnl_bps"]:
        if col in df.columns:
            pnl_col = col
            break
    
    if pnl_col is None:
        # Try to extract from nested structure
        if "metrics" in df.columns:
            df["pnl_bps"] = df["metrics"].apply(
                lambda x: x.get("pnl_bps") if isinstance(x, dict) else None
            )
            pnl_col = "pnl_bps"
        else:
            print(f"⚠️  Warning: No PnL column found in {tag}")
            return None
    
    pnl = pd.to_numeric(df[pnl_col], errors='coerce').dropna()
    
    if len(pnl) == 0:
        print(f"⚠️  Warning: No valid PnL values in {tag}")
        return None
    
    # Compute basic metrics
    n_trades = len(pnl)
    wins = (pnl > 0).sum()
    win_rate_pct = (wins / n_trades) * 100.0
    total_pnl_bps = float(pnl.sum())
    ev_per_trade_bps = float(pnl.mean())
    
    # Compute EV per day
    ev_per_day_bps = None
    
    # Try to compute from exit_time (preferred: actual trade exit dates)
    exit_time_col = None
    for col in ["exit_time", "exit.timestamp", "exit"]:
        if col in df.columns:
            exit_time_col = col
            break
    
    if exit_time_col:
        try:
            exit_times = pd.to_datetime(df[exit_time_col], errors='coerce', utc=True)
            exit_times = exit_times.dropna()
            if len(exit_times) > 0:
                # Get distinct dates
                exit_dates = exit_times.dt.date
                n_days = exit_dates.nunique()
                # Only use if we have multiple distinct days (single day might be replay execution date)
                if n_days > 1:
                    ev_per_day_bps = total_pnl_bps / n_days
                elif n_days == 1:
                    # Single day might be replay execution date, try entry_time as fallback
                    entry_time_col = None
                    for col in ["entry_time", "entry.timestamp", "entry"]:
                        if col in df.columns:
                            entry_time_col = col
                            break
                    if entry_time_col:
                        try:
                            entry_times = pd.to_datetime(df[entry_time_col], errors='coerce', utc=True)
                            entry_times = entry_times.dropna()
                            if len(entry_times) > 0:
                                entry_dates = entry_times.dt.date
                                n_entry_days = entry_dates.nunique()
                                if n_entry_days > 1:
                                    # Use entry date range as proxy for trading period
                                    ev_per_day_bps = total_pnl_bps / n_entry_days
                        except Exception:
                            pass
        except Exception:
            pass
    
    # If EV per day not computed, try results.json
    if ev_per_day_bps is None:
        results_json = run_dir / "results.json"
        if results_json.exists():
            try:
                with open(results_json, 'r') as f:
                    results = json.load(f)
                    # Look for EV/day in various possible locations
                    if "ev_per_day_bps" in results:
                        ev_per_day_bps = float(results["ev_per_day_bps"])
                    elif "metrics" in results and isinstance(results["metrics"], dict):
                        if "ev_per_day_bps" in results["metrics"]:
                            ev_per_day_bps = float(results["metrics"]["ev_per_day_bps"])
            except Exception:
                pass
    
    # Compute RULE6A share
    exit_profile_col = None
    for col in ["exit_profile", "exit_profile_name"]:
        if col in df.columns:
            exit_profile_col = col
            break
    
    rule6a_share_pct = 0.0
    if exit_profile_col:
        policies = df[exit_profile_col].apply(map_exit_profile_to_policy)
        n_rule6a = (policies == "RULE6A").sum()
        rule6a_share_pct = (n_rule6a / n_trades) * 100.0
    else:
        # Try to infer from other columns
        if "policy" in df.columns:
            policies = df["policy"].apply(lambda x: "RULE6A" if "RULE6A" in str(x).upper() or "ADAPTIVE" in str(x).upper() else "RULE5")
            n_rule6a = (policies == "RULE6A").sum()
            rule6a_share_pct = (n_rule6a / n_trades) * 100.0
    
    return {
        "tag": tag,
        "run_dir": str(run_dir),
        "n_trades": n_trades,
        "win_rate_pct": round(win_rate_pct, 2),
        "total_pnl_bps": round(total_pnl_bps, 2),
        "ev_per_trade_bps": round(ev_per_trade_bps, 2),
        "ev_per_day_bps": round(ev_per_day_bps, 2) if ev_per_day_bps is not None else None,
        "rule6a_share_pct": round(rule6a_share_pct, 2),
    }


def print_comparison_table(runs: list[Dict[str, Any]]) -> None:
    """Print a markdown-style comparison table."""
    if not runs:
        print("No runs to compare.")
        return
    
    # Sort by ev_per_day_bps (descending) if available, otherwise by ev_per_trade_bps
    def sort_key(r):
        ev_day = r.get("ev_per_day_bps")
        if ev_day is not None:
            return -ev_day  # Negative for descending
        return -r.get("ev_per_trade_bps", 0)
    
    runs_sorted = sorted(runs, key=sort_key)
    
    # Print header
    print("\n" + "=" * 100)
    print("EXIT ROUTER COMPARISON - FULLYEAR 2025")
    print("=" * 100)
    print()
    
    # Table header
    header = ["Tag", "Trades", "Win rate (%)", "EV/trade (bps)", "EV/day (bps)", "RULE6A (%)", "Total PnL (bps)"]
    
    # Calculate column widths based on header and data
    col_widths = []
    for i, h in enumerate(header):
        max_width = len(h)
        for run in runs_sorted:
            if i == 0:  # Tag
                val = str(run.get("tag", ""))
            elif i == 1:  # Trades
                val = str(run.get("n_trades", 0))
            elif i == 2:  # Win rate
                val = f"{run.get('win_rate_pct', 0.0):.1f}"
            elif i == 3:  # EV/trade
                val = f"{run.get('ev_per_trade_bps', 0.0):.1f}"
            elif i == 4:  # EV/day
                ev_day = run.get("ev_per_day_bps")
                val = f"{ev_day:.1f}" if ev_day is not None else "N/A"
            elif i == 5:  # RULE6A
                val = f"{run.get('rule6a_share_pct', 0.0):.1f}"
            else:  # Total PnL
                val = f"{run.get('total_pnl_bps', 0.0):.1f}"
            max_width = max(max_width, len(val))
        col_widths.append(max_width + 2)
    
    # Print header row
    header_row = " | ".join(h.ljust(w) for h, w in zip(header, col_widths))
    print(header_row)
    print("-" * len(header_row))
    
    # Print data rows
    for run in runs_sorted:
        tag = str(run.get("tag", ""))
        n_trades = run.get("n_trades", 0)
        win_rate = run.get("win_rate_pct", 0.0)
        ev_trade = run.get("ev_per_trade_bps", 0.0)
        ev_day = run.get("ev_per_day_bps")
        ev_day_str = f"{ev_day:.1f}" if ev_day is not None else "N/A"
        rule6a = run.get("rule6a_share_pct", 0.0)
        total_pnl = run.get("total_pnl_bps", 0.0)
        
        row = [
            tag.ljust(col_widths[0]),
            str(n_trades).rjust(col_widths[1]),
            f"{win_rate:.1f}".rjust(col_widths[2]),
            f"{ev_trade:.1f}".rjust(col_widths[3]),
            ev_day_str.rjust(col_widths[4]),
            f"{rule6a:.1f}".rjust(col_widths[5]),
            f"{total_pnl:.1f}".rjust(col_widths[6]),
        ]
        print(" | ".join(row))
    
    print()
    print("=" * 100)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple FARM exit-router runs side by side"
    )
    parser.add_argument(
        "--run",
        action="append",
        nargs=2,
        metavar=("TAG", "PATH"),
        help="Add a run to compare. Format: --run TAG PATH (can be used multiple times)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("gx1/analysis/exit_router_models_v3/exit_router_comparison_fullyear.json"),
        help="Path to save JSON summary (default: gx1/analysis/exit_router_models_v3/exit_router_comparison_fullyear.json)",
    )
    parser.add_argument(
        "--regime-analysis",
        type=Path,
        metavar="PATH",
        help="Run regime analysis (DEL A & B) on a single run directory. Skips baseline comparison.",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=14,
        help="Time window size in days for DEL B analysis (default: 14 = 2 weeks)",
    )
    parser.add_argument(
        "--intratrade-risk",
        action="store_true",
        help="Run intratrade risk analysis for overridden trades (requires --baseline and --guardrail)",
    )
    parser.add_argument(
        "--baseline-run",
        type=Path,
        metavar="PATH",
        help="Baseline run directory for intratrade risk analysis",
    )
    parser.add_argument(
        "--guardrail-run",
        type=Path,
        metavar="PATH",
        help="Guardrail run directory for intratrade risk analysis",
    )
    parser.add_argument(
        "--overridden-trades-csv",
        type=Path,
        metavar="PATH",
        help="CSV file with overridden trades (optional, will be found automatically if not provided)",
    )
    
    args = parser.parse_args()
    
    # If intratrade risk analysis mode, run that instead
    if args.intratrade_risk:
        if not args.baseline_run or not args.guardrail_run:
            parser.error("--intratrade-risk requires --baseline-run and --guardrail-run")
        
        baseline_run = Path(args.baseline_run)
        guardrail_run = Path(args.guardrail_run)
        
        if not baseline_run.exists():
            print(f"❌ Baseline run directory does not exist: {baseline_run}")
            return
        if not guardrail_run.exists():
            print(f"❌ Guardrail run directory does not exist: {guardrail_run}")
            return
        
        analyze_intratrade_risk(
            baseline_run,
            guardrail_run,
            args.overridden_trades_csv,
        )
        return
    
    # If regime analysis mode, run that instead
    if args.regime_analysis:
        run_dir = Path(args.regime_analysis)
        if not run_dir.exists():
            print(f"❌ Run directory does not exist: {run_dir}")
            return
        
        print("=" * 100)
        print("V3_RANGE PROD_BASELINE - REGIME ANALYSIS")
        print("=" * 100)
        print(f"Run directory: {run_dir}")
        print(f"Time window size: {args.window_days} days")
        print()
        
        run_regime_analysis(run_dir, window_days=args.window_days)
        return
    
    # Normal comparison mode
    if not args.run:
        parser.error("Either --run or --regime-analysis must be provided")
    
    # Parse runs
    runs_data = []
    for tag, path_str in args.run:
        run_dir = Path(path_str)
        metrics = compute_metrics_for_run(tag, run_dir)
        if metrics:
            runs_data.append(metrics)
    
    if not runs_data:
        print("❌ No valid runs found. Exiting.")
        return
    
    # Print comparison table
    print_comparison_table(runs_data)
    
    # Save JSON summary
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "runs": runs_data
    }
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ JSON summary saved to: {output_path}")


# ============================================================================
# DEL A & B: Regime Analysis for V3_RANGE PROD_BASELINE
# ============================================================================

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


def compute_drawdown_clusters(pnl_series: pd.Series) -> Dict[str, Any]:
    """
    Compute drawdown clusters (max loss in series).
    
    Returns dict with:
    - max_drawdown: Maximum single loss
    - max_consecutive_losses: Longest losing streak
    - max_cluster_loss: Maximum cumulative loss in any cluster
    """
    if len(pnl_series) == 0:
        return {
            "max_drawdown": 0.0,
            "max_consecutive_losses": 0,
            "max_cluster_loss": 0.0,
        }
    
    losses = pnl_series[pnl_series < 0]
    if len(losses) == 0:
        return {
            "max_drawdown": 0.0,
            "max_consecutive_losses": 0,
            "max_cluster_loss": 0.0,
        }
    
    max_drawdown = float(losses.min())
    
    # Find consecutive losses
    is_loss = (pnl_series < 0).astype(int)
    consecutive = []
    current = 0
    for val in is_loss:
        if val == 1:
            current += 1
        else:
            if current > 0:
                consecutive.append(current)
            current = 0
    if current > 0:
        consecutive.append(current)
    
    max_consecutive_losses = max(consecutive) if consecutive else 0
    
    # Find max cluster loss (cumulative loss in any consecutive sequence)
    max_cluster_loss = 0.0
    current_cluster = 0.0
    for val in pnl_series:
        if val < 0:
            current_cluster += val
            max_cluster_loss = min(max_cluster_loss, current_cluster)
        else:
            current_cluster = 0.0
    
    return {
        "max_drawdown": max_drawdown,
        "max_consecutive_losses": max_consecutive_losses,
        "max_cluster_loss": max_cluster_loss,
    }


def analyze_range_edge_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """
    DEL A: Analyze trades by range_edge_dist_atr buckets.
    
    Buckets: <1.0, 1.0-2.0, 2.0-4.0, >4.0
    For each bucket, compute metrics for RULE5 and RULE6A separately.
    """
    # Extract range_edge_dist_atr
    if "extra" not in df.columns:
        print("⚠️  Warning: No 'extra' column found. Skipping range-edge analysis.")
        return pd.DataFrame()
    
    df = df.copy()
    df["range_edge_dist_atr"] = df["extra"].apply(extract_range_edge_dist_atr)
    
    # Filter to trades with valid range_edge_dist_atr
    df_valid = df[df["range_edge_dist_atr"].notna()].copy()
    
    if len(df_valid) == 0:
        print("⚠️  Warning: No trades with valid range_edge_dist_atr. Skipping range-edge analysis.")
        return pd.DataFrame()
    
    # Map exit profiles
    df_valid["policy"] = df_valid["exit_profile"].apply(map_exit_profile_to_policy)
    
    # Extract PnL
    pnl_col = None
    for col in ["pnl_bps", "metrics.pnl_bps"]:
        if col in df_valid.columns:
            pnl_col = col
            break
    
    if pnl_col is None:
        print("⚠️  Warning: No PnL column found. Skipping range-edge analysis.")
        return pd.DataFrame()
    
    df_valid["pnl_bps"] = pd.to_numeric(df_valid[pnl_col], errors='coerce')
    df_valid = df_valid[df_valid["pnl_bps"].notna()].copy()
    
    # Define buckets
    def assign_bucket(reda: float) -> str:
        if reda < 1.0:
            return "<1.0"
        elif reda < 2.0:
            return "1.0-2.0"
        elif reda < 4.0:
            return "2.0-4.0"
        else:
            return ">4.0"
    
    df_valid["bucket"] = df_valid["range_edge_dist_atr"].apply(assign_bucket)
    
    # Compute metrics per bucket and policy
    results = []
    buckets = ["<1.0", "1.0-2.0", "2.0-4.0", ">4.0"]
    
    for bucket in buckets:
        df_bucket = df_valid[df_valid["bucket"] == bucket]
        if len(df_bucket) == 0:
            continue
        
        for policy in ["RULE5", "RULE6A"]:
            df_policy = df_bucket[df_bucket["policy"] == policy]
            if len(df_policy) == 0:
                continue
            
            pnl = df_policy["pnl_bps"]
            n_trades = len(pnl)
            wins = (pnl > 0).sum()
            win_rate = (wins / n_trades) * 100.0 if n_trades > 0 else 0.0
            ev_trade = float(pnl.mean())
            
            # Median bars held
            bars_held = None
            if "bars_held" in df_policy.columns:
                bars_held_vals = pd.to_numeric(df_policy["bars_held"], errors='coerce').dropna()
                if len(bars_held_vals) > 0:
                    bars_held = float(bars_held_vals.median())
            
            # Drawdown clusters
            dd = compute_drawdown_clusters(pnl)
            
            results.append({
                "bucket": bucket,
                "policy": policy,
                "n_trades": n_trades,
                "ev_trade_bps": round(ev_trade, 2),
                "win_rate_pct": round(win_rate, 2),
                "median_bars_held": round(bars_held, 0) if bars_held is not None else None,
                "max_drawdown_bps": round(dd["max_drawdown"], 2),
                "max_consecutive_losses": dd["max_consecutive_losses"],
                "max_cluster_loss_bps": round(dd["max_cluster_loss"], 2),
            })
    
    return pd.DataFrame(results)


def print_range_edge_analysis(results_df: pd.DataFrame) -> None:
    """Print DEL A: Range-edge bucket analysis."""
    if len(results_df) == 0:
        print("\n" + "=" * 100)
        print("DEL A: RANGE-EDGE BUCKET ANALYSIS")
        print("=" * 100)
        print("⚠️  No data available for range-edge analysis.")
        return
    
    print("\n" + "=" * 100)
    print("DEL A: RANGE-EDGE BUCKET ANALYSIS")
    print("=" * 100)
    print()
    print("Segmentering basert på range_edge_dist_atr:")
    print("  <1.0:   Very near edge")
    print("  1.0-2.0: Near edge")
    print("  2.0-4.0: Mid-range")
    print("  >4.0:   Far from edge")
    print()
    
    # Create comparison table
    buckets = ["<1.0", "1.0-2.0", "2.0-4.0", ">4.0"]
    policies = ["RULE5", "RULE6A"]
    
    # Header
    header = ["Bucket", "Policy", "Trades", "EV/trade (bps)", "Win rate (%)", "Median bars", "Max DD (bps)", "Max cluster loss (bps)"]
    col_widths = [10, 8, 8, 15, 12, 12, 15, 20]
    
    header_row = " | ".join(h.ljust(w) for h, w in zip(header, col_widths))
    print(header_row)
    print("-" * len(header_row))
    
    # Data rows
    for bucket in buckets:
        for policy in policies:
            row_data = results_df[(results_df["bucket"] == bucket) & (results_df["policy"] == policy)]
            if len(row_data) == 0:
                continue
            
            r = row_data.iloc[0]
            row = [
                bucket.ljust(col_widths[0]),
                policy.ljust(col_widths[1]),
                str(r["n_trades"]).rjust(col_widths[2]),
                f"{r['ev_trade_bps']:.2f}".rjust(col_widths[3]),
                f"{r['win_rate_pct']:.2f}".rjust(col_widths[4]),
                f"{r['median_bars_held']:.0f}".rjust(col_widths[5]) if r["median_bars_held"] is not None else "N/A".rjust(col_widths[5]),
                f"{r['max_drawdown_bps']:.2f}".rjust(col_widths[6]),
                f"{r['max_cluster_loss_bps']:.2f}".rjust(col_widths[7]),
            ]
            print(" | ".join(row))
    
    print()
    
    # Identify problematic buckets
    print("🔍 PROBLEMATIC BUCKETS (RULE6A underperforms RULE5):")
    print()
    
    problematic = []
    for bucket in buckets:
        rule5_data = results_df[(results_df["bucket"] == bucket) & (results_df["policy"] == "RULE5")]
        rule6a_data = results_df[(results_df["bucket"] == bucket) & (results_df["policy"] == "RULE6A")]
        
        if len(rule5_data) == 0 or len(rule6a_data) == 0:
            continue
        
        r5 = rule5_data.iloc[0]
        r6a = rule6a_data.iloc[0]
        
        issues = []
        if r6a["ev_trade_bps"] < r5["ev_trade_bps"]:
            issues.append(f"Lower EV/trade ({r6a['ev_trade_bps']:.2f} vs {r5['ev_trade_bps']:.2f} bps)")
        if r6a["max_drawdown_bps"] < r5["max_drawdown_bps"]:
            issues.append(f"Worse max drawdown ({r6a['max_drawdown_bps']:.2f} vs {r5['max_drawdown_bps']:.2f} bps)")
        if r6a["max_cluster_loss_bps"] < r5["max_cluster_loss_bps"]:
            issues.append(f"Worse cluster loss ({r6a['max_cluster_loss_bps']:.2f} vs {r5['max_cluster_loss_bps']:.2f} bps)")
        
        if issues:
            problematic.append((bucket, issues))
    
    if problematic:
        for bucket, issues in problematic:
            print(f"  ❌ {bucket}:")
            for issue in issues:
                print(f"     - {issue}")
    else:
        print("  ✅ No problematic buckets identified.")
    
    print()
    print("=" * 100)
    print()
    
    # Conclusion
    print("📝 KONKLUSJON DEL A:")
    print()
    if problematic:
        print("  RULE6A underperformer RULE5 i følgende range-edge buckets:")
        for bucket, _ in problematic:
            print(f"    - {bucket}")
        print()
        print("  Dette indikerer at adaptiv exit (RULE6A) ikke lønner seg i disse regime.")
        print("  Vurder å redusere RULE6A-allokering i disse buckets i fremtidige iterasjoner.")
    else:
        print("  RULE6A presterer minst like bra som RULE5 i alle range-edge buckets.")
        print("  Ingen tydelige problemer identifisert.")
    print()


def analyze_time_windows(df: pd.DataFrame, window_days: int = 14) -> pd.DataFrame:
    """
    DEL B: Analyze trades in time windows.
    
    Splits FULLYEAR into fixed time windows (default: 14 days = 2 weeks).
    For each window, reports: trades/day, EV/day, EV/trade, win rate, RULE5 vs RULE6A share.
    """
    # Extract entry_time
    entry_time_col = None
    for col in ["entry_time", "entry.timestamp", "entry"]:
        if col in df.columns:
            entry_time_col = col
            break
    
    if entry_time_col is None:
        print("⚠️  Warning: No entry_time column found. Skipping time-window analysis.")
        return pd.DataFrame()
    
    df = df.copy()
    df["entry_time_parsed"] = pd.to_datetime(df[entry_time_col], errors='coerce', utc=True)
    df = df[df["entry_time_parsed"].notna()].copy()
    
    if len(df) == 0:
        print("⚠️  Warning: No valid entry times. Skipping time-window analysis.")
        return pd.DataFrame()
    
    # Extract PnL
    pnl_col = None
    for col in ["pnl_bps", "metrics.pnl_bps"]:
        if col in df.columns:
            pnl_col = col
            break
    
    if pnl_col is None:
        print("⚠️  Warning: No PnL column found. Skipping time-window analysis.")
        return pd.DataFrame()
    
    df["pnl_bps"] = pd.to_numeric(df[pnl_col], errors='coerce')
    df = df[df["pnl_bps"].notna()].copy()
    
    # Map exit profiles
    df["policy"] = df["exit_profile"].apply(map_exit_profile_to_policy)
    
    # Assign time windows
    min_time = df["entry_time_parsed"].min()
    max_time = df["entry_time_parsed"].max()
    
    # Create windows
    windows = []
    current_start = min_time
    window_num = 0
    
    while current_start < max_time:
        window_end = current_start + timedelta(days=window_days)
        window_num += 1
        
        df_window = df[(df["entry_time_parsed"] >= current_start) & (df["entry_time_parsed"] < window_end)]
        
        if len(df_window) > 0:
            # Compute metrics
            pnl = df_window["pnl_bps"]
            n_trades = len(pnl)
            wins = (pnl > 0).sum()
            win_rate = (wins / n_trades) * 100.0 if n_trades > 0 else 0.0
            ev_trade = float(pnl.mean())
            total_pnl = float(pnl.sum())
            
            # Days in window (actual trading days)
            window_dates = df_window["entry_time_parsed"].dt.date.unique()
            n_days = len(window_dates)
            trades_per_day = n_trades / n_days if n_days > 0 else 0.0
            ev_per_day = total_pnl / n_days if n_days > 0 else 0.0
            
            # Policy distribution
            policies = df_window["policy"]
            n_rule5 = (policies == "RULE5").sum()
            n_rule6a = (policies == "RULE6A").sum()
            rule5_pct = (n_rule5 / n_trades) * 100.0 if n_trades > 0 else 0.0
            rule6a_pct = (n_rule6a / n_trades) * 100.0 if n_trades > 0 else 0.0
            
            windows.append({
                "window_num": window_num,
                "start_date": current_start.date(),
                "end_date": (window_end - timedelta(days=1)).date(),
                "n_trades": n_trades,
                "n_days": n_days,
                "trades_per_day": round(trades_per_day, 2),
                "ev_per_day_bps": round(ev_per_day, 2),
                "ev_trade_bps": round(ev_trade, 2),
                "win_rate_pct": round(win_rate, 2),
                "rule5_pct": round(rule5_pct, 2),
                "rule6a_pct": round(rule6a_pct, 2),
            })
        
        current_start = window_end
    
    return pd.DataFrame(windows)


def print_time_window_analysis(results_df: pd.DataFrame, window_days: int = 14) -> None:
    """Print DEL B: Time-window analysis."""
    if len(results_df) == 0:
        print("\n" + "=" * 100)
        print("DEL B: TIME-WINDOW REGIME ANALYSIS")
        print("=" * 100)
        print("⚠️  No data available for time-window analysis.")
        return
    
    print("\n" + "=" * 100)
    print("DEL B: TIME-WINDOW REGIME ANALYSIS")
    print("=" * 100)
    print()
    print(f"Tidsvinduer: {window_days} dager per vindu")
    print()
    
    # Header
    header = ["Window", "Period", "Trades", "Days", "Trades/day", "EV/day (bps)", "EV/trade (bps)", "Win rate (%)", "RULE5 (%)", "RULE6A (%)"]
    col_widths = [8, 25, 8, 6, 12, 15, 15, 12, 10, 10]
    
    header_row = " | ".join(h.ljust(w) for h, w in zip(header, col_widths))
    print(header_row)
    print("-" * len(header_row))
    
    # Data rows
    for _, row in results_df.iterrows():
        period_str = f"{row['start_date']} to {row['end_date']}"
        row_data = [
            str(row["window_num"]).ljust(col_widths[0]),
            period_str[:col_widths[1]].ljust(col_widths[1]),
            str(row["n_trades"]).rjust(col_widths[2]),
            str(row["n_days"]).rjust(col_widths[3]),
            f"{row['trades_per_day']:.2f}".rjust(col_widths[4]),
            f"{row['ev_per_day_bps']:.2f}".rjust(col_widths[5]),
            f"{row['ev_trade_bps']:.2f}".rjust(col_widths[6]),
            f"{row['win_rate_pct']:.2f}".rjust(col_widths[7]),
            f"{row['rule5_pct']:.1f}".rjust(col_widths[8]),
            f"{row['rule6a_pct']:.1f}".rjust(col_widths[9]),
        ]
        print(" | ".join(row_data))
    
    print()
    
    # Identify conservative periods
    print("🔍 CONSERVATIVE PERIODS (Low trades/day, high EV/trade, but low EV/day):")
    print()
    
    # Compute thresholds (median-based)
    median_trades_per_day = results_df["trades_per_day"].median()
    median_ev_trade = results_df["ev_trade_bps"].median()
    median_ev_per_day = results_df["ev_per_day_bps"].median()
    
    conservative = []
    for _, row in results_df.iterrows():
        is_low_trades = row["trades_per_day"] < median_trades_per_day * 0.5
        is_high_ev_trade = row["ev_trade_bps"] > median_ev_trade * 1.2
        is_low_ev_per_day = row["ev_per_day_bps"] < median_ev_per_day * 0.7
        
        if is_low_trades and is_high_ev_trade and is_low_ev_per_day:
            conservative.append(row)
    
    if conservative:
        for row in conservative:
            print(f"  ⚠️  Window {row['window_num']} ({row['start_date']} to {row['end_date']}):")
            print(f"     - Trades/day: {row['trades_per_day']:.2f} (low)")
            print(f"     - EV/trade: {row['ev_trade_bps']:.2f} bps (high)")
            print(f"     - EV/day: {row['ev_per_day_bps']:.2f} bps (low)")
            print(f"     - RULE6A: {row['rule6a_pct']:.1f}%")
    else:
        print("  ✅ No conservative periods identified.")
    
    print()
    print("=" * 100)
    print()
    
    # Conclusion
    print("📝 KONKLUSJON DEL B:")
    print()
    if conservative:
        print(f"  Identifisert {len(conservative)} periode(r) med potensielt over-konservativ routing:")
        for row in conservative:
            print(f"    - Window {row['window_num']}: {row['start_date']} to {row['end_date']}")
        print()
        print("  Disse periodene har:")
        print("    - Lav trade-frekvens (mulig missed opportunities)")
        print("    - Høy EV/trade (kvalitet er god)")
        print("    - Lav EV/day (volum er problemet)")
        print()
        print("  Dette kan indikere at routeren er for konservativ i disse periodene.")
        print("  Vurder å øke RULE6A-allokering eller justere entry-kriterier i fremtidige iterasjoner.")
    else:
        print("  Ingen tydelige perioder med over-konservativ routing identifisert.")
        print("  Routeren balanserer trade-frekvens og EV/trade godt.")
    print()


def run_regime_analysis(run_dir: Path, window_days: int = 14) -> None:
    """
    Run DEL A and DEL B regime analysis on a single run.
    
    This function loads the trade log and performs:
    - DEL A: Range-edge bucket analysis
    - DEL B: Time-window analysis
    
    Args:
        run_dir: Path to run directory containing trade logs
        window_days: Size of time windows for DEL B analysis (default: 14 days)
    """
    try:
        df = load_trades_from_run(run_dir)
    except Exception as e:
        print(f"❌ Error loading trades from {run_dir}: {e}")
        return
    
    if len(df) == 0:
        print(f"❌ No trades found in {run_dir}")
        return
    
    # DEL A: Range-edge bucket analysis
    range_results = analyze_range_edge_buckets(df)
    if len(range_results) > 0:
        print_range_edge_analysis(range_results)
    
    # DEL B: Time-window analysis
    time_results = analyze_time_windows(df, window_days=window_days)
    if len(time_results) > 0:
        print_time_window_analysis(time_results, window_days=window_days)


# ============================================================================
# INTRATRADE RISK ANALYSIS
# ============================================================================

def calculate_intratrade_metrics(
    entry_time: pd.Timestamp,
    exit_time: pd.Timestamp,
    entry_price: float,
    price_data: pd.DataFrame,
    side: str = "long",
) -> Dict[str, float]:
    """
    DEL 1: Calculate MFE, MAE, and intratrade drawdown for a trade.
    
    Args:
        entry_time: Trade entry timestamp
        exit_time: Trade exit timestamp
        entry_price: Entry price
        price_data: DataFrame with columns: time (index), high, low, close
        side: "long" or "short"
    
    Returns:
        dict with: mfe_bps, mae_bps, intratrade_dd_bps
    """
    # Filter bars between entry and exit
    bars = price_data[(price_data.index >= entry_time) & (price_data.index <= exit_time)].copy()
    
    if len(bars) == 0:
        return {
            "mfe_bps": 0.0,
            "mae_bps": 0.0,
            "intratrade_dd_bps": 0.0,
        }
    
    # Use bid/ask prices based on side
    if side.lower() == "long":
        # Long: use bid prices (sell price)
        high_col = "bid_high" if "bid_high" in bars.columns else "high"
        low_col = "bid_low" if "bid_low" in bars.columns else "low"
        close_col = "bid_close" if "bid_close" in bars.columns else "close"
    else:
        # Short: use ask prices (buy price)
        high_col = "ask_high" if "ask_high" in bars.columns else "high"
        low_col = "ask_low" if "ask_low" in bars.columns else "low"
        close_col = "ask_close" if "ask_close" in bars.columns else "close"
    
    highs = bars[high_col].values
    lows = bars[low_col].values
    closes = bars[close_col].values
    
    # Calculate PnL in bps for each bar
    if side.lower() == "long":
        # Long: profit when price goes up
        bar_highs_pnl = ((highs - entry_price) / entry_price) * 10000.0  # bps
        bar_lows_pnl = ((lows - entry_price) / entry_price) * 10000.0
        bar_closes_pnl = ((closes - entry_price) / entry_price) * 10000.0
    else:
        # Short: profit when price goes down
        bar_highs_pnl = ((entry_price - highs) / entry_price) * 10000.0
        bar_lows_pnl = ((entry_price - lows) / entry_price) * 10000.0
        bar_closes_pnl = ((entry_price - closes) / entry_price) * 10000.0
    
    # MFE: Max Favorable Excursion (best unrealized PnL)
    mfe_bps = float(np.max(bar_highs_pnl))
    
    # MAE: Max Adverse Excursion (worst unrealized drawdown from entry)
    mae_bps = float(np.min(bar_lows_pnl))
    
    # Intratrade Drawdown: Largest peak-to-trough drawdown within trade
    # Track running max (peak) and calculate drawdown from peak
    running_max = np.maximum.accumulate(bar_closes_pnl)
    drawdowns = bar_closes_pnl - running_max
    intratrade_dd_bps = float(np.min(drawdowns))  # Most negative = worst drawdown
    
    return {
        "mfe_bps": mfe_bps,
        "mae_bps": mae_bps,
        "intratrade_dd_bps": intratrade_dd_bps,
    }


def analyze_intratrade_risk(
    baseline_run_dir: Path,
    guardrail_run_dir: Path,
    overridden_trades_csv: Optional[Path] = None,
) -> None:
    """
    Analyze intratrade risk for overridden trades (RULE6A → RULE5).
    
    DEL 1-4: Calculate MFE/MAE/intratrade DD and compare baseline vs guardrail.
    """
    print("=" * 100)
    print("INTRATRADE RISK ANALYSIS - OVERRIDDEN TRADES")
    print("=" * 100)
    print()
    
    # Load trade logs
    df_baseline = load_trades_from_run(baseline_run_dir)
    df_guardrail = load_trades_from_run(guardrail_run_dir)
    
    # Load overridden trades if CSV provided, otherwise find them
    if overridden_trades_csv and overridden_trades_csv.exists():
        df_overridden = pd.read_csv(overridden_trades_csv)
    else:
        # Find overridden trades
        baseline_rule6a = df_baseline[
            df_baseline["exit_profile"] == "FARM_EXIT_V2_RULES_ADAPTIVE_v1"
        ].copy()
        guardrail_rule5 = df_guardrail[
            df_guardrail["exit_profile"] == "FARM_EXIT_V2_RULES_A"
        ].copy()
        
        # Match by entry_time + entry_price
        baseline_rule6a["trade_key"] = (
            baseline_rule6a["entry_time"].astype(str) + "_" +
            baseline_rule6a["entry_price"].astype(str) + "_" +
            baseline_rule6a.get("side", "").astype(str) + "_" +
            baseline_rule6a.get("direction", "").astype(str)
        )
        guardrail_rule5["trade_key"] = (
            guardrail_rule5["entry_time"].astype(str) + "_" +
            guardrail_rule5["entry_price"].astype(str) + "_" +
            guardrail_rule5.get("side", "").astype(str) + "_" +
            guardrail_rule5.get("direction", "").astype(str)
        )
        
        matched_keys = set(baseline_rule6a["trade_key"]) & set(guardrail_rule5["trade_key"])
        
        if len(matched_keys) == 0:
            print("⚠️  No overridden trades found. Skipping intratrade risk analysis.")
            return
        
        # Build comparison DataFrame
        results = []
        for key in matched_keys:
            baseline_row = baseline_rule6a[baseline_rule6a["trade_key"] == key].iloc[0]
            guardrail_row = guardrail_rule5[guardrail_rule5["trade_key"] == key].iloc[0]
            
            results.append({
                "trade_key": key,
                "baseline_row": baseline_row,
                "guardrail_row": guardrail_row,
            })
        
        df_overridden = pd.DataFrame(results)
    
    print(f"Analyzing {len(df_overridden)} overridden trades...")
    print()
    
    # Load price data (try both run directories)
    price_data = None
    for run_dir in [baseline_run_dir, guardrail_run_dir]:
        price_file = run_dir / "price_data_filtered.parquet"
        if price_file.exists():
            try:
                price_data = pd.read_parquet(price_file)
                print(f"✅ Loaded price data from: {price_file}")
                break
            except Exception as e:
                print(f"⚠️  Could not load price data from {price_file}: {e}")
    
    if price_data is None:
        print("❌ Could not load price data. Skipping intratrade risk analysis.")
        return
    
    # Ensure time index
    if price_data.index.name != "time" and "time" in price_data.columns:
        price_data = price_data.set_index("time")
    
    # Calculate intratrade metrics for each overridden trade
    results = []
    
    for idx, row in df_overridden.iterrows():
        if "baseline_row" in row:
            baseline_row = row["baseline_row"]
            guardrail_row = row["guardrail_row"]
        else:
            # Load from CSV format
            trade_key = row["trade_key"]
            baseline_row = df_baseline[
                (df_baseline["entry_time"].astype(str) + "_" +
                 df_baseline["entry_price"].astype(str) + "_" +
                 df_baseline.get("side", "").astype(str) + "_" +
                 df_baseline.get("direction", "").astype(str)) == trade_key
            ].iloc[0]
            guardrail_row = df_guardrail[
                (df_guardrail["entry_time"].astype(str) + "_" +
                 df_guardrail["entry_price"].astype(str) + "_" +
                 df_guardrail.get("side", "").astype(str) + "_" +
                 df_guardrail.get("direction", "").astype(str)) == trade_key
            ].iloc[0]
        
        # Parse timestamps
        baseline_entry_time = pd.to_datetime(baseline_row["entry_time"], utc=True)
        baseline_exit_time = pd.to_datetime(baseline_row["exit_time"], utc=True)
        guardrail_entry_time = pd.to_datetime(guardrail_row["entry_time"], utc=True)
        guardrail_exit_time = pd.to_datetime(guardrail_row["exit_time"], utc=True)
        
        entry_price = float(baseline_row["entry_price"])
        side = str(baseline_row.get("side", "long")).lower()
        
        # Calculate metrics
        baseline_metrics = calculate_intratrade_metrics(
            baseline_entry_time,
            baseline_exit_time,
            entry_price,
            price_data,
            side,
        )
        
        guardrail_metrics = calculate_intratrade_metrics(
            guardrail_entry_time,
            guardrail_exit_time,
            entry_price,
            price_data,
            side,
        )
        
        # Calculate deltas
        delta_mfe = guardrail_metrics["mfe_bps"] - baseline_metrics["mfe_bps"]
        delta_mae = guardrail_metrics["mae_bps"] - baseline_metrics["mae_bps"]
        delta_dd = guardrail_metrics["intratrade_dd_bps"] - baseline_metrics["intratrade_dd_bps"]
        
        results.append({
            "trade_key": row.get("trade_key", f"trade_{idx}"),
            "trade_id": baseline_row.get("trade_id", ""),
            "baseline_exit_profile": baseline_row["exit_profile"],
            "guardrail_exit_profile": guardrail_row["exit_profile"],
            "baseline_bars_held": float(pd.to_numeric(baseline_row.get("bars_held"), errors='coerce')) if pd.notna(baseline_row.get("bars_held")) else None,
            "guardrail_bars_held": float(pd.to_numeric(guardrail_row.get("bars_held"), errors='coerce')) if pd.notna(guardrail_row.get("bars_held")) else None,
            "baseline_MFE_bps": baseline_metrics["mfe_bps"],
            "guardrail_MFE_bps": guardrail_metrics["mfe_bps"],
            "baseline_MAE_bps": baseline_metrics["mae_bps"],
            "guardrail_MAE_bps": guardrail_metrics["mae_bps"],
            "baseline_intratrade_DD_bps": baseline_metrics["intratrade_dd_bps"],
            "guardrail_intratrade_DD_bps": guardrail_metrics["intratrade_dd_bps"],
            "delta_MFE": delta_mfe,
            "delta_MAE": delta_mae,
            "delta_intratrade_DD": delta_dd,
        })
    
    df_results = pd.DataFrame(results)
    
    # DEL 2: Print per-trade table summary
    print("=" * 100)
    print("DEL 2: PER-TRADE INTRATRADE METRICS")
    print("=" * 100)
    print()
    print(f"Total trades analyzed: {len(df_results)}")
    print()
    
    # DEL 3: Aggregates
    print("=" * 100)
    print("DEL 3: AGGREGATES AND RISK DIFFERENCES")
    print("=" * 100)
    print()
    
    print("MFE (Max Favorable Excursion):")
    print(f"  Baseline: mean={df_results['baseline_MFE_bps'].mean():.2f}, median={df_results['baseline_MFE_bps'].median():.2f}")
    print(f"  Guardrail: mean={df_results['guardrail_MFE_bps'].mean():.2f}, median={df_results['guardrail_MFE_bps'].median():.2f}")
    print(f"  Delta: mean={df_results['delta_MFE'].mean():.2f}, median={df_results['delta_MFE'].median():.2f}")
    print()
    
    print("MAE (Max Adverse Excursion):")
    print(f"  Baseline: mean={df_results['baseline_MAE_bps'].mean():.2f}, median={df_results['baseline_MAE_bps'].median():.2f}")
    print(f"  Guardrail: mean={df_results['guardrail_MAE_bps'].mean():.2f}, median={df_results['guardrail_MAE_bps'].median():.2f}")
    print(f"  Delta: mean={df_results['delta_MAE'].mean():.2f}, median={df_results['delta_MAE'].median():.2f}")
    print(f"  Max MAE baseline: {df_results['baseline_MAE_bps'].min():.2f}")
    print(f"  Max MAE guardrail: {df_results['guardrail_MAE_bps'].min():.2f}")
    print()
    
    print("Intratrade Drawdown:")
    print(f"  Baseline: mean={df_results['baseline_intratrade_DD_bps'].mean():.2f}, median={df_results['baseline_intratrade_DD_bps'].median():.2f}")
    print(f"  Guardrail: mean={df_results['guardrail_intratrade_DD_bps'].mean():.2f}, median={df_results['guardrail_intratrade_DD_bps'].median():.2f}")
    print(f"  Delta: mean={df_results['delta_intratrade_DD'].mean():.2f}, median={df_results['delta_intratrade_DD'].median():.2f}")
    print(f"  Max DD baseline: {df_results['baseline_intratrade_DD_bps'].min():.2f}")
    print(f"  Max DD guardrail: {df_results['guardrail_intratrade_DD_bps'].min():.2f}")
    print()
    
    # Key insights
    mae_worse_under_rule5 = (df_results['delta_MAE'] < 0).sum()  # Negative delta = worse MAE
    dd_reduced_by_guardrail = (df_results['delta_intratrade_DD'] > 0).sum()  # Positive delta = better DD
    
    print("KEY INSIGHTS:")
    print(f"  MAE worse under RULE5 (guardrail): {mae_worse_under_rule5}/{len(df_results)} ({100*mae_worse_under_rule5/len(df_results):.1f}%)")
    print(f"  Intratrade DD reduced by guardrail: {dd_reduced_by_guardrail}/{len(df_results)} ({100*dd_reduced_by_guardrail/len(df_results):.1f}%)")
    print()
    
    # DEL 4: Interpretation
    print("=" * 100)
    print("DEL 4: INTERPRETATION")
    print("=" * 100)
    print()
    
    print("ANALYSIS:")
    if mae_worse_under_rule5 > len(df_results) * 0.5:
        print("  ⚠️  RULE5 holds longer with more pain:")
        print(f"     {mae_worse_under_rule5}/{len(df_results)} trades had worse MAE under RULE5")
        print("     This suggests RULE5 allows larger adverse excursions before exit.")
    else:
        print("  ✅ Guardrail reduces intratrade pain:")
        print(f"     Only {mae_worse_under_rule5}/{len(df_results)} trades had worse MAE under RULE5")
        print("     Most trades had similar or better MAE under guardrail (RULE5).")
    print()
    
    if dd_reduced_by_guardrail > len(df_results) * 0.5:
        print("  ✅ Guardrail reduces intratrade drawdown:")
        print(f"     {dd_reduced_by_guardrail}/{len(df_results)} trades had reduced drawdown")
        print("     RULE5 (via guardrail) provides better risk control during trade lifetime.")
    else:
        print("  ⚠️  Guardrail does not consistently reduce drawdown:")
        print(f"     Only {dd_reduced_by_guardrail}/{len(df_results)} trades had reduced drawdown")
        print("     RULE5 and RULE6A show similar intratrade risk profiles.")
    print()
    
    # Check if RULE6A was "unnecessary" or "cosmetic"
    same_exit_result = ((df_results['delta_MFE'].abs() < 1.0) & (df_results['delta_MAE'].abs() < 1.0)).sum()
    if same_exit_result > len(df_results) * 0.8:
        print("  💡 RULE6A appears 'cosmetic' in these regimes:")
        print("     Most trades show identical MFE/MAE, suggesting same exit behavior.")
        print("     Guardrail correctly identifies that RULE6A adds no value here.")
    else:
        print("  💡 RULE6A shows different intratrade behavior:")
        print("     MFE/MAE differences suggest RULE6A and RULE5 handle trades differently.")
        print("     However, final PnL is identical, indicating both reach same endpoint.")
    print()
    
    # Save CSV
    output_dir = baseline_run_dir.parent / "guardrail_verification"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "intratrade_risk_analysis.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"✅ Saved intratrade risk analysis to: {csv_path}")
    print()


if __name__ == "__main__":
    main()

