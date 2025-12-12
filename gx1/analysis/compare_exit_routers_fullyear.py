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
from typing import Dict, Any, Optional

import pandas as pd


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
        required=True,
        help="Add a run to compare. Format: --run TAG PATH (can be used multiple times)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("gx1/analysis/exit_router_models_v3/exit_router_comparison_fullyear.json"),
        help="Path to save JSON summary (default: gx1/analysis/exit_router_models_v3/exit_router_comparison_fullyear.json)",
    )
    
    args = parser.parse_args()
    
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


if __name__ == "__main__":
    main()

