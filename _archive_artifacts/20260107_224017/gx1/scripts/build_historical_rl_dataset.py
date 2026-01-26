#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Historical RL Dataset from All Runs

Finds all run-dirs under gx1/wf_runs with trade_journal and shadow data,
runs analyze_shadow_and_trades.py logic on each, and combines into a master dataset.

Usage:
    python gx1/scripts/build_historical_rl_dataset.py
    python gx1/scripts/build_historical_rl_dataset.py --wf_runs_dir gx1/wf_runs --output_prefix FULLYEAR_2025
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Add parent directory to path for imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent.parent))

# Import functions from analyze_shadow_and_trades.py
from gx1.scripts.analyze_shadow_and_trades import (
    build_rl_dataset,
    join_shadow_and_trades,
    load_shadow_journal,
    load_trade_journal,
)


def find_runs_with_data(
    wf_runs_dir: Path,
    require_shadow: bool = True,
    require_trades: bool = False,
) -> List[Path]:
    """
    Find all run directories with shadow and/or trade journal data.
    
    Args:
        wf_runs_dir: Path to gx1/wf_runs directory (or runs/live_demo)
        require_shadow: If True, only include runs with shadow data
        require_trades: If True, only include runs with trade data
    
    Returns:
        List of run directory paths
    """
    if not wf_runs_dir.exists():
        print(f"WARNING: Directory not found: {wf_runs_dir}")
        return []
    
    runs = []
    
    # Find all subdirectories (recursively search first level)
    for run_dir in wf_runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        
        # Check for shadow data (try both direct and nested paths)
        shadow_path = run_dir / "shadow" / "shadow_hits.jsonl"
        if not shadow_path.exists():
            # Try alternative: runs/live_demo/SNIPER_*/shadow/shadow_hits.jsonl
            shadow_path = run_dir / "shadow" / "shadow" / "shadow_hits.jsonl"
        has_shadow = shadow_path.exists()
        
        # Check for trade journal
        trade_dir = run_dir / "trade_journal" / "trades"
        has_trades = trade_dir.exists() and len(list(trade_dir.glob("*.json"))) > 0
        
        # Apply filters
        if require_shadow and not has_shadow:
            continue
        if require_trades and not has_trades:
            continue
        
        if has_shadow or has_trades:
            runs.append(run_dir)
    
    print(f"Found {len(runs)} runs with data (shadow={require_shadow}, trades={require_trades})")
    return runs


def process_run(run_dir: Path) -> Optional[pd.DataFrame]:
    """
    Process a single run directory and return RL dataset.
    
    Args:
        run_dir: Path to run directory
    
    Returns:
        RL dataset DataFrame or None if processing fails
    """
    print(f"\nProcessing: {run_dir.name}")
    
    # Load shadow journal (try multiple paths)
    shadow_path = run_dir / "shadow" / "shadow_hits.jsonl"
    if not shadow_path.exists():
        # Try alternative path (runs/live_demo/SNIPER_*/shadow/shadow/shadow_hits.jsonl)
        shadow_path = run_dir / "shadow" / "shadow" / "shadow_hits.jsonl"
    if not shadow_path.exists():
        print(f"  ⚠️  No shadow data found, skipping")
        return None
    
    shadow_df = load_shadow_journal(shadow_path)
    if shadow_df.empty:
        print(f"  ⚠️  Shadow data is empty, skipping")
        return None
    
    # Load trade journal
    trades_df = load_trade_journal(run_dir)
    if trades_df.empty:
        print(f"  ℹ️  No trades found (shadow-only signals)")
    
    # Join shadow and trades
    joined_df = join_shadow_and_trades(shadow_df, trades_df)
    if joined_df.empty:
        print(f"  ⚠️  Joined data is empty, skipping")
        return None
    
    # Build RL dataset
    rl_df = build_rl_dataset(joined_df)
    if rl_df.empty:
        print(f"  ⚠️  RL dataset is empty, skipping")
        return None
    
    # Add run metadata
    rl_df["run_id"] = run_dir.name
    rl_df["run_dir"] = str(run_dir)
    
    print(f"  ✅ Processed: {len(rl_df)} records ({rl_df['action_taken'].sum()} trades)")
    
    return rl_df


def combine_datasets(datasets: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Combine multiple RL datasets into one master dataset.
    
    Args:
        datasets: List of RL dataset DataFrames
    
    Returns:
        Combined DataFrame
    """
    if not datasets:
        return pd.DataFrame()
    
    # Combine all datasets
    combined = pd.concat(datasets, ignore_index=True)
    
    # Remove duplicates (if any)
    initial_len = len(combined)
    combined = combined.drop_duplicates(subset=["candle_time", "run_id"], keep="first")
    if len(combined) < initial_len:
        print(f"Removed {initial_len - len(combined)} duplicate records")
    
    print(f"\nCombined dataset: {len(combined)} total records")
    print(f"  - Trades: {combined['action_taken'].sum()}")
    print(f"  - Shadow-only: {len(combined) - combined['action_taken'].sum()}")
    
    return combined


def generate_summary_report(
    combined_df: pd.DataFrame,
    output_path: Path,
    wf_runs_dir: Path,
) -> None:
    """Generate summary report for historical RL dataset."""
    lines = []
    
    lines.append("# Historical RL Dataset Summary - 2025")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"**Source:** {wf_runs_dir}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Basic statistics
    lines.append("## Dataset Statistics")
    lines.append("")
    total_rows = len(combined_df)
    total_trades = combined_df["action_taken"].sum()
    shadow_only = total_rows - total_trades
    
    lines.append(f"- **Total Shadow Rows:** {total_rows:,}")
    lines.append(f"- **Total Trades:** {total_trades:,}")
    lines.append(f"- **Shadow-Only Signals:** {shadow_only:,}")
    lines.append(f"- **Trade Rate:** {total_trades/total_rows*100:.2f}%")
    lines.append("")
    
    # Class balance
    if "label_profitable_10bps" in combined_df.columns:
        target_col = "label_profitable_10bps"
    elif "label_profitable" in combined_df.columns:
        target_col = "label_profitable"
    else:
        target_col = None
    
    if target_col and target_col in combined_df.columns:
        lines.append("## Class Balance")
        lines.append("")
        class_counts = combined_df[target_col].value_counts().sort_index()
        for cls, count in class_counts.items():
            pct = count / len(combined_df) * 100
            lines.append(f"- **Class {int(cls)}:** {count:,} ({pct:.2f}%)")
        lines.append("")
    
    # Regime distribution
    if "trend_regime" in combined_df.columns and "vol_regime" in combined_df.columns:
        lines.append("## Regime Distribution")
        lines.append("")
        regime_cross = pd.crosstab(
            combined_df["trend_regime"],
            combined_df["vol_regime"],
            margins=True,
        )
        lines.append("| Trend Regime | Vol Regime | Count |")
        lines.append("|--------------|------------|-------|")
        for idx, row in regime_cross.iterrows():
            for vol_regime, count in row.items():
                if vol_regime != "All":
                    lines.append(f"| {idx} | {vol_regime} | {int(count)} |")
        lines.append("")
    
    # Session distribution
    if "session" in combined_df.columns:
        lines.append("## Session Distribution")
        lines.append("")
        session_counts = combined_df["session"].value_counts()
        lines.append("| Session | Count | Percentage |")
        lines.append("|---------|-------|------------|")
        for session, count in session_counts.items():
            pct = count / len(combined_df) * 100
            lines.append(f"| {session} | {count:,} | {pct:.2f}% |")
        lines.append("")
    
    # p_long histogram
    if "p_long" in combined_df.columns:
        lines.append("## p_long Distribution")
        lines.append("")
        p_long_stats = combined_df["p_long"].describe()
        lines.append(f"- **Mean:** {p_long_stats.get('mean', 0):.3f}")
        lines.append(f"- **Median:** {p_long_stats.get('50%', 0):.3f}")
        lines.append(f"- **Min:** {p_long_stats.get('min', 0):.3f}")
        lines.append(f"- **Max:** {p_long_stats.get('max', 0):.3f}")
        lines.append(f"- **Std:** {p_long_stats.get('std', 0):.3f}")
        lines.append("")
        
        # p_long by action_taken
        if "action_taken" in combined_df.columns:
            lines.append("### p_long by Action")
            lines.append("")
            action_stats = combined_df.groupby("action_taken")["p_long"].describe()
            lines.append("| Action | Mean | Median | Min | Max |")
            lines.append("|--------|------|--------|-----|-----|")
            for action, stats in action_stats.iterrows():
                action_name = "Trade" if action == 1 else "Skip"
                lines.append(
                    f"| {action_name} | {stats.get('mean', 0):.3f} | "
                    f"{stats.get('50%', 0):.3f} | {stats.get('min', 0):.3f} | "
                    f"{stats.get('max', 0):.3f} |"
                )
            lines.append("")
    
    # Potential EV improvements
    if "pnl_bps" in combined_df.columns and "p_long" in combined_df.columns:
        lines.append("## Potential EV Improvements")
        lines.append("")
        
        # Calculate average PnL by p_long buckets
        combined_df["p_long_bucket"] = pd.cut(
            combined_df["p_long"],
            bins=[0, 0.55, 0.60, 0.65, 0.70, 1.0],
            labels=["<0.55", "0.55-0.60", "0.60-0.65", "0.65-0.70", "0.70+"],
        )
        
        bucket_stats = combined_df[combined_df["action_taken"] == 1].groupby("p_long_bucket", observed=True)["pnl_bps"].agg(
            ["mean", "count", "sum"]
        )
        
        lines.append("### Average PnL by p_long Bucket (Trades Only)")
        lines.append("")
        lines.append("| p_long Bucket | Avg PnL (bps) | Trade Count | Total PnL (bps) |")
        lines.append("|---------------|----------------|-------------|------------------|")
        for bucket, stats in bucket_stats.iterrows():
            lines.append(
                f"| {bucket} | {stats['mean']:.2f} | {int(stats['count'])} | "
                f"{stats['sum']:.2f} |"
            )
        lines.append("")
        
        # Shadow-only opportunities
        shadow_only_df = combined_df[combined_df["action_taken"] == 0].copy()
        if len(shadow_only_df) > 0:
            # Estimate potential trades at lower thresholds
            lines.append("### Shadow-Only Opportunities")
            lines.append("")
            lines.append("Estimated potential trades if threshold was lowered:")
            lines.append("")
            
            thresholds = [0.55, 0.58, 0.60, 0.62, 0.65]
            lines.append("| Threshold | Potential Trades | Avg p_long |")
            lines.append("|-----------|------------------|------------|")
            for thr in thresholds:
                potential = shadow_only_df[shadow_only_df["p_long"] >= thr]
                if len(potential) > 0:
                    avg_p_long = potential["p_long"].mean()
                    lines.append(f"| {thr:.2f} | {len(potential):,} | {avg_p_long:.3f} |")
            lines.append("")
    
    # Run distribution
    if "run_id" in combined_df.columns:
        lines.append("## Run Distribution")
        lines.append("")
        run_counts = combined_df["run_id"].value_counts()
        lines.append(f"**Total Runs:** {len(run_counts)}")
        lines.append("")
        lines.append("Top 10 runs by record count:")
        lines.append("")
        lines.append("| Run ID | Records | Trades |")
        lines.append("|--------|---------|--------|")
        for run_id, count in run_counts.head(10).items():
            trades = combined_df[combined_df["run_id"] == run_id]["action_taken"].sum()
            lines.append(f"| {run_id} | {count:,} | {trades} |")
        lines.append("")
    
    lines.append("---")
    lines.append("")
    lines.append(f"*Report generated by `gx1/scripts/build_historical_rl_dataset.py`*")
    
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    
    print(f"\nSummary report saved: {output_path}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build Historical RL Dataset from All Runs"
    )
    parser.add_argument(
        "--wf_runs_dir",
        type=Path,
        default=Path("gx1/wf_runs"),
        help="Path to wf_runs directory (default: gx1/wf_runs)",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="FULLYEAR_2025",
        help="Output prefix for dataset files (default: FULLYEAR_2025)",
    )
    parser.add_argument(
        "--require_trades",
        action="store_true",
        help="Only include runs with trade data (default: include all runs with shadow data)",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data/rl"),
        help="Output directory for datasets (default: data/rl)",
    )
    parser.add_argument(
        "--reports_dir",
        type=Path,
        default=Path("reports/rl"),
        help="Output directory for reports (default: reports/rl)",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Building Historical RL Dataset")
    print("=" * 60)
    print(f"Source: {args.wf_runs_dir}")
    print(f"Output prefix: {args.output_prefix}")
    print("")
    
    # Find runs with data
    runs = find_runs_with_data(
        args.wf_runs_dir,
        require_shadow=True,
        require_trades=args.require_trades,
    )
    
    if not runs:
        print("ERROR: No runs found with required data")
        return 1
    
    # Process each run
    datasets = []
    failed_runs = []
    
    for run_dir in runs:
        try:
            rl_df = process_run(run_dir)
            if rl_df is not None and not rl_df.empty:
                datasets.append(rl_df)
        except Exception as e:
            print(f"  ❌ Error processing {run_dir.name}: {e}")
            failed_runs.append((run_dir.name, str(e)))
    
    if not datasets:
        print("\nERROR: No datasets were successfully processed")
        return 1
    
    print(f"\n✅ Successfully processed {len(datasets)} runs")
    if failed_runs:
        print(f"⚠️  Failed to process {len(failed_runs)} runs")
    
    # Combine datasets
    combined_df = combine_datasets(datasets)
    
    if combined_df.empty:
        print("\nERROR: Combined dataset is empty")
        return 1
    
    # Save master dataset
    args.data_dir.mkdir(parents=True, exist_ok=True)
    
    parquet_path = args.data_dir / f"sniper_shadow_rl_dataset_{args.output_prefix}.parquet"
    combined_df.to_parquet(parquet_path, index=False)
    print(f"\n✅ Master dataset (Parquet) saved: {parquet_path}")
    
    csv_path = args.data_dir / f"sniper_shadow_rl_dataset_{args.output_prefix}.csv"
    combined_df.to_csv(csv_path, index=False)
    print(f"✅ Master dataset (CSV) saved: {csv_path}")
    
    # Generate summary report
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.reports_dir / f"HISTORICAL_RL_DATASET_SUMMARY_{args.output_prefix}.md"
    generate_summary_report(combined_df, report_path, args.wf_runs_dir)
    
    # Save metadata
    meta_path = args.data_dir / f"sniper_shadow_rl_dataset_{args.output_prefix}_meta.json"
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(args.wf_runs_dir),
        "output_prefix": args.output_prefix,
        "n_runs_processed": len(datasets),
        "n_runs_failed": len(failed_runs),
        "n_total_records": len(combined_df),
        "n_trades": int(combined_df["action_taken"].sum()),
        "n_shadow_only": int(len(combined_df) - combined_df["action_taken"].sum()),
        "columns": list(combined_df.columns),
        "failed_runs": failed_runs,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"✅ Metadata saved: {meta_path}")
    
    print("\n" + "=" * 60)
    print("✅ Historical RL Dataset Build Complete!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())

