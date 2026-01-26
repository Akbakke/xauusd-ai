#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SNIPER vs Shadow Analysis & Policy Simulation (2025)

Offline analysis script that:
1. Compares SNIPER trades vs shadow-only signals
2. Simulates alternative policy filters (offline only)

IMPORTANT: This script does NOT modify runtime code or trading logic.
It is purely analytical and reads from RL dataset only.
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    """Load RL dataset from Parquet file."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    log.info(f"Loading dataset: {dataset_path}")
    df = pd.read_parquet(dataset_path)
    log.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Check required columns
    required_cols = ["action_taken", "p_long"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Determine reward column
    reward_col = None
    for col in ["reward", "pnl_bps"]:
        if col in df.columns:
            reward_col = col
            break
    
    if reward_col is None:
        log.warning("No reward/pnl_bps column found, will use 0 for reward calculations")
        df["reward"] = 0.0
        reward_col = "reward"
    
    # Determine label column
    label_col = None
    for col in ["label_profitable_10bps", "label_profitable"]:
        if col in df.columns:
            label_col = col
            break
    
    if label_col is None:
        log.warning("No label column found, will infer from reward")
        df["label_profitable_10bps"] = (df[reward_col] >= 10.0).astype(int)
        label_col = "label_profitable_10bps"
    
    # Drop rows with NaN in critical fields
    initial_len = len(df)
    df = df.dropna(subset=["action_taken", "p_long", reward_col])
    dropped = initial_len - len(df)
    if dropped > 0:
        log.info(f"Dropped {dropped} rows with NaN in critical fields")
    
    # Ensure action_taken is binary
    df["action_taken"] = df["action_taken"].astype(int)
    
    log.info(f"Final dataset: {len(df)} rows")
    log.info(f"  - Trades (action_taken=1): {df['action_taken'].sum()}")
    log.info(f"  - Shadow-only (action_taken=0): {len(df) - df['action_taken'].sum()}")
    
    return df, reward_col, label_col


def calculate_statistics(
    df: pd.DataFrame,
    reward_col: str,
    label_col: str,
    name: str = "Dataset",
) -> Dict[str, float]:
    """Calculate statistics for a dataset subset."""
    if len(df) == 0:
        return {
            "n": 0,
            "win_rate": 0.0,
            "avg_reward": 0.0,
            "median_reward": 0.0,
            "p95_reward": 0.0,
            "p05_reward": 0.0,
            "avg_p_long": 0.0,
            "avg_spread_bps": 0.0,
            "avg_atr_bps": 0.0,
        }
    
    stats = {
        "n": len(df),
        "win_rate": df[label_col].mean() if label_col in df.columns else 0.0,
        "avg_reward": df[reward_col].mean(),
        "median_reward": df[reward_col].median(),
        "p95_reward": df[reward_col].quantile(0.95),
        "p05_reward": df[reward_col].quantile(0.05),
        "avg_p_long": df["p_long"].mean(),
    }
    
    # Optional columns
    if "spread_bps" in df.columns:
        stats["avg_spread_bps"] = df["spread_bps"].mean()
    else:
        stats["avg_spread_bps"] = None
    
    if "atr_bps" in df.columns:
        stats["avg_atr_bps"] = df["atr_bps"].mean()
    else:
        stats["avg_atr_bps"] = None
    
    return stats


def calculate_p_long_buckets(
    df: pd.DataFrame,
    reward_col: str,
    label_col: str,
    buckets: List[float] = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
) -> pd.DataFrame:
    """Calculate statistics per p_long bucket."""
    bucket_stats = []
    
    for i in range(len(buckets) - 1):
        lower = buckets[i]
        upper = buckets[i + 1]
        
        mask = (df["p_long"] >= lower) & (df["p_long"] < upper)
        bucket_df = df[mask]
        
        if len(bucket_df) > 0:
            bucket_stats.append({
                "p_long_range": f"{lower:.1f}-{upper:.1f}",
                "n": len(bucket_df),
                "win_rate": bucket_df[label_col].mean() if label_col in bucket_df.columns else 0.0,
                "avg_reward": bucket_df[reward_col].mean(),
                "median_reward": bucket_df[reward_col].median(),
            })
        else:
            bucket_stats.append({
                "p_long_range": f"{lower:.1f}-{upper:.1f}",
                "n": 0,
                "win_rate": 0.0,
                "avg_reward": 0.0,
                "median_reward": 0.0,
            })
    
    return pd.DataFrame(bucket_stats)


def simulate_policy(
    df: pd.DataFrame,
    p_long_threshold: float,
    critic_threshold: Optional[float],
    reward_col: str,
    label_col: str,
) -> Dict[str, float]:
    """
    Simulate a policy filter on the dataset.
    
    Args:
        df: Full dataset (both trades and shadow-only)
        p_long_threshold: Minimum p_long to take trade
        critic_threshold: Minimum entry_critic_score_v1 (None = no filter)
        reward_col: Column name for reward
        label_col: Column name for label
    
    Returns:
        Dictionary with simulation results
    """
    # Apply filters
    mask = df["p_long"] >= p_long_threshold
    
    if critic_threshold is not None:
        if "entry_critic_score_v1" in df.columns:
            critic_mask = df["entry_critic_score_v1"].notna() & (df["entry_critic_score_v1"] >= critic_threshold)
            mask = mask & critic_mask
        else:
            # If critic column doesn't exist, can't apply filter
            return {
                "n_trades_sim": 0,
                "win_rate_sim": 0.0,
                "avg_reward_sim": 0.0,
                "median_reward_sim": 0.0,
                "reward_sum_sim": 0.0,
                "reward_per_1k_trades_sim": 0.0,
            }
    
    filtered_df = df[mask]
    
    if len(filtered_df) == 0:
        return {
            "n_trades_sim": 0,
            "win_rate_sim": 0.0,
            "avg_reward_sim": 0.0,
            "median_reward_sim": 0.0,
            "reward_sum_sim": 0.0,
            "reward_per_1k_trades_sim": 0.0,
        }
    
    n_trades = len(filtered_df)
    win_rate = filtered_df[label_col].mean() if label_col in filtered_df.columns else 0.0
    avg_reward = filtered_df[reward_col].mean()
    median_reward = filtered_df[reward_col].median()
    reward_sum = filtered_df[reward_col].sum()
    reward_per_1k = (reward_sum / n_trades) * 1000 if n_trades > 0 else 0.0
    
    return {
        "n_trades_sim": n_trades,
        "win_rate_sim": win_rate,
        "avg_reward_sim": avg_reward,
        "median_reward_sim": median_reward,
        "reward_sum_sim": reward_sum,
        "reward_per_1k_trades_sim": reward_per_1k,
    }


def generate_report(
    df: pd.DataFrame,
    reward_col: str,
    label_col: str,
    dataset_path: Path,
    report_path: Path,
) -> None:
    """Generate markdown analysis report."""
    lines = []
    
    lines.append("# ENTRY CRITIC – 2025 Policy Eval (SNIPER vs Shadow)")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Metadata
    lines.append("## Metadata")
    lines.append("")
    lines.append(f"- **Dataset path:** {dataset_path}")
    lines.append(f"- **Total rows:** {len(df):,}")
    lines.append(f"- **Trades (action_taken=1):** {df['action_taken'].sum():,}")
    lines.append(f"- **Shadow-only (action_taken=0):** {len(df) - df['action_taken'].sum():,}")
    lines.append(f"- **Reward column:** {reward_col}")
    lines.append(f"- **Label column:** {label_col}")
    lines.append("")
    
    # Split dataset
    real_trades = df[df["action_taken"] == 1].copy()
    shadow_only = df[df["action_taken"] == 0].copy()
    
    # DEL 1 – Baseline comparison
    lines.append("## Del 1 – 2025: SNIPER trades vs shadow-only")
    lines.append("")
    
    # Overall statistics
    lines.append("### Overall Statistics")
    lines.append("")
    lines.append("| Metric | SNIPER Trades | Shadow-Only |")
    lines.append("|--------|---------------|-------------|")
    
    stats_trades = calculate_statistics(real_trades, reward_col, label_col, "Trades")
    stats_shadow = calculate_statistics(shadow_only, reward_col, label_col, "Shadow")
    
    metrics = [
        ("n", "Count", "{:,}"),
        ("win_rate", "Win Rate", "{:.2%}"),
        ("avg_reward", "Avg Reward (bps)", "{:.2f}"),
        ("median_reward", "Median Reward (bps)", "{:.2f}"),
        ("p95_reward", "P95 Reward (bps)", "{:.2f}"),
        ("p05_reward", "P05 Reward (bps)", "{:.2f}"),
        ("avg_p_long", "Avg p_long", "{:.3f}"),
    ]
    
    for key, label, fmt in metrics:
        val_trades = stats_trades.get(key, 0)
        val_shadow = stats_shadow.get(key, 0)
        lines.append(f"| {label} | {fmt.format(val_trades)} | {fmt.format(val_shadow)} |")
    
    # Optional metrics
    if stats_trades.get("avg_spread_bps") is not None:
        lines.append(f"| Avg Spread (bps) | {stats_trades['avg_spread_bps']:.2f} | {stats_shadow['avg_spread_bps']:.2f} |")
    if stats_trades.get("avg_atr_bps") is not None:
        lines.append(f"| Avg ATR (bps) | {stats_trades['avg_atr_bps']:.2f} | {stats_shadow['avg_atr_bps']:.2f} |")
    
    lines.append("")
    
    # p_long buckets
    lines.append("### p_long Buckets")
    lines.append("")
    
    buckets_trades = calculate_p_long_buckets(real_trades, reward_col, label_col)
    buckets_shadow = calculate_p_long_buckets(shadow_only, reward_col, label_col)
    
    lines.append("#### SNIPER Trades")
    lines.append("")
    lines.append("| p_long Range | Count | Win Rate | Avg Reward (bps) | Median Reward (bps) |")
    lines.append("|--------------|-------|----------|------------------|---------------------|")
    for _, row in buckets_trades.iterrows():
        if row["n"] > 0:
            lines.append(
                f"| {row['p_long_range']} | {int(row['n']):,} | "
                f"{row['win_rate']:.2%} | {row['avg_reward']:.2f} | {row['median_reward']:.2f} |"
            )
    lines.append("")
    
    lines.append("#### Shadow-Only")
    lines.append("")
    lines.append("| p_long Range | Count | Win Rate | Avg Reward (bps) | Median Reward (bps) |")
    lines.append("|--------------|-------|----------|------------------|---------------------|")
    for _, row in buckets_shadow.iterrows():
        if row["n"] > 0:
            lines.append(
                f"| {row['p_long_range']} | {int(row['n']):,} | "
                f"{row['win_rate']:.2%} | {row['avg_reward']:.2f} | {row['median_reward']:.2f} |"
            )
    lines.append("")
    
    # DEL 2 – Policy simulation
    lines.append("## Del 2 – Simulerte alternative policies (offline)")
    lines.append("")
    lines.append("**Note:** This is offline simulation only. No runtime code is modified.")
    lines.append("")
    
    # Policy sweep
    p_long_thresholds = [0.60, 0.65, 0.70, 0.75, 0.80]
    critic_thresholds = [None, 0.50, 0.60, 0.70]
    
    simulation_results = []
    
    for p_thr in p_long_thresholds:
        for critic_thr in critic_thresholds:
            result = simulate_policy(df, p_thr, critic_thr, reward_col, label_col)
            result["p_long_thr"] = p_thr
            result["critic_thr"] = critic_thr if critic_thr is not None else "None"
            simulation_results.append(result)
    
    sim_df = pd.DataFrame(simulation_results)
    
    # Filter out zero-trade policies
    sim_df = sim_df[sim_df["n_trades_sim"] > 0].copy()
    
    # Sort by reward_per_1k_trades
    sim_df = sim_df.sort_values("reward_per_1k_trades_sim", ascending=False)
    
    # Write table
    lines.append("### Policy Sweep Results")
    lines.append("")
    lines.append("| p_long_thr | critic_thr | n_trades | win_rate | avg_reward | reward_per_1k |")
    lines.append("|------------|-----------|----------|----------|------------|---------------|")
    
    for _, row in sim_df.iterrows():
        lines.append(
            f"| {row['p_long_thr']:.2f} | {row['critic_thr']} | "
            f"{int(row['n_trades_sim']):,} | {row['win_rate_sim']:.2%} | "
            f"{row['avg_reward_sim']:.2f} | {row['reward_per_1k_trades_sim']:.2f} |"
        )
    lines.append("")
    
    # Highlight candidates
    lines.append("### Top Candidates")
    lines.append("")
    
    # Highest reward_per_1k
    top_reward = sim_df.iloc[0]
    lines.append(f"**Highest reward_per_1k_trades:**")
    lines.append(f"- p_long_thr: {top_reward['p_long_thr']:.2f}")
    lines.append(f"- critic_thr: {top_reward['critic_thr']}")
    lines.append(f"- n_trades: {int(top_reward['n_trades_sim']):,}")
    lines.append(f"- win_rate: {top_reward['win_rate_sim']:.2%}")
    lines.append(f"- reward_per_1k: {top_reward['reward_per_1k_trades_sim']:.2f}")
    lines.append("")
    
    # Best win_rate with min trades
    min_trades = 2000
    candidates = sim_df[sim_df["n_trades_sim"] >= min_trades].copy()
    if len(candidates) > 0:
        best_winrate = candidates.sort_values("win_rate_sim", ascending=False).iloc[0]
        lines.append(f"**Best win_rate (≥{min_trades} trades):**")
        lines.append(f"- p_long_thr: {best_winrate['p_long_thr']:.2f}")
        lines.append(f"- critic_thr: {best_winrate['critic_thr']}")
        lines.append(f"- n_trades: {int(best_winrate['n_trades_sim']):,}")
        lines.append(f"- win_rate: {best_winrate['win_rate_sim']:.2%}")
        lines.append(f"- reward_per_1k: {best_winrate['reward_per_1k_trades_sim']:.2f}")
        lines.append("")
    
    # Notes
    lines.append("---")
    lines.append("")
    lines.append("## Notater")
    lines.append("")
    lines.append("- **Ingen runtime-logikk er endret.**")
    lines.append("- Dette er ren offline-analyse.")
    lines.append("- Resultater kan brukes til å foreslå fremtidige thresholds,")
    lines.append("  men ikke er aktivert noe sted i SNIPER/FARM.")
    lines.append("")
    lines.append(f"*Report generated by `gx1/scripts/eval_sniper_vs_shadow_2025.py`*")
    
    # Write report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    
    log.info(f"Report saved: {report_path}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SNIPER vs Shadow Analysis & Policy Simulation (2025)"
    )
    parser.add_argument(
        "--dataset_path",
        type=Path,
        default=Path("data/rl/sniper_shadow_rl_dataset_FULLYEAR_2025_PARALLEL.parquet"),
        help="Path to RL dataset Parquet file",
    )
    parser.add_argument(
        "--report_out",
        type=Path,
        default=Path("reports/rl/ENTRY_CRITIC_2025_POLICY_EVAL.md"),
        help="Output path for markdown report",
    )
    
    args = parser.parse_args()
    
    try:
        # Load dataset
        df, reward_col, label_col = load_dataset(args.dataset_path)
        
        # Generate report
        generate_report(df, reward_col, label_col, args.dataset_path, args.report_out)
        
        log.info("✅ Analysis complete!")
        return 0
    
    except Exception as e:
        log.error(f"❌ Analysis failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())













