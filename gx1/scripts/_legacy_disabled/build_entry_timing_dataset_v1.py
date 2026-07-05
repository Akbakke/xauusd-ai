#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Entry Timing Dataset V1

Joins RL dataset with shadow counterfactual V2 to create training dataset
for EntryTimingCriticV1 model.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def load_rl_dataset(dataset_path: Path) -> pd.DataFrame:
    """Load RL dataset."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"RL dataset not found: {dataset_path}")
    
    log.info(f"Loading RL dataset: {dataset_path}")
    df = pd.read_parquet(dataset_path)
    log.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Check required columns
    required_cols = ["candle_time", "p_long", "action_taken"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in RL dataset: {missing}")
    
    # Normalize timestamp column
    if "candle_time" in df.columns:
        df["ts"] = pd.to_datetime(df["candle_time"])
    elif "ts" not in df.columns:
        raise ValueError("Cannot find timestamp column (candle_time or ts)")
    
    if df["ts"].dt.tz is None:
        df["ts"] = df["ts"].dt.tz_localize("UTC")
    
    log.info(f"RL dataset: {len(df)} rows, shadow entries: {(df['action_taken'] == 0).sum()}")
    
    return df


def load_cf_dataset_v2(cf_path: Path) -> pd.DataFrame:
    """Load shadow counterfactual V2 dataset (shadow-only)."""
    if not cf_path.exists():
        raise FileNotFoundError(f"Counterfactual V2 dataset not found: {cf_path}")
    
    log.info(f"Loading counterfactual V2 dataset: {cf_path}")
    df = pd.read_parquet(cf_path)
    log.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Check required columns
    required_cols = ["ts", "timing_quality", "trade_type"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CF V2 dataset: {missing}")
    
    # Filter to shadow-only
    if "trade_type" in df.columns:
        df = df[df["trade_type"] == "SHADOW_SIM"].copy()
        log.info(f"Filtered to {len(df)} SHADOW_SIM trades")
    
    # Normalize timestamp
    df["ts"] = pd.to_datetime(df["ts"])
    if df["ts"].dt.tz is None:
        df["ts"] = df["ts"].dt.tz_localize("UTC")
    
    # Check timing_quality
    if "timing_quality" not in df.columns:
        raise ValueError("timing_quality column not found in CF V2 dataset")
    
    log.info(f"Timing quality distribution:")
    timing_counts = df["timing_quality"].value_counts()
    for quality, count in timing_counts.items():
        log.info(f"  {quality}: {count:,} ({count/len(df)*100:.1f}%)")
    
    return df


def join_datasets(rl_df: pd.DataFrame, cf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join RL dataset with counterfactual V2 dataset.
    
    Matches shadow entries (action_taken == 0) with V2 posts for same timestamp.
    """
    log.info("Joining RL dataset with counterfactual V2 dataset...")
    
    # Filter RL to shadow-only entries
    rl_shadow = rl_df[rl_df["action_taken"] == 0].copy()
    log.info(f"RL shadow entries: {len(rl_shadow):,}")
    
    # Merge on timestamp (exact match)
    merged = pd.merge(
        rl_shadow,
        cf_df,
        on="ts",
        how="inner",
        suffixes=("_rl", "_cf"),
    )
    
    log.info(f"Joined dataset: {len(merged)} rows")
    
    if len(merged) == 0:
        log.warning("No matches found! Check timestamp alignment between datasets.")
        return pd.DataFrame()
    
    # Build result with features from RL and timing from CF
    result_data = {}
    
    # Features from RL (prefer _rl suffix if exists, otherwise original)
    feature_cols = [
        "ts", "p_long", "spread_bps", "atr_bps",
        "trend_regime", "vol_regime", "session",
        "threshold_slot_numeric", "real_threshold",
    ]
    
    for col in feature_cols:
        if f"{col}_rl" in merged.columns:
            result_data[col] = merged[f"{col}_rl"]
        elif col in merged.columns:
            result_data[col] = merged[col]
        elif col in rl_shadow.columns:
            # If not in merged, try to get from original
            result_data[col] = None
    
    # Target and metrics from CF (prefer _cf suffix if exists, otherwise original)
    if "timing_quality" in merged.columns:
        result_data["timing_quality"] = merged["timing_quality"]
    elif "timing_quality_cf" in merged.columns:
        result_data["timing_quality"] = merged["timing_quality_cf"]
    
    if "label_profitable_10bps" in merged.columns:
        result_data["label_profitable_10bps"] = merged["label_profitable_10bps"]
    elif "label_profitable_10bps_cf" in merged.columns:
        result_data["label_profitable_10bps"] = merged["label_profitable_10bps_cf"]
    
    # Optional metrics from CF (for analysis)
    cf_metrics = [
        "mae_bps", "mfe_bps", "bars_to_first_profit",
        "mae_before_first_profit_bps", "bars_to_mfe_peak",
        "better_entry_offset_bps", "better_entry_possible",
        "pnl_bps", "hold_time_bars",
    ]
    for col in cf_metrics:
        if col in merged.columns:
            result_data[col] = merged[col]
        elif f"{col}_cf" in merged.columns:
            result_data[col] = merged[f"{col}_cf"]
    
    result_df = pd.DataFrame(result_data)
    
    # Clean up: drop rows with missing critical features
    initial_len = len(result_df)
    
    # Required features
    critical_features = ["p_long", "timing_quality"]
    result_df = result_df.dropna(subset=critical_features)
    
    dropped = initial_len - len(result_df)
    if dropped > 0:
        log.info(f"Dropped {dropped} rows with missing critical features")
    
    log.info(f"Final dataset: {len(result_df)} rows")
    
    return result_df


def generate_report(
    df: pd.DataFrame,
    dataset_path: Path,
    cf_path: Path,
    output_path: Path,
    report_path: Path,
) -> None:
    """Generate markdown report."""
    lines = []
    
    lines.append("# Entry Timing Dataset V1 – FULLYEAR 2025")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Metadata
    lines.append("## Metadata")
    lines.append("")
    lines.append(f"- **RL Dataset:** {dataset_path}")
    lines.append(f"- **Counterfactual V2 Dataset:** {cf_path}")
    lines.append(f"- **Output Dataset:** {output_path}")
    lines.append(f"- **Total Rows:** {len(df):,}")
    lines.append("")
    
    # Timing quality distribution
    if len(df) > 0 and "timing_quality" in df.columns:
        lines.append("## Timing Quality Distribution")
        lines.append("")
        timing_counts = df["timing_quality"].value_counts()
        lines.append("| Quality | Count | Percentage |")
        lines.append("|---------|-------|------------|")
        for quality, count in timing_counts.items():
            pct = count / len(df) * 100
            lines.append(f"| {quality} | {count:,} | {pct:.1f}% |")
        lines.append("")
        
        # Win rate per timing quality
        lines.append("## Win Rate by Timing Quality")
        lines.append("")
        if "label_profitable_10bps" in df.columns:
            lines.append("| Quality | Count | Win Rate | Avg PnL (bps) |")
            lines.append("|---------|-------|----------|---------------|")
            for quality in timing_counts.index:
                quality_df = df[df["timing_quality"] == quality]
                win_rate = quality_df["label_profitable_10bps"].mean() if "label_profitable_10bps" in quality_df.columns else 0.0
                avg_pnl = quality_df["pnl_bps"].mean() if "pnl_bps" in quality_df.columns else 0.0
                lines.append(f"| {quality} | {len(quality_df):,} | {win_rate:.2%} | {avg_pnl:.2f} |")
            lines.append("")
        
        # MAE before first profit stats
        if "mae_before_first_profit_bps" in df.columns:
            lines.append("## MAE Before First Profit (by Timing Quality)")
            lines.append("")
            lines.append("| Quality | Count | Avg MAE Before Profit | Median MAE |")
            lines.append("|---------|-------|------------------------|------------|")
            for quality in timing_counts.index:
                quality_df = df[df["timing_quality"] == quality]
                mae_before = quality_df["mae_before_first_profit_bps"].dropna()
                if len(mae_before) > 0:
                    avg_mae = mae_before.mean()
                    median_mae = mae_before.median()
                    lines.append(f"| {quality} | {len(mae_before):,} | {avg_mae:.2f} | {median_mae:.2f} |")
                else:
                    lines.append(f"| {quality} | 0 | N/A | N/A |")
            lines.append("")
    
    # Feature availability
    lines.append("## Feature Availability")
    lines.append("")
    feature_cols = [
        "p_long", "spread_bps", "atr_bps",
        "trend_regime", "vol_regime", "session",
        "threshold_slot_numeric", "real_threshold",
    ]
    lines.append("| Feature | Available | Non-Null Count |")
    lines.append("|---------|-----------|----------------|")
    for col in feature_cols:
        available = col in df.columns
        non_null = df[col].notna().sum() if available else 0
        lines.append(f"| {col} | {'Yes' if available else 'No'} | {non_null:,} |")
    lines.append("")
    
    # Notes
    lines.append("---")
    lines.append("")
    lines.append("## Notater")
    lines.append("")
    lines.append("- Dataset er bygget ved å joinne RL-datasettet med shadow counterfactual V2.")
    lines.append("- Kun shadow entries (action_taken == 0) er inkludert.")
    lines.append("- Target: timing_quality (IMMEDIATE_OK | DELAY_BETTER | AVOID_TRADE)")
    lines.append("")
    lines.append(f"*Report generated by `gx1/scripts/build_entry_timing_dataset_v1.py`*")
    
    # Write report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    
    log.info(f"Report saved: {report_path}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build Entry Timing Dataset V1"
    )
    parser.add_argument(
        "--rl_dataset",
        type=Path,
        default=Path("data/rl/sniper_shadow_rl_dataset_FULLYEAR_2025_PARALLEL.parquet"),
        help="Path to RL dataset Parquet file",
    )
    parser.add_argument(
        "--cf_dataset",
        type=Path,
        default=Path("data/rl/shadow_counterfactual_FULLYEAR_2025_V2_shadow_only.parquet"),
        help="Path to shadow counterfactual V2 dataset (shadow-only)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/rl/entry_timing_dataset_FULLYEAR_2025_V1.parquet"),
        help="Output path for timing dataset",
    )
    parser.add_argument(
        "--report_out",
        type=Path,
        default=Path("reports/rl/ENTRY_TIMING_DATASET_FULLYEAR_2025_V1.md"),
        help="Output path for markdown report",
    )
    
    args = parser.parse_args()
    
    try:
        # Load datasets
        rl_df = load_rl_dataset(args.rl_dataset)
        cf_df = load_cf_dataset_v2(args.cf_dataset)
        
        # Join datasets
        timing_df = join_datasets(rl_df, cf_df)
        
        if len(timing_df) == 0:
            log.error("No data after join - check dataset compatibility")
            return 1
        
        # Save output
        log.info(f"Saving timing dataset: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        timing_df.to_parquet(args.output)
        log.info(f"Saved {len(timing_df)} rows to {args.output}")
        
        # Generate report
        generate_report(timing_df, args.rl_dataset, args.cf_dataset, args.output, args.report_out)
        
        log.info("✅ Entry timing dataset build complete!")
        return 0
    
    except Exception as e:
        log.error(f"❌ Dataset build failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())

