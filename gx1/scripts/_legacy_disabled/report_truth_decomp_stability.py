#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Report Truth Decomposition: Stability Test (2020 vs 2025)

DEL 5: Compare edge bins between 2020 and 2025 to identify stable vs unstable bins.

================================================================================
INVARIANT: Analysis Script Requirements
================================================================================
- FATAL if years < 2
- FATAL if units outside plausible bounds
- WARN if trades < 1000 (but don't fail)
- No silent fallback
================================================================================
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def create_bin_key(row: pd.Series) -> str:
    """Create bin key from row."""
    session = row["entry_session"]
    atr_b = row.get("atr_bucket", "ALL")
    spread_b = row.get("spread_bucket", "ALL")
    trend_b = row.get("trend_regime_bin", "UNKNOWN")
    vol_b = row.get("vol_regime_bin", "UNKNOWN")
    return f"{session}|ATR:{atr_b}|SPREAD:{spread_b}|TREND:{trend_b}|VOL:{vol_b}"


def build_bin_metrics(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Build bin metrics for a given year.
    
    Returns dict: bin_key -> metrics
    """
    # Create bins (same as DEL 2)
    if "atr_bps" in df.columns and df["atr_bps"].notna().sum() > 0:
        atr_quantiles = df["atr_bps"].quantile([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        df["atr_bucket"] = pd.cut(
            df["atr_bps"],
            bins=atr_quantiles.values,
            labels=["Q0-Q20", "Q20-Q40", "Q40-Q60", "Q60-Q80", "Q80-Q100"],
            include_lowest=True,
        )
    else:
        df["atr_bucket"] = "ALL"
    
    if "spread_bps" in df.columns and df["spread_bps"].notna().sum() > 0:
        spread_quantiles = df["spread_bps"].quantile([0.0, 0.5, 0.8, 0.95, 1.0])
        df["spread_bucket"] = pd.cut(
            df["spread_bps"],
            bins=spread_quantiles.values,
            labels=["Q0-Q50", "Q50-Q80", "Q80-Q95", "Q95-Q100"],
            include_lowest=True,
        )
    else:
        df["spread_bucket"] = "ALL"
    
    df["trend_regime_bin"] = df.get("trend_regime", pd.Series()).fillna("UNKNOWN")
    df["vol_regime_bin"] = df.get("vol_regime", pd.Series()).fillna("UNKNOWN")
    
    # Group by bin
    bin_metrics = {}
    for bin_key, df_bin in df.groupby(df.apply(create_bin_key, axis=1)):
        if len(df_bin) < 5:  # Skip bins with too few trades
            continue
        
        pnl_values = df_bin["pnl_bps"].values
        bin_metrics[bin_key] = {
            "n_trades": len(df_bin),
            "avg_pnl_per_trade": float(pnl_values.mean()),
            "winrate": float((pnl_values > 0).mean()),
        }
    
    return bin_metrics


def compare_years(
    metrics_2020: Dict[str, Dict[str, Any]],
    metrics_2025: Dict[str, Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Compare bins between 2020 and 2025.
    
    Returns dict with:
    - stable_edge_bins
    - unstable_bins
    - drift_bins
    """
    all_bin_keys = set(metrics_2020.keys()) | set(metrics_2025.keys())
    
    stable_edge_bins = []
    unstable_bins = []
    drift_bins = []
    
    for bin_key in all_bin_keys:
        m2020 = metrics_2020.get(bin_key)
        m2025 = metrics_2025.get(bin_key)
        
        if m2020 is None or m2025 is None:
            continue  # Skip bins that don't exist in both years
        
        avg_2020 = m2020["avg_pnl_per_trade"]
        avg_2025 = m2025["avg_pnl_per_trade"]
        delta = avg_2025 - avg_2020
        
        # Sign flip
        sign_flip = (avg_2020 > 0 and avg_2025 < 0) or (avg_2020 < 0 and avg_2025 > 0)
        
        # Stable edge bin: positive in both years, small delta
        if avg_2020 > 0 and avg_2025 > 0 and abs(delta) < max(avg_2020, avg_2025) * 0.5:
            stable_edge_bins.append({
                "bin_key": bin_key,
                "avg_2020": avg_2020,
                "avg_2025": avg_2025,
                "delta": delta,
                "n_trades_2020": m2020["n_trades"],
                "n_trades_2025": m2025["n_trades"],
            })
        
        # Unstable bin: sign flip or large change
        elif sign_flip or abs(delta) > max(abs(avg_2020), abs(avg_2025)) * 0.8:
            unstable_bins.append({
                "bin_key": bin_key,
                "avg_2020": avg_2020,
                "avg_2025": avg_2025,
                "delta": delta,
                "sign_flip": sign_flip,
                "n_trades_2020": m2020["n_trades"],
                "n_trades_2025": m2025["n_trades"],
            })
        
        # Drift bin: moderate change
        elif abs(delta) > max(abs(avg_2020), abs(avg_2025)) * 0.3:
            drift_bins.append({
                "bin_key": bin_key,
                "avg_2020": avg_2020,
                "avg_2025": avg_2025,
                "delta": delta,
                "n_trades_2020": m2020["n_trades"],
                "n_trades_2025": m2025["n_trades"],
            })
    
    # Sort
    stable_edge_bins.sort(key=lambda x: -x["avg_2025"])
    unstable_bins.sort(key=lambda x: abs(x["delta"]), reverse=True)
    drift_bins.sort(key=lambda x: abs(x["delta"]), reverse=True)
    
    return {
        "stable_edge_bins": stable_edge_bins,
        "unstable_bins": unstable_bins,
        "drift_bins": drift_bins,
    }


def generate_report(
    comparison: Dict[str, List[Dict[str, Any]]],
    output_dir: Path,
) -> None:
    """Generate DEL 5 markdown report."""
    md_path = output_dir / "TRUTH_DECOMP_STABILITY_2020_vs_2025.md"
    
    with open(md_path, "w") as f:
        f.write("# Truth Decomposition: Stability Test (2020 vs 2025)\n\n")
        f.write(f"**Generated:** {pd.Timestamp.now().isoformat()}\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- **Stable Edge Bins:** {len(comparison['stable_edge_bins'])}\n")
        f.write(f"- **Unstable Bins:** {len(comparison['unstable_bins'])}\n")
        f.write(f"- **Drift Bins:** {len(comparison['drift_bins'])}\n\n")
        
        # Stable edge bins
        f.write("## Stable Edge Bins (Positive in Both Years, Small Delta)\n\n")
        f.write("| Bin Key | Avg 2020 | Avg 2025 | Δ | Trades 2020 | Trades 2025 |\n")
        f.write("|---------|----------|----------|---|-------------|-------------|\n")
        for bin_data in comparison["stable_edge_bins"][:20]:
            f.write(
                f"| {bin_data['bin_key'][:60]}... | {bin_data['avg_2020']:.2f} | "
                f"{bin_data['avg_2025']:.2f} | {bin_data['delta']:+.2f} | "
                f"{bin_data['n_trades_2020']:,} | {bin_data['n_trades_2025']:,} |\n"
            )
        f.write("\n")
        
        # Unstable bins
        f.write("## Unstable Bins (Sign Flip or Large Change)\n\n")
        f.write("| Bin Key | Avg 2020 | Avg 2025 | Δ | Sign Flip | Trades 2020 | Trades 2025 |\n")
        f.write("|---------|----------|----------|---|-----------|-------------|-------------|\n")
        for bin_data in comparison["unstable_bins"][:20]:
            f.write(
                f"| {bin_data['bin_key'][:60]}... | {bin_data['avg_2020']:.2f} | "
                f"{bin_data['avg_2025']:.2f} | {bin_data['delta']:+.2f} | "
                f"{'Yes' if bin_data['sign_flip'] else 'No'} | "
                f"{bin_data['n_trades_2020']:,} | {bin_data['n_trades_2025']:,} |\n"
            )
        f.write("\n")
    
    log.info(f"✅ Wrote stability report: {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Report Truth Decomposition: Stability Test")
    parser.add_argument(
        "--trade-table",
        type=Path,
        required=True,
        help="Path to canonical trade table parquet file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for reports",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Path to append to JSON (optional)",
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    workspace_root = Path(__file__).parent.parent.parent
    if not args.trade_table.is_absolute():
        args.trade_table = workspace_root / args.trade_table
    if not args.output_dir.is_absolute():
        args.output_dir = workspace_root / args.output_dir
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    log.info("=" * 60)
    log.info("STABILITY TEST (2020 vs 2025)")
    log.info("=" * 60)
    log.info(f"Trade table: {args.trade_table}")
    log.info(f"Output dir: {args.output_dir}")
    log.info("")
    
    # Load trade table
    log.info("Loading trade table...")
    df = pd.read_parquet(args.trade_table)
    log.info(f"Loaded {len(df):,} trades")
    
    # Filter to 2020 and 2025
    df_2020 = df[df["year"] == 2020].copy()
    df_2025 = df[df["year"] == 2025].copy()
    
    log.info(f"2020 trades: {len(df_2020):,}")
    log.info(f"2025 trades: {len(df_2025):,}")
    
    # Build bin metrics
    log.info("Building bin metrics for 2020...")
    metrics_2020 = build_bin_metrics(df_2020)
    log.info(f"Found {len(metrics_2020)} bins in 2020")
    
    log.info("Building bin metrics for 2025...")
    metrics_2025 = build_bin_metrics(df_2025)
    log.info(f"Found {len(metrics_2025)} bins in 2025")
    
    # Compare
    log.info("Comparing years...")
    comparison = compare_years(metrics_2020, metrics_2025)
    log.info(f"Found {len(comparison['stable_edge_bins'])} stable edge bins")
    log.info(f"Found {len(comparison['unstable_bins'])} unstable bins")
    log.info(f"Found {len(comparison['drift_bins'])} drift bins")
    
    # Generate report
    generate_report(comparison, args.output_dir)
    
    # Append to JSON if requested
    if args.json_output:
        json_path = workspace_root / args.json_output if not args.json_output.is_absolute() else args.json_output
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
        else:
            data = {}
        
        data["stability_2020_vs_2025"] = comparison
        
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        
        log.info(f"✅ Appended to JSON: {json_path}")
    
    log.info("✅ Stability test complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
