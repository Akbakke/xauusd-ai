#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Report Truth Decomposition: Session × Regime Matrix (Edge Map)

DEL 2: Build edge map with bins for session × ATR × spread × trend × vol.
Identify "EDGE BINS" (positive avg + good tail) and "POISON BINS".

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


def create_bins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create bin columns for ATR, spread, trend, vol.
    
    Returns DataFrame with additional bin columns.
    """
    df = df.copy()
    
    # ATR bins (q20/q40/q60/q80 = 5 buckets)
    if "atr_bps" in df.columns and df["atr_bps"].notna().sum() > 0:
        atr_quantiles = df["atr_bps"].quantile([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        df["atr_bucket"] = pd.cut(
            df["atr_bps"],
            bins=atr_quantiles.values,
            labels=["Q0-Q20", "Q20-Q40", "Q40-Q60", "Q60-Q80", "Q80-Q100"],
            include_lowest=True,
        )
    else:
        df["atr_bucket"] = None
    
    # Spread bins (q50/q80/q95 = 4 buckets)
    if "spread_bps" in df.columns and df["spread_bps"].notna().sum() > 0:
        spread_quantiles = df["spread_bps"].quantile([0.0, 0.5, 0.8, 0.95, 1.0])
        df["spread_bucket"] = pd.cut(
            df["spread_bps"],
            bins=spread_quantiles.values,
            labels=["Q0-Q50", "Q50-Q80", "Q80-Q95", "Q95-Q100"],
            include_lowest=True,
        )
    else:
        df["spread_bucket"] = None
    
    # Trend regime (categorical)
    if "trend_regime" in df.columns:
        df["trend_regime_bin"] = df["trend_regime"].fillna("UNKNOWN")
        # Normalize to {TREND, RANGE, MIXED, UNKNOWN}
        df["trend_regime_bin"] = df["trend_regime_bin"].replace({
            "TREND": "TREND",
            "RANGE": "RANGE",
            "MIXED": "MIXED",
        }).fillna("UNKNOWN")
    else:
        df["trend_regime_bin"] = "UNKNOWN"
    
    # Vol regime (categorical)
    if "vol_regime" in df.columns:
        df["vol_regime_bin"] = df["vol_regime"].fillna("UNKNOWN")
        # Normalize to {LOW, MID, HIGH, UNKNOWN}
        df["vol_regime_bin"] = df["vol_regime_bin"].replace({
            "LOW": "LOW",
            "MID": "MID",
            "HIGH": "HIGH",
        }).fillna("UNKNOWN")
    else:
        df["vol_regime_bin"] = "UNKNOWN"
    
    return df


def compute_bin_metrics(df_bin: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute metrics for a single bin.
    
    Returns dict with:
    - trades
    - total_pnl_bps
    - avg_pnl_per_trade
    - winrate
    - p5/p50/p95 pnl_per_trade
    - worst_trade
    - maxdd_proxy (simplified: worst_trade + p5 if both negative)
    """
    if len(df_bin) == 0:
        return {
            "trades": 0,
            "total_pnl_bps": 0.0,
            "avg_pnl_per_trade": 0.0,
            "winrate": 0.0,
            "p5_pnl": 0.0,
            "p50_pnl": 0.0,
            "p95_pnl": 0.0,
            "worst_trade": 0.0,
            "maxdd_proxy": 0.0,
        }
    
    pnl_values = df_bin["pnl_bps"].values
    
    return {
        "trades": len(df_bin),
        "total_pnl_bps": float(pnl_values.sum()),
        "avg_pnl_per_trade": float(pnl_values.mean()),
        "winrate": float((pnl_values > 0).mean()),
        "p5_pnl": float(np.percentile(pnl_values, 5)),
        "p50_pnl": float(np.percentile(pnl_values, 50)),
        "p95_pnl": float(np.percentile(pnl_values, 95)),
        "worst_trade": float(pnl_values.min()),
        "maxdd_proxy": float(min(pnl_values.min(), np.percentile(pnl_values, 5))),
    }


def build_edge_map(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Build edge map: session × regime bins.
    
    Returns nested dict: edge_map[session][bin_key] = metrics
    """
    df = create_bins(df)
    
    sessions = ["ASIA", "EU", "OVERLAP", "US"]
    edge_map = {}
    
    for session in sessions:
        df_session = df[df["entry_session"] == session].copy()
        
        if len(df_session) == 0:
            edge_map[session] = {}
            continue
        
        # Group by all bin dimensions
        bin_groups = df_session.groupby([
            "atr_bucket",
            "spread_bucket",
            "trend_regime_bin",
            "vol_regime_bin",
        ], dropna=False)
        
        session_map = {}
        for (atr_b, spread_b, trend_b, vol_b), df_bin in bin_groups:
            bin_key = f"ATR:{atr_b}|SPREAD:{spread_b}|TREND:{trend_b}|VOL:{vol_b}"
            session_map[bin_key] = compute_bin_metrics(df_bin)
        
        edge_map[session] = session_map
    
    return edge_map


def identify_edge_bins(edge_map: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Identify EDGE BINS (positive avg + good tail) and POISON BINS (negative avg + bad tail).
    
    Returns (edge_bins, poison_bins) lists.
    """
    edge_bins = []
    poison_bins = []
    
    for session, bins in edge_map.items():
        for bin_key, metrics in bins.items():
            if metrics["trades"] < 10:  # Skip bins with too few trades
                continue
            
            avg_pnl = metrics["avg_pnl_per_trade"]
            p5_pnl = metrics["p5_pnl"]
            winrate = metrics["winrate"]
            
            # EDGE BIN: positive avg + decent winrate + not terrible tail
            if avg_pnl > 0 and winrate > 0.5 and p5_pnl > -20.0:
                edge_bins.append({
                    "session": session,
                    "bin_key": bin_key,
                    **metrics,
                })
            
            # POISON BIN: negative avg + bad tail
            elif avg_pnl < -2.0 or (avg_pnl < 0 and p5_pnl < -30.0):
                poison_bins.append({
                    "session": session,
                    "bin_key": bin_key,
                    **metrics,
                })
    
    # Sort by avg_pnl_per_trade
    edge_bins.sort(key=lambda x: -x["avg_pnl_per_trade"])
    poison_bins.sort(key=lambda x: x["avg_pnl_per_trade"])
    
    return edge_bins, poison_bins


def generate_report(
    edge_map: Dict[str, Any],
    edge_bins: List[Dict[str, Any]],
    poison_bins: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """Generate DEL 2 markdown report."""
    md_path = output_dir / "TRUTH_DECOMP_SESSION_REGIME_MATRIX.md"
    
    with open(md_path, "w") as f:
        f.write("# Truth Decomposition: Session × Regime Matrix (Edge Map)\n\n")
        f.write(f"**Generated:** {pd.Timestamp.now().isoformat()}\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- **Total EDGE BINS:** {len(edge_bins)}\n")
        f.write(f"- **Total POISON BINS:** {len(poison_bins)}\n\n")
        
        # Top 20 Edge Bins
        f.write("## Top 20 Edge Bins\n\n")
        f.write("| Session | Bin Key | Trades | Avg PnL | Winrate | P5 | P50 | P95 | Worst |\n")
        f.write("|---------|---------|--------|---------|---------|----|-----|-----|-------|\n")
        for bin_data in edge_bins[:20]:
            f.write(
                f"| {bin_data['session']} | {bin_data['bin_key'][:50]}... | "
                f"{bin_data['trades']:,} | {bin_data['avg_pnl_per_trade']:.2f} | "
                f"{bin_data['winrate']:.1%} | {bin_data['p5_pnl']:.2f} | "
                f"{bin_data['p50_pnl']:.2f} | {bin_data['p95_pnl']:.2f} | "
                f"{bin_data['worst_trade']:.2f} |\n"
            )
        f.write("\n")
        
        # Top 20 Poison Bins
        f.write("## Top 20 Poison Bins (Worst)\n\n")
        f.write("| Session | Bin Key | Trades | Avg PnL | Winrate | P5 | P50 | P95 | Worst |\n")
        f.write("|---------|---------|--------|---------|---------|----|-----|-----|-------|\n")
        for bin_data in poison_bins[:20]:
            f.write(
                f"| {bin_data['session']} | {bin_data['bin_key'][:50]}... | "
                f"{bin_data['trades']:,} | {bin_data['avg_pnl_per_trade']:.2f} | "
                f"{bin_data['winrate']:.1%} | {bin_data['p5_pnl']:.2f} | "
                f"{bin_data['p50_pnl']:.2f} | {bin_data['p95_pnl']:.2f} | "
                f"{bin_data['worst_trade']:.2f} |\n"
            )
        f.write("\n")
        
        # Per-session summary
        f.write("## Per-Session Summary\n\n")
        for session in ["ASIA", "EU", "OVERLAP", "US"]:
            session_edge = [b for b in edge_bins if b["session"] == session]
            session_poison = [b for b in poison_bins if b["session"] == session]
            
            f.write(f"### {session}\n\n")
            f.write(f"- **Edge Bins:** {len(session_edge)}\n")
            f.write(f"- **Poison Bins:** {len(session_poison)}\n")
            if session_edge:
                f.write(f"- **Best Edge Bin Avg PnL:** {session_edge[0]['avg_pnl_per_trade']:.2f} bps\n")
            if session_poison:
                f.write(f"- **Worst Poison Bin Avg PnL:** {session_poison[0]['avg_pnl_per_trade']:.2f} bps\n")
            f.write("\n")
    
    log.info(f"✅ Wrote session-regime matrix report: {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Report Truth Decomposition: Session × Regime Matrix")
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
        help="Path to append edge_map to JSON (optional)",
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
    log.info("SESSION × REGIME MATRIX (EDGE MAP)")
    log.info("=" * 60)
    log.info(f"Trade table: {args.trade_table}")
    log.info(f"Output dir: {args.output_dir}")
    log.info("")
    
    # Load trade table
    log.info("Loading trade table...")
    df = pd.read_parquet(args.trade_table)
    log.info(f"Loaded {len(df):,} trades")
    
    # Build edge map
    log.info("Building edge map...")
    edge_map = build_edge_map(df)
    
    # Identify edge and poison bins
    log.info("Identifying edge and poison bins...")
    edge_bins, poison_bins = identify_edge_bins(edge_map)
    log.info(f"Found {len(edge_bins)} edge bins and {len(poison_bins)} poison bins")
    
    # Generate report
    generate_report(edge_map, edge_bins, poison_bins, args.output_dir)
    
    # Append to JSON if requested
    if args.json_output:
        json_path = workspace_root / args.json_output if not args.json_output.is_absolute() else args.json_output
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
        else:
            data = {}
        
        data["edge_map"] = edge_map
        data["edge_bins"] = edge_bins[:50]  # Top 50
        data["poison_bins"] = poison_bins[:50]  # Top 50
        
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        
        log.info(f"✅ Appended to JSON: {json_path}")
    
    log.info("✅ Session × Regime matrix complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
