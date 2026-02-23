#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Report Truth Decomposition: Winner vs Loser Separation

DEL 3: Analyze what makes bad trades bad by comparing winners vs losers per session.
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


def compute_separation_score(winner_values: np.ndarray, loser_values: np.ndarray) -> float:
    """
    Compute separation score: abs(median_winner - median_loser) / pooled_std
    """
    if len(winner_values) == 0 or len(loser_values) == 0:
        return 0.0
    
    median_winner = np.median(winner_values)
    median_loser = np.median(loser_values)
    
    # Pooled std
    pooled_std = np.sqrt(
        (np.var(winner_values) * len(winner_values) + np.var(loser_values) * len(loser_values))
        / (len(winner_values) + len(loser_values))
    )
    
    if pooled_std == 0:
        return 0.0
    
    return abs(median_winner - median_loser) / pooled_std


def analyze_session_separation(df: pd.DataFrame, session: str) -> Dict[str, Any]:
    """
    Analyze winner/loser separation for a single session.
    
    Returns dict with separation scores and distributions.
    """
    df_session = df[df["entry_session"] == session].copy()
    
    if len(df_session) == 0:
        return {
            "session": session,
            "n_trades": 0,
            "n_winners": 0,
            "n_losers": 0,
            "separators": [],
        }
    
    winners = df_session[df_session["pnl_bps"] > 0].copy()
    losers = df_session[df_session["pnl_bps"] <= 0].copy()
    
    separators = []
    
    # Features to analyze
    features = [
        ("atr_bps", "ATR (bps)"),
        ("spread_bps", "Spread (bps)"),
        ("bars_held", "Bars Held"),
        ("max_mae_bps", "Max MAE (bps)"),
    ]
    
    for feature_col, feature_name in features:
        if feature_col not in df_session.columns:
            continue
        
        winner_values = winners[feature_col].dropna().values
        loser_values = losers[feature_col].dropna().values
        
        if len(winner_values) == 0 or len(loser_values) == 0:
            continue
        
        separation = compute_separation_score(winner_values, loser_values)
        
        separators.append({
            "feature": feature_name,
            "feature_col": feature_col,
            "separation_score": float(separation),
            "winner_median": float(np.median(winner_values)),
            "winner_p75": float(np.percentile(winner_values, 75)),
            "winner_p90": float(np.percentile(winner_values, 90)),
            "loser_median": float(np.median(loser_values)),
            "loser_p75": float(np.percentile(loser_values, 75)),
            "loser_p90": float(np.percentile(loser_values, 90)),
        })
    
    # Sort by separation score
    separators.sort(key=lambda x: -x["separation_score"])
    
    return {
        "session": session,
        "n_trades": len(df_session),
        "n_winners": len(winners),
        "n_losers": len(losers),
        "separators": separators,
    }


def generate_report(
    all_analyses: Dict[str, Dict[str, Any]],
    output_dir: Path,
) -> None:
    """Generate DEL 3 markdown report."""
    md_path = output_dir / "TRUTH_DECOMP_WINNER_LOSER_SEPARATION.md"
    
    with open(md_path, "w") as f:
        f.write("# Truth Decomposition: Winner vs Loser Separation\n\n")
        f.write(f"**Generated:** {pd.Timestamp.now().isoformat()}\n\n")
        
        for session in ["ASIA", "EU", "OVERLAP", "US"]:
            analysis = all_analyses.get(session, {})
            
            f.write(f"## {session}\n\n")
            f.write(f"- **Total Trades:** {analysis.get('n_trades', 0):,}\n")
            f.write(f"- **Winners:** {analysis.get('n_winners', 0):,}\n")
            f.write(f"- **Losers:** {analysis.get('n_losers', 0):,}\n\n")
            
            separators = analysis.get("separators", [])
            if separators:
                f.write("### Top 10 Separators\n\n")
                f.write("| Feature | Separation Score | Winner Median | Loser Median | Δ |\n")
                f.write("|---------|------------------|---------------|--------------|---|\n")
                for sep in separators[:10]:
                    delta = sep["winner_median"] - sep["loser_median"]
                    f.write(
                        f"| {sep['feature']} | {sep['separation_score']:.2f} | "
                        f"{sep['winner_median']:.2f} | {sep['loser_median']:.2f} | "
                        f"{delta:+.2f} |\n"
                    )
                f.write("\n")
    
    log.info(f"✅ Wrote winner/loser separation report: {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Report Truth Decomposition: Winner vs Loser Separation")
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
    log.info("WINNER vs LOSER SEPARATION")
    log.info("=" * 60)
    log.info(f"Trade table: {args.trade_table}")
    log.info(f"Output dir: {args.output_dir}")
    log.info("")
    
    # Load trade table
    log.info("Loading trade table...")
    df = pd.read_parquet(args.trade_table)
    log.info(f"Loaded {len(df):,} trades")
    
    # Analyze per session
    all_analyses = {}
    for session in ["ASIA", "EU", "OVERLAP", "US"]:
        log.info(f"Analyzing {session}...")
        all_analyses[session] = analyze_session_separation(df, session)
    
    # Generate report
    generate_report(all_analyses, args.output_dir)
    
    # Append to JSON if requested
    if args.json_output:
        json_path = workspace_root / args.json_output if not args.json_output.is_absolute() else args.json_output
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
        else:
            data = {}
        
        data["top_separators_by_session"] = {
            session: {
                "top_10": analysis.get("separators", [])[:10]
            }
            for session, analysis in all_analyses.items()
        }
        
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        
        log.info(f"✅ Appended to JSON: {json_path}")
    
    log.info("✅ Winner/Loser separation complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
