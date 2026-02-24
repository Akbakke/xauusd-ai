#!/usr/bin/env python3
"""
ENTRY_V10.1 p_long Distribution Analysis

Analyzes the distribution of p_long_v10_1 values from UNGATED replay trades.
Generates quantile statistics and suggests threshold candidates based on label quality.

Usage:
    python -m gx1.tools.analysis.describe_entry_v10_1_p_long_distribution \
        --trade-journal data/replay/sniper/entry_v10_1_flat_ungated/2025/.../trade_journal \
        --label-quality-json data/entry_v10/entry_v10_1_label_quality_2025_ungated.json \
        --output-report reports/rl/entry_v10/ENTRY_V10_1_P_LONG_DISTRIBUTION_2025_UNGATED.md
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def load_trade_journal(trade_journal_path: Path) -> pd.DataFrame:
    """Load trade journal and extract p_long_v10_1 values."""
    print(f"[LOAD] Loading trade journal from: {trade_journal_path}")
    
    # Check if it's a directory (trade journal)
    if trade_journal_path.is_dir():
        # Try trade_journal_index.csv first
        index_csv = trade_journal_path / "trade_journal_index.csv"
        if index_csv.exists():
            print(f"[LOAD] Loading from trade journal index: {index_csv}")
            df = pd.read_csv(index_csv, on_bad_lines='skip', engine='python')
            
            # Try to load full trade data from JSON files
            trades_dir = trade_journal_path / "trades"
            if trades_dir.exists():
                print(f"[LOAD] Loading full trade data from: {trades_dir}")
                trades_data = []
                for json_file in sorted(trades_dir.glob("*.json")):
                    with open(json_file, 'r') as f:
                        trade_data = json.load(f)
                        trades_data.append(trade_data)
                
                if trades_data:
                    # Create DataFrame from trade data and merge with index
                    df_trades = pd.json_normalize(trades_data)
                    if 'trade_id' in df.columns and 'trade_id' in df_trades.columns:
                        df = df.merge(df_trades, on='trade_id', how='left', suffixes=('', '_json'))
                    else:
                        df = pd.concat([df, df_trades], axis=1)
            
            return df
        else:
            raise FileNotFoundError(f"trade_journal_index.csv not found in {trade_journal_path}")
    else:
        # Try as CSV file
        print(f"[LOAD] Loading from CSV file: {trade_journal_path}")
        return pd.read_csv(trade_journal_path, on_bad_lines='skip', engine='python')


def extract_p_long_v10_1(df: pd.DataFrame) -> pd.Series:
    """Extract p_long_v10_1 from trade journal DataFrame."""
    # Try various possible column names
    possible_cols = [
        'entry.p_long_v10_1',
        'entry.p_long_v10',
        'p_long_v10_1',
        'p_long_v10',
        'entry.p_long',
        'p_long',
    ]
    
    for col in possible_cols:
        if col in df.columns:
            p_long = df[col].copy()
            # Remove NaN values
            p_long = p_long.dropna()
            if len(p_long) > 0:
                print(f"[EXTRACT] Found p_long_v10_1 in column: {col} ({len(p_long)} values)")
                return p_long
    
    # Try nested dict access (if entry is a dict column)
    if 'entry' in df.columns:
        print("[EXTRACT] Trying to extract from nested 'entry' column...")
        p_long_list = []
        for entry_val in df['entry']:
            if isinstance(entry_val, dict):
                p_long_val = entry_val.get('p_long_v10_1') or entry_val.get('p_long_v10') or entry_val.get('p_long')
                if p_long_val is not None and not pd.isna(p_long_val):
                    p_long_list.append(p_long_val)
        
        if len(p_long_list) > 0:
            p_long = pd.Series(p_long_list)
            print(f"[EXTRACT] Extracted {len(p_long)} values from nested 'entry' column")
            return p_long
    
    raise ValueError(
        f"Could not find p_long_v10_1 in trade journal. Available columns: {list(df.columns)}"
    )


def compute_quantiles(p_long: pd.Series, quantiles: List[float]) -> Dict[str, float]:
    """Compute quantile values for p_long distribution."""
    quantile_values = {}
    for q in quantiles:
        quantile_values[f"q{q*100:.0f}"] = float(p_long.quantile(q))
    return quantile_values


def suggest_threshold_candidates(
    label_quality_json_path: Optional[Path],
    quantile_values: Dict[str, float],
) -> List[Dict[str, any]]:
    """Suggest threshold candidates based on label quality and distribution."""
    candidates = []
    
    if label_quality_json_path and label_quality_json_path.exists():
        print(f"[THRESHOLD] Loading label quality from: {label_quality_json_path}")
        with open(label_quality_json_path, 'r') as f:
            label_quality = json.load(f)
        
        quantile_stats = label_quality.get("quantile_analysis", {}).get("quantile_stats", [])
        
        if quantile_stats:
            # Find quantiles with positive edge (mean PnL > 0)
            positive_edge_quantiles = [
                q for q in quantile_stats
                if q.get("pnl_mean") is not None and q.get("pnl_mean", 0) > 0
            ]
            
            if positive_edge_quantiles:
                # Use p_long_min from first positive edge quantile as conservative candidate
                first_positive = positive_edge_quantiles[0]
                conservative_threshold = first_positive.get("p_long_min")
                if conservative_threshold is not None:
                    candidates.append({
                        "threshold": conservative_threshold,
                        "quantile": first_positive.get("quantile"),
                        "expected_ev": first_positive.get("pnl_mean", 0),
                        "comment": "≈ startpunkt for positiv edge (basert på første kvantil med positiv mean PnL)",
                    })
                
                # Use median of positive edge quantiles as moderate candidate
                if len(positive_edge_quantiles) > 1:
                    moderate_threshold = np.median([q.get("p_long_min", 0) for q in positive_edge_quantiles])
                    moderate_stats = positive_edge_quantiles[len(positive_edge_quantiles) // 2]
                    candidates.append({
                        "threshold": moderate_threshold,
                        "quantile": moderate_stats.get("quantile"),
                        "expected_ev": moderate_stats.get("pnl_mean", 0),
                        "comment": "moderat konservativ cut (median av positive edge kvantiler)",
                    })
            
            # Use highest quantile with sufficient trades as aggressive candidate
            high_quantiles = [q for q in quantile_stats if q.get("status") == "success"]
            if high_quantiles:
                highest = high_quantiles[-1]
                aggressive_threshold = highest.get("p_long_min")
                if aggressive_threshold is not None:
                    candidates.append({
                        "threshold": aggressive_threshold,
                        "quantile": highest.get("quantile"),
                        "expected_ev": highest.get("pnl_mean", 0),
                        "comment": "mer konservativ cut (høyeste kvantil med tilstrekkelig data)",
                    })
    
    # If no label quality available, suggest based on distribution quantiles
    if not candidates:
        q50 = quantile_values.get("q50", 0.5)
        q70 = quantile_values.get("q70", 0.7)
        q80 = quantile_values.get("q80", 0.8)
        q90 = quantile_values.get("q90", 0.9)
        
        candidates = [
            {
                "threshold": q50,
                "comment": "median (q50) - ingen label quality data tilgjengelig",
            },
            {
                "threshold": q70,
                "comment": "q70 - ingen label quality data tilgjengelig",
            },
            {
                "threshold": q80,
                "comment": "q80 - ingen label quality data tilgjengelig",
            },
            {
                "threshold": q90,
                "comment": "q90 - ingen label quality data tilgjengelig",
            },
        ]
    
    return candidates


def generate_report(
    p_long: pd.Series,
    quantile_values: Dict[str, float],
    threshold_candidates: List[Dict[str, any]],
    output_path: Path,
    label_quality_json_path: Optional[Path] = None,
) -> None:
    """Generate markdown report with p_long distribution and threshold candidates."""
    lines = [
        "# ENTRY_V10.1 p_long Distribution Analysis (UNGATED FULLYEAR 2025)",
        "",
        "**Date:** " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "",
        "**Source:** UNGATED FULLYEAR 2025 replay (no p_long threshold filtering)",
        "",
        "## Summary Statistics",
        "",
        f"- **Total Trades:** {len(p_long):,}",
        f"- **Min p_long:** {p_long.min():.4f}",
        f"- **Max p_long:** {p_long.max():.4f}",
        f"- **Mean p_long:** {p_long.mean():.4f}",
        f"- **Median p_long:** {p_long.median():.4f}",
        f"- **Std p_long:** {p_long.std():.4f}",
        "",
        "## Quantile Distribution",
        "",
        "| Quantile | p_long Value |",
        "|----------|--------------|",
    ]
    
    quantile_order = ["q50", "q70", "q80", "q90", "q95", "q99"]
    for q in quantile_order:
        if q in quantile_values:
            lines.append(f"| {q} | {quantile_values[q]:.4f} |")
    
    lines.extend([
        "",
        "## Threshold Candidates",
        "",
        "Suggested threshold candidates based on label quality analysis:",
        "",
        "| Threshold | Expected EV (bps) | Comment |",
        "|-----------|-------------------|---------|",
    ])
    
    for candidate in threshold_candidates:
        threshold = candidate.get("threshold", 0)
        expected_ev = candidate.get("expected_ev", "N/A")
        comment = candidate.get("comment", "")
        if isinstance(expected_ev, (int, float)):
            expected_ev_str = f"{expected_ev:.2f}"
        else:
            expected_ev_str = str(expected_ev)
        lines.append(f"| {threshold:.4f} | {expected_ev_str} | {comment} |")
    
    lines.extend([
        "",
        "### Notes",
        "",
        "- Threshold candidates are based on quantile analysis from label quality data.",
        "- Expected EV (Expected Value) is the mean PnL for trades in the corresponding quantile.",
        "- Use these candidates to update `min_prob_long` and `entry_gating.p_side_min.long` in FLAT config.",
        "",
    ])
    
    if label_quality_json_path and label_quality_json_path.exists():
        lines.extend([
            f"- Label quality data source: `{label_quality_json_path}`",
            "",
        ])
    
    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"\n[REPORT] Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze ENTRY_V10.1 p_long distribution")
    parser.add_argument(
        "--trade-journal",
        type=Path,
        required=True,
        help="Trade journal directory or CSV file",
    )
    parser.add_argument(
        "--label-quality-json",
        type=Path,
        default=None,
        help="Path to label quality JSON (optional, for threshold suggestions)",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        required=True,
        help="Output report path (Markdown)",
    )
    
    args = parser.parse_args()
    
    # Load trade journal
    df = load_trade_journal(args.trade_journal)
    
    # Extract p_long_v10_1
    p_long = extract_p_long_v10_1(df)
    
    if len(p_long) == 0:
        print("❌ ERROR: No p_long_v10_1 values found in trade journal")
        return 1
    
    print(f"[ANALYZE] Analyzing {len(p_long):,} p_long_v10_1 values...")
    
    # Compute quantiles
    quantiles = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
    quantile_values = compute_quantiles(p_long, quantiles)
    
    print("\n[QUANTILES] Distribution quantiles:")
    for q_name, q_value in sorted(quantile_values.items()):
        print(f"  {q_name}: {q_value:.4f}")
    
    # Suggest threshold candidates
    threshold_candidates = suggest_threshold_candidates(
        args.label_quality_json, quantile_values
    )
    
    # Generate report
    generate_report(
        p_long,
        quantile_values,
        threshold_candidates,
        args.output_report,
        args.label_quality_json,
    )
    
    print("\n✅ Analysis complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

