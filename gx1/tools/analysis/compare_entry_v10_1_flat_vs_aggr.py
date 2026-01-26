#!/usr/bin/env python3
"""
Compare ENTRY_V10.1 FLAT vs AGGR FULLYEAR 2025

Loads trade journals from both variants and generates comparison report.

Usage:
    python -m gx1.tools.analysis.compare_entry_v10_1_flat_vs_aggr \
        --flat-dir data/replay/sniper/entry_v10_1_flat/2025/.../trade_journal \
        --aggr-dir data/replay/sniper/entry_v10_1_aggr/2025/.../trade_journal \
        --output-report reports/rl/entry_v10/ENTRY_V10_1_FLAT_VS_AGGR_2025.md
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_trades_from_journal(journal_dir: Path) -> pd.DataFrame:
    """
    Load trades from trade journal directory.
    
    Supports both:
    - trade_journal_index.csv (merged index)
    - Individual JSON files in trade_journal/trades/
    """
    journal_dir = Path(journal_dir)
    
    # Try trade_journal_index.csv first (preferred)
    index_path = journal_dir / "trade_journal_index.csv"
    if index_path.exists():
        log.info(f"Loading trades from {index_path}")
        df = pd.read_csv(index_path)
        log.info(f"Loaded {len(df):,} trades from index")
        return df
    
    # Fallback: load from individual JSON files
    trades_dir = journal_dir / "trades"
    if trades_dir.exists():
        log.info(f"Loading trades from {trades_dir}")
        trades = []
        for json_file in sorted(trades_dir.glob("*.json")):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    trade = json.load(f)
                    trades.append(trade)
            except Exception as e:
                log.warning(f"Failed to load {json_file}: {e}")
        
        if not trades:
            raise FileNotFoundError(f"No trades found in {trades_dir}")
        
        # Convert to DataFrame
        df = pd.DataFrame(trades)
        log.info(f"Loaded {len(df):,} trades from JSON files")
        return df
    
    raise FileNotFoundError(f"No trade journal found in {journal_dir}")


def compute_metrics(df: pd.DataFrame, variant: str) -> Dict[str, Any]:
    """
    Compute metrics for a variant.
    
    Returns:
        Dictionary with metrics: total_pnl_bps, ev_per_trade_bps, sharpe_like, max_dd, etc.
    """
    # Filter to closed trades only
    if "exit_reason" in df.columns:
        df = df[df["exit_reason"].notna()].copy()
    
    if len(df) == 0:
        log.warning(f"No closed trades found for {variant}")
        return {
            "variant": variant,
            "n_trades": 0,
            "status": "no_trades",
        }
    
    # Extract PnL
    pnl_col = None
    for col in ["pnl_bps", "metrics.pnl_bps"]:
        if col in df.columns:
            pnl_col = col
            break
    
    if pnl_col is None:
        # Try nested structure
        if "metrics" in df.columns:
            df["pnl_bps"] = df["metrics"].apply(
                lambda x: x.get("pnl_bps") if isinstance(x, dict) else None
            )
            pnl_col = "pnl_bps"
        else:
            log.error(f"No PnL column found for {variant}")
            return {
                "variant": variant,
                "n_trades": len(df),
                "status": "no_pnl_column",
            }
    
    pnl = pd.to_numeric(df[pnl_col], errors='coerce').dropna()
    
    if len(pnl) == 0:
        log.warning(f"No valid PnL values for {variant}")
        return {
            "variant": variant,
            "n_trades": len(df),
            "status": "no_valid_pnl",
        }
    
    # Basic metrics
    n_trades = len(pnl)
    wins = (pnl > 0).sum()
    win_rate = (wins / n_trades) * 100.0
    total_pnl_bps = float(pnl.sum())
    ev_per_trade_bps = float(pnl.mean())
    median_pnl_bps = float(pnl.median())
    
    # Sharpe-like ratio (mean / std)
    pnl_std = float(pnl.std())
    sharpe_like = ev_per_trade_bps / pnl_std if pnl_std > 0 else 0.0
    
    # Max drawdown (cumulative)
    cumulative = pnl.cumsum()
    running_max = cumulative.expanding().max()
    drawdown = cumulative - running_max
    max_dd = float(drawdown.min())
    
    # Tail risk (P01, P05)
    p01 = float(np.percentile(pnl, 1))
    p05 = float(np.percentile(pnl, 5))
    p95 = float(np.percentile(pnl, 95))
    p99 = float(np.percentile(pnl, 99))
    
    # Average units per trade (if available)
    avg_units = None
    if "units" in df.columns:
        avg_units = float(df["units"].abs().mean())
    elif "entry" in df.columns:
        # Try nested entry.units
        try:
            units_list = [t.get("entry", {}).get("units") for t in df.to_dict("records")]
            units_list = [u for u in units_list if u is not None]
            if units_list:
                avg_units = float(np.mean(np.abs(units_list)))
        except Exception:
            pass
    
    return {
        "variant": variant,
        "n_trades": n_trades,
        "win_rate_pct": win_rate,
        "total_pnl_bps": total_pnl_bps,
        "ev_per_trade_bps": ev_per_trade_bps,
        "median_pnl_bps": median_pnl_bps,
        "sharpe_like": sharpe_like,
        "max_dd_bps": max_dd,
        "pnl_std_bps": pnl_std,
        "p01_bps": p01,
        "p05_bps": p05,
        "p95_bps": p95,
        "p99_bps": p99,
        "avg_units": avg_units,
        "status": "ok",
    }


def generate_report(flat_metrics: Dict[str, Any], aggr_metrics: Dict[str, Any], output_path: Path) -> None:
    """
    Generate Markdown comparison report.
    """
    lines = [
        "# ENTRY_V10.1 FLAT vs AGGR Comparison (FULLYEAR 2025)",
        "",
        f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        "This report compares ENTRY_V10.1 FLAT (baseline sizing) vs AGGR (aggressive sizing based on edge-buckets).",
        "",
        "### Configuration",
        "",
        "- **Entry Model:** ENTRY_V10.1 HYBRID (variant=v10_1, seq_len=90)",
        "- **FLAT:** Baseline sizing (same as P4.1)",
        "- **AGGR:** Aggressive sizing based on edge-buckets from 2025 FLAT label quality analysis",
        "- **Exit:** ExitCritic V1 + RULE5/RULE6A (same for both variants)",
        "- **Period:** FULLYEAR 2025 (Q1-Q4)",
        "",
        "## Metrics Comparison",
        "",
        "| Metric | FLAT | AGGR | Difference |",
        "|--------|------|------|------------|",
    ]
    
    # Check status
    if flat_metrics.get("status") != "ok":
        lines.append(f"| Status | {flat_metrics.get('status', 'unknown')} | - | - |")
        lines.append("")
        lines.append("⚠️  FLAT variant has issues - cannot generate full comparison.")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return
    
    if aggr_metrics.get("status") != "ok":
        lines.append(f"| Status | - | {aggr_metrics.get('status', 'unknown')} | - |")
        lines.append("")
        lines.append("⚠️  AGGR variant has issues - cannot generate full comparison.")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return
    
    # Number of trades
    n_flat = flat_metrics["n_trades"]
    n_aggr = aggr_metrics["n_trades"]
    n_diff = n_aggr - n_flat
    n_diff_pct = (n_diff / n_flat * 100.0) if n_flat > 0 else 0.0
    lines.append(f"| Number of Trades | {n_flat:,} | {n_aggr:,} | {n_diff:+,} ({n_diff_pct:+.1f}%) |")
    
    # Total PnL
    pnl_flat = flat_metrics["total_pnl_bps"]
    pnl_aggr = aggr_metrics["total_pnl_bps"]
    pnl_diff = pnl_aggr - pnl_flat
    pnl_diff_pct = (pnl_diff / abs(pnl_flat) * 100.0) if pnl_flat != 0 else 0.0
    lines.append(f"| Total PnL (bps) | {pnl_flat:,.1f} | {pnl_aggr:,.1f} | {pnl_diff:+,.1f} ({pnl_diff_pct:+.1f}%) |")
    
    # EV per trade
    ev_flat = flat_metrics["ev_per_trade_bps"]
    ev_aggr = aggr_metrics["ev_per_trade_bps"]
    ev_diff = ev_aggr - ev_flat
    ev_diff_pct = (ev_diff / abs(ev_flat) * 100.0) if ev_flat != 0 else 0.0
    lines.append(f"| EV per Trade (bps) | {ev_flat:.2f} | {ev_aggr:.2f} | {ev_diff:+.2f} ({ev_diff_pct:+.1f}%) |")
    
    # Win rate
    wr_flat = flat_metrics["win_rate_pct"]
    wr_aggr = aggr_metrics["win_rate_pct"]
    wr_diff = wr_aggr - wr_flat
    lines.append(f"| Win Rate (%) | {wr_flat:.1f} | {wr_aggr:.1f} | {wr_diff:+.1f}pp |")
    
    # Sharpe-like
    sharpe_flat = flat_metrics["sharpe_like"]
    sharpe_aggr = aggr_metrics["sharpe_like"]
    sharpe_diff = sharpe_aggr - sharpe_flat
    lines.append(f"| Sharpe-like (mean/std) | {sharpe_flat:.3f} | {sharpe_aggr:.3f} | {sharpe_diff:+.3f} |")
    
    # Max drawdown
    dd_flat = flat_metrics["max_dd_bps"]
    dd_aggr = aggr_metrics["max_dd_bps"]
    dd_diff = dd_aggr - dd_flat
    lines.append(f"| Max Drawdown (bps) | {dd_flat:,.1f} | {dd_aggr:,.1f} | {dd_diff:+,.1f} |")
    
    # Tail risk (P01, P05)
    p01_flat = flat_metrics["p01_bps"]
    p01_aggr = aggr_metrics["p01_bps"]
    p01_diff = p01_aggr - p01_flat
    lines.append(f"| P01 Loss (bps) | {p01_flat:.1f} | {p01_aggr:.1f} | {p01_diff:+.1f} |")
    
    p05_flat = flat_metrics["p05_bps"]
    p05_aggr = aggr_metrics["p05_bps"]
    p05_diff = p05_aggr - p05_flat
    lines.append(f"| P05 Loss (bps) | {p05_flat:.1f} | {p05_aggr:.1f} | {p05_diff:+.1f} |")
    
    # Average units
    if flat_metrics.get("avg_units") and aggr_metrics.get("avg_units"):
        units_flat = flat_metrics["avg_units"]
        units_aggr = aggr_metrics["avg_units"]
        units_diff = units_aggr - units_flat
        units_diff_pct = (units_diff / units_flat * 100.0) if units_flat > 0 else 0.0
        lines.append(f"| Avg Units per Trade | {units_flat:.1f} | {units_aggr:.1f} | {units_diff:+.1f} ({units_diff_pct:+.1f}%) |")
    
    lines.extend([
        "",
        "## Analysis",
        "",
    ])
    
    # Trade count check
    if abs(n_diff) > 0.01 * n_flat:  # More than 1% difference
        lines.append(f"⚠️  **Trade count mismatch:** AGGR has {n_diff:+,} trades ({n_diff_pct:+.1f}%) compared to FLAT.")
        lines.append("   This may indicate that AGGR overlay is dropping trades (size_multiplier=0) or changing entry timing.")
        lines.append("")
    
    # PnL analysis
    if pnl_diff > 0:
        lines.append(f"✅ **AGGR shows higher total PnL:** +{pnl_diff:,.1f} bps ({pnl_diff_pct:+.1f}%)")
    else:
        lines.append(f"❌ **AGGR shows lower total PnL:** {pnl_diff:,.1f} bps ({pnl_diff_pct:+.1f}%)")
    lines.append("")
    
    # Drawdown analysis
    if dd_diff < 0:
        lines.append(f"✅ **AGGR has lower max drawdown:** {dd_diff:,.1f} bps (better)")
    elif dd_diff > 0:
        lines.append(f"⚠️  **AGGR has higher max drawdown:** +{dd_diff:,.1f} bps (worse)")
    else:
        lines.append("ℹ️  **Max drawdown is similar** between variants")
    lines.append("")
    
    # Tail risk analysis
    if p01_diff < 0:
        lines.append(f"✅ **AGGR has lower P01 loss:** {p01_diff:.1f} bps (better tail risk)")
    elif p01_diff > 0:
        lines.append(f"⚠️  **AGGR has higher P01 loss:** +{p01_diff:.1f} bps (worse tail risk)")
    else:
        lines.append("ℹ️  **P01 loss is similar** between variants")
    lines.append("")
    
    if p05_diff < 0:
        lines.append(f"✅ **AGGR has lower P05 loss:** {p05_diff:.1f} bps (better tail risk)")
    elif p05_diff > 0:
        lines.append(f"⚠️  **AGGR has higher P05 loss:** +{p05_diff:.1f} bps (worse tail risk)")
    else:
        lines.append("ℹ️  **P05 loss is similar** between variants")
    lines.append("")
    
    # Sharpe analysis
    if sharpe_diff > 0:
        lines.append(f"✅ **AGGR has better risk-adjusted returns (Sharpe-like):** +{sharpe_diff:.3f}")
    elif sharpe_diff < 0:
        lines.append(f"⚠️  **AGGR has worse risk-adjusted returns (Sharpe-like):** {sharpe_diff:.3f}")
    else:
        lines.append("ℹ️  **Risk-adjusted returns are similar** between variants")
    lines.append("")
    
    lines.extend([
        "## Notes",
        "",
        "- This is an **OFFLINE ONLY** comparison. No live trading implications.",
        "- AGGR sizing is based on edge-buckets from 2025 FLAT label quality analysis.",
        "- Both variants use the same entry model (ENTRY_V10.1 HYBRID) and exit logic.",
        "- Differences in trade count may indicate overlay effects (size_multiplier=0 drops trades).",
        "- **No conclusion on live use** - this report provides data only for human review.",
        "",
    ])
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    log.info(f"✅ Comparison report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare ENTRY_V10.1 FLAT vs AGGR FULLYEAR 2025"
    )
    parser.add_argument(
        "--flat-dir",
        type=Path,
        required=True,
        help="Path to FLAT trade journal directory",
    )
    parser.add_argument(
        "--aggr-dir",
        type=Path,
        required=True,
        help="Path to AGGR trade journal directory",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        required=True,
        help="Output path for Markdown report",
    )
    
    args = parser.parse_args()
    
    # Load trades
    log.info("Loading FLAT trades...")
    try:
        df_flat = load_trades_from_journal(args.flat_dir)
    except Exception as e:
        log.error(f"Failed to load FLAT trades: {e}")
        return 1
    
    log.info("Loading AGGR trades...")
    try:
        df_aggr = load_trades_from_journal(args.aggr_dir)
    except Exception as e:
        log.error(f"Failed to load AGGR trades: {e}")
        return 1
    
    # Compute metrics
    log.info("Computing FLAT metrics...")
    flat_metrics = compute_metrics(df_flat, "FLAT")
    
    log.info("Computing AGGR metrics...")
    aggr_metrics = compute_metrics(df_aggr, "AGGR")
    
    # Generate report
    log.info("Generating comparison report...")
    generate_report(flat_metrics, aggr_metrics, args.output_report)
    
    log.info("✅ Comparison complete!")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

