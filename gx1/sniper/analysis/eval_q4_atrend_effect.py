"""
Evaluate Q4 A_TREND overlay effect from trade journals.

Reads chunk-level JSON files and reports metrics for:
- Q4 total trades
- Q4 A_TREND trades (where overlay applied)
- Q4 non-A_TREND trades

Outputs text summary and CSV dump.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def load_trades_from_chunks(run_dir: Path) -> List[Dict[str, Any]]:
    """Load all trades from chunk-level JSON files."""
    chunks_dir = run_dir / "parallel_chunks"
    if not chunks_dir.exists():
        print(f"ERROR: parallel_chunks not found in {run_dir}", file=sys.stderr)
        return []
    
    trades = []
    for chunk_dir in sorted(chunks_dir.glob("chunk_*")):
        trades_dir = chunk_dir / "trade_journal" / "trades"
        if not trades_dir.exists():
            continue
        
        for json_file in sorted(trades_dir.glob("*.json")):
            try:
                d = json.loads(json_file.read_text(encoding="utf-8"))
                trades.append(d)
            except Exception as e:
                print(f"WARNING: Failed to load {json_file}: {e}", file=sys.stderr)
    
    return trades


def extract_trade_metrics(trade: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract metrics from a trade JSON."""
    exit_summary = trade.get("exit_summary") or {}
    entry_snapshot = trade.get("entry_snapshot") or {}
    
    # Get PnL
    pnl_bps = exit_summary.get("realized_pnl_bps")
    if pnl_bps is None:
        return None  # Skip incomplete trades
    
    # Get regime info
    trend_regime = entry_snapshot.get("trend_regime")
    vol_regime = entry_snapshot.get("vol_regime")
    atr_bps = entry_snapshot.get("atr_bps")
    spread_bps = entry_snapshot.get("spread_bps")
    session = entry_snapshot.get("session")
    
    # Check for A_TREND overlay
    overlays = entry_snapshot.get("sniper_overlays") or []
    atrend_overlay = None
    for ov in overlays:
        if ov.get("overlay_name") == "Q4_A_TREND_SIZE":
            atrend_overlay = ov
            break
    
    # Classify regime (offline)
    regime_class = None
    try:
        from gx1.sniper.analysis.regime_classifier import classify_regime
        row = {
            "trend_regime": trend_regime,
            "vol_regime": vol_regime,
            "atr_bps": atr_bps,
            "spread_bps": spread_bps,
        }
        regime_class, _ = classify_regime(row)
    except Exception:
        pass
    
    # Get entry/exit times
    entry_time = entry_snapshot.get("entry_time") or trade.get("entry_time")
    exit_time = exit_summary.get("exit_time") or trade.get("exit_time")
    
    return {
        "trade_id": trade.get("trade_id"),
        "entry_time": entry_time,
        "exit_time": exit_time,
        "pnl_bps": float(pnl_bps),
        "trend_regime": trend_regime,
        "vol_regime": vol_regime,
        "atr_bps": atr_bps,
        "spread_bps": spread_bps,
        "session": session,
        "regime_class": regime_class,
        "atrend_overlay_applied": atrend_overlay.get("overlay_applied") if atrend_overlay else False,
        "atrend_size_before": atrend_overlay.get("size_before_units") if atrend_overlay else None,
        "atrend_size_after": atrend_overlay.get("size_after_units") if atrend_overlay else None,
        "atrend_multiplier": atrend_overlay.get("multiplier") if atrend_overlay else None,
        "atrend_reason": atrend_overlay.get("reason") if atrend_overlay else None,
    }


def compute_metrics(df: pd.DataFrame, label: str) -> Dict[str, Any]:
    """Compute metrics for a DataFrame."""
    if len(df) == 0:
        return {
            "label": label,
            "count": 0,
            "winrate": None,
            "mean_pnl_bps": None,
            "median_pnl_bps": None,
            "p90_loss_bps": None,
            "max_loss_bps": None,
            "avg_trades_per_day": None,
        }
    
    pnl = df["pnl_bps"].values
    wins = (pnl > 0).sum()
    losses = (pnl < 0).sum()
    
    # Compute days
    if "entry_time" in df.columns and df["entry_time"].notna().any():
        try:
            entry_times = pd.to_datetime(df["entry_time"], errors="coerce")
            days_span = (entry_times.max() - entry_times.min()).days + 1
            if days_span > 0:
                avg_trades_per_day = len(df) / days_span
            else:
                avg_trades_per_day = len(df)
        except Exception:
            avg_trades_per_day = None
    else:
        avg_trades_per_day = None
    
    return {
        "label": label,
        "count": len(df),
        "winrate": wins / len(df) if len(df) > 0 else None,
        "mean_pnl_bps": float(np.mean(pnl)),
        "median_pnl_bps": float(np.median(pnl)),
        "p90_loss_bps": float(np.percentile(pnl[pnl < 0], 90)) if (pnl < 0).sum() > 0 else None,
        "max_loss_bps": float(np.min(pnl)) if len(pnl) > 0 else None,
        "avg_trades_per_day": avg_trades_per_day,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Q4 A_TREND overlay effect")
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory path")
    parser.add_argument("--output-csv", type=str, help="Output CSV path (optional)")
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"ERROR: Run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loading trades from: {run_dir}")
    trades = load_trades_from_chunks(run_dir)
    print(f"Loaded {len(trades)} trades")
    
    # Extract metrics
    metrics_list = []
    for trade in trades:
        m = extract_trade_metrics(trade)
        if m:
            metrics_list.append(m)
    
    if not metrics_list:
        print("ERROR: No complete trades found", file=sys.stderr)
        sys.exit(1)
    
    df = pd.DataFrame(metrics_list)
    print(f"Extracted {len(df)} complete trades")
    
    # Filter Q4 (from entry_time)
    if "entry_time" in df.columns:
        try:
            df["entry_time_dt"] = pd.to_datetime(df["entry_time"], errors="coerce")
            df = df[df["entry_time_dt"].notna()]
            # Q4 2025: Oct 1 - Dec 31
            q4_start = pd.Timestamp("2025-10-01", tz="UTC")
            q4_end = pd.Timestamp("2025-12-31", tz="UTC")
            df = df[(df["entry_time_dt"] >= q4_start) & (df["entry_time_dt"] <= q4_end)]
        except Exception as e:
            print(f"WARNING: Failed to filter Q4: {e}", file=sys.stderr)
    
    print(f"Q4 trades: {len(df)}")
    
    # Compute metrics
    q4_total = compute_metrics(df, "Q4 Total")
    q4_atrend = compute_metrics(df[df["atrend_overlay_applied"] == True], "Q4 A_TREND (overlay applied)")
    q4_non_atrend = compute_metrics(df[df["atrend_overlay_applied"] != True], "Q4 Non-A_TREND")
    
    # Print summary
    print("\n" + "=" * 80)
    print("Q4 A_TREND Overlay Effect Analysis")
    print("=" * 80)
    print()
    
    for metrics in [q4_total, q4_atrend, q4_non_atrend]:
        print(f"{metrics['label']}:")
        print(f"  Count: {metrics['count']}")
        if metrics['count'] > 0:
            print(f"  Winrate: {metrics['winrate']:.1%}" if metrics['winrate'] is not None else "  Winrate: N/A")
            print(f"  Mean PnL: {metrics['mean_pnl_bps']:.2f} bps" if metrics['mean_pnl_bps'] is not None else "  Mean PnL: N/A")
            print(f"  Median PnL: {metrics['median_pnl_bps']:.2f} bps" if metrics['median_pnl_bps'] is not None else "  Median PnL: N/A")
            print(f"  P90 Loss: {metrics['p90_loss_bps']:.2f} bps" if metrics['p90_loss_bps'] is not None else "  P90 Loss: N/A")
            print(f"  Max Loss: {metrics['max_loss_bps']:.2f} bps" if metrics['max_loss_bps'] is not None else "  Max Loss: N/A")
            if metrics['avg_trades_per_day'] is not None:
                print(f"  Avg Trades/Day: {metrics['avg_trades_per_day']:.1f}")
        print()
    
    # Save CSV if requested
    if args.output_csv:
        df.to_csv(args.output_csv, index=False)
        print(f"Saved detailed metrics to: {args.output_csv}")


if __name__ == "__main__":
    main()

