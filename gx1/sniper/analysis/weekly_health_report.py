"""
Weekly Health Report for SNIPER trading system.

Generates health metrics from trade journals:
- Trades/day
- Regime distribution
- Overlay trigger rates
- PnL summary
- Top worst trades
- Operational alarms
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def load_trades_from_journals(journal_root: Path, start_date: str, end_date: str) -> List[Dict[str, Any]]:
    """Load trades from chunk-level JSON files within date range."""
    trades = []
    
    # Priority 1: parallel_chunks (preferred source)
    chunks_dir = journal_root / "parallel_chunks"
    if chunks_dir.exists():
        for chunk_dir in sorted(chunks_dir.glob("chunk_*")):
            trades_dir = chunk_dir / "trade_journal" / "trades"
            if not trades_dir.exists():
                continue
            for json_file in sorted(trades_dir.glob("*.json")):
                try:
                    d = json.loads(json_file.read_text(encoding="utf-8"))
                    trades.append(d)
                except Exception:
                    pass
    
    # Priority 2: direct trades directory (fallback)
    if not trades:  # Only use if parallel_chunks had no trades
        trades_dir = journal_root / "trade_journal" / "trades"
        if trades_dir.exists():
            for json_file in sorted(trades_dir.glob("*.json")):
                try:
                    d = json.loads(json_file.read_text(encoding="utf-8"))
                    trades.append(d)
                except Exception:
                    pass
    
    # Filter by date range (if dates provided)
    if start_date and end_date:
        start_dt = pd.Timestamp(start_date, tz="UTC")
        end_dt = pd.Timestamp(end_date, tz="UTC")
        # Add one day to end_date to make it inclusive
        end_dt = end_dt + pd.Timedelta(days=1)
        
        filtered_trades = []
        for trade in trades:
            entry_snapshot = trade.get("entry_snapshot") or {}
            entry_time = entry_snapshot.get("entry_time") or trade.get("entry_time")
            if entry_time:
                try:
                    entry_dt = pd.Timestamp(entry_time)
                    # Ensure timezone-aware comparison
                    if entry_dt.tzinfo is None:
                        entry_dt = entry_dt.tz_localize("UTC")
                    else:
                        entry_dt = entry_dt.tz_convert("UTC")
                    
                    # Include trades where entry_time is within range (inclusive)
                    if start_dt <= entry_dt < end_dt:
                        filtered_trades.append(trade)
                except Exception:
                    # Skip trades with invalid timestamps
                    continue
            else:
                # If no entry_time, include trade (let downstream handle it)
                filtered_trades.append(trade)
        
        return filtered_trades
    else:
        # No date filtering
        return trades


def extract_trade_metrics(trade: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract metrics from trade JSON."""
    exit_summary = trade.get("exit_summary") or {}
    entry_snapshot = trade.get("entry_snapshot") or {}
    
    pnl_bps = exit_summary.get("realized_pnl_bps")
    if pnl_bps is None:
        return None  # Skip incomplete trades
    
    # Regime classification
    regime_class = None
    try:
        from gx1.sniper.analysis.regime_classifier import classify_regime
        row = {
            "trend_regime": entry_snapshot.get("trend_regime"),
            "vol_regime": entry_snapshot.get("vol_regime"),
            "atr_bps": entry_snapshot.get("atr_bps"),
            "spread_bps": entry_snapshot.get("spread_bps"),
        }
        regime_class, _ = classify_regime(row)
    except Exception:
        pass
    
    # Overlay triggers
    overlays = entry_snapshot.get("sniper_overlays") or []
    overlay_triggers = {}
    for ov in overlays:
        overlay_name = ov.get("overlay_name", "unknown")
        overlay_triggers[overlay_name] = ov.get("overlay_applied", False)
    
    return {
        "trade_id": trade.get("trade_id"),
        "entry_time": entry_snapshot.get("entry_time") or trade.get("entry_time"),
        "exit_time": exit_summary.get("exit_time") or trade.get("exit_time"),
        "pnl_bps": float(pnl_bps),
        "regime_class": regime_class,
        "trend_regime": entry_snapshot.get("trend_regime"),
        "vol_regime": entry_snapshot.get("vol_regime"),
        "session": entry_snapshot.get("session"),
        "atr_bps": entry_snapshot.get("atr_bps"),
        "spread_bps": entry_snapshot.get("spread_bps"),
        "exit_reason": exit_summary.get("exit_reason"),
        "overlay_triggers": overlay_triggers,
        "overlays": overlays,
    }


def compute_regime_distribution(df: pd.DataFrame) -> Dict[str, float]:
    """Compute regime distribution percentages."""
    regime_counts = Counter(df["regime_class"].dropna())
    total = len(df)
    if total == 0:
        return {}
    return {regime: (count / total) * 100.0 for regime, count in regime_counts.items()}


def compute_overlay_trigger_rates(df: pd.DataFrame) -> Dict[str, float]:
    """Compute overlay trigger rates."""
    overlay_names = set()
    for triggers in df["overlay_triggers"]:
        overlay_names.update(triggers.keys())
    
    rates = {}
    for overlay_name in overlay_names:
        triggered = sum(1 for triggers in df["overlay_triggers"] if triggers.get(overlay_name, False))
        total = len(df)
        rates[overlay_name] = (triggered / total * 100.0) if total > 0 else 0.0
    
    return rates


def check_alarms(df: pd.DataFrame, baseline_regime_dist: Optional[Dict[str, float]] = None) -> List[str]:
    """Check operational alarms."""
    alarms = []
    
    # Coverage alarm: trades/day
    if "entry_time" in df.columns and df["entry_time"].notna().any():
        try:
            entry_times = pd.to_datetime(df["entry_time"], errors="coerce")
            days_span = (entry_times.max() - entry_times.min()).days + 1
            if days_span > 0:
                trades_per_day = len(df) / days_span
                if trades_per_day < 50:
                    alarms.append(f"ALARM: COVERAGE trades_per_day={trades_per_day:.1f} (threshold=50)")
        except Exception:
            pass
    
    # Tail risk alarm
    pnl = df["pnl_bps"].values
    if len(pnl) > 0:
        p95_loss = np.percentile(pnl[pnl < 0], 95) if (pnl < 0).sum() > 0 else None
        max_loss = np.min(pnl)
        
        if p95_loss is not None and p95_loss < -50.0:
            alarms.append(f"ALARM: TAIL_RISK p95_loss={p95_loss:.2f} bps (threshold=-50.0)")
        
        if max_loss < -200.0:
            alarms.append(f"ALARM: TAIL_RISK max_loss={max_loss:.2f} bps (threshold=-200.0)")
    
    # Regime drift alarm
    if baseline_regime_dist:
        current_regime_dist = compute_regime_distribution(df)
        for regime, baseline_pct in baseline_regime_dist.items():
            current_pct = current_regime_dist.get(regime, 0.0)
            drift = abs(current_pct - baseline_pct)
            if drift > 10.0:  # 10% threshold
                alarms.append(
                    f"ALARM: REGIME_DRIFT {regime} baseline={baseline_pct:.1f}% "
                    f"current={current_pct:.1f}% drift={drift:.1f}% (threshold=10%)"
                )
    
    # NO-TRADE alarm (check overlay triggers for disable actions)
    no_trade_count = 0
    for _, row in df.iterrows():
        overlays = row.get("overlays", [])
        for ov in overlays:
            if ov.get("size_after_units") == 0 and ov.get("overlay_applied"):
                no_trade_count += 1
                break
    
    no_trade_pct = (no_trade_count / len(df) * 100.0) if len(df) > 0 else 0.0
    if no_trade_pct > 5.0:  # 5% threshold
        alarms.append(f"ALARM: COVERAGE no_trade_pct={no_trade_pct:.1f}% (threshold=5%)")
    
    return alarms


def main():
    parser = argparse.ArgumentParser(description="Generate weekly health report from trade journals")
    parser.add_argument("--journal-root", type=str, required=True, help="Root directory containing trade journals")
    parser.add_argument("--start-date", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output-csv", type=str, help="Output CSV path (optional)")
    parser.add_argument("--baseline-regime-dist", type=str, help="Baseline regime distribution JSON (optional)")
    parser.add_argument("--archive-dir", type=str, help="Archive directory for reports (reports/weekly/YYYY-MM-DD/)")
    parser.add_argument("--baseline-commit", type=str, help="Baseline commit hash (optional)")
    parser.add_argument("--policy-name", type=str, help="Policy name (optional)")
    args = parser.parse_args()
    
    journal_root = Path(args.journal_root)
    if not journal_root.exists():
        print(f"ERROR: Journal root not found: {journal_root}", file=sys.stderr)
        sys.exit(1)
    
    # Load baseline regime distribution if provided
    baseline_regime_dist = None
    if args.baseline_regime_dist:
        try:
            with open(args.baseline_regime_dist) as f:
                baseline_regime_dist = json.load(f)
        except Exception as e:
            print(f"WARNING: Failed to load baseline regime distribution: {e}", file=sys.stderr)
    
    print(f"Loading trades from: {journal_root}")
    print(f"Date range: {args.start_date} to {args.end_date}")
    
    trades = load_trades_from_journals(journal_root, args.start_date, args.end_date)
    if not trades:
        print("ERROR: No trades found", file=sys.stderr)
        sys.exit(1)
    
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
    
    # Compute metrics
    entry_times = pd.to_datetime(df["entry_time"], errors="coerce")
    valid_times = entry_times.dropna()
    if len(valid_times) > 0:
        days_span = (valid_times.max() - valid_times.min()).days + 1
        trades_per_day = len(df) / days_span if days_span > 0 else 0.0
    else:
        days_span = 0
        trades_per_day = 0.0
    
    regime_dist = compute_regime_distribution(df)
    overlay_rates = compute_overlay_trigger_rates(df)
    
    pnl = df["pnl_bps"].values
    mean_pnl = float(np.mean(pnl))
    median_pnl = float(np.median(pnl))
    p90_loss = float(np.percentile(pnl[pnl < 0], 90)) if (pnl < 0).sum() > 0 else None
    p95_loss = float(np.percentile(pnl[pnl < 0], 95)) if (pnl < 0).sum() > 0 else None
    max_loss = float(np.min(pnl))
    winrate = (pnl > 0).sum() / len(pnl) if len(pnl) > 0 else 0.0
    
    # Top 5 worst trades
    worst_trades = df.nsmallest(5, "pnl_bps")
    
    # Check alarms
    alarms = check_alarms(df, baseline_regime_dist)
    
    # Print report
    print("\n" + "=" * 80)
    print("SNIPER Weekly Health Report")
    print("=" * 80)
    print(f"Period: {args.start_date} to {args.end_date}")
    print(f"Total trades: {len(df)}")
    print(f"Days: {days_span:.1f}")
    print(f"Trades/day: {trades_per_day:.1f}")
    print()
    
    print("Regime Distribution:")
    for regime, pct in sorted(regime_dist.items()):
        print(f"  {regime}: {pct:.1f}%")
    print()
    
    print("Overlay Trigger Rates:")
    for overlay_name, rate in sorted(overlay_rates.items()):
        print(f"  {overlay_name}: {rate:.1f}%")
    print()
    
    print("PnL Summary:")
    print(f"  Mean PnL: {mean_pnl:.2f} bps")
    print(f"  Median PnL: {median_pnl:.2f} bps")
    print(f"  P90 Loss: {p90_loss:.2f} bps" if p90_loss else "  P90 Loss: N/A")
    print(f"  P95 Loss: {p95_loss:.2f} bps" if p95_loss else "  P95 Loss: N/A")
    print(f"  Max Loss: {max_loss:.2f} bps")
    print(f"  Winrate: {winrate:.1%}")
    print()
    
    print("Top 5 Worst Trades:")
    for idx, row in worst_trades.iterrows():
        print(f"  {row['entry_time']}: {row['pnl_bps']:.2f} bps | "
              f"regime={row['regime_class']} session={row['session']} | "
              f"exit={row['exit_reason']}")
    print()
    
    if alarms:
        print("Operational Alarms:")
        for alarm in alarms:
            print(f"  {alarm}")
        print()
    else:
        print("Operational Alarms: None")
        print()
    
    # Save CSV if requested
    csv_path = args.output_csv
    if csv_path:
        df.to_csv(csv_path, index=False)
        print(f"Saved detailed metrics to: {csv_path}")
    
    # Archive report if requested
    if args.archive_dir:
        archive_dir = Path(args.archive_dir)
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary.txt
        summary_path = archive_dir / "summary.txt"
        with open(summary_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("SNIPER Weekly Health Report\n")
            f.write("=" * 80 + "\n")
            f.write(f"Period: {args.start_date} to {args.end_date}\n")
            if args.baseline_commit:
                f.write(f"Baseline Commit: {args.baseline_commit}\n")
            if args.policy_name:
                f.write(f"Policy: {args.policy_name}\n")
            f.write(f"Total trades: {len(df)}\n")
            f.write(f"Days: {days_span:.1f}\n")
            f.write(f"Trades/day: {trades_per_day:.1f}\n")
            f.write("\n")
            
            f.write("Regime Distribution:\n")
            for regime, pct in sorted(regime_dist.items()):
                f.write(f"  {regime}: {pct:.1f}%\n")
            f.write("\n")
            
            f.write("Overlay Trigger Rates:\n")
            for overlay_name, rate in sorted(overlay_rates.items()):
                f.write(f"  {overlay_name}: {rate:.1f}%\n")
            f.write("\n")
            
            f.write("PnL Summary:\n")
            f.write(f"  Mean PnL: {mean_pnl:.2f} bps\n")
            f.write(f"  Median PnL: {median_pnl:.2f} bps\n")
            f.write(f"  P90 Loss: {p90_loss:.2f} bps\n" if p90_loss else "  P90 Loss: N/A\n")
            f.write(f"  P95 Loss: {p95_loss:.2f} bps\n" if p95_loss else "  P95 Loss: N/A\n")
            f.write(f"  Max Loss: {max_loss:.2f} bps\n")
            f.write(f"  Winrate: {winrate:.1%}\n")
            f.write("\n")
            
            if alarms:
                f.write("Operational Alarms:\n")
                for alarm in alarms:
                    f.write(f"  {alarm}\n")
            else:
                f.write("Operational Alarms: None\n")
            f.write("\n")
        
        # Save metrics.csv
        metrics_path = archive_dir / "metrics.csv"
        df.to_csv(metrics_path, index=False)
        
        print(f"Archived report to: {archive_dir}")
    
    # Exit with error code if alarms present
    if alarms:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()

