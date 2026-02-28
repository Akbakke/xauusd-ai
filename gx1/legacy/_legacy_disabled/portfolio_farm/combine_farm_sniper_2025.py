#!/usr/bin/env python3
"""
Portfolio 2025: Combine FARM + SNIPER full-year replays.

Deterministic combination of FARM (Asia) and SNIPER (EU/London/NY) trades
into a single portfolio journal with risk stacking.

Router rules:
- ASIA session → FARM engine
- EU/LONDON/NY/OVERLAP sessions → SNIPER engine
- Conflicts (same entry_time): SNIPER wins
- Risk stacking: max_open_trades (default=1) prevents overlapping positions
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Session routing rules
FARM_SESSIONS = {"ASIA"}
SNIPER_SESSIONS = {"EU", "LONDON", "NY", "OVERLAP"}


def load_trades_from_journal(journal_root: Path) -> List[Dict[str, Any]]:
    """Load all trades from chunk-level JSON files or CSV."""
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
                except Exception as e:
                    print(f"WARNING: Failed to load {json_file}: {e}", file=sys.stderr)
    
    # Priority 2: direct trades directory (fallback)
    if not trades:
        trades_dir = journal_root / "trade_journal" / "trades"
        if trades_dir.exists():
            for json_file in sorted(trades_dir.glob("*.json")):
                try:
                    d = json.loads(json_file.read_text(encoding="utf-8"))
                    trades.append(d)
                except Exception as e:
                    print(f"WARNING: Failed to load {json_file}: {e}", file=sys.stderr)
    
    # Priority 3: CSV format (for FARM runs that use trade_log.csv)
    if not trades:
        csv_files = list(journal_root.glob("trade_log*.csv")) + list(journal_root.glob("*merged.csv"))
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                if len(df) == 0:
                    continue
                # Convert CSV rows to trade dict format
                for _, row in df.iterrows():
                    trade_dict = {
                        "trade_id": str(row.get("trade_id", "")),
                        "entry_snapshot": {
                            "trade_id": str(row.get("trade_id", "")),
                            "entry_time": str(row.get("entry_time", "")),
                            "session": str(row.get("session", row.get("session_entry", "UNKNOWN"))),
                            "side": str(row.get("side", "")),
                            "entry_price": float(row.get("entry_price", 0.0)) if pd.notna(row.get("entry_price")) else None,
                            "vol_regime": str(row.get("vol_regime", row.get("vol_regime_entry", ""))),
                            "trend_regime": str(row.get("trend_regime", "")),
                        },
                        "exit_summary": {
                            "exit_time": str(row.get("exit_time", "")) if pd.notna(row.get("exit_time")) else None,
                            "exit_price": float(row.get("exit_price", 0.0)) if pd.notna(row.get("exit_price")) else None,
                            "realized_pnl_bps": float(row.get("pnl_bps", 0.0)) if pd.notna(row.get("pnl_bps")) else None,
                            "exit_reason": str(row.get("exit_reason", row.get("primary_exit_reason", ""))),
                        }
                    }
                    trades.append(trade_dict)
                print(f"Loaded {len(df)} trades from CSV: {csv_file.name}")
                break  # Use first CSV found
            except Exception as e:
                print(f"WARNING: Failed to load CSV {csv_file}: {e}", file=sys.stderr)
    
    return trades


def normalize_trade(trade_json: Dict[str, Any], engine: str) -> Optional[Dict[str, Any]]:
    """
    Normalize trade JSON to portfolio format.
    
    Returns None if trade is incomplete (no exit_summary).
    """
    entry_snapshot = trade_json.get("entry_snapshot") or {}
    exit_summary = trade_json.get("exit_summary")
    
    if not exit_summary:
        return None  # Skip incomplete trades
    
    entry_time_str = entry_snapshot.get("entry_time")
    exit_time_str = exit_summary.get("exit_time")
    
    if not entry_time_str or not exit_time_str:
        return None
    
    # Parse timestamps
    try:
        entry_time = pd.to_datetime(entry_time_str)
        exit_time = pd.to_datetime(exit_time_str)
    except Exception:
        return None
    
    session = entry_snapshot.get("session", "UNKNOWN")
    pnl_bps = exit_summary.get("realized_pnl_bps")
    
    if pnl_bps is None:
        # Try alternative field names
        pnl_bps = exit_summary.get("pnl_bps")
    
    if pnl_bps is None:
        return None  # Skip trades without PnL
    
    # Extract overlay metadata
    overlays = entry_snapshot.get("sniper_overlays") or []
    overlay_names = [ov.get("overlay_name", "") for ov in overlays if ov.get("overlay_applied")]
    
    return {
        "trade_id": trade_json.get("trade_id", ""),
        "entry_time": entry_time,
        "exit_time": exit_time,
        "entry_time_str": entry_time_str,
        "exit_time_str": exit_time_str,
        "session": session,
        "engine": engine,
        "side": entry_snapshot.get("side", "unknown"),
        "entry_price": entry_snapshot.get("entry_price"),
        "exit_price": exit_summary.get("exit_price"),
        "pnl_bps": float(pnl_bps),
        "vol_regime": entry_snapshot.get("vol_regime"),
        "trend_regime": entry_snapshot.get("trend_regime"),
        "overlays": overlay_names,
        "exit_reason": exit_summary.get("exit_reason", ""),
        "source_trade_id": trade_json.get("trade_id", ""),
    }


def route_trade(trade: Dict[str, Any], farm_trades: List[Dict], sniper_trades: List[Dict]) -> Tuple[Optional[Dict], str]:
    """
    Route trade based on session.
    
    Returns:
        (selected_trade, reason)
        reason can be: "FARM", "SNIPER", "CONFLICT_SNIPER_WINS", "NO_MATCH"
    """
    session = trade.get("session", "").upper()
    entry_time = trade.get("entry_time")
    
    # Find matching trades in both engines
    farm_match = None
    sniper_match = None
    
    for ft in farm_trades:
        if ft.get("entry_time") == entry_time:
            farm_match = ft
            break
    
    for st in sniper_trades:
        if st.get("entry_time") == entry_time:
            sniper_match = st
            break
    
    # Router logic
    if session in FARM_SESSIONS:
        if farm_match:
            return farm_match, "FARM"
        elif sniper_match:
            # SNIPER trade in ASIA session - log conflict but use SNIPER
            return sniper_match, "CONFLICT_SNIPER_WINS"
        else:
            return None, "NO_MATCH"
    
    elif session in SNIPER_SESSIONS:
        if sniper_match:
            return sniper_match, "SNIPER"
        elif farm_match:
            # FARM trade in SNIPER session - log conflict but use SNIPER
            return sniper_match, "CONFLICT_SNIPER_WINS"
        else:
            return None, "NO_MATCH"
    
    else:
        # Unknown session - prefer SNIPER if available
        if sniper_match:
            return sniper_match, "SNIPER"
        elif farm_match:
            return farm_match, "FARM"
        else:
            return None, "NO_MATCH"


def apply_risk_stacking(
    routed_trades: List[Dict[str, Any]], 
    max_open_trades: Optional[int] = 1
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Apply risk stacking: only allow max_open_trades concurrent positions.
    
    If max_open_trades is None or <= 0, no stacking is applied (all trades accepted).
    
    Returns:
        (accepted_trades, dropped_trades)
    """
    if max_open_trades is None or max_open_trades <= 0:
        return routed_trades, []
    
    accepted = []
    dropped = []
    open_trades: List[Dict[str, Any]] = []
    
    # Sort by entry_time
    sorted_trades = sorted(routed_trades, key=lambda t: t.get("entry_time"))
    
    for trade in sorted_trades:
        entry_time = trade.get("entry_time")
        exit_time = trade.get("exit_time")
        
        # Close trades that have exited before this entry
        open_trades = [
            ot for ot in open_trades 
            if ot.get("exit_time") > entry_time
        ]
        
        # Check if we can open new trade
        if len(open_trades) >= max_open_trades:
            dropped_trade = trade.copy()
            dropped_trade["dropped_reason"] = "max_open_trades_limit"
            dropped.append(dropped_trade)
            continue
        
        # Accept trade
        open_trades.append(trade)
        accepted.append(trade)
    
    return accepted, dropped


def compute_metrics(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute portfolio metrics."""
    if not trades:
        return {
            "total_trades": 0,
            "mean_pnl": 0.0,
            "median_pnl": 0.0,
            "p90_loss": None,
            "p95_loss": None,
            "max_loss": 0.0,
            "winrate": 0.0,
        }
    
    pnl_array = np.array([t.get("pnl_bps", 0.0) for t in trades])
    
    losses = pnl_array[pnl_array < 0]
    
    metrics = {
        "total_trades": len(trades),
        "mean_pnl": float(np.mean(pnl_array)),
        "median_pnl": float(np.median(pnl_array)),
        "p90_loss": float(np.percentile(losses, 90)) if len(losses) > 0 else None,
        "p95_loss": float(np.percentile(losses, 95)) if len(losses) > 0 else None,
        "max_loss": float(np.min(pnl_array)),
        "winrate": float((pnl_array > 0).sum() / len(pnl_array)) if len(pnl_array) > 0 else 0.0,
    }
    
    return metrics


def compute_engine_metrics(trades: List[Dict[str, Any]], engine: str) -> Dict[str, Any]:
    """Compute metrics for specific engine."""
    engine_trades = [t for t in trades if t.get("engine") == engine]
    return compute_metrics(engine_trades)


def main():
    parser = argparse.ArgumentParser(
        description="Combine FARM + SNIPER full-year replays into portfolio"
    )
    parser.add_argument(
        "--farm-run-dir",
        type=str,
        required=True,
        help="FARM full-year run directory (wf_runs/PORTFOLIO_2025_FARM_FULLYEAR/...)"
    )
    parser.add_argument(
        "--sniper-run-dir",
        type=str,
        required=True,
        help="SNIPER full-year run directory (wf_runs/PORTFOLIO_2025_SNIPER_FULLYEAR/...)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/portfolio/2025",
        help="Output directory for portfolio reports"
    )
    def parse_max_open_trades(value: str) -> Optional[int]:
        if value.lower() in ("none", "unlimited", "0"):
            return None
        return int(value)
    
    parser.add_argument(
        "--max-open-trades",
        type=parse_max_open_trades,
        default=1,
        help="Maximum concurrent open trades (default: 1). Use 'none' or 'unlimited' for no limit."
    )
    args = parser.parse_args()
    
    farm_run_dir = Path(args.farm_run_dir)
    sniper_run_dir = Path(args.sniper_run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load trades
    print(f"Loading FARM trades from: {farm_run_dir}")
    farm_trades_raw = load_trades_from_journal(farm_run_dir)
    print(f"Loaded {len(farm_trades_raw)} FARM trade JSONs")
    
    print(f"Loading SNIPER trades from: {sniper_run_dir}")
    sniper_trades_raw = load_trades_from_journal(sniper_run_dir)
    print(f"Loaded {len(sniper_trades_raw)} SNIPER trade JSONs")
    
    # Normalize trades
    farm_trades = []
    for t in farm_trades_raw:
        normalized = normalize_trade(t, "FARM")
        if normalized:
            farm_trades.append(normalized)
    
    sniper_trades = []
    for t in sniper_trades_raw:
        normalized = normalize_trade(t, "SNIPER")
        if normalized:
            sniper_trades.append(normalized)
    
    print(f"Normalized: {len(farm_trades)} FARM trades, {len(sniper_trades)} SNIPER trades")
    
    # Route trades
    routed_trades = []
    conflicts = []
    routing_stats = defaultdict(int)
    
    # Build lookup by entry_time for efficient matching
    farm_by_time = {t.get("entry_time"): t for t in farm_trades}
    sniper_by_time = {t.get("entry_time"): t for t in sniper_trades}
    
    # Process all unique entry times
    all_entry_times = set(farm_by_time.keys()) | set(sniper_by_time.keys())
    
    # Track trade_ids to prevent duplicates
    seen_trade_ids = set()
    
    for entry_time in sorted(all_entry_times):
        farm_trade = farm_by_time.get(entry_time)
        sniper_trade = sniper_by_time.get(entry_time)
        
        # Determine session (prefer SNIPER if both exist)
        session = None
        if sniper_trade:
            session = sniper_trade.get("session", "").upper()
        elif farm_trade:
            session = farm_trade.get("session", "").upper()
        
        if not session:
            continue
        
        # Route
        selected_trade = None
        route_reason = None
        
        if session in FARM_SESSIONS:
            if farm_trade:
                selected_trade = farm_trade
                route_reason = "FARM"
            elif sniper_trade:
                # Conflict: SNIPER trade in ASIA session
                selected_trade = sniper_trade
                route_reason = "CONFLICT_SNIPER_WINS"
                conflicts.append({
                    "entry_time": entry_time,
                    "session": session,
                    "selected": "SNIPER",
                    "reason": "SNIPER trade in ASIA session (SNIPER wins)"
                })
        
        elif session in SNIPER_SESSIONS:
            if sniper_trade:
                selected_trade = sniper_trade
                route_reason = "SNIPER"
            elif farm_trade:
                # Conflict: FARM trade in SNIPER session
                selected_trade = farm_trade
                route_reason = "CONFLICT_FARM_IN_SNIPER_SESSION"
                conflicts.append({
                    "entry_time": entry_time,
                    "session": session,
                    "selected": "FARM",
                    "reason": "FARM trade in SNIPER session (SNIPER preferred but FARM only available)"
                })
        
        else:
            # Unknown session - prefer SNIPER
            if sniper_trade:
                selected_trade = sniper_trade
                route_reason = "SNIPER"
            elif farm_trade:
                selected_trade = farm_trade
                route_reason = "FARM"
        
        # Add trade if selected and not duplicate
        if selected_trade:
            trade_id = selected_trade.get("trade_id")
            if trade_id and trade_id not in seen_trade_ids:
                routed_trades.append(selected_trade)
                seen_trade_ids.add(trade_id)
                routing_stats[route_reason] = routing_stats.get(route_reason, 0) + 1
            elif trade_id:
                # Duplicate trade_id - skip
                routing_stats["DUPLICATE_SKIPPED"] = routing_stats.get("DUPLICATE_SKIPPED", 0) + 1
    
    print(f"Routed {len(routed_trades)} trades")
    print(f"Conflicts: {len(conflicts)}")
    print(f"Routing stats: {dict(routing_stats)}")
    
    # Apply risk stacking
    accepted_trades, dropped_trades = apply_risk_stacking(
        routed_trades, 
        max_open_trades=args.max_open_trades
    )
    
    print(f"After risk stacking: {len(accepted_trades)} accepted, {len(dropped_trades)} dropped")
    
    # Compute metrics
    portfolio_metrics = compute_metrics(accepted_trades)
    farm_metrics = compute_engine_metrics(accepted_trades, "FARM")
    sniper_metrics = compute_engine_metrics(accepted_trades, "SNIPER")
    
    # Compute days span
    if accepted_trades:
        entry_times = [t.get("entry_time") for t in accepted_trades]
        min_time = min(entry_times)
        max_time = max(entry_times)
        days_span = (max_time - min_time).days + 1
        trades_per_day = len(accepted_trades) / days_span if days_span > 0 else 0.0
    else:
        days_span = 0
        trades_per_day = 0.0
    
    # Write portfolio trades JSONL
    jsonl_path = output_dir / "portfolio_trades.jsonl"
    with open(jsonl_path, "w") as f:
        for trade in accepted_trades:
            # Convert datetime to ISO string for JSON
            trade_export = trade.copy()
            trade_export["entry_time"] = trade_export["entry_time_str"]
            trade_export["exit_time"] = trade_export["exit_time_str"]
            trade_export.pop("entry_time_str", None)
            trade_export.pop("exit_time_str", None)
            f.write(json.dumps(trade_export, default=str) + "\n")
    
    print(f"Wrote portfolio trades to: {jsonl_path}")
    
    # Write portfolio metrics CSV
    csv_path = output_dir / "portfolio_metrics.csv"
    df = pd.DataFrame(accepted_trades)
    if not df.empty:
        # Convert datetime columns to strings for CSV
        df["entry_time"] = df["entry_time"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        df["exit_time"] = df["exit_time"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        df.to_csv(csv_path, index=False)
        print(f"Wrote portfolio metrics CSV to: {csv_path}")
    
    # Write summary
    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("Portfolio 2025: FARM + SNIPER Combined\n")
        f.write("=" * 80 + "\n")
        f.write(f"Period: {min_time.strftime('%Y-%m-%d')} to {max_time.strftime('%Y-%m-%d')}\n")
        f.write(f"Days: {days_span}\n")
        f.write(f"Max open trades: {args.max_open_trades}\n")
        f.write("\n")
        
        f.write("Portfolio Total:\n")
        f.write(f"  Total trades: {portfolio_metrics['total_trades']}\n")
        f.write(f"  Trades/day: {trades_per_day:.1f}\n")
        f.write(f"  Mean PnL: {portfolio_metrics['mean_pnl']:.2f} bps\n")
        f.write(f"  Median PnL: {portfolio_metrics['median_pnl']:.2f} bps\n")
        f.write(f"  P90 Loss: {portfolio_metrics['p90_loss']:.2f} bps\n" if portfolio_metrics['p90_loss'] else "  P90 Loss: N/A\n")
        f.write(f"  P95 Loss: {portfolio_metrics['p95_loss']:.2f} bps\n" if portfolio_metrics['p95_loss'] else "  P95 Loss: N/A\n")
        f.write(f"  Max Loss: {portfolio_metrics['max_loss']:.2f} bps\n")
        f.write(f"  Winrate: {portfolio_metrics['winrate']:.1%}\n")
        f.write("\n")
        
        f.write("Engine Breakdown:\n")
        f.write(f"  FARM:\n")
        f.write(f"    Trades: {farm_metrics['total_trades']}\n")
        f.write(f"    Mean PnL: {farm_metrics['mean_pnl']:.2f} bps\n")
        f.write(f"    Median PnL: {farm_metrics['median_pnl']:.2f} bps\n")
        f.write(f"    P90 Loss: {farm_metrics['p90_loss']:.2f} bps\n" if farm_metrics['p90_loss'] else "    P90 Loss: N/A\n")
        f.write(f"    Winrate: {farm_metrics['winrate']:.1%}\n")
        f.write(f"  SNIPER:\n")
        f.write(f"    Trades: {sniper_metrics['total_trades']}\n")
        f.write(f"    Mean PnL: {sniper_metrics['mean_pnl']:.2f} bps\n")
        f.write(f"    Median PnL: {sniper_metrics['median_pnl']:.2f} bps\n")
        f.write(f"    P90 Loss: {sniper_metrics['p90_loss']:.2f} bps\n" if sniper_metrics['p90_loss'] else "    P90 Loss: N/A\n")
        f.write(f"    Winrate: {sniper_metrics['winrate']:.1%}\n")
        f.write("\n")
        
        f.write("Routing Stats:\n")
        for reason, count in sorted(routing_stats.items()):
            f.write(f"  {reason}: {count}\n")
        f.write("\n")
        
        f.write(f"Conflicts: {len(conflicts)}\n")
        if conflicts:
            f.write("  (SNIPER wins in conflicts)\n")
        f.write("\n")
        
        f.write(f"Dropped Stats:\n")
        f.write(f"  Dropped due to exposure: {len(dropped_trades)}\n")
        if dropped_trades:
            dropped_by_session = defaultdict(int)
            for dt in dropped_trades:
                dropped_by_session[dt.get("session", "UNKNOWN")] += 1
            f.write("  Dropped by session:\n")
            for session, count in sorted(dropped_by_session.items()):
                f.write(f"    {session}: {count}\n")
        f.write("\n")
    
    print(f"Wrote summary to: {summary_path}")
    print("\n" + "=" * 80)
    print("Portfolio combination complete!")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Total trades: {portfolio_metrics['total_trades']}")
    print(f"FARM trades: {farm_metrics['total_trades']}")
    print(f"SNIPER trades: {sniper_metrics['total_trades']}")
    print(f"Conflicts: {len(conflicts)}")
    print(f"Dropped: {len(dropped_trades)}")


if __name__ == "__main__":
    main()

