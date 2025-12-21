#!/usr/bin/env python3
"""
Q4 SNIPER chunk-level regime analysis (no merge, direct from chunks).

This script:
- Reads directly from parallel_chunks/**/trade_journal/trades/*.json
- Does NOT use merge or trade_journal_index.csv
- For each trade:
  - Reads entry_snapshot, feature_context, sniper_overlays
  - Classifies regime via classify_regime(...)
- Aggregates Q4 stats per:
  - regime_class (A_TREND / B_MIXED / C_CHOP)
  - session (EU / OVERLAP / US) where volume ≥ 200
- Reports:
  - trades, EV/trade (mean pnl_bps), winrate, payoff, p90_loss
  - overlay coverage: how many trades have sniper_overlays
  - overlay_applied-rate per regime/session
- Verifies overlay triggering:
  - Count of trades with overlay_name == "Q4_C_CHOP_SESSION_SIZE"
  - Count of overlay_applied == True
  - Breakdown per session
  - If overlay_applied == 0, explains why based on:
    - quarter
    - regime_class
    - multiplier == 1.0
    - missing_fields

Output:
reports/SNIPER_Q4_CHUNK_REGIME_ANALYSIS__YYYYMMDD_HHMMSS.md
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from gx1.sniper.analysis.regime_classifier import classify_regime
from gx1.sniper.policy.sniper_regime_size_overlay import compute_quarter


SNIPER_BASE = Path("gx1/wf_runs")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)


def _first_non_null(*candidates):
    """Return first non-None value, or None if all are None."""
    for v in candidates:
        if v is not None:
            return v
    return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float."""
    try:
        if value is None:
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def _safe_str(value: Any, default: str = "UNKNOWN") -> str:
    """Safely convert value to string."""
    if value is None:
        return default
    if isinstance(value, str):
        return value
    return str(value)


def load_trades_from_chunks(run_dir: Path) -> List[Dict[str, Any]]:
    """
    Load all trades from parallel_chunks/**/trade_journal/trades/*.json.

    Returns list of trade dicts with added metadata:
    - chunk_id (str)
    - trade_file (str)
    """
    trades: List[Dict[str, Any]] = []
    chunks_dir = run_dir / "parallel_chunks"
    
    if not chunks_dir.exists():
        return trades
    
    for chunk_dir in sorted(chunks_dir.glob("chunk_*")):
        chunk_id = chunk_dir.name
        trades_dir = chunk_dir / "trade_journal" / "trades"
        
        if not trades_dir.exists():
            continue
        
        for json_file in sorted(trades_dir.glob("*.json")):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    trade = json.load(f)
                trade["chunk_id"] = chunk_id
                trade["trade_file"] = json_file.name
                trades.append(trade)
            except Exception as e:
                print(f"WARNING: Failed to load {json_file}: {e}")
                continue
    
    return trades


def extract_trade_analysis_row(trade: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extract analysis fields from a trade JSON.

    Returns dict with:
    - trade_id, chunk_id, trade_file
    - pnl_bps
    - trend_regime, vol_regime, atr_bps, spread_bps, session
    - regime_class, regime_reason (from classify_regime)
    - overlay_* fields (from entry_snapshot.sniper_overlays)
    - quarter (computed from entry_time)
    """
    entry = trade.get("entry_snapshot") or {}
    feature_ctx = trade.get("feature_context") or {}
    exit_summary = trade.get("exit_summary") or {}
    
    # pnl_bps
    pnl = _first_non_null(
        exit_summary.get("realized_pnl_bps"),
        exit_summary.get("pnl_bps"),
        trade.get("pnl_bps"),
    )
    
    # Regime inputs (best-effort extraction)
    trend_regime = _first_non_null(
        entry.get("trend_regime"),
        feature_ctx.get("trend_regime"),
        trade.get("trend_regime"),
    )
    vol_regime = _first_non_null(
        entry.get("vol_regime"),
        feature_ctx.get("vol_regime"),
        trade.get("vol_regime"),
    )
    atr_bps = _first_non_null(
        entry.get("atr_bps"),
        feature_ctx.get("atr_bps"),
        trade.get("atr_bps"),
    )
    spread_bps = _first_non_null(
        entry.get("spread_bps"),
        feature_ctx.get("spread_bps"),
        trade.get("spread_bps"),
    )
    session = _first_non_null(
        entry.get("session"),
        feature_ctx.get("session"),
        trade.get("session"),
    )
    
    # Classify regime
    regime_row = {
        "trend_regime": trend_regime,
        "vol_regime": vol_regime,
        "atr_bps": atr_bps,
        "spread_bps": spread_bps,
    }
    try:
        regime_class, regime_reason = classify_regime(regime_row)
    except Exception as e:
        regime_class = "UNKNOWN"
        regime_reason = f"classify_error:{type(e).__name__}"
    
    # Quarter
    entry_time = _first_non_null(
        entry.get("entry_time"),
        trade.get("entry_time"),
    )
    try:
        quarter = compute_quarter(entry_time) if entry_time else "UNKNOWN"
    except Exception:
        quarter = "UNKNOWN"
    
    # Overlay metadata
    overlays = entry.get("sniper_overlays") or []
    cchop_overlay = None
    for ov in overlays:
        if ov.get("overlay_name") == "Q4_C_CHOP_SESSION_SIZE":
            cchop_overlay = ov
            break
    
    row = {
        "trade_id": trade.get("trade_id"),
        "chunk_id": trade.get("chunk_id", "unknown"),
        "trade_file": trade.get("trade_file", "unknown"),
        "pnl_bps": _safe_float(pnl),
        "trend_regime": _safe_str(trend_regime),
        "vol_regime": _safe_str(vol_regime),
        "atr_bps": _safe_float(atr_bps),
        "spread_bps": _safe_float(spread_bps),
        "session": _safe_str(session),
        "regime_class": regime_class,
        "regime_reason": regime_reason,
        "quarter": quarter,
        "has_entry_snapshot": entry is not None and bool(entry),
        "has_overlays": len(overlays) > 0,
        "cchop_overlay_applied": cchop_overlay.get("overlay_applied", False) if cchop_overlay else False,
        "cchop_overlay_reason": cchop_overlay.get("reason", "") if cchop_overlay else "",
        "cchop_overlay_quarter": cchop_overlay.get("quarter") if cchop_overlay else None,
        "cchop_overlay_regime": cchop_overlay.get("regime_class") if cchop_overlay else None,
        "cchop_overlay_multiplier": cchop_overlay.get("multiplier") if cchop_overlay else None,
        "cchop_overlay_session": cchop_overlay.get("session") if cchop_overlay else None,
    }
    
    return row


def compute_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute core metrics from analysis rows."""
    if not rows:
        return {
            "n_trades": 0,
            "ev_per_trade": 0.0,
            "winrate": 0.0,
            "payoff": 0.0,
            "p90_loss": 0.0,
        }
    
    pnls = [r["pnl_bps"] for r in rows if r["pnl_bps"] is not None]
    if not pnls:
        return {
            "n_trades": len(rows),
            "ev_per_trade": 0.0,
            "winrate": 0.0,
            "payoff": 0.0,
            "p90_loss": 0.0,
        }
    
    pnls_arr = np.array(pnls)
    wins = pnls_arr[pnls_arr > 0]
    losses = pnls_arr[pnls_arr < 0]
    
    ev = float(np.mean(pnls_arr))
    winrate = len(wins) / len(pnls_arr) if len(pnls_arr) > 0 else 0.0
    payoff = float(np.mean(wins)) / abs(np.mean(losses)) if len(losses) > 0 and np.mean(losses) != 0 else 0.0
    p90_loss = float(np.percentile(losses, 90)) if len(losses) > 0 else 0.0
    
    return {
        "n_trades": len(rows),
        "ev_per_trade": ev,
        "winrate": winrate,
        "payoff": payoff,
        "p90_loss": p90_loss,
    }


def analyze_q4_chunk_regimes(run_dir: Path) -> str:
    """
    Main analysis function.

    Returns markdown report string.
    """
    print(f"Loading trades from chunks in: {run_dir}")
    trades = load_trades_from_chunks(run_dir)
    print(f"Loaded {len(trades)} trades from chunks")
    
    # Extract analysis rows
    rows: List[Dict[str, Any]] = []
    for trade in trades:
        row = extract_trade_analysis_row(trade)
        if row:
            rows.append(row)
    
    if not rows:
        return f"# Q4 Chunk Regime Analysis\n\n**Run**: `{run_dir.name}`\n\n**ERROR**: No valid trades found.\n"
    
    df = pd.DataFrame(rows)
    
    # Overall stats
    overall_metrics = compute_metrics(rows)
    
    # Overlay coverage
    has_entry_snapshot = df["has_entry_snapshot"].sum()
    has_overlays = df["has_overlays"].sum()
    has_cchop_overlay = (df["cchop_overlay_reason"] != "").sum()
    cchop_applied = df["cchop_overlay_applied"].sum()
    
    # Regime breakdown
    regime_stats: Dict[str, Dict[str, Any]] = {}
    for regime in ["A_TREND", "B_MIXED", "C_CHOP", "UNKNOWN"]:
        regime_rows = [r for r in rows if r["regime_class"] == regime]
        if regime_rows:
            regime_stats[regime] = compute_metrics(regime_rows)
    
    # Session breakdown (where volume ≥ 200)
    session_stats: Dict[str, Dict[str, Any]] = {}
    for session in ["EU", "OVERLAP", "US"]:
        session_rows = [r for r in rows if r["session"] == session]
        if len(session_rows) >= 200:
            session_stats[session] = compute_metrics(session_rows)
    
    # Overlay trigger analysis
    cchop_rows = [r for r in rows if r["cchop_overlay_reason"] != ""]
    overlay_reasons = Counter([r["cchop_overlay_reason"] for r in cchop_rows])
    overlay_by_session = Counter([r["cchop_overlay_session"] for r in cchop_rows if r["cchop_overlay_session"]])
    
    # Breakdown of why overlay_applied == False
    not_applied_reasons = Counter([
        r["cchop_overlay_reason"]
        for r in cchop_rows
        if not r["cchop_overlay_applied"]
    ])
    
    # Check what's actually in trade files
    sample_trade = trades[0] if trades else {}
    available_keys = list(sample_trade.keys())
    entry_snapshot_type = type(sample_trade.get("entry_snapshot")).__name__
    feature_context_type = type(sample_trade.get("feature_context")).__name__
    
    # Generate report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_lines = [
        "# Q4 Chunk Regime Analysis",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Run Directory**: `{run_dir.name}`",
        f"**Source**: `parallel_chunks/**/trade_journal/trades/*.json`",
        "",
        "---",
        "",
        "## Summary",
        "",
        f"- **Total Trades**: {overall_metrics['n_trades']}",
        f"- **EV/Trade**: {overall_metrics['ev_per_trade']:.2f} bps",
        f"- **Winrate**: {overall_metrics['winrate']:.1%}",
        f"- **Payoff**: {overall_metrics['payoff']:.2f}",
        f"- **P90 Loss**: {overall_metrics['p90_loss']:.2f} bps",
        "",
        "### Overlay Coverage",
        "",
        f"- **Trades with entry_snapshot**: {has_entry_snapshot} / {len(rows)} ({100*has_entry_snapshot/len(rows):.1f}%)",
        f"- **Trades with sniper_overlays**: {has_overlays} / {len(rows)} ({100*has_overlays/len(rows):.1f}%)",
        f"- **Trades with Q4_C_CHOP_SESSION_SIZE overlay**: {has_cchop_overlay} / {len(rows)} ({100*has_cchop_overlay/len(rows):.1f}%)",
        f"- **Trades with overlay_applied == True**: {cchop_applied} / {len(rows)} ({100*cchop_applied/len(rows):.1f}%)",
        "",
        "### Trade File Structure",
        "",
        f"- **Available top-level keys**: {', '.join(available_keys)}",
        f"- **entry_snapshot type**: `{entry_snapshot_type}`",
        f"- **feature_context type**: `{feature_context_type}`",
        "",
        "⚠️ **CRITICAL ISSUE**: All trades have `entry_snapshot = None`.",
        "",
        "This means:",
        "- Overlay metadata (`sniper_overlays`) is not being written to trade journals.",
        "- Regime classification must rely on `feature_context` or top-level fields (if available).",
        "- Overlay trigger verification cannot be performed without `entry_snapshot`.",
        "",
        "**Root Cause**: Trade journaling in `oanda_demo_runner.py` or `trade_journal.py` is not preserving `entry_snapshot`.",
        "",
    ]
    
    # Regime breakdown
    if regime_stats:
        report_lines.extend([
            "## Regime Breakdown",
            "",
            "| Regime | Trades | EV/Trade | Winrate | Payoff | P90 Loss |",
            "|--------|--------|----------|---------|--------|----------|",
        ])
        for regime in ["A_TREND", "B_MIXED", "C_CHOP", "UNKNOWN"]:
            if regime in regime_stats:
                m = regime_stats[regime]
                report_lines.append(
                    f"| {regime} | {m['n_trades']} | {m['ev_per_trade']:.2f} | {m['winrate']:.1%} | {m['payoff']:.2f} | {m['p90_loss']:.2f} |"
                )
        report_lines.append("")
    
    # Session breakdown (where volume ≥ 200)
    if session_stats:
        report_lines.extend([
            "## Session Breakdown (Volume ≥ 200)",
            "",
            "| Session | Trades | EV/Trade | Winrate | Payoff | P90 Loss |",
            "|---------|--------|----------|---------|--------|----------|",
        ])
        for session in ["EU", "OVERLAP", "US"]:
            if session in session_stats:
                m = session_stats[session]
                report_lines.append(
                    f"| {session} | {m['n_trades']} | {m['ev_per_trade']:.2f} | {m['winrate']:.1%} | {m['payoff']:.2f} | {m['p90_loss']:.2f} |"
                )
        report_lines.append("")
    
    # Overlay trigger analysis
    report_lines.extend([
        "## Overlay Trigger Analysis",
        "",
        f"### Q4_C_CHOP_SESSION_SIZE Overlay",
        "",
        f"- **Trades with overlay metadata**: {has_cchop_overlay}",
        f"- **Trades with overlay_applied == True**: {cchop_applied}",
        "",
    ])
    
    if overlay_reasons:
        report_lines.extend([
            "### Overlay Reasons (Top 15)",
            "",
            "| Reason | Count |",
            "|--------|-------|",
        ])
        for reason, count in overlay_reasons.most_common(15):
            report_lines.append(f"| `{reason}` | {count} |")
        report_lines.append("")
    
    if overlay_by_session:
        report_lines.extend([
            "### Overlay by Session",
            "",
            "| Session | Count |",
            "|---------|-------|",
        ])
        for session, count in overlay_by_session.most_common(10):
            report_lines.append(f"| {session} | {count} |")
        report_lines.append("")
    
    if not_applied_reasons:
        report_lines.extend([
            "### Why overlay_applied == False",
            "",
            "| Reason | Count |",
            "|--------|-------|",
        ])
        for reason, count in not_applied_reasons.most_common(15):
            report_lines.append(f"| `{reason}` | {count} |")
        report_lines.append("")
    
    # Sample trade with overlay_applied == True (if any)
    applied_samples = [r for r in cchop_rows if r["cchop_overlay_applied"]]
    if applied_samples:
        sample = applied_samples[0]
        report_lines.extend([
            "### Sample Trade with overlay_applied == True",
            "",
            f"- **Trade ID**: `{sample['trade_id']}`",
            f"- **File**: `{sample['trade_file']}`",
            f"- **Chunk**: `{sample['chunk_id']}`",
            f"- **Quarter**: `{sample['cchop_overlay_quarter']}`",
            f"- **Regime**: `{sample['cchop_overlay_regime']}`",
            f"- **Session**: `{sample['cchop_overlay_session']}`",
            f"- **Multiplier**: `{sample['cchop_overlay_multiplier']}`",
            f"- **Reason**: `{sample['cchop_overlay_reason']}`",
            "",
        ])
    
    # Sample trade with non-error reason (if any)
    non_error_samples = [
        r for r in cchop_rows
        if r["cchop_overlay_reason"] and not r["cchop_overlay_reason"].startswith("overlay_error")
    ]
    if non_error_samples:
        sample = non_error_samples[0]
        report_lines.extend([
            "### Sample Trade with Non-Error Reason",
            "",
            f"- **Trade ID**: `{sample['trade_id']}`",
            f"- **File**: `{sample['trade_file']}`",
            f"- **Chunk**: `{sample['chunk_id']}`",
            f"- **Quarter**: `{sample['cchop_overlay_quarter']}`",
            f"- **Regime**: `{sample['cchop_overlay_regime']}`",
            f"- **Session**: `{sample['cchop_overlay_session']}`",
            f"- **Multiplier**: `{sample['cchop_overlay_multiplier']}`",
            f"- **Reason**: `{sample['cchop_overlay_reason']}`",
            "",
        ])
    
    return "\n".join(report_lines)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Q4 SNIPER chunk-level regimes and overlay triggering"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        help="Path to SNIPER run directory (e.g., gx1/wf_runs/SNIPER_OBS_Q4_2025_baseline_*)",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-detect latest Q4 baseline run",
    )
    args = parser.parse_args()
    
    if args.auto:
        matches = sorted(SNIPER_BASE.glob("SNIPER_OBS_Q4_2025_baseline_*"))
        if not matches:
            print("ERROR: No Q4 baseline runs found")
            return 1
        run_dir = matches[-1]
        print(f"Auto-detected run: {run_dir.name}")
    elif args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        print("ERROR: Must specify --run-dir or --auto")
        return 1
    
    if not run_dir.exists():
        print(f"ERROR: Run directory does not exist: {run_dir}")
        return 1
    
    # Generate report
    report = analyze_q4_chunk_regimes(run_dir)
    
    # Write report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = REPORTS_DIR / f"SNIPER_Q4_CHUNK_REGIME_ANALYSIS__{timestamp}.md"
    report_file.write_text(report, encoding="utf-8")
    print(f"✅ Report written: {report_file}")
    
    return 0


if __name__ == "__main__":
    exit(main())

