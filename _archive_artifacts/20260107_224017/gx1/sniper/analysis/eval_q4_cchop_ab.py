#!/usr/bin/env python3
"""
A/B evaluation of Q4 C_CHOP overlay impact (chunk-JSON only, no merge dependency).

This script compares two Q4 baseline runs:
- Run A: overlay OFF (control)
- Run B: overlay ON (treatment)

Both runs are read directly from parallel_chunks/**/trade_journal/trades/*.json
(no dependency on merge or trade_journal_index.csv).

Metrics reported:
- Q4 total: trades, EV, winrate, payoff, p90_loss
- Q4 × C_CHOP total
- Q4 × C_CHOP × US (expected largest effect)
- Effective risk reduction proxy (p90_loss comparison)

Output:
reports/SNIPER_Q4_CCHOP_OVERLAY_AB__YYYYMMDD_HHMMSS.md
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from gx1.sniper.analysis.regime_classifier import classify_regime
from gx1.sniper.policy.sniper_regime_size_overlay import compute_quarter


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


def extract_trade_row(trade: Dict[str, Any], run_label: str) -> Optional[Dict[str, Any]]:
    """
    Extract analysis row from trade JSON.
    
    Returns dict with:
    - pnl_bps
    - session, trend_regime, vol_regime, atr_bps, spread_bps
    - regime_class (from classify_regime)
    - overlay metadata (if available)
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
    
    # Regime inputs
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
    
    # Overlay metadata (if available)
    overlays = entry.get("sniper_overlays") or []
    cchop_overlay = None
    for ov in overlays:
        if ov.get("overlay_name") == "Q4_C_CHOP_SESSION_SIZE":
            cchop_overlay = ov
            break
    
    row = {
        "run_label": run_label,
        "trade_id": trade.get("trade_id"),
        "pnl_bps": _safe_float(pnl),
        "session": _safe_str(session),
        "trend_regime": _safe_str(trend_regime),
        "vol_regime": _safe_str(vol_regime),
        "atr_bps": _safe_float(atr_bps),
        "spread_bps": _safe_float(spread_bps),
        "regime_class": regime_class,
        "quarter": quarter,
        "overlay_applied": cchop_overlay.get("overlay_applied", False) if cchop_overlay else False,
        "overlay_multiplier": cchop_overlay.get("multiplier") if cchop_overlay else None,
        "size_before_units": cchop_overlay.get("size_before_units") if cchop_overlay else None,
        "size_after_units": cchop_overlay.get("size_after_units") if cchop_overlay else None,
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


def verify_overlay_sanity(df_b: pd.DataFrame) -> Dict[str, Any]:
    """
    Verify overlay activation sanity in run B (overlay ON).
    
    Checks:
    - overlay_applied count and distribution per session
    - size_after_units ≈ round(size_before_units * multiplier) sign-preserving
    """
    sanity: Dict[str, Any] = {}
    
    # Overlay applied count
    applied_true = df_b["overlay_applied"].sum()
    sanity["applied_true_count"] = int(applied_true)
    sanity["applied_true_pct"] = (applied_true / len(df_b) * 100.0) if len(df_b) > 0 else 0.0
    
    # Distribution per session
    applied_by_session = df_b[df_b["overlay_applied"] == True].groupby("session").size().to_dict()
    sanity["applied_by_session"] = applied_by_session
    
    # Size verification (only for trades where overlay was applied)
    applied_df = df_b[df_b["overlay_applied"] == True].copy()
    size_mismatches = 0
    size_checks = 0
    
    for _, row in applied_df.iterrows():
        size_before = row.get("size_before_units")
        size_after = row.get("size_after_units")
        multiplier = row.get("overlay_multiplier")
        
        if size_before is not None and size_after is not None and multiplier is not None:
            size_checks += 1
            # Sign-preserving round (matching overlay logic)
            sign = 1 if size_before >= 0 else -1
            units_abs = abs(int(size_before))
            units_out_abs = int(round(units_abs * multiplier))
            # Minimum 1 unit if base > 0 (matching overlay logic)
            if units_out_abs == 0 and units_abs > 0:
                units_out_abs = 1
            expected = sign * units_out_abs
            if abs(size_after - expected) > 0.5:  # Allow small floating point differences
                size_mismatches += 1
    
    sanity["size_checks"] = size_checks
    sanity["size_mismatches"] = size_mismatches
    sanity["size_match_pct"] = ((size_checks - size_mismatches) / size_checks * 100.0) if size_checks > 0 else 0.0
    
    return sanity


def generate_ab_report(
    run_a_dir: Path,
    run_b_dir: Path,
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    overlay_sanity: Dict[str, Any],
) -> str:
    """Generate A/B comparison report."""
    
    # Overall Q4 metrics
    metrics_a_total = compute_metrics(df_a.to_dict("records"))
    metrics_b_total = compute_metrics(df_b.to_dict("records"))
    
    # Q4 × C_CHOP metrics
    df_a_cchop = df_a[df_a["regime_class"] == "C_CHOP"]
    df_b_cchop = df_b[df_b["regime_class"] == "C_CHOP"]
    metrics_a_cchop = compute_metrics(df_a_cchop.to_dict("records"))
    metrics_b_cchop = compute_metrics(df_b_cchop.to_dict("records"))
    
    # Q4 × C_CHOP × US metrics
    df_a_cchop_us = df_a[(df_a["regime_class"] == "C_CHOP") & (df_a["session"] == "US")]
    df_b_cchop_us = df_b[(df_b["regime_class"] == "C_CHOP") & (df_b["session"] == "US")]
    metrics_a_cchop_us = compute_metrics(df_a_cchop_us.to_dict("records"))
    metrics_b_cchop_us = compute_metrics(df_b_cchop_us.to_dict("records"))
    
    # Calculate deltas
    delta_total = {
        "ev": metrics_b_total["ev_per_trade"] - metrics_a_total["ev_per_trade"],
        "winrate": metrics_b_total["winrate"] - metrics_a_total["winrate"],
        "payoff": metrics_b_total["payoff"] - metrics_a_total["payoff"],
        "p90_loss": metrics_b_total["p90_loss"] - metrics_a_total["p90_loss"],  # Negative is better
    }
    
    delta_cchop = {
        "ev": metrics_b_cchop["ev_per_trade"] - metrics_a_cchop["ev_per_trade"],
        "winrate": metrics_b_cchop["winrate"] - metrics_a_cchop["winrate"],
        "payoff": metrics_b_cchop["payoff"] - metrics_a_cchop["payoff"],
        "p90_loss": metrics_b_cchop["p90_loss"] - metrics_a_cchop["p90_loss"],
    }
    
    delta_cchop_us = {
        "ev": metrics_b_cchop_us["ev_per_trade"] - metrics_a_cchop_us["ev_per_trade"],
        "winrate": metrics_b_cchop_us["winrate"] - metrics_a_cchop_us["winrate"],
        "payoff": metrics_b_cchop_us["payoff"] - metrics_a_cchop_us["payoff"],
        "p90_loss": metrics_b_cchop_us["p90_loss"] - metrics_a_cchop_us["p90_loss"],
    }
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_lines = [
        "# Q4 C_CHOP Overlay A/B Evaluation",
        "",
        f"**Generated**: {timestamp}",
        f"**Run A (Overlay OFF)**: `{run_a_dir.name}`",
        f"**Run B (Overlay ON)**: `{run_b_dir.name}`",
        f"**Source**: `parallel_chunks/**/trade_journal/trades/*.json`",
        "",
        "---",
        "",
        "## Summary",
        "",
        "### Q4 Total",
        "",
        "| Metric | Run A (OFF) | Run B (ON) | Delta |",
        "|--------|-------------|------------|-------|",
        f"| Trades | {metrics_a_total['n_trades']} | {metrics_b_total['n_trades']} | {metrics_b_total['n_trades'] - metrics_a_total['n_trades']:+d} |",
        f"| EV/Trade | {metrics_a_total['ev_per_trade']:.2f} bps | {metrics_b_total['ev_per_trade']:.2f} bps | {delta_total['ev']:+.2f} bps |",
        f"| Winrate | {metrics_a_total['winrate']:.1%} | {metrics_b_total['winrate']:.1%} | {delta_total['winrate']:+.1%} |",
        f"| Payoff | {metrics_a_total['payoff']:.2f} | {metrics_b_total['payoff']:.2f} | {delta_total['payoff']:+.2f} |",
        f"| P90 Loss | {metrics_a_total['p90_loss']:.2f} bps | {metrics_b_total['p90_loss']:.2f} bps | {delta_total['p90_loss']:+.2f} bps |",
        "",
        "### Q4 × C_CHOP",
        "",
        "| Metric | Run A (OFF) | Run B (ON) | Delta |",
        "|--------|-------------|------------|-------|",
        f"| Trades | {metrics_a_cchop['n_trades']} | {metrics_b_cchop['n_trades']} | {metrics_b_cchop['n_trades'] - metrics_a_cchop['n_trades']:+d} |",
        f"| EV/Trade | {metrics_a_cchop['ev_per_trade']:.2f} bps | {metrics_b_cchop['ev_per_trade']:.2f} bps | {delta_cchop['ev']:+.2f} bps |",
        f"| Winrate | {metrics_a_cchop['winrate']:.1%} | {metrics_b_cchop['winrate']:.1%} | {delta_cchop['winrate']:+.1%} |",
        f"| Payoff | {metrics_a_cchop['payoff']:.2f} | {metrics_b_cchop['payoff']:.2f} | {delta_cchop['payoff']:+.2f} |",
        f"| P90 Loss | {metrics_a_cchop['p90_loss']:.2f} bps | {metrics_b_cchop['p90_loss']:.2f} bps | {delta_cchop['p90_loss']:+.2f} bps |",
        "",
        "### Q4 × C_CHOP × US (Expected Largest Effect)",
        "",
        "| Metric | Run A (OFF) | Run B (ON) | Delta |",
        "|--------|-------------|------------|-------|",
        f"| Trades | {metrics_a_cchop_us['n_trades']} | {metrics_b_cchop_us['n_trades']} | {metrics_b_cchop_us['n_trades'] - metrics_a_cchop_us['n_trades']:+d} |",
        f"| EV/Trade | {metrics_a_cchop_us['ev_per_trade']:.2f} bps | {metrics_b_cchop_us['ev_per_trade']:.2f} bps | {delta_cchop_us['ev']:+.2f} bps |",
        f"| Winrate | {metrics_a_cchop_us['winrate']:.1%} | {metrics_b_cchop_us['winrate']:.1%} | {delta_cchop_us['winrate']:+.1%} |",
        f"| Payoff | {metrics_a_cchop_us['payoff']:.2f} | {metrics_b_cchop_us['payoff']:.2f} | {delta_cchop_us['payoff']:+.2f} |",
        f"| P90 Loss | {metrics_a_cchop_us['p90_loss']:.2f} bps | {metrics_b_cchop_us['p90_loss']:.2f} bps | {delta_cchop_us['p90_loss']:+.2f} bps |",
        "",
        "## Overlay Activation Sanity (Run B Only)",
        "",
        f"- **Trades with overlay_applied == True**: {overlay_sanity['applied_true_count']} ({overlay_sanity['applied_true_pct']:.1f}%)",
        "",
        "### Distribution per Session",
        "",
        "| Session | Count |",
        "|---------|-------|",
    ]
    
    for session, count in overlay_sanity.get("applied_by_session", {}).items():
        report_lines.append(f"| {session} | {count} |")
    
    report_lines.extend([
        "",
        "### Size Verification",
        "",
        f"- **Size checks**: {overlay_sanity['size_checks']}",
        f"- **Size mismatches**: {overlay_sanity['size_mismatches']}",
        f"- **Match rate**: {overlay_sanity['size_match_pct']:.1f}%",
        "",
        "## Conclusions",
        "",
    ])
    
    # Generate conclusions
    conclusions = []
    
    # EV change
    if delta_cchop_us["ev"] > 0:
        conclusions.append(f"✅ **EV/Trade increased** in C_CHOP×US: {delta_cchop_us['ev']:+.2f} bps")
    elif delta_cchop_us["ev"] < 0:
        conclusions.append(f"⚠️  **EV/Trade decreased** in C_CHOP×US: {delta_cchop_us['ev']:+.2f} bps")
    else:
        conclusions.append("➡️  **EV/Trade unchanged** in C_CHOP×US")
    
    # P90 loss change (negative delta is better)
    if delta_cchop_us["p90_loss"] < 0:
        conclusions.append(f"✅ **P90 Loss reduced** in C_CHOP×US: {delta_cchop_us['p90_loss']:+.2f} bps (target achieved)")
    elif delta_cchop_us["p90_loss"] > 0:
        conclusions.append(f"⚠️  **P90 Loss increased** in C_CHOP×US: {delta_cchop_us['p90_loss']:+.2f} bps")
    else:
        conclusions.append("➡️  **P90 Loss unchanged** in C_CHOP×US")
    
    # Winrate/Payoff
    if abs(delta_cchop_us["winrate"]) > 0.01:
        conclusions.append(f"📊 **Winrate changed** in C_CHOP×US: {delta_cchop_us['winrate']:+.1%}")
    if abs(delta_cchop_us["payoff"]) > 0.1:
        conclusions.append(f"📊 **Payoff changed** in C_CHOP×US: {delta_cchop_us['payoff']:+.2f}")
    
    # Overlay activation
    if overlay_sanity["applied_true_count"] > 0:
        conclusions.append(f"✅ **Overlay activated** for {overlay_sanity['applied_true_count']} trades ({overlay_sanity['applied_true_pct']:.1f}%)")
    else:
        conclusions.append("⚠️  **Overlay not activated** for any trades")
    
    # Size verification
    if overlay_sanity["size_match_pct"] >= 99.0:
        conclusions.append(f"✅ **Size calculation verified**: {overlay_sanity['size_match_pct']:.1f}% match rate")
    elif overlay_sanity["size_match_pct"] >= 95.0:
        conclusions.append(f"⚠️  **Size calculation mostly correct**: {overlay_sanity['size_match_pct']:.1f}% match rate")
    else:
        conclusions.append(f"❌ **Size calculation issues**: {overlay_sanity['size_match_pct']:.1f}% match rate")
    
    report_lines.extend(conclusions)
    report_lines.append("")
    
    return "\n".join(report_lines)


def main():
    parser = argparse.ArgumentParser(
        description="A/B evaluation of Q4 C_CHOP overlay impact"
    )
    parser.add_argument(
        "--run-a",
        type=str,
        required=True,
        help="Path to run A (overlay OFF) directory",
    )
    parser.add_argument(
        "--run-b",
        type=str,
        required=True,
        help="Path to run B (overlay ON) directory",
    )
    args = parser.parse_args()
    
    run_a_dir = Path(args.run_a)
    run_b_dir = Path(args.run_b)
    
    if not run_a_dir.exists():
        print(f"ERROR: Run A directory does not exist: {run_a_dir}")
        return 1
    
    if not run_b_dir.exists():
        print(f"ERROR: Run B directory does not exist: {run_b_dir}")
        return 1
    
    print(f"Loading trades from Run A (OFF): {run_a_dir.name}")
    trades_a = load_trades_from_chunks(run_a_dir)
    print(f"Loaded {len(trades_a)} trades from Run A")
    
    print(f"Loading trades from Run B (ON): {run_b_dir.name}")
    trades_b = load_trades_from_chunks(run_b_dir)
    print(f"Loaded {len(trades_b)} trades from Run B")
    
    # Extract rows
    rows_a = []
    for trade in trades_a:
        row = extract_trade_row(trade, "A")
        if row:
            rows_a.append(row)
    
    rows_b = []
    for trade in trades_b:
        row = extract_trade_row(trade, "B")
        if row:
            rows_b.append(row)
    
    df_a = pd.DataFrame(rows_a)
    df_b = pd.DataFrame(rows_b)
    
    print(f"Extracted {len(df_a)} rows from Run A")
    print(f"Extracted {len(df_b)} rows from Run B")
    
    # Verify overlay sanity in Run B
    overlay_sanity = verify_overlay_sanity(df_b)
    
    # Generate report
    report = generate_ab_report(run_a_dir, run_b_dir, df_a, df_b, overlay_sanity)
    
    # Write report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = REPORTS_DIR / f"SNIPER_Q4_CCHOP_OVERLAY_AB__{timestamp}.md"
    report_file.write_text(report, encoding="utf-8")
    print(f"✅ Report written: {report_file}")
    
    return 0


if __name__ == "__main__":
    exit(main())

