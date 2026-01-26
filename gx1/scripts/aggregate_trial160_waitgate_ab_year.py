#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trial 160 Pre-Entry Wait Gate A/B Aggregator - Single Year

DEL 6: Rapport-generator for PreEntryWaitGate A/B backtest.

Generates:
- WAIT_GATE_SUMMARY_{YEAR}.md
- WAIT_GATE_METRICS_{YEAR}.json

Includes:
- Headline A vs B comparison
- Trades, PnL, MaxDD, P1/P5/P50
- WaitGate telemetry (ARM_B only)
- Per-session breakdown
- GO/NO-GO criteria
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def load_metrics(report_dir: Path, year: int) -> Dict[str, Any]:
    """Load metrics from FULLYEAR_TRIAL160_METRICS_{year}.json."""
    # Try exact filename first
    metrics_file = report_dir / f"FULLYEAR_TRIAL160_METRICS_{year}.json"
    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            return json.load(f)
    
    # Fallback: glob pattern
    metrics_files = list(report_dir.glob("FULLYEAR_TRIAL160_METRICS_*.json"))
    if not metrics_files:
        raise FileNotFoundError(f"Metrics file not found in {report_dir}")
    if len(metrics_files) > 1:
        log.warning(f"Multiple metrics files found, using: {metrics_files[0]}")
    
    with open(metrics_files[0], "r") as f:
        return json.load(f)


def load_run_identity(report_dir: Path) -> Dict[str, Any]:
    """Load RUN_IDENTITY.json."""
    identity_file = report_dir / "RUN_IDENTITY.json"
    if not identity_file.exists():
        raise FileNotFoundError(f"RUN_IDENTITY.json not found: {identity_file}")
    
    with open(identity_file, "r") as f:
        return json.load(f)


def load_trade_outcomes(report_dir: Path) -> pd.DataFrame:
    """Load trade outcomes from chunk_* directories."""
    # Look for merged file first
    merged_file = report_dir / "trade_outcomes_MERGED.parquet"
    if merged_file.exists():
        return pd.read_parquet(merged_file)
    
    # Otherwise, look in chunk_0
    chunk_0_dir = report_dir / "chunk_0"
    if chunk_0_dir.exists():
        trade_files = list(chunk_0_dir.glob("trade_outcomes_*.parquet"))
        if trade_files:
            return pd.read_parquet(trade_files[0])
    
    raise FileNotFoundError(f"Trade outcomes not found in {report_dir}")


def calculate_per_session_metrics(trades: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Calculate per-session metrics (EU/OVERLAP/US)."""
    if len(trades) == 0:
        return {
            "EU": {"trades": 0, "pnl_bps": 0.0, "maxdd_bps": 0.0},
            "OVERLAP": {"trades": 0, "pnl_bps": 0.0, "maxdd_bps": 0.0},
            "US": {"trades": 0, "pnl_bps": 0.0, "maxdd_bps": 0.0},
        }
    
    # Get session from entry_session or infer from timestamp
    if "entry_session" in trades.columns:
        trades = trades.copy()
        trades["session"] = trades["entry_session"]
    else:
        # Infer session from timestamp (simplified)
        trades = trades.copy()
        trades["session"] = "UNKNOWN"
    
    sessions = ["EU", "OVERLAP", "US"]
    result = {}
    
    for session in sessions:
        session_trades = trades[trades["session"] == session]
        if len(session_trades) == 0:
            result[session] = {"trades": 0, "pnl_bps": 0.0, "maxdd_bps": 0.0}
        else:
            pnl_bps = float(session_trades["pnl_bps"].sum())
            maxdd_bps = float(session_trades["maxdd_bps"].max()) if "maxdd_bps" in session_trades.columns else 0.0
            result[session] = {
                "trades": len(session_trades),
                "pnl_bps": pnl_bps,
                "maxdd_bps": maxdd_bps,
            }
    
    return result


def evaluate_go_nogo(metrics_a: Dict[str, Any], metrics_b: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate GO/NO-GO criteria.
    
    Criteria:
    - Trades_B <= Trades_A * 0.80 (we want to reduce noise)
    - P5_B >= P5_A * 1.10 (10% improvement in P5)
    - MaxDD_B >= MaxDD_A * 0.95 (not worse than +5% DD)
    - PnL_B >= PnL_A * 0.95 (we tolerate slightly lower total PnL if tail improves)
    """
    # Handle both field name variants
    trades_a = metrics_a.get("n_trades", metrics_a.get("trade_count", 0))
    trades_b = metrics_b.get("n_trades", metrics_b.get("trade_count", 0))
    
    pnl_a = metrics_a.get("total_pnl_bps", 0.0)
    pnl_b = metrics_b.get("total_pnl_bps", 0.0)
    
    maxdd_a = abs(metrics_a.get("max_dd", metrics_a.get("max_drawdown_bps", 0.0)))
    maxdd_b = abs(metrics_b.get("max_dd", metrics_b.get("max_drawdown_bps", 0.0)))
    
    # Get P5 from metrics (p5_pnl field)
    p5_a = metrics_a.get("p5_pnl", metrics_a.get("trade_pnl_p5", 0.0))
    p5_b = metrics_b.get("p5_pnl", metrics_b.get("trade_pnl_p5", 0.0))
    
    criteria = {
        "trades_reduction": {
            "criterion": f"Trades_B <= Trades_A * 0.80",
            "value_a": trades_a,
            "value_b": trades_b,
            "threshold": trades_a * 0.80,
            "passed": trades_b <= trades_a * 0.80,
        },
        "p5_improvement": {
            "criterion": f"P5_B >= P5_A * 1.10",
            "value_a": p5_a,
            "value_b": p5_b,
            "threshold": p5_a * 1.10,
            "passed": p5_b >= p5_a * 1.10,
        },
        "maxdd_not_worse": {
            "criterion": f"MaxDD_B >= MaxDD_A * 0.95",
            "value_a": maxdd_a,
            "value_b": maxdd_b,
            "threshold": maxdd_a * 0.95,
            "passed": maxdd_b >= maxdd_a * 0.95,
        },
        "pnl_not_collapse": {
            "criterion": f"PnL_B >= PnL_A * 0.95",
            "value_a": pnl_a,
            "value_b": pnl_b,
            "threshold": pnl_a * 0.95,
            "passed": pnl_b >= pnl_a * 0.95,
        },
    }
    
    all_passed = all(c["passed"] for c in criteria.values())
    
    return {
        "all_passed": all_passed,
        "verdict": "PASS" if all_passed else "FAIL",
        "criteria": criteria,
    }


def generate_summary(
    year: int,
    metrics_a: Dict[str, Any],
    metrics_b: Dict[str, Any],
    identity_a: Dict[str, Any],
    identity_b: Dict[str, Any],
    session_metrics_a: Dict[str, Dict[str, float]],
    session_metrics_b: Dict[str, Dict[str, float]],
    go_nogo: Dict[str, Any],
    output_path: Path,
    trades_a_df: Optional[pd.DataFrame] = None,
    trades_b_df: Optional[pd.DataFrame] = None,
) -> None:
    """Generate WAIT_GATE_SUMMARY_{YEAR}.md."""
    
    # Handle both field name variants
    trades_a = metrics_a.get("n_trades", metrics_a.get("trade_count", 0))
    trades_b = metrics_b.get("n_trades", metrics_b.get("trade_count", 0))
    trades_delta = trades_b - trades_a
    trades_delta_pct = (trades_delta / trades_a * 100) if trades_a > 0 else 0.0
    
    pnl_a = metrics_a.get("total_pnl_bps", 0.0)
    pnl_b = metrics_b.get("total_pnl_bps", 0.0)
    pnl_delta = pnl_b - pnl_a
    pnl_delta_pct = (pnl_delta / pnl_a * 100) if pnl_a != 0 else 0.0
    
    maxdd_a = abs(metrics_a.get("max_drawdown_bps", 0.0))
    maxdd_b = abs(metrics_b.get("max_drawdown_bps", 0.0))
    maxdd_delta = maxdd_b - maxdd_a
    maxdd_delta_pct = (maxdd_delta / maxdd_a * 100) if maxdd_a != 0 else 0.0
    
    # Get P1/P5/P50 from metrics first, then fallback to trade outcomes
    p1_a = metrics_a.get("p1_pnl", metrics_a.get("trade_pnl_p1", None))
    p5_a = metrics_a.get("p5_pnl", metrics_a.get("trade_pnl_p5", None))
    p50_a = metrics_a.get("p50_pnl", metrics_a.get("trade_pnl_p50", None))
    p1_b = metrics_b.get("p1_pnl", metrics_b.get("trade_pnl_p1", None))
    p5_b = metrics_b.get("p5_pnl", metrics_b.get("trade_pnl_p5", None))
    p50_b = metrics_b.get("p50_pnl", metrics_b.get("trade_pnl_p50", None))
    
    # Fallback: calculate from trade outcomes DataFrame if available
    if p1_a is None and trades_a_df is not None and len(trades_a_df) > 0:
        p1_a = float(trades_a_df["pnl_bps"].quantile(0.01))
        p5_a = float(trades_a_df["pnl_bps"].quantile(0.05))
        p50_a = float(trades_a_df["pnl_bps"].quantile(0.50))
    if p1_b is None and trades_b_df is not None and len(trades_b_df) > 0:
        p1_b = float(trades_b_df["pnl_bps"].quantile(0.01))
        p5_b = float(trades_b_df["pnl_bps"].quantile(0.05))
        p50_b = float(trades_b_df["pnl_bps"].quantile(0.50))
    
    # Default to 0.0 if still None
    p1_a = p1_a if p1_a is not None else 0.0
    p5_a = p5_a if p5_a is not None else 0.0
    p50_a = p50_a if p50_a is not None else 0.0
    p1_b = p1_b if p1_b is not None else 0.0
    p5_b = p5_b if p5_b is not None else 0.0
    p50_b = p50_b if p50_b is not None else 0.0
    
    pnl_per_trade_a = pnl_a / trades_a if trades_a > 0 else 0.0
    pnl_per_trade_b = pnl_b / trades_b if trades_b > 0 else 0.0
    
    # WaitGate telemetry (ARM_B only)
    wait_gate_counters = identity_b.get("pre_entry_wait_counters", {})
    wait_total = wait_gate_counters.get("pre_entry_wait_n_total", 0)
    wait_wait = wait_gate_counters.get("pre_entry_wait_n_wait", 0)
    wait_pass = wait_gate_counters.get("pre_entry_wait_n_pass", 0)
    wait_rate = (wait_wait / wait_total * 100) if wait_total > 0 else 0.0
    
    with open(output_path, "w") as f:
        f.write(f"# Pre-Entry Wait Gate A/B Backtest Summary - {year}\n\n")
        f.write(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Executive Delta (top-level summary)
        f.write("## Executive Delta\n\n")
        f.write("| Metric | ARM_A | ARM_B | Change |\n")
        f.write("|---|---|---|---|\n")
        f.write(f"| **Trades** | {trades_a:,} | {trades_b:,} | {trades_delta:+,} ({trades_delta_pct:+.1f}%) |\n")
        f.write(f"| **Total PnL (bps)** | {pnl_a:,.1f} | {pnl_b:,.1f} | {pnl_delta:+,.1f} ({pnl_delta_pct:+.1f}%) |\n")
        f.write(f"| **PnL/Trade (bps)** | {pnl_per_trade_a:.2f} | {pnl_per_trade_b:.2f} | {pnl_per_trade_b - pnl_per_trade_a:+.2f} ({((pnl_per_trade_b - pnl_per_trade_a) / pnl_per_trade_a * 100) if pnl_per_trade_a != 0 else 0.0:+.1f}%) |\n")
        f.write(f"| **P5 (bps)** | {p5_a:.2f} | {p5_b:.2f} | {p5_b - p5_a:+.2f} ({((p5_b - p5_a) / abs(p5_a) * 100) if p5_a != 0 else 0.0:+.1f}%) |\n")
        f.write(f"| **MaxDD (bps)** | {maxdd_a:,.1f} | {maxdd_b:,.1f} | {maxdd_delta:+,.1f} ({maxdd_delta_pct:+.1f}%) |\n")
        f.write(f"| **Wait Rate** | N/A | {wait_rate:.1f}% | {wait_wait:,} waits / {wait_total:,} total |\n")
        f.write("\n")
        
        # Wait reason breakdown summary
        wait_reasons_summary = []
        for reason_key in [
            "pre_entry_wait_n_pullback_depth",
            "pre_entry_wait_n_bars_since_low",
            "pre_entry_wait_n_adverse_move",
            "pre_entry_wait_n_distance_to_ema",
            "pre_entry_wait_n_volatility_cooling",
        ]:
            count = wait_gate_counters.get(reason_key, 0)
            if count > 0:
                reason_name = reason_key.replace("pre_entry_wait_n_", "").replace("_", " ").title()
                wait_reasons_summary.append(f"{reason_name}: {count:,}")
        
        f.write("**Wait Reason Breakdown (ARM_B):** " + ", ".join(wait_reasons_summary) if wait_reasons_summary else "N/A")
        f.write("\n\n")
        
        # GO/NO-GO verdict
        f.write(f"**GO/NO-GO:** {go_nogo['verdict']}\n\n")
        if go_nogo["verdict"] == "PASS":
            # Find the key improvement
            if p5_b >= p5_a * 1.10:
                f.write("✅ **PASS** - P5 improved by ≥10%, indicating better tail performance.\n\n")
            elif trades_b <= trades_a * 0.80 and maxdd_b >= maxdd_a * 0.95:
                f.write("✅ **PASS** - Trade count reduced by ≥20% while MaxDD maintained.\n\n")
            else:
                f.write("✅ **PASS** - Overall improvement meets criteria.\n\n")
        else:
            # Find the failure reason
            failed_criteria = [k for k, v in go_nogo["criteria"].items() if not v["passed"]]
            if "p5_improvement" in failed_criteria:
                f.write("❌ **FAIL** - P5 did not improve by ≥10%.\n\n")
            elif "maxdd_not_worse" in failed_criteria:
                f.write("❌ **FAIL** - MaxDD worsened by >5%.\n\n")
            elif "pnl_not_collapse" in failed_criteria:
                f.write("❌ **FAIL** - Total PnL fell by >5% without clear tail improvement.\n\n")
            else:
                f.write("❌ **FAIL** - One or more criteria not met.\n\n")
        f.write("\n")
        
        f.write("## Headline: ARM_A vs ARM_B\n\n")
        f.write("| Metric | ARM_A (Baseline) | ARM_B (Wait Gate) | Delta | Delta % |\n")
        f.write("|---|---|---|---|---|\n")
        f.write(f"| **Trades** | {trades_a:,} | {trades_b:,} | {trades_delta:+,} | {trades_delta_pct:+.1f}% |\n")
        f.write(f"| **Total PnL (bps)** | {pnl_a:,.1f} | {pnl_b:,.1f} | {pnl_delta:+,.1f} | {pnl_delta_pct:+.1f}% |\n")
        f.write(f"| **PnL/Trade (bps)** | {pnl_per_trade_a:.2f} | {pnl_per_trade_b:.2f} | {pnl_per_trade_b - pnl_per_trade_a:+.2f} | {((pnl_per_trade_b - pnl_per_trade_a) / pnl_per_trade_a * 100) if pnl_per_trade_a != 0 else 0.0:+.1f}% |\n")
        f.write(f"| **MaxDD (bps)** | {maxdd_a:,.1f} | {maxdd_b:,.1f} | {maxdd_delta:+,.1f} | {maxdd_delta_pct:+.1f}% |\n")
        f.write(f"| **P1 (bps)** | {p1_a:.2f} | {p1_b:.2f} | {p1_b - p1_a:+.2f} | {((p1_b - p1_a) / abs(p1_a) * 100) if p1_a != 0 else 0.0:+.1f}% |\n")
        f.write(f"| **P5 (bps)** | {p5_a:.2f} | {p5_b:.2f} | {p5_b - p5_a:+.2f} | {((p5_b - p5_a) / abs(p5_a) * 100) if p5_a != 0 else 0.0:+.1f}% |\n")
        f.write(f"| **P50 (bps)** | {p50_a:.2f} | {p50_b:.2f} | {p50_b - p50_a:+.2f} | {((p50_b - p50_a) / abs(p50_a) * 100) if p50_a != 0 else 0.0:+.1f}% |\n")
        f.write("\n")
        
        f.write("## WaitGate Telemetry (ARM_B only)\n\n")
        f.write("| Metric | Value |\n")
        f.write("|---|---|\n")
        f.write(f"| **Total Evaluations** | {wait_total:,} |\n")
        f.write(f"| **WAIT** | {wait_wait:,} |\n")
        f.write(f"| **PASS** | {wait_pass:,} |\n")
        f.write(f"| **Wait Rate** | {wait_rate:.1f}% |\n")
        f.write("\n")
        
        f.write("### Wait Reason Breakdown\n\n")
        f.write("| Reason | Count |\n")
        f.write("|---|---|\n")
        for reason_key in [
            "pre_entry_wait_n_pullback_depth",
            "pre_entry_wait_n_bars_since_low",
            "pre_entry_wait_n_adverse_move",
            "pre_entry_wait_n_distance_to_ema",
            "pre_entry_wait_n_volatility_cooling",
        ]:
            count = wait_gate_counters.get(reason_key, 0)
            reason_name = reason_key.replace("pre_entry_wait_n_", "").replace("_", " ").title()
            f.write(f"| {reason_name} | {count:,} |\n")
        f.write("\n")
        
        f.write("## Per-Session Breakdown\n\n")
        f.write("| Session | ARM_A Trades | ARM_A PnL (bps) | ARM_A MaxDD (bps) | ARM_B Trades | ARM_B PnL (bps) | ARM_B MaxDD (bps) | Delta Trades | Delta PnL (bps) |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for session in ["EU", "OVERLAP", "US"]:
            sess_a = session_metrics_a.get(session, {})
            sess_b = session_metrics_b.get(session, {})
            f.write(
                f"| {session} | {sess_a.get('trades', 0):,} | {sess_a.get('pnl_bps', 0.0):,.1f} | "
                f"{sess_a.get('maxdd_bps', 0.0):,.1f} | {sess_b.get('trades', 0):,} | "
                f"{sess_b.get('pnl_bps', 0.0):,.1f} | {sess_b.get('maxdd_bps', 0.0):,.1f} | "
                f"{sess_b.get('trades', 0) - sess_a.get('trades', 0):+,} | "
                f"{sess_b.get('pnl_bps', 0.0) - sess_a.get('pnl_bps', 0.0):+,.1f} |\n"
            )
        f.write("\n")
        
        f.write("## GO/NO-GO Evaluation\n\n")
        f.write(f"**Verdict:** {go_nogo['verdict']}\n\n")
        f.write("| Criterion | ARM_A | ARM_B | Threshold | Passed |\n")
        f.write("|---|---|---|---|---|\n")
        for key, criterion in go_nogo["criteria"].items():
            passed_mark = "✅" if criterion["passed"] else "❌"
            f.write(
                f"| {criterion['criterion']} | {criterion['value_a']:.2f} | {criterion['value_b']:.2f} | "
                f"{criterion['threshold']:.2f} | {passed_mark} |\n"
            )
        f.write("\n")
        
        f.write("## Conclusion\n\n")
        if go_nogo["verdict"] == "PASS":
            f.write("✅ **PASS**: Pre-Entry Wait Gate shows improvement.\n\n")
        else:
            f.write("❌ **FAIL**: Pre-Entry Wait Gate does not meet criteria.\n\n")
        
        # Identify which session and wait reason drives the change
        f.write("### Key Insights\n\n")
        f.write(f"- Wait rate: {wait_rate:.1f}% of entry evaluations were blocked\n")
        f.write(f"- Trade reduction: {trades_delta_pct:+.1f}% ({trades_delta:+,} trades)\n")
        f.write(f"- P5 improvement: {((p5_b - p5_a) / abs(p5_a) * 100) if p5_a != 0 else 0.0:+.1f}%\n")
        f.write(f"- MaxDD change: {maxdd_delta_pct:+.1f}%\n")
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Trial 160 Pre-Entry Wait Gate A/B Aggregator - Single Year"
    )
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Year (e.g., 2024)",
    )
    parser.add_argument(
        "--arm-a",
        type=Path,
        required=True,
        help="ARM_A report directory",
    )
    parser.add_argument(
        "--arm-b",
        type=Path,
        required=True,
        help="ARM_B report directory",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for summary and metrics",
    )
    
    args = parser.parse_args()
    
    log.info("=" * 60)
    log.info("TRIAL 160 PRE-ENTRY WAIT GATE A/B AGGREGATOR")
    log.info("=" * 60)
    log.info(f"Year: {args.year}")
    log.info(f"ARM_A: {args.arm_a}")
    log.info(f"ARM_B: {args.arm_b}")
    log.info("")
    
    # Load data
    log.info("Loading ARM_A data...")
    metrics_a = load_metrics(args.arm_a, args.year)
    identity_a = load_run_identity(args.arm_a)
    trades_a = load_trade_outcomes(args.arm_a)
    session_metrics_a = calculate_per_session_metrics(trades_a)
    
    log.info("Loading ARM_B data...")
    metrics_b = load_metrics(args.arm_b, args.year)
    identity_b = load_run_identity(args.arm_b)
    trades_b = load_trade_outcomes(args.arm_b)
    session_metrics_b = calculate_per_session_metrics(trades_b)
    
    # Evaluate GO/NO-GO
    log.info("Evaluating GO/NO-GO criteria...")
    go_nogo = evaluate_go_nogo(metrics_a, metrics_b)
    
    # Generate summary
    log.info("Generating summary...")
    args.out.mkdir(parents=True, exist_ok=True)
    summary_path = args.out / f"WAIT_GATE_SUMMARY_{args.year}.md"
    generate_summary(
        year=args.year,
        metrics_a=metrics_a,
        metrics_b=metrics_b,
        identity_a=identity_a,
        identity_b=identity_b,
        session_metrics_a=session_metrics_a,
        session_metrics_b=session_metrics_b,
        go_nogo=go_nogo,
        output_path=summary_path,
        trades_a_df=trades_a,
        trades_b_df=trades_b,
    )
    
    # Calculate deltas for JSON
    trades_a_count = metrics_a.get("n_trades", metrics_a.get("trade_count", 0))
    trades_b_count = metrics_b.get("n_trades", metrics_b.get("trade_count", 0))
    pnl_a_val = metrics_a.get("total_pnl_bps", 0.0)
    pnl_b_val = metrics_b.get("total_pnl_bps", 0.0)
    maxdd_a_val = abs(metrics_a.get("max_dd", metrics_a.get("max_drawdown_bps", 0.0)))
    maxdd_b_val = abs(metrics_b.get("max_dd", metrics_b.get("max_drawdown_bps", 0.0)))
    
    # Get P1/P5/P50 for deltas
    p1_a_val = metrics_a.get("p1_pnl", metrics_a.get("trade_pnl_p1", 0.0))
    p5_a_val = metrics_a.get("p5_pnl", metrics_a.get("trade_pnl_p5", 0.0))
    p50_a_val = metrics_a.get("p50_pnl", metrics_a.get("trade_pnl_p50", 0.0))
    p1_b_val = metrics_b.get("p1_pnl", metrics_b.get("trade_pnl_p1", 0.0))
    p5_b_val = metrics_b.get("p5_pnl", metrics_b.get("trade_pnl_p5", 0.0))
    p50_b_val = metrics_b.get("p50_pnl", metrics_b.get("trade_pnl_p50", 0.0))
    
    # Fallback to trade outcomes if not in metrics
    if p1_a_val == 0.0 and trades_a is not None and len(trades_a) > 0:
        p1_a_val = float(trades_a["pnl_bps"].quantile(0.01))
        p5_a_val = float(trades_a["pnl_bps"].quantile(0.05))
        p50_a_val = float(trades_a["pnl_bps"].quantile(0.50))
    if p1_b_val == 0.0 and trades_b is not None and len(trades_b) > 0:
        p1_b_val = float(trades_b["pnl_bps"].quantile(0.01))
        p5_b_val = float(trades_b["pnl_bps"].quantile(0.05))
        p50_b_val = float(trades_b["pnl_bps"].quantile(0.50))
    
    # Generate metrics JSON
    log.info("Generating metrics JSON...")
    metrics_path = args.out / f"WAIT_GATE_METRICS_{args.year}.json"
    metrics_json = {
        "year": args.year,
        "arm_a": {
            "metrics": metrics_a,
            "session_metrics": session_metrics_a,
        },
        "arm_b": {
            "metrics": metrics_b,
            "session_metrics": session_metrics_b,
            "wait_gate_counters": identity_b.get("pre_entry_wait_counters", {}),
        },
        "go_nogo": go_nogo,
        "deltas": {
            "trades": trades_b_count - trades_a_count,
            "pnl_bps": pnl_b_val - pnl_a_val,
            "maxdd_bps": maxdd_b_val - maxdd_a_val,
            "p1": p1_b_val - p1_a_val,
            "p5": p5_b_val - p5_a_val,
            "p50": p50_b_val - p50_a_val,
        },
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_json, f, indent=2)
    
    log.info("")
    log.info("=" * 60)
    log.info("✅ AGGREGATION COMPLETE")
    log.info("=" * 60)
    log.info(f"Summary: {summary_path}")
    log.info(f"Metrics: {metrics_path}")
    log.info(f"GO/NO-GO: {go_nogo['verdict']}")


if __name__ == "__main__":
    main()
