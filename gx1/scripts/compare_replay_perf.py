#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Replay Performance Comparison - A/B Test PreGate OFF vs ON

DEL 1: Compare replay performance metrics between PreGate OFF and ON.
DEL 3: Hard stop decision: Is 1-hour target achievable without HTF cache?

Usage:
    python gx1/scripts/compare_replay_perf.py \
        --off reports/perf/PREGATE_OFF_<run_id>.json \
        --on reports/perf/PREGATE_ON_<run_id>.json \
        --output reports/perf/PREGATE_COMPARISON_<run_id>.md
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def load_perf_json(path: Path) -> Dict[str, Any]:
    """
    Load performance JSON file.
    
    Raises ValueError if file is a stub (export_failed).
    """
    if not path.exists():
        raise FileNotFoundError(f"Perf JSON not found: {path}")
    
    with open(path, "r") as f:
        data = json.load(f)
    
    # CRITICAL: Check if this is a stub file (export failed)
    if data.get("status") == "export_failed":
        raise ValueError(
            f"Perf JSON is a stub file (export failed): {path}\n"
            f"Export error: {data.get('export_error', 'unknown')}\n"
            f"This file was written because perf export failed. "
            f"Comparison cannot proceed with incomplete data."
        )
    
    return data


def calculate_speedup(off_data: Dict[str, Any], on_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate speedup metrics from OFF vs ON comparison.
    
    Args:
        off_data: PreGate OFF performance data
        on_data: PreGate ON performance data
    
    Returns:
        Dict with speedup metrics
    """
    # Extract aggregate metrics (sum across all chunks)
    off_wall_clock_sec = sum(chunk.get("wall_clock_sec", 0.0) for chunk in off_data.get("chunks", []))
    on_wall_clock_sec = sum(chunk.get("wall_clock_sec", 0.0) for chunk in on_data.get("chunks", []))
    
    off_bars_total = sum(chunk.get("total_bars", 0) for chunk in off_data.get("chunks", []))
    on_bars_total = sum(chunk.get("total_bars", 0) for chunk in on_data.get("chunks", []))
    
    off_bars_per_sec = off_bars_total / off_wall_clock_sec if off_wall_clock_sec > 0 else 0.0
    on_bars_per_sec = on_bars_total / on_wall_clock_sec if on_wall_clock_sec > 0 else 0.0
    
    off_model_calls = sum(chunk.get("n_model_calls", 0) for chunk in off_data.get("chunks", []))
    on_model_calls = sum(chunk.get("n_model_calls", 0) for chunk in on_data.get("chunks", []))
    
    off_feature_time_total = sum(chunk.get("feature_time_mean_ms", 0.0) * chunk.get("n_model_calls", 0) / 1000.0 
                                  for chunk in off_data.get("chunks", []) if chunk.get("n_model_calls", 0) > 0)
    on_feature_time_total = sum(chunk.get("feature_time_mean_ms", 0.0) * chunk.get("n_model_calls", 0) / 1000.0 
                                 for chunk in on_data.get("chunks", []) if chunk.get("n_model_calls", 0) > 0)
    
    # DEL 1: Phase timing breakdown (sum across chunks)
    off_t_pregate = sum(chunk.get("t_pregate_total_sec", 0.0) for chunk in off_data.get("chunks", []))
    on_t_pregate = sum(chunk.get("t_pregate_total_sec", 0.0) for chunk in on_data.get("chunks", []))
    off_t_feature = sum(chunk.get("t_feature_build_total_sec", 0.0) for chunk in off_data.get("chunks", []))
    on_t_feature = sum(chunk.get("t_feature_build_total_sec", 0.0) for chunk in on_data.get("chunks", []))
    off_t_model = sum(chunk.get("t_model_total_sec", 0.0) for chunk in off_data.get("chunks", []))
    on_t_model = sum(chunk.get("t_model_total_sec", 0.0) for chunk in on_data.get("chunks", []))
    off_t_policy = sum(chunk.get("t_policy_total_sec", 0.0) for chunk in off_data.get("chunks", []))
    on_t_policy = sum(chunk.get("t_policy_total_sec", 0.0) for chunk in on_data.get("chunks", []))
    off_t_io = sum(chunk.get("t_io_total_sec", 0.0) for chunk in off_data.get("chunks", []))
    on_t_io = sum(chunk.get("t_io_total_sec", 0.0) for chunk in on_data.get("chunks", []))
    
    # Feature time mean (for hard-akseptkriterium)
    off_feature_time_mean_ms = (off_feature_time_total / off_model_calls * 1000.0) if off_model_calls > 0 else 0.0
    on_feature_time_mean_ms = (on_feature_time_total / on_model_calls * 1000.0) if on_model_calls > 0 else 0.0
    
    # Speedup factor (higher is better)
    speedup_factor = on_bars_per_sec / off_bars_per_sec if off_bars_per_sec > 0 else 0.0
    
    # Model call reduction (%)
    model_call_reduction = ((off_model_calls - on_model_calls) / off_model_calls * 100.0) if off_model_calls > 0 else 0.0
    
    # Feature time reduction (%)
    feature_time_reduction = ((off_feature_time_total - on_feature_time_total) / off_feature_time_total * 100.0) if off_feature_time_total > 0 else 0.0
    
    # PreGate skip ratio (from ON data)
    pregate_skips_total = sum(chunk.get("pregate_skips", 0) for chunk in on_data.get("chunks", []))
    pregate_skip_ratio = (pregate_skips_total / on_bars_total * 100.0) if on_bars_total > 0 else 0.0
    
    return {
        "speedup_factor": speedup_factor,
        "wall_clock_sec_off": off_wall_clock_sec,
        "wall_clock_sec_on": on_wall_clock_sec,
        "wall_clock_reduction_sec": off_wall_clock_sec - on_wall_clock_sec,
        "wall_clock_reduction_pct": ((off_wall_clock_sec - on_wall_clock_sec) / off_wall_clock_sec * 100.0) if off_wall_clock_sec > 0 else 0.0,
        "bars_per_sec_off": off_bars_per_sec,
        "bars_per_sec_on": on_bars_per_sec,
        "bars_total_off": off_bars_total,
        "bars_total_on": on_bars_total,
        "model_call_reduction": model_call_reduction,
        "model_calls_off": off_model_calls,
        "model_calls_on": on_model_calls,
        "feature_time_reduction": feature_time_reduction,
        "feature_time_total_sec_off": off_feature_time_total,
        "feature_time_total_sec_on": on_feature_time_total,
        "feature_time_mean_ms_off": off_feature_time_mean_ms,
        "feature_time_mean_ms_on": on_feature_time_mean_ms,
        "pregate_skip_ratio": pregate_skip_ratio,
        "pregate_skips_total": pregate_skips_total,
        # DEL 1: Phase timing breakdown
        "t_pregate_total_sec_off": off_t_pregate,
        "t_pregate_total_sec_on": on_t_pregate,
        "t_feature_build_total_sec_off": off_t_feature,
        "t_feature_build_total_sec_on": on_t_feature,
        "t_model_total_sec_off": off_t_model,
        "t_model_total_sec_on": on_t_model,
        "t_policy_total_sec_off": off_t_policy,
        "t_policy_total_sec_on": on_t_policy,
        "t_io_total_sec_off": off_t_io,
        "t_io_total_sec_on": on_t_io,
    }


def verify_invariants(off_data: Dict[str, Any], on_data: Dict[str, Any], metrics: Dict[str, float]) -> Dict[str, Any]:
    """
    Verify invariants: trading results should be identical (±0.1%), model calls should not increase.
    
    Args:
        off_data: PreGate OFF performance data
        on_data: PreGate ON performance data
        metrics: Calculated metrics
    
    Returns:
        Dict with invariant check results
    """
    off_trades = sum(chunk.get("n_trades_closed", 0) for chunk in off_data.get("chunks", []))
    on_trades = sum(chunk.get("n_trades_closed", 0) for chunk in on_data.get("chunks", []))
    
    trades_diff_pct = abs(off_trades - on_trades) / off_trades * 100.0 if off_trades > 0 else 0.0
    trades_match = trades_diff_pct <= 0.1
    
    model_calls_increased = metrics["model_calls_on"] > metrics["model_calls_off"]
    
    return {
        "trades_off": off_trades,
        "trades_on": on_trades,
        "trades_diff_pct": trades_diff_pct,
        "trades_match": trades_match,
        "model_calls_increased": model_calls_increased,
        "all_invariants_ok": trades_match and not model_calls_increased,
    }


def estimate_fullyear_runtime(speedup_factor: float, current_runtime_sec: float) -> float:
    """
    Estimate FULLYEAR runtime based on speedup factor.
    
    Assumes linear scaling (may not be accurate but gives ballpark).
    """
    fullyear_bars = 70000  # Approximate FULLYEAR bars
    fullyear_estimated_sec = (fullyear_bars / (70000 / current_runtime_sec)) / speedup_factor
    fullyear_estimated_minutes = fullyear_estimated_sec / 60.0
    return fullyear_estimated_minutes


def check_hard_acceptance_criteria(metrics: Dict[str, float], invariants: Dict[str, Any]) -> Dict[str, Any]:
    """
    DEL 2: Hard-akseptkriterium for "1-time plausibel".
    
    PASS (1-time plausibel) hvis:
    - speedup_factor >= 3.0x (mini-parallel, fullyear har bedre amortisering)
    - pregate_skip_ratio >= 0.80 (80% skips)
    - feature_time_mean_ms_on <= 0.30 * feature_time_mean_ms_off (70% reduksjon)
    - trades/model_calls innenfor invariants
    
    Returns:
        Dict with acceptance check results
    """
    speedup_ok = metrics["speedup_factor"] >= 3.0
    pregate_ratio_ok = metrics["pregate_skip_ratio"] >= 80.0
    feature_time_ok = metrics["feature_time_mean_ms_on"] <= (0.30 * metrics["feature_time_mean_ms_off"])
    invariants_ok = invariants["all_invariants_ok"]
    
    all_pass = speedup_ok and pregate_ratio_ok and feature_time_ok and invariants_ok
    
    return {
        "pass": all_pass,
        "speedup_ok": speedup_ok,
        "pregate_ratio_ok": pregate_ratio_ok,
        "feature_time_ok": feature_time_ok,
        "invariants_ok": invariants_ok,
        "speedup_factor": metrics["speedup_factor"],
        "pregate_skip_ratio": metrics["pregate_skip_ratio"],
        "feature_time_reduction_pct": ((metrics["feature_time_mean_ms_off"] - metrics["feature_time_mean_ms_on"]) / metrics["feature_time_mean_ms_off"] * 100.0) if metrics["feature_time_mean_ms_off"] > 0 else 0.0,
    }


def generate_report(
    off_path: Path,
    on_path: Path,
    metrics: Dict[str, float],
    invariants: Dict[str, Any],
    output_path: Path,
) -> None:
    """
    Generate markdown report with comparison and decision.
    
    Args:
        off_path: Path to PreGate OFF JSON
        on_path: Path to PreGate ON JSON
        metrics: Calculated metrics
        invariants: Invariant check results
        output_path: Output markdown file path
    """
    speedup_factor = metrics["speedup_factor"]
    
    # DEL 2: Check hard-akseptkriterium
    acceptance = check_hard_acceptance_criteria(metrics, invariants)
    
    # DEL 3: Hard stop decision (based on acceptance criteria)
    if acceptance["pass"]:
        status = "1-HOUR TARGET ACHIEVABLE WITHOUT HTF CACHE"
        recommendation = "GO 1-HOUR"
        fullyear_estimated_minutes = estimate_fullyear_runtime(speedup_factor, metrics["wall_clock_sec_on"])
    else:
        status = "REPLAY STILL DOMINATED BY FEATURE-BUILDING"
        recommendation = "HTF CACHE REQUIRED"
        fullyear_estimated_minutes = None
    
    report_lines = [
        "# Replay Performance Comparison: PreGate OFF vs ON",
        "",
        f"**Generated:** {datetime.now().isoformat()}",
        "",
        "## Summary",
        "",
        f"- **Speedup Factor:** {speedup_factor:.2f}x",
        f"- **Wall Clock Reduction:** {metrics['wall_clock_reduction_pct']:.1f}% ({metrics['wall_clock_reduction_sec']:.1f}s)",
        f"- **Model Call Reduction:** {metrics['model_call_reduction']:.1f}%",
        f"- **Feature Time Reduction:** {metrics['feature_time_reduction']:.1f}%",
        f"- **PreGate Skip Ratio:** {metrics['pregate_skip_ratio']:.1f}%",
        "",
        "## Detailed Metrics",
        "",
        "### Throughput",
        f"- **OFF:** {metrics['bars_per_sec_off']:.2f} bars/sec",
        f"- **ON:** {metrics['bars_per_sec_on']:.2f} bars/sec",
        "",
        "### Model Calls",
        f"- **OFF:** {metrics['model_calls_off']:,} calls",
        f"- **ON:** {metrics['model_calls_on']:,} calls",
        f"- **Reduction:** {metrics['model_call_reduction']:.1f}%",
        "",
        "### Feature Building Time",
        f"- **OFF:** {metrics['feature_time_total_sec_off']:.1f}s total ({metrics['feature_time_mean_ms_off']:.2f}ms mean)",
        f"- **ON:** {metrics['feature_time_total_sec_on']:.1f}s total ({metrics['feature_time_mean_ms_on']:.2f}ms mean)",
        f"- **Reduction:** {metrics['feature_time_reduction']:.1f}%",
        "",
        "### Phase Timing Breakdown (Pie Chart)",
        "",
        "**OFF:**",
        f"- PreGate: {metrics['t_pregate_total_sec_off']:.2f}s",
        f"- Feature Build: {metrics['t_feature_build_total_sec_off']:.2f}s",
        f"- Model: {metrics['t_model_total_sec_off']:.2f}s",
        f"- Policy: {metrics['t_policy_total_sec_off']:.2f}s",
        f"- I/O: {metrics['t_io_total_sec_off']:.2f}s",
        "",
        "**ON:**",
        f"- PreGate: {metrics['t_pregate_total_sec_on']:.2f}s",
        f"- Feature Build: {metrics['t_feature_build_total_sec_on']:.2f}s",
        f"- Model: {metrics['t_model_total_sec_on']:.2f}s",
        f"- Policy: {metrics['t_policy_total_sec_on']:.2f}s",
        f"- I/O: {metrics['t_io_total_sec_on']:.2f}s",
        "",
        "## Hard Acceptance Criteria (1-Time Plausible)",
        "",
        f"- **Speedup >= 3.0x:** {'✅ PASS' if acceptance['speedup_ok'] else '❌ FAIL'} ({acceptance['speedup_factor']:.2f}x)",
        f"- **PreGate Skip Ratio >= 80%:** {'✅ PASS' if acceptance['pregate_ratio_ok'] else '❌ FAIL'} ({acceptance['pregate_skip_ratio']:.1f}%)",
        f"- **Feature Time Reduction >= 70%:** {'✅ PASS' if acceptance['feature_time_ok'] else '❌ FAIL'} ({acceptance['feature_time_reduction_pct']:.1f}% reduction)",
        f"- **Invariants OK:** {'✅ PASS' if acceptance['invariants_ok'] else '❌ FAIL'}",
        "",
        f"- **Overall:** {'✅ PASS (1-Time Plausible)' if acceptance['pass'] else '❌ FAIL (HTF Cache Required)'}",
        "",
        "## Invariant Checks",
        "",
        f"- **Trades Match (±0.1%):** {'✅ PASS' if invariants['trades_match'] else '❌ FAIL'}",
        f"  - OFF: {invariants['trades_off']:,} trades",
        f"  - ON: {invariants['trades_on']:,} trades",
        f"  - Diff: {invariants['trades_diff_pct']:.3f}%",
        "",
        f"- **Model Calls Not Increased:** {'✅ PASS' if not invariants['model_calls_increased'] else '❌ FAIL'}",
        f"  - OFF: {metrics['model_calls_off']:,} calls",
        f"  - ON: {metrics['model_calls_on']:,} calls",
        "",
        f"- **All Invariants OK:** {'✅ YES' if invariants['all_invariants_ok'] else '❌ NO'}",
        "",
        "## Decision",
        "",
        f"**Status:** {status}",
        "",
        f"**Recommendation:** {recommendation}",
        "",
    ]
    
    if fullyear_estimated_minutes is not None:
        report_lines.extend([
            f"**Estimated FULLYEAR Runtime:** {fullyear_estimated_minutes:.1f} minutes",
            "",
        ])
    else:
        report_lines.extend([
            "**Next Steps:**",
            "1. Implement HTF alignment cache per chunk",
            "2. Precompute HTF mapping once per chunk (not per bar)",
            "3. Reduce HTF alignment overhead from O(n) to O(1) per bar",
            "",
        ])
    
    report_lines.extend([
        "## Input Files",
        "",
        f"- **PreGate OFF:** `{off_path}`",
        f"- **PreGate ON:** `{on_path}`",
        "",
    ])
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(report_lines))
    
    log.info(f"Report written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare replay performance: PreGate OFF vs ON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--off", type=Path, required=True, help="PreGate OFF JSON path")
    parser.add_argument("--on", type=Path, required=True, help="PreGate ON JSON path")
    parser.add_argument("--output", type=Path, required=True, help="Output markdown report path")
    
    args = parser.parse_args()
    
    log.info(f"Loading PreGate OFF: {args.off}")
    try:
        off_data = load_perf_json(args.off)
    except ValueError as e:
        log.error(f"❌ PreGate OFF file is a stub (export failed): {e}")
        log.error("   Cannot compare - OFF run did not complete successfully")
        return 1
    
    log.info(f"Loading PreGate ON: {args.on}")
    try:
        on_data = load_perf_json(args.on)
    except ValueError as e:
        log.error(f"❌ PreGate ON file is a stub (export failed): {e}")
        log.error("   Cannot compare - ON run did not complete successfully")
        return 1
    
    log.info("Calculating metrics...")
    metrics = calculate_speedup(off_data, on_data)
    
    log.info("Verifying invariants...")
    invariants = verify_invariants(off_data, on_data, metrics)
    
    if not invariants["all_invariants_ok"]:
        log.error("❌ Invariant checks FAILED:")
        if not invariants["trades_match"]:
            log.error(f"  - Trades diff: {invariants['trades_diff_pct']:.3f}% (> 0.1%)")
        if invariants["model_calls_increased"]:
            log.error(f"  - Model calls increased: {metrics['model_calls_on']} > {metrics['model_calls_off']}")
    else:
        log.info("✅ All invariants passed")
    
    log.info(f"Speedup factor: {metrics['speedup_factor']:.2f}x")
    
    log.info(f"Generating report: {args.output}")
    generate_report(args.off, args.on, metrics, invariants, args.output)
    
    # DEL 2: Hard-akseptkriterium check
    acceptance = check_hard_acceptance_criteria(metrics, invariants)
    
    # DEL 3: Hard stop decision (based on acceptance criteria)
    if acceptance["pass"]:
        log.info("🎯 STATUS: 1-HOUR TARGET ACHIEVABLE WITHOUT HTF CACHE")
        log.info(f"   Recommendation: GO 1-HOUR (estimated FULLYEAR: {estimate_fullyear_runtime(metrics['speedup_factor'], metrics['wall_clock_sec_on']):.1f} min)")
        log.info(f"   Acceptance: speedup={acceptance['speedup_factor']:.2f}x, pregate_ratio={acceptance['pregate_skip_ratio']:.1f}%, feature_reduction={acceptance['feature_time_reduction_pct']:.1f}%")
    else:
        log.warning("⚠️  STATUS: REPLAY STILL DOMINATED BY FEATURE-BUILDING")
        log.warning("   Recommendation: HTF CACHE REQUIRED")
        log.warning(f"   Acceptance FAIL: speedup={acceptance['speedup_factor']:.2f}x (need >=3.0x), pregate_ratio={acceptance['pregate_skip_ratio']:.1f}% (need >=80%), feature_reduction={acceptance['feature_time_reduction_pct']:.1f}% (need >=70%)")
    
    return 0 if acceptance["pass"] and invariants["all_invariants_ok"] else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
