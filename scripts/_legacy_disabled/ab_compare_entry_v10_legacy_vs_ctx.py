#!/usr/bin/env python3
"""
A/B Comparison: ENTRY_V10 Legacy vs ENTRY_V10_CTX

DEL 5: Mini A/B replay to prove ctx-modell actually changes decisions.

Runs same 1-week window twice:
- Run A: Legacy V10 (ENTRY_CONTEXT_FEATURES_ENABLED=false)
- Run B: Context V10 (ENTRY_CONTEXT_FEATURES_ENABLED=true, GX1_CTX_CONSUMPTION_PROOF=1)

Compares eligibility, volume, entry quality, distribution, performance, and CTX proof.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

# Import model loader worker
from gx1.inference.model_loader_worker import (
    ModelLoadConfig,
    ModelLoadResult,
    load_model_with_timeout,
)


def preflight_bundle_load(
    bundle_dir: Path,
    feature_meta_path: Path,
    seq_scaler_path: Optional[Path],
    snap_scaler_path: Optional[Path],
    model_variant: str,
    timeout_sec: float = 60.0,
    run_tag: str = "unknown",
) -> ModelLoadResult:
    """
    Preflight check: load bundle in isolated worker with timeout.
    
    Parameters
    ----------
    bundle_dir : Path
        Bundle directory
    feature_meta_path : Path
        Feature metadata path
    seq_scaler_path : Optional[Path]
        Sequence scaler path (optional)
    snap_scaler_path : Optional[Path]
        Snapshot scaler path (optional)
    model_variant : str
        Model variant ("v10" or "v10_ctx")
    timeout_sec : float
        Timeout in seconds (default: 60.0)
    run_tag : str
        Tag for logging (e.g., "legacy" or "ctx")
    
    Returns
    -------
    ModelLoadResult
        Result with success status, metadata, or error information
    """
    print(f"\n[PREFLIGHT] {run_tag} bundle load starting...")
    print(f"  Bundle dir: {bundle_dir}")
    print(f"  Variant: {model_variant}")
    print(f"  Timeout: {timeout_sec}s")
    
    start_time = time.perf_counter()
    
    config = ModelLoadConfig(
        bundle_dir=bundle_dir,
        feature_meta_path=feature_meta_path,
        seq_scaler_path=seq_scaler_path,
        snap_scaler_path=snap_scaler_path,
        model_variant=model_variant,
        device="cpu",
        timeout_sec=timeout_sec,
    )
    
    result = load_model_with_timeout(config, timeout_sec=timeout_sec)
    
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    
    if result.success:
        print(f"[PREFLIGHT] {run_tag} bundle load OK in {elapsed_ms:.0f}ms")
        print(f"  Model: {result.model_class_name}")
        print(f"  Params: {result.param_count:,}")
        print(f"  Hash: {result.model_hash}")
    else:
        print(f"[PREFLIGHT] {run_tag} bundle load FAIL")
        print(f"  Reason: {result.error_type}")
        print(f"  Message: {result.error_message}")
        print(f"  Time: {elapsed_ms:.0f}ms")
    
    return result


def run_replay(
    policy_path: Path,
    data_file: Path,
    start_date: str,
    end_date: str,
    output_dir: Path,
    env_vars: Dict[str, str],
    run_tag: str,
) -> Dict[str, Any]:
    """
    Run replay with specified environment variables.
    
    Returns:
        Dict with run results (output_dir, perf_summary_path, success)
    """
    print(f"\n{'='*80}")
    print(f"[A/B] Running {run_tag}")
    print(f"{'='*80}")
    print(f"Output dir: {output_dir}")
    print(f"Env vars: {env_vars}")
    
    # Set environment variables
    original_env = {}
    for key, value in env_vars.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value
        print(f"  {key}={value}")
    
    try:
        # Run replay
        # run_mini_replay_perf.py requires: policy_path data_file output_dir [--start START] [--end END]
        
        cmd = [
            sys.executable,
            "scripts/run_mini_replay_perf.py",
            str(policy_path),
            str(data_file),
            str(output_dir),
            "--start", start_date,
            "--end", end_date,
        ]
        
        print(f"\n[A/B] Command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,  # Don't fail-fast, we'll check results
        )
        
        if result.returncode != 0:
            print(f"\n❌ [A/B] {run_tag} failed with exit code {result.returncode}")
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            return {
                "output_dir": str(output_dir),
                "success": False,
                "error": result.stderr,
            }
        
        # Find perf summary
        perf_summary_path = output_dir / "REPLAY_PERF_SUMMARY.json"
        if not perf_summary_path.exists():
            print(f"\n❌ [A/B] {run_tag} perf summary not found: {perf_summary_path}")
            return {
                "output_dir": str(output_dir),
                "success": False,
                "error": "Perf summary not found",
            }
        
        # Load perf summary
        with open(perf_summary_path, "r") as f:
            perf_summary = json.load(f)
        
        print(f"\n✅ [A/B] {run_tag} completed successfully")
        
        # Verify fast_path_enabled
        fast_path_enabled = perf_summary.get("fast_path_enabled", False)
        if not fast_path_enabled:
            print(f"\n❌ [A/B] {run_tag} fast_path_enabled=false")
            print(f"   This indicates fast path environment variables were not set correctly.")
            print(f"   Expected: GX1_REPLAY_INCREMENTAL_FEATURES=1, GX1_REPLAY_NO_CSV=1, etc.")
            return {
                "output_dir": str(output_dir),
                "perf_summary_path": str(perf_summary_path),
                "perf_summary": perf_summary,
                "success": False,
                "error": "FAST_PATH_DISABLED",
            }
        
        return {
            "output_dir": str(output_dir),
            "perf_summary_path": str(perf_summary_path),
            "perf_summary": perf_summary,
            "success": True,
        }
        
    finally:
        # Restore original environment
        for key, original_value in original_env.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value


def extract_metrics(perf_summary: Dict[str, Any]) -> Dict[str, Any]:
    """Extract metrics from perf summary for comparison."""
    entry_counters = perf_summary.get("entry_counters", {})
    runner_perf = perf_summary.get("runner_perf_metrics", {})
    
    # Eligibility / volume
    metrics = {
        "n_cycles": entry_counters.get("n_cycles", 0),
        "n_eligible_hard": entry_counters.get("n_eligible_hard", 0),
        "n_eligible_cycles": entry_counters.get("n_eligible_cycles", 0),
        "n_candidates": entry_counters.get("n_entry_candidates", 0),
        "n_trades_created": entry_counters.get("n_trades_created", 0),
    }
    
    # Entry quality (p_long stats)
    p_long_values = entry_counters.get("p_long_values", [])
    if p_long_values:
        import numpy as np
        p_long_arr = np.array(p_long_values)
        metrics["p_long_mean"] = float(np.mean(p_long_arr))
        metrics["p_long_p50"] = float(np.median(p_long_arr))
        metrics["p_long_p90"] = float(np.percentile(p_long_arr, 90))
        metrics["p_long_min"] = float(np.min(p_long_arr))
        metrics["p_long_max"] = float(np.max(p_long_arr))
    else:
        metrics["p_long_mean"] = None
        metrics["p_long_p50"] = None
        metrics["p_long_p90"] = None
        metrics["p_long_min"] = None
        metrics["p_long_max"] = None
    
    # Distribution
    metrics["candidate_session_counts"] = entry_counters.get("candidate_session_counts", {})
    metrics["trade_session_counts"] = entry_counters.get("trade_session_counts", {})
    
    # Performance
    metrics["feat_time_sec"] = runner_perf.get("feat_time_sec", 0.0)
    metrics["duration_sec"] = perf_summary.get("duration_sec", 0.0)
    bars_processed = perf_summary.get("bars_processed", 0)
    if bars_processed > 0:
        metrics["feat_ms_per_bar"] = (metrics["feat_time_sec"] / bars_processed) * 1000.0
    else:
        metrics["feat_ms_per_bar"] = None
    
    # CTX-bevis
    metrics["n_ctx_model_calls"] = entry_counters.get("n_ctx_model_calls", 0)
    metrics["n_v10_calls"] = entry_counters.get("n_v10_calls", 0)
    metrics["n_context_built"] = entry_counters.get("n_context_built", 0)
    metrics["n_context_missing_or_invalid"] = entry_counters.get("n_context_missing_or_invalid", 0)
    
    # NEW: Additional ctx diagnostics
    metrics["ctx_expected"] = entry_counters.get("ctx_expected", False)
    metrics["ENTRY_CONTEXT_FEATURES_ENABLED"] = entry_counters.get("ENTRY_CONTEXT_FEATURES_ENABLED", False)
    metrics["entry_models_v10_ctx_enabled"] = entry_counters.get("entry_models_v10_ctx_enabled", False)
    metrics["bundle_supports_context_features"] = entry_counters.get("bundle_supports_context_features", False)
    metrics["model_class_name"] = entry_counters.get("model_class_name", "UNKNOWN")
    v10_none_reason_counts = entry_counters.get("v10_none_reason_counts", {})
    # Get top 5 reasons
    if v10_none_reason_counts:
        sorted_reasons = sorted(v10_none_reason_counts.items(), key=lambda x: x[1], reverse=True)
        metrics["v10_none_reason_counts_top5"] = dict(sorted_reasons[:5])
    else:
        metrics["v10_none_reason_counts_top5"] = {}
    
    # DEL D: CTX consumption proof telemetry
    metrics["ctx_proof_enabled"] = entry_counters.get("ctx_proof_enabled", False)
    metrics["ctx_proof_pass_count"] = entry_counters.get("ctx_proof_pass_count", 0)
    metrics["ctx_proof_fail_count"] = entry_counters.get("ctx_proof_fail_count", 0)
    
    # DEL D: Fast path
    metrics["fast_path_enabled"] = perf_summary.get("fast_path_enabled", False)
    
    return metrics


def compare_runs(run_a: Dict[str, Any], run_b: Dict[str, Any]) -> Dict[str, Any]:
    """Compare two runs and compute deltas."""
    if not run_a.get("success") or not run_b.get("success"):
        return {
            "comparison_valid": False,
            "error": "One or both runs failed",
        }
    
    metrics_a = extract_metrics(run_a["perf_summary"])
    metrics_b = extract_metrics(run_b["perf_summary"])
    
    comparison = {
        "comparison_valid": True,
        "run_a": {
            "output_dir": run_a["output_dir"],
            "metrics": metrics_a,
        },
        "run_b": {
            "output_dir": run_b["output_dir"],
            "metrics": metrics_b,
        },
        "deltas": {},
    }
    
    # Compute deltas
    for key in metrics_a.keys():
        if key in ["candidate_session_counts", "trade_session_counts"]:
            # Skip dict keys (handled separately)
            continue
        
        val_a = metrics_a.get(key)
        val_b = metrics_b.get(key)
        
        if val_a is None or val_b is None:
            comparison["deltas"][key] = None
        elif isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
            delta = val_b - val_a
            delta_pct = (delta / val_a * 100.0) if val_a != 0 else None
            comparison["deltas"][key] = {
                "absolute": delta,
                "percent": delta_pct,
            }
        else:
            comparison["deltas"][key] = {"a": val_a, "b": val_b}
    
    # DEL D: "Did ctx change decisions?" metric
    # Compare p_long distributions (aggregated, not per-bar)
    p_long_mean_a = metrics_a.get("p_long_mean")
    p_long_mean_b = metrics_b.get("p_long_mean")
    p_long_p50_a = metrics_a.get("p_long_p50")
    p_long_p50_b = metrics_b.get("p_long_p50")
    p_long_p90_a = metrics_a.get("p_long_p90")
    p_long_p90_b = metrics_b.get("p_long_p90")
    
    if p_long_mean_a is not None and p_long_mean_b is not None:
        # Simple metric: absolute difference in mean p_long
        p_long_mean_diff = abs(p_long_mean_b - p_long_mean_a)
        p_long_p50_diff = abs(p_long_p50_b - p_long_p50_a) if p_long_p50_a is not None and p_long_p50_b is not None else None
        p_long_p90_diff = abs(p_long_p90_b - p_long_p90_a) if p_long_p90_a is not None and p_long_p90_b is not None else None
        
        # Threshold: if diff > 1e-6, ctx changed decisions (at least in aggregate)
        ctx_changed_decisions = p_long_mean_diff > 1e-6 or (p_long_p50_diff is not None and p_long_p50_diff > 1e-6)
        
        comparison["ctx_decision_change"] = {
            "p_long_mean_diff": p_long_mean_diff,
            "p_long_p50_diff": p_long_p50_diff,
            "p_long_p90_diff": p_long_p90_diff,
            "ctx_changed_decisions": ctx_changed_decisions,
        }
    else:
        comparison["ctx_decision_change"] = {
            "p_long_mean_diff": None,
            "p_long_p50_diff": None,
            "p_long_p90_diff": None,
            "ctx_changed_decisions": None,
        }
    
    return comparison


def check_fail_fast(comparison: Dict[str, Any]) -> list[str]:
    """Check fail-fast conditions. Returns list of failures."""
    failures = []
    
    if not comparison.get("comparison_valid"):
        failures.append("Comparison invalid (one or both runs failed)")
        return failures
    
    metrics_b = comparison["run_b"]["metrics"]
    
    # HARD INVARIANT: If ctx_expected==True, require all ctx invariants
    ctx_expected = metrics_b.get("ctx_expected", False)
    
    if ctx_expected:
        # Fail-fast: n_ctx_model_calls == 0
        n_ctx_calls = metrics_b.get("n_ctx_model_calls", 0)
        if n_ctx_calls == 0:
            v10_none_reason_counts = metrics_b.get("v10_none_reason_counts_top5", {})
            dominant_reason = max(v10_none_reason_counts.items(), key=lambda x: x[1]) if v10_none_reason_counts else ("UNKNOWN", 0)
            failures.append(
                f"FAIL_FAST: CTX_INV_0 violated: ctx_expected=True but n_ctx_model_calls=0. "
                f"Dominant reason: {dominant_reason[0]} (count={dominant_reason[1]}). "
                f"All v10_none_reason_counts: {v10_none_reason_counts}"
            )
        
        # Fail-fast: CTX_INV_1: n_ctx_model_calls == n_v10_calls
        n_v10_calls = metrics_b.get("n_v10_calls", 0)
        if n_ctx_calls != n_v10_calls:
            failures.append(
                f"FAIL_FAST: CTX_INV_1 violated: n_ctx_model_calls ({n_ctx_calls}) != n_v10_calls ({n_v10_calls})"
            )
        
        # Fail-fast: CTX_INV_2: n_context_missing_or_invalid == 0
        n_context_missing = metrics_b.get("n_context_missing_or_invalid", 0)
        if n_context_missing > 0:
            failures.append(
                f"FAIL_FAST: CTX_INV_2 violated: n_context_missing_or_invalid ({n_context_missing}) > 0"
            )
        
        # Fail-fast: CTX_INV_3: n_context_built == n_v10_calls
        n_context_built = metrics_b.get("n_context_built", 0)
        if n_context_built != n_v10_calls:
            failures.append(
                f"FAIL_FAST: CTX_INV_3 violated: n_context_built ({n_context_built}) != n_v10_calls ({n_v10_calls})"
            )
    else:
        # If ctx_expected is False, we should still check if ctx was supposed to be enabled
        # This is a warning, not a hard failure
        if metrics_b.get("entry_models_v10_ctx_enabled", False):
            failures.append(
                f"WARNING: entry_models.v10_ctx.enabled=True but ctx_expected=False. "
                f"ENTRY_CONTEXT_FEATURES_ENABLED={metrics_b.get('ENTRY_CONTEXT_FEATURES_ENABLED', False)}, "
                f"bundle_supports_context_features={metrics_b.get('bundle_supports_context_features', False)}"
            )
    
    return failures


def generate_markdown_report(comparison: Dict[str, Any], output_path: Path) -> None:
    """Generate Markdown comparison report."""
    with open(output_path, "w") as f:
        f.write("# A/B Comparison: ENTRY_V10 Legacy vs ENTRY_V10_CTX\n\n")
        f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        if "baseline_mode" in comparison:
            f.write(f"**Baseline mode:** {comparison['baseline_mode']}\n\n")
        
        # Preflight section (if available)
        if "preflight" in comparison:
            f.write("## Preflight Checks\n\n")
            preflight = comparison["preflight"]
            if preflight.get("legacy", {}).get("success"):
                f.write("**Legacy Bundle:** ✅ OK\n")
                f.write(f"- Model: {preflight['legacy'].get('model_class_name', 'UNKNOWN')}\n")
                f.write(f"- Load Time: {preflight['legacy'].get('load_time_ms', 0):.0f}ms\n")
            else:
                f.write("**Legacy Bundle:** ❌ FAIL\n")
                f.write(f"- Reason: {preflight['legacy'].get('error_type', 'UNKNOWN')}\n")
            
            if preflight.get("ctx", {}).get("success"):
                f.write("**CTX Bundle:** ✅ OK\n")
                f.write(f"- Model: {preflight['ctx'].get('model_class_name', 'UNKNOWN')}\n")
                f.write(f"- Load Time: {preflight['ctx'].get('load_time_ms', 0):.0f}ms\n")
            else:
                f.write("**CTX Bundle:** ❌ FAIL\n")
                f.write(f"- Reason: {preflight['ctx'].get('error_type', 'UNKNOWN')}\n")
            f.write("\n")
        
        if not comparison.get("comparison_valid"):
            f.write("## ❌ Comparison Invalid\n\n")
            f.write(f"Error: {comparison.get('error', 'Unknown error')}\n")
            return
        
        run_a = comparison["run_a"]
        run_b = comparison["run_b"]
        deltas = comparison["deltas"]
        
        # VERDICT section (at the top, as requested)
        f.write("## VERDICT\n\n")
        
        # CTX consumption proof
        n_ctx_calls = run_b["metrics"].get("n_ctx_model_calls", 0)
        ctx_proof_status = "✅ PASS" if n_ctx_calls > 0 else "❌ FAIL"
        f.write(f"**CTX consumption proof:** {ctx_proof_status} (n_ctx_model_calls={n_ctx_calls})\n\n")
        
        # Invariants
        n_v10_calls = run_b["metrics"].get("n_v10_calls", 0)
        n_context_built = run_b["metrics"].get("n_context_built", 0)
        n_context_missing = run_b["metrics"].get("n_context_missing_or_invalid", 0)
        fast_path_a = run_a["metrics"].get("fast_path_enabled", False)
        fast_path_b = run_b["metrics"].get("fast_path_enabled", False)
        ctx_expected_b = run_b["metrics"].get("ctx_expected", False)
        ctx_proof_enabled_b = run_b["metrics"].get("ctx_proof_enabled", False)
        ctx_proof_fail_b = run_b["metrics"].get("ctx_proof_fail_count", 0)
        
        inv_1_pass = n_ctx_calls == n_v10_calls
        inv_2_pass = n_context_missing == 0
        inv_3_pass = n_context_built == n_v10_calls
        
        # CTX_INV_4: ctx_proof_fail_count == 0 if proof is enabled
        ctx_proof_enabled = run_b["metrics"].get("ctx_proof_enabled", False)
        ctx_proof_fail_count = run_b["metrics"].get("ctx_proof_fail_count", 0)
        inv_4_pass = not ctx_proof_enabled or ctx_proof_fail_count == 0
        
        f.write("**Invariants:**\n")
        f.write(f"- CTX_INV_1 (n_ctx_model_calls == n_v10_calls): {'✅ PASS' if inv_1_pass else '❌ FAIL'} ({n_ctx_calls} == {n_v10_calls})\n")
        f.write(f"- CTX_INV_2 (n_context_missing_or_invalid == 0): {'✅ PASS' if inv_2_pass else '❌ FAIL'} ({n_context_missing} == 0)\n")
        f.write(f"- CTX_INV_3 (n_context_built == n_v10_calls): {'✅ PASS' if inv_3_pass else '❌ FAIL'} ({n_context_built} == {n_v10_calls})\n")
        f.write(f"- CTX_INV_4 (ctx_proof_fail_count == 0 if proof enabled): {'✅ PASS' if inv_4_pass else '❌ FAIL'} (proof_enabled={ctx_proof_enabled}, fail_count={ctx_proof_fail_count})\n")
        f.write(f"- FAST_PATH enabled: A={'✅' if fast_path_a else '❌'}, B={'✅' if fast_path_b else '❌'}\n")
        f.write(f"- ctx_expected (B): {'✅' if ctx_expected_b else '❌'}\n")
        
        # Eligibility invariants
        n_cycles_a = run_a["metrics"].get("n_cycles", 0)
        n_eligible_a = run_a["metrics"].get("n_eligible_cycles", 0)
        n_cycles_b = run_b["metrics"].get("n_cycles", 0)
        n_eligible_b = run_b["metrics"].get("n_eligible_cycles", 0)
        
        elig_inv_1_a = 0 <= n_eligible_a <= n_cycles_a
        elig_inv_1_b = 0 <= n_eligible_b <= n_cycles_b
        
        f.write(f"- ELIGIBILITY_INV_1 (0 <= n_eligible <= n_cycles): A={'✅' if elig_inv_1_a else '❌'}, B={'✅' if elig_inv_1_b else '❌'}\n\n")
        
        # Perf regression
        feat_ms_a = run_a["metrics"].get("feat_ms_per_bar")
        feat_ms_b = run_b["metrics"].get("feat_ms_per_bar")
        duration_a = run_a["metrics"].get("duration_sec", 0.0)
        duration_b = run_b["metrics"].get("duration_sec", 0.0)
        
        if feat_ms_a is not None and feat_ms_b is not None:
            feat_ms_delta_pct = ((feat_ms_b - feat_ms_a) / feat_ms_a * 100.0) if feat_ms_a != 0 else None
            perf_regression_status = "✅ NO REGRESSION" if (feat_ms_delta_pct is None or feat_ms_delta_pct <= 10.0) else f"❌ REGRESSION ({feat_ms_delta_pct:.1f}% slower)"
            f.write(f"**Perf regression:** {perf_regression_status}\n")
            f.write(f"- feat_ms_per_bar: A={feat_ms_a:.2f}ms, B={feat_ms_b:.2f}ms, Δ={feat_ms_delta_pct:+.1f}%\n")
        else:
            f.write(f"**Perf regression:** ⚠️  UNAVAILABLE (feat_ms_per_bar missing)\n")
        
        if duration_a > 0 and duration_b > 0:
            duration_delta_pct = ((duration_b - duration_a) / duration_a * 100.0)
            f.write(f"- duration_sec: A={duration_a:.1f}s, B={duration_b:.1f}s, Δ={duration_delta_pct:+.1f}%\n\n")
        else:
            f.write(f"- duration_sec: A={duration_a:.1f}s, B={duration_b:.1f}s\n\n")
        
        # Candidate reduction
        n_candidates_a = run_a["metrics"].get("n_candidates", 0)
        n_candidates_b = run_b["metrics"].get("n_candidates", 0)
        candidates_delta = n_candidates_b - n_candidates_a
        candidates_delta_pct = (candidates_delta / n_candidates_a * 100.0) if n_candidates_a != 0 else None
        
        candidate_reduction_status = "✅ REDUCED" if candidates_delta < 0 else ("❌ INCREASED" if candidates_delta > 0 else "⚠️  NO CHANGE")
        f.write(f"**Candidate reduction:** {candidate_reduction_status}\n")
        f.write(f"- n_candidates: A={n_candidates_a}, B={n_candidates_b}, Δ={candidates_delta:+d} ({candidates_delta_pct:+.1f}%)\n\n")
        
        # Quality movement
        p_long_p50_a = run_a["metrics"].get("p_long_p50")
        p_long_p50_b = run_b["metrics"].get("p_long_p50")
        p_long_p90_a = run_a["metrics"].get("p_long_p90")
        p_long_p90_b = run_b["metrics"].get("p_long_p90")
        
        f.write(f"**Quality movement:**\n")
        if p_long_p50_a is not None and p_long_p50_b is not None:
            p50_delta = p_long_p50_b - p_long_p50_a
            f.write(f"- p_long_p50: A={p_long_p50_a:.4f}, B={p_long_p50_b:.4f}, Δ={p50_delta:+.4f}\n")
        else:
            f.write(f"- p_long_p50: UNAVAILABLE\n")
        
        if p_long_p90_a is not None and p_long_p90_b is not None:
            p90_delta = p_long_p90_b - p_long_p90_a
            f.write(f"- p_long_p90: A={p_long_p90_a:.4f}, B={p_long_p90_b:.4f}, Δ={p90_delta:+.4f}\n")
        else:
            f.write(f"- p_long_p90: UNAVAILABLE\n")
        
        # Check if goes_against_us / early_move available
        # These would be in entry_quality analysis, not in perf summary
        # For now, we'll note they're unavailable
        f.write(f"- goes_against_us_rate: UNAVAILABLE (requires entry_quality analysis)\n")
        f.write(f"- early_move hit-rate: UNAVAILABLE (requires entry_quality analysis)\n")
        f.write(f"\n*Note: Quality metrics (goes_against_us, early_move) unavailable in this window; decision based on distribution + candidate volume + proof + perf.*\n\n")
        
        # DEL D: "Did ctx change decisions?" section
        ctx_decision_change = comparison.get("ctx_decision_change", {})
        if ctx_decision_change.get("ctx_changed_decisions") is not None:
            f.write("**Did ctx change decisions?**\n")
            p_long_mean_diff = ctx_decision_change.get("p_long_mean_diff")
            p_long_p50_diff = ctx_decision_change.get("p_long_p50_diff")
            p_long_p90_diff = ctx_decision_change.get("p_long_p90_diff")
            ctx_changed = ctx_decision_change.get("ctx_changed_decisions", False)
            
            if ctx_changed:
                f.write(f"- ✅ YES: ctx changed decisions (aggregated)\n")
            else:
                f.write(f"- ⚠️  NO: ctx did not change decisions (aggregated)\n")
            
            if p_long_mean_diff is not None:
                f.write(f"- p_long_mean diff: {p_long_mean_diff:.6f}\n")
            if p_long_p50_diff is not None:
                f.write(f"- p_long_p50 diff: {p_long_p50_diff:.6f}\n")
            if p_long_p90_diff is not None:
                f.write(f"- p_long_p90 diff: {p_long_p90_diff:.6f}\n")
            f.write("\n")
        
        # GO/NO-GO decision
        f.write("### GO/NO-GO Decision\n\n")
        
        go_criteria_met = []
        no_go_criteria_met = []
        
        # GO: ctx proof passes
        if n_ctx_calls > 0 and inv_1_pass and inv_2_pass and inv_3_pass:
            go_criteria_met.append("CTX proof and invariants PASS")
        else:
            no_go_criteria_met.append("CTX proof or invariants FAIL")
        
        # GO: candidate reduction OR quality improvement
        if candidates_delta < 0:
            go_criteria_met.append(f"Reduced n_candidates by {abs(candidates_delta)} ({abs(candidates_delta_pct):.1f}%)")
        elif candidates_delta > 0:
            no_go_criteria_met.append(f"Increased n_candidates by {candidates_delta} ({candidates_delta_pct:.1f}%)")
        
        # NO-GO: perf regression >10%
        if feat_ms_delta_pct is not None and feat_ms_delta_pct > 10.0:
            no_go_criteria_met.append(f"Performance regression: {feat_ms_delta_pct:.1f}% slower")
        elif feat_ms_delta_pct is not None and feat_ms_delta_pct < -10.0:
            go_criteria_met.append(f"Performance improvement: {abs(feat_ms_delta_pct):.1f}% faster")
        
        if not no_go_criteria_met and go_criteria_met:
            f.write("**Decision: ✅ GO** - Context features can be activated as default.\n\n")
            f.write("**Reasoning:**\n")
            for criterion in go_criteria_met:
                f.write(f"- {criterion}\n")
        elif no_go_criteria_met:
            f.write("**Decision: ❌ NO-GO** - Context features should not be activated yet.\n\n")
            f.write("**Reasoning:**\n")
            for criterion in no_go_criteria_met:
                f.write(f"- {criterion}\n")
        else:
            f.write("**Decision: ⚠️  INCONCLUSIVE** - More testing needed.\n\n")
        
        f.write("\n---\n\n")
        
        run_a = comparison["run_a"]
        run_b = comparison["run_b"]
        deltas = comparison["deltas"]
        
        f.write("## CTX Configuration and Telemetry (Run B - Context)\n\n")
        run_b_metrics = run_b["metrics"]
        f.write(f"- **ctx_expected:** {run_b_metrics.get('ctx_expected', False)}\n")
        f.write(f"- **ENTRY_CONTEXT_FEATURES_ENABLED:** {run_b_metrics.get('ENTRY_CONTEXT_FEATURES_ENABLED', False)}\n")
        f.write(f"- **entry_models.v10_ctx.enabled:** {run_b_metrics.get('entry_models_v10_ctx_enabled', False)}\n")
        f.write(f"- **bundle.supports_context_features:** {run_b_metrics.get('bundle_supports_context_features', False)}\n")
        f.write(f"- **model_class_name:** {run_b_metrics.get('model_class_name', 'UNKNOWN')}\n")
        f.write(f"- **n_v10_calls:** {run_b_metrics.get('n_v10_calls', 0):,}\n")
        f.write(f"- **n_ctx_model_calls:** {run_b_metrics.get('n_ctx_model_calls', 0):,}\n")
        f.write(f"- **n_context_built:** {run_b_metrics.get('n_context_built', 0):,}\n")
        f.write(f"- **n_context_missing_or_invalid:** {run_b_metrics.get('n_context_missing_or_invalid', 0):,}\n")
        v10_none_reason_counts = run_b_metrics.get('v10_none_reason_counts_top5', {})
        if v10_none_reason_counts:
            f.write(f"- **v10_none_reason_counts (top 5):**\n")
            for reason, count in sorted(v10_none_reason_counts.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  - `{reason}`: {count}\n")
        else:
            f.write(f"- **v10_none_reason_counts:** (empty)\n")
        f.write("\n")
        
        # CTX Invariants (detailed)
        ctx_expected = run_b_metrics.get('ctx_expected', False)
        if ctx_expected:
            inv_0_pass = run_b_metrics.get('n_ctx_model_calls', 0) > 0
            f.write("**CTX Invariants (Run B):**\n")
            f.write(f"- CTX_INV_0 (if ctx_expected: n_ctx_model_calls > 0): {'✅ PASS' if inv_0_pass else '❌ FAIL'}\n")
            f.write(f"- CTX_INV_1 (n_ctx_model_calls == n_v10_calls): {'✅ PASS' if inv_1_pass else '❌ FAIL'}\n")
            f.write(f"- CTX_INV_2 (n_context_missing_or_invalid == 0): {'✅ PASS' if inv_2_pass else '❌ FAIL'}\n")
            f.write(f"- CTX_INV_3 (n_context_built == n_v10_calls): {'✅ PASS' if inv_3_pass else '❌ FAIL'}\n\n")
        else:
            f.write("**CTX Invariants:** ⚠️  ctx_expected=False, invariants not applicable\n\n")
        
        f.write("## Run Configuration\n\n")
        f.write("### Run A (Legacy)\n")
        f.write(f"- Output: `{run_a['output_dir']}`\n")
        f.write("- Config: `ENTRY_CONTEXT_FEATURES_ENABLED=false`, `entry_models.v10.enabled=true`\n\n")
        
        f.write("### Run B (Context)\n")
        f.write(f"- Output: `{run_b['output_dir']}`\n")
        f.write("- Config: `ENTRY_CONTEXT_FEATURES_ENABLED=true`, `entry_models.v10_ctx.enabled=true`, `GX1_CTX_CONSUMPTION_PROOF=1`\n\n")
        
        f.write("## Eligibility / Volume\n\n")
        f.write("| Metric | Legacy (A) | Context (B) | Delta | Delta % |\n")
        f.write("|--------|-----------|------------|-------|--------|\n")
        
        volume_metrics = ["n_cycles", "n_eligible_cycles", "n_candidates", "n_trades_created"]
        for metric in volume_metrics:
            val_a = run_a["metrics"].get(metric, 0)
            val_b = run_b["metrics"].get(metric, 0)
            delta_info = deltas.get(metric, {})
            delta_abs = delta_info.get("absolute", 0) if isinstance(delta_info, dict) else 0
            delta_pct = delta_info.get("percent", 0) if isinstance(delta_info, dict) else None
            
            delta_str = f"{delta_abs:+.0f}" if isinstance(delta_abs, (int, float)) else "N/A"
            delta_pct_str = f"{delta_pct:+.1f}%" if delta_pct is not None else "N/A"
            
            f.write(f"| {metric} | {val_a} | {val_b} | {delta_str} | {delta_pct_str} |\n")
        
        f.write("\n## Entry Quality (p_long)\n\n")
        f.write("| Metric | Legacy (A) | Context (B) | Delta |\n")
        f.write("|--------|-----------|------------|-------|\n")
        
        p_long_metrics = ["p_long_mean", "p_long_p50", "p_long_p90"]
        for metric in p_long_metrics:
            val_a = run_a["metrics"].get(metric)
            val_b = run_b["metrics"].get(metric)
            delta_info = deltas.get(metric, {})
            delta_abs = delta_info.get("absolute", 0.0) if isinstance(delta_info, dict) else 0.0
            
            val_a_str = f"{val_a:.4f}" if val_a is not None else "N/A"
            val_b_str = f"{val_b:.4f}" if val_b is not None else "N/A"
            delta_str = f"{delta_abs:+.4f}" if isinstance(delta_abs, (int, float)) else "N/A"
            
            f.write(f"| {metric} | {val_a_str} | {val_b_str} | {delta_str} |\n")
        
        f.write("\n## Distribution\n\n")
        f.write("### Candidate Session Distribution\n\n")
        f.write("| Session | Legacy (A) | Context (B) | Delta |\n")
        f.write("|---------|-----------|------------|-------|\n")
        
        sessions_a = run_a["metrics"].get("candidate_session_counts", {})
        sessions_b = run_b["metrics"].get("candidate_session_counts", {})
        all_sessions = set(sessions_a.keys()) | set(sessions_b.keys())
        
        for session in sorted(all_sessions):
            count_a = sessions_a.get(session, 0)
            count_b = sessions_b.get(session, 0)
            delta = count_b - count_a
            f.write(f"| {session} | {count_a} | {count_b} | {delta:+.0f} |\n")
        
        f.write("\n### Trade Session Distribution\n\n")
        f.write("| Session | Legacy (A) | Context (B) | Delta |\n")
        f.write("|---------|-----------|------------|-------|\n")
        
        trade_sessions_a = run_a["metrics"].get("trade_session_counts", {})
        trade_sessions_b = run_b["metrics"].get("trade_session_counts", {})
        all_trade_sessions = set(trade_sessions_a.keys()) | set(trade_sessions_b.keys())
        
        for session in sorted(all_trade_sessions):
            count_a = trade_sessions_a.get(session, 0)
            count_b = trade_sessions_b.get(session, 0)
            delta = count_b - count_a
            f.write(f"| {session} | {count_a} | {count_b} | {delta:+.0f} |\n")
        
        f.write("\n## Performance\n\n")
        f.write("| Metric | Legacy (A) | Context (B) | Delta | Delta % |\n")
        f.write("|--------|-----------|------------|-------|--------|\n")
        
        perf_metrics = ["feat_time_sec", "feat_ms_per_bar", "duration_sec"]
        for metric in perf_metrics:
            val_a = run_a["metrics"].get(metric)
            val_b = run_b["metrics"].get(metric)
            delta_info = deltas.get(metric, {})
            delta_abs = delta_info.get("absolute", 0.0) if isinstance(delta_info, dict) else 0.0
            delta_pct = delta_info.get("percent", 0.0) if isinstance(delta_info, dict) else None
            
            val_a_str = f"{val_a:.2f}" if val_a is not None else "N/A"
            val_b_str = f"{val_b:.2f}" if val_b is not None else "N/A"
            delta_str = f"{delta_abs:+.2f}" if isinstance(delta_abs, (int, float)) else "N/A"
            delta_pct_str = f"{delta_pct:+.1f}%" if delta_pct is not None else "N/A"
            
            f.write(f"| {metric} | {val_a_str} | {val_b_str} | {delta_str} | {delta_pct_str} |\n")
        
        f.write("\n## CTX-Bevis\n\n")
        f.write("| Metric | Context (B) | Status |\n")
        f.write("|--------|------------|--------|\n")
        
        n_ctx_calls = run_b["metrics"].get("n_ctx_model_calls", 0)
        n_v10_calls = run_b["metrics"].get("n_v10_calls", 0)
        n_context_built = run_b["metrics"].get("n_context_built", 0)
        n_context_missing = run_b["metrics"].get("n_context_missing_or_invalid", 0)
        
        f.write(f"| n_ctx_model_calls | {n_ctx_calls} | {'✅' if n_ctx_calls > 0 else '❌'} |\n")
        f.write(f"| n_v10_calls | {n_v10_calls} | - |\n")
        f.write(f"| n_context_built | {n_context_built} | - |\n")
        f.write(f"| n_context_missing_or_invalid | {n_context_missing} | {'✅' if n_context_missing == 0 else '❌'} |\n")
        
        # Check invariants
        f.write("\n### Invariants\n\n")
        f.write("| Invariant | Status |\n")
        f.write("|-----------|--------|\n")
        
        inv_1_pass = n_ctx_calls == n_v10_calls
        inv_2_pass = n_context_missing == 0
        inv_3_pass = n_context_built == n_v10_calls
        
        f.write(f"| CTX_INV_1: n_ctx_model_calls == n_v10_calls | {'✅ PASS' if inv_1_pass else '❌ FAIL'} |\n")
        f.write(f"| CTX_INV_2: n_context_missing_or_invalid == 0 | {'✅ PASS' if inv_2_pass else '❌ FAIL'} |\n")
        f.write(f"| CTX_INV_3: n_context_built == n_v10_calls | {'✅ PASS' if inv_3_pass else '❌ FAIL'} |\n")
        
        f.write("\n## Conclusion\n\n")
        
        # GO/NO-GO criteria
        candidates_delta = deltas.get("n_candidates", {}).get("absolute", 0) if isinstance(deltas.get("n_candidates"), dict) else 0
        perf_regression = deltas.get("feat_ms_per_bar", {}).get("percent", 0) if isinstance(deltas.get("feat_ms_per_bar"), dict) else 0
        
        go_criteria = []
        no_go_criteria = []
        
        if candidates_delta < 0:
            go_criteria.append(f"✅ Reduced n_candidates by {abs(candidates_delta)}")
        elif candidates_delta > 0:
            no_go_criteria.append(f"❌ Increased n_candidates by {candidates_delta}")
        
        if perf_regression > 10:
            no_go_criteria.append(f"❌ Performance regression: {perf_regression:.1f}% slower")
        elif perf_regression < -10:
            go_criteria.append(f"✅ Performance improvement: {abs(perf_regression):.1f}% faster")
        
        if n_ctx_calls == 0:
            no_go_criteria.append("❌ CTX not used (n_ctx_model_calls == 0)")
        else:
            go_criteria.append(f"✅ CTX used ({n_ctx_calls} calls)")
        
        if not inv_1_pass or not inv_2_pass or not inv_3_pass:
            no_go_criteria.append("❌ CTX invariants failed")
        else:
            go_criteria.append("✅ All CTX invariants PASS")
        
        if go_criteria:
            f.write("### ✅ GO Criteria\n\n")
            for criterion in go_criteria:
                f.write(f"- {criterion}\n")
            f.write("\n")
        
        if no_go_criteria:
            f.write("### ❌ NO-GO Criteria\n\n")
            for criterion in no_go_criteria:
                f.write(f"- {criterion}\n")
            f.write("\n")
        
        if not no_go_criteria and go_criteria:
            f.write("**Decision: ✅ GO** - Context features can be activated as default.\n")
        elif no_go_criteria:
            f.write("**Decision: ❌ NO-GO** - Context features should not be activated yet.\n")
        else:
            f.write("**Decision: ⚠️  INCONCLUSIVE** - More testing needed.\n")


def main():
    parser = argparse.ArgumentParser(description="A/B Compare ENTRY_V10 Legacy vs CTX")
    parser.add_argument("--policy", type=str, required=True, help="Policy YAML path")
    parser.add_argument("--data_file", type=str, default=None, help="M5 data file (Parquet). If not provided, will search for it.")
    parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output_base", type=str, default="gx1/wf_runs/ab_compare", help="Base output directory")
    parser.add_argument("--legacy_bundle_dir", type=str, default=None, help="Legacy V10 bundle dir (optional)")
    parser.add_argument("--ctx_bundle_dir", type=str, required=True, help="CTX bundle dir")
    parser.add_argument("--feature_meta_path", type=str, required=True, help="Feature metadata path")
    parser.add_argument("--seq_scaler_path", type=str, default=None, help="Seq scaler path (optional)")
    parser.add_argument("--snap_scaler_path", type=str, default=None, help="Snap scaler path (optional)")
    parser.add_argument("--model_load_timeout_sec", type=float, default=60.0, help="Model load timeout in seconds (default: 60.0)")
    parser.add_argument("--preflight_only", action="store_true", help="Only run preflight checks, do not run replay")
    parser.add_argument("--baseline_mode", type=str, choices=["legacy", "ctx_null"], default="legacy", help="Baseline mode: legacy or ctx_null (default: legacy)")
    
    args = parser.parse_args()
    
    policy_path = Path(args.policy)
    if not policy_path.exists():
        print(f"❌ Policy not found: {policy_path}")
        return 1
    
    # Find data file
    data_file = args.data_file
    if data_file:
        data_file = Path(data_file)
        if not data_file.exists():
            print(f"❌ Data file not found: {data_file}")
            return 1
    else:
        # Try to find data file from environment or search
        data_file_env = os.getenv("M5_DATA")
        if data_file_env and Path(data_file_env).exists():
            data_file = Path(data_file_env)
        else:
            # Try to find a recent data file
            import glob
            data_candidates = glob.glob("data/raw/*xauusd*m5*.parquet") + glob.glob("gx1/wf_runs/*/test_data*.parquet") + glob.glob("gx1/tests/data/*xauusd*m5*.parquet")
            if data_candidates:
                data_file = Path(sorted(data_candidates, key=lambda p: Path(p).stat().st_mtime, reverse=True)[0])
                print(f"[A/B] Using data file: {data_file}")
            else:
                print(f"❌ Could not find M5 data file. Set --data_file or M5_DATA environment variable.")
                return 1
    
    output_base = Path(args.output_base)
    output_base.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_a = output_base / f"legacy_{timestamp}"
    output_b = output_base / f"ctx_{timestamp}"
    
    # Load policy to modify
    import yaml
    with open(policy_path, "r") as f:
        policy = yaml.safe_load(f)
    
    # Prepare policies for Run A and Run B based on baseline_mode
    baseline_mode = args.baseline_mode
    policy_a = policy.copy()
    policy_b = policy.copy()
    entry_models_a = policy_a.get("entry_models", {})
    entry_models_b = policy_b.get("entry_models", {})
    if baseline_mode == "legacy":
        # Run A: Legacy V10
        entry_models_a["v10"] = entry_models_a.get("v10", {})
        entry_models_a["v10"]["enabled"] = True
        entry_models_a["v10_ctx"] = entry_models_a.get("v10_ctx", {})
        entry_models_a["v10_ctx"]["enabled"] = False
        # Run B: CTX V10
        entry_models_b["v10"] = entry_models_b.get("v10", {})
        entry_models_b["v10"]["enabled"] = False
        entry_models_b["v10_ctx"] = entry_models_b.get("v10_ctx", {})
        entry_models_b["v10_ctx"]["enabled"] = True
        entry_models_b["v10_ctx"]["bundle_dir"] = args.ctx_bundle_dir
        entry_models_b["v10_ctx"]["feature_meta_path"] = args.feature_meta_path
        if args.seq_scaler_path:
            entry_models_b["v10_ctx"]["seq_scaler_path"] = args.seq_scaler_path
        if args.snap_scaler_path:
            entry_models_b["v10_ctx"]["snap_scaler_path"] = args.snap_scaler_path
    else:
        # ctx_null baseline: both runs use ctx model, Run A uses null context
        entry_models_a["v10"] = entry_models_a.get("v10", {})
        entry_models_a["v10"]["enabled"] = False
        entry_models_a["v10_ctx"] = entry_models_a.get("v10_ctx", {})
        entry_models_a["v10_ctx"]["enabled"] = True
        entry_models_a["v10_ctx"]["bundle_dir"] = args.ctx_bundle_dir
        entry_models_a["v10_ctx"]["feature_meta_path"] = args.feature_meta_path
        entry_models_b["v10"] = entry_models_b.get("v10", {})
        entry_models_b["v10"]["enabled"] = False
        entry_models_b["v10_ctx"] = entry_models_b.get("v10_ctx", {})
        entry_models_b["v10_ctx"]["enabled"] = True
        entry_models_b["v10_ctx"]["bundle_dir"] = args.ctx_bundle_dir
        entry_models_b["v10_ctx"]["feature_meta_path"] = args.feature_meta_path
        if args.seq_scaler_path:
            entry_models_a["v10_ctx"]["seq_scaler_path"] = args.seq_scaler_path
            entry_models_b["v10_ctx"]["seq_scaler_path"] = args.seq_scaler_path
        if args.snap_scaler_path:
            entry_models_a["v10_ctx"]["snap_scaler_path"] = args.snap_scaler_path
            entry_models_b["v10_ctx"]["snap_scaler_path"] = args.snap_scaler_path
    
    policy_a_path = output_base / f"policy_A_{baseline_mode}_{timestamp}.yaml"
    with open(policy_a_path, "w") as f:
        yaml.dump(policy_a, f)
    policy_b_path = output_base / f"policy_B_ctx_{timestamp}.yaml"
    with open(policy_b_path, "w") as f:
        yaml.dump(policy_b, f)
    
    # Determine bundle paths for preflight
    feature_meta_path = Path(args.feature_meta_path)
    seq_scaler_path = Path(args.seq_scaler_path) if args.seq_scaler_path else None
    snap_scaler_path = Path(args.snap_scaler_path) if args.snap_scaler_path else None
    
    # Find legacy bundle dir (from policy or args)
    legacy_bundle_dir = None
    if args.legacy_bundle_dir:
        legacy_bundle_dir = Path(args.legacy_bundle_dir)
    else:
        # Try to find from policy
        legacy_model_path = entry_models_a.get("v10", {}).get("model_path")
        if legacy_model_path:
            legacy_bundle_dir = Path(legacy_model_path).parent
    
    ctx_bundle_dir = Path(args.ctx_bundle_dir)
    
    # Determine ctx_expected
    ctx_expected = (
        os.getenv("ENTRY_CONTEXT_FEATURES_ENABLED", "false").lower() == "true" or
        entry_models_b.get("v10_ctx", {}).get("enabled", False)
    )
    
    # Preflight checks
    print("\n" + "="*80)
    print("[PREFLIGHT] Model Loading Preflight Checks")
    print("="*80)
    
    preflight_results = {}
    
    # Preflight legacy bundle (if needed)
    if legacy_bundle_dir and legacy_bundle_dir.exists():
        legacy_result = preflight_bundle_load(
            bundle_dir=legacy_bundle_dir,
            feature_meta_path=feature_meta_path,
            seq_scaler_path=seq_scaler_path,
            snap_scaler_path=snap_scaler_path,
            model_variant="v10",
            timeout_sec=args.model_load_timeout_sec,
            run_tag="legacy",
        )
        preflight_results["legacy"] = legacy_result
        
        if not legacy_result.success:
            print(f"\n❌ [PREFLIGHT] Legacy bundle preflight FAILED")
            print(f"   Reason: {legacy_result.error_type}")
            print(f"   Message: {legacy_result.error_message}")
            if getattr(legacy_result, "traceback_excerpt", None):
                print("   Traceback (first 30 lines):")
                print(legacy_result.traceback_excerpt)
            print(f"   Bundle dir: {legacy_bundle_dir}")
            if args.preflight_only:
                return 2
            else:
                if baseline_mode == "legacy":
                    print(f"\n❌ Cannot proceed with replay: legacy bundle preflight failed")
                    sys.exit(2)
    else:
        print(f"[PREFLIGHT] Legacy bundle dir not found or not specified: {legacy_bundle_dir}")
        if not args.preflight_only:
            print(f"⚠️  Warning: Legacy bundle preflight skipped, but replay will attempt to load it")
    
    # Preflight ctx bundle (if ctx_expected or explicitly requested)
    if ctx_expected or args.preflight_only:
        if not ctx_bundle_dir.exists():
            print(f"\n❌ [PREFLIGHT] CTX bundle dir not found: {ctx_bundle_dir}")
            if args.preflight_only:
                return 2
            else:
                print(f"\n❌ Cannot proceed with replay: CTX bundle dir not found")
                sys.exit(2)
        
        ctx_result = preflight_bundle_load(
            bundle_dir=ctx_bundle_dir,
            feature_meta_path=feature_meta_path,
            seq_scaler_path=seq_scaler_path,
            snap_scaler_path=snap_scaler_path,
            model_variant="v10_ctx",
            timeout_sec=args.model_load_timeout_sec,
            run_tag="ctx",
        )
        preflight_results["ctx"] = ctx_result
        
        if not ctx_result.success:
            print(f"\n❌ [PREFLIGHT] CTX bundle preflight FAILED")
            print(f"   Reason: {ctx_result.error_type}")
            print(f"   Message: {ctx_result.error_message}")
            if getattr(ctx_result, "traceback_excerpt", None):
                print("   Traceback (first 30 lines):")
                print(ctx_result.traceback_excerpt)
            print(f"   Bundle dir: {ctx_bundle_dir}")
            if args.preflight_only:
                return 2
            else:
                print(f"\n❌ Cannot proceed with replay: CTX bundle preflight failed")
                sys.exit(2)
        
        # Verify ctx bundle metadata if ctx_expected
        if ctx_expected:
            # Try to load bundle metadata to verify supports_context_features
            try:
                bundle_metadata_path = ctx_bundle_dir / "bundle_metadata.json"
                if bundle_metadata_path.exists():
                    with open(bundle_metadata_path, "r") as f:
                        bundle_metadata = json.load(f)
                    
                    supports_context_features = bundle_metadata.get("supports_context_features", False)
                    expected_ctx_cat_dim = bundle_metadata.get("expected_ctx_cat_dim", 0)
                    expected_ctx_cont_dim = bundle_metadata.get("expected_ctx_cont_dim", 0)
                    feature_contract_hash = bundle_metadata.get("feature_contract_hash")
                    
                    if not supports_context_features:
                        print(f"\n❌ [PREFLIGHT] CTX bundle does not support context features")
                        print(f"   bundle_dir: {ctx_bundle_dir}")
                        print(f"   supports_context_features: {supports_context_features}")
                        if args.preflight_only:
                            return 2
                        else:
                            sys.exit(2)
                    
                    if expected_ctx_cat_dim != 5 or expected_ctx_cont_dim != 2:
                        print(f"\n❌ [PREFLIGHT] CTX bundle has incorrect context dimensions")
                        print(f"   Expected: ctx_cat_dim=5, ctx_cont_dim=2")
                        print(f"   Got: ctx_cat_dim={expected_ctx_cat_dim}, ctx_cont_dim={expected_ctx_cont_dim}")
                        if args.preflight_only:
                            return 2
                        else:
                            sys.exit(2)
                    
                    print(f"[PREFLIGHT] CTX bundle metadata verified:")
                    print(f"   supports_context_features: {supports_context_features}")
                    print(f"   ctx_cat_dim: {expected_ctx_cat_dim}")
                    print(f"   ctx_cont_dim: {expected_ctx_cont_dim}")
                    if feature_contract_hash:
                        print(f"   feature_contract_hash: {feature_contract_hash}")
            except Exception as e:
                print(f"⚠️  [PREFLIGHT] Could not verify CTX bundle metadata: {e}")
                if args.preflight_only:
                    return 2
    
    print("="*80)
    
    # If --preflight_only, generate preflight report and exit
    if args.preflight_only:
        preflight_report = {
            "timestamp": timestamp,
            "baseline_mode": baseline_mode,
            "legacy": {
                "bundle_dir": str(legacy_bundle_dir) if legacy_bundle_dir else None,
                "success": preflight_results.get("legacy", ModelLoadResult(
                    success=False,
                    model_class_name="UNKNOWN",
                    param_count=0,
                    model_hash="",
                    error_type="NOT_CHECKED",
                    error_message="Legacy bundle not checked",
                )).success,
                "model_class_name": preflight_results.get("legacy", ModelLoadResult(
                    success=False,
                    model_class_name="UNKNOWN",
                    param_count=0,
                    model_hash="",
                )).model_class_name,
                "param_count": preflight_results.get("legacy", ModelLoadResult(
                    success=False,
                    model_class_name="UNKNOWN",
                    param_count=0,
                    model_hash="",
                )).param_count,
                "model_hash": preflight_results.get("legacy", ModelLoadResult(
                    success=False,
                    model_class_name="UNKNOWN",
                    param_count=0,
                    model_hash="",
                )).model_hash,
                "load_time_ms": preflight_results.get("legacy", ModelLoadResult(
                    success=False,
                    model_class_name="UNKNOWN",
                    param_count=0,
                    model_hash="",
                )).load_time_sec * 1000.0,
                "error_type": preflight_results.get("legacy", ModelLoadResult(
                    success=False,
                    model_class_name="UNKNOWN",
                    param_count=0,
                    model_hash="",
                )).error_type,
                "error_message": preflight_results.get("legacy", ModelLoadResult(
                    success=False,
                    model_class_name="UNKNOWN",
                    param_count=0,
                    model_hash="",
                )).error_message,
                "traceback_excerpt": preflight_results.get("legacy", ModelLoadResult(
                    success=False,
                    model_class_name="UNKNOWN",
                    param_count=0,
                    model_hash="",
                )).traceback_excerpt,
            },
            "ctx": {
                "bundle_dir": str(ctx_bundle_dir),
                "success": preflight_results.get("ctx", ModelLoadResult(
                    success=False,
                    model_class_name="UNKNOWN",
                    param_count=0,
                    model_hash="",
                    error_type="NOT_CHECKED",
                    error_message="CTX bundle not checked",
                )).success,
                "model_class_name": preflight_results.get("ctx", ModelLoadResult(
                    success=False,
                    model_class_name="UNKNOWN",
                    param_count=0,
                    model_hash="",
                )).model_class_name,
                "param_count": preflight_results.get("ctx", ModelLoadResult(
                    success=False,
                    model_class_name="UNKNOWN",
                    param_count=0,
                    model_hash="",
                )).param_count,
                "model_hash": preflight_results.get("ctx", ModelLoadResult(
                    success=False,
                    model_class_name="UNKNOWN",
                    param_count=0,
                    model_hash="",
                )).model_hash,
                "load_time_ms": preflight_results.get("ctx", ModelLoadResult(
                    success=False,
                    model_class_name="UNKNOWN",
                    param_count=0,
                    model_hash="",
                )).load_time_sec * 1000.0,
                "error_type": preflight_results.get("ctx", ModelLoadResult(
                    success=False,
                    model_class_name="UNKNOWN",
                    param_count=0,
                    model_hash="",
                )).error_type,
                "error_message": preflight_results.get("ctx", ModelLoadResult(
                    success=False,
                    model_class_name="UNKNOWN",
                    param_count=0,
                    model_hash="",
                )).error_message,
                "traceback_excerpt": preflight_results.get("ctx", ModelLoadResult(
                    success=False,
                    model_class_name="UNKNOWN",
                    param_count=0,
                    model_hash="",
                )).traceback_excerpt,
            },
        }
        
        # Save preflight report
        preflight_json_path = output_base / "ab_compare_preflight.json"
        with open(preflight_json_path, "w") as f:
            json.dump(preflight_report, f, indent=2)
        print(f"\n✅ Preflight report saved: {preflight_json_path}")
        
        # Generate Markdown report
        preflight_md_path = output_base / "ab_compare_preflight.md"
        with open(preflight_md_path, "w") as f:
            f.write("# A/B Comparison Preflight Report\n\n")
            f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Baseline mode:** {baseline_mode}\n\n")
            
            f.write("## Legacy Bundle\n\n")
            legacy_info = preflight_report["legacy"]
            if legacy_info["success"]:
                f.write(f"- **Status:** ✅ OK\n")
                f.write(f"- **Model:** {legacy_info['model_class_name']}\n")
                f.write(f"- **Params:** {legacy_info['param_count']:,}\n")
                f.write(f"- **Hash:** {legacy_info['model_hash']}\n")
                f.write(f"- **Load Time:** {legacy_info['load_time_ms']:.0f}ms\n")
            else:
                f.write(f"- **Status:** ❌ FAIL\n")
                f.write(f"- **Reason:** {legacy_info['error_type']}\n")
                f.write(f"- **Message:** {legacy_info['error_message']}\n")
                if legacy_info.get("traceback_excerpt"):
                    f.write("\n### LEGACY_PREFLIGHT_DIAGNOSIS\n\n")
                    f.write("```\n")
                    f.write(legacy_info["traceback_excerpt"])
                    f.write("\n```\n")
            
            f.write("\n## CTX Bundle\n\n")
            ctx_info = preflight_report["ctx"]
            if ctx_info["success"]:
                f.write(f"- **Status:** ✅ OK\n")
                f.write(f"- **Model:** {ctx_info['model_class_name']}\n")
                f.write(f"- **Params:** {ctx_info['param_count']:,}\n")
                f.write(f"- **Hash:** {ctx_info['model_hash']}\n")
                f.write(f"- **Load Time:** {ctx_info['load_time_ms']:.0f}ms\n")
            else:
                f.write(f"- **Status:** ❌ FAIL\n")
                f.write(f"- **Reason:** {ctx_info['error_type']}\n")
                f.write(f"- **Message:** {ctx_info['error_message']}\n")
                if ctx_info.get("traceback_excerpt"):
                    f.write("\n### CTX_PREFLIGHT_DIAGNOSIS\n\n")
                    f.write("```\n")
                    f.write(ctx_info["traceback_excerpt"])
                    f.write("\n```\n")
        
        print(f"✅ Preflight Markdown report saved: {preflight_md_path}")
        
        # Exit with appropriate code
        all_ok = (
            (not legacy_bundle_dir or preflight_results.get("legacy", ModelLoadResult(
                success=False,
                model_class_name="UNKNOWN",
                param_count=0,
                model_hash="",
            )).success) and
            (not ctx_expected or preflight_results.get("ctx", ModelLoadResult(
                success=False,
                model_class_name="UNKNOWN",
                param_count=0,
                model_hash="",
            )).success)
        )
        
        if all_ok:
            print("\n✅ All preflight checks passed")
            return 0
        else:
            print("\n❌ Preflight checks failed")
            return 2
    
    # Enforce fast path environment variables
    print("\n[FAST_PATH] Enforcing fast path environment variables...")
    fast_path_vars = {
        "GX1_REPLAY_INCREMENTAL_FEATURES": "1",
        "GX1_REPLAY_NO_CSV": "1",
        "GX1_FEATURE_USE_NP_ROLLING": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    for key, value in fast_path_vars.items():
        os.environ[key] = value
        print(f"  {key}={value}")
    
    # Run A env vars
    if baseline_mode == "legacy":
        env_vars_a = {
            "ENTRY_CONTEXT_FEATURES_ENABLED": "false",
        }
    else:
        # ctx_null baseline: enable ctx but force null context
        env_vars_a = {
            "ENTRY_CONTEXT_FEATURES_ENABLED": "true",
            "GX1_CTX_NULL_BASELINE": "1",
        }
    run_a = run_replay(
        policy_path=policy_a_path,
        data_file=data_file,
        start_date=args.start,
        end_date=args.end,
        output_dir=output_a,
        env_vars=env_vars_a,
        run_tag="Legacy (A)",
    )
    
    # Run B: Context (always ctx run)
    env_vars_b = {
        "ENTRY_CONTEXT_FEATURES_ENABLED": "true",
        "GX1_CTX_CONSUMPTION_PROOF": "1",
    }
    run_b = run_replay(
        policy_path=policy_b_path,
        data_file=data_file,
        start_date=args.start,
        end_date=args.end,
        output_dir=output_b,
        env_vars=env_vars_b,
        run_tag="Context (B)",
    )
    
    # Check fast path for both runs
    if not run_a.get("success", False) or run_a.get("error") == "FAST_PATH_DISABLED":
        print("\n❌ Run A (Legacy) failed: FAST_PATH_DISABLED")
        sys.exit(3)
    if not run_b.get("success", False) or run_b.get("error") == "FAST_PATH_DISABLED":
        print("\n❌ Run B (Context) failed: FAST_PATH_DISABLED")
        sys.exit(3)
    
    # Compare
    comparison = compare_runs(run_a, run_b)
    comparison["baseline_mode"] = baseline_mode
    
    # Add preflight results to comparison
    if preflight_results:
        comparison["preflight"] = {
            "legacy": {
                "success": preflight_results.get("legacy", ModelLoadResult(
                    success=False,
                    model_class_name="UNKNOWN",
                    param_count=0,
                    model_hash="",
                )).success,
                "model_class_name": preflight_results.get("legacy", ModelLoadResult(
                    success=False,
                    model_class_name="UNKNOWN",
                    param_count=0,
                    model_hash="",
                )).model_class_name,
                "load_time_ms": preflight_results.get("legacy", ModelLoadResult(
                    success=False,
                    model_class_name="UNKNOWN",
                    param_count=0,
                    model_hash="",
                )).load_time_sec * 1000.0,
                "error_type": preflight_results.get("legacy", ModelLoadResult(
                    success=False,
                    model_class_name="UNKNOWN",
                    param_count=0,
                    model_hash="",
                )).error_type,
            },
            "ctx": {
                "success": preflight_results.get("ctx", ModelLoadResult(
                    success=False,
                    model_class_name="UNKNOWN",
                    param_count=0,
                    model_hash="",
                )).success,
                "model_class_name": preflight_results.get("ctx", ModelLoadResult(
                    success=False,
                    model_class_name="UNKNOWN",
                    param_count=0,
                    model_hash="",
                )).model_class_name,
                "load_time_ms": preflight_results.get("ctx", ModelLoadResult(
                    success=False,
                    model_class_name="UNKNOWN",
                    param_count=0,
                    model_hash="",
                )).load_time_sec * 1000.0,
                "error_type": preflight_results.get("ctx", ModelLoadResult(
                    success=False,
                    model_class_name="UNKNOWN",
                    param_count=0,
                    model_hash="",
                )).error_type,
            },
        }
    
    # Check fail-fast conditions
    failures = check_fail_fast(comparison)
    if failures:
        print("\n" + "="*80)
        print("❌ FAIL-FAST CONDITIONS FAILED")
        print("="*80)
        for failure in failures:
            print(f"  - {failure}")
        print("="*80 + "\n")
        return 1
    
    # Save comparison
    comparison_path = output_base / "ab_compare.json"
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n✅ Comparison saved: {comparison_path}")
    
    # Generate Markdown report
    report_path = output_base / "ab_compare.md"
    generate_markdown_report(comparison, report_path)
    print(f"✅ Report saved: {report_path}")
    
    print("\n" + "="*80)
    print("✅ A/B Comparison Complete")
    print("="*80)
    print(f"Comparison: {comparison_path}")
    print(f"Report: {report_path}")
    print("="*80 + "\n")
    
    return 0


if __name__ == "__main__":
    exit(main())

