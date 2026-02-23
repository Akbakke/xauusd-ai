#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Depth Ladder Smoke Compare: Baseline vs L+1

Compares baseline and L+1 smoke eval results to verify wiring and trade-universe match.

Usage:
    python gx1/scripts/compare_depth_ladder_smoke.py \
        --baseline-snapshot <BASELINE_SMOKE_SNAPSHOT.json> \
        --lplus1-root <LPLUS1_OUTPUT_DIR>/2025 \
        --out-root <OUTPUT_DIR>
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add workspace root to path
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_baseline_snapshot(snapshot_path: Path) -> Dict[str, Any]:
    """Load baseline smoke snapshot.
    
    Supports two modes:
    1. If snapshot_path points to BASELINE_SMOKE_SNAPSHOT.json, loads it and tries to find RUN_SUMMARY.json in parent dir
    2. If snapshot_path points to a directory, loads RUN_SUMMARY.json from that directory
    """
    snapshot = {}
    
    if snapshot_path.is_dir():
        # Mode 2: Directory path - load RUN_SUMMARY.json directly
        run_summary_path = snapshot_path / "RUN_SUMMARY.json"
        if not run_summary_path.exists():
            raise FileNotFoundError(
                f"[DEPTH_LADDER_COMPARE] FATAL: Baseline RUN_SUMMARY.json not found in {snapshot_path}"
            )
        with open(run_summary_path, "r") as f:
            snapshot["run_summary"] = json.load(f)
        
        # Also try to load smoke_stats.json if available
        smoke_stats_path = snapshot_path / "smoke_stats.json"
        if smoke_stats_path.exists():
            with open(smoke_stats_path, "r") as f:
                smoke_stats = json.load(f)
            fingerprint = smoke_stats.get("data_universe_fingerprint", {})
            completion = smoke_stats.get("run_completion_fingerprint", {})
            if fingerprint:
                snapshot["data_universe_fingerprint"] = fingerprint
                # Extract metadata from fingerprint
                snapshot["bundle_sha256"] = fingerprint.get("bundle_sha256")
                snapshot["transformer_layers"] = fingerprint.get("transformer_layers")
                snapshot["transformer_layers_baseline"] = fingerprint.get("transformer_layers_baseline")
                snapshot["depth_ladder_delta"] = fingerprint.get("depth_ladder_delta")
                snapshot["policy_id"] = fingerprint.get("policy_id")
                snapshot["replay_mode"] = fingerprint.get("replay_mode")
                snapshot["temperature_scaling_effective_enabled"] = fingerprint.get("temperature_scaling_effective_enabled")
            if completion:
                snapshot["run_completion_fingerprint"] = completion
        
        # Load metrics (check both root and year subdirectory and chunk_0)
        year_dir = snapshot_path / "2025"
        if year_dir.exists():
            metrics_files = list(year_dir.glob("metrics_TRIAL160_2025_*_MERGED.json"))
            if not metrics_files:
                chunk_0_dir = year_dir / "chunk_0"
                if chunk_0_dir.exists():
                    metrics_files = list(chunk_0_dir.glob("metrics_TRIAL160_2025_*.json"))
            if metrics_files:
                with open(metrics_files[0], "r") as f:
                    metrics = json.load(f)
                snapshot["trades"] = metrics.get("n_trades")
                snapshot["pnl_bps"] = metrics.get("total_pnl_bps")
                snapshot["mean_pnl_bps"] = metrics.get("mean_pnl_bps")
                snapshot["median_pnl_bps"] = metrics.get("median_pnl_bps")
                snapshot["maxdd_bps"] = metrics.get("max_dd", metrics.get("max_drawdown_bps"))
                snapshot["p1_loss_bps"] = metrics.get("p1_loss")
                snapshot["p5_loss_bps"] = metrics.get("p5_loss")
            
            # Load trades for winners/losers
            trade_files = list(year_dir.glob("trade_outcomes_*_MERGED.parquet"))
            if trade_files:
                import pandas as pd
                df = pd.read_parquet(trade_files[0])
                snapshot["winners"] = (df["pnl_bps"] > 0).sum()
                snapshot["losers"] = (df["pnl_bps"] < 0).sum()
        
        # Load RUN_IDENTITY if available
        identity_files = list(snapshot_path.glob("**/RUN_IDENTITY.json"))
        if identity_files:
            with open(identity_files[0], "r") as f:
                identity = json.load(f)
            if not snapshot.get("bundle_sha256"):
                snapshot["bundle_sha256"] = identity.get("bundle_sha256")
            if not snapshot.get("transformer_layers"):
                snapshot["transformer_layers"] = identity.get("transformer_layers")
            if not snapshot.get("transformer_layers_baseline"):
                snapshot["transformer_layers_baseline"] = identity.get("transformer_layers_baseline")
            if not snapshot.get("depth_ladder_delta"):
                snapshot["depth_ladder_delta"] = identity.get("depth_ladder_delta")
            if not snapshot.get("policy_id"):
                snapshot["policy_id"] = identity.get("policy_id")
            if not snapshot.get("replay_mode"):
                snapshot["replay_mode"] = identity.get("replay_mode")
            if not snapshot.get("temperature_scaling_enabled"):
                snapshot["temperature_scaling_enabled"] = identity.get("temperature_scaling_enabled")
            if not snapshot.get("feature_build_disabled"):
                snapshot["feature_build_disabled"] = identity.get("feature_build_disabled")
    else:
        # Mode 1: File path - load snapshot and find RUN_SUMMARY.json
        with open(snapshot_path, "r") as f:
            snapshot = json.load(f)
        
        # Try to load fingerprint and run_summary from smoke_stats.json if available
        snapshot_dir = snapshot_path.parent
        smoke_stats_path = snapshot_dir / "smoke_stats.json"
        if smoke_stats_path.exists():
            with open(smoke_stats_path, "r") as f:
                smoke_stats = json.load(f)
            fingerprint = smoke_stats.get("data_universe_fingerprint", {})
            completion = smoke_stats.get("run_completion_fingerprint", {})
            run_summary = smoke_stats.get("run_summary", {})
            if fingerprint:
                snapshot["data_universe_fingerprint"] = fingerprint
            if completion:
                snapshot["run_completion_fingerprint"] = completion
            if run_summary:
                snapshot["run_summary"] = run_summary
        
        # Also try to load RUN_SUMMARY.json directly (master-level truth source)
        # Check both snapshot_dir and snapshot_dir.parent (in case snapshot is in a subdirectory)
        run_summary_path = snapshot_dir / "RUN_SUMMARY.json"
        if not run_summary_path.exists():
            run_summary_path = snapshot_dir.parent / "RUN_SUMMARY.json"
        if run_summary_path.exists():
            with open(run_summary_path, "r") as f:
                run_summary = json.load(f)
            snapshot["run_summary"] = run_summary
        elif "run_summary" not in snapshot:
            # If snapshot file contains run_summary_path, use it
            run_summary_path_from_snapshot = snapshot.get("run_summary_path")
            if run_summary_path_from_snapshot:
                run_summary_path = Path(run_summary_path_from_snapshot)
                if run_summary_path.exists():
                    with open(run_summary_path, "r") as f:
                        snapshot["run_summary"] = json.load(f)
            # If still not found, try to construct from snapshot data (fallback for old snapshots)
            # This is a fallback - prefer RUN_SUMMARY.json or run_summary in smoke_stats.json
            if "run_summary" not in snapshot and snapshot.get("year") and snapshot.get("trades") is not None:
                log.warning("[DEPTH_LADDER_COMPARE] Baseline RUN_SUMMARY.json not found, constructing minimal run_summary from snapshot")
                # Construct minimal run_summary from snapshot (for backward compatibility)
                snapshot["run_summary"] = {
                    "arm": "baseline",
                    "year": snapshot.get("year"),
                    "bars_total_in_subset": None,  # Not available in old snapshot
                    "bars_iterated": None,
                    "bars_emitted_signals": snapshot.get("trades"),  # Approximate
                    "completed": True,  # Assume completed if snapshot exists
                    "timed_out": False,
                    "transformer_layers": snapshot.get("transformer_layers"),
                    "transformer_layers_baseline": snapshot.get("transformer_layers_baseline"),
                    "depth_ladder_delta": snapshot.get("depth_ladder_delta", 0),
                }
    
    return snapshot


def load_lplus1_results(lplus1_root: Path) -> Dict[str, Any]:
    """Load L+1 smoke eval results."""
    results = {
        "trades": None,
        "pnl_bps": None,
        "mean_pnl_bps": None,
        "median_pnl_bps": None,
        "maxdd_bps": None,
        "p1_loss_bps": None,
        "p5_loss_bps": None,
        "winners": None,
        "losers": None,
        "bundle_sha256": None,
        "transformer_layers": None,
        "transformer_layers_baseline": None,
        "depth_ladder_delta": None,
        "policy_id": None,
        "replay_mode": None,
        "temperature_scaling_enabled": None,
        "temperature_scaling_effective_enabled": None,
        "feature_build_disabled": None,
        "data_universe_fingerprint": None,
        "run_completion_fingerprint": None,
        "run_summary": None,
    }
    
    # Try to load smoke_stats.json first (contains fingerprint and run_summary)
    smoke_stats_path = lplus1_root.parent / "smoke_stats.json"
    if smoke_stats_path.exists():
        with open(smoke_stats_path, "r") as f:
            smoke_stats = json.load(f)
        fingerprint = smoke_stats.get("data_universe_fingerprint", {})
        completion = smoke_stats.get("run_completion_fingerprint", {})
        run_summary = smoke_stats.get("run_summary", {})
        if fingerprint:
            results["data_universe_fingerprint"] = fingerprint
            results["bundle_sha256"] = fingerprint.get("bundle_sha256")
            results["transformer_layers"] = fingerprint.get("transformer_layers")
            results["transformer_layers_baseline"] = fingerprint.get("transformer_layers_baseline")
            results["depth_ladder_delta"] = fingerprint.get("depth_ladder_delta")
            results["policy_id"] = fingerprint.get("policy_id")
            results["replay_mode"] = fingerprint.get("replay_mode")
            results["temperature_scaling_effective_enabled"] = fingerprint.get("temperature_scaling_effective_enabled")
        if completion:
            results["run_completion_fingerprint"] = completion
        if run_summary:
            results["run_summary"] = run_summary
    
    # Also try to load RUN_SUMMARY.json directly (master-level truth source)
    run_summary_path = lplus1_root.parent / "RUN_SUMMARY.json"
    if run_summary_path.exists():
        with open(run_summary_path, "r") as f:
            run_summary = json.load(f)
        results["run_summary"] = run_summary
    
    # Load metrics (check both root and chunk_0)
    metrics_files = list(lplus1_root.glob("metrics_TRIAL160_2025_*_MERGED.json"))
    if not metrics_files:
        # Try chunk_0 subdirectory
        chunk_0_dir = lplus1_root / "chunk_0"
        if chunk_0_dir.exists():
            metrics_files = list(chunk_0_dir.glob("metrics_TRIAL160_2025_*.json"))
    if metrics_files:
        with open(metrics_files[0], "r") as f:
            metrics = json.load(f)
        results["trades"] = metrics.get("n_trades")
        results["pnl_bps"] = metrics.get("total_pnl_bps")
        results["mean_pnl_bps"] = metrics.get("mean_pnl_bps")
        results["median_pnl_bps"] = metrics.get("median_pnl_bps")
        results["maxdd_bps"] = metrics.get("max_dd", metrics.get("max_drawdown_bps"))
        results["p1_loss_bps"] = metrics.get("p1_loss")
        results["p5_loss_bps"] = metrics.get("p5_loss")
    
    # Load trades for winners/losers
    trade_files = list(lplus1_root.glob("trade_outcomes_*_MERGED.parquet"))
    if trade_files:
        import pandas as pd
        df = pd.read_parquet(trade_files[0])
        results["winners"] = (df["pnl_bps"] > 0).sum()
        results["losers"] = (df["pnl_bps"] < 0).sum()
    
    # Load RUN_IDENTITY
    identity_files = list(lplus1_root.glob("**/RUN_IDENTITY.json"))
    if identity_files:
        with open(identity_files[0], "r") as f:
            identity = json.load(f)
        results["bundle_sha256"] = identity.get("bundle_sha256")
        results["transformer_layers"] = identity.get("transformer_layers")
        results["transformer_layers_baseline"] = identity.get("transformer_layers_baseline")
        results["depth_ladder_delta"] = identity.get("depth_ladder_delta")
        results["policy_id"] = identity.get("policy_id")
        results["replay_mode"] = identity.get("replay_mode")
        results["temperature_scaling_enabled"] = identity.get("temperature_scaling_enabled")
        results["feature_build_disabled"] = identity.get("feature_build_disabled")
    
    return results


def compare_results(baseline: Dict[str, Any], lplus1: Dict[str, Any]) -> Dict[str, Any]:
    """Compare baseline and L+1 results."""
    comparison = {
        "baseline": baseline,
        "lplus1": lplus1,
        "deltas": {},
        "verification": {},
        "go_nogo": "UNKNOWN",
    }
    
    # FAIL-FAST CHECKS (before any other processing)
    # These catch common mistakes like wrong paths or wrong runs
    
    # 1. Baseline RUN_SUMMARY must exist and be parseable
    baseline_summary = baseline.get("run_summary", {})
    if not baseline_summary:
        raise RuntimeError(
            "[DEPTH_LADDER_COMPARE] FATAL: Baseline RUN_SUMMARY.json not found or not parseable. "
            "Check --baseline-snapshot path or ensure baseline smoke eval completed."
        )
    
    # 2. L+1 RUN_SUMMARY must exist and be parseable
    lplus1_summary = lplus1.get("run_summary", {})
    if not lplus1_summary:
        raise RuntimeError(
            "[DEPTH_LADDER_COMPARE] FATAL: L+1 RUN_SUMMARY.json not found or not parseable. "
            "Check --lplus1-root path or ensure L+1 smoke eval completed."
        )
    
    # 3. Arm identity check: baseline must have depth_ladder_delta==0, L+1 must have depth_ladder_delta==1
    # Also check transformer_layers are different (baseline=3, L+1=4)
    baseline_delta = baseline_summary.get("depth_ladder_delta")
    lplus1_delta = lplus1_summary.get("depth_ladder_delta")
    
    # Try to get from fingerprint if not in summary
    if baseline_delta is None:
        baseline_fp = baseline.get("data_universe_fingerprint", {})
        baseline_delta = baseline_fp.get("depth_ladder_delta")
    if lplus1_delta is None:
        lplus1_fp = lplus1.get("data_universe_fingerprint", {})
        lplus1_delta = lplus1_fp.get("depth_ladder_delta")
    
    baseline_layers = baseline_summary.get("transformer_layers")
    lplus1_layers = lplus1_summary.get("transformer_layers")
    
    # Try to get from fingerprint if not in summary
    if baseline_layers is None:
        baseline_fp = baseline.get("data_universe_fingerprint", {})
        baseline_layers = baseline_fp.get("transformer_layers")
    if lplus1_layers is None:
        lplus1_fp = lplus1.get("data_universe_fingerprint", {})
        lplus1_layers = lplus1_fp.get("transformer_layers")
    
    # Arm identity validation
    if baseline_delta != 0:
        raise RuntimeError(
            f"[DEPTH_LADDER_COMPARE] FATAL: Baseline arm identity check failed. "
            f"Expected depth_ladder_delta=0, got {baseline_delta}. "
            f"Check --baseline-snapshot path (might be pointing to L+1 run)."
        )
    
    if lplus1_delta != 1:
        raise RuntimeError(
            f"[DEPTH_LADDER_COMPARE] FATAL: L+1 arm identity check failed. "
            f"Expected depth_ladder_delta=1, got {lplus1_delta}. "
            f"Check --lplus1-root path (might be pointing to baseline run)."
        )
    
    if baseline_layers == lplus1_layers:
        raise RuntimeError(
            f"[DEPTH_LADDER_COMPARE] FATAL: Arm identity check failed. "
            f"Baseline and L+1 have same transformer_layers ({baseline_layers}). "
            f"Check paths - might be comparing same arm twice."
        )
    
    if baseline_layers != 3:
        log.warning(f"[DEPTH_LADDER_COMPARE] Baseline transformer_layers={baseline_layers} (expected 3)")
    
    if lplus1_layers != 4:
        log.warning(f"[DEPTH_LADDER_COMPARE] L+1 transformer_layers={lplus1_layers} (expected 4)")
    
    log.info(f"[DEPTH_LADDER_COMPARE] ✅ Arm identity verified: baseline (delta=0, layers={baseline_layers}) vs L+1 (delta=1, layers={lplus1_layers})")
    
    # Calculate deltas
    if baseline.get("trades") is not None and lplus1.get("trades") is not None:
        comparison["deltas"]["trades"] = lplus1["trades"] - baseline["trades"]
        comparison["deltas"]["trades_pct"] = (comparison["deltas"]["trades"] / baseline["trades"]) * 100 if baseline["trades"] > 0 else 0.0
    
    if baseline.get("pnl_bps") is not None and lplus1.get("pnl_bps") is not None:
        comparison["deltas"]["pnl_bps"] = lplus1["pnl_bps"] - baseline["pnl_bps"]
        comparison["deltas"]["pnl_bps_pct"] = (comparison["deltas"]["pnl_bps"] / abs(baseline["pnl_bps"])) * 100 if baseline["pnl_bps"] != 0 else 0.0
    
    if baseline.get("maxdd_bps") is not None and lplus1.get("maxdd_bps") is not None:
        comparison["deltas"]["maxdd_bps"] = lplus1["maxdd_bps"] - baseline["maxdd_bps"]
    
    if baseline.get("p1_loss_bps") is not None and lplus1.get("p1_loss_bps") is not None:
        comparison["deltas"]["p1_loss_bps"] = lplus1["p1_loss_bps"] - baseline["p1_loss_bps"]
    
    if baseline.get("p5_loss_bps") is not None and lplus1.get("p5_loss_bps") is not None:
        comparison["deltas"]["p5_loss_bps"] = lplus1["p5_loss_bps"] - baseline["p5_loss_bps"]
    
    # Verification checks
    verification = comparison["verification"]
    
    # Universe match check (from fingerprints)
    baseline_fp = baseline.get("data_universe_fingerprint", {})
    lplus1_fp = lplus1.get("data_universe_fingerprint", {})
    
    # FATAL if baseline missing fingerprint (likely old full-year snapshot)
    if not baseline_fp:
        raise RuntimeError(
            "[DEPTH_LADDER_COMPARE] FATAL: Baseline RUN_SUMMARY mangler data_universe_fingerprint. "
            "Du bruker sannsynligvis et gammelt fullår-snapshot. "
            "Kjør baseline med samme smoke-date-range og RUN_SUMMARY."
        )
    
    if not lplus1_fp:
        raise RuntimeError(
            "[DEPTH_LADDER_COMPARE] FATAL: L+1 mangler data_universe_fingerprint. "
            "Kjør L+1 smoke eval på nytt for å generere fingerprint."
        )
    
    if baseline_fp and lplus1_fp:
        # Check critical universe keys
        universe_keys = [
            "candles_first_ts",
            "candles_last_ts",
            "candles_rowcount_loaded",
            "prebuilt_first_ts",
            "prebuilt_last_ts",
            "prebuilt_rowcount",
            "policy_sha256",
            "replay_mode",
            "temperature_scaling_effective_enabled",
        ]
        
        universe_mismatches = []
        for key in universe_keys:
            baseline_val = baseline_fp.get(key)
            lplus1_val = lplus1_fp.get(key)
            if baseline_val != lplus1_val:
                universe_mismatches.append({
                    "key": key,
                    "baseline": baseline_val,
                    "lplus1": lplus1_val,
                })
        
        verification["universe_match"] = len(universe_mismatches) == 0
        verification["universe_mismatches"] = universe_mismatches
    else:
        verification["universe_match"] = None
        verification["universe_mismatches"] = []
        log.warning("Could not compare universe fingerprints (missing in baseline or L+1)")
    
    # Bundle SHA256 must be different
    baseline_sha = baseline.get("bundle_sha256", "")
    lplus1_sha = lplus1.get("bundle_sha256", "")
    verification["bundle_sha256_different"] = (
        baseline_sha != "" and lplus1_sha != "" and baseline_sha != lplus1_sha
    )
    
    # Transformer layers
    verification["transformer_layers_baseline"] = (
        baseline.get("transformer_layers") == 3
    )
    verification["transformer_layers_lplus1"] = (
        lplus1.get("transformer_layers") == 4
    )
    verification["depth_ladder_delta"] = (
        lplus1.get("depth_ladder_delta") == 1
    )
    
    # Policy must be same (check policy_id from fingerprint or identity)
    baseline_policy_id = baseline.get("policy_id") or ""
    lplus1_policy_id = lplus1.get("policy_id") or ""
    baseline_policy_path = baseline.get("policy_path") or ""
    
    # Try to get policy_id from fingerprint if not directly available
    if not baseline_policy_id and baseline_fp:
        baseline_policy_id = baseline_fp.get("policy_id") or ""
    if not lplus1_policy_id and lplus1_fp:
        lplus1_policy_id = lplus1_fp.get("policy_id") or ""
    
    # Match if policy_id matches, or if policy_path ends with policy_id
    verification["policy_match"] = (
        (baseline_policy_id and lplus1_policy_id and baseline_policy_id == lplus1_policy_id) or
        (baseline_policy_path and lplus1_policy_id and baseline_policy_path.endswith(lplus1_policy_id)) or
        (baseline_policy_id and baseline_policy_path and baseline_policy_path.endswith(baseline_policy_id))
    )
    
    # Trade universe match (allow tolerance for smoke eval)
    # For Q1 smoke, we expect some variation due to model differences
    # Allow up to 5% difference for smoke eval (not full year)
    trades_match = False
    if baseline.get("trades") is not None and lplus1.get("trades") is not None:
        trades_diff = abs(comparison["deltas"]["trades"])
        trades_diff_pct = abs(comparison["deltas"]["trades_pct"])
        # Allow up to 5% difference for smoke eval (Q1 subset, not full year)
        trades_match = trades_diff_pct < 5.0
        verification["trades_match"] = trades_match
        verification["trades_diff"] = trades_diff
        verification["trades_diff_pct"] = trades_diff_pct
    
    # PREBUILT mode (check from verification dict, fingerprint, or fallback)
    baseline_verification = baseline.get("verification", {})
    lplus1_verification = lplus1.get("verification", {})
    
    # Check verification dict first (from smoke_stats.json)
    baseline_prebuilt_verified = baseline_verification.get("prebuilt_verified", False)
    lplus1_prebuilt_verified = lplus1_verification.get("prebuilt_verified", False)
    
    # Fallback to fingerprint/replay_mode if verification not available
    if not baseline_prebuilt_verified:
        baseline_prebuilt_verified = (
            baseline.get("replay_mode") == "PREBUILT" and
            baseline.get("feature_build_disabled") == True
        )
    if not lplus1_prebuilt_verified:
        lplus1_prebuilt_verified = (
            lplus1.get("replay_mode") == "PREBUILT" and
            lplus1.get("feature_build_disabled") == True
        )
    
    verification["prebuilt_mode"] = baseline_prebuilt_verified and lplus1_prebuilt_verified
    
    # Temperature scaling (check from verification dict, fingerprint, or fallback)
    baseline_temp_verified = baseline_verification.get("temperature_scaling_verified", False)
    lplus1_temp_verified = lplus1_verification.get("temperature_scaling_verified", False)
    
    # Fallback to fingerprint if verification not available
    if not baseline_temp_verified:
        baseline_temp_verified = (
            baseline.get("temperature_scaling_enabled") == True or
            baseline.get("temperature_scaling_effective_enabled") == True
        )
    if not lplus1_temp_verified:
        lplus1_temp_verified = (
            lplus1.get("temperature_scaling_enabled") == True or
            lplus1.get("temperature_scaling_effective_enabled") == True
        )
    
    verification["temperature_scaling_enabled"] = baseline_temp_verified and lplus1_temp_verified
    
    # GO/NO-GO decision with completion gates
    # Use RUN_SUMMARY (master-level truth source) if available, otherwise fallback to run_completion_fingerprint
    baseline_summary = baseline.get("run_summary", {})
    lplus1_summary = lplus1.get("run_summary", {})
    baseline_completion = baseline.get("run_completion_fingerprint", {})
    lplus1_completion = lplus1.get("run_completion_fingerprint", {})
    
    # Prefer run_summary (master-level truth), fallback to run_completion_fingerprint
    baseline_completed = baseline_summary.get("completed") if baseline_summary else baseline_completion.get("completed", False)
    lplus1_completed = lplus1_summary.get("completed") if lplus1_summary else lplus1_completion.get("completed", False)
    baseline_timed_out = baseline_summary.get("timed_out") if baseline_summary else baseline_completion.get("timed_out", False)
    lplus1_timed_out = lplus1_summary.get("timed_out") if lplus1_summary else lplus1_completion.get("timed_out", False)
    
    verification["baseline_completed"] = baseline_completed
    verification["lplus1_completed"] = lplus1_completed
    verification["baseline_timed_out"] = baseline_timed_out
    verification["lplus1_timed_out"] = lplus1_timed_out
    
    # Log completion details from run_summary
    if baseline_summary:
        verification["baseline_bars_processed"] = baseline_summary.get("bars_processed_total")
        verification["baseline_expected_candles"] = baseline_summary.get("expected_candles_in_subset")
    if lplus1_summary:
        verification["lplus1_bars_processed"] = lplus1_summary.get("bars_processed_total")
        verification["lplus1_expected_candles"] = lplus1_summary.get("expected_candles_in_subset")
    
    # Krav D: feature_build_call_count==0 + temp scaling true (already checked)
    
    # Hard fail if not completed or timed out
    if not baseline_completed or baseline_timed_out:
        raise RuntimeError(
            f"[DEPTH_LADDER_COMPARE] FATAL: Baseline smoke eval not completed "
            f"(completed={baseline_completed}, timed_out={baseline_timed_out}). "
            f"Cannot compare partial results."
        )
    
    if not lplus1_completed or lplus1_timed_out:
        raise RuntimeError(
            f"[DEPTH_LADDER_COMPARE] FATAL: L+1 smoke eval not completed "
            f"(completed={lplus1_completed}, timed_out={lplus1_timed_out}). "
            f"Cannot compare partial results."
        )
    
    # Sanity gate: check completion percentage (optional, but useful)
    baseline_pct = baseline_completion.get("pct_of_candles_processed")
    lplus1_pct = lplus1_completion.get("pct_of_candles_processed")
    if baseline_pct is not None and lplus1_pct is not None:
        verification["baseline_pct_processed"] = baseline_pct
        verification["lplus1_pct_processed"] = lplus1_pct
        # Warn if pct is not 100% (but don't fail - might be intentional subset)
        if baseline_pct < 100.0 or lplus1_pct < 100.0:
            log.warning(f"[DEPTH_LADDER_COMPARE] Completion percentage < 100%: baseline={baseline_pct:.1f}%, lplus1={lplus1_pct:.1f}%")
    
    go_conditions = [
        bool(verification.get("bundle_sha256_different", False)),
        bool(verification.get("transformer_layers_baseline", False)),
        bool(verification.get("transformer_layers_lplus1", False)),
        bool(verification.get("depth_ladder_delta", False)),
        bool(verification.get("policy_match", False)),
        bool(verification.get("universe_match", False)) if verification.get("universe_match") is not None else False,
        bool(verification.get("prebuilt_mode", False)),
        bool(verification.get("temperature_scaling_enabled", False)),
        bool(verification.get("baseline_completed", False)),
        bool(verification.get("lplus1_completed", False)),
        not bool(verification.get("baseline_timed_out", False)),  # Must be False (not timed out)
        not bool(verification.get("lplus1_timed_out", False)),  # Must be False (not timed out)
    ]
    
    comparison["go_nogo"] = "GO" if all(go_conditions) else "NO-GO"
    comparison["go_conditions_met"] = sum(go_conditions)
    comparison["go_conditions_total"] = len(go_conditions)
    
    return comparison


def generate_markdown_report(comparison: Dict[str, Any], output_path: Path, quarter: str = "Q1") -> None:
    """Generate markdown comparison report."""
    baseline = comparison["baseline"]
    lplus1 = comparison["lplus1"]
    deltas = comparison["deltas"]
    verification = comparison["verification"]
    
    # Helper functions for safe formatting
    def fmt_num(val, default="N/A"):
        if val is None:
            return default
        if isinstance(val, (int, float)):
            return f"{val:,}"
        return str(val)
    
    def fmt_float(val, fmt=".2f", default="N/A"):
        if val is None:
            return default
        if isinstance(val, (int, float)):
            return f"{val:{fmt}}"
        return str(val)
    
    def fmt_num_delta(val, default="N/A"):
        if val is None:
            return default
        if isinstance(val, (int, float)):
            return f"{val:+,}"
        return str(val)
    
    baseline_sha = baseline.get("bundle_sha256") or "N/A"
    lplus1_sha = lplus1.get("bundle_sha256") or "N/A"
    baseline_sha_short = baseline_sha[:16] + "..." if isinstance(baseline_sha, str) and len(baseline_sha) > 16 else baseline_sha
    lplus1_sha_short = lplus1_sha[:16] + "..." if isinstance(lplus1_sha, str) and len(lplus1_sha) > 16 else lplus1_sha
    
    # Get completion status for report
    baseline_summary = baseline.get("run_summary", {})
    lplus1_summary = lplus1.get("run_summary", {})
    baseline_completion = baseline.get("run_completion_fingerprint", {})
    lplus1_completion = lplus1.get("run_completion_fingerprint", {})
    
    baseline_completed = baseline_summary.get("completed") if baseline_summary else baseline_completion.get("completed", False)
    lplus1_completed = lplus1_summary.get("completed") if lplus1_summary else lplus1_completion.get("completed", False)
    baseline_timed_out = baseline_summary.get("timed_out") if baseline_summary else baseline_completion.get("timed_out", False)
    lplus1_timed_out = lplus1_summary.get("timed_out") if lplus1_summary else lplus1_completion.get("timed_out", False)
    
    # Get RUN_ID and date
    run_id = baseline.get("run_id") or lplus1.get("run_id") or "UNKNOWN"
    from datetime import datetime
    report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Get paths
    baseline_path_str = baseline.get("bundle_dir") or baseline.get("data_universe_fingerprint", {}).get("bundle_dir_resolved", "N/A")
    lplus1_path_str = lplus1.get("bundle_dir") or lplus1.get("data_universe_fingerprint", {}).get("bundle_dir_resolved", "N/A")
    
    # Determine quarter from output path or use default
    quarter_detected = quarter
    if "Q2" in str(output_path) or "q2" in str(output_path).lower():
        quarter_detected = "Q2"
    elif "Q3" in str(output_path) or "q3" in str(output_path).lower():
        quarter_detected = "Q3"
    
    report = f"""# Depth Ladder {quarter_detected} 2025 Compare: Baseline vs L+1

**Date:** {report_date}  
**RUN_ID:** {run_id}  
**Status:** {comparison["go_nogo"]}  
**Conditions Met:** {comparison["go_conditions_met"]}/{comparison["go_conditions_total"]}

**Paths:**
- Baseline: {baseline_path_str}
- L+1: {lplus1_path_str}

---

## Universe Match

**Status:** {"✅ PASS" if verification.get("universe_match") else "❌ FAIL" if verification.get("universe_match") is not None else "⚠️ N/A"}

{("**Universe fingerprints match**" if verification.get("universe_match") else "**⚠️ UNIVERSE MISMATCHES DETECTED:**") if verification.get("universe_match") is not None else "**⚠️ Universe fingerprints not available**"}

---

| Metric | Baseline | L+1 | Status |
|--------|----------|-----|--------|
| Bundle SHA256 | `{baseline_sha_short}` | `{lplus1_sha_short}` | {"✅" if verification.get("bundle_sha256_different") else "❌"} |
| Transformer Layers | {baseline.get("transformer_layers", "N/A")} | {lplus1.get("transformer_layers", "N/A")} | {"✅" if verification.get("transformer_layers_lplus1") else "❌"} |
| Depth Ladder Delta | {baseline.get("depth_ladder_delta", "N/A")} | {lplus1.get("depth_ladder_delta", "N/A")} | {"✅" if verification.get("depth_ladder_delta") else "❌"} |
| Policy Match | {baseline.get("policy_path", "N/A")} | {lplus1.get("policy_id", "N/A")} | {"✅" if verification.get("policy_match") else "❌"} |

---

## Completion

**Status:** {"✅ PASS" if (baseline_completed and lplus1_completed and not baseline_timed_out and not lplus1_timed_out) else "❌ FAIL"}

| Check | Baseline | L+1 |
|-------|----------|-----|
| Bars Total in Subset | {baseline_summary.get("bars_total_in_subset", "N/A") if baseline_summary else "N/A"} | {lplus1_summary.get("bars_total_in_subset", "N/A") if lplus1_summary else "N/A"} |
| Bars Iterated | {baseline_summary.get("bars_iterated", "N/A") if baseline_summary else "N/A"} | {lplus1_summary.get("bars_iterated", "N/A") if lplus1_summary else "N/A"} |
| Timed Out | {baseline_timed_out} | {lplus1_timed_out} |
| Candles First TS Iterated | {baseline_summary.get("candles_first_ts_iterated", "N/A") if baseline_summary else "N/A"} | {lplus1_summary.get("candles_first_ts_iterated", "N/A") if lplus1_summary else "N/A"} |
| Candles Last TS Iterated | {baseline_summary.get("candles_last_ts_iterated", "N/A") if baseline_summary else "N/A"} | {lplus1_summary.get("candles_last_ts_iterated", "N/A") if lplus1_summary else "N/A"} |

---

## Arm Identity

| Check | Baseline | L+1 | Status |
|-------|----------|-----|--------|
| Transformer Layers | {baseline.get("transformer_layers", "N/A")} | {lplus1.get("transformer_layers", "N/A")} | {"✅" if verification.get("transformer_layers_baseline") and verification.get("transformer_layers_lplus1") else "❌"} |
| Depth Ladder Delta | {baseline.get("depth_ladder_delta", "N/A")} | {lplus1.get("depth_ladder_delta", "N/A")} | {"✅" if verification.get("depth_ladder_delta") else "❌"} |
| Bundle SHA256 | `{baseline_sha_short}` | `{lplus1_sha_short}` | {"✅" if verification.get("bundle_sha256_different") else "❌"} |
| Policy ID/SHA | {baseline.get("policy_id", baseline.get("data_universe_fingerprint", {}).get("policy_id", "N/A"))} | {lplus1.get("policy_id", "N/A")} | {"✅" if verification.get("policy_match") else "❌"} |

---

## Trading Metrics

| Metric | Baseline | L+1 | Delta |
|--------|----------|-----|-------|
| Trades | {fmt_num(baseline.get("trades"))} | {fmt_num(lplus1.get("trades"))} | {fmt_num_delta(deltas.get("trades"))} ({fmt_float(deltas.get("trades_pct", 0), "+.2f")}%) |
| PnL (bps) | {fmt_float(baseline.get("pnl_bps"))} | {fmt_float(lplus1.get("pnl_bps"))} | {fmt_float(deltas.get("pnl_bps"), "+.2f", "N/A")} ({fmt_float(deltas.get("pnl_bps_pct", 0), "+.2f")}% hvis tilgjengelig) |
| MaxDD (bps) | {fmt_float(baseline.get("maxdd_bps"))} | {fmt_float(lplus1.get("maxdd_bps"))} | {fmt_float(deltas.get("maxdd_bps"), "+.2f", "N/A")} |
| P1 Loss (bps) | {fmt_float(baseline.get("p1_loss_bps"))} | {fmt_float(lplus1.get("p1_loss_bps"))} | {fmt_float(deltas.get("p1_loss_bps"), "+.2f", "N/A")} |
| P5 Loss (bps) | {fmt_float(baseline.get("p5_loss_bps"))} | {fmt_float(lplus1.get("p5_loss_bps"))} | {fmt_float(deltas.get("p5_loss_bps"), "+.2f", "N/A")} |

---

## Invariant Verification

| Check | Status |
|-------|--------|
| Bundle SHA256 Different | {"✅" if verification.get("bundle_sha256_different") else "❌"} |
| Transformer Layers (Baseline=3) | {"✅" if verification.get("transformer_layers_baseline") else "❌"} |
| Transformer Layers (L+1=4) | {"✅" if verification.get("transformer_layers_lplus1") else "❌"} |
| Depth Ladder Delta (=1) | {"✅" if verification.get("depth_ladder_delta") else "❌"} |
| **Universe Match** | {"✅" if verification.get("universe_match") else "❌" if verification.get("universe_match") is not None else "⚠️ N/A"} |
| Trade Universe Match | {"✅" if verification.get("trades_match") else "❌"} |
| PREBUILT Mode | {"✅" if verification.get("prebuilt_mode") else "❌"} |
| Temperature Scaling Enabled | {"✅" if verification.get("temperature_scaling_enabled") else "❌"} |

---

## Universe Fingerprint Comparison

"""
    
    # Add universe mismatch details if any
    universe_mismatches = verification.get("universe_mismatches", [])
    if universe_mismatches:
        report += "\n**⚠️ UNIVERSE MISMATCHES DETECTED:**\n\n"
        report += "| Key | Baseline | L+1 |\n"
        report += "|-----|----------|-----|\n"
        for mismatch in universe_mismatches:
            baseline_val = str(mismatch.get("baseline", "N/A"))
            lplus1_val = str(mismatch.get("lplus1", "N/A"))
            report += f"| {mismatch.get('key', 'N/A')} | {baseline_val} | {lplus1_val} |\n"
        report += "\n"
    else:
        baseline_fp = baseline.get("data_universe_fingerprint", {})
        lplus1_fp = lplus1.get("data_universe_fingerprint", {})
        if baseline_fp and lplus1_fp:
            report += "\n**✅ Universe fingerprints match**\n\n"
            report += "| Key | Value |\n"
            report += "|-----|-------|\n"
            report += f"| Candles Rowcount | {baseline_fp.get('candles_rowcount_loaded', 'N/A')} |\n"
            report += f"| Candles First TS | {baseline_fp.get('candles_first_ts', 'N/A')} |\n"
            report += f"| Candles Last TS | {baseline_fp.get('candles_last_ts', 'N/A')} |\n"
            report += f"| Prebuilt Rowcount | {baseline_fp.get('prebuilt_rowcount', 'N/A')} |\n"
            report += f"| Prebuilt First TS | {baseline_fp.get('prebuilt_first_ts', 'N/A')} |\n"
            report += f"| Prebuilt Last TS | {baseline_fp.get('prebuilt_last_ts', 'N/A')} |\n"
            report += f"| Policy SHA256 | {baseline_fp.get('policy_sha256', 'N/A')[:16]}... |\n"
            report += "\n"
        else:
            report += "\n**⚠️ Universe fingerprints not available**\n\n"
    
    decision_text = "✅ VALID COMPARE" if comparison["go_nogo"] == "GO" else "❌ INVALID COMPARE"
    report += f"""
---

## Konklusjon

**Decision:** {decision_text}

"""
    
    if comparison["go_nogo"] == "GO":
        trades_delta = deltas.get("trades", 0)
        trades_delta_pct = deltas.get("trades_pct", 0)
        pnl_delta = deltas.get("pnl_bps", 0)
        pnl_delta_pct = deltas.get("pnl_bps_pct", 0) if deltas.get("pnl_bps_pct") is not None else None
        
        if abs(trades_delta) > 0:
            report += f"""
✅ Alle gates passerte. L+1 bundle er klar for full multiyear eval.

**Trade-path endret:** L+1 produserte {abs(trades_delta)} {'færre' if trades_delta < 0 else 'flere'} trades ({abs(trades_delta_pct):.2f}%) i {quarter_detected} 2025,
noe som indikerer at modellendringen (3→4 layers) påvirker trade-path som forventet.
"""
        else:
            report += f"""
✅ Alle gates passerte. L+1 bundle er klar for full multiyear eval.

**Trade-path match:** L+1 produserte samme antall trades som baseline i {quarter_detected} 2025.
"""
        
        # Add decision rule (use quarter_detected which is set earlier in the function)
        report += f"""
---

## Beslutningsregel for Full Multiyear Eval

**{quarter_detected} Resultat:**
- PnL Delta: {pnl_delta:+.2f} bps ({pnl_delta_pct:+.2f}% hvis tilgjengelig) {"✅" if pnl_delta >= 0 else "❌"}
- Trades Delta: {trades_delta:+d} ({trades_delta_pct:+.2f}%)

**Regel:**
- Hvis Q1, Q2 og Q3 alle viser signifikant negativ PnL for L+1 → NO-GO for full multiyear
- Hvis minst én av Q2/Q3 er positiv eller flat → GO for full 2025 / multiyear eval

**{quarter_detected} Anbefaling:** {"✅ GO" if pnl_delta >= 0 else "⚠️ NEGATIV (vent på Q2/Q3)"}
"""
    else:
        failed_checks = []
        for k, v in verification.items():
            if isinstance(v, bool) and not v:
                failed_checks.append(k)
            elif isinstance(v, dict) and v.get("universe_match") is False:
                failed_checks.append(f"universe_match ({len(v.get('universe_mismatches', []))} mismatches)")
        
        report += f"""
❌ Verifikasjonssjekker feilet. L+1 bundle er IKKE klar for full eval.

**Eksakt grunn:**
"""
        for check in failed_checks:
            report += f"- {check}\n"
    
    with open(output_path, "w") as f:
        f.write(report)
    
    log.info(f"✅ Written markdown report: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare Depth Ladder Smoke Results")
    parser.add_argument("--baseline-snapshot", type=Path, required=False,
                        help="Path to baseline smoke snapshot JSON (or directory with RUN_SUMMARY.json)")
    parser.add_argument("--baseline-root", type=Path, required=False,
                        help="Path to baseline smoke eval output directory (alternative to --baseline-snapshot)")
    parser.add_argument("--lplus1-root", type=Path, required=True,
                        help="Path to L+1 smoke eval output directory (year subdirectory)")
    parser.add_argument("--out-root", type=Path, required=True,
                        help="Output directory for comparison results")
    
    args = parser.parse_args()
    
    # Determine baseline path
    if args.baseline_root:
        baseline_path = args.baseline_root
        log.info(f"Loading baseline from directory: {baseline_path}")
    elif args.baseline_snapshot:
        baseline_path = args.baseline_snapshot
        log.info(f"Loading baseline from snapshot: {baseline_path}")
    else:
        parser.error("Either --baseline-snapshot or --baseline-root must be provided")
    
    # Load data
    log.info("Loading baseline snapshot...")
    baseline = load_baseline_snapshot(baseline_path)
    
    log.info("Loading L+1 results...")
    lplus1 = load_lplus1_results(args.lplus1_root)
    
    # Compare
    log.info("Comparing results...")
    comparison = compare_results(baseline, lplus1)
    
    # Write outputs
    args.out_root.mkdir(parents=True, exist_ok=True)
    
    # JSON output (convert numpy/pandas types to native Python types)
    def convert_to_json_serializable(obj):
        """Convert numpy/pandas types to native Python types for JSON serialization."""
        import numpy as np
        import pandas as pd
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (pd.Timestamp, pd.DatetimeTZDtype)):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_json_serializable(item) for item in obj]
        else:
            return obj
    
    json_path = args.out_root / "SMOKE_COMPARE.json"
    comparison_serializable = convert_to_json_serializable(comparison)
    with open(json_path, "w") as f:
        json.dump(comparison_serializable, f, indent=2)
    log.info(f"✅ Written JSON: {json_path}")
    
    # Determine quarter from data_universe_fingerprint (most reliable source)
    quarter = "Q1"  # Default
    baseline_fp = baseline.get("data_universe_fingerprint", {})
    lplus1_fp = lplus1.get("data_universe_fingerprint", {})
    
    # Try to get smoke_date_range from fingerprint
    smoke_date_range = baseline_fp.get("smoke_date_range") or lplus1_fp.get("smoke_date_range")
    
    if smoke_date_range:
        if "2025-04-01" in smoke_date_range or "2025-05" in smoke_date_range or "2025-06" in smoke_date_range:
            quarter = "Q2"
        elif "2025-07-01" in smoke_date_range or "2025-08" in smoke_date_range or "2025-09" in smoke_date_range:
            quarter = "Q3"
        elif "2025-10-01" in smoke_date_range or "2025-11" in smoke_date_range or "2025-12" in smoke_date_range:
            quarter = "Q4"
    else:
        # Fallback to path-based detection
        if "Q2" in str(args.baseline_root) or "Q2" in str(args.lplus1_root) or "2025-04-01" in str(args.baseline_root) or "2025-04-01" in str(args.lplus1_root):
            quarter = "Q2"
        elif "Q3" in str(args.baseline_root) or "Q3" in str(args.lplus1_root) or "2025-07-01" in str(args.baseline_root) or "2025-07-01" in str(args.lplus1_root):
            quarter = "Q3"
        elif "Q4" in str(args.baseline_root) or "Q4" in str(args.lplus1_root) or "2025-10-01" in str(args.baseline_root) or "2025-10-01" in str(args.lplus1_root):
            quarter = "Q4"
    
    # Markdown report
    md_path = args.out_root / f"DEPTH_LADDER_{quarter}_COMPARE.md"
    generate_markdown_report(comparison, md_path, quarter)
    
    # Also write JSON with requested name
    json_path_q1 = args.out_root / f"DEPTH_LADDER_{quarter}_COMPARE.json"
    with open(json_path_q1, "w") as f:
        json.dump(comparison_serializable, f, indent=2)
    log.info(f"✅ Written Q1 compare JSON: {json_path_q1}")
    
    # Print summary
    log.info("\n" + "="*80)
    log.info(f"GO/NO-GO: {comparison['go_nogo']}")
    log.info(f"Conditions Met: {comparison['go_conditions_met']}/{comparison['go_conditions_total']}")
    log.info("="*80)
    
    if comparison["go_nogo"] == "NO-GO":
        log.error("❌ L+1 smoke eval failed verification checks")
        return 1
    
    log.info("✅ L+1 smoke eval passed all verification checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
