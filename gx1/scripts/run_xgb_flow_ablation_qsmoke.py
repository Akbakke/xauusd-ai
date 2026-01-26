#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGB Flow Ablation A/B Test Runner (Q1/Q2 Smoke)

Runs controlled A/B tests to measure the value of XGB→Transformer pipeline:
- Test 1: XGB channels → Transformer (baseline vs no_xgb_channels_in_transformer)

NOTE: Test 2 (XGB post-transformer calibration/veto) has been REMOVED as of 2026-01-24.
      XGB now only provides pre-predict channels to Transformer. No post-processing.

Usage:
    python gx1/scripts/run_xgb_flow_ablation_qsmoke.py \
        --arm test1_channels \
        --years 2025 \
        --data <CANDLES_PARQUET> \
        --prebuilt-parquet <PREBUILT_PARQUET> \
        --bundle-dir <BUNDLE_DIR> \
        --policy <POLICY_YAML> \
        --out-root <OUTPUT_ROOT> \
        --smoke-date-range "2025-01-01..2025-03-31"
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add workspace root to path
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        import numpy as np
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def load_run_summary(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Load RUN_SUMMARY.json from run directory."""
    # Try root first
    run_summary_path = run_dir / "RUN_SUMMARY.json"
    if not run_summary_path.exists():
        # Try year subdirectory
        year_dirs = [d for d in run_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        if year_dirs:
            run_summary_path = year_dirs[0] / "RUN_SUMMARY.json"
    if not run_summary_path.exists():
        log.warning(f"RUN_SUMMARY.json not found in {run_dir}")
        return None
    with open(run_summary_path, "r") as f:
        return json.load(f)


def load_entry_features_used(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Load ENTRY_FEATURES_USED.json from run directory."""
    # Priority 1: Try master file in run root
    master_path = run_dir / "ENTRY_FEATURES_USED_MASTER.json"
    if master_path.exists():
        log.info(f"Loading ENTRY_FEATURES_USED_MASTER.json from {master_path}")
        with open(master_path, "r") as f:
            return json.load(f)
    
    # Priority 2: Try manifest and load first chunk
    manifest_path = run_dir / "ENTRY_FEATURES_TELEMETRY_MANIFEST.json"
    if manifest_path.exists():
        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            chunks = manifest.get("chunks", [])
            # Find first chunk with telemetry
            for chunk_info in chunks:
                entry_features_rel = chunk_info.get("entry_features_used")
                if entry_features_rel and chunk_info.get("entry_features_used_loaded", False):
                    entry_features_path = run_dir / entry_features_rel
                    if entry_features_path.exists():
                        log.info(f"Loading ENTRY_FEATURES_USED.json from manifest chunk: {entry_features_path}")
                        with open(entry_features_path, "r") as f:
                            return json.load(f)
        except Exception as e:
            log.warning(f"Failed to load from manifest: {e}")
    
    # Priority 3: Fallback to globbing (backward compatibility)
    entry_features_paths = list(run_dir.glob("**/ENTRY_FEATURES_USED.json"))
    if not entry_features_paths:
        # Also try chunk_0 subdirectory
        chunk_0_dir = run_dir / "chunk_0"
        if chunk_0_dir.exists():
            entry_features_paths = list(chunk_0_dir.glob("ENTRY_FEATURES_USED.json"))
    if not entry_features_paths:
        raise FileNotFoundError(
            f"[XGB_ABLATION] FATAL: ENTRY_FEATURES_USED.json not found in {run_dir}. "
            f"Expected: ENTRY_FEATURES_USED_MASTER.json (run root) or ENTRY_FEATURES_USED.json (chunk directories). "
            f"Check that telemetry aggregation completed successfully."
        )
    # Use first found (should be unique)
    log.warning(f"Using fallback: loading ENTRY_FEATURES_USED.json from {entry_features_paths[0]}")
    with open(entry_features_paths[0], "r") as f:
        return json.load(f)


def load_metrics(run_dir: Path, year: int = None) -> Dict[str, Any]:
    """Load trading metrics from run directory (SSoT: chunk_0/metrics_*.json).
    
    Args:
        run_dir: Path to run output directory
        year: Optional year subdirectory (not typically used in smoke tests)
    
    Returns:
        Dict with trading metrics
        
    Raises:
        RuntimeError: If metrics file not found (fail-fast)
    """
    search_paths_tried = []
    
    # Priority 1: chunk_0/metrics_*.json (direct chunk output)
    chunk_0_dir = run_dir / "chunk_0"
    if chunk_0_dir.exists():
        metrics_files = list(chunk_0_dir.glob("metrics_*.json"))
        search_paths_tried.append(str(chunk_0_dir / "metrics_*.json"))
        if metrics_files:
            with open(metrics_files[0], "r") as f:
                log.info(f"Loaded metrics from {metrics_files[0]}")
                return json.load(f)
    
    # Priority 2: Run root metrics (aggregated)
    root_metrics_files = list(run_dir.glob("metrics_*.json"))
    search_paths_tried.append(str(run_dir / "metrics_*.json"))
    if root_metrics_files:
        with open(root_metrics_files[0], "r") as f:
            log.info(f"Loaded metrics from {root_metrics_files[0]}")
            return json.load(f)
    
    # Priority 3: Year subdirectory (legacy)
    if year:
        year_dir = run_dir / str(year)
        if year_dir.exists():
            year_metrics_files = list(year_dir.glob("metrics_*.json"))
            search_paths_tried.append(str(year_dir / "metrics_*.json"))
            if year_metrics_files:
                with open(year_metrics_files[0], "r") as f:
                    log.info(f"Loaded metrics from {year_metrics_files[0]}")
                    return json.load(f)
    
    # FATAL: Metrics are required for A/B tests
    raise RuntimeError(
        f"[METRICS_LOAD_FAIL] FATAL: Trading metrics not found in {run_dir}.\n"
        f"Searched paths:\n  - " + "\n  - ".join(search_paths_tried) + "\n"
        f"Expected: chunk_0/metrics_*.json (primary) or metrics_*.json (root).\n"
        f"Check that replay completed successfully and metrics were written."
    )


def run_replay_arm(
    arm_name: str,
    arm_config: Dict[str, str],
    data_path: Path,
    prebuilt_path: Path,
    bundle_dir: Path,
    policy_path: Path,
    output_dir: Path,
    smoke_date_range: Optional[str] = None,
    smoke_bars: Optional[int] = None,
    workers: int = 1,
) -> Dict[str, Any]:
    """Run replay for a single arm with specified environment variables."""
    log.info(f"[{arm_name}] Starting replay...")
    log.info(f"[{arm_name}] Environment variables:")
    for key, value in arm_config.items():
        log.info(f"  {key}={value}")
    
    # Create output directory for this arm
    arm_output_dir = output_dir / arm_name
    arm_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build command - ensure paths are absolute
    policy_path_abs = Path(policy_path).resolve()
    data_path_abs = Path(data_path).resolve()
    prebuilt_path_abs = Path(prebuilt_path).resolve()
    bundle_dir_abs = Path(bundle_dir).resolve()
    
    cmd = [
        sys.executable,
        str(workspace_root / "gx1" / "scripts" / "replay_eval_gated_parallel.py"),
        "--policy", str(policy_path_abs),
        "--data", str(data_path_abs),
        "--prebuilt-parquet", str(prebuilt_path_abs),
        "--bundle-dir", str(bundle_dir_abs),  # CLI override (highest priority)
        "--output-dir", str(arm_output_dir.resolve()),
        "--workers", str(workers),
    ]
    
    # Add slice-head if smoke_bars is specified (overrides date range)
    if smoke_bars is not None:
        cmd.extend(["--slice-head", str(smoke_bars)])
        log.info(f"[{arm_name}] Using --slice-head {smoke_bars} (overrides date range)")
    elif smoke_date_range:
        # Parse smoke_date_range (format: "2025-01-01..2025-03-31")
        start_date, end_date = smoke_date_range.split("..")
        cmd.extend(["--start-ts", start_date, "--end-ts", end_date])
        log.info(f"[{arm_name}] Using date range: {start_date} to {end_date}")
    else:
        raise ValueError("Either --smoke-bars or --smoke-date-range must be specified")
    
    # Set environment variables
    env = os.environ.copy()
    for key, value in arm_config.items():
        env[key] = value
    
    # Also set bundle directory in environment (as fallback, CLI takes precedence)
    env["GX1_BUNDLE_DIR"] = str(bundle_dir_abs)
    
    # Set PYTHONPATH to include workspace root
    pythonpath = env.get("PYTHONPATH", "")
    if pythonpath:
        env["PYTHONPATH"] = f"{workspace_root}:{pythonpath}"
    else:
        env["PYTHONPATH"] = str(workspace_root)
    
    log.info(f"[{arm_name}] Running command: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            env=env,
            cwd=workspace_root,
            capture_output=True,
            text=True,
            check=False,  # Don't raise on non-zero exit
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode != 0:
            log.error(f"[{arm_name}] Replay failed with exit code {result.returncode}")
            log.error(f"[{arm_name}] stderr: {result.stderr[:1000]}")
            return {
                "arm": arm_name,
                "success": False,
                "exit_code": result.returncode,
                "stderr": result.stderr[:1000],
                "elapsed_sec": elapsed,
            }
        
        log.info(f"[{arm_name}] Replay completed in {elapsed:.1f}s")
        
        # Load results
        run_summary = load_run_summary(arm_output_dir)
        entry_features = load_entry_features_used(arm_output_dir)
        
        # Extract year from smoke_date_range or default to 2025
        if smoke_date_range:
            start_date, end_date = smoke_date_range.split("..")
            year = int(start_date.split("-")[0])
        else:
            # Default to 2025 if using smoke-bars
            year = 2025
        metrics = load_metrics(arm_output_dir, year)
        
        # Track which range was actually used
        used_smoke_date_range = smoke_date_range
        used_smoke_bars = smoke_bars
        
        return {
            "arm": arm_name,
            "success": True,
            "output_dir": str(arm_output_dir),
            "run_summary": run_summary,
            "entry_features_used": entry_features,
            "metrics": metrics,
            "elapsed_sec": elapsed,
            "used_smoke_date_range": used_smoke_date_range,
            "used_smoke_bars": used_smoke_bars,
        }
    
    except Exception as e:
        elapsed = time.time() - start_time
        log.error(f"[{arm_name}] Replay failed with exception: {e}")
        import traceback
        return {
            "arm": arm_name,
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "elapsed_sec": elapsed,
        }


def check_transformer_called(entry_features: Dict[str, Any]) -> bool:
    """Check if transformer was actually called in this run."""
    if not entry_features:
        return False
    
    # Check transformer_forward_calls from master telemetry
    transformer_forward_calls = entry_features.get("transformer_forward_calls", 0)
    if transformer_forward_calls > 0:
        return True
    
    # Also check model_entry summary
    model_entry = entry_features.get("model_entry", {})
    model_forward_calls = sum(model_entry.get("model_forward_calls", {}).values())
    if model_forward_calls > 0:
        return True
    
    return False


def run_replay_arm_with_autorange(
    arm_name: str,
    arm_config: Dict[str, str],
    data_path: Path,
    prebuilt_path: Path,
    bundle_dir: Path,
    policy_path: Path,
    output_dir: Path,
    smoke_date_range: Optional[str] = None,
    smoke_bars: Optional[int] = None,
    workers: int = 1,
) -> Dict[str, Any]:
    """
    Run replay arm with automatic range expansion if transformer is not called.
    
    Tries progressively larger ranges until transformer_forward_calls > 0.
    """
    from datetime import datetime, timedelta
    
    # Initial range
    current_smoke_date_range = smoke_date_range
    current_smoke_bars = smoke_bars
    attempt = 0
    max_attempts = 4  # Initial + 3 fallbacks
    
    while attempt < max_attempts:
        log.info(f"[{arm_name}] Attempt {attempt + 1}/{max_attempts}")
        if current_smoke_date_range:
            log.info(f"[{arm_name}] Using date range: {current_smoke_date_range}")
        elif current_smoke_bars:
            log.info(f"[{arm_name}] Using smoke-bars: {current_smoke_bars}")
        
        # Use unique output directory for each attempt (to avoid PREBUILT_FAIL on reuse)
        attempt_output_dir = output_dir
        if attempt > 0:
            attempt_output_dir = output_dir.parent / f"{output_dir.name}_attempt{attempt + 1}"
        
        result = run_replay_arm(
            arm_name,
            arm_config,
            data_path,
            prebuilt_path,
            bundle_dir,
            policy_path,
            attempt_output_dir,
            smoke_date_range=current_smoke_date_range,
            smoke_bars=current_smoke_bars,
            workers=workers,
        )
        
        if not result.get("success"):
            log.error(f"[{arm_name}] Run failed, aborting autorange")
            return result
        
        # Check if transformer was called
        entry_features = result.get("entry_features_used", {})
        if check_transformer_called(entry_features):
            log.info(f"[{arm_name}] ✅ Transformer called (transformer_forward_calls > 0)")
            result["autorange_attempt"] = attempt + 1
            result["autorange_final_range"] = current_smoke_date_range or f"smoke-bars-{current_smoke_bars}"
            # Move final attempt output to main output_dir if it was a subdirectory
            if attempt > 0 and attempt_output_dir != output_dir:
                import shutil
                if output_dir.exists():
                    shutil.rmtree(output_dir)
                shutil.move(attempt_output_dir, output_dir)
                result["output_dir"] = str(output_dir)
            return result
        
        log.warning(
            f"[{arm_name}] ⚠️  Transformer not called (transformer_forward_calls=0). "
            f"Trying larger range..."
        )
        
        # Try next fallback range
        attempt += 1
        if attempt < max_attempts:
            if current_smoke_date_range:
                # Expand date range: +2 days, +5 days, then fallback to smoke-bars
                start_date, end_date = current_smoke_date_range.split("..")
                start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                
                if attempt == 1:
                    # +2 days
                    end_dt = end_dt + timedelta(days=2)
                    current_smoke_date_range = f"{start_date}..{end_dt.strftime('%Y-%m-%d')}"
                elif attempt == 2:
                    # +5 more days (total +7)
                    end_dt = end_dt + timedelta(days=5)
                    current_smoke_date_range = f"{start_date}..{end_dt.strftime('%Y-%m-%d')}"
                else:
                    # Fallback to smoke-bars
                    current_smoke_date_range = None
                    current_smoke_bars = 50000
            elif current_smoke_bars:
                # Increase bars: 2x, 5x, then 10x
                if attempt == 1:
                    current_smoke_bars = current_smoke_bars * 2
                elif attempt == 2:
                    current_smoke_bars = current_smoke_bars * 5
                else:
                    current_smoke_bars = 50000
            else:
                # Should not happen, but fallback
                current_smoke_bars = 50000
    
    # All attempts failed
    raise RuntimeError(
        f"[{arm_name}] FATAL: NO_TRANSFORMER_CALLS_IN_TEST_UNIVERSE\n"
        f"Tried {max_attempts} different ranges but transformer_forward_calls remained 0.\n"
        f"Last attempt: date_range={current_smoke_date_range}, smoke_bars={current_smoke_bars}\n"
        f"This indicates a fundamental issue with the test setup or data range."
    )


def check_policy_sanity(policy_path: Path) -> None:
    """Check that policy has entry_models.v10_ctx configured."""
    import yaml
    
    if not policy_path.exists():
        raise FileNotFoundError(f"[POLICY_SANITY] Policy file not found: {policy_path}")
    
    with open(policy_path, "r") as f:
        policy = yaml.safe_load(f)
    
    entry_models = policy.get("entry_models", {})
    v10_ctx_cfg = entry_models.get("v10_ctx", {})
    
    if not v10_ctx_cfg:
        # Check if it's in entry_config instead
        entry_config_str = policy.get("entry_config", "")
        if entry_config_str:
            entry_config_path = Path(entry_config_str)
            if not entry_config_path.is_absolute():
                entry_config_path = policy_path.parent / entry_config_path
            
            if entry_config_path.exists():
                with open(entry_config_path, "r") as f:
                    entry_config = yaml.safe_load(f)
                entry_models = entry_config.get("entry_models", {})
                v10_ctx_cfg = entry_models.get("v10_ctx", {})
    
    if not v10_ctx_cfg:
        raise RuntimeError(
            f"[POLICY_SANITY] FATAL: Policy {policy_path} does not have entry_models.v10_ctx configured. "
            f"This is required for XGB flow ablation tests. "
            f"Use a policy with v10_ctx enabled (e.g., GX1_SNIPER_REPLAY_V10_CTX_VERIFY.yaml)."
        )
    
    if not v10_ctx_cfg.get("enabled", False):
        raise RuntimeError(
            f"[POLICY_SANITY] FATAL: Policy {policy_path} has entry_models.v10_ctx but it is not enabled. "
            f"Set entry_models.v10_ctx.enabled: true in policy."
        )
    
    log.info(f"[POLICY_SANITY] ✅ Policy has entry_models.v10_ctx configured and enabled")


def extract_telemetry_sanity(entry_features: Dict[str, Any]) -> Dict[str, Any]:
    """Extract telemetry sanity checks from ENTRY_FEATURES_USED.json."""
    if not entry_features:
        return {}
    
    xgb_flow = entry_features.get("xgb_flow", {})
    toggles = entry_features.get("toggles", {})
    
    # Extract XGB channel names (seq + snap)
    xgb_seq_channels = entry_features.get("xgb_seq_channels", {})
    xgb_snap_channels = entry_features.get("xgb_snap_channels", {})
    xgb_seq_names = xgb_seq_channels.get("names", [])
    xgb_snap_names = xgb_snap_channels.get("names", [])
    xgb_channel_names = xgb_seq_names + xgb_snap_names
    
    return {
        "n_xgb_channels_in_transformer_input": xgb_flow.get("n_xgb_channels_in_transformer_input", 0),
        "xgb_channel_names": xgb_channel_names,
        "xgb_seq_channel_names": xgb_seq_names,
        "xgb_snap_channel_names": xgb_snap_names,
        "xgb_used_as": xgb_flow.get("xgb_used_as", "none"),
        "xgb_pre_predict_count": xgb_flow.get("xgb_pre_predict_count", 0),
        "xgb_post_predict_count": xgb_flow.get("xgb_post_predict_count", 0),
        "post_predict_called": xgb_flow.get("post_predict_called", False),
        "veto_applied_count": xgb_flow.get("veto_applied_count", 0),
        "disable_xgb_channels_in_transformer_effective": toggles.get("disable_xgb_channels_in_transformer_effective", False),
        "disable_xgb_post_transformer_effective": toggles.get("disable_xgb_post_transformer_effective", False),
    }


def extract_trading_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Extract trading metrics from metrics JSON."""
    if not metrics:
        return {}
    
    return {
        "n_trades": metrics.get("n_trades", 0),
        "total_pnl_bps": metrics.get("total_pnl_bps", 0.0),
        "mean_pnl_bps": metrics.get("mean_pnl_bps", 0.0),
        "median_pnl_bps": metrics.get("median_pnl_bps", 0.0),
        "max_dd": metrics.get("max_dd", metrics.get("max_drawdown_bps", 0.0)),
        "winrate": metrics.get("winrate", None),  # May not be available
    }


def compare_arms(baseline: Dict[str, Any], ablated: Dict[str, Any], test_name: str) -> Dict[str, Any]:
    """Compare baseline and ablated arms."""
    comparison = {
        "test_name": test_name,
        "baseline": baseline,
        "ablated": ablated,
        "deltas": {},
        "telemetry_sanity": {},
        "invariants": {},
    }
    
    # Extract metrics
    baseline_metrics = extract_trading_metrics(baseline.get("metrics", {}))
    ablated_metrics = extract_trading_metrics(ablated.get("metrics", {}))
    
    # Extract telemetry
    baseline_telemetry = extract_telemetry_sanity(baseline.get("entry_features_used", {}))
    ablated_telemetry = extract_telemetry_sanity(ablated.get("entry_features_used", {}))
    
    # Compute deltas
    for key in ["n_trades", "total_pnl_bps", "mean_pnl_bps", "median_pnl_bps", "max_dd"]:
        baseline_val = baseline_metrics.get(key)
        ablated_val = ablated_metrics.get(key)
        if baseline_val is not None and ablated_val is not None:
            comparison["deltas"][key] = ablated_val - baseline_val
            if baseline_val != 0:
                comparison["deltas"][f"{key}_pct"] = ((ablated_val - baseline_val) / abs(baseline_val)) * 100.0
    
    # Telemetry sanity comparison
    comparison["telemetry_sanity"] = {
        "baseline": baseline_telemetry,
        "ablated": ablated_telemetry,
    }
    
    # Invariants
    invariants = comparison["invariants"]
    
    # Test 1 invariants
    if test_name == "test1_channels":
        n_xgb_channels_baseline = baseline_telemetry.get("n_xgb_channels_in_transformer_input", -1)
        n_xgb_channels_ablated = ablated_telemetry.get("n_xgb_channels_in_transformer_input", -1)
        xgb_channel_names_baseline = baseline_telemetry.get("xgb_channel_names", [])
        xgb_channel_names_ablated = ablated_telemetry.get("xgb_channel_names", [])
        
        invariants["test1_n_xgb_channels_baseline"] = n_xgb_channels_baseline
        invariants["test1_n_xgb_channels_ablated"] = n_xgb_channels_ablated
        invariants["test1_n_xgb_channels_ablated_is_zero"] = (n_xgb_channels_ablated == 0)
        invariants["test1_baseline_has_xgb_channels"] = (n_xgb_channels_baseline > 0)
        invariants["test1_xgb_channel_names_baseline"] = xgb_channel_names_baseline
        invariants["test1_xgb_channel_names_ablated"] = xgb_channel_names_ablated
        invariants["test1_xgb_channel_names_ablated_is_empty"] = (len(xgb_channel_names_ablated) == 0)
        
        # FATAL if ablated arm has XGB channels when it shouldn't
        if n_xgb_channels_ablated != 0:
            raise RuntimeError(
                f"[XGB_ABLATION] FATAL: Test 1 ablated arm has n_xgb_channels_in_transformer_input={n_xgb_channels_ablated}, "
                f"expected 0. Check GX1_DISABLE_XGB_CHANNELS_IN_TRANSFORMER toggle."
            )
        
        # FATAL if ablated arm has XGB channel names when it shouldn't
        if len(xgb_channel_names_ablated) > 0:
            raise RuntimeError(
                f"[XGB_ABLATION] FATAL: Test 1 ablated arm has xgb_channel_names={xgb_channel_names_ablated}, "
                f"expected empty. Check GX1_DISABLE_XGB_CHANNELS_IN_TRANSFORMER toggle."
            )
        
        # Check if baseline has XGB channels (if not, test is no-op)
        if n_xgb_channels_baseline == 0:
            log.warning(
                "[XGB_ABLATION] Test 1 baseline has n_xgb_channels_in_transformer_input=0. "
                "This test may be a no-op (XGB channels not used in baseline)."
            )
            invariants["test1_is_noop"] = True
        else:
            invariants["test1_is_noop"] = False
    
    # Test 2 invariants
    if test_name == "test2_post":
        post_predict_baseline = baseline_telemetry.get("post_predict_called", False)
        post_predict_ablated = ablated_telemetry.get("post_predict_called", False)
        veto_count_ablated = ablated_telemetry.get("veto_applied_count", -1)
        
        invariants["test2_post_predict_baseline"] = post_predict_baseline
        invariants["test2_post_predict_ablated"] = post_predict_ablated
        invariants["test2_post_predict_ablated_is_false"] = (post_predict_ablated == False)
        invariants["test2_veto_count_ablated"] = veto_count_ablated
        invariants["test2_veto_count_ablated_is_zero"] = (veto_count_ablated == 0)
        
        # FATAL if ablated arm has post_predict_called when it shouldn't
        if post_predict_ablated:
            raise RuntimeError(
                f"[XGB_ABLATION] FATAL: Test 2 ablated arm has post_predict_called=True, "
                f"expected False. Check GX1_DISABLE_XGB_POST_TRANSFORMER toggle."
            )
        
        # FATAL if ablated arm has veto_applied_count > 0
        if veto_count_ablated > 0:
            raise RuntimeError(
                f"[XGB_ABLATION] FATAL: Test 2 ablated arm has veto_applied_count={veto_count_ablated}, "
                f"expected 0. Check GX1_DISABLE_XGB_POST_TRANSFORMER toggle."
            )
    
    return comparison


def generate_markdown_report(comparison: Dict[str, Any], output_path: Path, test_name: str) -> None:
    """Generate markdown comparison report."""
    baseline = comparison["baseline"]
    ablated = comparison["ablated"]
    deltas = comparison["deltas"]
    telemetry_sanity = comparison["telemetry_sanity"]
    invariants = comparison["invariants"]
    
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
    
    def fmt_delta(val, default="N/A"):
        if val is None:
            return default
        if isinstance(val, (int, float)):
            return f"{val:+,}"
        return str(val)
    
    report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""# XGB Flow Ablation: {test_name.upper()}

**Date:** {report_date}  
**Test:** {test_name}

---

## Trading Metrics

| Metric | Baseline | Ablated | Delta |
|--------|----------|---------|-------|
| Trades | {fmt_num((baseline.get('metrics') or {}).get('n_trades'))} | {fmt_num((ablated.get('metrics') or {}).get('n_trades'))} | {fmt_delta((deltas or {}).get('n_trades'))} ({fmt_float((deltas or {}).get('n_trades_pct', 0), '+.2f')}%) |
| PnL (bps) | {fmt_float((baseline.get('metrics') or {}).get('total_pnl_bps'))} | {fmt_float((ablated.get('metrics') or {}).get('total_pnl_bps'))} | {fmt_delta((deltas or {}).get('total_pnl_bps'))} ({fmt_float((deltas or {}).get('total_pnl_bps_pct', 0), '+.2f')}%) |
| Mean PnL (bps) | {fmt_float((baseline.get('metrics') or {}).get('mean_pnl_bps'))} | {fmt_float((ablated.get('metrics') or {}).get('mean_pnl_bps'))} | {fmt_delta((deltas or {}).get('mean_pnl_bps'))} |
| Median PnL (bps) | {fmt_float((baseline.get('metrics') or {}).get('median_pnl_bps'))} | {fmt_float((ablated.get('metrics') or {}).get('median_pnl_bps'))} | {fmt_delta((deltas or {}).get('median_pnl_bps'))} |
| MaxDD (bps) | {fmt_float((baseline.get('metrics') or {}).get('max_dd'))} | {fmt_float((ablated.get('metrics') or {}).get('max_dd'))} | {fmt_delta((deltas or {}).get('max_dd'))} |
| Winrate | {fmt_float((baseline.get('metrics') or {}).get('winrate'), '.1%')} | {fmt_float((ablated.get('metrics') or {}).get('winrate'), '.1%')} | {fmt_delta(((ablated.get('metrics') or {}).get('winrate', 0) or 0) - ((baseline.get('metrics') or {}).get('winrate', 0) or 0) if (baseline.get('metrics') or {}).get('winrate') is not None and (ablated.get('metrics') or {}).get('winrate') is not None else None, '+.1%')} |

---

## Telemetry Sanity

### Test 1 Specific Checks

| Check | Baseline | Ablated | Status |
|-------|----------|---------|--------|
| n_xgb_channels_in_transformer_input | {telemetry_sanity['baseline'].get('n_xgb_channels_in_transformer_input', 'N/A')} | {telemetry_sanity['ablated'].get('n_xgb_channels_in_transformer_input', 'N/A')} | {"✅" if (test_name == "test1_channels" and telemetry_sanity['ablated'].get('n_xgb_channels_in_transformer_input', -1) == 0) else ("⚠️" if test_name == "test1_channels" else "N/A")} |
| xgb_channel_names (count) | {len(telemetry_sanity['baseline'].get('xgb_channel_names', []))} | {len(telemetry_sanity['ablated'].get('xgb_channel_names', []))} | {"✅" if (test_name == "test1_channels" and len(telemetry_sanity['ablated'].get('xgb_channel_names', [])) == 0) else ("⚠️" if test_name == "test1_channels" else "N/A")} |
| xgb_channel_names (list) | {', '.join(telemetry_sanity['baseline'].get('xgb_channel_names', [])[:5])}{'...' if len(telemetry_sanity['baseline'].get('xgb_channel_names', [])) > 5 else ''} | {', '.join(telemetry_sanity['ablated'].get('xgb_channel_names', [])[:5]) if telemetry_sanity['ablated'].get('xgb_channel_names', []) else '[]'} | {"✅" if (test_name == "test1_channels" and len(telemetry_sanity['ablated'].get('xgb_channel_names', [])) == 0) else ("⚠️" if test_name == "test1_channels" else "N/A")} |
| xgb_pre_predict_count | {telemetry_sanity['baseline'].get('xgb_pre_predict_count', 'N/A')} | {telemetry_sanity['ablated'].get('xgb_pre_predict_count', 'N/A')} | - |

### Test 2 Specific Checks

| Check | Baseline | Ablated | Status |
|-------|----------|---------|--------|
| post_predict_called | {telemetry_sanity['baseline'].get('post_predict_called', 'N/A')} | {telemetry_sanity['ablated'].get('post_predict_called', 'N/A')} | {"✅" if (test_name == "test2_post" and telemetry_sanity['ablated'].get('post_predict_called', True) == False) else ("⚠️" if test_name == "test2_post" else "N/A")} |
| veto_applied_count | {telemetry_sanity['baseline'].get('veto_applied_count', 'N/A')} | {telemetry_sanity['ablated'].get('veto_applied_count', 'N/A')} | {"✅" if (test_name == "test2_post" and telemetry_sanity['ablated'].get('veto_applied_count', -1) == 0) else ("⚠️" if test_name == "test2_post" else "N/A")} |

### General Checks

| Check | Baseline | Ablated |
|-------|----------|---------|
| xgb_used_as | {telemetry_sanity['baseline'].get('xgb_used_as', 'N/A')} | {telemetry_sanity['ablated'].get('xgb_used_as', 'N/A')} |
| xgb_pre_predict_count | {telemetry_sanity['baseline'].get('xgb_pre_predict_count', 'N/A')} | {telemetry_sanity['ablated'].get('xgb_pre_predict_count', 'N/A')} |
| xgb_post_predict_count | {telemetry_sanity['baseline'].get('xgb_post_predict_count', 'N/A')} | {telemetry_sanity['ablated'].get('xgb_post_predict_count', 'N/A')} |

---

## Invariants

"""
    
    # Add test-specific invariants
    if test_name == "test1_channels":
        xgb_channel_names_baseline = invariants.get('test1_xgb_channel_names_baseline', [])
        xgb_channel_names_ablated = invariants.get('test1_xgb_channel_names_ablated', [])
        report += f"""
| Check | Value | Status |
|-------|-------|--------|
| Baseline has XGB channels | {invariants.get('test1_baseline_has_xgb_channels', 'N/A')} | {"✅" if invariants.get('test1_baseline_has_xgb_channels', False) else "⚠️ NO-OP"} |
| Baseline XGB channel names | {', '.join(xgb_channel_names_baseline[:5])}{'...' if len(xgb_channel_names_baseline) > 5 else ''} ({len(xgb_channel_names_baseline)} total) | - |
| Ablated n_xgb_channels == 0 | {invariants.get('test1_n_xgb_channels_ablated', 'N/A')} | {"✅" if invariants.get('test1_n_xgb_channels_ablated_is_zero', False) else "❌"} |
| Ablated xgb_channel_names is empty | {len(xgb_channel_names_ablated)} | {"✅" if invariants.get('test1_xgb_channel_names_ablated_is_empty', False) else "❌"} |
| Test is no-op | {invariants.get('test1_is_noop', 'N/A')} | {"⚠️ YES" if invariants.get('test1_is_noop', False) else "✅ NO"} |
"""
    elif test_name == "test2_post":
        report += f"""
| Check | Value | Status |
|-------|-------|--------|
| Ablated post_predict_called == False | {invariants.get('test2_post_predict_ablated', 'N/A')} | {"✅" if invariants.get('test2_post_predict_ablated_is_false', False) else "❌"} |
| Ablated veto_count == 0 | {invariants.get('test2_veto_count_ablated', 'N/A')} | {"✅" if invariants.get('test2_veto_count_ablated_is_zero', False) else "❌"} |
"""
    
    report += f"""
---

## Conclusion

**Test:** {test_name}  
**Status:** {"✅ PASS" if all(invariants.get(k, False) for k in invariants.keys() if k.endswith("_is_zero") or k.endswith("_is_false")) else "❌ FAIL"}

"""
    
    # Add no-op warning if applicable
    if test_name == "test1_channels" and invariants.get("test1_is_noop", False):
        report += """
⚠️ **NO-OP WARNING:** Baseline has n_xgb_channels_in_transformer_input=0. This test may not measure the intended effect.
"""
    
    with open(output_path, "w") as f:
        f.write(report)
    
    log.info(f"✅ Written markdown report: {output_path}")


def main():
    # LEGACY_GUARD: Check for legacy modes before proceeding
    try:
        from gx1.runtime.legacy_guard import assert_no_legacy_mode_enabled
        assert_no_legacy_mode_enabled()
    except ImportError:
        log.warning("[LEGACY_GUARD] legacy_guard not available - skipping check")
    except RuntimeError as e:
        log.error(f"[LEGACY_GUARD] {e}")
        raise
    
    # DEL 3: Truth-mode guard - hard-fail if compat-mode is enabled in truth/baseline runs
    compat_enabled = os.getenv("GX1_ALLOW_CLOSE_ALIAS_COMPAT", "0") == "1"
    truth_mode = os.getenv("GX1_TRUTH_RUN", "0") == "1" or os.getenv("TRUTH_BASELINE_LOCK", "0") == "1"
    
    if compat_enabled and truth_mode:
        raise RuntimeError(
            "[TRUTH_MODE_GUARD] GX1_ALLOW_CLOSE_ALIAS_COMPAT=1 is not allowed in truth/baseline runs. "
            "Compat-mode is only for emergency use, not for truth runs. "
            "If you see this error, the permanent fix (CLOSE alias from candles) should be working. "
            "Check that prebuilt features do not contain CLOSE and that transformer input assembly aliases CLOSE correctly."
        )
    
    if compat_enabled:
        log.warning("[WARN] GX1_ALLOW_CLOSE_ALIAS_COMPAT=1 is enabled. This is NOT for truth/baseline runs.")
        log.warning("[WARN] Compat-mode is an emergency workaround - permanent fix should be used instead.")
    
    parser = argparse.ArgumentParser(description="XGB Flow Ablation A/B Test Runner")
    parser.add_argument("--arm", type=str, required=True, choices=["test1_channels"],
                        help="Which test to run: test1_channels (Test 2 removed 2026-01-24)")
    parser.add_argument("--years", type=str, default="2025",
                        help="Comma-separated years (default: 2025)")
    parser.add_argument("--data", type=Path, required=True,
                        help="Path to candles parquet file")
    parser.add_argument("--prebuilt-parquet", type=Path, required=True,
                        help="Path to prebuilt features parquet file")
    parser.add_argument("--bundle-dir", type=Path, required=True,
                        help="Path to bundle directory")
    parser.add_argument("--policy", type=Path, required=True,
                        help="Path to policy YAML file")
    parser.add_argument("--out-root", type=Path, default=None,
                        help="Output root directory (default: GX1_REPORTS_ROOT or reports/replay_eval/XGB_FLOW_ABLATION)")
    parser.add_argument("--smoke-date-range", type=str, default="2025-01-01..2025-03-31",
                        help="Date range for smoke test (format: YYYY-MM-DD..YYYY-MM-DD)")
    parser.add_argument("--smoke-bars", type=int, default=None,
                        help="Use only first N bars (deterministic, overrides smoke-date-range)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of workers (default: 1 for determinism)")
    
    args = parser.parse_args()
    
    # Resolve paths to absolute
    args.policy = args.policy.resolve()
    args.data = args.data.resolve()
    args.prebuilt_parquet = args.prebuilt_parquet.resolve()
    args.bundle_dir = args.bundle_dir.resolve()
    
    # LEGACY_GUARD: Check policy file and output-dir after path resolution
    try:
        from gx1.runtime.legacy_guard import check_policy_for_legacy, assert_no_legacy_mode_enabled
        check_policy_for_legacy(args.policy)
        # Determine output_dir (may be set later)
        output_dir_resolved = None
        if args.out_root:
            output_dir_resolved = args.out_root.resolve()
        assert_no_legacy_mode_enabled(
            argv=sys.argv,
            bundle_dir_resolved=args.bundle_dir,
            output_dir_resolved=output_dir_resolved,
        )
    except ImportError:
        log.warning("[LEGACY_GUARD] legacy_guard not available - skipping check")
    except RuntimeError as e:
        log.error(f"[LEGACY_GUARD] {e}")
        raise
    
    # Policy sanity check (fail-fast if policy doesn't have v10_ctx)
    log.info("="*80)
    log.info("Policy Sanity Check")
    log.info("="*80)
    check_policy_sanity(args.policy)
    
    # Verify bundle_dir exists
    bundle_dir_abs = Path(args.bundle_dir).resolve()
    if not bundle_dir_abs.exists():
        raise FileNotFoundError(
            f"[BUNDLE_DIR] FATAL: Bundle directory does not exist: {bundle_dir_abs}. "
            f"Check --bundle-dir path."
        )
    log.info(f"[BUNDLE_DIR] ✅ Bundle directory exists: {bundle_dir_abs}")
    
    # Determine output root
    if args.out_root:
        output_root = args.out_root
    else:
        output_root = Path(os.getenv("GX1_REPORTS_ROOT", "reports/replay_eval/XGB_FLOW_ABLATION"))
    
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / f"xgb_ablation_{args.arm}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log.info(f"Output directory: {output_dir}")
    
    # Define arm configurations
    # CRITICAL: Always require telemetry for A/B tests (fail-fast if telemetry not collected)
    baseline_config = {
        "GX1_GATED_FUSION_ENABLED": "1",  # Required for replay eval
        "GX1_REPLAY_USE_PREBUILT_FEATURES": "1",  # Required for prebuilt features
        "GX1_ALLOW_PARALLEL_REPLAY": "1",  # Allow parallel execution (for A/B tests)
        "GX1_REQUIRE_ENTRY_TELEMETRY": "1",  # Required for A/B tests (fail-fast if telemetry missing)
        "GX1_PANIC_MODE": "0",  # Disable panic mode for smokes (write capsule before kill)
        "GX1_REQUIRE_XGB_CALIBRATION": "0",  # Allow uncalibrated XGB for smoke tests
    }  # No XGB ablation toggles (baseline)
    
    test1_ablated_config = {
        "GX1_GATED_FUSION_ENABLED": "1",
        "GX1_REPLAY_USE_PREBUILT_FEATURES": "1",
        "GX1_ALLOW_PARALLEL_REPLAY": "1",
        "GX1_REQUIRE_ENTRY_TELEMETRY": "1",  # Required for A/B tests
        "GX1_PANIC_MODE": "0",  # Disable panic mode for smokes
        "GX1_REQUIRE_XGB_CALIBRATION": "0",  # Allow uncalibrated XGB for smoke tests
        "GX1_DISABLE_XGB_CHANNELS_IN_TRANSFORMER": "1",
    }
    
    # NOTE: Test 2 configs REMOVED on 2026-01-24 (XGB POST removed from pipeline)
    
    # Run tests
    results = {}
    
    if args.arm in ["test1_channels", "both"]:
        log.info("="*80)
        log.info("Running Test 1: XGB channels → Transformer")
        log.info("="*80)
        
        # Run baseline with autorange (ensures transformer is called)
        baseline_result = run_replay_arm_with_autorange(
            "baseline",
            baseline_config,
            args.data,
            args.prebuilt_parquet,
            args.bundle_dir,
            args.policy,
            output_dir,
            smoke_date_range=args.smoke_date_range if not args.smoke_bars else None,
            smoke_bars=args.smoke_bars,
            workers=args.workers,
        )
        results["baseline"] = baseline_result
        
        if not baseline_result.get("success"):
            log.error("Baseline run failed, aborting")
            return 1
        
        # Run ablated (use same range as baseline for consistency)
        baseline_range = baseline_result.get("used_smoke_date_range") or baseline_result.get("used_smoke_bars")
        if baseline_range and isinstance(baseline_range, str) and ".." in baseline_range:
            ablated_smoke_date_range = baseline_range
            ablated_smoke_bars = None
        elif baseline_range:
            ablated_smoke_date_range = None
            ablated_smoke_bars = baseline_range
        else:
            ablated_smoke_date_range = args.smoke_date_range if not args.smoke_bars else None
            ablated_smoke_bars = args.smoke_bars
        
        ablated_result = run_replay_arm(
            "no_xgb_channels_in_transformer",
            test1_ablated_config,
            args.data,
            args.prebuilt_parquet,
            args.bundle_dir,
            args.policy,
            output_dir,
            smoke_date_range=ablated_smoke_date_range,
            smoke_bars=ablated_smoke_bars,
            workers=args.workers,
        )
        results["test1_ablated"] = ablated_result
        
        if not ablated_result.get("success"):
            log.error("Test 1 ablated run failed")
            return 1
        
        # Compare
        comparison = compare_arms(baseline_result, ablated_result, "test1_channels")
        results["test1_comparison"] = comparison
        
        # Write comparison
        json_path = output_dir / "XGB_FLOW_ABLATION_TEST1_COMPARE.json"
        with open(json_path, "w") as f:
            json.dump(comparison, f, indent=2, cls=NumpyEncoder)
        log.info(f"✅ Written comparison JSON: {json_path}")
        
        # Generate markdown report
        md_path = output_dir / "XGB_FLOW_ABLATION_TEST1_COMPARE.md"
        generate_markdown_report(comparison, md_path, "test1_channels")
    
    # NOTE: Test 2 (XGB POST calibration/veto) was REMOVED on 2026-01-24.
    # XGB now only provides pre-predict channels to Transformer. No post-processing.
    
    # Write overall results
    results_path = output_dir / "XGB_FLOW_ABLATION_RESULTS.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    log.info(f"✅ Written results JSON: {results_path}")
    
    log.info("="*80)
    log.info("✅ All tests completed")
    log.info(f"Results: {output_dir}")
    log.info("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
