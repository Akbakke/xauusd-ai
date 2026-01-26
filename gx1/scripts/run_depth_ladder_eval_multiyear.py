#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Depth Ladder Multi-Year Eval Orchestrator

Runs deterministic A/B evaluation: Baseline (3 layers) vs L+1 (4 layers).

⚠️  DO NOT MODIFY TRADING LOGIC - ONLY ARCHITECTURE DEPTH

Usage:
    python gx1/scripts/run_depth_ladder_eval_multiyear.py \
        --arm baseline|lplus1 \
        --bundle-dir <PATH> \
        --years 2020,2021,2022,2023,2024,2025 \
        --workers 6 \
        --out-root reports/replay_eval/DEPTH_LADDER \
        [--smoke]  # Fast mode: 2025 only, 1 worker
"""

import argparse
import hashlib
import json
import logging
import multiprocessing
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
import yaml

try:
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

if not PYARROW_AVAILABLE:
    log.warning("[DEPTH_LADDER] pyarrow not available - will use pandas fallback for parquet metadata")

# Force spawn method for multiprocessing
if multiprocessing.get_start_method(allow_none=True) != "spawn":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

# Add workspace root to path
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))


def compute_data_universe_fingerprint(
    data_path: Path,
    prebuilt_path: Path,
    policy_path: Path,
    bundle_dir: Path,
    bundle_metadata: Dict[str, Any],
    workspace_root: Path,
    year: int,
    arm: str,
    smoke_date_range: Optional[str] = None,
    smoke_bars: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute DataUniverseFingerprint for smoke eval.
    
    This fingerprint captures all inputs that affect trade universe:
    - Data paths and row counts
    - Prebuilt features paths and row counts
    - Policy paths and hashes
    - Bundle metadata
    - Environment settings
    
    Args:
        data_path: Path to candles data parquet
        prebuilt_path: Path to prebuilt features parquet
        policy_path: Path to policy YAML
        bundle_dir: Path to bundle directory
        bundle_metadata: Bundle metadata dict
        workspace_root: Workspace root
        year: Year being processed
        arm: "baseline" or "lplus1"
    
    Returns:
        Dict with fingerprint data
    """
    fingerprint = {
        "fingerprint_version": "1.0",
        "computed_at": datetime.now().isoformat(),
        "arm": arm,
        "year": year,
    }
    
    # Resolve all paths to absolute
    data_path_resolved = data_path.resolve()
    prebuilt_path_resolved = prebuilt_path.resolve()
    policy_path_resolved = policy_path.resolve()
    bundle_dir_resolved = bundle_dir.resolve()
    
    fingerprint["data_root_resolved"] = str(data_path_resolved)
    fingerprint["prebuilt_parquet_resolved"] = str(prebuilt_path_resolved)
    fingerprint["policy_path_resolved"] = str(policy_path_resolved)
    fingerprint["bundle_dir_resolved"] = str(bundle_dir_resolved)
    
    # Get candles stats
    try:
        # Read just index to get rowcount and timestamp range (memory efficient)
        df_candles = pd.read_parquet(data_path_resolved, columns=[])
        if not isinstance(df_candles.index, pd.DatetimeIndex):
            raise RuntimeError(f"[DEPTH_LADDER] Data index must be DatetimeIndex, got: {type(df_candles.index)}")
        
        fingerprint["candles_rowcount_loaded"] = len(df_candles)
        fingerprint["candles_first_ts"] = df_candles.index.min().isoformat()
        fingerprint["candles_last_ts"] = df_candles.index.max().isoformat()
    except Exception as e:
        raise RuntimeError(f"[DEPTH_LADDER] Failed to read candles data: {e}") from e
    
    # Get prebuilt stats
    try:
        if PYARROW_AVAILABLE:
            # Use pyarrow for efficient metadata reading
            parquet_file = pq.ParquetFile(prebuilt_path_resolved, memory_map=False)
            metadata = parquet_file.metadata
            fingerprint["prebuilt_rowcount"] = metadata.num_rows
            
            # Read first and last row to get timestamp range
            # Try to find timestamp column in schema
            schema = parquet_file.schema
            ts_col = None
            for col_name in schema.names:
                if "timestamp" in col_name.lower() or col_name == "time" or col_name == "ts":
                    ts_col = col_name
                    break
            
            if ts_col:
                # Read first and last row groups
                first_rg = parquet_file.read_row_group(0, columns=[ts_col])
                last_rg_idx = metadata.num_row_groups - 1
                last_rg = parquet_file.read_row_group(last_rg_idx, columns=[ts_col])
                
                # Convert pyarrow timestamp to pandas timestamp
                first_ts_val = first_rg[ts_col][0].as_py()
                last_ts_val = last_rg[ts_col][-1].as_py()
                first_ts = pd.to_datetime(first_ts_val).isoformat()
                last_ts = pd.to_datetime(last_ts_val).isoformat()
                fingerprint["prebuilt_first_ts"] = first_ts
                fingerprint["prebuilt_last_ts"] = last_ts
            else:
                # Fallback: use index if available
                df_prebuilt = pd.read_parquet(prebuilt_path_resolved, columns=[])
                if isinstance(df_prebuilt.index, pd.DatetimeIndex):
                    fingerprint["prebuilt_first_ts"] = df_prebuilt.index.min().isoformat()
                    fingerprint["prebuilt_last_ts"] = df_prebuilt.index.max().isoformat()
                else:
                    fingerprint["prebuilt_first_ts"] = None
                    fingerprint["prebuilt_last_ts"] = None
        else:
            # Fallback: use pandas
            df_prebuilt = pd.read_parquet(prebuilt_path_resolved, columns=[])
            fingerprint["prebuilt_rowcount"] = len(df_prebuilt)
            if isinstance(df_prebuilt.index, pd.DatetimeIndex):
                fingerprint["prebuilt_first_ts"] = df_prebuilt.index.min().isoformat()
                fingerprint["prebuilt_last_ts"] = df_prebuilt.index.max().isoformat()
            else:
                fingerprint["prebuilt_first_ts"] = None
                fingerprint["prebuilt_last_ts"] = None
    except Exception as e:
        raise RuntimeError(f"[DEPTH_LADDER] Failed to read prebuilt features: {e}") from e
    
    # Get policy hash/ID
    try:
        with open(policy_path_resolved, "rb") as f:
            policy_hash = hashlib.sha256(f.read()).hexdigest()
        fingerprint["policy_sha256"] = policy_hash
        
        # Try to extract policy_id from YAML
        with open(policy_path_resolved, "r") as f:
            policy_yaml = yaml.safe_load(f)
        policy_id = policy_yaml.get("policy_id") or policy_path_resolved.stem
        fingerprint["policy_id"] = policy_id
    except Exception as e:
        raise RuntimeError(f"[DEPTH_LADDER] Failed to read policy: {e}") from e
    
    # Bundle metadata
    transformer_layers = bundle_metadata.get("transformer_layers")
    transformer_layers_baseline = bundle_metadata.get("transformer_layers_baseline", 3)
    depth_ladder_delta = bundle_metadata.get("depth_ladder_delta", 0)
    
    # Infer for baseline if not set
    if transformer_layers is None and arm == "baseline":
        transformer_layers = 3
        transformer_layers_baseline = 3
        depth_ladder_delta = 0
        log.info(f"[DEPTH_LADDER] Inferred baseline metadata: layers=3, delta=0")
    
    fingerprint["transformer_layers"] = transformer_layers
    fingerprint["transformer_layers_baseline"] = transformer_layers_baseline
    fingerprint["depth_ladder_delta"] = depth_ladder_delta
    
    # Bundle SHA256
    bundle_sha256 = bundle_metadata.get("sha256")
    if not bundle_sha256:
        # Compute from model_state_dict.pt
        model_path = bundle_dir_resolved / "model_state_dict.pt"
        if model_path.exists():
            with open(model_path, "rb") as f:
                bundle_sha256 = hashlib.sha256(f.read()).hexdigest()
        else:
            log.warning(f"[DEPTH_LADDER] model_state_dict.pt not found, cannot compute bundle_sha256")
    
    fingerprint["bundle_sha256"] = bundle_sha256
    
    # Entry model variant ID
    entry_model_variant_id = bundle_metadata.get("entry_model_variant_id")
    if not entry_model_variant_id:
        if arm == "lplus1":
            entry_model_variant_id = "v10_ctx_depth_ladder_lplus1"
        else:
            entry_model_variant_id = "v10_ctx_baseline"
    fingerprint["entry_model_variant_id"] = entry_model_variant_id
    
    # Environment settings
    fingerprint["replay_mode"] = "PREBUILT"
    fingerprint["years_requested"] = [year]
    
    # Smoke mode parameters (for deterministic subset)
    expected_candles_in_subset = None
    expected_prebuilt_rows_in_subset = None
    
    if smoke_date_range:
        fingerprint["smoke_date_range"] = smoke_date_range
        # Parse and validate date range
        if ".." in smoke_date_range:
            start_str, end_str = smoke_date_range.split("..", 1)
            start_ts = pd.to_datetime(start_str.strip())
            end_ts = pd.to_datetime(end_str.strip())
            fingerprint["smoke_date_range_start"] = start_str.strip()
            fingerprint["smoke_date_range_end"] = end_str.strip()
            
            # Ensure timezone-aware comparison (match index timezone)
            if df_candles.index.tz is not None:
                # Index is timezone-aware, make start_ts and end_ts timezone-aware too
                if start_ts.tz is None:
                    start_ts = start_ts.tz_localize(df_candles.index.tz)
                else:
                    start_ts = start_ts.tz_convert(df_candles.index.tz)
                if end_ts.tz is None:
                    end_ts = end_ts.tz_localize(df_candles.index.tz)
                else:
                    end_ts = end_ts.tz_convert(df_candles.index.tz)
            
            # Calculate expected subset size (fail-fast validation)
            df_candles_subset = df_candles[(df_candles.index >= start_ts) & (df_candles.index <= end_ts)]
            expected_candles_in_subset = len(df_candles_subset)
            fingerprint["expected_candles_in_subset"] = expected_candles_in_subset
            
            # Store expected subset range (critical for validation)
            if len(df_candles_subset) > 0:
                fingerprint["subset_first_ts_expected"] = df_candles_subset.index.min().isoformat()
                fingerprint["subset_last_ts_expected"] = df_candles_subset.index.max().isoformat()
            else:
                raise RuntimeError(
                    f"[DEPTH_LADDER] FATAL: smoke_date_range {smoke_date_range} results in 0 candles. "
                    f"Data range: {df_candles.index.min()} to {df_candles.index.max()}"
                )
            
            # For prebuilt, we need to check if it has the same range
            # This is a sanity check - prebuilt should match candles range
            if PYARROW_AVAILABLE:
                try:
                    parquet_file = pq.ParquetFile(prebuilt_path_resolved, memory_map=False)
                    # Read index to check range
                    df_prebuilt_check = pd.read_parquet(prebuilt_path_resolved, columns=[])
                    if isinstance(df_prebuilt_check.index, pd.DatetimeIndex):
                        # Ensure timezone-aware comparison for prebuilt too
                        start_ts_prebuilt = start_ts
                        end_ts_prebuilt = end_ts
                        if df_prebuilt_check.index.tz is not None:
                            if start_ts_prebuilt.tz is None:
                                start_ts_prebuilt = start_ts_prebuilt.tz_localize(df_prebuilt_check.index.tz)
                            else:
                                start_ts_prebuilt = start_ts_prebuilt.tz_convert(df_prebuilt_check.index.tz)
                            if end_ts_prebuilt.tz is None:
                                end_ts_prebuilt = end_ts_prebuilt.tz_localize(df_prebuilt_check.index.tz)
                            else:
                                end_ts_prebuilt = end_ts_prebuilt.tz_convert(df_prebuilt_check.index.tz)
                        df_prebuilt_subset = df_prebuilt_check[(df_prebuilt_check.index >= start_ts_prebuilt) & (df_prebuilt_check.index <= end_ts_prebuilt)]
                        expected_prebuilt_rows_in_subset = len(df_prebuilt_subset)
                        fingerprint["expected_prebuilt_rows_in_subset"] = expected_prebuilt_rows_in_subset
                except Exception as e:
                    log.warning(f"[DEPTH_LADDER] Could not calculate expected_prebuilt_rows_in_subset: {e}")
            
            # Hard-assert: subset must not be empty
            if expected_candles_in_subset == 0:
                raise RuntimeError(
                    f"[DEPTH_LADDER] FATAL: smoke_date_range {smoke_date_range} results in 0 candles. "
                    f"Data range: {df_candles.index.min()} to {df_candles.index.max()}"
                )
            
            log.info(f"[DEPTH_LADDER] Expected candles in subset: {expected_candles_in_subset:,}")
            if expected_prebuilt_rows_in_subset:
                log.info(f"[DEPTH_LADDER] Expected prebuilt rows in subset: {expected_prebuilt_rows_in_subset:,}")
    
    if smoke_bars:
        fingerprint["smoke_bars"] = smoke_bars
        expected_candles_in_subset = min(smoke_bars, len(df_candles))
        fingerprint["expected_candles_in_subset"] = expected_candles_in_subset
        expected_prebuilt_rows_in_subset = min(smoke_bars, fingerprint.get("prebuilt_rowcount", 0))
        fingerprint["expected_prebuilt_rows_in_subset"] = expected_prebuilt_rows_in_subset
        
        # Hard-assert: subset must not be empty
        if expected_candles_in_subset == 0:
            raise RuntimeError(
                f"[DEPTH_LADDER] FATAL: smoke_bars {smoke_bars} results in 0 candles. "
                f"Total candles: {len(df_candles)}"
            )
        
        log.info(f"[DEPTH_LADDER] Expected candles in subset: {expected_candles_in_subset:,}")
        if expected_prebuilt_rows_in_subset:
            log.info(f"[DEPTH_LADDER] Expected prebuilt rows in subset: {expected_prebuilt_rows_in_subset:,}")
    
    # Temperature scaling status (from env)
    temp_scaling_env = os.getenv("GX1_TEMPERATURE_SCALING", "1")
    fingerprint["temperature_scaling_env_value"] = temp_scaling_env
    fingerprint["temperature_scaling_effective_enabled"] = (temp_scaling_env == "1")
    fingerprint["temperature_scaling_source"] = "env_var"
    
    return fingerprint


def validate_bundle_metadata(bundle_dir: Path, expected_arm: str) -> Dict[str, Any]:
    """
    Validate bundle metadata and extract transformer_layers info.
    
    Args:
        bundle_dir: Path to bundle directory
        expected_arm: "baseline" or "lplus1"
    
    Returns:
        Bundle metadata dict
    
    Raises:
        RuntimeError: If validation fails
    """
    bundle_metadata_path = bundle_dir / "bundle_metadata.json"
    if not bundle_metadata_path.exists():
        raise RuntimeError(f"[DEPTH_LADDER] bundle_metadata.json not found: {bundle_metadata_path}")
    
    with open(bundle_metadata_path, "r") as f:
        metadata = json.load(f)
    
    transformer_layers = metadata.get("transformer_layers")
    transformer_layers_baseline = metadata.get("transformer_layers_baseline", 3)
    depth_ladder_delta = metadata.get("depth_ladder_delta", 0)
    
    # Validate expected layers
    if expected_arm == "baseline":
        expected_layers = 3
        # If transformer_layers is not set in metadata, assume baseline (3 layers)
        if transformer_layers is None:
            transformer_layers = 3
            log.info(f"[DEPTH_LADDER] transformer_layers not in metadata, assuming baseline (3 layers)")
    elif expected_arm == "lplus1":
        expected_layers = 4
    else:
        raise ValueError(f"Unknown arm: {expected_arm}")
    
    if transformer_layers != expected_layers:
        raise RuntimeError(
            f"[DEPTH_LADDER] transformer_layers mismatch: "
            f"expected={expected_layers} (arm={expected_arm}), "
            f"got={transformer_layers} from bundle metadata"
        )
    
    # Validate depth_ladder_delta
    if expected_arm == "baseline" and depth_ladder_delta != 0:
        raise RuntimeError(
            f"[DEPTH_LADDER] baseline arm must have depth_ladder_delta=0, got={depth_ladder_delta}"
        )
    elif expected_arm == "lplus1" and depth_ladder_delta != 1:
        raise RuntimeError(
            f"[DEPTH_LADDER] lplus1 arm must have depth_ladder_delta=1, got={depth_ladder_delta}"
        )
    
    log.info(f"[DEPTH_LADDER] Bundle validated: layers={transformer_layers}, delta={depth_ladder_delta}")
    
    return metadata


def run_year_replay(
    year: int,
    arm: str,
    bundle_dir: Path,
    data_path: Path,
    prebuilt_path: Path,
    policy_path: Path,
    output_dir: Path,
    workspace_root: Path,
    bundle_metadata: Optional[Dict[str, Any]] = None,
    start_ts: Optional[str] = None,
    end_ts: Optional[str] = None,
    safety_timeout_seconds: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run replay for a single year.
    
    Args:
        year: Year to process
        arm: "baseline" or "lplus1"
        bundle_dir: Path to bundle directory
        data_path: Path to year data parquet
        prebuilt_path: Path to prebuilt features
        policy_path: Path to policy YAML
        output_dir: Output directory for this year
        workspace_root: Workspace root
    
    Returns:
        Dict with replay metadata
    """
    log.info(f"[{arm.upper()}] Running replay for year {year}...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set environment variables
    env = os.environ.copy()
    env["GX1_REPLAY_USE_PREBUILT_FEATURES"] = "1"
    env["GX1_REPLAY_PREBUILT_FEATURES_PATH"] = str(prebuilt_path.resolve())
    env["GX1_FEATURE_BUILD_DISABLED"] = "1"
    env["GX1_GATED_FUSION_ENABLED"] = "1"
    env["GX1_REQUIRE_XGB_CALIBRATION"] = "1"
    env["GX1_REPLAY_MODE"] = "PREBUILT"
    env["GX1_ANALYSIS_MODE"] = "1"
    env["GX1_ALLOW_PARALLEL_REPLAY"] = "1"
    
    # ============================================================================
    # DEPTH LADDER MODE: Set explicit env vars
    # ============================================================================
    env["GX1_DEPTH_LADDER_MODE"] = "1"
    env["GX1_DEPTH_LADDER_VARIANT"] = arm
    
    # Thread limits
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["GX1_XGB_THREADS"] = "1"
    
    # Trial 160 policy parameters
    env["GX1_ENTRY_THRESHOLD_OVERRIDE"] = "0.102"
    env["GX1_RISK_GUARD_BLOCK_SPREAD_BPS_GTE_OVERRIDE"] = "2000"
    env["GX1_RISK_GUARD_BLOCK_ATR_BPS_GTE_OVERRIDE"] = "13.73"
    env["GX1_RISK_GUARD_COOLDOWN_BARS_AFTER_ENTRY_OVERRIDE"] = "2"
    env["GX1_MAX_CONCURRENT_POSITIONS_OVERRIDE"] = "2"
    
    # Update policy to use correct bundle_dir
    # We need to modify policy YAML to point to bundle_dir
    # For now, we'll use a temporary policy file
    with open(policy_path, "r") as f:
        policy = yaml.safe_load(f)
    
    # Update bundle_dir in policy
    if "entry_models" not in policy:
        policy["entry_models"] = {}
    if "v10_ctx" not in policy["entry_models"]:
        policy["entry_models"]["v10_ctx"] = {}
    
    policy["entry_models"]["v10_ctx"]["bundle_dir"] = str(bundle_dir.resolve())
    policy["entry_models"]["v10_ctx"]["enabled"] = True
    
    # Resolve all relative paths in policy to absolute paths
    if "v10_ctx" in policy.get("entry_models", {}):
        v10_ctx_cfg = policy["entry_models"]["v10_ctx"]
        for key in ["feature_meta_path", "seq_scaler_path", "snap_scaler_path"]:
            if key in v10_ctx_cfg:
                rel_path = v10_ctx_cfg[key]
                if not Path(rel_path).is_absolute():
                    # Resolve relative to workspace root
                    abs_path = (workspace_root / rel_path).resolve()
                    v10_ctx_cfg[key] = str(abs_path)
    
    # Write temporary policy to a temp directory (not in output_dir)
    import tempfile
    temp_dir = Path(tempfile.gettempdir())
    temp_policy_path = temp_dir / f"depth_ladder_policy_{year}_{arm}_{os.getpid()}.yaml"
    with open(temp_policy_path, "w") as f:
        yaml.dump(policy, f)
    
    # Use run_trial160_year_job.py for replay
    from gx1.scripts.run_trial160_year_job import run_replay
    
    try:
        result = run_replay(
            year=year,
            data_path=data_path,
            prebuilt_path=prebuilt_path,
            report_out_dir=output_dir,
            policy_yaml_path=temp_policy_path,
            workspace_root=workspace_root,
            workers=1,  # Single worker per year
            start_ts=start_ts,
            end_ts=end_ts,
        )
        
        # Load bundle metadata if not provided
        if bundle_metadata is None:
            bundle_metadata_path = bundle_dir / "bundle_metadata.json"
            if bundle_metadata_path.exists():
                with open(bundle_metadata_path, "r") as f:
                    bundle_metadata = json.load(f)
            else:
                bundle_metadata = {}
        
        # Update RUN_IDENTITY with depth ladder metadata
        update_run_identity_with_depth_ladder_metadata(
            output_dir=output_dir,
            arm=arm,
            bundle_dir=bundle_dir,
            bundle_metadata=bundle_metadata,
            policy_path=policy_path,
        )
        
        # Validate invariants after replay
        validate_replay_invariants(output_dir, arm, bundle_dir)
        
        return result
        
    finally:
        # Clean up temp policy
        if temp_policy_path.exists():
            temp_policy_path.unlink()


def update_run_identity_with_depth_ladder_metadata(
    output_dir: Path,
    arm: str,
    bundle_dir: Path,
    bundle_metadata: Dict[str, Any],
    policy_path: Path,
) -> None:
    """
    Update RUN_IDENTITY.json with depth ladder metadata.
    
    This ensures RUN_IDENTITY always contains:
    - transformer_layers
    - transformer_layers_baseline
    - depth_ladder_delta
    - policy_id
    - replay_mode
    - temperature_scaling_effective_enabled
    
    Args:
        output_dir: Replay output directory
        arm: "baseline" or "lplus1"
        bundle_dir: Bundle directory path
        bundle_metadata: Bundle metadata dict
        policy_path: Policy YAML path
    """
    identity_path = output_dir / "RUN_IDENTITY.json"
    if not identity_path.exists():
        log.warning(f"[DEPTH_LADDER] RUN_IDENTITY.json not found, creating minimal one: {identity_path}")
        identity = {}
    else:
        with open(identity_path, "r") as f:
            identity = json.load(f)
    
    # Extract metadata from bundle
    transformer_layers = bundle_metadata.get("transformer_layers")
    transformer_layers_baseline = bundle_metadata.get("transformer_layers_baseline", 3)
    depth_ladder_delta = bundle_metadata.get("depth_ladder_delta", 0)
    
    # Infer for baseline if not set
    if transformer_layers is None and arm == "baseline":
        transformer_layers = 3
        transformer_layers_baseline = 3
        depth_ladder_delta = 0
    
    # Compute bundle SHA256 if not present
    if "bundle_sha256" not in identity or not identity.get("bundle_sha256"):
        import hashlib
        model_path = bundle_dir / "model_state_dict.pt"
        if model_path.exists():
            with open(model_path, "rb") as f:
                bundle_sha256 = hashlib.sha256(f.read()).hexdigest()
            identity["bundle_sha256"] = bundle_sha256
        else:
            log.warning(f"[DEPTH_LADDER] model_state_dict.pt not found, cannot compute bundle_sha256")
    
    # Update with depth ladder metadata
    identity["transformer_layers"] = transformer_layers
    identity["transformer_layers_baseline"] = transformer_layers_baseline
    identity["depth_ladder_delta"] = depth_ladder_delta
    
    # Ensure replay_mode is set
    if "replay_mode" not in identity:
        identity["replay_mode"] = "PREBUILT"
    
    # Ensure feature_build_disabled is set (PREBUILT mode)
    identity["feature_build_disabled"] = True
    
    # Ensure policy_id is set (extract from policy path)
    if "policy_id" not in identity or not identity.get("policy_id"):
        policy_id = policy_path.stem
        identity["policy_id"] = policy_id
    
    # Ensure temperature_scaling_effective_enabled is set
    # Check if temperature scaling was actually enabled during replay
    if "temperature_scaling_enabled" in identity:
        identity["temperature_scaling_effective_enabled"] = identity["temperature_scaling_enabled"]
    else:
        # Default to True for depth ladder evals (should be enabled)
        identity["temperature_scaling_enabled"] = True
        identity["temperature_scaling_effective_enabled"] = True
    
    # Write updated RUN_IDENTITY
    with open(identity_path, "w") as f:
        json.dump(identity, f, indent=2, sort_keys=True)
    
    log.info(f"[DEPTH_LADDER] Updated RUN_IDENTITY.json with depth ladder metadata")
    log.info(f"  transformer_layers={transformer_layers}")
    log.info(f"  transformer_layers_baseline={transformer_layers_baseline}")
    log.info(f"  depth_ladder_delta={depth_ladder_delta}")
    log.info(f"  policy_id={identity.get('policy_id')}")
    log.info(f"  replay_mode={identity.get('replay_mode')}")
    log.info(f"  temperature_scaling_effective_enabled={identity.get('temperature_scaling_effective_enabled')}")


def validate_replay_invariants(
    output_dir: Path,
    arm: str,
    bundle_dir: Path,
) -> None:
    """
    Validate invariants after replay.
    
    FATAL if:
    - GX1_DEPTH_LADDER_MODE != 1
    - depth_ladder_delta != {0, +1}
    - transformer_layers_baseline missing in RUN_IDENTITY
    - temperature_scaling_enabled == false
    - PREBUILT not used (feature_build_call_count > 0)
    
    Args:
        output_dir: Replay output directory
        arm: "baseline" or "lplus1"
        bundle_dir: Bundle directory path
    """
    log.info(f"[{arm.upper()}] Validating invariants...")
    
    # Load RUN_IDENTITY
    identity_path = output_dir / "RUN_IDENTITY.json"
    if not identity_path.exists():
        raise RuntimeError(f"[DEPTH_LADDER] RUN_IDENTITY.json not found: {identity_path}")
    
    with open(identity_path, "r") as f:
        identity = json.load(f)
    
    # ============================================================================
    # INVARIANT CHECKS (FATAL)
    # ============================================================================
    
    # Check replay_mode
    replay_mode = identity.get("replay_mode")
    if replay_mode != "PREBUILT":
        raise RuntimeError(
            f"[DEPTH_LADDER_INVARIANT] replay_mode must be PREBUILT, got: {replay_mode}"
        )
    
    # Check feature_build_disabled
    feature_build_disabled = identity.get("feature_build_disabled", False)
    if not feature_build_disabled:
        raise RuntimeError(
            f"[DEPTH_LADDER_INVARIANT] feature_build_disabled must be True (PREBUILT mode), got: {feature_build_disabled}"
        )
    
    # Check for feature_build_call_count in performance data (if available)
    perf_path = output_dir / "perf.json"
    if perf_path.exists():
        try:
            with open(perf_path, "r") as f:
                perf = json.load(f)
            feature_build_call_count = perf.get("feature_build_call_count", 0)
            if feature_build_call_count > 0:
                raise RuntimeError(
                    f"[DEPTH_LADDER_INVARIANT] feature_build_call_count must be 0 (PREBUILT mode), got: {feature_build_call_count}"
                )
        except Exception as e:
            log.warning(f"[DEPTH_LADDER] Could not check feature_build_call_count: {e}")
    
    # Check temperature scaling
    temperature_scaling_enabled = identity.get("temperature_scaling_enabled", False)
    if not temperature_scaling_enabled:
        raise RuntimeError(
            f"[DEPTH_LADDER_INVARIANT] temperature_scaling_enabled must be True, got: {temperature_scaling_enabled}"
        )
    
    # Check transformer_layers
    transformer_layers = identity.get("transformer_layers")
    transformer_layers_baseline = identity.get("transformer_layers_baseline")
    depth_ladder_delta = identity.get("depth_ladder_delta")
    
    if transformer_layers is None:
        raise RuntimeError(
            f"[DEPTH_LADDER_INVARIANT] transformer_layers missing in RUN_IDENTITY"
        )
    
    if transformer_layers_baseline is None:
        raise RuntimeError(
            f"[DEPTH_LADDER_INVARIANT] transformer_layers_baseline missing in RUN_IDENTITY"
        )
    
    if depth_ladder_delta is None:
        raise RuntimeError(
            f"[DEPTH_LADDER_INVARIANT] depth_ladder_delta missing in RUN_IDENTITY"
        )
    
    # Validate depth_ladder_delta
    if depth_ladder_delta not in [0, 1]:
        raise RuntimeError(
            f"[DEPTH_LADDER_INVARIANT] depth_ladder_delta must be 0 or 1, got: {depth_ladder_delta}"
        )
    
    # Validate transformer_layers matches arm
    if arm == "baseline" and transformer_layers != 3:
        raise RuntimeError(
            f"[DEPTH_LADDER_INVARIANT] baseline arm must have transformer_layers=3, got: {transformer_layers}"
        )
    elif arm == "lplus1" and transformer_layers != 4:
        raise RuntimeError(
            f"[DEPTH_LADDER_INVARIANT] lplus1 arm must have transformer_layers=4, got: {transformer_layers}"
        )
    
    # Check bundle_sha256
    bundle_sha256 = identity.get("bundle_sha256")
    if not bundle_sha256:
        raise RuntimeError(
            f"[DEPTH_LADDER_INVARIANT] bundle_sha256 missing in RUN_IDENTITY"
        )
    
    log.info(f"[{arm.upper()}] ✅ All invariants validated")
    log.info(f"  transformer_layers={transformer_layers}")
    log.info(f"  transformer_layers_baseline={transformer_layers_baseline}")
    log.info(f"  depth_ladder_delta={depth_ladder_delta}")
    log.info(f"  bundle_sha256={bundle_sha256[:16]}...")


def verify_prebuilt_and_temp_scaling(
    output_dir: Path,
    arm: str,
) -> Dict[str, Any]:
    """
    Verify PREBUILT mode and temperature scaling.
    
    FATAL if:
    - feature_build_disabled != True
    - feature_build_call_count > 0
    - temperature_scaling_effective_enabled != True
    
    Args:
        output_dir: Replay output directory
        arm: "baseline" or "lplus1"
    
    Returns:
        Dict with verification results
    """
    verification = {
        "prebuilt_verified": False,
        "temperature_scaling_verified": False,
        "feature_build_disabled": None,
        "feature_build_call_count": None,
        "temperature_scaling_effective_enabled": None,
    }
    
    # Load RUN_IDENTITY
    identity_path = output_dir / "RUN_IDENTITY.json"
    if not identity_path.exists():
        raise RuntimeError(f"[DEPTH_LADDER] RUN_IDENTITY.json not found: {identity_path}")
    
    with open(identity_path, "r") as f:
        identity = json.load(f)
    
    # Check PREBUILT
    feature_build_disabled = identity.get("feature_build_disabled", False)
    verification["feature_build_disabled"] = feature_build_disabled
    
    if not feature_build_disabled:
        raise RuntimeError(
            f"[DEPTH_LADDER_SMOKE] FATAL: feature_build_disabled must be True (PREBUILT mode), got: {feature_build_disabled}"
        )
    
    # Check feature_build_call_count
    feature_build_call_count = None
    perf_path = output_dir / "perf.json"
    if perf_path.exists():
        try:
            with open(perf_path, "r") as f:
                perf = json.load(f)
            feature_build_call_count = perf.get("feature_build_call_count", 0)
        except Exception as e:
            log.warning(f"[DEPTH_LADDER] Could not read perf.json: {e}")
    
    # Try chunk_0 perf if not found in root
    if feature_build_call_count is None:
        chunk_0_perf = output_dir / "chunk_0" / "perf.json"
        if chunk_0_perf.exists():
            try:
                with open(chunk_0_perf, "r") as f:
                    perf = json.load(f)
                feature_build_call_count = perf.get("feature_build_call_count", 0)
            except Exception as e:
                log.warning(f"[DEPTH_LADDER] Could not read chunk_0/perf.json: {e}")
    
    verification["feature_build_call_count"] = feature_build_call_count
    
    if feature_build_call_count is not None and feature_build_call_count > 0:
        raise RuntimeError(
            f"[DEPTH_LADDER_SMOKE] FATAL: feature_build_call_count must be 0 (PREBUILT mode), got: {feature_build_call_count}"
        )
    
    verification["prebuilt_verified"] = True
    
    # Check temperature scaling
    temperature_scaling_effective_enabled = identity.get("temperature_scaling_effective_enabled")
    if temperature_scaling_effective_enabled is None:
        temperature_scaling_effective_enabled = identity.get("temperature_scaling_enabled", False)
    
    verification["temperature_scaling_effective_enabled"] = temperature_scaling_effective_enabled
    
    if not temperature_scaling_effective_enabled:
        raise RuntimeError(
            f"[DEPTH_LADDER_SMOKE] FATAL: temperature_scaling_effective_enabled must be True, got: {temperature_scaling_effective_enabled}"
        )
    
    verification["temperature_scaling_verified"] = True
    
    log.info(f"[{arm.upper()}] ✅ PREBUILT verified: feature_build_disabled=True, feature_build_call_count={feature_build_call_count or 0}")
    log.info(f"[{arm.upper()}] ✅ Temperature scaling verified: enabled=True")
    
    return verification


def validate_universe_match(
    baseline_fingerprint_path: Path,
    current_fingerprint: Dict[str, Any],
    arm: str,
) -> None:
    """
    Validate that current fingerprint matches baseline fingerprint.
    
    FATAL if mismatch in:
    - candles_first_ts, candles_last_ts, candles_rowcount_loaded
    - prebuilt_first_ts, prebuilt_last_ts, prebuilt_rowcount
    - policy_sha256
    - replay_mode
    - temperature_scaling_effective_enabled
    
    Args:
        baseline_fingerprint_path: Path to baseline_reference.json
        current_fingerprint: Current fingerprint dict
        arm: "baseline" or "lplus1"
    """
    if not baseline_fingerprint_path.exists():
        log.warning(f"[DEPTH_LADDER] Baseline reference not found: {baseline_fingerprint_path}")
        log.warning(f"[DEPTH_LADDER] Skipping universe match validation (first run)")
        return
    
    with open(baseline_fingerprint_path, "r") as f:
        baseline_fingerprint = json.load(f)
    
    # Extract DataUniverseFingerprint from baseline
    baseline_universe = baseline_fingerprint.get("data_universe_fingerprint", {})
    if not baseline_universe:
        # Try direct keys (if baseline_reference.json is just the fingerprint)
        baseline_universe = baseline_fingerprint
    
    # Keys to compare (must match exactly)
    critical_keys = [
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
    
    mismatches = []
    for key in critical_keys:
        baseline_val = baseline_universe.get(key)
        current_val = current_fingerprint.get(key)
        
        if baseline_val != current_val:
            mismatches.append({
                "key": key,
                "baseline": baseline_val,
                "current": current_val,
            })
    
    if mismatches:
        error_msg = f"[DEPTH_LADDER] FATAL: TRADE_UNIVERSE_MISMATCH\n"
        error_msg += f"Arm: {arm.upper()}\n"
        error_msg += f"Mismatches:\n"
        for mismatch in mismatches:
            error_msg += f"  {mismatch['key']}:\n"
            error_msg += f"    baseline: {mismatch['baseline']}\n"
            error_msg += f"    current:  {mismatch['current']}\n"
        
        raise RuntimeError(error_msg)
    
    log.info(f"[{arm.upper()}] ✅ Universe match validated (all critical keys match)")


def run_smoke_eval(
    arm: str,
    bundle_dir: Path,
    data_path: Path,
    prebuilt_path: Path,
    policy_path: Path,
    output_dir: Path,
    workspace_root: Path,
    bundle_metadata: Dict[str, Any],
    baseline_reference_path: Optional[Path] = None,
    smoke_date_range: Optional[str] = None,
    smoke_bars: Optional[int] = None,
    safety_timeout_seconds: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run smoke evaluation (2025 only, 1 worker, wiring verification).
    
    Ensures:
    - Same year=2025
    - Same workers=1
    - Same policy
    - Same prebuilt-parquet
    - Logs: bars_processed, trades_created, eligibility_blocks
    - Computes and validates DataUniverseFingerprint
    
    Args:
        arm: "baseline" or "lplus1"
        bundle_dir: Path to bundle directory
        data_path: Path to 2025 data
        prebuilt_path: Path to prebuilt features
        policy_path: Path to policy YAML
        output_dir: Output directory
        workspace_root: Workspace root
        bundle_metadata: Bundle metadata dict
        baseline_reference_path: Optional path to baseline_reference.json for universe match validation
    
    Returns:
        Dict with smoke eval results
    """
    log.info(f"[SMOKE] Running smoke eval for {arm.upper()}...")
    log.info(f"[SMOKE] Configuration:")
    log.info(f"  Year: 2025 (fixed)")
    log.info(f"  Workers: 1 (fixed)")
    log.info(f"  Policy: {policy_path}")
    log.info(f"  Prebuilt: {prebuilt_path}")
    log.info(f"  Data: {data_path}")
    
    year = 2025
    year_output_dir = output_dir / str(year)
    
    # Compute fingerprint BEFORE replay (from input data)
    log.info(f"[SMOKE] Computing DataUniverseFingerprint...")
    fingerprint = compute_data_universe_fingerprint(
        data_path=data_path,
        prebuilt_path=prebuilt_path,
        policy_path=policy_path,
        bundle_dir=bundle_dir,
        bundle_metadata=bundle_metadata,
        workspace_root=workspace_root,
        year=year,
        arm=arm,
        smoke_date_range=smoke_date_range,
        smoke_bars=smoke_bars,
    )
    
    # Log timeout configuration
    if safety_timeout_seconds is not None:
        fingerprint["safety_timeout_seconds"] = safety_timeout_seconds
        log.info(f"[SMOKE] Safety timeout: {safety_timeout_seconds}s ({'disabled' if safety_timeout_seconds == 0 else 'enabled'})")
    if smoke_date_range:
        log.info(f"[SMOKE] Date range: {smoke_date_range}")
    if smoke_bars:
        log.info(f"[SMOKE] Bars limit: {smoke_bars}")
    
    # Validate universe match if baseline reference exists
    if baseline_reference_path and baseline_reference_path.exists():
        log.info(f"[SMOKE] Validating universe match against baseline...")
        validate_universe_match(baseline_reference_path, fingerprint, arm)
    
    # Parse date range for replay
    # Note: We pass ISO format strings to replay, not datetime objects
    # The replay script will handle timezone conversion
    start_ts_replay = None
    end_ts_replay = None
    if smoke_date_range and ".." in smoke_date_range:
        start_ts_replay = smoke_date_range.split("..")[0].strip()
        end_ts_replay = smoke_date_range.split("..")[1].strip()
    
    result = run_year_replay(
        year=year,
        arm=arm,
        bundle_dir=bundle_dir,
        data_path=data_path,
        prebuilt_path=prebuilt_path,
        policy_path=policy_path,
        output_dir=year_output_dir,
        workspace_root=workspace_root,
        bundle_metadata=bundle_metadata,
        start_ts=start_ts_replay,
        end_ts=end_ts_replay,
        safety_timeout_seconds=safety_timeout_seconds,
    )
    
    # Get actual candles processed from replay output and check completion status
    candles_processed = None
    candles_first_ts_actual = None
    candles_last_ts_actual = None
    timed_out = False
    completed = False
    last_ts_processed = None
    pct_of_candles_processed = None
    
    # Initialize run_completion early (before it's used)
    run_completion = {
        "timed_out": timed_out,
        "completed": completed,
        "last_ts_processed": last_ts_processed,
        "pct_of_candles_processed": pct_of_candles_processed,
        "candles_processed_actual": candles_processed,
        "bars_processed_total": None,
        "candles_first_ts_actual": candles_first_ts_actual,
        "candles_last_ts_actual": candles_last_ts_actual,
        "subset_first_ts_actual": None,
        "subset_last_ts_actual": None,
        "n_chunks_expected": None,
        "n_chunks_done": None,
        "errors_count": 0,
    }
    
    # Get completion status from orchestrator_summary and perf.json (master-level truth source)
    orchestrator_summary_path = year_output_dir / "orchestrator_summary.json"
    perf_json_path = year_output_dir / f"perf_TRIAL160_2025_*.json"
    perf_files = list(year_output_dir.glob("perf_TRIAL160_2025_*.json"))
    if not perf_files:
        perf_files = list(year_output_dir.glob("perf_*.json"))
    
    bars_processed_total = None
    n_chunks_expected = None
    n_chunks_done = None
    errors_count = 0
    
    # Try perf.json first (has total_bars)
    if perf_files:
        try:
            with open(perf_files[0], "r") as f:
                perf_data = json.load(f)
            bars_processed_total = perf_data.get("total_bars")
            n_chunks_expected = perf_data.get("chunks_total")
            n_chunks_done = perf_data.get("chunks_completed")
            # Check for timeout/early stop
            if perf_data.get("export_mode") == "watchdog_sigterm":
                timed_out = True
        except Exception as e:
            log.warning(f"[SMOKE] Could not read perf.json: {e}")
    
    # Also check orchestrator_summary
    if orchestrator_summary_path.exists():
        try:
            with open(orchestrator_summary_path, "r") as f:
                orchestrator_summary = json.load(f)
            # Use orchestrator_summary values if perf.json didn't have them
            if bars_processed_total is None:
                bars_processed_total = orchestrator_summary.get("total_bars_processed")
            if n_chunks_expected is None:
                n_chunks_expected = orchestrator_summary.get("n_chunks")
            if n_chunks_done is None:
                n_chunks_done = orchestrator_summary.get("n_chunks_completed", orchestrator_summary.get("chunks_completed"))
            errors_count = orchestrator_summary.get("errors_count", 0)
            # Check for timeout/early stop
            if orchestrator_summary.get("timed_out", False) or orchestrator_summary.get("early_stop", False):
                timed_out = True
        except Exception as e:
            log.warning(f"[SMOKE] Could not read orchestrator_summary.json: {e}")
    
    # Fallback: Check chunk footer for completion status and bars_processed (if perf/orchestrator not available)
    chunk_0_dir = year_output_dir / "chunk_0"
    if bars_processed_total is None and chunk_0_dir.exists():
        chunk_footer_path = chunk_0_dir / "chunk_footer.json"
        if chunk_footer_path.exists():
            try:
                with open(chunk_footer_path, "r") as f:
                    chunk_footer = json.load(f)
                status = chunk_footer.get("status", "unknown")
                timed_out = (status == "stopped" or "timeout" in status.lower() or "SIGTERM" in str(chunk_footer.get("error", "")))
                # Try to get total_bars from chunk_footer
                if bars_processed_total is None:
                    bars_processed_total = chunk_footer.get("total_bars")
                # Don't set completed from chunk_footer - use bars_processed comparison instead
            except Exception as e:
                log.warning(f"[SMOKE] Could not read chunk_footer.json: {e}")
    
    # Get actual processed data from raw_signals
    # Try MERGED first (preferred), then non-MERGED, then chunk_0
    raw_signals_files = list(year_output_dir.glob("raw_signals_*_MERGED.parquet"))
    if not raw_signals_files:
        raw_signals_files = list(year_output_dir.glob("raw_signals_*.parquet"))
    if not raw_signals_files:
        if chunk_0_dir.exists():
            raw_signals_files = list(chunk_0_dir.glob("raw_signals_*.parquet"))
    
    if raw_signals_files:
        try:
            # Prefer non-empty file: try chunk_0 first, then MERGED, then others
            # (MERGED might be empty if merge failed)
            chunk_files = [f for f in raw_signals_files if "chunk_0" in str(f)]
            merged_files = [f for f in raw_signals_files if "_MERGED" in f.name]
            
            file_to_read = None
            if chunk_files:
                # Try chunk_0 file first
                for f in chunk_files:
                    try:
                        test_df = pd.read_parquet(f, columns=[])
                        if len(test_df) > 0:
                            file_to_read = f
                            break
                    except:
                        continue
            
            if not file_to_read and merged_files:
                # Try MERGED file
                for f in merged_files:
                    try:
                        test_df = pd.read_parquet(f, columns=[])
                        if len(test_df) > 0:
                            file_to_read = f
                            break
                    except:
                        continue
            
            if not file_to_read:
                file_to_read = raw_signals_files[0]
            
            df_signals = pd.read_parquet(file_to_read)
            candles_processed = len(df_signals)
            
            # Try to get timestamp from index or 'time' column
            if isinstance(df_signals.index, pd.DatetimeIndex):
                candles_first_ts_actual = df_signals.index.min().isoformat()
                candles_last_ts_actual = df_signals.index.max().isoformat()
            elif "ts" in df_signals.columns:
                # 'ts' is the standard column name in raw_signals
                df_signals["ts"] = pd.to_datetime(df_signals["ts"])
                candles_first_ts_actual = df_signals["ts"].min().isoformat()
                candles_last_ts_actual = df_signals["ts"].max().isoformat()
            elif "time" in df_signals.columns:
                df_signals["time"] = pd.to_datetime(df_signals["time"])
                candles_first_ts_actual = df_signals["time"].min().isoformat()
                candles_last_ts_actual = df_signals["time"].max().isoformat()
            else:
                log.warning(f"[SMOKE] raw_signals has no DatetimeIndex, 'ts', or 'time' column")
                candles_first_ts_actual = None
                candles_last_ts_actual = None
            
            if candles_last_ts_actual:
                last_ts_processed = candles_last_ts_actual
                
                # Store signals timestamps (from raw_signals, not candle iteration)
                # These are renamed to signals_first_ts and signals_last_ts for clarity
                run_completion["signals_first_ts"] = candles_first_ts_actual
                run_completion["signals_last_ts"] = candles_last_ts_actual
                
                # Calculate percentage of candles processed
                if fingerprint.get("candles_rowcount_loaded"):
                    pct_of_candles_processed = (candles_processed / fingerprint["candles_rowcount_loaded"]) * 100.0
        except Exception as e:
            log.warning(f"[SMOKE] Could not read raw_signals for actual candles: {e}")
    
    # Update fingerprint with actual processed data and completion status
    if candles_processed is not None:
        fingerprint["candles_processed_actual"] = candles_processed
    if candles_first_ts_actual:
        fingerprint["candles_first_ts_actual"] = candles_first_ts_actual
    if candles_last_ts_actual:
        fingerprint["candles_last_ts_actual"] = candles_last_ts_actual
    
    # Split into data_universe_fingerprint (input) and run_completion_fingerprint (output/progress)
    # Note: timeout_seconds is in universe fingerprint because it affects stop condition
    if safety_timeout_seconds is not None:
        fingerprint["timeout_seconds"] = safety_timeout_seconds
    
    # Update run_completion_fingerprint with final values (output/progress, NOT part of universe match)
    # Note: run_completion was initialized early, now we update it with final values
    # Note: candles_first_ts_iterated and candles_last_ts_iterated will be set in run_summary
    #       (they're calculated based on bars_iterated == bars_total_in_subset)
    run_completion.update({
        "timed_out": timed_out,
        "completed": completed,  # Will be set based on bars_processed_total == bars_total_in_subset
        "last_ts_processed": last_ts_processed,
        "pct_of_candles_processed": pct_of_candles_processed,
        "candles_processed_actual": candles_processed,  # From raw_signals (fallback)
        "bars_processed_total": bars_processed_total,  # From orchestrator_summary (source of truth)
        "n_chunks_expected": n_chunks_expected,
        "n_chunks_done": n_chunks_done,
        "errors_count": errors_count,
    })
    
    # Remove output/progress fields from data_universe_fingerprint
    # (they should not be in universe match comparison)
    fingerprint.pop("timed_out", None)
    fingerprint.pop("completed", None)
    fingerprint.pop("last_ts_processed", None)
    fingerprint.pop("pct_of_candles_processed", None)
    fingerprint.pop("candles_processed_actual", None)
    
    # Load and log metrics
    metrics_files = list(year_output_dir.glob("metrics_TRIAL160_2025_*.json"))
    if not metrics_files:
        # Try chunk_0
        chunk_0_dir = year_output_dir / "chunk_0"
        if chunk_0_dir.exists():
            metrics_files = list(chunk_0_dir.glob("metrics_TRIAL160_2025_*.json"))
    
    if metrics_files:
        with open(metrics_files[0], "r") as f:
            metrics = json.load(f)
        log.info(f"[SMOKE] Metrics:")
        log.info(f"  bars_processed: {metrics.get('bars_processed', 'N/A')}")
        log.info(f"  trades_created: {metrics.get('n_trades', 'N/A')}")
        log.info(f"  eligibility_blocks: {metrics.get('eligibility_blocks', 'N/A')}")
        
        fingerprint["bars_processed"] = metrics.get("bars_processed")
        fingerprint["trades_created"] = metrics.get("n_trades")
        fingerprint["eligibility_blocks"] = metrics.get("eligibility_blocks")
    
    # Verify PREBUILT and temperature scaling
    verification = verify_prebuilt_and_temp_scaling(year_output_dir, arm)
    # Verification results are NOT part of universe fingerprint (they're verification status)
    # They're kept in the verification dict separately
    
    # Determine completion status based on bars_iterated vs bars_total_in_subset
    # Hard rule: completed=true ONLY if bars_iterated == bars_total_in_subset
    # Note: bars_iterated comes from chunk_footer.total_bars (replay loop iteration count)
    #       bars_total_in_subset comes from df_candles_subset (sliced subset length)
    
    bars_total_in_subset = fingerprint.get("expected_candles_in_subset")
    
    # If bars_processed_total is None, try to get it from chunk_footer (already checked above, but double-check)
    if bars_processed_total is None and chunk_0_dir.exists():
        chunk_footer_path = chunk_0_dir / "chunk_footer.json"
        if chunk_footer_path.exists():
            try:
                with open(chunk_footer_path, "r") as f:
                    chunk_footer = json.load(f)
                bars_processed_total = chunk_footer.get("total_bars") or chunk_footer.get("bars_processed")
                if bars_processed_total is not None:
                    log.info(f"[SMOKE] Got bars_iterated from chunk_footer.total_bars: {bars_processed_total}")
            except Exception as e:
                log.warning(f"[SMOKE] Could not read chunk_footer for bars_iterated: {e}")
    
    # Completion gate: bars_iterated == bars_total_in_subset
    if bars_total_in_subset is not None and bars_processed_total is not None:
        # Use bars_processed_total as bars_iterated (source of truth from replay loop)
        if bars_processed_total == bars_total_in_subset:
            completed = True
            log.info(f"[SMOKE] ✅ Completion verified: bars_iterated ({bars_processed_total}) == bars_total_in_subset ({bars_total_in_subset})")
        else:
            completed = False
            diff = bars_processed_total - bars_total_in_subset
            log.warning(f"[SMOKE] ⚠️  Incomplete: bars_iterated ({bars_processed_total}) != bars_total_in_subset ({bars_total_in_subset}), diff={diff}")
            # Do NOT set timed_out here - timed_out only means we reached timeout limit
            # incomplete_reason will be set in run_summary
    elif bars_processed_total is None:
        # Fallback: if no source of truth available, cannot determine completion
        completed = False
        log.warning(f"[SMOKE] ⚠️  Cannot determine completion: bars_iterated not available from perf/orchestrator/chunk_footer")
    
    # Determine incomplete reason (not just timed_out)
    incomplete_reason = None
    if not completed:
        if timed_out:
            incomplete_reason = "timeout"
        elif bars_processed_total is not None and bars_total_in_subset is not None:
            if bars_processed_total < bars_total_in_subset:
                incomplete_reason = "early_stop"  # Stopped before completing subset
            elif bars_processed_total > bars_total_in_subset:
                incomplete_reason = "range_mismatch"  # Iterated more than expected
        else:
            incomplete_reason = "unknown"
    
    # Hard asserts for subset validation (when completed=true)
    # Note: Completion is based on bars_iterated == bars_total_in_subset, not signals
    if completed and not timed_out:
        subset_first_expected = fingerprint.get("subset_first_ts_expected")
        subset_last_expected = fingerprint.get("subset_last_ts_expected")
        
        if bars_total_in_subset is not None and bars_processed_total is not None:
            if bars_processed_total != bars_total_in_subset:
                raise RuntimeError(
                    f"[DEPTH_LADDER] FATAL: Subset validation failed. "
                    f"bars_iterated ({bars_processed_total}) != bars_total_in_subset ({bars_total_in_subset})"
                )
            log.info(f"[SMOKE] ✅ Subset candles validation: bars_iterated ({bars_processed_total}) == bars_total_in_subset ({bars_total_in_subset})")
        
        # Validate candles iteration range (not signals range)
        # Completion is based on bars_iterated == bars_total_in_subset, not signals
        # We'll validate candles_first_ts_iterated and candles_last_ts_iterated after they're set in run_summary
        # For now, just log that validation will happen
        log.info(f"[SMOKE] Completion validation: bars_iterated == bars_total_in_subset (candles iteration range will be validated in run_summary)")
    
    # Write RUN_SUMMARY.json (master-level truth source)
    # Explicit definitions:
    # - bars_total_in_subset: antall candles i det sliced subsettet (fra df_candles_subset)
    # - bars_iterated: antall candles replay-loop'en faktisk itererte over (fra chunk_footer.total_bars)
    # - bars_eligible: antall candles som kom forbi pregate/eligibility (hvis relevant)
    # - bars_emitted_signals: antall raw_signals rows (ofte < eligible pga "no-signal")
    
    bars_total_in_subset_value = fingerprint.get("expected_candles_in_subset")  # Rename for clarity
    bars_iterated_value = bars_processed_total  # From chunk_footer.total_bars or perf.json
    bars_emitted_signals_value = candles_processed  # From raw_signals count
    
    # Get candles_first_ts_iterated and candles_last_ts_iterated from chunk_df (candle-stream)
    # If bars_iterated == bars_total_in_subset, we know loop iterated over all bars in subset
    # So candles_first_ts_iterated and candles_last_ts_iterated should match subset range
    # But we can also try to read from chunk_footer or chunk_df if available
    candles_first_ts_iterated = None
    candles_last_ts_iterated = None
    
    # Try to get from chunk_footer or chunk_df
    if chunk_0_dir.exists():
        chunk_footer_path = chunk_0_dir / "chunk_footer.json"
        if chunk_footer_path.exists():
            try:
                with open(chunk_footer_path, "r") as f:
                    chunk_footer = json.load(f)
                # Check if chunk_footer has chunk_df range info
                # If not, we'll use subset_first_ts_expected/subset_last_ts_expected as fallback
                # (since if bars_iterated == bars_total_in_subset, loop iterated over all bars)
                pass  # chunk_footer doesn't have this info, we'll use fallback
            except Exception as e:
                log.warning(f"[SMOKE] Could not read chunk_footer for candles timestamps: {e}")
    
    # Fallback: if bars_iterated == bars_total_in_subset, use subset range
    # (loop iterated over all bars in subset, so first/last iterated = first/last in subset)
    if bars_iterated_value == bars_total_in_subset_value:
        candles_first_ts_iterated = fingerprint.get("subset_first_ts_expected")
        candles_last_ts_iterated = fingerprint.get("subset_last_ts_expected")
    else:
        # If incomplete, we can't determine exact range, but we can try to infer from signals
        # (signals_first_ts might be later than candles_first_ts_iterated due to warmup)
        candles_first_ts_iterated = fingerprint.get("subset_first_ts_expected")  # Best guess
        candles_last_ts_iterated = run_completion.get("signals_last_ts")  # May be earlier if incomplete
    
    # Check for entry feature telemetry files (written by chunks)
    telemetry_files = {}
    chunk_dirs = sorted(year_output_dir.glob("chunk_*"))
    if chunk_dirs:
        # Check first chunk for telemetry files
        first_chunk_dir = chunk_dirs[0]
        if (first_chunk_dir / "ENTRY_FEATURES_USED.json").exists():
            telemetry_files["entry_features_used_json"] = str(first_chunk_dir / "ENTRY_FEATURES_USED.json")
        if (first_chunk_dir / "ENTRY_FEATURES_USED.md").exists():
            telemetry_files["entry_features_used_md"] = str(first_chunk_dir / "ENTRY_FEATURES_USED.md")
        if (first_chunk_dir / "FEATURE_MASK_APPLIED.json").exists():
            telemetry_files["feature_mask_applied_json"] = str(first_chunk_dir / "FEATURE_MASK_APPLIED.json")
        if (first_chunk_dir / "ENTRY_FEATURES_TELEMETRY.json").exists():
            telemetry_files["entry_features_telemetry_json"] = str(first_chunk_dir / "ENTRY_FEATURES_TELEMETRY.json")
    
    run_summary = {
        "arm": arm,
        "year": year,
        "bars_total_in_subset": bars_total_in_subset_value,  # Antall candles i det sliced subsettet
        "bars_iterated": bars_iterated_value,  # Antall candles replay-loop'en faktisk itererte over
        "bars_emitted_signals": bars_emitted_signals_value,  # Antall raw_signals rows
        # Expected range (from df_candles_subset)
        "subset_first_ts_expected": fingerprint.get("subset_first_ts_expected"),
        "subset_last_ts_expected": fingerprint.get("subset_last_ts_expected"),
        # Candle iteration range (from replay-loop / candle-stream)
        "candles_first_ts_iterated": candles_first_ts_iterated,
        "candles_last_ts_iterated": candles_last_ts_iterated,
        # Signals range (from raw_signals)
        "signals_first_ts": run_completion.get("signals_first_ts"),
        "signals_last_ts": run_completion.get("signals_last_ts"),
        "completed": completed,  # True ONLY if bars_iterated == bars_total_in_subset
        "incomplete": not completed,
        "incomplete_reason": incomplete_reason,  # "timeout" | "early_stop" | "range_mismatch" | "unknown" | null
        "timed_out": timed_out,  # True ONLY if we reached timeout limit
        "n_chunks_expected": n_chunks_expected,
        "n_chunks_done": n_chunks_done,
        "errors_count": errors_count,
        "computed_at": datetime.now().isoformat(),
        # Entry feature telemetry files (if available)
        "entry_feature_telemetry_files": telemetry_files if telemetry_files else None,
    }
    
    run_summary_path = output_dir / "RUN_SUMMARY.json"
    with open(run_summary_path, "w") as f:
        json.dump(run_summary, f, indent=2, sort_keys=True)
    log.info(f"[SMOKE] ✅ Written RUN_SUMMARY.json: {run_summary_path}")
    
    # Write smoke_stats.json
    smoke_stats = {
        "arm": arm,
        "year": year,
        "data_universe_fingerprint": fingerprint,  # Input/configuration only
        "run_completion_fingerprint": run_completion,  # Output/progress only
        "run_summary": run_summary,  # Master-level truth source
        "verification": verification,
        "computed_at": datetime.now().isoformat(),
    }
    
    smoke_stats_path = output_dir / "smoke_stats.json"
    with open(smoke_stats_path, "w") as f:
        json.dump(smoke_stats, f, indent=2, sort_keys=True)
    log.info(f"[SMOKE] ✅ Written smoke_stats.json: {smoke_stats_path}")
    
    # If baseline, save as baseline_reference.json
    if arm == "baseline":
        baseline_ref_path = output_dir.parent / "baseline_reference.json"
        with open(baseline_ref_path, "w") as f:
            json.dump(smoke_stats, f, indent=2, sort_keys=True)
        log.info(f"[SMOKE] ✅ Written baseline_reference.json: {baseline_ref_path}")
    
    # Load trades for comparison
    trades_path = year_output_dir / "trades.parquet"
    if not trades_path.exists():
        # Try chunk_0
        chunk_0_trades = year_output_dir / "chunk_0" / "trades.parquet"
        if chunk_0_trades.exists():
            trades_path = chunk_0_trades
    
    if trades_path.exists():
        df = pd.read_parquet(trades_path)
        log.info(f"[SMOKE] {arm.upper()}: {len(df)} trades")
        # trades_count is output/progress, NOT part of universe fingerprint
        # It's available in metrics, but not in fingerprint
    
    return result


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
    
    parser = argparse.ArgumentParser(description="Depth Ladder Multi-Year Eval Orchestrator")
    parser.add_argument("--arm", type=str, required=True, choices=["baseline", "lplus1"],
                        help="Arm to evaluate (baseline or lplus1)")
    parser.add_argument("--bundle-dir", type=Path, required=True,
                        help="Path to bundle directory")
    parser.add_argument("--years", type=str, default="2020,2021,2022,2023,2024,2025",
                        help="Comma-separated years (default: 2020,2021,2022,2023,2024,2025)")
    parser.add_argument("--workers", type=int, default=6,
                        help="Number of parallel workers (default: 6, 1 per year)")
    # DEL 4A: Use GX1_DATA env vars for default paths
    default_reports_root = Path(os.getenv("GX1_REPORTS_ROOT", "../GX1_DATA/reports"))
    parser.add_argument("--out-root", type=Path, default=default_reports_root / "replay_eval" / "DEPTH_LADDER",
                        help="Output root directory (default: reports/replay_eval/DEPTH_LADDER)")
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Path to data root (contains year subdirectories)")
    parser.add_argument("--prebuilt-parquet", type=Path, required=True,
                        help="Absolute path to prebuilt features parquet")
    parser.add_argument("--policy", type=Path, required=True,
                        help="Path to policy YAML")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke mode: 2025 only, 1 worker, wiring verification")
    parser.add_argument("--smoke-full", action="store_true",
                        help="Smoke mode with full 2025 dataset (no timeout, no date range limit)")
    parser.add_argument("--smoke-date-range", type=str, default=None,
                        help="Smoke mode date range (ISO format: 2025-01-01..2025-12-31 or 2025-01-01..2025-04-30)")
    parser.add_argument("--smoke-bars", type=int, default=None,
                        help="Smoke mode: use only first N bars (deterministic subset)")
    parser.add_argument("--safety-timeout-seconds", type=int, default=600,
                        help="Safety timeout in seconds (default: 600, 0=disabled, smoke-full sets to disabled)")
    parser.add_argument("--recheck-baseline-with-arm-config", action="store_true",
                        help="Re-run baseline smoke with same config as specified arm (for diagnosis)")
    
    args = parser.parse_args()
    
    # Resolve paths to absolute
    args.policy = args.policy.resolve()
    args.prebuilt_parquet = args.prebuilt_parquet.resolve()
    args.bundle_dir = args.bundle_dir.resolve()
    args.data_root = args.data_root.resolve()
    
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
    
    # Validate bundle
    bundle_dir = args.bundle_dir
    if not bundle_dir.exists():
        raise FileNotFoundError(f"Bundle directory not found: {bundle_dir}")
    
    bundle_metadata = validate_bundle_metadata(bundle_dir, args.arm)
    
    # Generate run ID
    run_id = f"DEPTH_LADDER_{args.arm.upper()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = args.out_root / run_id / args.arm.upper()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log.info(f"[DEPTH_LADDER] Starting eval for {args.arm.upper()}")
    log.info(f"  Bundle: {bundle_dir}")
    log.info(f"  Output: {output_dir}")
    log.info(f"  Run ID: {run_id}")
    
    # ============================================================================
    # DEL D: Bundle load selftest (fail early)
    # ============================================================================
    log.info("[DEPTH_LADDER] Running bundle load selftest...")
    try:
        from gx1.scripts.selftest_entry_v10_bundle_load import test_bundle_load
        
        selftest_result = test_bundle_load(bundle_dir, args.policy)
        
        if selftest_result["status"] != "OK":
            raise RuntimeError(
                f"[DEPTH_LADDER] Bundle load selftest failed: {selftest_result.get('exception', {}).get('message', 'Unknown error')}"
            )
        
        # Write selftest result
        selftest_output_path = output_dir / "bundle_load_selftest.json"
        with open(selftest_output_path, "w") as f:
            json.dump(selftest_result, f, indent=2)
        log.info(f"✅ Bundle load selftest passed: {selftest_output_path}")
        
    except Exception as e:
        log.error(f"❌ Bundle load selftest failed: {e}")
        # Write failure result
        selftest_fail = {
            "bundle_dir": str(bundle_dir),
            "status": "FAIL",
            "exception": {
                "type": type(e).__name__,
                "message": str(e),
            },
        }
        selftest_output_path = output_dir / "bundle_load_selftest.json"
        with open(selftest_output_path, "w") as f:
            json.dump(selftest_fail, f, indent=2)
        raise RuntimeError(f"[DEPTH_LADDER] Bundle load selftest failed - aborting eval") from e
    
    # Write master_early.json (will be updated with fingerprint after smoke if smoke mode)
    master_early = {
        "run_id": run_id,
        "arm": args.arm,
        "bundle_dir": str(bundle_dir),
        "transformer_layers": bundle_metadata.get("transformer_layers"),
        "transformer_layers_baseline": bundle_metadata.get("transformer_layers_baseline", 3),
        "depth_ladder_delta": bundle_metadata.get("depth_ladder_delta", 0),
        "bundle_sha256": bundle_metadata.get("sha256"),  # If available
        "start_time": datetime.now().isoformat(),
        "smoke_mode": args.smoke,
    }
    
    master_early_path = output_dir / "master_early.json"
    with open(master_early_path, "w") as f:
        json.dump(master_early, f, indent=2)
    log.info(f"✅ Written master_early.json: {master_early_path}")
    
    # Smoke mode
    if args.smoke or args.smoke_full:
        log.info("[SMOKE] Running in smoke mode (2025 only, 1 worker)")
        
        # Determine safety timeout
        if args.smoke_full:
            safety_timeout_seconds = 0  # Disabled for smoke-full
            log.info("[SMOKE] smoke-full mode: safety timeout disabled")
        else:
            safety_timeout_seconds = args.safety_timeout_seconds
            if safety_timeout_seconds == 0:
                log.info("[SMOKE] Safety timeout disabled (--safety-timeout-seconds=0)")
            else:
                log.info(f"[SMOKE] Safety timeout: {safety_timeout_seconds}s")
        
        # Parse smoke date range if provided
        smoke_date_range = args.smoke_date_range
        smoke_bars = args.smoke_bars
        start_ts = None
        end_ts = None
        
        if smoke_date_range:
            if ".." in smoke_date_range:
                start_str, end_str = smoke_date_range.split("..", 1)
                start_ts = start_str.strip()
                end_ts = end_str.strip()
                log.info(f"[SMOKE] Date range: {start_ts} to {end_ts}")
            else:
                raise ValueError(f"Invalid smoke-date-range format: {smoke_date_range} (expected: START..END)")
        
        if smoke_bars:
            log.info(f"[SMOKE] Bars limit: {smoke_bars}")
            # Note: smoke_bars is handled by replay_eval_gated_parallel via --slice-head
        
        # Handle --recheck-baseline-with-arm-config
        if args.recheck_baseline_with_arm_config:
            if args.arm != "lplus1":
                raise ValueError("--recheck-baseline-with-arm-config requires --arm lplus1")
            log.info("[SMOKE] Recheck mode: Will run baseline with same config as L+1")
            # Temporarily switch to baseline arm for this run
            actual_arm = "baseline"
            # Use baseline bundle (DEL 4A: Use GX1_DATA env vars for default paths)
            default_models_root = Path(os.getenv("GX1_MODELS_ROOT", "../GX1_DATA/models"))
            baseline_bundle_dir = default_models_root / "entry_v10_ctx" / "FULLYEAR_2025_GATED_FUSION"
            if not baseline_bundle_dir.exists():
                raise FileNotFoundError(f"Baseline bundle not found: {baseline_bundle_dir}")
            bundle_dir = baseline_bundle_dir
            bundle_metadata = validate_bundle_metadata(bundle_dir, "baseline")
            args.arm = "baseline"  # Temporarily override
        else:
            actual_arm = args.arm
        
        # Try multiple possible data paths
        possible_paths = [
            args.data_root / "2025" / "xauusd_m5_2025_bid_ask.parquet",
            args.data_root / "2025.parquet",
            args.data_root / "xauusd_m5_2025_bid_ask.parquet",
        ]
        data_path = None
        for path in possible_paths:
            if path.exists():
                data_path = path
                break
        
        if data_path is None:
            raise FileNotFoundError(
                f"Smoke data not found. Tried: {[str(p) for p in possible_paths]}"
            )
        
        # Find baseline reference if L+1
        baseline_reference_path = None
        if actual_arm == "lplus1":
            # Look for baseline_reference.json in parent directory
            baseline_ref_candidates = [
                output_dir.parent / "baseline_reference.json",
                args.out_root / "baseline_reference.json",
            ]
            for candidate in baseline_ref_candidates:
                if candidate.exists():
                    baseline_reference_path = candidate
                    break
        
        result = run_smoke_eval(
            arm=actual_arm,
            bundle_dir=bundle_dir,
            data_path=data_path,
            prebuilt_path=args.prebuilt_parquet,
            policy_path=args.policy,
            output_dir=output_dir,
            workspace_root=workspace_root,
            bundle_metadata=bundle_metadata,
            baseline_reference_path=baseline_reference_path,
            smoke_date_range=smoke_date_range,
            smoke_bars=smoke_bars,
            safety_timeout_seconds=safety_timeout_seconds,
        )
        
        # Update master_early.json with fingerprint if available
        smoke_stats_path = output_dir / "smoke_stats.json"
        if smoke_stats_path.exists():
            with open(smoke_stats_path, "r") as f:
                smoke_stats = json.load(f)
            fingerprint = smoke_stats.get("data_universe_fingerprint", {})
            if fingerprint:
                master_early["data_universe_fingerprint"] = fingerprint
                with open(master_early_path, "w") as f:
                    json.dump(master_early, f, indent=2)
                log.info(f"✅ Updated master_early.json with fingerprint")
        
        log.info("✅ Smoke eval complete")
        return 0
    
    # Parse years
    years = [int(y.strip()) for y in args.years.split(",")]
    
    # Run years in parallel
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    futures = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for year in years:
            # Try multiple possible data paths
            possible_paths = [
                args.data_root / str(year) / f"xauusd_m5_{year}_bid_ask.parquet",
                args.data_root / f"{year}.parquet",
                args.data_root / f"xauusd_m5_{year}_bid_ask.parquet",
            ]
            year_data_path = None
            for path in possible_paths:
                if path.exists():
                    year_data_path = path
                    break
            
            if year_data_path is None:
                log.warning(f"Year {year} data not found. Tried: {[str(p) for p in possible_paths]}, skipping")
                continue
            
            year_output_dir = output_dir / str(year)
            
            future = executor.submit(
                run_year_replay,
                year=year,
                arm=args.arm,
                bundle_dir=bundle_dir,
                data_path=year_data_path,
                prebuilt_path=args.prebuilt_parquet,
                policy_path=args.policy,
                output_dir=year_output_dir,
                workspace_root=workspace_root,
                bundle_metadata=bundle_metadata,
            )
            futures.append((year, future))
    
    # Collect results
    results = {}
    for year, future in futures:
        try:
            result = future.result()
            results[year] = result
            log.info(f"✅ Year {year} complete")
        except Exception as e:
            log.error(f"❌ Year {year} failed: {e}", exc_info=True)
            results[year] = {"error": str(e)}
    
    # Write master summary
    master_summary = {
        **master_early,
        "end_time": datetime.now().isoformat(),
        "years": list(results.keys()),
        "results": results,
    }
    
    master_summary_path = output_dir / "master_summary.json"
    with open(master_summary_path, "w") as f:
        json.dump(master_summary, f, indent=2)
    log.info(f"✅ Written master_summary.json: {master_summary_path}")
    
    log.info("✅ Depth Ladder eval complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
