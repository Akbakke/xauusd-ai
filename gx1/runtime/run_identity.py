#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runner Identity Invariant — RUN_IDENTITY.json Generator

Generates RUN_IDENTITY.json with all required identity fields.
Hard-fails if any required field is missing.

Dependencies (explicit install line):
  (no external dependencies beyond stdlib)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)


class ReplayMode(str, Enum):
    """Replay mode enum."""
    PREBUILT = "PREBUILT"  # Replay with prebuilt features
    LIVE = "LIVE"  # Live trading (builds features on-the-fly)
    BASELINE = "BASELINE"  # Legacy, not allowed for Trial 160


class OutputMode(str, Enum):
    """Output mode enum."""
    MINIMAL = "MINIMAL"  # Default: Only essential files (RUN_IDENTITY, RUN_COMPLETED, metrics)
    DEBUG = "DEBUG"  # MINIMAL + telemetry + error logs
    TRUTH = "TRUTH"  # Full artifacts, but no per-bar/per-tick logs


@dataclass
class RunIdentity:
    """Runner identity (fail-fast, no defaults)."""
    
    git_head_sha: str
    git_dirty: bool
    python_executable: str
    python_version: str
    bundle_sha256: Optional[str]
    bundle_dir_resolved: Optional[str]  # Resolved absolute path to bundle directory
    bundle_dir_source: Optional[str]  # "cli" | "env" | "policy" | None
    bundle_dir_exists: Optional[bool]  # Whether bundle_dir actually exists
    prebuilt_manifest_sha256: Optional[str]
    prebuilt_manifest_path: Optional[str]
    policy_id: str
    policy_sha256: str
    replay_mode: ReplayMode
    prebuilt_used: bool
    feature_build_disabled: bool
    windows_sha: Optional[str]
    feature_schema_fingerprint: Optional[str] = None  # Feature fingerprint hash
    # Exit Policy V2 fields
    exit_v2_enabled: bool = False
    exit_v2_yaml_path: Optional[str] = None
    exit_v2_yaml_sha256: Optional[str] = None
    exit_v2_params_summary: Optional[Dict[str, Any]] = None
    exit_v2_counters: Optional[Dict[str, int]] = None
    
    # DEL 3: Pre-Entry Wait Gate telemetry
    pre_entry_wait_enabled: bool = False
    pre_entry_wait_counters: Optional[Dict[str, int]] = None
    
    # DEL OVERLAP: OVERLAP Overlay telemetry
    overlap_overlay_enabled: bool = False
    overlap_overlay_config: Optional[Dict[str, Any]] = None
    overlap_overlay_counters: Optional[Dict[str, int]] = None
    overlap_override_applied_n: int = 0  # BUGFIX: Count of threshold overrides applied
    overlap_override_applied_outside_overlap: int = 0  # BUGFIX: FATAL if >0 (invariant violation)
    
    # CRITICAL: dt_module version stamp
    dt_module_version: Optional[str] = None
    
    # DEPTH LADDER: Transformer architecture info
    entry_model_variant_id: Optional[str] = None  # e.g., "ENTRY_V10_CTX_LPLUS1"
    transformer_layers: Optional[int] = None  # Actual num_layers
    transformer_layers_baseline: Optional[int] = None  # Baseline num_layers
    depth_ladder_delta: Optional[int] = None  # Delta from baseline
    
    # REPLAY SSoT: Entry model ID from replay_config (required for replay)
    entry_model_id: Optional[str] = None  # e.g., "ENTRY_V10_CTX_FULLYEAR_2025_GATED_FUSION"
    
    # PHASE 1 FIX: Exit invariant counters
    exit_monotonicity_violations: int = 0
    duplicate_exit_attempts: int = 0
    
    # PHASE 1 FIX: NaN/Inf prediction counters
    nan_inf_pred_count: int = 0
    nan_inf_pred_first_ts: Optional[str] = None
    nan_inf_pred_mode_handling: str = "UNKNOWN"
    
    # PHASE 1 FIX: Temperature scaling status
    temperature_scaling_enabled: bool = False
    temperature_map_loaded: bool = False
    temperature_defaults_used_count: int = 0
    
    # OUTPUT POLICY: Output mode (MINIMAL/DEBUG/TRUTH)
    output_mode: str = "MINIMAL"  # Default: MINIMAL (only essential files)

    # CHAIN/REPLAY SEMANTICS: Exit semantics tag (Phase 1 proof)
    # Examples: "CALENDAR_TRUE"
    replay_exit_semantics: Optional[str] = None
    
    # Session tokens / snap dim metadata (truth verification)
    session_tokens_enabled: Optional[bool] = None
    snap_dim_expected: Optional[int] = None
    snap_dim_effective: Optional[int] = None

    # SIGNAL-ONLY TRUTH (SSoT)
    signal_bridge_contract_sha256: Optional[str] = None
    seq_signal_dim: Optional[int] = None
    snap_signal_dim: Optional[int] = None

    # XGB LOCK (SSoT)
    xgb_lock_path: Optional[str] = None
    xgb_lock_file_sha256: Optional[str] = None
    xgb_lock_feature_contract_id: Optional[str] = None
    xgb_lock_feature_list_sha256: Optional[str] = None
    xgb_lock_model_sha256: Optional[str] = None

    # Prebuilt schema manifest (SSoT)
    prebuilt_schema_manifest_path: Optional[str] = None
    prebuilt_schema_manifest_sha256: Optional[str] = None
    
    # Diagnostic override metadata (replay truth only)
    diagnostic_enabled: Optional[bool] = None
    diagnostic_force_eval_sessions: Optional[str] = None
    diagnostic_bypassed_gate: Optional[str] = None
    diagnostic_mode: Optional[str] = None
    
    # Ritual warmup padding (replay truth only)
    ritual_warmup_padding_days: int = 0
    requested_eval_start_ts: Optional[str] = None
    requested_eval_end_ts: Optional[str] = None
    effective_replay_start_ts: Optional[str] = None
    effective_replay_end_ts: Optional[str] = None
    
    # XGB SESSION POLICY: Policy hash and US disable status
    xgb_session_policy_hash: Optional[str] = None
    xgb_us_disabled_enabled: bool = False  # True if US is disabled in policy
    xgb_us_disabled_count: int = 0  # Count of US rows that returned neutral outputs
    xgb_predict_count_total: int = 0  # Total XGB predict calls
    xgb_predict_count_eu: int = 0  # XGB predict calls for EU session
    xgb_predict_count_us: int = 0  # XGB predict calls for US session
    xgb_predict_count_overlap: int = 0  # XGB predict calls for OVERLAP session
    
    # PROVENANCE: Track policy loading provenance
    run_identity_impl_module: Optional[str] = None  # e.g., "gx1.runtime.run_identity"
    run_identity_impl_file: Optional[str] = None  # __file__ (absolute path)
    gx1_package_root: Optional[str] = None  # Path(gx1.__file__).resolve().parent
    resolved_policy_path: Optional[str] = None  # Absolute path attempted
    policy_load_status: Optional[str] = None  # "OK" | "MISSING" | "PARSE_ERROR" | "HASH_ERROR"
    policy_load_error: Optional[str] = None  # Short error string (only if error)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "git_head_sha": self.git_head_sha,
            "git_dirty": self.git_dirty,
            "python_executable": self.python_executable,
            "python_version": self.python_version,
            "bundle_sha256": self.bundle_sha256,
            "bundle_dir_resolved": self.bundle_dir_resolved,
            "bundle_dir_source": self.bundle_dir_source,
            "bundle_dir_exists": self.bundle_dir_exists,
            "prebuilt_manifest_sha256": self.prebuilt_manifest_sha256,
            "prebuilt_manifest_path": self.prebuilt_manifest_path,
            "policy_id": self.policy_id,
            "policy_sha256": self.policy_sha256,
            "replay_mode": self.replay_mode.value,
            "prebuilt_used": self.prebuilt_used,
            "feature_build_disabled": self.feature_build_disabled,
            "windows_sha": self.windows_sha,
            "feature_schema_fingerprint": self.feature_schema_fingerprint,
            "exit_v2_enabled": self.exit_v2_enabled,
            "pre_entry_wait_enabled": self.pre_entry_wait_enabled,
            "overlap_overlay_enabled": self.overlap_overlay_enabled,
            "exit_monotonicity_violations": self.exit_monotonicity_violations,
            "duplicate_exit_attempts": self.duplicate_exit_attempts,
            "nan_inf_pred_count": self.nan_inf_pred_count,
            "nan_inf_pred_first_ts": self.nan_inf_pred_first_ts,
            "nan_inf_pred_mode_handling": self.nan_inf_pred_mode_handling,
            "temperature_scaling_enabled": self.temperature_scaling_enabled,
            "temperature_map_loaded": self.temperature_map_loaded,
            "temperature_defaults_used_count": self.temperature_defaults_used_count,
            "output_mode": self.output_mode,
            "replay_exit_semantics": self.replay_exit_semantics,
            "xgb_session_policy_hash": self.xgb_session_policy_hash,  # Always include (None if not loaded)
            "xgb_us_disabled_enabled": self.xgb_us_disabled_enabled,  # Always include (False if not loaded)
            "session_tokens_enabled": self.session_tokens_enabled,
            "snap_dim_expected": self.snap_dim_expected,
            "snap_dim_effective": self.snap_dim_effective,
            "signal_bridge_contract_sha256": self.signal_bridge_contract_sha256,
            "seq_signal_dim": self.seq_signal_dim,
            "snap_signal_dim": self.snap_signal_dim,
            "xgb_lock_path": self.xgb_lock_path,
            "xgb_lock_file_sha256": self.xgb_lock_file_sha256,
            "xgb_lock_feature_contract_id": self.xgb_lock_feature_contract_id,
            "xgb_lock_feature_list_sha256": self.xgb_lock_feature_list_sha256,
            "xgb_lock_model_sha256": self.xgb_lock_model_sha256,
            "prebuilt_schema_manifest_path": self.prebuilt_schema_manifest_path,
            "prebuilt_schema_manifest_sha256": self.prebuilt_schema_manifest_sha256,
            "diagnostic_enabled": self.diagnostic_enabled,
            "diagnostic_force_eval_sessions": self.diagnostic_force_eval_sessions,
            "diagnostic_bypassed_gate": self.diagnostic_bypassed_gate,
            "diagnostic_mode": self.diagnostic_mode,
            "ritual_warmup_padding_days": self.ritual_warmup_padding_days,
            "requested_eval_start_ts": self.requested_eval_start_ts,
            "requested_eval_end_ts": self.requested_eval_end_ts,
            "effective_replay_start_ts": self.effective_replay_start_ts,
            "effective_replay_end_ts": self.effective_replay_end_ts,
        }
        # Always include provenance fields (even if None for debugging)
        result["run_identity_impl_module"] = self.run_identity_impl_module
        result["run_identity_impl_file"] = self.run_identity_impl_file
        result["gx1_package_root"] = self.gx1_package_root
        result["resolved_policy_path"] = self.resolved_policy_path
        result["policy_load_status"] = self.policy_load_status
        if self.policy_load_error:
            result["policy_load_error"] = self.policy_load_error
        if self.xgb_us_disabled_count > 0:
            result["xgb_us_disabled_count"] = self.xgb_us_disabled_count
        if self.xgb_predict_count_total > 0:
            result["xgb_predict_count_total"] = self.xgb_predict_count_total
        if self.xgb_predict_count_eu > 0:
            result["xgb_predict_count_eu"] = self.xgb_predict_count_eu
        if self.xgb_predict_count_us > 0:
            result["xgb_predict_count_us"] = self.xgb_predict_count_us
        if self.xgb_predict_count_overlap > 0:
            result["xgb_predict_count_overlap"] = self.xgb_predict_count_overlap
        if self.entry_model_variant_id:
            result["entry_model_variant_id"] = self.entry_model_variant_id
        if self.transformer_layers is not None:
            result["transformer_layers"] = self.transformer_layers
        if self.transformer_layers_baseline is not None:
            result["transformer_layers_baseline"] = self.transformer_layers_baseline
        if self.depth_ladder_delta is not None:
            result["depth_ladder_delta"] = self.depth_ladder_delta
        if self.dt_module_version:
            result["dt_module_version"] = self.dt_module_version
        if self.exit_v2_yaml_path:
            result["exit_v2_yaml_path"] = self.exit_v2_yaml_path
        if self.exit_v2_yaml_sha256:
            result["exit_v2_yaml_sha256"] = self.exit_v2_yaml_sha256
        if self.exit_v2_params_summary:
            result["exit_v2_params_summary"] = self.exit_v2_params_summary
        if self.exit_v2_counters:
            result["exit_v2_counters"] = self.exit_v2_counters
        if self.pre_entry_wait_counters:
            result["pre_entry_wait_counters"] = self.pre_entry_wait_counters
        if self.overlap_overlay_config:
            result["overlap_overlay_config"] = self.overlap_overlay_config
        if self.overlap_overlay_counters:
            result["overlap_overlay_counters"] = self.overlap_overlay_counters
        return result
    
    def ensure_xgb_policy_fields_loaded(self, truth_mode: bool = False) -> None:
        """
        Load XGB session policy and populate xgb_session_policy_hash and xgb_us_disabled_enabled.
        
        Sets provenance fields for debugging:
        - run_identity_impl_module
        - run_identity_impl_file
        - gx1_package_root
        - resolved_policy_path
        - policy_load_status ("OK" | "MISSING" | "PARSE_ERROR" | "HASH_ERROR" | "ENTERED" | "EXCEPTION")
        - policy_load_error (if error)
        
        Args:
            truth_mode: If True, hard-fail on missing/invalid policy
        
        Raises:
            RuntimeError: If truth_mode=True and policy is missing or invalid
        """
        import importlib
        
        # EXEC_SENTINEL: Set provenance and status IMMEDIATELY to prove method execution
        self.run_identity_impl_module = __name__
        self.run_identity_impl_file = str(Path(__file__).resolve())
        self.policy_load_status = "ENTERED"
        self.policy_load_error = "ENTERED_OK"  # Temporary, will be cleared on success
        
        # Resolve gx1 package root using importlib (not repo_root)
        try:
            gx1_module = importlib.import_module("gx1")
            # Handle namespace packages or __file__ being None
            if hasattr(gx1_module, "__file__") and gx1_module.__file__ is not None:
                gx1_package_root = Path(gx1_module.__file__).resolve().parent
            else:
                # Fallback: use this file's location to infer gx1 root
                # This file is at gx1/runtime/run_identity.py, so gx1 root is parent.parent
                gx1_package_root = Path(__file__).resolve().parent.parent
            self.gx1_package_root = str(gx1_package_root)
        except Exception as e:
            # Fallback: use this file's location to infer gx1 root
            try:
                gx1_package_root = Path(__file__).resolve().parent.parent
                self.gx1_package_root = str(gx1_package_root)
            except Exception as e2:
                if truth_mode:
                    raise RuntimeError(
                        f"RUN_IDENTITY_FAIL: Cannot determine gx1 package root: {e} (fallback also failed: {e2})"
                    ) from e2
                self.gx1_package_root = None
                self.policy_load_status = "MISSING"
                self.policy_load_error = f"Cannot determine gx1 root: {e}"
                return
        
        # Construct policy path relative to gx1 package root
        policy_path = gx1_package_root / "configs" / "xgb_session_policy.json"
        self.resolved_policy_path = str(policy_path.resolve())
        
        # Check if policy exists
        if not policy_path.exists():
            self.policy_load_status = "MISSING"
            self.policy_load_error = f"Policy file not found: {policy_path}"
            if truth_mode:
                raise RuntimeError(
                    f"RUN_IDENTITY_FAIL: XGB session policy not found: {policy_path} "
                    f"(required in TRUTH/SMOKE mode)"
                )
            return
        
        # Load and parse policy
        try:
            # Compute SHA256 hash
            xgb_session_policy_hash = _compute_file_sha256(policy_path)
            if len(xgb_session_policy_hash) != 64:
                self.policy_load_status = "HASH_ERROR"
                self.policy_load_error = f"Hash length is {len(xgb_session_policy_hash)}, expected 64"
                if truth_mode:
                    raise RuntimeError(
                        f"RUN_IDENTITY_FAIL: Invalid policy hash length: {len(xgb_session_policy_hash)}"
                    )
                return
            
            # Parse JSON
            with open(policy_path, "r", encoding="utf-8") as f:
                import json
                policy = json.load(f)
            
            # Extract US disable status
            us_config = policy.get("sessions", {}).get("US", {})
            xgb_us_disabled_enabled = not us_config.get("enabled", True)
            
            # Set fields
            self.xgb_session_policy_hash = xgb_session_policy_hash
            self.xgb_us_disabled_enabled = xgb_us_disabled_enabled
            self.policy_load_status = "OK"
            self.policy_load_error = None  # Clear temporary "ENTERED_OK"
            
        except json.JSONDecodeError as e:
            self.policy_load_status = "PARSE_ERROR"
            self.policy_load_error = repr(e)
            if truth_mode:
                raise RuntimeError(
                    f"RUN_IDENTITY_FAIL: Failed to parse XGB session policy JSON: {e} "
                    f"(required in TRUTH/SMOKE mode)"
                ) from e
        except Exception as e:
            self.policy_load_status = "EXCEPTION"
            self.policy_load_error = repr(e)
            if truth_mode:
                raise RuntimeError(
                    f"RUN_IDENTITY_FAIL: Failed to load XGB session policy: {e} "
                    f"(required in TRUTH/SMOKE mode)"
                ) from e


def _get_git_head_sha(repo_root: Path) -> tuple[str, bool]:
    """Get git HEAD SHA and dirty flag."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=repo_root,
            check=True,
        )
        head_sha = result.stdout.strip()
        
        # Check if dirty
        result = subprocess.run(
            ["git", "diff", "--quiet"],
            cwd=repo_root,
        )
        dirty = result.returncode != 0
        
        return head_sha, dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError(
            f"RUN_IDENTITY_FAIL: Cannot determine git HEAD SHA from {repo_root}"
        )


def _get_python_version() -> str:
    """Get Python version string."""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def _compute_file_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of file content."""
    if not file_path.exists():
        raise FileNotFoundError(f"RUN_IDENTITY_FAIL: File not found: {file_path}")
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _find_bundle_sha256(bundle_dir: Optional[Path]) -> Optional[str]:
    """Find bundle SHA256 from model_state_dict.pt."""
    if bundle_dir is None:
        return None
    
    model_state_path = bundle_dir / "model_state_dict.pt"
    if model_state_path.exists():
        return _compute_file_sha256(model_state_path)
    
    return None


def _find_prebuilt_manifest_sha256(prebuilt_path: Optional[Path]) -> tuple[Optional[str], Optional[str]]:
    """Find prebuilt manifest SHA256 and path."""
    if prebuilt_path is None:
        return None, None
    
    # Look for manifest file (common patterns)
    manifest_patterns = [
        prebuilt_path.parent / f"{prebuilt_path.stem}.manifest.json",
        prebuilt_path.parent / f"{prebuilt_path.stem}_manifest.json",
        prebuilt_path.parent / "manifest.json",
    ]
    
    for manifest_path in manifest_patterns:
        if manifest_path.exists():
            return _compute_file_sha256(manifest_path), str(manifest_path.resolve())
    
    # If no manifest found, compute SHA256 of prebuilt file itself
    if prebuilt_path.exists():
        return _compute_file_sha256(prebuilt_path), str(prebuilt_path.resolve())
    
    return None, None


def _get_windows_sha(windows_path: Optional[Path]) -> Optional[str]:
    """Get windows SHA256 if windows file exists."""
    if windows_path is None or not windows_path.exists():
        return None
    return _compute_file_sha256(windows_path)


def create_run_identity(
    output_dir: Path,
    policy_id: str,
    policy_sha256: str,
    *,
    repo_root: Optional[Path] = None,
    bundle_dir: Optional[Path] = None,
    bundle_dir_source: Optional[str] = None,  # "cli" | "env" | "policy" | None
    prebuilt_path: Optional[Path] = None,
    windows_path: Optional[Path] = None,
    allow_dirty: bool = False,
    is_live: bool = False,
    feature_schema_fingerprint: Optional[str] = None,
    exit_monotonicity_violations: int = 0,
    duplicate_exit_attempts: int = 0,
    nan_inf_pred_count: int = 0,
    nan_inf_pred_first_ts: Optional[str] = None,
    temperature_scaling_enabled: bool = False,  # PHASE 1 FIX
    temperature_map_loaded: bool = False,  # PHASE 1 FIX
    temperature_defaults_used_count: int = 0,  # PHASE 1 FIX
    xgb_us_disabled_count: int = 0,  # XGB session policy telemetry
    prebuilt_used: Optional[bool] = None,
    entry_model_id: Optional[str] = None,  # REPLAY SSoT: Entry model ID from replay_config
    session_tokens_enabled: Optional[bool] = None,
    snap_dim_expected: Optional[int] = None,
    snap_dim_effective: Optional[int] = None,
    signal_bridge_contract_sha256: Optional[str] = None,
    seq_signal_dim: Optional[int] = None,
    snap_signal_dim: Optional[int] = None,
    xgb_lock_path: Optional[str] = None,
    xgb_lock_file_sha256: Optional[str] = None,
    xgb_lock_feature_contract_id: Optional[str] = None,
    xgb_lock_feature_list_sha256: Optional[str] = None,
    xgb_lock_model_sha256: Optional[str] = None,
    prebuilt_schema_manifest_path: Optional[str] = None,
    prebuilt_schema_manifest_sha256: Optional[str] = None,
    diagnostic_enabled: Optional[bool] = None,
    diagnostic_force_eval_sessions: Optional[str] = None,
    diagnostic_bypassed_gate: Optional[str] = None,
    diagnostic_mode: Optional[str] = None,
    ritual_warmup_padding_days: int = 0,
    requested_eval_start_ts: Optional[str] = None,
    requested_eval_end_ts: Optional[str] = None,
    effective_replay_start_ts: Optional[str] = None,
    effective_replay_end_ts: Optional[str] = None,
) -> RunIdentity:
    """
    Create and write RUN_IDENTITY.json to output directory.
    
    Hard-fails if:
    - Any required field cannot be determined
    - git_dirty is True and allow_dirty is False
    - For PREBUILT mode: feature_build_disabled is not True
    - For LIVE mode: policy_id or policy_sha256 mismatch
    
    Args:
        output_dir: Output directory (must exist)
        policy_id: Policy ID (from policy loader)
        policy_sha256: Policy SHA256 (from policy loader)
        repo_root: Repository root (default: infer from current working directory)
        bundle_dir: Bundle directory (optional)
        prebuilt_path: Prebuilt features path (optional)
        windows_path: Windows file path (optional)
        allow_dirty: Allow dirty git (default: False, must be True for smokes)
    
    Returns:
        RunIdentity object
    
    Raises:
        RuntimeError: If any required field cannot be determined
        ValueError: If invariants are violated
    """
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine repo root
    if repo_root is None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                capture_output=True,
                text=True,
                check=True,
            )
            repo_root = Path(result.stdout.strip())
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback: try to infer from script location or use None
            # This allows runs outside git repos (e.g., in production environments)
            script_path = Path(__file__).resolve()
            # Try to find workspace root by looking for common markers
            for parent in script_path.parents:
                if (parent / ".git").exists() or (parent / "gx1").exists():
                    repo_root = parent
                    break
            if repo_root is None:
                # Last resort: use script's parent directory
                repo_root = script_path.parents[2] if len(script_path.parents) >= 3 else script_path.parent
                log.warning(f"[RUN_IDENTITY] Cannot determine git repo root, using inferred path: {repo_root}")
    
    # Get git info
    git_head_sha, git_dirty = _get_git_head_sha(repo_root)
    
    # Hard-fail if dirty and not allowed
    if git_dirty and not allow_dirty:
        raise RuntimeError(
            f"RUN_IDENTITY_FAIL: Working tree is dirty (set allow_dirty=True to override). "
            f"Repo root: {repo_root}"
        )
    
    # Get Python info
    python_executable = sys.executable
    python_version = _get_python_version()
    
    # Resolve bundle_dir and determine source
    bundle_dir_resolved = None
    bundle_dir_exists = None
    if bundle_dir:
        bundle_dir_resolved = str(Path(bundle_dir).resolve())
        bundle_dir_exists = Path(bundle_dir).exists()
        if not bundle_dir_source:
            # Infer source from environment (if not provided)
            if os.getenv("GX1_BUNDLE_DIR"):
                bundle_dir_source = "env"
            else:
                bundle_dir_source = "policy"
    
    # Get bundle SHA256 and extract transformer_layers from bundle metadata
    bundle_sha256 = _find_bundle_sha256(bundle_dir)
    transformer_layers = None
    transformer_layers_baseline = None
    depth_ladder_delta = None
    entry_model_variant_id = None
    
    if bundle_dir:
        bundle_metadata_path = Path(bundle_dir) / "bundle_metadata.json"
        if bundle_metadata_path.exists():
            try:
                with open(bundle_metadata_path, "r") as f:
                    bundle_metadata = json.load(f)
                transformer_layers = bundle_metadata.get("transformer_layers")
                transformer_layers_baseline = bundle_metadata.get("transformer_layers_baseline", 3)
                depth_ladder_delta = bundle_metadata.get("depth_ladder_delta", 0)
                
                # Determine variant ID
                if depth_ladder_delta and depth_ladder_delta != 0:
                    entry_model_variant_id = f"ENTRY_V10_CTX_LPLUS{depth_ladder_delta}"
                else:
                    entry_model_variant_id = "ENTRY_V10_CTX"
            except Exception as e:
                log.warning(f"[RUN_IDENTITY] Failed to extract transformer_layers from bundle metadata: {e}")
    
    # Get prebuilt manifest SHA256
    prebuilt_manifest_sha256, prebuilt_manifest_path = _find_prebuilt_manifest_sha256(prebuilt_path)
    
    # Get windows SHA
    windows_sha = _get_windows_sha(windows_path)
    
    # Get Exit Policy V2 info (if available from env)
    exit_v2_enabled = os.getenv("GX1_EXIT_POLICY_V2", "0") == "1"
    exit_v2_yaml_path = os.getenv("GX1_EXIT_POLICY_V2_YAML")
    exit_v2_yaml_sha256 = None
    exit_v2_params_summary = None
    exit_v2_counters = None
    
    if exit_v2_enabled and exit_v2_yaml_path:
        exit_v2_yaml_path_obj = Path(exit_v2_yaml_path)
        if not exit_v2_yaml_path_obj.is_absolute():
            exit_v2_yaml_path_obj = repo_root / exit_v2_yaml_path_obj
        if exit_v2_yaml_path_obj.exists():
            exit_v2_yaml_sha256 = _compute_file_sha256(exit_v2_yaml_path_obj)
            exit_v2_yaml_path = str(exit_v2_yaml_path_obj.resolve())
    
    # DEL 3: Get Pre-Entry Wait Gate info (if available from env)
    pre_entry_wait_enabled = os.getenv("GX1_PRE_ENTRY_WAIT_GATE_ENABLED", "0") == "1"
    pre_entry_wait_counters = None  # Will be updated by EntryManager after replay
    
    # DEL OVERLAP: Get OVERLAP Overlay info (if available from env)
    overlap_overlay_config_path = os.getenv("GX1_OVERLAP_OVERLAY_CONFIG")
    overlap_overlay_enabled = overlap_overlay_config_path is not None
    overlap_overlay_config = None
    overlap_overlay_counters = None  # Will be updated by EntryManager after replay
    
    if overlap_overlay_enabled and overlap_overlay_config_path:
        try:
            from gx1.entry.overlap_overlay import load_overlap_overlay_config
            # Path is already imported at module level, don't shadow it
            config_path_obj = Path(overlap_overlay_config_path)
            if not config_path_obj.is_absolute():
                config_path_obj = repo_root / config_path_obj
            if config_path_obj.exists():
                overlay_config = load_overlap_overlay_config(config_path_obj)
                overlap_overlay_config = overlay_config.to_dict()
        except Exception as e:
            log.warning(f"[RUN_IDENTITY] Failed to load OVERLAP overlay config: {e}")
            overlap_overlay_enabled = False
    
    # Determine replay mode (PREBUILT or LIVE)
    replay_mode_env = os.getenv("GX1_REPLAY_USE_PREBUILT_FEATURES", "0")
    is_live_mode = os.getenv("GX1_LIVE_MODE", "0") == "1" or not prebuilt_path
    
    if replay_mode_env == "1":
        replay_mode = ReplayMode.PREBUILT
        # For PREBUILT mode, feature_build_disabled must be True
        feature_build_disabled_env = os.getenv("GX1_FEATURE_BUILD_DISABLED", "0")
        if feature_build_disabled_env != "1":
            raise ValueError(
                f"RUN_IDENTITY_FAIL: feature_build_disabled must be True for PREBUILT mode "
                f"(GX1_FEATURE_BUILD_DISABLED=1), got: {feature_build_disabled_env}"
            )
        feature_build_disabled = True
    elif is_live_mode:
        replay_mode = ReplayMode.LIVE
        # For LIVE mode, feature building is allowed (builds on-the-fly)
        feature_build_disabled = False
    else:
        # Default: assume PREBUILT if prebuilt_path is provided
        if prebuilt_path:
            replay_mode = ReplayMode.PREBUILT
            feature_build_disabled = True
        else:
            replay_mode = ReplayMode.LIVE
            feature_build_disabled = False
    
    if prebuilt_used is None:
        prebuilt_used = replay_mode == ReplayMode.PREBUILT and prebuilt_path is not None
    
    # CRITICAL: Get dt_module version (fail-fast if import fails)
    try:
        from gx1.utils.dt_module import get_dt_module_version, validate_dt_module_version
        validate_dt_module_version()
        dt_module_version = get_dt_module_version()
    except Exception as e:
        raise RuntimeError(
            f"[RUN_IDENTITY_FAIL] Cannot import/validate dt_module: {e}. "
            f"This indicates a critical code issue. FATAL."
        ) from e
    
    # OUTPUT POLICY: Get output mode (MINIMAL/DEBUG/TRUTH)
    # Default: MINIMAL (only essential files)
    # TRUTH must be explicitly enabled via flag or env
    output_mode_env = os.getenv("GX1_OUTPUT_MODE", "").upper()
    if output_mode_env in ("MINIMAL", "DEBUG", "TRUTH"):
        output_mode = output_mode_env
    else:
        # Default: MINIMAL (new policy - prevents reports explosion)
        output_mode = "MINIMAL"
    
    # XGB SESSION POLICY: Policy loading is deferred to ensure_xgb_policy_fields_loaded()
    # This is called explicitly before writing RUN_IDENTITY.json in replay_eval_gated_parallel.py
    xgb_session_policy_hash = None
    xgb_us_disabled_enabled = False
    
    # CHAIN/REPLAY: Exit semantics tag (Phase 1 proof)
    replay_exit_semantics = os.getenv("GX1_REPLAY_EXIT_SEMANTICS") or None

    # Create RunIdentity
    identity = RunIdentity(
        git_head_sha=git_head_sha,
        git_dirty=git_dirty,
        python_executable=python_executable,
        python_version=python_version,
        bundle_sha256=bundle_sha256,
        bundle_dir_resolved=bundle_dir_resolved,
        bundle_dir_source=bundle_dir_source,
        bundle_dir_exists=bundle_dir_exists,
        prebuilt_manifest_sha256=prebuilt_manifest_sha256,
        prebuilt_manifest_path=prebuilt_manifest_path,
        policy_id=policy_id,
        policy_sha256=policy_sha256,
        replay_mode=replay_mode,
        prebuilt_used=prebuilt_used,
        feature_build_disabled=feature_build_disabled,
        windows_sha=windows_sha,
        feature_schema_fingerprint=feature_schema_fingerprint,
        exit_v2_enabled=exit_v2_enabled,
        exit_v2_yaml_path=exit_v2_yaml_path,
        exit_v2_yaml_sha256=exit_v2_yaml_sha256,
        exit_v2_params_summary=exit_v2_params_summary,
        exit_v2_counters=exit_v2_counters,
        pre_entry_wait_enabled=pre_entry_wait_enabled,
        pre_entry_wait_counters=pre_entry_wait_counters,
        overlap_overlay_enabled=overlap_overlay_enabled,
        overlap_overlay_config=overlap_overlay_config,
        overlap_overlay_counters=overlap_overlay_counters,
        dt_module_version=dt_module_version,  # CRITICAL: Version stamp
        entry_model_variant_id=entry_model_variant_id,
        entry_model_id=entry_model_id,  # REPLAY SSoT: Entry model ID from replay_config
        transformer_layers=transformer_layers,
        transformer_layers_baseline=transformer_layers_baseline,
        depth_ladder_delta=depth_ladder_delta,
        exit_monotonicity_violations=exit_monotonicity_violations,  # PHASE 1 FIX
        duplicate_exit_attempts=duplicate_exit_attempts,  # PHASE 1 FIX
        nan_inf_pred_count=nan_inf_pred_count,  # PHASE 1 FIX
        nan_inf_pred_first_ts=nan_inf_pred_first_ts,  # PHASE 1 FIX
        nan_inf_pred_mode_handling="FATAL_IN_REPLAY",  # PHASE 1 FIX
        temperature_scaling_enabled=temperature_scaling_enabled,  # PHASE 1 FIX
        temperature_map_loaded=temperature_map_loaded,  # PHASE 1 FIX
        temperature_defaults_used_count=temperature_defaults_used_count,  # PHASE 1 FIX
        output_mode=output_mode,  # SWEEP OPTIMIZATION
        replay_exit_semantics=replay_exit_semantics,
        session_tokens_enabled=session_tokens_enabled,
        snap_dim_expected=snap_dim_expected,
        snap_dim_effective=snap_dim_effective,
        signal_bridge_contract_sha256=signal_bridge_contract_sha256,
        seq_signal_dim=seq_signal_dim,
        snap_signal_dim=snap_signal_dim,
        xgb_lock_path=xgb_lock_path,
        xgb_lock_file_sha256=xgb_lock_file_sha256,
        xgb_lock_feature_contract_id=xgb_lock_feature_contract_id,
        xgb_lock_feature_list_sha256=xgb_lock_feature_list_sha256,
        xgb_lock_model_sha256=xgb_lock_model_sha256,
        prebuilt_schema_manifest_path=prebuilt_schema_manifest_path,
        prebuilt_schema_manifest_sha256=prebuilt_schema_manifest_sha256,
        diagnostic_enabled=diagnostic_enabled,
        diagnostic_force_eval_sessions=diagnostic_force_eval_sessions,
        diagnostic_bypassed_gate=diagnostic_bypassed_gate,
        diagnostic_mode=diagnostic_mode,
        ritual_warmup_padding_days=ritual_warmup_padding_days,
        requested_eval_start_ts=requested_eval_start_ts,
        requested_eval_end_ts=requested_eval_end_ts,
        effective_replay_start_ts=effective_replay_start_ts,
        effective_replay_end_ts=effective_replay_end_ts,
        xgb_session_policy_hash=xgb_session_policy_hash,  # XGB session policy
        xgb_us_disabled_enabled=xgb_us_disabled_enabled,  # XGB session policy
        xgb_us_disabled_count=xgb_us_disabled_count,  # XGB session policy telemetry
        xgb_predict_count_total=0,  # Will be incremented in runtime
        xgb_predict_count_eu=0,  # Will be incremented in runtime
        xgb_predict_count_us=0,  # Will be incremented in runtime
        xgb_predict_count_overlap=0,  # Will be incremented in runtime
    )
    
    # XGB SESSION POLICY: Load policy fields before writing (hard-fail in TRUTH/SMOKE)
    is_truth_mode = os.getenv("GX1_RUN_MODE", "").upper() == "TRUTH"
    is_smoke_mode = os.getenv("GX1_SMOKE_MODE", "0") == "1" or "smoke" in str(output_dir).lower()
    truth_or_smoke = is_truth_mode or is_smoke_mode
    
    try:
        log.info(f"[RUN_IDENTITY] Calling ensure_xgb_policy_fields_loaded(truth_mode={truth_or_smoke})")
        identity.ensure_xgb_policy_fields_loaded(truth_mode=truth_or_smoke)
        log.info(f"[RUN_IDENTITY] ensure_xgb_policy_fields_loaded completed: status={identity.policy_load_status}, hash={identity.xgb_session_policy_hash[:16] if identity.xgb_session_policy_hash else None}...")
    except Exception as e:
        if truth_or_smoke:
            raise  # Re-raise in TRUTH/SMOKE mode
        log.warning(f"[RUN_IDENTITY] Failed to load XGB policy fields: {e}")
    
    # Hard-fail on write-time if TRUTH/SMOKE and policy not loaded
    if truth_or_smoke:
        if identity.policy_load_status != "OK":
            raise RuntimeError(
                f"RUN_IDENTITY_FAIL: XGB policy load status is '{identity.policy_load_status}', "
                f"expected 'OK' (error: {identity.policy_load_error})"
            )
        if not identity.xgb_session_policy_hash or len(identity.xgb_session_policy_hash) != 64:
            raise RuntimeError(
                f"RUN_IDENTITY_FAIL: Invalid xgb_session_policy_hash: "
                f"{identity.xgb_session_policy_hash} (len={len(identity.xgb_session_policy_hash) if identity.xgb_session_policy_hash else 0})"
            )
    
    # Write RUN_IDENTITY.json (atomic write)
    identity_path = output_dir / "RUN_IDENTITY.json"
    identity_dict = identity.to_dict()
    
    # Write to temp file first, then rename (atomic)
    temp_path = output_dir / "RUN_IDENTITY.json.tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(identity_dict, f, indent=2, sort_keys=True)
    temp_path.replace(identity_path)
    
    log.info(f"[RUN_IDENTITY] Written to: {identity_path}")
    log.info(f"[RUN_IDENTITY] Policy ID: {policy_id}")
    log.info(f"[RUN_IDENTITY] Policy SHA256: {policy_sha256[:16]}...")
    log.info(f"[RUN_IDENTITY] Git HEAD: {git_head_sha[:8]}... (dirty={git_dirty})")
    log.info(f"[RUN_IDENTITY] Replay mode: {replay_mode.value}")
    log.info(f"[RUN_IDENTITY] Feature build disabled: {feature_build_disabled}")
    if feature_schema_fingerprint:
        log.info(f"[RUN_IDENTITY] Feature schema fingerprint: {feature_schema_fingerprint[:16]}...")
    
    return identity


def load_run_identity(identity_path: Path) -> RunIdentity:
    """Load RUN_IDENTITY.json from file."""
    if not identity_path.exists():
        raise FileNotFoundError(f"RUN_IDENTITY not found: {identity_path}")
    
    with open(identity_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return RunIdentity(
        git_head_sha=data["git_head_sha"],
        git_dirty=data["git_dirty"],
        python_executable=data["python_executable"],
        python_version=data["python_version"],
        bundle_sha256=data.get("bundle_sha256"),
        bundle_dir_resolved=data.get("bundle_dir_resolved"),
        bundle_dir_source=data.get("bundle_dir_source"),
        bundle_dir_exists=data.get("bundle_dir_exists"),
        prebuilt_manifest_sha256=data.get("prebuilt_manifest_sha256"),
        prebuilt_manifest_path=data.get("prebuilt_manifest_path"),
        policy_id=data["policy_id"],
        policy_sha256=data["policy_sha256"],
        replay_mode=ReplayMode(data["replay_mode"]),
        prebuilt_used=data.get("prebuilt_used", False),
        feature_build_disabled=data["feature_build_disabled"],
        windows_sha=data.get("windows_sha"),
        feature_schema_fingerprint=data.get("feature_schema_fingerprint"),
        exit_v2_enabled=data.get("exit_v2_enabled", False),
        exit_v2_yaml_path=data.get("exit_v2_yaml_path"),
        exit_v2_yaml_sha256=data.get("exit_v2_yaml_sha256"),
        exit_v2_params_summary=data.get("exit_v2_params_summary"),
        exit_v2_counters=data.get("exit_v2_counters"),
        pre_entry_wait_enabled=data.get("pre_entry_wait_enabled", False),
        pre_entry_wait_counters=data.get("pre_entry_wait_counters"),
        overlap_overlay_enabled=data.get("overlap_overlay_enabled", False),
        overlap_overlay_config=data.get("overlap_overlay_config"),
        overlap_overlay_counters=data.get("overlap_overlay_counters"),
        dt_module_version=data.get("dt_module_version"),  # CRITICAL: Version stamp
        entry_model_variant_id=data.get("entry_model_variant_id"),
        transformer_layers=data.get("transformer_layers"),
        transformer_layers_baseline=data.get("transformer_layers_baseline"),
        depth_ladder_delta=data.get("depth_ladder_delta"),
        exit_monotonicity_violations=data.get("exit_monotonicity_violations", 0),  # PHASE 1 FIX
        duplicate_exit_attempts=data.get("duplicate_exit_attempts", 0),  # PHASE 1 FIX
        nan_inf_pred_count=data.get("nan_inf_pred_count", 0),  # PHASE 1 FIX
        nan_inf_pred_first_ts=data.get("nan_inf_pred_first_ts"),  # PHASE 1 FIX
        nan_inf_pred_mode_handling=data.get("nan_inf_pred_mode_handling", "FATAL_IN_REPLAY"),  # PHASE 1 FIX
        temperature_scaling_enabled=data.get("temperature_scaling_enabled", True),  # PHASE 1 FIX
        temperature_map_loaded=data.get("temperature_map_loaded", False),  # PHASE 1 FIX
        temperature_defaults_used_count=data.get("temperature_defaults_used_count", 0),  # PHASE 1 FIX
        output_mode=data.get("output_mode", "MINIMAL"),  # OUTPUT POLICY (default MINIMAL)
        replay_exit_semantics=data.get("replay_exit_semantics"),
        ritual_warmup_padding_days=data.get("ritual_warmup_padding_days", 0),
        requested_eval_start_ts=data.get("requested_eval_start_ts"),
        requested_eval_end_ts=data.get("requested_eval_end_ts"),
        effective_replay_start_ts=data.get("effective_replay_start_ts"),
        effective_replay_end_ts=data.get("effective_replay_end_ts"),
        # XGB SESSION POLICY fields (preserved from JSON)
        xgb_session_policy_hash=data.get("xgb_session_policy_hash"),
        xgb_us_disabled_enabled=data.get("xgb_us_disabled_enabled", False),
        xgb_us_disabled_count=data.get("xgb_us_disabled_count", 0),
        xgb_predict_count_total=data.get("xgb_predict_count_total", 0),
        xgb_predict_count_eu=data.get("xgb_predict_count_eu", 0),
        xgb_predict_count_us=data.get("xgb_predict_count_us", 0),
        xgb_predict_count_overlap=data.get("xgb_predict_count_overlap", 0),
        # PROVENANCE fields (preserved from JSON)
        run_identity_impl_module=data.get("run_identity_impl_module"),
        run_identity_impl_file=data.get("run_identity_impl_file"),
        gx1_package_root=data.get("gx1_package_root"),
        resolved_policy_path=data.get("resolved_policy_path"),
        policy_load_status=data.get("policy_load_status"),
        policy_load_error=data.get("policy_load_error"),
    )


if __name__ == "__main__":
    # Test identity creation
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy-id", required=True)
    parser.add_argument("--policy-sha256", required=True)
    parser.add_argument("--prebuilt-path", type=Path)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    
    try:
        identity = create_run_identity(
            output_dir=args.output_dir,
            policy_id=args.policy_id,
            policy_sha256=args.policy_sha256,
            prebuilt_path=args.prebuilt_path,
            allow_dirty=args.allow_dirty,
        )
        print(f"✅ RUN_IDENTITY created successfully:")
        print(f"  Policy ID: {identity.policy_id}")
        print(f"  Policy SHA256: {identity.policy_sha256[:16]}...")
        print(f"  Git HEAD: {identity.git_head_sha[:8]}... (dirty={identity.git_dirty})")
        print(f"  Replay mode: {identity.replay_mode.value}")
        print(f"  Feature build disabled: {identity.feature_build_disabled}")
    except Exception as e:
        print(f"❌ Failed to create RUN_IDENTITY: {e}")
        sys.exit(1)
