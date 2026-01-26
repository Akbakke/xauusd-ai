#!/usr/bin/env python3
"""
Build ENTRY_V10_CTX training dataset with calibrated XGB features.

This script builds a pre-processed dataset with:
- Base features (V9)
- HTF features (H1/H4)
- XGB inference (per session)
- XGB calibration (Platt/Isotonic)
- Uncertainty signals (entropy, uncertainty_score)
- Sequence/snapshot/context feature packing
- Metadata with calibration_applied=true

Output: Parquet file ready for training (no feature building in training loop).
"""

import argparse
import copy
import hashlib
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from joblib import load as joblib_load

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from gx1.execution.entry_context_features import build_entry_context_features
from gx1.execution.live_features import infer_session_tag
from gx1.features.runtime_v9 import build_v9_runtime_features
from gx1.features.feature_state import FeatureState
from gx1.utils.feature_context import set_feature_state, reset_feature_state
from gx1.models.entry_v10.calibration_paths import (
    get_calibrator_path,
    get_calibrator_path_hierarchy,
)
from gx1.models.entry_v10.xgb_calibration import (
    apply_xgb_calibration,
    get_regime_bucket_key,
    load_xgb_calibrators,
)
from gx1.execution.telemetry import prob_entropy

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

EXPECTED_SESSION_TAGS = ("EU", "OVERLAP", "US", "ASIA", "UNKNOWN")


def _raw_schema_signature(df: pd.DataFrame) -> Dict[str, any]:
    """
    Create a deterministic signature of the raw input schema (name + dtype).

    Returns a dict with:
      - columns: list of {name, dtype}
      - schema_hash: sha256 over "name:dtype\\n" lines
    """
    cols = [{"name": str(c), "dtype": str(df[c].dtype)} for c in df.columns]
    payload = "\n".join([f"{c['name']}:{c['dtype']}" for c in cols]).encode("utf-8")
    return {"columns": cols, "schema_hash": hashlib.sha256(payload).hexdigest()}


def compute_session_histogram(
    df: pd.DataFrame,
    ts_col: str = "ts",
    session_col: str = "session_id",
    session_override: Optional[pd.Series] = None,
) -> Dict[str, any]:
    """
    Compute robust session histogram without mutating df.

    Session source precedence:
      1) session_override if provided (string tags)
      2) session_col if present and looks like tags or small int ids (0..3)
      3) infer_session_tag(ts) from ts_col
    """
    out: Dict[str, any] = {}
    n_rows = int(len(df))
    out["n_rows"] = n_rows

    if ts_col in df.columns:
        ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        out["n_ts_na"] = int(ts.isna().sum())
        out["ts_min"] = None if ts.dropna().empty else str(ts.dropna().min())
        out["ts_max"] = None if ts.dropna().empty else str(ts.dropna().max())
    else:
        ts = None
        out["n_ts_na"] = n_rows
        out["ts_min"] = None
        out["ts_max"] = None

    unexpected_values: List[str] = []

    def normalize_tag(x: any) -> str:
        try:
            s = str(x).upper()
        except Exception:
            return "UNKNOWN"
        if s in ("EU", "OVERLAP", "US", "ASIA"):
            return s
        return "UNKNOWN"

    # Determine session tags series
    if session_override is not None:
        tags = session_override.map(normalize_tag)
    elif session_col in df.columns:
        ser = df[session_col]
        if ser.dtype == object:
            tags = ser.map(normalize_tag)
            unexpected_values = [
                str(v) for v in ser.dropna().astype(str).unique().tolist()
                if str(v).upper() not in ("EU", "OVERLAP", "US", "ASIA")
            ][:5]
        else:
            # Numeric: only treat as session IDs if it looks like small integers
            ser_num = pd.to_numeric(ser, errors="coerce")
            uniq = ser_num.dropna().unique()
            if len(uniq) > 0 and set(np.unique(uniq).astype(int).tolist()).issubset({0, 1, 2, 3}):
                mapping = {0: "EU", 1: "OVERLAP", 2: "US", 3: "ASIA"}
                tags = ser_num.map(lambda v: mapping.get(int(v), "UNKNOWN") if pd.notna(v) else "UNKNOWN")
            else:
                # Not a categorical session ID; fall back to timestamp inference
                unexpected_values = [str(v) for v in ser.dropna().unique().tolist()[:5]]
                if ts is not None:
                    tags = ts.map(lambda t: infer_session_tag(t) if pd.notna(t) else "UNKNOWN").astype(str).str.upper()
                else:
                    tags = pd.Series(["UNKNOWN"] * n_rows)
    else:
        if ts is not None:
            tags = ts.map(lambda t: infer_session_tag(t) if pd.notna(t) else "UNKNOWN").astype(str).str.upper()
        else:
            tags = pd.Series(["UNKNOWN"] * n_rows)

    counts = {k: 0 for k in EXPECTED_SESSION_TAGS}
    vc = tags.value_counts(dropna=False).to_dict()
    for k, v in vc.items():
        k2 = str(k).upper()
        if k2 in counts:
            counts[k2] = int(v)
        else:
            counts["UNKNOWN"] += int(v)

    out["counts"] = counts
    out["pct"] = {k: (counts[k] / n_rows * 100.0) if n_rows > 0 else 0.0 for k in counts}

    if unexpected_values:
        out["unexpected_session_values_top5"] = unexpected_values[:5]

    # Sanity: ensure sums match
    out["counts_sum"] = int(sum(counts.values()))

    return out


def log_session_histogram(
    df: pd.DataFrame,
    label: str,
    ts_col: str = "ts",
    session_col: str = "session_id",
    session_override: Optional[pd.Series] = None,
) -> Dict[str, any]:
    """Log and return a robust session histogram summary."""
    h = compute_session_histogram(
        df=df,
        ts_col=ts_col,
        session_col=session_col,
        session_override=session_override,
    )
    msg = (
        f"[SESSION_HIST] {label}: n_rows={h['n_rows']} n_ts_na={h['n_ts_na']} "
        f"ts_min={h['ts_min']} ts_max={h['ts_max']} "
        f"counts={h['counts']}"
    )
    if "unexpected_session_values_top5" in h:
        msg += f" unexpected_top5={h['unexpected_session_values_top5']}"
    log.info(msg)
    return h


def load_xgb_models(policy_config: Dict) -> Dict[str, any]:
    """Load XGB models from policy config."""
    xgb_cfg = policy_config.get("entry_models", {}).get("v10_ctx", {}).get("xgb", {})
    
    if not xgb_cfg.get("enabled", False):
        raise RuntimeError("XGB not enabled in policy config")
    
    models = {}
    # PHASE 2: Resolve workspace root correctly
    script_path = Path(__file__).resolve()
    workspace_root = script_path.parent.parent.parent  # gx1/scripts -> gx1 -> project_root
    
    for session, model_path_key in [
        ("EU", "eu_model_path"),
        ("US", "us_model_path"),
        ("OVERLAP", "overlap_model_path"),
        ("ASIA", "asia_model_path"),
    ]:
        model_path = xgb_cfg.get(model_path_key)
        if not model_path:
            raise RuntimeError(f"XGB model path not configured for {session}")
        
        model_path_obj = Path(model_path)
        if not model_path_obj.is_absolute():
            # Try relative to workspace root
            model_path_obj = workspace_root / model_path_obj
            # If still not found, try relative to current working directory
            if not model_path_obj.exists():
                model_path_obj = Path(model_path).resolve()
        
        if not model_path_obj.exists():
            raise RuntimeError(
                f"XGB model not found for {session}: {model_path_obj}\n"
                f"  Searched: {workspace_root / model_path if not Path(model_path).is_absolute() else model_path}\n"
                f"  Workspace root: {workspace_root}\n"
                f"  Current dir: {Path.cwd()}"
            )
        
        models[session] = joblib_load(model_path_obj)
        log.info(f"✅ Loaded XGB model for {session}: {model_path_obj}")
    
    return models


def verify_calibrators_exist(
    calibration_dir: Path,
    policy_id: str,
    method: str = "platt",
) -> None:
    """
    Verify that calibrators exist and hard fail if missing.
    
    Logs exact paths searched and policy_id/session/bucket.
    Uses SSoT get_calibrator_path() function.
    """
    require_calibration = os.getenv("GX1_REQUIRE_XGB_CALIBRATION", "0") == "1"
    
    if not require_calibration:
        log.warning("GX1_REQUIRE_XGB_CALIBRATION=0, skipping calibrator verification")
        return
    
    # PHASE 2: Resolve calibration_dir path correctly
    if not calibration_dir.is_absolute():
        script_path = Path(__file__).resolve()
        # workspace_root should be the project root (where "gx1/" directory is)
        # script is at: gx1/scripts/build_entry_v10_ctx_training_dataset.py
        # So: parent.parent.parent = project root
        workspace_root = script_path.parent.parent.parent
        calibration_dir = workspace_root / calibration_dir
    else:
        # Already absolute, but ensure it's resolved
        calibration_dir = calibration_dir.resolve()
    
    log.info(f"[CALIBRATOR_VERIFY] Checking calibration directory: {calibration_dir}")
    log.info(f"[CALIBRATOR_VERIFY] Policy ID: {policy_id}, Method: {method}")
    
    missing_sessions = []
    for session in ["EU", "US", "OVERLAP", "ASIA"]:
        # Check session-only calibrator (required)
        session_cal_path = get_calibrator_path(calibration_dir, policy_id, session, None, method)
        # PHASE 2: Ensure path is absolute and resolved
        if not session_cal_path.is_absolute():
            session_cal_path = calibration_dir / session_cal_path
        session_cal_path = session_cal_path.resolve()
        
        log.info(f"[CALIBRATOR_VERIFY] Checking: {session_cal_path}")
        
        if not session_cal_path.exists():
            missing_sessions.append(session)
            log.error(f"  ❌ Session calibrator missing: {session_cal_path}")
            # PHASE 2: Debug - check if file exists with different path
            alt_path = calibration_dir / policy_id / session / f"calibrator_{method}.joblib"
            if alt_path.exists():
                log.warning(f"  ⚠️  Found at alternative path: {alt_path}")
        else:
            log.info(f"  ✅ Session calibrator found: {session_cal_path}")
        
        # Check for per-regime calibrators (optional, but log if found)
        policy_dir = calibration_dir / policy_id
        session_dir = policy_dir / session
        if session_dir.exists():
            for regime_dir in session_dir.iterdir():
                if not regime_dir.is_dir() or regime_dir.name.startswith("."):
                    continue
                
                bucket_key = regime_dir.name
                regime_cal_path = get_calibrator_path(calibration_dir, policy_id, session, bucket_key, method)
                if regime_cal_path.exists():
                    log.info(f"  ✅ Regime calibrator found: {regime_cal_path} (bucket: {bucket_key})")
    
    if missing_sessions:
        raise RuntimeError(
            f"CALIBRATOR_MISSING: Calibrators missing for sessions: {missing_sessions}\n"
            f"Expected paths (using SSoT get_calibrator_path()):\n"
            f"  - {get_calibrator_path(calibration_dir, policy_id, 'EU', None, method)} (session-only)\n"
            f"  - {get_calibrator_path(calibration_dir, policy_id, 'EU', 'EU_LOW', method)} (per-regime example)\n"
            f"Policy ID: {policy_id}\n"
            f"Method: {method}\n"
            f"Calibration directory: {calibration_dir}\n"
            f"Set GX1_REQUIRE_XGB_CALIBRATION=0 to allow uncalibrated XGB (not recommended for training)."
        )
    
    log.info(f"✅ All calibrators verified for policy_id={policy_id}, method={method}")


# PHASE 2: Calibrator usage stats tracking
_calibrator_usage_stats = {
    "session+bucket": {},
    "session-only": {},
    "raw": {},
}


def track_calibrator_usage(
    session: str,
    bucket_key: Optional[str],
    tier: str,
):
    """Track which calibrator tier was used."""
    if tier not in _calibrator_usage_stats:
        _calibrator_usage_stats[tier] = {}
    
    if session not in _calibrator_usage_stats[tier]:
        _calibrator_usage_stats[tier][session] = {}
    
    if bucket_key:
        if bucket_key not in _calibrator_usage_stats[tier][session]:
            _calibrator_usage_stats[tier][session][bucket_key] = 0
        _calibrator_usage_stats[tier][session][bucket_key] += 1
    else:
        if "total" not in _calibrator_usage_stats[tier][session]:
            _calibrator_usage_stats[tier][session]["total"] = 0
        _calibrator_usage_stats[tier][session]["total"] += 1


def get_calibrator_usage_stats() -> Dict:
    """Get calibrator usage statistics."""
    return _calibrator_usage_stats.copy()


def reset_calibrator_usage_stats():
    """Reset usage statistics."""
    global _calibrator_usage_stats
    _calibrator_usage_stats = {
        "session+bucket": {},
        "session-only": {},
        "raw": {},
    }


def build_xgb_features_for_row(
    row: pd.Series,
    xgb_models: Dict[str, any],
    calibrators: Dict[str, Dict[str, any]],
    snap_feature_names: List[str],
    calibration_dir: Path,
    policy_id: str,
    method: str = "platt",
) -> Tuple[float, float, float, float]:
    """
    Build XGB features for a single row.
    
    Returns:
        (p_cal, margin, p_hat, uncertainty_score)
    """
    # Session inference SSoT: infer from timestamp using infer_session_tag(ts),
    # matching XGB training / calibrator training / runtime.
    ts = None
    if "ts" in row.index:
        ts = pd.to_datetime(row["ts"], utc=True, errors="coerce")
    if (ts is None or pd.isna(ts)) and isinstance(row.name, pd.Timestamp):
        ts = row.name
    if ts is None or pd.isna(ts):
        raise RuntimeError("DATASET_BUILD_FAILED: Missing/invalid timestamp for session inference")
    session = infer_session_tag(ts).upper()
    
    # Get XGB model
    xgb_model = xgb_models.get(session)
    if xgb_model is None:
        raise RuntimeError(f"XGB model not found for session: {session}")
    
    # Prepare XGB features (snapshot features only)
    xgb_features = row[snap_feature_names].values.reshape(1, -1).astype(np.float32)
    xgb_features = np.nan_to_num(xgb_features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # XGB prediction
    try:
        xgb_proba = xgb_model.predict_proba(xgb_features)[0]
        
        # Extract p_long
        if hasattr(xgb_model, 'classes_') and len(xgb_model.classes_) == 2:
            if xgb_model.classes_[0] == 0:  # [prob_0, prob_1] where 0=short, 1=long
                p_long_raw = float(xgb_proba[1])
            else:
                p_long_raw = float(xgb_proba[0])
        else:
            p_long_raw = float(xgb_proba[1] if len(xgb_proba) > 1 else xgb_proba[0])
    except Exception as e:
        raise RuntimeError(f"XGB prediction failed for session={session}: {e}")
    
    # Get vol_regime_id
    vol_regime_id = 2  # Default to MEDIUM
    if "_v1_atr_regime_id" in row.index:
        vol_regime_id = int(row["_v1_atr_regime_id"])
    
    # Get regime bucket key
    bucket_key = get_regime_bucket_key(session, vol_regime_id)
    
    # PHASE 2: Try calibrator hierarchy (session+bucket -> session-only)
    calibrator_paths = get_calibrator_path_hierarchy(
        calibration_dir, policy_id, session, bucket_key, method
    )
    
    calibrator_used = None
    tier_used = None
    
    for cal_path, tier in calibrator_paths:
        if cal_path.exists():
            calibrator_used = cal_path
            tier_used = tier
            # PHASE 2: Track usage
            track_calibrator_usage(session, bucket_key if tier_used == "session+bucket" else None, tier_used)
            break
    
    # PHASE 2: If no calibrator found
    if tier_used is None:
        # No calibrator found - this should have been caught by verify_calibrators_exist
        require_cal = os.getenv("GX1_REQUIRE_XGB_CALIBRATION", "0") == "1"
        if require_cal:
            raise RuntimeError(
                f"CALIBRATOR_MISSING: No calibrator found for session={session}, bucket={bucket_key}. "
                f"Searched paths: {[str(p) for p, _ in calibrator_paths]}"
            )
        else:
            # Only log once per session/bucket combination to reduce spam
            if not hasattr(build_xgb_features_for_row, '_warned_raw'):
                build_xgb_features_for_row._warned_raw = set()
            warn_key = (session, bucket_key)
            if warn_key not in build_xgb_features_for_row._warned_raw:
                log.warning(f"No calibrator found for session={session}, bucket={bucket_key}, using raw (will suppress further warnings)")
                build_xgb_features_for_row._warned_raw.add(warn_key)
            tier_used = "raw"
            track_calibrator_usage(session, bucket_key, "raw")
    
    # Apply calibration (with SSoT path logging)
    calibrated_output = apply_xgb_calibration(
        p_raw=p_long_raw,
        session=session,
        vol_regime_id=vol_regime_id,
        calibrators=calibrators,
        method=method,
        calibration_dir=calibration_dir,
        policy_id=policy_id,
    )
    
    return (
        calibrated_output.p_cal,
        calibrated_output.margin,
        calibrated_output.p_hat,
        calibrated_output.uncertainty_score,
    )


def validate_built_dataset(
    df: pd.DataFrame,
    seq_feature_names: List[str],
    snap_feature_names: List[str],
    seq_len: int = 30,
    check_shapes: bool = False,  # PHASE 2: Can't check tensor shapes from DataFrame, skip for now
) -> Dict:
    """
    Validate built dataset (hard fail on errors, warnings on soft issues).
    
    Returns:
        Dict with validation statistics
    """
    log.info("[DATASET_VALIDATE] Starting dataset validation...")
    
    stats = {
        "n_rows": len(df),
        "p_cal_stats": {},
        "uncertainty_score_stats": {},
        "warnings": [],
    }
    
    # Hard fail: calibration_applied must be true
    if not df.attrs.get("calibration_applied", False):
        raise RuntimeError(
            "DATASET_VALIDATE_FAILED: calibration_applied != true. "
            "Expected prebuilt dataset with calibrated XGB features."
        )
    log.info("  ✅ calibration_applied = true")
    
    # Hard fail: Check for NaN/inf in critical columns
    critical_cols = seq_feature_names + snap_feature_names + ["p_cal", "margin", "p_hat", "uncertainty_score"]
    for col in critical_cols:
        if col not in df.columns:
            continue
        
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            raise RuntimeError(f"DATASET_VALIDATE_FAILED: Column '{col}' has {nan_count} NaN values")
        
        if df[col].dtype in [np.float32, np.float64, np.int32, np.int64]:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                raise RuntimeError(f"DATASET_VALIDATE_FAILED: Column '{col}' has {inf_count} inf values")
    
    log.info("  ✅ No NaN/inf in critical columns")
    
    # Hard fail: Range sanity
    if "p_cal" in df.columns:
        p_cal_min = df["p_cal"].min()
        p_cal_max = df["p_cal"].max()
        if p_cal_min < 0.0 or p_cal_max > 1.0:
            raise RuntimeError(
                f"DATASET_VALIDATE_FAILED: p_cal out of range [0, 1]: min={p_cal_min}, max={p_cal_max}"
            )
        stats["p_cal_stats"] = {
            "mean": float(df["p_cal"].mean()),
            "p5": float(df["p_cal"].quantile(0.05)),
            "p95": float(df["p_cal"].quantile(0.95)),
        }
        log.info(f"  ✅ p_cal range: [{p_cal_min:.4f}, {p_cal_max:.4f}]")
    
    if "p_hat" in df.columns:
        p_hat_min = df["p_hat"].min()
        p_hat_max = df["p_hat"].max()
        if p_hat_min < 0.0 or p_hat_max > 1.0:
            raise RuntimeError(
                f"DATASET_VALIDATE_FAILED: p_hat out of range [0, 1]: min={p_hat_min}, max={p_hat_max}"
            )
    
    if "uncertainty_score" in df.columns:
        u_min = df["uncertainty_score"].min()
        u_max = df["uncertainty_score"].max()
        if u_min < 0.0 or u_max > 1.0:
            raise RuntimeError(
                f"DATASET_VALIDATE_FAILED: uncertainty_score out of range [0, 1]: min={u_min}, max={u_max}"
            )
        stats["uncertainty_score_stats"] = {
            "mean": float(df["uncertainty_score"].mean()),
            "p5": float(df["uncertainty_score"].quantile(0.05)),
            "p95": float(df["uncertainty_score"].quantile(0.95)),
        }
        log.info(f"  ✅ uncertainty_score range: [{u_min:.4f}, {u_max:.4f}]")
    
    # Soft warning: p_cal vs p_raw (if p_raw exists)
    if "p_raw" in df.columns and "p_cal" in df.columns:
        corr = df["p_raw"].corr(df["p_cal"])
        mean_abs_diff = (df["p_cal"] - df["p_raw"]).abs().mean()
        
        if corr > 0.999 and mean_abs_diff < 1e-6:
            warning = (
                f"WARNING: p_cal and p_raw are nearly identical "
                f"(corr={corr:.6f}, mean_abs_diff={mean_abs_diff:.6e}). "
                f"Calibration may not have been applied."
            )
            stats["warnings"].append(warning)
            log.warning(f"  ⚠️  {warning}")
        else:
            log.info(f"  ✅ p_cal vs p_raw: corr={corr:.4f}, mean_abs_diff={mean_abs_diff:.6e}")
    
    # PHASE 2: Print statistics block
    print("\n[DATASET_VALIDATE STATISTICS]")
    if "p_cal" in df.columns:
        print(f"  p_cal: mean={stats['p_cal_stats']['mean']:.4f}, "
              f"p5={stats['p_cal_stats']['p5']:.4f}, p95={stats['p_cal_stats']['p95']:.4f}")
    if "uncertainty_score" in df.columns:
        print(f"  uncertainty_score: mean={stats['uncertainty_score_stats']['mean']:.4f}, "
              f"p5={stats['uncertainty_score_stats']['p5']:.4f}, p95={stats['uncertainty_score_stats']['p95']:.4f}")
    if stats["warnings"]:
        for warning in stats["warnings"]:
            print(f"  ⚠️  {warning}")
    
    log.info("[DATASET_VALIDATE] Validation complete")
    return stats


def build_dataset(
    df_raw: pd.DataFrame,
    policy_config: Dict,
    feature_meta_path: Path,
    seq_scaler_path: Optional[Path] = None,
    snap_scaler_path: Optional[Path] = None,
    calibration_dir: Path = Path("models/xgb_calibration"),
    calibration_method: str = "platt",
) -> Tuple[pd.DataFrame, Dict]:
    """
    Build complete dataset with calibrated XGB features.
    
    Returns:
        DataFrame with all features + XGB channels + metadata
    """
    log.info(f"[DATASET_BUILD] Starting dataset build for {len(df_raw)} rows")
    
    # Ensure ts is datetime for session inference
    if "ts" in df_raw.columns:
        df_raw["ts"] = pd.to_datetime(df_raw["ts"], utc=True)
        # Filter out rows with NaN timestamps (cannot infer session)
        n_nan_ts = df_raw["ts"].isna().sum()
        if n_nan_ts > 0:
            df_raw = df_raw[df_raw["ts"].notna()].copy()
            log.info(f"[DATASET_BUILD] Filtered out {n_nan_ts} rows with NaN timestamps")
    
    # Load XGB models
    xgb_models = load_xgb_models(policy_config)
    
    # Verify calibrators exist
    policy_id = policy_config.get("policy_name", "GX1_SNIPER_TRAIN_V10_CTX_GATED")
    verify_calibrators_exist(calibration_dir, policy_id, calibration_method)
    
    # PHASE 2: Resolve calibration_dir path correctly
    if not calibration_dir.is_absolute():
        script_path = Path(__file__).resolve()
        workspace_root = script_path.parent.parent.parent  # gx1/scripts -> gx1 -> project_root
        calibration_dir = workspace_root / calibration_dir
    
    # Load calibrators
    calibrators = load_xgb_calibrators(
        calibration_dir=calibration_dir,
        policy_id=policy_id,
        method=calibration_method,
    )
    
    require_calibration = os.getenv("GX1_REQUIRE_XGB_CALIBRATION", "0") == "1"
    
    if not calibrators:
        if require_calibration:
            raise RuntimeError(
                f"No calibrators loaded for policy_id={policy_id}. "
                f"Set GX1_REQUIRE_XGB_CALIBRATION=0 to allow uncalibrated XGB (not recommended for training)."
            )
        else:
            log.warning(
                f"No calibrators loaded for policy_id={policy_id}. "
                f"Will use raw XGB probabilities (uncalibrated)."
            )
    else:
        log.info(f"✅ Loaded calibrators for {len(calibrators)} sessions")
    
    # Session histogram (RAW_INPUT) - do not mutate df_raw
    session_histograms: Dict[str, Dict[str, any]] = {}
    session_histograms["RAW_INPUT"] = log_session_histogram(df_raw, label="RAW_INPUT", ts_col="ts", session_col="session_id")
    # Session histogram (POST_INFER) - SSoT = timestamp inference
    ts_series = pd.to_datetime(df_raw["ts"], utc=True, errors="coerce") if "ts" in df_raw.columns else pd.Series([pd.NaT] * len(df_raw))
    inferred_session = ts_series.map(lambda t: infer_session_tag(t) if pd.notna(t) else "UNKNOWN").astype(str).str.upper()
    session_histograms["POST_INFER"] = log_session_histogram(
        df_raw,
        label="POST_INFER",
        ts_col="ts",
        session_col="session_id",
        session_override=inferred_session,
    )

    # Build V9 features
    log.info("[DATASET_BUILD] Building V9 runtime features...")
    
    # Preserve label columns from raw data (they will be removed by build_v9_runtime_features as "leakage").
    # IMPORTANT: build_v9_runtime_features changes index to DatetimeIndex, so preserve by POSITION.
    label_cols_to_preserve = {}
    for col in ["mfe_bps", "MFE_bps", "mae_bps", "MAE_bps", "first_hit"]:
        if col in df_raw.columns:
            label_cols_to_preserve[col] = df_raw[col].to_numpy(copy=True)
    
    # Preserve raw columns needed for optional context feature building in training.
    # CONTRACT: Always preserve minimal raw columns (OHLC, ts, labels, spread) even if ctx is prebuilt.
    # IMPORTANT: build_v9_runtime_features changes index to DatetimeIndex, so preserve by POSITION.
    raw_cols_to_preserve = {}
    raw_cols_list = []
    
    # Required raw columns for context building
    required_raw_cols = ["open", "high", "low", "close", "ts"]
    # Optional raw columns (if available)
    optional_raw_cols = ["volume", "spread_bps", "spread", "bid_close", "ask_close"]
    
    for col in required_raw_cols + optional_raw_cols:
        if col in df_raw.columns:
            raw_cols_to_preserve[col] = df_raw[col].to_numpy(copy=True)
            raw_cols_list.append(col)
    
    # Also preserve session_id if it exists (may be used for context)
    if "session_id" in df_raw.columns:
        raw_cols_to_preserve["session_id"] = df_raw["session_id"].to_numpy(copy=True)
        raw_cols_list.append("session_id")
    
    log.info(f"[DATASET_BUILD] Preserving {len(raw_cols_list)} raw columns: {raw_cols_list}")
    if len(raw_cols_to_preserve) != len(raw_cols_list):
        raise RuntimeError(
            f"RAW_RESTORE_MISMATCH: raw_cols_to_preserve and raw_cols_list length mismatch: "
            f"{len(raw_cols_to_preserve)} != {len(raw_cols_list)}"
        )
    
    # Set FEATURE_STATE context for HTF features
    feature_state = FeatureState()
    feature_token = set_feature_state(feature_state)
    try:
        df_features, seq_feature_names, snap_feature_names = build_v9_runtime_features(
            df_raw=df_raw,
            feature_meta_path=feature_meta_path,
            seq_scaler_path=seq_scaler_path,
            snap_scaler_path=snap_scaler_path,
        )
    finally:
        reset_feature_state(feature_token)
    
    # Ensure raw timestamp column exists and is authoritative (SSoT = DatetimeIndex after build_v9_runtime_features)
    df_features["ts"] = df_features.index

    # Restore label columns by position (must match row order)
    for col, values in label_cols_to_preserve.items():
        if col not in df_features.columns:
            if len(values) != len(df_features):
                raise RuntimeError(
                    f"DATASET_BUILD_FAILED: Cannot restore label column '{col}': "
                    f"len(values)={len(values)} != len(df_features)={len(df_features)}"
                )
            df_features[col] = values
    
    # Restore raw columns by position (must match row order)
    for col, values in raw_cols_to_preserve.items():
        if col not in df_features.columns:
            if len(values) != len(df_features):
                raise RuntimeError(
                    f"DATASET_BUILD_FAILED: Cannot restore raw column '{col}': "
                    f"len(values)={len(values)} != len(df_features)={len(df_features)}"
                )
            df_features[col] = values

    # Restore safety asserts (fail-fast): ts not NaT, OHLC not NaN, ts monotonic increasing.
    try:
        ts_restored = pd.to_datetime(df_features["ts"], utc=True, errors="coerce")
        if int(ts_restored.isna().sum()) != 0:
            raise RuntimeError(f"ts has NaT count={int(ts_restored.isna().sum())}")
        for col in ["open", "high", "low", "close"]:
            if col in df_features.columns:
                n_nan = int(pd.to_numeric(df_features[col], errors="coerce").isna().sum())
                if n_nan != 0:
                    raise RuntimeError(f"{col} has NaN count={n_nan}")
        if not ts_restored.is_monotonic_increasing:
            raise RuntimeError("ts is not monotonic increasing")
    except Exception as e:
        cols_preview = list(df_features.columns)
        head_preview = df_features.head(5).to_dict(orient="list")
        raise RuntimeError(
            "RAW_RESTORE_MISMATCH: Post-restore invariants failed: "
            f"{e}. columns={cols_preview}. head={head_preview}"
        )
    
    # Sanity check: assert preserved raw columns exist in output
    missing_raw_cols = [col for col in raw_cols_list if col not in df_features.columns]
    if missing_raw_cols:
        raise RuntimeError(
            f"DATASET_BUILD_FAILED: Preserved raw columns missing in output: {missing_raw_cols}. "
            f"This indicates a bug in column preservation logic."
        )
    log.info(f"[DATASET_BUILD] ✅ Verified {len(raw_cols_list)} preserved raw columns in output")
    
    log.info(f"[DATASET_BUILD] Built {len(seq_feature_names)} seq features, {len(snap_feature_names)} snap features")
    
    # PHASE 2: Reset usage stats
    reset_calibrator_usage_stats()
    
    # Build XGB features for each row
    log.info("[DATASET_BUILD] Building XGB features with calibration...")
    xgb_cols = ["p_cal", "margin", "p_hat", "uncertainty_score"]
    
    for col in xgb_cols:
        df_features[col] = 0.0
    
    for idx, row in df_features.iterrows():
        try:
            p_cal, margin, p_hat, uncertainty_score = build_xgb_features_for_row(
                row=row,
                xgb_models=xgb_models,
                calibrators=calibrators,
                snap_feature_names=snap_feature_names,
                calibration_dir=calibration_dir,
                policy_id=policy_id,
                method=calibration_method,
            )
            
            df_features.at[idx, "p_cal"] = p_cal
            df_features.at[idx, "margin"] = margin
            df_features.at[idx, "p_hat"] = p_hat
            df_features.at[idx, "uncertainty_score"] = uncertainty_score
        except Exception as e:
            log.error(f"[DATASET_BUILD] Failed to build XGB features for row {idx}: {e}")
            raise
    
    log.info(f"[DATASET_BUILD] Built XGB features for {len(df_features)} rows")
    
    # Set metadata
    df_features.attrs["calibration_applied"] = True
    df_features.attrs["calibration_method"] = calibration_method
    df_features.attrs["policy_id"] = policy_id
    df_features.attrs["seq_features"] = seq_feature_names
    df_features.attrs["snap_features"] = snap_feature_names
    df_features.attrs["raw_columns_preserved"] = raw_cols_list  # Document preserved raw columns
    
    # PHASE 2: Generate labels if not present
    if "y_direction" not in df_features.columns or "y_early_move" not in df_features.columns or "y_quality_score" not in df_features.columns:
        log.info("[DATASET_BUILD] Labels not found, generating labels...")
        from gx1.models.entry_v9.entry_v9_labeler import generate_entry_v9_labels
        
        # Map column names: data may have lowercase (mfe_bps, mae_bps) or uppercase (MFE_bps, MAE_bps)
        mfe_col = "MFE_bps" if "MFE_bps" in df_features.columns else ("mfe_bps" if "mfe_bps" in df_features.columns else None)
        mae_col = "MAE_bps" if "MAE_bps" in df_features.columns else ("mae_bps" if "mae_bps" in df_features.columns else None)
        first_hit_col = "first_hit" if "first_hit" in df_features.columns else None
        
        if mfe_col is None or mae_col is None:
            raise RuntimeError(
                f"Missing required columns for label generation. "
                f"Expected MFE_bps/mfe_bps and MAE_bps/mae_bps. "
                f"Found columns: {sorted([c for c in df_features.columns if 'MFE' in c or 'MAE' in c or 'mfe' in c.lower() or 'mae' in c.lower()])}"
            )
        
        df_features = generate_entry_v9_labels(
            df_features,
            mfe_col=mfe_col,
            mae_col=mae_col,
            first_hit_col=first_hit_col,
        )
        log.info(f"[DATASET_BUILD] Generated labels: {len(df_features)} rows")
    else:
        log.info(f"[DATASET_BUILD] Labels found in dataset: {len(df_features)} rows")
    
    # PHASE 2: Validate dataset
    validation_stats = validate_built_dataset(df_features, seq_feature_names, snap_feature_names, seq_len=30)
    
    # PHASE 2: Get usage stats
    usage_stats = get_calibrator_usage_stats()
    
    # Get raw columns preserved from attrs
    raw_cols_preserved = df_features.attrs.get("raw_columns_preserved", raw_cols_list)
    
    return df_features, {
        "validation_stats": validation_stats,
        "calibrator_usage_stats": usage_stats,
        "raw_columns_preserved": raw_cols_preserved,
        "raw_restore_method": "by_position",
        "raw_input_schema_signature": _raw_schema_signature(df_raw),
        "session_histograms": session_histograms,
    }


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=project_root,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def write_manifest(
    output_path: Path,
    input_data_path: str,
    build_command: List[str],
    policy_config_path: str,
    calibration_dir: str,
    calibration_method: str,
    feature_meta_path: str,
    seq_scaler_path: Optional[str],
    snap_scaler_path: Optional[str],
    policy_id: str,
    seq_feature_names: List[str],
    snap_feature_names: List[str],
    seq_len: int,
    feature_contract_hash: Optional[str],
    splits: Optional[Dict],
    calibrator_usage_stats: Dict,
    xgb_model_paths: Dict[str, str],
    validation_stats: Dict,
    raw_columns_preserved: Optional[List[str]] = None,
    raw_restore_method: str = "",
    raw_input_schema_signature: Optional[Dict[str, any]] = None,
    session_histograms: Optional[Dict[str, Dict[str, any]]] = None,
    ts_min_max_by_split: Optional[Dict[str, Dict[str, Optional[str]]]] = None,
    notes: str = "",
) -> Path:
    """Write manifest file for prebuilt dataset."""
    manifest = {
        "created_at": datetime.now().isoformat(),
        "git_commit": get_git_commit(),
        "input_data_path": input_data_path,
        "output_data_path": str(output_path),
        "build_command": build_command,
        "policy_config_path": policy_config_path,
        "calibration_dir": calibration_dir,
        "calibration_method": calibration_method,
        "feature_meta_path": feature_meta_path,
        "seq_scaler_path": seq_scaler_path,
        "snap_scaler_path": snap_scaler_path,
        "feature_contract": {
            "seq_len": seq_len,
            "seq_dim": len(seq_feature_names) + 3,  # Base + 3 XGB channels
            "snap_dim": len(snap_feature_names) + 3,  # Base + 3 XGB channels
            "ctx_cat_dim": 5,
            "ctx_cont_dim": 2,
            "feature_contract_hash": feature_contract_hash,
        },
        "raw_columns_preserved": raw_columns_preserved or [],  # Document preserved raw columns
        "raw_restore_method": raw_restore_method,
        "raw_input_schema_signature": raw_input_schema_signature or {},
        "splits": splits,
        "ts_min_max_by_split": ts_min_max_by_split or {},
        "session_histograms": session_histograms or {},
        "calibrator_usage_stats": calibrator_usage_stats,
        "xgb_model_paths": xgb_model_paths,
        "validation_stats": validation_stats,
        "notes": notes,
    }
    
    manifest_path = output_path.parent / f"{output_path.stem}.manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    log.info(f"MANIFEST WRITTEN: {manifest_path}")
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description="Build ENTRY_V10_CTX training dataset")
    parser.add_argument("--data", type=str, required=True, help="Input data file (parquet)")
    parser.add_argument("--output", type=str, required=True, help="Output dataset file (parquet)")
    parser.add_argument("--policy_config", type=str, required=True, help="Policy config YAML/JSON")
    parser.add_argument("--feature_meta_path", type=str, required=True, help="Feature metadata JSON")
    parser.add_argument("--seq_scaler_path", type=str, default=None, help="Sequence scaler path")
    parser.add_argument("--snap_scaler_path", type=str, default=None, help="Snapshot scaler path")
    parser.add_argument("--calibration_dir", type=str, default="models/xgb_calibration", help="Calibration directory")
    parser.add_argument("--calibration_method", type=str, default="platt", choices=["platt", "isotonic"], help="Calibration method")
    parser.add_argument("--time_split", action="store_true", help="Apply time-based split (train/val/test)")
    # PHASE 2: Mini-build and time filtering
    parser.add_argument("--start", type=str, default=None, help="Start date/time (ISO format, e.g., 2025-01-01T00:00:00Z)")
    parser.add_argument("--end", type=str, default=None, help="End date/time (ISO format, e.g., 2025-01-31T23:59:59Z)")
    parser.add_argument("--max_rows", type=int, default=None, help="Maximum number of rows to process (deterministic, first N)")
    parser.add_argument("--dry_run", action="store_true", help="Dry run: only parse config and check calibrators, don't build")
    
    args = parser.parse_args()
    
    # Store build command for manifest
    build_command = sys.argv.copy()
    
    # Load policy config
    policy_config_path = Path(args.policy_config)
    if policy_config_path.suffix in [".yaml", ".yml"]:
        import yaml
        with open(policy_config_path, "r") as f:
            policy_config = yaml.safe_load(f)
    else:
        with open(policy_config_path, "r") as f:
            policy_config = json.load(f)
    
    policy_id = policy_config.get("policy_name", "GX1_SNIPER_TRAIN_V10_CTX_GATED")
    
    # PHASE 2: Dry run - only verify calibrators
    if args.dry_run:
        log.info("[DRY_RUN] Dry run mode: only verifying configuration...")
        calibration_dir = Path(args.calibration_dir)
        verify_calibrators_exist(calibration_dir, policy_id, args.calibration_method)
        
        # Also verify XGB models
        try:
            xgb_models = load_xgb_models(policy_config)
            log.info(f"[DRY_RUN] ✅ XGB models verified ({len(xgb_models)} sessions)")
        except Exception as e:
            log.error(f"[DRY_RUN] ❌ XGB models verification failed: {e}")
            raise
        
        log.info("[DRY_RUN] ✅ Configuration verified, exiting")
        return
    
    # Load raw data
    log.info(f"[DATASET_BUILD] Loading data from {args.data}")
    df_raw = pd.read_parquet(args.data)
    log.info(f"[DATASET_BUILD] Loaded {len(df_raw)} rows")
    
    # PHASE 2: Filter by time range (before feature building)
    if args.start or args.end:
        if "ts" not in df_raw.columns:
            raise RuntimeError("--start/--end requires 'ts' column in input data")
        
        df_raw["ts"] = pd.to_datetime(df_raw["ts"], utc=True)
        
        if args.start:
            start_ts = pd.Timestamp(args.start, tz="UTC")
            df_raw = df_raw[df_raw["ts"] >= start_ts].copy()
            log.info(f"[DATASET_BUILD] Filtered by start: {start_ts} ({len(df_raw)} rows remaining)")
        
        if args.end:
            end_ts = pd.Timestamp(args.end, tz="UTC")
            df_raw = df_raw[df_raw["ts"] <= end_ts].copy()
            log.info(f"[DATASET_BUILD] Filtered by end: {end_ts} ({len(df_raw)} rows remaining)")
    
    # PHASE 2: Limit rows (deterministic, first N)
    if args.max_rows:
        if len(df_raw) > args.max_rows:
            df_raw = df_raw.head(args.max_rows).copy()
            log.info(f"[DATASET_BUILD] Limited to {args.max_rows} rows (deterministic, first N)")
    
    # Get XGB model paths for manifest
    xgb_cfg = policy_config.get("entry_models", {}).get("v10_ctx", {}).get("xgb", {})
    workspace_root = Path(__file__).parent.parent.parent.parent
    xgb_model_paths = {}
    for session, model_path_key in [("EU", "eu_model_path"), ("US", "us_model_path"), ("OVERLAP", "overlap_model_path"), ("ASIA", "asia_model_path")]:
        model_path = xgb_cfg.get(model_path_key)
        if model_path:
            model_path_obj = Path(model_path)
            if not model_path_obj.is_absolute():
                model_path_obj = workspace_root / model_path_obj
            xgb_model_paths[session] = str(model_path_obj)
    
    # Time split if requested
    if args.time_split and "ts" in df_raw.columns:
        df_raw["ts"] = pd.to_datetime(df_raw["ts"], utc=True)
        
        train_start = pd.Timestamp("2025-01-01", tz="UTC")
        train_end = pd.Timestamp("2025-09-30 23:59:59", tz="UTC")
        val_start = pd.Timestamp("2025-10-01", tz="UTC")
        val_end = pd.Timestamp("2025-11-30 23:59:59", tz="UTC")
        test_start = pd.Timestamp("2025-12-01", tz="UTC")
        test_end = pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
        
        df_train = df_raw[(df_raw["ts"] >= train_start) & (df_raw["ts"] <= train_end)].copy()
        df_val = df_raw[(df_raw["ts"] >= val_start) & (df_raw["ts"] <= val_end)].copy()
        df_test = df_raw[(df_raw["ts"] >= test_start) & (df_raw["ts"] <= test_end)].copy()
        
        log.info(f"[DATASET_BUILD] Time split: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")

        # Session histograms (SSoT = timestamp inference; do not mutate the dfs)
        session_histograms_main: Dict[str, Dict[str, any]] = {}
        session_histograms_main["RAW_INPUT"] = log_session_histogram(df_raw, label="RAW_INPUT", ts_col="ts", session_col="session_id")
        ts_all = pd.to_datetime(df_raw["ts"], utc=True, errors="coerce")
        inferred_all = ts_all.map(lambda t: infer_session_tag(t) if pd.notna(t) else "UNKNOWN").astype(str).str.upper()
        session_histograms_main["POST_INFER"] = log_session_histogram(
            df_raw,
            label="POST_INFER",
            ts_col="ts",
            session_col="session_id",
            session_override=inferred_all,
        )
        session_histograms_main["TRAIN_SPLIT"] = log_session_histogram(
            df_train,
            label="TRAIN_SPLIT",
            ts_col="ts",
            session_override=pd.to_datetime(df_train["ts"], utc=True, errors="coerce").map(lambda t: infer_session_tag(t) if pd.notna(t) else "UNKNOWN").astype(str).str.upper()
        )
        session_histograms_main["VAL_SPLIT"] = log_session_histogram(
            df_val,
            label="VAL_SPLIT",
            ts_col="ts",
            session_override=pd.to_datetime(df_val["ts"], utc=True, errors="coerce").map(lambda t: infer_session_tag(t) if pd.notna(t) else "UNKNOWN").astype(str).str.upper()
        )
        session_histograms_main["TEST_SPLIT"] = log_session_histogram(
            df_test,
            label="TEST_SPLIT",
            ts_col="ts",
            session_override=pd.to_datetime(df_test["ts"], utc=True, errors="coerce").map(lambda t: infer_session_tag(t) if pd.notna(t) else "UNKNOWN").astype(str).str.upper()
        )

        def _split_min_max(d: pd.DataFrame) -> Dict[str, Optional[str]]:
            if "ts" not in d.columns:
                return {"ts_min": None, "ts_max": None}
            t = pd.to_datetime(d["ts"], utc=True, errors="coerce").dropna()
            return {"ts_min": None if t.empty else str(t.min()), "ts_max": None if t.empty else str(t.max())}

        ts_min_max_by_split = {
            "RAW_INPUT": _split_min_max(df_raw),
            "TRAIN_SPLIT": _split_min_max(df_train),
            "VAL_SPLIT": _split_min_max(df_val),
            "TEST_SPLIT": _split_min_max(df_test),
        }

        raw_schema_sig = _raw_schema_signature(df_raw)
        
        # Build datasets
        output_path = Path(args.output)
        output_dir = output_path.parent
        output_stem = output_path.stem
        
        # PHASE 2: Create output directory
        output_path = Path(args.output)
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Train
        log.info("[DATASET_BUILD] Building train dataset...")
        df_train_built, train_meta = build_dataset(
            df_raw=df_train,
            policy_config=policy_config,
            feature_meta_path=Path(args.feature_meta_path),
            seq_scaler_path=Path(args.seq_scaler_path) if args.seq_scaler_path else None,
            snap_scaler_path=Path(args.snap_scaler_path) if args.snap_scaler_path else None,
            calibration_dir=Path(args.calibration_dir),
            calibration_method=args.calibration_method,
        )
        # PHASE 2: Capture stats snapshot immediately after build (before any reset)
        train_usage_stats = copy.deepcopy(train_meta.get("calibrator_usage_stats", get_calibrator_usage_stats()))
        
        train_output = output_dir / f"{output_stem}_train.parquet"
        train_output.parent.mkdir(parents=True, exist_ok=True)
        df_train_built.to_parquet(train_output, index=False)
        log.info(f"✅ Saved train dataset: {train_output}")
        
        # Write manifest for train
        seq_feat_names = df_train_built.attrs.get("seq_features", [])
        snap_feat_names = df_train_built.attrs.get("snap_features", [])
        train_raw_cols = train_meta.get("raw_columns_preserved", [])
        write_manifest(
            output_path=train_output,
            input_data_path=args.data,
            build_command=build_command,
            policy_config_path=str(policy_config_path),
            calibration_dir=args.calibration_dir,
            calibration_method=args.calibration_method,
            feature_meta_path=args.feature_meta_path,
            seq_scaler_path=args.seq_scaler_path,
            snap_scaler_path=args.snap_scaler_path,
            policy_id=policy_id,
            seq_feature_names=seq_feat_names,
            snap_feature_names=snap_feat_names,
            seq_len=30,
            feature_contract_hash=None,  # TODO: compute if available
            splits={
                "train": {"start": str(train_start), "end": str(train_end), "count": len(df_train)},
                "val": {"start": str(val_start), "end": str(val_end), "count": len(df_val)},
                "test": {"start": str(test_start), "end": str(test_end), "count": len(df_test)},
            },
            calibrator_usage_stats=train_usage_stats,
            xgb_model_paths=xgb_model_paths,
            validation_stats=train_meta["validation_stats"],
            raw_columns_preserved=train_raw_cols,
            raw_restore_method=train_meta.get("raw_restore_method", ""),
            raw_input_schema_signature=raw_schema_sig,
            session_histograms=session_histograms_main,
            ts_min_max_by_split=ts_min_max_by_split,
        )
        
        # Val
        if len(df_val) > 0:
            log.info("[DATASET_BUILD] Building val dataset...")
            df_val_built, val_meta = build_dataset(
                df_raw=df_val,
                policy_config=policy_config,
                feature_meta_path=Path(args.feature_meta_path),
                seq_scaler_path=Path(args.seq_scaler_path) if args.seq_scaler_path else None,
                snap_scaler_path=Path(args.snap_scaler_path) if args.snap_scaler_path else None,
                calibration_dir=Path(args.calibration_dir),
                calibration_method=args.calibration_method,
            )
        else:
            log.info("[DATASET_BUILD] Skipping val dataset (0 rows)")
            df_val_built = None
            val_meta = {"validation_stats": {}, "calibrator_usage_stats": {}}
        # PHASE 2: Capture stats snapshot immediately after build (before any reset)
        val_usage_stats = copy.deepcopy(val_meta.get("calibrator_usage_stats", get_calibrator_usage_stats()))
        
        if df_val_built is not None and len(df_val_built) > 0:
            val_output = output_dir / f"{output_stem}_val.parquet"
            val_output.parent.mkdir(parents=True, exist_ok=True)
            df_val_built.to_parquet(val_output, index=False)
            log.info(f"✅ Saved val dataset: {val_output}")
            
            # Write manifest for val
            val_raw_cols = val_meta.get("raw_columns_preserved", [])
            write_manifest(
            output_path=val_output,
            input_data_path=args.data,
            build_command=build_command,
            policy_config_path=str(policy_config_path),
            calibration_dir=args.calibration_dir,
            calibration_method=args.calibration_method,
            feature_meta_path=args.feature_meta_path,
            seq_scaler_path=args.seq_scaler_path,
            snap_scaler_path=args.snap_scaler_path,
            policy_id=policy_id,
            seq_feature_names=seq_feat_names,
            snap_feature_names=snap_feat_names,
            seq_len=30,
            feature_contract_hash=None,
            splits={
                "train": {"start": str(train_start), "end": str(train_end), "count": len(df_train)},
                "val": {"start": str(val_start), "end": str(val_end), "count": len(df_val)},
                "test": {"start": str(test_start), "end": str(test_end), "count": len(df_test)},
            },
            calibrator_usage_stats=val_usage_stats,
            xgb_model_paths=xgb_model_paths,
            validation_stats=val_meta["validation_stats"],
            raw_columns_preserved=val_raw_cols,
            raw_restore_method=val_meta.get("raw_restore_method", ""),
            raw_input_schema_signature=raw_schema_sig,
            session_histograms=session_histograms_main,
            ts_min_max_by_split=ts_min_max_by_split,
            )
        else:
            log.info("[DATASET_BUILD] Skipping val manifest (no data)")
        
        # Test
        if len(df_test) > 0:
            log.info("[DATASET_BUILD] Building test dataset...")
            df_test_built, test_meta = build_dataset(
                df_raw=df_test,
                policy_config=policy_config,
                feature_meta_path=Path(args.feature_meta_path),
                seq_scaler_path=Path(args.seq_scaler_path) if args.seq_scaler_path else None,
                snap_scaler_path=Path(args.snap_scaler_path) if args.snap_scaler_path else None,
                calibration_dir=Path(args.calibration_dir),
                calibration_method=args.calibration_method,
            )
        else:
            log.info("[DATASET_BUILD] Skipping test dataset (0 rows)")
            df_test_built = None
            test_meta = {"validation_stats": {}, "calibrator_usage_stats": {}}
        
        if df_test_built is not None and len(df_test_built) > 0:
            test_output = output_dir / f"{output_stem}_test.parquet"
            test_output.parent.mkdir(parents=True, exist_ok=True)
            df_test_built.to_parquet(test_output, index=False)
            log.info(f"✅ Saved test dataset: {test_output}")
            
            # Write manifest for test
            test_raw_cols = test_meta.get("raw_columns_preserved", [])
            write_manifest(
            output_path=test_output,
            input_data_path=args.data,
            build_command=build_command,
            policy_config_path=str(policy_config_path),
            calibration_dir=args.calibration_dir,
            calibration_method=args.calibration_method,
            feature_meta_path=args.feature_meta_path,
            seq_scaler_path=args.seq_scaler_path,
            snap_scaler_path=args.snap_scaler_path,
            policy_id=policy_id,
            seq_feature_names=seq_feat_names,
            snap_feature_names=snap_feat_names,
            seq_len=30,
            feature_contract_hash=None,
            splits={
                "train": {"start": str(train_start), "end": str(train_end), "count": len(df_train)},
                "val": {"start": str(val_start), "end": str(val_end), "count": len(df_val)},
                "test": {"start": str(test_start), "end": str(test_end), "count": len(df_test)},
            },
            calibrator_usage_stats=test_meta["calibrator_usage_stats"],
            xgb_model_paths=xgb_model_paths,
            validation_stats=test_meta["validation_stats"],
            raw_columns_preserved=test_raw_cols,
            raw_restore_method=test_meta.get("raw_restore_method", ""),
            raw_input_schema_signature=raw_schema_sig,
            session_histograms=session_histograms_main,
            ts_min_max_by_split=ts_min_max_by_split,
            )
        else:
            log.info("[DATASET_BUILD] Skipping test manifest (no data)")
    else:
        # Build single dataset
        df_built, build_meta = build_dataset(
            df_raw=df_raw,
            policy_config=policy_config,
            feature_meta_path=Path(args.feature_meta_path),
            seq_scaler_path=Path(args.seq_scaler_path) if args.seq_scaler_path else None,
            snap_scaler_path=Path(args.snap_scaler_path) if args.snap_scaler_path else None,
            calibration_dir=Path(args.calibration_dir),
            calibration_method=args.calibration_method,
        )
        
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_built.to_parquet(output_path, index=False)
        log.info(f"✅ Saved dataset: {output_path}")
        
        # Write manifest
        seq_feat_names = df_built.attrs.get("seq_features", [])
        snap_feat_names = df_built.attrs.get("snap_features", [])
        build_raw_cols = build_meta.get("raw_columns_preserved", [])
        write_manifest(
            output_path=output_path,
            input_data_path=args.data,
            build_command=build_command,
            policy_config_path=str(policy_config_path),
            calibration_dir=args.calibration_dir,
            calibration_method=args.calibration_method,
            feature_meta_path=args.feature_meta_path,
            seq_scaler_path=args.seq_scaler_path,
            snap_scaler_path=args.snap_scaler_path,
            policy_id=policy_id,
            seq_feature_names=seq_feat_names,
            snap_feature_names=snap_feat_names,
            seq_len=30,
            feature_contract_hash=None,
            splits=None,
            calibrator_usage_stats=build_meta["calibrator_usage_stats"],
            xgb_model_paths=xgb_model_paths,
            validation_stats=build_meta["validation_stats"],
            raw_columns_preserved=build_raw_cols,
        )
    
    log.info("[DATASET_BUILD] Dataset build complete!")


if __name__ == "__main__":
    main()
