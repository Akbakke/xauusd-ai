#!/usr/bin/env python3
"""
Train XGB Output Calibrator on Multiyear Data (2020-2025)

Trains Platt scaling or Isotonic regression calibrators on XGB outputs
across all years to stabilize probabilities.

Usage:
    python gx1/scripts/train_xgb_calibrator_multiyear.py \
        --years 2020 2021 2022 2023 2024 2025 \
        --calibrator-type platt \
        --output-dir ../GX1_DATA/models/calibrators/
"""

import argparse
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Add workspace root to path
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE_ROOT))

try:
    import pandas as pd
    import joblib
except ImportError as e:
    print(f"ERROR: Missing required package: {e}")
    sys.exit(1)

try:
    from scipy.optimize import minimize
    from scipy.special import expit
except ImportError as e:
    print(f"ERROR: Missing scipy: {e}")
    print("Install with: pip install scipy")
    sys.exit(1)

try:
    from gx1.xgb.calibration import PlattScaler, IsotonicScaler, QuantileClipper
except ImportError as e:
    print(f"ERROR: Failed to import calibration module: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Default paths (will be resolved from env or relative to repo)
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]

XGB_OUTPUT_CHANNELS = ["p_long_xgb", "p_hat_xgb", "uncertainty_score"]


def resolve_gx1_data_dir() -> Path:
    """
    Resolve GX1_DATA directory.
    
    Priority:
    1. GX1_DATA_ROOT env var
    2. GX1_DATA_DATA env var (then parent)
    3. Check GX1_PATHS.md or gx1/paths.py
    4. Fallback: ../GX1_DATA relative to repo root
    """
    # Check env vars
    if "GX1_DATA_ROOT" in os.environ:
        path = Path(os.environ["GX1_DATA_ROOT"])
        if path.exists():
            return path
    
    if "GX1_DATA_DATA" in os.environ:
        path = Path(os.environ["GX1_DATA_DATA"]).parent
        if path.exists():
            return path
    
    # Check for GX1_PATHS.md or paths.py
    paths_md = WORKSPACE_ROOT / "GX1_PATHS.md"
    if paths_md.exists():
        # Default from GX1_PATHS.md is ../GX1_DATA
        default = WORKSPACE_ROOT.parent / "GX1_DATA"
        if default.exists():
            return default
    
    # Fallback: ../GX1_DATA
    default = WORKSPACE_ROOT.parent / "GX1_DATA"
    return default


def resolve_prebuilt_root_candidates(gx1_data_dir: Path) -> List[Path]:
    """
    Resolve prebuilt root directory candidates in prioritized order.
    
    Returns list of candidate prebuilt root directories.
    """
    candidates = []
    
    # Priority order (most specific first)
    patterns = [
        gx1_data_dir / "data" / "data" / "prebuilt",
        gx1_data_dir / "data" / "prebuilt",
        gx1_data_dir / "prebuilt",
        gx1_data_dir / "data" / "data" / "prebuilt" / "TRIAL160",
        gx1_data_dir / "data" / "prebuilt" / "TRIAL160",
        gx1_data_dir / "prebuilt" / "TRIAL160",
    ]
    
    for pattern in patterns:
        if pattern.exists() and pattern.is_dir():
            candidates.append(pattern)
    
    return candidates


def resolve_canonical_prebuilt(
    year: int,
    gx1_data_dir: Path,
    prebuilt_root: Optional[Path] = None,
    allow_noncanonical: bool = False,
) -> Tuple[Optional[Path], Optional[Path], List[Path]]:
    """
    Resolve canonical prebuilt parquet path for a year.
    
    Args:
        year: Year to resolve
        gx1_data_dir: GX1_DATA root directory
        prebuilt_root: Optional explicit prebuilt root (overrides auto-detection)
        allow_noncanonical: If True, allow paths under reports/ or archive/
    
    Returns:
        (selected_path, selected_root, candidate_paths_tried)
    """
    candidates = []
    
    # If explicit prebuilt_root provided, use it
    if prebuilt_root:
        root_candidates = [prebuilt_root]
    else:
        root_candidates = resolve_prebuilt_root_candidates(gx1_data_dir)
    
    # Try each root candidate
    for root in root_candidates:
        # Check if this root is non-canonical (reports/ or archive/)
        path_str = str(root)
        is_noncanonical = "/reports/" in path_str or "/archive/" in path_str
        
        if is_noncanonical and not allow_noncanonical:
            print(f"SKIP non-canonical prebuilt: {root}")
            continue
        
        # Try canonical structure: <root>/TRIAL160/<year>/xauusd_m5_<year>_features_v10_ctx.parquet
        canonical_path = root / "TRIAL160" / str(year) / f"xauusd_m5_{year}_features_v10_ctx.parquet"
        candidates.append(canonical_path)
        if canonical_path.exists():
            return canonical_path, root, candidates
        
        # If root already points to TRIAL160, try direct year structure
        if root.name == "TRIAL160":
            direct_path = root / str(year) / f"xauusd_m5_{year}_features_v10_ctx.parquet"
            if direct_path not in candidates:
                candidates.append(direct_path)
                if direct_path.exists():
                    return direct_path, root.parent, candidates
        
        # Try glob search within TRIAL160/<year> directory
        trial160_year_dir = root / "TRIAL160" / str(year)
        if trial160_year_dir.exists() and trial160_year_dir.is_dir():
            for parquet_file in trial160_year_dir.glob(f"*{year}*features*v10*ctx*.parquet"):
                if parquet_file not in candidates:
                    candidates.append(parquet_file)
                    if parquet_file.exists():
                        return parquet_file, root, candidates
        
        # If root is TRIAL160, try year subdirectory
        if root.name == "TRIAL160":
            year_dir = root / str(year)
            if year_dir.exists() and year_dir.is_dir():
                for parquet_file in year_dir.glob(f"*{year}*features*v10*ctx*.parquet"):
                    if parquet_file not in candidates:
                        candidates.append(parquet_file)
                        if parquet_file.exists():
                            return parquet_file, root.parent, candidates
    
    return None, None, candidates


def resolve_prebuilt_parquet_for_year(
    year: int,
    prebuilt_root: Optional[Path] = None,
    prebuilt_map: Optional[Dict[int, Path]] = None,
    gx1_data_dir: Optional[Path] = None,
    allow_noncanonical: bool = False,
) -> Tuple[Optional[Path], Optional[Path], List[Path]]:
    """
    Resolve prebuilt parquet path for a year.
    
    Priority:
    1. --prebuilt-parquet-map (hard lock, overrides everything)
    2. Canonical resolution within prebuilt roots
    
    Returns: (selected_path, selected_root, candidate_paths_tried)
    """
    candidates = []
    
    # Hard lock: If explicit map provided, use it (overrides everything)
    if prebuilt_map and year in prebuilt_map:
        path = Path(prebuilt_map[year])
        candidates.append(path)
        if path.exists():
            # Determine root from path
            path_str = str(path)
            if "/TRIAL160/" in path_str:
                # Extract root up to prebuilt
                parts = path.parts
                try:
                    prebuilt_idx = next(i for i, p in enumerate(parts) if p == "prebuilt")
                    root = Path(*parts[:prebuilt_idx + 1])
                except StopIteration:
                    root = path.parent.parent
            else:
                root = path.parent.parent
            return path, root, candidates
        return None, None, candidates
    
    # Resolve GX1_DATA if not provided
    if gx1_data_dir is None:
        gx1_data_dir = resolve_gx1_data_dir()
    
    # Use canonical resolution
    return resolve_canonical_prebuilt(
        year=year,
        gx1_data_dir=gx1_data_dir,
        prebuilt_root=prebuilt_root,
        allow_noncanonical=allow_noncanonical,
    )


def load_xgb_outputs_from_prebuilt(
    year: int,
    prebuilt_path: Path,
    xgb_model_path: Path,
    n_samples: int = 50000,
    warmup: int = 200,
    require_feature_names: bool = True,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Load or compute XGB outputs for a year.
    
    If XGB outputs exist in prebuilt, load them.
    Otherwise, run XGB inference on prebuilt features.
    
    Args:
        year: Year being processed
        prebuilt_path: Path to prebuilt parquet file
        xgb_model_path: Path to XGB model (already resolved)
        n_samples: Number of samples to collect
        warmup: Number of warmup bars to skip
    
    Returns:
        Dict with XGB output channels or None on error
    """
    if not prebuilt_path.exists():
        print(f"  ERROR: Prebuilt not found for {year}: {prebuilt_path}")
        return None
    
    try:
        df = pd.read_parquet(prebuilt_path)
        
        # Handle index
        if "ts" in df.columns:
            df = df.set_index("ts")
        
        # Skip warmup and sample
        if len(df) <= warmup:
            return None
        
        df = df.iloc[warmup:]
        
        # Deterministic subsample
        step = max(1, len(df) // n_samples)
        df = df.iloc[::step][:n_samples]
        
        # Check if XGB outputs already exist
        existing_outputs = [col for col in XGB_OUTPUT_CHANNELS if col in df.columns]
        
        if len(existing_outputs) == len(XGB_OUTPUT_CHANNELS):
            # All outputs exist
            return {col: df[col].values for col in XGB_OUTPUT_CHANNELS}
        
        # Need to run XGB inference
        print(f"  Running XGB inference for {year}...")
        
        # Load XGB model (path already resolved, should exist)
        if not xgb_model_path.exists():
            raise FileNotFoundError(f"XGB model path does not exist: {xgb_model_path}")
        
        xgb_model = joblib.load(xgb_model_path)
        
        # Get feature columns (try multiple methods)
        feature_cols = None
        
        # Method 1: Custom attribute
        if hasattr(xgb_model, "feature_cols"):
            feature_cols = xgb_model.feature_cols
        
        # Method 2: Scikit-learn style
        elif hasattr(xgb_model, "feature_names_in_"):
            feature_cols = list(xgb_model.feature_names_in_)
        
        # Method 3: XGBoost booster feature names
        elif hasattr(xgb_model, "get_booster"):
            try:
                booster = xgb_model.get_booster()
                if hasattr(booster, "feature_names"):
                    feature_names = booster.feature_names
                    if feature_names:
                        feature_cols = list(feature_names)
            except Exception:
                pass
        
        # Method 4: Check if it's a wrapped XGBoost model
        elif hasattr(xgb_model, "booster"):
            try:
                booster = xgb_model.booster
                if hasattr(booster, "feature_names"):
                    feature_names = booster.feature_names
                    if feature_names:
                        feature_cols = list(feature_names)
            except Exception:
                pass
        
        # Method 5: Try to infer from n_features_in_ and use all numeric columns
        # TRUTH MODE TRIPWIRE: Forbid fallback in truth/replay mode
        if feature_cols is None:
            if hasattr(xgb_model, "n_features_in_"):
                n_features = xgb_model.n_features_in_
                # Use all numeric columns from prebuilt (excluding index and reserved columns)
                numeric_cols = [col for col in df.columns if df[col].dtype in [np.float32, np.float64, np.int32, np.int64]]
                # Exclude reserved columns
                reserved = ["ts", "p_long_xgb", "p_hat_xgb", "uncertainty_score"]
                numeric_cols = [col for col in numeric_cols if col not in reserved]
                if len(numeric_cols) >= n_features:
                    feature_cols = numeric_cols[:n_features]
                    # TRUTH MODE TRIPWIRE: Hard fail if require_feature_names is set
                    if require_feature_names:
                        raise ValueError(
                            f"TRUTH_MODE_FALLBACK_FORBIDDEN: XGB model at {xgb_model_path} lacks feature names. "
                            f"Truth/replay mode requires explicit feature names. "
                            f"Fallback to 'first {n_features} numeric columns' is not allowed. "
                            f"Model attributes: {', '.join([a for a in dir(xgb_model) if not a.startswith('_')])}"
                        )
                    print(f"    WARNING: Using first {n_features} numeric columns as features (model doesn't specify feature names)")
        
        if feature_cols is None:
            raise ValueError(
                f"Cannot get XGB feature columns from model at {xgb_model_path}. "
                f"Model attributes: {', '.join([a for a in dir(xgb_model) if not a.startswith('_')])}"
            )
        
        # Check features exist
        missing = [f for f in feature_cols if f not in df.columns]
        if missing:
            raise ValueError(
                f"Missing {len(missing)} features in prebuilt data for {year}. "
                f"First 10 missing: {missing[:10]}"
            )
        
        # Run inference
        X = df[feature_cols].values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        if hasattr(xgb_model, "predict_proba"):
            proba = xgb_model.predict_proba(X)
            p_long = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        else:
            p_long = xgb_model.predict(X)
        
        # Validate outputs (no NaN/Inf)
        if np.any(np.isnan(p_long)) or np.any(np.isinf(p_long)):
            raise ValueError(
                f"XGB inference produced NaN/Inf values for {year}. "
                f"NaN count: {np.sum(np.isnan(p_long))}, "
                f"Inf count: {np.sum(np.isinf(p_long))}"
            )
        
        return {
            "p_long_xgb": p_long,
            "p_hat_xgb": p_long,  # Same without calibration
            "uncertainty_score": 1.0 - 2 * np.abs(p_long - 0.5),
        }
        
    except Exception as e:
        print(f"  ERROR loading {year}: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_smoke_xgb_infer(
    prebuilt_path: Path,
    xgb_model_path: Path,
    n_bars: int = 1000,
    require_feature_names: bool = True,
) -> None:
    """
    Run smoke test: load 1000 bars from prebuilt, run XGB inference, validate outputs.
    
    Hard fails on any error or NaN/Inf values.
    """
    print("=" * 60)
    print("SMOKE TEST: XGB Inference")
    print("=" * 60)
    print(f"Prebuilt: {prebuilt_path}")
    print(f"XGB Model: {xgb_model_path}")
    print(f"Sample size: {n_bars} bars")
    print()
    
    # Load prebuilt
    if not prebuilt_path.exists():
        raise FileNotFoundError(f"Prebuilt not found: {prebuilt_path}")
    
    df = pd.read_parquet(prebuilt_path)
    print(f"Loaded prebuilt: {len(df)} rows, {len(df.columns)} columns")
    
    # Handle index
    if "ts" in df.columns:
        df = df.set_index("ts")
    
    # Take first n_bars
    if len(df) < n_bars:
        raise ValueError(f"Prebuilt has only {len(df)} rows, need at least {n_bars}")
    
    df = df.iloc[:n_bars]
    print(f"Using {len(df)} bars for smoke test")
    
    # Load XGB model
    if not xgb_model_path.exists():
        raise FileNotFoundError(f"XGB model not found: {xgb_model_path}")
    
    xgb_model = joblib.load(xgb_model_path)
    print(f"Loaded XGB model: {xgb_model_path}")
    
    # Get feature columns (try multiple methods)
    feature_cols = None
    
    # Method 1: Custom attribute
    if hasattr(xgb_model, "feature_cols"):
        feature_cols = xgb_model.feature_cols
    
    # Method 2: Scikit-learn style
    elif hasattr(xgb_model, "feature_names_in_"):
        feature_cols = list(xgb_model.feature_names_in_)
    
    # Method 3: XGBoost booster feature names
    elif hasattr(xgb_model, "get_booster"):
        try:
            booster = xgb_model.get_booster()
            if hasattr(booster, "feature_names"):
                feature_names = booster.feature_names
                if feature_names:
                    feature_cols = list(feature_names)
        except Exception:
            pass
    
    # Method 4: Check if it's a wrapped XGBoost model
    elif hasattr(xgb_model, "booster"):
        try:
            booster = xgb_model.booster
            if hasattr(booster, "feature_names"):
                feature_names = booster.feature_names
                if feature_names:
                    feature_cols = list(feature_names)
        except Exception:
            pass
    
    # Method 5: Try to infer from n_features_in_ and use all numeric columns
    # TRUTH MODE TRIPWIRE: Forbid fallback in truth/replay mode
    if feature_cols is None:
        if hasattr(xgb_model, "n_features_in_"):
            n_features = xgb_model.n_features_in_
            # Use all numeric columns from prebuilt (excluding index and reserved columns)
            numeric_cols = [col for col in df.columns if df[col].dtype in [np.float32, np.float64, np.int32, np.int64]]
            # Exclude reserved columns
            reserved = ["ts", "p_long_xgb", "p_hat_xgb", "uncertainty_score"]
            numeric_cols = [col for col in numeric_cols if col not in reserved]
            if len(numeric_cols) >= n_features:
                feature_cols = numeric_cols[:n_features]
                # TRUTH MODE TRIPWIRE: Hard fail if require_feature_names is set
                if require_feature_names:
                    raise ValueError(
                        f"TRUTH_MODE_FALLBACK_FORBIDDEN: XGB model at {xgb_model_path} lacks feature names. "
                        f"Truth/replay mode requires explicit feature names. "
                        f"Fallback to 'first {n_features} numeric columns' is not allowed. "
                        f"Model attributes: {', '.join([a for a in dir(xgb_model) if not a.startswith('_')])}"
                    )
                print(f"  WARNING: Using first {n_features} numeric columns as features (model doesn't specify feature names)")
    
    if feature_cols is None:
        raise ValueError(
            "Cannot get XGB feature columns from model. "
            "Model attributes: " + ", ".join(dir(xgb_model))
        )
    
    print(f"XGB model expects {len(feature_cols)} features")
    
    # Check features exist
    missing = [f for f in feature_cols if f not in df.columns]
    if missing:
        raise ValueError(
            f"Missing {len(missing)} features in prebuilt data. "
            f"First 10 missing: {missing[:10]}"
        )
    
    # Run inference
    X = df[feature_cols].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    if hasattr(xgb_model, "predict_proba"):
        proba = xgb_model.predict_proba(X)
        p_long = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
    else:
        p_long = xgb_model.predict(X)
    
    # Validate outputs
    if np.any(np.isnan(p_long)):
        raise ValueError(f"XGB inference produced NaN values. Count: {np.sum(np.isnan(p_long))}")
    
    if np.any(np.isinf(p_long)):
        raise ValueError(f"XGB inference produced Inf values. Count: {np.sum(np.isinf(p_long))}")
    
    # Compute statistics
    p_hat = p_long  # Same without calibration
    uncertainty = 1.0 - 2 * np.abs(p_long - 0.5)
    
    print()
    print("=" * 60)
    print("SMOKE TEST RESULTS")
    print("=" * 60)
    print(f"p_long_xgb: min={p_long.min():.4f}, max={p_long.max():.4f}, mean={p_long.mean():.4f}")
    print(f"p_hat_xgb: min={p_hat.min():.4f}, max={p_hat.max():.4f}, mean={p_hat.mean():.4f}")
    print(f"uncertainty_score: min={uncertainty.min():.4f}, max={uncertainty.max():.4f}, mean={uncertainty.mean():.4f}")
    print()
    print("✅ SMOKE TEST PASSED: No NaN/Inf detected, inference successful")
    print("=" * 60)


def create_synthetic_labels(
    p_long: np.ndarray,
    noise_level: float = 0.1,
) -> np.ndarray:
    """
    Create synthetic binary labels for calibration training.
    
    Since we don't have true entry success labels, we use the
    probability itself with some noise as a proxy. This trains
    the calibrator to center the distribution.
    
    For real calibration, you would use actual trade outcomes.
    """
    # Simple approach: sample labels with probability = p_long
    np.random.seed(42)  # Deterministic
    y = (np.random.random(len(p_long)) < p_long).astype(float)
    
    # Add some label noise to prevent perfect calibration
    noise_mask = np.random.random(len(y)) < noise_level
    y[noise_mask] = 1 - y[noise_mask]
    
    return y


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def compute_file_sha256(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def resolve_xgb_model(
    bundle_dir: Optional[Path],
    gx1_data_dir: Path,
    override_path: Optional[Path] = None,
    model_glob: Optional[str] = None,
    session: str = "EU",
) -> Path:
    """
    Resolve XGB model path with prioritized search.
    
    Priority:
    1. --xgb-model-path (hard override)
    2. --xgb-model-glob pattern
    3. <bundle_dir>/**/xgb*.(pkl|joblib|json|bst)
    4. <GX1_DATA>/models/**/xgb*...
    
    Args:
        bundle_dir: Bundle directory to search
        gx1_data_dir: GX1_DATA root directory
        override_path: Hard override path (from --xgb-model-path)
        model_glob: Optional glob pattern (from --xgb-model-glob)
        session: Session to use (EU/US/OVERLAP), defaults to EU
    
    Returns:
        Path to XGB model file
    
    Raises:
        FileNotFoundError: If model not found, with detailed error message
    """
    candidates = []
    
    # Priority 1: Hard override
    if override_path:
        path = Path(override_path)
        candidates.append(path)
        if path.exists() and path.is_file():
            return path
        else:
            raise FileNotFoundError(
                f"XGB_MODEL_NOT_FOUND: Override path does not exist: {override_path}"
            )
    
    # Priority 2: Custom glob pattern
    if model_glob:
        # Try in bundle_dir first
        if bundle_dir and bundle_dir.exists():
            for match in bundle_dir.rglob(model_glob):
                if match.is_file():
                    candidates.append(match)
                    return match
        
        # Try in GX1_DATA/models
        models_root = gx1_data_dir / "models"
        if models_root.exists():
            for match in models_root.rglob(model_glob):
                if match.is_file():
                    candidates.append(match)
                    return match
    
    # Priority 3: Search in bundle_dir with standard patterns
    # Prioritize entry_v10_ctx bundles over entry_v10
    if bundle_dir and bundle_dir.exists():
        # Check if bundle_dir is entry_v10_ctx (preferred) or entry_v10 (legacy)
        bundle_path_str = str(bundle_dir)
        is_v10_ctx = "/entry_v10_ctx/" in bundle_path_str
        is_v10_legacy = "/entry_v10/" in bundle_path_str and not is_v10_ctx
        
        # Standard naming patterns (prioritize v10_ctx compatible)
        patterns = [
            f"xgb_{session}.pkl",
            f"xgb_{session}.joblib",
        ]
        
        # Only add legacy patterns if allow_legacy is True
        # (We'll check this via the path check later, not here)
        
        # Try direct files first
        for pattern in patterns:
            path = bundle_dir / pattern
            candidates.append(path)
            if path.exists() and path.is_file():
                return path
        
        # Try recursive search with glob (limited depth)
        # Prefer entry_v10_ctx compatible patterns
        extensions = ["pkl", "joblib"]
        for ext in extensions:
            # Search in bundle_dir and immediate subdirectories (max depth 2)
            for depth in range(0, 3):
                if depth == 0:
                    pattern = f"xgb*.{ext}"
                else:
                    pattern = "*/" * depth + f"xgb*.{ext}"
                
                for match in bundle_dir.glob(pattern):
                    if match.is_file():
                        # Prefer session-specific files
                        if session in match.name.upper():
                            # Skip legacy patterns if we're in v10_ctx bundle
                            if is_v10_ctx and ("xgb_entry_" in match.name and "_v10.joblib" in match.name):
                                # This is a legacy pattern, skip it in v10_ctx bundles
                                continue
                            candidates.append(match)
                            return match
                        # Also collect non-session-specific for fallback
                        if match not in candidates:
                            candidates.append(match)
    
    # Priority 4: Search in GX1_DATA/models (limited scope)
    # Prioritize entry_v10_ctx over entry_v10
    models_search_dirs = [
        gx1_data_dir / "models" / "models" / "entry_v10_ctx",
        gx1_data_dir / "models" / "entry_v10_ctx",
        # Legacy entry_v10 (only if allow_legacy, but we check path later)
        gx1_data_dir / "models" / "models" / "entry_v10",
        gx1_data_dir / "models" / "entry_v10",
    ]
    
    for search_dir in models_search_dirs:
        if not search_dir.exists():
            continue
        
        # Try standard patterns
        for pattern in [
            f"xgb_{session}.pkl",
            f"xgb_{session}.joblib",
            f"xgb_entry_{session}_v10.joblib",
            f"xgb_entry_{session}_v10.pkl",
        ]:
            path = search_dir / pattern
            candidates.append(path)
            if path.exists() and path.is_file():
                return path
        
        # Try recursive search (max depth 2)
        for ext in ["pkl", "joblib"]:
            for depth in range(0, 3):
                if depth == 0:
                    pattern = f"xgb*.{ext}"
                else:
                    pattern = "*/" * depth + f"xgb*.{ext}"
                
                for match in search_dir.glob(pattern):
                    if match.is_file() and session in match.name.upper():
                        if match not in candidates:
                            candidates.append(match)
                            return match
    
    # HARD FAIL with detailed error message
    bundle_info = f"bundle_dir={bundle_dir}" if bundle_dir else "bundle_dir=None"
    candidates_str = "\n".join(f"  - {c} ({'EXISTS' if c.exists() else 'NOT FOUND'})" for c in candidates[:20])
    
    raise FileNotFoundError(
        f"XGB_MODEL_NOT_FOUND: Could not find XGB model for session '{session}'\n\n"
        f"Bundle directory used: {bundle_info}\n"
        f"GX1_DATA directory: {gx1_data_dir}\n"
        f"Override path: {override_path or 'None'}\n"
        f"Model glob: {model_glob or 'None'}\n\n"
        f"Candidates tried ({len(candidates)}):\n{candidates_str}\n\n"
        f"HOW TO FIX:\n"
        f"  1. Use --xgb-model-path <path> to specify model directly\n"
        f"  2. Use --bundle-dir <path> to point to bundle directory\n"
        f"  3. Use --xgb-model-glob <pattern> for custom search pattern\n"
        f"  4. Set GX1_DATA_ROOT env var if GX1_DATA path is wrong\n"
        f"  5. Verify XGB model exists in bundle directory or GX1_DATA/models/"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train XGB Output Calibrator on Multiyear Data"
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2020, 2021, 2022, 2023, 2024, 2025],
        help="Years to include in training"
    )
    parser.add_argument(
        "--prebuilt-root",
        type=Path,
        default=None,
        help="Prebuilt root directory (auto-detected if not provided)"
    )
    parser.add_argument(
        "--prebuilt-parquet-map",
        type=Path,
        default=None,
        help="JSON file mapping years to prebuilt parquet paths (hard lock, overrides all)"
    )
    parser.add_argument(
        "--allow-noncanonical-prebuilt",
        type=int,
        default=0,
        help="Allow non-canonical prebuilt paths under reports/ or archive/ (default: 0)"
    )
    parser.add_argument(
        "--list-prebuilt",
        action="store_true",
        help="List prebuilt file mapping and exit"
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=None,
        help="Bundle directory with XGB models (default: GX1_DATA/models/models/entry_v10_ctx/FULLYEAR_2025_GATED_FUSION)"
    )
    parser.add_argument(
        "--xgb-model-path",
        type=Path,
        default=None,
        help="Hard override: Direct path to XGB model file (overrides all other resolution)"
    )
    parser.add_argument(
        "--xgb-model-glob",
        type=str,
        default=None,
        help="Custom glob pattern for XGB model search (e.g., '**/xgb_EU*.pkl')"
    )
    parser.add_argument(
        "--smoke-xgb-infer",
        action="store_true",
        help="Run smoke test: load 1000 bars from 2025 prebuilt, run XGB inference, validate outputs"
    )
    parser.add_argument(
        "--require-feature-names",
        type=int,
        default=1,
        help="Hard fail if XGB model lacks feature names (default: 1 for truth/replay mode)"
    )
    parser.add_argument(
        "--require-universal-xgb",
        type=int,
        default=1,
        help="Hard fail if XGB model path is under /entry_v10/ (default: 1 for universal mode)"
    )
    parser.add_argument(
        "--allow-legacy-xgb",
        type=int,
        default=0,
        help="Allow legacy entry_v10 XGB models (default: 0, requires --require-universal-xgb 0)"
    )
    parser.add_argument(
        "--calibrator-type",
        choices=["platt", "isotonic"],
        default="platt",
        help="Calibrator type (default: platt)"
    )
    parser.add_argument(
        "--n-samples-per-year",
        type=int,
        default=50000,
        help="Samples per year (default: 50000)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for calibrators (default: GX1_DATA/models/calibrators)"
    )
    parser.add_argument(
        "--train-clipper",
        action="store_true",
        default=True,
        help="Also train quantile clipper"
    )
    
    args = parser.parse_args()
    
    # Resolve GX1_DATA
    gx1_data = resolve_gx1_data_dir()
    print(f"GX1_DATA resolved to: {gx1_data}")
    
    # Resolve output directory if not provided
    if args.output_dir is None:
        args.output_dir = gx1_data / "models" / "calibrators"
        print(f"Output directory auto-detected: {args.output_dir}")
    
    # Resolve bundle directory if not provided
    if args.bundle_dir is None:
        args.bundle_dir = gx1_data / "models" / "models" / "entry_v10_ctx" / "FULLYEAR_2025_GATED_FUSION"
        print(f"Bundle directory auto-detected: {args.bundle_dir}")
    
    # Resolve XGB model (required for inference)
    print()
    print("Resolving XGB model...")
    try:
        xgb_model_path = resolve_xgb_model(
            bundle_dir=args.bundle_dir,
            gx1_data_dir=gx1_data,
            override_path=args.xgb_model_path,
            model_glob=args.xgb_model_glob,
            session="EU",  # Default to EU session
        )
        
        # Universal mode tripwire: forbid entry_v10/ paths
        if args.require_universal_xgb and not args.allow_legacy_xgb:
            path_str = str(xgb_model_path)
            if "/entry_v10/" in path_str and "/entry_v10_ctx/" not in path_str:
                print()
                print("=" * 60)
                print("FATAL: Legacy XGB model path detected")
                print("=" * 60)
                print(f"Selected path: {xgb_model_path}")
                print("Universal mode requires entry_v10_ctx bundles, not entry_v10.")
                print("Use --allow-legacy-xgb 1 to override (not recommended).")
                return 1
            
            # Also check filename pattern
            if "xgb_entry_" in xgb_model_path.name and "_v10.joblib" in xgb_model_path.name:
                print()
                print("=" * 60)
                print("FATAL: Legacy XGB model filename pattern detected")
                print("=" * 60)
                print(f"Selected path: {xgb_model_path}")
                print("Universal mode requires entry_v10_ctx bundle structure.")
                print("Use --allow-legacy-xgb 1 to override (not recommended).")
                return 1
        
        # Print selected model with metadata
        try:
            size = xgb_model_path.stat().st_size
            sha256 = compute_file_sha256(xgb_model_path)
            size_str = format_file_size(size)
            print(f"SELECTED XGB MODEL: {xgb_model_path}")
            print(f"  Size: {size_str}")
            print(f"  SHA256: {sha256}")
        except Exception as e:
            print(f"SELECTED XGB MODEL: {xgb_model_path}")
            print(f"  (Could not compute metadata: {e})")
    except FileNotFoundError as e:
        print()
        print("=" * 60)
        print("FATAL: XGB Model Resolution Failed")
        print("=" * 60)
        print(str(e))
        return 1
    
    # Load prebuilt map if provided (hard lock)
    prebuilt_map = None
    if args.prebuilt_parquet_map:
        with open(args.prebuilt_parquet_map) as f:
            prebuilt_map = {int(k): Path(v) for k, v in json.load(f).items()}
        print(f"Loaded prebuilt map from: {args.prebuilt_parquet_map} (HARD LOCK)")
    
    # Resolve prebuilt paths for all years
    year_to_prebuilt = {}
    year_to_root = {}
    allow_noncanonical = bool(args.allow_noncanonical_prebuilt)
    
    print()
    print("Resolving prebuilt paths...")
    for year in args.years:
        prebuilt_path, selected_root, candidates = resolve_prebuilt_parquet_for_year(
            year=year,
            prebuilt_root=args.prebuilt_root,
            prebuilt_map=prebuilt_map,
            gx1_data_dir=gx1_data,
            allow_noncanonical=allow_noncanonical,
        )
        
        if prebuilt_path:
            year_to_prebuilt[year] = prebuilt_path
            year_to_root[year] = selected_root
            # Get file size
            try:
                size = prebuilt_path.stat().st_size
                size_str = format_file_size(size)
                print(f"  ✅ {year}: {prebuilt_path}")
                print(f"     Size: {size_str}, Root: {selected_root}")
            except Exception:
                print(f"  ✅ {year}: {prebuilt_path}")
                print(f"     Root: {selected_root}")
        else:
            print(f"  ❌ {year}: NOT FOUND")
            print(f"     Canonical candidates tried:")
            for cand in candidates:
                exists = "✅" if cand.exists() else "❌"
                print(f"       {exists} {cand}")
    
    # Handle smoke test mode
    if args.smoke_xgb_infer:
        # Find 2025 prebuilt for smoke test
        year_2025 = 2025
        prebuilt_path_2025, _, _ = resolve_prebuilt_parquet_for_year(
            year=year_2025,
            prebuilt_root=args.prebuilt_root,
            prebuilt_map=prebuilt_map,
            gx1_data_dir=gx1_data,
            allow_noncanonical=bool(args.allow_noncanonical_prebuilt),
        )
        
        if not prebuilt_path_2025:
            print(f"ERROR: Could not find prebuilt for {year_2025} (required for smoke test)")
            print("Use --list-prebuilt to see what was tried")
            return 1
        
        try:
            run_smoke_xgb_infer(
                prebuilt_path=prebuilt_path_2025,
                xgb_model_path=xgb_model_path,
                n_bars=1000,
                require_feature_names=bool(args.require_feature_names),
            )
            return 0
        except Exception as e:
            print()
            print("=" * 60)
            print("SMOKE TEST FAILED")
            print("=" * 60)
            print(str(e))
            import traceback
            traceback.print_exc()
            return 1
    
    # If --list-prebuilt, print mapping and exit
    if args.list_prebuilt:
        print()
        print("=" * 60)
        print("PREBUILT MAPPING")
        print("=" * 60)
        
        # Build mapping with metadata
        mapping = {}
        for year in args.years:
            if year in year_to_prebuilt:
                path = year_to_prebuilt[year]
                root = year_to_root.get(year, "unknown")
                try:
                    size = path.stat().st_size
                except Exception:
                    size = 0
                mapping[str(year)] = {
                    "path": str(path),
                    "root": str(root),
                    "size_bytes": size,
                    "size_human": format_file_size(size),
                }
            else:
                mapping[str(year)] = {
                    "path": None,
                    "root": None,
                    "size_bytes": 0,
                    "size_human": "N/A",
                }
        
        print(json.dumps(mapping, indent=2))
        
        # Auto-generate JSON mapping file
        if year_to_prebuilt:
            # Create output directory if needed
            args.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            years_str = "_".join(str(y) for y in sorted(args.years))
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            map_filename = f"prebuilt_map_trial160_{years_str}_{timestamp}.json"
            map_path = args.output_dir / map_filename
            
            # Write simple mapping (year -> path)
            simple_mapping = {str(year): str(path) for year, path in year_to_prebuilt.items()}
            with open(map_path, "w") as f:
                json.dump(simple_mapping, f, indent=2)
            
            print()
            print(f"Auto-generated mapping file: {map_path}")
            print(f"Use with: --prebuilt-parquet-map {map_path}")
        
        return 0
    
    # Fail-fast if any year is missing
    missing_years = [y for y in args.years if y not in year_to_prebuilt]
    if missing_years:
        print()
        print(f"ERROR: Missing prebuilt files for years: {missing_years}")
        print("Use --list-prebuilt to see what was tried")
        return 1
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print()
    print("=" * 60)
    print("XGB CALIBRATOR TRAINING")
    print("=" * 60)
    print(f"Years: {args.years}")
    print(f"Calibrator type: {args.calibrator_type}")
    print(f"Samples per year: {args.n_samples_per_year}")
    print(f"Output: {args.output_dir}")
    print()
    
    # Collect XGB outputs from all years
    all_outputs = {channel: [] for channel in XGB_OUTPUT_CHANNELS}
    
    print("Loading XGB outputs...")
    for year in args.years:
        prebuilt_path = year_to_prebuilt[year]
        print(f"  {year}...")
        outputs = load_xgb_outputs_from_prebuilt(
            year=year,
            prebuilt_path=prebuilt_path,
            xgb_model_path=xgb_model_path,
            n_samples=args.n_samples_per_year,
            require_feature_names=bool(args.require_feature_names),
        )
        
        if outputs is None:
            continue
        
        for channel in XGB_OUTPUT_CHANNELS:
            if channel in outputs:
                all_outputs[channel].append(outputs[channel])
    
    # Concatenate
    for channel in XGB_OUTPUT_CHANNELS:
        if all_outputs[channel]:
            all_outputs[channel] = np.concatenate(all_outputs[channel])
        else:
            all_outputs[channel] = np.array([])
    
    n_total = len(all_outputs["p_long_xgb"])
    print(f"\nTotal samples collected: {n_total:,}")
    
    if n_total == 0:
        print("ERROR: No samples collected")
        return 1
    
    # Train calibrator for p_long_xgb
    print("\nTraining calibrator...")
    
    p_long = all_outputs["p_long_xgb"]
    y_labels = create_synthetic_labels(p_long)
    
    if args.calibrator_type == "platt":
        calibrator = PlattScaler(name=f"multiyear_{timestamp}")
    else:
        calibrator = IsotonicScaler(name=f"multiyear_{timestamp}")
    
    calibrator.stats.years_included = args.years
    calibrator.fit(p_long, y_labels)
    
    # Save calibrator
    calibrator_path = args.output_dir / f"xgb_calibrator_{args.calibrator_type}_{timestamp}.pkl"
    calibrator.save(calibrator_path)
    print(f"Saved calibrator: {calibrator_path}")
    print(f"  SHA: {calibrator.stats.calibrator_sha}")
    print(f"  Brier before: {calibrator.stats.brier_before:.4f}")
    print(f"  Brier after: {calibrator.stats.brier_after:.4f}")
    print(f"  ECE before: {calibrator.stats.ece_before:.4f}")
    print(f"  ECE after: {calibrator.stats.ece_after:.4f}")
    
    # Train quantile clipper
    if args.train_clipper:
        print("\nTraining quantile clipper...")
        
        clipper = QuantileClipper(lower_quantile=0.01, upper_quantile=0.99)
        clipper.fit(all_outputs, years=args.years)
        
        clipper_path = args.output_dir / f"xgb_clipper_{timestamp}.pkl"
        clipper.save(clipper_path)
        print(f"Saved clipper: {clipper_path}")
        print(f"  SHA: {clipper.stats.normalizer_sha}")
        
        for channel, bounds in clipper._bounds.items():
            print(f"  {channel}: [{bounds[0]:.4f}, {bounds[1]:.4f}]")
    
    # Write metadata
    # Get git commit if available
    git_commit = None
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(WORKSPACE_ROOT),
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            git_commit = result.stdout.strip()[:12]
    except Exception:
        pass
    
    metadata = {
        "calibrator_path": str(calibrator_path),
        "calibrator_sha": calibrator.stats.calibrator_sha,
        "calibrator_type": args.calibrator_type,
        "clipper_path": str(clipper_path) if args.train_clipper else None,
        "clipper_sha": clipper.stats.normalizer_sha if args.train_clipper else None,
        "years_included": args.years,
        "n_samples_per_year": args.n_samples_per_year,
        "n_samples_total": n_total,
        "trained_at": timestamp,
        "git_commit": git_commit,
        "outputs_calibrated": ["p_long_xgb", "p_hat_xgb"],
        "clipper_quantiles": {
            "lower": 0.01,
            "upper": 0.99,
        } if args.train_clipper else None,
        "clipper_bounds": clipper._bounds if args.train_clipper else None,
        "calibration_stats": {
            "brier_before": calibrator.stats.brier_before,
            "brier_after": calibrator.stats.brier_after,
            "brier_improvement": calibrator.stats.brier_before - calibrator.stats.brier_after,
            "ece_before": calibrator.stats.ece_before,
            "ece_after": calibrator.stats.ece_after,
            "ece_improvement": calibrator.stats.ece_before - calibrator.stats.ece_after,
            "input_mean": calibrator.stats.input_mean,
            "input_std": calibrator.stats.input_std,
            "input_p1": calibrator.stats.input_p1,
            "input_p99": calibrator.stats.input_p99,
            "output_mean": calibrator.stats.output_mean,
            "output_std": calibrator.stats.output_std,
            "output_p1": calibrator.stats.output_p1,
            "output_p99": calibrator.stats.output_p99,
        },
    }
    
    metadata_path = args.output_dir / f"calibration_metadata_{timestamp}.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved metadata: {metadata_path}")
    
    # Summary
    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print()
    print("Output files:")
    print(f"  - {calibrator_path.name}")
    if args.train_clipper:
        print(f"  - {clipper_path.name}")
    print(f"  - {metadata_path.name}")
    print()
    print("To use in runtime, set:")
    print(f"  GX1_XGB_CALIBRATOR_PATH={calibrator_path}")
    if args.train_clipper:
        print(f"  GX1_XGB_CLIPPER_PATH={clipper_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
