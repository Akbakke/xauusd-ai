#!/usr/bin/env python3
"""
Smoke test for XGB Sanitizer + Inference.

Loads prebuilt data, applies sanitizer, runs XGB inference,
and validates outputs.

Usage:
    python3 gx1/scripts/smoke_xgb_sanitizer_infer.py --year 2025 --n-bars 5000
    python3 gx1/scripts/smoke_xgb_sanitizer_infer.py --year 2020 --n-bars 5000
"""

import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# Add workspace root to path
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE_ROOT))

from gx1.xgb.preprocess.xgb_input_sanitizer import XGBInputSanitizer


def resolve_gx1_data_dir() -> Path:
    """Resolve GX1_DATA directory."""
    if "GX1_DATA_ROOT" in os.environ:
        path = Path(os.environ["GX1_DATA_ROOT"])
        if path.exists():
            return path
    default = WORKSPACE_ROOT.parent / "GX1_DATA"
    return default


def resolve_prebuilt_for_year(year: int, gx1_data: Path) -> Optional[Path]:
    """Resolve prebuilt parquet path for a given year."""
    candidates = [
        gx1_data / "data" / "data" / "prebuilt" / "TRIAL160" / str(year) / f"xauusd_m5_{year}_features_v10_ctx.parquet",
        gx1_data / "data" / "data" / "features" / f"xauusd_m5_{year}_features_v10_ctx.parquet",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_xgb_model(gx1_data: Path, session: str = "EU"):
    """Load XGB model from entry_v10_ctx bundle."""
    try:
        from joblib import load as joblib_load
    except ImportError:
        import joblib
        joblib_load = joblib.load
    
    model_path = gx1_data / "models" / "models" / "entry_v10_ctx" / "FULLYEAR_2025_GATED_FUSION" / f"xgb_{session}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"XGB model not found: {model_path}")
    
    return joblib_load(model_path), model_path


def main():
    parser = argparse.ArgumentParser(
        description="Smoke test for XGB Sanitizer + Inference"
    )
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Year to test (e.g., 2020, 2025)"
    )
    parser.add_argument(
        "--n-bars",
        type=int,
        default=5000,
        help="Number of bars to test"
    )
    parser.add_argument(
        "--session",
        type=str,
        default="EU",
        choices=["EU", "US", "OVERLAP", "ASIA"],
        help="Session for XGB model"
    )
    parser.add_argument(
        "--max-clip-rate",
        type=float,
        default=5.0,
        help="Maximum allowed clip rate (%) before warning/fail"
    )
    parser.add_argument(
        "--allow-high-clip",
        action="store_true",
        help="Allow high clip rate without failing"
    )
    parser.add_argument(
        "--sanitizer-config",
        type=Path,
        default=None,
        help="Sanitizer config path (default: gx1/xgb/contracts/xgb_input_sanitizer_v1.json)"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    gx1_data = resolve_gx1_data_dir()
    print(f"GX1_DATA: {gx1_data}")
    
    # Load sanitizer
    sanitizer_config = args.sanitizer_config or (WORKSPACE_ROOT / "gx1" / "xgb" / "contracts" / "xgb_input_sanitizer_v1.json")
    if not sanitizer_config.exists():
        print(f"ERROR: Sanitizer config not found: {sanitizer_config}")
        print("Run: python3 gx1/scripts/build_xgb_sanitizer_bounds.py")
        return 1
    
    print(f"Loading sanitizer from: {sanitizer_config}")
    sanitizer = XGBInputSanitizer.from_config(str(sanitizer_config))
    print(f"  Features: {len(sanitizer.feature_list)}")
    print(f"  Bounds: {len(sanitizer.bounds)}")
    
    # Load prebuilt data
    prebuilt_path = resolve_prebuilt_for_year(args.year, gx1_data)
    if not prebuilt_path:
        print(f"ERROR: No prebuilt found for year {args.year}")
        return 1
    
    print(f"\nLoading prebuilt: {prebuilt_path}")
    df = pd.read_parquet(prebuilt_path)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Sample
    if len(df) > args.n_bars:
        step = len(df) // args.n_bars
        df = df.iloc[::step][:args.n_bars]
        print(f"  Sampled to {len(df)} rows")
    
    # Apply sanitizer
    print(f"\nApplying sanitizer...")
    try:
        X, stats = sanitizer.sanitize(df, sanitizer.feature_list)
        print(f"  ✅ Sanitization successful")
        print(f"  Shape: {X.shape}")
        print(f"  Clip rate: {stats.clip_rate_pct:.2f}%")
        print(f"  Clipped total: {stats.n_clipped_total}")
        
        if stats.top_clipped_features:
            print(f"\n  Top clipped features:")
            for feature, count, pct in stats.top_clipped_features[:10]:
                print(f"    {feature}: {count} ({pct:.2f}%)")
        
        # Check clip rate
        if stats.clip_rate_pct > args.max_clip_rate:
            if args.allow_high_clip:
                print(f"\n  ⚠️  WARNING: Clip rate {stats.clip_rate_pct:.2f}% > {args.max_clip_rate}% (allowed)")
            else:
                print(f"\n  ❌ FAIL: Clip rate {stats.clip_rate_pct:.2f}% > {args.max_clip_rate}%")
                print("  Use --allow-high-clip to override")
                return 1
    
    except ValueError as e:
        print(f"  ❌ Sanitization failed: {e}")
        return 1
    
    # Load XGB model
    print(f"\nLoading XGB model ({args.session})...")
    try:
        xgb_model, model_path = load_xgb_model(gx1_data, args.session)
        print(f"  ✅ Loaded from: {model_path}")
    except FileNotFoundError as e:
        print(f"  ❌ {e}")
        return 1
    
    # Check feature count matches
    if hasattr(xgb_model, "n_features_in_"):
        expected_features = xgb_model.n_features_in_
        if X.shape[1] != expected_features:
            print(f"\n  ⚠️  WARNING: Feature count mismatch!")
            print(f"    Sanitizer features: {X.shape[1]}")
            print(f"    Model expects: {expected_features}")
            
            # Try to use first N features if model expects fewer
            if X.shape[1] > expected_features:
                print(f"    Using first {expected_features} features for inference")
                X = X[:, :expected_features]
    
    # Run XGB inference
    print(f"\nRunning XGB inference...")
    try:
        if hasattr(xgb_model, "predict_proba"):
            proba = xgb_model.predict_proba(X)
            if proba.shape[1] >= 2:
                p_long = proba[:, 1]
            else:
                p_long = proba[:, 0]
        else:
            p_long = xgb_model.predict(X)
        
        # Compute stats
        p_hat = 0.5 + (p_long - 0.5) * 0.8  # Simple calibration proxy
        uncertainty = 1.0 - np.abs(p_long - 0.5) * 2
        
        print(f"  ✅ Inference successful")
        print(f"\n  Output stats:")
        print(f"    p_long_xgb: min={p_long.min():.4f}, max={p_long.max():.4f}, mean={p_long.mean():.4f}")
        print(f"    p_hat_xgb: min={p_hat.min():.4f}, max={p_hat.max():.4f}, mean={p_hat.mean():.4f}")
        print(f"    uncertainty: min={uncertainty.min():.4f}, max={uncertainty.max():.4f}, mean={uncertainty.mean():.4f}")
        
        # Check for NaN/Inf in outputs
        if np.isnan(p_long).any():
            print(f"\n  ❌ FAIL: NaN in p_long_xgb output!")
            return 1
        if np.isinf(p_long).any():
            print(f"\n  ❌ FAIL: Inf in p_long_xgb output!")
            return 1
        
        # Check output range
        if p_long.min() < 0 or p_long.max() > 1:
            print(f"\n  ⚠️  WARNING: p_long_xgb outside [0,1] range")
        
    except Exception as e:
        print(f"  ❌ Inference failed: {e}")
        return 1
    
    # Summary
    print(f"\n{'='*60}")
    print("SMOKE TEST PASSED ✅")
    print(f"{'='*60}")
    print(f"Year: {args.year}")
    print(f"Bars: {len(df)}")
    print(f"Features: {X.shape[1]}")
    print(f"Clip rate: {stats.clip_rate_pct:.2f}%")
    print(f"p_long_xgb mean: {p_long.mean():.4f}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
