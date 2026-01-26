#!/usr/bin/env python3
"""
Train Universal XGB v1 on multiyear data.

Trains a single universal XGB model that replaces session-split legacy models.
Bound to SSoT contracts:
- xgb_input_features_v1.json (96 features, explicit order)
- xgb_input_sanitizer_v1.json (bounds + counters)

Usage:
    python3 gx1/scripts/train_xgb_universal_v1_multiyear.py --years 2020 2021 2022 2023 2024 2025
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
import pandas as pd

# Add workspace root to path
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE_ROOT))

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
    import xgboost as xgb
    from joblib import dump as joblib_dump
except ImportError as e:
    print(f"ERROR: Missing required package: {e}")
    print("Install with: pip install scikit-learn xgboost joblib")
    sys.exit(1)

from gx1.xgb.preprocess.xgb_input_sanitizer import XGBInputSanitizer, SanitizeStats


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
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_contracts() -> Tuple[List[str], str, XGBInputSanitizer, str]:
    """Load feature contract and sanitizer, return features, schema_hash, sanitizer, sanitizer_sha."""
    # Load feature contract
    feature_contract_path = WORKSPACE_ROOT / "gx1" / "xgb" / "contracts" / "xgb_input_features_v1.json"
    if not feature_contract_path.exists():
        raise FileNotFoundError(f"Feature contract not found: {feature_contract_path}")
    
    with open(feature_contract_path, "r") as f:
        feature_contract = json.load(f)
    
    features = feature_contract.get("features", [])
    schema_hash = feature_contract.get("schema_hash", "unknown")
    
    # Compute SHA256 of feature contract
    with open(feature_contract_path, "rb") as f:
        feature_contract_sha = hashlib.sha256(f.read()).hexdigest()
    
    # Load sanitizer
    sanitizer_config_path = WORKSPACE_ROOT / "gx1" / "xgb" / "contracts" / "xgb_input_sanitizer_v1.json"
    if not sanitizer_config_path.exists():
        raise FileNotFoundError(f"Sanitizer config not found: {sanitizer_config_path}")
    
    sanitizer = XGBInputSanitizer.from_config(str(sanitizer_config_path))
    
    # Compute SHA256 of sanitizer config
    with open(sanitizer_config_path, "rb") as f:
        sanitizer_sha = hashlib.sha256(f.read()).hexdigest()
    
    return features, schema_hash, sanitizer, sanitizer_sha, feature_contract_sha


def create_labels(df: pd.DataFrame, lookahead_bars: int = 12) -> np.ndarray:
    """
    Create binary labels for XGB training.
    
    Label = 1 if price goes up by at least some threshold within lookahead_bars
    This matches the legacy entry-v10 XGB label definition.
    
    Args:
        df: DataFrame with 'close' column and sorted by time
        lookahead_bars: Number of bars to look ahead for return
    
    Returns:
        Binary labels array
    """
    if "close" not in df.columns:
        # Try 'mid' as fallback
        if "mid" in df.columns:
            close = df["mid"].values
        else:
            raise ValueError("No 'close' or 'mid' column found for label creation")
    else:
        close = df["close"].values
    
    # Calculate future returns
    future_close = np.roll(close, -lookahead_bars)
    future_return = (future_close - close) / close
    
    # Mask the last lookahead_bars rows (no valid future)
    future_return[-lookahead_bars:] = np.nan
    
    # Binary label: 1 if positive return, 0 otherwise
    # Threshold of 0 for now (any positive return = 1)
    labels = (future_return > 0).astype(float)
    labels[-lookahead_bars:] = np.nan  # Mark last rows as NaN
    
    return labels


def compute_file_sha256(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Train Universal XGB v1 on multiyear data"
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2020, 2021, 2022, 2023, 2024, 2025],
        help="Years to train on"
    )
    parser.add_argument(
        "--holdout-year",
        type=int,
        default=None,
        help="Optional holdout year for final evaluation (not used in training)"
    )
    parser.add_argument(
        "--n-bars-per-year",
        type=int,
        default=None,
        help="Limit bars per year (default: use all)"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split ratio (default: 0.2)"
    )
    parser.add_argument(
        "--lookahead-bars",
        type=int,
        default=12,
        help="Lookahead bars for label creation (default: 12 = 1 hour)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: GX1_DATA/models/models/entry_v10_ctx/FULLYEAR_2025_GATED_FUSION/)"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="XGBoost max_depth (default: 6)"
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="XGBoost n_estimators (default: 200)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="XGBoost learning_rate (default: 0.1)"
    )
    
    args = parser.parse_args()
    np.random.seed(args.seed)
    
    print("=" * 60)
    print("TRAIN UNIVERSAL XGB V1")
    print("=" * 60)
    
    # Resolve paths
    gx1_data = resolve_gx1_data_dir()
    print(f"GX1_DATA: {gx1_data}")
    
    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = gx1_data / "models" / "models" / "entry_v10_ctx" / "FULLYEAR_2025_GATED_FUSION"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")
    
    # Load contracts
    print("\nLoading contracts...")
    features, schema_hash, sanitizer, sanitizer_sha, feature_contract_sha = load_contracts()
    print(f"  Features: {len(features)}")
    print(f"  Schema hash: {schema_hash}")
    print(f"  Feature contract SHA: {feature_contract_sha[:16]}...")
    print(f"  Sanitizer SHA: {sanitizer_sha[:16]}...")
    
    # Determine training years
    train_years = [y for y in args.years if y != args.holdout_year]
    holdout_year = args.holdout_year
    print(f"\nTraining years: {train_years}")
    if holdout_year:
        print(f"Holdout year: {holdout_year}")
    
    # Collect training data
    print("\nCollecting training data...")
    all_X = []
    all_y = []
    all_stats: Dict[int, Dict[str, Any]] = {}
    year_sample_counts = {}
    
    for year in train_years:
        prebuilt_path = resolve_prebuilt_for_year(year, gx1_data)
        if not prebuilt_path:
            print(f"  WARNING: No prebuilt for {year}, skipping")
            continue
        
        print(f"  Loading {year}...")
        df = pd.read_parquet(prebuilt_path)
        
        # Limit bars if specified
        if args.n_bars_per_year and len(df) > args.n_bars_per_year:
            step = len(df) // args.n_bars_per_year
            df = df.iloc[::step][:args.n_bars_per_year]
        
        print(f"    Rows: {len(df)}")
        
        # Create labels
        labels = create_labels(df, lookahead_bars=args.lookahead_bars)
        
        # Apply sanitizer
        try:
            X, stats = sanitizer.sanitize(df, features, allow_nan_fill=True, nan_fill_value=0.0)
            print(f"    Sanitized: {X.shape}, clip rate: {stats.clip_rate_pct:.2f}%")
            all_stats[year] = stats.to_dict()
        except Exception as e:
            print(f"    ERROR: Sanitization failed: {e}")
            continue
        
        # Remove rows with NaN labels
        valid_mask = ~np.isnan(labels)
        X = X[valid_mask]
        y = labels[valid_mask]
        
        print(f"    Valid samples: {len(y)} (label=1: {y.mean():.2%})")
        
        all_X.append(X)
        all_y.append(y)
        year_sample_counts[year] = len(y)
    
    if not all_X:
        print("ERROR: No training data collected")
        return 1
    
    # Concatenate all data
    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)
    print(f"\nTotal training data: {X_all.shape[0]} samples, {X_all.shape[1]} features")
    print(f"Label distribution: {y_all.mean():.2%} positive")
    
    # Time-based split (last val_split of each year's data)
    # For simplicity, we'll do a global split but ensure temporal ordering
    n_train = int(len(X_all) * (1 - args.val_split))
    X_train, X_val = X_all[:n_train], X_all[n_train:]
    y_train, y_val = y_all[:n_train], y_all[n_train:]
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Val set: {len(X_val)} samples")
    
    # Train XGBoost
    print("\nTraining XGBoost...")
    xgb_params = {
        "max_depth": args.max_depth,
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "use_label_encoder": False,
        "random_state": args.seed,
        "n_jobs": -1,
    }
    print(f"  Params: {xgb_params}")
    
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True,
    )
    
    # Evaluate
    print("\nEvaluating...")
    y_train_pred = model.predict_proba(X_train)[:, 1]
    y_val_pred = model.predict_proba(X_val)[:, 1]
    
    train_metrics = {
        "auc": roc_auc_score(y_train, y_train_pred),
        "logloss": log_loss(y_train, y_train_pred),
        "brier": brier_score_loss(y_train, y_train_pred),
    }
    val_metrics = {
        "auc": roc_auc_score(y_val, y_val_pred),
        "logloss": log_loss(y_val, y_val_pred),
        "brier": brier_score_loss(y_val, y_val_pred),
    }
    
    print(f"  Train AUC: {train_metrics['auc']:.4f}, LogLoss: {train_metrics['logloss']:.4f}")
    print(f"  Val AUC: {val_metrics['auc']:.4f}, LogLoss: {val_metrics['logloss']:.4f}")
    
    # Feature importance
    importance_df = pd.DataFrame({
        "feature": features[:len(model.feature_importances_)],
        "gain": model.feature_importances_,
    })
    importance_df = importance_df.sort_values("gain", ascending=False)
    
    print("\nTop 10 features by gain:")
    for _, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']}: {row['gain']:.4f}")
    
    # Save model
    print("\nSaving model...")
    model_path = output_dir / "xgb_universal_v1.joblib"
    joblib_dump(model, model_path)
    print(f"  Model: {model_path}")
    
    # Compute model SHA256
    model_sha = compute_file_sha256(model_path)
    print(f"  Model SHA256: {model_sha[:16]}...")
    
    # Save feature importance
    importance_path = output_dir / "FEATURE_IMPORTANCE.csv"
    importance_df.to_csv(importance_path, index=False)
    print(f"  Feature importance: {importance_path}")
    
    # Save metadata
    meta = {
        "version": "xgb_universal_v1",
        "created_at": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        "schema_hash": schema_hash,
        "feature_contract_sha256": feature_contract_sha,
        "sanitizer_sha256": sanitizer_sha,
        "model_sha256": model_sha,
        "n_features": len(features),
        "features": features,
        "training": {
            "years": train_years,
            "holdout_year": holdout_year,
            "n_samples_total": len(X_all),
            "n_samples_train": len(X_train),
            "n_samples_val": len(X_val),
            "samples_by_year": year_sample_counts,
            "val_split": args.val_split,
            "lookahead_bars": args.lookahead_bars,
            "seed": args.seed,
        },
        "xgb_params": xgb_params,
        "metrics": {
            "train": train_metrics,
            "val": val_metrics,
        },
        "sanitizer_stats_by_year": all_stats,
    }
    
    meta_path = output_dir / "xgb_universal_v1_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata: {meta_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Model SHA256: {model_sha}")
    print(f"Schema hash: {schema_hash}")
    print(f"Val AUC: {val_metrics['auc']:.4f}")
    print(f"Val LogLoss: {val_metrics['logloss']:.4f}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
