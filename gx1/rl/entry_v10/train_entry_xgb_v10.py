#!/usr/bin/env python3
"""
Train/retrain XGBoost snapshot models for ENTRY_V10.

Trains session-routed XGBoost models on snapshot features (85 features).
These models are used both for inference annotation and as ensemble components.

Input:
    - V9 entry dataset: data/entry_v9/full_2020_2025.parquet (default)
    - Feature metadata: gx1/models/entry_v9/nextgen_2020_2025_clean/entry_v9_feature_meta.json

Output:
    - models/entry_v10/xgb_entry_EU_v10.joblib
    - models/entry_v10/xgb_entry_US_v10.joblib
    - models/entry_v10/xgb_entry_OVERLAP_v10.joblib
    - models/entry_v10/xgb_entry_meta_v10.json

Usage:
    python -m gx1.rl.entry_v10.train_entry_xgb_v10 \
        --dataset data/entry_v9/full_2020_2025.parquet \
        --model-out-dir models/entry_v10
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_feature_meta(feature_meta_path: Path) -> List[str]:
    """Load snapshot feature names from metadata."""
    with open(feature_meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    snap_features = meta.get("snap_features", [])
    if not snap_features:
        raise ValueError(f"Missing snap_features in {feature_meta_path}")

    return snap_features


def infer_session(row: pd.Series) -> str:
    """Infer session from row data."""
    if "session" in row.index:
        session = str(row["session"]).upper()
        if session in ["EU", "US", "OVERLAP", "ASIA"]:
            return session

    if "_v1_session_tag_EU" in row.index and row.get("_v1_session_tag_EU", 0) > 0.5:
        return "EU"
    if "_v1_session_tag_US" in row.index and row.get("_v1_session_tag_US", 0) > 0.5:
        return "US"
    if "_v1_session_tag_OVERLAP" in row.index and row.get("_v1_session_tag_OVERLAP", 0) > 0.5:
        return "OVERLAP"
    # Note: No _v1_session_tag_ASIA feature exists, infer from timestamp if needed

    if "session_id" in row.index:
        session_id = int(row["session_id"])
        # Note: session_id mapping: 0=EU, 1=OVERLAP, 2=US (ASIA not in numeric mapping)
        # ASIA must be inferred from timestamp or session column
        if session_id == 0:
            return "EU"
        elif session_id == 1:
            return "OVERLAP"
        elif session_id == 2:
            return "US"

    # Fallback: try to infer from timestamp if available
    # Check both "ts" column and DatetimeIndex
    ts = None
    if "ts" in row.index:
        try:
            ts = pd.to_datetime(row["ts"])
        except:
            pass
    elif isinstance(row.name, pd.Timestamp):
        # Row index is a timestamp (DatetimeIndex)
        ts = row.name
    
    if ts is not None:
        from gx1.execution.live_features import infer_session_tag
        try:
            return infer_session_tag(ts)
        except:
            pass

    return "OVERLAP"  # Default


def train_xgb_v10(
    input_parquet: Path,
    model_out_dir: Path,
    feature_meta_path: Path,
    val_frac: float = 0.2,
) -> None:
    """
    Train session-routed XGBoost models for ENTRY_V10.

    Args:
        input_parquet: Path to V9 entry dataset (before XGB annotation)
        model_out_dir: Directory to save trained models
        feature_meta_path: Path to entry_v9_feature_meta.json
        val_frac: Validation fraction (default: 0.2)
    """
    log.info(f"Loading dataset: {input_parquet}")
    df = pd.read_parquet(input_parquet)
    log.info(f"Loaded {len(df)} rows")

    # Preserve label columns before feature building (they will be removed by build_v9_runtime_features)
    label_cols_to_preserve = {}
    for col in ["y_direction", "y", "mfe_bps", "MFE_bps", "mae_bps", "MAE_bps", "first_hit"]:
        if col in df.columns:
            label_cols_to_preserve[col] = df[col].copy()

    # Build V9 runtime features if not already built
    # Check if features are already built by checking for key features that are always present
    # Must have CLOSE (uppercase) AND session tag features to be considered fully built
    has_built_features = (
        "CLOSE" in df.columns and 
        "_v1_session_tag_EU" in df.columns and
        "_v1_session_tag_US" in df.columns
    )
    
    if not has_built_features:
        log.info("Features not found, building V9 runtime features...")
        from gx1.features.runtime_v9 import build_v9_runtime_features
        from gx1.features.feature_state import FeatureState
        from gx1.utils.feature_context import set_feature_state, reset_feature_state
        
        # Set feature state for HTF features
        feature_state = FeatureState()
        feature_token = set_feature_state(feature_state)
        try:
            df, _, _ = build_v9_runtime_features(
                df_raw=df,
                feature_meta_path=feature_meta_path,
                seq_scaler_path=None,  # Not needed for XGB training
                snap_scaler_path=None,  # Not needed for XGB training
            )
        finally:
            reset_feature_state(feature_token)
        log.info(f"Built features: {len(df)} rows")
        
        # Restore label columns (align by index)
        for col, series in label_cols_to_preserve.items():
            if col not in df.columns:
                # Try to align by index
                if isinstance(df.index, pd.DatetimeIndex) and isinstance(series.index, pd.DatetimeIndex):
                    # Both are DatetimeIndex - direct alignment
                    df[col] = series.reindex(df.index, fill_value=np.nan)
                elif len(df) == len(series):
                    # Same length - assume same order (if index was preserved)
                    df[col] = series.values
                else:
                    # Fallback: try reindex
                    df[col] = series.reindex(df.index, fill_value=np.nan)
                log.info(f"Restored label column: {col} (unique values: {df[col].unique()})")

    # Load feature metadata
    snap_features = load_feature_meta(feature_meta_path)
    log.info(f"Using {len(snap_features)} snapshot features")

    # Validate features
    missing = [f for f in snap_features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing snapshot features: {missing[:10]}...")

    # Ensure y_direction exists
    if "y_direction" not in df.columns:
        if "y" in df.columns:
            df["y_direction"] = (df["y"] > 0).astype(int)
        else:
            raise ValueError("Missing y_direction or y column")

    # Infer sessions
    log.info("Inferring sessions")
    # If df has DatetimeIndex, use it directly for faster inference
    if isinstance(df.index, pd.DatetimeIndex):
        from gx1.execution.live_features import infer_session_tag
        log.info("Using DatetimeIndex for session inference")
        df["session"] = df.index.map(lambda ts: infer_session_tag(ts))
    else:
        df["session"] = df.apply(infer_session, axis=1)
    
    # Sanity: Log session distribution
    session_counts = df["session"].value_counts()
    log.info(f"Session distribution: {dict(session_counts)}")
    
    # Warn if ASIA has low count or many UNKNOWN
    if "ASIA" in session_counts:
        asia_count = session_counts["ASIA"]
        total_count = len(df)
        asia_pct = (asia_count / total_count) * 100
        log.info(f"ASIA: {asia_count} rows ({asia_pct:.1f}% of total)")
        if asia_pct < 5:
            log.warning(f"ASIA represents only {asia_pct:.1f}% of data - verify session inference is correct")
    else:
        log.warning("ASIA session not found in data - verify session inference")
    
    if "UNKNOWN" in session_counts:
        unknown_count = session_counts["UNKNOWN"]
        unknown_pct = (unknown_count / len(df)) * 100
        log.warning(f"UNKNOWN sessions: {unknown_count} rows ({unknown_pct:.1f}%) - session inference may need improvement")

    # Train per session (including ASIA for FULLYEAR training)
    model_out_dir.mkdir(parents=True, exist_ok=True)
    all_metadata = {}

    for session in ["EU", "US", "OVERLAP", "ASIA"]:
        log.info(f"\n{'='*60}")
        log.info(f"Training XGBoost model for {session}")
        log.info(f"{'='*60}")

        # Filter to session
        session_df = df[df["session"] == session].copy()
        log.info(f"Session {session}: {len(session_df)} samples")

        if len(session_df) < 100:
            log.warning(f"Too few samples for {session}, skipping")
            continue

        # Prepare features and labels
        # Filter to only numeric features and handle non-numeric values
        X_df = session_df[snap_features].copy()
        
        # Convert non-numeric columns to numeric, filling NaN with 0
        for col in X_df.columns:
            if X_df[col].dtype == 'object':
                # Try to convert to numeric
                X_df[col] = pd.to_numeric(X_df[col], errors='coerce').fillna(0.0)
            # Ensure float32
            X_df[col] = X_df[col].astype(np.float32)
        
        X = X_df.values.astype(np.float32)
        y = session_df["y_direction"].values.astype(int)

        # Train/val split
        n_val = int(len(X) * val_frac)
        X_train, X_val = X[:-n_val], X[-n_val:]
        y_train, y_val = y[:-n_val], y[-n_val:]

        log.info(f"Train: {len(X_train)}, Val: {len(X_val)}")

        # Train XGBoost with 7 threads
        n_jobs = 7
        log.info(f"Training XGBoost with n_jobs={n_jobs}")
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            max_depth=6,
            n_estimators=200,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            n_jobs=n_jobs,
            random_state=42,
            tree_method="hist",  # Use hist for better CPU performance
            nthread=n_jobs,  # Explicit thread count
        )

        # Check label distribution
        unique_labels = np.unique(y_train)
        if len(unique_labels) < 2:
            log.warning(f"Session {session}: Only {len(unique_labels)} unique label(s) in training set, skipping")
            continue
        
        # Use callbacks for early stopping (newer XGBoost API)
        try:
            from xgboost import callback
            early_stop = callback.EarlyStopping(rounds=20, save_best=True)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[early_stop],
                verbose=False,
            )
        except (ImportError, AttributeError, TypeError):
            # Fallback for older XGBoost versions
            try:
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=20,
                    verbose=False,
                )
            except (TypeError, ValueError):
                # If early_stopping_rounds doesn't work, just fit without it
                try:
                    model.fit(X_train, y_train, verbose=False)
                except Exception as e:
                    log.error(f"XGBoost training failed for {session}: {e}")
                    continue

        # Evaluate
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)

        auc = roc_auc_score(y_val, y_pred_proba)
        accuracy = accuracy_score(y_val, y_pred)

        log.info(f"Val AUC: {auc:.4f}")
        log.info(f"Val Accuracy: {accuracy:.4f}")

        # Save model
        model_path = model_out_dir / f"xgb_entry_{session}_v10.joblib"
        joblib.dump(model, model_path)
        log.info(f"Saved model to {model_path}")

        # Store metadata
        all_metadata[session] = {
            "model_path": str(model_path),
            "n_train": len(X_train),
            "n_val": len(X_val),
            "val_auc": float(auc),
            "val_accuracy": float(accuracy),
            "n_features": len(snap_features),
            "feature_cols": snap_features,
        }

    # Save combined metadata
    metadata_path = model_out_dir / "xgb_entry_meta_v10.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_bundle_version": "entry_v10_v1",
                "feature_cols": snap_features,
                "label_name": "y_direction",
                "sessions": ["EU", "US", "OVERLAP", "ASIA"],
                "sessions_metadata": all_metadata,
                "hyperparameters": {
                    "objective": "binary:logistic",
                    "max_depth": 6,
                    "n_estimators": 200,
                    "learning_rate": 0.1,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                },
            },
            f,
            indent=2,
        )
    log.info(f"Saved metadata to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost snapshot models for ENTRY_V10")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/entry_v9/full_2020_2025.parquet"),
        help="Path to V9 entry dataset (default: data/entry_v9/full_2020_2025.parquet)",
    )
    parser.add_argument(
        "--model-out-dir",
        type=Path,
        default=Path("models/entry_v10"),
        help="Directory to save trained models (default: models/entry_v10)",
    )
    parser.add_argument(
        "--feature-meta",
        type=Path,
        default=Path("gx1/models/entry_v9/nextgen_2020_2025_clean/entry_v9_feature_meta.json"),
        help="Path to entry_v9_feature_meta.json",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.2,
        help="Validation fraction (default: 0.2)",
    )

    args = parser.parse_args()

    train_xgb_v10(
        input_parquet=args.dataset,
        model_out_dir=args.model_out_dir,
        feature_meta_path=args.feature_meta,
        val_frac=args.val_frac,
    )


if __name__ == "__main__":
    main()

