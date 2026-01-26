#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train Entry Timing Critic V1

Trains a multiclass classifier to predict timing_quality before entry.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    """Load timing dataset."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    log.info(f"Loading dataset: {dataset_path}")
    df = pd.read_parquet(dataset_path)
    log.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Check required columns
    required_cols = ["p_long", "timing_quality"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Drop rows with missing target
    initial_len = len(df)
    df = df.dropna(subset=["timing_quality"])
    dropped = initial_len - len(df)
    if dropped > 0:
        log.info(f"Dropped {dropped} rows with missing timing_quality")
    
    log.info(f"Final dataset: {len(df)} rows")
    
    # Check class distribution
    timing_counts = df["timing_quality"].value_counts()
    log.info("Timing quality distribution:")
    for quality, count in timing_counts.items():
        log.info(f"  {quality}: {count:,} ({count/len(df)*100:.1f}%)")
    
    return df


def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """
    Prepare feature matrix and encoders.
    
    Returns:
        feature_df: DataFrame with encoded features
        feature_names: List of feature names
        encoder_info: Dictionary with encoder information
    """
    log.info("Preparing features...")
    
    # Base numerical features
    numerical_features = ["p_long"]
    
    # Optional numerical features
    optional_numerical = ["spread_bps", "atr_bps", "threshold_slot_numeric", "real_threshold"]
    for col in optional_numerical:
        if col in df.columns:
            numerical_features.append(col)
    
    # Categorical features
    categorical_features = []
    if "session" in df.columns:
        categorical_features.append("session")
    if "trend_regime" in df.columns:
        categorical_features.append("trend_regime")
    if "vol_regime" in df.columns:
        categorical_features.append("vol_regime")
    
    # Build feature matrix
    feature_data = {}
    feature_names = []
    
    # Numerical features
    for col in numerical_features:
        if col in df.columns:
            feature_data[col] = df[col].fillna(0.0)
            feature_names.append(col)
        else:
            log.warning(f"Numerical feature {col} not found, skipping")
    
    # Encode categorical features
    encoders = {}
    encoder_info = {}
    
    for col in categorical_features:
        if col in df.columns:
            # Use ordinal encoding for simplicity (can be changed to one-hot)
            encoder = LabelEncoder()
            # Fill NaN with a default value
            col_data = df[col].fillna("UNKNOWN")
            encoded = encoder.fit_transform(col_data)
            feature_data[col] = encoded
            feature_names.append(col)
            encoders[col] = encoder
            encoder_info[col] = {
                "type": "LabelEncoder",
                "classes": encoder.classes_.tolist(),
            }
        else:
            log.warning(f"Categorical feature {col} not found, skipping")
    
    feature_df = pd.DataFrame(feature_data)
    
    # Fill any remaining NaN with 0
    feature_df = feature_df.fillna(0.0)
    
    log.info(f"Prepared {len(feature_names)} features: {feature_names}")
    
    return feature_df, feature_names, encoder_info


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_names: List[str],
) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """Train RandomForestClassifier model."""
    log.info("Training Entry Timing Critic V1 model...")
    
    # Initialize model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",  # Handle class imbalance
    )
    
    # Train
    model.fit(X_train, y_train)
    
    # Evaluate
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    
    log.info(f"Train accuracy: {train_acc:.4f}")
    log.info(f"Val accuracy: {val_acc:.4f}")
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_val, y_val_pred, average=None, zero_division=0
    )
    classes = model.classes_
    
    log.info("Per-class metrics (validation):")
    for i, cls in enumerate(classes):
        log.info(
            f"  {cls}: precision={precision[i]:.4f}, recall={recall[i]:.4f}, "
            f"f1={f1[i]:.4f}, support={support[i]}"
        )
    
    # Feature importance
    feature_importance = dict(zip(feature_names, model.feature_importances_))
    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    log.info("Top 10 features by importance:")
    for feat, imp in sorted_importance[:10]:
        log.info(f"  {feat}: {imp:.4f}")
    
    # Build metadata
    metadata = {
        "model_type": "RandomForestClassifier",
        "target": "timing_quality",
        "classes": classes.tolist(),
        "features": feature_names,
        "train_accuracy": float(train_acc),
        "val_accuracy": float(val_acc),
        "feature_importance": feature_importance,
        "n_estimators": 100,
        "max_depth": 10,
    }
    
    return model, metadata


def generate_report(
    model: RandomForestClassifier,
    metadata: Dict[str, Any],
    X_val: pd.DataFrame,
    y_val: pd.Series,
    report_path: Path,
) -> None:
    """Generate markdown training report."""
    lines = []
    
    lines.append("# Entry Timing Critic V1 – Training Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Metadata
    lines.append("## Metadata")
    lines.append("")
    lines.append(f"- **Model Type:** {metadata['model_type']}")
    lines.append(f"- **Target:** {metadata['target']}")
    lines.append(f"- **Classes:** {', '.join(metadata['classes'])}")
    lines.append(f"- **Features:** {len(metadata['features'])}")
    lines.append("")
    
    # Metrics
    lines.append("## Metrics")
    lines.append("")
    lines.append(f"- **Train Accuracy:** {metadata['train_accuracy']:.4f}")
    lines.append(f"- **Validation Accuracy:** {metadata['val_accuracy']:.4f}")
    lines.append("")
    
    # Per-class metrics
    y_val_pred = model.predict(X_val)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_val, y_val_pred, average=None, zero_division=0
    )
    
    lines.append("## Per-Class Metrics (Validation)")
    lines.append("")
    lines.append("| Class | Precision | Recall | F1 | Support |")
    lines.append("|-------|-----------|--------|----|---------|")
    for i, cls in enumerate(metadata['classes']):
        lines.append(
            f"| {cls} | {precision[i]:.4f} | {recall[i]:.4f} | "
            f"{f1[i]:.4f} | {support[i]} |"
        )
    lines.append("")
    
    # Confusion matrix
    cm = confusion_matrix(y_val, y_val_pred, labels=metadata['classes'])
    lines.append("## Confusion Matrix")
    lines.append("")
    lines.append("| Actual \\ Predicted | " + " | ".join(metadata['classes']) + " |")
    lines.append("|" + "---|" * (len(metadata['classes']) + 1))
    for i, actual in enumerate(metadata['classes']):
        row = f"| {actual} | " + " | ".join(str(cm[i, j]) for j in range(len(metadata['classes']))) + " |"
        lines.append(row)
    lines.append("")
    
    # Feature importance
    lines.append("## Top 10 Features by Importance")
    lines.append("")
    sorted_importance = sorted(
        metadata['feature_importance'].items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]
    lines.append("| Feature | Importance |")
    lines.append("|---------|------------|")
    for feat, imp in sorted_importance:
        lines.append(f"| {feat} | {imp:.4f} |")
    lines.append("")
    
    # Timing gain potential (focus on DELAY_BETTER)
    if "DELAY_BETTER" in metadata['classes']:
        delay_idx = metadata['classes'].index("DELAY_BETTER")
        delay_recall = recall[delay_idx]
        delay_precision = precision[delay_idx]
        
        lines.append("## Timing Gain Potential (DELAY_BETTER)")
        lines.append("")
        lines.append(f"- **Recall (sensitivity):** {delay_recall:.4f}")
        lines.append(f"  - How often the model correctly identifies DELAY_BETTER trades")
        lines.append(f"- **Precision:** {delay_precision:.4f}")
        lines.append(f"  - How often predicted DELAY_BETTER is actually DELAY_BETTER")
        lines.append("")
        lines.append("*High recall is important for identifying potential timing improvements.*")
        lines.append("")
    
    # Notes
    lines.append("---")
    lines.append("")
    lines.append("## Notater")
    lines.append("")
    lines.append("- Modellen predikerer timing_quality før entry.")
    lines.append("- Klassen AVOID_TRADE indikerer trades med høy MAE og dårlig slutt.")
    lines.append("- Klassen DELAY_BETTER indikerer trades som ville vært bedre med bedre entry-timing.")
    lines.append("")
    lines.append(f"*Report generated by `gx1/rl/train_entry_timing_critic_v1.py`*")
    
    # Write report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    
    log.info(f"Report saved: {report_path}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train Entry Timing Critic V1"
    )
    parser.add_argument(
        "--dataset_path",
        type=Path,
        default=Path("data/rl/entry_timing_dataset_FULLYEAR_2025_V1.parquet"),
        help="Path to timing dataset Parquet file",
    )
    parser.add_argument(
        "--output_model",
        type=Path,
        default=Path("gx1/models/entry_timing_critic_v1.joblib"),
        help="Output path for trained model",
    )
    parser.add_argument(
        "--output_meta",
        type=Path,
        default=Path("gx1/models/entry_timing_critic_v1_meta.json"),
        help="Output path for model metadata",
    )
    parser.add_argument(
        "--report_out",
        type=Path,
        default=Path("reports/rl/ENTRY_TIMING_CRITIC_V1_REPORT_FULLYEAR_2025.md"),
        help="Output path for training report",
    )
    
    args = parser.parse_args()
    
    try:
        # Load dataset
        df = load_dataset(args.dataset_path)
        
        if len(df) == 0:
            log.error("Dataset is empty")
            return 1
        
        # Prepare features
        feature_df, feature_names, encoder_info = prepare_features(df)
        
        # Prepare target
        target = df["timing_quality"]
        
        # Train/val split
        X_train, X_val, y_train, y_val = train_test_split(
            feature_df,
            target,
            test_size=0.3,
            random_state=42,
            stratify=target,  # Maintain class distribution
        )
        
        log.info(f"Train set: {len(X_train)} rows")
        log.info(f"Val set: {len(X_val)} rows")
        
        # Train model
        model, metadata = train_model(X_train, y_train, X_val, y_val, feature_names)
        
        # Add encoder info to metadata
        metadata["encoders"] = encoder_info
        
        # Save model
        log.info(f"Saving model: {args.output_model}")
        args.output_model.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, args.output_model)
        
        # Save metadata
        log.info(f"Saving metadata: {args.output_meta}")
        with open(args.output_meta, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Generate report
        generate_report(model, metadata, X_val, y_val, args.report_out)
        
        log.info("✅ Entry Timing Critic V1 training complete!")
        return 0
    
    except Exception as e:
        log.error(f"❌ Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())













