#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train Entry Critic V1 Model

Trains a binary classifier to predict whether a trade entry will be profitable
(>= 10 bps) based on entry state features.

Usage:
    # Use latest dataset
    python -m gx1.rl.train_entry_critic_v1

    # Specify dataset and target
    python -m gx1.rl.train_entry_critic_v1 \
        --dataset_path data/rl/sniper_shadow_rl_dataset_20251225_*.parquet \
        --target label_profitable

    # Debug with limited rows
    python -m gx1.rl.train_entry_critic_v1 --max_rows 1000
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

try:
    import joblib
except ImportError:
    print("ERROR: joblib not installed. Install with: pip install joblib")
    exit(1)


def find_latest_dataset(data_dir: Path = Path("data/rl")) -> Optional[Path]:
    """Find latest sniper_shadow_rl_dataset_*.parquet file."""
    if not data_dir.exists():
        return None
    
    datasets = sorted(
        [
            f
            for f in data_dir.glob("sniper_shadow_rl_dataset_*.parquet")
            if f.is_file()
        ],
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    
    if datasets:
        return datasets[0]
    return None


def load_dataset(
    dataset_path: Path,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Load RL dataset from Parquet file."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    print(f"Loading dataset: {dataset_path}")
    df = pd.read_parquet(dataset_path)
    
    if max_rows is not None and len(df) > max_rows:
        print(f"Truncating dataset to first {max_rows} rows (debug mode)")
        df = df.head(max_rows)
    
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
    """
    Prepare feature matrix from dataset.
    
    Returns:
        (features_df, feature_names)
    """
    # Base features
    feature_cols = [
        "p_long",
        "spread_bps",
        "atr_bps",
    ]
    
    # Regime one-hot features
    regime_cols = [
        "regime_trend_up",
        "regime_trend_down",
        "regime_vol_low",
        "regime_vol_high",
    ]
    
    # Session one-hot features
    session_cols = [
        "session_eu",
        "session_overlap",
        "session_us",
    ]
    
    # Shadow hit features
    shadow_cols = [
        "shadow_hit_0.55",
        "shadow_hit_0.58",
        "shadow_hit_0.60",
        "shadow_hit_0.62",
        "shadow_hit_0.65",
    ]
    
    # Threshold slot (convert to numeric)
    # We'll create a numeric version if threshold_slot exists
    
    # Collect all available feature columns
    all_feature_cols = []
    for col_list in [feature_cols, regime_cols, session_cols, shadow_cols]:
        for col in col_list:
            if col in df.columns:
                all_feature_cols.append(col)
    
    # Handle threshold_slot (categorical -> numeric mapping)
    if "threshold_slot" in df.columns:
        # Map threshold_slot to numeric (use mean p_long for each slot as proxy)
        threshold_map = {
            "<0.55": 0.50,
            "0.55-0.58": 0.565,
            "0.58-0.60": 0.59,
            "0.60-0.62": 0.61,
            "0.62-0.65": 0.635,
            "0.65-0.67": 0.66,
            "0.67+": 0.70,
        }
        df = df.copy()
        df["threshold_slot_numeric"] = df["threshold_slot"].map(threshold_map).fillna(0.60)
        all_feature_cols.append("threshold_slot_numeric")
    
    # Select features
    features_df = df[all_feature_cols].copy()
    
    # Fill missing values
    features_df = features_df.fillna(0.0).infer_objects(copy=False)
    
    # Convert boolean columns to int
    for col in features_df.columns:
        if features_df[col].dtype == bool:
            features_df[col] = features_df[col].astype(int)
    
    print(f"Prepared {len(all_feature_cols)} features: {all_feature_cols}")
    
    return features_df, all_feature_cols


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_type: str = "RandomForest",
) -> tuple[Any, Dict[str, float]]:
    """
    Train entry critic model.
    
    Returns:
        (trained_model, metrics_dict)
    """
    print(f"\nTraining {model_type} model...")
    
    if model_type == "RandomForest":
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_val_proba = model.predict_proba(X_val)[:, 1]
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    
    train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
        y_train, y_train_pred, average="binary", zero_division=0
    )
    val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
        y_val, y_val_pred, average="binary", zero_division=0
    )
    
    try:
        train_auc = roc_auc_score(y_train, y_train_proba)
        val_auc = roc_auc_score(y_val, y_val_proba)
    except ValueError:
        # Can happen if only one class in validation set
        train_auc = 0.0
        val_auc = 0.0
    
    # Calibration: average predicted prob for positive class vs actual positive rate
    train_pred_pos_rate = y_train_proba.mean()
    train_actual_pos_rate = y_train.mean()
    val_pred_pos_rate = y_val_proba.mean()
    val_actual_pos_rate = y_val.mean()
    
    metrics = {
        "train_accuracy": float(train_accuracy),
        "val_accuracy": float(val_accuracy),
        "train_precision": float(train_precision),
        "val_precision": float(val_precision),
        "train_recall": float(train_recall),
        "val_recall": float(val_recall),
        "train_f1": float(train_f1),
        "val_f1": float(val_f1),
        "train_auc": float(train_auc),
        "val_auc": float(val_auc),
        "train_pred_pos_rate": float(train_pred_pos_rate),
        "train_actual_pos_rate": float(train_actual_pos_rate),
        "val_pred_pos_rate": float(val_pred_pos_rate),
        "val_actual_pos_rate": float(val_actual_pos_rate),
    }
    
    return model, metrics


def get_feature_importance(model: Any, feature_names: List[str]) -> pd.DataFrame:
    """Extract feature importance from trained model."""
    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        return importance_df
    else:
        print("WARNING: Model does not support feature_importances_")
        return pd.DataFrame({"feature": feature_names, "importance": [0.0] * len(feature_names)})


def save_feature_importance_report(
    importance_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save feature importance report to markdown."""
    lines = []
    lines.append("# Entry Critic V1 - Feature Importance")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Top Features")
    lines.append("")
    lines.append("| Rank | Feature | Importance |")
    lines.append("|------|---------|------------|")
    
    for idx, row in importance_df.iterrows():
        rank = idx + 1
        lines.append(f"| {rank} | {row['feature']} | {row['importance']:.6f} |")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"*Report generated by `gx1/rl/train_entry_critic_v1.py`*")
    
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    
    print(f"Feature importance report saved: {output_path}")


def save_training_report(
    metrics: Dict[str, float],
    dataset_path: Path,
    target: str,
    n_rows_total: int,
    n_train: int,
    n_val: int,
    importance_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save training report to markdown."""
    lines = []
    lines.append("# Entry Critic V1 - Training Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Dataset Information")
    lines.append("")
    lines.append(f"- **Dataset Path:** {dataset_path}")
    lines.append(f"- **Target:** {target}")
    lines.append(f"- **Total Rows:** {n_rows_total}")
    lines.append(f"- **Train Rows:** {n_train}")
    lines.append(f"- **Validation Rows:** {n_val}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append("### Training Set")
    lines.append("")
    lines.append(f"- **Accuracy:** {metrics['train_accuracy']:.4f}")
    lines.append(f"- **Precision:** {metrics['train_precision']:.4f}")
    lines.append(f"- **Recall:** {metrics['train_recall']:.4f}")
    lines.append(f"- **F1 Score:** {metrics['train_f1']:.4f}")
    lines.append(f"- **ROC-AUC:** {metrics['train_auc']:.4f}")
    lines.append("")
    lines.append("### Validation Set")
    lines.append("")
    lines.append(f"- **Accuracy:** {metrics['val_accuracy']:.4f}")
    lines.append(f"- **Precision:** {metrics['val_precision']:.4f}")
    lines.append(f"- **Recall:** {metrics['val_recall']:.4f}")
    lines.append(f"- **F1 Score:** {metrics['val_f1']:.4f}")
    lines.append(f"- **ROC-AUC:** {metrics['val_auc']:.4f}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Calibration")
    lines.append("")
    lines.append("### Training Set")
    lines.append(f"- **Average Predicted Prob (Positive):** {metrics['train_pred_pos_rate']:.4f}")
    lines.append(f"- **Actual Positive Rate:** {metrics['train_actual_pos_rate']:.4f}")
    lines.append(f"- **Difference:** {abs(metrics['train_pred_pos_rate'] - metrics['train_actual_pos_rate']):.4f}")
    lines.append("")
    lines.append("### Validation Set")
    lines.append(f"- **Average Predicted Prob (Positive):** {metrics['val_pred_pos_rate']:.4f}")
    lines.append(f"- **Actual Positive Rate:** {metrics['val_actual_pos_rate']:.4f}")
    lines.append(f"- **Difference:** {abs(metrics['val_pred_pos_rate'] - metrics['val_actual_pos_rate']):.4f}")
    lines.append("")
    
    # Calibration comment
    train_diff = abs(metrics['train_pred_pos_rate'] - metrics['train_actual_pos_rate'])
    val_diff = abs(metrics['val_pred_pos_rate'] - metrics['val_actual_pos_rate'])
    
    if train_diff < 0.05 and val_diff < 0.05:
        calibration_comment = "✅ Model is well-calibrated (difference < 0.05)"
    elif train_diff < 0.10 and val_diff < 0.10:
        calibration_comment = "⚠️ Model is reasonably calibrated (difference < 0.10)"
    else:
        calibration_comment = "❌ Model may need calibration (difference >= 0.10)"
    
    lines.append(f"**Calibration Assessment:** {calibration_comment}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Top 10 Most Important Features")
    lines.append("")
    lines.append("| Rank | Feature | Importance |")
    lines.append("|------|---------|------------|")
    
    for idx, row in importance_df.head(10).iterrows():
        rank = idx + 1
        lines.append(f"| {rank} | {row['feature']} | {row['importance']:.6f} |")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"*Report generated by `gx1/rl/train_entry_critic_v1.py`*")
    
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    
    print(f"Training report saved: {output_path}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train Entry Critic V1 Model"
    )
    parser.add_argument(
        "--dataset_path",
        type=Path,
        default=None,
        help="Path to RL dataset Parquet file (default: latest in data/rl/)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="label_profitable_10bps",
        help="Target column name (default: label_profitable_10bps)",
    )
    parser.add_argument(
        "--model_out",
        type=Path,
        default=Path("gx1/models/entry_critic_v1.joblib"),
        help="Output path for trained model (default: gx1/models/entry_critic_v1.joblib)",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Maximum rows to use (for debugging, default: None = all)",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Validation set size (default: 0.2 = 20%%)",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random state for reproducibility (default: 42)",
    )
    
    args = parser.parse_args()
    
    # Find dataset
    if args.dataset_path:
        dataset_path = Path(args.dataset_path)
    else:
        dataset_path = find_latest_dataset()
        if not dataset_path:
            print("ERROR: No dataset found. Specify --dataset_path or ensure data/rl/sniper_shadow_rl_dataset_*.parquet exists")
            return 1
        print(f"Using latest dataset: {dataset_path}")
    
    # Load dataset
    try:
        df = load_dataset(dataset_path, max_rows=args.max_rows)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1
    
    # Check target column
    if args.target not in df.columns:
        print(f"ERROR: Target column '{args.target}' not found in dataset")
        print(f"Available columns: {list(df.columns)}")
        return 1
    
    # Prepare features
    features_df, feature_names = prepare_features(df)
    
    # Prepare target
    y = df[args.target].copy()
    
    # Check for missing values in target
    if y.isna().any():
        print(f"WARNING: {y.isna().sum()} missing values in target. Dropping rows.")
        mask = ~y.isna()
        features_df = features_df[mask]
        y = y[mask]
    
    # Check class balance
    class_counts = y.value_counts()
    print(f"\nTarget distribution:")
    for cls, count in class_counts.items():
        print(f"  Class {cls}: {count} ({count/len(y)*100:.1f}%%)")
    
    if len(class_counts) < 2:
        print(f"ERROR: Target has only one class. Cannot train binary classifier.")
        return 1
    
    # Train/validation split
    print(f"\nSplitting data: train={1-args.test_size:.0%}, val={args.test_size:.0%}")
    X_train, X_val, y_train, y_val = train_test_split(
        features_df,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,  # Stratified split for class balance
    )
    
    print(f"Train set: {len(X_train)} rows")
    print(f"Validation set: {len(X_val)} rows")
    
    # Train model
    model, metrics = train_model(X_train, y_train, X_val, y_val, model_type="RandomForest")
    
    # Print metrics
    print("\n" + "=" * 60)
    print("TRAINING METRICS")
    print("=" * 60)
    print(f"\nTraining Set:")
    print(f"  Accuracy:  {metrics['train_accuracy']:.4f}")
    print(f"  Precision: {metrics['train_precision']:.4f}")
    print(f"  Recall:    {metrics['train_recall']:.4f}")
    print(f"  F1 Score:  {metrics['train_f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['train_auc']:.4f}")
    print(f"\nValidation Set:")
    print(f"  Accuracy:  {metrics['val_accuracy']:.4f}")
    print(f"  Precision: {metrics['val_precision']:.4f}")
    print(f"  Recall:    {metrics['val_recall']:.4f}")
    print(f"  F1 Score:  {metrics['val_f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['val_auc']:.4f}")
    print(f"\nCalibration:")
    print(f"  Train: Pred={metrics['train_pred_pos_rate']:.4f}, Actual={metrics['train_actual_pos_rate']:.4f}")
    print(f"  Val:   Pred={metrics['val_pred_pos_rate']:.4f}, Actual={metrics['val_actual_pos_rate']:.4f}")
    
    # Feature importance
    importance_df = get_feature_importance(model, feature_names)
    print("\n" + "=" * 60)
    print("TOP 10 FEATURES BY IMPORTANCE")
    print("=" * 60)
    for idx, row in importance_df.head(10).iterrows():
        print(f"  {idx+1:2d}. {row['feature']:30s} {row['importance']:.6f}")
    
    # Save reports
    reports_dir = Path("reports/rl")
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    
    feature_report_path = reports_dir / f"ENTRY_CRITIC_V1_FEATURES_{timestamp}.md"
    save_feature_importance_report(importance_df, feature_report_path)
    
    training_report_path = reports_dir / f"ENTRY_CRITIC_V1_REPORT_{timestamp}.md"
    save_training_report(
        metrics,
        dataset_path,
        args.target,
        len(df),
        len(X_train),
        len(X_val),
        importance_df,
        training_report_path,
    )
    
    # Save model
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_out)
    print(f"\nModel saved: {args.model_out}")
    
    # Save metadata
    meta_path = args.model_out.parent / f"{args.model_out.stem}_meta.json"
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "target": args.target,
        "n_rows": len(df),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "features": feature_names,
        "model_type": "RandomForestClassifier",
        "metrics": metrics,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved: {meta_path}")
    
    print("\n✅ Training complete!")
    return 0


if __name__ == "__main__":
    exit(main())

