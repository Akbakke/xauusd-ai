#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FARM Entry Meta-Model Baseline

Trains and evaluates a meta-model that predicts whether a FARM trade
(LONG/SHORT, ASIA+LOW) will be profitable within 8 bars (y_profitable_8b).

Focuses on selection quality metrics, not just raw accuracy.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def compute_ece(y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        n_bins: Number of bins for calibration
    
    Returns:
        ECE value
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find predictions in this bin
        in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            # Accuracy in this bin
            accuracy_in_bin = y_true[in_bin].mean()
            # Average predicted probability in this bin
            avg_confidence_in_bin = y_pred_proba[in_bin].mean()
            # Add to ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return float(ece)


def load_entry_dataset(path: str) -> pd.DataFrame:
    """
    Load FARM entry dataset.
    
    Args:
        path: Path to parquet file
    
    Returns:
        DataFrame with cleaned data
    """
    logger.info(f"Loading dataset from {path}")
    df = pd.read_parquet(path)
    
    logger.info(f"Loaded {len(df)} rows, {df['trade_id'].nunique()} unique trades")
    
    # Drop rows with NaN in critical columns
    critical_cols = ['y_profitable_8b']
    before = len(df)
    df = df.dropna(subset=critical_cols)
    after = len(df)
    
    if before != after:
        logger.warning(f"Dropped {before - after} rows with NaN in critical columns")
    
    # Ensure y_profitable_8b is binary
    if 'y_profitable_8b' in df.columns:
        df['y_profitable_8b'] = df['y_profitable_8b'].astype(int)
    
    logger.info(f"Final dataset: {len(df)} rows")
    
    return df


def train_test_split_by_time_or_trade(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
    mode: str = "time",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train/test sets.
    
    Args:
        df: Full dataset
        test_ratio: Fraction for test set
        mode: "time" (time-based) or "trade" (trade-based)
    
    Returns:
        train_df, test_df
    """
    if mode == "time":
        # Sort by entry_ts
        if 'entry_ts' not in df.columns:
            raise ValueError("entry_ts column required for time-based split")
        
        df_sorted = df.sort_values('entry_ts').reset_index(drop=True)
        n_test = int(len(df_sorted) * test_ratio)
        
        train_df = df_sorted.iloc[:-n_test].copy()
        test_df = df_sorted.iloc[-n_test:].copy()
        
        logger.info(
            f"Time-based split: train={len(train_df)} ({train_df['entry_ts'].min()} to {train_df['entry_ts'].max()}), "
            f"test={len(test_df)} ({test_df['entry_ts'].min()} to {test_df['entry_ts'].max()})"
        )
    
    elif mode == "trade":
        # Group by trade_id
        if 'trade_id' not in df.columns:
            raise ValueError("trade_id column required for trade-based split")
        
        unique_trades = df['trade_id'].unique()
        n_test_trades = int(len(unique_trades) * test_ratio)
        
        np.random.seed(42)
        test_trade_ids = np.random.choice(unique_trades, size=n_test_trades, replace=False)
        
        train_df = df[~df['trade_id'].isin(test_trade_ids)].copy()
        test_df = df[df['trade_id'].isin(test_trade_ids)].copy()
        
        logger.info(
            f"Trade-based split: train={len(train_df)} ({train_df['trade_id'].nunique()} trades), "
            f"test={len(test_df)} ({test_df['trade_id'].nunique()} trades)"
        )
    
    else:
        raise ValueError(f"Unknown split mode: {mode}")
    
    return train_df, test_df


def train_meta_model(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "y_profitable_8b",
    model_type: str = "xgboost",
    random_state: int = 42,
) -> object:
    """
    Train meta-model on training data.
    
    Args:
        train_df: Training DataFrame
        feature_cols: List of feature column names
        target_col: Target column name
        model_type: "xgboost" or "gbdt"
        random_state: Random seed
    
    Returns:
        Fitted model
    """
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    
    logger.info(f"Training on {len(X_train)} samples with {len(feature_cols)} features")
    
    if model_type == "xgboost" and HAS_XGBOOST:
        logger.info("Using XGBoost")
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            eval_metric='logloss',
            use_label_encoder=False,
        )
    else:
        if model_type == "xgboost" and not HAS_XGBOOST:
            logger.warning("XGBoost not available, falling back to GradientBoosting")
        logger.info("Using GradientBoosting")
        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=random_state,
        )
    
    model.fit(X_train, y_train)
    logger.info("Training complete")
    
    return model


def evaluate_meta_model(
    model: object,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "y_profitable_8b",
) -> Dict:
    """
    Evaluate meta-model on test set.
    
    Args:
        model: Fitted model
        test_df: Test DataFrame
        feature_cols: List of feature column names
        target_col: Target column name
    
    Returns:
        Dictionary with metrics
    """
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Basic metrics
    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    
    # AUC
    try:
        auc = float(roc_auc_score(y_test, y_pred_proba))
    except ValueError:
        auc = 0.0
        logger.warning("Could not compute AUC (possibly only one class in test set)")
    
    # Calibration ECE
    ece = compute_ece(y_test, y_pred_proba)
    
    # Selection curves
    test_df_with_pred = test_df.copy()
    test_df_with_pred['p_pred'] = y_pred_proba
    
    # Sort by predicted probability (descending)
    test_df_sorted = test_df_with_pred.sort_values('p_pred', ascending=False).reset_index(drop=True)
    
    # Selection fractions
    fractions = [0.1, 0.2, 0.3, 0.5, 1.0]
    selection_curve = []
    
    # Compute period days for EV/day estimation
    if 'entry_ts' in test_df.columns:
        entry_ts_valid = test_df['entry_ts'].dropna()
        if len(entry_ts_valid) > 1:
            period_days = (entry_ts_valid.max() - entry_ts_valid.min()).total_seconds() / 86400.0
            period_days = max(period_days, 1.0)
        else:
            period_days = 1.0
    else:
        period_days = 1.0
    
    for f in fractions:
        n_sel = int(f * len(test_df_sorted))
        if n_sel == 0:
            continue
        
        df_sel = test_df_sorted.iloc[:n_sel].copy()
        
        # Compute metrics for selected subset
        mean_pnl_sel = float(df_sel['pnl_bps_signed'].mean()) if 'pnl_bps_signed' in df_sel.columns else 0.0
        mean_y_sel = float(df_sel[target_col].mean())
        ev_trade_sel = mean_pnl_sel
        
        # Approximate EV/day
        trades_per_day_sel = n_sel / period_days if period_days > 0 else 0.0
        ev_day_sel = mean_pnl_sel * trades_per_day_sel
        
        selection_curve.append({
            'fraction': float(f),
            'coverage': float(f),
            'n_selected': n_sel,
            'mean_pnl_bps_signed': mean_pnl_sel,
            'mean_y_profitable': mean_y_sel,
            'ev_trade_selected': ev_trade_sel,
            'ev_day_selected': ev_day_sel,
            'trades_per_day': trades_per_day_sel,
        })
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'ece': ece,
        'selection_curve': selection_curve,
        'n_test_samples': len(test_df),
        'n_features': len(feature_cols),
    }
    
    return metrics


def print_meta_summary(metrics: Dict) -> None:
    """
    Print summary of meta-model evaluation.
    
    Args:
        metrics: Metrics dictionary
    """
    print("\n" + "=" * 80)
    print("FARM ENTRY META-MODEL EVALUATION")
    print("=" * 80)
    print()
    
    print("Basic Metrics:")
    print(f"  AUC: {metrics['auc']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1: {metrics['f1']:.4f}")
    print(f"  ECE: {metrics['ece']:.4f}")
    print()
    
    print("Selection Curve (Top K% by Predicted Probability):")
    print()
    print(f"{'K%':<8} {'Coverage':<12} {'EV/trade':<12} {'EV/day':<12} {'Hit Rate':<12}")
    print("-" * 80)
    
    for point in metrics['selection_curve']:
        k_pct = int(point['fraction'] * 100)
        coverage = point['coverage']
        ev_trade = point['ev_trade_selected']
        ev_day = point['ev_day_selected']
        hit_rate = point['mean_y_profitable']
        
        print(
            f"{k_pct:<8} "
            f"{coverage:<12.1%} "
            f"{ev_trade:<12.2f} "
            f"{ev_day:<12.2f} "
            f"{hit_rate:<12.1%}"
        )
    
    print()
    print("=" * 80)


def auto_detect_feature_cols(df: pd.DataFrame) -> List[str]:
    """
    Auto-detect feature columns (numeric, excluding labels and IDs).
    
    Args:
        df: Dataset DataFrame
    
    Returns:
        List of feature column names
    """
    # Exclude labels and meta columns
    exclude_cols = {
        'y_profitable_8b',
        'pnl_bps_signed',
        'mfe_8b_signed',
        'mae_8b_signed',
        'trade_id',
        'entry_ts',
        'exit_ts',
    }
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter out excluded columns
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Always include side_sign, is_long, is_short if present
    for col in ['side_sign', 'is_long', 'is_short']:
        if col in df.columns and col not in feature_cols:
            feature_cols.append(col)
    
    logger.info(f"Auto-detected {len(feature_cols)} feature columns")
    
    return feature_cols


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train and evaluate FARM entry meta-model baseline"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="gx1/wf_runs/FARM_ENTRY_DATA/farm_entry_dataset_v1.parquet",
        help="Path to entry dataset parquet",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Fraction for test set (default: 0.2)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="time",
        choices=["time", "trade"],
        help="Split mode: 'time' (time-based) or 'trade' (trade-based) (default: time)",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("gx1/wf_runs/FARM_ENTRY_AI/entry_meta_baseline_metrics.json"),
        help="Output JSON path for metrics",
    )
    parser.add_argument(
        "--feature-cols",
        nargs="+",
        default=None,
        help="Optional list of feature columns (if not set, auto-detect)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="xgboost",
        choices=["xgboost", "gbdt"],
        help="Model type: 'xgboost' or 'gbdt' (default: xgboost)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--out-model",
        type=Path,
        default=Path("gx1/models/farm_entry_meta/baseline_model.pkl"),
        help="Output path for trained model (default: gx1/models/farm_entry_meta/baseline_model.pkl)",
    )
    parser.add_argument(
        "--out-feature-cols",
        type=Path,
        default=Path("gx1/models/farm_entry_meta/feature_cols.json"),
        help="Output path for feature columns JSON (default: gx1/models/farm_entry_meta/feature_cols.json)",
    )
    
    args = parser.parse_args()
    
    # Load dataset
    df = load_entry_dataset(args.dataset)
    
    if len(df) == 0:
        logger.error("No data loaded")
        return 1
    
    # Select feature columns
    if args.feature_cols:
        feature_cols = args.feature_cols
        # Verify all columns exist
        missing = [col for col in feature_cols if col not in df.columns]
        if missing:
            logger.error(f"Missing feature columns: {missing}")
            return 1
    else:
        feature_cols = auto_detect_feature_cols(df)
    
    if len(feature_cols) == 0:
        logger.error("No feature columns found")
        return 1
    
    logger.info(f"Using {len(feature_cols)} features: {feature_cols[:5]}...")
    
    # Split train/test
    train_df, test_df = train_test_split_by_time_or_trade(
        df,
        test_ratio=args.test_ratio,
        mode=args.mode,
    )
    
    if len(train_df) == 0 or len(test_df) == 0:
        logger.error("Train or test set is empty")
        return 1
    
    # Train model
    model = train_meta_model(
        train_df,
        feature_cols=feature_cols,
        target_col="y_profitable_8b",
        model_type=args.model_type,
        random_state=args.random_state,
    )
    
    # Evaluate
    metrics = evaluate_meta_model(
        model,
        test_df,
        feature_cols=feature_cols,
        target_col="y_profitable_8b",
    )
    
    # Print summary
    print_meta_summary(metrics)
    
    # Save metrics
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    
    logger.info(f"✅ Metrics saved to {args.out_json}")
    
    # Save model
    args.out_model.parent.mkdir(parents=True, exist_ok=True)
    try:
        import joblib
        joblib.dump(model, args.out_model)
        logger.info(f"✅ Model saved to {args.out_model}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        return 1
    
    # Save feature columns
    args.out_feature_cols.parent.mkdir(parents=True, exist_ok=True)
    feature_cols_data = {
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "dataset": str(args.dataset),
        "model_type": args.model_type,
        "random_state": args.random_state,
    }
    with open(args.out_feature_cols, 'w') as f:
        json.dump(feature_cols_data, f, indent=2)
    
    logger.info(f"✅ Feature columns saved to {args.out_feature_cols}")
    logger.info(f"   Total features: {len(feature_cols)}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

