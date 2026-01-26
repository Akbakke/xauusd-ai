#!/usr/bin/env python3
"""
Run XGBoost inference over historical entry dataset.

Annotates V9 entry dataset with XGBoost predictions (p_long_xgb, margin_xgb, etc.)
for use in ENTRY_V10 pipeline.

Input:
    - V9 entry dataset: data/entry_v9/full_2020_2025.parquet
    - XGBoost models: models/entry_v10/xgb_entry_{EU,US,OVERLAP}_v10.joblib
    - Feature metadata: gx1/models/entry_v9/nextgen_2020_2025_clean/entry_v9_feature_meta.json

Output:
    - data/entry_v10/xgb_annotated.parquet
        Adds columns:
        - p_long_xgb, p_short_xgb, p_neutral_xgb
        - margin_xgb, p_hat_xgb
        - p_long_xgb_ema_5 (EMA-smoothed)

Usage:
    python -m gx1.rl.entry_v10.run_xgb_inference_v10 \
        --input-parquet data/entry_v9/full_2020_2025.parquet \
        --output-parquet data/entry_v10/xgb_annotated.parquet \
        --xgb-model-dir models/entry_v10 \
        --feature-meta gx1/models/entry_v9/nextgen_2020_2025_clean/entry_v9_feature_meta.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_xgb_models(model_dir: Path) -> Dict[str, any]:
    """
    Load session-routed XGBoost models.

    Args:
        model_dir: Directory containing XGBoost models

    Returns:
        Dict mapping session -> model
    """
    models = {}
    sessions = ["EU", "US", "OVERLAP"]

    for session in sessions:
        # Try V10 model path first
        model_path = model_dir / f"xgb_entry_{session}_v10.joblib"
        if not model_path.exists():
            # Try legacy path structure
            model_path = model_dir / f"GX1_entry_{session}.joblib"
            if not model_path.exists():
                # Try alternative legacy path
                model_path = Path(f"gx1/models/GX1_entry_{session}.joblib")
                if not model_path.exists():
                    raise FileNotFoundError(
                        f"XGBoost model not found for {session}. "
                        f"Tried: {model_dir / f'xgb_entry_{session}_v10.joblib'}, "
                        f"{model_dir / f'GX1_entry_{session}.joblib'}, "
                        f"gx1/models/GX1_entry_{session}.joblib"
                    )

        log.info(f"Loading XGBoost model for {session}: {model_path}")
        model = joblib.load(model_path)
        models[session] = model
        log.info(f"  - Classes: {getattr(model, 'classes_', None)}")
        log.info(f"  - n_features_in_: {getattr(model, 'n_features_in_', 'unknown')}")

    return models


def infer_session(row: pd.Series) -> str:
    """
    Infer session from row data.

    Args:
        row: DataFrame row with session indicators

    Returns:
        Session string: "EU", "US", or "OVERLAP"
    """
    # Try explicit session column
    if "session" in row.index:
        session = str(row["session"]).upper()
        if session in ["EU", "US", "OVERLAP"]:
            return session

    # Try session tag columns
    if "_v1_session_tag_EU" in row.index and row.get("_v1_session_tag_EU", 0) > 0.5:
        return "EU"
    if "_v1_session_tag_US" in row.index and row.get("_v1_session_tag_US", 0) > 0.5:
        return "US"
    if "_v1_session_tag_OVERLAP" in row.index and row.get("_v1_session_tag_OVERLAP", 0) > 0.5:
        return "OVERLAP"

    # Try session_id (0=EU, 1=OVERLAP, 2=US)
    if "session_id" in row.index:
        session_id = int(row["session_id"])
        if session_id == 0:
            return "EU"
        elif session_id == 1:
            return "OVERLAP"
        elif session_id == 2:
            return "US"

    # Default to OVERLAP if uncertain
    log.warning(f"Could not infer session for row, defaulting to OVERLAP")
    return "OVERLAP"


def run_xgb_inference(
    input_parquet: Path,
    output_parquet: Path,
    xgb_model_dir: Path,
    feature_meta_path: Path,
    batch_size: int = 10000,
) -> None:
    """
    Run XGBoost inference over historical dataset.

    Args:
        input_parquet: Path to V9 entry dataset (Parquet)
        output_parquet: Path to save annotated dataset
        xgb_model_dir: Directory containing XGBoost models
        feature_meta_path: Path to entry_v9_feature_meta.json
        batch_size: Batch size for processing (default: 10000)
    """
    log.info(f"Loading input dataset: {input_parquet}")
    df = pd.read_parquet(input_parquet)
    log.info(f"Loaded {len(df)} rows")

    # Load feature metadata
    with open(feature_meta_path, "r", encoding="utf-8") as f:
        feature_meta = json.load(f)

    snap_features = feature_meta.get("snap_features", [])
    log.info(f"Using {len(snap_features)} snapshot features")

    # Validate that all snapshot features exist
    missing = [f for f in snap_features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing snapshot features: {missing[:10]}...")

    # Load XGBoost models
    models = load_xgb_models(xgb_model_dir)

    # Initialize XGB output columns
    df["p_long_xgb"] = np.nan
    df["p_short_xgb"] = np.nan
    df["p_neutral_xgb"] = np.nan
    df["margin_xgb"] = np.nan
    df["p_hat_xgb"] = np.nan

    # Process in batches
    n_batches = (len(df) + batch_size - 1) // batch_size
    log.info(f"Processing {len(df)} rows in {n_batches} batches")

    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(df))
        batch_df = df.iloc[start_idx:end_idx].copy()

        log.info(f"Processing batch {batch_idx + 1}/{n_batches} (rows {start_idx}-{end_idx})")

        # Group by session for efficient processing
        for session in ["EU", "US", "OVERLAP"]:
            # Infer session for each row
            session_mask = batch_df.apply(lambda row: infer_session(row) == session, axis=1)
            session_rows = batch_df[session_mask]

            if len(session_rows) == 0:
                continue

            # Extract snapshot features
            # Handle non-numeric values
            X_df = session_rows[snap_features].copy()
            for col in X_df.columns:
                if X_df[col].dtype == 'object':
                    X_df[col] = pd.to_numeric(X_df[col], errors='coerce').fillna(0.0)
                X_df[col] = X_df[col].astype(np.float32)
            X = X_df.values.astype(np.float32)

            # Run XGBoost prediction
            model = models[session]
            proba = model.predict_proba(X)  # [N, n_classes]

            # Map probabilities based on model classes
            classes = model.classes_
            n_classes = len(classes)

            if n_classes == 2:
                # Binary classification: [class_0, class_1]
                # Assume class_1 is LONG, class_0 is SHORT
                if "LONG" in str(classes[1]).upper() or "1" in str(classes[1]):
                    p_long = proba[:, 1]
                    p_short = proba[:, 0]
                else:
                    p_long = proba[:, 0]
                    p_short = proba[:, 1]
                p_neutral = np.zeros(len(session_rows))
            elif n_classes == 3:
                # Multi-class: [NEUTRAL, SHORT, LONG] or similar
                # Find indices
                long_idx = None
                short_idx = None
                neutral_idx = None
                for i, cls in enumerate(classes):
                    cls_str = str(cls).upper()
                    if "LONG" in cls_str or "1" in cls_str:
                        long_idx = i
                    elif "SHORT" in cls_str or "-1" in cls_str or "0" in cls_str:
                        short_idx = i
                    else:
                        neutral_idx = i

                if long_idx is None or short_idx is None:
                    # Fallback: assume order is [NEUTRAL, SHORT, LONG]
                    neutral_idx = 0
                    short_idx = 1
                    long_idx = 2

                p_long = proba[:, long_idx] if long_idx is not None else np.zeros(len(session_rows))
                p_short = proba[:, short_idx] if short_idx is not None else np.zeros(len(session_rows))
                p_neutral = proba[:, neutral_idx] if neutral_idx is not None else np.zeros(len(session_rows))
            else:
                raise ValueError(f"Unexpected number of classes: {n_classes}, classes: {classes}")

            # Compute derived metrics
            margin_xgb = np.abs(p_long - p_short)
            p_hat_xgb = np.maximum(p_long, p_short)

            # Update DataFrame
            df.loc[session_rows.index, "p_long_xgb"] = p_long
            df.loc[session_rows.index, "p_short_xgb"] = p_short
            df.loc[session_rows.index, "p_neutral_xgb"] = p_neutral
            df.loc[session_rows.index, "margin_xgb"] = margin_xgb
            df.loc[session_rows.index, "p_hat_xgb"] = p_hat_xgb

        if (batch_idx + 1) % 10 == 0:
            log.info(f"Completed {batch_idx + 1}/{n_batches} batches")

    # Compute EMA-smoothed versions
    log.info("Computing EMA-smoothed XGB signals")
    df = df.sort_values("ts" if "ts" in df.columns else df.index)
    df["p_long_xgb_ema_5"] = df.groupby("session" if "session" in df.columns else None)["p_long_xgb"].transform(
        lambda x: x.ewm(span=5, adjust=False).mean()
    )

    # Ensure output directory exists
    output_parquet.parent.mkdir(parents=True, exist_ok=True)

    # Save annotated dataset
    log.info(f"Saving annotated dataset to {output_parquet}")
    df.to_parquet(output_parquet, index=False)
    log.info(f"Saved {len(df)} rows with XGBoost annotations")

    # Print summary statistics
    log.info("XGBoost inference summary:")
    log.info(f"  - p_long_xgb: mean={df['p_long_xgb'].mean():.4f}, std={df['p_long_xgb'].std():.4f}")
    log.info(f"  - margin_xgb: mean={df['margin_xgb'].mean():.4f}, std={df['margin_xgb'].std():.4f}")
    log.info(f"  - p_hat_xgb: mean={df['p_hat_xgb'].mean():.4f}, std={df['p_hat_xgb'].std():.4f}")


def main():
    parser = argparse.ArgumentParser(description="Run XGBoost inference over historical entry dataset")
    parser.add_argument(
        "--input-parquet",
        type=Path,
        required=True,
        help="Path to V9 entry dataset (Parquet)",
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        required=True,
        help="Path to save annotated dataset",
    )
    parser.add_argument(
        "--xgb-model-dir",
        type=Path,
        default=Path("gx1/models"),
        help="Directory containing XGBoost models (default: gx1/models)",
    )
    parser.add_argument(
        "--feature-meta",
        type=Path,
        default=Path("gx1/models/entry_v9/nextgen_2020_2025_clean/entry_v9_feature_meta.json"),
        help="Path to entry_v9_feature_meta.json",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Batch size for processing (default: 10000)",
    )

    args = parser.parse_args()

    run_xgb_inference(
        input_parquet=args.input_parquet,
        output_parquet=args.output_parquet,
        xgb_model_dir=args.xgb_model_dir,
        feature_meta_path=args.feature_meta,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

