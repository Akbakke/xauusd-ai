#!/usr/bin/env python3
"""
Build ENTRY_V10 dataset from XGBoost-annotated entry data.

Creates sequences with:
- 13 sequence features + 3 XGB channels = 16 total seq features
- 85 snapshot features + 3 XGB-now = 88 total snap features

Input:
    - XGBoost-annotated dataset: data/entry_v10/xgb_annotated.parquet
    - Feature metadata: gx1/models/entry_v9/nextgen_2020_2025_clean/entry_v9_feature_meta.json

Output:
    - data/entry_v10/entry_v10_dataset.parquet (V10 with seq_len=30)
    - data/entry_v10/entry_v10_1_dataset_seq90.parquet (V10.1 with seq_len=90)
        Contains:
        - seq: [N, lookback, 16] sequences (13 seq + 3 XGB channels)
        - snap: [N, 88] snapshots (85 snap + 3 XGB-now)
        - session_id, vol_regime_id, trend_regime_id
        - y_direction (binary labels)

Usage:
    # V10 (seq_len=30)
    python -m gx1.rl.entry_v10.build_entry_v10_dataset \
        --input-parquet data/entry_v10/xgb_annotated.parquet \
        --output-parquet data/entry_v10/entry_v10_dataset.parquet \
        --feature-meta gx1/models/entry_v9/nextgen_2020_2025_clean/entry_v9_feature_meta.json \
        --lookback 30
    
    # V10.1 (seq_len=90)
    python -m gx1.rl.entry_v10.build_entry_v10_dataset \
        --input-parquet data/entry_v10/xgb_annotated.parquet \
        --output-parquet data/entry_v10/entry_v10_1_dataset_seq90.parquet \
        --feature-meta gx1/models/entry_v9/nextgen_2020_2025_clean/entry_v9_feature_meta.json \
        --lookback 90

Note:
    V10.1 uses longer sequences (90 bars = ~7.5 hours on M5) for improved temporal context.
    Rows without sufficient history (< lookback bars) are dropped (no dummy/zero padding).
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_feature_meta(feature_meta_path: Path) -> Tuple[List[str], List[str]]:
    """Load feature metadata and return (seq_features, snap_features)."""
    with open(feature_meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    seq_features = meta.get("seq_features", [])
    snap_features = meta.get("snap_features", [])

    if not seq_features or not snap_features:
        raise ValueError(f"Missing seq_features or snap_features in {feature_meta_path}")

    return seq_features, snap_features


def build_entry_v10_dataset(
    input_parquet: Path,
    output_parquet: Path,
    feature_meta_path: Path,
    lookback: int = 30,
) -> None:
    """
    Build ENTRY_V10 dataset with sequences and XGBoost annotations.

    Args:
        input_parquet: Path to XGBoost-annotated dataset (from run_xgb_inference_v10.py)
        output_parquet: Path to save V10 dataset
        feature_meta_path: Path to entry_v9_feature_meta.json
        lookback: Sequence length (default: 30)
    """
    log.info(f"Loading input dataset: {input_parquet}")
    df = pd.read_parquet(input_parquet)
    log.info(f"Loaded {len(df)} rows")

    # Load feature metadata
    seq_features, snap_features = load_feature_meta(feature_meta_path)
    log.info(f"Sequence features: {len(seq_features)}")
    log.info(f"Snapshot features: {len(snap_features)}")

    # Validate required columns
    required_cols = (
        seq_features
        + snap_features
        + ["p_long_xgb", "margin_xgb", "p_hat_xgb", "p_long_xgb_ema_5"]
    )
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing[:10]}...")

    # Sort by timestamp
    if "ts" in df.columns:
        df = df.sort_values("ts").reset_index(drop=True)
    elif "time" in df.columns:
        df = df.sort_values("time").reset_index(drop=True)
    else:
        log.warning("No timestamp column found, using index order")

    # Ensure regime columns exist
    if "session_id" not in df.columns:
        # Infer from session tags
        if "_v1_session_tag_EU" in df.columns:
            df["session_id"] = (
                df["_v1_session_tag_EU"].apply(lambda x: 0 if x > 0.5 else None)
                .fillna(df["_v1_session_tag_OVERLAP"].apply(lambda x: 1 if x > 0.5 else None))
                .fillna(df["_v1_session_tag_US"].apply(lambda x: 2 if x > 0.5 else None))
                .fillna(1)  # Default to OVERLAP
            )
        else:
            raise ValueError("Cannot infer session_id - missing session columns")

    if "atr_regime_id" not in df.columns:
        if "_v1_atr_regime_id" in df.columns:
            df["atr_regime_id"] = df["_v1_atr_regime_id"]
        else:
            log.warning("atr_regime_id not found, defaulting to 1 (MEDIUM)")
            df["atr_regime_id"] = 1

    if "trend_regime_tf24h" in df.columns:
        # Map trend_regime_tf24h to trend_regime_id
        # Assuming: 0=DOWN, 1=NEUTRAL, 2=UP
        df["trend_regime_id"] = df["trend_regime_tf24h"].fillna(1).astype(int)
    elif "trend_regime_id" not in df.columns:
        log.warning("trend_regime_id not found, defaulting to 1 (NEUTRAL)")
        df["trend_regime_id"] = 1

    # Ensure y_direction exists
    if "y_direction" not in df.columns:
        if "y" in df.columns:
            df["y_direction"] = (df["y"] > 0).astype(int)
        else:
            raise ValueError("Missing y_direction or y column for labels")

    # Build sequences
    log.info(f"Building sequences with lookback={lookback}")
    log.info(f"Rows with insufficient history (< {lookback} bars) will be dropped (no padding)")
    sequences = []
    snapshots = []
    session_ids = []
    vol_regime_ids = []
    trend_regime_ids = []
    y_directions = []

    n_valid = 0
    n_skipped = 0

    for i in range(lookback - 1, len(df)):
        # Extract sequence window
        seq_window = df.iloc[i - lookback + 1 : i + 1]

        # Check if we have enough history
        # V10.1 design: NO dummy/zero padding - drop rows without sufficient history
        if len(seq_window) < lookback:
            n_skipped += 1
            continue

        # Build sequence: [lookback, 16]
        # 13 seq features + 3 XGB channels
        seq_data = np.zeros((lookback, 16), dtype=np.float32)

        # Fill sequence features (13)
        for j, feat in enumerate(seq_features):
            if feat in seq_window.columns:
                # Handle non-numeric values
                seq_vals = seq_window[feat].copy()
                if seq_vals.dtype == 'object':
                    seq_vals = pd.to_numeric(seq_vals, errors='coerce').fillna(0.0)
                seq_data[:, j] = seq_vals.values.astype(np.float32)
            else:
                log.warning(f"Sequence feature {feat} not found at row {i}")
                seq_data[:, j] = 0.0

        # Fill XGB sequence channels (3)
        seq_data[:, 13] = seq_window["p_long_xgb"].values.astype(np.float32)
        seq_data[:, 14] = seq_window["margin_xgb"].values.astype(np.float32)
        seq_data[:, 15] = seq_window["p_long_xgb_ema_5"].values.astype(np.float32)

        # Build snapshot: [88]
        # 85 snap features + 3 XGB-now
        snap_data = np.zeros(88, dtype=np.float32)

        # Fill snapshot features (85)
        current_row = df.iloc[i]
        for j, feat in enumerate(snap_features):
            if feat in current_row.index:
                val = current_row[feat]
                # Handle non-numeric values
                try:
                    snap_data[j] = float(val)
                except (ValueError, TypeError):
                    # Try to convert to numeric, default to 0.0
                    snap_data[j] = float(pd.to_numeric(val, errors='coerce') or 0.0)
            else:
                log.warning(f"Snapshot feature {feat} not found at row {i}")
                snap_data[j] = 0.0

        # Fill XGB-now features (3)
        snap_data[85] = float(current_row["p_long_xgb"])
        snap_data[86] = float(current_row["margin_xgb"])
        snap_data[87] = float(current_row["p_hat_xgb"])

        # Extract regime and label
        session_id = int(current_row["session_id"])
        vol_regime_id = int(current_row["atr_regime_id"])
        trend_regime_id = int(current_row["trend_regime_id"])
        y_direction = int(current_row["y_direction"])

        sequences.append(seq_data)
        snapshots.append(snap_data)
        session_ids.append(session_id)
        vol_regime_ids.append(vol_regime_id)
        trend_regime_ids.append(trend_regime_id)
        y_directions.append(y_direction)

        n_valid += 1

        if (i + 1) % 10000 == 0:
            log.info(f"Processed {i + 1}/{len(df)} rows, {n_valid} valid sequences")

    log.info(f"Built {n_valid} valid sequences (skipped {n_skipped})")

    # Create output DataFrame
    output_df = pd.DataFrame({
        "seq": [seq.tolist() for seq in sequences],  # Convert to list for Parquet
        "snap": [snap.tolist() for snap in snapshots],
        "session_id": session_ids,
        "vol_regime_id": vol_regime_ids,
        "trend_regime_id": trend_regime_ids,
        "y_direction": y_directions,
    })

    # Add timestamp if available
    if "ts" in df.columns:
        output_df["ts"] = df.iloc[lookback - 1 :]["ts"].values
    elif "time" in df.columns:
        output_df["time"] = df.iloc[lookback - 1 :]["time"].values

    # Save dataset
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Saving V10 dataset to {output_parquet}")
    output_df.to_parquet(output_parquet, index=False)
    log.info(f"Saved {len(output_df)} sequences")

    # Print summary
    log.info("Dataset summary:")
    log.info(f"  - Sequences: {len(sequences)}")
    log.info(f"  - Sequence shape: ({lookback}, 16)")
    log.info(f"  - Snapshot shape: (88,)")
    log.info(f"  - Session distribution: {pd.Series(session_ids).value_counts().to_dict()}")
    log.info(f"  - Vol regime distribution: {pd.Series(vol_regime_ids).value_counts().to_dict()}")
    log.info(f"  - Trend regime distribution: {pd.Series(trend_regime_ids).value_counts().to_dict()}")
    log.info(f"  - Label distribution: {pd.Series(y_directions).value_counts().to_dict()}")


def main():
    parser = argparse.ArgumentParser(description="Build ENTRY_V10 dataset from XGBoost-annotated data")
    parser.add_argument(
        "--input-parquet",
        type=Path,
        required=True,
        help="Path to XGBoost-annotated dataset",
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        required=True,
        help="Path to save V10 dataset",
    )
    parser.add_argument(
        "--feature-meta",
        type=Path,
        default=Path("gx1/models/entry_v9/nextgen_2020_2025_clean/entry_v9_feature_meta.json"),
        help="Path to entry_v9_feature_meta.json",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=30,
        help="Sequence length (default: 30)",
    )

    args = parser.parse_args()

    build_entry_v10_dataset(
        input_parquet=args.input_parquet,
        output_parquet=args.output_parquet,
        feature_meta_path=args.feature_meta,
        lookback=args.lookback,
    )


if __name__ == "__main__":
    main()

