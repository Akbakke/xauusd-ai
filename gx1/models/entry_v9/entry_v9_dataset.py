"""
ENTRY_V9 Dataset Module

Handles data loading, feature preparation, and label generation for ENTRY_V9 training.
Combines sequence features, snapshot features, and regime embeddings.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from joblib import dump, load

from gx1.seq.sequence_features import build_sequence_features, get_sequence_feature_names
from gx1.tuning.entry_v4_train import load_dataset, time_split_by_dates
from gx1.models.entry_v9.entry_v9_labeler import generate_entry_v9_labels, compute_label_statistics


class EntryV9Dataset(Dataset):
    """
    PyTorch Dataset for ENTRY_V9 training.

    Returns:
        - seq_x: [seq_len, n_seq_features]
        - snap_x: [n_snapshot_features]
        - session_id: [1] (0=EU, 1=OVERLAP, 2=US)
        - vol_regime_id: [1] (0=LOW, 1=MEDIUM, 2=HIGH, 3=EXTREME)
        - trend_regime_id: [1] (0=UP, 1=DOWN, 2=RANGE)
        - y_direction: [1] (0 or 1)
        - y_early_move: [1] (0 or 1)
        - y_quality_score: [1] (float, -1 to 1)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        seq_feature_names: List[str],
        snap_feature_names: List[str],
        seq_len: int = 30,
        lookback: int = 30,
    ):
        """
        Args:
            df: DataFrame with features and labels
            seq_feature_names: List of sequence feature column names
            snap_feature_names: List of snapshot feature column names
            seq_len: Sequence length (default: 30)
            lookback: Lookback window for sequences (default: 30)
        """
        self.df = df.reset_index(drop=True)
        self.seq_feature_names = seq_feature_names
        self.snap_feature_names = snap_feature_names
        self.seq_len = seq_len
        self.lookback = lookback

        # Validate required columns
        required_cols = (
            seq_feature_names
            + snap_feature_names
            + ["session_id", "atr_regime_id", "trend_regime_tf24h", "y_direction", "y_early_move", "y_quality_score"]
        )
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise KeyError(f"Missing required columns: {missing}")

        # Filter to rows with enough history
        self.valid_indices = []
        for i in range(lookback - 1, len(self.df)):
            # Check if we have enough history
            if i >= lookback - 1:
                self.valid_indices.append(i)

        print(f"[ENTRY_V9_DATASET] Created dataset with {len(self.valid_indices)} valid samples")
        print(f"[ENTRY_V9_DATASET] Sequence features: {len(seq_feature_names)}")
        print(f"[ENTRY_V9_DATASET] Snapshot features: {len(snap_feature_names)}")

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Dict with tensors for model input and labels
        """
        i = self.valid_indices[idx]

        # Sequence features: [lookback, n_seq_features]
        # Get exactly lookback elements ending at index i (inclusive)
        seq_start = max(0, i - self.lookback + 1)
        seq_end = i
        # iloc is exclusive on end, so seq_end+1 gives us elements from seq_start to seq_end (inclusive)
        # This gives us exactly (seq_end - seq_start + 1) = lookback elements when seq_start = i - lookback + 1
        seq_data = self.df.iloc[seq_start:seq_end+1][self.seq_feature_names].values
        seq_x = torch.FloatTensor(seq_data)  # Should be [lookback, n_seq_features] or less

        # Pad if needed (for rows near the start where we don't have full lookback)
        if seq_x.shape[0] < self.lookback:
            padding = torch.zeros(self.lookback - seq_x.shape[0], seq_x.shape[1])
            seq_x = torch.cat([padding, seq_x], dim=0)
        
        # CRITICAL: Truncate if too long (iloc can sometimes give us one extra element)
        # This ensures we always have exactly lookback elements
        if seq_x.shape[0] > self.lookback:
            seq_x = seq_x[-self.lookback:]

        # Snapshot features: [n_snapshot_features]
        # Get as pandas Series first (so we can use fillna if needed)
        snap_series = self.df.iloc[i][self.snap_feature_names]
        # Convert to numeric, handling any object dtype
        snap_series = pd.to_numeric(snap_series, errors='coerce')
        # Fill NaN with 0 (pandas Series has fillna)
        snap_series = snap_series.fillna(0.0)
        # Convert to numpy array and then to float32
        snap_data = snap_series.values.astype(np.float32)
        snap_x = torch.FloatTensor(snap_data)

        # Regime IDs (convert to int)
        session_id = int(self.df.iloc[i]["session_id"])
        vol_regime_id = int(self.df.iloc[i]["atr_regime_id"])

        # Trend regime: map from float to int (0=UP, 1=DOWN, 2=RANGE)
        trend_val = float(self.df.iloc[i]["trend_regime_tf24h"])
        if trend_val > 0.001:
            trend_regime_id = 0  # UP
        elif trend_val < -0.001:
            trend_regime_id = 1  # DOWN
        else:
            trend_regime_id = 2  # RANGE

        # Labels
        y_direction = torch.tensor(int(self.df.iloc[i]["y_direction"]), dtype=torch.long)
        y_early_move = torch.tensor(int(self.df.iloc[i]["y_early_move"]), dtype=torch.long)
        y_quality_score = torch.tensor(float(self.df.iloc[i]["y_quality_score"]), dtype=torch.float32)

        return {
            "seq_x": seq_x,
            "snap_x": snap_x,
            "session_id": torch.tensor(session_id, dtype=torch.long),
            "vol_regime_id": torch.tensor(vol_regime_id, dtype=torch.long),
            "trend_regime_id": torch.tensor(trend_regime_id, dtype=torch.long),
            "y_direction": y_direction,
            "y_early_move": y_early_move,
            "y_quality_score": y_quality_score,
        }


def prepare_entry_v9_data(
    config: Dict[str, Any],
    generate_labels: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load and prepare data for ENTRY_V9 training.

    Args:
        config: Configuration dict with dataset settings
        generate_labels: If True, generate labels using entry_v9_labeler

    Returns:
        (df_prepared, metadata_dict)
    """
    # Load dataset
    df = load_dataset(config)

    if df is None or len(df) == 0:
        raise ValueError("Failed to load dataset or dataset is empty")

    print(f"[ENTRY_V9] Loaded {len(df)} rows from dataset")

    # Build sequence features
    df = build_sequence_features(df.copy())
    print(f"[ENTRY_V9] Built sequence features: {len(df)} rows")

    # Get feature names
    seq_feature_names = get_sequence_feature_names()
    print(f"[ENTRY_V9] Sequence features: {len(seq_feature_names)}")

    # Snapshot features: tabular features (exclude sequence features and metadata)
    exclude_cols = set(seq_feature_names) | {
        "ts",
        "y",
        "y_direction",
        "y_early_move",
        "y_quality_score",
        "MFE_bps",
        "MAE_bps",
        "first_hit",
        "session_id",
        "atr_regime_id",
        "trend_regime_tf24h",
        "brain_vol_regime",
        "brain_trend_regime",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    # Get snapshot features (all numeric columns not in exclude)
    # Filter to only numeric columns that can be converted to float
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    snap_feature_names = [c for c in numeric_cols if c not in exclude_cols]
    
    # LEAKAGE GUARD: Filter out features with forbidden substrings
    forbidden_substrings = ["mfe", "mae", "net_margin", "pnl", "label", "t_mfe", "t_mae", "first_hit"]
    original_count = len(snap_feature_names)
    
    leakage_features = []
    clean_snap_features = []
    for feat in snap_feature_names:
        feat_lower = feat.lower()
        is_leakage = any(forbidden in feat_lower for forbidden in forbidden_substrings)
        if is_leakage:
            leakage_features.append(feat)
        else:
            clean_snap_features.append(feat)
    
    snap_feature_names = clean_snap_features
    
    if leakage_features:
        print(f"[ENTRY_V9] ⚠️  LEAKAGE GUARD: Dropping {len(leakage_features)} potential leakage features:")
        for feat in sorted(leakage_features):
            print(f"     - {feat}")
        print(f"[ENTRY_V9] Features before leakage guard: {original_count}, after: {len(snap_feature_names)}")
    
    # Ensure all snapshot features are actually numeric (convert object columns)
    valid_snap_features = []
    for feat in snap_feature_names:
        if feat in df.columns:
            # Convert to numeric if needed
            if df[feat].dtype == 'object':
                # Try to convert to numeric, replace NaN with 0
                df[feat] = pd.to_numeric(df[feat], errors='coerce')
                df[feat] = df[feat].fillna(0.0)  # pandas Series fillna
            # Ensure float32
            df[feat] = df[feat].astype(np.float32)
            valid_snap_features.append(feat)
    snap_feature_names = valid_snap_features

    print(f"[ENTRY_V9] Snapshot features: {len(snap_feature_names)}")

    # Ensure regime columns exist
    if "session_id" not in df.columns:
        # Try to infer from session column
        if "session" in df.columns:
            session_map = {"EU": 0, "OVERLAP": 1, "US": 2}
            df["session_id"] = df["session"].map(session_map).fillna(1)  # Default to OVERLAP
        else:
            raise KeyError("Missing session_id or session column")

    if "atr_regime_id" not in df.columns:
        # Try to infer from brain_vol_regime
        if "brain_vol_regime" in df.columns:
            vol_map = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "EXTREME": 3}
            df["atr_regime_id"] = df["brain_vol_regime"].map(vol_map).fillna(2)  # Default to HIGH
        else:
            raise KeyError("Missing atr_regime_id or brain_vol_regime column")

    if "trend_regime_tf24h" not in df.columns:
        # Try to infer from brain_trend_regime
        if "brain_trend_regime" in df.columns:
            # Map to numeric: UP=positive, DOWN=negative, RANGE=zero
            trend_map = {"TREND_UP": 0.01, "TREND_DOWN": -0.01, "RANGE": 0.0}
            df["trend_regime_tf24h"] = df["brain_trend_regime"].map(trend_map).fillna(0.0)
        else:
            # Default to zero (RANGE)
            df["trend_regime_tf24h"] = 0.0

    # Generate labels if requested
    if generate_labels:
        label_cfg = config.get("labels", {})
        df = generate_entry_v9_labels(
            df,
            horizon_bars=label_cfg.get("horizon_bars", 10),
            mfe_col=label_cfg.get("mfe_col", "MFE_bps"),
            mae_col=label_cfg.get("mae_col", "MAE_bps"),
            first_hit_col=label_cfg.get("first_hit_col", "first_hit"),
            y_direction_col=label_cfg.get("y_direction_col", "y"),
            target_mfe_bps=label_cfg.get("target_mfe_bps", 20.0),
        )

        # Print label statistics
        stats = compute_label_statistics(df)
        print(f"[ENTRY_V9] Label statistics:")
        print(f"  Direction: coverage={stats.get('direction_coverage', 0):.4f}, "
              f"positive={stats.get('direction_positive', 0)}/{stats.get('direction_total', 0)}")
        print(f"  Early move: coverage={stats.get('early_move_coverage', 0):.4f}, "
              f"positive={stats.get('early_move_positive', 0)}/{stats.get('early_move_total', 0)}")
        print(f"  Quality score: mean={stats.get('quality_score_mean', 0):.4f}, "
              f"std={stats.get('quality_score_std', 0):.4f}")

    # Metadata
    metadata = {
        "seq_feature_names": seq_feature_names,
        "snap_feature_names": snap_feature_names,
        "n_seq_features": len(seq_feature_names),
        "n_snap_features": len(snap_feature_names),
        "n_samples": len(df),
    }

    return df, metadata


def create_entry_v9_dataloaders(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    seq_feature_names: List[str],
    snap_feature_names: List[str],
    seq_len: int = 30,
    batch_size: int = 64,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training and validation.

    Args:
        df_train: Training DataFrame
        df_val: Validation DataFrame
        seq_feature_names: List of sequence feature names
        snap_feature_names: List of snapshot feature names
        seq_len: Sequence length
        batch_size: Batch size
        num_workers: Number of DataLoader workers

    Returns:
        (train_loader, val_loader)
    """
    train_dataset = EntryV9Dataset(
        df_train, seq_feature_names, snap_feature_names, seq_len=seq_len, lookback=seq_len
    )
    val_dataset = EntryV9Dataset(
        df_val, seq_feature_names, snap_feature_names, seq_len=seq_len, lookback=seq_len
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader

