#!/usr/bin/env python3
"""
ENTRY_V10 Dataset Module

PyTorch Dataset and utilities for ENTRY_V10 training.
Handles sequences with XGBoost-annotated features.

LAZY LOADING: Data is loaded on-demand by workers for parallel processing.
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from torch.utils.data import Dataset
from dataclasses import dataclass


@dataclass
class EntryV10RowSchema:
    """Schema for a single row in ENTRY_V10 dataset."""
    seq_x: np.ndarray  # [seq_len, 16] - 13 seq + 3 XGB channels
    snap_x: np.ndarray  # [88] - 85 snap + 3 XGB-now
    session_id: int  # 0=EU, 1=OVERLAP, 2=US
    vol_regime_id: int  # 0=LOW, 1=MEDIUM, 2=HIGH, 3=EXTREME
    trend_regime_id: int  # 0=UP, 1=DOWN, 2=NEUTRAL
    y_direction: int  # 0 or 1 (binary LONG edge)


class EntryV10Dataset(Dataset):
    """
    PyTorch Dataset for ENTRY_V10 training with LAZY LOADING.
    
    Data is loaded on-demand in __getitem__ to allow DataLoader workers
    to process data in parallel. Each worker loads its own copy of the DataFrame.

    Expects Parquet file with columns:
    - seq: list/array of shape [seq_len, 16]
    - snap: array of shape [88]
    - session_id: int
    - vol_regime_id: int
    - trend_regime_id: int
    - y_direction: int (0 or 1)
    """

    def __init__(
        self,
        parquet_path: str | Path,
        seq_len: int = 30,
        device: str = "cpu",
    ):
        """
        Args:
            parquet_path: Path to Parquet file with V10 dataset
            seq_len: Expected sequence length (default: 30)
            device: Device for tensors (default: "cpu")
        """
        self.parquet_path = Path(parquet_path)
        self.seq_len = seq_len
        self.device = device

        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.parquet_path}")

        # LAZY LOADING: Only load metadata, not the actual data
        # Workers will load data in parallel via __getitem__
        print(f"[ENTRY_V10_DATASET] Preparing lazy loading from {self.parquet_path}")
        
        # Get length without loading full dataset
        try:
            import pyarrow.parquet as pq
            parquet_file = pq.ParquetFile(self.parquet_path)
            self._len = parquet_file.metadata.num_rows
        except ImportError:
            # Fallback: load just to get length
            df_temp = pd.read_parquet(self.parquet_path, nrows=1)
            self._len = len(pd.read_parquet(self.parquet_path))
        
        print(f"[ENTRY_V10_DATASET] Prepared {self._len} samples (lazy loading, seq_len={seq_len})")
        print(f"[ENTRY_V10_DATASET] Each worker will load its own copy of the data")

    def __len__(self) -> int:
        return self._len

    def _parse_sequence(self, seq, seq_len: int) -> np.ndarray:
        """Parse sequence from various formats and clean NaN/Inf values."""
        if isinstance(seq, np.ndarray):
            if seq.dtype == object:
                try:
                    seq = np.array([np.array(x, dtype=np.float32) for x in seq], dtype=np.float32)
                except (ValueError, TypeError):
                    flat = np.concatenate([np.array(x, dtype=np.float32).flatten() for x in seq])
                    seq = flat.reshape(seq_len, 16)
            else:
                seq = seq.astype(np.float32)
        elif isinstance(seq, list):
            if len(seq) > 0 and isinstance(seq[0], (list, np.ndarray)):
                seq = np.array([np.array(x, dtype=np.float32) for x in seq], dtype=np.float32)
            else:
                seq = np.array(seq, dtype=np.float32)
        
        if seq.ndim == 1:
            if len(seq) == seq_len * 16:
                seq = seq.reshape(seq_len, 16)
        elif seq.ndim == 2:
            if seq.shape != (seq_len, 16) and seq.size == seq_len * 16:
                seq = seq.reshape(seq_len, 16)
        
        # Replace NaN and Inf with 0.0
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
        
        return seq.astype(np.float32)

    def _parse_snapshot(self, snap) -> np.ndarray:
        """Parse snapshot from various formats and clean NaN/Inf values."""
        if isinstance(snap, list):
            snap = np.array(snap, dtype=np.float32)
        elif isinstance(snap, np.ndarray):
            snap = snap.astype(np.float32)
        if snap.ndim > 1:
            snap = snap.flatten()
        if len(snap) != 88:
            snap = snap[:88] if len(snap) > 88 else np.pad(snap, (0, 88 - len(snap)), 'constant', constant_values=0.0)
        
        # Replace NaN and Inf with 0.0
        snap = np.nan_to_num(snap, nan=0.0, posinf=0.0, neginf=0.0)
        
        return snap.astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample - called by DataLoader workers in parallel.
        Each worker process loads its own copy of the DataFrame.
        """
        # Lazy load DataFrame per worker (each worker process gets its own copy)
        if not hasattr(self, '_df') or self._df is None:
            self._df = pd.read_parquet(self.parquet_path)
        
        row = self._df.iloc[idx]
        
        seq = self._parse_sequence(row["seq"], self.seq_len)
        snap = self._parse_snapshot(row["snap"])
        
        return {
            "seq_x": torch.tensor(seq, dtype=torch.float32).to(self.device),
            "snap_x": torch.tensor(snap, dtype=torch.float32).to(self.device),
            "session_id": torch.tensor(int(row["session_id"]), dtype=torch.long).to(self.device),
            "vol_regime_id": torch.tensor(int(row["vol_regime_id"]), dtype=torch.long).to(self.device),
            "trend_regime_id": torch.tensor(int(row["trend_regime_id"]), dtype=torch.long).to(self.device),
            "y_direction": torch.tensor(int(row["y_direction"]), dtype=torch.float32).to(self.device),
        }


def train_val_split(
    parquet_path: str | Path,
    val_frac: float = 0.2,
    by_date: bool = True,
    output_dir: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """
    Splits a Parquet dataset into training and validation sets.
    
    Args:
        parquet_path: Path to the input Parquet dataset.
        val_frac: Fraction of data to use for validation (default: 0.2).
        by_date: If True, split by date; otherwise, split randomly.
        output_dir: Directory to save split datasets. If None, uses same dir as input.
    
    Returns:
        Tuple of (train_path, val_path) for the saved datasets.
    """
    from pathlib import Path
    import pandas as pd
    
    parquet_path = Path(parquet_path)
    if output_dir is None:
        output_dir = parquet_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[TRAIN_VAL_SPLIT] Loading {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    if by_date and "ts" in df.columns:
        df = df.sort_values("ts")
        split_idx = int(len(df) * (1 - val_frac))
        df_train = df.iloc[:split_idx]
        df_val = df.iloc[split_idx:]
    else:
        df_train = df.sample(frac=(1 - val_frac), random_state=42)
        df_val = df.drop(df_train.index)
    
    train_path = output_dir / f"{parquet_path.stem}_train.parquet"
    val_path = output_dir / f"{parquet_path.stem}_val.parquet"
    
    df_train.to_parquet(train_path, index=False)
    df_val.to_parquet(val_path, index=False)
    
    print(f"[TRAIN_VAL_SPLIT] Train: {len(df_train)} samples -> {train_path}")
    print(f"[TRAIN_VAL_SPLIT] Val: {len(df_val)} samples -> {val_path}")
    
    return train_path, val_path
