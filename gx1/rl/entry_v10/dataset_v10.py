#!/usr/bin/env python3
"""
ENTRY_V10_CTX Dataset Module (ONE UNIVERSE: 6/6 + signal bridge 7/7).

PyTorch Dataset for ENTRY_V10_CTX training only.
NO LEGACY. ctx_cont_dim=6, ctx_cat_dim=6. seq_x [seq_len, 7], snap_x [7] from signal_bridge_v1.

NO FALLBACKS:
- No NaN/Inf sanitizing (data must be clean upstream).
- No padding/truncation. Any schema/shape mismatch hard-fails early.

Expected parquet columns:
  - seq_x: [seq_len, SEQ_SIGNAL_DIM]  (7)
  - snap_x: [SNAP_SIGNAL_DIM]  (7)
  - ctx_cont: [6]
  - ctx_cat: [6]
  - y_direction: 0/1
  Optional: y_early_move, y_quality_score
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


PathLike = Union[str, Path]


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def _is_finite_array(x: np.ndarray) -> bool:
    if np.issubdtype(x.dtype, np.floating):
        return bool(np.isfinite(x).all())
    return True


def _as_np_array_strict(x: Any, *, dtype: np.dtype) -> np.ndarray:
    if isinstance(x, np.ndarray):
        arr = x
    elif isinstance(x, (list, tuple)):
        arr = np.asarray(x)
    else:
        raise TypeError(f"UNSUPPORTED_TYPE: expected list/tuple/np.ndarray, got {type(x)}")
    try:
        arr = arr.astype(dtype, copy=False)
    except Exception as e:
        raise TypeError(f"DTYPE_CAST_FAIL: cannot cast to {dtype}: {e}") from e
    return arr


def _reshape_exact(arr: np.ndarray, shape: Tuple[int, ...], *, label: str) -> np.ndarray:
    if arr.shape == shape:
        return arr
    if arr.size == int(np.prod(shape)):
        return arr.reshape(shape)
    raise ValueError(f"{label}_SHAPE_MISMATCH: got shape={arr.shape} size={arr.size} expected shape={shape}")


def _parse_binary_label(y: Any, *, label: str) -> float:
    try:
        v = float(y)
    except Exception as e:
        raise TypeError(f"{label}_TYPE_FAIL: cannot cast y to float: {e}") from e
    if v not in (0.0, 1.0):
        raise ValueError(f"{label}_VALUE_FAIL: expected 0/1 got {v}")
    return v


# =============================================================================
# ENTRY_V10_CTX dataset (6/6 only)
# =============================================================================

CTX_CONT_DIM = 6
CTX_CAT_DIM = 6


class EntryV10CtxDataset(Dataset):
    """
    Dataset for ENTRY_V10_CTX (STRICT 6/6 only).

    Expects parquet columns: seq_x [seq_len, 7], snap_x [7], ctx_cont [6], ctx_cat [6], y_direction.
    ONE UNIVERSE: no 2/4/6 or 5/6. Hard-fail on other dims.
    """

    REQUIRED_COLS = ("seq_x", "snap_x", "ctx_cont", "ctx_cat", "y_direction")

    def __init__(
        self,
        parquet_path: PathLike,
        *,
        seq_len: int = 30,
        ctx_cont_dim: int = 6,
        ctx_cat_dim: int = 6,
    ):
        from gx1.contracts.signal_bridge_v1 import (
            ORDERED_CTX_CAT_NAMES_EXTENDED,
            ORDERED_CTX_CONT_NAMES_EXTENDED,
            SEQ_SIGNAL_DIM,
            SNAP_SIGNAL_DIM,
        )

        self.parquet_path = Path(parquet_path).expanduser().resolve()
        _require_file(self.parquet_path, "EntryV10CtxDataset parquet_path")

        self.seq_len = int(seq_len)
        self.ctx_cont_dim = int(ctx_cont_dim)
        self.ctx_cat_dim = int(ctx_cat_dim)

        if self.seq_len <= 0:
            raise ValueError(f"seq_len must be > 0; got {self.seq_len}")
        if self.ctx_cont_dim != CTX_CONT_DIM:
            raise ValueError(f"ctx_cont_dim must be {CTX_CONT_DIM} (ONE UNIVERSE); got {self.ctx_cont_dim}")
        if self.ctx_cat_dim != CTX_CAT_DIM:
            raise ValueError(f"ctx_cat_dim must be {CTX_CAT_DIM} (ONE UNIVERSE); got {self.ctx_cat_dim}")

        self._seq_signal_dim = int(SEQ_SIGNAL_DIM)
        self._snap_signal_dim = int(SNAP_SIGNAL_DIM)
        self._cont_names = list(ORDERED_CTX_CONT_NAMES_EXTENDED[: self.ctx_cont_dim])
        self._cat_names = list(ORDERED_CTX_CAT_NAMES_EXTENDED[: self.ctx_cat_dim])

        df = pd.read_parquet(self.parquet_path)
        for c in self.REQUIRED_COLS:
            if c not in df.columns:
                raise RuntimeError(f"CTX_DATASET_MISSING_COL: {c}")
        if len(df) <= 0:
            raise RuntimeError("CTX_DATASET_EMPTY")

        self._df = df

        r0 = df.iloc[0]
        seq0 = _as_np_array_strict(r0["seq_x"], dtype=np.float32)
        snap0 = _as_np_array_strict(r0["snap_x"], dtype=np.float32)
        cont0 = _as_np_array_strict(r0["ctx_cont"], dtype=np.float32)
        cat0 = _as_np_array_strict(r0["ctx_cat"], dtype=np.int64)

        _ = _reshape_exact(seq0, (self.seq_len, self._seq_signal_dim), label="SEQ_X")
        _ = _reshape_exact(snap0, (self._snap_signal_dim,), label="SNAP_X")
        _ = _reshape_exact(cont0, (self.ctx_cont_dim,), label="CTX_CONT")
        _ = _reshape_exact(cat0, (self.ctx_cat_dim,), label="CTX_CAT")

        if not _is_finite_array(seq0) or not _is_finite_array(snap0) or not _is_finite_array(cont0):
            raise ValueError("NON_FINITE_INPUTS: ctx dataset contains NaN/Inf (no fallback)")

        y0 = _parse_binary_label(r0["y_direction"], label="Y_DIRECTION")

        log.info(
            f"[EntryV10CtxDataset] path={self.parquet_path} rows={len(df)} "
            f"seq_len={self.seq_len} seq_dim={self._seq_signal_dim} snap_dim={self._snap_signal_dim} "
            f"ctx_cont_dim={self.ctx_cont_dim} ctx_cat_dim={self.ctx_cat_dim} y0={y0}"
        )

    def __len__(self) -> int:
        return int(len(self._df))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        i = int(idx)
        if i < 0 or i >= len(self._df):
            raise IndexError(f"INDEX_OOB: idx={i} len={len(self._df)}")

        row = self._df.iloc[i]

        seq = _as_np_array_strict(row["seq_x"], dtype=np.float32)
        seq = _reshape_exact(seq, (self.seq_len, self._seq_signal_dim), label="SEQ_X")

        snap = _as_np_array_strict(row["snap_x"], dtype=np.float32)
        snap = _reshape_exact(snap, (self._snap_signal_dim,), label="SNAP_X")

        ctx_cont = _as_np_array_strict(row["ctx_cont"], dtype=np.float32)
        ctx_cont = _reshape_exact(ctx_cont, (self.ctx_cont_dim,), label="CTX_CONT")

        ctx_cat = _as_np_array_strict(row["ctx_cat"], dtype=np.int64)
        ctx_cat = _reshape_exact(ctx_cat, (self.ctx_cat_dim,), label="CTX_CAT")

        if not _is_finite_array(seq) or not _is_finite_array(snap) or not _is_finite_array(ctx_cont):
            raise ValueError(f"NON_FINITE_INPUTS: idx={i} has NaN/Inf (no fallback)")

        y_dir = _parse_binary_label(row["y_direction"], label="Y_DIRECTION")

        out: Dict[str, Any] = {
            "seq_x": torch.tensor(seq, dtype=torch.float32),
            "snap_x": torch.tensor(snap, dtype=torch.float32),
            "ctx_cont": torch.tensor(ctx_cont, dtype=torch.float32),
            "ctx_cat": torch.tensor(ctx_cat, dtype=torch.long),
            "y_direction": torch.tensor(y_dir, dtype=torch.float32),
        }

        if "y_early_move" in self._df.columns:
            out["y_early_move"] = torch.tensor(float(row["y_early_move"]), dtype=torch.float32)
        if "y_quality_score" in self._df.columns:
            out["y_quality_score"] = torch.tensor(float(row["y_quality_score"]), dtype=torch.float32)

        return out


# =============================================================================
# Train/val split utility
# =============================================================================

def train_val_split(
    parquet_path: PathLike,
    *,
    val_frac: float = 0.2,
    by_date: bool = True,
    output_dir: Optional[PathLike] = None,
    seed: int = 42,
    ts_col: str = "ts",
) -> Tuple[Path, Path]:
    """
    Split a parquet dataset into train/val (STRICT, deterministic).
    Returns (train_path, val_path).
    """
    parquet_path = Path(parquet_path).expanduser().resolve()
    _require_file(parquet_path, "train_val_split parquet_path")

    if not (0.0 < float(val_frac) < 1.0):
        raise ValueError(f"val_frac must be in (0,1); got {val_frac}")

    if output_dir is None:
        out_dir = parquet_path.parent
    else:
        out_dir = Path(output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(parquet_path)
    if len(df) <= 0:
        raise RuntimeError("SPLIT_FAIL: dataset empty")

    if by_date:
        if ts_col not in df.columns:
            raise RuntimeError(f"SPLIT_FAIL: by_date=1 requires column '{ts_col}'")
        df = df.sort_values(ts_col, kind="mergesort")
        split_idx = int(len(df) * (1.0 - float(val_frac)))
        if split_idx <= 0 or split_idx >= len(df):
            raise RuntimeError(f"SPLIT_FAIL: invalid split_idx={split_idx} for n={len(df)}")
        df_train = df.iloc[:split_idx].copy()
        df_val = df.iloc[split_idx:].copy()
    else:
        df_train = df.sample(frac=(1.0 - float(val_frac)), random_state=int(seed)).copy()
        df_val = df.drop(df_train.index).copy()

    train_path = out_dir / f"{parquet_path.stem}_train.parquet"
    val_path = out_dir / f"{parquet_path.stem}_val.parquet"

    df_train.to_parquet(train_path, index=False)
    df_val.to_parquet(val_path, index=False)

    log.info(f"[train_val_split] train={len(df_train)} -> {train_path}")
    log.info(f"[train_val_split] val={len(df_val)} -> {val_path}")

    return train_path, val_path
