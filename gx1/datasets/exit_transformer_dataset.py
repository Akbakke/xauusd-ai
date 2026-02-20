"""
Exit Transformer dataset – IOV3_CLEAN only (ONE PATH).

Builds (X, y) from exits_<run_id>.jsonl using gx1.contracts.exit_io.
Legacy IOV1/IOV2 disabled: use_io_v2 or IOV1/IOV2 → RuntimeError.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from gx1.contracts.exit_io import (
    EXIT_IO_FEATURE_COUNT,
    EXIT_IO_VERSION,
    row_to_feature_vector_v3,
    validate_window_v3,
)

DEFAULT_WINDOW_LEN = 64
HOLDOUT_MOD = 10
HOLDOUT_VAL_FOLD = 9
ONE_UNIVERSE_CTX_CONT_DIM = 6
ONE_UNIVERSE_CTX_CAT_DIM = 6


def _trade_id_to_fold(trade_id: str) -> int:
    """Deterministic fold 0..HOLDOUT_MOD-1 from trade_id."""
    return hash(trade_id) % HOLDOUT_MOD


def load_exits_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load and return list of records (one per line). Deterministic order (file order)."""
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def build_sequences_and_labels(
    records: List[Dict[str, Any]],
    window_len: int = DEFAULT_WINDOW_LEN,
    stride: int = 1,
    context: str = "unknown",
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], str, int]:
    """
    Group by trade_id, sort by ts within trade, build windows and labels.
    IOV3_CLEAN only: uses row_to_feature_vector_v3 (35 dims).
    Returns (X, y, metadata_list, io_version, feature_dim).
    """
    by_trade: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        tid = r.get("trade_id") or ""
        if tid not in by_trade:
            by_trade[tid] = []
        by_trade[tid].append(r)
    for tid in by_trade:
        by_trade[tid].sort(key=lambda x: (x.get("ts") or "", x.get("run_id") or ""))

    def row_to_vec(r: Dict[str, Any]) -> np.ndarray:
        return row_to_feature_vector_v3(
            r, ctx_cont_dim=ONE_UNIVERSE_CTX_CONT_DIM, ctx_cat_dim=ONE_UNIVERSE_CTX_CAT_DIM
        )

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    meta_list: List[Dict[str, Any]] = []
    for trade_id, bars in by_trade.items():
        if len(bars) < window_len:
            continue
        vectors = [row_to_vec(r) for r in bars]
        for start in range(0, len(bars) - window_len + 1, stride):
            end = start + window_len
            window_vecs = vectors[start:end]
            last_bar = bars[end - 1]
            comp = last_bar.get("computed") or {}
            decision = (comp.get("decision") or "").strip().upper()
            label = 1 if decision == "EXIT" else 0
            X_list.append(np.stack(window_vecs, axis=0))
            y_list.append(label)
            meta_list.append({
                "run_id": last_bar.get("run_id"),
                "trade_id": trade_id,
                "ts": last_bar.get("ts"),
            })
    if not X_list:
        empty = np.zeros((0, window_len, EXIT_IO_FEATURE_COUNT), dtype=np.float32)
        return empty, np.zeros(0, dtype=np.int64), [], EXIT_IO_VERSION, EXIT_IO_FEATURE_COUNT
    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    for i in range(X.shape[0]):
        validate_window_v3(X[i], window_len, EXIT_IO_FEATURE_COUNT, context=f"{context}[{i}]")
    return X, y, meta_list, EXIT_IO_VERSION, EXIT_IO_FEATURE_COUNT


def load_dataset_from_exits_jsonl(
    path: Path,
    window_len: int = DEFAULT_WINDOW_LEN,
    stride: int = 1,
    use_io_v2: bool = False,
    require_io_v2: bool = False,
    ctx_cont_dim: int = 6,
    ctx_cat_dim: int = 6,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], str, int]:
    """
    Load exits jsonl and build (X, y, metadata, io_version, feature_dim).
    IOV3_CLEAN only (35 features). use_io_v2 / require_io_v2 / IOV1/IOV2 → RuntimeError.
    """
    if use_io_v2 or require_io_v2:
        raise RuntimeError(
            "legacy exit IO disabled; use IOV3_CLEAN. Do not pass use_io_v2 or require_io_v2."
        )
    records = load_exits_jsonl(path)
    return build_sequences_and_labels(
        records,
        window_len=window_len,
        stride=stride,
        context=path.name,
    )


def train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    metadata_list: List[Dict[str, Any]],
    val_fold: int = HOLDOUT_VAL_FOLD,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Deterministic split by trade_id: val_fold (default 9) → val, rest → train.
    Returns (X_train, y_train, X_val, y_val, meta_train, meta_val).
    """
    if len(metadata_list) != len(y) or len(metadata_list) != X.shape[0]:
        raise ValueError("metadata_list length must match X.shape[0] and len(y)")
    train_idx = [i for i, m in enumerate(metadata_list) if _trade_id_to_fold(m.get("trade_id") or "") != val_fold]
    val_idx = [i for i, m in enumerate(metadata_list) if _trade_id_to_fold(m.get("trade_id") or "") == val_fold]
    X_train = X[train_idx] if train_idx else np.zeros((0,) + X.shape[1:], dtype=X.dtype)
    y_train = y[train_idx] if train_idx else np.zeros(0, dtype=y.dtype)
    X_val = X[val_idx] if val_idx else np.zeros((0,) + X.shape[1:], dtype=X.dtype)
    y_val = y[val_idx] if val_idx else np.zeros(0, dtype=y.dtype)
    meta_train = [metadata_list[i] for i in train_idx]
    meta_val = [metadata_list[i] for i in val_idx]
    return X_train, y_train, X_val, y_val, meta_train, meta_val
