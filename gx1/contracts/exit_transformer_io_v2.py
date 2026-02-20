"""
Exit Transformer IO V2 – SSoT input contract with slow context (ctx_cont + ctx_cat).

Extends IOV1 with same-order entry context: ctx_cont (dim 6), ctx_cat (dim 6). ONE UNIVERSE 6/6.
Validation, NaN-policy (missing → 0), stable feature_names_hash for TRAIN_REPORT/VERIFY.
"""

from __future__ import annotations

import hashlib
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from gx1.contracts.exit_transformer_io_v1 import (
    ORDERED_ENTRY_SNAPSHOT_FIELDS,
    ORDERED_SIGNAL_FIELDS,
    ORDERED_TRADE_STATE_FIELDS,
)

EXIT_TRANSFORMER_IO_V2_ID = "EXIT_TRANSFORMER_IO_V2"

# Same as IOV1: signal bridge (7) + entry snapshot (5) + trade state (7)
ORDERED_IOV1_NAMES: List[str] = (
    list(ORDERED_SIGNAL_FIELDS)
    + list(ORDERED_ENTRY_SNAPSHOT_FIELDS)
    + list(ORDERED_TRADE_STATE_FIELDS)
)
IOV1_DIM = 19

# ONE UNIVERSE: 6/6 only
DEFAULT_CTX_CONT_DIM = 6
DEFAULT_CTX_CAT_DIM = 6


def ordered_feature_names_v2(
    ctx_cont_dim: int = DEFAULT_CTX_CONT_DIM,
    ctx_cat_dim: int = DEFAULT_CTX_CAT_DIM,
) -> List[str]:
    """Ordered feature names for IOV2: IOV1 + ctx_cont_0..ctx_cont_{n-1} + ctx_cat_0..ctx_cat_{n-1}."""
    names = list(ORDERED_IOV1_NAMES)
    for i in range(ctx_cont_dim):
        names.append(f"ctx_cont_{i}")
    for i in range(ctx_cat_dim):
        names.append(f"ctx_cat_{i}")
    return names


def feature_dim_v2(
    ctx_cont_dim: int = DEFAULT_CTX_CONT_DIM,
    ctx_cat_dim: int = DEFAULT_CTX_CAT_DIM,
) -> int:
    return IOV1_DIM + ctx_cont_dim + ctx_cat_dim


def contract_sha256_v2(
    ctx_cont_dim: int = DEFAULT_CTX_CONT_DIM,
    ctx_cat_dim: int = DEFAULT_CTX_CAT_DIM,
) -> str:
    names = ordered_feature_names_v2(ctx_cont_dim=ctx_cont_dim, ctx_cat_dim=ctx_cat_dim)
    return hashlib.sha256(("|".join(names)).encode("utf-8")).hexdigest()


def config_hash_v2(
    window_len: int,
    d_model: int,
    n_layers: int,
    ctx_cont_dim: int = DEFAULT_CTX_CONT_DIM,
    ctx_cat_dim: int = DEFAULT_CTX_CAT_DIM,
    feature_names: Optional[Sequence[str]] = None,
) -> str:
    """Stable hash for IOV2 transformer config."""
    names = tuple(
        feature_names
        or ordered_feature_names_v2(ctx_cont_dim=ctx_cont_dim, ctx_cat_dim=ctx_cat_dim)
    )
    canonical = (
        f"io=IOV2|window_len={window_len}|d_model={d_model}|n_layers={n_layers}|"
        f"ctx_cont={ctx_cont_dim}|ctx_cat={ctx_cat_dim}|features={'|'.join(names)}"
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _row_to_iov1_slice(row: Any) -> np.ndarray:
    """Build first IOV1_DIM from row (state/signals/entry_snapshot). No context. Same order as IOV1."""
    def _f(d: Any, k: str, default: float = 0.0) -> float:
        if d is None:
            return default
        v = d.get(k) if isinstance(d, dict) else None
        if v is None:
            return default
        try:
            x = float(v)
            return x if np.isfinite(x) else default
        except (TypeError, ValueError):
            return default

    if not isinstance(row, dict):
        return np.zeros(IOV1_DIM, dtype=np.float32)
    state = row.get("state") or {}
    signals = row.get("signals") or {}
    entry_snapshot = row.get("entry_snapshot") or {}
    atr_bps_now = 1.0
    vec = np.zeros(IOV1_DIM, dtype=np.float32)
    vec[0] = _f(signals, "p_long_now")
    vec[1] = _f(signals, "p_short_now")
    vec[2] = 0.0
    vec[3] = _f(signals, "p_hat_now")
    vec[4] = _f(signals, "uncertainty_score")
    vec[5] = _f(signals, "margin_top1_top2")
    vec[6] = _f(signals, "entropy")
    vec[7] = _f(entry_snapshot, "p_long_entry")
    vec[8] = _f(entry_snapshot, "p_hat_entry")
    vec[9] = _f(entry_snapshot, "uncertainty_entry")
    vec[10] = _f(entry_snapshot, "entropy_entry")
    vec[11] = _f(entry_snapshot, "margin_entry")
    vec[12] = _f(state, "pnl_bps")
    vec[13] = _f(state, "mfe_bps")
    vec[14] = _f(state, "mae_bps")
    vec[15] = _f(state, "dd_from_mfe_bps")
    vec[16] = float(state.get("bars_held") or 0)
    vec[17] = float(state.get("time_since_mfe_bars") or 0)
    vec[18] = atr_bps_now
    return vec


def row_to_feature_vector_v2(
    row: Any,
    ctx_cont_dim: int = DEFAULT_CTX_CONT_DIM,
    ctx_cat_dim: int = DEFAULT_CTX_CAT_DIM,
    atr_bps_fill: float = 1.0,
) -> np.ndarray:
    """
    Build one bar's feature vector from jsonl row (state + signals + entry_snapshot + context).
    Uses same mapping as IOV1 for first IOV1_DIM; then ctx_cont[*], ctx_cat[*].
    Missing/None → 0.0. NaN/Inf → 0.0.
    """
    vec_iov1 = _row_to_iov1_slice(row)
    dim = feature_dim_v2(ctx_cont_dim=ctx_cont_dim, ctx_cat_dim=ctx_cat_dim)
    vec = np.zeros(dim, dtype=np.float32)
    vec[:IOV1_DIM] = vec_iov1

    context = row.get("context") if isinstance(row, dict) else None
    if not context:
        return vec
    ctx_cont = context.get("ctx_cont") if isinstance(context, dict) else None
    ctx_cat = context.get("ctx_cat") if isinstance(context, dict) else None

    def _float(x: Any, default: float = 0.0) -> float:
        if x is None:
            return default
        try:
            v = float(x)
            return v if np.isfinite(v) else default
        except (TypeError, ValueError):
            return default

    off = IOV1_DIM
    if isinstance(ctx_cont, (list, tuple)) and len(ctx_cont) >= ctx_cont_dim:
        for i in range(ctx_cont_dim):
            vec[off + i] = _float(ctx_cont[i] if i < len(ctx_cont) else None)
    off += ctx_cont_dim
    if isinstance(ctx_cat, (list, tuple)) and len(ctx_cat) >= ctx_cat_dim:
        for i in range(ctx_cat_dim):
            vec[off + i] = _float(ctx_cat[i] if i < len(ctx_cat) else None)
    return vec


def validate_window_v2(
    x: np.ndarray,
    window_len: int,
    feature_dim: int,
    context: str = "unknown",
) -> None:
    """Validate sequence [T, feature_dim]; no NaN/Inf."""
    arr = np.asarray(x)
    if arr.ndim == 2:
        if arr.shape[0] != window_len or arr.shape[1] != feature_dim:
            raise RuntimeError(
                f"[EXIT_TRANSFORMER_IO_V2] window shape expected ({window_len}, {feature_dim}), got {arr.shape} ({context})"
            )
    elif arr.ndim == 3:
        if arr.shape[1] != window_len or arr.shape[2] != feature_dim:
            raise RuntimeError(
                f"[EXIT_TRANSFORMER_IO_V2] batch window shape expected (B, {window_len}, {feature_dim}), got {arr.shape} ({context})"
            )
    else:
        raise RuntimeError(
            f"[EXIT_TRANSFORMER_IO_V2] expected 2D or 3D array, got ndim={arr.ndim} ({context})"
        )
    if not np.isfinite(arr).all():
        n_bad = int((~np.isfinite(arr)).sum())
        raise RuntimeError(
            f"[EXIT_TRANSFORMER_IO_V2] non-finite values: count={n_bad} ({context})"
        )


def has_context_in_row(row: Any) -> bool:
    """True if row has context.ctx_cont and context.ctx_cat (list-like)."""
    if isinstance(row, dict):
        ctx = row.get("context")
    else:
        ctx = getattr(row, "get", lambda k: None)("context") if hasattr(row, "get") else None
    if not ctx or not isinstance(ctx, dict):
        return False
    c_cont = ctx.get("ctx_cont")
    c_cat = ctx.get("ctx_cat")
    return (
        isinstance(c_cont, (list, tuple)) and len(c_cont) > 0
        and isinstance(c_cat, (list, tuple)) and len(c_cat) > 0
    )


__all__ = [
    "EXIT_TRANSFORMER_IO_V2_ID",
    "ORDERED_IOV1_NAMES",
    "IOV1_DIM",
    "DEFAULT_CTX_CONT_DIM",
    "DEFAULT_CTX_CAT_DIM",
    "ordered_feature_names_v2",
    "feature_dim_v2",
    "contract_sha256_v2",
    "config_hash_v2",
    "row_to_feature_vector_v2",
    "validate_window_v2",
    "has_context_in_row",
]
