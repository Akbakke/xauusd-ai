"""
Exit Transformer IO V1 – SSoT input contract for exit-transformer (imitation of score_v1).

Per-bar features (numeric), ordered. Validation, stable hashing for config.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

EXIT_TRANSFORMER_IO_ID = "EXIT_TRANSFORMER_IO_V1"

# Per-bar feature order (must match ExitMLContext / exits jsonl state+signals+entry_snapshot)
# Signal bridge (7)
ORDERED_SIGNAL_FIELDS: List[str] = [
    "p_long",
    "p_short",
    "p_flat",
    "p_hat",
    "uncertainty_score",
    "margin_top1_top2",
    "entropy",
]
# Entry snapshot frozen (5)
ORDERED_ENTRY_SNAPSHOT_FIELDS: List[str] = [
    "p_long_entry",
    "p_hat_entry",
    "uncertainty_entry",
    "entropy_entry",
    "margin_entry",
]
# Trade state (7)
ORDERED_TRADE_STATE_FIELDS: List[str] = [
    "pnl_bps_now",
    "mfe_bps",
    "mae_bps",
    "dd_from_mfe_bps",
    "bars_held",
    "time_since_mfe_bars",
    "atr_bps_now",
]

ORDERED_FEATURE_NAMES: List[str] = (
    ORDERED_SIGNAL_FIELDS + ORDERED_ENTRY_SNAPSHOT_FIELDS + ORDERED_TRADE_STATE_FIELDS
)
FEATURE_DIM = len(ORDERED_FEATURE_NAMES)

CONTRACT_SHA256 = hashlib.sha256(
    ("|".join(ORDERED_FEATURE_NAMES)).encode("utf-8")
).hexdigest()


@dataclass(frozen=True)
class ExitTransformerIOContract:
    contract_id: str
    ordered_feature_names: Tuple[str, ...]
    feature_dim: int
    sha256: str


CONTRACT = ExitTransformerIOContract(
    contract_id=EXIT_TRANSFORMER_IO_ID,
    ordered_feature_names=tuple(ORDERED_FEATURE_NAMES),
    feature_dim=int(FEATURE_DIM),
    sha256=str(CONTRACT_SHA256),
)


def config_hash(window_len: int, d_model: int, n_layers: int, feature_names: Optional[Sequence[str]] = None) -> str:
    """Stable hash for transformer config (window_len, d_model, layers, feature set)."""
    names = tuple(feature_names or ORDERED_FEATURE_NAMES)
    canonical = f"window_len={window_len}|d_model={d_model}|n_layers={n_layers}|features={'|'.join(names)}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def ml_ctx_to_feature_vector(ml_ctx: Any, atr_bps_fill: float = 1.0) -> np.ndarray:
    """
    Build one bar's feature vector from ExitMLContext (order matches ORDERED_FEATURE_NAMES).
    Missing/None → 0.0; atr_bps uses atr_bps_fill if None/<=0.
    """
    def _f(x: Any, default: float = 0.0) -> float:
        if x is None:
            return default
        try:
            v = float(x)
            return v if np.isfinite(v) else default
        except (TypeError, ValueError):
            return default

    atr = _f(getattr(ml_ctx, "atr_bps", None), atr_bps_fill)
    if atr <= 0:
        atr = atr_bps_fill

    vec = np.zeros(FEATURE_DIM, dtype=np.float32)
    vec[0] = _f(getattr(ml_ctx, "p_long", None))
    vec[1] = _f(getattr(ml_ctx, "p_short", None))
    vec[2] = _f(getattr(ml_ctx, "p_flat", None))
    vec[3] = _f(getattr(ml_ctx, "p_hat", None))
    vec[4] = _f(getattr(ml_ctx, "uncertainty_score", None))
    vec[5] = _f(getattr(ml_ctx, "margin_top1_top2", None))
    vec[6] = _f(getattr(ml_ctx, "entropy", None))
    vec[7] = _f(getattr(ml_ctx, "p_long_entry", None))
    vec[8] = _f(getattr(ml_ctx, "p_hat_entry", None))
    vec[9] = _f(getattr(ml_ctx, "uncertainty_entry", None))
    vec[10] = _f(getattr(ml_ctx, "entropy_entry", None))
    vec[11] = _f(getattr(ml_ctx, "margin_entry", None))
    vec[12] = _f(getattr(ml_ctx, "pnl_bps", None))
    vec[13] = _f(getattr(ml_ctx, "mfe_bps", None))
    vec[14] = _f(getattr(ml_ctx, "mae_bps", None))
    vec[15] = _f(getattr(ml_ctx, "drawdown_from_mfe_bps", None))
    vec[16] = float(getattr(ml_ctx, "bars_held", 0) or 0)
    vec[17] = float(getattr(ml_ctx, "time_since_mfe_bars", 0) or 0)
    vec[18] = atr
    return vec


def validate_window(x: np.ndarray, window_len: int, context: str = "unknown") -> None:
    """Validate sequence tensor [T, FEATURE_DIM] or [B, T, FEATURE_DIM]; T must be window_len; no NaN/Inf."""
    arr = np.asarray(x)
    if arr.ndim == 2:
        if arr.shape[0] != window_len or arr.shape[1] != FEATURE_DIM:
            raise RuntimeError(
                f"[EXIT_TRANSFORMER_IO] window shape mismatch: expected ({window_len}, {FEATURE_DIM}), "
                f"got {arr.shape} (context={context})"
            )
    elif arr.ndim == 3:
        if arr.shape[1] != window_len or arr.shape[2] != FEATURE_DIM:
            raise RuntimeError(
                f"[EXIT_TRANSFORMER_IO] batch window shape mismatch: expected (B, {window_len}, {FEATURE_DIM}), "
                f"got {arr.shape} (context={context})"
            )
    else:
        raise RuntimeError(
            f"[EXIT_TRANSFORMER_IO] expected 2D or 3D array, got ndim={arr.ndim} (context={context})"
        )
    if not np.isfinite(arr).all():
        n_bad = int((~np.isfinite(arr)).sum())
        raise RuntimeError(
            f"[EXIT_TRANSFORMER_IO] non-finite values: count={n_bad} (context={context})"
        )


__all__ = [
    "EXIT_TRANSFORMER_IO_ID",
    "ORDERED_FEATURE_NAMES",
    "ORDERED_SIGNAL_FIELDS",
    "ORDERED_ENTRY_SNAPSHOT_FIELDS",
    "ORDERED_TRADE_STATE_FIELDS",
    "FEATURE_DIM",
    "CONTRACT_SHA256",
    "CONTRACT",
    "config_hash",
    "ml_ctx_to_feature_vector",
    "validate_window",
]
