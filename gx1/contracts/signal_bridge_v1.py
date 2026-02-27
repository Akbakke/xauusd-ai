"""
XGB_SIGNAL_BRIDGE_V1 contract (SSoT).

Goal:
- Define the ONLY allowed Transformer input feature schema for ENTRY_V10_CTX going forward:
  - seq_x: per-bar XGB signal vectors (sequence)
  - snap_x: current-bar XGB signal vector (snapshot)
  - ctx_cat / ctx_cont: macro context tensors (order-sensitive; full contract lists ORDERED_CTX_CONT_NAMES_EXTENDED / ORDERED_CTX_CAT_NAMES_EXTENDED; bundle-meta expected dims define prefix)

Hard rules:
- Fixed, explicit ordered schema with fixed dimension.
- No fallback: in TRUTH/SMOKE, any mismatch must hard-fail.
- Independent of raw feature universes (no v10_ctx base 13/84 features).
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


SIGNAL_BRIDGE_ID = "XGB_SIGNAL_BRIDGE_V1"

# Class probability order (must match XGBMultiheadModel.predict_proba contract)
#
# xgb_multihead_model_v1.py documents:
# - proba[:, 0] = LONG
# - proba[:, 1] = SHORT
# - proba[:, 2] = FLAT
XGB_PROB_FIELDS_ORDERED: List[str] = [
    "p_long",
    "p_short",
    "p_flat",
]

# Signal bridge fields (ordered, fixed):
# - xgb probs (3)
# - p_hat (max of directional probs)
# - uncertainty_score (normalized entropy; already used in current runtime)
# - margin_top1_top2 (top1 - top2 across 3-class probs)
# - entropy (raw entropy in nats)
ORDERED_FIELDS: List[str] = [
    *XGB_PROB_FIELDS_ORDERED,
    "p_hat",
    "uncertainty_score",
    "margin_top1_top2",
    "entropy",
]

SEQ_SIGNAL_DIM = len(ORDERED_FIELDS)
SNAP_SIGNAL_DIM = len(ORDERED_FIELDS)

CONTRACT_SHA256 = hashlib.sha256(("|".join(ORDERED_FIELDS)).encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# ctx_cont (Transformer continuous context) - order-sensitive, no fallback
# ---------------------------------------------------------------------------
# Baseline: 2 dims [atr_bps, spread_bps]. Extended: +4 slow core = 6 dims (append only).
# Slow core is SLOW_CTX_CORE; each can be toggled via runtime ctx feature mask (ablation/ops).
ORDERED_CTX_CONT_NAMES_BASELINE: List[str] = ["atr_bps", "spread_bps"]
CTX_CONT_COL_D1_DIST = "D1_dist_from_ema200_atr"
CTX_CONT_COL_H1_COMP = "H1_range_compression_ratio"
CTX_CONT_COL_D1_ATR_PCTL252 = "D1_atr_percentile_252"
CTX_CONT_COL_M15_COMP = "M15_range_compression_ratio"
ORDERED_CTX_CONT_NAMES_SLOW_CORE: List[str] = [
    CTX_CONT_COL_D1_DIST,
    CTX_CONT_COL_H1_COMP,
    CTX_CONT_COL_D1_ATR_PCTL252,
    CTX_CONT_COL_M15_COMP,
]
ORDERED_CTX_CONT_NAMES_EXTENDED: List[str] = (
    ORDERED_CTX_CONT_NAMES_BASELINE + ORDERED_CTX_CONT_NAMES_SLOW_CORE
)
N_CTX_CONT_BASELINE = 2
N_CTX_CONT_EXTENDED = len(ORDERED_CTX_CONT_NAMES_EXTENDED)  # 6
# No global "default" dim: bundle-meta expected_ctx_cont_dim / expected_ctx_cat_dim and validate prefix.

# ---------------------------------------------------------------------------
# ctx_cat (Transformer categorical context) - order-sensitive, no fallback
# ---------------------------------------------------------------------------
# Baseline: 5 dims [session_id, trend_regime_id, vol_regime_id, atr_bucket, spread_bucket].
# Extended: +1 = 6 dims (append H4_trend_sign_cat, values 0/1/2).
ORDERED_CTX_CAT_NAMES_BASELINE: List[str] = [
    "session_id",
    "trend_regime_id",
    "vol_regime_id",
    "atr_bucket",
    "spread_bucket",
]
CTX_CAT_COL_H4_TREND_SIGN = "H4_trend_sign_cat"
ORDERED_CTX_CAT_NAMES_EXTENDED: List[str] = (
    ORDERED_CTX_CAT_NAMES_BASELINE + [CTX_CAT_COL_H4_TREND_SIGN]
)
N_CTX_CAT_BASELINE = 5
N_CTX_CAT_EXTENDED = len(ORDERED_CTX_CAT_NAMES_EXTENDED)  # 6
# No global "default" dim: bundle-meta drives expected dims; validate prefix against EXTENDED lists.

# ONE UNIVERSE: only 6/6
ALLOWED_CTX_CONT_DIMS = (6,)
ALLOWED_CTX_CAT_DIMS = (6,)


@dataclass(frozen=True)
class SignalBridgeContract:
    bridge_id: str
    ordered_fields: Tuple[str, ...]
    seq_dim: int
    snap_dim: int
    sha256: str


CONTRACT = SignalBridgeContract(
    bridge_id=SIGNAL_BRIDGE_ID,
    ordered_fields=tuple(ORDERED_FIELDS),
    seq_dim=int(SEQ_SIGNAL_DIM),
    snap_dim=int(SNAP_SIGNAL_DIM),
    sha256=str(CONTRACT_SHA256),
)


def _is_truth_or_smoke() -> bool:
    mode = os.getenv("GX1_RUN_MODE", "").upper()
    return os.getenv("GX1_TRUTH_MODE", "0") == "1" or mode in {"TRUTH", "SMOKE"}


def validate_seq_signal(seq_x: np.ndarray, *, context: str = "unknown") -> None:
    """
    Validate seq_x for SIGNAL_BRIDGE_V1.

    Expected:
    - seq_x is 3D: [B, T, SEQ_SIGNAL_DIM]
    - finite (no NaN/Inf)
    """
    if seq_x is None:
        raise RuntimeError(f"[SIGNAL_BRIDGE_FAIL] seq_x is None (context={context})")
    arr = np.asarray(seq_x)
    if _is_truth_or_smoke() and arr.dtype not in (np.float32, np.float64):
        raise RuntimeError(
            f"[SIGNAL_BRIDGE_FAIL] seq_x invalid dtype={arr.dtype} "
            f"(expected float32 or float64, context={context})"
        )
    if arr.ndim != 3:
        raise RuntimeError(
            f"[SIGNAL_BRIDGE_FAIL] seq_x.ndim mismatch: expected=3 got={arr.ndim} shape={getattr(arr,'shape',None)} (context={context})"
        )
    if int(arr.shape[-1]) != int(SEQ_SIGNAL_DIM):
        raise RuntimeError(
            f"[SIGNAL_BRIDGE_FAIL] seq_x feature dim mismatch: expected={SEQ_SIGNAL_DIM} got={int(arr.shape[-1])} (context={context})"
        )
    if not np.isfinite(arr).all():
        n_bad = int((~np.isfinite(arr)).sum())
        raise RuntimeError(f"[SIGNAL_BRIDGE_FAIL] seq_x contains non-finite values: count={n_bad} (context={context})")


def validate_snap_signal(snap_x: np.ndarray, *, context: str = "unknown") -> None:
    """
    Validate snap_x for SIGNAL_BRIDGE_V1.

    Expected:
    - snap_x is 2D: [B, SNAP_SIGNAL_DIM]
    - finite (no NaN/Inf)
    """
    if snap_x is None:
        raise RuntimeError(f"[SIGNAL_BRIDGE_FAIL] snap_x is None (context={context})")
    arr = np.asarray(snap_x)
    if _is_truth_or_smoke() and arr.dtype not in (np.float32, np.float64):
        raise RuntimeError(
            f"[SIGNAL_BRIDGE_FAIL] snap_x invalid dtype={arr.dtype} "
            f"(expected float32 or float64, context={context})"
        )
    if arr.ndim != 2:
        raise RuntimeError(
            f"[SIGNAL_BRIDGE_FAIL] snap_x.ndim mismatch: expected=2 got={arr.ndim} shape={getattr(arr,'shape',None)} (context={context})"
        )
    if int(arr.shape[-1]) != int(SNAP_SIGNAL_DIM):
        raise RuntimeError(
            f"[SIGNAL_BRIDGE_FAIL] snap_x feature dim mismatch: expected={SNAP_SIGNAL_DIM} got={int(arr.shape[-1])} (context={context})"
        )
    if not np.isfinite(arr).all():
        n_bad = int((~np.isfinite(arr)).sum())
        raise RuntimeError(f"[SIGNAL_BRIDGE_FAIL] snap_x contains non-finite values: count={n_bad} (context={context})")


def validate_contract_in_truth() -> None:
    """
    TRUTH/SMOKE invariant: contract must be stable and non-empty.
    """
    if not _is_truth_or_smoke():
        return
    if not ORDERED_FIELDS or len(set(ORDERED_FIELDS)) != len(ORDERED_FIELDS):
        raise RuntimeError("[SIGNAL_BRIDGE_FAIL] ORDERED_FIELDS invalid (empty or duplicates)")
    if SEQ_SIGNAL_DIM <= 0 or SNAP_SIGNAL_DIM <= 0:
        raise RuntimeError("[SIGNAL_BRIDGE_FAIL] signal dims invalid (<=0)")
    if CONTRACT.sha256 != CONTRACT_SHA256:
        raise RuntimeError("[SIGNAL_BRIDGE_FAIL] contract sha mismatch (internal)")
    if not ORDERED_CTX_CONT_NAMES_EXTENDED or len(set(ORDERED_CTX_CONT_NAMES_EXTENDED)) != len(ORDERED_CTX_CONT_NAMES_EXTENDED):
        raise RuntimeError("[SIGNAL_BRIDGE_FAIL] ORDERED_CTX_CONT_NAMES_EXTENDED invalid (empty or duplicates)")
    if not ORDERED_CTX_CAT_NAMES_EXTENDED or len(set(ORDERED_CTX_CAT_NAMES_EXTENDED)) != len(ORDERED_CTX_CAT_NAMES_EXTENDED):
        raise RuntimeError("[SIGNAL_BRIDGE_FAIL] ORDERED_CTX_CAT_NAMES_EXTENDED invalid (empty or duplicates)")
    if len(ORDERED_CTX_CONT_NAMES_BASELINE) != N_CTX_CONT_BASELINE:
        raise RuntimeError(
            f"[SIGNAL_BRIDGE_FAIL] N_CTX_CONT_BASELINE={N_CTX_CONT_BASELINE} != len(ORDERED_CTX_CONT_NAMES_BASELINE)={len(ORDERED_CTX_CONT_NAMES_BASELINE)}"
        )
    if len(ORDERED_CTX_CAT_NAMES_BASELINE) != N_CTX_CAT_BASELINE:
        raise RuntimeError(
            f"[SIGNAL_BRIDGE_FAIL] N_CTX_CAT_BASELINE={N_CTX_CAT_BASELINE} != len(ORDERED_CTX_CAT_NAMES_BASELINE)={len(ORDERED_CTX_CAT_NAMES_BASELINE)}"
        )


def get_canonical_ctx_contract() -> Dict[str, object]:
    """
    Return the canonical ONE-UNIVERSE ctx contract for ENTRY_V10_CTX.
    
    Canonical tag: CTX6CAT6
    ctx_cont_dim = 6 (ORDERED_CTX_CONT_NAMES_EXTENDED)
    ctx_cat_dim  = 6 (ORDERED_CTX_CAT_NAMES_EXTENDED)
    """
    return {
        "ctx_cont_dim": int(N_CTX_CONT_EXTENDED),
        "ctx_cat_dim": int(N_CTX_CAT_EXTENDED),
        "ctx_cont_names": list(ORDERED_CTX_CONT_NAMES_EXTENDED),
        "ctx_cat_names": list(ORDERED_CTX_CAT_NAMES_EXTENDED),
        "tag": "CTX6CAT6",
        "source": "signal_bridge_v1_contract",
    }


def validate_bundle_ctx_contract_in_strict(
    expected_ctx_cont_dim: int,
    expected_ctx_cat_dim: int,
    ordered_ctx_cont_names: Sequence[str],
    ordered_ctx_cat_names: Sequence[str],
    *,
    context: str = "bundle_meta",
) -> None:
    """
    TRUTH/SMOKE hard-gate: ONE UNIVERSE 6/6 only; meta list length >= dim; ordered names match contract prefix.
    - ctx_cont_dim must be 6, ctx_cat_dim must be 6; RuntimeError otherwise.
    - meta ordered_ctx_cont_names / ordered_ctx_cat_names must have length >= expected dim, else RuntimeError with meta_cont_len / meta_cat_len.
    - meta_*[:dim] must equal contract ORDERED_CTX_*_EXTENDED[:dim] (always, including baseline 2/5).
    """
    if not _is_truth_or_smoke():
        return
    if expected_ctx_cont_dim not in ALLOWED_CTX_CONT_DIMS:
        raise RuntimeError(
            f"[SIGNAL_BRIDGE_FAIL] {context} expected_ctx_cont_dim={expected_ctx_cont_dim} not in {ALLOWED_CTX_CONT_DIMS}"
        )
    if expected_ctx_cat_dim not in ALLOWED_CTX_CAT_DIMS:
        raise RuntimeError(
            f"[SIGNAL_BRIDGE_FAIL] {context} expected_ctx_cat_dim={expected_ctx_cat_dim} not in {ALLOWED_CTX_CAT_DIMS}"
        )
    if _is_truth_or_smoke() and len(ordered_ctx_cont_names) > len(ORDERED_CTX_CONT_NAMES_EXTENDED):
        raise RuntimeError(
            f"[SIGNAL_BRIDGE_FAIL] {context} ordered_ctx_cont_names longer than contract: "
            f"meta_len={len(ordered_ctx_cont_names)} contract_len={len(ORDERED_CTX_CONT_NAMES_EXTENDED)}"
        )
    if _is_truth_or_smoke() and len(ordered_ctx_cat_names) > len(ORDERED_CTX_CAT_NAMES_EXTENDED):
        raise RuntimeError(
            f"[SIGNAL_BRIDGE_FAIL] {context} ordered_ctx_cat_names longer than contract: "
            f"meta_len={len(ordered_ctx_cat_names)} contract_len={len(ORDERED_CTX_CAT_NAMES_EXTENDED)}"
        )
    if expected_ctx_cont_dim > len(ORDERED_CTX_CONT_NAMES_EXTENDED):
        raise RuntimeError(
            f"[SIGNAL_BRIDGE_FAIL] {context} expected_ctx_cont_dim={expected_ctx_cont_dim} > "
            f"len(ORDERED_CTX_CONT_NAMES_EXTENDED)={len(ORDERED_CTX_CONT_NAMES_EXTENDED)}"
        )
    if expected_ctx_cat_dim > len(ORDERED_CTX_CAT_NAMES_EXTENDED):
        raise RuntimeError(
            f"[SIGNAL_BRIDGE_FAIL] {context} expected_ctx_cat_dim={expected_ctx_cat_dim} > "
            f"len(ORDERED_CTX_CAT_NAMES_EXTENDED)={len(ORDERED_CTX_CAT_NAMES_EXTENDED)}"
        )
    meta_cont_len = len(ordered_ctx_cont_names)
    if meta_cont_len < expected_ctx_cont_dim:
        raise RuntimeError(
            f"[SIGNAL_BRIDGE_FAIL] {context} ordered_ctx_cont_names too short: "
            f"meta_cont_len={meta_cont_len} expected_ctx_cont_dim={expected_ctx_cont_dim}"
        )
    meta_cat_len = len(ordered_ctx_cat_names)
    if meta_cat_len < expected_ctx_cat_dim:
        raise RuntimeError(
            f"[SIGNAL_BRIDGE_FAIL] {context} ordered_ctx_cat_names too short: "
            f"meta_cat_len={meta_cat_len} expected_ctx_cat_dim={expected_ctx_cat_dim}"
        )
    contract_cont = ORDERED_CTX_CONT_NAMES_EXTENDED[:expected_ctx_cont_dim]
    meta_cont = list(ordered_ctx_cont_names)[:expected_ctx_cont_dim]
    if meta_cont != contract_cont:
        raise RuntimeError(
            f"[SIGNAL_BRIDGE_FAIL] bundle {context} ordered_ctx_cont_names does not match "
            f"contract (expected_ctx_cont_dim={expected_ctx_cont_dim}): "
            f"contract_prefix={contract_cont!r} meta_prefix={meta_cont!r}"
        )
    contract_cat = ORDERED_CTX_CAT_NAMES_EXTENDED[:expected_ctx_cat_dim]
    meta_cat = list(ordered_ctx_cat_names)[:expected_ctx_cat_dim]
    if meta_cat != contract_cat:
        raise RuntimeError(
            f"[SIGNAL_BRIDGE_FAIL] bundle {context} ordered_ctx_cat_names does not match "
            f"contract (expected_ctx_cat_dim={expected_ctx_cat_dim}): "
            f"contract_prefix={contract_cat!r} meta_prefix={meta_cat!r}"
        )


__all__ = [
    "SIGNAL_BRIDGE_ID",
    "ORDERED_FIELDS",
    "SEQ_SIGNAL_DIM",
    "SNAP_SIGNAL_DIM",
    "CONTRACT_SHA256",
    "CONTRACT",
    "CTX_CONT_COL_D1_DIST",
    "CTX_CONT_COL_H1_COMP",
    "CTX_CONT_COL_D1_ATR_PCTL252",
    "CTX_CONT_COL_M15_COMP",
    "ORDERED_CTX_CONT_NAMES_BASELINE",
    "ORDERED_CTX_CONT_NAMES_SLOW_CORE",
    "ORDERED_CTX_CONT_NAMES_EXTENDED",
    "N_CTX_CONT_BASELINE",
    "N_CTX_CONT_EXTENDED",
    "ALLOWED_CTX_CONT_DIMS",
    "ALLOWED_CTX_CAT_DIMS",
    "CTX_CAT_COL_H4_TREND_SIGN",
    "ORDERED_CTX_CAT_NAMES_BASELINE",
    "ORDERED_CTX_CAT_NAMES_EXTENDED",
    "N_CTX_CAT_BASELINE",
    "N_CTX_CAT_EXTENDED",
    "validate_seq_signal",
    "validate_snap_signal",
    "validate_contract_in_truth",
    "validate_bundle_ctx_contract_in_strict",
]