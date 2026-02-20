"""
Runtime ctx feature mask (env/config) for Transformer entry.

Ablation/ops: turn off slow ctx features without rebuilding prebuilt.

GX1_CTX_CONT_MASK (contract order):
  - Length must be either N_CTX_CONT_EXTENDED (full) or expected_ctx_cont_dim (bundle prefix).
  - If length == expected_ctx_cont_dim: pad with 1s to full length, then slice to bundle dim.
  - Any other length: strict → RuntimeError; non-strict → default all 1s.
  - Indices 0,1 (atr_bps, spread_bps) must be 1; strict fails if set to 0.

GX1_CTX_CAT_MASK:
  - Length must equal expected_ctx_cat_dim. Strict → RuntimeError on mismatch; non-strict → default all 1s.

TRUTH/SMOKE or GX1_STRICT_MASK=1: hard-fail on wrong length, invalid values (not 0/1), or cont[0]/cont[1]=0.
Returns mask arrays and mask_id (sha256 prefix) for footer/diagnostics.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Tuple

import numpy as np

from gx1.contracts.signal_bridge_v1 import N_CTX_CONT_EXTENDED

log = logging.getLogger(__name__)


def _is_truth_or_smoke() -> bool:
    mode = os.getenv("GX1_RUN_MODE", "").upper()
    return os.getenv("GX1_TRUTH_MODE", "0") == "1" or mode in {"TRUTH", "SMOKE"}


def _strict_mask() -> bool:
    """True if mask errors must raise RuntimeError (TRUTH/SMOKE or GX1_STRICT_MASK=1)."""
    return _is_truth_or_smoke() or os.getenv("GX1_STRICT_MASK", "").strip() == "1"


def _parse_mask_string(env_key: str, expected_len: int) -> np.ndarray:
    """Parse env as comma-separated 0/1; length must equal expected_len. Empty → all 1s."""
    raw = os.getenv(env_key, "").strip()
    if not raw:
        return np.ones(expected_len, dtype=np.float32)
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != expected_len:
        if _strict_mask():
            raise RuntimeError(
                f"[CTX_MASK_FAIL] {env_key} length={len(parts)} does not match expected={expected_len}. "
                f"TRUTH/SMOKE or GX1_STRICT_MASK=1 requires exact match."
            )
        log.warning(
            "[CTX_MASK] %s length=%d != expected=%d; using default all 1",
            env_key, len(parts), expected_len,
        )
        return np.ones(expected_len, dtype=np.float32)
    out = np.zeros(expected_len, dtype=np.float32)
    for i, p in enumerate(parts):
        if p == "1":
            out[i] = 1.0
        elif p != "0":
            if _strict_mask():
                raise RuntimeError(
                    f"[CTX_MASK_FAIL] {env_key} value at index {i} must be 0 or 1, got {p!r}"
                )
            out[i] = 1.0
    return out


def _parts_to_cont_array(parts: list, length: int, env_key: str) -> np.ndarray:
    """Parse parts to float32 0/1 array of given length. Invalid value: strict raise, else 1."""
    out = np.zeros(length, dtype=np.float32)
    for i, p in enumerate(parts):
        if i >= length:
            break
        if p == "1":
            out[i] = 1.0
        elif p != "0":
            if _strict_mask():
                raise RuntimeError(
                    f"[CTX_MASK_FAIL] {env_key} value at index {i} must be 0 or 1, got {p!r}"
                )
            out[i] = 1.0
    return out


def _parse_cont_mask(expected_ctx_cont_dim: int) -> np.ndarray:
    """
    Parse GX1_CTX_CONT_MASK. Allowed: length == N_CTX_CONT_EXTENDED (full) or == expected_ctx_cont_dim (prefix).
    Prefix is padded with 1s to full length. Other lengths: strict fail, non-strict → all 1s.
    """
    raw = os.getenv("GX1_CTX_CONT_MASK", "").strip()
    if not raw:
        return np.ones(N_CTX_CONT_EXTENDED, dtype=np.float32)
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) == expected_ctx_cont_dim:
        prefix = _parts_to_cont_array(parts, expected_ctx_cont_dim, "GX1_CTX_CONT_MASK")
        full = np.ones(N_CTX_CONT_EXTENDED, dtype=np.float32)
        full[:expected_ctx_cont_dim] = prefix
        return full
    if len(parts) == N_CTX_CONT_EXTENDED:
        return _parts_to_cont_array(parts, N_CTX_CONT_EXTENDED, "GX1_CTX_CONT_MASK")
    if _strict_mask():
        raise RuntimeError(
            f"[CTX_MASK_FAIL] GX1_CTX_CONT_MASK length={len(parts)} must be "
            f"{expected_ctx_cont_dim} (bundle prefix) or {N_CTX_CONT_EXTENDED} (full contract)."
        )
    log.warning(
        "[CTX_MASK] GX1_CTX_CONT_MASK length=%d not %d or %d; using default all 1",
        len(parts), expected_ctx_cont_dim, N_CTX_CONT_EXTENDED,
    )
    return np.ones(N_CTX_CONT_EXTENDED, dtype=np.float32)


def _mask_id(mask: np.ndarray) -> str:
    s = ",".join("1" if float(x) != 0 else "0" for x in mask.tolist())
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def get_ctx_feature_masks(
    expected_ctx_cont_dim: int,
    expected_ctx_cat_dim: int,
) -> Tuple[np.ndarray, np.ndarray, str, str]:
    """
    Parse env masks and return (cont_mask, cat_mask, cont_mask_id, cat_mask_id).

    - cont_mask: length expected_ctx_cont_dim, float32 0/1. Indices 0,1 forced to 1.
    - cat_mask: length expected_ctx_cat_dim, float32 0/1.
    - mask_id: sha256 prefix for footer/diagnostics (cont_id from sliced cont mask).
    """
    # Invariants: expected dims in range
    if expected_ctx_cont_dim <= 0 or expected_ctx_cont_dim > N_CTX_CONT_EXTENDED or expected_ctx_cat_dim <= 0:
        if _strict_mask():
            raise RuntimeError(
                f"[CTX_MASK_FAIL] invalid expected dims: expected_ctx_cont_dim={expected_ctx_cont_dim} "
                f"(must be 1..{N_CTX_CONT_EXTENDED}), expected_ctx_cat_dim={expected_ctx_cat_dim} (must be > 0)."
            )
        log.warning(
            "[CTX_MASK] invalid expected dims cont=%s cat=%s; using default all 1",
            expected_ctx_cont_dim, expected_ctx_cat_dim,
        )
        cont_mask = np.ones(expected_ctx_cont_dim, dtype=np.float32)
        cat_mask = np.ones(expected_ctx_cat_dim, dtype=np.float32)
        return cont_mask, cat_mask, _mask_id(cont_mask), _mask_id(cat_mask)
    # Cont: parse (full or prefix), enforce 0,1 == 1, slice to bundle dim
    cont_full = _parse_cont_mask(expected_ctx_cont_dim)
    if cont_full[0] != 1.0 or cont_full[1] != 1.0:
        if _strict_mask():
            raise RuntimeError(
                "[CTX_MASK_FAIL] atr_bps and spread_bps (indices 0,1) must be 1. "
                f"GX1_CTX_CONT_MASK: got cont[0]={cont_full[0]}, cont[1]={cont_full[1]}."
            )
        cont_full[0] = 1.0
        cont_full[1] = 1.0
    cont_mask = cont_full[:expected_ctx_cont_dim].astype(np.float32).copy()
    cont_mask_id = _mask_id(cont_mask)

    cat_mask = _parse_mask_string("GX1_CTX_CAT_MASK", expected_ctx_cat_dim)
    cat_mask_id = _mask_id(cat_mask)

    return cont_mask, cat_mask, cont_mask_id, cat_mask_id


def apply_ctx_cont_mask(ctx_cont: np.ndarray, cont_mask: np.ndarray) -> np.ndarray:
    """Apply cont mask: ctx_cont * cont_mask (broadcast)."""
    if ctx_cont.shape[-1] != cont_mask.shape[0]:
        if _strict_mask():
            raise RuntimeError(
                f"[CTX_MASK_FAIL] ctx_cont dim={ctx_cont.shape[-1]} != cont_mask dim={cont_mask.shape[0]}"
            )
        return ctx_cont
    return ctx_cont * np.asarray(cont_mask, dtype=ctx_cont.dtype)


# Neutral values when cat feature is masked (mask=0). Order matches ORDERED_CTX_CAT_NAMES_EXTENDED.
# session_id=0 (ASIA), trend_regime_id=1 (NEUTRAL), vol_regime_id=1 (MEDIUM), atr_bucket=0, spread_bucket=0,
# H4_trend_sign_cat=1 (flat).
NEUTRAL_CTX_CAT_VALUES: Tuple[int, ...] = (0, 1, 1, 0, 0, 1)


def apply_ctx_cat_mask(ctx_cat: np.ndarray, cat_mask: np.ndarray) -> np.ndarray:
    """
    Apply cat mask: where mask=0, set to neutral class (no lookahead).
    ctx_cat and cat_mask must have same length (last dim).
    """
    if ctx_cat.shape[-1] != cat_mask.shape[0]:
        if _strict_mask():
            raise RuntimeError(
                f"[CTX_MASK_FAIL] ctx_cat dim={ctx_cat.shape[-1]} != cat_mask dim={cat_mask.shape[0]}"
            )
        return ctx_cat
    out = np.asarray(ctx_cat, dtype=np.int64).copy()
    neutrals = np.array(
        [NEUTRAL_CTX_CAT_VALUES[i] if i < len(NEUTRAL_CTX_CAT_VALUES) else 0 for i in range(out.shape[-1])],
        dtype=np.int64,
    )
    for i in range(out.shape[-1]):
        if cat_mask[i] == 0:
            if out.ndim == 1:
                out[i] = neutrals[i]
            else:
                out[..., i] = neutrals[i]
    return out
