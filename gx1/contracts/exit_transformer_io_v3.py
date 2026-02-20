#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exit Transformer IO Version 3 – IOV3_CLEAN contract (frozen).

Single representation per concept; append-only. TRUTH/SMOKE: only IOV3_CLEAN allowed.
Row layout: state (entry_price, price_now, spread_bps_now, pnl_bps_now, mfe_bps, atr_bps_now,
time_since_mfe_bars, ...), signals, entry_snapshot, context (ctx_cont 6, ctx_cat 6).
See gx1/docs/EXIT_IOV3_CLEAN.md.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List

import numpy as np

from gx1.contracts.exit_transformer_io_v2 import (
    DEFAULT_CTX_CAT_DIM,
    DEFAULT_CTX_CONT_DIM,
    feature_dim_v2,
    ordered_feature_names_v2,
    row_to_feature_vector_v2,
)

EXIT_IO_VERSION = "IOV3_CLEAN"
EXIT_TRANSFORMER_IO_V3_ID = "EXIT_TRANSFORMER_IO_V3"

# ONE UNIVERSE: fixed ctx dims
ONE_UNIVERSE_CTX_CONT_DIM = 6
ONE_UNIVERSE_CTX_CAT_DIM = 6

IOV2_DIM = feature_dim_v2(ctx_cont_dim=DEFAULT_CTX_CONT_DIM, ctx_cat_dim=DEFAULT_CTX_CAT_DIM)

# V3 extras (append after IOV2); one representation per concept (duplicates removed)
# - drawdown: only in IOV2 base (dd_from_mfe_bps, idx 15). Not duplicated here.
# - time since MFE: only in IOV2 base (time_since_mfe_bars, idx 17). Not duplicated here.
ORDERED_EXIT_EXTRA_FIELDS_V3: List[str] = [
    "pnl_over_atr",                     # pnl_bps_now / atr_bps_now
    "spread_over_atr",                  # spread_bps_now / atr_bps_now
    "price_dist_from_entry_bps",        # (price_now - entry_price) / entry_price * 10000
    "price_dist_from_entry_over_atr",   # price_dist_from_entry_bps / atr_bps_now
]
V3_EXTRAS_DIM = len(ORDERED_EXIT_EXTRA_FIELDS_V3)

# Full ordered feature list (35): IOV2 names + V3 extras
ORDERED_EXIT_FEATURES_V3: List[str] = (
    ordered_feature_names_v2(ctx_cont_dim=ONE_UNIVERSE_CTX_CONT_DIM, ctx_cat_dim=ONE_UNIVERSE_CTX_CAT_DIM)
    + list(ORDERED_EXIT_EXTRA_FIELDS_V3)
)
EXIT_IO_FEATURE_COUNT = len(ORDERED_EXIT_FEATURES_V3)
FEATURE_DIM_V3 = EXIT_IO_FEATURE_COUNT

MIN_ATR_BPS = 0.1


def ordered_feature_names_v3(
    ctx_cont_dim: int = DEFAULT_CTX_CONT_DIM,
    ctx_cat_dim: int = DEFAULT_CTX_CAT_DIM,
) -> List[str]:
    """Ordered feature names for IOV3_CLEAN (EXIT_IO_FEATURE_COUNT for ONE UNIVERSE 6/6)."""
    if ctx_cont_dim != ONE_UNIVERSE_CTX_CONT_DIM or ctx_cat_dim != ONE_UNIVERSE_CTX_CAT_DIM:
        raise RuntimeError(
            f"[EXIT_TRANSFORMER_IO_V3] ONE UNIVERSE: ctx must be 6/6, got {ctx_cont_dim}/{ctx_cat_dim}"
        )
    return list(ORDERED_EXIT_FEATURES_V3)


def _require_dict(obj: Any, ctx: str) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        raise RuntimeError(f"[EXIT_TRANSFORMER_IO_V3] expected dict for {ctx}, got {type(obj).__name__}")
    return obj


def _require_key(d: Dict[str, Any], key: str, ctx: str) -> Any:
    if key not in d:
        raise RuntimeError(f"[EXIT_TRANSFORMER_IO_V3] missing required key {ctx}.{key!r}")
    return d[key]


def _require_float(d: Dict[str, Any], key: str, ctx: str) -> float:
    v = _require_key(d, key, ctx)
    try:
        x = float(v)
    except (TypeError, ValueError) as e:
        raise RuntimeError(
            f"[EXIT_TRANSFORMER_IO_V3] {ctx}.{key!r} must be numeric, got {type(v).__name__}: {e}"
        ) from e
    if not np.isfinite(x):
        raise RuntimeError(f"[EXIT_TRANSFORMER_IO_V3] {ctx}.{key!r} must be finite, got {v!r}")
    return x


def _require_int_nonneg(d: Dict[str, Any], key: str, ctx: str) -> int:
    v = _require_key(d, key, ctx)
    try:
        i = int(v)
    except (TypeError, ValueError) as e:
        raise RuntimeError(
            f"[EXIT_TRANSFORMER_IO_V3] {ctx}.{key!r} must be int-compatible, got {v!r}: {e}"
        ) from e
    if i < 0:
        raise RuntimeError(f"[EXIT_TRANSFORMER_IO_V3] {ctx}.{key!r} must be >= 0, got {i}")
    return i


def validate_row_v3(
    row: Any,
    ctx_cont_dim: int = ONE_UNIVERSE_CTX_CONT_DIM,
    ctx_cat_dim: int = ONE_UNIVERSE_CTX_CAT_DIM,
) -> None:
    """Validate row for IOV3. Raises RuntimeError on first violation."""
    if ctx_cont_dim != ONE_UNIVERSE_CTX_CONT_DIM or ctx_cat_dim != ONE_UNIVERSE_CTX_CAT_DIM:
        raise RuntimeError(
            f"[EXIT_TRANSFORMER_IO_V3] ONE UNIVERSE: ctx dims must be 6/6, got {ctx_cont_dim}/{ctx_cat_dim}"
        )

    row_d = _require_dict(row, "row")
    for top in ("state", "signals", "entry_snapshot", "context"):
        if top not in row_d:
            raise RuntimeError(f"[EXIT_TRANSFORMER_IO_V3] missing required top-level key {top!r}")

    state = _require_dict(row_d["state"], "row.state")
    ctx = _require_dict(row_d["context"], "row.context")
    ctx_cont = ctx.get("ctx_cont")
    ctx_cat = ctx.get("ctx_cat")
    if not isinstance(ctx_cont, (list, tuple)) or len(ctx_cont) != ctx_cont_dim:
        raise RuntimeError(
            f"[EXIT_TRANSFORMER_IO_V3] row.context.ctx_cont must be length-{ctx_cont_dim}, "
            f"got {type(ctx_cont).__name__} len={len(ctx_cont) if hasattr(ctx_cont, '__len__') else 'N/A'}"
        )
    if not isinstance(ctx_cat, (list, tuple)) or len(ctx_cat) != ctx_cat_dim:
        raise RuntimeError(
            f"[EXIT_TRANSFORMER_IO_V3] row.context.ctx_cat must be length-{ctx_cat_dim}, "
            f"got {type(ctx_cat).__name__} len={len(ctx_cat) if hasattr(ctx_cat, '__len__') else 'N/A'}"
        )

    for i, v in enumerate(ctx_cont):
        try:
            x = float(v)
        except Exception:
            x = float("nan")
        if not np.isfinite(x):
            raise RuntimeError(f"[EXIT_TRANSFORMER_IO_V3] row.context.ctx_cont[{i}] must be finite, got {v!r}")
    for i, v in enumerate(ctx_cat):
        try:
            x = float(v)
        except Exception:
            x = float("nan")
        if not np.isfinite(x):
            raise RuntimeError(f"[EXIT_TRANSFORMER_IO_V3] row.context.ctx_cat[{i}] must be finite, got {v!r}")

    if "pnl_bps_now" not in state and "pnl_bps" not in state:
        raise RuntimeError("[EXIT_TRANSFORMER_IO_V3] state must have pnl_bps_now or pnl_bps")
    _require_float(state, "mfe_bps", "state")
    atr_bps_now = _require_float(state, "atr_bps_now", "state")
    if atr_bps_now < MIN_ATR_BPS:
        raise RuntimeError(
            f"[EXIT_TRANSFORMER_IO_V3] state.atr_bps_now too small: {atr_bps_now} < {MIN_ATR_BPS}"
        )
    _require_int_nonneg(state, "time_since_mfe_bars", "state")
    entry_price = _require_float(state, "entry_price", "state")
    price_now = _require_float(state, "price_now", "state")
    _require_float(state, "spread_bps_now", "state")
    if entry_price <= 0.0 or price_now <= 0.0:
        raise RuntimeError("[EXIT_TRANSFORMER_IO_V3] state.entry_price and price_now must be > 0")


def validate_window_v3(
    x: np.ndarray,
    window_len: int,
    feature_dim: int,
    context: str = "unknown",
) -> None:
    """Validate IOV3 window: shape (window_len, EXIT_IO_FEATURE_COUNT), no NaN/Inf."""
    if feature_dim != EXIT_IO_FEATURE_COUNT:
        raise RuntimeError(
            f"[EXIT_TRANSFORMER_IO_V3] feature_dim must be {EXIT_IO_FEATURE_COUNT}, got {feature_dim} ({context})"
        )
    arr = np.asarray(x)
    if arr.ndim == 2:
        if arr.shape != (window_len, feature_dim):
            raise RuntimeError(
                f"[EXIT_TRANSFORMER_IO_V3] window shape expected ({window_len}, {feature_dim}), got {arr.shape} ({context})"
            )
    elif arr.ndim == 3:
        if arr.shape[1] != window_len or arr.shape[2] != feature_dim:
            raise RuntimeError(
                f"[EXIT_TRANSFORMER_IO_V3] batch shape expected (B, {window_len}, {feature_dim}), got {arr.shape} ({context})"
            )
    else:
        raise RuntimeError(f"[EXIT_TRANSFORMER_IO_V3] expected 2D or 3D array, got ndim={arr.ndim} ({context})")
    if not np.isfinite(arr).all():
        n_bad = int((~np.isfinite(arr)).sum())
        raise RuntimeError(f"[EXIT_TRANSFORMER_IO_V3] non-finite values: count={n_bad} ({context})")


def row_to_feature_vector_v3(
    row: Any,
    ctx_cont_dim: int = DEFAULT_CTX_CONT_DIM,
    ctx_cat_dim: int = DEFAULT_CTX_CAT_DIM,
) -> np.ndarray:
    """Build one bar IOV3 feature vector (length EXIT_IO_FEATURE_COUNT). No fallback; RuntimeError on missing/invalid."""
    if ctx_cont_dim != ONE_UNIVERSE_CTX_CONT_DIM or ctx_cat_dim != ONE_UNIVERSE_CTX_CAT_DIM:
        raise RuntimeError(
            f"[EXIT_TRANSFORMER_IO_V3] ONE UNIVERSE: ctx must be 6/6, got {ctx_cont_dim}/{ctx_cat_dim}"
        )
    validate_row_v3(row, ctx_cont_dim=ctx_cont_dim, ctx_cat_dim=ctx_cat_dim)

    vec_v2 = row_to_feature_vector_v2(row, ctx_cont_dim=ctx_cont_dim, ctx_cat_dim=ctx_cat_dim)
    state = row["state"]
    pnl_bps_now = float(state.get("pnl_bps_now", state.get("pnl_bps", 0.0)))
    atr_bps_now = float(state["atr_bps_now"])
    spread_bps_now = float(state["spread_bps_now"])
    entry_price = float(state["entry_price"])
    price_now = float(state["price_now"])

    pnl_over_atr = pnl_bps_now / atr_bps_now
    spread_over_atr = spread_bps_now / atr_bps_now
    price_dist_from_entry_bps = (price_now - entry_price) / entry_price * 10000.0
    price_dist_from_entry_over_atr = price_dist_from_entry_bps / atr_bps_now

    extras = np.array(
        [
            pnl_over_atr,
            spread_over_atr,
            price_dist_from_entry_bps,
            price_dist_from_entry_over_atr,
        ],
        dtype=np.float32,
    )
    if not np.isfinite(extras).all():
        raise RuntimeError("[EXIT_TRANSFORMER_IO_V3] computed V3 extras contain non-finite values")

    out = np.zeros(FEATURE_DIM_V3, dtype=np.float32)
    out[:IOV2_DIM] = vec_v2
    out[IOV2_DIM:] = extras

    if out.shape != (FEATURE_DIM_V3,) or not np.isfinite(out).all():
        raise RuntimeError("[EXIT_TRANSFORMER_IO_V3] output shape or finiteness check failed")
    return out


def config_hash_v3(window_len: int, d_model: int, n_layers: int) -> str:
    """Stable hash for IOV3_CLEAN transformer config (feature set + architecture)."""
    names = "|".join(ORDERED_EXIT_FEATURES_V3)
    canonical = (
        f"io={EXIT_IO_VERSION}|window_len={window_len}|d_model={d_model}|n_layers={n_layers}|"
        f"features={names}"
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def assert_exit_io_v3_clean_in_truth() -> None:
    """TRUTH/SMOKE: hard-fail if exit IO is not IOV3_CLEAN. Call at runtime when building/using exit features."""
    try:
        from gx1.utils.truth_banlist import is_truth_or_smoke
        if not is_truth_or_smoke():
            return
    except Exception:
        return
    if EXIT_IO_VERSION != "IOV3_CLEAN":
        raise RuntimeError(
            f"[EXIT_TRANSFORMER_IO_V3] TRUTH violation: exit IO must be IOV3_CLEAN, got {EXIT_IO_VERSION!r}"
        )


__all__ = [
    "EXIT_IO_VERSION",
    "EXIT_IO_FEATURE_COUNT",
    "EXIT_TRANSFORMER_IO_V3_ID",
    "ORDERED_EXIT_FEATURES_V3",
    "ORDERED_EXIT_EXTRA_FIELDS_V3",
    "IOV2_DIM",
    "V3_EXTRAS_DIM",
    "FEATURE_DIM_V3",
    "MIN_ATR_BPS",
    "config_hash_v3",
    "ordered_feature_names_v3",
    "validate_row_v3",
    "validate_window_v3",
    "row_to_feature_vector_v3",
    "assert_exit_io_v3_clean_in_truth",
]
