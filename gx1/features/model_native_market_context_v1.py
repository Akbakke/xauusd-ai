"""Strict observed-market primitives shared by model-native build and serve."""

from __future__ import annotations

import numpy as np
import pandas as pd


MODEL_NATIVE_TREND_REGIME_D1_BOUNDS = (-1.0, 1.0)


def _finite_or_fail(values: np.ndarray, *, label: str) -> None:
    if not np.isfinite(values).all():
        raise RuntimeError(
            f"[MODEL_NATIVE_CONTEXT_NONFINITE] {label}: "
            f"count={int(np.count_nonzero(~np.isfinite(values)))}"
        )


def derive_model_native_trend_regime_id(
    d1_distance_from_ema200_atr: np.ndarray,
) -> np.ndarray:
    """Derive the shared three-state trend context from exact D1 evidence."""

    values = np.asarray(d1_distance_from_ema200_atr, dtype=np.float64)
    if values.ndim != 1:
        raise RuntimeError(
            "[MODEL_NATIVE_TREND_REGIME_SHAPE] "
            f"expected one dimension, observed={values.shape}"
        )
    _finite_or_fail(values, label="D1_dist_from_ema200_atr")
    lower, upper = MODEL_NATIVE_TREND_REGIME_D1_BOUNDS
    return np.where(
        values < lower,
        0,
        np.where(values <= upper, 1, 2),
    ).astype(np.int64)


def derive_observed_spread_bps(frame: pd.DataFrame) -> np.ndarray:
    """Return causal spread bps without defaults, clipping or zero fill."""

    if "spread_bps" in frame.columns:
        spread_bps = frame["spread_bps"].to_numpy(dtype=np.float64)
        _finite_or_fail(spread_bps, label="spread_bps")
        if np.any(spread_bps < 0.0):
            raise RuntimeError(
                "[MODEL_NATIVE_CONTEXT_INVALID] spread_bps contains negative values: "
                f"count={int(np.count_nonzero(spread_bps < 0.0))}"
            )
        return spread_bps

    if {"bid_close", "ask_close"}.issubset(frame.columns):
        bid = frame["bid_close"].to_numpy(dtype=np.float64)
        ask = frame["ask_close"].to_numpy(dtype=np.float64)
        _finite_or_fail(bid, label="bid_close")
        _finite_or_fail(ask, label="ask_close")
        invalid = (bid <= 0.0) | (ask < bid)
        if np.any(invalid):
            raise RuntimeError(
                "[MODEL_NATIVE_CONTEXT_INVALID] invalid bid/ask spread rows: "
                f"count={int(np.count_nonzero(invalid))}"
            )
        spread_bps = (ask - bid) / bid * 1e4
        _finite_or_fail(spread_bps, label="spread_bps_from_bid_ask")
        return spread_bps

    if {"spread", "close"}.issubset(frame.columns):
        spread = frame["spread"].to_numpy(dtype=np.float64)
        close = frame["close"].to_numpy(dtype=np.float64)
        _finite_or_fail(spread, label="spread")
        _finite_or_fail(close, label="close")
        invalid = (spread < 0.0) | (close <= 0.0)
        if np.any(invalid):
            raise RuntimeError(
                "[MODEL_NATIVE_CONTEXT_INVALID] invalid spread/close rows: "
                f"count={int(np.count_nonzero(invalid))}"
            )
        spread_bps = spread / close * 1e4
        _finite_or_fail(spread_bps, label="spread_bps_from_spread_close")
        return spread_bps

    raise RuntimeError(
        "[MODEL_NATIVE_CONTEXT_MISSING] observed spread requires spread_bps, "
        "bid_close+ask_close, or spread+close"
    )
