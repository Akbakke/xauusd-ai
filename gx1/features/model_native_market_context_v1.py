"""Strict observed-market primitives shared by model-native build and serve."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _finite_or_fail(values: np.ndarray, *, label: str) -> None:
    if not np.isfinite(values).all():
        raise RuntimeError(
            f"[MODEL_NATIVE_CONTEXT_NONFINITE] {label}: "
            f"count={int(np.count_nonzero(~np.isfinite(values)))}"
        )


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


def derive_model_native_atr_spread_bps(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the exact raw ATR/spread context without any rank or bucket."""

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise RuntimeError("[MODEL_NATIVE_CONTEXT_EMPTY] ATR/spread source")
    required = ("high", "low", "close")
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise RuntimeError(
            f"[MODEL_NATIVE_CONTEXT_MISSING] ATR source fields: {missing}"
        )
    numeric = {
        name: pd.to_numeric(frame[name], errors="raise").to_numpy(
            dtype=np.float64
        )
        for name in required
    }
    for name, values in numeric.items():
        _finite_or_fail(values, label=name)
    high = numeric["high"]
    low = numeric["low"]
    close = numeric["close"]
    invalid = (
        (high <= 0.0)
        | (low <= 0.0)
        | (close <= 0.0)
        | (high < low)
        | (high < close)
        | (low > close)
    )
    if invalid.any():
        raise RuntimeError(
            "[MODEL_NATIVE_CONTEXT_INVALID] OHLC rows: "
            f"count={int(np.count_nonzero(invalid))}"
        )
    high_s = pd.Series(high, index=frame.index, dtype=np.float64)
    low_s = pd.Series(low, index=frame.index, dtype=np.float64)
    close_s = pd.Series(close, index=frame.index, dtype=np.float64)
    true_range = pd.concat(
        [
            high_s - low_s,
            (high_s - close_s.shift(1)).abs(),
            (low_s - close_s.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(14, min_periods=1).mean().to_numpy(
        dtype=np.float64
    )
    atr_bps = atr / ((high + low) * 0.5) * 1e4
    spread_bps = derive_observed_spread_bps(frame)
    _finite_or_fail(atr_bps, label="atr_bps")
    if np.any(atr_bps <= 0.0):
        raise RuntimeError("[MODEL_NATIVE_CONTEXT_INVALID] nonpositive atr_bps")
    return pd.DataFrame(
        {"atr": atr, "atr_bps": atr_bps, "spread_bps": spread_bps},
        index=frame.index,
    )
