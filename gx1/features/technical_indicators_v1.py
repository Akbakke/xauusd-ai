"""Canonical causal technical-indicator primitives shared by every clock.

This module owns formula semantics only.  Callers must still pass independently
closed OHLCV for each timeframe and keep separate state/history per clock.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


TECHNICAL_INDICATOR_PRIMITIVES_VERSION = (
    "technical_indicator_primitives_v2_classic_seed_raw_20260814"
)
EMA50_200_FAST_SPAN = 50
EMA50_200_SLOW_SPAN = 200
EMA50_200_ATR_PERIOD = 14
EMA50_200_SPREAD_ATR_BLOCK_COLUMNS = (
    "atr14",
    "atr14_positive",
    "ema50",
    "ema200",
    "spread",
    "spread_atr",
)
TECHNICAL_INDICATOR_FORMULA_CONTRACT = (
    "ema=classic_sma_seed_after_exact_span_then_alpha_2_over_span_plus_1",
    "atr=classic_wilder_sma_seed_period14",
    "atr_denominator=defined_strictly_positive_else_unavailable",
    "ema50_200_spread_atr=raw_spread_over_positive_wilder_atr_no_clip",
    "warmup=causal_nan_prefix_no_partial_window_no_neutral_fill",
)
TECHNICAL_INDICATOR_FORMULA_SHA256 = hashlib.sha256(
    "\n".join(TECHNICAL_INDICATOR_FORMULA_CONTRACT).encode("utf-8")
).hexdigest()


def technical_indicator_contract_metadata() -> dict[str, object]:
    """Immutable formula identity consumed by dataset/cache contracts."""

    return {
        "owner": "gx1.features.technical_indicators_v1",
        "schema_version": TECHNICAL_INDICATOR_PRIMITIVES_VERSION,
        "formula_sha256": TECHNICAL_INDICATOR_FORMULA_SHA256,
        "formula_contract": list(TECHNICAL_INDICATOR_FORMULA_CONTRACT),
        "ema50_200_block_columns": list(EMA50_200_SPREAD_ATR_BLOCK_COLUMNS),
    }


def classic_ema(series: pd.Series, span: int) -> pd.Series:
    """Classic EMA with an SMA seed after exactly ``span`` observations."""

    if isinstance(span, bool) or not isinstance(span, int) or span <= 0:
        raise RuntimeError("[CLASSIC_EMA_SPAN_INVALID]")
    if not isinstance(series, pd.Series):
        raise RuntimeError("[CLASSIC_EMA_SOURCE_TYPE_INVALID]")
    source = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)
    if source.ndim != 1 or not np.isfinite(source).all():
        raise RuntimeError("[CLASSIC_EMA_SOURCE_NONFINITE]")
    out = np.full(len(source), np.nan, dtype=np.float64)
    if len(source) < span:
        return pd.Series(out, index=series.index, dtype=np.float64)
    value = float(np.mean(source[:span], dtype=np.float64))
    out[span - 1] = value
    alpha = 2.0 / (span + 1.0)
    for row in range(span, len(source)):
        value += alpha * (float(source[row]) - value)
        out[row] = value
    return pd.Series(out, index=series.index, dtype=np.float64)


def wilder_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int,
) -> pd.Series:
    """Return classic Wilder ATR with an SMA seed and recursive RMA updates.

    True range on the first row is ``high-low``.  The first defined ATR is at
    row ``period-1`` and is the arithmetic mean of the first ``period`` true
    ranges.  Later rows use ``((period-1)*previous + TR)/period``.  No neutral
    prefix is fabricated; unavailable warmup rows remain NaN for the caller's
    common-history owner to trim.
    """

    if isinstance(period, bool) or not isinstance(period, int) or period <= 0:
        raise RuntimeError("[WILDER_ATR_PERIOD_INVALID]")
    if not isinstance(high, pd.Series) or not isinstance(low, pd.Series) or not isinstance(close, pd.Series):
        raise RuntimeError("[WILDER_ATR_SOURCE_TYPE_INVALID]")
    if not high.index.equals(low.index) or not high.index.equals(close.index):
        raise RuntimeError("[WILDER_ATR_INDEX_MISMATCH]")
    high_values = pd.to_numeric(high, errors="coerce").to_numpy(dtype=np.float64)
    low_values = pd.to_numeric(low, errors="coerce").to_numpy(dtype=np.float64)
    close_values = pd.to_numeric(close, errors="coerce").to_numpy(dtype=np.float64)
    if not (
        len(high_values) == len(low_values) == len(close_values)
        and np.isfinite(high_values).all()
        and np.isfinite(low_values).all()
        and np.isfinite(close_values).all()
    ):
        raise RuntimeError("[WILDER_ATR_SOURCE_NONFINITE]")
    if (
        np.any(high_values <= 0.0)
        or np.any(low_values <= 0.0)
        or np.any(close_values <= 0.0)
        or np.any(high_values < low_values)
        or np.any(high_values < close_values)
        or np.any(low_values > close_values)
    ):
        raise RuntimeError("[WILDER_ATR_SOURCE_GEOMETRY_INVALID]")

    true_range = high_values - low_values
    if len(true_range) > 1:
        previous_close = close_values[:-1]
        true_range[1:] = np.maximum.reduce(
            (
                true_range[1:],
                np.abs(high_values[1:] - previous_close),
                np.abs(low_values[1:] - previous_close),
            )
        )
    out = np.full(len(true_range), np.nan, dtype=np.float64)
    if len(true_range) >= period:
        seed_index = period - 1
        out[seed_index] = float(np.mean(true_range[:period], dtype=np.float64))
        for row in range(period, len(true_range)):
            out[row] = (
                (period - 1) * out[row - 1] + true_range[row]
            ) / period
    return pd.Series(out, index=high.index, dtype=np.float64)


def wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Return classic Wilder RSI with SMA seed and recursive smoothing."""

    if isinstance(period, bool) or not isinstance(period, int) or period <= 0:
        raise RuntimeError("[WILDER_RSI_PERIOD_INVALID]")
    if not isinstance(close, pd.Series):
        raise RuntimeError("[WILDER_RSI_SOURCE_TYPE_INVALID]")
    values = pd.to_numeric(close, errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("[WILDER_RSI_SOURCE_NONFINITE]")
    out = np.full(len(values), np.nan, dtype=np.float64)
    if len(values) <= period:
        return pd.Series(out, index=close.index, dtype=np.float64)
    changes = np.diff(values)
    gains = np.maximum(changes, 0.0)
    losses = np.maximum(-changes, 0.0)
    average_gain = float(np.mean(gains[:period], dtype=np.float64))
    average_loss = float(np.mean(losses[:period], dtype=np.float64))

    def _value(gain: float, loss: float) -> float:
        if loss == 0.0:
            return 50.0 if gain == 0.0 else 100.0
        if gain == 0.0:
            return 0.0
        strength = gain / loss
        return 100.0 - 100.0 / (1.0 + strength)

    out[period] = _value(average_gain, average_loss)
    for row in range(period + 1, len(values)):
        change_index = row - 1
        average_gain = (
            (period - 1) * average_gain + gains[change_index]
        ) / period
        average_loss = (
            (period - 1) * average_loss + losses[change_index]
        ) / period
        out[row] = _value(average_gain, average_loss)
    return pd.Series(out, index=close.index, dtype=np.float64)


def wilder_atr14_positive(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Return Wilder-14 ATR and a positive-only honest denominator.

    A defined zero ATR carries no meaningful ATR-normalized quotient. It is
    unavailable (NaN), never replaced by a price floor or epsilon.
    """

    atr14 = wilder_atr(high, low, close, EMA50_200_ATR_PERIOD)
    atr14_positive = atr14.where(atr14 > 0.0)
    return atr14, atr14_positive


def ema50_200_spread_atr_block(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> pd.DataFrame:
    """Return the one float64 EMA50/200 ATR-normalized formula block.

    ATR is classic Wilder-14 from :func:`wilder_atr`; a zero ATR is
    unavailable. EMA50/200 both use :func:`classic_ema`, the same SMA-seeded
    convention as ``basic_v1``. The raw normalized spread is not clipped.
    Every local M1/M5 and per-TF caller consumes this exact float64 block and
    casts to its immutable float32 storage representation only afterwards.
    """

    atr14, atr14_positive = wilder_atr14_positive(high, low, close)
    close_float = pd.Series(
        close.to_numpy(dtype=np.float64),
        index=close.index,
        dtype=np.float64,
    )
    ema50 = classic_ema(close_float, EMA50_200_FAST_SPAN)
    ema200 = classic_ema(close_float, EMA50_200_SLOW_SPAN)
    spread = ema50 - ema200
    spread_atr = spread / atr14_positive
    out = pd.DataFrame(
        {
            "atr14": atr14,
            "atr14_positive": atr14_positive,
            "ema50": ema50,
            "ema200": ema200,
            "spread": spread,
            "spread_atr": spread_atr,
        },
        index=close.index,
        dtype=np.float64,
    )
    if tuple(out.columns) != EMA50_200_SPREAD_ATR_BLOCK_COLUMNS:
        raise RuntimeError("[EMA50_200_SPREAD_ATR_BLOCK_ORDER_INVALID]")
    return out
