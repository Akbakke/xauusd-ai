"""Entry candlestick-pattern challenger features.

The layer encodes single, double and triple candle patterns as continuous
numeric scores. Outputs are shifted one bar so row t only sees patterns that
were fully closed before t.
"""
from __future__ import annotations

import numpy as np


CANDLESTICK_PATTERN_FEATURE_VERSION = "entry_candlestick_patterns_v1_20260630_closed_bar_numeric_patterns"
CANDLESTICK_PATTERN_FEATURE_PREFIX = "candle.pattern_"
CANDLESTICK_PATTERN_SOURCE_FIELDS = ("time", "open", "high", "low", "close")


def _arr(frame: object, name: str) -> np.ndarray:
    try:
        values = frame[name]  # type: ignore[index]
    except Exception as exc:
        raise RuntimeError(f"candlestick source field missing: {name}") from exc
    if hasattr(values, "to_numpy"):
        out = values.to_numpy(dtype=np.float64)
    else:
        out = np.asarray(values, dtype=np.float64)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _clip(arr: np.ndarray, lo: float = -25.0, hi: float = 25.0) -> np.ndarray:
    return np.clip(np.nan_to_num(arr, nan=0.0, posinf=hi, neginf=lo), lo, hi).astype(np.float32, copy=False)


def _clip01(arr: np.ndarray) -> np.ndarray:
    return _clip(arr, 0.0, 1.0)


def _shift1(arr: np.ndarray) -> np.ndarray:
    out = np.zeros_like(arr, dtype=np.float32)
    if arr.size > 1:
        out[1:] = arr[:-1]
    return out


def _lag(arr: np.ndarray, periods: int) -> np.ndarray:
    out = np.zeros_like(arr, dtype=np.float32)
    if periods <= 0:
        return arr.astype(np.float32, copy=False)
    if arr.size > periods:
        out[periods:] = arr[:-periods]
    return out


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return (num / np.maximum(np.abs(den), 1e-12)).astype(np.float32, copy=False)


def _add(arrays: list[np.ndarray], names: list[str], name: str, arr: np.ndarray, *, lo: float = -25.0, hi: float = 25.0) -> None:
    clean = _shift1(_clip(np.asarray(arr, dtype=np.float32), lo, hi))
    if clean.ndim != 1:
        raise RuntimeError(f"candlestick feature {name} is not 1D: {clean.shape}")
    if not np.isfinite(clean).all():
        raise RuntimeError(f"candlestick feature {name} contains non-finite values")
    arrays.append(clean)
    names.append(f"{CANDLESTICK_PATTERN_FEATURE_PREFIX}{name}")


def missing_candlestick_source_fields(columns: object) -> list[str]:
    available = {str(name) for name in columns}
    return [name for name in CANDLESTICK_PATTERN_SOURCE_FIELDS if name not in available]


def build_entry_candlestick_pattern_layer(frame: object) -> tuple[np.ndarray, list[str]]:
    open_ = _arr(frame, "open")
    high = _arr(frame, "high")
    low = _arr(frame, "low")
    close = _arr(frame, "close")
    n = len(close)
    if not (len(open_) == len(high) == len(low) == n):
        raise RuntimeError("candlestick source arrays have incompatible lengths")

    raw_range = np.maximum(high - low, 1e-12)
    body_signed = close - open_
    body = np.abs(body_signed)
    upper = np.maximum(high - np.maximum(open_, close), 0.0)
    lower = np.maximum(np.minimum(open_, close) - low, 0.0)
    body_share = _clip01(body / raw_range)
    upper_share = _clip01(upper / raw_range)
    lower_share = _clip01(lower / raw_range)
    close_loc = _clip01((close - low) / raw_range)
    direction = np.sign(body_signed).astype(np.float32)
    bull = (body_signed > 0.0).astype(np.float32)
    bear = (body_signed < 0.0).astype(np.float32)
    prev_open = _lag(open_, 1)
    prev_close = _lag(close, 1)
    prev_high = _lag(high, 1)
    prev_low = _lag(low, 1)
    prev_body = _lag(body, 1)
    prev_bull = _lag(bull, 1)
    prev_bear = _lag(bear, 1)
    lag2_open = _lag(open_, 2)
    lag2_close = _lag(close, 2)
    lag2_body = _lag(body, 2)
    lag2_bull = _lag(bull, 2)
    lag2_bear = _lag(bear, 2)

    doji = _clip01((0.18 - body_share) / 0.18)
    long_body = _clip01((body_share - 0.55) / 0.35)
    small_body = _clip01((0.35 - body_share) / 0.35)
    hammer = _clip01(lower_share * 1.8 - upper_share * 0.7 - body_share * 0.35) * _clip01(close_loc)
    shooting_star = _clip01(upper_share * 1.8 - lower_share * 0.7 - body_share * 0.35) * _clip01(1.0 - close_loc)
    marubozu_bull = bull * long_body * _clip01(close_loc)
    marubozu_bear = bear * long_body * _clip01(1.0 - close_loc)

    bullish_engulfing = (
        bull
        * prev_bear
        * _clip01(_safe_div(body, np.maximum(prev_body, 1e-12)) - 0.85)
        * (open_ <= prev_close).astype(np.float32)
        * (close >= prev_open).astype(np.float32)
    )
    bearish_engulfing = (
        bear
        * prev_bull
        * _clip01(_safe_div(body, np.maximum(prev_body, 1e-12)) - 0.85)
        * (open_ >= prev_close).astype(np.float32)
        * (close <= prev_open).astype(np.float32)
    )
    inside_bar = ((high <= prev_high) & (low >= prev_low)).astype(np.float32) * small_body
    outside_bar = ((high >= prev_high) & (low <= prev_low)).astype(np.float32) * _clip01(raw_range / np.maximum(prev_high - prev_low, 1e-12) - 0.8)
    piercing_line = (
        bull
        * prev_bear
        * (open_ < prev_close).astype(np.float32)
        * _clip01((close - (prev_open + prev_close) * 0.5) / np.maximum(prev_body, 1e-12))
    )
    dark_cloud = (
        bear
        * prev_bull
        * (open_ > prev_close).astype(np.float32)
        * _clip01(((prev_open + prev_close) * 0.5 - close) / np.maximum(prev_body, 1e-12))
    )
    tweezer_bottom = _clip01(1.0 - np.abs(low - prev_low) / raw_range) * prev_bear * bull
    tweezer_top = _clip01(1.0 - np.abs(high - prev_high) / raw_range) * prev_bull * bear

    morning_star = (
        lag2_bear
        * _clip01(lag2_body / np.maximum(raw_range, 1e-12) - 0.35)
        * _lag(small_body, 1)
        * bull
        * _clip01((close - (lag2_open + lag2_close) * 0.5) / np.maximum(lag2_body, 1e-12))
    )
    evening_star = (
        lag2_bull
        * _clip01(lag2_body / np.maximum(raw_range, 1e-12) - 0.35)
        * _lag(small_body, 1)
        * bear
        * _clip01(((lag2_open + lag2_close) * 0.5 - close) / np.maximum(lag2_body, 1e-12))
    )
    three_white_soldiers = bull * prev_bull * lag2_bull * _clip01(long_body + _lag(long_body, 1) + _lag(long_body, 2) - 0.8) * (close > prev_close).astype(np.float32) * (prev_close > lag2_close).astype(np.float32)
    three_black_crows = bear * prev_bear * lag2_bear * _clip01(long_body + _lag(long_body, 1) + _lag(long_body, 2) - 0.8) * (close < prev_close).astype(np.float32) * (prev_close < lag2_close).astype(np.float32)

    bull_reversal = _clip01(hammer + bullish_engulfing + piercing_line + tweezer_bottom + morning_star)
    bear_reversal = _clip01(shooting_star + bearish_engulfing + dark_cloud + tweezer_top + evening_star)
    bull_continuation = _clip01(marubozu_bull + three_white_soldiers + outside_bar * bull)
    bear_continuation = _clip01(marubozu_bear + three_black_crows + outside_bar * bear)
    indecision_setup = _clip01(doji + inside_bar)
    tail_risk = _clip01(doji + upper_share + lower_share + outside_bar)

    arrays: list[np.ndarray] = []
    names: list[str] = []
    _add(arrays, names, "body_share", body_share, lo=0.0, hi=1.0)
    _add(arrays, names, "upper_wick_share", upper_share, lo=0.0, hi=1.0)
    _add(arrays, names, "lower_wick_share", lower_share, lo=0.0, hi=1.0)
    _add(arrays, names, "close_location", close_loc, lo=0.0, hi=1.0)
    _add(arrays, names, "body_direction", direction, lo=-1.0, hi=1.0)
    _add(arrays, names, "doji_score", doji, lo=0.0, hi=1.0)
    _add(arrays, names, "hammer_bull_reversal_score", hammer, lo=0.0, hi=1.0)
    _add(arrays, names, "shooting_star_bear_reversal_score", shooting_star, lo=0.0, hi=1.0)
    _add(arrays, names, "marubozu_bull_score", marubozu_bull, lo=0.0, hi=1.0)
    _add(arrays, names, "marubozu_bear_score", marubozu_bear, lo=0.0, hi=1.0)
    _add(arrays, names, "bullish_engulfing_score", bullish_engulfing, lo=0.0, hi=1.0)
    _add(arrays, names, "bearish_engulfing_score", bearish_engulfing, lo=0.0, hi=1.0)
    _add(arrays, names, "inside_bar_compression_score", inside_bar, lo=0.0, hi=1.0)
    _add(arrays, names, "outside_bar_expansion_score", outside_bar, lo=0.0, hi=1.0)
    _add(arrays, names, "piercing_line_bull_score", piercing_line, lo=0.0, hi=1.0)
    _add(arrays, names, "dark_cloud_bear_score", dark_cloud, lo=0.0, hi=1.0)
    _add(arrays, names, "tweezer_bottom_score", tweezer_bottom, lo=0.0, hi=1.0)
    _add(arrays, names, "tweezer_top_score", tweezer_top, lo=0.0, hi=1.0)
    _add(arrays, names, "morning_star_bull_score", morning_star, lo=0.0, hi=1.0)
    _add(arrays, names, "evening_star_bear_score", evening_star, lo=0.0, hi=1.0)
    _add(arrays, names, "three_white_soldiers_score", three_white_soldiers, lo=0.0, hi=1.0)
    _add(arrays, names, "three_black_crows_score", three_black_crows, lo=0.0, hi=1.0)
    _add(arrays, names, "bull_reversal_pressure", bull_reversal, lo=0.0, hi=1.0)
    _add(arrays, names, "bear_reversal_pressure", bear_reversal, lo=0.0, hi=1.0)
    _add(arrays, names, "bull_continuation_pressure", bull_continuation, lo=0.0, hi=1.0)
    _add(arrays, names, "bear_continuation_pressure", bear_continuation, lo=0.0, hi=1.0)
    _add(arrays, names, "indecision_breakout_setup", indecision_setup, lo=0.0, hi=1.0)
    _add(arrays, names, "tail_rejection_risk", tail_risk, lo=0.0, hi=1.0)

    out = np.column_stack(arrays).astype(np.float32, copy=False) if arrays else np.empty((n, 0), dtype=np.float32)
    if not np.isfinite(out).all():
        raise RuntimeError("candlestick pattern layer contains non-finite values")
    return out, names


_toy = {
    "open": np.asarray([1.0, 1.0, 1.2], dtype=np.float64),
    "high": np.asarray([1.2, 1.3, 1.4], dtype=np.float64),
    "low": np.asarray([0.9, 0.95, 1.1], dtype=np.float64),
    "close": np.asarray([1.1, 1.25, 1.35], dtype=np.float64),
}
CANDLESTICK_PATTERN_FEATURE_NAMES = tuple(build_entry_candlestick_pattern_layer(_toy)[1])
