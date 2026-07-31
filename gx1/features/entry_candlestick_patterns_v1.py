"""Canonical Entry candlestick-pattern features.

The layer encodes single, double and triple candle patterns as continuous
numeric scores. Row ``t`` describes the bar that closed at ``t + timeframe``;
the decision-time availability shift is owned by the dataset/live MTF slicer.
No second feature-layer shift is allowed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


CANDLESTICK_PATTERN_FEATURE_VERSION = (
    "entry_candlestick_patterns_v2_20260729_single_closed_bar_alignment_failclosed"
)
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
    if out.ndim != 1:
        raise RuntimeError(f"candlestick source field is not 1D: {name} shape={out.shape}")
    if out.size == 0:
        raise RuntimeError(f"candlestick source field is empty: {name}")
    if not np.isfinite(out).all():
        bad = int(np.flatnonzero(~np.isfinite(out))[0])
        raise RuntimeError(f"candlestick source field is non-finite: {name} row={bad}")
    return out


def _time_index(frame: object) -> pd.DatetimeIndex:
    try:
        values = frame["time"]  # type: ignore[index]
    except Exception as exc:
        raise RuntimeError("candlestick source field missing: time") from exc
    try:
        index = pd.DatetimeIndex(values)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("candlestick time source is invalid") from exc
    if index.empty:
        raise RuntimeError("candlestick time source is empty")
    if index.hasnans:
        raise RuntimeError("candlestick time source contains NaT")
    if index.tz is None:
        raise RuntimeError("candlestick time source must be timezone-aware UTC")
    if str(index.tz) != "UTC":
        raise RuntimeError(f"candlestick time source must use UTC: tz={index.tz}")
    if not index.is_monotonic_increasing or not index.is_unique:
        raise RuntimeError("candlestick time source must be strictly increasing and unique")
    return index


def _clip(arr: np.ndarray, lo: float = -25.0, hi: float = 25.0) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float32)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise RuntimeError(f"candlestick derived value is invalid: shape={values.shape}")
    return np.clip(values, lo, hi).astype(np.float32, copy=False)


def _clip01(arr: np.ndarray) -> np.ndarray:
    return _clip(arr, 0.0, 1.0)


def _lag(arr: np.ndarray, periods: int) -> np.ndarray:
    out = np.zeros_like(arr, dtype=np.float32)
    if periods <= 0:
        return arr.astype(np.float32, copy=False)
    if arr.size > periods:
        out[periods:] = arr[:-periods]
    return out


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return (num / np.maximum(np.abs(den), 1e-12)).astype(np.float32, copy=False)


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise RuntimeError(f"candlestick rolling source is invalid: shape={values.shape}")
    if values.size == 0:
        return values.astype(np.float32, copy=False)
    width = max(int(window), 1)
    csum = np.cumsum(np.concatenate(([0.0], values)))
    idx = np.arange(values.size)
    start = np.maximum(0, idx - width + 1)
    count = idx - start + 1
    return ((csum[idx + 1] - csum[start]) / count).astype(np.float32, copy=False)


def _add(arrays: list[np.ndarray], names: list[str], name: str, arr: np.ndarray, *, lo: float = -25.0, hi: float = 25.0) -> None:
    clean = _clip(np.asarray(arr, dtype=np.float32), lo, hi)
    if clean.ndim != 1:
        raise RuntimeError(f"candlestick feature {name} is not 1D: {clean.shape}")
    if not np.isfinite(clean).all():
        raise RuntimeError(f"candlestick feature {name} contains non-finite values")
    arrays.append(clean)
    names.append(f"{CANDLESTICK_PATTERN_FEATURE_PREFIX}{name}")


def missing_candlestick_source_fields(columns: object) -> list[str]:
    available = set(columns)
    return [name for name in CANDLESTICK_PATTERN_SOURCE_FIELDS if name not in available]


def build_entry_candlestick_pattern_layer(frame: object) -> tuple[np.ndarray, list[str]]:
    columns = getattr(frame, "columns", None)
    if columns is None:
        try:
            columns = frame.keys()  # type: ignore[union-attr]
        except Exception as exc:
            raise RuntimeError("candlestick source does not expose canonical columns") from exc
    missing = missing_candlestick_source_fields(columns)
    if missing:
        raise RuntimeError(f"CANDLESTICK_PATTERN_SOURCE_FIELDS_MISSING: {missing}")
    times = _time_index(frame)
    open_ = _arr(frame, "open")
    high = _arr(frame, "high")
    low = _arr(frame, "low")
    close = _arr(frame, "close")
    n = len(close)
    if not (len(times) == len(open_) == len(high) == len(low) == n):
        raise RuntimeError("candlestick source arrays have incompatible lengths")
    if np.any(open_ <= 0.0) or np.any(high <= 0.0) or np.any(low <= 0.0) or np.any(close <= 0.0):
        raise RuntimeError("candlestick OHLC source prices must be positive")
    invalid_geometry = (
        (high < low)
        | (high < open_)
        | (high < close)
        | (low > open_)
        | (low > close)
    )
    if invalid_geometry.any():
        row = int(np.flatnonzero(invalid_geometry)[0])
        raise RuntimeError(f"candlestick OHLC geometry is invalid: row={row}")

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
    prev_range = _lag(raw_range, 1)
    prev_body = _lag(body, 1)
    prev_bull = _lag(bull, 1)
    prev_bear = _lag(bear, 1)
    lag2_open = _lag(open_, 2)
    lag2_close = _lag(close, 2)
    lag2_range = _lag(raw_range, 2)
    lag2_body = _lag(body, 2)
    lag2_bull = _lag(bull, 2)
    lag2_bear = _lag(bear, 2)

    doji = _clip01((0.18 - body_share) / 0.18)
    long_body = _clip01((body_share - 0.55) / 0.35)
    small_body = _clip01((0.35 - body_share) / 0.35)
    prev_long_body = _lag(long_body, 1)
    lag2_long_body = _lag(long_body, 2)
    prev_small_body = _lag(small_body, 1)
    prev_doji = _lag(doji, 1)
    avg_range_5 = _rolling_mean(raw_range, 5)
    avg_body_5 = _rolling_mean(body, 5)
    prev_avg_range_5 = _lag(avg_range_5, 1)
    prev_avg_body_5 = _lag(avg_body_5, 1)
    range_baseline = np.where(prev_avg_range_5 > 1e-12, prev_avg_range_5, raw_range)
    body_baseline = np.where(prev_avg_body_5 > 1e-12, prev_avg_body_5, np.maximum(body, 1e-12))
    range_expansion_vs_prev5 = _clip01(_safe_div(raw_range, range_baseline) - 1.0)
    range_compression_vs_prev5 = _clip01(1.0 - _safe_div(raw_range, range_baseline))
    body_expansion_vs_prev5 = _clip01(_safe_div(body, body_baseline) - 1.0)
    body_compression_vs_prev5 = _clip01(1.0 - _safe_div(body, body_baseline))
    close_pressure_signed = _clip(close_loc * 2.0 - 1.0, -1.0, 1.0)
    wick_imbalance_signed = _clip(lower_share - upper_share, -1.0, 1.0)
    upper_wick_to_body = _clip01(_safe_div(upper, np.maximum(body, raw_range * 0.08)) / 3.0)
    lower_wick_to_body = _clip01(_safe_div(lower, np.maximum(body, raw_range * 0.08)) / 3.0)
    wick_dominance = _clip01(np.maximum(upper_share, lower_share) * 1.35 - body_share * 0.35)
    lower_rejection_quality = _clip01(
        lower_share * 1.55 + close_loc * 0.55 + lower_wick_to_body * 0.35 - upper_share * 0.65 - body_share * 0.20
    )
    upper_rejection_quality = _clip01(
        upper_share * 1.55 + (1.0 - close_loc) * 0.55 + upper_wick_to_body * 0.35 - lower_share * 0.65 - body_share * 0.20
    )
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
    inside_event = ((high <= prev_high) & (low >= prev_low)).astype(np.float32)
    outside_event = ((high >= prev_high) & (low <= prev_low)).astype(np.float32)
    inside_range_quality = _clip01(1.0 - _safe_div(raw_range, prev_range))
    inside_mid_close = _clip01(1.0 - np.abs(close_loc - 0.5) * 2.0)
    inside_bar = inside_event * small_body
    inside_bar_quality = inside_event * _clip01(0.45 * inside_range_quality + 0.25 * small_body + 0.20 * inside_mid_close + 0.10 * body_compression_vs_prev5)
    inside_bar_chain = inside_event * _clip01(0.50 * inside_range_quality + 0.35 * _lag(inside_event, 1) + 0.15 * _lag(inside_event, 2))
    outside_range_quality = _clip01(_safe_div(raw_range, prev_range) - 0.8)
    outside_bar = outside_event * outside_range_quality
    outside_bar_quality = outside_event * _clip01(
        0.35 * outside_range_quality + 0.25 * body_share + 0.20 * np.abs(close_pressure_signed) + 0.20 * range_expansion_vs_prev5
    )
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
    tweezer_low_match = _clip01(1.0 - np.abs(low - prev_low) / np.maximum(np.maximum(raw_range, prev_range) * 0.25, 1e-12))
    tweezer_high_match = _clip01(1.0 - np.abs(high - prev_high) / np.maximum(np.maximum(raw_range, prev_range) * 0.25, 1e-12))
    tweezer_bottom = _clip01(1.0 - np.abs(low - prev_low) / raw_range) * prev_bear * bull
    tweezer_top = _clip01(1.0 - np.abs(high - prev_high) / raw_range) * prev_bull * bear
    tweezer_bottom_quality = (
        prev_bear
        * bull
        * tweezer_low_match
        * _clip01(0.35 * lower_rejection_quality + 0.25 * _lag(lower_rejection_quality, 1) + 0.25 * close_loc + 0.15 * body_expansion_vs_prev5)
    )
    tweezer_top_quality = (
        prev_bull
        * bear
        * tweezer_high_match
        * _clip01(0.35 * upper_rejection_quality + 0.25 * _lag(upper_rejection_quality, 1) + 0.25 * (1.0 - close_loc) + 0.15 * body_expansion_vs_prev5)
    )
    engulf_body_quality = _clip01(_safe_div(body, np.maximum(prev_body, 1e-12)) - 0.75)
    engulf_range_quality = _clip01(_safe_div(raw_range, np.maximum(prev_range, 1e-12)) - 0.75)
    bullish_engulfing_quality = bullish_engulfing * _clip01(0.50 * engulf_body_quality + 0.20 * engulf_range_quality + 0.20 * close_loc + 0.10 * body_expansion_vs_prev5)
    bearish_engulfing_quality = bearish_engulfing * _clip01(0.50 * engulf_body_quality + 0.20 * engulf_range_quality + 0.20 * (1.0 - close_loc) + 0.10 * body_expansion_vs_prev5)
    bullish_engulfing_followthrough = _lag(bullish_engulfing_quality, 1) * bull * _clip01(
        0.60 * _safe_div(close - prev_high, prev_range) + 0.40 * _safe_div(close - prev_close, prev_range)
    )
    bearish_engulfing_followthrough = _lag(bearish_engulfing_quality, 1) * bear * _clip01(
        0.60 * _safe_div(prev_low - close, prev_range) + 0.40 * _safe_div(prev_close - close, prev_range)
    )
    outside_after_inside_bull = outside_event * _lag(inside_event, 1) * bull * _clip01(
        0.60 * _safe_div(close - prev_high, prev_range) + 0.25 * close_loc + 0.15 * body_share
    )
    outside_after_inside_bear = outside_event * _lag(inside_event, 1) * bear * _clip01(
        0.60 * _safe_div(prev_low - close, prev_range) + 0.25 * (1.0 - close_loc) + 0.15 * body_share
    )

    morning_star = (
        lag2_bear
        * _clip01(lag2_body / np.maximum(raw_range, 1e-12) - 0.35)
        * prev_small_body
        * bull
        * _clip01((close - (lag2_open + lag2_close) * 0.5) / np.maximum(lag2_body, 1e-12))
    )
    evening_star = (
        lag2_bull
        * _clip01(lag2_body / np.maximum(raw_range, 1e-12) - 0.35)
        * prev_small_body
        * bear
        * _clip01(((lag2_open + lag2_close) * 0.5 - close) / np.maximum(lag2_body, 1e-12))
    )
    star_middle_pause = _clip01(0.60 * prev_small_body + 0.40 * prev_doji)
    morning_star_quality = (
        lag2_bear
        * lag2_long_body
        * star_middle_pause
        * bull
        * _clip01(
            0.45 * _safe_div(close - (lag2_open + lag2_close) * 0.5, np.maximum(lag2_body, 1e-12))
            + 0.25 * close_loc
            + 0.15 * body_expansion_vs_prev5
            + 0.15 * lower_rejection_quality
        )
    )
    evening_star_quality = (
        lag2_bull
        * lag2_long_body
        * star_middle_pause
        * bear
        * _clip01(
            0.45 * _safe_div((lag2_open + lag2_close) * 0.5 - close, np.maximum(lag2_body, 1e-12))
            + 0.25 * (1.0 - close_loc)
            + 0.15 * body_expansion_vs_prev5
            + 0.15 * upper_rejection_quality
        )
    )
    three_white_soldiers = bull * prev_bull * lag2_bull * _clip01(long_body + prev_long_body + lag2_long_body - 0.8) * (close > prev_close).astype(np.float32) * (prev_close > lag2_close).astype(np.float32)
    three_black_crows = bear * prev_bear * lag2_bear * _clip01(long_body + prev_long_body + lag2_long_body - 0.8) * (close < prev_close).astype(np.float32) * (prev_close < lag2_close).astype(np.float32)
    three_bar_bull_reversal = lag2_bear * prev_bear * bull * _clip01(
        0.35 * lower_rejection_quality
        + 0.30 * _safe_div(close - prev_close, np.maximum(prev_range, 1e-12))
        + 0.20 * close_loc
        + 0.15 * body_expansion_vs_prev5
    )
    three_bar_bear_reversal = lag2_bull * prev_bull * bear * _clip01(
        0.35 * upper_rejection_quality
        + 0.30 * _safe_div(prev_close - close, np.maximum(prev_range, 1e-12))
        + 0.20 * (1.0 - close_loc)
        + 0.15 * body_expansion_vs_prev5
    )
    three_bar_bull_continuation = three_white_soldiers * _clip01(
        0.30 * long_body + 0.25 * prev_long_body + 0.20 * lag2_long_body + 0.15 * close_loc + 0.10 * range_expansion_vs_prev5
    )
    three_bar_bear_continuation = three_black_crows * _clip01(
        0.30 * long_body + 0.25 * prev_long_body + 0.20 * lag2_long_body + 0.15 * (1.0 - close_loc) + 0.10 * range_expansion_vs_prev5
    )
    prev_close_slope_up = _clip01(_safe_div(prev_close - lag2_close, np.maximum(lag2_range, 1e-12)))
    prev_close_slope_down = _clip01(_safe_div(lag2_close - prev_close, np.maximum(lag2_range, 1e-12)))
    bull_rejection_continuation = lower_rejection_quality * bull * _clip01(prev_bull + prev_close_slope_up) * _clip01(
        0.55 * _safe_div(close - prev_close, np.maximum(prev_range, 1e-12)) + 0.30 * close_loc + 0.15 * body_share
    )
    bear_rejection_continuation = upper_rejection_quality * bear * _clip01(prev_bear + prev_close_slope_down) * _clip01(
        0.55 * _safe_div(prev_close - close, np.maximum(prev_range, 1e-12)) + 0.30 * (1.0 - close_loc) + 0.15 * body_share
    )
    bull_rejection_followthrough = _lag(lower_rejection_quality, 1) * bull * _clip01(
        0.55 * _safe_div(close - prev_high, np.maximum(prev_range, 1e-12)) + 0.30 * _safe_div(close - prev_close, np.maximum(prev_range, 1e-12)) + 0.15 * close_loc
    )
    bear_rejection_followthrough = _lag(upper_rejection_quality, 1) * bear * _clip01(
        0.55 * _safe_div(prev_low - close, np.maximum(prev_range, 1e-12)) + 0.30 * _safe_div(prev_close - close, np.maximum(prev_range, 1e-12)) + 0.15 * (1.0 - close_loc)
    )

    bull_reversal = _clip01(
        hammer
        + bullish_engulfing_quality
        + piercing_line
        + tweezer_bottom_quality
        + morning_star_quality
        + lower_rejection_quality * bull
        + three_bar_bull_reversal
    )
    bear_reversal = _clip01(
        shooting_star
        + bearish_engulfing_quality
        + dark_cloud
        + tweezer_top_quality
        + evening_star_quality
        + upper_rejection_quality * bear
        + three_bar_bear_reversal
    )
    bull_continuation = _clip01(
        marubozu_bull
        + three_white_soldiers
        + outside_bar_quality * bull
        + bull_rejection_continuation
        + bullish_engulfing_followthrough
        + three_bar_bull_continuation
    )
    bear_continuation = _clip01(
        marubozu_bear
        + three_black_crows
        + outside_bar_quality * bear
        + bear_rejection_continuation
        + bearish_engulfing_followthrough
        + three_bar_bear_continuation
    )
    indecision_setup = _clip01(doji + inside_bar_quality + inside_bar_chain + wick_dominance * small_body)
    tail_risk = _clip01(doji + upper_share + lower_share + outside_bar_quality + wick_dominance)

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
    _add(arrays, names, "close_pressure_signed", close_pressure_signed, lo=-1.0, hi=1.0)
    _add(arrays, names, "wick_imbalance_signed", wick_imbalance_signed, lo=-1.0, hi=1.0)
    _add(arrays, names, "upper_wick_to_body_score", upper_wick_to_body, lo=0.0, hi=1.0)
    _add(arrays, names, "lower_wick_to_body_score", lower_wick_to_body, lo=0.0, hi=1.0)
    _add(arrays, names, "body_expansion_vs_prev5", body_expansion_vs_prev5, lo=0.0, hi=1.0)
    _add(arrays, names, "body_compression_vs_prev5", body_compression_vs_prev5, lo=0.0, hi=1.0)
    _add(arrays, names, "range_expansion_vs_prev5", range_expansion_vs_prev5, lo=0.0, hi=1.0)
    _add(arrays, names, "range_compression_vs_prev5", range_compression_vs_prev5, lo=0.0, hi=1.0)
    _add(arrays, names, "wick_dominance_score", wick_dominance, lo=0.0, hi=1.0)
    _add(arrays, names, "bull_lower_wick_rejection_quality", lower_rejection_quality, lo=0.0, hi=1.0)
    _add(arrays, names, "bear_upper_wick_rejection_quality", upper_rejection_quality, lo=0.0, hi=1.0)
    _add(arrays, names, "bullish_engulfing_quality", bullish_engulfing_quality, lo=0.0, hi=1.0)
    _add(arrays, names, "bearish_engulfing_quality", bearish_engulfing_quality, lo=0.0, hi=1.0)
    _add(arrays, names, "bullish_engulfing_followthrough_quality", bullish_engulfing_followthrough, lo=0.0, hi=1.0)
    _add(arrays, names, "bearish_engulfing_followthrough_quality", bearish_engulfing_followthrough, lo=0.0, hi=1.0)
    _add(arrays, names, "inside_bar_quality", inside_bar_quality, lo=0.0, hi=1.0)
    _add(arrays, names, "inside_bar_chain_compression_score", inside_bar_chain, lo=0.0, hi=1.0)
    _add(arrays, names, "outside_bar_quality", outside_bar_quality, lo=0.0, hi=1.0)
    _add(arrays, names, "outside_after_inside_bull_breakout_score", outside_after_inside_bull, lo=0.0, hi=1.0)
    _add(arrays, names, "outside_after_inside_bear_breakout_score", outside_after_inside_bear, lo=0.0, hi=1.0)
    _add(arrays, names, "tweezer_bottom_quality", tweezer_bottom_quality, lo=0.0, hi=1.0)
    _add(arrays, names, "tweezer_top_quality", tweezer_top_quality, lo=0.0, hi=1.0)
    _add(arrays, names, "morning_star_bull_quality", morning_star_quality, lo=0.0, hi=1.0)
    _add(arrays, names, "evening_star_bear_quality", evening_star_quality, lo=0.0, hi=1.0)
    _add(arrays, names, "bull_rejection_continuation_score", bull_rejection_continuation, lo=0.0, hi=1.0)
    _add(arrays, names, "bear_rejection_continuation_score", bear_rejection_continuation, lo=0.0, hi=1.0)
    _add(arrays, names, "bull_rejection_followthrough_score", bull_rejection_followthrough, lo=0.0, hi=1.0)
    _add(arrays, names, "bear_rejection_followthrough_score", bear_rejection_followthrough, lo=0.0, hi=1.0)
    _add(arrays, names, "three_bar_bull_reversal_quality", three_bar_bull_reversal, lo=0.0, hi=1.0)
    _add(arrays, names, "three_bar_bear_reversal_quality", three_bar_bear_reversal, lo=0.0, hi=1.0)
    _add(arrays, names, "three_bar_bull_continuation_quality", three_bar_bull_continuation, lo=0.0, hi=1.0)
    _add(arrays, names, "three_bar_bear_continuation_quality", three_bar_bear_continuation, lo=0.0, hi=1.0)

    out = np.column_stack(arrays).astype(np.float32, copy=False) if arrays else np.empty((n, 0), dtype=np.float32)
    if not np.isfinite(out).all():
        raise RuntimeError("candlestick pattern layer contains non-finite values")
    if len(names) != len(set(names)):
        raise RuntimeError("candlestick pattern layer contains duplicate feature names")
    return out, names


_toy = {
    "time": pd.date_range("2000-01-01", periods=3, freq="5min", tz="UTC"),
    "open": np.asarray([1.0, 1.0, 1.2], dtype=np.float64),
    "high": np.asarray([1.2, 1.3, 1.4], dtype=np.float64),
    "low": np.asarray([0.9, 0.95, 1.1], dtype=np.float64),
    "close": np.asarray([1.1, 1.25, 1.35], dtype=np.float64),
}
CANDLESTICK_PATTERN_FEATURE_NAMES = tuple(build_entry_candlestick_pattern_layer(_toy)[1])
