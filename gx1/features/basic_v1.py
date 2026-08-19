# gx1/features/basic_v1.py
# -*- coding: utf-8 -*-
# Runtime hot-path: pandas forbidden
# All per-bar operations must use NumPy arrays directly.
# Pandas operations (.clip(), .reindex(), .rolling(), .fillna(), .shift(), .replace(), etc.) are removed from hot-path.
# Input must be float32/float64 - hard fail in replay mode on wrong dtype.
import hashlib
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import njit

from gx1.features.model_native_market_context_v1 import derive_observed_spread_bps
from gx1.features.technical_indicators_v1 import wilder_atr

M5_DECISION_DELAY_SECONDS = 5 * 60

PLUS5_FEATURES = (
    "atr",
    "std50",
    "roc20",
    "_v1_vwap_drift48",
)
PLUS5_FIRST_FINITE_ROW = {
    "atr": 13,
    "std50": 50,
    "roc20": 20,
    "_v1_vwap_drift48": 47,
}
BASIC_V1_FEATURES = (
    "_v1_atr14",
    "_v1_pk_sigma20",
    "_v1_ema_diff",
    "_v1_vwap_drift48",
    "_v1_bb10_bandwidth_change_3",
    "_v1_ema3_ema6_spread_atr",
    "_v1_range_z",
    "_v1_kurt_r",
    "_v1_tema20_change_3_atr",
    "_v1_bb_squeeze_20_2",
    "_v1_kama30_change_5_atr",
)
BASIC_V1_SCHEMA_VERSION = "gx1_basic_v1_active_surface_v4"
BASIC_V1_FEATURES_SHA256 = hashlib.sha256(
    "\n".join(BASIC_V1_FEATURES).encode("utf-8")
).hexdigest()
BASIC_V1_FORMULA_CONTRACT = (
    "atr14=classic_wilder_sma_seed_period14_current_closed_bar",
    "pk_sigma20=full_window20_current_closed_bar",
    "ema_diff=classic_ema_sma_seed_spans12_26_over_wilder_atr14_current_closed_bar",
    "vwap_drift48=full_volume_weighted_window48",
    "bb10_bandwidth_change3=full_window10_change3_current_closed_bar",
    "ema3_ema6_spread_atr=classic_ema_sma_seed_spans3_6_over_wilder_atr14_current_closed_bar",
    "range_z=full_window48_ddof0_current_closed_bar",
    "kurt_r=48_observed_returns_unbiased_fisher_current_closed_bar",
    "tema20_change3_atr=three_classic_ema_sma_seed_layers_change3_over_atr_current_closed_bar",
    "bb_squeeze20_2=full_band20_over_full_100_bandwidth_baseline_current_closed_bar",
    "kama30_change5_atr=talib_kama_price_seed_period30_change5_over_atr_current_closed_bar",
    "plus5_atr=classic_wilder_sma_seed_period14_current_closed_bar",
    "plus5_std50=50_observed_close_returns_sample_std",
    "plus5_roc20=exact_20bar_close_return",
    "plus5_volume=positive_whole_oanda_tick_count",
    "ratios=positive_defined_denominator_no_epsilon_or_neutral_fill",
    "warmup=causal_nan_prefix_no_neutral_fill",
)
BASIC_V1_FIRST_FINITE_ROW = {
    "_v1_atr14": 13,
    "_v1_pk_sigma20": 19,
    "_v1_ema_diff": 25,
    "_v1_vwap_drift48": 47,
    "_v1_bb10_bandwidth_change_3": 12,
    "_v1_ema3_ema6_spread_atr": 13,
    "_v1_range_z": 47,
    "_v1_kurt_r": 48,
    "_v1_tema20_change_3_atr": 60,
    "_v1_bb_squeeze_20_2": 118,
    "_v1_kama30_change_5_atr": 35,
}
BASIC_V1_FORMULA_SHA256 = hashlib.sha256(
    "\n".join(
        BASIC_V1_FORMULA_CONTRACT
        + tuple(
            f"basic_first_finite:{name}={BASIC_V1_FIRST_FINITE_ROW[name]}"
            for name in BASIC_V1_FEATURES
        )
        + tuple(
            f"plus5_first_finite:{name}={PLUS5_FIRST_FINITE_ROW[name]}"
            for name in PLUS5_FEATURES
        )
    ).encode("utf-8")
).hexdigest()


def compute_plus5_features(df: pd.DataFrame) -> pd.DataFrame:
    """One train/serve owner for the retained local-clock auxiliary fields."""
    required = ("open", "high", "low", "close", "volume")
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise RuntimeError(
            f"[PLUS5_SOURCE_MISSING] required OHLCV columns missing: {missing}"
        )
    if df["volume"].map(
        lambda value: isinstance(value, (bool, np.bool_))
    ).any():
        raise RuntimeError("[PLUS5_VOLUME_INVALID] boolean tick count")
    numeric = {
        column: pd.to_numeric(df[column], errors="coerce").to_numpy(
            dtype=np.float64
        )
        for column in required
    }
    unavailable = {
        column: int(np.count_nonzero(~np.isfinite(values)))
        for column, values in numeric.items()
        if not np.isfinite(values).all()
    }
    if unavailable:
        raise RuntimeError(
            f"[PLUS5_SOURCE_NONFINITE] unavailable OHLCV values: {unavailable}"
        )
    invalid_ohlc = (
        (numeric["open"] <= 0.0)
        | (numeric["high"] <= 0.0)
        | (numeric["low"] <= 0.0)
        | (numeric["close"] <= 0.0)
        | (numeric["high"] < numeric["low"])
        | (numeric["high"] < numeric["open"])
        | (numeric["high"] < numeric["close"])
        | (numeric["low"] > numeric["open"])
        | (numeric["low"] > numeric["close"])
    )
    if invalid_ohlc.any():
        raise RuntimeError(
            "[PLUS5_OHLC_INVALID] prices must be positive with valid OHLC "
            f"geometry; count={int(np.count_nonzero(invalid_ohlc))}"
        )
    invalid_volume = (numeric["volume"] <= 0.0) | (
        numeric["volume"] != np.floor(numeric["volume"])
    )
    if invalid_volume.any():
        raise RuntimeError(
            "[PLUS5_VOLUME_INVALID] volume must be observed, strictly positive "
            "whole OANDA tick counts; "
            f"count={int(np.count_nonzero(invalid_volume))}"
        )

    out = df.copy()
    close = pd.Series(numeric["close"], index=out.index)
    high = pd.Series(numeric["high"], index=out.index)
    low = pd.Series(numeric["low"], index=out.index)
    # A 48-bar VWAP is undefined before all 48 price-volume observations
    # exist. Fixed-order window reductions make a 47-row overlap bit-exact.
    pv_48 = _rolling_sum_full(numeric["close"] * numeric["volume"], 48)
    volume_48 = _rolling_sum_full(numeric["volume"], 48)
    vwap_48 = pv_48 / volume_48
    out["_v1_vwap_drift48"] = (
        (numeric["close"] - vwap_48) / vwap_48
    ).astype(np.float32)

    out["atr"] = wilder_atr(high, low, close, 14).astype(np.float32)
    returns = np.full(len(close), np.nan, dtype=np.float64)
    returns[1:] = numeric["close"][1:] / numeric["close"][:-1] - 1.0
    _return_mean, return_std_population = _rolling_mean_std_full(returns, 50)
    out["std50"] = (
        return_std_population * np.sqrt(50.0 / 49.0)
    ).astype(np.float32)
    roc20 = np.full(len(close), np.nan, dtype=np.float64)
    roc20[20:] = numeric["close"][20:] / numeric["close"][:-20] - 1.0
    out["roc20"] = roc20.astype(np.float32)
    out.attrs["plus5_contract"] = {
        "schema_version": BASIC_V1_SCHEMA_VERSION,
        "features": list(PLUS5_FEATURES),
        "formula_sha256": BASIC_V1_FORMULA_SHA256,
        "first_finite_row": dict(PLUS5_FIRST_FINITE_ROW),
    }
    return out


def _require_observed_spread_input(df: pd.DataFrame) -> None:
    """Materialize the only causal transaction-cost input: observed spread."""
    if "spread_pct" in df.columns:
        spread_pct = pd.to_numeric(df["spread_pct"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        if not np.isfinite(spread_pct).all() or np.any(spread_pct < 0.0):
            raise RuntimeError(
                "[BASIC_V1_SPREAD_SOURCE_INVALID] spread_pct must be finite and "
                "non-negative on every row"
            )
    else:
        spread_pct = derive_observed_spread_bps(df) / 1e4
    df["spread_pct"] = spread_pct


def _validate_causal_feature_column(values: np.ndarray, *, name: str) -> np.ndarray:
    """Preserve a causal NaN prefix, but reject all other unavailable output."""
    array = np.asarray(values, dtype=np.float64)
    if np.isinf(array).any():
        raise RuntimeError(
            f"[BASIC_V1_FEATURE_NONFINITE] {name} contains infinite values"
        )
    finite = np.isfinite(array)
    if not finite.any():
        raise RuntimeError(
            f"[BASIC_V1_FEATURE_UNAVAILABLE] {name} has no finite observations"
        )
    first_finite = int(np.flatnonzero(finite)[0])
    invalid_tail = ~finite[first_finite:]
    if invalid_tail.any():
        raise RuntimeError(
            f"[BASIC_V1_FEATURE_NONFINITE_GAP] {name} has unavailable values "
            "after causal warmup"
        )
    return array


def _divide_where_positive(
    numerator: np.ndarray,
    denominator: np.ndarray,
) -> np.ndarray:
    """Divide only where both inputs exist and the denominator is positive."""

    top = np.asarray(numerator, dtype=np.float64)
    bottom = np.asarray(denominator, dtype=np.float64)
    if top.shape != bottom.shape or np.isinf(top).any() or np.isinf(bottom).any():
        raise RuntimeError("[BASIC_V1_RATIO_SOURCE_INVALID]")
    out = np.full(top.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(top) & np.isfinite(bottom) & (bottom > 0.0)
    out[valid] = top[valid] / bottom[valid]
    return out


@njit(cache=True, fastmath=False)
def _rolling_sum_full(values: np.ndarray, window: int) -> np.ndarray:
    """Fixed-order full-window sum, independent of preceding chunk history."""

    out = np.full(len(values), np.nan, dtype=np.float64)
    for row in range(window - 1, len(values)):
        total = 0.0
        for position in range(row - window + 1, row + 1):
            total += values[position]
        out[row] = total
    return out


@njit(cache=True, fastmath=False)
def _rolling_mean_std_full(
    values: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-order full-window mean and population std (ddof=0)."""

    means = np.full(len(values), np.nan, dtype=np.float64)
    standard_deviations = np.full(len(values), np.nan, dtype=np.float64)
    for row in range(window - 1, len(values)):
        start = row - window + 1
        total = 0.0
        finite = True
        for position in range(start, row + 1):
            value = values[position]
            if not np.isfinite(value):
                finite = False
                break
            total += value
        if not finite:
            continue
        mean = total / window
        squared_deviations = 0.0
        for position in range(start, row + 1):
            deviation = values[position] - mean
            squared_deviations += deviation * deviation
        means[row] = mean
        standard_deviations[row] = np.sqrt(squared_deviations / window)
    return means, standard_deviations


@dataclass(frozen=True)
class _EmaState:
    """Exact classic-EMA continuation state for immutable chunk boundaries."""

    span: int
    observations: int = 0
    seed_sum: float = 0.0
    value: float | None = None


def _ema_np_chunk(
    values: np.ndarray,
    span: int,
    *,
    state: _EmaState | None = None,
) -> tuple[np.ndarray, _EmaState]:
    """Classic EMA: SMA seed after ``span`` observations, then recursion."""

    if isinstance(span, bool) or not isinstance(span, int) or span <= 0:
        raise RuntimeError("[BASIC_V1_EMA_SPAN_INVALID]")
    current = state or _EmaState(span=span)
    if current.span != span:
        raise RuntimeError("[BASIC_V1_EMA_STATE_SPAN_MISMATCH]")
    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 1 or np.isinf(source).any():
        raise RuntimeError("[BASIC_V1_EMA_SOURCE_INVALID]")

    out = np.full(len(source), np.nan, dtype=np.float64)
    observations = current.observations
    seed_sum = current.seed_sum
    ema_value = current.value
    alpha = 2.0 / (span + 1.0)
    for position, value in enumerate(source):
        if np.isnan(value):
            if observations:
                raise RuntimeError("[BASIC_V1_EMA_NONFINITE_GAP]")
            continue
        observations += 1
        if observations < span:
            seed_sum += float(value)
            continue
        if observations == span:
            seed_sum += float(value)
            ema_value = seed_sum / span
        else:
            if ema_value is None:
                raise RuntimeError("[BASIC_V1_EMA_STATE_INVALID]")
            ema_value += alpha * (float(value) - ema_value)
        out[position] = ema_value
    return out, _EmaState(
        span=span,
        observations=observations,
        seed_sum=seed_sum,
        value=ema_value,
    )


def _ema(s, span):
    source = s.to_numpy(dtype=np.float64) if hasattr(s, "to_numpy") else s
    values, _state = _ema_np_chunk(source, span)
    if hasattr(s, "index"):
        return pd.Series(values, index=s.index, dtype=np.float64)
    return values

def _zscore(s, win):
    """
    Compute z-score using Numba-accelerated implementation.
    Only win==48 is supported (hard assert, no fallback).
    """
    assert win == 48, "Only w48 zscore supported"
    
    from gx1.utils.perf_timer import perf_add, perf_inc
    t_start = time.perf_counter()
    arr = s.to_numpy(dtype=np.float64)
    mean, standard_deviation = _rolling_mean_std_full(arr, 48)
    z = _divide_where_positive(arr - mean, standard_deviation)
    result = pd.Series(z, index=s.index, dtype=np.float64)
    t_end = time.perf_counter()
    
    perf_add(f"rolling.numpy.zscore.w{win}", t_end - t_start)
    perf_inc(f"rolling.numpy.zscore.w{win}")
    
    return result


@njit(cache=True, fastmath=False)
def _rolling_kurtosis_full48(values: np.ndarray) -> np.ndarray:
    """Unbiased Fisher kurtosis on exactly 48 observed returns.

    Central deviations are normalized by their window maximum before moments
    are formed.  Kurtosis is scale invariant, so this avoids the retired
    retired absolute variance threshold that turned genuine small gold
    returns into fabricated zero kurtosis.
    """

    window = 48
    out = np.full(len(values), np.nan, dtype=np.float64)
    for row in range(window - 1, len(values)):
        start = row - window + 1
        finite = True
        mean = 0.0
        for position in range(start, row + 1):
            value = values[position]
            if not np.isfinite(value):
                finite = False
                break
            mean += value
        if not finite:
            continue
        mean /= window
        maximum_deviation = 0.0
        for position in range(start, row + 1):
            deviation = abs(values[position] - mean)
            if deviation > maximum_deviation:
                maximum_deviation = deviation
        if maximum_deviation == 0.0:
            continue
        sum2 = 0.0
        sum4 = 0.0
        for position in range(start, row + 1):
            deviation = (values[position] - mean) / maximum_deviation
            squared = deviation * deviation
            sum2 += squared
            sum4 += squared * squared
        sample_variance = sum2 / (window - 1)
        factor1 = window * (window + 1) / (
            (window - 1) * (window - 2) * (window - 3)
        )
        factor2 = 3.0 * (window - 1) * (window - 1) / (
            (window - 2) * (window - 3)
        )
        out[row] = factor1 * sum4 / (sample_variance * sample_variance) - factor2
    return out

def _parkinson_sigma(high, low):
    """
    Parkinson volatility estimator using NumPy (no pandas rolling).
    DEL 2: Replaced .replace(), .rolling().mean(), .pow() with NumPy equivalents.
    """
    # Convert to NumPy arrays
    high_arr = high.to_numpy(dtype=np.float64) if hasattr(high, 'to_numpy') else np.asarray(high, dtype=np.float64)
    low_arr = low.to_numpy(dtype=np.float64) if hasattr(low, 'to_numpy') else np.asarray(low, dtype=np.float64)
    
    # Replace 0 with NaN: use mask
    low_safe = low_arr.copy()
    low_safe[low_safe == 0.0] = np.nan
    
    # sqrt(1/(4ln2)) * ln(high/low)
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.log(high_arr / low_safe)
    
    # x^2
    x_sq = x * x
    
    # Exactly 20 observations, reduced in fixed window order.
    x_sq_mean_arr, _unused_std = _rolling_mean_std_full(x_sq, 20)
    
    # sqrt: use np.sqrt instead of .pow(0.5)
    result_arr = np.sqrt(x_sq_mean_arr) / np.sqrt(4*np.log(2))
    
    # Return as Series if input was Series
    if hasattr(high, 'index'):
        return pd.Series(result_arr, index=high.index, dtype=np.float64)
    return result_arr

def add_session_features(
    df,
    *,
    decision_delay_seconds: int = M5_DECISION_DELAY_SECONDS,
):
    """
    Deriver session features fra timestamp (ASIA/EU/OVERLAP/US).
    Del 2: Uses DatetimeIndex directly, no pd.to_datetime in hot path.
    
    Expects either:
    - df.index is DatetimeIndex (preferred)
    - df["ts"] is datetime-like (fallback, converts once)
    """
    # Del 1+2: Use index if it's DatetimeIndex, otherwise use ts column (convert once)
    if isinstance(df.index, pd.DatetimeIndex):
        # Index is already DatetimeIndex - use directly (no conversion needed)
        idx = df.index
    elif "ts" in df.columns:
        # Fallback: convert ts column to DatetimeIndex once (not per bar)
        # This should only happen if caller didn't set index properly
        ts_col = df["ts"]
        
        # Check if already DatetimeIndex or datetime Series
        if isinstance(ts_col, pd.DatetimeIndex):
            idx = ts_col
        elif len(ts_col) > 0 and isinstance(ts_col.iloc[0], pd.Timestamp):
            # Already datetime Series - convert to DatetimeIndex
            # Convert to UTC if tz-naive, convert to UTC if tz-aware
            if ts_col.dt.tz is None:
                # Tz-naive, assume UTC and convert to DatetimeIndex
                idx = pd.DatetimeIndex(ts_col).tz_localize('UTC')
            else:
                # Tz-aware, convert to UTC and convert to DatetimeIndex
                idx = pd.DatetimeIndex(ts_col).tz_convert('UTC')
        else:
            # Not datetime - convert once (should be rare)
            idx = pd.to_datetime(ts_col, utc=True, errors="coerce")
            if not isinstance(idx, pd.DatetimeIndex):
                # Convert to DatetimeIndex if not already
                idx = pd.DatetimeIndex(idx)
    else:
        # No timestamp source: synthetic zero sessions are a forbidden
        # decision input, so this fails closed instead of degrading.
        raise RuntimeError(
            "[SESSION_TIME_SOURCE_MISSING] add_session_features requires a "
            "DatetimeIndex or 'ts' column"
        )
    
    if (
        isinstance(decision_delay_seconds, bool)
        or not isinstance(decision_delay_seconds, (int, np.integer))
        or int(decision_delay_seconds) <= 0
    ):
        raise RuntimeError("[SESSION_DECISION_DELAY_INVALID] positive integer required")
    idx = idx + pd.Timedelta(seconds=int(decision_delay_seconds))

    # Del 2: Session inference (SSoT)
    from gx1.time.session_detector import (
        get_session_vectorized,
        get_session_id_vectorized,
        get_session_minutes_since_open_vectorized,
        get_session_minutes_to_next_boundary_vectorized,
    )
    
    sessions = get_session_vectorized(idx)
    session_id = get_session_id_vectorized(idx)
    
    # Del 3: Create boolean masks directly (NumPy-friendly).  The detector
    # returns Series indexed by the decision instant (T+delay), while ``df``
    # remains labelled by the bar start (T).  These are row-wise features;
    # assign the values positionally or pandas will align on the shifted
    # labels and create a NaN prefix at every decision delay.
    is_asia_mask = np.asarray(sessions == "ASIA")
    is_eu_mask = np.asarray(sessions == "EU")
    is_overlap_mask = np.asarray(sessions == "OVERLAP")
    is_us_mask = np.asarray(sessions == "US")
    
    # Assign as integer (0/1) arrays directly
    df["is_ASIA"] = is_asia_mask.astype(int)
    df["is_EU"] = is_eu_mask.astype(int)
    df["is_OVERLAP"] = is_overlap_mask.astype(int)
    df["is_US"] = is_us_mask.astype(int)
    # Canonical session_id (observerable context)
    session_id_values = np.asarray(session_id)
    df["session_id"] = session_id_values
    
    # Session timing features (minutes)
    df["minutes_since_session_open"] = np.asarray(
        get_session_minutes_since_open_vectorized(idx),
        dtype=np.float32,
    )
    df["minutes_to_next_session_boundary"] = np.asarray(
        get_session_minutes_to_next_boundary_vectorized(idx),
        dtype=np.float32,
    )
    
    # Session change flag (1 if session changes vs previous bar)
    df["session_change_flag"] = (
        pd.Series(session_id_values, index=df.index).diff().fillna(0) != 0
    ).astype(int).to_numpy()
    
    return df

def build_basic_v1(
    df,
    *,
    decision_delay_seconds: int = M5_DECISION_DELAY_SECONDS,
):
    """
    Forventer kolonner: ts, open, high, low, close.
    Spread krever spread_pct, spread_bps, bid_close+ask_close eller spread+close.
    Realisert slippage er et execution/evalueringsutfall, ikke en kausal feature.
    Row t is a closed native bar made available at bar-end; every formula uses
    that current closed bar plus causal history, never a forming/future bar.
    Returnerer df med nye _v1_* features; originalkolonner beholdes.
    
    Input OHLC must be float32/float64. Every caller uses the same strict
    feature path; ambient environment flags cannot change formulas or fields.
    """
    from gx1.utils.perf_timer import perf_add
    t_start = time.perf_counter()

    df = df.copy()
    for c in ["open", "high", "low", "close"]:
        if c not in df:
            raise ValueError(f"Column '{c}' missing for basic_v1 features")
        if df[c].dtype not in [np.float32, np.float64]:
            raise ValueError(
                f"OHLC dtype mismatch: column '{c}' has dtype {df[c].dtype}; "
                "expected float32/float64"
            )
        df[c] = df[c].astype(np.float64)

    _require_observed_spread_input(df)
    plus5 = compute_plus5_features(df)

    # --- Volatilitet / Regime ---
    # Del: Time atr_family block
    t_atr_start = time.perf_counter()
    # Classic Wilder ATR has an indicator-defined SMA seed at row 13 and a
    # recursive RMA thereafter. Row t is already a closed, available bar, so
    # the shared owner's row-t value is the decision-time value (no lag copy).
    atr14 = wilder_atr(df["high"], df["low"], df["close"], 14)
    atr14_arr = atr14.to_numpy(dtype=np.float64) if hasattr(atr14, 'to_numpy') else np.asarray(atr14, dtype=np.float64)
    df["_v1_atr14"] = atr14_arr
    
    # Del: Time parkinson separately (may have pandas rolling)
    t_parkinson_start = time.perf_counter()
    pk_sigma = _parkinson_sigma(df["high"], df["low"])
    pk_sigma_arr = pk_sigma.to_numpy(dtype=np.float64) if hasattr(pk_sigma, 'to_numpy') else np.asarray(pk_sigma, dtype=np.float64)
    df["_v1_pk_sigma20"] = pk_sigma_arr
    t_parkinson_end = time.perf_counter()
    perf_add("feat.basic_v1.parkinson", t_parkinson_end - t_parkinson_start)
    
    t_atr_end = time.perf_counter()
    perf_add("feat.basic_v1.atr_family", t_atr_end - t_atr_start)

    # --- Trend / Mean-reversion ---
    # Del: Time trend_ema block
    t_trend_start = time.perf_counter()
    ema12 = _ema(df["close"], 12)
    ema26 = _ema(df["close"], 26)
    ema12_arr = ema12.to_numpy(dtype=np.float64) if hasattr(ema12, 'to_numpy') else np.asarray(ema12, dtype=np.float64)
    ema26_arr = ema26.to_numpy(dtype=np.float64) if hasattr(ema26, 'to_numpy') else np.asarray(ema26, dtype=np.float64)
    # ATR-relative trend spread. The raw USD spread (ema12 - ema26) scales
    # with the price level (gold ran 1454 -> 5588 over the declared tape) and
    # acted as an era/date proxy rather than market evidence. Dividing by the
    # file's own positive _v1_atr14 source (undefined zero-ATR rows remain
    # unavailable instead of being divided by an arbitrary epsilon)
    # makes it era-stable.
    # Both numerator and denominator use row t's closed bar. The ATR/EMA
    # warmup NaN prefix is preserved as honest causal warmup.
    ema_diff = _divide_where_positive(ema12_arr - ema26_arr, atr14_arr)
    df["_v1_ema_diff"] = ema_diff

    # Shared local-clock owner for the normalized volume-VWAP fraction.
    df["_v1_vwap_drift48"] = plus5["_v1_vwap_drift48"].to_numpy(
        dtype=np.float64
    )
    t_trend_end = time.perf_counter()
    perf_add("feat.basic_v1.trend_ema", t_trend_end - t_trend_start)

    # BB bandwidth delta 10: endring i squeeze (opløser stramt regime)
    # Del: Time BB block (already exists, keep it)
    t_bb_start = time.perf_counter()
    close_values = df["close"].to_numpy(dtype=np.float64)
    bb_mean10_arr, bb_std10_arr = _rolling_mean_std_full(close_values, 10)
    bb_upper10 = bb_mean10_arr + 2.0 * bb_std10_arr
    bb_lower10 = bb_mean10_arr - 2.0 * bb_std10_arr
    bb_width10 = _divide_where_positive(
        bb_upper10 - bb_lower10,
        bb_mean10_arr,
    )
    # 2026-08-13 noise-amplifier repair: ``np.diff(..., n=3)`` was the 3rd-order
    # finite difference (coefficients 1,-3,3,-1 — a noise amplifier, the exact
    # bug class repaired in htf_features._model_native_htf_slope_v4 on
    # 2026-08-09), not a bandwidth change.  The intended quantity keeps the
    # field's 3-bar timescale: the plain k-bar change x[t] - x[t-3] (k=3 is
    # the field's own pre-existing lookback, no new number).  Convention: no
    # further normalization — bb_width10 is already price-relative and the
    # raw change is routed directly to the learned model. The rolling-warmup NaN prefix is kept as
    # honest causal warmup (one contiguous prefix; the old code fabricated
    # live zeros inside warmup).
    bb_width10_delta3 = bb_width10 - np.concatenate(
        [np.full(3, np.nan), bb_width10[:-3]]
    )
    df["_v1_bb10_bandwidth_change_3"] = bb_width10_delta3
    t_bb_end = time.perf_counter()
    perf_add("feat.basic_v1.bb_family", t_bb_end - t_bb_start)

    # 3) Pre-trend booster: kort EMA-spread på close (3 vs 6)
    # 2026-08-19 volatility-coupling repair. The field was
    # (ema3 - ema6) / ema6 -- a PRICE fraction, hence its retired ``_frac``
    # name; the retired literal itself is not restated inside ``gx1/`` because
    # the stale-path scanner in tests/test_basic_v1_features.py owns it.
    # A price fraction of a trend displacement carries the volatility regime
    # inside its magnitude: over the declared TRAIN window the realised
    # volatility roughly doubles, and the measured IQR width of this field
    # doubled with it (last third / first third = 1.96 on 70,554 TRAIN rows,
    # 2026-08-19), while every ATR-normalized sibling in this file stayed flat
    # (_v1_ema_diff 0.93, _v1_tema20_change_3_atr 0.94,
    # _v1_kama30_change_5_atr 0.94). "45 bps above the slow EMA" cannot be
    # told apart from "a loud market"; "1.2 ATR" can.
    #
    # No magnitude is invented: the denominator is switched to the exact
    # ATR-multiple convention this file already uses three times above -
    # ``_divide_where_positive(numerator, atr14_arr)`` on the file's own single
    # ``wilder_atr(high, low, close, 14)`` source. That construct is bit-identical
    # to technical_indicators_v1.wilder_atr14_positive (same wilder_atr call,
    # same strictly-positive mask, NaN where ATR <= 0, never an epsilon floor);
    # calling it here would create a second ATR evaluation for one clock, so the
    # in-file owner is reused (rule 19).
    #
    # Rule 4 - no market evidence is removed. The retired denominator ema6 is
    # a smoothing of ``close``, which stays a model input in its own right, and
    # the numerator (ema3 - ema6) is preserved exactly; only the unit changes,
    # from price-fraction to ATR multiples. The price-relative view of the same
    # displacement family is still carried by _v1_vwap_drift48 and
    # _v1_bb10_bandwidth_change_3, and the volatility level itself by
    # _v1_atr14 / _v1_pk_sigma20 / _v1_range_z, so the model can still recover
    # (ema3 - ema6)/close as (ATR-multiple) x (_v1_atr14 / close) if that is
    # what it wants.
    #
    # Warmup: the numerator is finite from row 5 (ema6 SMA seed) and atr14 from
    # row 13 (Wilder SMA seed, BASIC_V1_FIRST_FINITE_ROW["_v1_atr14"]), so the
    # emitted first finite row is max(5, 13) = 13. Both derived, not chosen.
    ema3 = _ema(df["close"], 3)
    ema6 = _ema(df["close"], 6)
    ema3_arr = ema3.to_numpy(dtype=np.float64) if hasattr(ema3, 'to_numpy') else np.asarray(ema3, dtype=np.float64)
    ema6_arr = ema6.to_numpy(dtype=np.float64) if hasattr(ema6, 'to_numpy') else np.asarray(ema6, dtype=np.float64)
    ema_spread_atr = _divide_where_positive(ema3_arr - ema6_arr, atr14_arr)
    df["_v1_ema3_ema6_spread_atr"] = ema_spread_atr

    # Local range state.
    high_arr = df["high"].to_numpy(dtype=np.float64)
    low_arr = df["low"].to_numpy(dtype=np.float64)
    range_series = pd.Series(high_arr - low_arr, index=df.index)

    # --- Session ---
    # A frame without any exact time source cannot silently lose the canonical
    # session evidence consumed by the current context contract.
    if not isinstance(df.index, pd.DatetimeIndex) and "ts" not in df.columns:
        raise RuntimeError(
            "[BASIC_V1_SESSION_TIME_SOURCE_MISSING] build_basic_v1 requires a "
            "DatetimeIndex or exact 'ts' column for session features"
        )
    df = add_session_features(
        df,
        decision_delay_seconds=decision_delay_seconds,
    )
    # Range z-score
    range_z = _zscore(range_series, 48)
    range_z_arr = range_z.to_numpy(dtype=np.float64) if hasattr(range_z, 'to_numpy') else np.asarray(range_z, dtype=np.float64)
    df["_v1_range_z"] = range_z_arr
    
    # Kurtosis av returer (vol-form)
    t_misc_roll_start = time.perf_counter()
    from gx1.features.rolling_np import pct_change_np

    close_arr = df["close"].to_numpy(dtype=np.float64)
    ret1_array = pct_change_np(close_arr, k=1)
    kurt_result_np = _rolling_kurtosis_full48(ret1_array)
    df["_v1_kurt_r"] = kurt_result_np
    t_misc_roll_end = time.perf_counter()
    perf_add("feat.basic_v1.misc_roll", t_misc_roll_end - t_misc_roll_start)

    # TEMA slope 20 (trend-skarphet)
    # 2026-08-13 noise-amplifier repair: ``np.diff(..., n=3)`` was the
    # 3rd-order finite difference (a noise amplifier — the bug class repaired
    # in htf_features._model_native_htf_slope_v4 on 2026-08-09), not a slope.
    # Repaired to the k-bar change x[t] - x[t-3] (k=3 is the field's own
    # pre-existing lookback), normalized by this file's own current atr14
    # source wherever ATR is defined and positive — the exact ATR-multiple
    # convention `_v1_ema_diff` above. That keeps the
    # output O(1) in ATR units, the scale class for which the consumer's
    # tanh(scale=1.0) is dimensionally sane for this ATR-normalized field,
    # instead of a raw-USD magnitude that tracks the multi-year price level.
    # Numerator and denominator use row t's closed bar; the ATR/TEMA warmup
    # NaN prefix is kept as honest causal warmup instead of neutral zeros.
    tema20 = _tema(df["close"], 20)
    tema20_arr = tema20.to_numpy(dtype=np.float64) if hasattr(tema20, 'to_numpy') else np.asarray(tema20, dtype=np.float64)
    atr14_current_arr = atr14.to_numpy(dtype=np.float64)
    tema20_change3 = tema20_arr - np.concatenate(
        [np.full(3, np.nan), tema20_arr[:-3]]
    )
    tema20_slope = _divide_where_positive(tema20_change3, atr14_current_arr)
    df["_v1_tema20_change_3_atr"] = tema20_slope
    
    # Bollinger Band squeeze 20_2 (volatilitetssqueeze)
    bb_mean_arr, bb_std_arr = _rolling_mean_std_full(close_values, 20)
    bb_upper = bb_mean_arr + 2.0 * bb_std_arr
    bb_lower = bb_mean_arr - 2.0 * bb_std_arr
    bb_width = _divide_where_positive(bb_upper - bb_lower, bb_mean_arr)
    bb_width_mean_arr, _unused_std = _rolling_mean_std_full(bb_width, 100)
    bb_squeeze = _divide_where_positive(bb_width, bb_width_mean_arr) - 1.0
    # The old fill fabricated "exactly average bandwidth" throughout the
    # incomplete 20/100-bar prefix.  Unknown history stays NaN and is trimmed
    # by the bounded surface materializer.
    df["_v1_bb_squeeze_20_2"] = bb_squeeze
    
    # KAMA slope 30 (trend med støyfilter)
    # 2026-08-13 noise-amplifier repair: ``np.diff(..., n=5)`` was the
    # 5th-order finite difference (coefficients 1,-5,10,-10,5,-1 — the exact
    # noise amplifier repaired in htf_features._model_native_htf_slope_v4 on
    # 2026-08-09, measured there as ~15.9x noise), not a slope.  Repaired to
    # the k-bar change x[t] - x[t-5] (k=5 is the field's own pre-existing
    # lookback, the same k as the repaired HTF slope5 fields), normalized by
    # this file's own defined positive current atr14 source — the
    # `_v1_ema_diff` ATR-multiple convention. Same rationale as
    # `_v1_tema20_change_3_atr` directly above: O(1) ATR units keep the consumer's
    # tanh(scale=1.0) sane for this ATR-normalized field and remove the
    # era-proxy USD magnitude. Numerator/denominator use the current closed
    # bar; honest causal NaN warmup prefix is preserved.
    kama30 = _kama(df["close"], 30)
    kama30_arr = kama30.to_numpy(dtype=np.float64) if hasattr(kama30, 'to_numpy') else np.asarray(kama30, dtype=np.float64)
    kama30_change5 = kama30_arr - np.concatenate(
        [np.full(5, np.nan), kama30_arr[:-5]]
    )
    kama30_slope = _divide_where_positive(kama30_change5, atr14_current_arr)
    df["_v1_kama30_change_5_atr"] = kama30_slope
    
    # Preserve causal warmup. Never convert missing/non-finite model evidence
    # into a live zero signal.
    # Del: Time final_pack block (df operations, cleanup)
    t_final_start = time.perf_counter()
    newcols = list(BASIC_V1_FEATURES)
    missing = [name for name in newcols if name not in df.columns]
    extras = sorted(
        name
        for name in df.columns
        if name.startswith("_v1_") and name not in BASIC_V1_FEATURES
    )
    if missing or extras:
        raise RuntimeError(
            "[BASIC_V1_OUTPUT_CONTRACT_INVALID] "
            f"missing={missing} extras={extras}"
        )
    for col in newcols:
        col_arr = df[col].to_numpy(dtype=np.float64)
        df[col] = _validate_causal_feature_column(col_arr, name=col)
    df.attrs["basic_v1_contract"] = {
        "schema_version": BASIC_V1_SCHEMA_VERSION,
        "features": list(BASIC_V1_FEATURES),
        "features_sha256": BASIC_V1_FEATURES_SHA256,
        "formula_contract": list(BASIC_V1_FORMULA_CONTRACT),
        "formula_sha256": BASIC_V1_FORMULA_SHA256,
        "first_finite_row": dict(BASIC_V1_FIRST_FINITE_ROW),
    }
    t_final_end = time.perf_counter()
    perf_add("feat.basic_v1.final_pack", t_final_end - t_final_start)
    
    t_end = time.perf_counter()
    feature_build_time_ms = (t_end - t_start) * 1000.0
    perf_add("feat.basic_v1.total", t_end - t_start)
    perf_add("feat.basic_v1.total_ms", feature_build_time_ms)
    return df, newcols


@dataclass(frozen=True)
class _TemaState:
    """The three exact EMA continuation states that define one TEMA."""

    period: int
    ema1: _EmaState
    ema2: _EmaState
    ema3: _EmaState


def _tema_np_chunk(
    values: np.ndarray,
    period: int,
    *,
    state: _TemaState | None = None,
) -> tuple[np.ndarray, _TemaState]:
    if state is not None and state.period != period:
        raise RuntimeError("[BASIC_V1_TEMA_STATE_PERIOD_MISMATCH]")
    ema1, state1 = _ema_np_chunk(
        values,
        period,
        state=None if state is None else state.ema1,
    )
    ema2, state2 = _ema_np_chunk(
        ema1,
        period,
        state=None if state is None else state.ema2,
    )
    ema3, state3 = _ema_np_chunk(
        ema2,
        period,
        state=None if state is None else state.ema3,
    )
    return 3.0 * ema1 - 3.0 * ema2 + ema3, _TemaState(
        period=period,
        ema1=state1,
        ema2=state2,
        ema3=state3,
    )


def _tema(series, period):
    """Classic triple EMA, with an SMA seed for each recursive layer."""

    source = (
        series.to_numpy(dtype=np.float64)
        if hasattr(series, "to_numpy")
        else np.asarray(series, dtype=np.float64)
    )
    values, _state = _tema_np_chunk(source, period)
    if hasattr(series, "index"):
        return pd.Series(values, index=series.index, dtype=np.float64)
    return values


@dataclass(frozen=True)
class _KamaState:
    """Exact KAMA continuation state, including its period-price ring."""

    period: int
    fast: int
    slow: int
    observations: int = 0
    history: tuple[float, ...] = ()
    value: float | None = None


def _kama_np_chunk(
    prices: np.ndarray,
    period: int,
    fast: int = 2,
    slow: int = 30,
    *,
    state: _KamaState | None = None,
) -> tuple[np.ndarray, _KamaState]:
    """TA-Lib-compatible KAMA initialization and exact chunk continuation.

    The first ``period`` prices establish the period history.  The price at
    ``period-1`` is the previous KAMA; the first emitted value at ``period``
    is one adaptive update using the first complete efficiency-ratio window.
    Unknown prefix rows remain NaN.  No partial-window EMA and no smoothing-
    constant clamp are part of the indicator definition.
    """

    parameters = (period, fast, slow)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in parameters):
        raise RuntimeError("[BASIC_V1_KAMA_PARAMETER_INVALID]")
    source = np.asarray(prices, dtype=np.float64)
    if source.ndim != 1 or not np.isfinite(source).all():
        raise RuntimeError("[BASIC_V1_KAMA_SOURCE_INVALID]")
    current = state or _KamaState(period=period, fast=fast, slow=slow)
    if (current.period, current.fast, current.slow) != parameters:
        raise RuntimeError("[BASIC_V1_KAMA_STATE_PARAMETER_MISMATCH]")

    out = np.full(len(source), np.nan, dtype=np.float64)
    observations = current.observations
    history = list(current.history)
    kama_value = current.value
    if period == 1:
        for position, price in enumerate(source):
            kama_value = float(price)
            observations += 1
            history = [kama_value]
            out[position] = kama_value
        return out, _KamaState(
            period=period,
            fast=fast,
            slow=slow,
            observations=observations,
            history=tuple(history),
            value=kama_value,
        )

    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    sc_diff = fast_sc - slow_sc
    for position, price_value in enumerate(source):
        price = float(price_value)
        if observations < period:
            history.append(price)
            observations += 1
            if observations == period:
                kama_value = price
            continue
        if len(history) != period or kama_value is None:
            raise RuntimeError("[BASIC_V1_KAMA_STATE_INVALID]")
        extended = np.asarray((*history, price), dtype=np.float64)
        change = abs(price - history[0])
        volatility = float(np.sum(np.abs(np.diff(extended)), dtype=np.float64))
        efficiency_ratio = 1.0 if volatility == 0.0 else change / volatility
        smoothing_constant = (
            efficiency_ratio * sc_diff + slow_sc
        ) ** 2
        kama_value += smoothing_constant * (price - kama_value)
        out[position] = kama_value
        history = history[1:] + [price]
        observations += 1
    return out, _KamaState(
        period=period,
        fast=fast,
        slow=slow,
        observations=observations,
        history=tuple(history),
        value=kama_value,
    )


def kama_np(
    prices: np.ndarray,
    period: int,
    fast: int = 2,
    slow: int = 30,
) -> np.ndarray:
    values, _state = _kama_np_chunk(prices, period, fast, slow)
    return values

def _kama(series, period):
    """
    Del 1: KAMA wrapper that uses incremental NumPy implementation.
    
    Replaces pandas rolling-based implementation with streaming NumPy version.
    """
    # Convert to numpy array
    if isinstance(series, pd.Series):
        prices = series.values
        index = series.index
    else:
        prices = np.asarray(series)
        index = None
    
    # Use exact incremental NumPy implementation.
    kama_values = kama_np(prices, period)
    
    # Return as Series if input was Series, otherwise array
    if index is not None:
        return pd.Series(kama_values, index=index, dtype=np.float64)
    else:
        return kama_values
