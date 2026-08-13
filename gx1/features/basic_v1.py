# gx1/features/basic_v1.py
# -*- coding: utf-8 -*-
# Runtime hot-path: pandas forbidden
# All per-bar operations must use NumPy arrays directly.
# Pandas operations (.clip(), .reindex(), .rolling(), .fillna(), .shift(), .replace(), etc.) are removed from hot-path.
# Input must be float32/float64 - hard fail in replay mode on wrong dtype.
import numpy as np
import pandas as pd
import time

from gx1.features.model_native_market_context_v1 import derive_observed_spread_bps
from gx1.time.session_detector import ASIA_SESSION_ID

M5_DECISION_DELAY_SECONDS = 5 * 60
M1_DECISION_DELAY_SECONDS = 1 * 60
LOCAL_DECISION_DELAYS_SECONDS = (
    M1_DECISION_DELAY_SECONDS,
    M5_DECISION_DELAY_SECONDS,
)

BASIC_V1_OBSERVED_SPREAD_FEATURES = (
    "_v1_spread_p",
    "_v1_spread_z",
)
PLUS5_FEATURES = (
    "atr",
    "std50",
    "roc20",
    "_v1_vwap_drift48",
)


def compute_plus5_features(df: pd.DataFrame) -> pd.DataFrame:
    """One train/serve owner for the retained local-clock auxiliary fields."""
    required = ("open", "high", "low", "close", "volume")
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise RuntimeError(
            f"[PLUS5_SOURCE_MISSING] required OHLCV columns missing: {missing}"
        )
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
    invalid_volume = numeric["volume"] <= 0.0
    if invalid_volume.any():
        raise RuntimeError(
            "[PLUS5_VOLUME_INVALID] volume must be observed and strictly positive; "
            f"count={int(np.count_nonzero(invalid_volume))}"
        )

    out = df.copy()
    close = pd.Series(numeric["close"], index=out.index)
    high = pd.Series(numeric["high"], index=out.index)
    low = pd.Series(numeric["low"], index=out.index)
    volume = pd.Series(numeric["volume"], index=out.index)
    pv = close * volume

    pv_48 = pv.rolling(48, min_periods=1).sum()
    volume_48 = volume.rolling(48, min_periods=1).sum()
    vwap_48 = pv_48 / volume_48
    out["_v1_vwap_drift48"] = ((close - vwap_48) / vwap_48).astype(
        np.float32
    )

    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr"] = true_range.ewm(alpha=1 / 14, adjust=False).mean().astype(
        np.float32
    )
    returns = close.pct_change(fill_method=None)
    out["std50"] = returns.rolling(50, min_periods=2).std().astype(np.float32)
    out["roc20"] = close.pct_change(20, fill_method=None).astype(np.float32)
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


def _local_bars_for_days(days: int, *, decision_delay_seconds: int) -> int:
    """Convert an explicitly day-based feature horizon to native M1/M5 rows."""

    if (
        isinstance(days, bool)
        or not isinstance(days, (int, np.integer))
        or int(days) <= 0
        or isinstance(decision_delay_seconds, bool)
        or not isinstance(decision_delay_seconds, (int, np.integer))
        or int(decision_delay_seconds) not in LOCAL_DECISION_DELAYS_SECONDS
        or 86_400 % int(decision_delay_seconds) != 0
    ):
        raise RuntimeError("[BASIC_V1_PHYSICAL_HORIZON_INVALID]")
    return int(days) * (86_400 // int(decision_delay_seconds))

def _assert_valid_datetime_index(df: pd.DataFrame, ctx: str = "unknown") -> pd.Timestamp:
    """
    Assert that df has a valid DatetimeIndex and return current timestamp.
    
    PHASE 1 FIX: Removes pd.Timestamp.now() fallback to ensure determinism.
    
    Args:
        df: DataFrame to validate
        ctx: Context string for error messages (e.g., "H1 RSI z-score")
    
    Returns:
        pd.Timestamp: Current timestamp from df.index[-1]
    
    Raises:
        RuntimeError: If the deterministic timestamp contract is invalid.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        error_msg = (
            f"[DETERMINISM_BLOCKER] DataFrame index is not DatetimeIndex in context '{ctx}'. "
            f"Index type: {type(df.index)}, df.shape: {df.shape}. "
            f"This breaks determinism (cannot use pd.Timestamp.now() fallback)."
        )
        raise RuntimeError(error_msg)
    
    if len(df.index) == 0:
        error_msg = (
            f"[DETERMINISM_BLOCKER] DataFrame index is empty in context '{ctx}'. "
            f"df.shape: {df.shape}. "
            f"This breaks determinism (cannot use pd.Timestamp.now() fallback)."
        )
        raise RuntimeError(error_msg)
    
    return df.index[-1]

def _roll(s, win, fn, minp=None):
    """
    Helper for NumPy rolling operations (NO PANDAS).
    DEL 2: All rolling operations use NumPy directly - no pandas fallback.
    
    Args:
        s: pd.Series or np.ndarray (will convert to array)
        win: Window size
        fn: Function name ("mean" or "std")
        minp: Minimum periods (default: win)
    
    Returns:
        pd.Series with NumPy-computed rolling values
    """
    # DEL 2: Always use NumPy - no pandas fallback
    from gx1.features.rolling_np import rolling_mean, rolling_std
    from gx1.utils.perf_timer import perf_add, perf_inc
    
    # Convert to NumPy array
    if hasattr(s, 'to_numpy'):
        arr = s.to_numpy(dtype=np.float64)
        index = s.index
    else:
        arr = np.asarray(s, dtype=np.float64)
        index = None
    
    if minp is None:
        minp = max(1, int(win * 0.8))
    
    # Time NumPy rolling
    t_rolling_start = time.perf_counter()
    try:
        if fn == "mean":
            result_arr = rolling_mean(arr, win, min_periods=minp)
        elif fn == "std":
            result_arr = rolling_std(arr, win, min_periods=minp, ddof=0)
        else:
            raise ValueError(f"Unsupported rolling function: {fn} (only 'mean' and 'std' supported)")
    finally:
        t_rolling_end = time.perf_counter()
        perf_add(f"rolling.numpy.{fn}.w{win}", t_rolling_end - t_rolling_start)
        perf_inc(f"rolling.numpy.{fn}.w{win}")
    
    # Return as Series if input was Series
    if index is not None:
        return pd.Series(result_arr, index=index, dtype=np.float64)
    return result_arr

def _ema(s, span):
    return s.ewm(span=span, adjust=False, min_periods=max(1, span//2)).mean()

def _zscore(s, win):
    """
    Compute z-score using Numba-accelerated implementation.
    Only win==48 is supported (hard assert, no fallback).
    """
    assert win == 48, "Only w48 zscore supported"
    
    from gx1.utils.perf_timer import perf_add, perf_inc
    from gx1.features.rolling_np import zscore_w48
    
    t_start = time.perf_counter()
    arr = s.to_numpy(dtype=np.float64)
    z = zscore_w48(arr, min_periods=24)
    result = pd.Series(z, index=s.index, dtype=np.float64)
    t_end = time.perf_counter()
    
    perf_add(f"rolling.numpy.zscore.w{win}", t_end - t_start)
    perf_inc(f"rolling.numpy.zscore.w{win}")
    
    return result

def _true_range(high, low, close):
    """
    True Range calculation using NumPy (no pandas).
    DEL 2: Replaced .shift(), .concat(), .max() with NumPy equivalents.
    """
    # Convert to NumPy arrays
    high_arr = high.to_numpy(dtype=np.float64) if hasattr(high, 'to_numpy') else np.asarray(high, dtype=np.float64)
    low_arr = low.to_numpy(dtype=np.float64) if hasattr(low, 'to_numpy') else np.asarray(low, dtype=np.float64)
    close_arr = close.to_numpy(dtype=np.float64) if hasattr(close, 'to_numpy') else np.asarray(close, dtype=np.float64)
    
    # Shift(1): prev_close = close shifted by 1 (first element is NaN/0)
    prev_close = np.roll(close_arr, 1)
    prev_close[0] = close_arr[0]  # First element uses current close (no previous)
    
    # Compute three components
    tr1 = high_arr - low_arr
    tr2 = np.abs(high_arr - prev_close)
    tr3 = np.abs(low_arr - prev_close)
    
    # Max of three components
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Return as Series if input was Series
    if hasattr(close, 'index'):
        return pd.Series(tr, index=close.index, dtype=np.float64)
    return tr

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
    
    # Rolling mean w20 uses the timed rolling owner.
    from gx1.features.rolling_timer import timed_rolling
    x_sq_series = pd.Series(x_sq, index=high.index if hasattr(high, 'index') else range(len(x_sq)))
    x_sq_mean = timed_rolling(x_sq_series, 20, "mean", min_periods=10)
    x_sq_mean_arr = x_sq_mean.to_numpy(dtype=np.float64) if hasattr(x_sq_mean, 'to_numpy') else np.asarray(x_sq_mean, dtype=np.float64)
    
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
    # Historical model-field names are emitted by the same session owner at
    # the same decision instant.  There is no second shift or second formula.
    df["_v1_is_EU"] = is_eu_mask.astype(np.float64)
    df["_v1_is_US"] = is_us_mask.astype(np.float64)
    
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
    
    # Tradable flag (policy can still restrict to EU/OVERLAP/US)
    df["session_tradable"] = (
        session_id_values != ASIA_SESSION_ID
    ).astype(int)
    
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
    Bruker kun fortid: shift(1) og ruller bakover.
    Returnerer df med nye _v1_* features; originalkolonner beholdes.
    
    Input OHLC must be float32/float64. Every caller uses the same strict
    feature path; ambient environment flags cannot change formulas or fields.
    """
    from gx1.utils.perf_timer import perf_add
    t_start = time.perf_counter()

    bars_per_day = _local_bars_for_days(
        1,
        decision_delay_seconds=decision_delay_seconds,
    )

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

    # --- Momentum (laggede avkastninger) ---
    # Del: Time returns_and_vol block
    t_returns_start = time.perf_counter()
    
    # DEL 2: Always use NumPy path (no pandas fallback)
    from gx1.features.rolling_np import pct_change_np
    import logging
    log = logging.getLogger(__name__)
    
    if not hasattr(build_basic_v1, '_logged_np_pct_change'):
        log.info("[FEATURES] Using NumPy pct_change_np for percentage change operations (pandas removed)")
        build_basic_v1._logged_np_pct_change = True
    
    # Use NumPy pct_change for ret1 (k=1)
    close_array = df["close"].to_numpy(dtype=np.float64)
    ret1_array = pct_change_np(close_array, k=1)
    ret1 = pd.Series(ret1_array, index=df.index, dtype=np.float64)
    
    # Use NumPy pct_change for k in [3,5,8,12,24]
    for k in [3,5,8,12,24]:
        rk_array = pct_change_np(close_array, k=k)
        # Shift(1): unavailable history remains a causal NaN prefix.
        rk_shifted = np.roll(rk_array, 1)
        rk_shifted[0] = np.nan
        df[f"_v1_r{k}"] = rk_shifted
    
    # Shift(1) for ret1
    ret1_shifted = np.roll(ret1_array, 1)
    ret1_shifted[0] = np.nan
    df["_v1_r1"] = ret1_shifted
    
    # Z-score and shift(1)
    ret1_z = _zscore(ret1, 48)
    ret1_z_arr = ret1_z.to_numpy(dtype=np.float64) if hasattr(ret1_z, 'to_numpy') else np.asarray(ret1_z, dtype=np.float64)
    ret1_z_shifted = np.roll(ret1_z_arr, 1)
    ret1_z_shifted[0] = np.nan
    df["_v1_r48_z"] = ret1_z_shifted
    t_returns_end = time.perf_counter()
    perf_add("feat.basic_v1.returns_and_vol", t_returns_end - t_returns_start)

    # --- Volatilitet / Regime ---
    # Del: Time atr_family block
    t_atr_start = time.perf_counter()
    tr = _true_range(df["high"], df["low"], df["close"])
    atr14 = _roll(tr, 14, "mean")
    # DEL 2: Shift(1) using NumPy
    atr14_arr = atr14.to_numpy(dtype=np.float64) if hasattr(atr14, 'to_numpy') else np.asarray(atr14, dtype=np.float64)
    atr14_shifted = np.roll(atr14_arr, 1)
    atr14_shifted[0] = atr14_arr[0]  # First element uses current value
    df["_v1_atr14"] = atr14_shifted
    
    # Del: Time parkinson separately (may have pandas rolling)
    t_parkinson_start = time.perf_counter()
    pk_sigma = _parkinson_sigma(df["high"], df["low"])
    pk_sigma_arr = pk_sigma.to_numpy(dtype=np.float64) if hasattr(pk_sigma, 'to_numpy') else np.asarray(pk_sigma, dtype=np.float64)
    pk_sigma_shifted = np.roll(pk_sigma_arr, 1)
    pk_sigma_shifted[0] = pk_sigma_arr[0] if len(pk_sigma_arr) > 0 else 0.0
    df["_v1_pk_sigma20"] = pk_sigma_shifted
    t_parkinson_end = time.perf_counter()
    perf_add("feat.basic_v1.parkinson", t_parkinson_end - t_parkinson_start)
    
    # ATR-regime-ID (kvantiler på rullende 20 dager) - CAUSAL VERSION
    # FIX: Replace global quantile ranking with rolling quantiles to prevent future leakage
    # Window: 20 days = 288 bars/day * 20 = 5760 bars for M5
    # Del: Time zscore_family (includes _zscore calls)
    t_zscore_start = time.perf_counter()
    atr14_arr_for_regime = atr14_arr.copy()
    
    # Use rolling quantiles (causal) instead of global ranking (non-causal).
    # This field is explicitly day-based, so M1 and M5 retain the same
    # physical 20/10-day semantics while remaining independently computed.
    from gx1.features.rolling_np import rolling_quantile
    from gx1.utils.perf_timer import perf_add, perf_inc
    import time as time_module
    
    # Compute rolling quantiles: q33 and q67 (for 3 bins: LOW, MEDIUM, HIGH)
    regime_window = 20 * bars_per_day
    regime_min_periods = 10 * bars_per_day
    
    t_q33_start = time_module.perf_counter()
    q33_arr = rolling_quantile(atr14_arr_for_regime, regime_window, q=0.333, min_periods=regime_min_periods)
    t_q33_end = time_module.perf_counter()
    perf_add("feat.basic_v1.atr_regime.q33", t_q33_end - t_q33_start)
    
    t_q67_start = time_module.perf_counter()
    q67_arr = rolling_quantile(atr14_arr_for_regime, regime_window, q=0.667, min_periods=regime_min_periods)
    t_q67_end = time_module.perf_counter()
    perf_add("feat.basic_v1.atr_regime.q67", t_q67_end - t_q67_start)
    
    # Classify into 3 bins based on rolling quantiles (causal)
    # LOW: atr14 < q33, MEDIUM: q33 <= atr14 < q67, HIGH: atr14 >= q67
    regime_id = np.zeros(len(atr14_arr_for_regime), dtype=np.float64)
    regime_id[:] = 1.0  # Default to MEDIUM
    
    # Only classify where we have enough data (q33 and q67 are not NaN)
    valid_mask = ~(np.isnan(q33_arr) | np.isnan(q67_arr))
    if valid_mask.sum() > 0:
        atr_valid = atr14_arr_for_regime[valid_mask]
        q33_valid = q33_arr[valid_mask]
        q67_valid = q67_arr[valid_mask]
        
        # LOW regime: atr14 < q33 ; HIGH regime: atr14 >= q67 ; MEDIUM: q33<=atr14<q67 (already 1.0)
        low_mask = atr_valid < q33_valid
        high_mask = atr_valid >= q67_valid
        # Write through integer indices. Chained boolean indexing writes a
        # temporary copy and previously made this field constant MEDIUM.
        valid_idx = np.flatnonzero(valid_mask)
        regime_id[valid_idx[low_mask]] = 0.0
        regime_id[valid_idx[high_mask]] = 2.0
    
    # Fill NaN with 1.0 (middle regime)
    regime_id = np.nan_to_num(regime_id, nan=1.0)
    # Shift(1)
    regime_id_shifted = np.roll(regime_id, 1)
    regime_id_shifted[0] = 1.0
    df["_v1_atr_regime_id"] = regime_id_shifted
    t_zscore_end = time.perf_counter()
    perf_add("feat.basic_v1.zscore_family", t_zscore_end - t_zscore_start)
    t_atr_end = time.perf_counter()
    perf_add("feat.basic_v1.atr_family", t_atr_end - t_atr_start)

    # --- Trend / Mean-reversion ---
    # Del: Time trend_ema block
    t_trend_start = time.perf_counter()
    ema12 = _ema(df["close"], 12)
    ema26 = _ema(df["close"], 26)
    # DEL 2: Replace .shift() with NumPy
    ema12_arr = ema12.to_numpy(dtype=np.float64) if hasattr(ema12, 'to_numpy') else np.asarray(ema12, dtype=np.float64)
    ema26_arr = ema26.to_numpy(dtype=np.float64) if hasattr(ema26, 'to_numpy') else np.asarray(ema26, dtype=np.float64)
    # ATR-relative trend spread. The raw USD spread (ema12 - ema26) scales
    # with the price level (gold ran 1454 -> 5588 over the declared tape) and
    # acted as an era/date proxy rather than market evidence. Dividing by the
    # file's own _v1_atr14 source (additive 1e-12 epsilon, the same ATR
    # division convention as _v1_tr_1_over_atr_14 below) makes it era-stable.
    # Both numerator and denominator are unshifted here and shifted once
    # together, so row t reads bar t-1's ratio; the ATR rolling warmup NaN
    # prefix is preserved as honest causal warmup.
    ema_diff = (ema12_arr - ema26_arr) / (atr14_arr + 1e-12)
    ema_diff_shifted = np.roll(ema_diff, 1)
    ema_diff_shifted[0] = ema_diff[0] if len(ema_diff) > 0 else 0.0
    df["_v1_ema_diff"] = ema_diff_shifted

    # Shared local-clock owner for the normalized volume-VWAP fraction.
    df["_v1_vwap_drift48"] = plus5["_v1_vwap_drift48"].to_numpy(
        dtype=np.float64
    )
    t_trend_end = time.perf_counter()
    perf_add("feat.basic_v1.trend_ema", t_trend_end - t_trend_start)

    # RSI14 z-score (klassisk)
    # Del: Time rsi_family block
    t_rsi_start = time.perf_counter()
    # DEL 2: Replace .diff(), .clip(), .rolling() with NumPy
    close_arr = df["close"].to_numpy(dtype=np.float64)
    delta = np.diff(close_arr, prepend=close_arr[0])
    
    # Wilder RSI: avg_gain = mean(positive deltas), avg_loss = mean(|negative deltas|)
    # Both must be POSITIVE for rs = up / dn to behave (rs in [0, +inf), rsi in [0, 100]).
    up = np.clip(delta, 0, np.inf)
    dn_neg = np.clip(-delta, 0, np.inf)
    dn = dn_neg  # FIX 2026-05-03: was `dn = -dn_neg`, which made rs negative and could
    # produce rsi values up to 1e17 when (1 + rs) ≈ 0. Now dn is positive absolute loss
    # per Wilder's classic formula. Affects: _v1_rsi14_z, _v1_rsi14, _v1_rsi2, _v1_rsi2_gt_rsi14.
    
    # Rolling mean w14: use timed_rolling (will use pandas fallback but timed)
    from gx1.features.rolling_timer import timed_rolling
    up_series = pd.Series(up, index=df.index)
    dn_series = pd.Series(dn, index=df.index)
    up_roll = timed_rolling(up_series, 14, "mean", min_periods=7)
    dn_roll = timed_rolling(dn_series, 14, "mean", min_periods=7)
    up_arr = up_roll.to_numpy(dtype=np.float64) if hasattr(up_roll, 'to_numpy') else np.asarray(up_roll, dtype=np.float64)
    dn_arr = dn_roll.to_numpy(dtype=np.float64) if hasattr(dn_roll, 'to_numpy') else np.asarray(dn_roll, dtype=np.float64)
    
    rs = up_arr / (dn_arr + 1e-12)
    rsi = 100 - 100/(1 + rs)
    rsi_series = pd.Series(rsi, index=df.index)
    rsi_z = _zscore(rsi_series, 48)
    rsi_z_arr = rsi_z.to_numpy(dtype=np.float64) if hasattr(rsi_z, 'to_numpy') else np.asarray(rsi_z, dtype=np.float64)
    rsi_z_shifted = np.roll(rsi_z_arr, 1)
    rsi_z_shifted[0] = np.nan
    df["_v1_rsi14_z"] = rsi_z_shifted
    
    # RSI2 and RSI14 (raw, for mini-featurepack)
    up2_roll = timed_rolling(up_series, 2, "mean", min_periods=1)
    dn2_roll = timed_rolling(dn_series, 2, "mean", min_periods=1)
    up2_arr = up2_roll.to_numpy(dtype=np.float64) if hasattr(up2_roll, 'to_numpy') else np.asarray(up2_roll, dtype=np.float64)
    dn2_arr = dn2_roll.to_numpy(dtype=np.float64) if hasattr(dn2_roll, 'to_numpy') else np.asarray(dn2_roll, dtype=np.float64)
    rs2 = up2_arr / (dn2_arr + 1e-12)
    rsi2 = 100 - 100/(1 + rs2)
    
    # Shift(1) and fillna
    rsi2_shifted = np.roll(rsi2, 1)
    rsi2_shifted[0] = 50.0
    rsi2_shifted = np.nan_to_num(rsi2_shifted, nan=50.0)
    df["_v1_rsi2"] = rsi2_shifted
    
    rsi_shifted = np.roll(rsi, 1)
    rsi_shifted[0] = 50.0
    rsi_shifted = np.nan_to_num(rsi_shifted, nan=50.0)
    df["_v1_rsi14"] = rsi_shifted
    
    # rsi2 > rsi comparison
    rsi2_gt_rsi = (rsi2 > rsi).astype(np.float64)
    rsi2_gt_rsi_shifted = np.roll(rsi2_gt_rsi, 1)
    rsi2_gt_rsi_shifted[0] = 0.0
    rsi2_gt_rsi_shifted = np.nan_to_num(rsi2_gt_rsi_shifted, nan=0.0)
    df["_v1_rsi2_gt_rsi14"] = rsi2_gt_rsi_shifted
    t_rsi_end = time.perf_counter()
    perf_add("feat.basic_v1.rsi_family", t_rsi_end - t_rsi_start)

    # --- Raskere features (tempo) ---
    # 1) micro-momentum: diff mellom EMA av returns (2 vs 5)
    # DEL 2: Replace .pct_change(), .fillna(), .shift() with NumPy
    close_arr = df["close"].to_numpy(dtype=np.float64)
    ret_array = pct_change_np(close_arr, k=1)
    ret_array = np.nan_to_num(ret_array, nan=0.0)
    ret = pd.Series(ret_array, index=df.index)
    ema_ret2 = _ema(ret, 2)
    ema_ret5 = _ema(ret, 5)
    ema_ret2_arr = ema_ret2.to_numpy(dtype=np.float64) if hasattr(ema_ret2, 'to_numpy') else np.asarray(ema_ret2, dtype=np.float64)
    ema_ret5_arr = ema_ret5.to_numpy(dtype=np.float64) if hasattr(ema_ret5, 'to_numpy') else np.asarray(ema_ret5, dtype=np.float64)
    ret_ema_diff = ema_ret2_arr - ema_ret5_arr
    ret_ema_diff_shifted = np.roll(ret_ema_diff, 1)
    ret_ema_diff_shifted[0] = 0.0
    ret_ema_diff_shifted = np.nan_to_num(ret_ema_diff_shifted, nan=0.0)
    df["_v1_ret_ema_diff_2_5"] = ret_ema_diff_shifted
    
    # 2) BB bandwidth delta 10: endring i squeeze (opløser stramt regime)
    # Del: Time BB block (already exists, keep it)
    t_bb_start = time.perf_counter()
    from gx1.features.rolling_timer import timed_rolling
    bb_mean10 = timed_rolling(df["close"], 10, "mean", min_periods=5)
    bb_std10 = timed_rolling(df["close"], 10, "std", min_periods=5, ddof=0)
    bb_mean10_arr = bb_mean10.to_numpy(dtype=np.float64) if hasattr(bb_mean10, 'to_numpy') else np.asarray(bb_mean10, dtype=np.float64)
    bb_std10_arr = bb_std10.to_numpy(dtype=np.float64) if hasattr(bb_std10, 'to_numpy') else np.asarray(bb_std10, dtype=np.float64)
    bb_upper10 = bb_mean10_arr + 2.0 * bb_std10_arr
    bb_lower10 = bb_mean10_arr - 2.0 * bb_std10_arr
    bb_width10 = (bb_upper10 - bb_lower10) / (bb_mean10_arr + 1e-12)
    # 2026-08-13 noise-amplifier repair: ``np.diff(..., n=3)`` was the 3rd-order
    # finite difference (coefficients 1,-3,3,-1 — a noise amplifier, the exact
    # bug class repaired in htf_features._model_native_htf_slope_v4 on
    # 2026-08-09), not a bandwidth change.  The intended quantity keeps the
    # field's 3-bar timescale: the plain k-bar change x[t] - x[t-3] (k=3 is
    # the field's own pre-existing lookback, no new number).  Convention: no
    # further normalization — bb_width10 is already the price-relative
    # (dimensionless) bandwidth, so its plain change stays in the same output
    # scale class the consumer's tanh(scale=1.0) expects
    # (entry_vol_compression_v1).  The rolling-warmup NaN prefix is kept as
    # honest causal warmup (one contiguous prefix; the nan_to_num masking of
    # the old code fabricated live zeros inside warmup).
    bb_width10_delta3 = bb_width10 - np.concatenate(
        [np.full(3, np.nan), bb_width10[:-3]]
    )
    bb_width10_shifted = np.roll(bb_width10_delta3, 1)
    bb_width10_shifted[0] = np.nan
    df["_v1_bb_bandwidth_delta_10"] = bb_width10_shifted
    t_bb_end = time.perf_counter()
    perf_add("feat.basic_v1.bb_family", t_bb_end - t_bb_start)

    # 3) Pre-trend booster: kort EMA-slope på close (3 vs 6)
    ema3 = _ema(df["close"], 3)
    ema6 = _ema(df["close"], 6)
    ema3_arr = ema3.to_numpy(dtype=np.float64) if hasattr(ema3, 'to_numpy') else np.asarray(ema3, dtype=np.float64)
    ema6_arr = ema6.to_numpy(dtype=np.float64) if hasattr(ema6, 'to_numpy') else np.asarray(ema6, dtype=np.float64)
    ema_slope = (ema3_arr - ema6_arr) / (ema6_arr + 1e-12)
    ema_slope_shifted = np.roll(ema_slope, 1)
    ema_slope_shifted[0] = 0.0
    ema_slope_shifted = np.nan_to_num(ema_slope_shifted, nan=0.0)
    df["_v1_close_ema_slope_3"] = ema_slope_shifted

    # --- Candle shape / struktur ---
    # Del: Time range_features block
    t_range_start = time.perf_counter()
    # DEL 2: Replace .abs(), .replace(), .max(), .min(), .clip(), .shift() with NumPy
    open_arr = df["open"].to_numpy(dtype=np.float64)
    high_arr = df["high"].to_numpy(dtype=np.float64)
    low_arr = df["low"].to_numpy(dtype=np.float64)
    close_arr = df["close"].to_numpy(dtype=np.float64)
    
    body = np.abs(close_arr - open_arr)
    range_arr = high_arr - low_arr
    # A real zero-range bar (high == low on quiet holiday tape) has exactly
    # zero body and zero wicks, so 0/eps = 0.0 is the honest defined value.
    # Masking zero range to NaN created mid-series non-finite gaps that the
    # causal output validator rejects.

    # Max/min of open and close
    open_close_max = np.maximum(open_arr, close_arr)
    open_close_min = np.minimum(open_arr, close_arr)
    upper = np.clip(high_arr - open_close_max, 0, np.inf)
    lower = np.clip(open_close_min - low_arr, 0, np.inf)

    body_tr = body / (range_arr + 1e-12)
    body_tr_shifted = np.roll(body_tr, 1)
    body_tr_shifted[0] = 0.0
    df["_v1_body_tr"] = body_tr_shifted
    
    upper_tr = upper / (range_arr + 1e-12)
    upper_tr_shifted = np.roll(upper_tr, 1)
    upper_tr_shifted[0] = 0.0
    df["_v1_upper_tr"] = upper_tr_shifted
    
    lower_tr = lower / (range_arr + 1e-12)
    lower_tr_shifted = np.roll(lower_tr, 1)
    lower_tr_shifted[0] = 0.0
    df["_v1_lower_tr"] = lower_tr_shifted
    
    # Wick imbalance (mini-featurepack)
    # Upper wick vs lower wick imbalance
    wick_upper = upper
    wick_lower = lower
    wick_imbalance = (wick_upper - wick_lower) / (range_arr + 1e-12)
    wick_imbalance_shifted = np.roll(wick_imbalance, 1)
    wick_imbalance_shifted[0] = 0.0
    wick_imbalance_shifted = np.nan_to_num(wick_imbalance_shifted, nan=0.0)
    df["_v1_wick_imbalance"] = wick_imbalance_shifted
    
    # Range comparison 20 vs 100 periods (mini-featurepack)
    from gx1.features.rolling_timer import timed_rolling
    range_series = pd.Series(range_arr, index=df.index)
    range_20 = timed_rolling(range_series, 20, "mean", min_periods=10)
    range_100 = timed_rolling(range_series, 100, "mean", min_periods=50)
    range_20_arr = range_20.to_numpy(dtype=np.float64) if hasattr(range_20, 'to_numpy') else np.asarray(range_20, dtype=np.float64)
    range_100_arr = range_100.to_numpy(dtype=np.float64) if hasattr(range_100, 'to_numpy') else np.asarray(range_100, dtype=np.float64)
    range_comp = (range_20_arr / (range_100_arr + 1e-12)) - 1.0
    range_comp_shifted = np.roll(range_comp, 1)
    range_comp_shifted[0] = 0.0
    range_comp_shifted = np.nan_to_num(range_comp_shifted, nan=0.0)
    df["_v1_range_comp_20_100"] = range_comp_shifted
    # Range relative to the same physical ADR(20) horizon on native M1/M5.
    adr20 = _roll(range_series, 20 * bars_per_day, "mean")
    adr20_arr = adr20.to_numpy(dtype=np.float64) if hasattr(adr20, 'to_numpy') else np.asarray(adr20, dtype=np.float64)
    range_adr = range_arr / (adr20_arr + 1e-12)
    range_adr_shifted = np.roll(range_adr, 1)
    range_adr_shifted[0] = np.nan
    df["_v1_range_adr"] = range_adr_shifted
    t_range_end = time.perf_counter()
    perf_add("feat.basic_v1.range_features", t_range_end - t_range_start)

    # --- Kost-proxies ---
    # DEL 2: Replace .shift(), .astype() with NumPy
    spread_pct_arr = df["spread_pct"].to_numpy(dtype=np.float64)
    spread_bps = spread_pct_arr * 1e4
    spread_bps_shifted = np.roll(spread_bps, 1)
    spread_bps_shifted[0] = np.nan
    df["_v1_spread_p"] = spread_bps_shifted

    # --- Session ---
    # Mandatory: `_v1_is_EU`/`_v1_is_US` feed interaction features and the
    # required session-volatility-pressure block below. A frame without any
    # exact time source cannot silently lose session evidence.
    if not isinstance(df.index, pd.DatetimeIndex) and "ts" not in df.columns:
        raise RuntimeError(
            "[BASIC_V1_SESSION_TIME_SOURCE_MISSING] build_basic_v1 requires a "
            "DatetimeIndex or exact 'ts' column for session features"
        )
    df = add_session_features(
        df,
        decision_delay_seconds=decision_delay_seconds,
    )
    if "_v1_is_EU" not in df.columns or "_v1_is_US" not in df.columns:
        raise RuntimeError(
            "[BASIC_V1_SESSION_FLAGS_MISSING] add_session_features did not "
            "emit _v1_is_EU/_v1_is_US"
        )
    if (
        not np.isfinite(df["_v1_is_EU"].to_numpy(dtype=np.float64)).all()
        or not np.isfinite(df["_v1_is_US"].to_numpy(dtype=np.float64)).all()
    ):
        raise RuntimeError("[BASIC_V1_SESSION_FLAGS_NONFINITE]")

    # --- Mikrostruktur ---
    # DEL 2: Replace ALL pandas operations with NumPy
    # CLV (Close Location Value)
    high_arr = df["high"].to_numpy(dtype=np.float64)
    low_arr = df["low"].to_numpy(dtype=np.float64)
    close_arr = df["close"].to_numpy(dtype=np.float64)
    open_arr = df["open"].to_numpy(dtype=np.float64)
    
    range_arr = high_arr - low_arr
    # Replace 0 with NaN: use mask
    range_safe = range_arr.copy()
    range_safe[range_safe == 0.0] = np.nan
    
    clv = (close_arr - low_arr) / (range_safe + 1e-12)
    clv_shifted = np.roll(clv, 1)
    # CLV convention, deliberately different from the body/wick-share
    # convention above: a zero-range bar has a genuinely zero body and zero
    # wicks (0.0 is the honest share), but its close LOCATION inside a
    # zero-width range is undefined, and 0.5 is the true neutral midpoint of
    # the [0, 1] close-location domain. Keep nan -> 0.5 here.
    clv_shifted[0] = 0.5
    clv_shifted = np.nan_to_num(clv_shifted, nan=0.5)
    df["_v1_clv"] = clv_shifted
    
    # Range z-score
    range_for_z = high_arr - low_arr
    range_series = pd.Series(range_for_z, index=df.index)
    range_z = _zscore(range_series, 48)
    range_z_arr = range_z.to_numpy(dtype=np.float64) if hasattr(range_z, 'to_numpy') else np.asarray(range_z, dtype=np.float64)
    range_z_shifted = np.roll(range_z_arr, 1)
    range_z_shifted[0] = 0.0
    range_z_shifted = np.nan_to_num(range_z_shifted, nan=0.0)
    df["_v1_range_z"] = range_z_shifted
    
    # Spread z-score (rullende z for robusthet)
    sp_arr = df["spread_pct"].to_numpy(dtype=np.float64)
    # Sjekk om spread varierer (ikke konstant) - use NumPy std
    sp_std_val = np.std(sp_arr)
    if sp_std_val > 1e-6:
        # Rolling mean/std w144: use timed_rolling (will use pandas fallback but timed)
        from gx1.features.rolling_timer import timed_rolling
        sp_series = pd.Series(sp_arr, index=df.index)
        sp_mean_roll = timed_rolling(sp_series, 144, "mean", min_periods=72)
        sp_std_roll = timed_rolling(sp_series, 144, "std", min_periods=72, ddof=0)
        sp_mean_arr = sp_mean_roll.to_numpy(dtype=np.float64) if hasattr(sp_mean_roll, 'to_numpy') else np.asarray(sp_mean_roll, dtype=np.float64)
        sp_std_arr = sp_std_roll.to_numpy(dtype=np.float64) if hasattr(sp_std_roll, 'to_numpy') else np.asarray(sp_std_roll, dtype=np.float64)
        sp_std_arr = sp_std_arr + 1e-12
        spread_z = (sp_arr - sp_mean_arr) / sp_std_arr
        spread_z_shifted = np.roll(spread_z, 1)
        spread_z_shifted[0] = np.nan
        df["_v1_spread_z"] = spread_z_shifted
    else:
        # Constant observed spread has a defined zero z-score after the first
        # causally unavailable bar.
        spread_z_shifted = np.zeros(len(df), dtype=np.float64)
        spread_z_shifted[0] = np.nan
        df["_v1_spread_z"] = spread_z_shifted

    # Kurtosis av returer (vol-form)
    # Del: Time misc_roll block (includes quantiles, rolling operations)
    t_misc_roll_start = time.perf_counter()
    from gx1.features.rolling_timer import timed_rolling
    from gx1.utils.perf_timer import perf_add  # perf_inc already imported above in this fn (F811 fix)
    # DEL 2: Replace .pct_change(), .fillna() with NumPy
    close_arr = df["close"].to_numpy(dtype=np.float64)
    ret1_array = pct_change_np(close_arr, k=1)
    ret1_array = np.nan_to_num(ret1_array, nan=0.0)
    ret1 = pd.Series(ret1_array, index=df.index)  # Only for index, not for operations
    
    # Del 1: Start timing for moments sub-block
    t_misc_moments_start = time.perf_counter()
    
    # Del 1: Kurtosis sub-block (Numba-accelerated, no fallback)
    t_misc_moments_kurtosis_start = time.perf_counter()
    from gx1.features.rolling_np import rolling_kurtosis_w48
    kurt_result_np = rolling_kurtosis_w48(ret1_array, min_periods=12)
    # DEL 2: Replace .shift(), .fillna() with NumPy
    kurt_result_shifted = np.roll(kurt_result_np, 1)
    kurt_result_shifted[0] = 0.0
    kurt_result_shifted = np.nan_to_num(kurt_result_shifted, nan=0.0)
    df["_v1_kurt_r"] = kurt_result_shifted
    t_misc_moments_kurtosis_end = time.perf_counter()
    perf_add("feat.basic_v1.misc_roll.moments.kurtosis", t_misc_moments_kurtosis_end - t_misc_moments_kurtosis_start)
    
    t_misc_moments_end = time.perf_counter()
    perf_add("feat.basic_v1.misc_roll.moments", t_misc_moments_end - t_misc_moments_start)
    
    # Local-clock interactions only. MTF values and their local projected
    # slopes are attached later by the sole native-M5 V4 owner.
    t_misc_interactions_start = time.perf_counter()
    from gx1.features.array_utils import safe_clip, safe_mul
    
    # Array-first batch processing: fetch all inputs once, compute in NumPy, assign back

    # Del 1: Base interaksjoner
    t_misc_interactions_base_start = time.perf_counter()
    if "_v1_r5" in df.columns and "_v1_atr_regime_id" in df.columns:
        r5_arr = df["_v1_r5"].to_numpy(dtype=np.float64, na_value=0.0)
        atr_regime_arr = df["_v1_atr_regime_id"].to_numpy(dtype=np.float64, na_value=0.0)
        result = safe_clip(safe_mul(r5_arr, 1.0 + atr_regime_arr))
        df["_v1_int_r5_atr"] = result
    
    if "_v1_ema_diff" in df.columns and "_v1_is_US" in df.columns:
        ema_diff_arr = df["_v1_ema_diff"].to_numpy(dtype=np.float64, na_value=0.0)
        is_us_arr = df["_v1_is_US"].to_numpy(dtype=np.float64, na_value=0.0)
        result = safe_clip(safe_mul(ema_diff_arr, is_us_arr))
        df["_v1_int_ema_us"] = result
    
    t_misc_interactions_base_end = time.perf_counter()
    perf_add("feat.basic_v1.misc_roll.interactions.base", t_misc_interactions_base_end - t_misc_interactions_base_start)
    
    # Del 3: Ekstra interaksjoner (AUC-løft)
    t_misc_interactions_extra_start = time.perf_counter()
    if "_v1_clv" in df.columns and "_v1_atr_regime_id" in df.columns:
        clv_arr = df["_v1_clv"].to_numpy(dtype=np.float64, na_value=0.0)
        atr_regime_arr = df["_v1_atr_regime_id"].to_numpy(dtype=np.float64, na_value=0.0)
        result = safe_clip(safe_mul(clv_arr, 1.0 + atr_regime_arr))
        df["_v1_int_clv_atr"] = result
    
    if "_v1_range_z" in df.columns and "_v1_is_US" in df.columns:
        range_z_arr = df["_v1_range_z"].to_numpy(dtype=np.float64, na_value=0.0)
        is_us_arr = df["_v1_is_US"].to_numpy(dtype=np.float64, na_value=0.0)
        result = safe_clip(safe_mul(range_z_arr, is_us_arr))
        df["_v1_int_range_us"] = result
    t_misc_interactions_extra_end = time.perf_counter()
    perf_add("feat.basic_v1.misc_roll.interactions.extra", t_misc_interactions_extra_end - t_misc_interactions_extra_start)
    
    t_misc_interactions_end = time.perf_counter()
    perf_add("feat.basic_v1.misc_roll.interactions", t_misc_interactions_end - t_misc_interactions_start)
    
    # Causal session/volatility pressure. This is deliberately not named or
    # interpreted as spread/slippage/cost; those semantics require observations.
    t_misc_cost_proxy_start = time.perf_counter()
    pressure_required = ("close", "high", "low", "_v1_atr14", "_v1_is_US")
    pressure_missing = [
        column for column in pressure_required if column not in df.columns
    ]
    if pressure_missing:
        raise RuntimeError(
            "[BASIC_V1_SESSION_VOLATILITY_PRESSURE_SOURCE_MISSING] "
            f"{pressure_missing}"
        )
    atr14_arr = df["_v1_atr14"].to_numpy(dtype=np.float64)
    close_arr = df["close"].to_numpy(dtype=np.float64)
    high_arr = df["high"].to_numpy(dtype=np.float64)
    low_arr = df["low"].to_numpy(dtype=np.float64)
    is_us_arr = df["_v1_is_US"].to_numpy(dtype=np.float64)
    if (
        not np.isfinite(close_arr).all()
        or not np.isfinite(high_arr).all()
        or not np.isfinite(low_arr).all()
        or not np.isfinite(is_us_arr).all()
        or np.any(close_arr <= 0.0)
    ):
        raise RuntimeError(
            "[BASIC_V1_SESSION_VOLATILITY_PRESSURE_SOURCE_INVALID]"
        )
    atr_bps_arr = atr14_arr / close_arr * 1e4
    rng_z_arr = _zscore(
        pd.Series(high_arr - low_arr, index=df.index),
        48,
    ).to_numpy(dtype=np.float64)
    session_scale = 1.0 + 0.15 * is_us_arr
    pressure = session_scale * (
        np.log1p(np.maximum(atr_bps_arr, 0.0))
        + 0.25 * np.tanh(rng_z_arr)
    )
    pressure_shifted = np.roll(pressure, 1)
    pressure_shifted[0] = np.nan
    df["_v1_session_volatility_pressure"] = pressure_shifted
    t_misc_cost_proxy_end = time.perf_counter()
    perf_add(
        "feat.basic_v1.misc_roll.session_volatility_pressure",
        t_misc_cost_proxy_end - t_misc_cost_proxy_start,
    )
    
    # Time-of-day encoding (sin/cos) og rolling quantil-momenter
    t_misc_time_of_day_start = time.perf_counter()
    # DEL 2: Replace pd.to_datetime, .dt.hour, .shift(), .fillna() with NumPy
    if "ts" in df.columns or isinstance(df.index, pd.DatetimeIndex):
        if isinstance(df.index, pd.DatetimeIndex):
            # Extract hour/minute directly from DatetimeIndex (NumPy-friendly)
            h = df.index.hour.values if hasattr(df.index.hour, 'values') else np.array([ts.hour for ts in df.index])
            m = df.index.minute.values if hasattr(df.index.minute, 'values') else np.array([ts.minute for ts in df.index])
        elif "ts" in df.columns:
            # Convert once (not per bar) - this is OK as it's outside hot-path loop
            ts_col = pd.to_datetime(df["ts"], utc=True, errors="coerce")
            if isinstance(ts_col, pd.DatetimeIndex):
                h = ts_col.hour.values if hasattr(ts_col.hour, 'values') else np.array([ts.hour for ts in ts_col])
                m = ts_col.minute.values if hasattr(ts_col.minute, 'values') else np.array([ts.minute for ts in ts_col])
            else:
                h = ts_col.dt.hour.values
                m = ts_col.dt.minute.values
        else:
            h = np.zeros(len(df), dtype=np.int32)
            m = np.zeros(len(df), dtype=np.int32)
        
        h_float = (h.astype(np.float64) + m.astype(np.float64) / 60.0)
        tod_sin = np.sin(2 * np.pi * h_float / 24.0)
        tod_cos = np.cos(2 * np.pi * h_float / 24.0)
        # DEL 2: Replace .shift(), .fillna() with NumPy
        tod_sin_shifted = np.roll(tod_sin, 1)
        tod_sin_shifted[0] = 0.0
        tod_sin_shifted = np.nan_to_num(tod_sin_shifted, nan=0.0)
        df["_v1_tod_sin"] = tod_sin_shifted
        
        tod_cos_shifted = np.roll(tod_cos, 1)
        tod_cos_shifted[0] = 0.0
        tod_cos_shifted = np.nan_to_num(tod_cos_shifted, nan=0.0)
        df["_v1_tod_cos"] = tod_cos_shifted
    t_misc_time_of_day_end = time.perf_counter()
    perf_add("feat.basic_v1.misc_roll.time_of_day", t_misc_time_of_day_end - t_misc_time_of_day_start)
    
    # Rolling quantil-momenter
    t_misc_quantiles_start = time.perf_counter()
    
    # One formula path: quantiles are always materialized by this owner.
    if "close" in df.columns:
        from gx1.features.rolling_np import rolling_quantile_w48
        from gx1.utils.perf_timer import perf_add, perf_inc
        
        # DEL 2: Replace .pct_change(), .fillna() with NumPy
        close_arr = df["close"].to_numpy(dtype=np.float64)
        r1_arr = pct_change_np(close_arr, k=1)
        # Keep NaN for quantile computation (don't fill here)
        
        # Compute quantiles using Numba
        t_q90_start = time.perf_counter()
        q90_arr = rolling_quantile_w48(r1_arr, q=0.90, min_periods=24)
        t_q90_end = time.perf_counter()
        perf_add("rolling.numpy.quantile.w48", t_q90_end - t_q90_start)
        perf_inc("rolling.numpy.quantile.w48")
        
        t_q10_start = time.perf_counter()
        q10_arr = rolling_quantile_w48(r1_arr, q=0.10, min_periods=24)
        t_q10_end = time.perf_counter()
        perf_add("rolling.numpy.quantile.w48", t_q10_end - t_q10_start)
        perf_inc("rolling.numpy.quantile.w48")
        
        # DEL 2: Replace .shift(), .fillna() with NumPy
        q90_shifted = np.roll(q90_arr, 1)
        q90_shifted[0] = 0.0
        q90_shifted = np.nan_to_num(q90_shifted, nan=0.0)
        df["_v1_r1_q90_48"] = q90_shifted
        
        q10_shifted = np.roll(q10_arr, 1)
        q10_shifted[0] = 0.0
        q10_shifted = np.nan_to_num(q10_shifted, nan=0.0)
        df["_v1_r1_q10_48"] = q10_shifted
    t_misc_quantiles_end = time.perf_counter()
    perf_add("feat.basic_v1.misc_roll.quantiles", t_misc_quantiles_end - t_misc_quantiles_start)
    
    t_misc_roll_end = time.perf_counter()
    perf_add("feat.basic_v1.misc_roll", t_misc_roll_end - t_misc_roll_start)
    
    # --- Featurepack v2 (lav risiko, høy effekt) ---
    # DEL 2: Replace ALL pandas operations with NumPy
    # ATR z-score 10 vs 100 (vol-regime)
    tr_series = pd.Series(tr, index=df.index) if not isinstance(tr, pd.Series) else tr
    atr10 = _roll(tr_series, 10, "mean")
    atr100 = _roll(tr_series, 100, "mean")
    atr10_arr = atr10.to_numpy(dtype=np.float64) if hasattr(atr10, 'to_numpy') else np.asarray(atr10, dtype=np.float64)
    atr100_arr = atr100.to_numpy(dtype=np.float64) if hasattr(atr100, 'to_numpy') else np.asarray(atr100, dtype=np.float64)
    atr_z = (atr10_arr - atr100_arr) / (atr100_arr + 1e-12)
    atr_z_shifted = np.roll(atr_z, 1)
    atr_z_shifted[0] = 0.0
    atr_z_shifted = np.nan_to_num(atr_z_shifted, nan=0.0)
    df["_v1_atr_z_10_100"] = atr_z_shifted
    
    # TEMA slope 20 (trend-skarphet)
    # 2026-08-13 noise-amplifier repair: ``np.diff(..., n=3)`` was the
    # 3rd-order finite difference (a noise amplifier — the bug class repaired
    # in htf_features._model_native_htf_slope_v4 on 2026-08-09), not a slope.
    # Repaired to the k-bar change x[t] - x[t-3] (k=3 is the field's own
    # pre-existing lookback), normalized by this file's own unshifted atr14
    # source with the additive 1e-12 epsilon — the exact ATR-multiple
    # convention `_v1_ema_diff` above adopted on 2026-08-09.  That keeps the
    # output O(1) in ATR units, the scale class for which the consumer's
    # tanh(scale=1.0) is dimensionally sane (entry_trend_ema_v1 unit note),
    # instead of a raw-USD magnitude that tracks the multi-year price level.
    # Numerator and denominator are unshifted and shifted once together (row
    # t reads bar t-1's value); the ATR/TEMA warmup NaN prefix is kept as
    # honest causal warmup instead of nan_to_num zeros.
    tema20 = _tema(df["close"], 20)
    tema20_arr = tema20.to_numpy(dtype=np.float64) if hasattr(tema20, 'to_numpy') else np.asarray(tema20, dtype=np.float64)
    atr14_unshifted_arr = atr14.to_numpy(dtype=np.float64)
    tema20_change3 = tema20_arr - np.concatenate(
        [np.full(3, np.nan), tema20_arr[:-3]]
    )
    tema20_slope = tema20_change3 / (atr14_unshifted_arr + 1e-12)
    tema20_slope_shifted = np.roll(tema20_slope, 1)
    tema20_slope_shifted[0] = np.nan
    df["_v1_tema_slope_20"] = tema20_slope_shifted
    
    # Bollinger Band squeeze 20_2 (volatilitetssqueeze)
    from gx1.features.rolling_timer import timed_rolling
    bb_mean = timed_rolling(df["close"], 20, "mean", min_periods=10)
    bb_std = timed_rolling(df["close"], 20, "std", min_periods=10, ddof=0)
    bb_mean_arr = bb_mean.to_numpy(dtype=np.float64) if hasattr(bb_mean, 'to_numpy') else np.asarray(bb_mean, dtype=np.float64)
    bb_std_arr = bb_std.to_numpy(dtype=np.float64) if hasattr(bb_std, 'to_numpy') else np.asarray(bb_std, dtype=np.float64)
    bb_upper = bb_mean_arr + 2.0 * bb_std_arr
    bb_lower = bb_mean_arr - 2.0 * bb_std_arr
    bb_width = (bb_upper - bb_lower) / (bb_mean_arr + 1e-12)
    bb_width_series = pd.Series(bb_width, index=df.index)
    bb_width_mean = timed_rolling(bb_width_series, 100, "mean", min_periods=50)
    bb_width_mean_arr = bb_width_mean.to_numpy(dtype=np.float64) if hasattr(bb_width_mean, 'to_numpy') else np.asarray(bb_width_mean, dtype=np.float64)
    bb_squeeze = (bb_width / (bb_width_mean_arr + 1e-12)) - 1.0
    bb_squeeze_shifted = np.roll(bb_squeeze, 1)
    bb_squeeze_shifted[0] = 0.0
    bb_squeeze_shifted = np.nan_to_num(bb_squeeze_shifted, nan=0.0)
    df["_v1_bb_squeeze_20_2"] = bb_squeeze_shifted
    
    # KAMA slope 30 (trend med støyfilter)
    # 2026-08-13 noise-amplifier repair: ``np.diff(..., n=5)`` was the
    # 5th-order finite difference (coefficients 1,-5,10,-10,5,-1 — the exact
    # noise amplifier repaired in htf_features._model_native_htf_slope_v4 on
    # 2026-08-09, measured there as ~15.9x noise), not a slope.  Repaired to
    # the k-bar change x[t] - x[t-5] (k=5 is the field's own pre-existing
    # lookback, the same k as the repaired HTF slope5 fields), normalized by
    # this file's own unshifted atr14 source with the additive 1e-12 epsilon
    # — the `_v1_ema_diff` ATR-multiple convention.  Same rationale as
    # `_v1_tema_slope_20` directly above: O(1) ATR units keep the consumer's
    # tanh(scale=1.0) sane (entry_trend_ema_v1 unit note) and remove the
    # era-proxy USD magnitude.  Numerator/denominator shifted once together;
    # honest causal NaN warmup prefix preserved.
    kama30 = _kama(df["close"], 30)
    kama30_arr = kama30.to_numpy(dtype=np.float64) if hasattr(kama30, 'to_numpy') else np.asarray(kama30, dtype=np.float64)
    kama30_change5 = kama30_arr - np.concatenate(
        [np.full(5, np.nan), kama30_arr[:-5]]
    )
    kama30_slope = kama30_change5 / (atr14_unshifted_arr + 1e-12)
    kama30_slope_shifted = np.roll(kama30_slope, 1)
    kama30_slope_shifted[0] = np.nan
    df["_v1_kama_slope_30"] = kama30_slope_shifted
    
    # Return EMA ratio 5 vs 34 (momentum vs medium trend)
    # DEL 2: Replace .pct_change(), .fillna(), .abs() with NumPy
    close_arr = df["close"].to_numpy(dtype=np.float64)
    ret1_array = pct_change_np(close_arr, k=1)
    ret1_array = np.nan_to_num(ret1_array, nan=0.0)
    ret1_abs = np.abs(ret1_array)
    ret1_abs_series = pd.Series(ret1_abs, index=df.index)
    ret_ema5 = _ema(ret1_abs_series, 5)
    ret_ema34 = _ema(ret1_abs_series, 34)
    ret_ema5_arr = ret_ema5.to_numpy(dtype=np.float64) if hasattr(ret_ema5, 'to_numpy') else np.asarray(ret_ema5, dtype=np.float64)
    ret_ema34_arr = ret_ema34.to_numpy(dtype=np.float64) if hasattr(ret_ema34, 'to_numpy') else np.asarray(ret_ema34, dtype=np.float64)
    ret_ema_ratio = ret_ema5_arr / (ret_ema34_arr + 1e-12)
    ret_ema_ratio_shifted = np.roll(ret_ema_ratio, 1)
    ret_ema_ratio_shifted[0] = 1.0
    ret_ema_ratio_shifted = np.nan_to_num(ret_ema_ratio_shifted, nan=1.0)
    df["_v1_ret_ema_ratio_5_34"] = ret_ema_ratio_shifted
    
    # --- Raskere features (compression→expansion) ---
    # 1) body_share_1: store kropp vs range (signal på sterk momentum)
    # DEL 2: Replace .abs(), .replace(), .shift(), .fillna() with NumPy
    body = np.abs(close_arr - open_arr)
    range_arr = high_arr - low_arr
    # Zero-range bar: zero body over zero range -> 0/eps = 0.0 is the honest
    # defined value, per this file's stated candle-shape convention (see the
    # _v1_body_tr block: same quantity, same 1e-12 epsilon, row-0 seed 0.0).
    # The previous NaN mask + nan_to_num(0.5) fabricated a "half body" on
    # quiet holiday tape. Contrast with _v1_clv below, where 0.5 genuinely is
    # the neutral value of a [0, 1] close-location.
    body_share = body / (range_arr + 1e-12)
    body_share_shifted = np.roll(body_share, 1)
    body_share_shifted[0] = 0.0
    df["_v1_body_share_1"] = body_share_shifted
    
    # 2) tr_1_over_atr_14: eksplosjon relativt ATR (breakout-styrke)
    tr1 = _true_range(df["high"], df["low"], df["close"])
    tr1_arr = tr1.to_numpy(dtype=np.float64) if hasattr(tr1, 'to_numpy') else np.asarray(tr1, dtype=np.float64)
    # Use atr14_arr from earlier in function (defined in ATR section)
    atr14_arr_for_tr = df["_v1_atr14"].to_numpy(dtype=np.float64)
    tr1_over_atr = tr1_arr / (atr14_arr_for_tr + 1e-12)
    tr1_over_atr_shifted = np.roll(tr1_over_atr, 1)
    tr1_over_atr_shifted[0] = 1.0
    tr1_over_atr_shifted = np.nan_to_num(tr1_over_atr_shifted, nan=1.0)
    df["_v1_tr_1_over_atr_14"] = tr1_over_atr_shifted
    
    # 3) comp3_ratio: 3-bar kompresjon (lave std indikerer squeeze)
    # Del 2C: Time comp3_ratio block
    from gx1.utils.perf_timer import perf_add, perf_inc
    t_comp3_start = time.perf_counter()
    
    # DEL 2: Always use NumPy rolling (no pandas fallback)
    from gx1.features.rolling_np import rolling_std_3
    close_std3_np = rolling_std_3(df["close"].to_numpy(dtype=np.float64), min_periods=2, ddof=0)
    close_std3_arr = close_std3_np
    
    from gx1.features.rolling_timer import timed_rolling
    close_std20 = timed_rolling(df["close"], 20, "std", min_periods=10, ddof=0)
    close_std20_arr = close_std20.to_numpy(dtype=np.float64) if hasattr(close_std20, 'to_numpy') else np.asarray(close_std20, dtype=np.float64)
    # DEL 2: Replace .shift(), .fillna() with NumPy
    comp3_ratio = close_std3_arr / (close_std20_arr + 1e-12)
    comp3_ratio_shifted = np.roll(comp3_ratio, 1)
    comp3_ratio_shifted[0] = 1.0
    comp3_ratio_shifted = np.nan_to_num(comp3_ratio_shifted, nan=1.0)
    df["_v1_comp3_ratio"] = comp3_ratio_shifted
    
    # Del 2C: Record comp3_ratio time
    t_comp3_end = time.perf_counter()
    perf_add("feat.basic_v1.comp3_ratio", t_comp3_end - t_comp3_start)
    
    # Preserve causal warmup. Never convert missing/non-finite model evidence
    # into a live zero signal.
    # Del: Time final_pack block (df operations, cleanup)
    t_final_start = time.perf_counter()
    newcols = [c for c in df.columns if c.startswith("_v1")]
    for col in newcols:
        col_arr = df[col].to_numpy(dtype=np.float64)
        df[col] = _validate_causal_feature_column(col_arr, name=col)
    t_final_end = time.perf_counter()
    perf_add("feat.basic_v1.final_pack", t_final_end - t_final_start)
    
    t_end = time.perf_counter()
    feature_build_time_ms = (t_end - t_start) * 1000.0
    perf_add("feat.basic_v1.total", t_end - t_start)
    perf_add("feat.basic_v1.total_ms", feature_build_time_ms)
    return df, newcols


def _tema(series, period):
    """Triple Exponential Moving Average (TEMA)."""
    ema1 = _ema(series, period)
    ema2 = _ema(ema1, period)
    ema3 = _ema(ema2, period)
    tema = 3 * ema1 - 3 * ema2 + ema3
    return tema


def kama_np(prices: np.ndarray, period: int, fast: int = 2, slow: int = 30) -> np.ndarray:
    """
    Del 1: Incremental KAMA implementation using NumPy (no pandas rolling).
    
    Kaufman Adaptive Moving Average (KAMA) - streaming/incremental version.
    
    Args:
        prices: 1D numpy array of prices
        period: Period for volatility calculation
        fast: Fast smoothing constant period (default 2)
        slow: Slow smoothing constant period (default 30)
    
    Returns:
        1D numpy array of KAMA values (same length as prices)
    
    Formula:
        ER = abs(price[t] - price[t-n]) / sum_{i=1..n} abs(price[t-i+1] - price[t-i])
        SC = (ER * (fastSC - slowSC) + slowSC)^2
        KAMA[t] = KAMA[t-1] + SC * (price[t] - KAMA[t-1])
        fastSC = 2/(fast+1), slowSC = 2/(slow+1)
    """
    n = len(prices)
    if n == 0:
        return np.array([])
    
    # Precompute constants
    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    sc_diff = fast_sc - slow_sc
    
    # Convert to float64, handle NaN/Inf defensively
    prices_clean = np.asarray(prices, dtype=np.float64)
    if np.any(np.isnan(prices_clean)) or np.any(np.isinf(prices_clean)):
        # Handle NaN/Inf: forward-fill where possible
        valid_mask = np.isfinite(prices_clean)
        if not np.any(valid_mask):
            return np.full(n, np.nan, dtype=np.float64)
        # Forward-fill from first valid value
        first_valid_idx = np.argmax(valid_mask)
        prices_clean[:first_valid_idx] = prices_clean[first_valid_idx]
        # Forward-fill remaining NaNs
        for i in range(first_valid_idx + 1, n):
            if not np.isfinite(prices_clean[i]):
                prices_clean[i] = prices_clean[i-1]
    
    # Initialize output
    kama = np.zeros(n, dtype=np.float64)
    if n > 0:
        kama[0] = prices_clean[0]
    
    # Compute price changes (for volatility calculation)
    price_diffs = np.abs(np.diff(prices_clean, prepend=prices_clean[0]))
    
    # Streaming calculation
    for i in range(1, n):
        if i < period:
            # Not enough data for ER calculation - use simple EMA-like initialization
            alpha = 2.0 / (i + 1.0)
            kama[i] = kama[i-1] + alpha * (prices_clean[i] - kama[i-1])
        else:
            # Calculate Efficiency Ratio (ER)
            change = abs(prices_clean[i] - prices_clean[i - period])
            # Volatility = sum of absolute price changes over period
            volatility = np.sum(price_diffs[(i - period + 1):(i + 1)])
            
            # Avoid division by zero
            if volatility < 1e-12:
                er = 0.0
            else:
                er = change / volatility
            
            # Calculate Smoothing Constant (SC)
            sc = (er * sc_diff + slow_sc) ** 2
            # Clamp SC to reasonable range [0, 1] (use min/max for scalars to avoid numpy clip segfault)
            sc = max(0.0, min(1.0, float(sc)))
            
            # Update KAMA
            kama[i] = kama[i-1] + sc * (prices_clean[i] - kama[i-1])
    
    return kama

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
    
    # Use incremental NumPy implementation
    kama_values = kama_np(prices, period)
    
    # Return as Series if input was Series, otherwise array
    if index is not None:
        return pd.Series(kama_values, index=index, dtype=np.float64)
    else:
        return kama_values
