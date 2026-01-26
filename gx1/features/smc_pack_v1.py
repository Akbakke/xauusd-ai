#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SMC (Smart Money Concepts) Starter Pack v1

PREBUILT-only feature pack for market structure, liquidity, and displacement analysis.
All features are deterministic, causal, and ATR-normalized where applicable.

FASE 0.3: Global kill-switch - GX1_FEATURE_BUILD_DISABLED=1 forbydder ALL feature-building.
This module is ONLY used in prebuilt builder, NOT in runtime replay.

Features:
- Market Structure: swings, HH/HL/LH/LL, BOS, CHOCH
- Liquidity: EQ levels, sweeps, reclaims
- Displacement/Compression: TR, ATR, compression ratio
- Microstructure: wick ratios, body ratio, CLV
- Context Anchors: HTF EMA distance, premium/discount

All distance/size features are ATR-normalized in bps.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import os
import logging

log = logging.getLogger(__name__)

# FASE 0.3: Global kill-switch - hard-fail hvis feature-building er deaktivert
def _check_feature_build_enabled():
    """Check if feature building is enabled (must be False in PREBUILT mode)."""
    feature_build_disabled = os.getenv("GX1_FEATURE_BUILD_DISABLED", "0") == "1"
    if feature_build_disabled:
        raise RuntimeError(
            "[PREBUILT_FAIL] build_smc_pack_v1() called while GX1_FEATURE_BUILD_DISABLED=1. "
            "Feature-building is completely disabled in prebuilt mode. "
            "This is a hard invariant - prebuilt features must be used directly."
        )


def _get_atr_ref(df: pd.DataFrame, use_htf: bool = True) -> pd.Series:
    """
    Get ATR reference for normalization.
    
    Priority:
    1. H1 ATR (if available and use_htf=True)
    2. M5 ATR (fallback)
    
    Returns:
        ATR series in bps (same length as df)
    """
    # Try H1 ATR first (if available)
    if use_htf and "_h1_atr" in df.columns:
        atr_ref = df["_h1_atr"].fillna(method="ffill").fillna(method="bfill")
        log.debug("[SMC] Using H1 ATR for normalization")
        return atr_ref
    
    # Fallback to M5 ATR
    if "_v1_atr14" in df.columns:
        atr_ref = df["_v1_atr14"].fillna(method="ffill").fillna(method="bfill")
        log.debug("[SMC] Using M5 ATR (_v1_atr14) for normalization")
        return atr_ref
    
    # Compute M5 ATR if not available
    from gx1.execution.live_features import compute_atr_bps
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    atr_series = compute_atr_bps(pd.DataFrame({"high": high, "low": low, "close": close}), period=14)
    log.warning("[SMC] Computed M5 ATR on-the-fly (should be in prebuilt)")
    return atr_series


def _find_pivots(
    high: np.ndarray,
    low: np.ndarray,
    L: int = 5,
    R: int = 5,
    k_atr: float = 0.5,
    atr_ref: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find pivot highs and lows with causal confirmation.
    
    Args:
        high: High prices
        low: Low prices
        L: Left window size (bars before pivot)
        R: Right window size (bars after pivot for confirmation)
        k_atr: Minimum amplitude filter (move must be >= k_atr * ATR)
        atr_ref: ATR reference for amplitude filter (if None, uses simple range)
    
    Returns:
        (pivot_highs, pivot_lows) - boolean arrays indicating pivot locations
    """
    n = len(high)
    pivot_highs = np.zeros(n, dtype=bool)
    pivot_lows = np.zeros(n, dtype=bool)
    
    # Find local extrema
    for i in range(L, n - R):
        # Check for pivot high
        is_pivot_high = True
        for j in range(i - L, i + R + 1):
            if j != i and high[j] >= high[i]:
                is_pivot_high = False
                break
        
        if is_pivot_high:
            # Amplitude filter: check if move from pivot to next swing is significant
            if atr_ref is not None and k_atr > 0:
                # Find next swing (simplified: next local extremum)
                next_swing_high = high[i]
                next_swing_low = low[i]
                for j in range(i + 1, min(i + R * 2, n)):
                    if high[j] > next_swing_high:
                        next_swing_high = high[j]
                    if low[j] < next_swing_low:
                        next_swing_low = low[j]
                
                move_size = max(next_swing_high - high[i], high[i] - next_swing_low)
                min_move = k_atr * atr_ref[i] if i < len(atr_ref) else 0
                
                if move_size < min_move:
                    continue  # Skip this pivot (amplitude too small)
            
            # Mark pivot at confirmation time (i + R)
            if i + R < n:
                pivot_highs[i + R] = True
        
        # Check for pivot low
        is_pivot_low = True
        for j in range(i - L, i + R + 1):
            if j != i and low[j] <= low[i]:
                is_pivot_low = False
                break
        
        if is_pivot_low:
            # Amplitude filter
            if atr_ref is not None and k_atr > 0:
                next_swing_high = high[i]
                next_swing_low = low[i]
                for j in range(i + 1, min(i + R * 2, n)):
                    if high[j] > next_swing_high:
                        next_swing_high = high[j]
                    if low[j] < next_swing_low:
                        next_swing_low = low[j]
                
                move_size = max(next_swing_high - low[i], low[i] - next_swing_low)
                min_move = k_atr * atr_ref[i] if i < len(atr_ref) else 0
                
                if move_size < min_move:
                    continue
            
            if i + R < n:
                pivot_lows[i + R] = True
    
    return pivot_highs, pivot_lows


def _compute_swing_state(
    pivot_highs: np.ndarray,
    pivot_lows: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
) -> np.ndarray:
    """
    Compute HH/HL/LH/LL state.
    
    Returns:
        State array: 0=HH, 1=HL, 2=LH, 3=LL, 4=unknown
    """
    n = len(pivot_highs)
    state = np.full(n, 4, dtype=np.int32)  # 4 = unknown
    
    last_swing_high_idx = -1
    last_swing_low_idx = -1
    
    for i in range(n):
        if pivot_highs[i]:
            if last_swing_high_idx >= 0:
                if high[i] > high[last_swing_high_idx]:
                    state[i] = 0  # HH
                else:
                    state[i] = 1  # HL
            last_swing_high_idx = i
        
        if pivot_lows[i]:
            if last_swing_low_idx >= 0:
                if low[i] > low[last_swing_low_idx]:
                    state[i] = 2  # LH
                else:
                    state[i] = 3  # LL
            last_swing_low_idx = i
        
        # Propagate state forward until next swing
        if i > 0 and state[i] == 4:
            state[i] = state[i - 1]
    
    return state


def _find_eq_levels(
    high: np.ndarray,
    low: np.ndarray,
    eq_tol_atr: float = 0.3,
    atr_ref: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find Equal Highs (EQ High) and Equal Lows (EQ Low) clusters.
    
    Args:
        high: High prices
        low: Low prices
        eq_tol_atr: Tolerance for "equal" in ATR units
        atr_ref: ATR reference for tolerance
    
    Returns:
        (eq_high, eq_low, cluster_size) - boolean arrays and cluster size
    """
    n = len(high)
    eq_high = np.zeros(n, dtype=bool)
    eq_low = np.zeros(n, dtype=bool)
    cluster_size = np.zeros(n, dtype=np.int32)
    
    if atr_ref is None:
        # Use simple price-based tolerance
        price_range = np.max(high) - np.min(low)
        eq_tol = eq_tol_atr * price_range * 0.01  # Approximate
    else:
        eq_tol = eq_tol_atr * np.median(atr_ref) * 1e-4  # Convert bps to price units
    
    # Find EQ High clusters
    for i in range(n):
        cluster = [i]
        for j in range(i + 1, min(i + 20, n)):  # Look ahead max 20 bars
            if abs(high[j] - high[i]) <= eq_tol:
                cluster.append(j)
        
        if len(cluster) >= 2:  # At least 2 touches
            for idx in cluster:
                eq_high[idx] = True
                cluster_size[idx] = len(cluster)
    
    # Find EQ Low clusters
    for i in range(n):
        cluster = [i]
        for j in range(i + 1, min(i + 20, n)):
            if abs(low[j] - low[i]) <= eq_tol:
                cluster.append(j)
        
        if len(cluster) >= 2:
            for idx in cluster:
                eq_low[idx] = True
                if cluster_size[idx] < len(cluster):
                    cluster_size[idx] = len(cluster)
    
    return eq_high, eq_low, cluster_size


def build_smc_pack_v1(
    df: pd.DataFrame,
    L: int = 5,
    R: int = 5,
    k_atr: float = 0.5,
    eq_tol_atr: float = 0.3,
    use_htf_atr: bool = True,
) -> pd.DataFrame:
    """
    Build SMC Starter Pack v1 features.
    
    Args:
        df: DataFrame with OHLCV data (must have DatetimeIndex)
        L: Left window for pivot detection (default: 5)
        R: Right window for pivot confirmation (default: 5)
        k_atr: Minimum amplitude filter in ATR units (default: 0.5)
        eq_tol_atr: EQ level tolerance in ATR units (default: 0.3)
        use_htf_atr: Use H1 ATR if available (default: True)
    
    Returns:
        DataFrame with SMC features added (smc_* columns)
    
    FASE 0.3: Global kill-switch - hard-fail hvis feature-building er deaktivert.
    """
    _check_feature_build_enabled()
    
    if df is None or len(df) == 0:
        raise ValueError("[SMC] df is empty – cannot build SMC features")
    
    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("[SMC] df must have DatetimeIndex")
    
    df = df.copy()
    df = df.sort_index()
    
    # Get required columns
    required_cols = ['open', 'high', 'low', 'close']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"[SMC] Missing required columns: {missing}")
    
    # Get ATR reference for normalization
    atr_ref = _get_atr_ref(df, use_htf=use_htf_atr)
    atr_ref_arr = atr_ref.values if isinstance(atr_ref, pd.Series) else atr_ref
    
    # Convert to numpy for efficiency
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    close = df["close"].values.astype(np.float64)
    open_price = df["open"].values.astype(np.float64)
    
    n = len(df)
    
    # ============================================================
    # Market Structure: Swings and Pivots
    # ============================================================
    
    pivot_highs, pivot_lows = _find_pivots(high, low, L=L, R=R, k_atr=k_atr, atr_ref=atr_ref_arr)
    
    # Swing high/low (at confirmation time)
    df["smc_swing_high"] = pivot_highs.astype(np.float64)
    df["smc_swing_low"] = pivot_lows.astype(np.float64)
    
    # Distance to last swing (ATR-normalized in bps)
    last_swing_high_price = np.full(n, np.nan)
    last_swing_low_price = np.full(n, np.nan)
    
    for i in range(n):
        if pivot_highs[i]:
            last_swing_high_price[i] = high[i]
        elif i > 0:
            last_swing_high_price[i] = last_swing_high_price[i - 1]
        
        if pivot_lows[i]:
            last_swing_low_price[i] = low[i]
        elif i > 0:
            last_swing_low_price[i] = last_swing_low_price[i - 1]
    
    # Forward fill for distance calculation
    last_swing_high_price = pd.Series(last_swing_high_price).fillna(method="ffill").values
    last_swing_low_price = pd.Series(last_swing_low_price).fillna(method="ffill").values
    
    # Distance in ATR-normalized bps
    dist_to_swing_high = (close - last_swing_high_price) / (atr_ref_arr + 1e-8) * 10000  # Convert to bps
    dist_to_swing_low = (close - last_swing_low_price) / (atr_ref_arr + 1e-8) * 10000
    
    df["smc_dist_to_last_swing_high_atr"] = np.where(
        np.isnan(last_swing_high_price), np.nan, dist_to_swing_high
    )
    df["smc_dist_to_last_swing_low_atr"] = np.where(
        np.isnan(last_swing_low_price), np.nan, dist_to_swing_low
    )
    
    # HH/HL/LH/LL state
    swing_state = _compute_swing_state(pivot_highs, pivot_lows, high, low)
    df["smc_hh_hl_lh_ll_state"] = swing_state.astype(np.float64)
    
    # BOS (Break of Structure) - simplified: break of last swing high/low
    df["smc_bos_up"] = (close > last_swing_high_price).astype(np.float64)
    df["smc_bos_down"] = (close < last_swing_low_price).astype(np.float64)
    
    # CHOCH (Change of Character) - simplified: state change
    state_changed = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if swing_state[i] != swing_state[i - 1] and swing_state[i] != 4:
            state_changed[i] = True
    df["smc_choch"] = state_changed.astype(np.float64)
    
    # ============================================================
    # Liquidity: EQ Levels and Sweeps
    # ============================================================
    
    eq_high, eq_low, cluster_size = _find_eq_levels(high, low, eq_tol_atr=eq_tol_atr, atr_ref=atr_ref_arr)
    
    df["smc_eq_high"] = eq_high.astype(np.float64)
    df["smc_eq_low"] = eq_low.astype(np.float64)
    df["smc_eq_cluster_size"] = cluster_size.astype(np.float64)
    
    # Distance to EQ levels (ATR-normalized in bps)
    eq_high_price = np.full(n, np.nan)
    eq_low_price = np.full(n, np.nan)
    
    for i in range(n):
        if eq_high[i]:
            eq_high_price[i] = high[i]
        elif i > 0:
            eq_high_price[i] = eq_high_price[i - 1]
        
        if eq_low[i]:
            eq_low_price[i] = low[i]
        elif i > 0:
            eq_low_price[i] = eq_low_price[i - 1]
    
    eq_high_price = pd.Series(eq_high_price).fillna(method="ffill").values
    eq_low_price = pd.Series(eq_low_price).fillna(method="ffill").values
    
    dist_to_eq_high = (close - eq_high_price) / (atr_ref_arr + 1e-8) * 10000
    dist_to_eq_low = (close - eq_low_price) / (atr_ref_arr + 1e-8) * 10000
    
    df["smc_dist_to_eq_high_atr"] = np.where(np.isnan(eq_high_price), np.nan, dist_to_eq_high)
    df["smc_dist_to_eq_low_atr"] = np.where(np.isnan(eq_low_price), np.nan, dist_to_eq_low)
    
    # Sweeps (simplified: break of EQ level followed by reclaim)
    sweep_up = np.zeros(n, dtype=bool)
    sweep_down = np.zeros(n, dtype=bool)
    sweep_size = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        # Sweep up: break EQ high, then reclaim
        if eq_high[i - 1] and high[i] > eq_high_price[i - 1] * 1.001:  # 0.1% break
            # Look for reclaim (price comes back below EQ)
            for j in range(i + 1, min(i + 20, n)):
                if close[j] < eq_high_price[i - 1]:
                    sweep_up[i] = True
                    sweep_size[i] = (high[i] - eq_high_price[i - 1]) / (atr_ref_arr[i] + 1e-8) * 10000
                    break
        
        # Sweep down: break EQ low, then reclaim
        if eq_low[i - 1] and low[i] < eq_low_price[i - 1] * 0.999:
            for j in range(i + 1, min(i + 20, n)):
                if close[j] > eq_low_price[i - 1]:
                    sweep_down[i] = True
                    sweep_size[i] = (eq_low_price[i - 1] - low[i]) / (atr_ref_arr[i] + 1e-8) * 10000
                    break
    
    df["smc_sweep_up"] = sweep_up.astype(np.float64)
    df["smc_sweep_down"] = sweep_down.astype(np.float64)
    df["smc_sweep_size_atr"] = sweep_size
    
    # Reclaim speed (bars until reclaim after sweep)
    reclaim_speed = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if sweep_up[i] or sweep_down[i]:
            # Find reclaim
            for j in range(i + 1, min(i + 50, n)):
                if sweep_up[i] and close[j] < eq_high_price[i]:
                    reclaim_speed[i] = j - i
                    break
                elif sweep_down[i] and close[j] > eq_low_price[i]:
                    reclaim_speed[i] = j - i
                    break
    df["smc_reclaim_speed_bars"] = reclaim_speed
    
    # ============================================================
    # Displacement / Compression
    # ============================================================
    
    # True Range
    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - np.roll(close, 1)),
            np.abs(low - np.roll(close, 1))
        )
    )
    tr[0] = high[0] - low[0]  # First bar
    df["smc_tr"] = tr
    
    # ATR (M5, 14-period)
    atr_period = 14
    atr = np.full(n, np.nan)
    for i in range(atr_period - 1, n):
        atr[i] = np.mean(tr[i - atr_period + 1:i + 1])
    atr_bps = atr * 10000  # Convert to bps
    df["smc_atr"] = atr_bps
    
    # ATR slope (bps per bar)
    # Note: atr_bps is already in bps, so slope is just the difference
    atr_slope = np.full(n, np.nan)
    for i in range(1, n):
        if not np.isnan(atr_bps[i]) and not np.isnan(atr_bps[i - 1]):
            atr_slope[i] = atr_bps[i] - atr_bps[i - 1]  # Change in bps per bar
    df["smc_atr_slope"] = atr_slope
    
    # Range compression ratio (current range vs ATR)
    range_compression = (high - low) / (atr_ref_arr + 1e-8)
    df["smc_range_compression_ratio"] = range_compression
    
    # ============================================================
    # Microstructure
    # ============================================================
    
    range_price = high - low
    df["smc_upper_wick_ratio"] = np.where(range_price > 0, (high - np.maximum(open_price, close)) / range_price, 0.0)
    df["smc_lower_wick_ratio"] = np.where(range_price > 0, (np.minimum(open_price, close) - low) / range_price, 0.0)
    df["smc_body_ratio"] = np.where(range_price > 0, np.abs(close - open_price) / range_price, 0.0)
    
    # CLV (Close Location Value)
    df["smc_clv"] = np.where(range_price > 0, (close - low) / range_price - 0.5, 0.0)
    
    # ============================================================
    # Context Anchors
    # ============================================================
    
    # HTF EMA distance (ATR-normalized in bps)
    if "_h1_ema" in df.columns or "_h4_ema" in df.columns:
        htf_ema = df.get("_h1_ema", df.get("_h4_ema", None))
        if htf_ema is not None:
            htf_ema_arr = htf_ema.values
            dist_to_htf_ema = (close - htf_ema_arr) / (atr_ref_arr + 1e-8) * 10000
            df["smc_htf_ema_dist_atr"] = dist_to_htf_ema
        else:
            df["smc_htf_ema_dist_atr"] = np.nan
    else:
        df["smc_htf_ema_dist_atr"] = np.nan
    
    # Premium/Discount (simplified: position relative to recent range)
    lookback = 50
    recent_high = pd.Series(high).rolling(window=lookback, min_periods=1).max().values
    recent_low = pd.Series(low).rolling(window=lookback, min_periods=1).min().values
    recent_range = recent_high - recent_low
    premium_discount = np.where(
        recent_range > 0,
        (close - recent_low) / recent_range * 2 - 1,  # -1 (discount) to +1 (premium)
        0.0
    )
    df["smc_premium_discount"] = premium_discount
    
    # ============================================================
    # Validation: Fail-fast on NaN/Inf
    # ============================================================
    
    smc_cols = [col for col in df.columns if col.startswith("smc_")]
    for col in smc_cols:
        if df[col].isna().all():
            log.warning(f"[SMC] Column {col} is all NaN (may be expected for early bars)")
        elif df[col].isna().any():
            nan_count = df[col].isna().sum()
            log.debug(f"[SMC] Column {col} has {nan_count} NaN values (may be expected)")
        
        # Check for Inf
        if np.isinf(df[col]).any():
            inf_count = np.isinf(df[col]).sum()
            raise RuntimeError(
                f"[SMC_FATAL] Column {col} has {inf_count} Inf values. "
                "This indicates a computation error (division by zero or overflow)."
            )
    
    log.info(f"[SMC] Built {len(smc_cols)} SMC features")
    
    return df


def get_smc_feature_names() -> List[str]:
    """Return list of SMC feature names."""
    return [
        "smc_swing_high",
        "smc_swing_low",
        "smc_dist_to_last_swing_high_atr",
        "smc_dist_to_last_swing_low_atr",
        "smc_hh_hl_lh_ll_state",
        "smc_bos_up",
        "smc_bos_down",
        "smc_choch",
        "smc_eq_high",
        "smc_eq_low",
        "smc_eq_cluster_size",
        "smc_dist_to_eq_high_atr",
        "smc_dist_to_eq_low_atr",
        "smc_sweep_up",
        "smc_sweep_down",
        "smc_sweep_size_atr",
        "smc_reclaim_speed_bars",
        "smc_tr",
        "smc_atr",
        "smc_atr_slope",
        "smc_range_compression_ratio",
        "smc_upper_wick_ratio",
        "smc_lower_wick_ratio",
        "smc_body_ratio",
        "smc_clv",
        "smc_htf_ema_dist_atr",
        "smc_premium_discount",
    ]
