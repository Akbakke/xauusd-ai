"\"\"\"Live feature helpers for GX1 demo execution.\"\"\""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from gx1.features import basic_v1
from gx1.tuning.feature_manifest import align_features, load_manifest

log = logging.getLogger(__name__)

ATR_PERIOD = 14
ADR_WINDOW = 288  # ≈ 1 trading day on M5
VOL_BUCKET_Q33 = 0.033
VOL_BUCKET_Q66 = 0.050
PIPS_PER_PERCENT = 10000.0  # basis points helper


@dataclass
class EntryFeatureBundle:
    features: pd.DataFrame
    raw_row: pd.Series
    close_price: float
    atr_bps: float
    vol_bucket: str
    bid_open: Optional[float] = None
    bid_close: Optional[float] = None
    ask_open: Optional[float] = None
    ask_close: Optional[float] = None


def _compute_true_range(df: pd.DataFrame) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr_components = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    return tr_components.max(axis=1)


def compute_atr(series: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    tr = _compute_true_range(series)
    atr = tr.rolling(window=period, min_periods=max(1, period // 2)).mean()
    return atr


def compute_atr_bps(series: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    atr = compute_atr(series, period=period)
    close = series["close"]
    atr_bps = (atr / close) * PIPS_PER_PERCENT
    return atr_bps


def infer_session_tag(timestamp: pd.Timestamp) -> str:
    """
    Robust session classification based on UTC hour.
    
    SAFETY_V4: Returns one of 'EU', 'US', 'OVERLAP', 'ASIA' - never 'UNKNOWN' or 'ALL'.
    
    Session boundaries (UTC):
    - ASIA: 22:00-07:00 (Asian session, includes late US close and early EU pre-open)
    - EU: 07:00-12:00 (European session)
    - OVERLAP: 12:00-16:00 (EU/US overlap)
    - US: 16:00-22:00 (US session)
    
    Parameters
    ----------
    timestamp : pd.Timestamp
        Timestamp to classify (timezone-aware or naive, will be converted to UTC).
    
    Returns
    -------
    str
        Session tag: 'EU', 'US', 'OVERLAP', or 'ASIA'. Never 'UNKNOWN' or 'ALL'.
    """
    if pd.isna(timestamp):
        # SAFETY_V4: Never return UNKNOWN - default to ASIA for invalid timestamps
        log.warning("[SESSION_INFERENCE] Invalid timestamp (NaN), defaulting to ASIA")
        return "ASIA"
    
    # Convert to UTC if timezone-aware, assume UTC if naive
    if timestamp.tzinfo is not None:
        hour = timestamp.tz_convert("UTC").hour
    else:
        hour = timestamp.hour
    
    # SAFETY_V4: Explicit session boundaries with no gaps
    if 7 <= hour < 12:
        return "EU"
    elif 12 <= hour < 16:
        return "OVERLAP"
    elif 16 <= hour < 22:
        return "US"
    else:
        # Hours 22:00-06:59 UTC: ASIA session (covers late US close to early EU pre-open)
        return "ASIA"


def infer_vol_bucket(atr_adr_ratio: float) -> str:
    if np.isnan(atr_adr_ratio):
        return "MID"
    if atr_adr_ratio <= VOL_BUCKET_Q33:
        return "LOW"
    if atr_adr_ratio <= VOL_BUCKET_Q66:
        return "MID"
    return "HIGH"


def build_live_entry_features(candles: pd.DataFrame) -> EntryFeatureBundle:
    """
    Build aligned entry features for the most recent M5 candle window.

    Parameters
    ----------
    candles : pd.DataFrame
        Columns: open, high, low, close, volume. Index = UTC timestamps.
        Must contain at least ATR_PERIOD rows.
    """
    if candles.empty:
        raise ValueError("Candles DataFrame is empty")
    candles = candles.sort_index()

    # CRITICAL: Use basic_v1.build_basic_v1() (same as offline evaluation)
    # NOT make_features_v2.build_features() (which doesn't build _v1_r5, _v1_r8)
    # Add 'ts' column (required by basic_v1)
    candles_with_ts = candles.copy()
    candles_with_ts["ts"] = candles_with_ts.index
    
    # Build features using basic_v1 (same as offline evaluation)
    feature_df, _ = basic_v1.build_basic_v1(candles_with_ts)
    # Extract only _v1_* feature columns (drop OHLC and ts)
    feature_cols = [c for c in feature_df.columns if c.startswith("_v1_")]
    feature_df = feature_df[feature_cols].copy()
    feature_df.index = candles.index
    last_row = feature_df.iloc[-1].copy()

    manifest = load_manifest()
    aligned = align_features(feature_df, manifest=manifest, training_stats=manifest.get("training_stats"))
    aligned_last = aligned.tail(1).copy()
    
    # Volummaske i features (unngå falske signaler)
    # Bruk volume_mask til å nullstille/ignorere volum-drevne features per bar
    if "volume_mask" in candles.columns:
        volume_mask = candles["volume_mask"].iloc[-1] == 1
        if volume_mask:
            # Ekskluder/sett til NaN kun volum-baserte features
            vol_cols = ["vol_z", "tick_vol", "vratio_h1", "obv_delta", "volume_ma", "volume_ratio"]
            for col in vol_cols:
                if col in aligned_last.columns:
                    aligned_last.loc[:, col] = np.nan
            log.debug(
                "[VOLUME_MASK] Applied volume mask to bar %s: set %d volume-driven features to NaN",
                candles.index[-1],
                sum(1 for col in vol_cols if col in aligned_last.columns),
            )
    else:
        # Check if volume is 0 and apply mask
        if "volume" in candles.columns and candles["volume"].iloc[-1] == 0.0:
            vol_cols = ["vol_z", "tick_vol", "vratio_h1", "obv_delta", "volume_ma", "volume_ratio"]
            for col in vol_cols:
                if col in aligned_last.columns:
                    aligned_last.loc[:, col] = np.nan
            log.debug(
                "[VOLUME_MASK] Applied volume mask (volume=0) to bar %s: set %d volume-driven features to NaN",
                candles.index[-1],
                sum(1 for col in vol_cols if col in aligned_last.columns),
            )

    atr_series = compute_atr_bps(candles[["high", "low", "close"]])
    atr_bps = float(atr_series.iloc[-1])

    recent_window = candles.tail(ADR_WINDOW)
    adr = (recent_window["high"].max() - recent_window["low"].min()) if len(recent_window) >= 2 else np.nan
    adr_bps = (adr / recent_window["close"].iloc[-1]) * PIPS_PER_PERCENT if not np.isnan(adr) else np.nan
    atr_adr_ratio = float(atr_bps / adr_bps) if adr_bps and adr_bps > 0 else np.nan
    vol_bucket = infer_vol_bucket(atr_adr_ratio)

    close_price = float(candles["close"].iloc[-1])
    bid_open = float(candles["bid_open"].iloc[-1]) if "bid_open" in candles.columns else None
    bid_close = float(candles["bid_close"].iloc[-1]) if "bid_close" in candles.columns else None
    ask_open = float(candles["ask_open"].iloc[-1]) if "ask_open" in candles.columns else None
    ask_close = float(candles["ask_close"].iloc[-1]) if "ask_close" in candles.columns else None
    bundle = EntryFeatureBundle(
        features=aligned_last,
        raw_row=last_row,
        close_price=close_price,
        atr_bps=atr_bps,
        vol_bucket=vol_bucket,
        bid_open=bid_open,
        bid_close=bid_close,
        ask_open=ask_open,
        ask_close=ask_close,
    )
    return bundle


def build_live_exit_snapshot(
    trade: Dict[str, Any],
    candles: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a snapshot matching the exit model schema for a single open trade.

    Parameters
    ----------
    trade : dict
        Keys required:
            entry_time (pd.Timestamp)
            entry_price (float)
            side (str, 'long' or 'short')
            units (int)
            trade_id (str/int)
    candles : pd.DataFrame
        OHLCV frame used for excursions; same schema as build_live_entry_features.
    """
    entry_time = pd.to_datetime(trade["entry_time"])
    entry_price = float(trade["entry_price"])
    side_str = trade.get("side", "long").lower()
    side = 1 if side_str == "long" else -1

    candles = candles.sort_index()
    now_ts = candles.index[-1]
    window = candles[candles.index >= entry_time]
    if window.empty:
        window = candles.tail(ATR_PERIOD).copy()

    current_close = float(candles["close"].iloc[-1])
    high_since_entry = float(window["high"].max())
    low_since_entry = float(window["low"].min())

    if side > 0:
        mfe = (high_since_entry - entry_price) / entry_price
        mae = (entry_price - low_since_entry) / entry_price
    else:
        mfe = (entry_price - low_since_entry) / entry_price
        mae = (high_since_entry - entry_price) / entry_price

    mfe_bps = float(mfe * PIPS_PER_PERCENT)
    mae_bps = float(mae * PIPS_PER_PERCENT)
    net_move_bps = float((current_close - entry_price) / entry_price * PIPS_PER_PERCENT * side)

    atr_series = compute_atr_bps(candles[["high", "low", "close"]])
    atr_bps = float(atr_series.iloc[-1])

    delta_minutes = max((now_ts - entry_time).total_seconds() / 60.0, 0.0)
    bars_in_trade = int(max(1, round(delta_minutes / 5.0)))

    # Ensure UTC-aware timestamp for session_tag
    if now_ts.tzinfo is None:
        now_ts_utc = pd.Timestamp(now_ts, tz="UTC")
    else:
        now_ts_utc = now_ts.tz_convert("UTC")
    
    session_tag = infer_session_tag(now_ts_utc)
    # Create period_tag from tz-naive timestamp to avoid warning
    period_tag = str(now_ts_utc.tz_localize(None).to_period("Q"))
    vol_bucket = infer_vol_bucket(atr_bps / max(abs(net_move_bps) + 1e-6, 1.0) if atr_bps else np.nan)

    snapshot = pd.DataFrame(
        [
            {
                "entry_id": str(trade.get("trade_id", "")),
                "entry_time": entry_time,
                "snapshot_ts": now_ts_utc,
                "bars_in_trade": bars_in_trade,
                "side": side,
                "atr_bps": atr_bps,
                "mfe_so_far": mfe_bps,
                "mae_so_far": mae_bps,
                "net_move_bps": net_move_bps,
                "cost_bps": float(trade.get("cost_bps", 0.0)),
                "pnl_so_far": net_move_bps,  # cost already deducted
                "session_tag": session_tag,
                "period_tag": period_tag,
                "vol_bucket": vol_bucket,
                "trade_date": now_ts_utc.normalize(),
            }
        ]
    )
    return snapshot
