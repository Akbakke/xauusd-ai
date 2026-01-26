#!/usr/bin/env python3
"""
exit_snapshot_v1.py

Helper-modul for å bygge exit-snapshots som brukes både i:
- build_exit_dataset_v1.py (fra trade log CSV)
- runtime ExitCritic (fra aktiv trade i exit_manager)

Dette sikrer at feature-strukturen er konsistent mellom trening og runtime.
"""
from __future__ import annotations

from typing import Any, Dict, Optional
import pandas as pd
import numpy as np


def make_exit_snapshot(trade_row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Bygger exit-snapshot fra trade-row (enten fra CSV eller runtime trade).
    
    Args:
        trade_row: Dictionary med trade-data. Kan komme fra:
            - CSV trade log (pd.Series.to_dict())
            - Runtime trade object (konvertert til dict)
    
    Returns:
        Dictionary med features som matcher det datasettet forventer.
    """
    snapshot: Dict[str, Any] = {}
    
    # Core PnL metrics
    snapshot["pnl_bps"] = float(trade_row.get("pnl_bps", 0.0) or 0.0)
    snapshot["mfe_bps"] = float(trade_row.get("mfe_bps", 0.0) or 0.0)
    snapshot["mae_bps"] = float(trade_row.get("mae_bps", 0.0) or 0.0)
    
    # Bars held
    snapshot["bars_held"] = int(trade_row.get("bars_held", 0) or 0)
    
    # ATR/Spread (defensive defaults)
    snapshot["atr_bps_entry"] = float(trade_row.get("atr_bps_entry", trade_row.get("atr_bps", 0.0)) or 0.0)
    snapshot["spread_bps_entry"] = float(trade_row.get("spread_bps_entry", trade_row.get("spread_bps", 0.0)) or 0.0)
    
    # RR-type features
    eps = 1e-6
    mae_abs = abs(snapshot["mae_bps"])
    mfe_abs = abs(snapshot["mfe_bps"])
    snapshot["mae_abs"] = mae_abs
    snapshot["mfe_abs"] = mfe_abs
    snapshot["rr_mfe_over_mae"] = mfe_abs / (mae_abs + eps) if mae_abs > eps else 0.0
    
    # Regime encoding (trend)
    trend_regime = str(trade_row.get("trend_regime", "UNKNOWN") or "UNKNOWN")
    if trend_regime == "TREND_DOWN":
        snapshot["trend_regime_id"] = 0
    elif trend_regime == "TREND_NEUTRAL":
        snapshot["trend_regime_id"] = 1
    elif trend_regime == "TREND_UP":
        snapshot["trend_regime_id"] = 2
    else:
        snapshot["trend_regime_id"] = 1  # default to NEUTRAL
    
    # Regime encoding (vol)
    vol_regime = str(trade_row.get("vol_regime", "UNKNOWN") or "UNKNOWN")
    if vol_regime == "LOW":
        snapshot["vol_regime_id"] = 0
    elif vol_regime == "MEDIUM":
        snapshot["vol_regime_id"] = 1
    elif vol_regime == "HIGH":
        snapshot["vol_regime_id"] = 2
    elif vol_regime == "EXTREME":
        snapshot["vol_regime_id"] = 3
    else:
        snapshot["vol_regime_id"] = 1  # default to MEDIUM
    
    # Session encoding
    session = str(trade_row.get("session", "UNKNOWN") or "UNKNOWN")
    if session == "EU":
        snapshot["session_id"] = 0
    elif session == "OVERLAP":
        snapshot["session_id"] = 1
    elif session == "US":
        snapshot["session_id"] = 2
    elif session == "ASIA":
        snapshot["session_id"] = 3
    else:
        snapshot["session_id"] = 1  # default to OVERLAP
    
    # Entry probability
    snapshot["p_long"] = float(trade_row.get("p_long", trade_row.get("entry_p_long", 0.0)) or 0.0)
    
    # Time features (hvis tilgjengelig)
    entry_time = trade_row.get("entry_time")
    if entry_time is not None:
        try:
            if isinstance(entry_time, str):
                ts = pd.Timestamp(entry_time)
            else:
                ts = pd.Timestamp(entry_time)
            snapshot["entry_hour"] = int(ts.hour)
            snapshot["entry_dow"] = int(ts.dayofweek)
        except Exception:
            snapshot["entry_hour"] = 12  # default noon
            snapshot["entry_dow"] = 0  # default Monday
    else:
        snapshot["entry_hour"] = 12
        snapshot["entry_dow"] = 0
    
    # Categorical features (for backward compatibility med eksisterende dataset)
    snapshot["session"] = session
    snapshot["trend_regime"] = trend_regime
    snapshot["vol_regime"] = vol_regime
    
    return snapshot


def snapshot_to_feature_array(snapshot: Dict[str, Any], feature_cols: list[str]) -> list[float]:
    """
    Konverterer snapshot til feature-array i samme rekkefølge som feature_cols.
    
    Args:
        snapshot: Exit snapshot dictionary
        feature_cols: Liste med feature-kolonnenavn (fra metadata)
    
    Returns:
        Liste med feature-verdier i samme rekkefølge som feature_cols
    """
    return [float(snapshot.get(col, 0.0) or 0.0) for col in feature_cols]


__all__ = ["make_exit_snapshot", "snapshot_to_feature_array"]

