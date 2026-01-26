#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entry Critic Runtime Loader

Loads Entry Critic V1 model for runtime scoring in SNIPER.
Used in shadow-only mode: calculates score but does not gate trades.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

log = logging.getLogger(__name__)


def load_entry_critic_v1(
    model_path: Optional[Path] = None,
    meta_path: Optional[Path] = None,
) -> Tuple[Optional[Any], Optional[Dict[str, Any]], Optional[List[str]]]:
    """
    Load Entry Critic V1 model and metadata.
    
    Args:
        model_path: Path to model file (default: gx1/models/entry_critic_v1.joblib)
        meta_path: Path to metadata file (default: gx1/models/entry_critic_v1_meta.json)
    
    Returns:
        (model, meta, feature_order) tuple
        - model: Loaded model object or None if not found
        - meta: Metadata dict or None if not found
        - feature_order: List of feature names in correct order for prediction
    
    If model or meta not found, returns (None, None, None) and logs warning.
    """
    if model_path is None:
        model_path = Path("gx1/models/entry_critic_v1.joblib")
    if meta_path is None:
        meta_path = Path("gx1/models/entry_critic_v1_meta.json")
    
    # Check if files exist
    if not model_path.exists():
        log.warning(f"[ENTRY_CRITIC] Model not found: {model_path}, disabled")
        return None, None, None
    
    if not meta_path.exists():
        log.warning(f"[ENTRY_CRITIC] Metadata not found: {meta_path}, disabled")
        return None, None, None
    
    try:
        # Load metadata
        with open(meta_path, "r") as f:
            meta = json.load(f)
        
        # Load model
        try:
            import joblib
            model = joblib.load(model_path)
        except ImportError:
            log.error("[ENTRY_CRITIC] joblib not available, cannot load model")
            return None, None, None
        
        # Extract feature order from metadata
        feature_order = meta.get("features", [])
        
        log.info(
            f"[ENTRY_CRITIC] Loaded Entry Critic V1: "
            f"model={model_path.name}, target={meta.get('target', 'N/A')}, "
            f"features={len(feature_order)}"
        )
        
        return model, meta, feature_order
    
    except Exception as e:
        log.error(f"[ENTRY_CRITIC] Failed to load model: {e}", exc_info=True)
        return None, None, None


def prepare_entry_critic_features(
    p_long: float,
    spread_bps: Optional[float],
    atr_bps: Optional[float],
    trend_regime: Optional[str],
    vol_regime: Optional[str],
    session: Optional[str],
    shadow_hits: Optional[Dict[float, bool]] = None,
    real_threshold: Optional[float] = None,
    feature_order: Optional[List[str]] = None,
) -> Optional[List[float]]:
    """
    Prepare feature vector for Entry Critic prediction.
    
    Args:
        p_long: Probability of long trade
        spread_bps: Spread in basis points
        atr_bps: ATR in basis points
        trend_regime: Trend regime (TREND_UP, TREND_DOWN, TREND_NEUTRAL)
        vol_regime: Volatility regime (LOW, MEDIUM, HIGH)
        session: Session tag (EU, OVERLAP, US, ASIA)
        shadow_hits: Dict mapping threshold -> bool for shadow hits
        real_threshold: Real threshold used (for threshold_slot_numeric)
        feature_order: List of feature names in correct order
    
    Returns:
        Feature vector as list of floats, or None if feature_order not provided
    """
    if feature_order is None:
        return None
    
    # Initialize feature dict
    features: Dict[str, float] = {}
    
    # Numeric features
    features["p_long"] = float(p_long)
    features["spread_bps"] = float(spread_bps) if spread_bps is not None else 0.0
    features["atr_bps"] = float(atr_bps) if atr_bps is not None else 0.0
    
    # One-hot encode trend regime
    features["regime_trend_up"] = 1.0 if trend_regime == "TREND_UP" else 0.0
    features["regime_trend_down"] = 1.0 if trend_regime == "TREND_DOWN" else 0.0
    # TREND_NEUTRAL is implicit (both 0)
    
    # One-hot encode vol regime
    features["regime_vol_low"] = 1.0 if vol_regime == "LOW" else 0.0
    features["regime_vol_high"] = 1.0 if vol_regime == "HIGH" else 0.0
    # MEDIUM is implicit (both 0)
    
    # One-hot encode session
    features["session_eu"] = 1.0 if session == "EU" else 0.0
    features["session_overlap"] = 1.0 if session == "OVERLAP" else 0.0
    features["session_us"] = 1.0 if session == "US" else 0.0
    # ASIA is implicit (all 0)
    
    # Shadow hits (binary features)
    if shadow_hits is not None:
        for thr in [0.55, 0.58, 0.60, 0.62, 0.65]:
            features[f"shadow_hit_{thr:.2f}"] = 1.0 if shadow_hits.get(thr, False) else 0.0
    else:
        # Default: all False
        for thr in [0.55, 0.58, 0.60, 0.62, 0.65]:
            features[f"shadow_hit_{thr:.2f}"] = 0.0
    
    # Threshold slot numeric (map threshold to numeric slot)
    if real_threshold is not None:
        # Map threshold to slot: 0.55=0, 0.58=1, 0.60=2, 0.62=3, 0.65=4, 0.67=5, etc.
        threshold_slots = {0.55: 0, 0.58: 1, 0.60: 2, 0.62: 3, 0.65: 4, 0.67: 5}
        features["threshold_slot_numeric"] = float(
            threshold_slots.get(real_threshold, 5)
        )
    else:
        features["threshold_slot_numeric"] = 5.0  # Default to 0.67 slot
    
    # Build feature vector in correct order
    try:
        feature_vector = [features.get(feat, 0.0) for feat in feature_order]
        return feature_vector
    except Exception as e:
        log.warning(f"[ENTRY_CRITIC] Failed to prepare features: {e}")
        return None


def score_entry_critic(
    model: Any,
    feature_vector: List[float],
) -> Optional[float]:
    """
    Score entry using Entry Critic model.
    
    Args:
        model: Loaded Entry Critic model
        feature_vector: Feature vector in correct order
    
    Returns:
        Score (probability of profitable trade) as float 0-1, or None if error
    """
    if model is None or feature_vector is None:
        return None
    
    try:
        import numpy as np
        
        # Convert to numpy array and reshape for prediction
        X = np.array(feature_vector).reshape(1, -1)
        
        # Predict probability of positive class (class 1 = profitable)
        proba = model.predict_proba(X)
        
        # Return probability of class 1 (profitable)
        if proba.shape[1] >= 2:
            score = float(proba[0, 1])  # Class 1 probability
        else:
            score = float(proba[0, 0])  # Fallback if only one class
        
        return score
    
    except Exception as e:
        log.warning(f"[ENTRY_CRITIC] Failed to score entry: {e}")
        return None













