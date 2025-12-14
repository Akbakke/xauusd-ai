#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid exit router implementations.

Provides V1 (rule-based), V2 (ML decision tree), V2B (conservative ML),
V3 (ML decision tree RAW, conservative), and Adaptive routers for choosing
between RULE5 and RULE6A exit policies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

ExitPolicyName = Literal["RULE5", "RULE6A"]

# Standardization parameters from training data
# These match the StandardScaler used during training
ATR_PCT_MEAN = 50.687833
ATR_PCT_STD = 29.005795


@dataclass
class ExitRouterContext:
    """Context for exit routing decisions."""
    atr_pct: Optional[float]
    spread_pct: Optional[float]
    atr_bucket: str
    regime: str
    session: str
    # V4 range features
    range_pos: Optional[float] = None
    distance_to_range: Optional[float] = None
    range_edge_dist_atr: Optional[float] = None


def _standardize_atr_pct(atr_pct: float) -> float:
    """Standardize atr_pct using training data statistics."""
    return (atr_pct - ATR_PCT_MEAN) / ATR_PCT_STD


def _router_v1_condition(ctx: ExitRouterContext) -> bool:
    """
    V1 condition helper: checks if V1 rules would allow RULE6A.
    
    V1 rules:
      - atr_pct >= 75%
      - spread_pct <= 40%
      - regime contains "MEDIUM" (handles both "MEDIUM" and "FARM_ASIA_MEDIUM")
    """
    regime_ok = "MEDIUM" in ctx.regime.upper() if ctx.regime else False
    return (
        ctx.atr_pct is not None
        and ctx.atr_pct >= 75.0
        and ctx.spread_pct is not None
        and ctx.spread_pct <= 40.0
        and regime_ok
    )


def hybrid_exit_router_v1(ctx: ExitRouterContext) -> ExitPolicyName:
    """
    HYBRID_ROUTER_V1 - Rule-based router (production baseline).
    
    Simple rules:
      if atr_pct >= 75% and spread_pct <= 40% and regime == MEDIUM:
          RULE6A
      else:
          RULE5
    """
    if _router_v1_condition(ctx):
        return "RULE6A"
    return "RULE5"


def hybrid_exit_router_v2(ctx: ExitRouterContext) -> ExitPolicyName:
    """
    HYBRID_ROUTER_V2 - ML decision tree router (research only, p ~ 0.5 default).
    
    Hardcoded version of decision tree from:
      gx1/analysis/exit_router_models/exit_router_decision_tree_rules.txt
    
    class 0 = RULE5
    class 1 = RULE6A
    
    Tree structure:
    |--- atr_pct <= -0.56
    |   |--- atr_pct <= -1.53
    |   |   |--- class: 0 (RULE5)
    |   |--- atr_pct >  -1.53
    |   |   |--- atr_pct <= -1.21
    |   |   |   |--- class: 1 (RULE6A)
    |   |   |--- atr_pct >  -1.21
    |   |   |   |--- class: 1 (RULE6A)
    |--- atr_pct >  -0.56
    |   |--- atr_pct <= 0.97
    |   |   |--- atr_pct <= 0.85
    |   |   |   |--- class: 0 (RULE5)
    |   |   |--- atr_pct >  0.85
    |   |   |   |--- class: 1 (RULE6A)
    |   |--- atr_pct >  0.97
    |   |   |--- class: 0 (RULE5)
    
    NOTE: This version is for research only. Use V2B for production.
    """
    # Fail-safe: if atr_pct is missing, default to RULE5
    if ctx.atr_pct is None:
        return "RULE5"
    
    # Standardize atr_pct (tree was trained on standardized values)
    atr_std = _standardize_atr_pct(ctx.atr_pct)
    
    # Decision tree logic (hardcoded from export_text output)
    if atr_std <= -0.56:
        # Left branch
        if atr_std <= -1.53:
            return "RULE5"  # class: 0
        else:
            # atr_std > -1.53
            if atr_std <= -1.21:
                return "RULE6A"  # class: 1
            else:
                return "RULE6A"  # class: 1
    else:
        # Right branch: atr_std > -0.56
        if atr_std <= 0.97:
            if atr_std <= 0.85:
                return "RULE5"  # class: 0
            else:
                return "RULE6A"  # class: 1
        else:
            return "RULE5"  # class: 0


def hybrid_exit_router_v2b(ctx: ExitRouterContext) -> ExitPolicyName:
    """
    HYBRID_ROUTER_V2B - Conservative ML router (production candidate).
    
    RULE6A brukes KUN der:
      - V1-ruter ville tillate RULE6A (høy ATR >= 75%, lav spread <= 40%, MEDIUM-regime)
      - OG tree-logikken (V2) sier RULE6A i high-confidence sonen
    
    Dette tilsvarer praktisk p_RULE6A ~ 0.7-0.85 og ~10% RULE6A-andel.
    
    Fra threshold sweep: threshold 0.70 gir:
      - EV/trade: 176.97 bps (best)
      - % RULE6A: 10.0%
      - Uplift vs baseline: +7.10 bps
    
    High-confidence sonen fra treet:
      - atr_std > 0.85 og <= 0.97 (tilsvarer atr_pct ~75-78%)
      - Dette er sweet spot hvor både V1 og V2 er enige
    """
    # Fail-safe: if atr_pct is missing, default to RULE5
    if ctx.atr_pct is None:
        return "RULE5"
    
    # Først: V1 må være fornøyd (grov-reglene)
    if not _router_v1_condition(ctx):
        return "RULE5"
    
    # Så: tre-logikken må indikere RULE6A i high-confidence sonen
    # Standardize atr_pct
    atr_std = _standardize_atr_pct(ctx.atr_pct)
    
    # High-confidence sonen fra treet: atr_std > 0.85 og <= 0.97
    # Dette tilsvarer threshold >= 0.70 i modellen
    # Konvertert til rå atr_pct: ~75.35% til ~78.84%
    if atr_std > 0.85 and atr_std <= 0.97:
        # Både V1 og V2 er enige om RULE6A i denne sonen
        return "RULE6A"
    
    # Ellers faller vi tilbake til RULE5
    return "RULE5"


def hybrid_exit_router_v3(ctx: ExitRouterContext) -> ExitPolicyName:
    """
    HYBRID_ROUTER_V3 - ML decision tree router with range features (V4-C).
    
    Laster trent decision tree fra exit_router_models_v3/exit_router_v3_tree.pkl
    og bruker range features (range_pos, distance_to_range, range_edge_dist_atr).
    
    Tree structure (fra training med range features):
    |--- atr_pct <= 6.81 → RULE5
    |--- atr_pct > 6.81:
    |   |--- atr_pct <= 33.50:
    |   |   |--- spread_pct <= 98.47:
    |   |   |   |--- atr_pct <= 10.65 → RULE6A
    |   |   |   |--- atr_pct > 10.65 → RULE6A
    |   |   |--- spread_pct > 98.47:
    |   |   |   |--- range_edge_dist_atr <= 0.08 → RULE5
    |   |   |   |--- range_edge_dist_atr > 0.08 → RULE6A
    |   |--- atr_pct > 33.50:
    |   |   |--- spread_pct <= 0.50:
    |   |   |   |--- atr_pct <= 61.34 → RULE5
    |   |   |   |--- atr_pct > 61.34 → RULE6A
    |   |   |--- spread_pct > 0.50 → RULE5
    """
    import joblib
    from pathlib import Path
    
    atr = ctx.atr_pct
    spr = ctx.spread_pct
    regime = ctx.regime or "UNKNOWN"
    atr_bucket = ctx.atr_bucket or "UNKNOWN"
    
    # Manglende nøkkelfeatures → alltid RULE5
    if atr is None or spr is None:
        return "RULE5"
    
    # Lazy load trained model (cache after first load)
    if not hasattr(hybrid_exit_router_v3, "_model_cache"):
        model_path = Path(__file__).parent.parent / "analysis" / "exit_router_models_v3" / "exit_router_v3_tree.pkl"
        if model_path.exists():
            try:
                hybrid_exit_router_v3._model_cache = joblib.load(model_path)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"[ROUTER_V3] Failed to load model: {e}, falling back to hardcoded logic")
                hybrid_exit_router_v3._model_cache = None
        else:
            hybrid_exit_router_v3._model_cache = None
    
    model = hybrid_exit_router_v3._model_cache
    
    # If model loaded, use it
    if model is not None:
        try:
            import pandas as pd
            import numpy as np
            
            # Prepare features as DataFrame (matching training format)
            # Features: atr_pct, spread_pct, distance_to_range, range_edge_dist_atr, micro_volatility, volume_stability
            # Categorical: atr_bucket, farm_regime, session
            features = {
                "atr_pct": [atr],
                "spread_pct": [spr if spr is not None else 1.0],
                "distance_to_range": [ctx.distance_to_range if ctx.distance_to_range is not None else 0.5],
                "range_edge_dist_atr": [ctx.range_edge_dist_atr if ctx.range_edge_dist_atr is not None else 0.0],
                "micro_volatility": [0.0],  # Not available in context, use default
                "volume_stability": [0.0],  # Not available in context, use default
                "atr_bucket": [atr_bucket],
                "farm_regime": [regime],
                "session": [ctx.session or "UNKNOWN"],
            }
            
            X = pd.DataFrame(features)
            
            # Predict using pipeline
            prediction = model.predict(X)
            policy_name = prediction[0] if len(prediction) > 0 else "RULE5"
            
            # Map to ExitPolicyName
            if isinstance(policy_name, str):
                if policy_name.upper() == "RULE6A":
                    return "RULE6A"
            return "RULE5"
            
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"[ROUTER_V3] Prediction failed: {e}, falling back to hardcoded logic")
    
    # Fallback: hardcoded tree logic (from training)
    # 1) Ultralav ATR → alltid RULE5
    if atr <= 6.81:
        return "RULE5"
    
    # 2) Moderat ATR (6.81–33.50)
    if atr <= 33.50:
        if spr <= 98.47:
            # Low spread → RULE6A if atr > 10.65
            if atr > 10.65:
                return "RULE6A"
            return "RULE5"
        else:
            # High spread (> 98.47) → use range_edge_dist_atr
            reda = ctx.range_edge_dist_atr if ctx.range_edge_dist_atr is not None else 0.0
            if reda <= 0.08:
                return "RULE5"
            else:
                return "RULE6A"
    
    # 3) Høy ATR (> 33.50)
    if spr <= 0.50:
        if atr > 61.34:
            return "RULE6A"
        return "RULE5"
    
    # Alt annet → RULE5
    return "RULE5"


def hybrid_exit_router_adaptive(
    ctx: ExitRouterContext,
    *,
    high_vol_atr_threshold: float = 65.0,
    low_spread_threshold: float = 40.0,
) -> ExitPolicyName:
    """
    HYBRID_ROUTER_ADAPTIVE - System-level adaptive router switching.
    
    Velger mellom V1 og V2B basert på market conditions:
    
    Routing logic:
      if atr_pct >= high_vol_atr_threshold (default 65%) and spread_pct <= low_spread_threshold (default 40%):
          use HYBRID_ROUTER_V1 (mer aggressiv i høyvolatil markeder)
      else:
          use HYBRID_ROUTER_V2B (konservativ i roligere markeder)
    
    Rationale:
      - Høyvolatil Q2 → bruk V1 (mer aggressiv RULE6A)
      - Roligere Q3/Q4 → bruk V2B (konservativ, kun sweet spots)
      - Nyhetsperioder → fall back til V2B (sikkerhet)
      - Stille Asia → V1 kan være bra (hyperaktiv RULE6A)
    
    Dette gir system-level adaptivitet, ikke bare exit-level adaptivitet.
    """
    # Fail-safe: if atr_pct is missing, default to V2B (conservative)
    if ctx.atr_pct is None:
        return hybrid_exit_router_v2b(ctx)
    
    # Check if we're in high-volatility, low-spread conditions
    # This is where V1 (more aggressive) might be better
    high_vol = ctx.atr_pct >= high_vol_atr_threshold
    low_spread = ctx.spread_pct is not None and ctx.spread_pct <= low_spread_threshold
    
    if high_vol and low_spread:
        # High volatility + low spread → use V1 (more aggressive)
        return hybrid_exit_router_v1(ctx)
    else:
        # Otherwise → use V2B (conservative)
        return hybrid_exit_router_v2b(ctx)

