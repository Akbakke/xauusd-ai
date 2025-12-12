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
    HYBRID_ROUTER_V3 - ML decision tree router (RAW, konservativ variant).
    
    Basert på V3 decision tree trent på rå atr_pct / spread_pct.
    Konservativ variant: vi klipper bort de mest tvilsomme high-spread RULE6A-grenene.
    
    Tree structure (fra training):
    |--- atr_pct <= 6.81 → RULE5
    |--- 6.81 < atr_pct <= 31.34:
    |   |--- spread_pct <= 78.10:
    |   |   |--- atr_pct <= 18.75 → RULE5
    |   |   |--- atr_pct > 18.75 → RULE6A
    |   |--- spread_pct > 78.10:
    |   |   |--- spread_pct <= 99.26 → RULE6A (KUTTET UT - for høy spread)
    |   |   |--- spread_pct > 99.26 → RULE5
    |--- atr_pct > 31.34:
    |   |--- spread_pct <= 0.50:
    |   |   |--- atr_pct <= 61.03 → RULE5
    |   |   |--- atr_pct > 61.03 → RULE6A
    |   |--- spread_pct > 0.50 → RULE5
    
    Konservative endringer:
    - Kutter high-spread RULE6A (spread_pct > 78.10)
    - Krever MEDIUM-regime for RULE6A
    - Beholder RULE5 som default
    """
    atr = ctx.atr_pct
    spr = ctx.spread_pct
    regime = ctx.regime or "UNKNOWN"
    
    # Manglende nøkkelfeatures → alltid RULE5
    if atr is None or spr is None:
        return "RULE5"
    
    # Vi tillater RULE6A kun i MEDIUM-regime (der vi vet det fungerer best)
    is_medium_regime = "MEDIUM" in regime.upper()
    
    # 1) Ultralav ATR → alltid RULE5
    if atr <= 6.81:
        return "RULE5"
    
    # 2) Moderat ATR (6.81–31.34)
    if atr <= 31.34:
        # Spreaddritt: over ~80-percentilen → vi tvinger RULE5
        if spr > 78.10:
            return "RULE5"
        
        # spread ok-ish og atr i øvre del av medium → RULE6A *kun* i MEDIUM regime
        if atr > 18.75 and spr <= 78.10 and is_medium_regime:
            return "RULE6A"
        
        # Resten → RULE5
        return "RULE5"
    
    # 3) Høy ATR (>31.34)
    # Her tillater vi RULE6A bare når spread er ekstremt lav og ATR veldig høy
    if spr <= 0.50 and atr > 61.03 and is_medium_regime:
        return "RULE6A"
    
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

