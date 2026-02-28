#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid exit controller helper.

Chooses between RULE5 (sniper) and RULE6A (adaptive scalper) per trade using
simple regime heuristics so downstream exit plumbing can remain untouched.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import logging

logger = logging.getLogger(__name__)

RULE5_PROFILE = "FARM_EXIT_V2_RULES_A"
RULE6A_PROFILE = "FARM_EXIT_V2_RULES_ADAPTIVE_v1"


@dataclass
class HybridThresholds:
    atr_low_pct: float = 30.0
    atr_high_pct: float = 70.0
    max_spread_pct: float = 60.0


class ExitModeSelector:
    """
    Stateless helper that maps regime metrics to exit profile names.
    """

    def __init__(self, config: Dict[str, float] | None = None, policy_meta: Dict[str, str] | None = None, runner=None) -> None:
        cfg = config or {}
        self.config = cfg  # Store config for adaptive router thresholds
        self.thresholds = HybridThresholds(
            atr_low_pct=float(cfg.get("atr_low_pct", 30.0)),
            atr_high_pct=float(cfg.get("atr_high_pct", 70.0)),
            max_spread_pct=float(cfg.get("max_spread_pct", 60.0)),
        )
        self.name = cfg.get("mode", "RULE5_RULE6A_ATR_SPREAD_V1")
        self.version = cfg.get("version", "HYBRID_ROUTER_V1")
        # V3 guardrail: cutoff for range_edge_dist_atr (default: 1.0)
        self.v3_range_edge_cutoff = float(cfg.get("v3_range_edge_cutoff", 1.0))
        # V3 model path (from config, if provided)
        self.v3_model_path = cfg.get("model_path")
        # PROD_BASELINE mode: check policy meta.role
        policy_meta = policy_meta or {}
        self.prod_baseline = policy_meta.get("role", "").upper() == "PROD_BASELINE"
        self.runner = runner  # Store runner reference for journal access

    def choose_exit_profile(
        self,
        *,
        atr_bps: float | None,
        atr_pct: float | None,
        spread_bps: float | None,
        spread_pct: float | None,
        session: str,
        regime: str,
        range_pos: float | None = None,
        distance_to_range: float | None = None,
        range_edge_dist_atr: float | None = None,
    ) -> str:
        """
        Decide which exit profile to assign to a trade.
        
        Supports router versions:
        - HYBRID_ROUTER_V1: Rule-based (production baseline)
        - HYBRID_ROUTER_V2: ML decision tree-based (research only, p ~ 0.5)
        - HYBRID_ROUTER_V2B: Conservative ML router (production candidate, p ~ 0.7)
        - HYBRID_ROUTER_V3: ML decision tree RAW (conservative, uses raw atr_pct/spread_pct)
        - HYBRID_ROUTER_ADAPTIVE: System-level adaptive switching between V1 and V2B
        """
        # Use V3 router if configured
        if self.version == "HYBRID_ROUTER_V3" or self.version == "V3":
            from gx1.core.hybrid_exit_router import (
                ExitRouterContext,
                hybrid_exit_router_v3,
            )
            
            # Determine atr_bucket from atr_pct
            atr_bucket = "UNKNOWN"
            if atr_pct is not None:
                if atr_pct <= 50.0:
                    atr_bucket = "LOW"
                elif atr_pct <= 75.0:
                    atr_bucket = "MEDIUM"
                else:
                    atr_bucket = "HIGH"
            
            ctx = ExitRouterContext(
                atr_pct=atr_pct,
                spread_pct=spread_pct,
                atr_bucket=atr_bucket,
                regime=regime,
                session=session,
                range_pos=range_pos,
                distance_to_range=distance_to_range,
                range_edge_dist_atr=range_edge_dist_atr,
                model_path=self.v3_model_path,
                prod_baseline=self.prod_baseline,
            )
            
            # Use V3 router
            raw_prediction = hybrid_exit_router_v3(ctx)
            policy = raw_prediction
            
            # V3 guardrail: Override RULE6A if range_edge_dist_atr >= cutoff
            guardrail_override = False
            if policy == "RULE6A":
                reda = range_edge_dist_atr if range_edge_dist_atr is not None else float("inf")
                if reda >= self.v3_range_edge_cutoff:
                    policy = "RULE5"
                    guardrail_override = True
                    logger.debug(
                        "[HYBRID_EXIT] V3 guardrail: RULE6A overridden to RULE5 (range_edge_dist_atr=%.3f >= cutoff=%.3f)",
                        reda,
                        self.v3_range_edge_cutoff,
                    )
            
            # Log router decision to trade journal (structured)
            if hasattr(self, "runner") and hasattr(self.runner, "trade_journal") and self.runner.trade_journal:
                try:
                    from gx1.monitoring.trade_journal import EVENT_ROUTER_DECISION, EVENT_GUARDRAIL_OVERRIDE
                    
                    # Get trade_id from context (if available)
                    trade_id = getattr(ctx, "trade_id", None)
                    if not trade_id:
                        # Try to get from runner's open trades (last created)
                        if hasattr(self.runner, "open_trades") and self.runner.open_trades:
                            trade_id = self.runner.open_trades[-1].trade_id
                    
                    if trade_id:
                        # Log structured router decision
                        self.runner.trade_journal.log_router_decision(
                            trade_id=trade_id,
                            router_version=self.version,
                            router_raw_decision=raw_prediction,
                            final_exit_profile=policy,
                            router_model_hash=self.router_sha256 if hasattr(self, "router_sha256") else None,
                            router_features_used={
                                "atr_pct": atr_pct,
                                "spread_pct": spread_pct,
                                "atr_bucket": atr_bucket,
                                "regime": regime,
                                "session": session,
                                "range_pos": range_pos,
                                "distance_to_range": distance_to_range,
                                "range_edge_dist_atr": range_edge_dist_atr,
                            },
                            guardrail_applied=guardrail_override,
                            guardrail_reason="range_edge_dist_atr >= cutoff" if guardrail_override else None,
                            guardrail_cutoff=self.v3_range_edge_cutoff if guardrail_override else None,
                            range_edge_dist_atr=range_edge_dist_atr,
                        )
                    
                    # Log router decision (backward compatibility JSONL)
                    self.runner.trade_journal.log(
                        EVENT_ROUTER_DECISION,
                        {
                            "router_version": self.version,
                            "model_path": str(self.v3_model_path) if self.v3_model_path else None,
                            "ctx_features": {
                                "atr_pct": ctx.atr_pct,
                                "spread_pct": ctx.spread_pct,
                                "atr_bucket": ctx.atr_bucket,
                                "regime": ctx.regime,
                                "session": ctx.session,
                                "range_pos": ctx.range_pos,
                                "distance_to_range": ctx.distance_to_range,
                                "range_edge_dist_atr": ctx.range_edge_dist_atr,
                            },
                            "raw_prediction": raw_prediction,
                            "final_decision": policy,
                            "guardrail_cutoff": self.v3_range_edge_cutoff,
                            "range_edge_dist_atr": range_edge_dist_atr,
                        },
                    )
                    
                    # Log guardrail override if it happened
                    if guardrail_override:
                        self.runner.trade_journal.log(
                            EVENT_GUARDRAIL_OVERRIDE,
                            {
                                "raw_prediction": raw_prediction,
                                "final_decision": policy,
                                "range_edge_dist_atr": range_edge_dist_atr,
                                "cutoff": self.v3_range_edge_cutoff,
                            },
                        )
                except Exception as e:
                    logger.warning("[TRADE_JOURNAL] Failed to log router decision: %s", e)
            
            profile = RULE6A_PROFILE if policy == "RULE6A" else RULE5_PROFILE
            
            logger.debug(
                "[HYBRID_EXIT] mode=%s version=%s session=%s regime=%s atr_bps=%.2f atr_pct=%.1f spread_bps=%.2f spread_pct=%.1f range_edge_dist_atr=%.3f -> %s (V3)",
                self.name,
                self.version,
                session,
                regime,
                atr_bps if atr_bps is not None else float("nan"),
                atr_pct if atr_pct is not None else float("nan"),
                spread_bps if spread_bps is not None else float("nan"),
                spread_pct if spread_pct is not None else float("nan"),
                range_edge_dist_atr if range_edge_dist_atr is not None else float("nan"),
                profile,
            )
            return profile
        
        # Use adaptive router if configured
        if self.version == "HYBRID_ROUTER_ADAPTIVE":
            from gx1.core.hybrid_exit_router import (
                ExitRouterContext,
                hybrid_exit_router_adaptive,
            )
            
            # Determine atr_bucket from atr_pct
            atr_bucket = "UNKNOWN"
            if atr_pct is not None:
                if atr_pct <= 50.0:
                    atr_bucket = "LOW"
                elif atr_pct <= 75.0:
                    atr_bucket = "MEDIUM"
                else:
                    atr_bucket = "HIGH"
            
            ctx = ExitRouterContext(
                atr_pct=atr_pct,
                spread_pct=spread_pct,
                atr_bucket=atr_bucket,
                regime=regime,
                session=session,
            )
            
            # Get adaptive thresholds from config (with defaults)
            high_vol_threshold = float(self.config.get("high_vol_atr_threshold", 65.0))
            low_spread_threshold = float(self.config.get("low_spread_threshold", 40.0))
            
            # Use adaptive router
            policy = hybrid_exit_router_adaptive(
                ctx,
                high_vol_atr_threshold=high_vol_threshold,
                low_spread_threshold=low_spread_threshold,
            )
            
            profile = RULE6A_PROFILE if policy == "RULE6A" else RULE5_PROFILE
            
            logger.debug(
                "[HYBRID_EXIT] mode=%s version=%s session=%s regime=%s atr_bps=%.2f atr_pct=%.1f spread_bps=%.2f spread_pct=%.1f -> %s (adaptive)",
                self.name,
                self.version,
                session,
                regime,
                atr_bps if atr_bps is not None else float("nan"),
                atr_pct if atr_pct is not None else float("nan"),
                spread_bps if spread_bps is not None else float("nan"),
                spread_pct if spread_pct is not None else float("nan"),
                profile,
            )
            return profile
        
        # Use V2 or V2B router if configured
        if self.version in ("HYBRID_ROUTER_V2", "HYBRID_ROUTER_V2B"):
            from gx1.core.hybrid_exit_router import (
                ExitRouterContext,
                hybrid_exit_router_v2,
                hybrid_exit_router_v2b,
            )
            
            # Determine atr_bucket from atr_pct
            atr_bucket = "UNKNOWN"
            if atr_pct is not None:
                if atr_pct <= 50.0:
                    atr_bucket = "LOW"
                elif atr_pct <= 75.0:
                    atr_bucket = "MEDIUM"
                else:
                    atr_bucket = "HIGH"
            
            ctx = ExitRouterContext(
                atr_pct=atr_pct,
                spread_pct=spread_pct,
                atr_bucket=atr_bucket,
                regime=regime,
                session=session,
            )
            
            # Choose router function based on version
            if self.version == "HYBRID_ROUTER_V2B":
                policy = hybrid_exit_router_v2b(ctx)
            else:  # HYBRID_ROUTER_V2
                policy = hybrid_exit_router_v2(ctx)
            
            profile = RULE6A_PROFILE if policy == "RULE6A" else RULE5_PROFILE
            
            logger.debug(
                "[HYBRID_EXIT] mode=%s version=%s session=%s regime=%s atr_bps=%.2f atr_pct=%.1f spread_bps=%.2f spread_pct=%.1f -> %s",
                self.name,
                self.version,
                session,
                regime,
                atr_bps if atr_bps is not None else float("nan"),
                atr_pct if atr_pct is not None else float("nan"),
                spread_bps if spread_bps is not None else float("nan"),
                spread_pct if spread_pct is not None else float("nan"),
                profile,
            )
            return profile
        
        # V1 router (original rule-based logic)
        # Only hybridise inside FARM Asia regimes; fall back to RULE5 otherwise
        if session != "ASIA" or regime not in {"FARM_ASIA_LOW", "FARM_ASIA_MEDIUM"}:
            return RULE5_PROFILE

        atr_pct_val = 50.0 if atr_pct is None else atr_pct
        spread_pct_val = 50.0 if spread_pct is None else spread_pct

        if (
            self.thresholds.atr_low_pct
            <= atr_pct_val
            <= self.thresholds.atr_high_pct
            and spread_pct_val <= self.thresholds.max_spread_pct
        ):
            profile = RULE6A_PROFILE
        else:
            profile = RULE5_PROFILE

        logger.debug(
            "[HYBRID_EXIT] mode=%s version=%s session=%s regime=%s atr_bps=%.2f atr_pct=%.1f spread_bps=%.2f spread_pct=%.1f -> %s",
            self.name,
            self.version,
            session,
            regime,
            atr_bps if atr_bps is not None else float("nan"),
            atr_pct_val,
            spread_bps if spread_bps is not None else float("nan"),
            spread_pct_val,
            profile,
        )
        return profile
