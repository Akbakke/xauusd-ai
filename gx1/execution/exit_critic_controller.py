#!/usr/bin/env python3
"""
exit_critic_controller.py

Controller for ExitCritic V1 runtime-integrasjon.
Brukes i exit_manager.py for å evaluere ExitCritic på aktive trades.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from gx1.rl.exit_critic.integrate_exit_critic_runtime_v1 import (
    ExitCriticRuntimeV1,
    ACTION_EXIT_NOW,
    ACTION_SCALP_PROFIT,
    ACTION_HOLD,
)
from gx1.rl.exit_critic.exit_snapshot_v1 import make_exit_snapshot

log = logging.getLogger(__name__)


class ExitCriticController:
    """
    Controller for ExitCritic V1.
    
    Håndterer:
    - Initialisering av ExitCritic fra config
    - Guard-checks (min_bars_held, pnl-thresholds, session/regime filters)
    - Snapshot-bygging og evaluering
    - Returnerer ACTION_* eller None
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: ExitCritic config fra policy YAML (exit_critic-seksjonen)
        """
        self.config = config or {}
        self.enabled = bool(self.config.get("enabled", False))
        self.critic: Optional[ExitCriticRuntimeV1] = None
        
        if not self.enabled:
            log.info("[EXIT_CRITIC] Disabled in config")
            return
        
        # Load model
        model_path = self.config.get("model_path", "models/exit_critic/exit_critic_xgb_v1.json")
        metadata_path = self.config.get("metadata_path", "models/exit_critic/exit_critic_xgb_v1.meta.json")
        
        exit_now_threshold = float(self.config.get("exit_now_threshold", 0.60))
        scalp_threshold = float(self.config.get("scalp_threshold", 0.40))
        
        try:
            self.critic = ExitCriticRuntimeV1(
                model_path=model_path,
                metadata_path=metadata_path,
                exit_now_threshold=exit_now_threshold,
                scalp_threshold=scalp_threshold,
            )
            log.info("[EXIT_CRITIC] Initialized successfully")
        except Exception as e:
            log.error("[EXIT_CRITIC] Failed to initialize: %s", e, exc_info=True)
            self.enabled = False
            self.critic = None
    
    def maybe_apply_exit_critic(
        self,
        trade_snapshot: Dict[str, Any],
        bars_held: int,
        current_pnl_bps: float,
        session: Optional[str] = None,
        vol_regime: Optional[str] = None,
        trend_regime: Optional[str] = None,
    ) -> Optional[str]:
        """
        Evaluerer ExitCritic på trade-snapshot.
        
        Args:
            trade_snapshot: Trade snapshot dict (fra build_live_exit_snapshot eller make_exit_snapshot)
            bars_held: Antall bars trade har vært åpen
            current_pnl_bps: Nåværende PnL i bps
            session: Session (EU/OVERLAP/US/ASIA)
            vol_regime: Volatility regime (LOW/MEDIUM/HIGH/EXTREME)
            trend_regime: Trend regime (TREND_UP/TREND_NEUTRAL/TREND_DOWN)
        
        Returns:
            ACTION_EXIT_NOW, ACTION_SCALP_PROFIT, eller None (HOLD)
        """
        if not self.enabled or self.critic is None:
            return None
        
        # Guard checks
        guards = self.config.get("guards", {})
        
        # Min bars held
        min_bars_held = guards.get("min_bars_held", 5)
        if bars_held < min_bars_held:
            return None
        
        # PnL thresholds
        apply_for_loss_leq = guards.get("apply_for_loss_leq", -40.0)
        apply_for_profit_ge = guards.get("apply_for_profit_ge", 40.0)
        
        # Session filter
        allowed_sessions = guards.get("allowed_sessions", ["EU", "OVERLAP", "US"])
        if session and session not in allowed_sessions:
            return None
        
        # Vol regime filter
        allowed_vol_regimes = guards.get("allowed_vol_regimes", ["LOW", "MEDIUM", "HIGH"])
        if vol_regime and vol_regime not in allowed_vol_regimes:
            return None
        
        # Trend regime filter
        allowed_trend_regimes = guards.get("allowed_trend_regimes", ["TREND_UP", "TREND_NEUTRAL", "TREND_DOWN"])
        if trend_regime and trend_regime not in allowed_trend_regimes:
            return None
        
        # Build exit snapshot for ExitCritic
        try:
            # Convert trade_snapshot to format expected by make_exit_snapshot
            exit_snapshot_dict = {
                "pnl_bps": current_pnl_bps,
                "mfe_bps": trade_snapshot.get("mfe_so_far", trade_snapshot.get("mfe_bps", 0.0)),
                "mae_bps": trade_snapshot.get("mae_so_far", trade_snapshot.get("mae_bps", 0.0)),
                "bars_held": bars_held,
                "atr_bps": trade_snapshot.get("atr_bps", 0.0),
                "spread_bps": trade_snapshot.get("spread_bps", 0.0),
                "session": session or trade_snapshot.get("session_tag", "UNKNOWN"),
                "trend_regime": trend_regime or trade_snapshot.get("trend_regime", "UNKNOWN"),
                "vol_regime": vol_regime or trade_snapshot.get("vol_bucket", "UNKNOWN"),
                "p_long": trade_snapshot.get("p_long", 0.0),
                "entry_time": trade_snapshot.get("entry_time"),
            }
            
            # Use make_exit_snapshot to ensure feature consistency
            critic_snapshot = make_exit_snapshot(exit_snapshot_dict)
            
            # Evaluate ExitCritic
            action = self.critic.evaluate(critic_snapshot)
            
            # Apply PnL thresholds
            if action == ACTION_EXIT_NOW:
                if current_pnl_bps > apply_for_loss_leq:
                    # EXIT_NOW only for trades with significant loss
                    return None
            elif action == ACTION_SCALP_PROFIT:
                if current_pnl_bps < apply_for_profit_ge:
                    # SCALP_PROFIT only for trades with significant profit
                    return None
            
            log.debug(
                "[EXIT_CRITIC] Trade %s: action=%s, pnl=%.2f bps, bars=%d",
                trade_snapshot.get("entry_id", "unknown"),
                action,
                current_pnl_bps,
                bars_held,
            )
            
            return action
            
        except Exception as e:
            log.warning("[EXIT_CRITIC] Failed to evaluate: %s", e, exc_info=True)
            return None

