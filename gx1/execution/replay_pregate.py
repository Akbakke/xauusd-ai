"""
Replay PreGate - Early skip before expensive feature building.

DEL A: Skip bars that will definitely result in NO-TRADE before building features.
Uses only cheap inputs (no pandas, no HTF, no rolling).

This is a replay-only optimization. Does not affect trading logic.
"""

import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

log = logging.getLogger(__name__)


def replay_pregate_should_skip(
    ts: datetime,
    session: Optional[str] = None,
    warmup_ready: bool = False,
    degraded: bool = False,
    spread_bps: Optional[float] = None,
    atr_bps: Optional[float] = None,
    in_scope: bool = True,
    policy_config: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    """
    Determine if bar should be skipped before feature building.
    
    Uses only cheap inputs (no pandas operations, no HTF, no rolling).
    
    Args:
        ts: Current timestamp (datetime)
        session: Session tag (EU/US/OVERLAP/ASIA) - optional, will infer if missing
        warmup_ready: Whether warmup is ready (from runner state)
        degraded: Whether degraded warmup is allowed (from runner state)
        spread_bps: Spread in basis points (from row data) - optional
        atr_bps: ATR in basis points (from row data or cheap proxy) - optional
        in_scope: Whether bar is in scope (from eligibility checks) - optional
        policy_config: Policy config dict with replay_pregate section - optional
    
    Returns:
        (should_skip: bool, reason: str)
        - If should_skip=True: reason contains skip reason (for attribution)
        - If should_skip=False: reason is empty string (continue to feature build)
    """
    # Default: conservative (don't skip if data missing)
    if policy_config is None:
        policy_config = {}
    
    # Get pregate config (defaults to conservative if not configured)
    pregate_cfg = policy_config.get("replay_pregate", {})
    if not isinstance(pregate_cfg, dict):
        pregate_cfg = {}
    
    # Default: disabled (conservative - don't skip unless explicitly enabled)
    enabled = pregate_cfg.get("enabled", False)
    if not enabled:
        return False, ""
    
    # DEL A1: Infer session if missing (cheap operation - just time-based)
    if session is None and ts is not None:
        try:
            # Use hour of day (UTC) to infer session (cheap, no pandas)
            hour = ts.hour if hasattr(ts, "hour") else ts.hour if hasattr(ts, "hour") else None
            if hour is not None:
                # EU: 7-15, US: 13-20, OVERLAP: 7-20 overlap, ASIA: 0-7, 20-24
                if 7 <= hour <= 15:
                    session = "EU"
                elif 13 <= hour <= 20:
                    session = "US"
                elif 7 <= hour <= 20:
                    session = "OVERLAP"
                else:
                    session = "ASIA"
        except Exception:
            pass  # Keep session as None if inference fails
    
    # DEL A1: Check allowed sessions (cheap - just string comparison)
    allowed_sessions = pregate_cfg.get("allow_sessions", [])
    if allowed_sessions and session:
        if session not in allowed_sessions:
            return True, f"replay_pregate_skip:session_not_allowed:{session}"
    
    # DEL A1: Check warmup requirement (cheap - just boolean)
    require_warmup_ready = pregate_cfg.get("require_warmup_ready", True)
    if require_warmup_ready and not warmup_ready:
        # Allow degraded warmup if configured
        allow_degraded = pregate_cfg.get("allow_degraded", False)
        if not (allow_degraded and degraded):
            return True, "replay_pregate_skip:warmup_not_ready"
    
    # DEL A1: Check in_scope (cheap - just boolean)
    if not in_scope:
        return True, "replay_pregate_skip:out_of_scope"
    
    # DEL A1: Check spread cap (cheap - just float comparison)
    max_spread_bps = pregate_cfg.get("max_spread_bps")
    if max_spread_bps is not None and spread_bps is not None:
        try:
            spread_val = float(spread_bps)
            if spread_val > float(max_spread_bps):
                return True, f"replay_pregate_skip:spread_too_high:{spread_val:.2f}"
        except (ValueError, TypeError):
            pass  # Skip check if conversion fails (conservative)
    
    # DEL A1: Check ATR bounds (cheap - just float comparison)
    min_atr_bps = pregate_cfg.get("min_atr_bps")
    max_atr_bps = pregate_cfg.get("max_atr_bps")
    if atr_bps is not None:
        try:
            atr_val = float(atr_bps)
            if min_atr_bps is not None and atr_val < float(min_atr_bps):
                return True, f"replay_pregate_skip:atr_too_low:{atr_val:.2f}"
            if max_atr_bps is not None and atr_val > float(max_atr_bps):
                return True, f"replay_pregate_skip:atr_too_high:{atr_val:.2f}"
        except (ValueError, TypeError):
            pass  # Skip check if conversion fails (conservative)
    
    # All checks passed - don't skip
    return False, ""
