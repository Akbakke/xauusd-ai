#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OVERLAP Overlay - Session-specific controls for OVERLAP trades only

This module implements three types of OVERLAP overlays:
A) OVERLAP threshold override (baseline + delta)
B) OVERLAP cost-gate (veto based on spread_bps and atr_bps)
C) Partial overlap window (time-based filtering for OVERLAP)

These overlays ONLY affect OVERLAP session trades. EU/US/ASIA are unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any
from pathlib import Path
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class OverlapOverlayConfig:
    """Configuration for OVERLAP overlays."""
    
    # A) Threshold override
    overlap_threshold_delta: Optional[float] = None  # Added to baseline threshold (e.g., +0.02)
    
    # B) Cost-gate
    overlap_cost_gate_enabled: bool = False
    overlap_cost_gate_spread_bps: Optional[float] = None  # Veto if spread_bps > this
    overlap_cost_gate_atr_bps: Optional[float] = None  # Veto if atr_bps > this
    
    # C) Partial overlap window
    overlap_window_mode: Optional[str] = None  # "W0" (full), "W1" (tight), "W2" (mid)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for RUN_IDENTITY logging."""
        return {
            "overlap_threshold_delta": self.overlap_threshold_delta,
            "overlap_cost_gate_enabled": self.overlap_cost_gate_enabled,
            "overlap_cost_gate_spread_bps": self.overlap_cost_gate_spread_bps,
            "overlap_cost_gate_atr_bps": self.overlap_cost_gate_atr_bps,
            "overlap_window_mode": self.overlap_window_mode,
        }


def load_overlap_overlay_config(overlay_config_path: Optional[Path]) -> OverlapOverlayConfig:
    """
    Load OVERLAP overlay config from YAML file.
    
    Args:
        overlay_config_path: Path to YAML config file (optional)
    
    Returns:
        OverlapOverlayConfig instance
    """
    import yaml
    
    if overlay_config_path is None or not overlay_config_path.exists():
        # Return default (all overlays disabled)
        return OverlapOverlayConfig()
    
    try:
        with open(overlay_config_path, "r") as f:
            config_dict = yaml.safe_load(f) or {}
        
        overlay_dict = config_dict.get("overlap_overlay", {})
        
        return OverlapOverlayConfig(
            overlap_threshold_delta=overlay_dict.get("threshold_delta"),
            overlap_cost_gate_enabled=overlay_dict.get("cost_gate", {}).get("enabled", False),
            overlap_cost_gate_spread_bps=overlay_dict.get("cost_gate", {}).get("spread_bps"),
            overlap_cost_gate_atr_bps=overlay_dict.get("cost_gate", {}).get("atr_bps"),
            overlap_window_mode=overlay_dict.get("window_mode"),
        )
    except Exception as e:
        log.error(f"[OVERLAP_OVERLAY] Failed to load config from {overlay_config_path}: {e}")
        return OverlapOverlayConfig()


def apply_overlap_threshold_override(
    baseline_threshold: float,
    session: str,
    config: OverlapOverlayConfig,
) -> float:
    """
    Apply OVERLAP threshold override (A).
    
    Args:
        baseline_threshold: Baseline threshold from policy
        session: Current session (EU, US, OVERLAP, ASIA)
        config: Overlap overlay config
    
    Returns:
        Adjusted threshold (baseline + delta for OVERLAP, unchanged for others)
    """
    if session != "OVERLAP":
        return baseline_threshold
    
    if config.overlap_threshold_delta is None:
        return baseline_threshold
    
    adjusted = baseline_threshold + config.overlap_threshold_delta
    # Clamp to reasonable range [0.0, 1.0]
    adjusted = max(0.0, min(1.0, adjusted))
    
    return adjusted


def check_overlap_cost_gate(
    session: str,
    spread_bps: Optional[float],
    atr_bps: Optional[float],
    config: OverlapOverlayConfig,
) -> Tuple[bool, Optional[str]]:
    """
    Check OVERLAP cost-gate (B) - veto if spread/ATR too high.
    
    Args:
        session: Current session
        spread_bps: Spread in basis points (required if cost-gate enabled)
        atr_bps: ATR in basis points (required if cost-gate enabled)
        config: Overlap overlay config
    
    Returns:
        Tuple of (pass, reason)
        - pass=True: Entry allowed
        - pass=False: Entry vetoed (reason explains why)
    """
    if session != "OVERLAP":
        return (True, None)
    
    if not config.overlap_cost_gate_enabled:
        return (True, None)
    
    # Hard fail if required fields missing
    if spread_bps is None:
        raise RuntimeError(
            "[OVERLAP_COST_GATE] spread_bps is None but cost-gate is enabled. "
            "spread_bps must be available for OVERLAP cost-gate."
        )
    
    if atr_bps is None:
        raise RuntimeError(
            "[OVERLAP_COST_GATE] atr_bps is None but cost-gate is enabled. "
            "atr_bps must be available for OVERLAP cost-gate."
        )
    
    # Check spread threshold
    if config.overlap_cost_gate_spread_bps is not None:
        if spread_bps > config.overlap_cost_gate_spread_bps:
            return (False, f"spread_bps={spread_bps:.2f} > {config.overlap_cost_gate_spread_bps:.2f}")
    
    # Check ATR threshold
    if config.overlap_cost_gate_atr_bps is not None:
        if atr_bps > config.overlap_cost_gate_atr_bps:
            return (False, f"atr_bps={atr_bps:.2f} > {config.overlap_cost_gate_atr_bps:.2f}")
    
    return (True, None)


def check_overlap_window(
    session: str,
    entry_time: pd.Timestamp,
    config: OverlapOverlayConfig,
) -> Tuple[bool, Optional[str]]:
    """
    Check partial overlap window (C) - time-based filtering for OVERLAP.
    
    Args:
        session: Current session
        entry_time: Entry timestamp (timezone-aware, UTC)
        config: Overlap overlay config
    
    Returns:
        Tuple of (pass, reason)
        - pass=True: Entry allowed
        - pass=False: Entry vetoed (reason explains why)
    """
    if session != "OVERLAP":
        return (True, None)
    
    if config.overlap_window_mode is None:
        return (True, None)
    
    # Convert to UTC if needed
    if entry_time.tzinfo is None:
        entry_time = entry_time.tz_localize("UTC")
    else:
        entry_time = entry_time.tz_convert("UTC")
    
    hour = entry_time.hour
    minute = entry_time.minute
    
    # Window definitions (UTC):
    # W0: full overlap (12:00-16:00 UTC) - no filtering
    # W1: tight (13:30-14:30 UTC)
    # W2: mid (14:00-15:30 UTC)
    
    if config.overlap_window_mode == "W0":
        # Full overlap - no filtering
        return (True, None)
    elif config.overlap_window_mode == "W1":
        # Tight: 13:30-14:30 UTC
        if hour == 13 and minute >= 30:
            return (True, None)
        elif hour == 14 and minute < 30:
            return (True, None)
        else:
            return (False, f"outside W1 window (13:30-14:30 UTC), current={hour:02d}:{minute:02d}")
    elif config.overlap_window_mode == "W2":
        # Mid: 14:00-15:30 UTC
        if hour == 14:
            return (True, None)
        elif hour == 15 and minute < 30:
            return (True, None)
        else:
            return (False, f"outside W2 window (14:00-15:30 UTC), current={hour:02d}:{minute:02d}")
    else:
        log.warning(f"[OVERLAP_WINDOW] Unknown window mode: {config.overlap_window_mode}, allowing entry")
        return (True, None)


def generate_variant_id(config: OverlapOverlayConfig) -> str:
    """
    Generate deterministic variant ID from config.
    
    Format: THR+0.02__COST_S15_A100__WIN_W1
    """
    parts = []
    
    # Threshold delta
    if config.overlap_threshold_delta is not None:
        delta_str = f"+{config.overlap_threshold_delta:.2f}".replace("+", "PLUS").replace("-", "MINUS")
        parts.append(f"THR{delta_str}")
    else:
        parts.append("THR_BASE")
    
    # Cost-gate
    if config.overlap_cost_gate_enabled:
        spread_str = f"S{int(config.overlap_cost_gate_spread_bps)}" if config.overlap_cost_gate_spread_bps else "S_"
        atr_str = f"A{int(config.overlap_cost_gate_atr_bps)}" if config.overlap_cost_gate_atr_bps else "A_"
        parts.append(f"COST_{spread_str}_{atr_str}")
    else:
        parts.append("COST_OFF")
    
    # Window
    if config.overlap_window_mode:
        parts.append(f"WIN_{config.overlap_window_mode}")
    else:
        parts.append("WIN_W0")
    
    return "__".join(parts)
