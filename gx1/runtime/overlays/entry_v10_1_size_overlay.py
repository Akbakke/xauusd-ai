#!/usr/bin/env python3
"""
ENTRY_V10.1 Size Overlay (OFFLINE ONLY)

Applies aggressive sizing based on edge-buckets computed from label quality analysis.
Uses expected_bps per quantile bin to determine size multiplier.

⚠️  OFFLINE ONLY: This overlay is for replay testing only. Do not use for live trading.

Design:
    - NO dummy values: If no bin found or insufficient data → fallback to 1.0 (policy decision, not model prediction)
    - Direct mapping from expected_bps to size_multiplier (no smoothing/interpolation)
    - Caps enforced: max_size_multiplier, max_risk_bps
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

log = logging.getLogger(__name__)


class EntryV10_1SizeOverlay:
    """
    ENTRY_V10.1 size overlay based on edge-buckets.
    
    ⚠️  OFFLINE ONLY - for replay testing only.
    """
    
    def __init__(
        self,
        edge_buckets_path: Path,
        max_size_multiplier: float = 2.0,
        max_risk_bps: float = 100.0,
    ):
        """
        Initialize overlay.
        
        Args:
            edge_buckets_path: Path to edge buckets JSON file
            max_size_multiplier: Maximum size multiplier cap (default: 2.0)
            max_risk_bps: Maximum risk in bps (sl_bps * size_multiplier) (default: 100.0)
        """
        self.edge_buckets_path = Path(edge_buckets_path)
        self.max_size_multiplier = max_size_multiplier
        self.max_risk_bps = max_risk_bps
        
        if not self.edge_buckets_path.exists():
            raise FileNotFoundError(f"Edge buckets file not found: {self.edge_buckets_path}")
        
        # Load edge buckets
        with open(self.edge_buckets_path, "r", encoding="utf-8") as f:
            self.edge_buckets = json.load(f)
        
        log.info(f"[ENTRY_V10_1_SIZE_OVERLAY] Loaded edge buckets from {self.edge_buckets_path}")
        log.info(f"[ENTRY_V10_1_SIZE_OVERLAY] Max size multiplier: {self.max_size_multiplier}")
        log.info(f"[ENTRY_V10_1_SIZE_OVERLAY] Max risk bps: {self.max_risk_bps}")
    
    def _find_bin(
        self,
        p_long_v10_1: float,
        session: str,
        regime: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Find edge bucket bin for given p_long, session, and regime.
        
        Returns:
            Bin dict with p_min, p_max, expected_bps, n_trades, or None if not found
        """
        session_buckets = self.edge_buckets.get(session, {})
        regime_buckets = session_buckets.get(regime, [])
        
        if not regime_buckets:
            return None
        
        # Find bin where p_min <= p_long_v10_1 < p_max
        for bin_data in regime_buckets:
            p_min = bin_data.get("p_min", 0.0)
            p_max = bin_data.get("p_max", 1.0)
            
            if p_min <= p_long_v10_1 < p_max:
                return bin_data
        
        # Check if p_long is exactly at max boundary (include it)
        if regime_buckets:
            last_bin = regime_buckets[-1]
            if p_long_v10_1 == last_bin.get("p_max", 1.0):
                return last_bin
        
        return None
    
    def _expected_bps_to_multiplier(self, expected_bps: float) -> float:
        """
        Map expected_bps to size multiplier.
        
        Hardcoded mapping (first version):
        - expected_bps < 0 → 0.0 (drop trade)
        - 0 <= expected_bps < 2 → 0.5
        - 2 <= expected_bps < 5 → 1.0
        - 5 <= expected_bps < 8 → 1.5
        - expected_bps >= 8 → 2.0
        """
        if expected_bps < 0:
            return 0.0
        elif expected_bps < 2:
            return 0.5
        elif expected_bps < 5:
            return 1.0
        elif expected_bps < 8:
            return 1.5
        else:
            return 2.0
    
    def get_size_multiplier(
        self,
        p_long_v10_1: float,
        session: str,
        regime: str,
        sl_bps: float,
        base_units: float,
    ) -> Dict[str, Any]:
        """
        Get size multiplier based on edge-buckets.
        
        Args:
            p_long_v10_1: V10.1 p_long prediction
            session: Trading session (EU/OVERLAP/US)
            regime: Regime string (e.g., "UP×LOW", "NEUTRAL×MEDIUM")
            sl_bps: Stop loss in bps
            base_units: Base units before overlay
        
        Returns:
            Dict with:
            - size_multiplier: float
            - reason: str (e.g., "low_edge", "medium_edge", "high_edge", "no_data_fallback")
            - expected_bps: float (from bin, or None)
            - bin_data: dict (bin data, or None)
            - caps_applied: dict (which caps were applied, if any)
        """
        # Find bin
        bin_data = self._find_bin(p_long_v10_1, session, regime)
        
        if bin_data is None:
            # No bin found - fallback to 1.0 (policy decision, not dummy model prediction)
            return {
                "size_multiplier": 1.0,
                "reason": "no_data_fallback",
                "expected_bps": None,
                "bin_data": None,
                "caps_applied": {},
            }
        
        expected_bps = bin_data.get("expected_bps", 0.0)
        
        # Map expected_bps to multiplier
        raw_multiplier = self._expected_bps_to_multiplier(expected_bps)
        
        # Apply caps
        caps_applied = {}
        
        # Cap 1: max_size_multiplier
        if raw_multiplier > self.max_size_multiplier:
            caps_applied["max_size_multiplier"] = {
                "raw": raw_multiplier,
                "capped": self.max_size_multiplier,
            }
            raw_multiplier = self.max_size_multiplier
        
        # Cap 2: max_risk_bps (sl_bps * multiplier)
        risk_bps = abs(sl_bps) * raw_multiplier
        if risk_bps > self.max_risk_bps:
            # Adjust multiplier to respect max_risk_bps
            adjusted_multiplier = self.max_risk_bps / abs(sl_bps) if sl_bps != 0 else raw_multiplier
            caps_applied["max_risk_bps"] = {
                "raw_multiplier": raw_multiplier,
                "raw_risk_bps": risk_bps,
                "adjusted_multiplier": adjusted_multiplier,
                "capped_risk_bps": self.max_risk_bps,
            }
            raw_multiplier = adjusted_multiplier
        
        # Determine reason
        if expected_bps < 0:
            reason = "negative_edge_drop"
        elif expected_bps < 2:
            reason = "low_edge"
        elif expected_bps < 5:
            reason = "medium_edge"
        elif expected_bps < 8:
            reason = "high_edge"
        else:
            reason = "very_high_edge"
        
        return {
            "size_multiplier": float(raw_multiplier),
            "reason": reason,
            "expected_bps": float(expected_bps),
            "bin_data": bin_data,
            "caps_applied": caps_applied,
        }
    
    def apply_overlay(
        self,
        base_units: int,
        p_long_v10_1: float,
        session: str,
        regime: str,
        sl_bps: float,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Apply size overlay to base units.
        
        Args:
            base_units: Base units before overlay
            p_long_v10_1: V10.1 p_long prediction
            session: Trading session (EU/OVERLAP/US)
            regime: Regime string (e.g., "UP×LOW", "NEUTRAL×MEDIUM")
            sl_bps: Stop loss in bps
        
        Returns:
            Tuple of (units_out, overlay_meta)
        """
        # Get size multiplier
        multiplier_info = self.get_size_multiplier(
            p_long_v10_1=p_long_v10_1,
            session=session,
            regime=regime,
            sl_bps=sl_bps,
            base_units=base_units,
        )
        
        size_multiplier = multiplier_info["size_multiplier"]
        
        # If multiplier is 0, drop trade (return 0 units)
        if size_multiplier == 0.0:
            overlay_meta = {
                "overlay_applied": True,
                "overlay_name": "ENTRY_V10_1_SIZE",
                "size_before_units": base_units,
                "size_after_units": 0,
                "multiplier": 0.0,
                "reason": multiplier_info["reason"],
                "expected_bps": multiplier_info["expected_bps"],
                "caps_applied": multiplier_info["caps_applied"],
            }
            return 0, overlay_meta
        
        # Apply multiplier
        sign = 1 if base_units >= 0 else -1
        units_abs = abs(int(base_units))
        units_out_abs = int(round(units_abs * size_multiplier))
        
        # Minimum 1 unit if base was > 0
        if units_out_abs == 0 and units_abs > 0:
            log.warning(
                "[ENTRY_V10_1_SIZE_OVERLAY] Size overlay produced 0 units "
                "(base=%s, mult=%.3f); keeping minimum of 1 unit.",
                base_units,
                size_multiplier,
            )
            units_out_abs = 1
        
        units_out = sign * units_out_abs
        
        overlay_meta = {
            "overlay_applied": True,
            "overlay_name": "ENTRY_V10_1_SIZE",
            "size_before_units": base_units,
            "size_after_units": units_out,
            "multiplier": size_multiplier,
            "reason": multiplier_info["reason"],
            "expected_bps": multiplier_info["expected_bps"],
            "bin_data": multiplier_info["bin_data"],
            "caps_applied": multiplier_info["caps_applied"],
        }
        
        return units_out, overlay_meta


def load_entry_v10_1_size_overlay(cfg: Mapping[str, Any] | None) -> Optional[EntryV10_1SizeOverlay]:
    """
    Load EntryV10_1SizeOverlay from config.
    
    Config structure:
        entry_v10_1_size_overlay:
          enabled: true
          edge_buckets_path: "data/entry_v10/entry_v10_1_edge_buckets_2025_flat.json"
          max_size_multiplier: 2.0
          max_risk_bps: 100.0
    """
    if cfg is None:
        return None
    
    enabled = cfg.get("enabled", False)
    if not enabled:
        return None
    
    edge_buckets_path = cfg.get("edge_buckets_path")
    if not edge_buckets_path:
        log.warning("[ENTRY_V10_1_SIZE_OVERLAY] Enabled but edge_buckets_path not provided")
        return None
    
    max_size_multiplier = float(cfg.get("max_size_multiplier", 2.0))
    max_risk_bps = float(cfg.get("max_risk_bps", 100.0))
    
    try:
        overlay = EntryV10_1SizeOverlay(
            edge_buckets_path=Path(edge_buckets_path),
            max_size_multiplier=max_size_multiplier,
            max_risk_bps=max_risk_bps,
        )
        log.info("[ENTRY_V10_1_SIZE_OVERLAY] Loaded successfully")
        return overlay
    except Exception as e:
        log.error(f"[ENTRY_V10_1_SIZE_OVERLAY] Failed to load: {e}")
        return None


__all__ = ["EntryV10_1SizeOverlay", "load_entry_v10_1_size_overlay"]

