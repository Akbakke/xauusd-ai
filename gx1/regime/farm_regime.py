"""
FARM-only regime inference based on session + ATR regime.

This module provides a simple regime classification for FARM_V2B that does not
depend on Big Brain V1 runtime. It uses session ID and ATR regime ID to determine
if a bar is within FARM scope (ASIA + LOW/MEDIUM volatility).
"""

from typing import Union


def infer_farm_regime(session_id: Union[str, int], atr_regime_id: Union[str, int]) -> str:
    """
    Return a simple FARM-regime string based on session + ATR-regime.
    
    FARM scope:
    - ASIA + LOW    -> "FARM_ASIA_LOW"
    - ASIA + MEDIUM -> "FARM_ASIA_MEDIUM"
    - All other     -> "FARM_OUT_OF_SCOPE"
    
    Parameters
    ----------
    session_id : str | int
        Session identifier. Expected values: "ASIA", "EU", "US", "OVERLAP", or numeric IDs.
        Will be normalized to uppercase string.
    atr_regime_id : str | int
        ATR regime identifier. Expected values: "LOW", "MEDIUM", "HIGH", "EXTREME", or numeric IDs.
        Will be normalized to uppercase string.
    
    Returns
    -------
    str
        FARM regime string:
        - "FARM_ASIA_LOW" if ASIA + LOW
        - "FARM_ASIA_MEDIUM" if ASIA + MEDIUM
        - "FARM_OUT_OF_SCOPE" otherwise
    """
    # Normalize inputs to uppercase strings
    session = str(session_id).upper()
    atr_regime = str(atr_regime_id).upper()
    
    # Map numeric ATR regime IDs to names (if needed)
    # Common mappings: 0=LOW, 1=MEDIUM, 2=HIGH, 3=EXTREME
    atr_regime_map = {
        "0": "LOW",
        "1": "MEDIUM",
        "2": "HIGH",
        "3": "EXTREME",
    }
    if atr_regime in atr_regime_map:
        atr_regime = atr_regime_map[atr_regime]
    
    # FARM scope: ASIA session + LOW or MEDIUM volatility
    if session == "ASIA":
        if atr_regime == "LOW":
            return "FARM_ASIA_LOW"
        elif atr_regime == "MEDIUM":
            return "FARM_ASIA_MEDIUM"
    
    # All other combinations are out of scope
    return "FARM_OUT_OF_SCOPE"

