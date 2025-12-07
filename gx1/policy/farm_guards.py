"""
FARM_V1 Brutal Guards

Centralized guards to ensure FARM_V1 trades are ONLY opened in ASIA + LOW volatility.
This is the single source of truth for FARM entry validation.
"""

import logging
from typing import Union, Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)

# Version identifiers for guard implementations
FARM_GUARD_VERSION_V1 = "FARM_V1_BRUTAL_V1"
FARM_GUARD_VERSION_V2 = "FARM_V2_BRUTAL_V2"


def farm_brutal_guard(row: Union[pd.Series, Dict[str, Any]], context: str = "unknown") -> bool:
    """
    Brutal guard for FARM_V1: Only allows ASIA + LOW volatility.
    
    This is the SINGLE SOURCE OF TRUTH for FARM entry validation.
    All FARM entry paths MUST pass through this guard.
    
    Args:
        row: DataFrame row (pd.Series) or dict with session and vol_regime information
        context: Context string for logging (e.g., "policy", "live_runner", "replay")
    
    Returns:
        True if guard passes (ASIA + LOW), raises AssertionError otherwise
    
    Raises:
        AssertionError: If session != "ASIA" or vol_regime != "LOW"
    """
    # Extract session - try multiple column names
    session = None
    for col in ["session", "session_entry", "_v1_session_tag", "session_tag"]:
        if isinstance(row, pd.Series):
            if col in row.index:
                session = row[col]
                break
        elif isinstance(row, dict):
            if col in row:
                session = row[col]
                break
    
    # Fallback: try session_id mapping
    if session is None:
        if isinstance(row, pd.Series):
            if "session_id" in row.index:
                session_map = {0: "EU", 1: "OVERLAP", 2: "US"}
                session_id = row["session_id"]
                session = session_map.get(int(session_id), "UNKNOWN")
        elif isinstance(row, dict):
            if "session_id" in row:
                session_map = {0: "EU", 1: "OVERLAP", 2: "US"}
                session_id = row["session_id"]
                session = session_map.get(int(session_id), "UNKNOWN")
    
    # Extract vol_regime - try multiple column names
    vol_regime = None
    for col in ["vol_regime", "vol_regime_entry", "atr_regime"]:
        if isinstance(row, pd.Series):
            if col in row.index:
                vol_regime = row[col]
                break
        elif isinstance(row, dict):
            if col in row:
                vol_regime = row[col]
                break
    
    # Fallback: try atr_regime_id mapping
    if vol_regime is None:
        ATR_ID_TO_VOL = {0: "LOW", 1: "MEDIUM", 2: "HIGH", 3: "EXTREME"}
        for col in ["_v1_atr_regime_id", "atr_regime_id"]:
            if isinstance(row, pd.Series):
                if col in row.index:
                    atr_id = int(row[col])
                    vol_regime = ATR_ID_TO_VOL.get(atr_id, "UNKNOWN")
                    break
            elif isinstance(row, dict):
                if col in row:
                    atr_id = int(row[col])
                    vol_regime = ATR_ID_TO_VOL.get(atr_id, "UNKNOWN")
                    break
    
    # BRUTAL ASSERT: Session must be ASIA
    if session != "ASIA":
        error_msg = (
            f"[FARM_BRUTAL_GUARD] {context}: session={session} != ASIA. "
            f"FARM_V1 only allows ASIA session."
        )
        logger.error(error_msg)
        raise AssertionError(error_msg)
    
    # BRUTAL ASSERT: Vol regime must be LOW
    if vol_regime != "LOW":
        error_msg = (
            f"[FARM_BRUTAL_GUARD] {context}: vol_regime={vol_regime} != LOW. "
            f"FARM_V1 only allows LOW volatility."
        )
        logger.error(error_msg)
        raise AssertionError(error_msg)
    
    # Guard passed
    logger.debug(
        f"[FARM_BRUTAL_GUARD] {context}: PASSED - session={session}, vol_regime={vol_regime}"
    )
    return True


def get_farm_entry_metadata(row: Union[pd.Series, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract FARM entry metadata from a row.
    
    This is used to log explicit FARM entry regime in trade metadata.
    
    Args:
        row: DataFrame row (pd.Series) or dict with session and vol_regime information
    
    Returns:
        Dict with:
            - farm_entry_session: Session at entry (should be "ASIA")
            - farm_entry_vol_regime: Vol regime at entry (should be "LOW")
            - farm_guard_version: Version identifier for guard implementation
    """
    # Extract session (same logic as guard)
    session = None
    for col in ["session", "session_entry", "_v1_session_tag", "session_tag"]:
        if isinstance(row, pd.Series):
            if col in row.index:
                session = row[col]
                break
        elif isinstance(row, dict):
            if col in row:
                session = row[col]
                break
    
    if session is None and (isinstance(row, pd.Series) and "session_id" in row.index) or (isinstance(row, dict) and "session_id" in row):
        session_map = {0: "EU", 1: "OVERLAP", 2: "US"}
        session_id = row["session_id"] if isinstance(row, pd.Series) else row.get("session_id")
        session = session_map.get(int(session_id), "UNKNOWN")
    
    # Extract vol_regime (same logic as guard)
    vol_regime = None
    for col in ["vol_regime", "vol_regime_entry", "atr_regime"]:
        if isinstance(row, pd.Series):
            if col in row.index:
                vol_regime = row[col]
                break
        elif isinstance(row, dict):
            if col in row:
                vol_regime = row[col]
                break
    
    if vol_regime is None:
        ATR_ID_TO_VOL = {0: "LOW", 1: "MEDIUM", 2: "HIGH", 3: "EXTREME"}
        for col in ["_v1_atr_regime_id", "atr_regime_id"]:
            if isinstance(row, pd.Series):
                if col in row.index:
                    atr_id = int(row[col])
                    vol_regime = ATR_ID_TO_VOL.get(atr_id, "UNKNOWN")
                    break
            elif isinstance(row, dict):
                if col in row:
                    atr_id = int(row[col])
                    vol_regime = ATR_ID_TO_VOL.get(atr_id, "UNKNOWN")
                    break
    
    return {
        "farm_entry_session": session,
        "farm_entry_vol_regime": vol_regime,
        "farm_guard_version": FARM_GUARD_VERSION_V1,
    }


def farm_brutal_guard_v2(
    row: Union[pd.Series, Dict[str, Any]], 
    context: str = "unknown",
    allow_medium_vol: bool = True
) -> bool:
    """
    Brutal guard for FARM_V2: Allows ASIA + (LOW ∪ MEDIUM) volatility.
    
    This is the guard for FARM_V2_PLUS which allows both LOW and MEDIUM volatility
    regimes, while FARM_V1 remains ASIA+LOW only.
    
    Args:
        row: DataFrame row (pd.Series) or dict with session and vol_regime information
        context: Context string for logging (e.g., "policy", "live_runner", "replay")
        allow_medium_vol: Whether to allow MEDIUM volatility (default: True)
    
    Returns:
        True if guard passes (ASIA + (LOW or MEDIUM)), raises AssertionError otherwise
    
    Raises:
        AssertionError: If session != "ASIA" or vol_regime not in allowed set
    """
    # Extract session - same logic as V1 guard
    session = None
    for col in ["session", "session_entry", "_v1_session_tag", "session_tag"]:
        if isinstance(row, pd.Series):
            if col in row.index:
                session = row[col]
                break
        elif isinstance(row, dict):
            if col in row:
                session = row[col]
                break
    
    # Fallback: try session_id mapping
    if session is None:
        if isinstance(row, pd.Series):
            if "session_id" in row.index:
                session_map = {0: "EU", 1: "OVERLAP", 2: "US"}
                session_id = row["session_id"]
                session = session_map.get(int(session_id), "UNKNOWN")
        elif isinstance(row, dict):
            if "session_id" in row:
                session_map = {0: "EU", 1: "OVERLAP", 2: "US"}
                session_id = row["session_id"]
                session = session_map.get(int(session_id), "UNKNOWN")
    
    # Extract vol_regime - same logic as V1 guard
    vol_regime = None
    for col in ["vol_regime", "vol_regime_entry", "atr_regime"]:
        if isinstance(row, pd.Series):
            if col in row.index:
                vol_regime = row[col]
                break
        elif isinstance(row, dict):
            if col in row:
                vol_regime = row[col]
                break
    
    # Fallback: try atr_regime_id mapping
    if vol_regime is None:
        ATR_ID_TO_VOL = {0: "LOW", 1: "MEDIUM", 2: "HIGH", 3: "EXTREME"}
        for col in ["_v1_atr_regime_id", "atr_regime_id"]:
            if isinstance(row, pd.Series):
                if col in row.index:
                    atr_id = int(row[col])
                    vol_regime = ATR_ID_TO_VOL.get(atr_id, "UNKNOWN")
                    break
            elif isinstance(row, dict):
                if col in row:
                    atr_id = int(row[col])
                    vol_regime = ATR_ID_TO_VOL.get(atr_id, "UNKNOWN")
                    break
    
    # BRUTAL ASSERT: Session must be ASIA
    if session != "ASIA":
        error_msg = (
            f"[FARM_BRUTAL_GUARD_V2] {context}: session={session} != ASIA. "
            f"FARM_V2 only allows ASIA session."
        )
        logger.error(error_msg)
        raise AssertionError(error_msg)
    
    # BRUTAL ASSERT: Vol regime must be LOW or (MEDIUM if allowed)
    allowed_vol_regimes = ["LOW"]
    if allow_medium_vol:
        allowed_vol_regimes.append("MEDIUM")
    
    if vol_regime not in allowed_vol_regimes:
        error_msg = (
            f"[FARM_BRUTAL_GUARD_V2] {context}: vol_regime={vol_regime} not in {allowed_vol_regimes}. "
            f"FARM_V2 only allows {allowed_vol_regimes} volatility."
        )
        logger.error(error_msg)
        raise AssertionError(error_msg)
    
    # Guard passed
    logger.debug(
        f"[FARM_BRUTAL_GUARD_V2] {context}: PASSED - session={session}, vol_regime={vol_regime}"
    )
    return True


def get_farm_entry_metadata_v2(
    row: Union[pd.Series, Dict[str, Any]],
    allow_medium_vol: bool = True
) -> Dict[str, Any]:
    """
    Extract FARM entry metadata from a row (V2 version).
    
    This is used to log explicit FARM entry regime in trade metadata for V2.
    
    Args:
        row: DataFrame row (pd.Series) or dict with session and vol_regime information
        allow_medium_vol: Whether MEDIUM volatility is allowed (default: True)
    
    Returns:
        Dict with:
            - farm_entry_session: Session at entry (should be "ASIA")
            - farm_entry_vol_regime: Vol regime at entry ("LOW" or "MEDIUM")
            - farm_guard_version: Version identifier for guard implementation ("FARM_V2_BRUTAL_V2")
    """
    # Extract session (same logic as guard)
    session = None
    for col in ["session", "session_entry", "_v1_session_tag", "session_tag"]:
        if isinstance(row, pd.Series):
            if col in row.index:
                session = row[col]
                break
        elif isinstance(row, dict):
            if col in row:
                session = row[col]
                break
    
    if session is None and (isinstance(row, pd.Series) and "session_id" in row.index) or (isinstance(row, dict) and "session_id" in row):
        session_map = {0: "EU", 1: "OVERLAP", 2: "US"}
        session_id = row["session_id"] if isinstance(row, pd.Series) else row.get("session_id")
        session = session_map.get(int(session_id), "UNKNOWN")
    
    # Extract vol_regime (same logic as guard)
    vol_regime = None
    for col in ["vol_regime", "vol_regime_entry", "atr_regime"]:
        if isinstance(row, pd.Series):
            if col in row.index:
                vol_regime = row[col]
                break
        elif isinstance(row, dict):
            if col in row:
                vol_regime = row[col]
                break
    
    if vol_regime is None:
        ATR_ID_TO_VOL = {0: "LOW", 1: "MEDIUM", 2: "HIGH", 3: "EXTREME"}
        for col in ["_v1_atr_regime_id", "atr_regime_id"]:
            if isinstance(row, pd.Series):
                if col in row.index:
                    atr_id = int(row[col])
                    vol_regime = ATR_ID_TO_VOL.get(atr_id, "UNKNOWN")
                    break
            elif isinstance(row, dict):
                if col in row:
                    atr_id = int(row[col])
                    vol_regime = ATR_ID_TO_VOL.get(atr_id, "UNKNOWN")
                    break
    
    return {
        "farm_entry_session": session,
        "farm_entry_vol_regime": vol_regime,
        "farm_guard_version": FARM_GUARD_VERSION_V2,
    }

