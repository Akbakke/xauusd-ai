"""
FARM_V1 Brutal Guards

Centralized guards to ensure FARM_V1 trades are ONLY opened in ASIA + LOW volatility.
This is the single source of truth for FARM entry validation.
"""

import logging
from typing import Union, Dict, Any, List
import pandas as pd

logger = logging.getLogger(__name__)

# Version identifiers for guard implementations
FARM_GUARD_VERSION_V1 = "FARM_V1_BRUTAL_V1"
FARM_GUARD_VERSION_V2 = "FARM_V2_BRUTAL_V2"

# SNIPER_GUARD_V1 UNKNOWN policy telemetry (module-level; per-process / per-worker)
# Policy B (PASS-THROUGH): vol_regime=UNKNOWN is NOT allowed to block by itself.
SNIPER_GUARD_UNKNOWN_PASS_COUNT = 0
SNIPER_GUARD_UNKNOWN_BLOCK_COUNT = 0  # Should remain 0 for Policy B
SNIPER_GUARD_UNKNOWN_LOG_COUNT = 0  # Rate-limit log spam


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


def _extract_session_vol_regime(row: Union[pd.Series, Dict[str, Any]]) -> tuple[str, str]:
    """
    Extract session and vol_regime from a row (shared logic for all guards).
    
    Returns:
        (session, vol_regime) tuple
    """
    # Extract session
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
    
    # Extract vol_regime
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
    
    return (session or "UNKNOWN", vol_regime or "UNKNOWN")


def session_vol_guard(
    row: Union[pd.Series, Dict[str, Any]],
    allowed_sessions: List[str],
    allowed_vol_regimes: List[str],
    allow_extreme: bool = False,
    context: str = "unknown",
    guard_name: str = "SESSION_VOL_GUARD"
) -> bool:
    """
    Generalized session + volatility guard.
    
    Args:
        row: DataFrame row (pd.Series) or dict with session and vol_regime information
        allowed_sessions: List of allowed session IDs (e.g., ["ASIA"] or ["EU", "OVERLAP", "US"])
        allowed_vol_regimes: List of allowed volatility regimes (e.g., ["LOW", "MEDIUM"])
        allow_extreme: Whether to allow EXTREME volatility (default: False)
        context: Context string for logging
        guard_name: Name of guard for logging (e.g., "FARM_BRUTAL_GUARD_V2" or "SNIPER_GUARD_V1")
    
    Returns:
        True if guard passes, raises AssertionError otherwise
    
    Raises:
        AssertionError: If session or vol_regime not in allowed sets
    """
    session, vol_regime = _extract_session_vol_regime(row)
    
    # Build allowed vol regimes list (include EXTREME if allowed)
    effective_allowed_vol = allowed_vol_regimes.copy()
    if allow_extreme and "EXTREME" not in effective_allowed_vol:
        effective_allowed_vol.append("EXTREME")
    
    # BRUTAL ASSERT: Session must be in allowed list
    if session not in allowed_sessions:
        error_msg = (
            f"[{guard_name}] {context}: session={session} not in {allowed_sessions}. "
            f"Only {allowed_sessions} sessions allowed."
        )
        logger.error(error_msg)
        raise AssertionError(error_msg)
    
    # BRUTAL ASSERT: Vol regime must be in allowed list
    if vol_regime not in effective_allowed_vol:
        error_msg = (
            f"[{guard_name}] {context}: vol_regime={vol_regime} not in {effective_allowed_vol}. "
            f"Only {effective_allowed_vol} volatility allowed."
        )
        # SNIPER_GUARD_V1 Policy B: UNKNOWN = PASS-THROUGH (NOT a blocker by itself)
        # Rationale: UNKNOWN means "missing or unmappable regime fields" in replay; hard-blocking makes replay deterministic-dead.
        if guard_name == "SNIPER_GUARD_V1" and vol_regime == "UNKNOWN":
            global SNIPER_GUARD_UNKNOWN_PASS_COUNT, SNIPER_GUARD_UNKNOWN_LOG_COUNT
            SNIPER_GUARD_UNKNOWN_PASS_COUNT += 1
            # Rate-limit to first N occurrences per process to avoid spam
            if SNIPER_GUARD_UNKNOWN_LOG_COUNT < 3:
                SNIPER_GUARD_UNKNOWN_LOG_COUNT += 1
                logger.warning(
                    f"[{guard_name}] {context}: vol_regime=UNKNOWN -> PASS (policy=B) "
                    f"reason=missing_or_unmappable_regime_fields"
                )
            return True

        # Default: hard-block on disallowed vol regime
        logger.error(error_msg)
        raise AssertionError(error_msg)
    
    # Guard passed
    logger.debug(
        f"[{guard_name}] {context}: PASSED - session={session}, vol_regime={vol_regime}"
    )
    return True


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
    allowed_vol_regimes = ["LOW"]
    if allow_medium_vol:
        allowed_vol_regimes.append("MEDIUM")
    
    return session_vol_guard(
        row=row,
        allowed_sessions=["ASIA"],
        allowed_vol_regimes=allowed_vol_regimes,
        allow_extreme=False,
        context=context,
        guard_name="FARM_BRUTAL_GUARD_V2"
    )


def sniper_guard_v1(
    row: Union[pd.Series, Dict[str, Any]],
    context: str = "unknown",
    allow_high_vol: bool = True,
    allow_extreme_vol: bool = False
) -> bool:
    """
    Guard for SNIPER: Allows EU/OVERLAP/US + (LOW ∪ MEDIUM ∪ HIGH) volatility.
    
    Args:
        row: DataFrame row (pd.Series) or dict with session and vol_regime information
        context: Context string for logging
        allow_high_vol: Whether to allow HIGH volatility (default: True)
        allow_extreme_vol: Whether to allow EXTREME volatility (default: False)
    
    Returns:
        True if guard passes, raises AssertionError otherwise
    
    Raises:
        AssertionError: If session not in [EU, OVERLAP, US] or vol_regime not allowed
    """
    allowed_vol_regimes = ["LOW", "MEDIUM"]
    if allow_high_vol:
        allowed_vol_regimes.append("HIGH")
    
    return session_vol_guard(
        row=row,
        allowed_sessions=["EU", "OVERLAP", "US"],
        allowed_vol_regimes=allowed_vol_regimes,
        allow_extreme=allow_extreme_vol,
        context=context,
        guard_name="SNIPER_GUARD_V1"
    )


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

