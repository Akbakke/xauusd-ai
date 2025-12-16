#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OANDA credentials management.

Loads credentials from environment variables and validates them.
Fail-closed in PROD_BASELINE mode.
"""
from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class OandaCredentials:
    """OANDA credentials configuration."""
    env: str  # "practice" or "live"
    api_token: str
    account_id: str
    api_url: str  # Base REST API URL
    stream_url: str  # Base streaming API URL


def load_oanda_credentials(prod_baseline: bool = False, require_live_latch: bool = False) -> OandaCredentials:
    """
    Load OANDA credentials from environment variables.
    
    Expected environment variables:
    - OANDA_ENV: "practice" or "live" (default: "practice")
    - OANDA_API_TOKEN: API token (required)
    - OANDA_ACCOUNT_ID: Account ID (required)
    
    Args:
        prod_baseline: If True, fail-closed on missing credentials
        require_live_latch: If True, require I_UNDERSTAND_LIVE_TRADING=YES for live environment
        
    Returns:
        OandaCredentials object
        
    Raises:
        ValueError: If credentials are missing/invalid (in PROD_BASELINE mode)
        ValueError: If live trading latch not set (when require_live_latch=True and OANDA_ENV=live)
    """
    # Read environment variables
    oanda_env = os.getenv("OANDA_ENV", "practice").strip().lower()
    # Support both OANDA_API_TOKEN (preferred) and OANDA_API_KEY (legacy)
    api_token = os.getenv("OANDA_API_TOKEN", "").strip() or os.getenv("OANDA_API_KEY", "").strip()
    account_id = os.getenv("OANDA_ACCOUNT_ID", "").strip()
    
    # Validate environment
    if oanda_env not in ("practice", "live"):
        error_msg = f"Invalid OANDA_ENV: {oanda_env} (must be 'practice' or 'live')"
        if prod_baseline:
            logger.error(f"[OANDA_CREDENTIALS] {error_msg}")
            raise ValueError(error_msg)
        else:
            logger.warning(f"[OANDA_CREDENTIALS] {error_msg}, defaulting to 'practice'")
            oanda_env = "practice"
    
    # Live trading safety latch
    if oanda_env == "live" and require_live_latch:
        live_latch = os.getenv("I_UNDERSTAND_LIVE_TRADING", "").strip().upper()
        if live_latch != "YES":
            error_msg = (
                "Live trading requires explicit confirmation: "
                "Set I_UNDERSTAND_LIVE_TRADING=YES in environment variables. "
                "This is a safety measure to prevent accidental live trading."
            )
            logger.error(f"[OANDA_CREDENTIALS] {error_msg}")
            raise ValueError(error_msg)
        logger.warning(
            "[OANDA_CREDENTIALS] ⚠️  LIVE TRADING ENABLED - Real money at risk! "
            "env=live, account_id=%s",
            _mask_account_id(account_id) if account_id else "MISSING",
        )
    
    # Validate credentials
    missing = []
    if not api_token:
        missing.append("OANDA_API_TOKEN")
    if not account_id:
        missing.append("OANDA_ACCOUNT_ID")
    
    if missing:
        error_msg = f"Missing OANDA credentials: {', '.join(missing)}"
        if prod_baseline:
            logger.error(f"[OANDA_CREDENTIALS] {error_msg}")
            logger.error("[OANDA_CREDENTIALS] PROD_BASELINE mode: Trading disabled due to missing credentials")
            raise ValueError(error_msg)
        else:
            logger.warning(f"[OANDA_CREDENTIALS] {error_msg}")
            # Return empty credentials in dev mode (will fail later if actually used)
            api_token = api_token or ""
            account_id = account_id or ""
    
    # Determine API URLs based on environment
    if oanda_env == "practice":
        api_url = "https://api-fxpractice.oanda.com"
        stream_url = "https://stream-fxpractice.oanda.com"
    else:  # live
        api_url = "https://api-fxtrade.oanda.com"
        stream_url = "https://stream-fxtrade.oanda.com"
    
    credentials = OandaCredentials(
        env=oanda_env,
        api_token=api_token,
        account_id=account_id,
        api_url=api_url,
        stream_url=stream_url,
    )
    
    # Log credentials loaded (mask sensitive info)
    masked_account_id = _mask_account_id(account_id) if account_id else "MISSING"
    logger.info(
        "[OANDA_CREDENTIALS] Loaded credentials: env=%s, account_id=%s, api_url=%s",
        oanda_env,
        masked_account_id,
        api_url,
    )
    
    return credentials


def _mask_account_id(account_id: str) -> str:
    """
    Mask account ID for logging (e.g., "101-004-12345-001" -> "101-***-001").
    
    Args:
        account_id: Full account ID
        
    Returns:
        Masked account ID
    """
    if not account_id:
        return "MISSING"
    
    # OANDA account IDs are typically: XXX-XXX-XXXXX-XXX
    parts = account_id.split("-")
    if len(parts) >= 3:
        # Mask middle parts
        masked = f"{parts[0]}-***-{parts[-1]}"
        return masked
    
    # Fallback: mask middle characters
    if len(account_id) > 6:
        return f"{account_id[:3]}***{account_id[-3:]}"
    
    return "***"


def validate_oanda_credentials(credentials: OandaCredentials) -> bool:
    """
    Validate OANDA credentials by making a test API call.
    
    Args:
        credentials: OandaCredentials object
        
    Returns:
        True if credentials are valid, False otherwise
    """
    try:
        import requests
        
        # Test endpoint: Get account details
        url = f"{credentials.api_url}/v3/accounts/{credentials.account_id}"
        headers = {
            "Authorization": f"Bearer {credentials.api_token}",
            "Content-Type": "application/json",
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            logger.info("[OANDA_CREDENTIALS] Credentials validated successfully")
            return True
        else:
            logger.warning(
                "[OANDA_CREDENTIALS] Credential validation failed: status_code=%d, response=%s",
                response.status_code,
                response.text[:200],
            )
            return False
            
    except Exception as e:
        logger.warning("[OANDA_CREDENTIALS] Credential validation error: %s", e)
        return False

