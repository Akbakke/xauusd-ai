"""
Centralized trade log schema definition.

This is the SINGLE SOURCE OF TRUTH for all trade log CSV columns.
All trade log operations MUST use this schema.
"""

# Trade log field names in order
TRADE_LOG_FIELDS = [
    # Trade identification
    "trade_id",
    "run_id",
    
    # Timestamps
    "entry_time",
    "exit_time",
    "entry_ts",  # Alias for entry_time (backward compatibility)
    "exit_ts",   # Alias for exit_time (backward compatibility)
    
    # Trade details
    "side",
    "direction",  # Alias for side (backward compatibility)
    "entry_price",
    "exit_price",
    "units",
    
    # PnL
    "pnl_bps",
    "pnl_currency",
    
    # Entry/exit probabilities
    "entry_prob_long",
    "entry_prob_short",
    "exit_prob_close",
    "entry_p_long",        # ENTRY_V9 p_long (for FARM_V2)
    "entry_p_profitable",  # Meta-model p_profitable (for FARM_V2)
    "entry_policy_version", # Policy version identifier (e.g., "FARM_V2")
    
    # Exit information
    "exit_reason",
    "primary_exit_reason",
    "bars_held",
    
    # Regime information (backward compatibility - uses entry)
    "session",      # Session (ASIA/EU/US/OVERLAP) - backward compatibility (uses entry)
    "vol_regime",   # Volatility regime (LOW/MEDIUM/HIGH/EXTREME) - backward compatibility (uses entry)
    "trend_regime", # Trend regime (TREND_UP/TREND_DOWN/MR)
    
    # Entry-regime (explicit entry-regime fields)
    "session_entry",      # Session at entry (ASIA/EU/US/OVERLAP)
    "vol_regime_entry",   # Volatility regime at entry (LOW/MEDIUM/HIGH/EXTREME)
    
    # Exit-regime (explicit exit-regime fields)
    "session_exit",      # Session at exit (ASIA/EU/US/OVERLAP)
    "vol_regime_exit",   # Volatility regime at exit (LOW/MEDIUM/HIGH/EXTREME)
    
    # FARM-specific fields (SINGLE SOURCE OF TRUTH for FARM trades)
    "farm_entry_session",    # FARM entry session (SINGLE SOURCE OF TRUTH for FARM trades)
    "farm_entry_vol_regime", # FARM entry vol regime (SINGLE SOURCE OF TRUTH for FARM trades)
    "farm_guard_version",    # FARM guard version identifier
    
    # Additional metadata
    "vol_bucket",
    "atr_bps",
    "notes",
    "policy_name",  # Policy version/name for traceability
    "model_name",   # Model version/name used for entry prediction
    "exit_profile", # Exit strategy/profile identifier (e.g., "FARM_EXIT_V1_STABLE")
    
    # Extra JSON (must be last)
    "extra",
]

# Field aliases for backward compatibility
FIELD_ALIASES = {
    "entry_ts": "entry_time",
    "exit_ts": "exit_time",
    "direction": "side",
}

def get_field_name(field: str) -> str:
    """
    Get canonical field name, resolving aliases.
    
    Args:
        field: Field name (may be alias)
    
    Returns:
        Canonical field name
    """
    return FIELD_ALIASES.get(field, field)

def normalize_field_names(fields: list) -> list:
    """
    Normalize field names, resolving aliases and ensuring all are in TRADE_LOG_FIELDS.
    
    Args:
        fields: List of field names (may include aliases)
    
    Returns:
        List of canonical field names
    """
    normalized = []
    for field in fields:
        canonical = get_field_name(field)
        if canonical in TRADE_LOG_FIELDS:
            normalized.append(canonical)
    return normalized

