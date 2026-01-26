#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trial 160 Policy Loader — Fail-Fast, No Defaults

Loads Trial 160 production policy from JSON with strict validation.
Hard-fails on any missing fields, unknown fields, or type mismatches.

Dependencies (explicit install line):
  (no external dependencies beyond stdlib)
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

log = logging.getLogger(__name__)


@dataclass
class PolicyConfig:
    """Trial 160 policy configuration (fail-fast, no defaults)."""
    
    policy_id: str
    policy_sha256: str
    entry_threshold: float
    max_concurrent_positions: int
    risk_guard_block_atr_bps_gte: float
    risk_guard_block_spread_bps_gte: int
    risk_guard_cooldown_bars_after_entry: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "policy_id": self.policy_id,
            "policy_sha256": self.policy_sha256,
            "entry_threshold": self.entry_threshold,
            "max_concurrent_positions": self.max_concurrent_positions,
            "risk_guard_block_atr_bps_gte": self.risk_guard_block_atr_bps_gte,
            "risk_guard_block_spread_bps_gte": self.risk_guard_block_spread_bps_gte,
            "risk_guard_cooldown_bars_after_entry": self.risk_guard_cooldown_bars_after_entry,
        }


def _compute_file_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of file content."""
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _validate_type(value: Any, expected_type: type, field_name: str) -> None:
    """Hard-fail if value is not of expected type."""
    if not isinstance(value, expected_type):
        raise TypeError(
            f"POLICY_LOADER_FAIL: Field '{field_name}' has wrong type. "
            f"Expected {expected_type.__name__}, got {type(value).__name__}: {value}"
        )


def load_policy(policy_path: Path) -> PolicyConfig:
    """
    Load Trial 160 policy from JSON file with strict validation.
    
    Hard-fails on:
    - Missing required fields
    - Unknown fields (forbids extra)
    - Wrong types
    - Missing policy_id
    
    Returns:
        PolicyConfig with policy_sha256 computed from file content.
    
    Raises:
        FileNotFoundError: If policy file doesn't exist
        KeyError: If required field is missing
        TypeError: If field has wrong type
        ValueError: If unknown fields are present
    """
    if not policy_path.exists():
        raise FileNotFoundError(
            f"POLICY_LOADER_FAIL: Policy file not found: {policy_path}"
        )
    
    # Load JSON
    try:
        with open(policy_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"POLICY_LOADER_FAIL: Invalid JSON in policy file {policy_path}: {e}"
        ) from e
    
    # Compute SHA256
    policy_sha256 = _compute_file_sha256(policy_path)
    
    # Required fields (no defaults, all must be present)
    required_fields = {
        "policy_id": str,
        "parameters": dict,
    }
    
    # Check required top-level fields
    for field, expected_type in required_fields.items():
        if field not in data:
            raise KeyError(
                f"POLICY_LOADER_FAIL: Missing required field '{field}' in policy file {policy_path}"
            )
        _validate_type(data[field], expected_type, field)
    
    # Check policy_id
    policy_id = data["policy_id"]
    if not policy_id or not isinstance(policy_id, str):
        raise ValueError(
            f"POLICY_LOADER_FAIL: policy_id must be non-empty string, got: {policy_id}"
        )
    
    # Check parameters dict
    params = data["parameters"]
    if not isinstance(params, dict):
        raise TypeError(
            f"POLICY_LOADER_FAIL: 'parameters' must be dict, got {type(params).__name__}"
        )
    
    # Required parameters (no defaults)
    required_params = {
        "entry_threshold": float,
        "max_concurrent_positions": int,
        "risk_guard_block_atr_bps_gte": float,
        "risk_guard_block_spread_bps_gte": int,
        "risk_guard_cooldown_bars_after_entry": int,
    }
    
    # Check all required parameters are present
    for param_name, expected_type in required_params.items():
        if param_name not in params:
            raise KeyError(
                f"POLICY_LOADER_FAIL: Missing required parameter '{param_name}' in policy file {policy_path}"
            )
        _validate_type(params[param_name], expected_type, param_name)
    
    # Forbid unknown parameters (strict validation)
    allowed_params = set(required_params.keys())
    actual_params = set(params.keys())
    unknown_params = actual_params - allowed_params
    if unknown_params:
        raise ValueError(
            f"POLICY_LOADER_FAIL: Unknown parameters found: {unknown_params}. "
            f"Allowed parameters: {allowed_params}"
        )
    
    # Forbid unknown top-level fields (strict validation)
    allowed_top_level = {"policy_id", "policy_name", "policy_version", "created", "source", 
                         "description", "parameters", "promotion_results", "invariants", "notes"}
    actual_top_level = set(data.keys())
    unknown_top_level = actual_top_level - allowed_top_level
    if unknown_top_level:
        raise ValueError(
            f"POLICY_LOADER_FAIL: Unknown top-level fields found: {unknown_top_level}. "
            f"Allowed fields: {allowed_top_level}"
        )
    
    # Extract values (no defaults, all must be present)
    entry_threshold = float(params["entry_threshold"])
    max_concurrent_positions = int(params["max_concurrent_positions"])
    risk_guard_block_atr_bps_gte = float(params["risk_guard_block_atr_bps_gte"])
    risk_guard_block_spread_bps_gte = int(params["risk_guard_block_spread_bps_gte"])
    risk_guard_cooldown_bars_after_entry = int(params["risk_guard_cooldown_bars_after_entry"])
    
    # Validate value ranges (sanity checks)
    if not (0.0 < entry_threshold < 1.0):
        raise ValueError(
            f"POLICY_LOADER_FAIL: entry_threshold must be in (0, 1), got: {entry_threshold}"
        )
    if max_concurrent_positions < 1:
        raise ValueError(
            f"POLICY_LOADER_FAIL: max_concurrent_positions must be >= 1, got: {max_concurrent_positions}"
        )
    if risk_guard_block_atr_bps_gte < 0:
        raise ValueError(
            f"POLICY_LOADER_FAIL: risk_guard_block_atr_bps_gte must be >= 0, got: {risk_guard_block_atr_bps_gte}"
        )
    if risk_guard_block_spread_bps_gte < 0:
        raise ValueError(
            f"POLICY_LOADER_FAIL: risk_guard_block_spread_bps_gte must be >= 0, got: {risk_guard_block_spread_bps_gte}"
        )
    if risk_guard_cooldown_bars_after_entry < 0:
        raise ValueError(
            f"POLICY_LOADER_FAIL: risk_guard_cooldown_bars_after_entry must be >= 0, got: {risk_guard_cooldown_bars_after_entry}"
        )
    
    # Create PolicyConfig
    config = PolicyConfig(
        policy_id=policy_id,
        policy_sha256=policy_sha256,
        entry_threshold=entry_threshold,
        max_concurrent_positions=max_concurrent_positions,
        risk_guard_block_atr_bps_gte=risk_guard_block_atr_bps_gte,
        risk_guard_block_spread_bps_gte=risk_guard_block_spread_bps_gte,
        risk_guard_cooldown_bars_after_entry=risk_guard_cooldown_bars_after_entry,
    )
    
    log.info(f"[POLICY_LOADER] Loaded policy: {policy_id} (SHA256: {policy_sha256[:16]}...)")
    log.info(f"[POLICY_LOADER] Parameters: entry_threshold={entry_threshold}, "
             f"max_positions={max_concurrent_positions}, spread_bps={risk_guard_block_spread_bps_gte}, "
             f"atr_bps={risk_guard_block_atr_bps_gte}, cooldown={risk_guard_cooldown_bars_after_entry}")
    
    return config


if __name__ == "__main__":
    # Test loader
    import sys
    from pathlib import Path
    
    if len(sys.argv) < 2:
        print("Usage: python3 -m gx1.policy.trial160_loader <policy_path>")
        sys.exit(1)
    
    policy_path = Path(sys.argv[1])
    try:
        config = load_policy(policy_path)
        print(f"✅ Policy loaded successfully:")
        print(f"  Policy ID: {config.policy_id}")
        print(f"  Policy SHA256: {config.policy_sha256}")
        print(f"  Entry threshold: {config.entry_threshold}")
        print(f"  Max positions: {config.max_concurrent_positions}")
        print(f"  Spread BPS: {config.risk_guard_block_spread_bps_gte}")
        print(f"  ATR BPS: {config.risk_guard_block_atr_bps_gte}")
        print(f"  Cooldown: {config.risk_guard_cooldown_bars_after_entry}")
    except Exception as e:
        print(f"❌ Failed to load policy: {e}")
        sys.exit(1)
