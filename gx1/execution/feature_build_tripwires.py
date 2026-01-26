"""
PREBUILT-safe feature build call counter for tripwire checks.

This module is safe to import in PREBUILT mode because it does NOT import
any feature-building modules (basic_v1, live_features, etc.).

Builders call bump_feature_build_call_count() to increment the counter.
PREBUILT checks read FEATURE_BUILD_CALL_COUNT to verify no feature-building occurred.
"""

import threading
from typing import Dict

# Global counter for feature build calls (thread-safe for multiprocessing)
_feature_build_call_lock = threading.Lock()
_feature_build_call_count = 0
_feature_build_call_details: Dict[str, int] = {}  # name -> count


def bump_feature_build_call_count(name: str) -> None:
    """
    Increment global feature build call counter (thread-safe).
    
    Called by feature builders (basic_v1.build_basic_v1, etc.) to track usage.
    
    Args:
        name: Identifier for the builder function (e.g., "basic_v1.build_basic_v1")
    """
    global _feature_build_call_count, _feature_build_call_details
    with _feature_build_call_lock:
        _feature_build_call_count += 1
        _feature_build_call_details[name] = _feature_build_call_details.get(name, 0) + 1


def get_feature_build_call_count() -> int:
    """
    Get global count of feature build calls (thread-safe).
    
    Returns:
        Total number of feature build calls since last reset
    """
    with _feature_build_call_lock:
        return _feature_build_call_count


def get_feature_build_call_details() -> Dict[str, int]:
    """
    Get detailed breakdown of feature build calls by name (thread-safe).
    
    Returns:
        Dictionary mapping builder name to call count
    """
    with _feature_build_call_lock:
        return _feature_build_call_details.copy()


def reset_feature_build_call_count() -> None:
    """
    Reset global count of feature build calls (thread-safe).
    
    Called at run start to reset counters for verification.
    """
    global _feature_build_call_count, _feature_build_call_details
    with _feature_build_call_lock:
        _feature_build_call_count = 0
        _feature_build_call_details = {}


# Expose counter for direct access (read-only, use get_feature_build_call_count() for thread-safe access)
FEATURE_BUILD_CALL_COUNT = 0  # This is a constant reference, actual value is in _feature_build_call_count
