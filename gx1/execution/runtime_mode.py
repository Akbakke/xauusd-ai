"""
Runtime Mode Helper - DEL 6: Fail-fast policy (replay) vs tolerant policy (live).

Provides helper functions to determine if we're in replay or live mode,
and to apply appropriate fail-fast vs tolerant behavior.
"""

import os
import logging
from typing import Optional, Any

log = logging.getLogger(__name__)


def is_replay_mode(runner: Optional[Any] = None) -> bool:
    """
    Determine if we're in replay mode.
    
    DEL 6: Fail-fast policy helper.
    
    Checks (in order):
    1. runner.replay_mode attribute (if runner provided)
    2. GX1_REPLAY_MODE environment variable
    3. Default: False (live mode)
    
    Args:
        runner: Optional GX1DemoRunner instance
    
    Returns:
        True if replay mode, False if live mode
    """
    if runner is not None:
        if hasattr(runner, "replay_mode"):
            return bool(runner.replay_mode)
    
    # Check environment variable
    env_replay = os.getenv("GX1_REPLAY_MODE", "").lower()
    if env_replay in ("true", "1", "yes"):
        return True
    elif env_replay in ("false", "0", "no"):
        return False
    
    # Default: live mode (conservative)
    return False


def is_live_mode(runner: Optional[Any] = None) -> bool:
    """
    Determine if we're in live mode.
    
    DEL 6: Tolerant policy helper.
    
    Args:
        runner: Optional GX1DemoRunner instance
    
    Returns:
        True if live mode, False if replay mode
    """
    return not is_replay_mode(runner)


def fail_fast_if_replay(
    error_msg: str,
    runner: Optional[Any] = None,
    exception_class: type = RuntimeError,
) -> None:
    """
    Fail-fast in replay mode, log warning in live mode.
    
    DEL 6: Fail-fast policy implementation.
    
    Args:
        error_msg: Error message to raise/log
        runner: Optional GX1DemoRunner instance
        exception_class: Exception class to raise (default: RuntimeError)
    
    Raises:
        exception_class: In replay mode
    """
    if is_replay_mode(runner):
        raise exception_class(error_msg)
    else:
        log.warning(f"[LIVE_MODE] {error_msg} (live mode: continuing with degraded behavior)")


def log_degraded_if_live(
    warning_msg: str,
    runner: Optional[Any] = None,
    degraded_flag: Optional[str] = None,
) -> None:
    """
    Log degraded mode warning in live mode.
    
    DEL 6: Tolerant policy implementation.
    
    Args:
        warning_msg: Warning message to log
        runner: Optional GX1DemoRunner instance
        degraded_flag: Optional flag name to track (e.g., "DEGRADED_CTX")
    """
    if is_live_mode(runner):
        log.warning(f"[LIVE_MODE] {warning_msg}")
        if degraded_flag:
            log.warning(f"[LIVE_MODE] Degraded flag set: {degraded_flag}")

