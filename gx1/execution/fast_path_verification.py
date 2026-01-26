"""
Fast path verification for replay mode.

Verifies that fast path is enabled with correct environment variables and flags.
Hard fails in replay mode if fast path is not active.
"""

import os
import logging

log = logging.getLogger(__name__)

def verify_fast_path_enabled(is_replay: bool = False) -> dict:
    """
    Verify fast path is enabled.
    
    Checks:
    - GX1_REPLAY_INCREMENTAL_FEATURES=1 (incremental features)
    - GX1_REPLAY_NO_CSV=1 (CSV disabled)
    - GX1_FEATURE_USE_NP_ROLLING=1 (NumPy rolling for features)
    - Single-thread BLAS (OMP_NUM_THREADS=1, etc.)
    
    Args:
        is_replay: Whether we're in replay mode (hard fail if fast path not enabled)
    
    Returns:
        dict with fast_path_enabled (bool) and details
    """
    checks = {
        "GX1_REPLAY_INCREMENTAL_FEATURES": os.getenv("GX1_REPLAY_INCREMENTAL_FEATURES") == "1",
        "GX1_REPLAY_NO_CSV": os.getenv("GX1_REPLAY_NO_CSV") == "1",
        "GX1_FEATURE_USE_NP_ROLLING": os.getenv("GX1_FEATURE_USE_NP_ROLLING") == "1",
        "OMP_NUM_THREADS": os.getenv("OMP_NUM_THREADS") == "1",
        "OPENBLAS_NUM_THREADS": os.getenv("OPENBLAS_NUM_THREADS") == "1",
        "MKL_NUM_THREADS": os.getenv("MKL_NUM_THREADS") == "1",
        "VECLIB_MAXIMUM_THREADS": os.getenv("VECLIB_MAXIMUM_THREADS") == "1",
        "NUMEXPR_NUM_THREADS": os.getenv("NUMEXPR_NUM_THREADS") == "1",
    }
    
    all_passed = all(checks.values())
    
    result = {
        "fast_path_enabled": all_passed,
        "checks": checks,
        "missing_checks": [k for k, v in checks.items() if not v],
    }
    
    if is_replay and not all_passed:
        missing = ", ".join(result["missing_checks"])
        error_msg = (
            f"FAST_PATH_NOT_ENABLED: Fast path is not enabled in replay mode. "
            f"Missing: {missing}. "
            f"This is a hard contract violation. "
            f"Set required environment variables before running replay."
        )
        log.error(error_msg)
        raise RuntimeError(error_msg)
    
    if not all_passed:
        missing = ", ".join(result["missing_checks"])
        log.warning(
            f"Fast path not fully enabled. Missing: {missing}. "
            f"This may impact replay performance."
        )
    else:
        log.debug("Fast path verification passed")
    
    return result



