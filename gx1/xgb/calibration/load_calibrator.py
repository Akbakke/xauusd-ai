"""
Calibrator loading utilities.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .calibrator_base import CalibratorBase, CalibrationStats


def load_calibrator(path: Path) -> CalibratorBase:
    """
    Load a calibrator from file.
    
    Args:
        path: Path to calibrator file (.pkl or .joblib)
        
    Returns:
        Loaded calibrator instance
    """
    return CalibratorBase.load(path)


def get_calibrator_metadata(path: Path) -> Dict[str, Any]:
    """
    Get calibrator metadata without loading full model.
    
    Args:
        path: Path to calibrator file
        
    Returns:
        Metadata dict with sha, type, stats summary
    """
    import joblib
    
    state = joblib.load(path)
    
    stats = state.get("stats", {})
    
    return {
        "sha": state.get("sha", ""),
        "calibrator_type": stats.get("calibrator_type", "unknown"),
        "n_samples": stats.get("n_samples", 0),
        "trained_at": stats.get("trained_at", ""),
        "brier_before": stats.get("brier_before", 0),
        "brier_after": stats.get("brier_after", 0),
        "ece_before": stats.get("ece_before", 0),
        "ece_after": stats.get("ece_after", 0),
        "input_mean": stats.get("input_mean", 0),
        "input_std": stats.get("input_std", 0),
        "output_mean": stats.get("output_mean", 0),
        "output_std": stats.get("output_std", 0),
    }


def validate_calibrator(
    calibrator: CalibratorBase,
    expected_sha: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Validate a loaded calibrator.
    
    Args:
        calibrator: Loaded calibrator
        expected_sha: Expected SHA (optional)
        
    Returns:
        (is_valid, error_message)
    """
    if not calibrator.is_fitted:
        return False, "Calibrator is not fitted"
    
    if expected_sha is not None:
        actual_sha = calibrator.compute_sha()
        if actual_sha != expected_sha:
            return False, f"SHA mismatch: expected {expected_sha}, got {actual_sha}"
    
    return True, ""
