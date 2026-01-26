"""
Base class for XGB output calibrators.

Calibrators transform raw XGB probabilities into calibrated probabilities
that are consistent across different years/regimes.
"""

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class CalibrationStats:
    """Statistics about calibration training data and results."""
    
    # Training data info
    n_samples: int = 0
    years_included: List[int] = field(default_factory=list)
    sessions_included: List[str] = field(default_factory=list)
    
    # Input distribution
    input_mean: float = 0.0
    input_std: float = 0.0
    input_min: float = 0.0
    input_max: float = 0.0
    input_p1: float = 0.0
    input_p5: float = 0.0
    input_p50: float = 0.0
    input_p95: float = 0.0
    input_p99: float = 0.0
    
    # Output distribution
    output_mean: float = 0.0
    output_std: float = 0.0
    output_min: float = 0.0
    output_max: float = 0.0
    output_p1: float = 0.0
    output_p5: float = 0.0
    output_p50: float = 0.0
    output_p95: float = 0.0
    output_p99: float = 0.0
    
    # Calibration quality
    brier_before: float = 0.0
    brier_after: float = 0.0
    ece_before: float = 0.0  # Expected Calibration Error
    ece_after: float = 0.0
    
    # Metadata
    trained_at: str = ""
    calibrator_type: str = ""
    calibrator_sha: str = ""


class CalibratorBase(ABC):
    """
    Base class for probability calibrators.
    
    Calibrators transform raw XGB probabilities p_raw ∈ [0,1] into
    calibrated probabilities p_cal ∈ [0,1] that better reflect true
    outcome probabilities across different data distributions.
    """
    
    def __init__(self, name: str = "base"):
        self.name = name
        self.is_fitted = False
        self.stats = CalibrationStats(calibrator_type=name)
        self._fit_data_hash: str = ""
    
    @abstractmethod
    def fit(
        self,
        p_raw: np.ndarray,
        y_true: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "CalibratorBase":
        """
        Fit the calibrator on training data.
        
        Args:
            p_raw: Raw probabilities from XGB, shape (n_samples,)
            y_true: True binary labels, shape (n_samples,)
            sample_weight: Optional sample weights
            
        Returns:
            self
        """
        pass
    
    @abstractmethod
    def transform(self, p_raw: np.ndarray) -> np.ndarray:
        """
        Transform raw probabilities to calibrated probabilities.
        
        Args:
            p_raw: Raw probabilities, shape (n_samples,) or scalar
            
        Returns:
            Calibrated probabilities, same shape as input
        """
        pass
    
    def fit_transform(
        self,
        p_raw: np.ndarray,
        y_true: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(p_raw, y_true, sample_weight)
        return self.transform(p_raw)
    
    def _compute_stats(
        self,
        p_raw: np.ndarray,
        p_cal: np.ndarray,
        y_true: Optional[np.ndarray] = None,
    ) -> None:
        """Compute calibration statistics."""
        # Input stats
        self.stats.input_mean = float(np.mean(p_raw))
        self.stats.input_std = float(np.std(p_raw))
        self.stats.input_min = float(np.min(p_raw))
        self.stats.input_max = float(np.max(p_raw))
        self.stats.input_p1 = float(np.percentile(p_raw, 1))
        self.stats.input_p5 = float(np.percentile(p_raw, 5))
        self.stats.input_p50 = float(np.percentile(p_raw, 50))
        self.stats.input_p95 = float(np.percentile(p_raw, 95))
        self.stats.input_p99 = float(np.percentile(p_raw, 99))
        
        # Output stats
        self.stats.output_mean = float(np.mean(p_cal))
        self.stats.output_std = float(np.std(p_cal))
        self.stats.output_min = float(np.min(p_cal))
        self.stats.output_max = float(np.max(p_cal))
        self.stats.output_p1 = float(np.percentile(p_cal, 1))
        self.stats.output_p5 = float(np.percentile(p_cal, 5))
        self.stats.output_p50 = float(np.percentile(p_cal, 50))
        self.stats.output_p95 = float(np.percentile(p_cal, 95))
        self.stats.output_p99 = float(np.percentile(p_cal, 99))
        
        # Quality metrics (if labels available)
        if y_true is not None:
            # Brier score
            self.stats.brier_before = float(np.mean((p_raw - y_true) ** 2))
            self.stats.brier_after = float(np.mean((p_cal - y_true) ** 2))
            
            # Expected Calibration Error (ECE)
            self.stats.ece_before = self._compute_ece(p_raw, y_true)
            self.stats.ece_after = self._compute_ece(p_cal, y_true)
        
        self.stats.trained_at = datetime.now().isoformat()
    
    def _compute_ece(
        self,
        p: np.ndarray,
        y: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Compute Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            mask = (p >= bin_boundaries[i]) & (p < bin_boundaries[i + 1])
            if np.sum(mask) > 0:
                bin_acc = np.mean(y[mask])
                bin_conf = np.mean(p[mask])
                bin_weight = np.sum(mask) / len(p)
                ece += bin_weight * abs(bin_acc - bin_conf)
        
        return float(ece)
    
    def compute_sha(self) -> str:
        """Compute SHA256 of calibrator state."""
        state_str = json.dumps(self.get_state(), sort_keys=True, default=str)
        sha = hashlib.sha256(state_str.encode()).hexdigest()[:16]
        self.stats.calibrator_sha = sha
        return sha
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get calibrator state for serialization."""
        pass
    
    @abstractmethod
    def set_state(self, state: Dict[str, Any]) -> None:
        """Set calibrator state from deserialization."""
        pass
    
    def save(self, path: Path) -> None:
        """Save calibrator to file."""
        import joblib
        
        state = {
            "name": self.name,
            "is_fitted": self.is_fitted,
            "stats": asdict(self.stats),
            "calibrator_state": self.get_state(),
            "sha": self.compute_sha(),
        }
        
        joblib.dump(state, path)
    
    @classmethod
    def load(cls, path: Path) -> "CalibratorBase":
        """Load calibrator from file."""
        import joblib
        
        state = joblib.load(path)
        
        # Create instance based on type
        calibrator_type = state.get("stats", {}).get("calibrator_type", "platt")
        
        if calibrator_type == "platt":
            from .platt_scaler import PlattScaler
            instance = PlattScaler()
        elif calibrator_type == "isotonic":
            from .isotonic_scaler import IsotonicScaler
            instance = IsotonicScaler()
        else:
            raise ValueError(f"Unknown calibrator type: {calibrator_type}")
        
        instance.name = state["name"]
        instance.is_fitted = state["is_fitted"]
        instance.stats = CalibrationStats(**state["stats"])
        instance.set_state(state["calibrator_state"])
        
        return instance
