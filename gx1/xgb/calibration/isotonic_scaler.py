"""
Isotonic Regression for probability calibration.

Isotonic regression fits a non-decreasing function to transform raw
probabilities into calibrated probabilities. It's more flexible than
Platt scaling but requires more data.
"""

import numpy as np
from typing import Any, Dict, Optional

from .calibrator_base import CalibratorBase


class IsotonicScaler(CalibratorBase):
    """
    Isotonic Regression calibrator.
    
    Fits a non-parametric, monotonically increasing function that
    maps raw probabilities to calibrated probabilities.
    
    Uses sklearn's IsotonicRegression internally.
    """
    
    def __init__(self, name: str = "isotonic"):
        super().__init__(name=name)
        self.stats.calibrator_type = "isotonic"
        
        # Isotonic model
        self._isotonic = None
        
        # Fallback for OOD values
        self._x_min: float = 0.0
        self._x_max: float = 1.0
        self._y_min: float = 0.0
        self._y_max: float = 1.0
    
    def fit(
        self,
        p_raw: np.ndarray,
        y_true: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "IsotonicScaler":
        """
        Fit isotonic regression.
        """
        try:
            from sklearn.isotonic import IsotonicRegression
        except ImportError as e:
            raise ImportError(f"sklearn required for IsotonicScaler: {e}")
        
        p_raw = np.asarray(p_raw).flatten()
        y_true = np.asarray(y_true).flatten()
        
        if len(p_raw) != len(y_true):
            raise ValueError("p_raw and y_true must have same length")
        
        n = len(p_raw)
        self.stats.n_samples = n
        
        # Fit isotonic regression
        self._isotonic = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            out_of_bounds="clip",
        )
        
        if sample_weight is not None:
            self._isotonic.fit(p_raw, y_true, sample_weight=sample_weight)
        else:
            self._isotonic.fit(p_raw, y_true)
        
        # Store bounds for OOD handling
        self._x_min = float(np.min(p_raw))
        self._x_max = float(np.max(p_raw))
        self._y_min = float(self._isotonic.transform([self._x_min])[0])
        self._y_max = float(self._isotonic.transform([self._x_max])[0])
        
        self.is_fitted = True
        
        # Compute stats
        p_cal = self.transform(p_raw)
        self._compute_stats(p_raw, p_cal, y_true)
        
        return self
    
    def transform(self, p_raw: np.ndarray) -> np.ndarray:
        """Transform raw probabilities using fitted isotonic regression."""
        if not self.is_fitted:
            raise RuntimeError("Calibrator not fitted. Call fit() first.")
        
        p_raw = np.asarray(p_raw)
        is_scalar = p_raw.ndim == 0
        p_raw = p_raw.flatten()
        
        # Transform using isotonic regression
        p_cal = self._isotonic.transform(p_raw)
        
        # Ensure output is in [0, 1]
        p_cal = np.clip(p_cal, 0.0, 1.0)
        
        if is_scalar:
            return float(p_cal[0])
        return p_cal
    
    def get_state(self) -> Dict[str, Any]:
        """Get calibrator state."""
        # Serialize isotonic regression
        if self._isotonic is not None:
            isotonic_state = {
                "X_thresholds_": self._isotonic.X_thresholds_.tolist() if hasattr(self._isotonic, "X_thresholds_") else [],
                "y_thresholds_": self._isotonic.y_thresholds_.tolist() if hasattr(self._isotonic, "y_thresholds_") else [],
                "X_min_": float(self._isotonic.X_min_) if hasattr(self._isotonic, "X_min_") else 0.0,
                "X_max_": float(self._isotonic.X_max_) if hasattr(self._isotonic, "X_max_") else 1.0,
            }
        else:
            isotonic_state = {}
        
        return {
            "isotonic": isotonic_state,
            "x_min": self._x_min,
            "x_max": self._x_max,
            "y_min": self._y_min,
            "y_max": self._y_max,
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Set calibrator state."""
        try:
            from sklearn.isotonic import IsotonicRegression
        except ImportError as e:
            raise ImportError(f"sklearn required for IsotonicScaler: {e}")
        
        isotonic_state = state.get("isotonic", {})
        
        self._isotonic = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            out_of_bounds="clip",
        )
        
        if isotonic_state:
            self._isotonic.X_thresholds_ = np.array(isotonic_state.get("X_thresholds_", []))
            self._isotonic.y_thresholds_ = np.array(isotonic_state.get("y_thresholds_", []))
            self._isotonic.X_min_ = isotonic_state.get("X_min_", 0.0)
            self._isotonic.X_max_ = isotonic_state.get("X_max_", 1.0)
        
        self._x_min = state.get("x_min", 0.0)
        self._x_max = state.get("x_max", 1.0)
        self._y_min = state.get("y_min", 0.0)
        self._y_max = state.get("y_max", 1.0)
        
        self.is_fitted = True
