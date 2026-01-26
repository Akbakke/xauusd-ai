"""
Platt Scaling for probability calibration.

Platt scaling fits a sigmoid function to transform raw probabilities
into calibrated probabilities. It's effective when the calibration
curve is roughly sigmoidal.
"""

import numpy as np
from typing import Any, Dict, Optional
from scipy.optimize import minimize
from scipy.special import expit  # sigmoid

from .calibrator_base import CalibratorBase


class PlattScaler(CalibratorBase):
    """
    Platt Scaling calibrator.
    
    Fits parameters A and B such that:
        p_calibrated = sigmoid(A * p_raw + B)
    
    This is equivalent to fitting a logistic regression on the raw
    probabilities to predict the true labels.
    """
    
    def __init__(self, name: str = "platt"):
        super().__init__(name=name)
        self.stats.calibrator_type = "platt"
        
        # Platt parameters
        self.A: float = 1.0  # Scale
        self.B: float = 0.0  # Shift
    
    def fit(
        self,
        p_raw: np.ndarray,
        y_true: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "PlattScaler":
        """
        Fit Platt scaling parameters.
        
        Uses maximum likelihood to find A and B that minimize
        cross-entropy loss.
        """
        p_raw = np.asarray(p_raw).flatten()
        y_true = np.asarray(y_true).flatten()
        
        if len(p_raw) != len(y_true):
            raise ValueError("p_raw and y_true must have same length")
        
        n = len(p_raw)
        self.stats.n_samples = n
        
        # Convert to log-odds (with clipping for numerical stability)
        eps = 1e-7
        p_clipped = np.clip(p_raw, eps, 1 - eps)
        log_odds = np.log(p_clipped / (1 - p_clipped))
        
        # Objective function: negative log-likelihood
        def neg_log_likelihood(params):
            A, B = params
            z = A * log_odds + B
            p = expit(z)
            p = np.clip(p, eps, 1 - eps)
            
            if sample_weight is not None:
                nll = -np.sum(sample_weight * (y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))
            else:
                nll = -np.sum(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
            
            return nll
        
        # Optimize
        result = minimize(
            neg_log_likelihood,
            x0=[1.0, 0.0],
            method="L-BFGS-B",
            bounds=[(0.01, 10.0), (-5.0, 5.0)],
        )
        
        self.A, self.B = result.x
        self.is_fitted = True
        
        # Compute stats
        p_cal = self.transform(p_raw)
        self._compute_stats(p_raw, p_cal, y_true)
        
        return self
    
    def transform(self, p_raw: np.ndarray) -> np.ndarray:
        """Transform raw probabilities using fitted Platt parameters."""
        if not self.is_fitted:
            raise RuntimeError("Calibrator not fitted. Call fit() first.")
        
        p_raw = np.asarray(p_raw)
        is_scalar = p_raw.ndim == 0
        p_raw = p_raw.flatten()
        
        # Convert to log-odds
        eps = 1e-7
        p_clipped = np.clip(p_raw, eps, 1 - eps)
        log_odds = np.log(p_clipped / (1 - p_clipped))
        
        # Apply Platt transformation
        z = self.A * log_odds + self.B
        p_cal = expit(z)
        
        if is_scalar:
            return float(p_cal[0])
        return p_cal
    
    def get_state(self) -> Dict[str, Any]:
        """Get calibrator state."""
        return {
            "A": self.A,
            "B": self.B,
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Set calibrator state."""
        self.A = state["A"]
        self.B = state["B"]
        self.is_fitted = True
