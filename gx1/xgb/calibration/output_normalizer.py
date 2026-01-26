"""
XGB Output Normalization and Clipping.

Provides OOD damping through quantile clipping and z-score normalization.
"""

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class NormalizationStats:
    """Statistics for normalization."""
    
    # Training data info
    n_samples: int = 0
    years_included: List[int] = field(default_factory=list)
    
    # Per-channel stats
    channel_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Metadata
    trained_at: str = ""
    normalizer_type: str = ""
    normalizer_sha: str = ""


class QuantileClipper:
    """
    Clips XGB outputs to training quantiles.
    
    This prevents extreme OOD values from affecting downstream models.
    """
    
    def __init__(
        self,
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
    ):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        
        self.is_fitted = False
        self.stats = NormalizationStats(normalizer_type="quantile_clip")
        
        # Per-channel bounds
        self._bounds: Dict[str, Tuple[float, float]] = {}
    
    def fit(
        self,
        data: Dict[str, np.ndarray],
        years: Optional[List[int]] = None,
    ) -> "QuantileClipper":
        """
        Fit clipper on training data.
        
        Args:
            data: Dict mapping channel names to value arrays
            years: List of years included in training
        """
        total_samples = 0
        
        for channel, values in data.items():
            values = np.asarray(values).flatten()
            valid = values[~(np.isnan(values) | np.isinf(values))]
            
            if len(valid) == 0:
                self._bounds[channel] = (0.0, 1.0)
                continue
            
            lower = float(np.percentile(valid, self.lower_quantile * 100))
            upper = float(np.percentile(valid, self.upper_quantile * 100))
            
            self._bounds[channel] = (lower, upper)
            
            # Store stats
            self.stats.channel_stats[channel] = {
                "lower_bound": lower,
                "upper_bound": upper,
                "mean": float(np.mean(valid)),
                "std": float(np.std(valid)),
                "min": float(np.min(valid)),
                "max": float(np.max(valid)),
            }
            
            total_samples = max(total_samples, len(valid))
        
        self.stats.n_samples = total_samples
        self.stats.years_included = years or []
        self.stats.trained_at = datetime.now().isoformat()
        
        self.is_fitted = True
        self._compute_sha()
        
        return self
    
    def transform(
        self,
        channel: str,
        value: float,
    ) -> float:
        """
        Clip a single value.
        
        Args:
            channel: Channel name (e.g., "p_long_xgb")
            value: Raw value to clip
            
        Returns:
            Clipped value
        """
        if not self.is_fitted:
            raise RuntimeError("Clipper not fitted. Call fit() first.")
        
        if channel not in self._bounds:
            return value
        
        lower, upper = self._bounds[channel]
        return float(np.clip(value, lower, upper))
    
    def transform_batch(
        self,
        data: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        Clip multiple channels.
        
        Args:
            data: Dict mapping channel names to value arrays
            
        Returns:
            Dict with clipped values
        """
        result = {}
        
        for channel, values in data.items():
            if channel not in self._bounds:
                result[channel] = values
                continue
            
            lower, upper = self._bounds[channel]
            result[channel] = np.clip(values, lower, upper)
        
        return result
    
    def get_bounds(self, channel: str) -> Tuple[float, float]:
        """Get bounds for a channel."""
        return self._bounds.get(channel, (0.0, 1.0))
    
    def _compute_sha(self) -> str:
        """Compute SHA of clipper state."""
        state_str = json.dumps(self._bounds, sort_keys=True)
        sha = hashlib.sha256(state_str.encode()).hexdigest()[:16]
        self.stats.normalizer_sha = sha
        return sha
    
    def save(self, path: Path) -> None:
        """Save clipper to file."""
        import joblib
        
        state = {
            "lower_quantile": self.lower_quantile,
            "upper_quantile": self.upper_quantile,
            "bounds": self._bounds,
            "stats": asdict(self.stats),
            "sha": self.stats.normalizer_sha,
        }
        
        joblib.dump(state, path)
    
    @classmethod
    def load(cls, path: Path) -> "QuantileClipper":
        """Load clipper from file."""
        import joblib
        
        state = joblib.load(path)
        
        instance = cls(
            lower_quantile=state["lower_quantile"],
            upper_quantile=state["upper_quantile"],
        )
        instance._bounds = state["bounds"]
        instance.stats = NormalizationStats(**state["stats"])
        instance.is_fitted = True
        
        return instance


class XGBOutputNormalizer:
    """
    Combined normalizer for XGB outputs.
    
    Applies calibration + quantile clipping in sequence.
    """
    
    def __init__(
        self,
        calibrator: Optional[Any] = None,  # CalibratorBase
        clipper: Optional[QuantileClipper] = None,
    ):
        self.calibrator = calibrator
        self.clipper = clipper
    
    def transform(
        self,
        channel: str,
        value: float,
    ) -> Tuple[float, float]:
        """
        Transform a single value.
        
        Args:
            channel: Channel name
            value: Raw XGB output
            
        Returns:
            (raw_value, normalized_value)
        """
        raw = value
        
        # Step 1: Calibration (if available and channel is probability)
        if self.calibrator is not None and channel in ["p_long_xgb", "p_hat_xgb"]:
            value = self.calibrator.transform(np.array([value]))[0]
        
        # Step 2: Clipping (if available)
        if self.clipper is not None:
            value = self.clipper.transform(channel, value)
        
        return raw, float(value)
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get normalizer metadata for logging."""
        return {
            "calibrator_sha": self.calibrator.stats.calibrator_sha if self.calibrator else None,
            "calibrator_type": self.calibrator.stats.calibrator_type if self.calibrator else None,
            "clipper_sha": self.clipper.stats.normalizer_sha if self.clipper else None,
            "clipper_bounds": self.clipper._bounds if self.clipper else None,
        }
