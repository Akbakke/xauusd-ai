"""
XGB Input Sanitizer - SSoT for feature sanitization.

Ensures XGB inputs are clean, bounded, and in correct order.
Hard-fails on NaN/Inf in truth mode to prevent garbage-in-garbage-out.

Usage:
    from gx1.xgb.preprocess.xgb_input_sanitizer import XGBInputSanitizer
    
    sanitizer = XGBInputSanitizer.from_config("gx1/xgb/contracts/xgb_input_sanitizer_v1.json")
    X_clean, stats = sanitizer.sanitize(df, feature_list)
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class SanitizeStats:
    """Statistics from sanitization run."""
    n_rows: int = 0
    n_features: int = 0
    n_nan_total: int = 0
    n_inf_total: int = 0
    n_clipped_total: int = 0
    n_clipped_by_feature: Dict[str, int] = field(default_factory=dict)
    n_nan_by_feature: Dict[str, int] = field(default_factory=dict)
    n_inf_by_feature: Dict[str, int] = field(default_factory=dict)
    clip_rate_pct: float = 0.0
    top_clipped_features: List[Tuple[str, int, float]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_features": self.n_features,
            "n_nan_total": self.n_nan_total,
            "n_inf_total": self.n_inf_total,
            "n_clipped_total": self.n_clipped_total,
            "clip_rate_pct": self.clip_rate_pct,
            "top_clipped_features": [
                {"feature": f, "count": c, "pct": p}
                for f, c, p in self.top_clipped_features[:10]
            ],
        }


class XGBInputSanitizer:
    """
    Sanitizes XGB inputs by:
    1. Checking for NaN/Inf (hard-fail in truth mode)
    2. Clipping values to bounds (quantile or fixed)
    3. Ensuring correct feature order
    """
    
    def __init__(
        self,
        feature_list: List[str],
        bounds: Dict[str, Tuple[float, float]],
        clip_method: str = "quantile",
        hard_fail_on_nan: bool = True,
        hard_fail_on_inf: bool = True,
        hard_fail_abs_max: float = 1e9,
        max_clip_rate_pct: float = 10.0,
        config_sha256: Optional[str] = None,
    ):
        """
        Initialize sanitizer.
        
        Args:
            feature_list: Ordered list of feature names
            bounds: Dict of feature -> (lower, upper) bounds
            clip_method: "quantile" or "fixed"
            hard_fail_on_nan: Raise on NaN values
            hard_fail_on_inf: Raise on Inf values
            hard_fail_abs_max: Fail if any value exceeds this absolute max
            max_clip_rate_pct: Warn/fail if clip rate exceeds this
            config_sha256: SHA256 of config file for provenance
        """
        self.feature_list = feature_list
        self.bounds = bounds
        self.clip_method = clip_method
        self.hard_fail_on_nan = hard_fail_on_nan
        self.hard_fail_on_inf = hard_fail_on_inf
        self.hard_fail_abs_max = hard_fail_abs_max
        self.max_clip_rate_pct = max_clip_rate_pct
        self.config_sha256 = config_sha256
    
    @classmethod
    def from_config(cls, config_path: str) -> "XGBInputSanitizer":
        """Load sanitizer from config file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Sanitizer config not found: {config_path}")
        
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Compute SHA256 of config file
        with open(config_path, "rb") as f:
            config_sha256 = hashlib.sha256(f.read()).hexdigest()
        
        # Load feature list
        feature_list = config.get("feature_list", [])
        if not feature_list:
            # Try to load from separate contract file
            feature_contract_path = config.get("feature_contract_path")
            if feature_contract_path:
                with open(feature_contract_path, "r") as f:
                    feature_contract = json.load(f)
                feature_list = feature_contract.get("features", [])
        
        # Load bounds
        bounds = {}
        bounds_config = config.get("bounds", {})
        for feature, bound_info in bounds_config.items():
            if isinstance(bound_info, list) and len(bound_info) == 2:
                bounds[feature] = tuple(bound_info)
            elif isinstance(bound_info, dict):
                bounds[feature] = (bound_info.get("lower", -np.inf), bound_info.get("upper", np.inf))
        
        return cls(
            feature_list=feature_list,
            bounds=bounds,
            clip_method=config.get("clip_method", "quantile"),
            hard_fail_on_nan=config.get("hard_fail_on_nan", True),
            hard_fail_on_inf=config.get("hard_fail_on_inf", True),
            hard_fail_abs_max=config.get("hard_fail_abs_max", 1e9),
            max_clip_rate_pct=config.get("max_clip_rate_pct", 10.0),
            config_sha256=config_sha256,
        )
    
    def sanitize(
        self,
        df: pd.DataFrame,
        feature_list: Optional[List[str]] = None,
        allow_nan_fill: bool = False,
        nan_fill_value: float = 0.0,
    ) -> Tuple[np.ndarray, SanitizeStats]:
        """
        Sanitize DataFrame to numpy array.
        
        Args:
            df: Input DataFrame with feature columns
            feature_list: Override feature list (default: use self.feature_list)
            allow_nan_fill: If True, fill NaN with nan_fill_value instead of failing
            nan_fill_value: Value to fill NaN with (if allow_nan_fill=True)
        
        Returns:
            Tuple of (sanitized numpy array, statistics)
        
        Raises:
            ValueError: If NaN/Inf found and hard_fail enabled
            KeyError: If required features missing from DataFrame
        """
        features = feature_list or self.feature_list
        if not features:
            raise ValueError("No feature list provided")
        
        stats = SanitizeStats(
            n_rows=len(df),
            n_features=len(features),
        )
        
        # Check for missing features
        missing = [f for f in features if f not in df.columns]
        if missing:
            raise KeyError(
                f"Missing {len(missing)} features in DataFrame: {missing[:10]}..."
            )
        
        # Extract feature matrix in correct order
        X = df[features].values.astype(np.float64)
        
        # Check for NaN
        nan_mask = np.isnan(X)
        if nan_mask.any():
            stats.n_nan_total = int(nan_mask.sum())
            for i, feature in enumerate(features):
                n_nan = int(nan_mask[:, i].sum())
                if n_nan > 0:
                    stats.n_nan_by_feature[feature] = n_nan
            
            if self.hard_fail_on_nan and not allow_nan_fill:
                top_nan = sorted(stats.n_nan_by_feature.items(), key=lambda x: -x[1])[:5]
                raise ValueError(
                    f"SANITIZER_NAN_FAIL: Found {stats.n_nan_total} NaN values. "
                    f"Top features: {top_nan}"
                )
            
            if allow_nan_fill:
                X = np.where(nan_mask, nan_fill_value, X)
                log.warning(f"[SANITIZER] Filled {stats.n_nan_total} NaN values with {nan_fill_value}")
        
        # Check for Inf
        inf_mask = np.isinf(X)
        if inf_mask.any():
            stats.n_inf_total = int(inf_mask.sum())
            for i, feature in enumerate(features):
                n_inf = int(inf_mask[:, i].sum())
                if n_inf > 0:
                    stats.n_inf_by_feature[feature] = n_inf
            
            if self.hard_fail_on_inf:
                top_inf = sorted(stats.n_inf_by_feature.items(), key=lambda x: -x[1])[:5]
                raise ValueError(
                    f"SANITIZER_INF_FAIL: Found {stats.n_inf_total} Inf values. "
                    f"Top features: {top_inf}"
                )
        
        # Apply clipping FIRST (before absurd value check)
        for i, feature in enumerate(features):
            if feature in self.bounds:
                lower, upper = self.bounds[feature]
                col = X[:, i]
                
                # Count clips
                n_clip_low = int((col < lower).sum())
                n_clip_high = int((col > upper).sum())
                n_clipped = n_clip_low + n_clip_high
                
                if n_clipped > 0:
                    stats.n_clipped_by_feature[feature] = n_clipped
                    stats.n_clipped_total += n_clipped
                    X[:, i] = np.clip(col, lower, upper)
        
        # Check for absurdly large values AFTER clipping
        # (This catches features that weren't in bounds config)
        abs_max = np.abs(X).max()
        if abs_max > self.hard_fail_abs_max:
            # Find which feature has the max
            max_idx = np.unravel_index(np.abs(X).argmax(), X.shape)
            max_feature = features[max_idx[1]]
            max_value = X[max_idx]
            raise ValueError(
                f"SANITIZER_ABSURD_VALUE_FAIL: Value {max_value:.2e} in feature '{max_feature}' "
                f"exceeds hard_fail_abs_max={self.hard_fail_abs_max:.2e}. "
                f"This feature may not have bounds defined."
            )
        
        # Compute clip rate
        total_values = stats.n_rows * stats.n_features
        if total_values > 0:
            stats.clip_rate_pct = (stats.n_clipped_total / total_values) * 100
        
        # Top clipped features
        stats.top_clipped_features = [
            (feature, count, (count / stats.n_rows) * 100)
            for feature, count in sorted(
                stats.n_clipped_by_feature.items(),
                key=lambda x: -x[1]
            )[:10]
        ]
        
        # Log summary
        if stats.n_clipped_total > 0:
            log.info(
                f"[SANITIZER] Clipped {stats.n_clipped_total} values ({stats.clip_rate_pct:.2f}%) "
                f"across {len(stats.n_clipped_by_feature)} features"
            )
        
        return X, stats
    
    def get_provenance(self) -> Dict[str, Any]:
        """Get provenance info for capsule logging."""
        return {
            "sanitizer_config_sha256": self.config_sha256,
            "clip_method": self.clip_method,
            "n_features": len(self.feature_list),
            "n_bounded_features": len(self.bounds),
            "hard_fail_abs_max": self.hard_fail_abs_max,
        }
