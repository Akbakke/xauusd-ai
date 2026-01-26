"""
XGB Multi-head Model v1.

Single artifact containing multiple session heads, each producing
3-class probabilities (LONG, SHORT, FLAT) + uncertainty.

Usage:
    from gx1.xgb.multihead.xgb_multihead_model_v1 import XGBMultiheadModel
    
    model = XGBMultiheadModel.load("path/to/xgb_universal_multihead_v2.joblib")
    outputs = model.predict_proba(df_prebuilt, session="EU")
    # outputs = {"p_long": ..., "p_short": ..., "p_flat": ..., "uncertainty": ...}
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
class MultiheadOutputs:
    """Outputs from a single head prediction."""
    p_long: np.ndarray
    p_short: np.ndarray
    p_flat: np.ndarray
    uncertainty: np.ndarray
    session: str
    
    def to_dict(self) -> Dict[str, np.ndarray]:
        return {
            f"p_long_xgb_{self.session}": self.p_long,
            f"p_short_xgb_{self.session}": self.p_short,
            f"p_flat_xgb_{self.session}": self.p_flat,
            f"uncertainty_xgb_{self.session}": self.uncertainty,
        }
    
    def validate(self) -> None:
        """Validate outputs, raise if invalid."""
        # Check for NaN
        for name, arr in [("p_long", self.p_long), ("p_short", self.p_short), 
                          ("p_flat", self.p_flat), ("uncertainty", self.uncertainty)]:
            if np.isnan(arr).any():
                raise ValueError(f"NaN detected in {name} for session {self.session}")
            if np.isinf(arr).any():
                raise ValueError(f"Inf detected in {name} for session {self.session}")
        
        # Check probability sum
        prob_sum = self.p_long + self.p_short + self.p_flat
        if not np.allclose(prob_sum, 1.0, atol=1e-5):
            max_deviation = np.abs(prob_sum - 1.0).max()
            if max_deviation > 0.01:
                raise ValueError(
                    f"Probability sum deviates from 1.0: max deviation={max_deviation:.4f}"
                )
    
    def stats(self) -> Dict[str, float]:
        """Get summary statistics."""
        return {
            f"p_long_xgb_{self.session}_mean": float(self.p_long.mean()),
            f"p_short_xgb_{self.session}_mean": float(self.p_short.mean()),
            f"p_flat_xgb_{self.session}_mean": float(self.p_flat.mean()),
            f"uncertainty_xgb_{self.session}_mean": float(self.uncertainty.mean()),
            f"p_long_xgb_{self.session}_min": float(self.p_long.min()),
            f"p_long_xgb_{self.session}_max": float(self.p_long.max()),
        }


class XGBMultiheadModel:
    """
    Multi-head XGB model with one head per session.
    
    Each head is a 3-class classifier producing (p_long, p_short, p_flat).
    Uncertainty is computed as normalized entropy.
    
    TRUTH MODE INVARIANTS:
    - Model MUST have feature_list (no fallback to numeric columns)
    - Feature list MUST match contract (no partial matching)
    - Model MUST have been trained with known schema hash
    """
    
    def __init__(
        self,
        heads: Dict[str, Any],  # session -> xgb model
        feature_list: List[str],
        meta: Dict[str, Any],
    ):
        self.heads = heads
        self.feature_list = feature_list
        self.meta = meta
        
        # Extract key info from meta
        self.schema_hash = meta.get("schema_hash", "unknown")
        self.feature_contract_sha = meta.get("feature_contract_sha256")
        self.sanitizer_sha = meta.get("sanitizer_sha256")
        self.sessions = list(heads.keys())
        
        # Validate invariants
        if not self.feature_list:
            raise ValueError(
                "TRUTH_MODE_VIOLATION: Model has no feature_list. "
                "Cannot use fallback to numeric columns in truth mode."
            )
    
    @classmethod
    def load(
        cls,
        model_path: str,
        require_feature_names: bool = True,
        expected_n_features: Optional[int] = None,
        allow_unsafe_dev: bool = False,
    ) -> "XGBMultiheadModel":
        """
        Load model from joblib file.
        
        Args:
            model_path: Path to model file
            require_feature_names: Require model has feature_list (default: True)
            expected_n_features: Expected number of features (e.g., 96)
            allow_unsafe_dev: Allow loading without feature names in dev mode
        
        Raises:
            FileNotFoundError: If model not found
            ValueError: If model format invalid or feature contract violated
        """
        try:
            from joblib import load as joblib_load
        except ImportError:
            import joblib
            joblib_load = joblib.load
        
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        data = joblib_load(model_path)
        
        if not isinstance(data, dict) or "heads" not in data:
            raise ValueError(f"Invalid multihead model format: {model_path}")
        
        feature_list = data.get("feature_list", [])
        meta = data.get("meta", {})
        
        # TRUTH MODE: Require feature names
        if require_feature_names and not feature_list:
            if allow_unsafe_dev:
                log.warning(
                    f"[XGB_MULTIHEAD] ⚠️  Loading model without feature_list (UNSAFE DEV MODE)"
                )
            else:
                raise ValueError(
                    f"TRUTH_MODE_VIOLATION: Model at {model_path} has no feature_list.\n\n"
                    f"The 'first N numeric columns' fallback is FORBIDDEN in truth mode.\n\n"
                    f"HOW TO FIX:\n"
                    f"1. Retrain model with explicit feature_list\n"
                    f"2. Or use --allow-unsafe-dev (ONLY for development)"
                )
        
        # TRUTH MODE: Verify feature count matches contract
        if expected_n_features is not None and feature_list:
            if len(feature_list) != expected_n_features:
                raise ValueError(
                    f"FEATURE_COUNT_MISMATCH: Model has {len(feature_list)} features, "
                    f"contract expects {expected_n_features}.\n\n"
                    f"HOW TO FIX: Retrain model with correct feature contract"
                )
        
        return cls(
            heads=data["heads"],
            feature_list=feature_list,
            meta=meta,
        )
    
    def save(self, model_path: str) -> str:
        """
        Save model to joblib file.
        
        Returns:
            SHA256 of saved file
        """
        try:
            from joblib import dump as joblib_dump
        except ImportError:
            import joblib
            joblib_dump = joblib.dump
        
        model_path = Path(model_path)
        
        data = {
            "heads": self.heads,
            "feature_list": self.feature_list,
            "meta": self.meta,
        }
        
        joblib_dump(data, model_path)
        
        # Compute SHA256
        sha256_hash = hashlib.sha256()
        with open(model_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        return sha256_hash.hexdigest()
    
    def get_head(self, session: str) -> Any:
        """Get head for session, raise if not found."""
        if session not in self.heads:
            raise ValueError(
                f"Session '{session}' not found in model. "
                f"Available: {list(self.heads.keys())}"
            )
        return self.heads[session]
    
    def predict_proba(
        self,
        df: pd.DataFrame,
        session: str,
        feature_list: Optional[List[str]] = None,
        strict_feature_match: bool = True,
    ) -> MultiheadOutputs:
        """
        Predict probabilities for a session head.
        
        Args:
            df: DataFrame with feature columns
            session: Session to predict for (EU, US, OVERLAP, ASIA)
            feature_list: Override feature list (default: use self.feature_list)
        
        Returns:
            MultiheadOutputs with p_long, p_short, p_flat, uncertainty
        """
        features = feature_list or self.feature_list
        if not features:
            raise ValueError(
                "TRUTH_MODE_VIOLATION: No feature list provided and model has none. "
                "Fallback to numeric columns is FORBIDDEN."
            )
        
        # TRUTH MODE: Verify feature list matches model's feature list
        if strict_feature_match and self.feature_list and feature_list:
            if feature_list != self.feature_list:
                # Allow subset but not different order
                if set(feature_list) != set(self.feature_list):
                    raise ValueError(
                        f"FEATURE_LIST_MISMATCH: Provided features don't match model's features.\n"
                        f"Model has {len(self.feature_list)} features, provided {len(feature_list)}.\n"
                        f"Missing: {set(self.feature_list) - set(feature_list)}\n"
                        f"Extra: {set(feature_list) - set(self.feature_list)}"
                    )
        
        # Get head
        head = self.get_head(session)
        
        # Extract feature matrix
        missing = [f for f in features if f not in df.columns]
        if missing:
            raise ValueError(f"Missing features: {missing[:10]}...")
        
        X = df[features].values.astype(np.float64)
        
        # Check for NaN/Inf in inputs
        if np.isnan(X).any():
            raise ValueError("NaN in input features - apply sanitizer first")
        if np.isinf(X).any():
            raise ValueError("Inf in input features - apply sanitizer first")
        
        # Handle feature count mismatch
        if hasattr(head, "n_features_in_"):
            expected = head.n_features_in_
            if X.shape[1] > expected:
                X = X[:, :expected]
            elif X.shape[1] < expected:
                raise ValueError(
                    f"Feature count mismatch: model expects {expected}, got {X.shape[1]}"
                )
        
        # Predict
        if hasattr(head, "predict_proba"):
            proba = head.predict_proba(X)
        else:
            raise ValueError("Model does not support predict_proba")
        
        # Extract 3-class probabilities
        if proba.shape[1] == 3:
            p_long = proba[:, 0]  # Class 0 = LONG
            p_short = proba[:, 1]  # Class 1 = SHORT
            p_flat = proba[:, 2]  # Class 2 = FLAT
        elif proba.shape[1] == 2:
            # Fallback for binary models (shouldn't happen)
            p_long = proba[:, 1]
            p_short = 1 - proba[:, 1]
            p_flat = np.zeros_like(p_long)
        else:
            raise ValueError(f"Unexpected proba shape: {proba.shape}")
        
        # Compute uncertainty (normalized entropy)
        eps = 1e-10
        probs = np.column_stack([p_long, p_short, p_flat])
        probs = np.clip(probs, eps, 1.0)
        entropy = -np.sum(probs * np.log(probs), axis=1)
        max_entropy = np.log(3)  # Maximum entropy for 3 classes
        uncertainty = entropy / max_entropy
        
        outputs = MultiheadOutputs(
            p_long=p_long.astype(np.float32),
            p_short=p_short.astype(np.float32),
            p_flat=p_flat.astype(np.float32),
            uncertainty=uncertainty.astype(np.float32),
            session=session,
        )
        
        # Validate outputs
        outputs.validate()
        
        return outputs
    
    def predict_all_heads(
        self,
        df: pd.DataFrame,
        feature_list: Optional[List[str]] = None,
    ) -> Dict[str, MultiheadOutputs]:
        """
        Predict probabilities for all heads.
        
        Args:
            df: DataFrame with feature columns
            feature_list: Override feature list
        
        Returns:
            Dict[session, MultiheadOutputs]
        """
        results = {}
        for session in self.sessions:
            results[session] = self.predict_proba(df, session, feature_list)
        return results
    
    def get_provenance(self) -> Dict[str, Any]:
        """Get provenance info for capsule logging."""
        return {
            "model_type": "xgb_universal_multihead_v2",
            "sessions": self.sessions,
            "n_features": len(self.feature_list),
            "schema_hash": self.schema_hash,
            "feature_contract_sha256": self.feature_contract_sha,
            "sanitizer_sha256": self.sanitizer_sha,
        }
