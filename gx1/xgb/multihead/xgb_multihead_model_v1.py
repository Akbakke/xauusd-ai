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
import logging
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from gx1.contracts.signal_bridge_v1 import XGB_PROB_FIELDS_ORDERED

log = logging.getLogger(__name__)

def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


def proba_to_signal_bridge_v1(proba: np.ndarray) -> np.ndarray:
    """
    Normalize XGB predict_proba output to canonical 7-dim XGB_SIGNAL_BRIDGE_V1.

    Allowed shapes:
    - (N, 7): already bridge; validated for finiteness and basic bounds
    - (N, 3): map to bridge:
        0 p_long  = proba[:,0]
        1 p_short = proba[:,1]
        2 p_flat  = proba[:,2]
        3 p_hat   = max(row)
        4 uncertainty_score = 1 - p_hat
        5 margin_top1_top2  = top1 - top2 (row-wise)
        6 entropy = -sum(p_i * log(p_i)), eps=1e-12
    - (N, 2): map to bridge:
        0 p_long  = proba[:,0]
        1 p_short = proba[:,1]
        2 p_flat  = 0.0
        3 p_hat   = max(row)
        4 uncertainty_score = 1 - p_hat
        5 margin_top1_top2  = top1 - top2 (row-wise)
        6 entropy = -sum(p_i * log(p_i)), eps=1e-12
    Any other shape → RuntimeError [XGB_PROBA_DIM_MISMATCH].
    """
    arr = np.asarray(proba)
    if arr.ndim != 2:
        raise RuntimeError(f"[XGB_PROBA_DIM_MISMATCH] Expected 2-D proba, got shape={arr.shape}")
    n_rows, n_cols = arr.shape
    if n_cols not in (2, 3, 7):
        raise RuntimeError(
            f"[XGB_PROBA_DIM_MISMATCH] Expected proba dim in {{2,3,7}}, got {n_cols} with shape={arr.shape}"
        )
    if not np.isfinite(arr).all():
        raise RuntimeError("[XGB_PROBA_INVALID] Non-finite values in proba output")

    if n_cols == 7:
        first3 = arr[:, :3]
        if (first3 < -1e-6).any() or (first3 > 1 + 1e-6).any():
            raise RuntimeError("[XGB_PROBA_INVALID] Bridge proba outside [0,1]")
        return arr.astype(np.float32, copy=False)

    # n_cols == 3 -> build bridge
    if n_cols == 3:
        p_long = arr[:, 0]
        p_short = arr[:, 1]
        p_flat = arr[:, 2]
    else:
        # n_cols == 2 -> build bridge, no FLAT
        log.info("[XGB_BRIDGE_2CLASS_MODE] enabled=True")
        p_long = arr[:, 0]
        p_short = arr[:, 1]
        p_flat = np.zeros_like(p_long)
    p_hat = np.maximum.reduce([p_long, p_short, p_flat])
    if n_cols == 2:
        top_two = np.sort(arr, axis=1)[:, ::-1][:, :2]
    else:
        top_two = np.sort(arr, axis=1)[:, ::-1][:, :2]
    margin_top1_top2 = top_two[:, 0] - top_two[:, 1]
    eps = 1e-12
    probs = np.clip(arr[:, :3] if n_cols == 3 else np.column_stack([p_long, p_short, p_flat]), eps, 1.0)
    entropy = -np.sum(probs * np.log(probs), axis=1)
    bridge = np.column_stack(
        [
            p_long,
            p_short,
            p_flat,
            p_hat,
            1.0 - p_hat,
            margin_top1_top2,
            entropy,
        ]
    ).astype(np.float32)
    return bridge


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
        calibrators: Optional[Dict[str, Any]] = None,
        calibration_method: Optional[str] = None,
    ):
        self.heads = heads
        self.feature_list = feature_list
        self.meta = meta
        self.calibrators = calibrators or {}
        self.calibration_method = calibration_method
        
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
        
        calibrators = None
        calibration_method = None
        if _bool_env("GX1_XGB_HEAD_CALIBRATE", False):
            calibration_method = (
                meta.get("calibration", {}).get("method")
                or os.getenv("GX1_XGB_CALIBRATOR", "isotonic")
            ).lower()
            calib_path = model_path.parent / "CALIBRATION.json"
            if calib_path.exists():
                try:
                    raw_state = json.loads(calib_path.read_text(encoding="utf-8"))
                    if isinstance(raw_state, dict) and "method" in raw_state and "heads" in raw_state:
                        calibration_method = str(raw_state.get("method") or calibration_method).lower()
                        heads_state = raw_state.get("heads") or {}
                    else:
                        heads_state = raw_state
                    if calibration_method == "platt":
                        from gx1.xgb.calibration.platt_scaler import PlattScaler
                        CalibCls = PlattScaler
                    elif calibration_method == "isotonic_interp":
                        CalibCls = None
                    else:
                        from gx1.xgb.calibration.isotonic_scaler import IsotonicScaler
                        CalibCls = IsotonicScaler
                    calibrators = {}
                    for head, state in heads_state.items():
                        if calibration_method == "isotonic_interp":
                            calibrators[head] = {"x": state.get("x", []), "y": state.get("y", [])}
                        else:
                            cal = CalibCls(name=calibration_method)
                            cal.set_state(state)
                            calibrators[head] = cal
                    log.info(
                        "[XGB_HEAD_CALIBRATION] enabled=True method=%s heads=%s source=%s",
                        calibration_method,
                        list(calibrators.keys()),
                        calib_path,
                    )
                except Exception as e:
                    log.warning(
                        "[XGB_HEAD_CALIBRATION] enabled=True but failed to load CALIBRATION.json error=%s path=%s",
                        e,
                        calib_path,
                    )
                    calibrators = None
            else:
                log.info(
                    "[XGB_HEAD_CALIBRATION] enabled=True but CALIBRATION.json missing path=%s",
                    calib_path,
                )

        return cls(
            heads=data["heads"],
            feature_list=feature_list,
            meta=meta,
            calibrators=calibrators,
            calibration_method=calibration_method,
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
        if not hasattr(self, "_class_order_logged") or session not in getattr(self, "_class_order_logged", set()):
            contract_class_order = list(XGB_PROB_FIELDS_ORDERED)
            bridge_mapping = ["p_long", "p_short", "p_flat"]
            meta_class_order = self.meta.get("class_labels") or self.meta.get("class_order")
            head_classes_raw = getattr(head, "classes_", None)
            if head_classes_raw is not None:
                if hasattr(head_classes_raw, "tolist"):
                    head_classes_order = head_classes_raw.tolist()
                else:
                    head_classes_order = list(head_classes_raw)
            else:
                head_classes_order = None

            print(f"[XGB_CLASS_ORDER_PROOF] contract_class_order={contract_class_order}", flush=True)
            print(f"[XGB_CLASS_ORDER_PROOF] bridge_interpretation={{'p_long': 0, 'p_short': 1, 'p_flat': 2}}", flush=True)
            print(f"[XGB_CLASS_ORDER_PROOF] model_meta_class_order={meta_class_order}", flush=True)
            print(f"[XGB_CLASS_ORDER_PROOF] head_classes_order={head_classes_order}", flush=True)

            if bridge_mapping != contract_class_order:
                raise RuntimeError("[XGB_CLASS_ORDER_FAIL] bridge mapping != contract order")
            if meta_class_order is not None:
                meta_as_list = list(meta_class_order)
                if meta_as_list != contract_class_order:
                    raise RuntimeError(
                        "[XGB_CLASS_ORDER_FAIL] model meta class order != contract order "
                        f"(meta={meta_as_list} contract={contract_class_order})"
                    )

            if not hasattr(self, "_class_order_logged"):
                self._class_order_logged = set()
            self._class_order_logged.add(session)
        
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

        if not hasattr(self, "_raw_proba_runtime_logged"):
            self._raw_proba_runtime_logged = {}
        if self._raw_proba_runtime_logged.get(session, 0) < 10:
            head_classes_raw = getattr(head, "classes_", None)
            if head_classes_raw is not None:
                if hasattr(head_classes_raw, "tolist"):
                    head_classes_order = head_classes_raw.tolist()
                else:
                    head_classes_order = list(head_classes_raw)
            else:
                head_classes_order = None
            sample_rows = proba[: min(10, len(proba))].tolist()
            log.info(
                "[XGB_RAW_PROBA_RUNTIME] head=%s proba_dim=%s classes_order=%s sample_rows=%s",
                session,
                proba.shape[1] if hasattr(proba, "shape") and len(proba.shape) > 1 else None,
                head_classes_order,
                sample_rows,
            )
            self._raw_proba_runtime_logged[session] = self._raw_proba_runtime_logged.get(session, 0) + len(sample_rows)

        if not hasattr(self, "_proba_logged"):
            self._proba_logged = set()
        if session not in self._proba_logged:
            try:
                arr = np.asarray(proba)
                n_cols = arr.shape[1] if arr.ndim == 2 else None
                if n_cols == 2:
                    p_long = arr[:, 0]
                    p_short = arr[:, 1]
                    p_flat = np.zeros_like(p_long)
                    mapping = "proba[:,0]->p_long proba[:,1]->p_short p_flat=0.0"
                elif n_cols == 3:
                    p_long = arr[:, 0]
                    p_short = arr[:, 1]
                    p_flat = arr[:, 2]
                    mapping = "proba[:,0]->p_long proba[:,1]->p_short proba[:,2]->p_flat"
                else:
                    p_long = None
                    p_short = None
                    p_flat = None
                    mapping = f"proba_dim={n_cols}"
                if p_long is not None and p_short is not None:
                    prefer_long = int(np.sum(p_long > p_short))
                    prefer_short = int(np.sum(p_short > p_long))
                    prefer_equal = int(np.sum(p_long == p_short))
                    def _q(vals: np.ndarray, q: float) -> float:
                        return float(np.quantile(vals, q)) if vals.size else float("nan")
                    log.info(
                        "[XGB_RAW_PROBA] session=%s proba_dim=%s p_long_mean=%.6f p_short_mean=%.6f p_flat_mean=%.6f "
                        "p_long_p10=%.6f p_long_p50=%.6f p_long_p90=%.6f "
                        "p_short_p10=%.6f p_short_p50=%.6f p_short_p90=%.6f "
                        "p_flat_p10=%.6f p_flat_p50=%.6f p_flat_p90=%.6f "
                        "prefer_long=%d prefer_short=%d prefer_equal=%d",
                        session,
                        n_cols,
                        float(np.mean(p_long)),
                        float(np.mean(p_short)),
                        float(np.mean(p_flat)),
                        _q(p_long, 0.10),
                        _q(p_long, 0.50),
                        _q(p_long, 0.90),
                        _q(p_short, 0.10),
                        _q(p_short, 0.50),
                        _q(p_short, 0.90),
                        _q(p_flat, 0.10),
                        _q(p_flat, 0.50),
                        _q(p_flat, 0.90),
                        prefer_long,
                        prefer_short,
                        prefer_equal,
                    )
                    log.info(
                        "[XGB_RAW_PROBA_MAPPING] session=%s mapping=%s",
                        session,
                        mapping,
                    )
                    log.info("[NO_RUNTIME_CALIBRATION] session=%s stage=predict_proba->bridge", session)
            except Exception as e:
                log.warning("[XGB_RAW_PROBA] failed session=%s error=%s", session, e)
            self._proba_logged.add(session)

        log.info(
            "[XGB_BRIDGE_INPUT] head=%s proba_dim=%s",
            session,
            proba.shape[1] if hasattr(proba, "shape") and len(proba.shape) > 1 else None,
        )
        bridge = proba_to_signal_bridge_v1(proba)
        calibrator = self.calibrators.get(session) if self.calibrators else None
        if calibrator is not None:
            try:
                p_hat_raw = bridge[:, 3]
                if self.calibration_method == "isotonic_interp":
                    x = np.asarray(calibrator.get("x", []), dtype=np.float32)
                    y = np.asarray(calibrator.get("y", []), dtype=np.float32)
                    if x.size < 2 or y.size < 2:
                        raise RuntimeError("isotonic_interp requires x/y with at least 2 points")
                    p_hat_cal = np.interp(p_hat_raw, x, y)
                else:
                    p_hat_cal = calibrator.transform(p_hat_raw)
                p_hat_cal = np.clip(p_hat_cal, 0.0, 1.0)
                bridge[:, 3] = p_hat_cal
                bridge[:, 4] = 1.0 - p_hat_cal
                if not hasattr(self, "_calibration_logged"):
                    self._calibration_logged = set()
                if session not in self._calibration_logged:
                    log.info(
                        "[XGB_HEAD_CALIBRATION_APPLIED] session=%s method=%s",
                        session,
                        self.calibration_method or "unknown",
                    )
                    self._calibration_logged.add(session)
            except Exception as e:
                log.warning("[XGB_HEAD_CALIBRATION_FAILED] session=%s error=%s", session, e)
        log.info(
            "[XGB_BRIDGE_OUTPUT] head=%s bridge_shape=%s p_flat_zero_rate=%.6f",
            session,
            bridge.shape,
            float(np.mean(bridge[:, 2] == 0.0)) if bridge.size else 0.0,
        )
        if not hasattr(self, "_bridge_logged"):
            self._bridge_logged = set()
        if session not in self._bridge_logged:
            log.info(
                "[XGB_BRIDGE] proba_dim=%d -> bridge_dim=7 source=CANONICAL_BUNDLE head=%s",
                proba.shape[1],
                session,
            )
            self._bridge_logged.add(session)
        if not hasattr(self, "_signal7_logged"):
            self._signal7_logged = set()
        if session not in self._signal7_logged:
            p_flat = bridge[:, 2]
            zero_rate = float(np.mean(p_flat == 0.0)) if p_flat.size else 0.0
            log.info(
                "[XGB_SIGNAL7_PROOF] head=%s shape=%s p_flat_zero_rate=%.6f",
                session,
                bridge.shape,
                zero_rate,
            )
            self._signal7_logged.add(session)

        p_long = bridge[:, 0]
        p_short = bridge[:, 1]
        p_flat = bridge[:, 2]
        uncertainty = bridge[:, 4]
        
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
