"""
ARCHIVED: legacy diagnostics; removed from runtime. Non-canonical.

Contains the former parity/mirror_offline_predict helpers from
`gx1/execution/oanda_demo_runner.py` (offline diagnostics only).
"""

import numpy as np
from typing import Any, Dict, List, Tuple
import pandas as pd

from gx1.execution.oanda_demo_runner import extract_entry_probabilities  # type: ignore


def mirror_offline_predict(
    feat_row: pd.Series,
    session_tag: str,
    models: Dict[str, Any],
    feature_cols: List[str],
    temperature_map: Dict[str, float],
    manifest: Dict,
) -> Tuple[float, float, str]:
    """
    LEGACY (non-canonical): Mirror offline prediction for parity verification (offline diagnostics only).
    Uses same model and logic as live runner.
    """
    model = models.get(session_tag)
    if model is None:
        raise ValueError(f"Model for session '{session_tag}' not found")

    row = feat_row[feature_cols].astype(np.float32).to_numpy().reshape(1, -1)
    probs = model.predict_proba(row)
    classes = getattr(model, "classes_", None)
    prediction_off = extract_entry_probabilities(probs, classes)

    T = float(temperature_map.get(session_tag, 1.0))
    if T != 1.0:
        prediction_off.prob_long = _apply_temperature_static(prediction_off.prob_long, T)
        prediction_off.prob_short = _apply_temperature_static(prediction_off.prob_short, T)
        prediction_off.p_hat = float(max(prediction_off.prob_long, prediction_off.prob_short))
        prediction_off.margin = float(abs(prediction_off.prob_long - prediction_off.prob_short))

    dir_off = "LONG" if prediction_off.prob_long >= prediction_off.prob_short else "SHORT"
    return prediction_off.p_hat, prediction_off.margin, dir_off


def _apply_temperature_static(p: float, T: float) -> float:
    """Apply temperature scaling (static version for parity-run)."""
    eps = 1e-8
    p = min(max(p, eps), 1.0 - eps)
    logit_p = np.log(p) - np.log(1.0 - p)
    logit_T = logit_p / max(T, 1e-6)
    return 1.0 / (1.0 + np.exp(-logit_T))
