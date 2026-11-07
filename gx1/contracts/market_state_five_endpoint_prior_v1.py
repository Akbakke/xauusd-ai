"""Predeclared constant market-state prior from earlier mature TRAIN targets.

The caller owns exact fold partitioning. This pure function never reads files,
opens TEST, scores a later fold, or grants model-fit authority.
"""
from __future__ import annotations

import numpy as np

from gx1.contracts.market_state_five_endpoint_contract_v1 import TIMEFRAMES, VALID_BITS


def fit_activity_side_prior(allowed: np.ndarray, eligible: np.ndarray,
                            mature: np.ndarray) -> dict:
    """Fit five constant [DOWN, FLAT, UP] priors with fixed binary +1 smoothing.

    Bit 5 counts as ACTIVE for activity but supplies no direction. Bit 7 and
    ineligible or immature cells supply no counts. Zero side support keeps an
    explicit count of zero and a neutral 0.5 conditional UP probability.
    """
    shape = getattr(allowed, "shape", None)
    if not (isinstance(allowed, np.ndarray) and allowed.dtype == np.uint8
            and allowed.ndim == 2 and allowed.shape[0] > 0
            and allowed.shape[1] == len(TIMEFRAMES)
            and np.isin(allowed, VALID_BITS).all()
            and all(isinstance(mask, np.ndarray) and mask.dtype == np.bool_
                    and mask.shape == shape for mask in (eligible, mature))):
        raise ValueError("FIVE_ENDPOINT_PRIOR_INPUT")
    ready = eligible & mature
    activity = ready & (allowed != 7)
    active = ready & np.isin(allowed, (1, 4, 5))
    side = ready & np.isin(allowed, (1, 4))
    up = ready & (allowed == 4)
    activity_count = activity.sum(axis=0, dtype=np.int64)
    active_count = active.sum(axis=0, dtype=np.int64)
    side_count = side.sum(axis=0, dtype=np.int64)
    up_count = up.sum(axis=0, dtype=np.int64)
    p_active = (active_count + 1) / (activity_count + 2)
    p_up_given_active = (up_count + 1) / (side_count + 2)
    probabilities = np.stack((p_active * (1 - p_up_given_active),
                              1 - p_active,
                              p_active * p_up_given_active), axis=-1)
    if not (probabilities.dtype == np.float64
            and probabilities.shape == (len(TIMEFRAMES), 3)
            and np.isfinite(probabilities).all()
            and np.all(probabilities > 0)
            and np.allclose(probabilities.sum(axis=-1), 1, rtol=0, atol=1e-15)):
        raise RuntimeError("FIVE_ENDPOINT_PRIOR_SIMPLEX")
    counts = {tf: {"activity": int(activity_count[j]),
                   "active": int(active_count[j]),
                   "certified_side": int(side_count[j]),
                   "up": int(up_count[j])}
              for j, tf in enumerate(TIMEFRAMES)}
    return {"probabilities": probabilities, "counts": counts,
            "smoothing": "laplace_plus_one_separate_binary_activity_and_conditional_side"}
