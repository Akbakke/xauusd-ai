"""TRAIN-only probability-level diagnostic; no model, I/O or fit authority."""
import math

import numpy as np
from scipy.special import expit

BISECTION_STEPS = 64
MEAN_TOLERANCE = 32 * np.finfo(np.float64).eps


def solve_prefix_prior_offset(prefix_logits, full_prefix_up_fraction, *, partition):
    """Match mean saved prefix probability to the original full-prefix prior.

    One scalar offset is estimated per frozen model and horizon. Labels from
    later periods must never determine the prior or the prediction sample.
    This changes probability level, not within-fold direction ranking.
    """
    if partition != "train":
        raise ValueError("PRIOR_OFFSET_TRAIN_ONLY")
    z = prefix_logits
    if not (isinstance(z, np.ndarray) and z.dtype == np.float64
            and z.ndim == 1 and len(z) >= 2 and z.flags.c_contiguous
            and np.isfinite(z).all()):
        raise ValueError("PRIOR_OFFSET_PREFIX_LOGITS")
    p = full_prefix_up_fraction
    if not (isinstance(p, (float, np.floating)) and math.isfinite(p) and 0 < p < 1):
        raise ValueError("PRIOR_OFFSET_PRIOR")
    p = float(p)
    target_logit = math.log(p) - math.log1p(-p)
    lo, hi = target_logit - float(z.max()), target_logit - float(z.min())
    if not all(math.isfinite(v) for v in (lo, hi, hi - lo)):
        raise ValueError("PRIOR_OFFSET_BRACKET")
    for _ in range(BISECTION_STEPS):
        middle = lo + (hi - lo) / 2
        if float(expit(z + middle).mean()) < p:
            lo = middle
        else:
            hi = middle
    offset = lo + (hi - lo) / 2
    after = float(expit(z + offset).mean())
    residual = after - p
    if not math.isfinite(residual) or abs(residual) > MEAN_TOLERANCE:
        raise ValueError("PRIOR_OFFSET_NUMERICAL_RESIDUAL")
    return {
        "offset": offset, "prefix_rows": len(z),
        "full_prefix_up_fraction": p,
        "mean_probability_before": float(expit(z).mean()),
        "mean_probability_after": after, "mean_residual": residual,
        "bisection_steps": BISECTION_STEPS, "fit_partition": "train",
    }
