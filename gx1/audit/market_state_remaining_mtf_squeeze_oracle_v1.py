"""Independent causal two-state squeeze oracle for V4 MTF columns 26..28.

The caller binds frozen per-TF HMM parameters and exact closed-TF prices. This
module neither fits a model nor invokes the production squeeze decoder.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

FIELDS = ("volatility.bars_in_squeeze", "volatility.squeeze_release_age_bars",
          "volatility.bandwidth_rel")
CACHE_COLUMNS = (26, 27, 28)
WINDOW = 20


def _parameters(means, variances, transition, initial):
    mu = np.asarray(means, dtype=np.float64)
    var = np.asarray(variances, dtype=np.float64)
    move = np.asarray(transition, dtype=np.float64)
    start = np.asarray(initial, dtype=np.float64)
    if (mu.shape != (2,) or var.shape != (2,) or move.shape != (2, 2)
            or start.shape != (2,) or not np.isfinite(mu).all()
            or not np.isfinite(var).all() or not np.isfinite(move).all()
            or not np.isfinite(start).all() or np.any(var <= 0)
            or np.any(move <= 0) or np.any(start <= 0)
            or not np.allclose(move.sum(axis=1), 1.0, rtol=0, atol=1e-12)
            or not np.isclose(start.sum(), 1.0, rtol=0, atol=1e-12)
            or mu[0] >= mu[1]):
        raise ValueError("SQUEEZE3_FROZEN_PARAMETERS_INVALID")
    return mu, var, move, start


def _decode_widths(widths: np.ndarray, mu: np.ndarray, var: np.ndarray,
                   move: np.ndarray, start: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalized probability recursion, independent of owner's log-odds path."""
    durations = np.empty(len(widths), dtype=np.float64)
    ages = np.empty(len(widths), dtype=np.float64)
    posterior = None
    previous_active = None
    bars = 0
    age = None
    for i, width in enumerate(widths):
        prior = start if posterior is None else posterior @ move
        log_density = -0.5 * (np.log(2.0 * np.pi * var) +
                              (width - mu) ** 2 / var)
        weights = np.log(prior) + log_density
        likelihood = np.exp(weights - np.max(weights))
        posterior = likelihood / likelihood.sum()
        active = bool(posterior[0] >= posterior[1])
        age = 0 if age is None else age + 1
        if active:
            bars = bars + 1 if previous_active else 1
        else:
            if previous_active:
                age = 0
            bars = 0
        durations[i] = bars
        ages[i] = age
        previous_active = active
    return durations, ages


def closed_squeeze3_reference(close, *, means, variances,
                              transition_probability, initial_probability) -> dict[str, np.ndarray]:
    c = np.asarray(close, dtype=np.float64)
    if c.ndim != 1 or not len(c) or not np.isfinite(c).all() or np.any(c <= 0):
        raise ValueError("SQUEEZE3_CLOSED_CLOSE_INVALID")
    mu, var, move, start = _parameters(means, variances, transition_probability,
                                       initial_probability)
    duration = np.full(len(c), np.nan, dtype=np.float64)
    age = np.full(len(c), np.nan, dtype=np.float64)
    bandwidth = np.full(len(c), np.nan, dtype=np.float64)
    if len(c) >= WINDOW:
        windows = sliding_window_view(c, WINDOW)
        centers = windows.mean(axis=1)
        spread = np.sqrt(((windows - centers[:, None]) ** 2).mean(axis=1))
        widths = 4.0 * spread / centers
        if not np.isfinite(widths).all() or np.any(widths < 0):
            raise ValueError("SQUEEZE3_BANDWIDTH_INVALID")
        bandwidth[WINDOW - 1:] = widths
        duration[WINDOW - 1:], age[WINDOW - 1:] = _decode_widths(
            widths, mu, var, move, start)
    return dict(zip(FIELDS, (duration.astype(np.float32), age.astype(np.float32),
                             bandwidth.astype(np.float32)), strict=True))
