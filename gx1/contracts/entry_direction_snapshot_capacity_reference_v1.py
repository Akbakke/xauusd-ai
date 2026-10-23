"""Fixed213 mean-SSE snapshot ridge reference; no data loader or fit authority.

The caller owns frozen feature order/units, prefix and observed maturity masks,
policy/caps, and numerical qualification. Existing193 SUM-SSE and native owners
remain unchanged. This fixed capacity convention is not an optimum or edge claim.
"""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np

from gx1.contracts.entry_direction_snapshot_reference_v1 import _require_matrix, _ridge_owners

SCHEMA_VERSION = "gx1_direction_snapshot_capacity_reference_v1"
HORIZONS_MINUTES = (1, 5, 15)
MEAN_SSE_LAMBDA = 1.0
OBJECTIVE = "mean_SSE_plus_lambda_beta_squared_free_intercept"
FEATURE_SCALE = "ones_in_supplied_snapshot_coordinates"


def _require(condition, reason):
    if not condition:
        raise RuntimeError("DIRECTION_SNAPSHOT_CAPACITY_REFERENCE_" + reason)


def fit_direction_snapshot_capacity_ridge(matrix, targets, observed, *, split):
    """Fit fixed mean-SSE+||beta||², using each horizon's own observed TRAIN N."""
    _require(split == "train", "TRAIN_ONLY_FIT")
    _require_matrix(matrix)
    _require(isinstance(targets, np.ndarray) and targets.shape == (len(matrix), 3)
        and targets.dtype == np.float64 and isinstance(observed, np.ndarray)
        and observed.shape == targets.shape and observed.dtype == np.bool_, "TARGET_MASK_SHAPE_DTYPE")
    _require(np.all(observed.sum(axis=0) > 0) and np.isfinite(targets[observed]).all(),
        "OBSERVED_TRAIN_TARGETS")
    chunk, gram_owner, xty_owner, _ = _ridge_owners()
    fits = {}
    for column, horizon in enumerate(HORIZONS_MINUTES):
        rows = np.flatnonzero(observed[:, column])
        own_n = int(len(rows))
        sum_sse_alpha = float(own_n) * MEAN_SSE_LAMBDA
        mean = np.zeros(matrix.shape[1], dtype=np.float64)
        for start in range(0, own_n, chunk):
            mean += matrix[rows[start:start+chunk]].sum(axis=0, dtype=np.float64)
        mean /= own_n
        target_mean = float(np.mean(targets[rows, column], dtype=np.float64))
        #Supplied snapshot coordinates already have their declared prefix units.
        scale = np.ones(matrix.shape[1], dtype=np.float64)
        gram = gram_owner(matrix, rows, mean=mean, scale=scale)
        gram.flat[::len(mean)+1] += sum_sse_alpha
        centered_y = targets[:, column].astype(np.float64) - target_mean
        rhs = xty_owner(matrix, centered_y, rows, mean=mean, scale=scale)
        _require(np.isfinite(gram).all() and np.isfinite(rhs).all(), "NONFINITE_SUFFICIENT_STATISTICS")
        beta = np.linalg.solve(gram, rhs)
        residual = gram @ beta - rhs
        rhs_norm = float(np.linalg.norm(rhs))
        residual_norm = float(np.linalg.norm(residual))
        intercept = target_mean - float(mean @ beta)
        _require(np.isfinite(beta).all() and np.isfinite(mean).all() and np.isfinite(residual).all()
            and all(np.isfinite(value) for value in (target_mean, intercept, rhs_norm, residual_norm)),
            "NONFINITE_SOLVE")
        fits[horizon] = {"beta": beta, "feature_mean": mean, "target_mean_bps": target_mean,
            "intercept_bps": intercept, "observed_train_rows": own_n, "sum_sse_alpha": sum_sse_alpha,
            "solver_residual_l2": residual_norm, "rhs_l2": rhs_norm,
            "solver_relative_residual": residual_norm/rhs_norm if rhs_norm else None}
        del gram, rhs, centered_y
    return {"schema_version": SCHEMA_VERSION + "_ridge", "horizons_minutes": HORIZONS_MINUTES,
        "mean_sse_lambda": MEAN_SSE_LAMBDA, "objective": OBJECTIVE,
        "feature_width": int(matrix.shape[1]), "feature_scale": FEATURE_SCALE,
        "fit_partition": "train", "fits": fits}


def predict_direction_snapshot_capacity_ridge(matrix, state):
    """Predict with frozen candidate centering/coefficient state; never recalibrate."""
    _require_matrix(matrix)
    _require(isinstance(state, Mapping) and state.get("schema_version") == SCHEMA_VERSION + "_ridge"
        and type(state.get("mean_sse_lambda")) is float and state["mean_sse_lambda"] == MEAN_SSE_LAMBDA
        and state.get("objective") == OBJECTIVE and state.get("feature_scale") == FEATURE_SCALE
        and state.get("fit_partition") == "train"
        and tuple(state.get("horizons_minutes", ())) == HORIZONS_MINUTES
        and type(state.get("feature_width")) is int and state["feature_width"] == matrix.shape[1]
        and isinstance(state.get("fits"), Mapping) and set(state["fits"]) == set(HORIZONS_MINUTES),
        "CAPACITY_STATE")
    _, _, _, matvec_owner = _ridge_owners()
    predictions = []
    for horizon in HORIZONS_MINUTES:
        fit = state["fits"][horizon]
        _require(isinstance(fit, Mapping), "CAPACITY_FIT")
        beta, mean = fit.get("beta"), fit.get("feature_mean")
        own_n, alpha = fit.get("observed_train_rows"), fit.get("sum_sse_alpha")
        target_mean, intercept = fit.get("target_mean_bps"), fit.get("intercept_bps")
        _require(type(own_n) is int and own_n > 0 and type(alpha) is float
            and np.isfinite(alpha) and alpha == float(own_n) * MEAN_SSE_LAMBDA,
            "OWN_N_SUM_SSE_ALPHA")
        _require(isinstance(beta, np.ndarray) and isinstance(mean, np.ndarray)
            and beta.dtype == mean.dtype == np.float64 and beta.shape == mean.shape == (matrix.shape[1],)
            and np.isfinite(beta).all() and np.isfinite(mean).all()
            and isinstance(target_mean, (float, np.floating)) and isinstance(intercept, (float, np.floating))
            and np.isfinite(target_mean) and np.isfinite(intercept), "CAPACITY_FIT_STATE")
        _require(intercept == target_mean - float(mean @ beta), "INTERCEPT_IDENTITY")
        predictions.append(matvec_owner(matrix, beta, mean=mean, scale=np.ones_like(mean)) + target_mean)
    result = np.column_stack(predictions)
    _require(np.isfinite(result).all(), "NONFINITE_PREDICTIONS")
    return result
