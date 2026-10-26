"""Fixed frozen-head BCE probe; no native model, raw input or trading authority.

The same independent linear heads minimize global eligible-cell BCE when the
backbone is frozen. Prefix-only affine coordinates improve conditioning without
adding a penalty, a feature or a future statistic. Actual data and fit admissions
belong to the caller. This module enforces capped CPU execution for every fit.
"""
from __future__ import annotations

import warnings
import numpy as np

SCHEMA_VERSION = 'gx1_native_direction_frozen_readout_v1'
COORDINATE_SCHEMA = SCHEMA_VERSION + '_prefix_coordinates'
WIDTH = 128
MINUTE_NS = 60_000_000_000
TRAIN_END_NS = 1_767_225_600_000_000_000
HORIZONS = (1, 5, 15)
SCIPY_VERSION = '1.15.3'
SOLVER_OPTIONS = {'gtol': 1e-7, 'ftol': 1e-12, 'maxiter': 2000,
                  'maxfun': 4000, 'maxls': 40, 'maxcor': 20}
GRADIENT_MAX = 1e-6
OBJECTIVE_TOLERANCE = 1e-10


def _require(condition, reason):
    if not condition:
        raise RuntimeError('FROZEN_READOUT_' + reason)


def _array(value, dtype, shape, label):
    _require(isinstance(value, np.ndarray) and value.dtype == np.dtype(dtype)
             and value.shape == shape, label)
    return value


def _hidden(value):
    _require(isinstance(value, np.ndarray) and value.dtype == np.float32
             and value.ndim == 2 and value.shape[0] > 0 and value.shape[1] == WIDTH
             and np.isfinite(value).all(), 'HIDDEN_FP32_128')
    return value


def _readonly(value):
    result = np.ascontiguousarray(value)
    result.setflags(write=False)
    return result


def fit_prefix_coordinates(hidden):
    """Caller supplies only own prefix rows; no label or later data is read."""
    raw = _hidden(hidden).astype(np.float64)
    center = raw.mean(axis=0)
    deviation = raw - center
    scale = np.sqrt(np.mean(deviation * deviation, axis=0))
    constant = scale == 0
    scale[constant] = 1.
    _require(np.isfinite(center).all() and np.isfinite(scale).all()
             and np.all(scale > 0), 'PREFIX_COORDINATES_FINITE')
    return {'schema_version': COORDINATE_SCHEMA, 'fit_rows': len(raw),
            'center': _readonly(center), 'scale': _readonly(scale),
            'constant': _readonly(constant)}


def transform_hidden(hidden, coordinates):
    raw = _hidden(hidden)
    _require(set(coordinates) == {'schema_version', 'fit_rows', 'center', 'scale', 'constant'}
             and coordinates['schema_version'] == COORDINATE_SCHEMA
             and type(coordinates['fit_rows']) is int and coordinates['fit_rows'] > 0,
             'COORDINATE_SCHEMA')
    center = _array(coordinates['center'], 'float64', (WIDTH,), 'CENTER')
    scale = _array(coordinates['scale'], 'float64', (WIDTH,), 'SCALE')
    constant = _array(coordinates['constant'], 'bool', (WIDTH,), 'CONSTANT_MASK')
    _require(np.isfinite(center).all() and np.isfinite(scale).all()
             and np.all(scale > 0) and np.all(scale[constant] == 1.), 'COORDINATE_VALUES')
    value = np.ascontiguousarray((raw.astype(np.float64) - center) / scale)
    _require(np.isfinite(value).all(), 'TRANSFORM_FINITE')
    return value


def native_head_coordinates(weight, bias, coordinates):
    _array(weight, 'float32', (3, WIDTH), 'NATIVE_WEIGHT')
    _array(bias, 'float32', (3,), 'NATIVE_BIAS')
    _require(np.isfinite(weight).all() and np.isfinite(bias).all(), 'NATIVE_HEAD_FINITE')
    coef = weight.astype(np.float64) * coordinates['scale'][None, :]
    intercept = bias.astype(np.float64) + weight.astype(np.float64) @ coordinates['center']
    _require(np.isfinite(coef).all() and np.isfinite(intercept).all(), 'NATIVE_COORDINATE_FINITE')
    return coef, intercept


def _value_gradient(theta, matrix, labels):
    _require(np.isfinite(theta).all(), 'THETA_FINITE')
    logit = matrix @ theta[:-1] + theta[-1]
    _require(np.isfinite(logit).all(), 'LOGIT_FINITE')
    value = float(np.mean(np.logaddexp(0., logit) - labels * logit))
    probability = np.empty_like(logit)
    positive = logit >= 0
    probability[positive] = 1. / (1. + np.exp(-logit[positive]))
    exponential = np.exp(logit[~positive])
    probability[~positive] = exponential / (1. + exponential)
    residual = probability - labels
    gradient = np.empty_like(theta)
    gradient[:-1] = matrix.T @ residual / len(labels)
    gradient[-1] = residual.mean()
    _require(np.isfinite(value) and value >= 0 and np.isfinite(gradient).all(), 'OBJECTIVE_FINITE')
    return value, gradient


def binary_bce_gradient(theta, matrix, labels):
    """Mean unweighted BCE and analytic gradient, including an unpenalized bias."""
    _require(isinstance(matrix, np.ndarray) and matrix.dtype == np.float64
             and matrix.ndim == 2 and matrix.shape[0] > 0 and 0 < matrix.shape[1] <= WIDTH
             and np.isfinite(matrix).all(), 'OBJECTIVE_MATRIX')
    _array(theta, 'float64', (matrix.shape[1] + 1,), 'THETA')
    _array(labels, 'float64', (len(matrix),), 'LABELS')
    _require(np.isin(labels, (0., 1.)).all(), 'LABEL_DOMAIN')
    return _value_gradient(theta, matrix, labels)


def eligible_prefix_labels(parent_rows, sign, observed, decision_ns, terminal_time_ns, *, cutoff_ns):
    n = len(parent_rows)
    _array(parent_rows, 'int64', (n,), 'PARENT_DTYPE')
    _require(n > 0 and np.array_equal(parent_rows, np.arange(n, dtype=np.int64)), 'PREFIX_PARENT_IDENTITY')
    _array(sign, 'int8', (n, 3), 'SIGN')
    _array(observed, 'bool', (n, 3), 'OBSERVED')
    _array(decision_ns, 'int64', (n,), 'DECISION')
    _array(terminal_time_ns, 'int64', (n, 3), 'TERMINAL')
    _require(type(cutoff_ns) is int and 0 < cutoff_ns <= TRAIN_END_NS, 'PREFIX_CUTOFF')
    _require(np.all(decision_ns > 0) and np.all(decision_ns < cutoff_ns)
             and np.all(np.diff(decision_ns) > 0)
             and np.all(decision_ns <= np.iinfo(np.int64).max - 15 * MINUTE_NS), 'DECISION_CUTOFF')
    expected = decision_ns[:, None] + np.asarray(HORIZONS, np.int64)[None, :] * MINUTE_NS
    _require(np.array_equal(terminal_time_ns, expected), 'TARGET_CLOCK')
    _require(np.isin(sign, (-1, 0, 1)).all() and np.all(sign[~observed] == 0), 'SIGN_MISSING_STORAGE')
    eligible = observed & (sign != 0) & (terminal_time_ns < cutoff_ns)
    for j in range(3):
        _require(np.any(eligible[:, j] & (sign[:, j] == 1))
                 and np.any(eligible[:, j] & (sign[:, j] == -1)), 'PREFIX_BOTH_CLASSES')
    return eligible


def fit_frozen_direction_readout(hidden, *, parent_rows, sign, observed, decision_ns,
                                 terminal_time_ns, cutoff_ns, original_weight, original_bias, split):
    """Three fixed convex fits on own prefix; caller must bind genuine scope.

    Results establish numerical head fitting only. Candidate equality is FP64
    coordinate algebra, not saved native GPU prediction parity. There are no
    native parameter changes, PyTorch optimizer steps or model forwards here.
    """
    _require(split == 'train', 'TRAIN_ONLY')
    _hidden(hidden)
    _require(len(hidden) == len(parent_rows), 'HIDDEN_PARENT_JOIN')
    eligible = eligible_prefix_labels(parent_rows, sign, observed, decision_ns,
                                     terminal_time_ns, cutoff_ns=cutoff_ns)
    coordinates = fit_prefix_coordinates(hidden)
    matrix = transform_hidden(hidden, coordinates)
    native_coef, native_intercept = native_head_coordinates(original_weight, original_bias, coordinates)
    from gx1.contracts.gx1_capped_execution_v1 import (
        require_capped_cpu_audit_execution, require_capped_lock_ancestry,
    )
    execution = require_capped_cpu_audit_execution()
    lock = require_capped_lock_ancestry()
    _require(lock['scope'] == execution['cgroup_path'][len('/sys/fs/cgroup'):], 'LOCK_SCOPE')
    import scipy
    from scipy.optimize import minimize
    _require(scipy.__version__ == SCIPY_VERSION, 'SCIPY_VERSION')
    coefficients = np.empty((3, WIDTH), np.float64)
    intercepts = np.empty(3, np.float64)
    raw = hidden.astype(np.float64)
    diagnostics = []
    for j, horizon in enumerate(HORIZONS):
        selected = eligible[:, j]
        x = _readonly(matrix[selected])
        y = _readonly((sign[selected, j] == 1).astype(np.float64))
        up = int(y.sum())
        down = len(y) - up
        initial = np.zeros(WIDTH + 1, np.float64)
        initial[-1] = np.log(float(up) / down)
        prior_value = _value_gradient(initial, x, y)[0]
        native_theta = np.r_[native_coef[j], native_intercept[j]]
        candidate_value = _value_gradient(native_theta, x, y)[0]
        reconstructed = x @ native_theta[:-1] + native_theta[-1]
        original = raw[selected] @ original_weight[j].astype(np.float64) + float(original_bias[j])
        max_error = float(np.max(np.abs(reconstructed - original)))
        mass = (np.max(np.abs(raw[selected]), axis=0) @ np.abs(original_weight[j].astype(np.float64))
                + np.abs(coordinates['center']) @ np.abs(original_weight[j].astype(np.float64))
                + abs(float(original_bias[j])) + 1.)
        rounding_bound = float(64 * np.finfo(np.float64).eps * (WIDTH + 4) * mass)
        _require(max_error <= rounding_bound, 'NATIVE_COORDINATE_ALGEBRA')
        calls = 0
        def objective(theta):
            nonlocal calls
            calls += 1
            _require(calls <= SOLVER_OPTIONS['maxfun'], 'OBJECTIVE_CALL_BUDGET')
            return _value_gradient(theta, x, y)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            fitted = minimize(objective, initial, jac=True, method='L-BFGS-B', options=dict(SOLVER_OPTIONS))
        _require(not caught and bool(fitted.success) and int(fitted.status) == 0, 'SOLVER_SUCCESS')
        _require(0 <= int(fitted.nit) <= SOLVER_OPTIONS['maxiter']
                 and 0 < int(fitted.nfev) == calls <= SOLVER_OPTIONS['maxfun'], 'SOLVER_COUNTS')
        value, gradient = _value_gradient(fitted.x, x, y)
        gradient_norm = float(np.max(np.abs(gradient)))
        _require(gradient_norm <= GRADIENT_MAX, 'SOLVER_GRADIENT')
        for reference in (prior_value, candidate_value):
            _require(value <= reference + OBJECTIVE_TOLERANCE * max(1., abs(reference)), 'CANDIDATE_OBJECTIVE_GAP')
        coefficients[j] = fitted.x[:-1]
        intercepts[j] = fitted.x[-1]
        diagnostics.append({'horizon_minutes': horizon, 'parent_rows': len(hidden),
            'eligible_rows': len(y), 'eligible_up': up, 'eligible_down': down,
            'fit_mean_bce': value, 'prior_mean_bce': prior_value,
            'original_head_mean_bce': candidate_value,
            'gradient_infinity_norm': gradient_norm, 'iterations': int(fitted.nit),
            'objective_evaluations': calls, 'solver_message': str(fitted.message),
            'native_coordinate_max_error': max_error, 'native_coordinate_rounding_bound': rounding_bound})
    total = sum(item['eligible_rows'] for item in diagnostics)
    return {'schema_version': SCHEMA_VERSION, 'fit_partition': 'train',
        'prefix_cutoff_ns': cutoff_ns, 'prefix_parent_rows': len(hidden),
        'coordinates': coordinates, 'coefficient': _readonly(coefficients),
        'intercept': _readonly(intercepts), 'fit_diagnostics': diagnostics,
        'global_eligible_cell_bce': sum(d['eligible_rows'] * d['fit_mean_bce'] for d in diagnostics) / total,
        'objective': 'unweighted mature observed nonzero BCE; no penalty',
        'solver': {'version': SCIPY_VERSION, 'method': 'L-BFGS-B', 'options': dict(SOLVER_OPTIONS)},
        'readout_fit_calls': 3, 'native_optimizer_steps': 0,
        'predictive_value_qualified': False, 'economic_edge_qualified': False}


def predict_frozen_direction_readout(hidden, state):
    _require(isinstance(state, dict) and state.get('schema_version') == SCHEMA_VERSION
             and state.get('fit_partition') == 'train', 'FIT_STATE')
    coefficients = _array(state['coefficient'], 'float64', (3, WIDTH), 'FIT_COEFFICIENT')
    intercept = _array(state['intercept'], 'float64', (3,), 'FIT_INTERCEPT')
    _require(np.isfinite(coefficients).all() and np.isfinite(intercept).all(), 'FIT_FINITE')
    value = transform_hidden(hidden, state['coordinates']) @ coefficients.T + intercept
    _require(np.isfinite(value).all(), 'PREDICTION_FINITE')
    return np.ascontiguousarray(value)
