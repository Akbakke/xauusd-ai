"""Pure target and paired scoring adapter for a tabular five-endpoint diagnostic.

Only target arrays enter this module. A caller must write the returned prior
record before any model updates; this module grants no training or TEST access.
"""
from __future__ import annotations

import json

import numpy as np

from gx1.contracts.market_state_five_endpoint_contract_v1 import (
    TIMEFRAMES, VALID_BITS, proper_score_sums,
)
from gx1.contracts.market_state_five_endpoint_prior_v1 import fit_activity_side_prior
from gx1.contracts.market_state_five_endpoint_target_binding_v1 import BoundFiveEndpointTargets


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError('MARKET_STATE_TABULAR_' + code)


def binary_fit_targets(allowed: np.ndarray, eligible: np.ndarray,
                       mature: np.ndarray) -> dict:
    """Return separate activity and certified-side labels with exact masks.

    Bit 5 contributes activity=1 but has no side label. Bit 7 contributes
    neither. Unused label slots are zero and must never be fitted without masks.
    """
    shape = getattr(allowed, 'shape', None)
    _require(isinstance(allowed, np.ndarray) and allowed.dtype == np.uint8
             and allowed.ndim == 2 and allowed.shape[0] > 0
             and allowed.shape[1] == len(TIMEFRAMES)
             and np.isin(allowed, VALID_BITS).all()
             and all(isinstance(mask, np.ndarray) and mask.dtype == np.bool_
                     and mask.shape == shape for mask in (eligible, mature)),
             'TARGET_INPUT')
    ready = eligible & mature
    activity_mask = ready & (allowed != 7)
    side_mask = ready & np.isin(allowed, (1, 4))
    activity_label = (activity_mask & np.isin(allowed, (1, 4, 5))).astype(np.uint8)
    side_up_label = (side_mask & (allowed == 4)).astype(np.uint8)
    for value in (activity_mask, activity_label, side_mask, side_up_label):
        value.setflags(write=False)
    return {'activity_mask': activity_mask, 'activity_label': activity_label,
            'certified_side_mask': side_mask, 'side_up_label': side_up_label,
            'activity_counts': {tf: int(activity_mask[:, j].sum())
                                for j, tf in enumerate(TIMEFRAMES)},
            'certified_side_counts': {tf: int(side_mask[:, j].sum())
                                      for j, tf in enumerate(TIMEFRAMES)}}


def _prior_record(fold_id: int, cohort: dict, truth: dict) -> dict:
    prior = fit_activity_side_prior(truth['allowed'], truth['eligible'], truth['mature'])
    return {'fold_id': fold_id, 'fit_rows': cohort['joined_fit_rows'],
            'fit_parent_rows_sha256': cohort['fit_parent_rows_sha256'],
            'fit_decision_ns_sha256': cohort['fit_decision_ns_sha256'],
            'smoothing': prior['smoothing'], 'counts': prior['counts'],
            'probabilities': prior['probabilities'].tolist()}


def mature_fold_training_targets(bound: BoundFiveEndpointTargets, *, fold_id: int) -> dict:
    """Bind target-only fit masks and a constant prior to one strict fit prefix."""
    _require(isinstance(bound, BoundFiveEndpointTargets), 'BOUND_TARGETS')
    cohort = bound.mature_oof_axes(fold_id=fold_id)
    truth = bound.gather(cohort['fit_parent_rows'], cohort['fit_decision_ns'],
                         fold_id=fold_id, partition='fit')
    _require(bool(truth['mature'].all()), 'IMMATURE_FIT')
    return {'cohort': cohort,
            'binary': binary_fit_targets(truth['allowed'], truth['eligible'], truth['mature']),
            'prior': _prior_record(fold_id, cohort, truth)}


def combine_binary_probabilities(p_active: np.ndarray,
                                 p_up_given_active: np.ndarray) -> np.ndarray:
    """Map two binary heads to [DOWN, FLAT, UP] without inventing side labels."""
    _require(isinstance(p_active, np.ndarray)
             and isinstance(p_up_given_active, np.ndarray)
             and p_active.dtype in (np.float32, np.float64)
             and p_up_given_active.dtype in (np.float32, np.float64)
             and p_active.ndim == p_up_given_active.ndim == 2
             and p_active.shape == p_up_given_active.shape
             and p_active.shape[0] > 0 and p_active.shape[1] == len(TIMEFRAMES)
             and np.isfinite(p_active).all() and np.isfinite(p_up_given_active).all()
             and np.all((p_active > 0) & (p_active <= 1))
             and np.all((p_up_given_active >= 0) & (p_up_given_active <= 1)),
             'BINARY_PROBABILITIES')
    active = p_active.astype(np.float64, copy=False)
    up = p_up_given_active.astype(np.float64, copy=False)
    probabilities = np.stack((active * (1 - up), 1 - active, active * up), axis=-1)
    _require(np.isfinite(probabilities).all()
             and np.allclose(probabilities.sum(axis=-1), 1.0, rtol=0, atol=1e-15),
             'THREE_CLASS_SIMPLEX')
    return probabilities


def paired_mature_fold_scores(bound: BoundFiveEndpointTargets, *, fold_id: int,
                              prediction_parent_rows: np.ndarray,
                              prediction_decision_ns: np.ndarray,
                              probabilities: np.ndarray, prior_record: dict) -> dict:
    """Score model and predeclared fit prior on identical original score rows."""
    _require(isinstance(bound, BoundFiveEndpointTargets), 'BOUND_TARGETS')
    cohort = bound.mature_oof_axes(fold_id=fold_id)
    _require(isinstance(prediction_parent_rows, np.ndarray)
             and isinstance(prediction_decision_ns, np.ndarray)
             and prediction_parent_rows.dtype == prediction_decision_ns.dtype == np.int64
             and np.array_equal(prediction_parent_rows, cohort['score_parent_rows'])
             and np.array_equal(prediction_decision_ns, cohort['score_decision_ns']),
             'SCORE_AXES')
    fit_truth = bound.gather(cohort['fit_parent_rows'], cohort['fit_decision_ns'],
                             fold_id=fold_id, partition='fit')
    expected_prior = _prior_record(fold_id, cohort, fit_truth)
    try:
        prior_equal = (json.dumps(prior_record, sort_keys=True, allow_nan=False)
                       == json.dumps(expected_prior, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError):
        prior_equal = False
    _require(prior_equal, 'FIT_PRIOR_IDENTITY')
    truth = bound.gather(cohort['score_parent_rows'], cohort['score_decision_ns'],
                         fold_id=fold_id, partition='score')
    _require(isinstance(probabilities, np.ndarray)
             and probabilities.shape == (len(cohort['score_parent_rows']), len(TIMEFRAMES), 3),
             'SCORE_PROBABILITY_SHAPE')
    model = proper_score_sums(probabilities, truth['allowed'], truth['eligible'], truth['mature'])
    prior_probabilities = np.broadcast_to(np.asarray(prior_record['probabilities'], dtype=np.float64),
                                          probabilities.shape)
    prior = proper_score_sums(prior_probabilities, truth['allowed'], truth['eligible'], truth['mature'])
    for tf in TIMEFRAMES:
        for metric in ('activity', 'conditional_side', 'three_class'):
            _require(model['per_tf'][tf][metric]['count'] == prior['per_tf'][tf][metric]['count'],
                     'PAIRED_SUPPORT:' + tf + ':' + metric)
    return {'fold_id': fold_id, 'score_rows': len(cohort['score_parent_rows']),
            'score_parent_rows_sha256': cohort['score_parent_rows_sha256'],
            'score_decision_ns_sha256': cohort['score_decision_ns_sha256'],
            'model': model, 'fit_prior': prior}
