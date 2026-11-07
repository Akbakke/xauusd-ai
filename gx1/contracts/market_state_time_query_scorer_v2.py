"""Stable proper scores for factorized market-state time queries.

The two primitive logits retain conditional side even when the active class
probability underflows in a finite floating-point representation. Matched
Brier aggregation reuses the v1 issue/TF/week reducer with an explicit side
probability; no future target, fit, TEST or trade authority lives here.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from gx1.contracts.market_state_five_endpoint_contract_v1 import CLASS_NAMES
from gx1.contracts.market_state_time_query_loss_v2 import factorized_time_query_loss_contract
from gx1.contracts.market_state_time_query_scorer_v1 import (
    TimeQueryMatchedScorer, time_query_scorer_contract,
)
from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K, TIMEFRAMES, VALID_BITS

_K = tuple(int(value) for value in NATIVE_K)


def factorized_time_query_scorer_contract() -> dict:
    value = {
        'schema_version': 'gx1_market_state_time_query_scorer_v2',
        'role': 'research_only_no_entry_exit_or_trade_authority',
        'loss_contract_sha256': factorized_time_query_loss_contract()['contract_sha256'],
        'matched_reducer_contract_sha256': time_query_scorer_contract()['contract_sha256'],
        'timeframes': list(TIMEFRAMES), 'native_k': list(_K),
        'class_order': list(CLASS_NAMES),
        'primitive_logit_order': ['ACTIVE', 'UP_GIVEN_ACTIVE'],
        'probabilities': 'derived_in_float64_from_primitive_logits_without_epsilon_floor',
        'conditional_side': 'direct_sigmoid_of_side_logit_even_when_active_mass_underflows',
        'proper_log': 'stable_softplus_of_primitive_logits',
        'matched_reference': 'same_frozen_activity_conditional_side_three_class_priors',
        'reduction': 'query_Brier_sums_to_issue_TF_mean_then_UTC_issue_week_and_fold',
        'primary': 'M5_k1_certified_side_Brier_prior_minus_model',
        'unknown': 'bit5_activity_only_bit7_unscored',
        'test_data_used': False,
    }
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return {**value, 'contract_sha256': hashlib.sha256(raw).hexdigest()}


def _require(ok: bool, code: str) -> None:
    if not ok:
        raise ValueError('TIME_QUERY_FACTORIZED_SCORER_' + code)


def factorized_probabilities(raw_logits: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return class probabilities, activity, and conditional UP in float64."""
    _require(isinstance(raw_logits, np.ndarray)
             and raw_logits.dtype in (np.dtype('float32'), np.dtype('float64'))
             and raw_logits.ndim in (3, 4) and raw_logits.shape[-2:] == (len(TIMEFRAMES), 2)
             and all(size > 0 for size in raw_logits.shape)
             and bool(np.isfinite(raw_logits).all()), 'RAW_LOGITS')
    activity_logit = raw_logits[..., 0].astype(np.float64)
    side_logit = raw_logits[..., 1].astype(np.float64)
    active = np.exp(-np.logaddexp(0.0, -activity_logit))
    flat = np.exp(-np.logaddexp(0.0, activity_logit))
    side_up = np.exp(-np.logaddexp(0.0, -side_logit))
    side_down = np.exp(-np.logaddexp(0.0, side_logit))
    classes = np.stack((active * side_down, flat, active * side_up), axis=-1)
    _require(bool(np.isfinite(classes).all())
             and bool(np.all((classes >= 0) & (classes <= 1)))
             and bool(np.allclose(classes.sum(axis=-1, dtype=np.float64), 1.0,
                                  rtol=0, atol=1e-6)), 'CLASS_PROBABILITIES')
    return classes, active, side_up


def proper_score_sums(raw_logits: np.ndarray, allowed: np.ndarray,
                      eligible: np.ndarray, mature: np.ndarray,
                      query_k: np.ndarray) -> dict:
    """Proper Brier and log sums on the exact qualified cells, from raw logits."""
    classes, active, side_up = factorized_probabilities(raw_logits)
    shape = raw_logits.shape[:2]
    _require(raw_logits.ndim == 3 and isinstance(allowed, np.ndarray)
             and allowed.dtype == np.uint8 and allowed.shape == shape
             and bool(np.isin(allowed, VALID_BITS).all()), 'ALLOWED')
    _require(all(isinstance(mask, np.ndarray) and mask.dtype == np.bool_
                 and mask.shape == shape for mask in (eligible, mature)), 'MASKS')
    _require(isinstance(query_k, np.ndarray) and query_k.dtype == np.int64
             and query_k.shape == shape
             and bool(np.all((query_k >= 1)
                             & (query_k <= np.asarray(_K, dtype=np.int64)[None, :]))),
             'QUERY_K')
    common = eligible & mature
    report = {}
    for j, tf in enumerate(TIMEFRAMES):
        bits = allowed[:, j]
        activity_mask = common[:, j] & (bits != 7)
        side_mask = common[:, j] & np.isin(bits, (1, 4))
        singleton_mask = common[:, j] & np.isin(bits, (1, 2, 4))
        activity_truth = (bits[activity_mask] != 2).astype(np.float64)
        activity_brier = np.square(active[activity_mask, j] - activity_truth).sum(dtype=np.float64)
        side_truth = (bits[side_mask] == 4)
        side_brier = np.square(side_up[side_mask, j] - side_truth).sum(dtype=np.float64)
        side_logits = raw_logits[side_mask, j, 1].astype(np.float64)
        side_log = np.logaddexp(0.0, np.where(side_truth, -side_logits, side_logits))
        labels = bits[singleton_mask]
        truth = np.eye(3, dtype=np.float64)[np.where(labels == 1, 0,
                                                   np.where(labels == 2, 1, 2))]
        three_brier = np.square(classes[singleton_mask, j] - truth).sum(dtype=np.float64)
        activity_logits = raw_logits[singleton_mask, j, 0].astype(np.float64)
        singleton_side_logits = raw_logits[singleton_mask, j, 1].astype(np.float64)
        with np.errstate(over='ignore'):
            three_log = np.where(labels == 2, np.logaddexp(0.0, activity_logits),
                        np.logaddexp(0.0, -activity_logits)
                        + np.logaddexp(0.0, np.where(labels == 4,
                                                     -singleton_side_logits,
                                                     singleton_side_logits)))
        side_log_sum = float(side_log.sum(dtype=np.float64))
        three_log_sum = float(three_log.sum(dtype=np.float64))
        _require(np.isfinite(side_log_sum) and np.isfinite(three_log_sum),
                 'NONFINITE_PROPER_LOG')
        report[tf] = {
            'activity': {'count': int(activity_mask.sum()), 'brier_sum': float(activity_brier)},
            'conditional_side': {'count': int(side_mask.sum()),
                                 'brier_sum': float(side_brier),
                                 'log_sum': side_log_sum,
                                 'zero_truth_probability_rows': 0},
            'three_class': {'count': int(singleton_mask.sum()),
                            'brier_sum': float(three_brier),
                            'log_sum': three_log_sum,
                            'zero_truth_probability_rows': 0},
        }
    return {'per_tf': report,
            'contract_sha256': factorized_time_query_scorer_contract()['contract_sha256'],
            'query_k_sha256': hashlib.sha256(query_k.tobytes()).hexdigest()}


class FactorizedTimeQueryMatchedScorer(TimeQueryMatchedScorer):
    """Exact v1 matched reducer with a primitive-logit side score path."""

    def add_block(self, step_start: int, *, query_k: np.ndarray,
                  validity_ns: np.ndarray, raw_logits: np.ndarray,
                  prior_activity: np.ndarray, prior_side_up: np.ndarray,
                  prior_three: np.ndarray, allowed: np.ndarray,
                  eligible: np.ndarray, mature: np.ndarray) -> None:
        classes, _, side_up = factorized_probabilities(raw_logits)
        _require(isinstance(allowed, np.ndarray) and allowed.ndim == 3
                 and raw_logits.ndim == 4
                 and raw_logits.shape == allowed.shape + (2,), 'BLOCK_SHAPE')
        super().add_block(step_start, query_k=query_k, validity_ns=validity_ns,
                          model_p=classes, model_side_up=side_up,
                          prior_activity=prior_activity, prior_side_up=prior_side_up,
                          prior_three=prior_three, allowed=allowed,
                          eligible=eligible, mature=mature)

    def result(self) -> dict:
        report = super().result()
        return {**report,
                'schema_version': 'gx1_market_state_time_query_matched_score_v2',
                'contract_sha256': factorized_time_query_scorer_contract()['contract_sha256']}
