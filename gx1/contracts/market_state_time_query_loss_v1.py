"""Pure allowed-set loss and scores for future M5-decision-time state queries.

The bound target owner supplies exact clocks and masks. Query times are known at
issue; future labels never enter the model forward. No fit or trade authority.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import torch

from gx1.contracts import market_state_five_endpoint_contract_v1 as five
from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K, TIMEFRAMES

K = tuple(int(value) for value in NATIVE_K)


def time_query_loss_contract() -> dict:
    value = {
        'schema_version': 'gx1_market_state_time_query_loss_v1',
        'role': 'research_only_no_entry_exit_or_trade_authority',
        'query_domain': 'future_M5_decision_ticks_after_closed_M5_issue',
        'native_k': list(K), 'timeframes': list(TIMEFRAMES),
        'query_k': 'int64_Bx5_future_M5_ticks_1_through_native_k_per_TF',
        'class_order': list(five.CLASS_NAMES),
        'eligible_mask': 'bool_Bx5_target_source_available',
        'mature_mask': 'bool_Bx5_validity_inside_same_fold',
        'informative_mask': 'eligible_and_mature_and_allowed_not_7',
        'allowed_bits': {'DOWN': 1, 'FLAT': 2, 'UP': 4,
                         'ACTIVE_SIDE_UNKNOWN': 5, 'ALL_UNKNOWN': 7},
        'loss': 'sum_of_per_TF_mean_negative_log_allowed_probability_divided_by_five',
        'empty_support': 'zero_loss_anchored_to_logits_skip_update',
        'activity_score': 'binary_Brier_on_bits_1_2_4_5',
        'side_score': 'conditional_UP_binary_Brier_and_log_on_bits_1_4',
        'three_class_score': 'multiclass_Brier_and_log_on_bits_1_2_4',
        'score_population': 'same_eligible_mature_cells_for_each_comparison',
        'future_labels_as_model_forward_inputs': False,
        'marginals_are_joint_episode_probabilities': False,
    }
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'),
                     allow_nan=False).encode()
    return {**value, 'contract_sha256': hashlib.sha256(raw).hexdigest()}


def _torch_query(logits: torch.Tensor, query_k: torch.Tensor) -> None:
    if not isinstance(logits, torch.Tensor):
        raise ValueError('FIVE_ENDPOINT_LOGITS')
    if not (isinstance(query_k, torch.Tensor)
            and query_k.dtype == torch.int64
            and query_k.shape == logits.shape[:2]
            and query_k.device == logits.device
            and bool(((query_k >= 1) & (query_k <= torch.tensor(
                K, dtype=torch.int64, device=query_k.device)[None, :])).all())):
        raise ValueError('TIME_QUERY_K')


def _numpy_query(probabilities: np.ndarray, query_k: np.ndarray) -> None:
    if not isinstance(probabilities, np.ndarray):
        raise ValueError('FIVE_ENDPOINT_PROBABILITIES')
    if not (isinstance(query_k, np.ndarray)
            and query_k.dtype == np.int64
            and query_k.shape == probabilities.shape[:2]
            and np.all((query_k >= 1) &
                       (query_k <= np.asarray(K, dtype=np.int64)[None, :]))):
        raise ValueError('TIME_QUERY_K')


def allowed_set_loss(logits: torch.Tensor, allowed: torch.Tensor,
                     eligible: torch.Tensor, mature: torch.Tensor,
                     query_k: torch.Tensor) -> dict:
    """Equal TF weighting; a TF without labels contributes zero, not extra weight."""
    _torch_query(logits, query_k)
    original = five.allowed_set_loss(logits, allowed, eligible, mature)
    zero = logits.mul(0).sum()
    terms = []
    for j, tf in enumerate(TIMEFRAMES):
        if original['per_tf_counts'][tf]:
            only_tf = torch.zeros_like(eligible)
            only_tf[:, j] = eligible[:, j]
            terms.append(five.allowed_set_loss(
                logits, allowed, only_tf, mature)['loss'])
        else:
            terms.append(zero)
    loss = sum(terms, zero) / len(TIMEFRAMES)
    if not bool(torch.isfinite(loss)):
        raise ValueError('TIME_QUERY_NONFINITE_LOSS')
    return {'loss': loss, 'count': original['count'],
            'per_tf_counts': original['per_tf_counts'],
            'per_tf_mean_loss': dict(zip(TIMEFRAMES, terms)),
            'informative_mask': original['informative_mask'],
            'skip_update': original['skip_update'],
            'contract_sha256': time_query_loss_contract()['contract_sha256']}


def proper_score_sums(probabilities: np.ndarray, allowed: np.ndarray,
                      eligible: np.ndarray, mature: np.ndarray,
                      query_k: np.ndarray) -> dict:
    """Score each TF on the same qualified cells as its loss contribution.

    Caller owns per-issue/TF, offset and week aggregation for dependent queries.
    """
    _numpy_query(probabilities, query_k)
    report = five.proper_score_sums(probabilities, allowed, eligible, mature)
    return {**report,
            'contract_sha256': time_query_loss_contract()['contract_sha256'],
            'query_k_sha256': hashlib.sha256(query_k.tobytes()).hexdigest()}
