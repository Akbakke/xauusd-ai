"""Separate S and native F losses and paired chronological F proper scores.

The caller prebinds the positive F weight; this owner cannot admit a fit or
select a weight from later-fold scores. Both heads use existing factorized math.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import torch

from gx1.contracts import market_state_five_endpoint_prior_v1 as prior
from gx1.contracts import market_state_time_query_loss_v2 as state_loss
from gx1.contracts import market_state_time_query_scorer_v2 as scorer
from gx1.contracts.market_state_future_displacement_candidate_v1 import (
    BoundFutureDisplacement, contract as future_contract,
)
from gx1.contracts.market_state_time_query_native_F_model_v1 import contract as model_contract
from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K, TIMEFRAMES

NATIVE_QUERY = np.asarray(NATIVE_K, dtype=np.int64)[None, :]
METRICS = ('activity', 'conditional_side', 'three_class')


def contract() -> dict:
    value = {
        'schema_version': 'gx1_market_state_native_F_objective_candidate_v1',
        'role': 'research_only_no_fit_or_trade_authority',
        'S_loss_contract_sha256': state_loss.side_normalized_time_query_loss_contract()['contract_sha256'],
        'F_target_contract_sha256': future_contract()['contract_sha256'],
        'dual_model_contract_sha256': model_contract()['contract_sha256'],
        'F_score_contract_sha256': scorer.factorized_time_query_scorer_contract()['contract_sha256'],
        'combined_loss': 'side_normalized_S_loss_plus_prebound_positive_weight_times_side_normalized_F_loss',
        'F_query': 'fixed_native_k_per_TF_every_row',
        'F_reference': 'earlier_mature_TRAIN_plus_one_activity_and_conditional_side_prior',
        'F_score': 'later_TRAIN_paired_Brier_and_log_by_TF_and_UTC_issue_week',
        'S_scoring': 'existing_v2_scorer_unchanged',
        'F_target_as_forward_input': False,
    }
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return {**value, 'contract_sha256': hashlib.sha256(raw).hexdigest()}


def combined_loss(*, state_raw: torch.Tensor, future_raw: torch.Tensor,
                  state_truth: dict, future_truth: dict,
                  state_query_k: torch.Tensor, future_weight: float) -> dict:
    """Reuse v2 masked loss on two distinct heads; weight is plan-owned."""
    if not (type(future_weight) is float and np.isfinite(future_weight)
            and 0 < future_weight <= 1
            and isinstance(state_raw, torch.Tensor)
            and isinstance(future_raw, torch.Tensor)
            and state_raw.shape == future_raw.shape
            and state_raw.ndim == 3 and state_raw.shape[1:] == (5, 2)
            and isinstance(state_query_k, torch.Tensor)
            and state_query_k.shape == state_raw.shape[:2]
            and all(set(truth) >= {'allowed', 'eligible', 'mature'}
                    for truth in (state_truth, future_truth))):
        raise ValueError('NATIVE_F_LOSS_INPUT')
    device = state_raw.device
    native = torch.tensor(NATIVE_K, dtype=torch.int64, device=device)[None, :].expand(
        state_raw.shape[0], -1)
    def tensor(value, dtype):
        if not (isinstance(value, np.ndarray) and value.shape == state_raw.shape[:2]
                and value.dtype == np.dtype(dtype)):
            raise ValueError('NATIVE_F_TRUTH_SHAPE')
        return torch.from_numpy(value).to(device=device)
    s = state_loss.allowed_set_loss(
        state_raw, tensor(state_truth['allowed'], 'uint8'),
        tensor(state_truth['eligible'], 'bool'),
        tensor(state_truth['mature'], 'bool'), state_query_k, normalize_known_side=True)
    f = state_loss.allowed_set_loss(
        future_raw, tensor(future_truth['allowed'], 'uint8'),
        tensor(future_truth['eligible'], 'bool'),
        tensor(future_truth['mature'], 'bool'), native, normalize_known_side=True)
    total = s['loss'] + future_weight * f['loss']
    if not bool(torch.isfinite(total)):
        raise ValueError('NATIVE_F_LOSS_NONFINITE')
    return {'loss': total, 'state': s, 'future': f,
            'skip_update': s['skip_update'] and f['skip_update'],
            'future_weight': future_weight,
            'contract_sha256': contract()['contract_sha256']}


def _weeks(issues: np.ndarray) -> np.ndarray:
    days = issues // (24 * 60 * 60 * 1_000_000_000)
    monday = days - (days + 3) % 7
    return (np.datetime64('1970-01-01', 'D')
            + monday.astype('timedelta64[D]')).astype(str)


def _prior_raw(fit_truth: dict) -> tuple[np.ndarray, dict]:
    fitted = prior.fit_activity_side_prior(
        fit_truth['allowed'], fit_truth['eligible'], fit_truth['mature'])
    classes = fitted['probabilities']
    active = classes[:, 0] + classes[:, 2]
    side = classes[:, 2] / active
    raw = np.column_stack((np.log(active / (1 - active)),
                           np.log(side / (1 - side)))).astype(np.float64)
    if not (raw.shape == (len(TIMEFRAMES), 2) and np.isfinite(raw).all()):
        raise ValueError('NATIVE_F_PRIOR_NUMERICS')
    return raw, fitted['counts']


def score_native_F_fold(*, bound: BoundFutureDisplacement, axes: dict,
                        fold_id: int, score_raw: np.ndarray) -> dict:
    """Compare one frozen later-TRAIN score to its earlier-TRAIN F prior."""
    if not (type(bound) is BoundFutureDisplacement
            and type(fold_id) is int and 1 <= fold_id <= 5
            and isinstance(axes, dict)
            and isinstance(score_raw, np.ndarray)
            and score_raw.dtype in (np.dtype('float32'), np.dtype('float64'))
            and score_raw.shape == (len(axes['score_parent_rows']), 5, 2)
            and np.isfinite(score_raw).all()):
        raise ValueError('NATIVE_F_SCORE_INPUT')
    fit = bound.gather(axes['fit_parent_rows'], axes['fit_decision_ns'],
                       fold_id=fold_id, partition='fit')
    score = bound.gather(axes['score_parent_rows'], axes['score_decision_ns'],
                         fold_id=fold_id, partition='score')
    prior_raw, prior_counts = _prior_raw(fit)
    repeated = np.broadcast_to(prior_raw[None, :, :], score_raw.shape)
    native = np.broadcast_to(NATIVE_QUERY, score['allowed'].shape).copy()
    def paired(rows):
        labels = score['allowed'][rows]
        eligible = score['eligible'][rows]
        mature = score['mature'][rows]
        actual = scorer.proper_score_sums(score_raw[rows], labels, eligible,
                                          mature, native[rows])['per_tf']
        reference = scorer.proper_score_sums(repeated[rows], labels, eligible,
                                             mature, native[rows])['per_tf']
        result = {}
        for tf in TIMEFRAMES:
            result[tf] = {}
            for metric in METRICS:
                m, p = actual[tf][metric], reference[tf][metric]
                if m['count'] != p['count']:
                    raise RuntimeError('NATIVE_F_SCORE_PAIR')
                result[tf][metric] = {'count': m['count'],
                                      'model_brier_sum': m['brier_sum'],
                                      'prior_brier_sum': p['brier_sum'],
                                      'prior_minus_model_brier_sum': p['brier_sum']-m['brier_sum']}
                if 'log_sum' in m:
                    result[tf][metric]['model_log_sum'] = m['log_sum']
                    result[tf][metric]['prior_log_sum'] = p['log_sum']
        return result
    overall = paired(slice(None))
    weeks = _weeks(score['issue_ns'])
    weekly = {str(week): paired(weeks == week) for week in np.unique(weeks)}
    for tf in TIMEFRAMES:
        for metric in METRICS:
            full = overall[tf][metric]
            for key in ('count', 'model_brier_sum', 'prior_brier_sum'):
                total = sum(week[tf][metric][key] for week in weekly.values())
                if not np.isclose(total, full[key], atol=1e-7, rtol=0):
                    raise RuntimeError('NATIVE_F_WEEK_PARTITION')
    return {'schema_version': 'gx1_market_state_native_F_fold_score_candidate_v1',
            'fold_id': fold_id,
            'fit_parent_rows_sha256': axes['fit_parent_rows_sha256'],
            'score_parent_rows_sha256': axes['score_parent_rows_sha256'],
            'prefix_counts': prior_counts,
            'overall': overall, 'by_utc_week': weekly,
            'contract_sha256': contract()['contract_sha256'],
            'test_data_used': False}
