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
from gx1.contracts import market_state_time_query_side_normalized_loss_v1 as state_loss
from gx1.contracts import market_state_time_query_scorer_v2 as scorer
from gx1.contracts.market_state_future_displacement_candidate_v1 import (
    BoundFutureDisplacement, contract as future_contract,
)
from gx1.contracts.market_state_time_query_native_F_model_v1 import contract as model_contract
from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K, TIMEFRAMES, TICK_NS

NATIVE_QUERY = np.asarray(NATIVE_K, dtype=np.int64)[None, :]
METRICS = ('activity', 'conditional_side', 'three_class')


def m5_equal_observation_profile() -> dict:
    """Fixed fold-1 TRAIN population from the verified paired weight contrast."""
    value = {
        'schema_version': 'gx1_ta_fold1_m5_equal_observation_weighting_v1',
        'fold_id': 1, 'pass_index': 0, 'seed': 20260911,
        'batch_size': 16, 'fit_rows': 29574, 'fit_batches': 1849,
        'parent_axis_sha256': '1c5a8c36a94b8fa39d613f0bc122cf869dc9fa3c2f87dc64c0506acd2f55a90f',
        'issue_axis_sha256': 'ebb3489c45b01b6fd8e77bf27529d650beb6883ba84b57ee936dc549ec74e427',
        'feature_view_sha256': '72b650198f5044d809957a7805b9f1d9160a1400f7026487d63c5bb78979bbb2',
        'fit_cache_sha256': 'e8a562314d81b5821931df0ff78ff563b0968015eac0eaa141e909c80d6fc22f',
        'weight_result_sha256': '4d0189738896d76a7a459f74a0c1a262ce5cf16f19432f3513015c9bd45fef5c',
        'weight_verification_sha256': '85482fd6fabd6d97ec14769a50f44b47e95d9b4102f0d9c4a06e6b5ae4a3c9fe',
        'state': {'known_side_rows': 10010, 'nonempty_batches': 1799},
        'future': {'known_side_rows': 11909, 'nonempty_batches': 1844},
        'S_observations': 'sampled_pass0_query_M5_k1_equal_native_M5',
        'F_observations': 'native_M5_k1',
        'known_side_mask': 'eligible_and_mature_and_allowed_in_1_4',
        'batch_side_term_before_TF_divisor': 'sum_known_side_BCE_times_q_over_N',
        'mass_preserved_over_mean_of_fit_batches': 'q_over_5B',
        'later_data_used_for_normalization': False,
    }
    return _seal(value)


def require_m5_equal_observation_axes(parent_rows, issue_ns) -> dict:
    """Verify the complete immutable fit order once, before any fit forward."""
    from gx1.audit.qualify_market_state_mature_prefix_normalization_v1 import _axis_sha
    profile = m5_equal_observation_profile()
    if not (isinstance(parent_rows, np.ndarray) and isinstance(issue_ns, np.ndarray)
            and parent_rows.dtype == issue_ns.dtype == np.dtype('int64')
            and parent_rows.shape == issue_ns.shape == (profile['fit_rows'],)
            and _axis_sha(parent_rows) == profile['parent_axis_sha256']
            and _axis_sha(issue_ns) == profile['issue_axis_sha256']):
        raise ValueError('M5_EQUAL_OBSERVATION_FIT_AXES')
    return profile


def _m5_training_profile(ta_only, dedicated_m5_side, enabled):
    if type(enabled) is not bool or (enabled and (
            ta_only is not True or dedicated_m5_side is not True)):
        raise ValueError('NATIVE_F_M5_EQUAL_OBSERVATION_MODE')
    return m5_equal_observation_profile() if enabled else None


def _require_m5_direction_mode(enabled, profile):
    if type(enabled) is not bool or (enabled and profile is None):
        raise ValueError('NATIVE_F_M5_DIRECTION_ONLY_MODE')


def contract(*, ta_only: bool = False, dedicated_m5_side: bool = False,
             m5_equal_observation: bool = False,
             dedicated_non_m5_side: bool = False,
             dedicated_tf_activity: bool = False,
             m5_direction_only: bool = False) -> dict:
    if type(ta_only) is not bool:
        raise ValueError('NATIVE_F_OBJECTIVE_PROFILE')
    profile = _m5_training_profile(ta_only, dedicated_m5_side, m5_equal_observation)
    _require_m5_direction_mode(m5_direction_only, profile)
    value = {
        'schema_version': 'gx1_market_state_native_F_objective_candidate_v1',
        'role': 'research_only_no_fit_or_trade_authority',
        'S_loss_contract_sha256': state_loss.contract()['contract_sha256'],
        'F_target_contract_sha256': future_contract()['contract_sha256'],
        'dual_model_contract_sha256': model_contract(ta_only=ta_only, dedicated_m5_side=dedicated_m5_side,
            dedicated_non_m5_side=dedicated_non_m5_side,
            dedicated_tf_activity=dedicated_tf_activity)['contract_sha256'],
        'F_score_contract_sha256': scorer.factorized_time_query_scorer_contract()['contract_sha256'],
        'combined_loss': 'side_normalized_S_loss_plus_prebound_positive_weight_times_side_normalized_F_loss',
        'F_query': 'fixed_native_k_per_TF_every_row',
        'F_reference': 'earlier_mature_TRAIN_plus_one_activity_and_conditional_side_prior',
        'F_score': 'later_TRAIN_paired_Brier_and_log_by_TF_and_UTC_issue_week',
        'S_scoring': 'existing_v2_scorer_unchanged',
        'F_target_as_forward_input': False,
    }
    if profile is not None:
        value['m5_equal_observation'] = True
        value['m5_side_weighting_profile'] = profile
        value['S_loss_contract_sha256'] = state_loss.contract(
            m5_side_normalization=profile['state'])['contract_sha256']
        value['F_loss_contract_sha256'] = state_loss.contract(
            m5_side_normalization=profile['future'])['contract_sha256']
        value['combined_loss'] = 'M5_equal_observation_S_loss_plus_prebound_positive_weight_times_M5_equal_observation_F_loss'
    if m5_direction_only:
        value['m5_direction_only'] = True
        value['combined_loss'] = 'existing_S_M5_side_plus_prebound_F_weight_times_existing_F_M5_side'
        value['unchanged_M5_coefficients'] = 'original_q_over_N_and_TF_divisor_five'
        value['empty_M5_side_support'] = 'skip_optimizer_update_without_imputing_direction'
        value['other_tasks'] = 'forward_and_reporting_retained_no_direct_loss_gradient'
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return {**value, 'contract_sha256': hashlib.sha256(raw).hexdigest()}


def combined_loss(*, state_raw: torch.Tensor, future_raw: torch.Tensor,
                  state_truth: dict, future_truth: dict,
                  state_query_k: torch.Tensor, future_weight: float,
                  ta_only: bool = False, dedicated_m5_side: bool = False,
                  m5_equal_observation: bool = False,
                  dedicated_non_m5_side: bool = False,
                  dedicated_tf_activity: bool = False,
                  m5_direction_only: bool = False) -> dict:
    """Reuse v2 masked loss on two distinct heads; weight is plan-owned."""
    profile = _m5_training_profile(ta_only, dedicated_m5_side, m5_equal_observation)
    _require_m5_direction_mode(m5_direction_only, profile)
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
        tensor(state_truth['mature'], 'bool'), state_query_k,
        m5_side_normalization=profile['state'] if profile is not None else None)
    f = state_loss.allowed_set_loss(
        future_raw, tensor(future_truth['allowed'], 'uint8'),
        tensor(future_truth['eligible'], 'bool'),
        tensor(future_truth['mature'], 'bool'), native,
        m5_side_normalization=profile['future'] if profile is not None else None)
    total = s['loss'] + future_weight * f['loss']
    skip_update = s['skip_update'] and f['skip_update']
    if m5_direction_only:
        # Keep the canonical observation masks, coefficients and TF divisor.
        def component(raw, truth, report, normalizer):
            target = tensor(truth['allowed'], 'uint8')[:, 0] == 4
            cells = torch.nn.functional.binary_cross_entropy_with_logits(
                raw[:, 0, 1], target.to(raw.dtype), reduction='none')
            known = report['side_known_mask'][:, 0]
            value = torch.where(known, cells, torch.zeros_like(cells)).sum()
            return value * (normalizer['nonempty_batches']
                            / normalizer['known_side_rows']) / len(TIMEFRAMES)
        total = (component(state_raw, state_truth, s, profile['state'])
                 + future_weight * component(future_raw, future_truth, f, profile['future']))
        skip_update = not (bool(s['side_known_mask'][:, 0].any())
                           or bool(f['side_known_mask'][:, 0].any()))
    if not bool(torch.isfinite(total)):
        raise ValueError('NATIVE_F_LOSS_NONFINITE')
    return {'loss': total, 'state': s, 'future': f,
            'skip_update': skip_update,
            'future_weight': future_weight,
            'contract_sha256': contract(ta_only=ta_only, dedicated_m5_side=dedicated_m5_side,
                m5_equal_observation=m5_equal_observation,
                dedicated_non_m5_side=dedicated_non_m5_side,
                dedicated_tf_activity=dedicated_tf_activity,
                m5_direction_only=m5_direction_only)['contract_sha256']}


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
                        fold_id: int, score_raw: np.ndarray, ta_only: bool = False, dedicated_m5_side: bool = False) -> dict:
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
            'contract_sha256': contract(ta_only=ta_only, dedicated_m5_side=dedicated_m5_side)['contract_sha256'],
            'test_data_used': False}

# Endpoint measurements use the same fixed five queries before/after fit.
# Their global support denominators are deliberately separate from minibatch
# optimization losses and from sampled-S training-query observations.
def _seal(value: dict) -> dict:
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return {**value, 'contract_sha256': hashlib.sha256(raw).hexdigest()}


def fit_endpoint_priors(state_bound, future_bound, *, axes: dict, fold_id: int,
                        ta_only: bool = False, dedicated_m5_side: bool = False) -> dict:
    """Freeze both endpoint priors on exactly the declared mature fit axis."""
    from gx1.contracts.market_state_five_endpoint_target_binding_v1 import BoundFiveEndpointTargets
    if not (type(state_bound) is BoundFiveEndpointTargets
            and type(future_bound) is BoundFutureDisplacement
            and type(fold_id) is int and 1 <= fold_id <= 5):
        raise ValueError('NATIVE_F_ENDPOINT_PRIOR_OWNERS')
    expected = state_bound.mature_oof_axes(fold_id=fold_id)
    for key in ('fit_parent_rows', 'fit_decision_ns'):
        if not (isinstance(axes.get(key), np.ndarray)
                and np.array_equal(axes[key], expected[key])):
            raise ValueError('NATIVE_F_ENDPOINT_PRIOR_EXACT_FIT_AXIS')
    rows, issues = expected['fit_parent_rows'], expected['fit_decision_ns']
    result = {
        'schema_version': 'gx1_native_S_F_frozen_endpoint_priors_v1',
        'fold_id': fold_id, 'fit_rows': len(rows),
        'fit_parent_rows_sha256': expected['fit_parent_rows_sha256'],
        'fit_decision_ns_sha256': expected['fit_decision_ns_sha256'],
        'native_query_k': list(NATIVE_K),
        'loss_contract_sha256': contract(ta_only=ta_only, dedicated_m5_side=dedicated_m5_side)['contract_sha256'],
        'tasks': {},
    }
    for name, bound in (('state', state_bound), ('future', future_bound)):
        truth = bound.gather(rows, issues, fold_id=fold_id, partition='fit')
        if not bool(truth['mature'].all()):
            raise ValueError('NATIVE_F_ENDPOINT_PRIOR_IMMATURE')
        raw, counts = _prior_raw(truth)
        result['tasks'][name] = {'raw_logits': raw.tolist(), 'counts': counts}
    return _seal(result)


def _endpoint_prior_arrays(value: dict, fold_id: int, *, ta_only: bool = False, dedicated_m5_side: bool = False) -> dict:
    if not (isinstance(value, dict)
            and value.get('schema_version') == 'gx1_native_S_F_frozen_endpoint_priors_v1'
            and value.get('fold_id') == fold_id
            and value.get('native_query_k') == list(NATIVE_K)
            and value.get('loss_contract_sha256') == contract(ta_only=ta_only, dedicated_m5_side=dedicated_m5_side)['contract_sha256']
            and value.get('contract_sha256') == _seal(
                {k: v for k, v in value.items() if k != 'contract_sha256'})['contract_sha256']
            and set(value.get('tasks', {})) == {'state', 'future'}):
        raise ValueError('NATIVE_F_ENDPOINT_PRIOR_BINDING')
    result = {}
    for name in ('state', 'future'):
        raw = np.asarray(value['tasks'][name]['raw_logits'], dtype=np.float64)
        if not (raw.shape == (len(TIMEFRAMES), 2) and np.isfinite(raw).all()):
            raise ValueError('NATIVE_F_ENDPOINT_PRIOR_LOGITS')
        result[name] = raw
    return result


def require_endpoint_priors(value: dict, *, fold_id: int = 1,
                            ta_only: bool = False, dedicated_m5_side: bool = False) -> dict:
    """Validate the frozen prior seal, endpoint axes and explicit model route."""
    _endpoint_prior_arrays(value, fold_id, ta_only=ta_only, dedicated_m5_side=dedicated_m5_side)
    return value


def _endpoint_task_sums(raw: np.ndarray, truth: dict, query: np.ndarray) -> dict:
    report = scorer.proper_score_sums(
        raw, truth['allowed'], truth['eligible'], truth['mature'], query)['per_tf']
    for j, tf in enumerate(TIMEFRAMES):
        mask = truth['eligible'][:, j] & truth['mature'][:, j] & (truth['allowed'][:, j] != 7)
        active = truth['allowed'][mask, j] != 2
        logits = raw[mask, j, 0].astype(np.float64)
        nll = np.logaddexp(0.0, np.where(active, -logits, logits))
        total = float(nll.sum(dtype=np.float64))
        if not np.isfinite(total):
            raise ValueError('NATIVE_F_ENDPOINT_ACTIVITY_LOG')
        report[tf]['activity']['log_sum'] = total
    return report


def endpoint_measurement_sums(*, state_raw: np.ndarray, future_raw: np.ndarray,
                              state_truth: dict, future_truth: dict,
                              parent_rows: np.ndarray, issue_ns: np.ndarray,
                              fold_id: int, partition: str, future_weight: float,
                              model_contract_sha256: str, feature_view_sha256: str,
                              frozen_priors: dict | None = None, ta_only: bool = False, dedicated_m5_side: bool = False) -> dict:
    """Additive endpoint sufficient statistics; never average minibatch losses."""
    if not (type(fold_id) is int and 1 <= fold_id <= 5 and partition in ('fit', 'score')
            and type(future_weight) is float and np.isfinite(future_weight)
            and 0 < future_weight <= 1
            and isinstance(parent_rows, np.ndarray) and parent_rows.dtype == np.int64
            and isinstance(issue_ns, np.ndarray) and issue_ns.dtype == np.int64
            and parent_rows.ndim == 1 and issue_ns.shape == parent_rows.shape
            and len(parent_rows) > 0
            and bool(np.all(np.diff(parent_rows) > 0))
            and bool(np.all(np.diff(issue_ns) > 0))):
        raise ValueError('NATIVE_F_ENDPOINT_MEASUREMENT_AXIS')
    query = np.broadcast_to(NATIVE_QUERY, (len(parent_rows), len(TIMEFRAMES))).copy()
    digest = hashlib.sha256()
    for value in (parent_rows, issue_ns, query):
        digest.update(np.ascontiguousarray(value).tobytes())
    for truth in (state_truth, future_truth):
        if not (np.array_equal(truth['parent_rows'], parent_rows)
                and np.array_equal(truth['issue_ns'], issue_ns)
                and isinstance(truth['validity_ns'], np.ndarray)
                and truth['validity_ns'].dtype == np.int64
                and np.array_equal(truth['validity_ns'], issue_ns[:, None] + query * TICK_NS)):
            raise ValueError('NATIVE_F_ENDPOINT_MEASUREMENT_TRUTH_AXIS')
        for key in ('allowed', 'eligible', 'mature', 'validity_ns'):
            digest.update(np.ascontiguousarray(truth[key]).tobytes())
    scores = {'model': {
        'state': _endpoint_task_sums(state_raw, state_truth, query),
        'future': _endpoint_task_sums(future_raw, future_truth, query),
    }}
    prior_sha = None
    if frozen_priors is not None:
        priors = _endpoint_prior_arrays(frozen_priors, fold_id, ta_only=ta_only, dedicated_m5_side=dedicated_m5_side)
        scores['prior'] = {}
        for name, truth in (('state', state_truth), ('future', future_truth)):
            repeated = np.broadcast_to(priors[name][None, :, :], state_raw.shape)
            scores['prior'][name] = _endpoint_task_sums(repeated, truth, query)
        prior_sha = frozen_priors['contract_sha256']
    return {
        'schema_version': 'gx1_native_S_F_endpoint_batch_measurement_v1',
        'rows': len(parent_rows), 'fold_id': fold_id, 'partition': partition,
        'future_weight': future_weight, 'native_query_k': list(NATIVE_K),
        'loss_contract_sha256': contract(ta_only=ta_only, dedicated_m5_side=dedicated_m5_side)['contract_sha256'],
        'model_contract_sha256': model_contract_sha256,
        'feature_view_sha256': feature_view_sha256,
        'frozen_prior_contract_sha256': prior_sha,
        'observation_sha256': digest.hexdigest(), 'scores': scores,
    }


_MEASUREMENT_BINDINGS = (
    'fold_id', 'partition', 'future_weight', 'native_query_k',
    'loss_contract_sha256', 'model_contract_sha256', 'feature_view_sha256',
    'frozen_prior_contract_sha256',
)


def merge_endpoint_measurements(total: dict | None, batch: dict) -> dict:
    """Merge counts and numerators, retaining exact batch/observation identity."""
    import copy
    if not (isinstance(batch, dict)
            and batch.get('schema_version') == 'gx1_native_S_F_endpoint_batch_measurement_v1'
            and type(batch.get('rows')) is int and batch['rows'] > 0):
        raise ValueError('NATIVE_F_ENDPOINT_MERGE_BATCH')
    if total is None:
        total = {key: copy.deepcopy(batch[key]) for key in _MEASUREMENT_BINDINGS}
        total.update(schema_version='gx1_native_S_F_endpoint_accumulator_v1',
                     rows=batch['rows'], batches=1,
                     batch_observation_sha256s=[batch['observation_sha256']],
                     scores=copy.deepcopy(batch['scores']))
        return total
    if not (total.get('schema_version') == 'gx1_native_S_F_endpoint_accumulator_v1'
            and all(total[key] == batch[key] for key in _MEASUREMENT_BINDINGS)
            and set(total['scores']) == set(batch['scores'])):
        raise ValueError('NATIVE_F_ENDPOINT_MERGE_BINDING')
    for role in total['scores']:
        for task in ('state', 'future'):
            for tf in TIMEFRAMES:
                for metric in METRICS:
                    destination = total['scores'][role][task][tf][metric]
                    source = batch['scores'][role][task][tf][metric]
                    if set(destination) != set(source):
                        raise ValueError('NATIVE_F_ENDPOINT_MERGE_METRIC')
                    for key, value in source.items():
                        destination[key] += value
    total['rows'] += batch['rows']
    total['batches'] += 1
    total['batch_observation_sha256s'].append(batch['observation_sha256'])
    return total


def finalize_endpoint_measurement(total: dict) -> dict:
    """Compute full-cohort means with separate activity and side support."""
    import copy
    if not (isinstance(total, dict)
            and total.get('schema_version') == 'gx1_native_S_F_endpoint_accumulator_v1'
            and total.get('rows', 0) > 0 and total.get('batches', 0) > 0):
        raise ValueError('NATIVE_F_ENDPOINT_FINALIZE')
    result = copy.deepcopy(total)
    result['schema_version'] = 'gx1_native_S_F_endpoint_measurement_v1'
    batches = result.pop('batch_observation_sha256s')
    result['observation_chain_sha256'] = hashlib.sha256(
        json.dumps(batches, separators=(',', ':')).encode()).hexdigest()
    result['means'] = {}
    result['endpoint_loss'] = {}
    for role, tasks in result['scores'].items():
        result['means'][role] = {}
        task_loss = {}
        for task in ('state', 'future'):
            result['means'][role][task] = {}
            tf_losses = []
            for tf in TIMEFRAMES:
                metrics = tasks[task][tf]
                means = {}
                for metric, stats in metrics.items():
                    if stats['count']:
                        means[metric] = {'brier': stats['brier_sum'] / stats['count']}
                        if 'log_sum' in stats:
                            means[metric]['log'] = stats['log_sum'] / stats['count']
                result['means'][role][task][tf] = means
                tf_losses.append(sum(
                    metrics[metric]['log_sum'] / metrics[metric]['count']
                    if metrics[metric]['count'] else 0.0
                    for metric in ('activity', 'conditional_side')))
            task_loss[task] = sum(tf_losses) / len(TIMEFRAMES)
        result['endpoint_loss'][role] = {
            **task_loss,
            'combined': task_loss['state'] + total['future_weight'] * task_loss['future'],
        }
    return _seal(result)


def rebind_endpoint_priors_for_dedicated_m5(value: dict) -> dict:
    """Reuse exact frozen TRAIN counts/logits under the explicit new model route."""
    import copy
    require_endpoint_priors(value, fold_id=1, ta_only=True)
    rebound = copy.deepcopy({k: v for k, v in value.items() if k != 'contract_sha256'})
    rebound['loss_contract_sha256'] = contract(
        ta_only=True, dedicated_m5_side=True)['contract_sha256']
    rebound = _seal(rebound)
    require_endpoint_priors(rebound, fold_id=1, ta_only=True, dedicated_m5_side=True)
    return rebound
