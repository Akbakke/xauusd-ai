"""One fixed Fold2 non-GC S/F plus auxiliary M5 high/low TRAIN loss."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import torch

from gx1.contracts.market_state_ta_integrated_model_v1 import masked_structure_hit_sums
from gx1.contracts.market_state_native_m5_targets_v1 import TIMEFRAMES, contract as native_target_contract

STRUCTURE_WEIGHT = 0.25
FUTURE_WEIGHT = 1.0


def contract() -> dict:
    from gx1.contracts.market_state_time_query_native_F_objective_v1 import contract as dual_loss_contract
    base = dual_loss_contract(ta_only=True, dedicated_m5_side=True,
        dedicated_non_m5_side=True, dedicated_tf_activity=True,
        direct_tf_hidden=True, m5_equal_observation=True, fold_id=2)
    value = {
        'schema_version': 'gx1_ta_integrated_fold2_objective_v1',
        'role': 'research_only_no_fit_or_trade_authority',
        'S_F_loss_contract_sha256': base['contract_sha256'],
        'S_F': 'existing_five_TF_activity_and_known_conditional_side_masks_with_fold2_M5_equal_observation',
        'F_weight': FUTURE_WEIGHT,
        'structure_targets': ['next_high_hits_frozen_high', 'next_low_hits_frozen_low'],
        'structure_unknowns': 'masked_zero_gradient_no_imputed_negative_or_flat',
        'structure_loss': 'mean_observed_high_BCE_plus_mean_observed_low_BCE_divided_by_two_per_batch',
        'structure_weight': STRUCTURE_WEIGHT,
        'total': 'S_F_loss_plus_structure_weight_times_structure_loss',
        'test_data_used': False,
    }
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return {**value, 'contract_sha256': hashlib.sha256(raw).hexdigest()}


def integrated_loss(*, model_output: dict, state_truth: dict, future_truth: dict,
                    state_query_k: torch.Tensor, structure_hit: np.ndarray,
                    structure_observed: np.ndarray) -> dict:
    from gx1.contracts.market_state_time_query_model_v2 import factorized_log_probabilities
    from gx1.contracts.market_state_time_query_native_F_objective_v1 import combined_loss
    if not (type(model_output) is dict and set(model_output) == {
            'state_activity_side_logits', 'state_log_probs',
            'future_activity_side_logits', 'future_log_probs',
            'm5_structure_hit_logits'}):
        raise ValueError('TA_INTEGRATED_LOSS_MODEL_OUTPUT')
    raw = model_output['m5_structure_hit_logits']
    for task in ('state', 'future'):
        logits = model_output[task + '_activity_side_logits']
        log_probs = model_output[task + '_log_probs']
        if not (isinstance(logits, torch.Tensor) and logits.dtype == torch.float32
                and logits.ndim == 3 and logits.shape[1:] == (5, 2)
                and isinstance(log_probs, torch.Tensor)
                and log_probs.shape == (logits.shape[0], 5, 3)
                and log_probs.device == logits.device
                and torch.equal(log_probs, factorized_log_probabilities(logits))):
            raise ValueError('TA_INTEGRATED_LOSS_FACTOR_PROBABILITIES')
    if not (isinstance(raw, torch.Tensor) and raw.dtype == torch.float32
            and raw.ndim == 2 and raw.shape[1] == 2
            and isinstance(structure_hit, np.ndarray)
            and isinstance(structure_observed, np.ndarray)
            and structure_hit.dtype == structure_observed.dtype == np.bool_
            and structure_hit.shape == structure_observed.shape == tuple(raw.shape)
            and bool(np.all(~structure_hit | structure_observed))):
        raise ValueError('TA_INTEGRATED_LOSS_STRUCTURE')
    sf = combined_loss(state_raw=model_output['state_activity_side_logits'],
        future_raw=model_output['future_activity_side_logits'],
        state_truth=state_truth, future_truth=future_truth,
        state_query_k=state_query_k, future_weight=FUTURE_WEIGHT,
        ta_only=True, dedicated_m5_side=True, dedicated_non_m5_side=True,
        dedicated_tf_activity=True, direct_tf_hidden=True,
        m5_equal_observation=True, fold_id=2)
    structure = masked_structure_hit_sums(raw,
        torch.from_numpy(structure_hit).to(device=raw.device),
        torch.from_numpy(structure_observed).to(device=raw.device))
    per_side = structure['loss_sums'] / structure['observed_counts'].clamp(min=1)
    auxiliary = per_side.mean()
    total = sf['loss'] + STRUCTURE_WEIGHT * auxiliary
    if not bool(torch.isfinite(total)):
        raise ValueError('TA_INTEGRATED_LOSS_NONFINITE')
    return {'loss': total, 'S_F_loss': sf['loss'], 'structure_loss': auxiliary,
            'structure_loss_per_side': per_side,
            'structure_observed_counts': structure['observed_counts'],
            'state_per_tf_counts': sf['state']['per_tf_counts'],
            'future_per_tf_counts': sf['future']['per_tf_counts'],
            'skip_update': sf['skip_update'] and not bool(structure['observed_counts'].any()),
            'contract_sha256': contract()['contract_sha256']}


def native_bar_objective_contract():
    value = {
        'schema_version': 'gx1_native_m5_bar_direction_objective_v1',
        'target_contract_sha256': native_target_contract()['contract_sha256'],
        'timeframes': list(TIMEFRAMES), 'class_order': [-1, 0, 1],
        'close_mask': 'target_present_and_mature_and_observed_previous',
        'open_prediction_mask': 'close_mask_and_not_open_known_at_issue',
        'known_open': 'reported_as_observed_context_separately_no_prediction_credit',
        'loss': 'unweighted_cross_entropy_sum_over_predictive_cells_divided_by_predictive_count',
        'aggregation': 'per_comparison_per_TF_loss_sums_and_counts_before_division',
        'scores': 'same_masks_three_class_log_loss_Brier_confusion_and_unique_target_rows',
        'future_metadata_as_model_input': False,
    }
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return {**value, 'contract_sha256': hashlib.sha256(raw).hexdigest()}


def _native_truth(truth, *, rows, fold_upper_ns, target_contract_sha256):
    def need(ok, reason):
        if not bool(ok):
            raise ValueError('NATIVE_BAR_OBJECTIVE_' + reason)
    need(target_contract_sha256 == native_target_contract()['contract_sha256'],
         'TARGET_CONTRACT')
    need(type(truth) is dict and type(fold_upper_ns) is int and fold_upper_ns > 0, 'TRUTH')
    shape = (rows, len(TIMEFRAMES))
    for key in ('target_present', 'observed_previous', 'mature', 'eligible', 'open_known_at_issue'):
        need(isinstance(truth.get(key), np.ndarray) and truth[key].dtype == np.bool_
             and truth[key].shape == shape, 'MASK_' + key)
    for key in ('open_sign', 'close_sign'):
        need(isinstance(truth.get(key), np.ndarray) and truth[key].dtype == np.int8
             and truth[key].shape == shape and np.isin(truth[key], (-1, 0, 1)).all(),
             'SIGN_' + key)
    for key in ('target_row', 'validity_ns'):
        need(isinstance(truth.get(key), np.ndarray) and truth[key].dtype == np.int64
             and truth[key].shape == shape, 'AXIS_' + key)
    issues = truth.get('issue_ns')
    need(isinstance(issues, np.ndarray) and issues.dtype == np.int64
         and issues.shape == (rows,) and np.all(issues > 0)
         and np.all(issues < fold_upper_ns) and np.all(issues % 300_000_000_000 == 0)
         and fold_upper_ns % 300_000_000_000 == 0, 'ISSUE_AXIS')
    present, mature = truth['target_present'], truth['mature']
    need(np.all(truth['validity_ns'][present] % 300_000_000_000 == 0)
         and np.all(truth['validity_ns'][present] > np.broadcast_to(issues[:, None], shape)[present])
         and np.array_equal(mature, present & (truth['validity_ns'] < fold_upper_ns))
         and np.array_equal(truth['eligible'], mature & truth['observed_previous'])
         and np.array_equal(truth['target_row'] >= 0, mature)
         and not np.any(truth['open_known_at_issue'] & ~present), 'OBSERVABILITY')
    return {'close': truth['eligible'],
            'open': truth['eligible'] & ~truth['open_known_at_issue'],
            'open_known': truth['eligible'] & truth['open_known_at_issue']}


def native_bar_direction_loss(*, model_output, truth, fold_upper_ns, target_contract_sha256):
    if not (type(model_output) is dict and set(model_output) == {'open_logits', 'close_logits'}):
        raise ValueError('NATIVE_BAR_OBJECTIVE_OUTPUT')
    logits = model_output['close_logits']
    if not (isinstance(logits, torch.Tensor) and logits.dtype == torch.float32
            and logits.ndim == 3 and logits.shape[0] > 0 and logits.shape[1:] == (len(TIMEFRAMES), 3)):
        raise ValueError('NATIVE_BAR_OBJECTIVE_LOGITS')
    masks = _native_truth(truth, rows=len(logits), fold_upper_ns=fold_upper_ns,
                          target_contract_sha256=target_contract_sha256)
    sums, counts = {}, {}
    for name in ('open', 'close'):
        raw = model_output[name + '_logits']
        if not (isinstance(raw, torch.Tensor) and raw.shape == logits.shape
                and raw.dtype == torch.float32 and raw.device == logits.device
                and bool(torch.isfinite(raw).all())):
            raise ValueError('NATIVE_BAR_OBJECTIVE_LOGITS')
        labels = torch.as_tensor(truth[name + '_sign'].astype(np.int64) + 1, device=raw.device)
        mask = torch.as_tensor(masks[name], device=raw.device)
        cells = torch.nn.functional.cross_entropy(
            raw.reshape(-1, 3), labels.reshape(-1), reduction='none').reshape(labels.shape)
        sums[name] = torch.where(mask, cells, 0.0).sum(dim=0)
        counts[name] = mask.sum(dim=0)
    total_count = counts['open'].sum() + counts['close'].sum()
    total_sum = sums['open'].sum() + sums['close'].sum()
    if not bool(torch.isfinite(total_sum)):
        raise ValueError('NATIVE_BAR_OBJECTIVE_NONFINITE')
    return {'loss': total_sum / total_count.clamp(min=1),
            'loss_sums': sums, 'counts': counts, 'skip_update': not bool(total_count),
            'contract_sha256': native_bar_objective_contract()['contract_sha256']}


def native_bar_direction_score_sums(*, model_output, truth, fold_upper_ns,
                                    target_contract_sha256):
    """Add proper-score sums on the same predictive cells used by the loss."""
    if not (type(model_output) is dict and set(model_output) == {'open_logits', 'close_logits'}):
        raise ValueError('NATIVE_BAR_OBJECTIVE_OUTPUT')
    raw_by_task = {}
    for name in ('open', 'close'):
        raw = model_output[name + '_logits']
        raw = raw.detach().cpu().numpy() if isinstance(raw, torch.Tensor) else np.asarray(raw)
        if not (raw.dtype == np.float32 and raw.ndim == 3 and raw.shape[0] > 0
                and raw.shape[1:] == (len(TIMEFRAMES), 3) and np.isfinite(raw).all()):
            raise ValueError('NATIVE_BAR_OBJECTIVE_LOGITS')
        raw_by_task[name] = raw
    if raw_by_task['open'].shape != raw_by_task['close'].shape:
        raise ValueError('NATIVE_BAR_OBJECTIVE_LOGITS')
    masks = _native_truth(truth, rows=len(raw_by_task['close']), fold_upper_ns=fold_upper_ns,
                          target_contract_sha256=target_contract_sha256)
    report = {}
    for name, mask in masks.items():
        task = 'close' if name == 'close' else 'open'
        logits = raw_by_task[task].astype(np.float64)
        shifted = logits - logits.max(axis=2, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=2, keepdims=True))
        probs = np.exp(log_probs)
        labels = truth[task + '_sign'].astype(np.int64) + 1
        task_report = {}
        for j, tf in enumerate(TIMEFRAMES):
            selected = mask[:, j]
            target = labels[selected, j]
            predictions = logits[selected, j].argmax(axis=1)
            confusion = np.zeros((3, 3), dtype=np.int64)
            np.add.at(confusion, (target, predictions), 1)
            one_hot = np.eye(3)[target]
            maximum = logits[selected, j].max(axis=1, keepdims=True)
            tied = (logits[selected, j] == maximum).sum(axis=1) > 1
            task_report[tf] = {
                'count': int(selected.sum()),
                'unique_target_rows': int(np.unique(truth['target_row'][selected, j]).size),
                'log_loss_sum': float(-log_probs[selected, j, target].sum()),
                'brier_sum': float(np.square(probs[selected, j] - one_hot).sum()),
                'confusion_truth_by_prediction': confusion.tolist(),
                'max_logit_ties': int(tied.sum()),
            }
        report[name] = task_report
    return {'schema_version': 'gx1_native_m5_bar_direction_score_sums_v1',
            'contract_sha256': native_bar_objective_contract()['contract_sha256'],
            'class_order': [-1, 0, 1], 'open_known_role': 'observed_context_diagnostic',
            'per_task': report}
