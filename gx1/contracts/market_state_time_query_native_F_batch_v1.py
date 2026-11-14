"""One shared-encoder S sampled query and F native query TRAIN batch adapter.

Feature tensors alone enter forward. Both future target owners are gathered
only after the model output exists. No optimizer, checkpoint or run authority.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch

from gx1.contracts.market_state_future_displacement_candidate_v1 import BoundFutureDisplacement
from gx1.contracts.market_state_time_query_batch_v2 import FEATURE_KEYS
from gx1.contracts.market_state_time_query_binding_v1 import (
    BoundTimeQueryTargets, sampled_query_k,
)
from gx1.contracts.market_state_time_query_native_F_model_v1 import NativeFutureDualModel
from gx1.contracts.market_state_time_query_model_v2 import factorized_log_probabilities
from gx1.contracts.market_state_time_query_native_F_objective_v1 import combined_loss



def _ta_only(model) -> bool:
    from gx1.features.entry_specialist_feature_groups_v1 import FULL_TA_ENTRY_EXECUTION_PROFILE_NAME
    backbone = getattr(model, 'backbone', None)
    profile = getattr(backbone, 'entry_execution_profile', None)
    return isinstance(profile, Mapping) and profile.get('name') == FULL_TA_ENTRY_EXECUTION_PROFILE_NAME


def _feature_keys(model) -> frozenset:
    """Select the exact declared input signature; never mask absent GC inputs."""
    from gx1.features.entry_specialist_feature_groups_v1 import FULL_TA_ENTRY_FEATURE_KEYS
    return FULL_TA_ENTRY_FEATURE_KEYS if _ta_only(model) else FEATURE_KEYS


def evaluate_fit_batch(model: NativeFutureDualModel, features: Mapping,
                       state_bound: BoundTimeQueryTargets,
                       future_bound: BoundFutureDisplacement,
                       parent_rows: np.ndarray, issue_ns: np.ndarray, *,
                       fold_id: int, pass_index: int, seed: int,
                       future_weight: float, m5_equal_observation: bool = False,
                       fit_batch_start: int | None = None,
                       fit_batch_positions: np.ndarray | None = None,
                       m5_direction_only: bool = False) -> dict:
    if type(m5_direction_only) is not bool or (m5_direction_only and not m5_equal_observation):
        raise ValueError('NATIVE_F_M5_DIRECTION_ONLY_MODE')
    if type(m5_equal_observation) is not bool or (
            not m5_equal_observation
            and (fit_batch_start is not None or fit_batch_positions is not None)) or (
            fit_batch_start is not None and fit_batch_positions is not None):
        raise ValueError('NATIVE_F_M5_EQUAL_OBSERVATION_MODE')
    if not (type(model) is NativeFutureDualModel
            and type(state_bound) is BoundTimeQueryTargets
            and type(future_bound) is BoundFutureDisplacement
            and isinstance(features, Mapping) and set(features) == _feature_keys(model)
            and all(isinstance(value, torch.Tensor) for value in features.values())
            and isinstance(parent_rows, np.ndarray) and parent_rows.dtype == np.int64
            and isinstance(issue_ns, np.ndarray) and issue_ns.dtype == np.int64
            and parent_rows.ndim == issue_ns.ndim == 1
            and len(parent_rows) == len(issue_ns) > 0
            and features['seq_x'].shape[0] == len(issue_ns)
            and type(fold_id) is int and 1 <= fold_id <= 5):
        raise ValueError('NATIVE_F_BATCH_INPUT')
    if m5_equal_observation:
        from gx1.contracts.market_state_time_query_native_F_objective_v1 import m5_equal_observation_profile
        profile = m5_equal_observation_profile(fold_id=fold_id)
        if fit_batch_positions is None:
            if not (_ta_only(model)
                    and getattr(model.state_model, '_dedicated_m5_side', False) is True
                    and fold_id == profile['fold_id'] and pass_index == profile['pass_index']
                    and seed == profile['seed'] and type(fit_batch_start) is int
                    and 0 <= fit_batch_start < profile['fit_rows']
                    and fit_batch_start % profile['batch_size'] == 0
                    and len(parent_rows) == min(profile['batch_size'],
                                                profile['fit_rows'] - fit_batch_start)):
                raise ValueError('NATIVE_F_M5_EQUAL_OBSERVATION_BATCH')
            stop = fit_batch_start + len(parent_rows)
            # Alignment is a position in the bound axis, never parent-ID modulo B.
            for bound in (state_bound, future_bound):
                if not (np.array_equal(parent_rows, bound.parent_rows[fit_batch_start:stop])
                        and np.array_equal(issue_ns, bound.decision_ns[fit_batch_start:stop])):
                    raise ValueError('NATIVE_F_M5_EQUAL_OBSERVATION_AXIS')
        else:
            if not (_ta_only(model)
                    and getattr(model.state_model, '_dedicated_m5_side', False) is True
                    and fold_id == profile['fold_id'] and pass_index == profile['pass_index']
                    and seed == profile['seed']
                    and isinstance(fit_batch_positions, np.ndarray)
                    and fit_batch_positions.dtype == np.int64
                    and fit_batch_positions.ndim == 1
                    and len(fit_batch_positions) == len(parent_rows) == profile['batch_size']
                    and np.all((fit_batch_positions >= 0)
                               & (fit_batch_positions < profile['fit_rows']))
                    and np.unique(fit_batch_positions).size == profile['batch_size']):
                raise ValueError('NATIVE_F_M5_EQUAL_OBSERVATION_BATCH')
            # Positions index each original bound axis; no reordering or synthetic start.
            for bound in (state_bound, future_bound):
                if not (np.array_equal(parent_rows, bound.parent_rows[fit_batch_positions])
                        and np.array_equal(issue_ns, bound.decision_ns[fit_batch_positions])):
                    raise ValueError('NATIVE_F_M5_EQUAL_OBSERVATION_AXIS')
    query_np = sampled_query_k(issue_ns, pass_index=pass_index, seed=seed)
    device = model.state_model.query_native_k.device
    query = torch.from_numpy(query_np).to(device=device)
    output = model(**features, query_k=query)
    if not (isinstance(output, dict)
            and set(output) == {'state_activity_side_logits', 'state_log_probs',
                                'future_activity_side_logits', 'future_log_probs'}
            and all(isinstance(value, torch.Tensor) and value.device == device
                    and value.dtype == torch.float32 and bool(torch.isfinite(value).all())
                    for value in output.values())
            and output['state_activity_side_logits'].shape
                == output['future_activity_side_logits'].shape == (len(issue_ns), 5, 2)
            and output['state_log_probs'].shape
                == output['future_log_probs'].shape == (len(issue_ns), 5, 3)
            and torch.equal(output['state_log_probs'], factorized_log_probabilities(
                output['state_activity_side_logits']))
            and torch.equal(output['future_log_probs'], factorized_log_probabilities(
                output['future_activity_side_logits']))):
        raise RuntimeError('NATIVE_F_BATCH_MODEL_OUTPUT')
    state_truth = state_bound.gather_fit_sampled(
        parent_rows, issue_ns, fold_id=fold_id,
        pass_index=pass_index, seed=seed)
    future_truth = future_bound.gather(
        parent_rows, issue_ns, fold_id=fold_id, partition='fit')
    if not (np.array_equal(state_truth['query_k'], query_np)
            and np.array_equal(state_truth['parent_rows'], parent_rows)
            and np.array_equal(future_truth['parent_rows'], parent_rows)
            and np.array_equal(state_truth['issue_ns'], issue_ns)
            and np.array_equal(future_truth['issue_ns'], issue_ns)):
        raise RuntimeError('NATIVE_F_BATCH_TARGET_CLOCK')
    report = combined_loss(
        state_raw=output['state_activity_side_logits'],
        future_raw=output['future_activity_side_logits'],
        state_truth=state_truth, future_truth=future_truth,
        state_query_k=query, future_weight=future_weight, ta_only=_ta_only(model),
        dedicated_m5_side=getattr(model.state_model, '_dedicated_m5_side', False),
        dedicated_non_m5_side=getattr(model.state_model, '_dedicated_non_m5_side', False),
        dedicated_tf_activity=getattr(model.state_model, '_dedicated_tf_activity', False),
        m5_equal_observation=m5_equal_observation,
        m5_direction_only=m5_direction_only, fold_id=fold_id)
    return {**output, 'loss': report['loss'],
            'state_count': report['state']['count'],
            'future_count': report['future']['count'],
            'state_per_tf_counts': report['state']['per_tf_counts'],
            'future_per_tf_counts': report['future']['per_tf_counts'],
            'skip_update': report['skip_update'],
            'state_query_k': query_np.copy(),
            'future_native_k': model.state_model.query_native_k.detach().cpu().numpy().copy(),
            'state_allowed': state_truth['allowed'],
            'future_allowed': future_truth['allowed'],
            'state_mature': state_truth['mature'],
            'future_mature': future_truth['mature'],
            'loss_contract_sha256': report['contract_sha256'],
            'model_contract_sha256': model.research_objective['contract_sha256']}


def _m5_readout_hooks(model, rows):
    """Observe the existing M5 Linear inputs; never alter a forward argument."""
    if not (_ta_only(model) and getattr(model.state_model, '_dedicated_m5_side', False)):
        raise ValueError('NATIVE_F_M5_CAPTURE_REQUIRES_DEDICATED_TA')
    heads = {'state_representation': model.state_model.m5_side_head,
             'future_representation': model.future_m5_side_head}
    if not all(type(head) is torch.nn.Linear and head.in_features == 128
               and head.out_features == 1 for head in heads.values()):
        raise ValueError('NATIVE_F_M5_CAPTURE_HEAD_SHAPE')
    captured, handles = {}, []
    def observe(name):
        def hook(_module, inputs):
            if not (name not in captured and len(inputs) == 1
                    and isinstance(inputs[0], torch.Tensor)
                    and inputs[0].shape == (
                        (rows, 1, 128) if name == 'state_representation' else (rows, 128))
                    and inputs[0].dtype == torch.float32
                    and bool(torch.isfinite(inputs[0]).all())):
                raise RuntimeError('NATIVE_F_M5_CAPTURE_INPUT')
            # S decodes a singleton query grid; F decodes the native endpoints directly.
            value = inputs[0][:, 0, :] if name == 'state_representation' else inputs[0]
            captured[name] = value.detach().cpu().numpy().copy()
        return hook
    try:
        for name, head in heads.items():
            handles.append(head.register_forward_pre_hook(observe(name)))
    except BaseException:
        for handle in handles:
            handle.remove()
        raise
    return captured, handles


def evaluate_endpoint_batch(model: NativeFutureDualModel, features: Mapping,
                            state_endpoint_bound, future_bound: BoundFutureDisplacement,
                            parent_rows: np.ndarray, issue_ns: np.ndarray, *,
                            fold_id: int, partition: str, future_weight: float,
                            frozen_priors: dict | None = None,
                            capture_m5_readout: bool = False) -> dict:
    """One eval-only S/F forward, optionally retaining its existing M5 inputs."""
    from gx1.contracts.market_state_five_endpoint_target_binding_v1 import BoundFiveEndpointTargets
    from gx1.contracts.market_state_time_query_native_F_objective_v1 import endpoint_measurement_sums
    from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K
    if not (type(capture_m5_readout) is bool
            and type(model) is NativeFutureDualModel
            and type(state_endpoint_bound) is BoundFiveEndpointTargets
            and type(future_bound) is BoundFutureDisplacement
            and isinstance(features, Mapping) and set(features) == _feature_keys(model)
            and all(isinstance(value, torch.Tensor) for value in features.values())
            and isinstance(parent_rows, np.ndarray) and parent_rows.dtype == np.int64
            and isinstance(issue_ns, np.ndarray) and issue_ns.dtype == np.int64
            and parent_rows.ndim == issue_ns.ndim == 1
            and len(parent_rows) == len(issue_ns) > 0
            and features['seq_x'].shape[0] == len(issue_ns)
            and type(fold_id) is int and 1 <= fold_id <= 5
            and partition in ('fit', 'score')):
        raise ValueError('NATIVE_F_ENDPOINT_BATCH_INPUT')
    if any(module.training for module in model.modules()):
        raise ValueError('NATIVE_F_ENDPOINT_REQUIRES_EVAL')
    device = model.state_model.query_native_k.device
    query = torch.tensor(NATIVE_K, dtype=torch.int64, device=device)[None, :].expand(
        len(issue_ns), -1)
    captured, handles = (_m5_readout_hooks(model, len(issue_ns))
                         if capture_m5_readout else ({}, []))
    with torch.inference_mode():
        try:
            output = model(**features, query_k=query)
        finally:
            for handle in handles:
                handle.remove()
        if capture_m5_readout and set(captured) != {
                'state_representation', 'future_representation'}:
            raise RuntimeError('NATIVE_F_M5_CAPTURE_MISSING_HEAD')
        if not (set(output) == {'state_activity_side_logits', 'state_log_probs',
                               'future_activity_side_logits', 'future_log_probs'}
                and all(isinstance(value, torch.Tensor) and value.device == device
                        and value.dtype == torch.float32 and bool(torch.isfinite(value).all())
                        for value in output.values())
                and output['state_activity_side_logits'].shape
                    == output['future_activity_side_logits'].shape == (len(issue_ns), 5, 2)
                and torch.equal(output['state_log_probs'], factorized_log_probabilities(
                    output['state_activity_side_logits']))
                and torch.equal(output['future_log_probs'], factorized_log_probabilities(
                    output['future_activity_side_logits']))):
            raise RuntimeError('NATIVE_F_ENDPOINT_MODEL_OUTPUT')
        state_raw = output['state_activity_side_logits'].cpu().numpy().copy()
        future_raw = output['future_activity_side_logits'].cpu().numpy().copy()
    # This endpoint owner has certified parity with the time-query target tape.
    # It permits fixed native TRAIN evaluation without weakening sampled fit.
    s = state_endpoint_bound.gather(parent_rows, issue_ns, fold_id=fold_id,
                                     partition=partition)
    f = future_bound.gather(parent_rows, issue_ns, fold_id=fold_id, partition=partition)
    if partition == 'fit' and not (bool(s['mature'].all()) and bool(f['mature'].all())):
        raise ValueError('NATIVE_F_ENDPOINT_IMMATURE_FIT')
    measurement = endpoint_measurement_sums(
        state_raw=state_raw, future_raw=future_raw, state_truth=s, future_truth=f,
        parent_rows=parent_rows, issue_ns=issue_ns, fold_id=fold_id,
        partition=partition, future_weight=future_weight,
        model_contract_sha256=model.research_objective['contract_sha256'],
        feature_view_sha256=model.state_model.feature_view_identity['contract_sha256'],
        frozen_priors=frozen_priors, ta_only=_ta_only(model),
        dedicated_m5_side=getattr(model.state_model, '_dedicated_m5_side', False))
    report = {'measurement': measurement,
              'state_activity_side_logits': state_raw,
              'future_activity_side_logits': future_raw}
    if capture_m5_readout:
        captured.update(parents=parent_rows.copy(), issues=issue_ns.copy(),
                        query_native=query.cpu().numpy().copy(),
                        state_activity_side_logits=state_raw.copy(),
                        future_activity_side_logits=future_raw.copy())
        for task, truth in (('state', s), ('future', f)):
            for key in ('allowed', 'eligible', 'mature'):
                captured[task + '_' + key] = truth[key].copy()
        report['m5_readout'] = captured
    return report
