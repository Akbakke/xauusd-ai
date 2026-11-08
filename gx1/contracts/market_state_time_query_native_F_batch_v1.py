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


def evaluate_fit_batch(model: NativeFutureDualModel, features: Mapping,
                       state_bound: BoundTimeQueryTargets,
                       future_bound: BoundFutureDisplacement,
                       parent_rows: np.ndarray, issue_ns: np.ndarray, *,
                       fold_id: int, pass_index: int, seed: int,
                       future_weight: float) -> dict:
    if not (type(model) is NativeFutureDualModel
            and type(state_bound) is BoundTimeQueryTargets
            and type(future_bound) is BoundFutureDisplacement
            and isinstance(features, Mapping) and set(features) == FEATURE_KEYS
            and all(isinstance(value, torch.Tensor) for value in features.values())
            and isinstance(parent_rows, np.ndarray) and parent_rows.dtype == np.int64
            and isinstance(issue_ns, np.ndarray) and issue_ns.dtype == np.int64
            and parent_rows.ndim == issue_ns.ndim == 1
            and len(parent_rows) == len(issue_ns) > 0
            and features['seq_x'].shape[0] == len(issue_ns)
            and type(fold_id) is int and 1 <= fold_id <= 5):
        raise ValueError('NATIVE_F_BATCH_INPUT')
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
        state_query_k=query, future_weight=future_weight)
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
