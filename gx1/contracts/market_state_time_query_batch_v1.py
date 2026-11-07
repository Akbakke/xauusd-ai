"""Fit-batch bridge for future market-state queries; no optimizer authority.

The query schedule is fixed from issue identity before model forward. Target
cells are gathered only after that forward and never enter its keyword arguments.
A caller must skip backward and optimizer.step when ``skip_update`` is true.
"""
from __future__ import annotations

from collections.abc import Mapping
import inspect

import numpy as np
import torch

from gx1.contracts.market_state_time_query_binding_v1 import (
    BoundTimeQueryTargets, sampled_query_k,
)
from gx1.contracts.market_state_time_query_loss_v1 import allowed_set_loss
from gx1.contracts.market_state_time_query_model_v1 import TimeQueryMarketStateModel

FEATURE_KEYS = frozenset(inspect.signature(
    TimeQueryMarketStateModel.encode_features).parameters) - {'self'}


def evaluate_fit_batch(
    model: TimeQueryMarketStateModel,
    features: Mapping[str, torch.Tensor],
    bound: BoundTimeQueryTargets,
    parent_rows: np.ndarray,
    issue_ns: np.ndarray,
    *,
    fold_id: int,
    pass_index: int,
    seed: int,
) -> dict:
    """Return differentiable allowed-set loss on one exact mature fit batch.

    The caller owns canonical FP32/MHA execution context, ONLINE baseline and
    optimizer skip behavior. This function does not read files or start a fit.
    """
    if type(model) is not TimeQueryMarketStateModel:
        raise TypeError('TIME_QUERY_BATCH_MODEL_OWNER')
    if type(bound) is not BoundTimeQueryTargets:
        raise TypeError('TIME_QUERY_BATCH_TARGET_OWNER')
    if not isinstance(features, Mapping) or set(features) != FEATURE_KEYS:
        raise ValueError('TIME_QUERY_BATCH_FEATURE_KEYS')
    if not all(isinstance(value, torch.Tensor) for value in features.values()):
        raise ValueError('TIME_QUERY_BATCH_FEATURE_TENSORS')
    if not (isinstance(parent_rows, np.ndarray) and parent_rows.dtype == np.int64
            and parent_rows.ndim == 1 and isinstance(issue_ns, np.ndarray)
            and issue_ns.dtype == np.int64 and issue_ns.ndim == 1
            and len(parent_rows) == len(issue_ns) > 0
            and features['seq_x'].shape[0] == len(issue_ns)
            and type(fold_id) is int and 1 <= fold_id <= 5):
        raise ValueError('TIME_QUERY_BATCH_AXES')

    # No future target cell is gathered until the model has consumed only features
    # and the issue-derived query coordinates.
    query_np = sampled_query_k(issue_ns, pass_index=pass_index, seed=seed)
    query = torch.from_numpy(query_np).to(device=model.query_native_k.device)
    output = model(**features, query_k=query)
    if not isinstance(output, dict) or set(output) != {'logits'}:
        raise RuntimeError('TIME_QUERY_BATCH_OUTPUT_KEYS')
    logits = output['logits']
    truth = bound.gather_fit_sampled(parent_rows, issue_ns,
                                     fold_id=fold_id, pass_index=pass_index,
                                     seed=seed)
    if not (np.array_equal(truth['query_k'], query_np)
            and np.array_equal(truth['issue_ns'], issue_ns)
            and np.array_equal(truth['parent_rows'], parent_rows)):
        raise RuntimeError('TIME_QUERY_BATCH_SCHEDULE_OR_AXIS_MISMATCH')
    allowed = torch.from_numpy(truth['allowed']).to(device=logits.device)
    eligible = torch.from_numpy(truth['eligible']).to(device=logits.device)
    mature = torch.from_numpy(truth['mature']).to(device=logits.device)
    report = allowed_set_loss(logits, allowed, eligible, mature, query)
    if (report['skip_update'] is not (report['count'] == 0)
            or sum(report['per_tf_counts'].values()) != report['count']):
        raise RuntimeError('TIME_QUERY_BATCH_SUPPORT_MISMATCH')
    return {
        'loss': report['loss'], 'count': report['count'],
        'per_tf_counts': report['per_tf_counts'],
        'skip_update': report['skip_update'],
        'query_k': query_np.copy(), 'validity_ns': truth['validity_ns'],
        'allowed': truth['allowed'], 'eligible': truth['eligible'],
        'mature': truth['mature'],
        'loss_contract_sha256': report['contract_sha256'],
        'model_contract_sha256': model.research_objective['contract_sha256'],
    }
