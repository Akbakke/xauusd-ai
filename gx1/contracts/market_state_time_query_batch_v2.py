"""Fit-batch bridge for the factorized future market-state candidate.

The issue clock fixes each query before model forward. Targets are gathered only
following that forward. This owner has no optimizer or fit admission authority.
"""
from __future__ import annotations

from collections.abc import Mapping
import inspect

import numpy as np
import torch

from gx1.contracts.market_state_time_query_binding_v1 import (
    BoundTimeQueryTargets, sampled_query_k,
)
from gx1.contracts.market_state_time_query_loss_v2 import allowed_set_loss
from gx1.contracts.market_state_time_query_model_v2 import (
    FactorizedTimeQueryMarketStateModel,
)

FEATURE_KEYS = frozenset(inspect.signature(
    FactorizedTimeQueryMarketStateModel.encode_features).parameters) - {'self'}


def evaluate_fit_batch(
    model: FactorizedTimeQueryMarketStateModel,
    features: Mapping[str, torch.Tensor],
    bound: BoundTimeQueryTargets,
    parent_rows: np.ndarray,
    issue_ns: np.ndarray,
    *,
    fold_id: int,
    pass_index: int,
    seed: int,
) -> dict:
    """Return the factorized loss on one mature, eligible TRAIN batch."""
    if type(model) is not FactorizedTimeQueryMarketStateModel:
        raise TypeError('TIME_QUERY_V2_BATCH_MODEL_OWNER')
    if type(bound) is not BoundTimeQueryTargets:
        raise TypeError('TIME_QUERY_V2_BATCH_TARGET_OWNER')
    if not isinstance(features, Mapping) or set(features) != FEATURE_KEYS:
        raise ValueError('TIME_QUERY_V2_BATCH_FEATURE_KEYS')
    if not all(isinstance(value, torch.Tensor) for value in features.values()):
        raise ValueError('TIME_QUERY_V2_BATCH_FEATURE_TENSORS')
    if not (isinstance(parent_rows, np.ndarray) and parent_rows.dtype == np.int64
            and parent_rows.ndim == 1 and isinstance(issue_ns, np.ndarray)
            and issue_ns.dtype == np.int64 and issue_ns.ndim == 1
            and len(parent_rows) == len(issue_ns) > 0
            and features['seq_x'].shape[0] == len(issue_ns)
            and type(fold_id) is int and 1 <= fold_id <= 5):
        raise ValueError('TIME_QUERY_V2_BATCH_AXES')

    query_np = sampled_query_k(issue_ns, pass_index=pass_index, seed=seed)
    query = torch.from_numpy(query_np).to(device=model.query_native_k.device)
    output = model(**features, query_k=query)
    if not isinstance(output, dict) or set(output) != {'logits', 'activity_side_logits'}:
        raise RuntimeError('TIME_QUERY_V2_BATCH_OUTPUT_KEYS')
    raw_logits = output['activity_side_logits']
    class_logp = output['logits']
    if not (isinstance(raw_logits, torch.Tensor)
            and isinstance(class_logp, torch.Tensor)
            and raw_logits.shape == (len(issue_ns), 5, 2)
            and class_logp.shape == (len(issue_ns), 5, 3)
            and raw_logits.dtype == class_logp.dtype == torch.float32
            and raw_logits.device == class_logp.device == query.device
            and bool(torch.isfinite(raw_logits).all() & torch.isfinite(class_logp).all())):
        raise RuntimeError('TIME_QUERY_V2_BATCH_OUTPUT_PROBABILITY_PARITY')
    truth = bound.gather_fit_sampled(parent_rows, issue_ns,
                                     fold_id=fold_id, pass_index=pass_index,
                                     seed=seed)
    if not (np.array_equal(truth['query_k'], query_np)
            and np.array_equal(truth['issue_ns'], issue_ns)
            and np.array_equal(truth['parent_rows'], parent_rows)):
        raise RuntimeError('TIME_QUERY_V2_BATCH_SCHEDULE_OR_AXIS_MISMATCH')
    allowed = torch.from_numpy(truth['allowed']).to(device=raw_logits.device)
    eligible = torch.from_numpy(truth['eligible']).to(device=raw_logits.device)
    mature = torch.from_numpy(truth['mature']).to(device=raw_logits.device)
    report = allowed_set_loss(raw_logits, allowed, eligible, mature, query)
    if (report['skip_update'] is not (report['count'] == 0)
            or sum(report['per_tf_counts'].values()) != report['count']):
        raise RuntimeError('TIME_QUERY_V2_BATCH_SUPPORT_MISMATCH')
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
