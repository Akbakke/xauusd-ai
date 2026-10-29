"""Research-only 5-minute-priority candidate for the 60-head direction loss.

No data access, model execution or training authority. The candidate retains
all 60 heads and their existing observed/nonzero/maturity mask. It changes only
the BCE cell weights and normalizes by the sum of eligible weights. Five is an
experimental prediction horizon, not the M5 decision cadence or a hold rule.
"""
from __future__ import annotations
import hashlib
import json
from typing import Any
import torch
from torch.nn import functional as F
from gx1.contracts.entry_native_direction_dense_v1 import dense_direction_binary_loss, dense_direction_research_contract

PRIMARY_MINUTES = 5
PRIMARY_WEIGHT = 10


_PRIORITY_OBJECTIVE = {k:v for k,v in dense_direction_research_contract().items() if k != 'contract_sha256'}
_PRIORITY_OBJECTIVE.update(schema_version='gx1_entry_native_direction_dense_five_priority_v1',
                           loss='BCE_with_logits_weight_10_on_5min_other_59_weight_1_divided_by_eligible_weight_sum',
                           primary_horizon_minutes=PRIMARY_MINUTES, primary_horizon_weight=PRIMARY_WEIGHT)
_PRIORITY_OBJECTIVE_JSON = json.dumps(_PRIORITY_OBJECTIVE, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
_PRIORITY_OBJECTIVE_SHA = hashlib.sha256(_PRIORITY_OBJECTIVE_JSON).hexdigest()


def five_priority_direction_contract() -> dict[str, Any]:
    """Return a copy of the distinct immutable objective identity."""
    return {**json.loads(_PRIORITY_OBJECTIVE_JSON), 'contract_sha256':_PRIORITY_OBJECTIVE_SHA}


def five_priority_direction_binary_loss(logit_up: torch.Tensor, *, sign: torch.Tensor,
        observed: torch.Tensor, decision_ns: torch.Tensor, terminal_time_ns: torch.Tensor,
        cutoff_ns: int) -> dict[str, Any]:
    """One weighted BCE on exactly the ordinary dense loss's eligible cells."""
    result = dense_direction_binary_loss(logit_up, sign=sign, observed=observed,
        decision_ns=decision_ns, terminal_time_ns=terminal_time_ns, cutoff_ns=cutoff_ns)
    result['research_objective_sha256'] = five_priority_direction_contract()['contract_sha256']
    n = result['counts']['eligible_cells']
    if not n:
        return result
    mask = result['eligible_mask']
    n5 = result['per_horizon_counts'][PRIMARY_MINUTES-1]['eligible_cells']
    logits = logit_up[mask]
    labels = (sign[mask] == 1).to(dtype=logits.dtype)
    cell_weights = torch.ones_like(logit_up)
    cell_weights[:, PRIMARY_MINUTES-1] = float(PRIMARY_WEIGHT)
    weights = cell_weights[mask]
    denominator = n + (PRIMARY_WEIGHT-1)*n5
    loss = F.binary_cross_entropy_with_logits(logits, labels, weight=weights, reduction='sum') / denominator
    if not bool(torch.isfinite(loss)) or not bool(loss >= 0):
        raise RuntimeError('NATIVE_DIRECTION_FIVE_PRIORITY_FINITE_LOSS')
    result['loss'] = loss
    return result
