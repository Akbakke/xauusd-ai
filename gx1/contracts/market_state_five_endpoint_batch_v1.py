"""Join one post-forward batch to corrected TRAIN targets; no fit authority.

Call only after feature-only model forward. The returned skip_update flag must
prevent backward and optimizer.step when the mature informative count is zero.
"""
from __future__ import annotations

import torch

from gx1.contracts.market_state_five_endpoint_contract_v1 import (
    TIMEFRAMES, allowed_set_loss, proper_score_sums,
)
from gx1.contracts.market_state_five_endpoint_target_binding_v1 import BoundFiveEndpointTargets


def evaluate_after_forward(logits: torch.Tensor, bound: BoundFiveEndpointTargets,
                           parent_rows, decision_ns, *, fold_id: int,
                           partition: str) -> dict:
    """Return differentiable loss and detached proper-score sums on exact rows."""
    if type(bound) is not BoundFiveEndpointTargets:
        raise TypeError("FIVE_ENDPOINT_BOUND_TARGET_OWNER")
    truth = bound.gather(parent_rows, decision_ns, fold_id=fold_id,
                         partition=partition)
    allowed = torch.from_numpy(truth["allowed"]).to(device=logits.device)
    eligible = torch.from_numpy(truth["eligible"]).to(device=logits.device)
    mature = torch.from_numpy(truth["mature"]).to(device=logits.device)
    loss = allowed_set_loss(logits, allowed, eligible, mature)
    probabilities = torch.softmax(logits.detach().to(torch.float64), dim=-1).cpu().numpy()
    scores = proper_score_sums(probabilities, truth["allowed"],
                               truth["eligible"], truth["mature"])
    if (scores["contract_sha256"] != loss["contract_sha256"]
            or any(scores["per_tf"][tf]["activity"]["count"] != loss["per_tf_counts"][tf]
                   for tf in TIMEFRAMES)):
        raise RuntimeError("FIVE_ENDPOINT_BATCH_SCORE_SUPPORT_MISMATCH")
    return {"loss": loss["loss"], "count": loss["count"],
            "per_tf_counts": loss["per_tf_counts"],
            "skip_update": loss["skip_update"],
            "score_sums": scores["per_tf"],
            "contract_sha256": loss["contract_sha256"]}
