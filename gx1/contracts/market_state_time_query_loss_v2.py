"""Factorized activity and conditional-side loss for time-query market state.

This research loss uses the two primitive head logits directly. In particular,
an ACTIVE_SIDE_UNKNOWN target updates activity and has exactly zero direct
gradient to the conditional-side logit. The three-class probabilities used by
the proper scorer come from the same factorization in the v2 model.
"""
from __future__ import annotations

import hashlib
import json

import torch
import torch.nn.functional as F

from gx1.contracts import market_state_five_endpoint_contract_v1 as five
from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K, TIMEFRAMES

K = tuple(int(value) for value in NATIVE_K)


def factorized_time_query_loss_contract() -> dict:
    value = {
        "schema_version": "gx1_market_state_time_query_loss_v2",
        "role": "research_only_no_entry_exit_or_trade_authority",
        "target_contract_sha256": five.five_endpoint_contract()["contract_sha256"],
        "timeframes": list(TIMEFRAMES),
        "native_k": list(K),
        "primitive_logit_order": ["ACTIVE", "UP_GIVEN_ACTIVE"],
        "allowed_bits": {"DOWN": 1, "FLAT": 2, "UP": 4,
                         "ACTIVE_SIDE_UNKNOWN": 5, "ALL_UNKNOWN": 7},
        "loss": "mean_per_TF_activity_BCE_plus_known_conditional_side_BCE_divided_by_five",
        "side_gradient": "zero_direct_side_logit_gradient_for_bits_2_5_7_or_immature",
        "empty_support": "zero_loss_anchored_to_primitive_logits_skip_update",
        "proper_score": "factorized_time_query_scorer_v2_from_primitive_logits",
        "future_labels_as_model_forward_inputs": False,
    }
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode()
    return {**value, "contract_sha256": hashlib.sha256(raw).hexdigest()}


def _torch_inputs(raw_logits: torch.Tensor, allowed: torch.Tensor,
                  eligible: torch.Tensor, mature: torch.Tensor,
                  query_k: torch.Tensor) -> None:
    if not (isinstance(raw_logits, torch.Tensor)
            and raw_logits.dtype == torch.float32
            and raw_logits.ndim == 3 and raw_logits.shape[0] > 0
            and tuple(raw_logits.shape[1:]) == (len(TIMEFRAMES), 2)
            and bool(torch.isfinite(raw_logits).all())):
        raise ValueError("TIME_QUERY_FACTORIZED_LOGITS")
    shape = raw_logits.shape[:2]
    if not (isinstance(allowed, torch.Tensor)
            and allowed.dtype == torch.uint8 and allowed.shape == shape
            and allowed.device == raw_logits.device
            and bool(torch.isin(allowed, torch.tensor(
                five.VALID_BITS, dtype=torch.uint8, device=raw_logits.device)).all())):
        raise ValueError("TIME_QUERY_FACTORIZED_ALLOWED")
    if not all(isinstance(mask, torch.Tensor) and mask.dtype == torch.bool
               and mask.shape == shape and mask.device == raw_logits.device
               for mask in (eligible, mature)):
        raise ValueError("TIME_QUERY_FACTORIZED_ELIGIBILITY")
    if not (isinstance(query_k, torch.Tensor) and query_k.dtype == torch.int64
            and query_k.shape == shape and query_k.device == raw_logits.device
            and bool(((query_k >= 1) & (query_k <= torch.tensor(
                K, dtype=torch.int64, device=raw_logits.device)[None, :])).all())):
        raise ValueError("TIME_QUERY_K")


def allowed_set_loss(raw_logits: torch.Tensor, allowed: torch.Tensor,
                     eligible: torch.Tensor, mature: torch.Tensor,
                     query_k: torch.Tensor) -> dict:
    """Equal TF weighting with activity and certified-side masks kept separate."""
    _torch_inputs(raw_logits, allowed, eligible, mature, query_k)
    informative = eligible & mature & (allowed != 7)
    side_known = informative & ((allowed == 1) | (allowed == 4))
    per_tf = {tf: int(informative[:, j].sum().item())
              for j, tf in enumerate(TIMEFRAMES)}
    count = sum(per_tf.values())
    zero = raw_logits.mul(0).sum()
    if not count:
        terms = [zero for _ in TIMEFRAMES]
    else:
        active_truth = (allowed != 2).to(dtype=raw_logits.dtype)
        up_truth = (allowed == 4).to(dtype=raw_logits.dtype)
        activity_nll = F.binary_cross_entropy_with_logits(
            raw_logits[..., 0], active_truth, reduction="none")
        side_nll = F.binary_cross_entropy_with_logits(
            raw_logits[..., 1], up_truth, reduction="none")
        per_cell = (torch.where(informative, activity_nll, torch.zeros_like(activity_nll))
                    + torch.where(side_known, side_nll, torch.zeros_like(side_nll)))
        terms = [(per_cell[:, j].sum() / per_tf[tf] if per_tf[tf] else zero)
                 for j, tf in enumerate(TIMEFRAMES)]
    loss = sum(terms, zero) / len(TIMEFRAMES)
    if not bool(torch.isfinite(loss)):
        raise ValueError("TIME_QUERY_FACTORIZED_NONFINITE_LOSS")
    return {
        "loss": loss,
        "count": count,
        "per_tf_counts": per_tf,
        "per_tf_mean_loss": dict(zip(TIMEFRAMES, terms)),
        "informative_mask": informative,
        "side_known_mask": side_known,
        "skip_update": count == 0,
        "contract_sha256": factorized_time_query_loss_contract()["contract_sha256"],
    }
