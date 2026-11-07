"""Pure five-endpoint market-state loss and proper scores; no fit authority.

Inputs are already joined to exact M5 issue and five future validity clocks by
an external, hash-bound target owner. Future masks never enter model forward.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import torch
import torch.nn.functional as F

TIMEFRAMES = ("M5", "M15", "H1", "H4", "D1")
LEAD_MINUTES = (5, 15, 60, 240, 1440)
CLASS_NAMES = ("DOWN", "FLAT", "UP")
VALID_BITS = (1, 2, 4, 5, 7)


def five_endpoint_contract() -> dict:
    value = {
        "schema_version": "gx1_market_state_five_endpoint_contract_v1",
        "role": "research_only_no_entry_exit_or_trade_authority",
        "timeframes": list(TIMEFRAMES),
        "future_validity_minutes_after_closed_M5_issue": list(LEAD_MINUTES),
        "class_order": list(CLASS_NAMES),
        "allowed_bits": {"DOWN": 1, "FLAT": 2, "UP": 4,
                         "ACTIVE_SIDE_UNKNOWN": 5, "ALL_UNKNOWN": 7},
        "loss": "mean_negative_log_allowed_probability_over_eligible_mature_informative_cells",
        "activity_score": "binary_Brier_on_bits_1_2_4_5",
        "side_score": "conditional_UP_binary_Brier_and_log_on_bits_1_4",
        "three_class_score": "multiclass_Brier_and_log_on_bits_1_2_4",
        "empty_support": "zero_loss_anchored_to_logits_skip_update",
        "future_labels_as_forward_inputs": False,
    }
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {**value, "contract_sha256": hashlib.sha256(raw).hexdigest()}


def _torch_inputs(logits: torch.Tensor, allowed: torch.Tensor,
                  eligible: torch.Tensor, mature: torch.Tensor) -> None:
    if not (isinstance(logits, torch.Tensor) and logits.dtype == torch.float32
            and logits.ndim == 3 and logits.shape[0] > 0
            and tuple(logits.shape[1:]) == (5, 3)
            and bool(torch.isfinite(logits).all())):
        raise ValueError("FIVE_ENDPOINT_LOGITS")
    shape = logits.shape[:2]
    if not (isinstance(allowed, torch.Tensor) and allowed.dtype == torch.uint8
            and allowed.shape == shape and allowed.device == logits.device
            and bool(torch.isin(allowed, torch.tensor(VALID_BITS, dtype=torch.uint8,
                                                     device=logits.device)).all())):
        raise ValueError("FIVE_ENDPOINT_ALLOWED")
    if not all(isinstance(mask, torch.Tensor) and mask.dtype == torch.bool
               and mask.shape == shape and mask.device == logits.device
               for mask in (eligible, mature)):
        raise ValueError("FIVE_ENDPOINT_ELIGIBILITY")


def allowed_set_loss(logits: torch.Tensor, allowed: torch.Tensor,
                     eligible: torch.Tensor, mature: torch.Tensor) -> dict:
    """Return a differentiable mean, with mask 7 excluded from its denominator.

    Mask 5 constrains only total DOWN+UP mass. It imposes no preference between
    DOWN and UP when their combined mass is fixed.
    """
    _torch_inputs(logits, allowed, eligible, mature)
    informative = eligible & mature & (allowed != 7)
    per_tf = {tf: int(informative[:, j].sum().item())
              for j, tf in enumerate(TIMEFRAMES)}
    count = sum(per_tf.values())
    if not count:
        loss = logits.mul(0).sum()
    else:
        table = torch.tensor((
            (False, False, False), (True, False, False),
            (False, True, False), (False, False, False),
            (False, False, True), (True, False, True),
            (False, False, False), (True, True, True),
        ), dtype=torch.bool, device=logits.device)
        logp = F.log_softmax(logits, dim=-1)
        nll = -torch.logsumexp(logp.masked_fill(~table[allowed.long()], -torch.inf), dim=-1)
        loss = torch.where(informative, nll, torch.zeros_like(nll)).sum() / count
    if not bool(torch.isfinite(loss)):
        raise ValueError("FIVE_ENDPOINT_NONFINITE_LOSS")
    return {"loss": loss, "count": count, "per_tf_counts": per_tf,
            "informative_mask": informative, "skip_update": count == 0,
            "contract_sha256": five_endpoint_contract()["contract_sha256"]}


def _numpy_inputs(probabilities: np.ndarray, allowed: np.ndarray,
                  eligible: np.ndarray, mature: np.ndarray) -> np.ndarray:
    if not (isinstance(probabilities, np.ndarray)
            and probabilities.dtype in (np.dtype("float32"), np.dtype("float64"))
            and probabilities.ndim == 3 and probabilities.shape[0] > 0
            and probabilities.shape[1:] == (5, 3)
            and np.isfinite(probabilities).all()
            and np.all((probabilities >= 0) & (probabilities <= 1))
            and np.allclose(probabilities.astype(np.float64).sum(axis=-1), 1.0,
                            rtol=0, atol=1e-6)):
        raise ValueError("FIVE_ENDPOINT_PROBABILITIES")
    shape = probabilities.shape[:2]
    if not (isinstance(allowed, np.ndarray) and allowed.dtype == np.uint8
            and allowed.shape == shape and np.isin(allowed, VALID_BITS).all()):
        raise ValueError("FIVE_ENDPOINT_ALLOWED")
    if not all(isinstance(mask, np.ndarray) and mask.dtype == np.bool_
               and mask.shape == shape for mask in (eligible, mature)):
        raise ValueError("FIVE_ENDPOINT_ELIGIBILITY")
    return eligible & mature


def _log_sum(truth_probability: np.ndarray) -> tuple[float, int]:
    """Zero assigned probability has its exact infinite proper log loss."""
    zeros = int(np.count_nonzero(truth_probability == 0))
    with np.errstate(divide="ignore"):
        return float((-np.log(truth_probability)).sum(dtype=np.float64)), zeros


def proper_score_sums(probabilities: np.ndarray, allowed: np.ndarray,
                      eligible: np.ndarray, mature: np.ndarray) -> dict:
    """Separate per-TF sums on the same mature and eligible rows as the loss."""
    common = _numpy_inputs(probabilities, allowed, eligible, mature)
    p = probabilities.astype(np.float64, copy=False)
    report = {}
    for j, tf in enumerate(TIMEFRAMES):
        bits = allowed[:, j]
        ready = common[:, j]
        activity_mask = ready & (bits != 7)
        side_mask = ready & np.isin(bits, (1, 4))
        singleton_mask = ready & np.isin(bits, (1, 2, 4))

        active_probability = p[activity_mask, j, 0] + p[activity_mask, j, 2]
        active_truth = (bits[activity_mask] != 2).astype(np.float64)
        activity = {"count": int(activity_mask.sum()),
                    "brier_sum": float(np.square(active_probability - active_truth).sum(dtype=np.float64))}

        side_probabilities = p[side_mask, j, :]
        denominator = side_probabilities[:, 0] + side_probabilities[:, 2]
        if not np.all(denominator > 0):
            raise ValueError("FIVE_ENDPOINT_UNDEFINED_CONDITIONAL_SIDE")
        up_probability = side_probabilities[:, 2] / denominator
        up_truth = bits[side_mask] == 4
        side_true_probability = (np.where(up_truth, side_probabilities[:, 2],
                                          side_probabilities[:, 0]) / denominator)
        side_log, side_zeros = _log_sum(side_true_probability)
        side = {"count": int(side_mask.sum()),
                "brier_sum": float(np.square(up_probability - up_truth).sum(dtype=np.float64)),
                "log_sum": side_log, "zero_truth_probability_rows": side_zeros}

        singleton_probabilities = p[singleton_mask, j, :]
        singleton_bits = bits[singleton_mask]
        class_index = np.where(singleton_bits == 1, 0,
                               np.where(singleton_bits == 2, 1, 2))
        truth = np.eye(3, dtype=np.float64)[class_index]
        class_log, class_zeros = _log_sum(
            singleton_probabilities[np.arange(len(class_index)), class_index])
        three = {"count": int(singleton_mask.sum()),
                 "brier_sum": float(np.square(singleton_probabilities - truth).sum(dtype=np.float64)),
                 "log_sum": class_log, "zero_truth_probability_rows": class_zeros}
        report[tf] = {"activity": activity, "conditional_side": side,
                      "three_class": three}
    return {"per_tf": report,
            "contract_sha256": five_endpoint_contract()["contract_sha256"]}
