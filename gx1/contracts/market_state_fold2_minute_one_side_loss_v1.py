"""Fixed Fold 2 minute-one S loss with equal certified-side row weight.

Source-only candidate. It changes no target, model forward, existing v1 loss,
optimizer, TEST or trading state. A fit needs its own fresh baseline and gate.
"""
from __future__ import annotations

import hashlib
import json

import torch
import torch.nn.functional as F

from gx1.contracts.market_state_minute_query_candidate_v1 import (
    contract as minute_contract, minute_allowed_set_loss,
)
from gx1.contracts.market_state_time_query_target_v1 import TIMEFRAMES

FIT_ROWS = 47338
BATCH_SIZE = 16
FIT_BATCHES = (FIT_ROWS + BATCH_SIZE - 1) // BATCH_SIZE
KNOWN_SIDE_FIT = {
    "M5": 15891, "M15": 18184, "H1": 18140, "H4": 19108, "D1": 16360,
}
SUPPORT_REVIEW_SHA256 = "06dd85b07d5d0c5b26995251f9b410b903826d1d450c4e77dfff473094488b2c"


def contract() -> dict:
    value = {
        "schema_version": "gx1_market_state_fold2_minute_one_side_loss_v1",
        "role": "source_candidate_no_fit_or_trade_authority",
        "base_minute_contract_sha256": minute_contract()["contract_sha256"],
        "support_review_sha256": SUPPORT_REVIEW_SHA256,
        "fold_id": 2,
        "partition": "fit",
        "query_minute_all_timeframes": 1,
        "fit_rows": FIT_ROWS,
        "batch_size": BATCH_SIZE,
        "fit_batches": FIT_BATCHES,
        "known_side_fit_rows": KNOWN_SIDE_FIT,
        "required_epoch": "each_bound_fit_row_once_in_fixed_chronological_batch_order",
        "loss": "equal_TF_activity_mean_over_informative_plus_side_sum_times_fit_batches_over_known_side_fit_rows",
        "unknown_side_gradient": "zero_for_bits_2_5_7_or_immature",
        "future_label_as_forward_input": False,
        "fresh_initial_baseline_required_before_fit": True,
        "test_data_used": False,
    }
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode()
    return {**value, "contract_sha256": hashlib.sha256(raw).hexdigest()}


def allowed_set_loss(raw: torch.Tensor, allowed: torch.Tensor,
                     eligible: torch.Tensor, mature: torch.Tensor,
                     query_minutes: torch.Tensor) -> dict:
    """One fixed minute per TF; equal TF and certified-side row weights."""
    if not (isinstance(raw, torch.Tensor) and raw.ndim == 3
            and 1 <= raw.shape[0] <= BATCH_SIZE
            and isinstance(query_minutes, torch.Tensor)
            and query_minutes.shape == (raw.shape[0], len(TIMEFRAMES))
            and bool((query_minutes == 1).all())):
        raise ValueError("FOLD2_MINUTE_ONE_FIXED_QUERY_OR_BATCH")
    base = minute_allowed_set_loss(
        raw, allowed, eligible, mature, query_minutes)
    informative = base["informative_mask"]
    side_known = base["side_known_mask"]
    activity_nll = F.binary_cross_entropy_with_logits(
        raw[..., 0], (allowed != 2).to(raw.dtype), reduction="none")
    side_nll = F.binary_cross_entropy_with_logits(
        raw[..., 1], (allowed == 4).to(raw.dtype), reduction="none")
    zero = raw.mul(0).sum()
    terms, side_counts = {}, {}
    for j, tf in enumerate(TIMEFRAMES):
        n_info = base["per_tf_counts"][tf]
        n_side = int(side_known[:, j].sum().item())
        side_counts[tf] = n_side
        activity = (
            torch.where(informative[:, j], activity_nll[:, j],
                        torch.zeros_like(activity_nll[:, j])).sum() / n_info
            if n_info else zero)
        side = torch.where(
            side_known[:, j], side_nll[:, j],
            torch.zeros_like(side_nll[:, j])).sum()
        terms[tf] = activity + side * (FIT_BATCHES / KNOWN_SIDE_FIT[tf])
    loss = sum(terms.values(), zero) / len(TIMEFRAMES)
    if not bool(torch.isfinite(loss)):
        raise ValueError("FOLD2_MINUTE_ONE_NONFINITE_LOSS")
    return {**base, "loss": loss, "per_tf_mean_loss": terms,
            "per_tf_side_counts": side_counts,
            "contract_sha256": contract()["contract_sha256"]}
