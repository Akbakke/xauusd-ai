"""Research S/F loss with conditional side normalized on certified side rows.

Reuses the unchanged v2 activity/side loss and only corrects the side term's
support denominator. No new labels, masks, model input or trading authority.
"""
from __future__ import annotations

import hashlib
import json

import torch
import torch.nn.functional as F

from gx1.contracts import market_state_time_query_loss_v2 as base
from gx1.contracts.market_state_time_query_target_v1 import TIMEFRAMES


def contract() -> dict:
    value = {
        'schema_version': 'gx1_market_state_time_query_side_normalized_loss_v1',
        'role': 'research_only_no_entry_exit_or_trade_authority',
        'base_loss_contract_sha256': base.factorized_time_query_loss_contract()['contract_sha256'],
        'loss': 'mean_per_TF_activity_BCE_over_informative_plus_conditional_side_BCE_over_known_side_divided_by_five',
        'side_gradient': 'zero_direct_side_logit_gradient_for_bits_2_5_7_or_immature',
        'empty_side_support': 'zero_side_term_no_imputed_direction',
        'future_labels_as_model_forward_inputs': False,
    }
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return {**value, 'contract_sha256': hashlib.sha256(raw).hexdigest()}


def allowed_set_loss(raw_logits: torch.Tensor, allowed: torch.Tensor,
                     eligible: torch.Tensor, mature: torch.Tensor,
                     query_k: torch.Tensor) -> dict:
    """Preserve v2 masks, then divide each side term by its known-side count."""
    result = base.allowed_set_loss(raw_logits, allowed, eligible, mature, query_k)
    side_known = result['side_known_mask']
    side_nll = F.binary_cross_entropy_with_logits(
        raw_logits[..., 1], (allowed == 4).to(raw_logits.dtype), reduction='none')
    side_cell = torch.where(side_known, side_nll, torch.zeros_like(side_nll))
    zero = raw_logits.mul(0).sum()
    terms = {}
    side_counts = {}
    for j, tf in enumerate(TIMEFRAMES):
        n_side = int(side_known[:, j].sum().item())
        n_info = result['per_tf_counts'][tf]
        side_counts[tf] = n_side
        correction = (side_cell[:, j].sum() * (1 / n_side - 1 / n_info)
                      if n_side else zero)
        terms[tf] = result['per_tf_mean_loss'][tf] + correction
    loss = sum(terms.values(), zero) / len(TIMEFRAMES)
    if not bool(torch.isfinite(loss)):
        raise ValueError('TIME_QUERY_SIDE_NORMALIZED_NONFINITE_LOSS')
    return {**result, 'loss': loss, 'per_tf_mean_loss': terms,
            'per_tf_side_counts': side_counts,
            'contract_sha256': contract()['contract_sha256']}
