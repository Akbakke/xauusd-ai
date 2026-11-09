"""Research S/F loss with conditional side normalized on certified side rows.

Reuses the unchanged v2 activity/side loss and only corrects the side term's
support denominator. No new labels, masks, model input or trading authority.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

import torch
import torch.nn.functional as F

from gx1.contracts import market_state_time_query_loss_v2 as base
from gx1.contracts.market_state_time_query_target_v1 import TIMEFRAMES


def _m5_normalization(value):
    if value is None:
        return None
    if not (isinstance(value, Mapping)
            and set(value) == {'known_side_rows', 'nonempty_batches'}
            and all(type(v) is int for v in value.values())
            and 1 <= value['nonempty_batches'] <= value['known_side_rows']
                <= 16 * value['nonempty_batches']):
        raise ValueError('M5_EQUAL_OBSERVATION_NORMALIZATION')
    return dict(value)


def contract(*, m5_side_normalization=None) -> dict:
    normalizer = _m5_normalization(m5_side_normalization)
    value = {
        'schema_version': 'gx1_market_state_time_query_side_normalized_loss_v1',
        'role': 'research_only_no_entry_exit_or_trade_authority',
        'base_loss_contract_sha256': base.factorized_time_query_loss_contract()['contract_sha256'],
        'loss': 'mean_per_TF_activity_BCE_over_informative_plus_conditional_side_BCE_over_known_side_divided_by_five',
        'side_gradient': 'zero_direct_side_logit_gradient_for_bits_2_5_7_or_immature',
        'empty_side_support': 'zero_side_term_no_imputed_direction',
        'future_labels_as_model_forward_inputs': False,
    }
    if normalizer is not None:
        value['m5_side_normalization'] = normalizer
        value['m5_side_loss'] = 'sum_known_M5_side_BCE_times_nonempty_fit_batches_over_known_fit_rows_before_existing_TF_divisor'
        value['unchanged_terms'] = 'M5_activity_and_all_other_TF_terms_and_observation_masks'
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return {**value, 'contract_sha256': hashlib.sha256(raw).hexdigest()}


def allowed_set_loss(raw_logits: torch.Tensor, allowed: torch.Tensor,
                     eligible: torch.Tensor, mature: torch.Tensor,
                     query_k: torch.Tensor, *, m5_side_normalization=None) -> dict:
    """Preserve v2 masks, then divide each side term by its known-side count."""
    normalizer = _m5_normalization(m5_side_normalization)
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
        if j == 0 and normalizer is not None:
            if n_side > normalizer['known_side_rows']:
                raise ValueError('M5_EQUAL_OBSERVATION_BATCH_SUPPORT')
            # Rebuild this one activity term to avoid subtractive cancellation
            # in the side gradient. Every known M5 side gets exactly q/N;
            # activity and all other TF gradients keep their existing scale.
            activity_nll = F.binary_cross_entropy_with_logits(
                raw_logits[:, 0, 0], (allowed[:, 0] != 2).to(raw_logits.dtype),
                reduction='none')
            activity_sum = torch.where(result['informative_mask'][:, 0],
                                       activity_nll, torch.zeros_like(activity_nll)).sum()
            activity = activity_sum / n_info if n_info else zero
            terms[tf] = activity + side_cell[:, 0].sum() * (
                normalizer['nonempty_batches'] / normalizer['known_side_rows'])
    loss = sum(terms.values(), zero) / len(TIMEFRAMES)
    if not bool(torch.isfinite(loss)):
        raise ValueError('TIME_QUERY_SIDE_NORMALIZED_NONFINITE_LOSS')
    return {**result, 'loss': loss, 'per_tf_mean_loss': terms,
            'per_tf_side_counts': side_counts,
            'contract_sha256': contract(m5_side_normalization=normalizer)['contract_sha256']}
