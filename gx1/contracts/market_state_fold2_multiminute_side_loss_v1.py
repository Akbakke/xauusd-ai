"""Fold 2 fixed multiminute S loss with equal certified-row task means.

The five clock points per TF are source-bound. Activity and known side are
normalized on their own full FIT supports for each point/TF; unknown side has
zero side gradient. Source-only candidate; no ONLINE change or fit authority.
"""
from __future__ import annotations

import hashlib
import json

import torch
import torch.nn.functional as F

from gx1.contracts.market_state_fold2_multiminute_path_v1 import (
    QUERY_MINUTES, QUERY_SHA256,
)
from gx1.contracts.market_state_time_query_target_v1 import TIMEFRAMES

FIT_ROWS = 47338
BATCH_SIZE = 16
FIT_BATCHES = (FIT_ROWS + BATCH_SIZE - 1) // BATCH_SIZE
SUPPORT_REVIEW_SHA256 = 'eb926b6324dbacb7e2c4bf536b5d49bd669475655c30a2c88a8aab32743c8065'
# Five fixed query points x five TF; exact certified Fold 2 fit support.
FIT_INFORMATIVE = (
    (47299, 47209, 46855, 45626, 37940),
    (47301, 47216, 46540, 43955, 34476),
    (47302, 47046, 46067, 43604, 32172),
    (47306, 46880, 45599, 43232, 29728),
    (47302, 46884, 45142, 42784, 29142),
)
FIT_KNOWN_SIDE = (
    (15891, 18184, 18140, 19108, 16360),
    (15933, 18267, 18127, 18848, 14595),
    (16057, 18203, 18078, 18756, 13385),
    (16085, 18120, 18023, 18625, 12353),
    (16101, 18182, 17973, 18391, 12291),
)


def contract() -> dict:
    value = {
        'schema_version': 'gx1_market_state_fold2_multiminute_side_loss_v1',
        'role': 'source_candidate_no_fit_or_trade_authority',
        'fold_id': 2, 'partition': 'fit', 'fit_rows': FIT_ROWS,
        'batch_size': BATCH_SIZE, 'fit_batches': FIT_BATCHES,
        'query_sha256': QUERY_SHA256, 'query_minutes': QUERY_MINUTES.tolist(),
        'support_review_sha256': SUPPORT_REVIEW_SHA256,
        'informative_support': FIT_INFORMATIVE,
        'known_side_support': FIT_KNOWN_SIDE,
        'reduction': 'mean_over_25_query_TF_cells_of_full_epoch_activity_and_certified_side_row_means',
        'unknown_side_gradient': 'zero_for_bits_2_5_7_ineligible_or_immature',
        'future_label_as_forward_input': False,
        'fresh_initial_baseline_required_before_fit': True,
        'test_data_used': False,
    }
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'),
                     allow_nan=False).encode()
    return {**value, 'contract_sha256': hashlib.sha256(raw).hexdigest()}


def allowed_set_loss(raw: torch.Tensor, allowed: torch.Tensor,
                     eligible: torch.Tensor, mature: torch.Tensor,
                     query_minutes: torch.Tensor) -> dict:
    shape = raw.shape[:3] if isinstance(raw, torch.Tensor) and raw.ndim == 4 else ()
    if not (isinstance(raw, torch.Tensor) and raw.dtype == torch.float32
            and raw.ndim == 4 and raw.shape[1:] == (len(QUERY_MINUTES), len(TIMEFRAMES), 2)
            and 1 <= raw.shape[0] <= BATCH_SIZE
            and bool(torch.isfinite(raw).all())
            and isinstance(query_minutes, torch.Tensor)
            and query_minutes.dtype == torch.int64
            and query_minutes.shape == shape and query_minutes.device == raw.device
            and bool((query_minutes == torch.as_tensor(
                QUERY_MINUTES.copy(), dtype=torch.int64, device=raw.device)[None]).all())
            and isinstance(allowed, torch.Tensor) and allowed.dtype == torch.uint8
            and allowed.shape == shape and allowed.device == raw.device
            and bool(torch.isin(allowed, torch.tensor(
                (1, 2, 4, 5, 7), dtype=torch.uint8, device=raw.device)).all())
            and all(isinstance(mask, torch.Tensor) and mask.dtype == torch.bool
                    and mask.shape == shape and mask.device == raw.device
                    for mask in (eligible, mature))
            and bool((~eligible | mature).all())
            and bool((allowed[~mature] == 7).all())):
        raise ValueError('FOLD2_MULTIMINUTE_LOSS_SHAPE_MASK_OR_QUERY')
    informative = eligible & mature & (allowed != 7)
    side_known = informative & ((allowed == 1) | (allowed == 4))
    activity_nll = F.binary_cross_entropy_with_logits(
        raw[..., 0], (allowed != 2).to(raw.dtype), reduction='none')
    side_nll = F.binary_cross_entropy_with_logits(
        raw[..., 1], (allowed == 4).to(raw.dtype), reduction='none')
    activity_sum = torch.where(informative, activity_nll,
                               torch.zeros_like(activity_nll)).sum(dim=0)
    side_sum = torch.where(side_known, side_nll,
                           torch.zeros_like(side_nll)).sum(dim=0)
    info_support = torch.tensor(FIT_INFORMATIVE, dtype=raw.dtype, device=raw.device)
    side_support = torch.tensor(FIT_KNOWN_SIDE, dtype=raw.dtype, device=raw.device)
    terms = activity_sum * (FIT_BATCHES / info_support) + side_sum * (FIT_BATCHES / side_support)
    loss = terms.mean()
    if not bool(torch.isfinite(loss)):
        raise ValueError('FOLD2_MULTIMINUTE_NONFINITE_LOSS')
    return {
        'loss': loss,
        'per_query_tf_terms': terms,
        'informative_counts': informative.sum(dim=0),
        'known_side_counts': side_known.sum(dim=0),
        'informative_mask': informative, 'side_known_mask': side_known,
        'skip_update': not bool(informative.any()),
        'contract_sha256': contract()['contract_sha256'],
    }
