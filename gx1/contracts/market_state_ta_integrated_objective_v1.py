"""One fixed Fold2 non-GC S/F plus auxiliary M5 high/low TRAIN loss."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import torch

from gx1.contracts.market_state_ta_integrated_model_v1 import masked_structure_hit_sums
from gx1.contracts.market_state_time_query_model_v2 import factorized_log_probabilities
from gx1.contracts.market_state_time_query_native_F_objective_v1 import (
    combined_loss, contract as dual_loss_contract,
)

STRUCTURE_WEIGHT = 0.25
FUTURE_WEIGHT = 1.0


def contract() -> dict:
    base = dual_loss_contract(ta_only=True, dedicated_m5_side=True,
        dedicated_non_m5_side=True, dedicated_tf_activity=True,
        direct_tf_hidden=True, m5_equal_observation=True, fold_id=2)
    value = {
        'schema_version': 'gx1_ta_integrated_fold2_objective_v1',
        'role': 'research_only_no_fit_or_trade_authority',
        'S_F_loss_contract_sha256': base['contract_sha256'],
        'S_F': 'existing_five_TF_activity_and_known_conditional_side_masks_with_fold2_M5_equal_observation',
        'F_weight': FUTURE_WEIGHT,
        'structure_targets': ['next_high_hits_frozen_high', 'next_low_hits_frozen_low'],
        'structure_unknowns': 'masked_zero_gradient_no_imputed_negative_or_flat',
        'structure_loss': 'mean_observed_high_BCE_plus_mean_observed_low_BCE_divided_by_two_per_batch',
        'structure_weight': STRUCTURE_WEIGHT,
        'total': 'S_F_loss_plus_structure_weight_times_structure_loss',
        'test_data_used': False,
    }
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return {**value, 'contract_sha256': hashlib.sha256(raw).hexdigest()}


def integrated_loss(*, model_output: dict, state_truth: dict, future_truth: dict,
                    state_query_k: torch.Tensor, structure_hit: np.ndarray,
                    structure_observed: np.ndarray) -> dict:
    if not (type(model_output) is dict and set(model_output) == {
            'state_activity_side_logits', 'state_log_probs',
            'future_activity_side_logits', 'future_log_probs',
            'm5_structure_hit_logits'}):
        raise ValueError('TA_INTEGRATED_LOSS_MODEL_OUTPUT')
    raw = model_output['m5_structure_hit_logits']
    for task in ('state', 'future'):
        logits = model_output[task + '_activity_side_logits']
        log_probs = model_output[task + '_log_probs']
        if not (isinstance(logits, torch.Tensor) and logits.dtype == torch.float32
                and logits.ndim == 3 and logits.shape[1:] == (5, 2)
                and isinstance(log_probs, torch.Tensor)
                and log_probs.shape == (logits.shape[0], 5, 3)
                and log_probs.device == logits.device
                and torch.equal(log_probs, factorized_log_probabilities(logits))):
            raise ValueError('TA_INTEGRATED_LOSS_FACTOR_PROBABILITIES')
    if not (isinstance(raw, torch.Tensor) and raw.dtype == torch.float32
            and raw.ndim == 2 and raw.shape[1] == 2
            and isinstance(structure_hit, np.ndarray)
            and isinstance(structure_observed, np.ndarray)
            and structure_hit.dtype == structure_observed.dtype == np.bool_
            and structure_hit.shape == structure_observed.shape == tuple(raw.shape)
            and bool(np.all(~structure_hit | structure_observed))):
        raise ValueError('TA_INTEGRATED_LOSS_STRUCTURE')
    sf = combined_loss(state_raw=model_output['state_activity_side_logits'],
        future_raw=model_output['future_activity_side_logits'],
        state_truth=state_truth, future_truth=future_truth,
        state_query_k=state_query_k, future_weight=FUTURE_WEIGHT,
        ta_only=True, dedicated_m5_side=True, dedicated_non_m5_side=True,
        dedicated_tf_activity=True, direct_tf_hidden=True,
        m5_equal_observation=True, fold_id=2)
    structure = masked_structure_hit_sums(raw,
        torch.from_numpy(structure_hit).to(device=raw.device),
        torch.from_numpy(structure_observed).to(device=raw.device))
    per_side = structure['loss_sums'] / structure['observed_counts'].clamp(min=1)
    auxiliary = per_side.mean()
    total = sf['loss'] + STRUCTURE_WEIGHT * auxiliary
    if not bool(torch.isfinite(total)):
        raise ValueError('TA_INTEGRATED_LOSS_NONFINITE')
    return {'loss': total, 'S_F_loss': sf['loss'], 'structure_loss': auxiliary,
            'structure_loss_per_side': per_side,
            'structure_observed_counts': structure['observed_counts'],
            'state_per_tf_counts': sf['state']['per_tf_counts'],
            'future_per_tf_counts': sf['future']['per_tf_counts'],
            'skip_update': sf['skip_update'] and not bool(structure['observed_counts'].any()),
            'contract_sha256': contract()['contract_sha256']}
