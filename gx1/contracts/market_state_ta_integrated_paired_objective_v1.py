"""One fixed paired S query loss plus native F and masked M5 structure."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import torch

from gx1.contracts.market_state_ta_integrated_model_v1 import masked_structure_hit_sums
from gx1.contracts.market_state_time_query_model_v2 import factorized_log_probabilities
from gx1.contracts.market_state_time_query_native_F_objective_v1 import (
    combined_loss, m5_equal_observation_profile,
)
from gx1.contracts.market_state_time_query_side_normalized_loss_v1 import allowed_set_loss
from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K

S_SAMPLED_WEIGHT = 0.5
S_NATIVE_WEIGHT = 0.5
F_WEIGHT = 1.0
STRUCTURE_WEIGHT = 0.25


def contract() -> dict:
    value = {'schema_version': 'gx1_ta_fold2_paired_S_integrated_objective_v1',
        'role': 'research_only_no_fit_or_trade_authority',
        'sampled_S_query': 'label_blind_k_in_1_to_native_k_per_TF_per_issue',
        'native_S_query': list(NATIVE_K),
        'sampled_S_weight': S_SAMPLED_WEIGHT,
        'native_S_weight': S_NATIVE_WEIGHT,
        'native_F_weight': F_WEIGHT,
        'M5_structure_weight': STRUCTURE_WEIGHT,
        'S_F_relation': 'mean_of_sampled_and_native_S_plus_one_native_F',
        'M5_native_S': 'duplicates_sampled_M5_k1_but_average_preserves_mass',
        'unknowns': 'separate_activity_side_and_structure_masks_zero_gradient',
        'test_data_used': False}
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return {**value, 'contract_sha256': hashlib.sha256(raw).hexdigest()}


def paired_integrated_loss(*, model_output: dict, sampled_truth: dict,
                           native_truth: dict, future_truth: dict,
                           query_grid: torch.Tensor, structure_hit: np.ndarray,
                           structure_observed: np.ndarray) -> dict:
    expected = {'state_activity_side_logits', 'state_log_probs',
        'state_native_activity_side_logits', 'state_native_log_probs',
        'future_activity_side_logits', 'future_log_probs',
        'm5_structure_hit_logits'}
    if type(model_output) is not dict or set(model_output) != expected:
        raise ValueError('TA_PAIRED_OUTPUT_KEYS')
    sampled = model_output['state_activity_side_logits']
    native = model_output['state_native_activity_side_logits']
    future = model_output['future_activity_side_logits']
    structure = model_output['m5_structure_hit_logits']
    if not (all(isinstance(x, torch.Tensor) and x.dtype == torch.float32
                and x.shape == sampled.shape for x in (sampled, native, future))
            and sampled.ndim == 3 and sampled.shape[1:] == (5, 2)
            and all(model_output[key + '_log_probs'].shape == (len(sampled), 5, 3)
                    and torch.equal(model_output[key + '_log_probs'],
                                    factorized_log_probabilities(raw))
                    for key, raw in (('state', sampled),
                                     ('state_native', native), ('future', future)))
            and isinstance(query_grid, torch.Tensor) and query_grid.dtype == torch.int64
            and query_grid.device == sampled.device
            and query_grid.shape == (len(sampled), 2, 5)
            and bool(torch.equal(query_grid[:, 1],
                torch.tensor(NATIVE_K, dtype=torch.int64, device=sampled.device)
                    .expand(len(sampled), -1)))
            and bool(torch.equal(query_grid[:, 0, 0], query_grid[:, 1, 0]))):
        raise ValueError('TA_PAIRED_OUTPUT_QUERY')
    def truth_tensor(truth, key, dtype):
        value = truth[key]
        if not (isinstance(value, np.ndarray) and value.dtype == np.dtype(dtype)
                and value.shape == (len(sampled), 5)):
            raise ValueError('TA_PAIRED_TRUTH_SHAPE')
        return torch.from_numpy(value).to(device=sampled.device)
    if not (np.array_equal(sampled_truth['allowed'][:, 0], native_truth['allowed'][:, 0])
            and np.array_equal(sampled_truth['eligible'][:, 0], native_truth['eligible'][:, 0])
            and np.array_equal(sampled_truth['mature'][:, 0], native_truth['mature'][:, 0])):
        raise ValueError('TA_PAIRED_M5_TARGET_PARITY')
    sf = combined_loss(state_raw=sampled, future_raw=future,
        state_truth=sampled_truth, future_truth=future_truth,
        state_query_k=query_grid[:, 0], future_weight=F_WEIGHT,
        ta_only=True, dedicated_m5_side=True, dedicated_non_m5_side=True,
        dedicated_tf_activity=True, direct_tf_hidden=True,
        m5_equal_observation=True, fold_id=2)
    profile = m5_equal_observation_profile(fold_id=2)['state']
    native_s = allowed_set_loss(native,
        truth_tensor(native_truth, 'allowed', 'uint8'),
        truth_tensor(native_truth, 'eligible', 'bool'),
        truth_tensor(native_truth, 'mature', 'bool'),
        query_grid[:, 1], m5_side_normalization=profile)
    if not (isinstance(structure, torch.Tensor) and structure.dtype == torch.float32
            and structure.shape == (len(sampled), 2)
            and isinstance(structure_hit, np.ndarray)
            and isinstance(structure_observed, np.ndarray)
            and structure_hit.dtype == structure_observed.dtype == np.bool_
            and structure_hit.shape == structure_observed.shape == tuple(structure.shape)
            and bool(np.all(~structure_hit | structure_observed))):
        raise ValueError('TA_PAIRED_STRUCTURE')
    aux = masked_structure_hit_sums(structure,
        torch.from_numpy(structure_hit.copy()).to(device=structure.device),
        torch.from_numpy(structure_observed.copy()).to(device=structure.device))
    aux_loss = (aux['loss_sums'] / aux['observed_counts'].clamp(min=1)).mean()
    total = (S_SAMPLED_WEIGHT * sf['state']['loss']
             + S_NATIVE_WEIGHT * native_s['loss']
             + F_WEIGHT * sf['future']['loss']
             + STRUCTURE_WEIGHT * aux_loss)
    if not bool(torch.isfinite(total)):
        raise ValueError('TA_PAIRED_NONFINITE')
    return {'loss': total, 'sampled_S_loss': sf['state']['loss'],
            'native_S_loss': native_s['loss'], 'native_F_loss': sf['future']['loss'],
            'structure_loss': aux_loss,
            'sampled_S_counts': sf['state']['per_tf_counts'],
            'native_S_counts': native_s['per_tf_counts'],
            'native_F_counts': sf['future']['per_tf_counts'],
            'structure_observed_counts': aux['observed_counts'],
            'skip_update': sf['skip_update'] and native_s['skip_update']
                           and not bool(aux['observed_counts'].any()),
            'contract_sha256': contract()['contract_sha256']}
