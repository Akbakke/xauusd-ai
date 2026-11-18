"""Fixed costed M5 decision head over the existing five-TF TA model."""
from __future__ import annotations

import hashlib
import json
from typing import Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from gx1.contracts.market_state_ta_integrated_model_v1 import IntegratedTAMarketStateModel
from gx1.contracts.market_state_ta_integrated_paired_objective_v1 import (
    contract as paired_contract, paired_integrated_loss,
)

HORIZONS = (1, 5, 15, 60, 240, 480)
FIELDS = ('net_bps', 'mfe_bps', 'pre_mfe_mae_upper_bps')
AUX_WEIGHT = 0.10
CORE_KEYS = frozenset(('state_activity_side_logits', 'state_log_probs',
    'state_native_activity_side_logits', 'state_native_log_probs',
    'future_activity_side_logits', 'future_log_probs', 'm5_structure_hit_logits'))


def require(ok, reason):
    if not bool(ok):
        raise ValueError('TA_M5_DECISION_' + reason)


def fit_statistics(targets: Mapping[str, np.ndarray]) -> dict:
    """Fit component and paired-side population scales on observed FIT cells."""
    term, path = targets['terminal_observed'], targets['path_observed']
    n = len(term)
    require(term.shape == path.shape == (n, 6) and n > 0
            and term.dtype == path.dtype == np.bool_
            and np.array_equal(targets['horizon_minutes'], HORIZONS)
            and np.all(~path | term), 'FIT_MASKS')
    mean = np.empty((6, 2, 3), dtype=np.float32)
    std = np.empty_like(mean)
    counts = np.empty((6, 2, 3), dtype=np.int64)
    values = []
    for k, field in enumerate(FIELDS):
        value = targets[field]
        mask = term if k == 0 else path
        require(value.shape == (n, 6, 2)
                and np.array_equal(np.isfinite(value).all(axis=2), mask),
                'FIT_' + field.upper())
        values.append(value)
        for j in range(6):
            for side in range(2):
                observed = value[mask[:, j], j, side].astype(np.float64)
                require(len(observed) > 1, 'FIT_COUNT')
                mu = float(observed.mean())
                scale = float(np.sqrt(np.mean((observed - mu) ** 2)))
                require(np.isfinite(mu) and np.isfinite(scale) and scale > 0,
                        'FIT_SCALE')
                mean[j, side, k], std[j, side, k] = mu, scale
                counts[j, side, k] = len(observed)
    qstd = np.empty((6, 2), dtype=np.float32)
    for j in range(6):
        for side in range(2):
            q = (values[0][path[:, j], j, side]
                 - values[2][path[:, j], j, side]).astype(np.float64)
            scale = float(np.sqrt(np.mean((q - q.mean()) ** 2)))
            require(np.isfinite(scale) and scale > 0, 'FIT_QUALITY_SCALE')
            qstd[j, side] = scale
    paired_std = np.empty(6, dtype=np.float32)
    for j in range(6):
        quality = (values[0][path[:, j], j, :]
                   - values[2][path[:, j], j, :]).astype(np.float64)
        difference = quality[:, 0] - quality[:, 1]
        scale = float(np.sqrt(np.mean((difference - difference.mean()) ** 2)))
        require(np.isfinite(scale) and scale > 0, 'FIT_PAIRED_QUALITY_SCALE')
        paired_std[j] = scale
    result = {'mean_bps': mean, 'std_bps': std,
              'quality_std_bps': qstd, 'paired_quality_std_bps': paired_std,
              'observed_count': counts}
    payload = {key: value.tolist() for key, value in result.items()}
    result['sha256'] = hashlib.sha256(json.dumps(payload, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return result


def contract(core_sha256: str, statistics_sha256: str, *, aux_fold_id: int = 2) -> dict:
    require(type(aux_fold_id) is int and aux_fold_id in (1, 2), 'AUX_FOLD_ID')
    value = {'schema_version': 'gx1_ta_m5_cost_path_decision_v3',
        'role': 'research_only_no_fit_or_trade_authority',
        'core_sha256': core_sha256, 'fit_statistics_sha256': statistics_sha256,
        'input': 'all_five_post_attention_M5_MTF_vectors',
        'horizon_minutes': list(HORIZONS), 'side_order': ['LONG', 'SHORT'],
        'field_order': list(FIELDS), 'head': 'single_linear_5TF_to_36',
        'head_initial': 'zero_units_equals_FIT_component_mean',
        'target_masks': 'net_terminal_MFE_MAE_complete_path_separate',
        'loss': 'six_equal_horizon_terms_unit_Huber_net_MFE_MAE_quality_side_pair_plus_one_sided_unit_MAE_underprediction',
        'side_pair_scale': 'population_std_of_FIT_path_observed_paired_LONG_minus_SHORT_quality',
        'decision_weight': 1.0, 'paired_S_F_structure_aux_weight': AUX_WEIGHT,
        'action': 'argmax_predicted_net_minus_nonnegative_MAE_or_FLAT_if_nonpositive',
        'tie': 'earliest_horizon_then_LONG; FLAT_at_zero',
        'test_data_used': False}
    # Each Fold1 objective keeps its explicitly bound auxiliary loss.
    if aux_fold_id == 1:
        value['cold_aux_fold_id'] = aux_fold_id
        value['cold_aux_contract_sha256'] = paired_contract(fold_id=aux_fold_id)['contract_sha256']
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return {**value, 'contract_sha256': hashlib.sha256(raw).hexdigest()}


class IntegratedTAM5DecisionModel(nn.Module):
    def __init__(self, core: IntegratedTAMarketStateModel, statistics: dict, *,
                 aux_fold_id: int = 2):
        super().__init__()
        require(type(core) is IntegratedTAMarketStateModel, 'CORE_TYPE')
        require(type(aux_fold_id) is int and aux_fold_id in (1, 2), 'AUX_FOLD_ID')
        self.aux_fold_id = aux_fold_id
        require(all(statistics[key].shape == shape for key, shape in (
            ('mean_bps', (6, 2, 3)), ('std_bps', (6, 2, 3)),
            ('quality_std_bps', (6, 2)),
            ('paired_quality_std_bps', (6,)))), 'STAT_SHAPE')
        require(all(np.isfinite(statistics[key]).all() for key in
                    ('mean_bps', 'std_bps', 'quality_std_bps',
                     'paired_quality_std_bps'))
                and np.all(statistics['std_bps'] > 0)
                and np.all(statistics['quality_std_bps'] > 0)
                and np.all(statistics['paired_quality_std_bps'] > 0), 'STAT_FINITE')
        self.core = core
        device = core.dual.state_model.query_head.weight.device
        width = core.dual.state_model._hidden_width
        self.decision_head = nn.Linear(5 * width, 6 * 2 * 3,
                                       device=device, dtype=torch.float32)
        nn.init.zeros_(self.decision_head.weight)
        nn.init.zeros_(self.decision_head.bias)
        self.register_buffer('target_mean_bps', torch.as_tensor(
            statistics['mean_bps'], device=device).clone())
        self.register_buffer('target_std_bps', torch.as_tensor(
            statistics['std_bps'], device=device).clone())
        self.register_buffer('quality_std_bps', torch.as_tensor(
            statistics['quality_std_bps'], device=device).clone())
        self.register_buffer('paired_quality_std_bps', torch.as_tensor(
            statistics['paired_quality_std_bps'], device=device).clone())
        self.statistics_sha256 = statistics['sha256']

    @property
    def research_objective(self) -> dict:
        return contract(self.core.research_objective['contract_sha256'],
                        self.statistics_sha256, aux_fold_id=self.aux_fold_id)

    def forward(self, features: Mapping[str, torch.Tensor], *, query_k: torch.Tensor) -> dict:
        out = self.core(features, query_k=query_k, return_fused_tf_hidden=True)
        hidden = out.pop('fused_tf_hidden')
        require(hidden.ndim == 3 and hidden.shape[1] == 5, 'FUSED_SHAPE')
        unit = self.decision_head(hidden.reshape(len(hidden), -1)).reshape(-1, 6, 2, 3)
        out['m5_cost_path_bps'] = self.target_mean_bps + self.target_std_bps * unit
        return out


def _horizon_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    counts = mask.sum(dim=(0, 2))
    means = (value * mask).sum(dim=(0, 2)) / counts.clamp(min=1)
    active = counts > 0
    return means[active].mean() if bool(active.any()) else value.sum() * 0


def decision_loss(*, model_output: dict, targets: Mapping[str, np.ndarray],
                  model: IntegratedTAM5DecisionModel, sampled_truth: dict,
                  native_truth: dict, future_truth: dict, query_grid: torch.Tensor,
                  structure_hit: np.ndarray, structure_observed: np.ndarray) -> dict:
    require(type(model_output) is dict and set(model_output) == CORE_KEYS | {'m5_cost_path_bps'},
            'OUTPUT_KEYS')
    pred = model_output['m5_cost_path_bps']
    n = len(pred)
    require(pred.shape == (n, 6, 2, 3) and pred.dtype == torch.float32
            and bool(torch.isfinite(pred).all()), 'PREDICTION')
    term, path = targets['terminal_observed'], targets['path_observed']
    require(term.shape == path.shape == (n, 6)
            and np.all(~path | term), 'TARGET_MASKS')
    masks = [term, path, path]
    field_losses = []
    tensors = []
    for k, field in enumerate(FIELDS):
        value = targets[field]
        require(value.shape == (n, 6, 2)
                and np.array_equal(np.isfinite(value).all(axis=2), masks[k]),
                'TARGET_' + field.upper())
        target = torch.from_numpy(np.nan_to_num(value, nan=0).copy()).to(
            device=pred.device, dtype=torch.float32)
        mask = torch.from_numpy(masks[k].copy()).to(pred.device)[:, :, None].expand(-1, -1, 2)
        error = (pred[..., k] - target) / model.target_std_bps[..., k]
        unit_huber = F.smooth_l1_loss(error, torch.zeros_like(error), reduction='none')
        field_losses.append(_horizon_mean(unit_huber, mask))
        tensors.append(target)
    q_target = tensors[0] - tensors[2]
    q_pred = pred[..., 0] - pred[..., 2].clamp_min(0)
    q_error = (q_pred - q_target) / model.quality_std_bps
    q_mask = torch.from_numpy(path.copy()).to(pred.device)[:, :, None].expand(-1, -1, 2)
    q_loss = _horizon_mean(F.smooth_l1_loss(q_error, torch.zeros_like(q_error),
                                           reduction='none'), q_mask)
    side_scale = model.paired_quality_std_bps
    side_error = ((q_pred[..., 0] - q_pred[..., 1])
                  - (q_target[..., 0] - q_target[..., 1])) / side_scale
    side_pair_loss = _horizon_mean(
        F.smooth_l1_loss(side_error, torch.zeros_like(side_error),
                         reduction='none')[:, :, None].expand(-1, -1, 2), q_mask)
    mae_under_loss = _horizon_mean(
        ((tensors[2] - pred[..., 2]) / model.target_std_bps[..., 2]).clamp_min(0),
        q_mask)
    primary = (sum(field_losses) + q_loss + side_pair_loss + mae_under_loss) / 6
    aux = paired_integrated_loss(model_output={key: model_output[key] for key in CORE_KEYS},
        sampled_truth=sampled_truth, native_truth=native_truth,
        future_truth=future_truth, query_grid=query_grid,
        structure_hit=structure_hit, structure_observed=structure_observed,
        fold_id=model.aux_fold_id)
    require(aux['contract_sha256'] == paired_contract(
        fold_id=model.aux_fold_id)['contract_sha256'], 'AUX_CONTRACT')
    total = primary + AUX_WEIGHT * aux['loss']
    require(bool(torch.isfinite(total)), 'LOSS_FINITE')
    return {'loss': total, 'decision_loss': primary, 'net_loss': field_losses[0],
            'mfe_loss': field_losses[1], 'mae_loss': field_losses[2],
            'quality_loss': q_loss, 'side_pair_loss': side_pair_loss,
            'mae_under_loss': mae_under_loss, 'aux_loss': aux['loss'],
            'skip_update': aux['skip_update'] and not bool(term.any()),
            'contract_sha256': model.research_objective['contract_sha256']}


def choose_actions(predicted_bps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    require(predicted_bps.ndim == 4 and predicted_bps.shape[1:] == (6, 2, 3)
            and np.isfinite(predicted_bps).all(), 'ACTION_INPUT')
    quality = predicted_bps[..., 0] - np.maximum(predicted_bps[..., 2], 0)
    flat_choice = quality.reshape(len(quality), -1).argmax(axis=1)
    best = np.take_along_axis(quality.reshape(len(quality), -1),
                              flat_choice[:, None], axis=1)[:, 0]
    action = np.where(best > 0, flat_choice % 2, 2).astype(np.int8)
    horizon = np.where(best > 0, flat_choice // 2, -1).astype(np.int8)
    return action, horizon
