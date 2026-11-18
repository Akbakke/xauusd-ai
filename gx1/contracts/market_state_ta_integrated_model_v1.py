"""M5/MTF TA S/F and M5 structure readout over one native encode.

The eight TA specialist families and closed M5/M15/H1/H4/D1 windows feed
the S/F heads directly. M1 trade management belongs to a later phase.
Targets enter only loss and scoring. This contract grants no fit authority.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from gx1.contracts.market_state_time_query_model_v2 import factorized_log_probabilities
from gx1.contracts.market_state_time_query_native_F_model_v1 import NativeFutureDualModel
from gx1.contracts.market_state_time_query_target_v1 import TIMEFRAMES


STRUCTURE_FIELDS = ('next_high_hits_frozen_high', 'next_low_hits_frozen_low')


def _need(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError('MARKET_STATE_TA_INTEGRATED_' + reason)


def contract(dual: NativeFutureDualModel) -> dict:
    _need(type(dual) is NativeFutureDualModel, 'DUAL_TYPE')
    state = dual.state_model
    _need(state._ta_only and state._direct_tf_hidden
          and state._dedicated_m5_side and state._dedicated_non_m5_side
          and state._dedicated_tf_activity and not dual._shared_m5_side,
          'FIVE_TF_MODE')
    value = {
        'schema_version': 'gx1_ta_integrated_m5_mtf_structure_model_v3',
        'role': 'research_only_no_fit_or_trade_authority',
        'dual_S_F_contract_sha256': dual.research_objective['contract_sha256'],
        'backbone': 'one_unchanged_full_TA_encode_with_five_post_attention_TF_vectors',
        'forecast_input_timeframes': list(TIMEFRAMES),
        'forecast_cadence': 'closed_M5',
        'S_F_input': 'direct_five_post_attention_TF_vectors',
        'S_F_direction_heads': 'five_distinct_per_task_conditional_side_heads',
        'M5_structure_logits': list(STRUCTURE_FIELDS),
        'M5_structure_truth_as_forward_input': False,
        'test_data_used': False,
    }
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return {**value, 'contract_sha256': hashlib.sha256(raw).hexdigest()}


class IntegratedTAMarketStateModel(nn.Module):
    """One M5/MTF encode shared by S/F and M5 high/low heads."""

    def __init__(self, dual: NativeFutureDualModel):
        super().__init__()
        objective = contract(dual)
        dual._require_binding()
        self.dual = dual
        width = dual.state_model._hidden_width
        device = dual.state_model.query_head.weight.device
        self.m5_structure_head = nn.Linear(width, len(STRUCTURE_FIELDS),
                                           device=device, dtype=torch.float32)
        self.register_buffer('integrated_objective_sha256', torch.tensor(
            list(bytes.fromhex(objective['contract_sha256'])),
            device=device, dtype=torch.uint8))

    @property
    def research_objective(self) -> dict:
        return contract(self.dual)

    def _require_binding(self) -> None:
        self.dual._require_binding()
        width = self.dual.state_model._hidden_width
        device = self.dual.state_model.query_head.weight.device
        digest = torch.tensor(list(bytes.fromhex(self.research_objective['contract_sha256'])),
                              device=device, dtype=torch.uint8)
        _need(type(self.m5_structure_head) is nn.Linear
              and self.m5_structure_head.in_features == width
              and self.m5_structure_head.out_features == len(STRUCTURE_FIELDS)
              and all(p.device == device and p.dtype == torch.float32
                      for p in self.m5_structure_head.parameters())
              and torch.equal(self.integrated_objective_sha256, digest),
              'MODEL_BINDING')

    def forward(self, features: Mapping[str, torch.Tensor], *, query_k: torch.Tensor,
                return_fused_tf_hidden: bool = False) -> dict[str, torch.Tensor]:
        self._require_binding()
        _need(isinstance(features, Mapping), 'FEATURE_MAPPING')
        hidden = self.dual.encode_features(**features)
        batch, tf_count, _ = hidden.shape
        paired = isinstance(query_k, torch.Tensor) and query_k.ndim == 3
        query_grid = None
        if isinstance(query_k, torch.Tensor):
            query_grid = query_k if paired else query_k[:, None, :]
        _need(tf_count == len(TIMEFRAMES)
              and isinstance(query_k, torch.Tensor) and query_k.dtype == torch.int64
              and query_grid.shape == (batch, 2 if paired else 1, tf_count)
              and query_k.device == hidden.device,
              'QUERY_GRID')
        state_grid = self.dual.state_model.decode_raw_query_grid(hidden, query_grid)
        state_raw = state_grid[:, 0]
        future_raw = self.dual.decode_native_F_raw(hidden)
        structure_raw = self.m5_structure_head(hidden[:, 0])
        _need(structure_raw.shape == (batch, len(STRUCTURE_FIELDS))
              and bool(torch.isfinite(structure_raw).all()), 'STRUCTURE_LOGITS')
        output = {'state_activity_side_logits': state_raw,
                  'state_log_probs': factorized_log_probabilities(state_raw),
                  'future_activity_side_logits': future_raw,
                  'future_log_probs': factorized_log_probabilities(future_raw),
                  'm5_structure_hit_logits': structure_raw}
        if paired:
            native_raw = state_grid[:, 1]
            output['state_native_activity_side_logits'] = native_raw
            output['state_native_log_probs'] = factorized_log_probabilities(native_raw)
        if return_fused_tf_hidden:
            output['fused_tf_hidden'] = hidden
        return output


def masked_structure_hit_sums(logits: torch.Tensor, hits: torch.Tensor,
                              observed: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return unweighted high/low loss sums for plan-bound global scaling."""
    _need(isinstance(logits, torch.Tensor) and logits.dtype == torch.float32
          and logits.ndim == 2 and logits.shape[1] == len(STRUCTURE_FIELDS)
          and logits.shape[0] > 0 and bool(torch.isfinite(logits).all())
          and isinstance(hits, torch.Tensor) and hits.dtype == torch.bool
          and hits.shape == logits.shape and hits.device == logits.device
          and isinstance(observed, torch.Tensor) and observed.dtype == torch.bool
          and observed.shape == logits.shape and observed.device == logits.device,
          'STRUCTURE_TRUTH')
    cells = F.binary_cross_entropy_with_logits(
        logits, hits.to(torch.float32), reduction='none')
    sums = torch.where(observed, cells, torch.zeros_like(cells)).sum(dim=0)
    return {'loss_sums': sums, 'observed_counts': observed.sum(dim=0)}
