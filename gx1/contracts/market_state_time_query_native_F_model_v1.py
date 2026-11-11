"""Research wrapper: unchanged S query head plus native-endpoint F head.

One causal feature encoding feeds both heads. Future targets are never passed to
forward. The existing v2 state model, checkpoint format and score stay intact.
"""
from __future__ import annotations

import hashlib
import json

import torch
from torch import nn

from gx1.contracts.market_state_future_displacement_candidate_v1 import contract as future_contract
from gx1.contracts.market_state_time_query_model_v2 import (
    FactorizedTimeQueryMarketStateModel, factorized_log_probabilities,
    factorized_time_query_model_contract, _clone_side_head,
)
from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K, TIMEFRAMES


def contract(*, ta_only: bool = False, dedicated_m5_side: bool = False,
             dedicated_non_m5_side: bool = False) -> dict:
    if type(ta_only) is not bool:
        raise ValueError('NATIVE_F_TA_MODE')
    if type(dedicated_m5_side) is not bool or (dedicated_m5_side and not ta_only):
        raise ValueError('NATIVE_F_DEDICATED_M5_SIDE_MODE')
    if type(dedicated_non_m5_side) is not bool or (
            dedicated_non_m5_side and not dedicated_m5_side):
        raise ValueError('NATIVE_F_DEDICATED_NON_M5_SIDE_MODE')
    value = {
        'schema_version': 'gx1_market_state_native_F_dual_model_candidate_v1',
        'role': 'research_only_no_fit_or_trade_authority',
        'S_model_contract_sha256': factorized_time_query_model_contract(
            ta_only=ta_only, dedicated_m5_side=dedicated_m5_side,
            dedicated_non_m5_side=dedicated_non_m5_side)['contract_sha256'],
        'F_target_contract_sha256': future_contract()['contract_sha256'],
        'encoder': 'one_existing_v2_causal_native_backbone_encode_features_call',
        'S': 'unchanged_v2_sampled_query_projection_and_state_head',
        'F': 'one_fixed_native_k_query_per_TF_from_same_hidden_and_projection_with_distinct_two_logit_head',
        'native_k': list(NATIVE_K),
        'F_primitive_logits': ['NET_ACTIVE', 'UP_GIVEN_NET_ACTIVE'],
        'S_and_F_targets_as_forward_inputs': False,
    }
    if dedicated_m5_side:
        value['dedicated_m5_side'] = True
        value['S'] = 'v2_sampled_query_projection_and_state_head_with_dedicated_M5_side'
        value['F'] = (
            'fixed_native_k_per_TF_shared_projection_distinct_two_logit_head'
            '_with_dedicated_M5_conditional_side_Linear_hidden_to_1')
        value['M5_side_initialization'] = 'clone_each_task_shared_side_row_preserving_CPU_and_active_CUDA_RNG'
    if dedicated_non_m5_side:
        value['dedicated_non_m5_side'] = True
        value['non_M5_side_timeframes'] = list(TIMEFRAMES[1:])
        value['S'] += '_and_independent_non_M5_conditional_side_heads'
        value['F'] += '_and_independent_non_M5_conditional_side_heads'
        value['non_M5_side_initialization'] = 'clone_each_task_shared_side_row_preserving_existing_M5_and_CPU_and_active_CUDA_RNG'
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return {**value, 'contract_sha256': hashlib.sha256(raw).hexdigest()}


class NativeFutureDualModel(nn.Module):
    """Keep the v2 S submodel exact while adding one small F decoder head."""

    def __init__(self, state_model: FactorizedTimeQueryMarketStateModel):
        super().__init__()
        if type(state_model) is not FactorizedTimeQueryMarketStateModel:
            raise TypeError('NATIVE_F_STATE_MODEL_OWNER')
        state_model._require_binding()
        self.state_model = state_model
        width = state_model._hidden_width
        device = state_model.query_head.weight.device
        self.future_head = nn.Linear(width, 2, device=device, dtype=torch.float32)
        if state_model._dedicated_m5_side:
            self.future_m5_side_head = _clone_side_head(self.future_head)
        if state_model._dedicated_non_m5_side:
            self.future_non_m5_side_heads = nn.ModuleDict({
                timeframe: _clone_side_head(self.future_head)
                for timeframe in TIMEFRAMES[1:]})
        self.register_buffer('native_F_objective_sha256', torch.tensor(
            list(bytes.fromhex(self.research_objective['contract_sha256'])),
            device=device, dtype=torch.uint8))

    @property
    def research_objective(self) -> dict:
        return contract(ta_only=self.state_model._ta_only,
                        dedicated_m5_side=self.state_model._dedicated_m5_side,
                        dedicated_non_m5_side=self.state_model._dedicated_non_m5_side)

    @property
    def backbone(self) -> nn.Module:
        return self.state_model.backbone

    def encode_features(self, **features: torch.Tensor) -> torch.Tensor:
        return self.state_model.encode_features(**features)

    def _require_binding(self) -> None:
        self.state_model._require_binding()
        device = self.state_model.query_head.weight.device
        width = self.state_model._hidden_width
        reference = torch.tensor(list(bytes.fromhex(self.research_objective['contract_sha256'])),
                                 dtype=torch.uint8, device=device)
        if not (self.native_F_objective_sha256.shape == reference.shape
                and self.native_F_objective_sha256.dtype == reference.dtype
                and self.native_F_objective_sha256.device == device
                and not self.native_F_objective_sha256.requires_grad
                and torch.equal(self.native_F_objective_sha256, reference)
                and hasattr(self, 'future_m5_side_head') == self.state_model._dedicated_m5_side
                and hasattr(self, 'future_non_m5_side_heads') == self.state_model._dedicated_non_m5_side):
            raise RuntimeError('NATIVE_F_HEAD_BINDING')
        heads = [(self.future_head, 2)]
        if self.state_model._dedicated_m5_side:
            heads.append((self.future_m5_side_head, 1))
        if self.state_model._dedicated_non_m5_side:
            if not (type(self.future_non_m5_side_heads) is nn.ModuleDict
                    and tuple(self.future_non_m5_side_heads) == tuple(TIMEFRAMES[1:])):
                raise RuntimeError("NATIVE_F_NON_M5_SIDE_HEAD_BINDING")
            heads.extend((head, 1) for head in self.future_non_m5_side_heads.values())
        for head, outputs in heads:
            if not (type(head) is nn.Linear and head.in_features == width
                    and head.out_features == outputs and head.weight.shape == (outputs, width)
                    and head.bias is not None and head.bias.shape == (outputs,)
                    and head.weight.dtype == head.bias.dtype == torch.float32
                    and head.weight.device == head.bias.device == device
                    and bool(torch.isfinite(head.weight).all())
                    and bool(torch.isfinite(head.bias).all())):
                raise RuntimeError('NATIVE_F_HEAD_BINDING')
        if self.state_model._dedicated_non_m5_side:
            storages = [parameter.untyped_storage().data_ptr()
                        for head, _ in heads
                        for parameter in (head.weight, head.bias)]
            if len(set(storages)) != len(storages):
                raise RuntimeError('NATIVE_F_NON_M5_SIDE_HEAD_ALIASES')

    def decode_native_F_raw(self, hidden: torch.Tensor) -> torch.Tensor:
        self._require_binding()
        if not (isinstance(hidden, torch.Tensor) and hidden.dtype == torch.float32
                and hidden.ndim == 2 and hidden.shape[0] > 0
                and hidden.shape[1] == self.state_model._hidden_width
                and hidden.device == self.future_head.weight.device
                and bool(torch.isfinite(hidden).all())):
            raise ValueError('NATIVE_F_HIDDEN')
        device = hidden.device
        batch = hidden.shape[0]
        source = hidden[:, None, :].expand(-1, len(TIMEFRAMES), -1)
        tf = self.state_model.query_tf_coordinates[None].expand(batch, -1, -1)
        native_fraction = torch.ones((batch, len(TIMEFRAMES), 1),
                                     dtype=torch.float32, device=device)
        representation = self.state_model.query_projection(
            torch.cat((source, tf, native_fraction), dim=-1))
        raw = self.future_head(representation)
        if self.state_model._dedicated_m5_side:
            raw = raw.clone()
            raw[:, 0, 1] = self.future_m5_side_head(representation[:, 0, :]).squeeze(-1)
        if self.state_model._dedicated_non_m5_side:
            for index, timeframe in enumerate(TIMEFRAMES[1:], start=1):
                raw[:, index, 1] = self.future_non_m5_side_heads[timeframe](
                    representation[:, index, :]).squeeze(-1)
        if not (raw.shape == (batch, len(TIMEFRAMES), 2)
                and bool(torch.isfinite(raw).all())):
            raise RuntimeError('NATIVE_F_LOGITS')
        return raw

    def forward(self, seq_x: torch.Tensor, snap_x: torch.Tensor, *,
                query_k: torch.Tensor, ctx_cat: torch.Tensor,
                ctx_cont: torch.Tensor, seq_m5: torch.Tensor,
                seq_m15: torch.Tensor, seq_h1: torch.Tensor,
                seq_h4: torch.Tensor, seq_d1: torch.Tensor,
                **gc_features: torch.Tensor) -> dict[str, torch.Tensor]:
        features = dict(seq_x=seq_x, snap_x=snap_x, ctx_cat=ctx_cat,
                        ctx_cont=ctx_cont, seq_m5=seq_m5, seq_m15=seq_m15,
                        seq_h1=seq_h1, seq_h4=seq_h4, seq_d1=seq_d1,
                        **gc_features)
        hidden = self.encode_features(**features)
        if not (isinstance(query_k, torch.Tensor) and query_k.dtype == torch.int64
                and query_k.shape == (hidden.shape[0], len(TIMEFRAMES))
                and query_k.device == hidden.device):
            raise ValueError('NATIVE_F_S_QUERY')
        state_raw = self.state_model.decode_raw_query_grid(
            hidden, query_k[:, None, :])[:, 0]
        future_raw = self.decode_native_F_raw(hidden)
        return {'state_activity_side_logits': state_raw,
                'state_log_probs': factorized_log_probabilities(state_raw),
                'future_activity_side_logits': future_raw,
                'future_log_probs': factorized_log_probabilities(future_raw)}
