"""M5/MTF TA S/F and M5 structure readout over one native encode.

The eight TA specialist families and closed M5/M15/H1/H4/D1 windows feed
the S/F heads directly. M1 trade management belongs to a later phase.
Targets enter only loss and scoring. This contract grants no fit authority.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from gx1.contracts.market_state_native_m5_targets_v1 import TIMEFRAMES, contract as native_target_contract
from gx1.features.entry_specialist_feature_groups_v1 import (
    FULL_TA_ENTRY_FEATURE_KEYS,
)
from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import EntryV10CtxHybridTransformer


STRUCTURE_FIELDS = ('next_high_hits_frozen_high', 'next_low_hits_frozen_low')


def _need(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError('MARKET_STATE_TA_INTEGRATED_' + reason)


def contract(dual: NativeFutureDualModel) -> dict:
    from gx1.contracts.market_state_time_query_native_F_model_v1 import NativeFutureDualModel
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
        from gx1.contracts.market_state_time_query_model_v2 import factorized_log_probabilities
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


def _seal_native(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return {**value, 'contract_sha256': hashlib.sha256(raw).hexdigest()}


def require_native_bar_input_binding(value):
    if isinstance(value,Mapping) and value.get('schema_version')=='gx1_native_m5_full_gc_bar_direction_inputs_v1':
        from gx1.contracts.market_state_native_m5_gc_feature_inputs_v1 import require_native_m5_gc_input_binding
        return require_native_m5_gc_input_binding(value)
    from gx1.contracts.market_state_native_m5_feature_inputs_v1 import require_native_m5_input_binding
    return require_native_m5_input_binding(value)


def _native_bar_feature_keys(binding):
    if binding['schema_version']=='gx1_native_m5_full_gc_bar_direction_inputs_v1':
        from gx1.features.entry_specialist_feature_groups_v1 import NATIVE_M5_FULL_GC_FEATURE_KEYS
        return NATIVE_M5_FULL_GC_FEATURE_KEYS
    return FULL_TA_ENTRY_FEATURE_KEYS


def native_bar_prior_initialization_contract(train_prior,*,target_contract=None):
    from gx1.contracts.market_state_ta_integrated_objective_v1 import require_native_bar_train_prior
    prior=require_native_bar_train_prior(train_prior,target_contract=target_contract)
    support={task:{tf:{'predictive_train_cells':cell['count'],
        'train_observations_available':cell['train_observations_available'],
        'role':('predictive_TRAIN_prior' if cell['train_observations_available']
                else 'UNFITTED_uniform_pseudocount_reference_no_predictive_TRAIN_support')}
        for tf,cell in prior['per_task'][task].items()} for task in ('open','close')}
    value={'schema_version':'gx1_native_m5_predictive_train_prior_head_initialization_v1',
        'prior_sha256':prior['contract_sha256'],'prior_contract_sha256':prior['prior_contract_sha256'],
        'target_contract_sha256':prior['target_contract_sha256'],
        'objective_contract_sha256':prior['objective_contract_sha256'],
        'class_order':[-1,0,1],'timeframes':list(TIMEFRAMES),
        'method':'zero_final_Linear_weights_plus_FP32_log_validated_TRAIN_probabilities_bias',
        'per_head_support':support,'backbone_and_heads_trainable':True,
        'first_step_backbone_data_gradient':'zero_until_final_head_weights_update',
        'checkpoint_load_reinitializes_heads':False}
    if target_contract is not None:
        value.update(schema_version='gx1_native_m5_noise_margin_train_prior_head_initialization_v1',
            class_names=prior['class_names'],class_names_sha256=prior['class_names_sha256'])
    return _seal_native(value)


M5_CLOSE_ONLY_BACKBONE_GRADIENT_POLICY = 'm5_close_only_backbone_v1'


def native_bar_backbone_gradient_policy_contract(policy):
    _need(type(policy) is str and policy == M5_CLOSE_ONLY_BACKBONE_GRADIENT_POLICY,
          'NATIVE_BACKBONE_GRADIENT_POLICY')
    return _seal_native({
        'schema_version': 'gx1_native_m5_close_only_backbone_gradient_policy_v1',
        'name': policy, 'attached_heads': ['close.M5'],
        'detached_heads': ['open.' + tf for tf in TIMEFRAMES]
            + ['close.' + tf for tf in TIMEFRAMES if tf != 'M5'],
        'method': 'detach_only_at_auxiliary_Linear_head_inputs',
        'all_head_parameters_trainable': True,
        'loss_labels_masks_and_total_denominator': 'unchanged',
        'checkpoint_load_reinitializes_heads': False,
    })


def _require_backbone_gradient_policy(policy, binding, train_prior):
    if policy is not None:
        native_bar_backbone_gradient_policy_contract(policy)
        _need(len(_native_bar_feature_keys(binding)) == 9 and train_prior is not None,
              'NATIVE_BACKBONE_GRADIENT_POLICY_TA_PRIOR_REQUIRED')


def _native_bar_target_spec(target_contract,binding,train_prior):
    if target_contract is None:return None
    from gx1.contracts.market_state_native_m5_targets_v1 import require_noise_margin_dataset_contract
    spec=require_noise_margin_dataset_contract(target_contract)
    _need(len(_native_bar_feature_keys(binding))==9 and train_prior is not None,
          'NATIVE_NOISE_TARGET_TA_PRIOR_REQUIRED')
    return spec


def native_bar_model_contract(input_binding,*,train_prior=None,backbone_gradient_policy=None,target_contract=None):
    binding = require_native_bar_input_binding(input_binding)
    return _native_bar_model_contract_from_binding(binding,train_prior=train_prior,
        backbone_gradient_policy=backbone_gradient_policy,target_contract=target_contract)


def _native_bar_model_contract_from_binding(binding,*,train_prior=None,backbone_gradient_policy=None,target_contract=None):
    # Only the public validator or one still-equal, freshly validated local
    # binding may enter here. No proof survives its current validation call.
    _require_backbone_gradient_policy(backbone_gradient_policy, binding, train_prior)
    target_spec=_native_bar_target_spec(target_contract,binding,train_prior)
    value={
        'schema_version': 'gx1_native_m5_bar_direction_model_v1',
        'role': 'research_forecast_no_fit_entry_or_trade_authority',
        'target_contract_sha256': native_target_contract()['contract_sha256'],
        'input_binding_sha256': binding['contract_sha256'],
        'timeframes': list(TIMEFRAMES), 'class_order': [-1, 0, 1],
        'comparisons': ['open_vs_previous_observed_open', 'close_vs_previous_observed_close'],
        'backbone': 'one_existing_native_eight_family_macro_five_post_cross_TF_encode',
        'heads': 'independent_fresh_Linear_hidden_to_3_per_comparison_and_timeframe',
        'forward_inputs': sorted(_native_bar_feature_keys(binding)),
        'target_metadata_as_forward_input': False,
        'initialization': 'fresh_random_no_checkpoint',
    }
    if train_prior is not None:
        _need(len(_native_bar_feature_keys(binding))==9,'NATIVE_PRIOR_TA_ONLY')
        value['schema_version']='gx1_native_m5_bar_direction_prior_initialized_model_v1'
        value['initialization']=native_bar_prior_initialization_contract(train_prior,target_contract=target_spec)
    if backbone_gradient_policy is not None:
        value['schema_version']='gx1_native_m5_bar_direction_m5_close_backbone_model_v1'
        value['backbone_gradient_policy']=native_bar_backbone_gradient_policy_contract(backbone_gradient_policy)
    if target_spec is not None:
        from gx1.contracts.market_state_ta_integrated_objective_v1 import native_bar_objective_contract
        objective=native_bar_objective_contract(target_contract=target_spec)
        value.update(schema_version='gx1_native_m5_bar_noise_margin_model_v1',target_contract=target_spec,
            target_contract_sha256=target_spec['contract_sha256'],
            raw_target_contract_sha256=target_spec['raw_target_contract_sha256'],
            objective_contract_sha256=objective['contract_sha256'],
            class_names=objective['class_names'],class_names_sha256=objective['class_names_sha256'])
    return _seal_native(value)


class NativeM5BarDirectionModel(nn.Module):
    """Two independent three-class bar readouts over each shared native TF token."""

    def __init__(self, backbone, *, input_binding,train_prior=None,backbone_gradient_policy=None,target_contract=None):
        super().__init__()
        self.input_binding = require_native_bar_input_binding(input_binding)
        _require_backbone_gradient_policy(backbone_gradient_policy, self.input_binding, train_prior)
        self._backbone_gradient_policy = backbone_gradient_policy
        self._target_contract=_native_bar_target_spec(target_contract,self.input_binding,train_prior)
        self._train_prior=None
        if train_prior is not None:
            _need(len(_native_bar_feature_keys(self.input_binding))==9,'NATIVE_PRIOR_TA_ONLY')
            from gx1.contracts.market_state_ta_integrated_objective_v1 import require_native_bar_train_prior
            self._train_prior=require_native_bar_train_prior(train_prior,target_contract=self._target_contract)
        _need(type(backbone) is EntryV10CtxHybridTransformer, 'NATIVE_BACKBONE_TYPE')
        self.backbone = backbone
        if self._train_prior is not None:
            prior_population=self._train_prior['fit_population']
            normalization_population=backbone._input_normalization_contract['fit_population']
            _need(prior_population['cutoff_ns']==normalization_population['cutoff_ns']
                and normalization_population['first_issue_ns']<=prior_population['first_issue_ns']
                and prior_population['last_issue_ns']<=normalization_population['last_issue_ns']
                and prior_population['issue_rows']<=normalization_population['issue_rows'],
                'NATIVE_PRIOR_TRAIN_POPULATION')
        width = int(backbone.cfg.d_model)
        device = backbone.entry_q_joint_in.weight.device
        self.open_heads = nn.ModuleDict({tf: nn.Linear(width, 3, device=device, dtype=torch.float32) for tf in TIMEFRAMES})
        self.close_heads = nn.ModuleDict({tf: nn.Linear(width, 3, device=device, dtype=torch.float32) for tf in TIMEFRAMES})
        if self._train_prior is not None:
            from gx1.contracts.market_state_ta_integrated_objective_v1 import native_bar_train_prior_logits
            logits=native_bar_train_prior_logits(self._train_prior,rows=1,target_contract=self._target_contract)
            with torch.no_grad():
                for task,heads in (('open',self.open_heads),('close',self.close_heads)):
                    for j,tf in enumerate(TIMEFRAMES):
                        heads[tf].weight.zero_()
                        heads[tf].bias.copy_(torch.from_numpy(logits[task+'_logits'][0,j]).to(device))
            for name,digest in (('native_bar_initialization_sha256',native_bar_prior_initialization_contract(self._train_prior,target_contract=self._target_contract)['contract_sha256']),
                                ('native_bar_train_prior_sha256',self._train_prior['contract_sha256'])):
                self.register_buffer(name,torch.tensor(list(bytes.fromhex(digest)),device=device,dtype=torch.uint8))
        if self._backbone_gradient_policy is not None:
            digest = native_bar_backbone_gradient_policy_contract(self._backbone_gradient_policy)['contract_sha256']
            self.register_buffer('native_bar_backbone_gradient_policy_sha256', torch.tensor(
                list(bytes.fromhex(digest)), device=device, dtype=torch.uint8))
        if self._target_contract is not None:
            self.register_buffer('native_bar_target_contract_sha256',torch.tensor(
                list(bytes.fromhex(self._target_contract['contract_sha256'])),device=device,dtype=torch.uint8))
        self.register_buffer('native_bar_objective_sha256', torch.tensor(
            list(bytes.fromhex(self.research_objective['contract_sha256'])),
            device=device, dtype=torch.uint8))
        self._require_binding()
        if self._train_prior is not None:self.require_prior_initialization_state()

    @property
    def research_objective(self):
        return native_bar_model_contract(self.input_binding,train_prior=self._train_prior,
            backbone_gradient_policy=self._backbone_gradient_policy,target_contract=self._target_contract)

    def require_prior_initialization_state(self):
        # Fresh-state assertion only; checkpoint restore never calls this.
        from gx1.contracts.market_state_ta_integrated_objective_v1 import native_bar_train_prior_logits
        _need(self._train_prior is not None,'NATIVE_PRIOR_INITIALIZATION_REQUIRED')
        logits=native_bar_train_prior_logits(self._train_prior,rows=1,target_contract=self._target_contract)
        for task,heads in (('open',self.open_heads),('close',self.close_heads)):
            for j,tf in enumerate(TIMEFRAMES):
                head=heads[tf]
                expected=torch.from_numpy(logits[task+'_logits'][0,j]).to(head.bias.device)
                _need(torch.count_nonzero(head.weight).item()==0 and torch.equal(head.bias,expected),
                      'NATIVE_FRESH_PRIOR_HEAD_STATE')
        return self.research_objective['initialization']

    def _require_binding(self):
        binding = require_native_bar_input_binding(self.input_binding)
        _need(self.backbone.entry_execution_profile == binding['execution_profile']
              and self._gc_binding_matches(binding)
              and self.backbone._native_bar_direction_only is True
              and self.backbone.cfg.dynamic_m5_query_minutes is None
              and not hasattr(self.backbone, 'dynamic_m5_query_horizons')
              and self.backbone._input_normalization_contract.get('schema_version')
                  == ('gx1_native_m5_full_gc_input_normalization_v1' if len(_native_bar_feature_keys(binding))==14 else 'gx1_native_m5_input_normalization_v2')
              and self.backbone._input_normalization_contract.get('source_axes_sha256')
                  == binding['source_axes_sha256']
              and self.backbone._input_normalization_contract.get('feature_provider_contract_sha256')
                  == binding['feature_provider_contract_sha256']
              and bytes(self.backbone.input_norm_contract_sha256.detach().cpu().tolist()).hex()
                  == binding['input_normalization_sha256'], 'NATIVE_BACKBONE_IDENTITY')
        if self._train_prior is not None:
            for name,digest in (('native_bar_initialization_sha256',native_bar_prior_initialization_contract(self._train_prior,target_contract=self._target_contract)['contract_sha256']),
                                ('native_bar_train_prior_sha256',self._train_prior['contract_sha256'])):
                actual=getattr(self,name)
                _need(torch.equal(actual.detach().cpu(),torch.tensor(list(bytes.fromhex(digest)),dtype=torch.uint8)),
                      'NATIVE_PRIOR_STATE_BINDING')
        if self._backbone_gradient_policy is not None:
            digest = native_bar_backbone_gradient_policy_contract(self._backbone_gradient_policy)['contract_sha256']
            expected = torch.tensor(list(bytes.fromhex(digest)), dtype=torch.uint8)
            _need(torch.equal(self.native_bar_backbone_gradient_policy_sha256.detach().cpu(), expected),
                  'NATIVE_BACKBONE_GRADIENT_POLICY_STATE_BINDING')
        if self._target_contract is not None:
            expected=torch.tensor(list(bytes.fromhex(self._target_contract['contract_sha256'])),dtype=torch.uint8)
            _need(torch.equal(self.native_bar_target_contract_sha256.detach().cpu(),expected),
                  'NATIVE_NOISE_TARGET_STATE_BINDING')
        self.backbone.require_input_normalization_state()
        # The validator returned a deep JSON copy. Detect mutation before
        # reusing that local proof instead of recompiling the same provider.
        _need(self.input_binding == binding, 'NATIVE_INPUT_BINDING_MUTATED_DURING_VALIDATION')
        objective = _native_bar_model_contract_from_binding(binding,train_prior=self._train_prior,
            backbone_gradient_policy=self._backbone_gradient_policy,target_contract=self._target_contract)
        device = self.backbone.entry_q_joint_in.weight.device
        expected = torch.tensor(list(bytes.fromhex(objective['contract_sha256'])),
                                device=device, dtype=torch.uint8)
        _need(torch.equal(self.native_bar_objective_sha256, expected), 'NATIVE_MODEL_BINDING')
        for heads in (self.open_heads, self.close_heads):
            _need(type(heads) is nn.ModuleDict and tuple(heads) == tuple(TIMEFRAMES),
                  'NATIVE_HEADS')
            for head in heads.values():
                _need(type(head) is nn.Linear and head.in_features == self.backbone.cfg.d_model
                      and head.out_features == 3
                      and all(p.device == device and p.dtype == torch.float32
                              and bool(torch.isfinite(p).all()) for p in head.parameters()),
                      'NATIVE_HEAD_BINDING')

    def _gc_binding_matches(self,binding):
        history = self.backbone.gc_mtf_history
        if len(_native_bar_feature_keys(binding))==9:
            return history is None
        if not isinstance(history,Mapping):
            return False
        from gx1.contracts.market_state_native_m5_gc_feature_inputs_v1 import native_m5_gc_feature_provider_contract
        provider = native_m5_gc_feature_provider_contract(history)
        return (provider['contract_sha256']==binding['feature_provider_contract_sha256']
            and history['layout']['schema_version']=='gx1_native_m5_gc_mtf_history_layout_v1'
            and history['layout']['contract_sha256']==binding['gc_layout_sha256']
            and history['normalization']['contract_sha256']==binding['gc_normalization_sha256']
            and history['normalization']['source_binding_sha256']==binding['native_gc_source_binding_sha256']
            and history==self.backbone._input_normalization_contract.get('gc_mtf_history'))

    def load_state_dict(self, state_dict, strict=True, assign=False):
        _need(strict is True, 'NATIVE_STRICT_CHECKPOINT_REQUIRED')
        names = ['native_bar_objective_sha256','backbone.input_norm_contract_sha256']
        if self._train_prior is not None:
            names += ['native_bar_initialization_sha256','native_bar_train_prior_sha256']
        if self._backbone_gradient_policy is not None:
            names += ['native_bar_backbone_gradient_policy_sha256']
        if self._target_contract is not None:
            names += ['native_bar_target_contract_sha256']
        if len(_native_bar_feature_keys(self.input_binding))==14:
            names += ['backbone.gc_mtf_layout_sha256','backbone.gc_mtf_normalization_sha256','backbone.entry_execution_profile_contract_sha256']
        for name in names:
            expected = self.state_dict()[name]
            actual = state_dict.get(name)
            _need(isinstance(actual, torch.Tensor)
                  and torch.equal(actual.detach().cpu(), expected.detach().cpu()),
                  'NATIVE_CHECKPOINT_IDENTITY')
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self._require_binding()
        return result

    def forward(self, features):
        self._require_binding()
        _need(isinstance(features, Mapping) and set(features) == _native_bar_feature_keys(self.input_binding),
              'NATIVE_FEATURE_KEYS')
        _need(all(isinstance(value, torch.Tensor) for value in features.values()),
              'NATIVE_FEATURE_TENSORS')
        output = self.backbone(**features, research_representation_only=True,
                               research_timeframe_representations=True)
        hidden = output['entry_q_timeframe_hidden']
        _need(hidden.dtype == torch.float32 and hidden.ndim == 3
              and hidden.shape[0] == features['seq_x'].shape[0] > 0
              and hidden.shape[1:] == (len(TIMEFRAMES), self.backbone.cfg.d_model)
              and bool(torch.isfinite(hidden).all()), 'NATIVE_HIDDEN')
        auxiliary_hidden = hidden if self._backbone_gradient_policy is None else hidden.detach()
        return {
            'open_logits': torch.stack([self.open_heads[tf](auxiliary_hidden[:, j])
                                        for j, tf in enumerate(TIMEFRAMES)], dim=1),
            'close_logits': torch.stack([self.close_heads[tf](hidden[:, j] if tf == 'M5' else auxiliary_hidden[:, j])
                                         for j, tf in enumerate(TIMEFRAMES)], dim=1),
        }


def native_bar_backbone_geometry():
    from gx1.contracts.entry_exit_feature_base_v1 import ENTRY_FEATURE_SEQUENCE_BARS
    from gx1.contracts.entry_exit_production_architecture_v1 import PRODUCTION_MTF_PER_TF_WINDOW_BARS
    return {'local': ENTRY_FEATURE_SEQUENCE_BARS, **dict(PRODUCTION_MTF_PER_TF_WINDOW_BARS)}


def build_native_bar_direction_model(*, input_normalization, input_binding, architecture, device, gc_mtf_history=None,train_prior=None,backbone_gradient_policy=None,target_contract=None,return_backbone_only=False):
    """Build fresh native parameters from explicit architecture and native statistics."""
    _need(type(return_backbone_only) is bool, 'NATIVE_BACKBONE_ONLY_FLAG')
    _need(not return_backbone_only or (train_prior is None and target_contract is None
          and backbone_gradient_policy is None), 'NATIVE_BACKBONE_ONLY_RAW_ARGUMENTS')
    from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
    from gx1.features import entry_specialist_feature_groups_v1 as owners
    binding = require_native_bar_input_binding(input_binding)
    _need(train_prior is None or len(_native_bar_feature_keys(binding))==9,'NATIVE_PRIOR_TA_ONLY')
    target_spec=_native_bar_target_spec(target_contract,binding,train_prior)
    _require_backbone_gradient_policy(backbone_gradient_policy, binding, train_prior)
    if len(_native_bar_feature_keys(binding))==14:
        _need(gc_mtf_history is not None and input_normalization.get('gc_mtf_history')==gc_mtf_history,'NATIVE_GC_FACTORY_HISTORY')
        profile = owners.compile_native_m5_full_gc_entry_execution_profile(gc_mtf_history)
    else:
        _need(gc_mtf_history is None,'NATIVE_TA_GC_FACTORY_FORBIDDEN')
        profile = owners.compile_full_ta_entry_execution_profile()
    expected = {'dropout', 'multi_tf_num_layers', 'multi_tf_scale',
                'specialist_num_layers', 'specialist_fusion_scale', 'cross_family_fusion_scale'}
    _need(type(architecture) is dict and set(architecture) == expected, 'NATIVE_ARCHITECTURE')
    for key, value in architecture.items():
        if key.endswith('num_layers'):
            _need(type(value) is int and value > 0, 'NATIVE_ARCHITECTURE')
        else:
            _need(type(value) in (int, float) and math.isfinite(value)
                  and (0 <= value < 1 if key == 'dropout' else value > 0), 'NATIVE_ARCHITECTURE')
    context = owners.model_native_context_specialist_routing_contract()
    if gc_mtf_history is None:
        mapping = {old:new for new,old in enumerate(profile['source_ctx_cont_indices'])}
        context = {key:{family:[mapping[i] for i in indices if i in mapping]
                   for family,indices in context[key].items()} for key in ('ctx_cont_indices','ctx_cont_nominal_indices')}
    else:
        from gx1.contracts.entry_model_native_signal_v1 import MODEL_NATIVE_CTX_CONT_FIELDS
        lookup = {name:i for i,name in enumerate(profile['physical_field_names']['ctx_cont'])}
        context = {'ctx_cont_indices':{family:list(indices) for family,indices in profile['ctx_cont_indices_by_specialist'].items()},
            'ctx_cont_nominal_indices':{family:[lookup[MODEL_NATIVE_CTX_CONT_FIELDS[i]] for i in indices if MODEL_NATIVE_CTX_CONT_FIELDS[i] in lookup]
                for family,indices in context['ctx_cont_nominal_indices'].items()}}
    cat_context = owners.model_native_context_specialist_routing_contract()['ctx_cat_indices']
    widths, geometry = profile['physical_widths'], native_bar_backbone_geometry()
    alias_pairs = sorted(zip(profile['temporal_alias_signal_indices'],
                             profile['temporal_alias_ctx_cont_indices']))
    kwargs = dict(architecture)
    kwargs.update(
        seq_input_dim=widths['signal'], snap_input_dim=widths['signal'],
        seq_len=geometry['local'], ctx_cont_dim=widths['ctx_cont'], ctx_cat_dim=widths['ctx_cat'],
        specialist_input_indices={family: list(indices) for family, indices
                                  in profile['local_signal_indices_by_specialist'].items()},
        specialist_ctx_cont_indices=context['ctx_cont_indices'],
        specialist_ctx_cont_nominal_indices=context['ctx_cont_nominal_indices'],
        specialist_ctx_cat_indices=cat_context,
        multi_tf_specialist_input_indices={family: list(indices) for family, indices
                                           in profile['mtf_indices_by_specialist'].items()},
        temporal_alias_signal_indices=[signal for signal, _ in alias_pairs],
        temporal_alias_ctx_cont_indices=[context for _, context in alias_pairs],
        input_normalization=input_normalization,
        entry_execution_profile=binding['execution_profile'], gc_mtf_history=gc_mtf_history,
        entry_training_phase=ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE,
        dynamic_m5_query_minutes=None)
    for tf in TIMEFRAMES:
        kwargs[tf.lower() + '_seq_dim'] = widths['mtf']
        kwargs[tf.lower() + '_seq_len'] = geometry[tf]
    backbone = EntryV10CtxHybridTransformer(**kwargs).to(device)
    if return_backbone_only:
        return backbone
    return NativeM5BarDirectionModel(backbone, input_binding=binding,train_prior=train_prior,
        backbone_gradient_policy=backbone_gradient_policy,target_contract=target_spec).to(device)


def native_future_state_readout_contract(*, d_model, target_contract, train_prior):
    """Bind a hidden-only component; no backbone, label recipe or fit authority."""
    from gx1.contracts.market_state_native_m5_targets_v1 import require_fully_future_state_contract
    from gx1.contracts.market_state_ta_integrated_objective_v1 import (
        native_future_state_objective_contract, require_native_future_state_train_prior,
    )
    _need(type(d_model) is int and d_model > 0, 'FUTURE_READOUT_WIDTH')
    target = require_fully_future_state_contract(target_contract)
    prior = require_native_future_state_train_prior(train_prior, target_contract=target)
    value = {
        'schema_version': 'gx1_native_future_state_readout_component_v1',
        'role': 'hidden_only_component_not_full_native_model_or_production_fit',
        'd_model': d_model, 'timeframes': list(TIMEFRAMES),
        'hidden': 'float32_N_by_5_by_explicit_d_model',
        'heads': 'independent_per_TF_Linear_d_model_to_1_for_activity_and_conditional_side',
        'output': ['activity_logits', 'side_logits'],
        'target_contract_sha256': target['contract_sha256'],
        'objective_contract_sha256': native_future_state_objective_contract(
            target_contract=target)['contract_sha256'],
        'train_prior_sha256': prior['contract_sha256'],
        'initial_weights': 'exact_zero',
        'initial_biases': 'exact_frozen_TRAIN_binary_prior_float32_logits',
        'backbone_routing_and_task_weights': 'not_defined_by_this_component',
        'feature_backbone_forward_qualified': False,
        'production_training_admitted': False,
    }
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return {**value, 'contract_sha256': hashlib.sha256(raw).hexdigest()}


class NativeFutureStateReadout(nn.Module):
    """Actual two-coordinate heads on supplied TF hidden states, without a backbone."""

    def __init__(self, *, d_model, target_contract, train_prior):
        super().__init__()
        from gx1.contracts.market_state_ta_integrated_objective_v1 import native_future_state_train_prior_logits
        bound = native_future_state_readout_contract(d_model=d_model,
            target_contract=target_contract, train_prior=train_prior)
        self.d_model = d_model
        self._readout_contract = bound
        self._identity_hex = {
            'future_state_target_sha256': bound['target_contract_sha256'],
            'future_state_objective_sha256': bound['objective_contract_sha256'],
            'future_state_prior_sha256': bound['train_prior_sha256'],
            'future_state_readout_sha256': bound['contract_sha256'],
        }
        # Linear's temporary initialization is discarded; preserve caller RNG.
        with torch.random.fork_rng(devices=[]):
            self.activity_heads = nn.ModuleDict({tf: nn.Linear(d_model, 1,
                device='cpu', dtype=torch.float32) for tf in TIMEFRAMES})
            self.side_heads = nn.ModuleDict({tf: nn.Linear(d_model, 1,
                device='cpu', dtype=torch.float32) for tf in TIMEFRAMES})
        initial = native_future_state_train_prior_logits(train_prior, rows=1,
            target_contract=target_contract)
        with torch.no_grad():
            for task, heads in (('activity', self.activity_heads), ('side', self.side_heads)):
                for j, tf in enumerate(TIMEFRAMES):
                    heads[tf].weight.zero_()
                    heads[tf].bias.copy_(initial[task + '_logits'][0, j:j + 1])
        for name, digest in self._identity_hex.items():
            self.register_buffer(name, torch.tensor(list(bytes.fromhex(digest)),
                device='cpu', dtype=torch.uint8))
        self._require_binding()

    @property
    def readout_contract(self):
        return json.loads(json.dumps(self._readout_contract))

    def _require_binding(self):
        for name, digest in self._identity_hex.items():
            value = getattr(self, name)
            _need(isinstance(value, torch.Tensor) and value.dtype == torch.uint8
                and value.shape == (32,) and value.device.type != 'meta'
                and torch.equal(value.detach().cpu(), torch.tensor(
                    list(bytes.fromhex(digest)), device='cpu', dtype=torch.uint8)), 'FUTURE_READOUT_IDENTITY')

    def _require_state(self, state_dict):
        self._require_binding()
        expected = self.state_dict()
        _need(isinstance(state_dict, Mapping) and set(state_dict) == set(expected),
              'FUTURE_READOUT_STATE_KEYS')
        for name, value in state_dict.items():
            _need(isinstance(value, torch.Tensor) and value.shape == expected[name].shape
                and value.dtype == expected[name].dtype and value.device.type != 'meta'
                and bool(torch.isfinite(value).all()), 'FUTURE_READOUT_STATE_TENSOR')
        for name, digest in self._identity_hex.items():
            _need(torch.equal(state_dict[name].detach().cpu(), torch.tensor(
                list(bytes.fromhex(digest)), device='cpu', dtype=torch.uint8)), 'FUTURE_READOUT_STATE_IDENTITY')
    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Reject identity/header defects before copying any component parameter."""
        _need(strict is True and type(assign) is bool, 'FUTURE_READOUT_STRICT_STATE')
        _need(isinstance(state_dict, Mapping), 'FUTURE_READOUT_STATE_KEYS')
        return super().load_state_dict(state_dict, strict=True, assign=assign)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Also runs during recursive parent restoration, before this component's
        # buffers or any child Linear parameters can be copied.
        self._require_state({key[len(prefix):]: value for key, value in state_dict.items()
                             if key.startswith(prefix)})
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def forward(self, hidden):
        parameter = self.activity_heads[TIMEFRAMES[0]].weight
        _need(isinstance(hidden, torch.Tensor) and hidden.dtype == torch.float32
            and hidden.ndim == 3 and hidden.shape[0] > 0
            and hidden.shape[1:] == (len(TIMEFRAMES), self.d_model), 'FUTURE_READOUT_HIDDEN')
        _need(hidden.device == parameter.device, 'FUTURE_READOUT_HIDDEN_DEVICE')
        self._require_binding()
        _need(bool(torch.isfinite(hidden).all()), 'FUTURE_READOUT_HIDDEN_FINITE')
        _need(all(p.dtype == torch.float32 and p.device == hidden.device
                  and bool(torch.isfinite(p).all()) for p in self.parameters()),
              'FUTURE_READOUT_PARAMETERS')
        return {task + '_logits': torch.stack([
            heads[tf](hidden[:, j]).squeeze(-1) for j, tf in enumerate(TIMEFRAMES)], dim=1)
            for task, heads in (('activity', self.activity_heads), ('side', self.side_heads))}


def native_future_state_model_contract(input_binding, *, architecture, train_prior, target_contract):
    """Full native forward identity; no gradient-routing or fit admission."""
    from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import CtxModelConfig
    from gx1.contracts.market_state_native_m5_targets_v1 import require_fully_future_state_contract
    from gx1.contracts.market_state_ta_integrated_objective_v1 import require_native_future_state_train_prior
    binding = require_native_bar_input_binding(input_binding)
    _need(len(_native_bar_feature_keys(binding)) == 9, 'FUTURE_MODEL_TA_ONLY')
    target = require_fully_future_state_contract(target_contract)
    prior = require_native_future_state_train_prior(train_prior, target_contract=target)
    keys = {'dropout', 'multi_tf_num_layers', 'multi_tf_scale',
            'specialist_num_layers', 'specialist_fusion_scale', 'cross_family_fusion_scale'}
    _need(type(architecture) is dict and set(architecture) == keys, 'FUTURE_MODEL_ARCHITECTURE')
    for key, value in architecture.items():
        if key.endswith('num_layers'):
            _need(type(value) is int and value > 0, 'FUTURE_MODEL_ARCHITECTURE')
        else:
            _need(type(value) in (int, float) and math.isfinite(value)
                  and (0 <= value < 1 if key == 'dropout' else value > 0), 'FUTURE_MODEL_ARCHITECTURE')
    # The existing factory leaves this field to its sole backbone owner.
    width = CtxModelConfig.__dataclass_fields__['d_model'].default
    readout = native_future_state_readout_contract(d_model=width,
        target_contract=target, train_prior=prior)
    return _seal_native({
        'schema_version': 'gx1_native_future_state_forward_model_v1',
        'role': 'native_forward_qualification_only_no_fit_or_trade_authority',
        'input_binding_sha256': binding['contract_sha256'],
        'input_normalization_sha256': binding['input_normalization_sha256'],
        'target_contract_sha256': target['contract_sha256'],
        'objective_contract_sha256': readout['objective_contract_sha256'],
        'train_prior_sha256': prior['contract_sha256'], 'readout': readout,
        'timeframes': list(TIMEFRAMES), 'forward_inputs': sorted(_native_bar_feature_keys(binding)),
        'architecture': json.loads(json.dumps(architecture)),
        'backbone_geometry': native_bar_backbone_geometry(),
        'backbone': 'same_fresh_native_eight_family_macro_five_post_cross_TF_representation',
        'd_model_owner': 'CtxModelConfig.d_model_used_by_existing_native_factory',
        'outputs': ['activity_logits', 'side_logits'],
        'raw_open_close_heads_constructed': False,
        'target_metadata_as_model_input': False,
        'nonparameter_checkpoint_state': 'immutable_exact_match_before_any_parameter_copy',
        'gradient_routing_and_task_weights': 'unqualified_requires_separate_admission',
        'production_training_admitted': False,
    })


class NativeFutureStateModel(nn.Module):
    """One genuine native encode followed by the independent state readout."""

    def __init__(self, backbone, *, input_binding, architecture, train_prior, target_contract):
        super().__init__()
        from gx1.contracts.market_state_native_m5_targets_v1 import require_fully_future_state_contract
        from gx1.contracts.market_state_ta_integrated_objective_v1 import require_native_future_state_train_prior
        _need(type(backbone) is EntryV10CtxHybridTransformer, 'FUTURE_MODEL_BACKBONE_TYPE')
        self.input_binding = require_native_bar_input_binding(input_binding)
        self._architecture = json.loads(json.dumps(architecture))
        self._target_contract = require_fully_future_state_contract(target_contract)
        self._train_prior = require_native_future_state_train_prior(train_prior,
            target_contract=self._target_contract)
        self.backbone = backbone
        spec = self.research_objective
        _need(backbone.cfg.d_model == spec['readout']['d_model'], 'FUTURE_MODEL_BACKBONE_WIDTH')
        self.readout = NativeFutureStateReadout(d_model=backbone.cfg.d_model,
            target_contract=self._target_contract, train_prior=self._train_prior).to(
                backbone.entry_q_joint_in.weight.device)
        self.register_buffer('future_state_model_sha256', torch.tensor(
            list(bytes.fromhex(spec['contract_sha256'])), dtype=torch.uint8,
            device=backbone.entry_q_joint_in.weight.device))
        self._require_binding()

    @property
    def research_objective(self):
        return native_future_state_model_contract(self.input_binding,
            architecture=self._architecture, train_prior=self._train_prior,
            target_contract=self._target_contract)

    def _require_binding(self):
        binding = require_native_bar_input_binding(self.input_binding)
        body = self.backbone
        norm = body._input_normalization_contract
        _need(len(_native_bar_feature_keys(binding)) == 9
            and body.entry_execution_profile == binding['execution_profile']
            and body.gc_mtf_history is None and body._native_bar_direction_only is True
            and body.cfg.dynamic_m5_query_minutes is None
            and not hasattr(body, 'dynamic_m5_query_horizons')
            and norm.get('schema_version') == 'gx1_native_m5_input_normalization_v2'
            and norm.get('source_axes_sha256') == binding['source_axes_sha256']
            and norm.get('feature_provider_contract_sha256') == binding['feature_provider_contract_sha256']
            and bytes(body.input_norm_contract_sha256.detach().cpu().tolist()).hex()
                == binding['input_normalization_sha256'], 'FUTURE_MODEL_BACKBONE_BINDING')
        _need(all(getattr(body.cfg, key) == value for key, value in self._architecture.items()),
              'FUTURE_MODEL_ARCHITECTURE_BINDING')
        body.require_input_normalization_state()
        _need(binding == self.input_binding, 'FUTURE_MODEL_INPUT_BINDING_MUTATED')
        pop, fitted = self._train_prior['fit_population'], norm['fit_population']
        _need(all(pop[key] == fitted[key] for key in
              ('cutoff_ns', 'first_issue_ns', 'last_issue_ns', 'issue_rows'))
            and pop['issue_ns_sha256'] == fitted['selected_issue_ns_sha256'],
            'FUTURE_MODEL_PRIOR_NORMALIZATION_POPULATION')
        spec = self.research_objective
        _need(self.readout.readout_contract == spec['readout'], 'FUTURE_MODEL_READOUT_BINDING')
        self.readout._require_binding()
        expected = torch.tensor(list(bytes.fromhex(spec['contract_sha256'])),
            device=self.future_state_model_sha256.device, dtype=torch.uint8)
        _need(self.future_state_model_sha256.dtype == torch.uint8
            and torch.equal(self.future_state_model_sha256, expected), 'FUTURE_MODEL_IDENTITY')

    def load_state_dict(self, state_dict, strict=True, assign=False):
        _need(strict is True and type(assign) is bool, 'FUTURE_MODEL_STRICT_STATE')
        _need(isinstance(state_dict, Mapping), 'FUTURE_MODEL_STATE_KEYS')
        return super().load_state_dict(state_dict, strict=True, assign=assign)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        self._require_binding()
        local = {key[len(prefix):]: value for key, value in state_dict.items() if key.startswith(prefix)}
        expected = self.state_dict()
        _need(set(local) == set(expected), 'FUTURE_MODEL_STATE_KEYS')
        for name, value in local.items():
            _need(isinstance(value, torch.Tensor) and value.shape == expected[name].shape
                and value.dtype == expected[name].dtype and value.device.type != 'meta'
                and bool(torch.isfinite(value).all()), 'FUTURE_MODEL_STATE_TENSOR')
        for name, value in self.named_buffers():
            # Nonpersistent constructor buffers are never checkpoint inputs.
            if name in expected:
                _need(torch.equal(local[name].detach().cpu(), value.detach().cpu()),
                      'FUTURE_MODEL_STATE_IDENTITY')
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def forward(self, **features):
        self._require_binding()
        _need(set(features) == _native_bar_feature_keys(self.input_binding), 'FUTURE_MODEL_FEATURE_KEYS')
        _need(all(isinstance(value, torch.Tensor) for value in features.values()), 'FUTURE_MODEL_FEATURE_TENSORS')
        output = self.backbone(**features, research_representation_only=True,
            research_timeframe_representations=True)
        hidden = output['entry_q_timeframe_hidden']
        _need(isinstance(hidden, torch.Tensor) and hidden.shape[0] == features['seq_x'].shape[0],
              'FUTURE_MODEL_HIDDEN_ROWS')
        return self.readout(hidden)


def build_native_future_state_model(*, input_normalization, input_binding, architecture,
                                    device, train_prior, target_contract):
    """Fresh body from the existing factory; never construct unused raw heads."""
    native_future_state_model_contract(input_binding, architecture=architecture,
        train_prior=train_prior, target_contract=target_contract)
    backbone = build_native_bar_direction_model(input_normalization=input_normalization,
        input_binding=input_binding, architecture=architecture, device=device,
        return_backbone_only=True)
    return NativeFutureStateModel(backbone, input_binding=input_binding,
        architecture=architecture, train_prior=train_prior, target_contract=target_contract).to(device)
