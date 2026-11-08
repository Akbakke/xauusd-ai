"""Synthetic exact S/F model AdamW state and resume; no market input or fit."""
import copy
import random

import numpy as np
import pytest
import torch
from torch import nn

from gx1.contracts import market_state_native_F_private_state_v1 as state
from gx1.contracts import entry_native_direction_research_state_v1 as old
from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256 as model_sha
from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
from gx1.contracts.market_state_time_query_model_v2 import FactorizedTimeQueryMarketStateModel
from gx1.contracts.market_state_time_query_native_F_model_v1 import NativeFutureDualModel
from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K
from gx1.features.entry_specialist_feature_groups_v1 import FULL_GC_MTF_ENTRY_EXECUTION_PROFILE_NAME
from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import CtxModelConfig, EntryV10CtxHybridTransformer

DEVICE = torch.device('cpu')
ORDER = torch.tensor(state.PARENT_ROWS, dtype=torch.int64)


def _seed():
    seed = state.numerical_recipe()['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _model(monkeypatch):
    monkeypatch.setattr(EntryV10CtxHybridTransformer,
                        'require_input_normalization_state', lambda self: None)
    backbone = EntryV10CtxHybridTransformer.__new__(EntryV10CtxHybridTransformer)
    nn.Module.__init__(backbone)
    cfg = CtxModelConfig.__new__(CtxModelConfig)
    object.__setattr__(cfg, 'd_model', 4)
    object.__setattr__(cfg, 'entry_training_phase', ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE)
    backbone.cfg = cfg
    backbone.entry_training_phase = cfg.entry_training_phase
    backbone.entry_q_joint_in = nn.Linear(16, 4)
    backbone._drop = nn.Dropout(0.25)
    backbone.head_forecast = nn.Linear(4, 3)
    backbone.dynamic_m5_query_projection = nn.Linear(4, 3)
    backbone.task_log_variances = nn.ParameterDict({'old': nn.Parameter(torch.zeros(()))})
    backbone.entry_decision_token = nn.Linear(4, 4)
    backbone.exit_probe = nn.Linear(4, 1)
    backbone.register_buffer('input_norm_signal_center', torch.zeros(4))
    backbone.register_buffer('research_unobserved_units_profile_sha256',
                             torch.tensor([3] * 32, dtype=torch.uint8))
    backbone.entry_execution_profile = {'name': FULL_GC_MTF_ENTRY_EXECUTION_PROFILE_NAME}
    backbone.gc_mtf_history = {'normalization': {}}
    base = FactorizedTimeQueryMarketStateModel(backbone)
    base.feature_view_identity = {'contract_sha256': '1' * 64,
        'prefix_normalization': {'contract_sha256': '2' * 64},
        'research_objective_sha256': base.research_objective['contract_sha256']}
    dual = NativeFutureDualModel(base)
    dual.train()
    return dual


def _case(monkeypatch):
    _seed()
    model = _model(monkeypatch)
    recipe = state.numerical_recipe()
    optimizer, binding = state.build_optimizer(model, recipe=recipe)
    _seed()
    return model, optimizer, recipe, binding


def _update(model, optimizer, step):
    x = torch.tensor([[1., 2., 3., 4.]]).expand(16, -1) + float(step)
    hidden = model.backbone._drop(model.backbone.entry_q_joint_in(torch.cat((x,) * 4, dim=1)))
    hidden = hidden + (random.random() + float(np.random.random())) * torch.rand_like(hidden)
    query = torch.tensor(NATIVE_K, dtype=torch.int64).expand(16, -1)
    s = model.state_model.decode_raw_query_grid(hidden, query[:, None, :])[:, 0]
    f = model.decode_native_F_raw(hidden)
    (s.square().mean() + f.square().mean()).backward()
    trainer._optimizer_step_with_finite_gradients(model=model, optimizer=optimizer)


def _capture(model, optimizer, recipe, updates):
    return state.capture(model, optimizer, recipe=recipe,
        feature_view_identity=model.state_model.feature_view_identity,
        parent_order=ORDER, cursor=16*updates, optimizer_updates=updates,
        device=DEVICE)


def test_partition_includes_both_heads_and_freezes_unused_backbone_heads(monkeypatch):
    model, optimizer, recipe, binding = _case(monkeypatch)
    active = set(binding['active_parameter_names'])
    inactive = set(binding['inactive_parameter_names'])
    assert {'state_model.query_head.weight', 'future_head.weight',
            'state_model.backbone.entry_q_joint_in.weight'} <= active
    assert 'state_model.backbone.head_forecast.weight' in inactive
    assert all(not p.requires_grad for name, p in model.named_parameters() if name in inactive)
    assert len(optimizer.param_groups) == 1 and recipe['max_updates'] == 2


def test_two_logical_steps_match_checkpoint_restore(monkeypatch):
    reference, opt, recipe, _ = _case(monkeypatch)
    for step in range(2):
        _update(reference, opt, step)
    expected = _capture(reference, opt, recipe, 2)

    interrupted, opt, _, _ = _case(monkeypatch)
    _update(interrupted, opt, 0)
    saved = _capture(interrupted, opt, recipe, 1)
    resumed, opt, _, _ = _case(monkeypatch)
    restored = state.restore(saved, resumed, opt, recipe=recipe,
        feature_view_identity=resumed.state_model.feature_view_identity,
        parent_order=ORDER, device=DEVICE)
    assert restored['cursor'] == 16 and restored['optimizer_updates'] == 1
    _update(resumed, opt, 1)
    actual = _capture(resumed, opt, recipe, 2)
    for key in ('model_state_sha256', 'buffer_state_sha256',
                'optimizer_state_sha256', 'rng_state_sha256'):
        assert actual[key] == expected[key]
    assert actual['module_training'] == expected['module_training']


def test_bad_checkpoint_rejected_before_model_mutation(monkeypatch):
    source, optimizer, recipe, _ = _case(monkeypatch)
    _update(source, optimizer, 0)
    saved = _capture(source, optimizer, recipe, 1)
    bad = copy.deepcopy(saved)
    bad['cursor'] = 32
    fresh, optimizer, _, _ = _case(monkeypatch)
    before = _capture(fresh, optimizer, recipe, 0)['model_state_sha256']
    with pytest.raises(RuntimeError, match='SEAL'):
        state.restore(bad, fresh, optimizer, recipe=recipe,
            feature_view_identity=fresh.state_model.feature_view_identity,
            parent_order=ORDER, device=DEVICE)
    after = _capture(fresh, optimizer, recipe, 0)['model_state_sha256']
    assert before == after


def test_resealed_wrong_objective_buffer_rejected_before_model_mutation(monkeypatch):
    source, optimizer, recipe, _ = _case(monkeypatch)
    _update(source, optimizer, 0)
    bad = copy.deepcopy(_capture(source, optimizer, recipe, 1))
    name = 'native_F_objective_sha256'
    bad['model_state'][name][0] ^= 1
    bad['buffer_state'][name][0] ^= 1
    bad['model_state_sha256'] = model_sha(bad['model_state'])
    bad['buffer_state_sha256'] = old._tree_sha(bad['buffer_state'])
    bad['contract_sha256'] = old._tree_sha({k: v for k, v in bad.items()
                                            if k != 'contract_sha256'})
    fresh, optimizer, _, _ = _case(monkeypatch)
    before = _capture(fresh, optimizer, recipe, 0)['model_state_sha256']
    with pytest.raises(RuntimeError, match='IMMUTABLE_BUFFER'):
        state.restore(bad, fresh, optimizer, recipe=recipe,
            feature_view_identity=fresh.state_model.feature_view_identity,
            parent_order=ORDER, device=DEVICE)
    assert _capture(fresh, optimizer, recipe, 0)['model_state_sha256'] == before
