"""Synthetic CPU checks of private time-query AdamW and exact resume mechanics.

No market data, physical native job, fitted normalizer or quality claim.
"""
from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from torch import nn

from gx1.contracts.market_state_time_query_private_campaign_v1 import PARENT_ROWS, time_query_private_numerical_recipe
from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
from gx1.contracts import entry_native_direction_research_state_v1 as old
from gx1.contracts import market_state_time_query_private_state_v1 as state_owner
from gx1.contracts.market_state_time_query_model_v1 import TimeQueryMarketStateModel
from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K
from gx1.features.entry_specialist_feature_groups_v1 import FULL_GC_MTF_ENTRY_EXECUTION_PROFILE_NAME
from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import CtxModelConfig, EntryV10CtxHybridTransformer

DEVICE = torch.device('cpu')
ORDER = torch.tensor(PARENT_ROWS, dtype=torch.int64)


def _seed():
    seed = time_query_private_numerical_recipe()['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _recipe():
    return time_query_private_numerical_recipe()


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
    model = TimeQueryMarketStateModel(backbone)
    model.feature_view_identity = {'contract_sha256': '1' * 64,
        'prefix_normalization': {'contract_sha256': '2' * 64},
        'research_objective_sha256': model.research_objective['contract_sha256']}
    model.train()
    return model


def _case(monkeypatch):
    _seed()
    model = _model(monkeypatch)
    recipe = _recipe()
    optimizer, binding = state_owner.build_time_query_private_optimizer(
        model, numerical_recipe=recipe)
    _seed()
    return model, optimizer, recipe, binding


def _update(model, optimizer, step):
    x = torch.tensor([[1., 2., 3., 4.]]).expand(16, -1) + float(step)
    hidden = model.backbone._drop(model.backbone.entry_q_joint_in(torch.cat((x,) * 4, dim=1)))
    hidden = hidden + (random.random() + float(np.random.random())) * torch.rand_like(hidden)
    query = torch.tensor(NATIVE_K, dtype=torch.int64).expand(16, -1)
    logits = model.decode_queries(hidden, query)
    loss = (logits.square().mean() + 0.1 * logits[..., 0].mean())
    loss.backward()
    trainer._optimizer_step_with_finite_gradients(model=model, optimizer=optimizer)


def _capture(model, optimizer, recipe, cursor, updates):
    return state_owner.capture_time_query_private_state(model, optimizer,
        numerical_recipe=recipe, feature_view_identity=model.feature_view_identity,
        parent_order=ORDER, cursor=cursor, optimizer_updates=updates, device=DEVICE)


def test_optimizer_freezes_old_heads_and_preserves_query_decoder(monkeypatch):
    model, optimizer, recipe, binding = _case(monkeypatch)
    assert binding['schema_version'] == state_owner.BINDING_SCHEMA
    assert set(binding['optimizer_group_parameter_names'][0]) == set(binding['active_parameter_names'])
    assert {'query_projection.0.weight', 'query_head.weight'} <= set(binding['active_parameter_names'])
    assert 'backbone.entry_q_joint_in.weight' in binding['active_parameter_names']
    assert 'backbone.head_forecast.weight' in binding['inactive_parameter_names']
    assert all(not p.requires_grad for name, p in model.named_parameters()
               if name in binding['inactive_parameter_names'])
    assert len(optimizer.param_groups) == 1
    assert recipe['max_updates'] == 2


def test_two_steps_bit_identical_to_interrupted_resume(monkeypatch):
    reference, reference_opt, recipe, _ = _case(monkeypatch)
    _update(reference, reference_opt, 0)
    _update(reference, reference_opt, 1)
    expected = _capture(reference, reference_opt, recipe, 32, 2)

    interrupted, interrupted_opt, _, _ = _case(monkeypatch)
    _update(interrupted, interrupted_opt, 0)
    saved = _capture(interrupted, interrupted_opt, recipe, 16, 1)
    resumed, resumed_opt, _, _ = _case(monkeypatch)
    restored = state_owner.restore_time_query_private_state(saved, resumed,
        resumed_opt, numerical_recipe=recipe,
        feature_view_identity=resumed.feature_view_identity,
        parent_order=ORDER, device=DEVICE)
    assert restored['cursor'] == 16 and restored['optimizer_updates'] == 1
    _update(resumed, resumed_opt, 1)
    actual = _capture(resumed, resumed_opt, recipe, 32, 2)
    assert actual['model_state_sha256'] == expected['model_state_sha256']
    assert actual['buffer_state_sha256'] == expected['buffer_state_sha256']
    assert actual['optimizer_state_sha256'] == expected['optimizer_state_sha256']
    assert actual['rng_state_sha256'] == expected['rng_state_sha256']
    assert actual['module_training'] == expected['module_training']


@pytest.mark.parametrize('fault', ['order', 'immutable', 'partial_gradient'])
def test_restore_rejects_changed_identity_before_model_mutation(monkeypatch, fault):
    source, source_opt, recipe, _ = _case(monkeypatch)
    _update(source, source_opt, 0)
    saved = _capture(source, source_opt, recipe, 16, 1)
    fresh, fresh_opt, _, _ = _case(monkeypatch)
    before = old._tree_sha(fresh.state_dict())
    if fault == 'order':
        order = ORDER.flip(0)
    else:
        order = ORDER
    if fault == 'immutable':
        fresh.backbone.research_unobserved_units_profile_sha256[0] = 4
        before = old._tree_sha(fresh.state_dict())
    if fault == 'partial_gradient':
        fresh.query_head.weight.grad = torch.ones_like(fresh.query_head.weight)
    with pytest.raises(RuntimeError):
        state_owner.restore_time_query_private_state(saved, fresh, fresh_opt,
            numerical_recipe=recipe, feature_view_identity=fresh.feature_view_identity,
            parent_order=order, device=DEVICE)
    assert old._tree_sha(fresh.state_dict()) == before
