"""S-only checkpoint compatibility on a small synthetic native shell."""
import copy

import pytest
import torch
from torch import nn

from gx1.contracts.entry_model_native_training_objective_v1 import (
    ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE,
)
from gx1.contracts.market_state_fold2_multiminute_private_v1 import (
    PARENT_ROWS, numerical_recipe,
)
from gx1.contracts.market_state_fold2_multiminute_side_loss_v1 import (
    contract as loss_contract,
)
from gx1.contracts.market_state_time_query_model_v2 import (
    FactorizedTimeQueryMarketStateModel,
)
from gx1.contracts import market_state_time_query_private_state_v1 as state
from gx1.features.entry_specialist_feature_groups_v1 import (
    FULL_GC_MTF_ENTRY_EXECUTION_PROFILE_NAME,
)
from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import (
    CtxModelConfig, EntryV10CtxHybridTransformer,
)


def model():
    native = EntryV10CtxHybridTransformer.__new__(EntryV10CtxHybridTransformer)
    nn.Module.__init__(native)
    cfg = CtxModelConfig.__new__(CtxModelConfig)
    object.__setattr__(cfg, 'd_model', 4)
    object.__setattr__(cfg, 'entry_training_phase',
                       ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE)
    native.cfg = cfg
    native.entry_training_phase = cfg.entry_training_phase
    native.entry_q_joint_in = nn.Linear(16, 4)
    native.head_inactive = nn.Linear(4, 2)
    native.entry_execution_profile = {
        'name': FULL_GC_MTF_ENTRY_EXECUTION_PROFILE_NAME}
    native.gc_mtf_history = {'normalization': {}}
    native.require_input_normalization_state = lambda: None
    result = FactorizedTimeQueryMarketStateModel(native)
    result.feature_view_identity = {
        'contract_sha256': 'a' * 64,
        'prefix_normalization': {},
        'research_objective_sha256': result.research_objective['contract_sha256']}
    return result


def test_s_only_recipe_and_parameter_partition_are_exact():
    recipe = numerical_recipe()
    assert recipe['batch_size'] == 16 and recipe['max_updates'] == 2
    assert recipe['fold2_25_cell_s_loss_contract_sha256'] == loss_contract()['contract_sha256']
    candidate = model()
    optimizer, binding = state.build_time_query_private_optimizer(
        candidate, numerical_recipe=recipe)
    assert type(optimizer) is torch.optim.AdamW
    assert {'query_projection.0.weight', 'query_head.weight',
            'backbone.entry_q_joint_in.weight'} <= set(binding['active_parameter_names'])
    assert 'backbone.head_inactive.weight' in binding['inactive_parameter_names']
    assert not candidate.backbone.head_inactive.weight.requires_grad
    wrong = copy.deepcopy(recipe)
    wrong['fold2_25_cell_s_loss_contract_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='PRIVATE_SCOPE'):
        state.build_time_query_private_optimizer(model(), numerical_recipe=wrong)


def test_s_only_checkpoint_restores_model_optimizer_and_rng():
    recipe = numerical_recipe()
    torch.manual_seed(recipe['seed'])
    original = model()
    optimizer, _ = state.build_time_query_private_optimizer(
        original, numerical_recipe=recipe)
    order = torch.tensor(PARENT_ROWS, dtype=torch.int64)
    start = state.capture_time_query_private_state(
        original, optimizer, numerical_recipe=recipe,
        feature_view_identity=original.feature_view_identity,
        parent_order=order, cursor=0, optimizer_updates=0,
        device=torch.device('cpu'))
    active = [p for p in original.parameters() if p.requires_grad]
    sum((p.square().sum() for p in active)).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    saved = state.capture_time_query_private_state(
        original, optimizer, numerical_recipe=recipe,
        feature_view_identity=original.feature_view_identity,
        parent_order=order, cursor=16, optimizer_updates=1,
        device=torch.device('cpu'))
    restored_model = model()
    restored_optimizer, _ = state.build_time_query_private_optimizer(
        restored_model, numerical_recipe=recipe)
    cursor = state.restore_time_query_private_state(
        saved, restored_model, restored_optimizer,
        numerical_recipe=recipe,
        feature_view_identity=restored_model.feature_view_identity,
        parent_order=order, device=torch.device('cpu'))
    again = state.capture_time_query_private_state(
        restored_model, restored_optimizer, numerical_recipe=recipe,
        feature_view_identity=restored_model.feature_view_identity,
        parent_order=order, cursor=16, optimizer_updates=1,
        device=torch.device('cpu'))
    assert cursor['cursor'] == 16 and cursor['optimizer_updates'] == 1
    assert saved['contract_sha256'] == again['contract_sha256']
    assert start['model_state_sha256'] != saved['model_state_sha256']
    assert start['optimizer_state_sha256'] != saved['optimizer_state_sha256']
