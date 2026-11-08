"""Narrow TRAIN-only time-query fit boundaries; no model or GPU run."""
from __future__ import annotations

import copy

import pytest
import torch

from gx1.audit import verify_market_state_time_query_zero_step_v1 as zero
from gx1.contracts import market_state_time_query_fold1_fit_v1 as scope
from gx1.contracts import market_state_time_query_fold1_fit_state_v1 as state
from gx1.scripts import run_market_state_time_query_fold1_fit_window_v1 as runner


def test_numerical_and_frozen_fold1_reference_match_selected_scope():
    recipe = scope.fold1_numerical_recipe()
    assert recipe['batch_size'] == 16
    assert recipe['max_updates'] == 1849
    assert recipe['seed'] == 20260911
    assert recipe['research_objective_sha256'] == scope.MODEL_CONTRACT_SHA256
    assert recipe['time_query_loss_contract_sha256'] == scope.LOSS_CONTRACT_SHA256
    report = scope._reference_report()
    fold = report['references']['folds'][0]
    assert fold['fit_rows'] == 29574
    assert fold['score_rows'] == 17753
    assert fold['official_fit_cutoff_ns'] == 1727740800000000000
    assert fold['score_end_exclusive_ns'] == scope.FOLD['oof_cutoff_ns']


def _bound_fold1_checkpoint_identity():
    plan = zero._read_ref(scope.private.ZERO_STEP_REFS['plan'])
    view = zero._bind_view(plan, zero._inspect_receipts(), 1)
    return {
        'contract_sha256': view['contract_sha256'],
        'prefix_normalization': copy.deepcopy(view['prefix_normalization']),
        'research_objective_sha256': view['research_objective']['contract_sha256'],
    }


def test_fold1_checkpoint_view_accepts_exact_mature_units_profile():
    identity = _bound_fold1_checkpoint_identity()
    assert scope.require_checkpoint_view(identity, verify_files=True) == identity

    missing = copy.deepcopy(identity)
    del missing['prefix_normalization']['unobserved_units_profile_sha256']
    with pytest.raises(RuntimeError, match='CHECKPOINT_NORMALIZATION_SHAPE'):
        scope.require_checkpoint_view(missing, verify_files=False)

    tampered = copy.deepcopy(identity)
    tampered['prefix_normalization']['unobserved_units_profile_sha256'] = '0' * 64
    with pytest.raises(RuntimeError, match='CHECKPOINT_UNOBSERVED_UNITS_PROFILE'):
        scope.require_checkpoint_view(tampered, verify_files=False)


def test_legacy_prefix_identity_still_rejects_unobserved_units_field():
    from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as prefix
    identity = _bound_fold1_checkpoint_identity()
    with pytest.raises(RuntimeError, match='PREFIX_IDENTITY'):
        prefix._view_identity(identity, verify_files=False)


def test_state_cursor_counts_skips_and_final_six_row_batch_exactly():
    order = torch.arange(scope.FIT_ROWS, dtype=torch.int64)
    recipe = scope.fold1_numerical_recipe()
    assert state._progress(0, 0, 0, order, recipe) == 0
    assert state._progress(16, 0, 1, order, recipe) == 1
    assert state._progress(scope.FIT_ROWS, 1848, 1, order, recipe) == 1849
    with pytest.raises(RuntimeError, match='EXACT_CONSUMED_BATCH_CURSOR'):
        state._progress(scope.FIT_ROWS - 1, 1848, 1, order, recipe)
    with pytest.raises(RuntimeError, match='EXACT_CONSUMED_BATCH_CURSOR'):
        state._progress(scope.FIT_ROWS, 1848, 0, order, recipe)
    with pytest.raises(RuntimeError, match='EXACT_CONSUMED_BATCH_CURSOR'):
        state._progress(16, 1, 1, order, recipe)


def test_score_only_resume_requires_completed_fit_checkpoint_and_measurement():
    identity = {'contract_sha256': scope.VIEW_SHA256}
    prior = {
        'outcome': 'RESUMABLE', 'parent_order_sha256': 'a' * 64,
        'feature_view_identity': identity, 'oof_cursor': 0,
        'phase_state': 'SCORING', 'parent_cursor': scope.FIT_ROWS,
        'optimizer_updates': 1848,
        'full_state': {'path': '/runs/states/FINAL_FOLD1_FIT_STATE.pt',
                       'sha256': 'b' * 64},
        'final_train_measurement': {'path': '/runs/FINAL_TRAIN.json',
                                    'sha256': 'c' * 64},
    }
    assert runner._require_prior_cursor(
        prior, order_sha='a' * 64, view_identity=identity) is True
    with pytest.raises(RuntimeError, match='SCORE_ONLY_RESUME_FROZEN_FIT'):
        runner._require_prior_cursor(
            {**prior, 'final_train_measurement': None},
            order_sha='a' * 64, view_identity=identity)
    with pytest.raises(RuntimeError, match='SCORE_ONLY_RESUME_FROZEN_FIT'):
        runner._require_prior_cursor(
            {**prior, 'full_state': {'path': '/runs/states/w1_p029574.pt',
                                     'sha256': 'b' * 64}},
            order_sha='a' * 64, view_identity=identity)
    with pytest.raises(RuntimeError, match='RESUME_CURSOR_IDENTITY'):
        runner._require_prior_cursor(
            {**prior, 'oof_cursor': 1},
            order_sha='a' * 64, view_identity=identity)


def test_private_probe_gate_cannot_authorize_fold1_predictor_fit():
    recipe = {'run_scope': scope.SCOPE, 'admission_stage': scope.STAGE}
    admitted = {
        'decision': 'ADMITTED', 'status': 'ADMITTED',
        'time_query_fold1_fit_authorized': True,
        'fold_id': 1, 'fit_rows': scope.FIT_ROWS,
        'score_rows': scope.SCORE_ROWS,
        'max_optimizer_updates': scope.MAX_UPDATES,
        'genuine_predictor_fit_authorized': True,
        'test_data_used': False,
    }
    scope.require_fit_gate(admitted, recipe)
    with pytest.raises(RuntimeError, match='POLICY_GATE'):
        scope.require_fit_gate({
            **admitted, 'genuine_predictor_fit_authorized': False,
        }, recipe)
    with pytest.raises(RuntimeError, match='POLICY_GATE'):
        scope.require_fit_gate({
            **admitted, 'time_query_fold1_fit_authorized': False,
        }, recipe)


def _ta_sf_model(monkeypatch):
    from torch import nn
    from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
    from gx1.contracts.market_state_time_query_model_v2 import FactorizedTimeQueryMarketStateModel
    from gx1.contracts.market_state_time_query_native_F_model_v1 import NativeFutureDualModel
    from gx1.features.entry_specialist_feature_groups_v1 import compile_full_ta_entry_execution_profile
    from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import CtxModelConfig, EntryV10CtxHybridTransformer
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
    backbone._drop = nn.Dropout(.25)
    backbone.head_forecast = nn.Linear(4, 3)
    backbone.dynamic_m5_query_projection = nn.Linear(4, 3)
    backbone.task_log_variances = nn.ParameterDict({'old': nn.Parameter(torch.zeros(()))})
    backbone.entry_decision_token = nn.Linear(4, 4)
    backbone.exit_probe = nn.Linear(4, 1)
    backbone.register_buffer('input_norm_signal_center', torch.zeros(4))
    backbone.register_buffer('research_unobserved_units_profile_sha256',
                             torch.tensor([3]*32, dtype=torch.uint8))
    profile = compile_full_ta_entry_execution_profile()
    backbone.entry_execution_profile = {key: profile[key] for key in ('name', 'contract_sha256')}
    backbone.gc_mtf_history = None
    base = FactorizedTimeQueryMarketStateModel(backbone)
    base.feature_view_identity = {'contract_sha256': '1'*64,
        'prefix_normalization': {'contract_sha256': '2'*64},
        'research_objective_sha256': base.research_objective['contract_sha256']}
    return NativeFutureDualModel(base)


def _ta_sf_case(monkeypatch):
    import random
    import numpy as np
    recipe = scope.ta_sf_numerical_recipe()
    seed = recipe['seed']
    def seed_all():
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    seed_all()
    model = _ta_sf_model(monkeypatch)
    optimizer, binding = state.build_fold1_fit_optimizer(model, numerical_recipe=recipe)
    seed_all()
    return model, optimizer, recipe, binding


def _ta_sf_update(model, optimizer, step):
    import random
    import numpy as np
    from gx1.contracts.market_state_time_query_target_v1 import NATIVE_K
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    optimizer.zero_grad(set_to_none=True)
    x = torch.tensor([[1., 2., 3., 4.]]).expand(16, -1) + float(step)
    hidden = model.backbone._drop(model.backbone.entry_q_joint_in(torch.cat((x,)*4, dim=1)))
    hidden = hidden + (random.random() + float(np.random.random())) * torch.rand_like(hidden)
    q = torch.tensor(NATIVE_K, dtype=torch.int64).expand(16, -1)
    s = model.state_model.decode_raw_query_grid(hidden, q[:, None, :])[:, 0]
    f = model.decode_native_F_raw(hidden)
    (s.square().mean() + f.square().mean()).backward()
    trainer._optimizer_step_with_finite_gradients(model=model, optimizer=optimizer)
    optimizer.zero_grad(set_to_none=True)


def _ta_sf_capture(model, optimizer, recipe, updates):
    return state.capture_fold1_fit_state(
        model, optimizer, numerical_recipe=recipe,
        feature_view_identity=model.state_model.feature_view_identity,
        parent_order=torch.arange(scope.FIT_ROWS, dtype=torch.int64),
        cursor=16*updates, optimizer_updates=updates, skipped_batches=0,
        source_commit='c'*40, device=torch.device('cpu'))


def test_ta_sf_state_keeps_view_and_dual_objective_identities_separate(monkeypatch):
    model, optimizer, recipe, binding = _ta_sf_case(monkeypatch)
    assert binding['schema_version'] == state.TA_BINDING_SCHEMA
    assert binding['feature_view_identity']['research_objective_sha256'] == model.state_model.research_objective['contract_sha256']
    assert binding['research_objective']['contract_sha256'] == recipe['research_objective_sha256']
    assert binding['feature_view_identity']['research_objective_sha256'] != recipe['research_objective_sha256']
    assert {'state_model.query_head.weight', 'future_head.weight',
            'state_model.backbone.entry_q_joint_in.weight'} <= set(binding['active_parameter_names'])
    assert 'state_model.backbone.head_forecast.weight' in binding['inactive_parameter_names']
    saved = _ta_sf_capture(model, optimizer, recipe, 0)
    assert saved['schema_version'] == state.TA_STATE_SCHEMA
    bad = dict(recipe, future_weight=.5)
    with pytest.raises(RuntimeError, match='TA_SF_NUMERICAL_RECIPE'):
        state.build_fold1_fit_optimizer(model, numerical_recipe=bad)


def test_ta_sf_full_cursor_resume_keeps_model_adamw_rng_and_modes_bit_exact(monkeypatch):
    reference, opt, recipe, _ = _ta_sf_case(monkeypatch)
    for step in range(2):
        _ta_sf_update(reference, opt, step)
    expected = _ta_sf_capture(reference, opt, recipe, 2)
    source, opt, _, _ = _ta_sf_case(monkeypatch)
    _ta_sf_update(source, opt, 0)
    saved = _ta_sf_capture(source, opt, recipe, 1)
    resumed, opt, _, _ = _ta_sf_case(monkeypatch)
    restored = state.restore_fold1_fit_state(
        saved, resumed, opt, numerical_recipe=recipe,
        feature_view_identity=resumed.state_model.feature_view_identity,
        parent_order=torch.arange(scope.FIT_ROWS, dtype=torch.int64),
        source_commit='c'*40, device=torch.device('cpu'))
    assert restored == {'cursor': 16, 'optimizer_updates': 1, 'skipped_batches': 0}
    _ta_sf_update(resumed, opt, 1)
    actual = _ta_sf_capture(resumed, opt, recipe, 2)
    for key in ('model_state_sha256', 'buffer_state_sha256',
                'optimizer_state_sha256', 'rng_state_sha256', 'module_training'):
        assert actual[key] == expected[key]


@pytest.mark.parametrize('name', [
    'native_F_objective_sha256', 'state_model.query_native_k',
    'state_model.backbone.input_norm_signal_center',
    'state_model.backbone.research_unobserved_units_profile_sha256',
])
def test_ta_sf_resealed_immutable_buffer_rejected_before_restore(monkeypatch, name):
    from gx1.contracts import entry_native_direction_research_state_v1 as old
    from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256 as digest
    source, opt, recipe, _ = _ta_sf_case(monkeypatch)
    _ta_sf_update(source, opt, 0)
    bad = copy.deepcopy(_ta_sf_capture(source, opt, recipe, 1))
    bad['model_state'][name].view(-1)[0] += 1
    bad['buffer_state'][name].view(-1)[0] += 1
    bad['model_state_sha256'] = digest(bad['model_state'])
    bad['buffer_state_sha256'] = old._tree_sha(bad['buffer_state'])
    bad['contract_sha256'] = old._tree_sha({k: v for k, v in bad.items() if k != 'contract_sha256'})
    fresh, opt, _, _ = _ta_sf_case(monkeypatch)
    before = _ta_sf_capture(fresh, opt, recipe, 0)
    with pytest.raises(RuntimeError, match='IMMUTABLE_BUFFER'):
        state.restore_fold1_fit_state(
            bad, fresh, opt, numerical_recipe=recipe,
            feature_view_identity=fresh.state_model.feature_view_identity,
            parent_order=torch.arange(scope.FIT_ROWS, dtype=torch.int64),
            source_commit='c'*40, device=torch.device('cpu'))
    assert _ta_sf_capture(fresh, opt, recipe, 0)['model_state_sha256'] == before['model_state_sha256']
