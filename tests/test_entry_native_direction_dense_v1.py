"""Synthetic minute-head wiring and cutoff tests; not genuine model evidence."""
import inspect

import pytest
import torch

from gx1.contracts import entry_native_direction_dense_v1 as dense
from gx1.contracts import entry_native_direction_research_v1 as legacy
from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import EntryV10CtxHybridTransformer
from tests import test_entry_native_direction_research_v1 as fixture

MINUTE = 60 * 10**9


def _model(monkeypatch):
    monkeypatch.setattr(EntryV10CtxHybridTransformer, 'forward', fixture._native_forward_spy)
    native = fixture._native_shell()
    return dense.DenseNativeEntryDirectionResearch(native), native


def _case():
    decision = torch.tensor([100 * MINUTE, 101 * MINUTE], dtype=torch.int64)
    return {'logit_up': torch.zeros((2, 60), dtype=torch.float32, requires_grad=True),
        'sign': torch.ones((2, 60), dtype=torch.int8), 'observed': torch.ones((2, 60), dtype=torch.bool),
        'decision_ns': decision, 'terminal_time_ns': decision[:, None] + torch.arange(1, 61)[None, :] * MINUTE,
        'cutoff_ns': 200 * MINUTE}


def test_original_anchor_initialization_and_rng_are_identical(monkeypatch):
    monkeypatch.setattr(EntryV10CtxHybridTransformer, 'forward', fixture._native_forward_spy)
    torch.manual_seed(503)
    old = legacy.NativeEntryDirectionResearch(fixture._native_shell())
    old_rng = torch.get_rng_state().clone()
    old_out = old(**fixture._inputs())['logit_up']
    torch.manual_seed(503)
    new = dense.DenseNativeEntryDirectionResearch(fixture._native_shell())
    assert torch.equal(old_rng, torch.get_rng_state())
    assert torch.equal(old.direction_head.weight, new.direction_head.weight)
    assert torch.equal(old.direction_head.bias, new.direction_head.bias)
    output = new(**fixture._inputs())
    assert output['horizons_minutes'] == tuple(range(1, 61))
    assert torch.equal(old_out, output['logit_up'][:, [0, 4, 14]])


def test_all_minutes_have_distinct_heads_and_shared_hidden_gradients(monkeypatch):
    model, native = _model(monkeypatch)
    with torch.no_grad():
        model.direction_head.weight.zero_(); model.dense_aux_head.weight.zero_()
        model.direction_head.bias.copy_(torch.tensor(dense.ANCHORS, dtype=torch.float32))
        model.dense_aux_head.bias.copy_(torch.tensor(dense.AUX_HORIZONS, dtype=torch.float32))
    result = model(**fixture._inputs())
    assert len(native._spy_calls) == 1
    assert torch.equal(result['logit_up'][0], torch.arange(1, 61, dtype=torch.float32))
    result['logit_up'].sum().backward()
    assert torch.equal(model.direction_head.bias.grad, torch.full((3,), 2.))
    assert torch.equal(model.dense_aux_head.bias.grad, torch.full((57,), 2.))
    assert native._synthetic_hidden.grad is not None
    assert all(p.grad is None for layer in (native.head_dynamic_mfe, native.head_forecast, native.task_log_variances)
               for p in layer.parameters())


def test_no_future_values_can_enter_forward(monkeypatch):
    model, native = _model(monkeypatch)
    assert not set(inspect.signature(model.forward).parameters).intersection(
        {'sign', 'observed', 'decision_ns', 'terminal_time_ns', 'cutoff_ns', 'target'})
    with pytest.raises(TypeError):
        model(**fixture._inputs(), sign=torch.ones((2, 60), dtype=torch.int8))
    assert not native._spy_calls


@pytest.mark.parametrize('fault', ['head', 'horizons', 'reorder', 'objective', 'hidden'])
def test_binding_or_native_hidden_fault_fails(monkeypatch, fault):
    model, native = _model(monkeypatch)
    if fault == 'head': model.dense_aux_head.bias.data[0] = float('nan')
    if fault == 'horizons': model.dense_horizon_minutes[0] = 2
    if fault == 'reorder': model.dense_reorder[0] = 1
    if fault == 'objective': model._dense_objective_json = b'{}'
    if fault == 'hidden': native._spy_fault = 'nonfinite'
    with pytest.raises(RuntimeError):
        model(**fixture._inputs())
    assert len(native._spy_calls) == (1 if fault == 'hidden' else 0)


def test_cutoff_loss_masks_equality_missing_and_ties():
    case = _case(); case['cutoff_ns'] = 105 * MINUTE
    case['sign'][0, 1] = -1
    case['sign'][0, 2] = 0
    case['observed'][1, 1] = False; case['sign'][1, 1] = 0
    result = dense.dense_direction_binary_loss(**case)
    expected = (case['terminal_time_ns'] < case['cutoff_ns']) & case['observed'] & (case['sign'] != 0)
    assert torch.equal(result['eligible_mask'], expected)
    assert result['counts']['eligible_cells'] == 5
    assert result['counts']['mature_tie_cells'] == 1
    assert result['counts']['mature_missing_cells'] == 1
    assert result['per_horizon_counts'][4]['eligible_cells'] == 0
    torch.testing.assert_close(result['loss'], torch.log(torch.tensor(2.)))
    result['loss'].backward()
    assert torch.equal(case['logit_up'].grad != 0, expected)
    assert case['logit_up'].grad[0, 0] < 0 and case['logit_up'].grad[0, 1] > 0


def test_empty_targets_do_not_create_update():
    case = _case(); case['sign'].zero_()
    result = dense.dense_direction_binary_loss(**case)
    assert result['loss'] is None and not result['update_allowed']
    assert result['counts']['eligible_cells'] == 0 and case['logit_up'].grad is None


@pytest.mark.parametrize('fault', ['shape', 'finite', 'domain', 'missing', 'cutoff', 'clock'])
def test_loss_rejects_invalid_inputs(fault):
    case = _case()
    if fault == 'shape': case['logit_up'] = torch.zeros(2, 3)
    if fault == 'finite': case['logit_up'] = torch.full((2, 60), float('nan'))
    if fault == 'domain': case['sign'][0, 0] = 2
    if fault == 'missing': case['observed'][0, 0] = False
    if fault == 'cutoff': case['cutoff_ns'] = True
    if fault == 'clock': case['terminal_time_ns'][0, 59] += 1
    with pytest.raises(RuntimeError):
        dense.dense_direction_binary_loss(**case)


def test_dense_recipe_is_explicit_and_old_recipe_remains_unchanged():
    from gx1.contracts import entry_model_native_train_recipe_v1 as recipes
    old = recipes.native_direction_research_recipe(batch_size=2, max_updates=4)
    new = recipes.native_direction_research_recipe(batch_size=2, max_updates=4, dense_minutes=True)
    assert 'dense_minute_supervision' not in old and new['dense_minute_supervision'] is True
    assert new['research_objective_sha256'] == dense.dense_direction_research_contract()['contract_sha256']
    assert recipes.require_native_direction_research_recipe(new) == new
    assert recipes.native_direction_research_recipe(batch_size=2, max_updates=4) == old
    for kwargs in ({'dense_minutes': 1}, {'dense_minutes': True, 'signed_bps': True},
                   {'dense_minutes': True, 'magnitude_weighted': True}):
        with pytest.raises(RuntimeError):
            recipes.native_direction_research_recipe(batch_size=2, max_updates=4, **kwargs)


def _state_case(monkeypatch):
    from gx1.contracts import entry_model_native_train_recipe_v1 as recipes
    from gx1.contracts import entry_native_direction_research_state_v1 as state
    from tests import test_entry_native_direction_research_state_v1 as existing
    case = existing._make(monkeypatch)
    model = dense.DenseNativeEntryDirectionResearch(case.model.backbone)
    identity = existing._identity()
    identity['research_objective_sha256'] = dense.dense_direction_research_contract()['contract_sha256']
    model.feature_view_identity = identity
    recipe = recipes.native_direction_research_recipe(batch_size=2, max_updates=4, dense_minutes=True)
    optimizer, binding = state.build_native_direction_research_optimizer(model, recipe=recipe)
    model.train(); existing._seed(recipe['seed'])
    return model, optimizer, recipe, identity, case.order, binding


def _dense_update(parts, step):
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    from tests import test_entry_native_direction_research_state_v1 as existing
    model, optimizer, *_ = parts
    logits = model(**existing._features(step))['logit_up']
    decision = torch.tensor([100 + 2 * step, 101 + 2 * step], dtype=torch.int64) * MINUTE
    sign = torch.where(torch.arange(60)[None, :] % 2 == 0, 1, -1).to(torch.int8).repeat(2, 1)
    loss = dense.dense_direction_binary_loss(logits, sign=sign, observed=torch.ones((2, 60), dtype=torch.bool),
        decision_ns=decision, terminal_time_ns=decision[:, None] + torch.arange(1, 61)[None, :] * MINUTE,
        cutoff_ns=200 * MINUTE)
    loss['loss'].backward()
    assert model.dense_aux_head.weight.grad is not None and torch.count_nonzero(model.dense_aux_head.weight.grad) > 0
    trainer._optimizer_step_with_finite_gradients(model=model, optimizer=optimizer)


def _dense_capture(parts, cursor, updates):
    from gx1.contracts import entry_native_direction_research_state_v1 as state
    model, optimizer, recipe, identity, order, _ = parts
    return state.capture_native_direction_research_state(model, optimizer, recipe=recipe,
        feature_view_identity=identity, parent_order=order, cursor=cursor, optimizer_updates=updates,
        device=torch.device('cpu'))


def test_dense_stochastic_optimizer_resume_is_exact(monkeypatch):
    from gx1.contracts import entry_native_direction_research_state_v1 as state
    reference = _state_case(monkeypatch); _dense_update(reference, 0); _dense_update(reference, 1)
    expected = _dense_capture(reference, 4, 2)
    interrupted = _state_case(monkeypatch); _dense_update(interrupted, 0); saved = _dense_capture(interrupted, 2, 1)
    resumed = _state_case(monkeypatch)
    model, optimizer, recipe, identity, order, binding = resumed
    assert {'dense_aux_head.weight', 'dense_aux_head.bias'}.issubset(binding['active_parameter_names'])
    state.restore_native_direction_research_state(saved, model, optimizer, recipe=recipe,
        feature_view_identity=identity, parent_order=order, device=torch.device('cpu'))
    _dense_update(resumed, 1); actual = _dense_capture(resumed, 4, 2)
    assert actual['contract_sha256'] == expected['contract_sha256']
    assert actual['optimizer_state_sha256'] == expected['optimizer_state_sha256']
    assert actual['rng_state_sha256'] == expected['rng_state_sha256']


def test_dense_report_matches_existing_three_anchor_panels_and_ranks_own_output():
    import numpy as np
    from gx1.contracts import entry_native_direction_context_report_v1 as old
    from gx1.contracts import entry_native_direction_dense_report_v1 as report
    n = 8; decision = np.arange(100, 100 + n, dtype=np.int64) * MINUTE
    terminal = decision[:, None] + np.arange(1, 61, dtype=np.int64)[None, :] * MINUTE
    signs = np.tile(np.array([1, -1] * 4, dtype=np.int8)[:, None], (1, 60))
    observed = np.ones((n, 60), dtype=np.bool_)
    logits = signs.astype(np.float32) * np.arange(1, 61, dtype=np.float32)[None, :]
    logits[:, 1] *= -1
    scored = report.score_dense_direction_logits(logits, sign=signs, observed=observed,
        decision_ns=decision, terminal_time_ns=terminal, cutoff_ns=200 * MINUTE)
    assert len(scored['per_horizon']) == 60
    assert scored['per_horizon'][0]['rank']['auc'] == 1.0
    assert scored['per_horizon'][1]['rank']['auc'] == 0.0
    assert scored['per_horizon'][0]['rank']['distinct_scores'] == 2
    idx = [0, 4, 14]
    expected = old.score_native_direction_logits(logits[:, idx], sign=signs[:, idx], observed=observed[:, idx],
        decision_ns=decision, terminal_time_ns=terminal[:, idx], cutoff_ns=200 * MINUTE)
    for i, j in enumerate(idx):
        new = dict(scored['per_horizon'][j]); new.pop('rank')
        assert new == expected['per_horizon'][i]


def test_dense_report_excludes_cutoff_equal_and_wrong_clocks():
    import numpy as np
    from gx1.contracts import entry_native_direction_dense_report_v1 as report
    decision = np.array([100 * MINUTE, 101 * MINUTE], dtype=np.int64)
    terminal = decision[:, None] + np.arange(1, 61, dtype=np.int64)[None, :] * MINUTE
    sign = np.ones((2, 60), dtype=np.int8); observed = np.ones((2, 60), dtype=np.bool_)
    z = np.zeros((2, 60), dtype=np.float32)
    result = report.score_dense_direction_logits(z, sign=sign, observed=observed, decision_ns=decision,
        terminal_time_ns=terminal, cutoff_ns=105 * MINUTE)
    assert result['per_horizon'][3]['counts']['eligible_cells'] == 1
    assert result['per_horizon'][4]['counts']['eligible_cells'] == 0
    assert result['per_horizon'][4]['rank']['auc'] is None
    terminal[0, 0] += 1
    with pytest.raises(RuntimeError):
        report.score_dense_direction_logits(z, sign=sign, observed=observed, decision_ns=decision,
            terminal_time_ns=terminal, cutoff_ns=105 * MINUTE)


def test_prefix_train_batch_selects_sixty_minute_loss(monkeypatch):
    from gx1.scripts import run_entry_native_direction_prefix_fit_window_v1 as window
    from tests import test_entry_native_direction_research_state_v1 as existing
    model, optimizer, *_ = _state_case(monkeypatch)
    monkeypatch.setattr(window, '_forward', lambda model, features, **kwargs: model(**features)['logit_up'])
    case = _case(); case.pop('logit_up'); cutoff = case.pop('cutoff_ns')
    counters = {key: 0 for key in ('backward_calls', 'backward_returns', 'optimizer_step_calls', 'optimizer_step_returns', 'empty_batches')}
    before = model.dense_aux_head.weight.detach().clone()
    record, updated = window._train_batch(model, optimizer, existing._features(0), case,
        cutoff=cutoff, counters=counters, ceiling=1, torch=torch, diagnostics=False)
    assert updated and record['eligible_cells'] == 120 and len(record['per_horizon_counts']) == 60
    assert counters['optimizer_step_returns'] == 1 and not torch.equal(before, model.dense_aux_head.weight)


def test_dense_data_joins_real_shape_semantics_and_rejects_drift():
    import copy
    from gx1.contracts import entry_native_direction_dense_data_v1 as bind
    from gx1.scripts import materialize_entry_direction_dense_v1 as materialize
    from tests.test_materialize_entry_direction_dense_v1 import fixture as candle_fixture
    source, saved, cutoff, _ = candle_fixture()
    targets = materialize.build_dense_targets(source, saved, cutoff_ns=cutoff)
    axes = {k: saved[k] for k in bind.AXES}
    joined = bind.bind_dense_targets(axes=axes, legacy=saved, dense=targets)
    assert set(joined) == {'terminal_time_ns', 'observed', 'sign'}
    assert all(not value.flags.writeable for value in joined.values())
    for key in ('parent_rows', 'bar_start_ns', 'decision_ns', 'dynamic_entry_m1_row',
                'terminal_time_ns', 'terminal_source_row', 'observed', 'mid_move', 'mid_return_bps', 'sign'):
        wrong = copy.deepcopy(targets)
        wrong[key] = wrong[key].copy()
        if wrong[key].dtype == bool: wrong[key].flat[0] = not wrong[key].flat[0]
        else: wrong[key].flat[0] += 1
        with pytest.raises(RuntimeError):
            bind.bind_dense_targets(axes=axes, legacy=saved, dense=wrong)
