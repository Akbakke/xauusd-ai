"""Initial native measurement admission and state preservation; synthetic only."""
import copy
import json
import time
from pathlib import Path

import pytest
import torch

from gx1.contracts import unified_exit_bounded_val_cohort_v1 as cohorts
from gx1.contracts import entry_model_native_train_launch_v1 as budgets
from tests.test_native_prefix_recipe import prefix_scope, scope, native, runner, _bind, _write
from tests.test_native_prefix_recipe import test_native_campaign_materialization_window_and_final_review_stop as campaign_check
from tests.test_native_prefix_coordinator import PrefixHarness, _prepared, equal_tree, digest, trainer


@pytest.fixture
def initial_scope(prefix_scope, tmp_path):
    policy, recipe, files, seal = prefix_scope
    policy.pop('chronological_learning_run')
    source = tmp_path / 'gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py'
    source.parent.mkdir(parents=True)
    source.write_text('synthetic model source')
    state = tmp_path / 'INITIAL_STATE.pt'
    state.write_bytes(b'synthetic state; loaded only in separate restore test')
    initial = {'schema_version': 'gx1_prefix_fresh_initialization_v1',
        'decision': 'ACTUAL_NATIVE_FRESH_COMPONENTS_READY_NO_LEARNING_MEASURED',
        'chronological_prefix': recipe['chronological_prefix'], 'files': recipe['files'],
        'inherited_checkpoint_weights': False, 'optimizer_state_empty': True, 'target_frozen': True,
        'model_forwards': 0, 'optimizer_steps': 0, 'ema_steps': 0, 'normalization_refits': 0,
        'online_model_state_sha256': 'a'*64, 'target_model_state_sha256': 'a'*64,
        'ema_model_state_sha256': 'a'*64, 'initial_state': _bind(state),
        'source_bindings': {str(source.relative_to(tmp_path)): digest(source)}}
    coordinates = _write(tmp_path / 'coordinates.json', {'design': recipe['chronological_prefix']['design']})
    measurement = {'schema_version': 'gx1_prefix_native_measurement_binding_result_v1',
        'decision': 'FROZEN_TRAIN_CONTROL_COORDINATES_AND_NATIVE_MEASUREMENT_BINDING_READY',
        'actual_model_forwards': 0, 'optimizer_steps': 0, 'test_data_used': False,
        'coordinate_result': coordinates}
    artifacts = {'initialization_result': _write(tmp_path / 'initial.json', initial),
                 'measurement_binding_result': _write(tmp_path / 'measurement.json', measurement)}
    recipe['chronological_initial_measurement'] = artifacts
    policy['chronological_initial_measurement'] = {**artifacts,
        'chronological_prefix': recipe['chronological_prefix'], 'run_id': recipe['run_id'],
        'out_bundle_dir': recipe['out_bundle_dir'], 'source_bindings_sha256': recipe['source_bindings_sha256'],
        'optimizer_steps': 0, 'max_invocations': 1, 'teacher_refresh_allowed': False,
        'full_epoch_training_allowed': False, 'full_val_allowed': False, 'test_data_used': False}
    seal()
    return policy, recipe, files, seal


def test_initial_scope_admits_only_one_zero_step_window(initial_scope, tmp_path):
    policy, recipe, _, seal = initial_scope
    rb = seal()
    budget = {'schema_version': budgets.CANDIDATE_EXECUTION_BUDGET_SCHEMA,
        'recipe_json': rb['path'], 'recipe_sha256': rb['sha256'],
        'stop_after_optimizer_steps': 0, 'stop_after_completed_val_epochs': None,
        'max_invocation_seconds': 12000, 'expected_active_pointer_sha256': None}
    def checked(value, recipe_value=recipe):
        binding = _write(tmp_path / 'budget.json', value)
        return budgets.require_candidate_execution_budget(Path(binding['path']), binding['sha256'],
            recipe_path=Path(rb['path']), recipe_sha256=rb['sha256'], recipe=recipe_value)
    assert checked(budget) == budget
    assert native.require_native_run_scope(recipe, invocation_number=1, execution_budget=budget) == 0
    for number in (0, 2, True):
        with pytest.raises(RuntimeError, match='INITIAL_INVOCATION'):
            native.require_native_run_scope(recipe, invocation_number=number)
    for field, value in [('stop_after_optimizer_steps', 1), ('stop_after_optimizer_steps', False),
                         ('expected_active_pointer_sha256', 'a'*64), ('stop_after_completed_val_epochs', 1)]:
        with pytest.raises(RuntimeError, match='INITIAL_BUDGET'):
            native.require_native_run_scope(recipe, execution_budget={**budget, field: value})
    old = {k:v for k,v in recipe.items() if k != 'chronological_initial_measurement'}
    with pytest.raises(ValueError, match='CEILING_INVALID'): checked(budget, old)
    with pytest.raises(ValueError, match='CEILING_INVALID'): checked({**budget, 'stop_after_optimizer_steps': False})
    policy['chronological_learning_run'] = {}
    seal()
    with pytest.raises(RuntimeError, match='NOT_AUTHORIZED'): native.require_native_run_scope(recipe)


@pytest.mark.parametrize('fault', ['initial_bytes', 'architecture', 'weights', 'design', 'full_training'])
def test_initial_scope_rejects_changed_artifacts_and_authority(initial_scope, fault):
    policy, recipe, _, seal = initial_scope
    artifacts = recipe['chronological_initial_measurement']
    if fault == 'full_training': policy['training_enabled'] = True
    elif fault == 'architecture':
        (Path(recipe['source_repo']) / 'gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py').write_text('changed')
    else:
        initial = json.loads(Path(artifacts['initialization_result']['path']).read_text())
        if fault == 'initial_bytes': Path(artifacts['initialization_result']['path']).write_text('{}')
        elif fault == 'weights': Path(initial['initial_state']['path']).write_bytes(b'changed')
        else:
            measurement = json.loads(Path(artifacts['measurement_binding_result']['path']).read_text())
            Path(measurement['coordinate_result']['path']).write_text('{}')
    seal()
    with pytest.raises(RuntimeError): native.require_native_run_scope(recipe)


def test_initial_native_campaign_stops_for_review_at_zero(initial_scope, monkeypatch, tmp_path):
    campaign_check(initial_scope, monkeypatch, tmp_path)


def test_guarded_dispatch_restores_then_saves_then_measures(initial_scope, monkeypatch, tmp_path):
    _, recipe, files, seal = initial_scope; rb = seal(); seen = []
    budget = {'stop_after_optimizer_steps': 0, 'stop_after_completed_val_epochs': None,
              'max_invocation_seconds': 12000, 'expected_active_pointer_sha256': None}
    monkeypatch.setattr(runner.trainer, '_require_cuda_trainer_guard_execution', lambda **kw: seen.append('guard'))
    monkeypatch.setattr(runner.trainer, '_resolve_device', lambda _: torch.device('cpu'))
    monkeypatch.setattr(runner, '_require_native_full_train_recipe', lambda *a: (recipe, files, {}))
    monkeypatch.setattr(runner.launch_owner, 'require_candidate_execution_budget', lambda *a, **kw: budget)
    monkeypatch.setattr(runner, '_build_bound_full_train_components', lambda **kw: {'chronological_prefix': kw['chronological_prefix']})
    monkeypatch.setattr(runner, '_restore_prefix_initial_measurement_state', lambda **kw: seen.append('restore'))
    def save(**kw):
        assert kw['execution_budget']['stop_after_optimizer_steps'] == 0
        seen.append('save')
        raise runner.trainer._CandidateExecutionPaused({'reason': 'optimizer_step_ceiling', 'global_optimizer_steps': 0})
    monkeypatch.setattr(runner, '_run_bound_full_train_candidate', save)
    def measure(**kw):
        seen.append('measure'); return {'path': 'synthetic', 'sha256': 'a'*64}
    monkeypatch.setattr(runner, '_run_prefix_initial_measurement', measure)
    receipt = tmp_path / 'pause.json'; receipt.write_text('{}')
    monkeypatch.setattr(runner.trainer, '_write_candidate_execution_pause_receipt', lambda *a, **kw: receipt)
    monkeypatch.setattr(runner, '_native_resume_state', lambda **kw: {'global_optimizer_steps': 0})
    result = runner.run_guarded_native_candidate_invocation(recipe_path=Path(rb['path']), recipe_file_sha256=rb['sha256'],
        execution_budget_path=tmp_path/'budget.json', execution_budget_file_sha256='b'*64)
    assert seen == ['guard','restore','save','measure']
    assert result['decision'] == 'PAUSED_RESUMABLE' and result['pause']['chronological_initial_measurement']['sha256'] == 'a'*64


def test_zero_step_native_save_and_resume_equal_uninterrupted_training(tmp_path):
    artifacts = tmp_path / 'artifacts'; artifacts.mkdir()
    h = PrefixHarness(tmp_path / 'run', _prepared(artifacts))
    direct, paused = h.root / 'DIRECT', h.root / 'MEASURED'
    h.run_prefix(direct, 2)
    expected, batches = h.state(direct), list(h.batches)
    h.batches.clear()
    _, pause = h.run_prefix(paused, 0)
    zero = h.state(paused)
    assert pause['global_optimizer_steps'] == pause['next_batch_offset'] == 0
    assert pause['phase'] == 'train' and h.batches == [] and h.validation_batches == 0
    assert zero['optimizer_state']['state'] == {} and zero['weight_ema_state']['steps'] == 0
    equal_tree(zero['model_state'], zero['target_model_state'])
    equal_tree(zero['model_state'], zero['weight_ema_state']['shadow'])
    h.run_prefix(paused, 2, expected_pointer=digest(h.pointer(paused)))
    actual = h.state(paused)
    for key in ('model_state','target_model_state','optimizer_state','weight_ema_state',
                'lr_scheduler_state','rng_state','epoch_order','training_progress'):
        equal_tree(expected[key], actual[key])
    assert h.batches == batches


def test_saved_fresh_state_restore_includes_optimizer_scheduler_ema_and_rng(tmp_path):
    torch.manual_seed(20260911)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0001)
    ema = trainer._WeightEma(model, .5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    model_hash = trainer._model_state_sha256(model)
    rng = trainer._attended_session_rng_state(device=torch.device('cpu'))
    state = {'schema_version': 'gx1_prefix_fresh_initial_state_v1', 'chronological_prefix': {'test': True},
        'model_forwards': 0, 'optimizer_steps': 0, 'checkpoint_loaded': False,
        'model_state': copy.deepcopy(model.state_dict()), 'target_model_state': copy.deepcopy(model.state_dict()),
        'optimizer_state': optimizer.state_dict(), 'weight_ema_state': ema.checkpoint_state(),
        'lr_scheduler_state': scheduler.state_dict(), 'rng_state': rng}
    path = tmp_path / 'INITIAL_STATE.pt'; torch.save(state, path)
    components = {'model': model, 'optimizer': optimizer, 'weight_ema': ema, 'lr_scheduler': scheduler,
        'chronological_prefix': {'test': True}, 'weight_ema_derivation': {'test': True},
        'seed_binding': {'model_state_sha256': model_hash}}
    initial = {'initial_state': _bind(path), 'online_model_state_sha256': model_hash,
               'ema_derivation': {'test': True}, 'seed_binding': {'seed': 20260911}}
    with torch.no_grad(): model.weight.add_(3.)
    torch.rand(3)
    runner._restore_prefix_initial_measurement_state(components=components, scope={'initialization': initial},
                                                    device=torch.device('cpu'))
    assert trainer._model_state_sha256(model) == model_hash
    equal_tree(optimizer.state_dict(), state['optimizer_state'])
    equal_tree(ema.checkpoint_state(), state['weight_ema_state'])
    equal_tree(scheduler.state_dict(), state['lr_scheduler_state'])
    equal_tree(trainer._attended_session_rng_state(device=torch.device('cpu')), rng)


@pytest.mark.parametrize('fail_control', [False, True])
def test_initial_measurement_preserves_native_state_even_on_partial_failure(tmp_path, monkeypatch, fail_control):
    monkeypatch.setattr(trainer, "_copy_frozen_prefix_reference_model", lambda m: copy.deepcopy(m).eval().requires_grad_(False))
    artifacts = tmp_path / 'artifacts'; artifacts.mkdir()
    h = PrefixHarness(tmp_path / 'run', _prepared(artifacts))
    output = h.root / 'MEASURED'
    model, pause = h.run_prefix(output, 0)
    initial_hash = trainer._model_state_sha256(model)
    pointer = h.pointer(output).read_bytes()
    saved = h.state(output)
    rng = trainer._attended_session_rng_state(device=torch.device('cpu'))
    components = {**h.last_kwargs, 'train_probe_ds': object()}
    scope_value = {'initialization': {'online_model_state_sha256': initial_hash},
        'measurement': {'coordinate_result': {'test': True}},
        'artifacts': {'initialization_result': {'test': True}, 'measurement_binding_result': {'test': True}}}
    monkeypatch.setattr(cohorts, 'build_chronological_measurement_cohort', lambda *a, role:
        {'role': role, 'parent_entry_row_indices': list(range(256)), 'entry_row_indices': list(range(256))})
    seen = []
    def measure(**kw):
        role = kw['evaluation_cohort']['role']; seen.append(role)
        assert not model.training and not kw['candidate_target_model'].training
        assert kw['candidate_target_model'] is kw['exit_boundary_model']
        assert trainer._model_state_sha256(kw['candidate_target_model']) == initial_hash
        assert not any(p.requires_grad for p in kw['candidate_target_model'].parameters())
        assert kw['dataset'] is components['train_probe_ds' if role == 'train' else 'val_ds']
        torch.rand(5)
        if role == 'control' and fail_control: raise RuntimeError('injected control failure')
        return None, {'bounded_entry_observations': [{}]*256, 'bounded_exit_anchor_observations': [{}]*256,
                      'bounded_exit_sampled_observations': [{}]*1024}, None
    monkeypatch.setattr(runner.val, '_entry_representations', measure)
    kwargs = dict(components=components, scope=scope_value, recipe={'chronological_prefix': h.data.value},
        output=output, device=torch.device('cpu'), invocation_started=time.monotonic(), pause_evidence=pause)
    if fail_control:
        with pytest.raises(RuntimeError, match='injected control failure'): runner._run_prefix_initial_measurement(**kwargs)
    else:
        binding = runner._run_prefix_initial_measurement(**kwargs)
        result = json.loads(Path(binding['path']).read_text())
        assert result['optimizer_steps'] == 0 and result['teacher_refreshed'] is False
        assert set(result['observations']) == {'train','control'}
    assert seen == ['train','control'] and model.training
    assert h.pointer(output).read_bytes() == pointer and trainer._model_state_sha256(model) == initial_hash
    equal_tree(h.state(output), saved)
    equal_tree(trainer._attended_session_rng_state(device=torch.device('cpu')), rng)
    assert h.batches == [] and h.validation_batches == 0
    with pytest.raises(RuntimeError, match='MEASUREMENT_EXISTS'): runner._run_prefix_initial_measurement(**kwargs)
