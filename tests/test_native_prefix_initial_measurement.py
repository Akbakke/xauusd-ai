"""Initial native measurement admission and state preservation; synthetic only."""
import copy
import json
import time
from pathlib import Path

import pytest
import torch

from gx1.contracts import unified_exit_bounded_val_cohort_v1 as cohorts
from gx1.contracts import entry_model_native_train_launch_v1 as budgets
from tests.test_native_prefix_recipe import prefix_scope, scope, native, runner, _bind, _write, dynamic_native_recipe_metadata
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
        'seed_binding': {'model_state_sha256': model_hash},
        'model_functions': dict(trainer._PREFIX_MODEL_FUNCTIONS)}
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


@pytest.mark.parametrize('train_only,fail_control', [(False,False),(False,True),(True,False)])
def test_initial_measurement_preserves_native_state_even_on_partial_failure(tmp_path, monkeypatch, train_only, fail_control):
    monkeypatch.setattr(trainer, "_copy_frozen_prefix_reference_model", lambda m, **kw: copy.deepcopy(m).eval().requires_grad_(False))
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
    measurement_recipe = {'chronological_prefix': h.data.value}
    if train_only: measurement_recipe['chronological_train_only_measurement'] = True
    kwargs = dict(components=components, scope=scope_value, recipe=measurement_recipe,
        output=output, device=torch.device('cpu'), invocation_started=time.monotonic(), pause_evidence=pause)
    if fail_control:
        with pytest.raises(RuntimeError, match='injected control failure'): runner._run_prefix_initial_measurement(**kwargs)
    else:
        binding = runner._run_prefix_initial_measurement(**kwargs)
        result = json.loads(Path(binding['path']).read_text())
        assert result['optimizer_steps'] == 0 and result['teacher_refreshed'] is False
        assert set(result['observations']) == ({'train'} if train_only else {'train','control'})
    assert seen == (['train'] if train_only else ['train','control']) and model.training
    assert h.pointer(output).read_bytes() == pointer and trainer._model_state_sha256(model) == initial_hash
    equal_tree(h.state(output), saved)
    equal_tree(trainer._attended_session_rng_state(device=torch.device('cpu')), rng)
    assert h.batches == [] and h.validation_batches == 0
    with pytest.raises(RuntimeError, match='MEASUREMENT_EXISTS'): runner._run_prefix_initial_measurement(**kwargs)


@pytest.mark.parametrize('fault', [None, 'false', 'unbound', 'mixed_learning', 'diagnostic'])
def test_initial_train_only_is_bound_and_cannot_enable_updates(initial_scope, fault):
    policy, recipe, _, seal = initial_scope
    recipe['chronological_train_only_measurement'] = False if fault == 'false' else True
    if fault != 'unbound':
        policy['chronological_initial_measurement']['chronological_train_only_measurement'] = True
    if fault == 'mixed_learning': recipe['chronological_learning_measurement'] = {}
    if fault == 'diagnostic': recipe['entry_gradient_diagnostic'] = {}
    seal()
    if fault:
        with pytest.raises(RuntimeError): native.require_native_run_scope(recipe, invocation_number=1)
    else:
        assert native.require_native_run_scope(recipe, invocation_number=1) == 0
        budget = {'stop_after_optimizer_steps':0, 'stop_after_completed_val_epochs':None,
                  'max_invocation_seconds':12000, 'expected_active_pointer_sha256':None}
        assert native.require_native_run_scope(recipe, invocation_number=1, execution_budget=budget) == 0
        with pytest.raises(RuntimeError, match='INITIAL_BUDGET'):
            native.require_native_run_scope(recipe, execution_budget={**budget, 'stop_after_optimizer_steps':1})



@pytest.mark.parametrize("fault", [None, "initial_missing", "state_missing", "state_legacy", "unknown", "component_missing"])
def test_current_initial_restore_binds_function_even_with_identical_weight_hashes(tmp_path, fault):
    torch.manual_seed(20260911)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0001)
    ema = trainer._WeightEma(model, .5)
    functions = dict(trainer._PREFIX_CURRENT_MODEL_FUNCTIONS)
    model_hash = trainer._model_state_sha256(model)
    state = {
        "schema_version": "gx1_prefix_fresh_initial_state_v1",
        "chronological_prefix": {"test": True}, "model_functions": functions,
        "model_forwards": 0, "optimizer_steps": 0, "checkpoint_loaded": False,
        "model_state": copy.deepcopy(model.state_dict()),
        "target_model_state": copy.deepcopy(model.state_dict()),
        "optimizer_state": optimizer.state_dict(), "weight_ema_state": ema.checkpoint_state(),
        "lr_scheduler_state": None, "rng_state": trainer._attended_session_rng_state(device=torch.device("cpu")),
    }
    if fault == "state_missing":
        state.pop("model_functions")
    elif fault == "state_legacy":
        state["model_functions"] = dict(trainer._PREFIX_MODEL_FUNCTIONS)
    path = tmp_path / "INITIAL_STATE.pt"
    torch.save(state, path)
    initial = {
        "initial_state": _bind(path), "online_model_state_sha256": model_hash,
        "ema_derivation": {"test": True}, "seed_binding": {"seed": 20260911},
        "model_functions": functions,
    }
    if fault == "initial_missing":
        initial.pop("model_functions")
    components = {
        "model": model, "optimizer": optimizer, "weight_ema": ema, "lr_scheduler": None,
        "chronological_prefix": {"test": True}, "weight_ema_derivation": {"test": True},
        "seed_binding": {"model_state_sha256": model_hash},
        "model_functions": functions if fault != "unknown" else {"online": "unknown", "target": "unknown"},
    }
    if fault == "component_missing":
        components.pop("model_functions")
    if fault is None:
        runner._restore_prefix_initial_measurement_state(
            components=components, scope={"initialization": initial}, device=torch.device("cpu"))
    else:
        with pytest.raises(RuntimeError, match="INITIAL_FUNCTION_BINDING_INVALID"):
            runner._restore_prefix_initial_measurement_state(
                components=components, scope={"initialization": initial}, device=torch.device("cpu"))
    assert trainer._model_state_sha256(model) == model_hash



def test_measurement_rejects_function_drift_before_teacher_copy_or_forward(tmp_path, monkeypatch):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    h = PrefixHarness(tmp_path / "run", _prepared(artifacts))
    output = h.root / "FUNCTION_DRIFT"
    model, pause = h.run_prefix(output, 0)
    original_pointer = h.pointer(output).read_bytes()
    original_hash = trainer._model_state_sha256(model)
    functions = dict(trainer._PREFIX_CURRENT_MODEL_FUNCTIONS)
    monkeypatch.setattr(trainer, "_prefix_reference_model_functions", lambda design: functions)

    def forbidden(*args, **kwargs):
        raise AssertionError("no teacher copy before exact function admission")

    monkeypatch.setattr(trainer, "_copy_frozen_prefix_reference_model", forbidden)
    scope_value = {
        "initialization": {"online_model_state_sha256": original_hash, "model_functions": functions},
    }
    with pytest.raises(RuntimeError, match="MEASUREMENT_FUNCTION_BINDING_INVALID"):
        runner._run_prefix_initial_measurement(
            components={**h.last_kwargs, "train_probe_ds": object()},
            scope=scope_value, recipe={"chronological_prefix": h.data.value},
            output=output, device=torch.device("cpu"), invocation_started=time.monotonic(),
            pause_evidence=pause,
        )
    assert h.pointer(output).read_bytes() == original_pointer
    assert trainer._model_state_sha256(model) == original_hash
    assert not (Path(pause["session_directory"]) / "initial_measurement").exists()


@pytest.mark.parametrize("fault", [
    "learning_rate", "weight_decay", "betas", "parameter_order",
    "scheduler_position", "scheduler_horizon", "scheduler_presence",
    "ema_decay", "ema_schema",
])
def test_fresh_restore_rejects_changed_training_controls_before_any_mutation(tmp_path, fault):
    torch.manual_seed(20260911)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0001, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    ema = trainer._WeightEma(model, .5)
    functions = dict(trainer._PREFIX_CURRENT_MODEL_FUNCTIONS)
    initial_hash = trainer._model_state_sha256(model)
    state = {"schema_version":"gx1_prefix_fresh_initial_state_v1",
        "chronological_prefix":{"synthetic":True}, "model_functions":functions,
        "model_forwards":0, "optimizer_steps":0, "checkpoint_loaded":False,
        "model_state":copy.deepcopy(model.state_dict()), "target_model_state":copy.deepcopy(model.state_dict()),
        "optimizer_state":copy.deepcopy(optimizer.state_dict()),
        "lr_scheduler_state":copy.deepcopy(scheduler.state_dict()),
        "weight_ema_state":copy.deepcopy(ema.checkpoint_state()),
        "rng_state":trainer._attended_session_rng_state(device=torch.device("cpu"))}
    group = state["optimizer_state"]["param_groups"][0]
    if fault == "learning_rate": group["lr"] *= 10
    elif fault == "weight_decay": group["weight_decay"] *= 10
    elif fault == "betas": group["betas"] = (.5, .9)
    elif fault == "parameter_order": group["params"].reverse()
    elif fault == "scheduler_position": state["lr_scheduler_state"]["last_epoch"] += 1
    elif fault == "scheduler_horizon": state["lr_scheduler_state"]["T_max"] += 1
    elif fault == "scheduler_presence": state["lr_scheduler_state"] = None
    elif fault == "ema_decay": state["weight_ema_state"]["decay"] = .75
    else: state["weight_ema_state"]["unused"] = True
    path = tmp_path/"INITIAL_STATE.pt"; torch.save(state,path)
    components = {"model":model, "optimizer":optimizer, "weight_ema":ema, "lr_scheduler":scheduler,
        "chronological_prefix":state["chronological_prefix"], "model_functions":functions,
        "seed_binding":{"model_state_sha256":initial_hash}, "weight_ema_derivation":{"synthetic":True}}
    initial = {"initial_state":_bind(path), "online_model_state_sha256":initial_hash,
        "model_functions":functions, "ema_derivation":{"synthetic":True}, "seed_binding":{"seed":20260911}}
    # A partial restore would visibly mutate the model before detecting a later mismatch.
    with torch.no_grad(): model.weight.add_(3.)
    torch.rand(5)
    before_model = copy.deepcopy(model.state_dict())
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    before_scheduler = copy.deepcopy(scheduler.state_dict())
    before_ema = copy.deepcopy(ema.checkpoint_state())
    before_rng = trainer._attended_session_rng_state(device=torch.device("cpu"))
    with pytest.raises(RuntimeError, match="NATIVE_PREFIX_INITIAL_(OPTIMIZER|SCHEDULER|EMA)"):
        runner._restore_prefix_initial_measurement_state(
            components=components, scope={"initialization":initial}, device=torch.device("cpu"))
    equal_tree(model.state_dict(),before_model)
    equal_tree(optimizer.state_dict(),before_optimizer)
    equal_tree(scheduler.state_dict(),before_scheduler)
    equal_tree(ema.checkpoint_state(),before_ema)
    equal_tree(trainer._attended_session_rng_state(device=torch.device("cpu")),before_rng)


@pytest.mark.parametrize("fault", [None, "missing", "stale_digest", "changed_target_source", "old_measurement", "changed_source_bytes"])
def test_physical_initial_observation_binds_complete_recipe_source_not_only_model(tmp_path, fault):
    model = tmp_path/"model.py"; model.write_text("synthetic model unchanged")
    targets = tmp_path/"targets.py"; targets.write_text("synthetic reference owner")
    sources = {"python:model.py":budgets.artifact_binding(model), "python:targets.py":budgets.artifact_binding(targets)}
    recipe = {"chronological_prefix":{"native_coordinates":{"synthetic":True}},
              "source_bindings":sources,"source_bindings_sha256":budgets.canonical_json_sha256(sources)}
    observation = native.require_prefix_measurement_source_binding(recipe)
    if fault == "missing": recipe.pop("source_bindings")
    elif fault == "stale_digest": recipe["source_bindings_sha256"] = "0"*64
    elif fault == "changed_target_source":
        targets.write_text("changed target implementation with identical model weights")
        sources["python:targets.py"] = budgets.artifact_binding(targets)
        recipe["source_bindings_sha256"] = budgets.canonical_json_sha256(sources)
    elif fault == "old_measurement": observation.pop("native_recipe_source_bindings")
    elif fault == "changed_source_bytes": targets.write_text("bytes changed without resealing")
    if fault:
        with pytest.raises(RuntimeError):
            native.require_prefix_measurement_source_binding(recipe,measurement=observation)
    else:
        assert native.require_prefix_measurement_source_binding(recipe,measurement=observation) == observation
        assert observation["native_recipe_source_bindings"] == sources
    assert native.require_prefix_measurement_source_binding(
        {"chronological_prefix":{"design":{"legacy":True}}}, measurement={}) == {}


@pytest.fixture
def dynamic_initial_observer(tmp_path, monkeypatch):
    import hashlib
    import numpy as np
    import pandas as pd
    from tests.test_entry_observed_market_native import _dynamic_training_fixture, SmallDynamicEntry
    from gx1.contracts.entry_causal_m1_outcomes_v1 import DYNAMIC_M1_PREFIX_HORIZON_MINUTES
    from tests.test_entry_dynamic_m5_loss_scale import _synthetic_loaded_loss_scales
    loader, template = _dynamic_training_fixture(monkeypatch, tmp_path,
        loss_scales=_synthetic_loaded_loss_scales(tmp_path))
    dataset = loader.dataset
    clock = pd.date_range("2025-12-01T00:00Z", periods=300, freq="5min").as_unit("ns")
    dataset.df = pd.DataFrame({"time": clock, "dynamic_entry_m1_row": np.arange(300, dtype=np.int64) * 5})
    dataset.indices = np.arange(300, dtype=np.int64)
    dataset.parquet_path = tmp_path / "entry.parquet"
    dataset.df.to_parquet(dataset.parquet_path, index=False)
    source = {"parquet": _bind(dataset.parquet_path), "manifest": {"path": "synthetic", "sha256": "a"*64},
        "physical_rows": 300, "clock_sha256": hashlib.sha256(np.asarray(clock.asi8, dtype="<i8").tobytes()).hexdigest()}
    dataset._entry_observed_market_binding.update(physical_source=source, role="TRAIN")
    identity = dataset._entry_observed_market_binding["identity"]
    parents = list(range(255, -1, -1))
    cohort = {"training_phase": identity["training_phase"], "exit_evaluation": False,
        "entry_row_indices": parents, "parent_entry_row_indices": parents, "measurement_role": "train",
        "source_split": "train", "parent_entry_parquet": source["parquet"], "parent_entry_manifest": source["manifest"],
        "population_rows": 300, "source_clock_sha256": source["clock_sha256"], "entry_time_ns": clock.asi8[parents].tolist()}
    monkeypatch.setattr(cohorts, "require_bounded_val_cohort", lambda value: value)
    def bound_loader(observed, **kwargs):
        assert observed is dataset and kwargs["batch_size"] == 16 and kwargs["num_workers"] == 0
        class Loader:
            def __iter__(self):
                for offset in range(0, len(parents), 16):
                    rows = torch.tensor(parents[offset:offset+16])
                    batch = {k:torch.cat([v]*8) for k,v in template.items()}
                    batch.update(entry_row_index=rows, dynamic_entry_parent_row_index=rows,
                        dynamic_entry_m1_row=rows*5)
                    yield batch
        return Loader()
    monkeypatch.setattr(torch.utils.data, "DataLoader", bound_loader)
    class Model(SmallDynamicEntry):
        def __init__(self):
            super().__init__()
            self.forwards = 0
        def forward(self, *args, **kwargs):
            self.forwards += 1
            out = super().forward(*args, **kwargs)
            quality = out["dynamic_terminal_net_markout_bps"][..., :2] - out["dynamic_pre_mfe_mae_upper_bps"]
            out["dynamic_entry_quality_best_horizon_mask"] = quality == quality.amax(dim=1, keepdim=True)
            out["dynamic_query_horizon_minutes"] = torch.tensor(DYNAMIC_M1_PREFIX_HORIZON_MINUTES)
            return out
    model = Model().eval()
    with torch.no_grad():
        model.head_dynamic_terminal_net_markout.weight.zero_()
        model.head_dynamic_terminal_net_markout.bias.fill_(-2)
    context = {"physical_sources": {"train": source}, "entry_supervision_identity": identity,
        "max_model_forwards": 16, "max_state_views": 256}
    return model, dataset, cohort, context


def test_dynamic_initial_observer_reuses_one_provider_and_encoder_forward_per_batch(dynamic_initial_observer):
    model, dataset, cohort, context = dynamic_initial_observer
    before = copy.deepcopy(model.state_dict())
    result = runner._dynamic_m5_prefix_observations(model=model, dataset=dataset, cohort=cohort,
        context=context, device=torch.device("cpu"), deadline=time.monotonic()+30)
    assert model.forwards == result["entry_model_forward_count"] == 16
    calls = dataset._entry_dynamic_m5_target_provider.calls
    assert len(calls) == 16 and [int(row) for batch in calls for row in batch] == cohort["parent_entry_row_indices"]
    errors = result["dynamic_component_errors"]
    assert errors["dynamic_terminal_net_markout_bps"]["observed_cells"] == 384
    assert errors["dynamic_mfe_bps"]["observed_cells"] == errors["dynamic_closed_mfe_bps"]["observed_cells"] == 256
    assert errors["dynamic_pre_mfe_mae_upper_bps"]["observed_cells"] == 256
    assert errors["dynamic_closed_pre_mfe_mae_upper_bps"]["observed_cells"] == 256
    traces = result["dynamic_entry_observations"]
    assert len(traces) == 256 and all(row["diagnostic_action"] == "FLAT" and row["diagnostic_abstained"] for row in traces)
    assert all(len(row["side_diagnostics"]) == 2 for row in traces)
    assert result["exit_model_forward_count"] == 0 and result["realized_cash_or_native_policy_evaluated"] is False
    assert result["dynamic_frozen_targets"]["parent_entry_rows"] == cohort["parent_entry_row_indices"]
    equal_tree(model.state_dict(), before)


@pytest.mark.parametrize("fault", ["source", "clock", "lifecycle", "budget", "deadline"])
def test_dynamic_initial_observer_rejects_source_or_compute_drift_before_forward(dynamic_initial_observer, fault):
    model, dataset, cohort, context = dynamic_initial_observer
    deadline = time.monotonic()+30
    if fault == "source": dataset.parquet_path = Path("wrong.parquet")
    elif fault == "clock": cohort["source_clock_sha256"] = "0"*64
    elif fault == "lifecycle": dataset._unified_exit_lifecycle_v2 = object()
    elif fault == "budget": context["max_model_forwards"] = 15
    else: deadline = time.monotonic()-1
    with pytest.raises(RuntimeError):
        runner._dynamic_m5_prefix_observations(model=model, dataset=dataset, cohort=cohort,
            context=context, device=torch.device("cpu"), deadline=deadline)
    assert model.forwards == 0 and dataset._entry_dynamic_m5_target_provider.calls == []


@pytest.mark.parametrize("fault", [None, "target", "mask", "partial_failure"])
def test_dynamic_initial_final_measurement_freezes_targets_and_restores_native_state(tmp_path, monkeypatch, fault):
    from types import SimpleNamespace
    artifacts = tmp_path / "artifacts"; artifacts.mkdir()
    h = PrefixHarness(tmp_path / "run", _prepared(artifacts))
    output = h.root / "DYNAMIC_MEASURED"
    model, pause = h.run_prefix(output, 0)
    initial_hash = trainer._model_state_sha256(model)
    state = h.state(output)
    pointer = h.pointer(output).read_bytes()
    phase = "entry_only_dynamic_m5_path"
    model.entry_training_phase = phase
    identity = {"training_phase": phase}
    functions = dict(trainer._PREFIX_CURRENT_MODEL_FUNCTIONS)
    from tests.test_entry_dynamic_m5_loss_scale import _synthetic_loaded_loss_scales
    from gx1.contracts.entry_dynamic_m5_loss_scale_v1 import dynamic_m5_loss_scale_identity
    loss_units = dynamic_m5_loss_scale_identity(_synthetic_loaded_loss_scales(tmp_path))
    prefix = {"entry_training_phase": phase, "model_functions": functions,
        "dynamic_m5_loss_scales": loss_units,
        "entry_observed_market": {role:{"identity":identity} for role in ("train", "control")}}
    context = {"entry_training_phase": phase, "entry_supervision_identity": identity, "max_wall_seconds": 30}
    context_binding = {"synthetic": "dynamic context"}
    fake_session = SimpleNamespace(_active_path=h.pointer(output),
        _contract={"chronological_prefix":prefix, "native_bounded_control":context_binding}, load_checkpoint=lambda:state)
    monkeypatch.setattr(trainer, "_CandidateTrainingSession", lambda **kw: fake_session)
    monkeypatch.setattr(trainer, "_prefix_reference_model_functions", lambda design:functions)
    monkeypatch.setattr(trainer, "_native_candidate_val_context_binding", lambda value:context_binding)
    def forbidden(*args, **kwargs): raise AssertionError("dynamic observation used Exit/teacher")
    monkeypatch.setattr(trainer, "_copy_frozen_prefix_reference_model", forbidden)
    monkeypatch.setattr(runner.val, "_entry_representations", forbidden)
    monkeypatch.setattr(cohorts, "build_chronological_measurement_cohort", lambda *args, role:
        {"measurement_role":role, "parent_entry_row_indices":list(range(256)), "entry_row_indices":list(range(256))})
    calls = []
    def observe(**kwargs):
        role = kwargs["cohort"]["measurement_role"]; calls.append(role)
        torch.rand(5)
        if fault == "partial_failure" and role == "control": raise RuntimeError("dynamic control failure")
        return {"entry_model_forward_count":16,
            "dynamic_frozen_targets":{"target_sha":"changed" if fault == "target" and state["global_optimizer_steps"] else "target",
                "mask_sha":"changed" if fault == "mask" and state["global_optimizer_steps"] else "mask"},
            "dynamic_entry_observations":[{"prediction":float(model.weight[0,0])}]}
    monkeypatch.setattr(runner, "_dynamic_m5_prefix_observations", observe)
    ds = SimpleNamespace(_entry_observed_market_binding={"identity":identity})
    components = {**h.last_kwargs, "train_probe_ds":ds, "val_ds":ds, "native_val_context":context,
        "entry_observed_market":identity, "entry_only_session_binding":prefix,
        "dynamic_m5_loss_scales":loss_units}
    scope_value = {"initialization":{"online_model_state_sha256":initial_hash,"model_functions":functions},
        "measurement":{"coordinate_result":{"synthetic":True}},
        "artifacts":{"initialization_result":{"synthetic":True},"measurement_binding_result":{"synthetic":True}}}
    recipe = {"chronological_prefix":h.data.value,"entry_observed_market":{"synthetic":True}}
    rng = trainer._attended_session_rng_state(device=torch.device("cpu"))
    kwargs = dict(components=components,scope=scope_value,recipe=recipe,output=output,
        device=torch.device("cpu"),invocation_started=time.monotonic(),pause_evidence=pause)
    if fault == "partial_failure":
        with pytest.raises(RuntimeError,match="dynamic control failure"): runner._run_prefix_initial_measurement(**kwargs)
    else:
        initial = runner._run_prefix_initial_measurement(**kwargs)
        scope_value["initial_measurement"] = json.loads(Path(initial["path"]).read_text())
        scope_value["artifacts"].update(initial_measurement_result=initial,initial_measurement_audit={"synthetic":True})
        with torch.no_grad(): model.weight.add_(.25)
        state.update(global_optimizer_steps=256,model_state=copy.deepcopy(model.state_dict()),
            optimizer_state={"state":{0:{}}},weight_ema_state={"steps":256})
        final_pause = {**pause,"next_batch_offset":256,"global_optimizer_steps":256}
        if fault:
            with pytest.raises(RuntimeError,match="FINAL_FROZEN_TARGET_CHANGED"):
                runner._run_prefix_initial_measurement(**{**kwargs,"pause_evidence":final_pause},optimizer_steps=256)
        else:
            final = runner._run_prefix_initial_measurement(**{**kwargs,"pause_evidence":final_pause},optimizer_steps=256)
            report = json.loads(Path(final["path"]).read_text())
            assert report["frozen_targets_exactly_preserved"] is True and report["exit_evaluation"] is False
            assert report["target_model_role"] == "frozen_constructor_anchor_no_forward"
    assert h.pointer(output).read_bytes() == pointer and model.training
    equal_tree(trainer._attended_session_rng_state(device=torch.device("cpu")),rng)


@pytest.fixture
def dynamic_initial_scope(dynamic_native_recipe_metadata, tmp_path, monkeypatch):
    case = dynamic_native_recipe_metadata
    policy, recipe = case["policy"], case["recipe"]
    policy.pop("chronological_learning_run")
    state = tmp_path/"DYNAMIC_INITIAL_STATE.pt"; state.write_bytes(b"unit state; real restore is separately tested")
    closure = native.require_prefix_measurement_source_binding(recipe)
    common = {"entry_training_phase":case["witness"]["training_phase"],
        "dynamic_m5_loss_scales":case["loss_scale_identity"],
        "target_model_role":"frozen_constructor_anchor_no_forward", "exit_evaluation":False,
        "native_launch_authorized":False, "exit_model_forward_count":0, **closure}
    initial = {"schema_version":"gx1_prefix_fresh_initialization_v1",
        "decision":"ACTUAL_NATIVE_FRESH_COMPONENTS_READY_NO_LEARNING_MEASURED", **common,
        "chronological_prefix":recipe["chronological_prefix"], "files":recipe["files"],
        "inherited_checkpoint_weights":False, "optimizer_state_empty":True, "target_frozen":True,
        "model_forwards":0, "optimizer_steps":0, "ema_steps":0, "normalization_refits":0,
        "online_model_state_sha256":"a"*64, "target_model_state_sha256":"a"*64, "ema_model_state_sha256":"a"*64,
        "initial_state":_bind(state), "test_data_used":False,
        "model_functions":{"online":"unit current ONLINE", "target":"unit current ONLINE"},
        "source_bindings":{key:value["sha256"] for key,value in recipe["source_bindings"].items()},
        "entry_only_base_normalization":case["physical"]["entry_only_base_normalization"],
        "base_contract_sha256":case["physical"]["base_contract_sha256"],
        "population_witness":recipe["entry_m5_path_single_epoch"]}
    measurement = {"schema_version":"gx1_prefix_native_measurement_binding_result_v1",
        "decision":"FROZEN_TRAIN_CONTROL_COORDINATES_AND_NATIVE_MEASUREMENT_BINDING_READY", **common,
        "actual_model_forwards":0, "optimizer_steps":0, "test_data_used":False,
        "coordinate_result":recipe["files"]["entry_measurement_coordinates"]}
    monkeypatch.setattr(cohorts, "build_chronological_measurement_cohort", lambda *args, role: {"role":role})
    artifacts = {"initialization_result":{}, "measurement_binding_result":{}}
    def seal():
        artifacts["initialization_result"] = _write(tmp_path/"dynamic-initial.json", initial)
        artifacts["measurement_binding_result"] = _write(tmp_path/"dynamic-measurement.json", measurement)
        recipe["chronological_initial_measurement"] = dict(artifacts)
        policy["chronological_initial_measurement"] = {**artifacts,
            "chronological_prefix":recipe["chronological_prefix"], "run_id":recipe["run_id"],
            "out_bundle_dir":recipe["out_bundle_dir"], "source_bindings_sha256":recipe["source_bindings_sha256"],
            "entry_m5_path_single_epoch":recipe["entry_m5_path_single_epoch"],
            "entry_observed_market":recipe["entry_observed_market"], "dynamic_m5_loss_scales":recipe["dynamic_m5_loss_scales"],
            "optimizer_steps":0, "max_invocations":1, "teacher_refresh_allowed":False,
            "full_epoch_training_allowed":False, "full_val_allowed":False, "test_data_used":False}
        return case["seal"]()
    seal()
    return {**case, "initial":initial, "measurement":measurement, "initial_artifacts":artifacts, "seal_initial":seal}


def test_dynamic_initial_scope_uses_current_entry_coordinates_and_zero_step_budget(dynamic_initial_scope):
    case = dynamic_initial_scope; recipe = case["recipe"]
    checked = native.require_chronological_initial_measurement(recipe, invocation_number=1,
        execution_budget={"stop_after_optimizer_steps":0, "expected_active_pointer_sha256":None,
                          "stop_after_completed_val_epochs":None, "max_invocation_seconds":12000})
    assert checked["initialization"]["population_witness"] == recipe["entry_m5_path_single_epoch"]
    assert checked["measurement"]["coordinate_result"] == recipe["files"]["entry_measurement_coordinates"]
    assert "design" not in json.loads(Path(checked["measurement"]["coordinate_result"]["path"]).read_text())


@pytest.fixture
def dynamic_initial_campaign_review(dynamic_initial_scope, tmp_path, monkeypatch):
    from gx1.contracts import local_random_access_campaign_v2 as campaign
    from tests.test_native_prefix_recipe import _cursor

    case = dynamic_initial_scope
    # Isolate receipt-chain transport; actual scope and cursor byte guards remain real.
    monkeypatch.setattr(campaign, "require_receipt_chain", lambda plan, receipts, **kw: receipts)

    def completed_cursor(*, steps=0, recipe_binding=None):
        binding = case["seal_initial"]() if recipe_binding is None else recipe_binding
        _, cursor = _cursor(tmp_path, binding, steps)
        plan = {"phase": "native_candidate", "checked_invocations": [{}], "native_recipe": binding}
        receipt = {"outcome": "RESUMABLE", "checkpoint_pointer_snapshot": cursor}
        return plan, [receipt]

    return case, campaign, completed_cursor


def test_dynamic_initial_campaign_with_m5_witness_stops_for_review_at_zero(
    dynamic_initial_campaign_review, monkeypatch,
):
    case, campaign, completed_cursor = dynamic_initial_campaign_review
    plan, receipts = completed_cursor()
    assert "entry_m5_path_single_epoch" in case["recipe"]
    real_scope = native.require_native_run_scope
    checked = []

    def scope(recipe, **kw):
        checked.append(recipe)
        return real_scope(recipe, **kw)

    def epoch_forbidden(*args, **kw):
        raise AssertionError("initial0 must not claim a completed learning epoch")

    monkeypatch.setattr(native, "require_native_run_scope", scope)
    monkeypatch.setattr(native, "require_m5_single_epoch_run", epoch_forbidden)
    assert campaign.next_action(plan, receipts, current_boot={}) == {
        "decision": "BLOCKED_NATIVE_LEARNING_REVIEW_REQUIRED"}
    assert checked == [case["recipe"]]


@pytest.mark.parametrize("fault", ["mixed_measurement", "training_enabled", "nonzero_cursor"])
def test_dynamic_initial_campaign_review_keeps_scope_and_zero_step_guards(
    dynamic_initial_campaign_review, fault,
):
    case, campaign, completed_cursor = dynamic_initial_campaign_review
    if fault == "mixed_measurement":
        case["recipe"]["chronological_learning_measurement"] = case["recipe"]["files"]["entry_measurement_coordinates"]
        reason = "NATIVE_M5_SINGLE_EPOCH_MIXED_OR_UNMEASURED_SCOPE"
    elif fault == "training_enabled":
        case["policy"]["training_enabled"] = True
        reason = "NATIVE_PREFIX_INITIAL_MEASUREMENT_NOT_AUTHORIZED"
    else:
        reason = "native prefix cursor exceeded fixed experiment"
    plan, receipts = completed_cursor(steps=1 if fault == "nonzero_cursor" else 0)
    with pytest.raises(RuntimeError, match=reason):
        campaign.next_action(plan, receipts, current_boot={})


@pytest.mark.parametrize("scope_rejected", [False, True])
def test_dynamic_campaign_technical_completion_requires_validated_scope_without_epoch_claim(
    dynamic_initial_campaign_review, tmp_path, monkeypatch, scope_rejected,
):
    case, campaign, completed_cursor = dynamic_initial_campaign_review
    recipe = copy.deepcopy(case["recipe"])
    recipe.pop("chronological_initial_measurement")
    recipe["entry_dynamic_m5_technical_qualification"] = {"unit_dispatch_only": True}
    ceiling = native.DYNAMIC_M5_TECHNICAL_CURSOR_CEILING
    plan, receipts = completed_cursor(steps=ceiling, recipe_binding=_write(tmp_path / "technical-review-recipe.json", recipe))
    checked = []

    def scope(value, **kw):
        checked.append(value)
        if scope_rejected:
            raise RuntimeError("synthetic strict technical scope rejected")
        return ceiling

    def epoch_forbidden(*args, **kw):
        raise AssertionError("technical qualification must not claim a completed learning epoch")

    # Only dispatch is tested here; the technical scope owner has separate strict fixtures.
    monkeypatch.setattr(native, "require_native_run_scope", scope)
    monkeypatch.setattr(native, "require_m5_single_epoch_run", epoch_forbidden)
    if scope_rejected:
        with pytest.raises(RuntimeError, match="strict technical scope rejected"):
            campaign.next_action(plan, receipts, current_boot={})
    else:
        assert campaign.next_action(plan, receipts, current_boot={}) == {
            "decision": "BLOCKED_NATIVE_LEARNING_REVIEW_REQUIRED"}
    assert checked == [recipe]


@pytest.mark.parametrize("fault", [None, "epoch_gate_rejects", "batch_offset"])
def test_dynamic_campaign_epoch_completion_keeps_full_epoch_gate(
    dynamic_initial_campaign_review, tmp_path, monkeypatch, fault,
):
    case, campaign, completed_cursor = dynamic_initial_campaign_review
    recipe = copy.deepcopy(case["recipe"])
    recipe.pop("chronological_initial_measurement")
    recipe["chronological_learning_measurement"] = recipe["files"]["entry_measurement_coordinates"]
    steps = case["witness"]["optimizer_steps"]
    plan, receipts = completed_cursor(steps=steps, recipe_binding=_write(tmp_path / "epoch-review-recipe.json", recipe))
    # Dispatch-only unit: prerequisite admission owners have their own strict tests.
    # This proves the completed epoch still reaches its full owner and offset check.
    monkeypatch.setattr(native, "require_native_run_scope", lambda value, **kw: steps)
    checked = []
    single = {"binding": recipe["entry_m5_path_single_epoch"], "train_rows": case["witness"]["train_rows"],
              "optimizer_steps": steps + (fault == "batch_offset")}

    def epoch(value):
        checked.append(value)
        if fault == "epoch_gate_rejects":
            raise RuntimeError("synthetic full epoch owner rejected admission")
        return {"entry_m5_path_single_epoch": single}

    monkeypatch.setattr(native, "require_m5_single_epoch_run", epoch)
    if fault is None:
        assert campaign.next_action(plan, receipts, current_boot={}) == {
            "decision": "BLOCKED_NATIVE_LEARNING_REVIEW_REQUIRED", "train_epochs_consumed": 1,
            "unique_train_rows": single["train_rows"], "population_witness": single["binding"]}
    else:
        reason = "full epoch owner rejected admission" if fault == "epoch_gate_rejects" else "M5 single epoch final batch offset differs"
        with pytest.raises(RuntimeError, match=reason):
            campaign.next_action(plan, receipts, current_boot={})
    assert checked == [recipe]


@pytest.mark.parametrize("fault", ["anchor", "exit", "normalization", "witness", "phase", "measurement_phase",
    "closure", "function", "coordinate", "zero_alias"])
def test_dynamic_initial_scope_rejects_unbound_constructor_or_exit_work(dynamic_initial_scope, fault):
    case = dynamic_initial_scope; initial, measurement = case["initial"], case["measurement"]
    if fault == "anchor": initial["target_model_role"] = "teacher"
    elif fault == "exit": measurement["exit_model_forward_count"] = 1
    elif fault == "normalization": initial["base_contract_sha256"] = "f"*64
    elif fault == "witness": initial["population_witness"] = case["recipe"]["files"]["source_bundle_metadata"]
    elif fault == "phase": initial["entry_training_phase"] = "joint_fitted_q"
    elif fault == "measurement_phase": measurement["entry_training_phase"] = "joint_fitted_q"
    elif fault == "closure": measurement["native_recipe_source_bindings_sha256"] = "f"*64
    elif fault == "function": initial["model_functions"]["target"] = "different"
    elif fault == "coordinate": measurement["coordinate_result"] = case["recipe"]["files"]["source_bundle_metadata"]
    elif fault == "zero_alias": measurement["actual_model_forwards"] = False
    case["seal_initial"]()
    with pytest.raises(RuntimeError): native.require_chronological_initial_measurement(case["recipe"])


@pytest.fixture
def dynamic_observation_admission_case(dynamic_initial_observer):
    model, dataset, cohort, context = dynamic_initial_observer
    diagnostics = runner._dynamic_m5_prefix_observations(model=model, dataset=dataset, cohort=cohort,
        context=context, device=torch.device("cpu"), deadline=time.monotonic()+30)
    observation = {"target_model_role":"frozen_constructor_anchor_no_forward", "exit_evaluation":False,
                   "native_launch_authorized":False, "diagnostics":diagnostics}
    return observation, cohort, context["entry_supervision_identity"]


@pytest.mark.parametrize("fault", [None, "hash", "mask_pair", "counts", "row", "pointer", "horizon", "candidate",
    "exit", "coverage", "error", "forwards"])
def test_dynamic_observation_admission_rejoins_actual_component_masks_and_trace(dynamic_observation_admission_case, fault):
    observation, cohort, identity = dynamic_observation_admission_case
    diagnostics = observation["diagnostics"]; frozen = diagnostics["dynamic_frozen_targets"]
    components = frozen["components"]; traces = diagnostics["dynamic_entry_observations"]
    if fault == "hash": components["dynamic_mfe_bps"]["target_storage_sha256"] = "bad"
    elif fault == "mask_pair": components["dynamic_mfe_bps"]["observed_mask_sha256"] = "f"*64
    elif fault == "counts": components["dynamic_mfe_bps"]["observed_cells"] += 1
    elif fault == "row": traces[0]["entry_row_index"] += 1
    elif fault == "pointer": frozen["dynamic_entry_m1_rows"][0] += 1
    elif fault == "horizon": traces[0]["side_diagnostics"][0]["predicted_best_horizon_minutes"] = [481]
    elif fault == "candidate": traces[0]["diagnostic_action"] = "LONG"
    elif fault == "exit": diagnostics["exit_model_forward_count"] = 1
    elif fault == "coverage": diagnostics["dynamic_candidate_diagnostic_coverage"]["abstained_rows"] -= 1
    elif fault == "error": diagnostics["dynamic_component_errors"]["dynamic_mfe_bps"]["sum_squared_error_bps2"] = float("nan")
    elif fault == "forwards": diagnostics["entry_model_forward_count"] = 17
    if fault:
        with pytest.raises(RuntimeError): native._require_dynamic_m5_observation(observation, identity=identity, cohort=cohort)
    else:
        assert native._require_dynamic_m5_observation(observation, identity=identity, cohort=cohort) == observation


@pytest.mark.parametrize("fault", ["recipe", "initial", "measurement", "policy"])
def test_dynamic_initial_scope_requires_same_loss_units_in_all_frozen_bindings(dynamic_initial_scope, fault):
    case = dynamic_initial_scope
    if fault == "recipe":
        case["recipe"]["dynamic_m5_loss_scales"] = {**case["recipe"]["dynamic_m5_loss_scales"], "sha256": "e" * 64}
    elif fault in ("initial", "measurement"):
        case[fault]["dynamic_m5_loss_scales"]["artifact_sha256"] = "e" * 64
    case["seal_initial"]()
    if fault == "policy":
        case["policy"]["chronological_initial_measurement"]["dynamic_m5_loss_scales"] = {
            **case["recipe"]["dynamic_m5_loss_scales"], "sha256": "e" * 64}
        case["seal"]()
    with pytest.raises(RuntimeError):
        native.require_chronological_initial_measurement(case["recipe"], execution_budget={
            "stop_after_optimizer_steps": 0, "expected_active_pointer_sha256": None,
            "stop_after_completed_val_epochs": None, "max_invocation_seconds": 12000})
