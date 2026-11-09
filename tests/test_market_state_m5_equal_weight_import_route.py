"""Synthetic metadata routing for the one-use zero-step M5 import."""
import copy
from pathlib import Path

import pytest

from gx1.contracts import entry_native_direction_research_campaign_v1 as outer
from gx1.contracts import local_random_access_campaign_v2 as campaign
from gx1.contracts import market_state_native_F_private_campaign_v1 as scope


def _ref(path):
    return {'path': str(path), 'sha256': 'a' * 64}


def _recipe(final_readout=False):
    stage = scope.TA_M5_FINAL_IMPORT_STAGE if final_readout else scope.TA_M5_IMPORT_STAGE
    _, runtime, _ = scope.ta_m5_import_paths(stage)
    return {
        'run_scope': scope.TA_M5_IMPORT_SCOPE, 'admission_stage': stage,
        'runtime_root': str(runtime), 'fold_id': 1,
        'numerical_recipe': {'batch_size': 16}, 'recipe_sha256': 'b' * 64,
        'training_enabled': False, 'test_data_used': False,
        'max_private_updates': 0, 'max_backwards': 0,
    }


def _gate(recipe_ref, final_readout=False):
    runtime = scope.TA_M5_FINAL_IMPORT_RUNTIME_ROOT if final_readout else scope.TA_M5_IMPORT_RUNTIME_ROOT
    return {
        'decision': 'ADMITTED', 'status': 'ADMITTED', 'ta_m5_import_authorized': True,
        'recipe': recipe_ref, 'run_root': str(runtime),
        'max_invocations': 1, 'claimed_invocations': 0, 'completed_invocations': 0,
        'relaunch_allowed': False, 'fold_id': 1, 'parent_selection': scope.TA_PARENT_SELECTION,
        'selected_fit_rows': 32, 'model_constructions_authorized': 3,
        'optimizer_constructions_authorized': 3, 'model_forwards_authorized': 6,
        'backward_calls_authorized': 0, 'optimizer_steps_authorized': 0,
        **{key: True for key in ('execution_authorized', 'data_reads_authorized',
                                'genuine_constructor_authorized', 'genuine_forward_authorized')},
        **{key: False for key in ('training_enabled', 'native_training_enabled',
            'genuine_backward_authorized', 'genuine_updates_authorized',
            'genuine_predictor_fit_authorized', 'native_launch_authorized',
            'normalization_fit_authorized', 'synthetic_test_execution_authorized',
            'test_data_used', 'exit_evaluation', 'broker_access_allowed')},
    }


def _window():
    runtime = scope.TA_M5_IMPORT_RUNTIME_ROOT
    return {'recipe': _ref(scope.TA_M5_IMPORT_RECIPE_PATH), 'policy_sha256': 'c' * 64,
        'progress_path': str(runtime / 'progress/invocation-0001.json'),
        'campaign_cursor_path': str(runtime / 'research-cursor/RESUME_CURSOR.json')}


def _invocation(window):
    policy_ref = _ref('/synthetic/WINDOW.json')
    return {
        'requires_fresh_windows_boot': True, 'kind': outer.KIND, 'batch_size': 16,
        'epoch_index': None, 'invocation_number': 1, 'window_index': 0, 'window_count': 1,
        'optimizer_step_budget': 0, 'expected_success_outcome': 'COMPLETE',
        'maximum_wall_seconds': 13800, 'invocation_sha256': 'd' * 64,
        'direction_research_recipe': window['recipe'], 'progress_path': window['progress_path'],
        'checkpoint': {'pointer_path': window['campaign_cursor_path'], 'before_mode': 'GENESIS',
                       'predecessor_invocation_number': None, 'write_mode': 'ATOMIC_UPDATE'},
        'launcher_argv': ['guard', '--', str(outer.CANONICAL_REPO / '.venv/bin/python'),
            '-m', outer.MODULE, '--window-policy', policy_ref['path'],
            '--window-policy-file-sha256', policy_ref['sha256'],
            '--progress-path', window['progress_path']],
    }


def test_import_recipe_requires_both_exact_scope_and_exact_path(monkeypatch):
    value = _recipe()
    monkeypatch.setattr(campaign, 'require_binding', lambda value, **kwargs: value)
    monkeypatch.setattr(campaign, 'read_bound_json', lambda *args: value)
    calls = []
    def checked(recipe, **kwargs):
        calls.append((recipe, kwargs))
        return recipe
    monkeypatch.setattr(scope, 'require_ta_m5_import_recipe', checked)
    args = {'source_repo': outer.CANONICAL_REPO, 'source_commit': 'e' * 40}
    assert outer.require_research_launch_recipe(_ref(scope.TA_M5_IMPORT_RECIPE_PATH), **args) is value
    assert len(calls) == 1 and calls[0][1]['verify_files'] is True
    with pytest.raises(Exception, match='M5_IMPORT_RECIPE_PATH'):
        outer.require_research_launch_recipe(_ref('/synthetic/other/RECIPE.json'), **args)
    value['run_scope'] = outer.TA_M5_MECHANICAL_SCOPE
    with pytest.raises(Exception, match='M5_IMPORT_RECIPE_PATH'):
        outer.require_research_launch_recipe(_ref(scope.TA_M5_IMPORT_RECIPE_PATH), **args)
    assert len(calls) == 1


def test_import_cursor_and_progress_report_zero_without_weakening_legacy():
    ref, artifact = _ref(scope.TA_M5_IMPORT_RECIPE_PATH), _ref('/synthetic/state.pt')
    cursor_args = dict(recipe_ref=ref, outcome='COMPLETE', optimizer_updates=0,
        parent_cursor=0, parent_order_sha256='e' * 64, full_state=artifact,
        result=_ref('/synthetic/RESULT.json'), batch_size=16)
    cursor = outer.build_research_cursor(**cursor_args)
    assert cursor['schema_version'] == scope.TA_M5_IMPORT_CURSOR_SCHEMA
    for field in ('optimizer_updates', 'parent_cursor'):
        with pytest.raises(Exception, match='M5_IMPORT_CURSOR_SCOPE'):
            outer.build_research_cursor(**{**cursor_args, field: 1})
    with pytest.raises(Exception, match='M5_IMPORT_RECIPE_PATH'):
        outer.require_research_cursor(cursor, expected_recipe=_ref('/synthetic/legacy.json'),
                                      verify_files=False, batch_size=16)
    with pytest.raises(Exception, match='CURSOR_COMPLETE'):
        outer.build_research_cursor(**{**cursor_args, 'recipe_ref': _ref('/synthetic/legacy.json')})
    window = _window()
    invocation = _invocation(window)
    args = dict(window=window, plan={'plan_sha256': 'f' * 64}, invocation=invocation,
        checkpoint_pointer=_ref(window['campaign_cursor_path']), optimizer_updates=0,
        parent_cursor=0, parent_order_sha256='e' * 64, outcome='COMPLETE')
    progress = outer.build_research_progress(**args)
    assert all(progress[key] == 0 for key in
               ('completed_units', 'total_units', 'optimizer_updates', 'parent_cursor'))
    for field in ('optimizer_updates', 'parent_cursor'):
        with pytest.raises(Exception, match='M5_IMPORT_PROGRESS_ZERO'):
            outer.build_research_progress(**{**args, field: 1})
    invocation['optimizer_step_budget'] = 4
    with pytest.raises(Exception, match='M5_IMPORT_PROGRESS_ZERO'):
        outer.build_research_progress(**args)


@pytest.mark.parametrize('fault', [None, 'budget', 'boot'])
def test_bound_import_invocation_requires_zero_updates_and_fresh_boot(monkeypatch, fault):
    window, recipe = _window(), _recipe()
    invocation = _invocation(window)
    if fault == 'budget':
        invocation['optimizer_step_budget'] = 4
    elif fault == 'boot':
        invocation['requires_fresh_windows_boot'] = False
    monkeypatch.setattr(campaign, 'require_binding', lambda value, **kwargs: value)
    monkeypatch.setattr(campaign, 'read_bound_json', lambda *args: window)
    monkeypatch.setattr(outer, 'require_research_launch_recipe', lambda *args, **kwargs: recipe)
    monkeypatch.setattr(outer, 'require_research_window_policy', lambda value: value)
    execution = {'prelaunch_manifest': window['recipe'],
        'train_session_manifest': _ref('/synthetic/WINDOW.json'),
        'prelaunch_manifest_sha256': recipe['recipe_sha256'],
        'train_session_manifest_sha256': window['policy_sha256']}
    def check():
        outer.require_research_invocation_execution(invocation, execution,
            source_repo=outer.CANONICAL_REPO, source_commit='e' * 40)
    if fault:
        with pytest.raises(Exception, match='INVOCATION_BINDING|FAMILY_ONLY_BOOT_MODE'):
            check()
    else:
        check()
        assert invocation['direction_research_recipe'] == window['recipe']


@pytest.mark.parametrize('fault', [None, 'global_training', 'gate_training', 'backward', 'constructor'])
def test_import_readiness_keeps_training_closed(monkeypatch, fault):
    import subprocess
    window, recipe = _window(), _recipe()
    gate = _gate(window['recipe'])
    policy = {'schema_version': 'gx1_next_native_run_policy_v1',
        'canonical_source_repo': str(outer.CANONICAL_REPO), 'canonical_branch': 'work/gx1-current',
        'training_module': outer.MODULE, 'training_enabled': False, 'native_training_enabled': False,
        'full_epoch_training_allowed': False, scope.TA_M5_IMPORT_STAGE: gate,
        'gpu_clock_launcher': _ref('/synthetic/clock')}
    if fault == 'global_training':
        policy['training_enabled'] = True
    elif fault == 'gate_training':
        gate['training_enabled'] = True
    elif fault == 'backward':
        gate['genuine_backward_authorized'] = True
    elif fault == 'constructor':
        gate['genuine_constructor_authorized'] = False
    monkeypatch.setattr(campaign, 'file_sha256', lambda path: 'a' * 64)
    monkeypatch.setattr(campaign, 'read_bound_json',
        lambda path, sha: policy if Path(path).name == 'NEXT_RUN_POLICY.json' else window)
    monkeypatch.setattr(campaign, 'require_binding', lambda value, **kwargs: value)
    monkeypatch.setattr(outer, 'require_research_window_policy', lambda value: value)
    monkeypatch.setattr(outer, 'require_research_launch_recipe', lambda *args, **kwargs: recipe)
    monkeypatch.setattr(subprocess, 'check_output', lambda args, **kwargs:
        'work/gx1-current' if '--show-current' in args else 'e' * 40 if 'rev-parse' in args else '')
    def check():
        return outer.require_native_direction_research_readiness(outer.CANONICAL_REPO,
            window_path=Path('/synthetic/WINDOW.json'), window_file_sha256='a' * 64)
    if fault:
        with pytest.raises(Exception, match='READINESS_GLOBAL|READINESS_GATE'):
            check()
    else:
        assert check()['native_window_scope']['optimizer_step_ceiling'] == 0


def test_import_250w_scope_and_guarded_dispatch(monkeypatch):
    recipe = _recipe()
    assert outer.guarded_research_power_limit(recipe) == 250
    with pytest.raises(Exception, match='M5_IMPORT_POWER_SCOPE'):
        outer.guarded_research_power_limit({**recipe, 'training_enabled': True})
    from gx1.contracts import gx1_capped_execution_v1 as capped
    from gx1.scripts import run_entry_native_direction_research_window_v1 as common
    from gx1.scripts import run_market_state_native_F_private_window_v1 as private
    calls = []
    monkeypatch.setattr(capped, 'require_guarded_cuda_trainer_execution', lambda: {})
    monkeypatch.setattr(outer, 'require_native_direction_research_context',
                        lambda *args: ({}, {}, {}, {}, recipe))
    monkeypatch.setattr(private, 'run_ta_m5_import', lambda **kwargs: calls.append(kwargs) or {'routed': True})
    args = dict(policy_path=Path('/synthetic/WINDOW.json'), policy_file_sha256='a' * 64,
                progress_path=Path('/synthetic/progress.json'))
    assert common.run_native_direction_research_window(**args) == {'routed': True}
    assert calls == [args]


@pytest.mark.parametrize('final_readout', [False, True])
def test_materializer_emits_one_zero_step_invocation_with_training_closed(tmp_path, monkeypatch, final_readout):
    from gx1.scripts import materialize_market_state_time_query_fold1_fit_campaign_v1 as materializer
    root, runtime = tmp_path, tmp_path / 'RUNTIME'
    prefix = 'TA_M5_FINAL_IMPORT' if final_readout else 'TA_M5_IMPORT'
    stage = scope.TA_M5_FINAL_IMPORT_STAGE if final_readout else scope.TA_M5_IMPORT_STAGE
    monkeypatch.setattr(scope, prefix + '_ROOT', root)
    monkeypatch.setattr(scope, prefix + '_RUNTIME_ROOT', runtime)
    monkeypatch.setattr(scope, prefix + '_RECIPE_PATH', root / 'RECIPE.json')
    binding_path = root / 'PREPARATION_EXECUTION_BINDING.json'
    binding = {'materializer': _ref(scope.CANONICAL_REPO /
        'gx1/scripts/materialize_market_state_time_query_fold1_fit_campaign_v1.py'),
        'recipe': _ref(root / 'RECIPE.json'), 'policy': _ref(scope.CANONICAL_REPO / 'NEXT_RUN_POLICY.json'),
        'prepared_boot': _ref(root / 'PREPARED_WINDOWS_BOOT.json'),
        'guard_template': _ref(root / 'GUARD_TEMPLATE.json'), 'source_head': 'e' * 40}
    recipe = _recipe(final_readout)
    gate = {**_gate(binding['recipe'], final_readout), 'campaign_preparation_authorized': True,
            **{key: binding[key] for key in ('prepared_boot', 'guard_template', 'materializer')}}
    objects = {str(binding_path): binding,
        binding['policy']['path']: {'training_enabled': False, 'native_training_enabled': False,
                                   stage: gate},
        binding['prepared_boot']['path']: {'synthetic_boot': True},
        binding['guard_template']['path']: {'phase': outer.PHASE, 'test_data_used': False,
            'gpu_uuid': 'GPU-synthetic', 'signed_guard_sources': {}, 'controller_sources': {}}}
    published = {}
    def publish(path, value):
        published[str(path)] = copy.deepcopy(value)
        return _ref(path)
    monkeypatch.setattr(materializer, '_ref', _ref)
    monkeypatch.setattr(materializer, '_publish', publish)
    monkeypatch.setattr(materializer, 'require_capped_cpu_audit_execution', lambda: None)
    monkeypatch.setattr(materializer, 'require_capped_lock_ancestry', lambda: None)
    monkeypatch.setattr(materializer.subprocess, 'check_output',
        lambda args, **kwargs: 'e' * 40 if args[1] == 'rev-parse' else '')
    monkeypatch.setattr(materializer.os, 'sync', lambda: None)
    monkeypatch.setattr(campaign, 'file_sha256', lambda path: 'a' * 64)
    monkeypatch.setattr(campaign, 'read_bound_json', lambda path, sha: objects[str(path)])
    monkeypatch.setattr(campaign, 'require_boot_identity', lambda value: value)
    monkeypatch.setattr(campaign, 'campaign_hardware_policy', lambda phase: {})
    monkeypatch.setattr(campaign, 'require_plan', lambda plan, **kwargs: {
        **plan, 'checked_invocations': [published[str(root / 'CAMPAIGN/INVOCATION_0001.json')]]})
    monkeypatch.setattr(campaign, 'next_action', lambda *args, **kwargs: {'decision': 'REBOOT_REQUIRED'})
    monkeypatch.setattr(outer, 'require_research_launch_recipe', lambda *args, **kwargs: recipe)
    monkeypatch.setattr(outer, 'require_native_direction_research_readiness',
                        lambda *args, **kwargs: {'decision': 'READY_FOR_EXISTING_BOUND_CAMPAIGN_GATES'})
    result = materializer.materialize(binding_path)
    assert result['genuine_job_started'] is False and len(result['windows']) == 1
    invocation = published[str(root / 'CAMPAIGN/INVOCATION_0001.json')]
    plan = published[str(root / 'CAMPAIGN/CAMPAIGN_PLAN.json')]
    assert invocation['optimizer_step_budget'] == 0
    assert invocation['requires_fresh_windows_boot'] is True
    assert invocation['expected_success_outcome'] == 'COMPLETE'
    assert invocation['window_count'] == 1 and invocation['kind'] == outer.KIND
    assert invocation['checkpoint']['pointer_path'] == str(runtime / 'research-cursor/RESUME_CURSOR.json')
    assert plan['policy']['physical_power_limit_w'] == 250
    assert result['decision'] == 'PASS_METADATA_PREPARED_REBOOT_REQUIRED'


def test_final_import_exact_recipe_stage_and_250w(monkeypatch):
    value = _recipe(True)
    path = scope.TA_M5_FINAL_IMPORT_RECIPE_PATH
    monkeypatch.setattr(campaign, 'require_binding', lambda value, **kwargs: value)
    monkeypatch.setattr(campaign, 'read_bound_json', lambda *args: value)
    monkeypatch.setattr(scope, 'require_ta_m5_import_recipe', lambda value, **kwargs: value)
    args = dict(source_repo=scope.CANONICAL_REPO, source_commit='e' * 40)
    assert outer.require_research_launch_recipe(_ref(path), **args) is value
    assert outer.guarded_research_power_limit(value) == 250
    scope.require_ta_m5_import_gate(_gate(_ref(path), True), value)
    with pytest.raises(Exception, match='M5_IMPORT_RECIPE_PATH'):
        outer.require_research_launch_recipe(_ref(scope.TA_M5_IMPORT_RECIPE_PATH), **args)
    with pytest.raises(Exception, match='M5_IMPORT_POWER_SCOPE'):
        outer.guarded_research_power_limit({**value, 'runtime_root': str(scope.TA_M5_IMPORT_RUNTIME_ROOT)})
    with pytest.raises(Exception, match='M5_IMPORT_POWER_SCOPE'):
        outer.guarded_research_power_limit({**value, 'max_private_updates': 1})
    with pytest.raises(Exception, match='M5_IMPORT_POLICY_GATE'):
        scope.require_ta_m5_import_gate({**_gate(_ref(path), True), 'training_enabled': True}, value)


def test_final_import_cursor_has_no_training_progress():
    recipe = _ref(scope.TA_M5_FINAL_IMPORT_RECIPE_PATH)
    kwargs = dict(recipe_ref=recipe, outcome='COMPLETE', optimizer_updates=0,
        parent_cursor=0, parent_order_sha256='e' * 64, full_state=_ref('/synthetic/state.pt'),
        result=_ref('/synthetic/result.json'), batch_size=16)
    cursor = outer.build_research_cursor(**kwargs)
    assert cursor['recipe'] == recipe and cursor['optimizer_updates'] == cursor['parent_cursor'] == 0
    for key in ('optimizer_updates', 'parent_cursor'):
        with pytest.raises(Exception, match='M5_IMPORT_CURSOR_SCOPE'):
            outer.build_research_cursor(**{**kwargs, key: 1})
