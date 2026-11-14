"""A failed admitted window is consumed with honest lower-bound counters."""
import json
import subprocess

import numpy as np

import pytest

from gx1.contracts import entry_native_direction_research_campaign_v1 as authority
from gx1.contracts import gx1_capped_execution_v1 as guard
from gx1.contracts import local_random_access_campaign_v2 as campaign
from gx1.contracts import market_state_ta_fold2_m5_fit_v1 as scope
from gx1.scripts import run_market_state_ta_fold2_m5_fit_window_v1 as worker


def test_synthetic_campaign_terminal_paths_and_refuses_relaunch(
        monkeypatch, tmp_path):
    root = tmp_path / 'FOLD2'
    root.mkdir()
    runtime = root / 'RUNTIME'
    recipe_path = root / 'RECIPE.json'
    cursor_path = runtime / 'research-cursor/RESUME_CURSOR.json'
    progress_path = runtime / 'progress/invocation-0001.json'
    monkeypatch.setattr(scope, 'ROOT', root)
    monkeypatch.setattr(scope, 'RUNTIME_ROOT', runtime)
    monkeypatch.setattr(scope, 'RECIPE_PATH', recipe_path)
    monkeypatch.setattr(scope, 'require_recipe', lambda *args, **kwargs: args[0])
    monkeypatch.setattr(guard, 'require_guarded_cuda_trainer_execution',
                        lambda: {'synthetic_guard': True})
    monkeypatch.setattr(campaign, 'require_clean_source', lambda plan: None)

    recipe_ref = {'path': str(recipe_path), 'sha256': 'a' * 64}
    result_input = {'path': str(root / 'INITIAL_RESULT.json'), 'sha256': 'b' * 64}
    recipe = {'run_scope': scope.SCOPE, 'admission_stage': scope.STAGE,
              'one_window': scope.ONE_WINDOW,
              'max_model_seconds': 10500,
              'max_published_json_bytes': 4 * 1024 * 1024,
              'input_refs': {'frozen_design': {'path': str(root / 'DESIGN.json'),
                                               'sha256': 'e' * 64},
                             'initial_plan': {'path': str(root / 'INITIAL_PLAN.json'),
                                              'sha256': 'f' * 64},
                             'initial_result': result_input}}
    window = {'recipe': recipe_ref, 'max_invocation_seconds': 12000,
              'output_directory': str(runtime / 'private-research'),
              'campaign_cursor_path': str(cursor_path)}
    plan = {'plan_sha256': 'c' * 64,
            'source_commit': subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=scope.REPO, text=True).strip()}
    invocation = {'invocation_sha256': 'd' * 64, 'kind': authority.KIND,
                  'batch_size': 16, 'optimizer_step_budget': 2959,
                  'direction_research_recipe': recipe_ref,
                  'checkpoint': {'pointer_path': str(cursor_path)}}
    monkeypatch.setattr(authority, 'require_native_direction_research_context',
                        lambda *args: (window, plan, invocation, {}, recipe))

    def fail_mid_step(**kwargs):
        kwargs['progress_state'].update(batch_visits=7, optimizer_updates=6,
            skipped_batches=1, parent_cursor=112, batch_visit_in_flight=True,
            optimizer_step_in_flight=True)
        raise RuntimeError('SYNTHETIC_MID_STEP')

    monkeypatch.setattr(worker, '_execute_fixed_fit_and_readout', fail_mid_step)
    args = {'policy_path': root / 'WINDOW.json',
            'policy_file_sha256': '0' * 64, 'progress_path': progress_path}
    with pytest.raises(RuntimeError, match='FOLD2_M5_FIT_CONSUMED_FAILURE'):
        worker.run_fold2_m5_fit_window(**args)
    result = json.loads((root / 'RESULT.json').read_text())
    terminal = json.loads((root / 'TERMINAL.json').read_text())
    progress = json.loads(progress_path.read_text())
    assert result['outcome'] == terminal['status'] == progress['outcome'] == 'FAILED'
    assert result['batch_visits'] == 7 and result['optimizer_updates'] == 6
    assert result['optimizer_updates_is_lower_bound'] is True
    assert result['batch_visit_in_flight_at_terminal'] is True
    assert result['checkpoint_writes'] == 0
    assert terminal['no_relaunch_under_this_plan'] is True
    assert progress['completed_units'] == 7 and progress['total_units'] == 2959
    assert not list(root.glob('FINAL_*_READOUT.npz'))
    before = (root / 'RESULT.json').read_bytes()
    with pytest.raises(RuntimeError, match='ONE_USE_GENESIS'):
        worker.run_fold2_m5_fit_window(**args)
    assert (root / 'RESULT.json').read_bytes() == before

    success_root = tmp_path / 'SUCCESS'
    success_root.mkdir()
    success_runtime = success_root / 'RUNTIME'
    success_cursor = success_runtime / 'research-cursor/RESUME_CURSOR.json'
    success_progress = success_runtime / 'progress/invocation-0001.json'
    monkeypatch.setattr(scope, 'ROOT', success_root)
    monkeypatch.setattr(scope, 'RUNTIME_ROOT', success_runtime)
    monkeypatch.setattr(scope, 'RECIPE_PATH', success_root / 'RECIPE.json')
    recipe_ref['path'] = str(scope.RECIPE_PATH)
    window.update(output_directory=str(success_runtime / 'private-research'),
                  campaign_cursor_path=str(success_cursor))
    invocation['checkpoint']['pointer_path'] = str(success_cursor)

    axes = np.arange(4096, dtype=np.int64)
    raw = np.zeros((4096, 5, 2), dtype=np.float32)
    truth = np.ones((4096, 5), dtype=np.int8)
    capture = {'parents': axes, 'issues': axes,
               'state_raw': raw, 'future_raw': raw,
               **{task + '_' + key: value
                  for task in ('state', 'future')
                  for key, value in (('allowed', truth),
                                     ('eligible', truth.astype(bool)),
                                     ('mature', truth.astype(bool)))}}

    def complete_synthetically(**kwargs):
        kwargs['progress_state'].update(batch_visits=2959, optimizer_updates=2950,
            skipped_batches=9, parent_cursor=47338, batch_visit_in_flight=False,
            optimizer_step_in_flight=False)
        return {'fit': {'batch_visits': 2959, 'optimizer_updates': 2950,
                        'skipped_batches': 9, 'parent_cursor': 47338,
                        'training_records': [{} for _ in range(2959)]},
                'final': {'captures': {'TRAIN': capture, 'LATER': capture},
                          'readouts': {'TRAIN': {}, 'LATER': {}},
                          'measurements': {'TRAIN': {}, 'LATER': {}},
                          'final_model_state_sha256': '1' * 64},
                'paired_later_f_m5': {
                    'schemes': {'UTC_DAY': {}, 'UTC_MONDAY_WEEK': {}},
                    'development_ranking_signal_qualified': False,
                    'test_data_used': False}}

    monkeypatch.setattr(worker, '_execute_fixed_fit_and_readout',
                        complete_synthetically)
    completed = worker.run_fold2_m5_fit_window(
        policy_path=success_root / 'WINDOW.json',
        policy_file_sha256='0' * 64, progress_path=success_progress)
    assert completed['predictive_edge_proven'] is False
    success = json.loads((success_root / 'RESULT.json').read_text())
    assert success['outcome'] == 'COMPLETE'
    assert success['batch_visits'] == 2959 and success['optimizer_updates'] == 2950
    assert set(success['final_captures']) == {'TRAIN', 'LATER'}
    assert success['economic_value_measured'] is False


def test_candidate_recipe_preparation_is_one_use_and_keeps_training_closed(
        monkeypatch, tmp_path):
    from gx1.audit import preflight_market_state_ta_fold2_m5_fit_v1 as preflight
    from gx1.scripts import prepare_market_state_ta_fold2_m5_fit_recipe_v1 as prep

    repo = tmp_path / 'REPO'
    repo.mkdir()
    root = tmp_path / 'RUNS/FOLD2'
    root.parent.mkdir()
    (repo / 'NEXT_RUN_POLICY.json').write_text(json.dumps({
        'status': 'TA_FOLD2_M5_003_CONSUMED_004_SOURCE_QUALIFIED_RECIPE_PENDING',
        'training_enabled': False, 'native_training_enabled': False,
        'full_epoch_training_allowed': False, 'full_val_allowed': False}))
    monkeypatch.setattr(scope, 'REPO', repo)
    monkeypatch.setattr(scope, 'ROOT', root)
    monkeypatch.setattr(scope, 'RECIPE_PATH', root / 'RECIPE.json')
    head = '1' * 40

    def git_output(argv, **kwargs):
        if argv[1:3] == ['branch', '--show-current']:
            return 'work/gx1-current'
        if argv[1:3] == ['rev-parse', 'HEAD']:
            return head
        if argv[1] == 'status':
            return ''
        raise AssertionError(argv)

    monkeypatch.setattr(prep.subprocess, 'check_output', git_output)
    monkeypatch.setattr(preflight, 'inspect',
                        lambda: {'status': 'PASS_METADATA_ONLY_FIT_UNADMITTED'})
    monkeypatch.setattr(scope, 'build_recipe', lambda *, source_commit: {
        'mechanical_source_commit': source_commit,
        'training_enabled': True, 'full_epoch_training_allowed': True,
        'test_data_used': False})
    prepared = prep.prepare()
    assert prepared['status'] == 'CANDIDATE_RECIPE_ONLY_NO_ADMISSION'
    assert prepared['optimizer_updates'] == 0
    assert (root / 'RECIPE.json').is_file()
    assert json.loads((repo / 'NEXT_RUN_POLICY.json').read_text())['training_enabled'] is False
    recipe_before = (root / 'RECIPE.json').read_bytes()
    with pytest.raises(RuntimeError, match='ONE_USE_GENESIS'):
        prep.prepare()
    assert (root / 'RECIPE.json').read_bytes() == recipe_before


def test_existing_campaign_router_selects_fold2_worker(monkeypatch, tmp_path):
    from gx1.scripts import run_entry_native_direction_research_window_v1 as router

    monkeypatch.setattr(guard, 'require_guarded_cuda_trainer_execution',
                        lambda: {'synthetic_guard': True})
    monkeypatch.setattr(authority, 'require_native_direction_research_context',
                        lambda *args: ({}, {}, {}, {}, {'run_scope': scope.SCOPE}))
    called = []

    def selected(**kwargs):
        called.append(kwargs)
        return {'selected': True}

    monkeypatch.setattr(worker, 'run_fold2_m5_fit_window', selected)
    args = {'policy_path': tmp_path / 'WINDOW.json',
            'policy_file_sha256': 'a' * 64,
            'progress_path': tmp_path / 'PROGRESS.json'}
    assert router.run_native_direction_research_window(**args) == {'selected': True}
    assert called == [args]


def test_existing_materializer_recognizes_fold2_root_before_metadata(
        monkeypatch):
    from gx1.scripts import materialize_market_state_time_query_fold1_fit_campaign_v1 as materializer

    monkeypatch.setattr(campaign, 'file_sha256', lambda path: '0' * 64)

    def reached(*args, **kwargs):
        raise RuntimeError('FOLD2_BINDING_REACHED')

    monkeypatch.setattr(campaign, 'read_bound_json', reached)
    with pytest.raises(RuntimeError, match='FOLD2_BINDING_REACHED'):
        materializer.materialize(scope.ROOT / 'PREPARATION_EXECUTION_BINDING.json')
