"""Fold-1 time-query counters must not inherit the legacy prefix geometry."""
from __future__ import annotations

import pytest

from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as prefix
from gx1.contracts import market_state_time_query_fold1_fit_v1 as fold1


REF = {'path': str(fold1.RECIPE_PATH), 'sha256': 'a' * 64}
ARTIFACT = {'path': '/tmp/gx1-fold1-campaign-test.json', 'sha256': 'b' * 64}


def _route(ta_sf):
    return {
        False: (fold1.SCOPE, fold1.STAGE, fold1.RECIPE_PATH),
        True: (fold1.TA_SF_SCOPE, fold1.TA_SF_STAGE, fold1.TA_SF_RECIPE_PATH),
        'recovery': (fold1.TA_RECOVERY_SCOPE, fold1.TA_RECOVERY_STAGE,
                     fold1.TA_RECOVERY_RECIPE_PATH),
    }[ta_sf]


def _recipe_ref(ta_sf):
    return {**REF, 'path': str(_route(ta_sf)[2])}

def _complete_cursor(monkeypatch, **changes):
    monkeypatch.setattr(fold1, 'require_checkpoint_view',
                        lambda value, **kwargs: value)
    args = dict(recipe_ref=REF, fold=fold1.FOLD,
                feature_view_identity={'synthetic': True},
                phase_state='COMPLETE', outcome='COMPLETE',
                optimizer_updates=fold1.MAX_UPDATES,
                parent_cursor=fold1.FIT_ROWS,
                parent_order_sha256='c' * 64,
                full_state=ARTIFACT, own_initial_state=ARTIFACT,
                initial_measurement=ARTIFACT,
                final_train_measurement=ARTIFACT,
                oof_cursor=fold1.SCORE_ROWS, result=ARTIFACT)
    return prefix.build_prefix_fit_cursor(**{**args, **changes})


@pytest.mark.parametrize('ta_sf', [False, True, 'recovery'])
def test_fold1_cursor_counts_only_later_score_rows(monkeypatch, ta_sf):
    ref = _recipe_ref(ta_sf)
    cursor = _complete_cursor(monkeypatch, recipe_ref=ref)
    assert cursor['parent_cursor'] == 29574
    assert cursor['oof_cursor'] == 17753
    with pytest.raises(Exception, match='CURSOR_COUNTER:oof_cursor'):
        _complete_cursor(monkeypatch, recipe_ref=ref, oof_cursor=18028)
    with pytest.raises(Exception, match='TIME_QUERY_CURSOR_FOLD'):
        _complete_cursor(monkeypatch, recipe_ref=ref, fold={**fold1.FOLD,
                                           'oof_start_row': fold1.FIT_ROWS})


@pytest.mark.parametrize('ta_sf', [False, True, 'recovery'])
def test_fold1_progress_uses_mature_fit_and_gap(monkeypatch, ta_sf):
    ref = _recipe_ref(ta_sf)
    cursor = _complete_cursor(monkeypatch, recipe_ref=ref)
    invocation = {
        'direction_prefix_fit_recipe': ref,
        'optimizer_step_budget': fold1.MAX_UPDATES,
        'kind': prefix.KIND,
        'invocation_sha256': 'd' * 64,
        'checkpoint': {'pointer_path': '/tmp/gx1-fold1-campaign-cursor.json'},
    }
    window = {'recipe': ref}
    plan = {'plan_sha256': 'e' * 64}
    pointer = {'path': invocation['checkpoint']['pointer_path'], 'sha256': 'f' * 64}
    progress = prefix.build_prefix_fit_progress(
        window=window, plan=plan, invocation=invocation,
        checkpoint_pointer=pointer, cursor=cursor)
    assert progress['total_units'] == 29574
    assert progress['oof_cursor'] == 17753
    assert progress['outcome'] == 'COMPLETE'
    bad = dict(progress, oof_cursor=18028)
    bad.pop('progress_sha256')
    bad['progress_sha256'] = prefix.campaign.canonical_sha256(bad)
    with pytest.raises(Exception, match='PROGRESS_COUNTER:oof_cursor'):
        prefix.require_prefix_fit_progress(
            bad, plan_sha256=plan['plan_sha256'], invocation=invocation,
            expected_selection_receipt_sha256=None, verify_file=False)


def test_one_use_receipt_shapes_accept_honest_complete_and_failed():
    from gx1.contracts import local_random_access_campaign_v2 as campaign
    boot_body = {
        'schema_version': campaign.BOOT_SCHEMA, 'computer_name': 'ANDRE2',
        'last_boot_utc': '2026-11-07T00:00:00Z', 'boot_id': 1,
    }
    boot = {**boot_body, 'identity_sha256': campaign.canonical_sha256(boot_body)}
    pointer = str(fold1.RUNTIME_ROOT / 'prefix-cursor' / 'RESUME_CURSOR.json')
    evidence = (fold1.RUNTIME_ROOT / 'invocation-evidence' /
                'invocation-0001')
    invocation = {
        'invocation_sha256': 'd' * 64,
        'invocation_number': 1,
        'invocation_id': 'invocation-0001',
        'kind': prefix.KIND,
        'requires_fresh_windows_boot': True,
        'expected_success_outcome': 'RESUMABLE_OR_COMPLETE',
        'checkpoint': {'before_mode': 'GENESIS', 'pointer_path': pointer},
        'progress_path': str(fold1.RUNTIME_ROOT / 'progress' /
                             'invocation-0001.json'),
    }
    unsigned = {
        'schema_version': campaign.RECEIPT_SCHEMA,
        'plan_sha256': 'e' * 64,
        'invocation_sha256': invocation['invocation_sha256'],
        'invocation_number': 1,
        'invocation_id': 'invocation-0001',
        'kind': prefix.KIND,
        'selection_receipt_sha256': None,
        'boot': boot,
        'started_utc': '2026-11-07T15:00:00+00:00',
        'finished_utc': '2026-11-07T15:01:00+00:00',
        'outcome': 'COMPLETE',
        'trainer_guard_exit_code': 0,
        'progress_observer_exit_code': 0,
        'pointer_before_sha256': 'GENESIS',
        'checkpoint_pointer_after': {'path': pointer, 'sha256': 'a' * 64},
        'checkpoint_pointer_snapshot': {
            'path': str(evidence / 'CHECKPOINT_POINTER.json'),
            'sha256': 'a' * 64,
        },
        'progress': {'path': str(evidence / 'PROGRESS.json'),
                     'sha256': 'b' * 64},
        'guard_log': {'path': str(evidence / 'SIGNED_GUARD.log'),
                      'sha256': 'c' * 64},
        'guard_decision': 'PASS',
        'signed_guard_telemetry_owner': 'gx1_guarded_trainer_exec.sh',
        'active_marker_sha256': 'f' * 64,
        'test_data_used': False,
    }

    def checked(value):
        sealed = {**value, 'receipt_sha256': campaign.canonical_sha256(value)}
        return campaign.require_receipt(
            sealed, plan_sha256=unsigned['plan_sha256'],
            invocation=invocation, verify_files=False)

    assert checked(unsigned)['outcome'] == 'COMPLETE'
    failed = {**unsigned, 'outcome': 'FAILED',
              'trainer_guard_exit_code': 1, 'guard_decision': 'FAILED'}
    assert checked(failed)['outcome'] == 'FAILED'
    with pytest.raises(Exception, match='failed receipt has successful processes'):
        checked({**failed, 'trainer_guard_exit_code': 0})


@pytest.mark.parametrize('ta_sf', [False, True, 'recovery'])
def test_fold1_recipe_dispatch_requires_its_own_path(monkeypatch, ta_sf):
    ref = _recipe_ref(ta_sf)
    scope = _route(ta_sf)[0]
    value = {'run_scope': scope}
    monkeypatch.setattr(prefix.campaign, 'require_binding',
                        lambda value, **kwargs: value)
    monkeypatch.setattr(prefix.campaign, 'read_bound_json',
                        lambda *args: value)
    calls = []
    def checked(recipe, **kwargs):
        calls.append((recipe, kwargs))
        return recipe
    monkeypatch.setattr(fold1, 'require_launch_recipe', checked)
    assert prefix.require_prefix_fit_launch_recipe(
        ref, source_repo=fold1.CANONICAL_REPO, source_commit='a' * 40) is value
    assert calls[0][1]['verify_files'] is True
    for other in (False, True, 'recovery'):
        if other == ta_sf:
            continue
        with pytest.raises(Exception, match='TIME_QUERY_RECIPE_PATH'):
            prefix.require_prefix_fit_launch_recipe(
                _recipe_ref(other), source_repo=fold1.CANONICAL_REPO,
                source_commit='a' * 40)
    assert len(calls) == 1


@pytest.mark.parametrize('ta_sf', [False, True, 'recovery'])
def test_fold1_power_scope_keeps_exact_stage_and_250w(ta_sf):
    scope = _route(ta_sf)[0]
    stage = _route(ta_sf)[1]
    other_stage = fold1.STAGE if ta_sf else fold1.TA_SF_STAGE
    assert prefix.guarded_prefix_power_limit({
        'run_scope': scope, 'admission_stage': stage}) == 250
    with pytest.raises(Exception, match='TIME_QUERY_POWER_SCOPE'):
        prefix.guarded_prefix_power_limit({
            'run_scope': scope, 'admission_stage': other_stage})


def test_ta_prefix_view_extension_requires_explicit_opt_in():
    prefix_identity = {
        'binding': ARTIFACT, 'contract_sha256': '1' * 64,
        'base_sha256': '2' * 64, 'gc_layout_sha256': '3' * 64,
        'gc_normalization_sha256': '4' * 64,
        'qualification_sha256': '5' * 64,
        'unobserved_units_profile_sha256': '6' * 64,
        'ta_base_normalization_sha256': '7' * 64,
        'source_owner_profile_sha256': '8' * 64,
    }
    view = {'contract_sha256': '9' * 64,
            'research_objective_sha256': 'a' * 64,
            'prefix_normalization': prefix_identity}
    assert prefix._view_identity(view, verify_files=False,
        allow_unobserved_units_profile=True, allow_ta_profile=True) == view
    with pytest.raises(Exception, match='PREFIX_IDENTITY'):
        prefix._view_identity(view, verify_files=False,
            allow_unobserved_units_profile=True)
    with pytest.raises(Exception, match='TA_PROFILE_SCOPE'):
        prefix._view_identity(view, verify_files=False, allow_ta_profile=True)
    for missing in ('ta_base_normalization_sha256', 'source_owner_profile_sha256'):
        bad = {**view, 'prefix_normalization': {
            key: value for key, value in prefix_identity.items() if key != missing}}
        with pytest.raises(Exception, match='PREFIX_IDENTITY'):
            prefix._view_identity(bad, verify_files=False,
                allow_unobserved_units_profile=True, allow_ta_profile=True)


@pytest.mark.parametrize('ta_sf', [True, 'recovery'])
def test_ta_cursor_binds_state_view_and_dual_recipe_separately(monkeypatch, ta_sf):
    from gx1.contracts.market_state_time_query_model_v2 import factorized_time_query_model_contract
    numerical = fold1.ta_sf_numerical_recipe()
    state_sha = factorized_time_query_model_contract(ta_only=True)['contract_sha256']
    assert state_sha != numerical['research_objective_sha256']
    ref = _recipe_ref(ta_sf)
    view = {'prefix_normalization': {},
            'research_objective_sha256': state_sha}
    cursor = _complete_cursor(monkeypatch, recipe_ref=ref,
        feature_view_identity=view)
    calls = []
    def checked_view(value, **kwargs):
        assert kwargs['ta_sf'] is True
        assert value['research_objective_sha256'] == state_sha
        calls.append(kwargs)
        return value
    monkeypatch.setattr(fold1, 'require_checkpoint_view', checked_view)
    require_binding = prefix.campaign.require_binding
    monkeypatch.setattr(prefix.campaign, 'require_binding',
        lambda value, **kwargs: require_binding(value,
            label=kwargs['label'], verify_file=False))
    launch = {
        'schema_version': prefix.RECIPE_SCHEMA, 'fold': fold1.FOLD,
        'train_passes': 1, 'run_scope': _route(ta_sf)[0],
        'numerical_recipe': numerical,
    }
    def sealed_launch():
        return {**launch, 'recipe_sha256': prefix.campaign.canonical_sha256(launch)}
    monkeypatch.setattr(prefix.campaign, 'read_bound_json',
                        lambda *args: sealed_launch())
    assert prefix.require_prefix_fit_cursor(cursor, expected_recipe=ref,
        verify_files=True)['outcome'] == 'COMPLETE'
    assert len(calls) == 2 and calls[-1]['recipe'] == sealed_launch()
    launch['numerical_recipe'] = {**numerical, 'research_objective_sha256': state_sha}
    with pytest.raises(Exception, match='CURSOR_INPUT_VIEW_IDENTITY'):
        prefix.require_prefix_fit_cursor(cursor, expected_recipe=ref,
            verify_files=True)
    launch['numerical_recipe'] = numerical
    for other in (False, True, 'recovery'):
        if other == ta_sf:
            continue
        launch['run_scope'] = _route(other)[0]
        with pytest.raises(Exception, match='CURSOR_INPUT_VIEW_IDENTITY'):
            prefix.require_prefix_fit_cursor(cursor, expected_recipe=ref,
                verify_files=True)


@pytest.mark.parametrize('ta_sf', [False, True, 'recovery'])
def test_guarded_fold1_dispatch_preserves_checked_context(monkeypatch, tmp_path, ta_sf):
    from gx1.contracts import gx1_capped_execution_v1 as capped
    from gx1.scripts import run_entry_native_direction_prefix_fit_window_v1 as dispatch
    from gx1.scripts import run_market_state_time_query_fold1_fit_window_v1 as fit_runner
    calls = []
    limits = {'synthetic_guard': True}
    parts = ({'window': 1}, {'plan': 2}, {'invocation': 3}, {'marker': 4},
             {'run_scope': _route(ta_sf)[0]})
    monkeypatch.setattr(capped, 'require_guarded_cuda_trainer_execution',
                        lambda: calls.append('guard') or limits)
    monkeypatch.setattr(prefix, 'require_native_direction_prefix_fit_context',
                        lambda *args: calls.append('context') or parts)
    output = {'synthetic_dispatch': True}
    seen = {}
    def fit(**kwargs):
        calls.append('fit')
        seen.update(kwargs)
        return output
    monkeypatch.setattr(fit_runner, 'run_market_state_time_query_fold1_fit_window', fit)
    progress_path = tmp_path / 'PROGRESS.json'
    assert dispatch.run_native_direction_prefix_fit_window(
        policy_path=tmp_path / 'WINDOW.json', policy_file_sha256='a' * 64,
        progress_path=progress_path) is output
    assert calls == ['guard', 'context', 'fit']
    assert seen == dict(zip(('window', 'plan', 'invocation', 'marker', 'launch'), parts),
                        progress_path=progress_path, limits=limits)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('ta_sf', [False, True, 'recovery'])
def test_fold1_materializer_accepts_only_exact_preparation_binding_path(monkeypatch, ta_sf):
    from gx1.scripts import materialize_market_state_time_query_fold1_fit_campaign_v1 as materializer
    binding_path = _route(ta_sf)[2].parent / 'PREPARATION_EXECUTION_BINDING.json'
    reads = []
    monkeypatch.setattr(materializer.campaign, 'file_sha256', lambda path: 'a' * 64)
    def read(path, sha256):
        reads.append((path, sha256))
        return {}  # Stop before any claim, directory, model or data access.
    monkeypatch.setattr(materializer.campaign, 'read_bound_json', read)
    with pytest.raises(Exception, match='MATERIALIZER_BINDING_FIELDS'):
        materializer.materialize(binding_path)
    assert reads == [(binding_path, 'a' * 64)]
    with pytest.raises(Exception, match='MATERIALIZER_BINDING_PATH'):
        materializer.materialize(binding_path.with_name('OTHER_BINDING.json'))
    assert reads == [(binding_path, 'a' * 64)]

def test_top_level_window_cli_parses_and_forwards_without_launch(monkeypatch, tmp_path, capsys):
    import json
    from gx1.scripts import run_unified_exit_native_candidate_window_v1 as cli
    policy_path = tmp_path / 'WINDOW_POLICY.json'
    progress_path = tmp_path / 'PROGRESS.json'
    policy_sha256 = 'a' * 64
    output = {'decision': 'SYNTHETIC_CLI_SMOKE', 'test_data_used': False}
    calls = []
    def run_window(**kwargs):
        calls.append(kwargs)
        return output
    monkeypatch.setattr(cli, 'run_window', run_window)
    assert cli.main([
        '--window-policy', str(policy_path),
        '--window-policy-file-sha256', policy_sha256,
        '--progress-path', str(progress_path),
    ]) == 0
    assert calls == [{'policy_path': policy_path, 'policy_file_sha256': policy_sha256,
                      'progress_path': progress_path}]
    captured = capsys.readouterr()
    assert captured.out == json.dumps(output, sort_keys=True, allow_nan=False) + '\n'
    assert captured.err == ''
    assert list(tmp_path.iterdir()) == []
