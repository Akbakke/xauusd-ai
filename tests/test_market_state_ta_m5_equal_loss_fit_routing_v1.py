"""Exact equal-loss full-fit routing using synthetic metadata only."""
import copy
from pathlib import Path

import pytest

from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as prefix
from gx1.contracts import local_random_access_campaign_v2 as campaign
from gx1.contracts import market_state_time_query_fold1_fit_v1 as scope


def _seal(value, key):
    return {**value, key: campaign.canonical_sha256({k: v for k, v in value.items() if k != key})}


def _ref(path):
    return {'path': str(path), 'sha256': 'a' * 64}


def test_new_recipe_path_dispatches_to_fullfit_scope_and_keeps_250w(monkeypatch):
    ref = _ref(scope.TA_M5_EQUAL_LOSS_FIT_RECIPE_PATH)
    value = {'run_scope': scope.TA_M5_EQUAL_LOSS_FIT_SCOPE,
             'admission_stage': scope.TA_M5_EQUAL_LOSS_FIT_STAGE}
    calls = []
    monkeypatch.setattr(campaign, 'require_binding', lambda ref, **kw: ref)
    monkeypatch.setattr(campaign, 'read_bound_json', lambda *args: value)

    def require(recipe, **kwargs):
        calls.append((recipe, kwargs))
        return recipe

    monkeypatch.setattr(scope, 'require_launch_recipe', require)
    assert prefix.require_prefix_fit_launch_recipe(
        ref, source_repo=scope.CANONICAL_REPO, source_commit='c' * 40) is value
    assert calls == [(value, dict(source_repo=scope.CANONICAL_REPO,
                                 source_commit='c' * 40, verify_files=True))]
    assert prefix.guarded_prefix_power_limit(value) == 250
    with pytest.raises(Exception, match='TIME_QUERY_RECIPE_PATH'):
        prefix.require_prefix_fit_launch_recipe(
            _ref(scope.TA_M5_FIT_RECIPE_PATH), source_repo=scope.CANONICAL_REPO,
            source_commit='c' * 40)
    with pytest.raises(Exception, match='TIME_QUERY_POWER_SCOPE'):
        prefix.guarded_prefix_power_limit({**value, 'admission_stage': scope.TA_M5_FIT_STAGE})


def _plan():
    artifact = _ref('/tmp/equal-loss-fit-metadata.json')
    policy = campaign.campaign_hardware_policy(prefix.PHASE)
    policy['physical_power_limit_w'] = 250
    return _seal({
        'schema_version': campaign.PLAN_SCHEMA, 'decision': 'PASS_PREPARED',
        'campaign_id': scope.TA_M5_EQUAL_LOSS_FIT_STAGE.upper(), 'phase': prefix.PHASE,
        'created_utc': '2026-11-09T00:00:00+00:00',
        'source_repo': str(scope.CANONICAL_REPO), 'source_commit': 'c' * 40,
        'runtime_root': str(scope.TA_M5_EQUAL_LOSS_FIT_RUNTIME_ROOT),
        'gpu_uuid': 'GPU-synthetic',
        'prepared_windows_boot': _seal({
            'schema_version': campaign.BOOT_SCHEMA, 'computer_name': 'SYNTHETIC',
            'last_boot_utc': '2026-11-08T23:00:00+00:00', 'boot_id': 1,
        }, 'identity_sha256'),
        'recipe': _ref(scope.TA_M5_EQUAL_LOSS_FIT_RECIPE_PATH), 'invocations': [artifact],
        'signed_guard_sources': {k: artifact for k in ('runner', 'guard', 'query', 'certificate')},
        'controller_sources': {k: artifact for k in ('controller', 'observer', 'campaign_cli')},
        'policy': policy, 'authority': {
            'test': False, 'promotion': False, 'paper': False, 'live': False, 'cloud_spend': False},
        'test_data_used': False,
    }, 'plan_sha256')


def test_new_plan_retains_fresh_boot_and_250w():
    plan = _plan()
    checked = prefix.require_prefix_fit_plan(plan, verify_files=False)
    assert len(checked['invocations']) == scope.TA_M5_EQUAL_LOSS_FIT_MAX_WINDOWS == 1
    assert checked['policy']['fresh_physical_windows_boot_before_every_invocation'] is True
    bad = copy.deepcopy(plan)
    bad['policy']['physical_power_limit_w'] = 300
    with pytest.raises(Exception, match='PLAN_AUTHORITY'):
        prefix.require_prefix_fit_plan(_seal(bad, 'plan_sha256'), verify_files=False)
    bad = copy.deepcopy(plan)
    bad['family_same_boot'] = {
        'protocol': _ref(campaign.FAMILY_PROTOCOL_PATH),
        'family_arm': 'synthetic', 'require_idle_review': True}
    with pytest.raises(Exception, match='TIME_QUERY_FRESH_BOOT_REQUIRED'):
        prefix.require_prefix_fit_plan(_seal(bad, 'plan_sha256'), verify_files=False)


def _cursor(monkeypatch):
    calls = []
    monkeypatch.setattr(scope, 'require_checkpoint_view',
                        lambda value, **kwargs: calls.append(kwargs) or value)
    ref = _ref(scope.TA_M5_EQUAL_LOSS_FIT_RECIPE_PATH)
    artifact = _ref('/tmp/equal-loss-fit-state.json')
    cursor = prefix.build_prefix_fit_cursor(
        recipe_ref=ref, fold=scope.FOLD, feature_view_identity={'prefix_normalization': {}},
        phase_state='COMPLETE', outcome='COMPLETE', optimizer_updates=1849,
        parent_cursor=29574, parent_order_sha256='b' * 64,
        full_state=artifact, own_initial_state=artifact, initial_measurement=artifact,
        final_train_measurement=artifact, oof_cursor=17753, result=artifact)
    return cursor, ref, calls


def test_equal_loss_cursor_and_progress_accept_tail6_and_exact_later_gap(monkeypatch):
    cursor, ref, calls = _cursor(monkeypatch)
    assert calls and all(c['ta_sf'] is True and c['dedicated_m5_side'] is True for c in calls)
    invocation = {
        'direction_prefix_fit_recipe': ref, 'optimizer_step_budget': 1849,
        'kind': prefix.KIND, 'invocation_sha256': 'd' * 64,
        'checkpoint': {'pointer_path': '/tmp/equal-loss-fit-cursor.json'}}
    progress = prefix.build_prefix_fit_progress(
        window={'recipe': ref}, plan={'plan_sha256': 'e' * 64}, invocation=invocation,
        checkpoint_pointer=_ref(invocation['checkpoint']['pointer_path']), cursor=cursor)
    assert (progress['total_units'], progress['completed_units'], progress['optimizer_updates'],
            progress['oof_cursor']) == (29574, 29574, 1849, 17753)
    original = campaign.require_binding
    monkeypatch.setattr(campaign, 'require_binding',
        lambda ref, **kw: original(ref, label=kw['label'], verify_file=False))
    launch = {
        'schema_version': prefix.RECIPE_SCHEMA, 'fold': scope.FOLD, 'train_passes': 1,
        'run_scope': scope.TA_M5_EQUAL_LOSS_FIT_SCOPE,
        'numerical_recipe': scope.ta_sf_numerical_recipe(
            dedicated_m5_side=True, m5_equal_observation=True)}
    monkeypatch.setattr(campaign, 'read_bound_json',
                        lambda *args: _seal(copy.deepcopy(launch), 'recipe_sha256'))
    assert prefix.require_prefix_fit_cursor(cursor, expected_recipe=ref,
                                            verify_files=True)['outcome'] == 'COMPLETE'
    assert calls[-1]['recipe']['run_scope'] == scope.TA_M5_EQUAL_LOSS_FIT_SCOPE
    launch['numerical_recipe'] = scope.ta_sf_numerical_recipe(dedicated_m5_side=True)
    with pytest.raises(Exception, match='CURSOR_INPUT_VIEW_IDENTITY'):
        prefix.require_prefix_fit_cursor(cursor, expected_recipe=ref, verify_files=True)


@pytest.mark.parametrize(('field', 'value', 'reason'), [
    ('parent_cursor', 29573, 'CURSOR_BATCH_BOUNDARY'),
    ('oof_cursor', 17752, 'CURSOR_COMPLETE'),
    ('optimizer_updates', 1850, 'CURSOR_COUNTER:optimizer_updates'),
])
def test_equal_loss_cursor_rejects_wrong_tail_or_score_count(monkeypatch, field, value, reason):
    cursor, ref, _ = _cursor(monkeypatch)
    cursor[field] = value
    with pytest.raises(Exception, match=reason):
        prefix.require_prefix_fit_cursor(_seal(cursor, 'cursor_sha256'),
                                         expected_recipe=ref, verify_files=False)


def test_guarded_dispatch_forwards_only_checked_context(monkeypatch, tmp_path):
    from gx1.contracts import gx1_capped_execution_v1 as capped
    from gx1.scripts import run_entry_native_direction_prefix_fit_window_v1 as dispatch
    from gx1.scripts import run_market_state_time_query_fold1_fit_window_v1 as fit
    calls = []
    limits = {'synthetic_guard': True}
    parts = ({'window': 1}, {'plan': 2}, {'invocation': 3}, {'marker': 4},
             {'run_scope': scope.TA_M5_EQUAL_LOSS_FIT_SCOPE})
    monkeypatch.setattr(capped, 'require_guarded_cuda_trainer_execution',
                        lambda: calls.append('guard') or limits)
    monkeypatch.setattr(prefix, 'require_native_direction_prefix_fit_context',
                        lambda *args: calls.append('context') or parts)

    def run(**kwargs):
        calls.append('fit')
        assert kwargs == dict(zip(('window', 'plan', 'invocation', 'marker', 'launch'), parts),
                              progress_path=tmp_path / 'PROGRESS.json', limits=limits)
        return {'synthetic': True}

    monkeypatch.setattr(fit, 'run_market_state_time_query_fold1_fit_window', run)
    assert dispatch.run_native_direction_prefix_fit_window(
        policy_path=tmp_path / 'WINDOW.json', policy_file_sha256='a' * 64,
        progress_path=tmp_path / 'PROGRESS.json') == {'synthetic': True}
    assert calls == ['guard', 'context', 'fit'] and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('windows', [1, 4])
def test_materializer_selects_one_new_window_before_any_claim(monkeypatch, windows):
    from gx1.scripts import materialize_market_state_time_query_fold1_fit_campaign_v1 as materializer
    root = scope.TA_M5_EQUAL_LOSS_FIT_ROOT
    path = root / 'PREPARATION_EXECUTION_BINDING.json'
    policy_path = scope.CANONICAL_REPO / 'NEXT_RUN_POLICY.json'
    binding = {
        'materializer': _ref(scope.CANONICAL_REPO /
            'gx1/scripts/materialize_market_state_time_query_fold1_fit_campaign_v1.py'),
        'recipe': _ref(scope.TA_M5_EQUAL_LOSS_FIT_RECIPE_PATH),
        'policy': _ref(policy_path), 'prepared_boot': _ref(root / 'PREPARED_WINDOWS_BOOT.json'),
        'guard_template': _ref('/tmp/equal-loss-guard-template.json'), 'source_head': 'c' * 40}
    gate = {
        'campaign_preparation_authorized': True, 'recipe': binding['recipe'],
        'prepared_boot': binding['prepared_boot'], 'guard_template': binding['guard_template'],
        'materializer': binding['materializer'],
        'run_root': str(scope.TA_M5_EQUAL_LOSS_FIT_RUNTIME_ROOT),
        'claimed_invocations': 0, 'completed_invocations': 0, 'max_invocations': windows,
        'relaunch_allowed': False, 'test_data_used': False}
    docs = {path: binding, policy_path: {scope.TA_M5_EQUAL_LOSS_FIT_STAGE: gate}}
    reads = []
    monkeypatch.setattr(campaign, 'file_sha256', lambda path: 'a' * 64)
    monkeypatch.setattr(campaign, 'read_bound_json',
                        lambda path, sha: reads.append(path) or copy.deepcopy(docs[path]))
    monkeypatch.setattr(materializer.subprocess, 'check_output',
                        lambda args, **kw: 'c' * 40 if args[1] == 'rev-parse' else '')

    class ReachedCappedGate(Exception):
        pass

    def stop_at_cap():
        raise ReachedCappedGate

    monkeypatch.setattr(materializer, 'require_capped_cpu_audit_execution', stop_at_cap)
    if windows == 1:
        with pytest.raises(ReachedCappedGate):
            materializer.materialize(path)
    else:
        with pytest.raises(Exception, match='MATERIALIZER_POLICY_GATE'):
            materializer.materialize(path)
    assert reads == [path, policy_path]
    with pytest.raises(Exception, match='MATERIALIZER_BINDING_PATH'):
        materializer.materialize(path.with_name('OTHER_BINDING.json'))
    assert reads == [path, policy_path]
