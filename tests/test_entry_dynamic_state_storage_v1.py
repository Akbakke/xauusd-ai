"""Storage layouts and fail-closed metadata only; no model, optimizer or training."""
import numpy as np
import pytest

from gx1.contracts import entry_dynamic_mtf_state_report_v1 as report


@pytest.mark.parametrize('rows,leads', [(1, 3), (7, 1441), (2, 10080)])
def test_prediction_estimate_matches_actual_small_npy_files(tmp_path, rows, leads):
    layout = report.dynamic_state_prediction_layout(rows, leads)
    predicted = report.dynamic_state_prediction_file_bytes(rows, leads)
    for name, (shape, dtype) in layout.items():
        path = tmp_path / (name + '.npy')
        value = np.lib.format.open_memmap(path, mode='w+', dtype=dtype, shape=shape)
        value.flush()
        del value
        assert path.stat().st_size == predicted[name]
    assert set(predicted) == {'probabilities', 'sequence_nll', 'parent_rows', 'issue_ns', 'bar_start_ns'}


@pytest.mark.parametrize('rows,leads', [(0, 1), (1, 0), (True, 1), (1, False), (1.0, 5)])
def test_prediction_layout_rejects_invalid_sizes(rows, leads):
    with pytest.raises(RuntimeError, match='OUTPUT_LAYOUT'):
        report.dynamic_state_prediction_file_bytes(rows, leads)


def _recipe(*, private=False, horizon=10080, reserve=5 * 1024**3):
    from gx1.contracts import entry_native_direction_research_campaign_v1 as private_owner
    numerical = {'extended_forecast_minutes': horizon, 'extended_storage_reserve_bytes': reserve, 'current_state_first_loss_weight': 1.0}
    value = {'numerical_recipe': numerical, 'fold': private_owner.BPS_FOLD,
             'max_checkpoint_bytes': 1073741824, 'max_published_json_bytes': 33554432 if private else 67108864}
    if not private:
        value.update(initial_measurement_rows=256, final_train_measurement_rows=47602,
                     max_checkpoint_total_bytes=4294967296)
    return value


def test_extended_storage_plan_accounts_for_all_remaining_arrays_and_reserve():
    from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as owner
    recipe = _recipe()
    preflight = owner._extended_storage_plan(recipe, private_scope=False)
    expected = sum(sum(report.dynamic_state_prediction_file_bytes(n, 10080).values()) for n in (256, 47602, 17288))
    assert preflight['logical_npy_bytes'] == expected and expected > 39400300800
    assert preflight['required_free_bytes'] == expected + 5 * 1024**3
    final = owner._extended_storage_plan(recipe, private_scope=False, stage='FINAL_TRAIN', stage_rows=47602)
    oof = owner._extended_storage_plan(recipe, private_scope=False, stage='OOF', stage_rows=17288)
    assert preflight['required_free_bytes'] > final['required_free_bytes'] > oof['required_free_bytes']
    with pytest.raises(RuntimeError, match='STATE_STORAGE_STAGE'):
        owner._extended_storage_plan(recipe, private_scope=False, stage='OOF', stage_rows=1)
    with pytest.raises(RuntimeError, match='STATE_STORAGE_RESERVE_FLOOR'):
        owner._extended_storage_plan(_recipe(reserve=1), private_scope=False)
    assert owner._extended_storage_plan({'numerical_recipe': {}}, private_scope=False) is None


def test_storage_guard_checks_guest_host_and_mapping_without_model(monkeypatch):
    from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as owner
    recipe = _recipe()
    plan = owner._extended_storage_plan(recipe, private_scope=False)
    needed = plan['required_free_bytes']
    observed = {'guest_available_bytes': needed + 5, 'host_available_bytes': needed + 4,
                'mounted_host_available_bytes': needed + 3, 'host_base_path': 'C:\\wsl', 'host_drive_root': 'C:\\'}
    monkeypatch.setattr(owner, '_observe_wsl_host_storage', lambda path: dict(observed))
    actual = owner.require_dynamic_state_storage_headroom(recipe, '/synthetic', private_scope=False)
    assert actual['plan'] == plan and actual['scope'] == 'timestamped_capacity_check_not_reserved_space'
    assert actual['host_identity'] == ('C:\\wsl', 'C:\\')
    observed['host_available_bytes'] = needed - 1
    with pytest.raises(RuntimeError, match='STATE_STORAGE_INSUFFICIENT'):
        owner.require_dynamic_state_storage_headroom(recipe, '/synthetic', private_scope=False)
    observed['host_available_bytes'] = needed + 4
    with pytest.raises(RuntimeError, match='STATE_STORAGE_HOST_CHANGED'):
        owner.require_dynamic_state_storage_headroom(recipe, '/synthetic', private_scope=False, expected_host=('D:\\other', 'D:\\'))


def test_private_and_recovery_storage_scopes_are_distinct():
    from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as owner
    private = owner._extended_storage_plan(_recipe(private=True), private_scope=True)
    assert private['remaining_stage_rows'] == {'INITIAL_TRAIN': 256}
    recovery = owner._extended_storage_plan(_recipe(), private_scope=False, recovery=True)
    assert recovery['remaining_stage_rows'] == {'FINAL_TRAIN': 47602, 'OOF': 17288}
    with pytest.raises(RuntimeError, match='STATE_STORAGE_SCOPE'):
        owner._extended_storage_plan(_recipe(private=True), private_scope=True, recovery=True)


def test_extended_recipe_requires_explicit_positive_reserve():
    from gx1.contracts import entry_model_native_train_recipe_v1 as recipes
    args = dict(batch_size=16, max_updates=2, dynamic_state=True,
                current_state_first_loss_weight=1.0, extended_forecast_minutes=10080)
    for bad in (None, 0, -1, True, 1.5):
        with pytest.raises(RuntimeError, match='EXTENDED_STORAGE_RESERVE'):
            recipes.native_direction_research_recipe(**args, extended_storage_reserve_bytes=bad)
    candidate = recipes.native_direction_research_recipe(**args, extended_storage_reserve_bytes=5 * 1024**3)
    assert recipes.require_native_direction_research_recipe(candidate) == candidate
    with pytest.raises(RuntimeError, match='EXTENDED_STORAGE_RESERVE'):
        recipes.native_direction_research_recipe(batch_size=16, max_updates=2, dynamic_state=True,
            extended_storage_reserve_bytes=5 * 1024**3)


def test_wsl_host_observer_rejects_unknown_or_reparse_mapping(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as owner
    host = {'distribution': 'Ubuntu-22.04', 'base_path': 'C:\\Users\\Andre\\LocalState',
            'drive_root': 'C:\\', 'no_reparse': True, 'host_available_bytes': 1000}
    monkeypatch.setenv('WSL_DISTRO_NAME', 'Ubuntu-22.04')
    monkeypatch.setattr(owner.os, 'statvfs', lambda p: SimpleNamespace(f_bavail=100, f_frsize=4))
    monkeypatch.setattr(owner.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps(host).encode()))
    measured = owner._observe_wsl_host_storage(tmp_path)
    assert measured['guest_available_bytes'] == measured['mounted_host_available_bytes'] == 400
    assert measured['host_available_bytes'] == 1000 and measured['host_drive_root'] == 'C:\\'
    host['no_reparse'] = False
    with pytest.raises(RuntimeError, match='STATE_STORAGE_HOST_IDENTITY'):
        owner._observe_wsl_host_storage(tmp_path)
    host['no_reparse'] = True
    host['drive_root'] = '\\\\server\\share\\'
    with pytest.raises(RuntimeError, match='STATE_STORAGE_HOST_IDENTITY'):
        owner._observe_wsl_host_storage(tmp_path)
    monkeypatch.setenv('WSL_DISTRO_NAME', 'unexpected')
    with pytest.raises(RuntimeError, match='STATE_STORAGE_DISTRO'):
        owner._observe_wsl_host_storage(tmp_path)


def test_evaluation_checks_storage_before_any_model_or_output(tmp_path, monkeypatch):
    from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as owner
    from gx1.scripts import run_entry_native_direction_prefix_fit_window_v1 as worker
    recipe = _recipe()
    guard = {'recipe': recipe, 'private_scope': False, 'recovery': False, 'host_identity': ('C:\\wsl', 'C:\\')}
    prepared = {'state_artifact_directory': tmp_path, 'extended_storage_guard': guard}
    def refuse(*args, **kwargs):
        raise RuntimeError('STATE_STORAGE_INSUFFICIENT')
    monkeypatch.setattr(owner, 'require_dynamic_state_storage_headroom', refuse)
    with pytest.raises(RuntimeError, match='STATE_STORAGE_INSUFFICIENT'):
        worker._evaluate_dynamic_state(object(), prepared, {}, {'extended_forecast_minutes': 10080}, np.arange(17288), cutoff=1, kind='OOF',
                                       counters={}, ceiling=1, device=None, torch=None, np=np, tree_sha=None)
    assert list(tmp_path.iterdir()) == []


def test_wsl_host_observer_fails_closed_when_windows_query_fails(tmp_path, monkeypatch):
    from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as owner
    monkeypatch.setenv('WSL_DISTRO_NAME', 'Ubuntu-22.04')
    def failed(*args, **kwargs):
        raise FileNotFoundError('synthetic missing host query')
    monkeypatch.setattr(owner.subprocess, 'run', failed)
    with pytest.raises(RuntimeError, match='STATE_STORAGE_HOST_UNAVAILABLE'):
        owner._observe_wsl_host_storage(tmp_path)


@pytest.mark.parametrize('reserve,accepted', [(5 * 1024**3, True), (1, False), (None, False)])
def test_extended_launch_cannot_bypass_storage_reserve(tmp_path, monkeypatch, reserve, accepted):
    import hashlib
    import json
    from gx1.contracts import entry_model_native_train_recipe_v1 as recipes
    from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as owner
    from gx1.contracts import entry_native_direction_research_campaign_v1 as private
    from gx1.contracts import local_random_access_campaign_v2 as canonical
    numerical = recipes.native_direction_research_recipe(batch_size=16, max_updates=2, dynamic_state=True,
        current_state_first_loss_weight=1.0, extended_forecast_minutes=1441,
        extended_storage_reserve_bytes=5 * 1024**3)
    if reserve is None:
        del numerical['extended_storage_reserve_bytes']
    else:
        numerical['extended_storage_reserve_bytes'] = reserve
    unsigned = {k: v for k, v in numerical.items() if k != 'contract_sha256'}
    numerical['contract_sha256'] = hashlib.sha256(json.dumps(unsigned, sort_keys=True,
        separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()).hexdigest()
    repo = tmp_path/'source'
    repo.mkdir()
    runtime = tmp_path/'runtime'
    dummy = {'path': str(tmp_path/'unused.json'), 'sha256': 'a'*64}
    monkeypatch.setattr(owner, 'CANONICAL_REPO', repo)
    monkeypatch.setattr(owner, 'REQUIRED_SOURCE_PATHS', ())
    monkeypatch.setattr(owner, 'DYNAMIC_STATE_EXTRA_SOURCE_PATHS', ())
    monkeypatch.setattr(owner, 'require_dynamic_state_input_manifest', lambda *a, **kw: None)
    monkeypatch.setattr(owner, 'require_dynamic_state_target_binding', lambda *a, **kw: None)
    launch = {'schema_version': private.RECIPE_SCHEMA, 'source_repo': str(repo), 'runtime_root': str(runtime),
        'admission_stage': 'dynamic_mtf_state_synthetic_private', 'run_scope': private.DYNAMIC_STATE_PRIVATE_SCOPE,
        'numerical_recipe': numerical, 'input_manifest': dummy, 'source_owners': {},
        'execution_binding_path': str(runtime/'EXECUTION_BINDING.json'), 'fold': private.BPS_FOLD,
        'max_checkpoint_bytes': 1073741824, 'max_published_json_bytes': 33554432,
        'training_enabled': True, 'full_epoch_training_allowed': False, 'test_data_used': False,
        'predictive_value_claimed': False, 'qualification_weights_for_main_fit': False,
        'target_manifest': dummy, 'target_sources': {'m1': dummy, 'm5_atr': dummy}, 'label_audit': dummy,
        'full_feature_profile': True, 'max_private_updates': 4, 'max_model_constructions': 3,
        'max_forwards': 21, 'max_backwards': 4, 'qualification_parent_rows': list(range(32))}
    launch['recipe_sha256'] = canonical.canonical_sha256(launch)
    if accepted:
        checked = owner.require_dynamic_state_launch_recipe(launch, source_repo=repo, source_commit='b'*40, private_scope=True)
        assert checked['numerical_recipe']['extended_storage_reserve_bytes'] == reserve
    else:
        with pytest.raises(RuntimeError, match='STATE_STORAGE_RESERVE'):
            owner.require_dynamic_state_launch_recipe(launch, source_repo=repo, source_commit='b'*40, private_scope=True)


def test_extended_evaluation_refuses_missing_guard_before_output(tmp_path):
    from gx1.scripts import run_entry_native_direction_prefix_fit_window_v1 as worker
    with pytest.raises(RuntimeError, match='STATE_EXTENDED_STORAGE_GUARD_MISMATCH'):
        worker._evaluate_dynamic_state(object(), {'state_artifact_directory': tmp_path}, {},
            {'extended_forecast_minutes': 10080}, np.arange(1), cutoff=1, kind='OOF',
            counters={}, ceiling=1, device=None, torch=None, np=np, tree_sha=None)
    assert list(tmp_path.iterdir()) == []


def test_extended_evaluation_refuses_missing_host_identity_before_output(tmp_path):
    from gx1.scripts import run_entry_native_direction_prefix_fit_window_v1 as worker
    guard = {'recipe': _recipe(), 'private_scope': False, 'recovery': False, 'host_identity': None}
    with pytest.raises(RuntimeError, match='STATE_EXTENDED_STORAGE_IDENTITY_REQUIRED'):
        worker._evaluate_dynamic_state(object(), {'state_artifact_directory': tmp_path, 'extended_storage_guard': guard}, {},
            {'extended_forecast_minutes': 10080}, np.arange(1), cutoff=1, kind='OOF',
            counters={}, ceiling=1, device=None, torch=None, np=np, tree_sha=None)
    assert list(tmp_path.iterdir()) == []
