"""One exact failed-capture recovery; JSON/source metadata, no model execution."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as prefix
from gx1.contracts import local_random_access_campaign_v2 as campaign
from gx1.contracts import market_state_native_F_private_campaign_v1 as mechanical
from gx1.contracts import market_state_time_query_fold1_fit_v1 as scope


def _seal(value, key='recipe_sha256'):
    return {**value, key: campaign.canonical_sha256({k: v for k, v in value.items() if k != key})}


@pytest.fixture(scope='module')
def metadata():
    docs = {}
    for name, ref in scope.TA_M5_SCORE_RECOVERY_REFS.items():
        if name in ('final_checkpoint', 'initial_state'):
            continue
        raw = Path(ref['path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == ref['sha256']
        docs[ref['path']] = json.loads(raw)
    original = docs[scope.TA_M5_SCORE_RECOVERY_REFS['original_recipe']['path']]
    current = {name: campaign.file_sha256(scope.CANONICAL_REPO / name)
               for name in scope.required_source_paths(ta_sf=True)}
    historical = {name: mechanical._ta_m5_git_source_sha(
        scope.TA_M5_SCORE_RECOVERY_SOURCE_COMMIT, name)
        for name in scope.TA_M5_SCORE_RECOVERY_CHANGED_SOURCE_PATHS}
    return docs, original, current, historical


@pytest.fixture
def bound(monkeypatch, metadata):
    docs, original, current, historical = copy.deepcopy(metadata)
    require_binding = campaign.require_binding
    monkeypatch.setattr(campaign, 'require_binding', lambda ref, **kw:
        require_binding(ref, label=kw.get('label', 'test'), verify_file=False))
    monkeypatch.setattr(mechanical, '_read', lambda ref, label: copy.deepcopy(docs[ref['path']]))
    monkeypatch.setattr(scope, '_ta_m5_next_native_pass_qualification',
                        lambda: (copy.deepcopy(original['input_refs']), copy.deepcopy(original)))
    monkeypatch.setattr(scope, 'require_checkpoint_view', lambda identity, **kw: identity)
    monkeypatch.setattr(mechanical, '_ta_m5_git_source_sha',
        lambda commit, name: historical[name] if commit == scope.TA_M5_SCORE_RECOVERY_SOURCE_COMMIT else None)
    monkeypatch.setattr(campaign, 'file_sha256',
        lambda path: current[str(Path(path).relative_to(scope.CANONICAL_REPO))])
    value = scope.build_ta_m5_score_recovery_launch_recipe(
        source_commit=scope.TA_M5_SCORE_RECOVERY_SOURCE_COMMIT)
    return value, docs


def _require(value, **kwargs):
    return scope.require_launch_recipe(value, source_repo=scope.CANONICAL_REPO,
        source_commit=scope.TA_M5_SCORE_RECOVERY_SOURCE_COMMIT, **kwargs)


def test_recovery_reuses_initial_and_final_without_refit_or_new_initial(bound):
    value, _ = bound
    assert _require(value) == value
    recovered = scope.read_ta_m5_score_recovery_origin(value)
    assert recovered['origin'] == value['scoring_recovery_origin']
    assert recovered['result']['outcome'] == recovered['receipt']['outcome'] == 'FAILED'
    assert recovered['result']['optimizer_updates'] == 1849
    assert recovered['initial_measurement'] == recovered['origin']['initial_measurement']
    assert recovered['initial_payload']['measurement']['rows'] == 29574
    assert recovered['cursor']['full_state'] == recovered['origin']['final_checkpoint']
    assert value['numerical_recipe'] == recovered['original_recipe']['numerical_recipe']
    assert value['max_forwards'] == 2959 and value['max_windows'] == 1
    assert value['max_optimizer_updates'] == value['max_backwards'] == value['max_initial_forwards'] == 0
    assert value['training_enabled'] is value['full_epoch_training_allowed'] is False
    assert value['measurement_scope'] == {
        'initial_fit_rows': 0, 'reused_initial_fit_rows': 29574,
        'final_fit_rows': 29574, 'later_train_rows': 17753, 'matched_initial_final': True}
    assert value['capture_m5_readout'] is True
    assert value['m5_readout_capture_measurements'] == ['FINAL_TRAIN', 'OOF']


@pytest.mark.parametrize(('key', 'replacement'), [
    ('training_enabled', True), ('full_epoch_training_allowed', True),
    ('max_optimizer_updates', 1), ('max_initial_forwards', 1849), ('max_backwards', 1),
    ('max_forwards', 4808), ('max_windows', 4), ('capture_m5_readout', False),
])
def test_resealed_recipe_cannot_authorize_refit_or_extra_baseline(bound, key, replacement):
    value, _ = bound
    with pytest.raises(Exception, match='M5_SCORE_RECOVERY_RECIPE_SCOPE'):
        _require(_seal({**value, key: replacement}))


@pytest.mark.parametrize(('fault', 'reason'), [
    ('receipt_pass', 'FAILED_TERMINAL'), ('observer_failed', 'FAILED_TERMINAL'),
    ('wrong_error', 'COMPLETED_FIT'), ('one_skip', 'COMPLETED_FIT'),
    ('old_final_present', 'COMPLETED_FIT'), ('incomplete_fit', 'COMPLETED_FIT'),
    ('initial_rows32', 'REUSED_INITIAL'),
])
def test_only_exact_failed_capture_after_completed_fit_is_recoverable(bound, fault, reason):
    value, docs = bound
    refs = scope.TA_M5_SCORE_RECOVERY_REFS
    result = docs[refs['failed_result']['path']]
    receipt = docs[refs['failed_receipt']['path']]
    if fault == 'receipt_pass':
        receipt['outcome'] = 'COMPLETE'
    elif fault == 'observer_failed':
        receipt['progress_observer_exit_code'] = 1
    elif fault == 'wrong_error':
        result['error'] = 'different_failure'
    elif fault == 'one_skip':
        result['skipped_batches'] = 1
    elif fault == 'old_final_present':
        result['final_train_measurement'] = refs['initial_measurement']
    elif fault == 'incomplete_fit':
        result['parent_cursor'] = 29568
    else:
        docs[refs['initial_measurement']['path']]['measurement']['rows'] = 32
    with pytest.raises(Exception, match='M5_SCORE_RECOVERY_' + reason):
        scope.read_ta_m5_score_recovery_origin(value)


@pytest.mark.parametrize('name', [
    'gx1/contracts/market_state_time_query_fold1_fit_state_v1.py',
    'gx1/contracts/market_state_time_query_native_F_objective_v1.py',
    'gx1/contracts/market_state_time_query_native_F_model_v1.py',
])
def test_numeric_model_and_state_owners_remain_unchanged(bound, name):
    value, _ = bound
    value['source_owners'][name]['sha256'] = '0' * 64
    with pytest.raises(Exception, match='M5_SCORE_RECOVERY_UNQUALIFIED_SOURCE'):
        _require(_seal(value))


def test_original_failed_origin_cannot_be_reset_or_substituted(bound):
    value, _ = bound
    value['scoring_recovery_origin']['final_checkpoint'] = value['scoring_recovery_origin']['initial_state']
    with pytest.raises(Exception, match='M5_SCORE_RECOVERY_ORIGIN'):
        scope.read_ta_m5_score_recovery_origin(value)
    with pytest.raises(Exception, match='M5_SCORE_RECOVERY_RECIPE_SCOPE'):
        _require(_seal(value))
    with pytest.raises(Exception, match='M5_SCORE_RECOVERY_PHYSICAL_METADATA_REQUIRED'):
        _require(value, verify_files=False)


def test_dedicated_scoring_gate_requires_closed_training_and_same_origin(bound):
    value, _ = bound
    gate = dict(decision='ADMITTED', status='ADMITTED',
        ta_m5_score_recovery_authorized=True, checkpoint_reads_authorized=True,
        zero_update_scoring_recovery_authorized=True, capture_m5_readout=True,
        scoring_recovery_origin=value['scoring_recovery_origin'], physical_power_limit_w=250,
        fit_rows=29574, score_rows=17753, max_optimizer_updates=0, max_initial_forwards=0,
        training_enabled=False, native_training_enabled=False, full_epoch_training_allowed=False,
        genuine_backward_authorized=False, genuine_updates_authorized=False,
        genuine_predictor_fit_authorized=False, test_data_used=False)
    scope.require_ta_m5_score_recovery_gate(gate, value)
    for key, replacement in (
        ('training_enabled', True), ('genuine_updates_authorized', True),
        ('max_optimizer_updates', 1), ('max_initial_forwards', 1),
        ('scoring_recovery_origin', {}), ('physical_power_limit_w', 160),
    ):
        with pytest.raises(Exception, match='M5_SCORE_RECOVERY_POLICY'):
            scope.require_ta_m5_score_recovery_gate({**gate, key: replacement}, value)
    with pytest.raises(Exception, match='POLICY_GATE'):
        scope.require_fit_gate(gate, value)


def _invocation(value, monkeypatch):
    ref = {'path': str(scope.TA_M5_SCORE_RECOVERY_RECIPE_PATH), 'sha256': 'a' * 64}
    window_ref = {'path': '/tmp/score-recovery-window.json', 'sha256': 'b' * 64}
    pointer = str(scope.TA_M5_SCORE_RECOVERY_RUNTIME_ROOT / 'prefix-cursor/RESUME_CURSOR.json')
    progress = str(scope.TA_M5_SCORE_RECOVERY_RUNTIME_ROOT / 'progress/invocation-0001.json')
    window = _seal(dict(schema_version=prefix.WINDOW_SCHEMA, recipe=ref, invocation_number=1,
        max_invocation_seconds=12000, progress_path=progress, campaign_cursor_path=pointer,
        output_directory=str(scope.TA_M5_SCORE_RECOVERY_RUNTIME_ROOT / 'prefix-fit'),
        test_data_used=False), 'policy_sha256')
    execution = dict(prelaunch_manifest=ref, train_session_manifest=window_ref,
        prelaunch_manifest_sha256=value['recipe_sha256'],
        train_session_manifest_sha256=window['policy_sha256'])
    invocation = dict(kind=prefix.KIND, invocation_number=1, batch_size=16, epoch_index=None,
        window_index=0, window_count=1, optimizer_step_budget=0, expected_success_outcome='COMPLETE',
        maximum_wall_seconds=13800, requires_fresh_windows_boot=True,
        checkpoint=dict(pointer_path=pointer, before_mode='GENESIS',
                        predecessor_invocation_number=None, write_mode='ATOMIC_UPDATE'),
        progress_path=progress, invocation_sha256='c' * 64,
        launcher_argv=['capped', '--', str(scope.CANONICAL_REPO / '.venv/bin/python'),
            '-m', prefix.MODULE, '--window-policy', window_ref['path'],
            '--window-policy-file-sha256', window_ref['sha256'], '--progress-path', progress])
    monkeypatch.setattr(prefix, 'require_prefix_fit_launch_recipe', lambda *args, **kw: value)
    monkeypatch.setattr(campaign, 'read_bound_json', lambda *args: window)
    prefix.require_prefix_fit_invocation_execution(
        invocation, execution, source_repo=scope.CANONICAL_REPO, source_commit='a' * 40)
    return invocation, execution, window


def test_zero_physical_budget_retains_historical_progress_1849(monkeypatch, bound):
    value, _ = bound
    invocation, execution, window = _invocation(value, monkeypatch)
    assert invocation['direction_prefix_fit_scoring_recovery'] is True
    assert prefix.guarded_prefix_power_limit(value) == 250
    origin = value['scoring_recovery_origin']
    artifact = {'path': '/tmp/score-recovery-final.json', 'sha256': 'd' * 64}
    cursor = prefix.build_prefix_fit_cursor(
        recipe_ref=window['recipe'], fold=scope.FOLD, feature_view_identity={'prefix_normalization': {}},
        phase_state='COMPLETE', outcome='COMPLETE', optimizer_updates=1849,
        parent_cursor=29574, parent_order_sha256='e' * 64,
        full_state=origin['final_checkpoint'], own_initial_state=origin['initial_state'],
        initial_measurement=origin['initial_measurement'], final_train_measurement=artifact,
        oof_cursor=17753, result=artifact)
    progress = prefix.build_prefix_fit_progress(window=window, plan={'plan_sha256': 'f' * 64},
        invocation=invocation,
        checkpoint_pointer={'path': invocation['checkpoint']['pointer_path'], 'sha256': 'd' * 64},
        cursor=cursor)
    assert progress['optimizer_updates'] == 1849 and progress['outcome'] == 'COMPLETE'
    invocation['optimizer_step_budget'] = 1
    with pytest.raises(Exception, match='INVOCATION_BINDING'):
        prefix.require_prefix_fit_invocation_execution(
            invocation, execution, source_repo=scope.CANONICAL_REPO, source_commit='a' * 40)


def test_origin_and_dispatch_select_only_new_scope(monkeypatch, bound, tmp_path):
    value, _ = bound
    assert prefix.require_scoring_recovery_origin(
        value, source_repo=scope.CANONICAL_REPO,
        source_commit=scope.TA_M5_SCORE_RECOVERY_SOURCE_COMMIT)['origin'] == value['scoring_recovery_origin']
    from gx1.contracts import gx1_capped_execution_v1 as capped
    from gx1.scripts import run_entry_native_direction_prefix_fit_window_v1 as dispatch
    from gx1.scripts import run_market_state_time_query_fold1_fit_window_v1 as fit
    calls = []
    limits = {'synthetic': True}
    parts = ({'window': 1}, {'plan': 2}, {'invocation': 3}, {'marker': 4}, value)
    monkeypatch.setattr(capped, 'require_guarded_cuda_trainer_execution',
                        lambda: calls.append('guard') or limits)
    monkeypatch.setattr(prefix, 'require_native_direction_prefix_fit_context',
                        lambda *args: calls.append('context') or parts)
    def run(**kwargs):
        calls.append('score')
        assert kwargs['launch'] is value and kwargs['limits'] is limits
        return {'synthetic': True}
    monkeypatch.setattr(fit, 'run_market_state_time_query_fold1_fit_window', run)
    assert dispatch.run_native_direction_prefix_fit_window(
        policy_path=tmp_path / 'WINDOW.json', policy_file_sha256='a' * 64,
        progress_path=tmp_path / 'PROGRESS.json') == {'synthetic': True}
    assert calls == ['guard', 'context', 'score'] and list(tmp_path.iterdir()) == []


def test_materializer_recognizes_only_exact_score_recovery_root(monkeypatch):
    from gx1.scripts import materialize_market_state_time_query_fold1_fit_campaign_v1 as materializer
    path = scope.TA_M5_SCORE_RECOVERY_ROOT / 'PREPARATION_EXECUTION_BINDING.json'
    reads = []
    monkeypatch.setattr(campaign, 'file_sha256', lambda path: 'a' * 64)
    monkeypatch.setattr(campaign, 'read_bound_json', lambda path, sha: reads.append(path) or {})
    with pytest.raises(Exception, match='MATERIALIZER_BINDING_FIELDS'):
        materializer.materialize(path)
    with pytest.raises(Exception, match='MATERIALIZER_BINDING_PATH'):
        materializer.materialize(path.with_name('OTHER_BINDING.json'))
    assert reads == [path]
