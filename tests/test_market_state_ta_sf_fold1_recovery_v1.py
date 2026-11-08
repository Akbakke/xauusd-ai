"""Small recovery metadata and import tests; no model, data or checkpoint IO."""
import copy
from pathlib import Path

import pytest

from gx1.contracts import market_state_time_query_fold1_fit_v1 as scope
from gx1.scripts import run_market_state_time_query_fold1_fit_window_v1 as runner


def _seal(value, key='contract_sha256'):
    value = copy.deepcopy(value)
    if key == 'contract_sha256':
        from gx1.contracts.market_state_time_query_native_F_objective_v1 import _seal as endpoint_seal
        return endpoint_seal(value)
    value[key] = scope.campaign.canonical_sha256(value)
    return value


@pytest.fixture
def origin_case(monkeypatch):
    def ref(path):
        return {'path': str(path), 'sha256': 'a' * 64}
    root = scope.TA_SF_RUNTIME_ROOT
    origin = {
        'schema_version': scope.TA_RECOVERY_ORIGIN_SCHEMA,
        'source_commit': scope.TA_RECOVERY_SOURCE_COMMIT,
        'original_recipe': ref(scope.TA_SF_RECIPE_PATH),
        'original_plan': ref(root.parent / 'CAMPAIGN/CAMPAIGN_PLAN.json'),
        'interruption_terminal': ref(root.parent / 'OPERATOR_INTERRUPTION_TERMINAL.json'),
        'initial_state': ref(root / 'prefix-fit/states/INITIAL_FOLD1_STATE.pt'),
        'initial_measurement': ref(root / 'prefix-fit/window-0001/INITIAL_TRAIN.json'),
        'checkpoint': ref(root / 'prefix-fit/states/w0001_p004096.pt'),
    }
    view_sha = 'b' * 64
    original = {'expected_model_contract_sha256': 'c' * 64,
                'expected_loss_contract_sha256': 'd' * 64}
    metadata = []
    for name, cursor, updates, seal in (
            ('initial_state', 0, 0, 'e'), ('checkpoint', 4096, 256, 'f')):
        metadata.append({
            **origin[name], 'schema_version': 'gx1_market_state_ta_sf_fold1_fit_state_v1',
            'source_commit': origin['source_commit'], 'cursor': cursor,
            'optimizer_updates': updates, 'skipped_batches': 0,
            'contract_sha256': seal * 64, 'model_state_sha256': seal * 64,
            'feature_view_sha256': view_sha, 'state_and_component_digest_checks': 'PASS'})
    terminal = {
        'schema_version': 'gx1_operator_interruption_terminal_v1',
        'status': 'ABORTED_CONSUMED', 'source_commit_at_interruption': origin['source_commit'],
        'source_changed_during_trainer': False, 'old_plan_relaunch_allowed': False,
        'originals_modified': False, 'test_data_used': False, 'full_fit_complete': False,
        'final_train_measurement_complete': False, 'later_train_score_complete': False,
        'last_preserved_parent_cursor': 4096, 'last_preserved_optimizer_updates': 256,
        'initial_measurement_optimizer_updates': 0, 'preserved_checkpoints': metadata,
        'initial_measurement': origin['initial_measurement'],
    }
    measured = _seal({
        'schema_version': 'gx1_native_S_F_endpoint_measurement_v1',
        'partition': 'fit', 'fold_id': 1, 'rows': scope.FIT_ROWS,
        'batches': scope.MAX_UPDATES, 'feature_view_sha256': view_sha,
        'model_contract_sha256': original['expected_model_contract_sha256'],
        'loss_contract_sha256': original['expected_loss_contract_sha256']})
    initial = {
        'schema_version': 'gx1_market_state_ta_sf_fold1_evaluation_v1',
        'fold_id': 1, 'optimizer_updates': 0, 'test_data_used': False,
        'predictive_value_claimed': False, 'evaluated_checkpoint': origin['initial_state'],
        'evaluated_model_state_sha256': metadata[0]['model_state_sha256'],
        'measurement': measured}
    plan = _seal({'source_commit': origin['source_commit'], 'source_repo': str(scope.CANONICAL_REPO),
                  'runtime_root': str(root), 'recipe': origin['original_recipe']}, 'plan_sha256')
    documents = {origin[key]['path']: value for key, value in (
        ('original_recipe', original), ('original_plan', plan),
        ('interruption_terminal', terminal), ('initial_measurement', initial))}
    def require_binding(value, **kwargs):
        assert kwargs['verify_file'] is True
        assert set(value) == {'path', 'sha256'}
        if value['sha256'] != 'a' * 64:
            raise ValueError('STALE_FILE')
        return dict(value)
    monkeypatch.setattr(scope.campaign, 'require_binding', require_binding)
    monkeypatch.setattr(scope.campaign, 'read_bound_json',
                        lambda path, digest: documents[str(path)])
    return origin, view_sha, documents


def test_consumed_origin_keeps_complete_initial_and_exact_checkpoint(origin_case):
    origin, view, _ = origin_case
    result = scope.read_ta_recovery_origin(origin, expected_view_sha256=view)
    assert result['checkpoint_metadata']['cursor'] == 4096
    assert result['checkpoint_metadata']['optimizer_updates'] == 256
    assert result['initial_payload']['measurement']['rows'] == 29574
    assert result['origin'] == origin


@pytest.mark.parametrize('change', [
    'wrong_checkpoint_path', 'stale_checkpoint_sha', 'wrong_source', 'extra_origin_key',
    'terminal_complete', 'wrong_cursor', 'wrong_updates', 'wrong_state_sha',
    'partial_initial', 'wrong_initial_state', 'changed_initial_seal', 'wrong_plan_source'])
def test_origin_rejects_unqualified_import(origin_case, change):
    origin, view, docs = origin_case
    terminal = docs[origin['interruption_terminal']['path']]
    initial = docs[origin['initial_measurement']['path']]
    if change == 'wrong_checkpoint_path':
        origin['checkpoint']['path'] = str(scope.TA_RECOVERY_RUNTIME_ROOT / 'states/arbitrary.pt')
    elif change == 'stale_checkpoint_sha':
        origin['checkpoint']['sha256'] = '9' * 64
    elif change == 'wrong_source':
        origin['source_commit'] = '9' * 40
    elif change == 'extra_origin_key':
        origin['allow_generic_import'] = True
    elif change == 'terminal_complete':
        terminal['status'] = 'COMPLETE'
    elif change == 'wrong_cursor':
        terminal['preserved_checkpoints'][1]['cursor'] = 4112
    elif change == 'wrong_updates':
        terminal['preserved_checkpoints'][1]['optimizer_updates'] = 255
    elif change == 'wrong_state_sha':
        terminal['preserved_checkpoints'][1]['contract_sha256'] = 'invalid'
    elif change == 'partial_initial':
        initial['measurement']['rows'] -= 1
    elif change == 'wrong_initial_state':
        initial['evaluated_checkpoint'] = origin['checkpoint']
    elif change == 'changed_initial_seal':
        initial['measurement']['contract_sha256'] = '9' * 64
    else:
        docs[origin['original_plan']['path']]['source_commit'] = '9' * 40
    with pytest.raises((scope.campaign.RandomAccessCampaignError, ValueError)):
        scope.read_ta_recovery_origin(origin, expected_view_sha256=view)


def _import(origin_case, *, fresh_change=None, checkpoint_change=None, restored_change=None):
    origin, view, _ = origin_case
    recovery = scope.read_ta_recovery_origin(origin, expected_view_sha256=view)
    initial = copy.deepcopy(recovery['initial_metadata'])
    checkpoint = copy.deepcopy(recovery['checkpoint_metadata'])
    fresh = copy.deepcopy(initial)
    if fresh_change:
        fresh.update(fresh_change)
    if checkpoint_change:
        checkpoint.update(checkpoint_change)
    calls = []
    def load(ref):
        calls.append(('load', ref))
        return initial if ref == origin['initial_state'] else checkpoint
    def restore(loaded, model, optimizer, **kwargs):
        calls.append(('restore', kwargs))
        assert loaded is checkpoint
        assert model == 'model' and optimizer == 'adamw'
        return {'cursor': 4096, 'optimizer_updates': 256, 'skipped_batches': 0,
                **(restored_change or {})}
    result = runner._ta_restore_recovery(
        recovery=recovery, fresh=fresh, model='model', optimizer='adamw',
        numerical={'recipe': 'unchanged'}, view_identity={'contract_sha256': view},
        order='exact-parent-order', device='cuda', load_state=load, restore_state=restore)
    return result, calls


def test_full_restore_uses_old_provenance_and_exact_order_without_evaluation(origin_case):
    result, calls = _import(origin_case)
    assert result == {'cursor': 4096, 'optimizer_updates': 256, 'skipped_batches': 0}
    assert [call[0] for call in calls] == ['load', 'load', 'restore']
    assert calls[-1][1]['source_commit'] == scope.TA_RECOVERY_SOURCE_COMMIT
    assert calls[-1][1]['parent_order'] == 'exact-parent-order'
    assert calls[-1][1]['numerical_recipe'] == {'recipe': 'unchanged'}


@pytest.mark.parametrize('change', [
    {'source_commit': '9' * 40}, {'contract_sha256': '9' * 64},
    {'model_state_sha256': '9' * 64}])
def test_fresh_initial_must_match_original_full_state(origin_case, change):
    with pytest.raises(RuntimeError, match='FRESH_INITIAL_BASELINE'):
        _import(origin_case, fresh_change=change)


@pytest.mark.parametrize('change', [
    {'source_commit': '9' * 40}, {'contract_sha256': '9' * 64},
    {'cursor': 4112}, {'optimizer_updates': 257}, {'skipped_batches': 1},
    {'feature_view_sha256': '9' * 64}, {'model_state_sha256': '9' * 64}])
def test_checkpoint_cannot_change_bound_state_or_skip_rows(origin_case, change):
    with pytest.raises(RuntimeError, match='TA_RECOVERY_CHECKPOINT'):
        _import(origin_case, checkpoint_change=change)


def test_restore_return_cannot_skip_an_unconsumed_batch(origin_case):
    with pytest.raises(RuntimeError, match='TA_RECOVERY_EXACT_PROGRESS'):
        _import(origin_case, restored_change={'cursor': 4112, 'optimizer_updates': 257})


def test_recovery_needs_physical_metadata_and_explicit_policy():
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='PHYSICAL_METADATA'):
        scope.require_launch_recipe({'run_scope': scope.TA_RECOVERY_SCOPE},
            source_repo=scope.CANONICAL_REPO, source_commit='9' * 40, verify_files=False)
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='RECOVERY_POLICY'):
        scope.require_fit_gate({}, {'run_scope': scope.TA_RECOVERY_SCOPE,
                                   'recovery_origin': {}})


@pytest.mark.parametrize('changed_owner', [
    None,
    'gx1/contracts/market_state_time_query_native_F_model_v1.py',
    'gx1/contracts/market_state_time_query_native_F_objective_v1.py',
    'gx1/contracts/market_state_time_query_target_v1.py',
    'gx1/contracts/entry_model_native_input_normalization_v1.py'])
def test_recovery_source_bridge_retains_original_qualification(monkeypatch, changed_owner):
    signal = 'gx1/contracts/entry_model_native_signal_v1.py'
    model = changed_owner or 'gx1/contracts/market_state_time_query_native_F_model_v1.py'
    owners = {name: {'path': str(scope.CANONICAL_REPO / name), 'sha256': 'a' * 64}
              for name in (signal, model)}
    original = _seal({'schema_version': scope.RECIPE_SCHEMA, 'source_owners': owners,
                     'run_scope': scope.TA_SF_SCOPE}, 'recipe_sha256')
    new_owners = copy.deepcopy(owners)
    new_owners[signal]['sha256'] = scope.TA_RECOVERY_SIGNAL_SHA256
    if changed_owner:
        new_owners[changed_owner]['sha256'] = '9' * 64
    raw = {**original, 'run_id': scope.TA_RECOVERY_STAGE,
        'admission_stage': scope.TA_RECOVERY_STAGE, 'run_scope': scope.TA_RECOVERY_SCOPE,
        'runtime_root': str(scope.TA_RECOVERY_RUNTIME_ROOT),
        'execution_binding_path': str(scope.TA_RECOVERY_RUNTIME_ROOT / 'EXECUTION_BINDING.json'),
        'source_owners': new_owners, 'recovery_origin': {}}
    raw.pop('recipe_sha256')
    recipe = _seal(raw, 'recipe_sha256')
    monkeypatch.setattr(scope, 'read_ta_recovery_origin',
                        lambda *args, **kwargs: {'original_recipe': original})
    qualified = []
    def require_old(value, **kwargs):
        qualified.append(kwargs)
        assert value == original
        return value
    monkeypatch.setattr(scope, '_require_ta_launch_recipe', require_old)
    monkeypatch.setattr(scope, '_require_ta_source_owners', lambda *args, **kwargs: None)
    if changed_owner:
        with pytest.raises(scope.campaign.RandomAccessCampaignError, match='SOURCE_BRIDGE'):
            scope.require_launch_recipe(recipe, source_repo=scope.CANONICAL_REPO,
                                        source_commit='9' * 40)
    else:
        assert scope.require_launch_recipe(recipe, source_repo=scope.CANONICAL_REPO,
                                          source_commit='9' * 40) == recipe
    assert qualified[0]['historical_source_commit'] == scope.TA_RECOVERY_SOURCE_COMMIT
    assert qualified[0]['verify_files'] is True
