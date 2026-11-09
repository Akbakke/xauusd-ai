"""Dedicated M5 full-fit metadata/route checks; no model, checkpoint payload or fit."""
import copy
from pathlib import Path

import pytest

from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as prefix
from gx1.contracts import local_random_access_campaign_v2 as campaign
from gx1.contracts import market_state_native_F_private_campaign_v1 as mechanical
from gx1.contracts import market_state_time_query_fold1_fit_v1 as scope


def _seal(value):
    value.pop('recipe_sha256', None)
    value['recipe_sha256'] = campaign.canonical_sha256(value)
    return value


@pytest.fixture(scope='module')
def metadata():
    refs = scope.ta_m5_fit_bound_input_refs()
    names = ('ta_input_audit', 'ta_input_terminal', 'original_candidate_view',
        'ta_candidate_view', 'source_review', 'source_metadata_check',
        'original_recipe', 'original_result', 'original_final_measurement',
        'original_endpoint_priors', 'ta_endpoint_priors', *scope.TA_M5_FIT_MECHANICAL_REFS)
    docs = {refs[name]['path']: mechanical._read(refs[name], name) for name in names}
    current = {name: campaign.file_sha256(scope.CANONICAL_REPO / name)
               for name in scope.required_source_paths(ta_sf=True)}
    old = {(scope.TA_M5_FIT_MECHANICAL_COMMIT, name):
            mechanical._ta_m5_git_source_sha(scope.TA_M5_FIT_MECHANICAL_COMMIT, name)
           for name in scope.TA_M5_FIT_CHANGED_SOURCE_PATHS}
    model_path = 'gx1/contracts/market_state_time_query_model_v2.py'
    old[mechanical.TA_M5_REVIEWED_COMMIT, model_path] = mechanical._ta_m5_git_source_sha(
        mechanical.TA_M5_REVIEWED_COMMIT, model_path)
    return refs, docs, current, old


@pytest.fixture
def bound(monkeypatch, metadata):
    refs, docs, current, old = copy.deepcopy(metadata)
    original_require = campaign.require_binding
    # Preserve binding syntax; no test hashes or deserializes the large checkpoint.
    monkeypatch.setattr(campaign, 'require_binding',
        lambda ref, **kw: original_require(ref, label=kw.get('label', 'test'), verify_file=False))
    monkeypatch.setattr(mechanical, '_read', lambda ref, label: copy.deepcopy(docs[ref['path']]))
    monkeypatch.setattr(mechanical, '_ta_m5_git_source_sha', lambda commit, name: old[commit, name])
    monkeypatch.setattr(campaign, 'file_sha256',
        lambda path: current[str(Path(path).relative_to(scope.CANONICAL_REPO))])
    value = scope.build_ta_m5_fit_launch_recipe(source_commit=scope.TA_M5_FIT_MECHANICAL_COMMIT)
    return value, refs, docs


def _require(value):
    return scope.require_launch_recipe(value, source_repo=scope.CANONICAL_REPO,
                                      source_commit=scope.TA_M5_FIT_MECHANICAL_COMMIT)


def test_new_full_fit_uses_original_final_fresh_optimizer_and_same_population(bound):
    value, refs, docs = bound
    assert _require(value) == value
    assert value['run_scope'] == scope.TA_M5_FIT_SCOPE
    assert value['runtime_root'] == str(scope.TA_M5_FIT_RUNTIME_ROOT)
    assert value['fold'] == scope.FOLD
    assert (scope.FIT_ROWS, scope.SCORE_ROWS, value['train_passes'], value['max_windows']) == (29574, 17753, 1, 4)
    assert value['numerical_recipe'] == scope.ta_sf_numerical_recipe(dedicated_m5_side=True)
    assert value['numerical_recipe']['max_updates'] == 1849
    qualified = docs[refs['ta_mechanical_recipe']['path']]
    report = docs[refs['ta_mechanical_qualification']['path']]
    assert value['warmstart_origin'] == qualified['warmstart_origin']
    assert value['warmstart_origin']['original_checkpoint'] == refs['original_checkpoint']
    assert value['warmstart_origin']['optimizer_migration'] == 'fresh_empty_AdamW_cursor0'
    assert value['expected_initial_model_state_sha256'] == report['weight_migrations'][0]['model_state_sha256']
    assert value['qualification_weights_for_main_fit'] is False
    assert scope.ta_sf_numerical_recipe() == docs[refs['original_recipe']['path']]['numerical_recipe']
    changed = {name for name, ref in value['source_owners'].items()
               if ref != qualified['source_owners'][name]}
    assert changed == scope.TA_M5_FIT_CHANGED_SOURCE_PATHS


@pytest.mark.parametrize(('key', 'replacement'), [
    ('train_passes', 2), ('max_windows', 5), ('checkpoint_every_updates', 512),
    ('expected_initial_model_state_sha256', '0' * 64),
    ('mechanical_source_commit', '0' * 40), ('qualification_weights_for_main_fit', True),
])
def test_resealed_recipe_cannot_change_scope_or_initial_identity(bound, key, replacement):
    value, _, _ = bound
    value[key] = replacement
    with pytest.raises(Exception, match='TA_M5_FIT_RECIPE_SCOPE'):
        _require(_seal(value))


def test_resealed_origin_cannot_inherit_old_optimizer_or_qualification_weights(bound):
    value, _, _ = bound
    value['warmstart_origin']['optimizer_migration'] = 'inherit_old_moments'
    with pytest.raises(Exception, match='TA_M5_FIT_RECIPE_SCOPE'):
        _require(_seal(value))


@pytest.mark.parametrize('damage', ['terminal', 'origin', 'migration', 'parity'])
def test_completed_mechanics_requires_actual_restore_parity_and_terminal(bound, damage):
    _, refs, docs = bound
    report = docs[refs['ta_mechanical_qualification']['path']]
    if damage == 'terminal':
        docs[refs['ta_mechanical_terminal']['path']]['status'] = 'FAILED'
        reason = 'TA_M5_FIT_QUALIFICATION'
    elif damage == 'origin':
        report['legacy_import']['strict_full_restore'] = False
        reason = 'TA_M5_FIT_ORIGINAL_RESTORE'
    elif damage == 'migration':
        report['weight_migrations'][1]['model_state_sha256'] = '0' * 64
        reason = 'TA_M5_FIT_INITIAL_PARITY'
    else:
        report['baseline_output_parity'][0]['tasks']['future']['unchanged_coordinates_bit_equal'] = False
        reason = 'TA_M5_FIT_OUTPUT_PARITY'
    with pytest.raises(Exception, match=reason):
        scope._ta_m5_fit_qualification(refs)


@pytest.mark.parametrize('name', [
    'gx1/contracts/market_state_time_query_native_F_model_v1.py',
    'gx1/contracts/market_state_native_F_private_campaign_v1.py',
    '.venv/lib/python3.10/site-packages/torch/optim/adamw.py',
])
def test_only_five_fit_orchestration_owners_may_change(bound, name):
    value, _, _ = bound
    value['source_owners'][name]['sha256'] = '0' * 64
    with pytest.raises(Exception, match='TA_M5_FIT_UNQUALIFIED_SOURCE'):
        _require(_seal(value))


def test_changed_owner_requires_exact_historical_git_binding(bound):
    value, refs, docs = bound
    qualified = copy.deepcopy(docs[refs['ta_mechanical_recipe']['path']]['source_owners'])
    name = 'gx1/contracts/market_state_time_query_fold1_fit_v1.py'
    qualified[name]['sha256'] = '0' * 64
    with pytest.raises(Exception, match='TA_M5_FIT_HISTORICAL_SOURCE'):
        scope._require_ta_m5_fit_sources(value['source_owners'], qualified,
            source_commit=scope.TA_M5_FIT_MECHANICAL_COMMIT)


def test_physical_validation_cannot_be_disabled(bound):
    with pytest.raises(Exception, match='TA_M5_FIT_PHYSICAL_METADATA_REQUIRED'):
        scope.require_launch_recipe(bound[0], source_repo=scope.CANONICAL_REPO,
            source_commit=scope.TA_M5_FIT_MECHANICAL_COMMIT, verify_files=False)


def test_fit_gate_does_not_reuse_consumed_mechanical_admission(bound):
    value, refs, _ = bound
    gate = dict(decision='ADMITTED', status='ADMITTED',
        time_query_fold1_fit_authorized=True, ta_sf_fold1_fit_authorized=True,
        ta_m5_fold1_fit_authorized=True, warmstart_origin=value['warmstart_origin'],
        input_audit=refs['ta_input_audit'], mechanical_qualification=refs['ta_mechanical_qualification'],
        fold_id=1, fit_rows=29574, score_rows=17753, max_optimizer_updates=1849,
        genuine_predictor_fit_authorized=True, test_data_used=False)
    scope.require_fit_gate(gate, value)
    with pytest.raises(Exception, match='TA_M5_FIT_POLICY'):
        scope.require_fit_gate({**gate, 'ta_m5_fold1_fit_authorized': False}, value)
    with pytest.raises(Exception, match='POLICY_GATE'):
        scope.require_fit_gate({**gate, 'genuine_predictor_fit_authorized': False}, value)


def test_checkpoint_mode_is_derived_from_explicit_scope_and_numerical_recipe(monkeypatch, bound):
    value, _, _ = bound
    calls = []
    monkeypatch.setattr(scope, '_require_ta_checkpoint_view',
        lambda identity, **kw: calls.append(kw) or identity)
    identity = {'synthetic': True}
    assert scope.require_checkpoint_view(identity, recipe=value, ta_sf=True) is identity
    assert calls[-1]['dedicated_m5_side'] is True
    scope.require_checkpoint_view(identity, ta_sf=True)
    assert calls[-1]['dedicated_m5_side'] is False
    with pytest.raises(Exception, match='TA_M5_VIEW_NUMERICAL'):
        scope.require_checkpoint_view(identity, recipe={
            **value, 'numerical_recipe': scope.ta_sf_numerical_recipe()}, ta_sf=True)


def test_m5_recipe_path_and_power_are_explicit(monkeypatch):
    ref = {'path': str(scope.TA_M5_FIT_RECIPE_PATH), 'sha256': 'a' * 64}
    value = {'run_scope': scope.TA_M5_FIT_SCOPE, 'admission_stage': scope.TA_M5_FIT_STAGE}
    monkeypatch.setattr(campaign, 'require_binding', lambda ref, **kw: ref)
    monkeypatch.setattr(campaign, 'read_bound_json', lambda *args: value)
    monkeypatch.setattr(scope, 'require_launch_recipe', lambda recipe, **kw: recipe)
    assert prefix.require_prefix_fit_launch_recipe(ref, source_repo=scope.CANONICAL_REPO,
        source_commit=scope.TA_M5_FIT_MECHANICAL_COMMIT) is value
    assert prefix.guarded_prefix_power_limit(value) == 250
    with pytest.raises(Exception, match='TIME_QUERY_RECIPE_PATH'):
        prefix.require_prefix_fit_launch_recipe(
            {**ref, 'path': str(scope.TA_RECOVERY_RECIPE_PATH)}, source_repo=scope.CANONICAL_REPO,
            source_commit=scope.TA_M5_FIT_MECHANICAL_COMMIT)
    with pytest.raises(Exception, match='TIME_QUERY_POWER_SCOPE'):
        prefix.guarded_prefix_power_limit({**value, 'admission_stage': scope.TA_SF_STAGE})


def test_m5_cursor_progress_preserves_gap_tail_and_dedicated_view(monkeypatch):
    ref = {'path': str(scope.TA_M5_FIT_RECIPE_PATH), 'sha256': 'a' * 64}
    artifact = {'path': '/tmp/gx1-m5-fit-test.json', 'sha256': 'b' * 64}
    calls = []
    monkeypatch.setattr(scope, 'require_checkpoint_view',
        lambda value, **kw: calls.append(kw) or value)
    cursor = prefix.build_prefix_fit_cursor(recipe_ref=ref, fold=scope.FOLD,
        feature_view_identity={'prefix_normalization': {}}, phase_state='COMPLETE', outcome='COMPLETE',
        optimizer_updates=1849, parent_cursor=29574, parent_order_sha256='c' * 64,
        full_state=artifact, own_initial_state=artifact, initial_measurement=artifact,
        final_train_measurement=artifact, oof_cursor=17753, result=artifact)
    assert calls and all(c['dedicated_m5_side'] is True for c in calls)
    invocation = {'direction_prefix_fit_recipe': ref, 'optimizer_step_budget': 1849,
        'kind': prefix.KIND, 'invocation_sha256': 'd' * 64,
        'checkpoint': {'pointer_path': '/tmp/gx1-m5-fit-cursor.json'}}
    progress = prefix.build_prefix_fit_progress(window={'recipe': ref},
        plan={'plan_sha256': 'e' * 64}, invocation=invocation,
        checkpoint_pointer={'path': invocation['checkpoint']['pointer_path'], 'sha256': 'f' * 64},
        cursor=cursor)
    assert progress['total_units'] == 29574 and progress['oof_cursor'] == 17753
    original_require = campaign.require_binding
    monkeypatch.setattr(campaign, 'require_binding',
        lambda ref, **kw: original_require(ref, label=kw['label'], verify_file=False))
    launch = {'schema_version': prefix.RECIPE_SCHEMA, 'fold': scope.FOLD, 'train_passes': 1,
        'run_scope': scope.TA_M5_FIT_SCOPE,
        'numerical_recipe': scope.ta_sf_numerical_recipe(dedicated_m5_side=True)}
    monkeypatch.setattr(campaign, 'read_bound_json', lambda *args: _seal(copy.deepcopy(launch)))
    assert prefix.require_prefix_fit_cursor(cursor, expected_recipe=ref,
        verify_files=True)['outcome'] == 'COMPLETE'
    assert calls[-1]['recipe']['run_scope'] == scope.TA_M5_FIT_SCOPE
    launch['numerical_recipe'] = scope.ta_sf_numerical_recipe()
    with pytest.raises(Exception, match='CURSOR_INPUT_VIEW_IDENTITY'):
        prefix.require_prefix_fit_cursor(cursor, expected_recipe=ref, verify_files=True)


def test_m5_guarded_dispatch_preserves_checked_context(monkeypatch, tmp_path):
    from gx1.contracts import gx1_capped_execution_v1 as capped
    from gx1.scripts import run_entry_native_direction_prefix_fit_window_v1 as dispatch
    from gx1.scripts import run_market_state_time_query_fold1_fit_window_v1 as fit
    calls = []
    limits = {'synthetic_guard': True}
    parts = ({'window': 1}, {'plan': 2}, {'invocation': 3}, {'marker': 4},
             {'run_scope': scope.TA_M5_FIT_SCOPE})
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


def test_m5_materializer_accepts_only_exact_preparation_binding(monkeypatch):
    from gx1.scripts import materialize_market_state_time_query_fold1_fit_campaign_v1 as materializer
    path = scope.TA_M5_FIT_ROOT / 'PREPARATION_EXECUTION_BINDING.json'
    reads = []
    monkeypatch.setattr(campaign, 'file_sha256', lambda path: 'a' * 64)
    def read(path, sha):
        reads.append(path)
        return {}
    monkeypatch.setattr(campaign, 'read_bound_json', read)
    with pytest.raises(Exception, match='MATERIALIZER_BINDING_FIELDS'):
        materializer.materialize(path)
    with pytest.raises(Exception, match='MATERIALIZER_BINDING_PATH'):
        materializer.materialize(path.with_name('OTHER_BINDING.json'))
    assert reads == [path]
