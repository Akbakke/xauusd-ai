"""One additional native pass: immutable metadata and synthetic routing only."""
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
    # Only small immutable JSON and source bytes; never a tensor/checkpoint payload.
    refs = scope.TA_M5_NEXT_NATIVE_PASS_IMPORT_REFS
    docs = {}
    for name, ref in refs.items():
        if name == 'next_imported_checkpoint':
            continue
        raw = Path(ref['path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == ref['sha256']
        docs[ref['path']] = json.loads(raw)
    imported = docs[refs['next_import_recipe']['path']]
    expected = {k: v for k, v in imported.items() if k not in ('source_owners', 'recipe_sha256')}
    ref = scope.TA_M5_EQUAL_LOSS_FIT_MECHANICAL_REFS['ta_mechanical_recipe']
    raw = Path(ref['path']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == ref['sha256']
    base = json.loads(raw)
    base_refs = {**base['input_refs'], **scope.TA_M5_EQUAL_LOSS_FIT_MECHANICAL_REFS}
    current = {name: campaign.file_sha256(scope.CANONICAL_REPO / name)
               for name in scope.required_source_paths(ta_sf=True)}
    historical = {name: mechanical._ta_m5_git_source_sha(
        scope.TA_M5_NEXT_NATIVE_PASS_IMPORT_SOURCE_COMMIT, name)
        for name in scope.TA_M5_NEXT_NATIVE_PASS_CHANGED_SOURCE_PATHS}
    return expected, base_refs, base, docs, current, historical


@pytest.fixture
def bound(monkeypatch, metadata):
    expected, base_refs, base, docs, current, historical = copy.deepcopy(metadata)
    original = campaign.require_binding
    monkeypatch.setattr(campaign, 'require_binding', lambda ref, **kw:
        original(ref, label=kw.get('label', 'test'), verify_file=False))
    monkeypatch.setattr(scope, '_ta_m5_equal_loss_fit_qualification',
                        lambda: (copy.deepcopy(base_refs), copy.deepcopy(base)))
    monkeypatch.setattr(mechanical, '_ta_m5_import_expected',
                        lambda **kw: copy.deepcopy(expected))
    monkeypatch.setattr(mechanical, '_read', lambda ref, label: copy.deepcopy(docs[ref['path']]))
    monkeypatch.setattr(mechanical, '_ta_m5_git_source_sha', lambda commit, name:
                        historical[name] if commit == scope.TA_M5_NEXT_NATIVE_PASS_IMPORT_SOURCE_COMMIT else None)
    monkeypatch.setattr(campaign, 'file_sha256',
                        lambda path: current[str(Path(path).relative_to(scope.CANONICAL_REPO))])
    value = scope.build_ta_m5_next_native_pass_launch_recipe(
        source_commit=scope.TA_M5_NEXT_NATIVE_PASS_IMPORT_SOURCE_COMMIT)
    return value, docs


def _require(value, **kwargs):
    return scope.require_launch_recipe(
        value, source_repo=scope.CANONICAL_REPO,
        source_commit=scope.TA_M5_NEXT_NATIVE_PASS_IMPORT_SOURCE_COMMIT, **kwargs)


def test_next_pass_binds_corrected_final_import_and_same_numerics(bound):
    value, _ = bound
    assert _require(value) == value
    assert value['run_scope'] == scope.TA_M5_NEXT_NATIVE_PASS_SCOPE
    assert value['run_id'] == value['admission_stage'] == scope.TA_M5_NEXT_NATIVE_PASS_STAGE
    assert value['runtime_root'] == str(scope.TA_M5_NEXT_NATIVE_PASS_RUNTIME_ROOT)
    assert value['train_passes'] == value['max_windows'] == 1
    assert value['numerical_recipe'] == scope.ta_sf_numerical_recipe(
        dedicated_m5_side=True, m5_equal_observation=True)
    assert value['numerical_recipe']['max_updates'] == 1849
    assert value['imported_origin']['imported_checkpoint'] == value['input_refs']['next_imported_checkpoint']
    assert value['imported_origin']['imported_checkpoint'] != value['input_refs']['imported_checkpoint']
    assert value['expected_initial_model_state_sha256'] == scope.TA_M5_NEXT_NATIVE_PASS_MODEL_SHA256
    assert value['measurement_scope'] == {
        'initial_fit_rows': 29574, 'final_fit_rows': 29574,
        'later_train_rows': 17753, 'matched_initial_final': True}
    assert value['capture_m5_readout'] is True
    assert value['m5_readout_capture_measurements'] == ['FINAL_TRAIN', 'OOF']
    assert value['max_m5_readout_array_bytes'] == 41943040
    assert value['import_campaign_outcome'] == 'CONTROLLER_FAILED_CONSUMED'
    assert value['qualification_weights_for_main_fit'] is False


@pytest.mark.parametrize(('key', 'replacement'), [
    ('train_passes', 2), ('max_windows', 4),
    ('capture_m5_readout', False), ('capture_m5_readout', 1),
    ('m5_readout_capture_measurements', ['INITIAL_TRAIN', 'FINAL_TRAIN', 'OOF']),
    ('max_m5_readout_array_bytes', 67108864),
    ('import_campaign_outcome', 'COMPLETE'),
    ('expected_initial_model_state_sha256', '0' * 64),
])
def test_resealed_next_pass_cannot_expand_scope_or_change_capture(bound, key, replacement):
    value, _ = bound
    with pytest.raises(Exception, match='M5_NEXT_NATIVE_PASS_RECIPE_SCOPE'):
        _require(_seal({**value, key: replacement}))


def test_old_import_and_old_loss_are_rejected(bound):
    value, _ = bound
    bad = copy.deepcopy(value)
    bad['imported_origin']['imported_checkpoint'] = bad['input_refs']['imported_checkpoint']
    with pytest.raises(Exception, match='M5_NEXT_NATIVE_PASS_RECIPE_SCOPE'):
        _require(_seal(bad))
    bad = {**value, 'numerical_recipe': scope.ta_sf_numerical_recipe(dedicated_m5_side=True)}
    with pytest.raises(Exception, match='M5_NEXT_NATIVE_PASS_RECIPE_SCOPE'):
        _require(_seal(bad))
    with pytest.raises(Exception, match='M5_NEXT_NATIVE_PASS_PHYSICAL_METADATA_REQUIRED'):
        _require(value, verify_files=False)


@pytest.mark.parametrize(('fault', 'reason'), [
    ('old_model', 'IMPORT_RESULT'), ('wrong_coefficients', 'IMPORT_RESULT'),
    ('nonzero_progress', 'IMPORT_RESULT'), ('resume', 'IMPORT_RESULT'),
    ('terminal', 'IMPORT_TERMINAL'), ('optimizer', 'IMPORT_REVIEW'),
    ('controller_pass', 'IMPORT_REVIEW'), ('coefficient_tensor', 'IMPORT_REVIEW'),
    ('cache_rows', 'IMPORT_CACHE_ROWS'),
])
def test_actual_final_import_and_controller_failure_are_required(bound, fault, reason):
    value, docs = bound
    refs = value['input_refs']
    report = docs[refs['next_import_result']['path']]
    review = docs[refs['next_import_review']['path']]
    if fault == 'old_model':
        report['head_import']['model_state_sha256'] = mechanical.TA_M5_EQUAL_LOSS_MODEL_SHA256
    elif fault == 'wrong_coefficients':
        report['head_import']['coefficient_artifact'] = refs['fit_cache']
    elif fault == 'nonzero_progress':
        report['parent_cursor'] = 16
    elif fault == 'resume':
        report['bit_exact_resume'] = False
    elif fault == 'terminal':
        docs[refs['next_import_terminal']['path']]['status'] = 'FAILED'
    elif fault == 'optimizer':
        review['fresh_empty_optimizer_verified'] = False
    elif fault == 'controller_pass':
        review['original_campaign_receipt_present'] = True
    elif fault == 'coefficient_tensor':
        report['head_import']['tensor_changes']['future_m5_side_head.weight']['coefficient_sha256'] = '0' * 64
    else:
        report['cached_head_baseline']['fit']['future']['rows'] = 32
        review['scores'] = copy.deepcopy(report['cached_head_baseline'])
    with pytest.raises(Exception, match='M5_NEXT_NATIVE_PASS_' + reason):
        scope._ta_m5_next_native_pass_qualification()


@pytest.mark.parametrize('name', [
    'gx1/contracts/market_state_time_query_fold1_fit_state_v1.py',
    'gx1/contracts/market_state_time_query_native_F_objective_v1.py',
    'gx1/contracts/market_state_time_query_native_F_model_v1.py',
    '.venv/lib/python3.10/site-packages/torch/optim/adamw.py',
])
def test_only_six_explicit_fit_capture_owners_may_change(bound, name):
    value, _ = bound
    value['source_owners'][name]['sha256'] = '0' * 64
    with pytest.raises(Exception, match='M5_NEXT_NATIVE_PASS_UNQUALIFIED_SOURCE'):
        _require(_seal(value))


def test_capture_owner_requires_exact_import_git_source(bound):
    value, docs = bound
    qualified = copy.deepcopy(docs[value['input_refs']['next_import_recipe']['path']]['source_owners'])
    qualified['gx1/contracts/market_state_time_query_native_F_batch_v1.py']['sha256'] = '0' * 64
    with pytest.raises(Exception, match='M5_NEXT_NATIVE_PASS_HISTORICAL_SOURCE'):
        scope._require_ta_m5_next_native_pass_sources(value['source_owners'], qualified,
            source_commit=scope.TA_M5_NEXT_NATIVE_PASS_IMPORT_SOURCE_COMMIT)


def test_new_positive_gate_and_view_require_exact_origin_and_capture(monkeypatch, bound):
    value, _ = bound
    refs = value['input_refs']
    gate = dict(decision='ADMITTED', status='ADMITTED',
        time_query_fold1_fit_authorized=True, ta_sf_fold1_fit_authorized=True,
        ta_m5_next_native_pass_authorized=True, imported_origin=value['imported_origin'],
        import_review=refs['next_import_review'], physical_power_limit_w=250, capture_m5_readout=True,
        input_audit=refs['ta_input_audit'], mechanical_qualification=refs['ta_mechanical_qualification'],
        fold_id=1, fit_rows=29574, score_rows=17753, max_optimizer_updates=1849,
        genuine_predictor_fit_authorized=True, test_data_used=False)
    scope.require_fit_gate(gate, value)
    for key, replacement in (
        ('ta_m5_next_native_pass_authorized', False), ('imported_origin', {}),
        ('capture_m5_readout', False), ('import_review', {}), ('physical_power_limit_w', 160),
    ):
        with pytest.raises(Exception, match='M5_NEXT_NATIVE_PASS_POLICY'):
            scope.require_fit_gate({**gate, key: replacement}, value)
    calls = []
    monkeypatch.setattr(scope, '_require_ta_checkpoint_view',
                        lambda identity, **kw: calls.append(kw) or identity)
    assert scope.require_checkpoint_view({'synthetic': True}, recipe=value) == {'synthetic': True}
    assert calls[-1]['dedicated_m5_side'] is True


def test_new_path_power_cursor_and_progress_route(monkeypatch, bound):
    value, _ = bound
    ref = {'path': str(scope.TA_M5_NEXT_NATIVE_PASS_RECIPE_PATH), 'sha256': 'a' * 64}
    monkeypatch.setattr(campaign, 'read_bound_json', lambda *args: value)
    monkeypatch.setattr(scope, 'require_launch_recipe', lambda candidate, **kw: candidate)
    assert prefix.require_prefix_fit_launch_recipe(ref, source_repo=scope.CANONICAL_REPO,
        source_commit='a' * 40) is value
    with pytest.raises(Exception, match='TIME_QUERY_RECIPE_PATH'):
        prefix.require_prefix_fit_launch_recipe(
            {**ref, 'path': str(scope.TA_M5_EQUAL_LOSS_FIT_RECIPE_PATH)},
            source_repo=scope.CANONICAL_REPO, source_commit='a' * 40)
    assert prefix.guarded_prefix_power_limit(value) == 250
    with pytest.raises(Exception, match='TIME_QUERY_POWER_SCOPE'):
        prefix.guarded_prefix_power_limit({**value, 'admission_stage': scope.TA_M5_EQUAL_LOSS_FIT_STAGE})
    monkeypatch.setattr(scope, 'require_checkpoint_view', lambda identity, **kw: identity)
    artifact = {'path': '/tmp/next-native-pass-metadata.json', 'sha256': 'b' * 64}
    cursor = prefix.build_prefix_fit_cursor(
        recipe_ref=ref, fold=scope.FOLD, feature_view_identity={'prefix_normalization': {}},
        phase_state='COMPLETE', outcome='COMPLETE', optimizer_updates=1849,
        parent_cursor=29574, parent_order_sha256='c' * 64, full_state=artifact,
        own_initial_state=artifact, initial_measurement=artifact, final_train_measurement=artifact,
        oof_cursor=17753, result=artifact)
    assert prefix.require_prefix_fit_cursor(cursor, expected_recipe=ref, verify_files=True) == cursor
    invocation = {'direction_prefix_fit_recipe': ref, 'optimizer_step_budget': 1849,
        'kind': prefix.KIND, 'invocation_sha256': 'd' * 64,
        'checkpoint': {'pointer_path': artifact['path']}}
    progress = prefix.build_prefix_fit_progress(window={'recipe': ref},
        plan={'plan_sha256': 'e' * 64}, invocation=invocation,
        checkpoint_pointer=artifact, cursor=cursor)
    assert (progress['total_units'], progress['oof_cursor']) == (29574, 17753)


def test_new_dispatch_preserves_guard_and_checked_context(monkeypatch, tmp_path):
    from gx1.contracts import gx1_capped_execution_v1 as capped
    from gx1.scripts import run_entry_native_direction_prefix_fit_window_v1 as dispatch
    from gx1.scripts import run_market_state_time_query_fold1_fit_window_v1 as fit
    calls = []
    limits = {'synthetic_guard': True}
    parts = ({'window': 1}, {'plan': 2}, {'invocation': 3}, {'marker': 4},
             {'run_scope': scope.TA_M5_NEXT_NATIVE_PASS_SCOPE})
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


def test_materializer_recognizes_only_exact_next_pass_binding(monkeypatch):
    from gx1.scripts import materialize_market_state_time_query_fold1_fit_campaign_v1 as materializer
    path = scope.TA_M5_NEXT_NATIVE_PASS_ROOT / 'PREPARATION_EXECUTION_BINDING.json'
    reads = []
    monkeypatch.setattr(campaign, 'file_sha256', lambda path: 'a' * 64)
    monkeypatch.setattr(campaign, 'read_bound_json', lambda path, sha: reads.append(path) or {})
    with pytest.raises(Exception, match='MATERIALIZER_BINDING_FIELDS'):
        materializer.materialize(path)
    with pytest.raises(Exception, match='MATERIALIZER_BINDING_PATH'):
        materializer.materialize(path.with_name('OTHER_BINDING.json'))
    assert reads == [path]
