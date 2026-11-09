"""Exact equal-observation full-fit admission; metadata only, no model or fit."""
import copy
from pathlib import Path

import pytest

from gx1.contracts import local_random_access_campaign_v2 as campaign
from gx1.contracts import market_state_native_F_private_campaign_v1 as mechanical
from gx1.contracts import market_state_time_query_fold1_fit_v1 as scope


def _seal(value):
    value.pop('recipe_sha256', None)
    value['recipe_sha256'] = campaign.canonical_sha256(value)
    return value


@pytest.fixture(scope='module')
def metadata():
    # Reuse immutable small qualification documents; never load a checkpoint.
    expected, _, _ = mechanical._ta_m5_equal_loss_expected()
    refs = scope.TA_M5_EQUAL_LOSS_FIT_MECHANICAL_REFS
    docs = {ref['path']: mechanical._read(ref, name) for name, ref in refs.items()}
    current = {name: campaign.file_sha256(scope.CANONICAL_REPO / name)
               for name in scope.required_source_paths(ta_sf=True)}
    historical = {name: mechanical._ta_m5_git_source_sha(
        scope.TA_M5_EQUAL_LOSS_FIT_MECHANICAL_COMMIT, name)
        for name in scope.TA_M5_EQUAL_LOSS_FIT_CHANGED_SOURCE_PATHS}
    return expected, docs, current, historical


@pytest.fixture
def bound(monkeypatch, metadata):
    expected, docs, current, historical = copy.deepcopy(metadata)
    original_require = campaign.require_binding
    monkeypatch.setattr(campaign, 'require_binding', lambda ref, **kw:
        original_require(ref, label=kw.get('label', 'test'), verify_file=False))
    monkeypatch.setattr(mechanical, '_ta_m5_equal_loss_expected',
        lambda: (copy.deepcopy(expected), None, None))
    monkeypatch.setattr(mechanical, '_read',
        lambda ref, label: copy.deepcopy(docs[ref['path']]))
    monkeypatch.setattr(mechanical, '_ta_m5_git_source_sha',
        lambda commit, name: historical[name] if commit == scope.TA_M5_EQUAL_LOSS_FIT_MECHANICAL_COMMIT else None)
    monkeypatch.setattr(campaign, 'file_sha256',
        lambda path: current[str(Path(path).relative_to(scope.CANONICAL_REPO))])
    value = scope.build_ta_m5_equal_loss_fit_launch_recipe(
        source_commit=scope.TA_M5_EQUAL_LOSS_FIT_MECHANICAL_COMMIT)
    return value, docs


def _require(value, **kwargs):
    return scope.require_launch_recipe(value, source_repo=scope.CANONICAL_REPO,
        source_commit=scope.TA_M5_EQUAL_LOSS_FIT_MECHANICAL_COMMIT, **kwargs)


def test_exact_original_import_one_pass_and_full_matched_measurement(bound):
    value, docs = bound
    assert _require(value) == value
    assert value['run_scope'] == scope.TA_M5_EQUAL_LOSS_FIT_SCOPE
    assert value['run_id'] == value['admission_stage'] == scope.TA_M5_EQUAL_LOSS_FIT_STAGE
    assert value['runtime_root'] == str(scope.TA_M5_EQUAL_LOSS_FIT_RUNTIME_ROOT)
    assert value['fold'] == scope.FOLD
    assert value['train_passes'] == value['max_windows'] == 1
    assert scope.MAX_WINDOWS == 4
    assert value['measurement_scope'] == {
        'initial_fit_rows': 29574, 'final_fit_rows': 29574,
        'later_train_rows': 17753, 'matched_initial_final': True}
    numerical = scope.ta_sf_numerical_recipe(dedicated_m5_side=True, m5_equal_observation=True)
    assert value['numerical_recipe'] == numerical
    assert numerical['max_updates'] == 1849 and numerical['batch_size'] == 16
    assert numerical['precision_policy'] == 'deterministic_fp32'
    assert numerical['optimizer'] == scope.ta_sf_numerical_recipe(dedicated_m5_side=True)['optimizer']
    refs = value['input_refs']
    qualified = docs[refs['ta_mechanical_recipe']['path']]
    report = docs[refs['ta_mechanical_qualification']['path']]
    assert value['imported_origin'] == qualified['imported_origin']
    assert value['imported_origin']['imported_checkpoint'] == refs['imported_checkpoint']
    assert value['imported_origin']['imported_checkpoint'] != report['final_state']
    assert value['expected_initial_model_state_sha256'] == mechanical.TA_M5_EQUAL_LOSS_MODEL_SHA256
    assert value['reference_result'] == refs['ta_endpoint_priors']
    assert value['qualification_weights_for_main_fit'] is False
    assert value['expected_model_contract_sha256'] == qualified['expected_model_contract_sha256']
    assert {name for name, ref in value['source_owners'].items()
            if ref != qualified['source_owners'][name]} <= scope.TA_M5_EQUAL_LOSS_FIT_CHANGED_SOURCE_PATHS


@pytest.mark.parametrize('key, replacement', [
    ('train_passes', 2), ('max_windows', 4), ('checkpoint_every_updates', 512),
    ('expected_initial_model_state_sha256', '0' * 64),
    ('mechanical_source_commit', '0' * 40), ('qualification_weights_for_main_fit', True),
    ('measurement_scope', {'initial_fit_rows': 32, 'final_fit_rows': 29574,
                           'later_train_rows': 17753, 'matched_initial_final': False}),
])
def test_resealed_recipe_cannot_change_scope_or_evidence(bound, key, replacement):
    value, _ = bound
    value[key] = replacement
    with pytest.raises(Exception, match='M5_EQUAL_LOSS_FIT_RECIPE_SCOPE'):
        _require(_seal(value))


def test_private_updated_checkpoint_and_default_loss_are_rejected(bound):
    value, docs = bound
    bad = copy.deepcopy(value)
    bad['imported_origin']['imported_checkpoint'] = docs[
        value['input_refs']['ta_mechanical_qualification']['path']]['final_state']
    with pytest.raises(Exception, match='M5_EQUAL_LOSS_FIT_RECIPE_SCOPE'):
        _require(_seal(bad))
    bad = copy.deepcopy(value)
    bad['numerical_recipe'] = scope.ta_sf_numerical_recipe(dedicated_m5_side=True)
    with pytest.raises(Exception, match='M5_EQUAL_LOSS_FIT_RECIPE_SCOPE'):
        _require(_seal(bad))


@pytest.mark.parametrize('fault, reason', [
    ('native_result', 'M5_EQUAL_LOSS_FIT_MECHANICAL_RESULT'),
    ('terminal', 'M5_EQUAL_LOSS_FIT_TERMINAL'),
    ('receipt', 'M5_EQUAL_LOSS_FIT_TERMINAL'),
    ('review', 'M5_EQUAL_LOSS_FIT_REVIEW'),
    ('artifacts', 'M5_EQUAL_LOSS_FIT_REVIEW'),
    ('rebind_parity', 'M5_EQUAL_LOSS_FIT_REBINDING_PARITY'),
    ('rebind_progress', 'M5_EQUAL_LOSS_FIT_REBINDING_STATE'),
    ('qualified_recipe', 'M5_EQUAL_LOSS_FIT_MECHANICAL_RECIPE'),
])
def test_full_native_qualification_chain_is_required(bound, fault, reason):
    value, docs = bound
    refs = value['input_refs']
    report = docs[refs['ta_mechanical_qualification']['path']]
    if fault == 'native_result':
        report['bit_exact_resume'] = False
    elif fault == 'terminal':
        docs[refs['ta_mechanical_terminal']['path']]['status'] = 'FAILED'
    elif fault == 'receipt':
        docs[refs['ta_mechanical_campaign_receipt']['path']]['trainer_guard_exit_code'] = 1
    elif fault == 'review':
        docs[refs['ta_mechanical_review']['path']]['model_buffers_optimizer_rng_modes_order_preserved_at_rebinding'] = False
    elif fault == 'artifacts':
        docs[refs['ta_mechanical_artifact_verification']['path']]['original_import_for_main_fit'] = False
    elif fault == 'rebind_parity':
        report['loss_rebindings'][1]['rng_state_sha256'] = '0' * 64
    elif fault == 'rebind_progress':
        for proof in report['loss_rebindings']:
            proof['cursor'] = 16
    else:
        qualified = docs[refs['ta_mechanical_recipe']['path']]
        qualified['max_private_updates'] = 5
        _seal(qualified)
    with pytest.raises(Exception, match=reason):
        scope._ta_m5_equal_loss_fit_qualification()


@pytest.mark.parametrize('name', [
    'gx1/contracts/market_state_time_query_fold1_fit_state_v1.py',
    'gx1/contracts/market_state_time_query_native_F_model_v1.py',
    'gx1/contracts/market_state_time_query_native_F_objective_v1.py',
    'gx1/contracts/market_state_time_query_side_normalized_loss_v1.py',
    'gx1/contracts/market_state_native_F_private_campaign_v1.py',
    '.venv/lib/python3.10/site-packages/torch/optim/adamw.py',
])
def test_only_five_fullfit_orchestration_sources_can_change(bound, name):
    value, _ = bound
    value['source_owners'][name]['sha256'] = '0' * 64
    with pytest.raises(Exception, match='M5_EQUAL_LOSS_FIT_UNQUALIFIED_SOURCE'):
        _require(_seal(value))


def test_changed_owner_still_requires_qualified_historical_git_bytes(bound):
    value, docs = bound
    qualified = copy.deepcopy(docs[value['input_refs']['ta_mechanical_recipe']['path']]['source_owners'])
    qualified['gx1/contracts/market_state_time_query_fold1_fit_v1.py']['sha256'] = '0' * 64
    with pytest.raises(Exception, match='M5_EQUAL_LOSS_FIT_HISTORICAL_SOURCE'):
        scope._require_ta_m5_equal_loss_fit_sources(value['source_owners'], qualified,
            source_commit=scope.TA_M5_EQUAL_LOSS_FIT_MECHANICAL_COMMIT)


def test_physical_metadata_validation_cannot_be_disabled(bound):
    with pytest.raises(Exception, match='M5_EQUAL_LOSS_FIT_PHYSICAL_METADATA_REQUIRED'):
        _require(bound[0], verify_files=False)


def test_fullfit_needs_distinct_positive_gate_and_same_import_origin(bound):
    value, _ = bound
    refs = value['input_refs']
    gate = dict(decision='ADMITTED', status='ADMITTED',
        time_query_fold1_fit_authorized=True, ta_sf_fold1_fit_authorized=True,
        ta_m5_equal_loss_fold1_fit_authorized=True, imported_origin=value['imported_origin'],
        mechanical_review=refs['ta_mechanical_review'], physical_power_limit_w=250,
        input_audit=refs['ta_input_audit'], mechanical_qualification=refs['ta_mechanical_qualification'],
        fold_id=1, fit_rows=29574, score_rows=17753, max_optimizer_updates=1849,
        genuine_predictor_fit_authorized=True, test_data_used=False)
    scope.require_fit_gate(gate, value)
    for key, replacement in (
        ('ta_m5_equal_loss_fold1_fit_authorized', False),
        ('imported_origin', {}), ('physical_power_limit_w', 160), ('mechanical_review', {}),
    ):
        with pytest.raises(Exception, match='M5_EQUAL_LOSS_FIT_POLICY'):
            scope.require_fit_gate({**gate, key: replacement}, value)
    with pytest.raises(Exception, match='POLICY_GATE'):
        scope.require_fit_gate({**gate, 'genuine_predictor_fit_authorized': False}, value)


def test_checkpoint_view_reuses_model_identity_and_requires_exact_opt_in_recipe(monkeypatch, bound):
    value, _ = bound
    calls = []
    monkeypatch.setattr(scope, '_require_ta_checkpoint_view',
        lambda identity, **kwargs: calls.append(kwargs) or identity)
    identity = {'synthetic_view': True}
    assert scope.require_checkpoint_view(identity, recipe=value, ta_sf=True) is identity
    assert calls[-1]['dedicated_m5_side'] is True
    with pytest.raises(Exception, match='TA_M5_VIEW_NUMERICAL'):
        scope.require_checkpoint_view(identity, recipe={
            **value, 'numerical_recipe': scope.ta_sf_numerical_recipe(dedicated_m5_side=True)},
            ta_sf=True)
    scope.require_checkpoint_view(identity, recipe={
        'run_scope': scope.TA_M5_FIT_SCOPE,
        'numerical_recipe': scope.ta_sf_numerical_recipe(dedicated_m5_side=True)}, ta_sf=True)
    assert calls[-1]['dedicated_m5_side'] is True
