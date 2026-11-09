"""Exact imported-state admission and cursor routing; no model or artifact load."""
import copy

import pytest

from gx1.contracts import entry_native_direction_research_campaign_v1 as research
from gx1.contracts import market_state_native_F_private_campaign_v1 as scope


def _proof():
    refs = copy.deepcopy(scope.TA_M5_EQUAL_LOSS_FIXED_REFS)
    imported = {
        'schema_version': scope.TA_M5_IMPORT_RESULT_SCHEMA,
        'decision': 'PASS_TA_M5_EQUAL_WEIGHT_IMPORT',
        'recipe': refs['import_recipe'], 'final_state': refs['imported_checkpoint'],
        'source_commit': scope.TA_M5_EQUAL_LOSS_IMPORT_COMMIT,
        'source_unchanged': True, 'error': None, 'test_data_used': False,
        'bit_exact_resume': True, 'physical_optimizer_updates': 0, 'parent_cursor': 0,
        'parent_order_sha256': scope.TA_M5_EQUAL_LOSS_ORDER_SHA256,
        'head_import': {'model_state_sha256': scope.TA_M5_EQUAL_LOSS_MODEL_SHA256,
                        'rng_state_sha256': scope.TA_M5_EQUAL_LOSS_RNG_SHA256},
    }
    checked = {
        'status': 'PASS_INDEPENDENT_MODEL_ARTIFACTS_CAMPAIGN_CONTROLLER_FAILED',
        'result': refs['import_result'], 'checkpoint': refs['imported_checkpoint'],
        'imported_model_state_sha256': scope.TA_M5_EQUAL_LOSS_MODEL_SHA256,
        'unchanged_buffers_optimizer_rng_modes_parent_order': True,
        'coefficient_tensors_bit_equal': True,
        'original_campaign_receipt_present': False, 'test_data_used': False,
    }
    review = {
        'status': 'MODEL_IMPORT_VERIFIED_CAMPAIGN_CONTROLLER_FAILED_CONSUMED',
        'result': refs['import_result'], 'artifact_verification': refs['import_verification'],
        'native_terminal': refs['import_terminal'],
        'imported_checkpoint': refs['imported_checkpoint'],
        'imported_model_state_sha256': scope.TA_M5_EQUAL_LOSS_MODEL_SHA256,
        'checkpoint_resume_bit_exact': True, 'test_data_used': False,
    }
    terminal = {'status': 'COMPLETE', 'exit_code': 0, 'result': refs['import_result'],
                'recipe': refs['import_recipe'], 'no_relaunch_under_this_plan': True,
                'test_data_used': False}
    return refs, imported, checked, review, terminal


def test_exact_import_proof_preserves_original_controller_failure():
    args = _proof()
    got = scope._ta_m5_equal_loss_import_origin(*args)
    assert got == {
        'imported_checkpoint': args[0]['imported_checkpoint'],
        'source_commit': scope.TA_M5_EQUAL_LOSS_IMPORT_COMMIT,
        'model_state_sha256': scope.TA_M5_EQUAL_LOSS_MODEL_SHA256,
        'rng_state_sha256': scope.TA_M5_EQUAL_LOSS_RNG_SHA256,
        'parent_order_sha256': scope.TA_M5_EQUAL_LOSS_ORDER_SHA256,
    }


@pytest.mark.parametrize(('key', 'value'), [
    ('source_commit', '0' * 40),
    ('parent_order_sha256', '0' * 64),
    ('source_unchanged', False),
    ('physical_optimizer_updates', 1),
    ('physical_optimizer_updates', False),
    ('parent_cursor', 16),
    ('bit_exact_resume', False),
    ('final_state', {'path': '/tmp/other.pt', 'sha256': '0' * 64}),
])
def test_reject_changed_import_origin(key, value):
    args = _proof()
    args[1][key] = value
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='IMPORTED_ORIGIN'):
        scope._ta_m5_equal_loss_import_origin(*args)


@pytest.mark.parametrize('key', ['model_state_sha256', 'rng_state_sha256'])
def test_reject_changed_import_tensor_identity(key):
    args = _proof()
    args[1]['head_import'][key] = '0' * 64
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='IMPORTED_ORIGIN'):
        scope._ta_m5_equal_loss_import_origin(*args)


@pytest.mark.parametrize(('index', 'key', 'value'), [
    (2, 'coefficient_tensors_bit_equal', False),
    (2, 'unchanged_buffers_optimizer_rng_modes_parent_order', False),
    (2, 'original_campaign_receipt_present', True),
    (3, 'status', 'COMPLETE'),
    (4, 'exit_code', 1),
])
def test_reject_missing_independent_import_proof(index, key, value):
    args = _proof()
    args[index][key] = value
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='IMPORTED_PROOF'):
        scope._ta_m5_equal_loss_import_origin(*args)


def _recipe_ref():
    return {'path': str(scope.TA_M5_EQUAL_LOSS_RECIPE_PATH), 'sha256': 'a' * 64}


def _cursor(**modes):
    return scope.build_cursor(
        recipe_ref=_recipe_ref(), outcome='COMPLETE', optimizer_updates=4, parent_cursor=32,
        parent_order_sha256=scope.TA_M5_EQUAL_LOSS_ORDER_SHA256,
        full_state={'path': str(scope.TA_M5_EQUAL_LOSS_ROOT / 'state.pt'), 'sha256': 'b' * 64},
        result={'path': str(scope.TA_M5_EQUAL_LOSS_ROOT / 'RESULT.json'), 'sha256': 'c' * 64},
        **modes)


def test_equal_loss_cursor_routes_with_ordinary_four_update_progress():
    cursor = _cursor(ta_sf=True, dedicated_m5_side=True, m5_equal_observation=True)
    assert cursor['schema_version'] == scope.TA_M5_EQUAL_LOSS_CURSOR_SCHEMA
    assert research.require_research_cursor(
        cursor, expected_recipe=_recipe_ref(), verify_files=False, batch_size=16) == cursor
    assert cursor['optimizer_updates'] == 4 and cursor['parent_cursor'] == 32


def test_equal_loss_cursor_requires_dedicated_ta_mode():
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='CURSOR_MODE'):
        _cursor(ta_sf=True, dedicated_m5_side=False, m5_equal_observation=True)


def test_new_recipe_rejects_old_cursor_schema():
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='CURSOR_RECIPE_PATH'):
        _cursor(ta_sf=True, dedicated_m5_side=True)


def test_new_cursor_rejects_old_recipe_path():
    cursor = _cursor(ta_sf=True, dedicated_m5_side=True, m5_equal_observation=True)
    old_ref = dict(_recipe_ref(), path=str(scope.TA_M5_RECIPE_PATH))
    cursor['recipe'] = old_ref
    cursor['cursor_sha256'] = scope.campaign.canonical_sha256({
        k: v for k, v in cursor.items() if k != 'cursor_sha256'})
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='CURSOR_RECIPE_PATH'):
        scope.require_cursor(cursor, expected_recipe=old_ref, verify_files=False)


def test_launch_recipe_dispatches_exact_new_path_and_scope(monkeypatch):
    binding = _recipe_ref()
    value = {'run_scope': scope.TA_M5_EQUAL_LOSS_SCOPE}
    calls = []
    monkeypatch.setattr(research.campaign, 'require_binding', lambda value, **_: dict(value))
    monkeypatch.setattr(research.campaign, 'read_bound_json', lambda *a: value)
    def require(candidate, **kwargs):
        calls.append((candidate, kwargs))
        return candidate
    monkeypatch.setattr(scope, 'require_ta_m5_equal_loss_mechanical_recipe', require)
    assert research.require_research_launch_recipe(
        binding, source_repo=scope.CANONICAL_REPO, source_commit='a' * 40) is value
    assert calls == [(value, {'source_repo': scope.CANONICAL_REPO,
                             'source_commit': 'a' * 40, 'verify_files': True})]
    with pytest.raises(scope.campaign.RandomAccessCampaignError, match='RECIPE_PATH'):
        research.require_research_launch_recipe(
            dict(binding, path=str(scope.TA_M5_RECIPE_PATH)),
            source_repo=scope.CANONICAL_REPO, source_commit='a' * 40)


def test_equal_loss_scope_retains_existing_250w_four_step_guard():
    recipe = {'run_scope': scope.TA_M5_EQUAL_LOSS_SCOPE, 'fold_id': 1,
              'training_enabled': True, 'test_data_used': False,
              'numerical_recipe': {'batch_size': 16}}
    assert research.guarded_research_power_limit(recipe) == 250
    recipe['training_enabled'] = False
    with pytest.raises(scope.campaign.RandomAccessCampaignError):
        research.guarded_research_power_limit(recipe)
