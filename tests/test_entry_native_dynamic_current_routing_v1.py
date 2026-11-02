"""Source/data routing checks only: no model construction, forward or optimizer."""
from copy import deepcopy

import numpy as np
import pytest

from gx1.contracts import entry_dynamic_mtf_state_targets_v1 as target
from gx1.contracts import entry_model_native_train_recipe_v1 as recipes
from gx1.contracts import entry_native_direction_feature_view_v1 as views
from gx1.scripts import run_entry_native_direction_prefix_fit_window_v1 as prefix
from tests.test_entry_dynamic_mtf_state_targets import START, build, frame_from_closes
from tests.test_entry_native_direction_feature_view_v1 import bound_case


def _arguments(bound):
    args = deepcopy(bound['arguments'])
    args['source_bindings']['dynamic_state_target_source'] = {
        'path': '/synthetic/entry_dynamic_mtf_state_targets_v1.py', 'sha256': 'b' * 64}
    args.update(full_feature_profile=True, dynamic_state_supervision=True)
    return args


def test_new_recipe_view_and_old_view_bind_distinct_exact_objectives(bound_case):
    args = _arguments(bound_case)
    old = views.bind_native_direction_feature_view(**args)
    args['current_state_first_loss_weight'] = 1.0  # illustrative test, not a selected training weight
    new = views.bind_native_direction_feature_view(**args)
    recipe = recipes.native_direction_research_recipe(batch_size=16, max_updates=2,
        dynamic_state=True, current_state_first_loss_weight=1.0)
    assert recipes.require_native_direction_research_recipe(recipe) == recipe
    assert new['research_objective']['contract_sha256'] == recipe['research_objective_sha256']
    assert new['research_objective']['contract_sha256'] != old['research_objective']['contract_sha256']
    assert new['current_state_first_loss_weight'] == 1.0
    assert new['physical_field_names_sha256'] == old['physical_field_names_sha256']
    assert new['entry_execution_profile'] == old['entry_execution_profile']
    assert views.bind_native_direction_feature_view(**{**args, 'current_state_first_loss_weight': None}) == old
    with pytest.raises(RuntimeError):
        recipes.native_direction_research_recipe(batch_size=16, max_updates=2,
            dynamic_state=False, current_state_first_loss_weight=1.0)
    with pytest.raises(RuntimeError):
        views.bind_native_direction_feature_view(**{**args, 'dynamic_state_supervision': False})


def _batch_fixture():
    parents = np.array([0], np.int64)
    issue = np.array([START + 5 * target.MINUTE_NS], np.int64)
    series = target.bind_dynamic_mtf_state_parents(build(frame_from_closes([100.0] * 12)),
                                                    parent_rows=parents, issue_ns=issue)
    prepared = {'arrays': {'parent_rows': parents, 'decision_ns': issue, 'bar_start_ns': issue - 5 * target.MINUTE_NS},
        'entry_sequence_source': None, 'current': {'snap': None, 'ctx_cont': None, 'ctx_cat': None},
        'cache': None, 'gc': {'surfaces': None, 'layout': None}}
    targets = {'dynamic_state_series': series, 'train_cutoff_ns': int(START + 100 * target.MINUTE_NS),
               'lead_minutes': np.array([1, 2], np.int64)}
    return prepared, targets, parents


def test_batch_routes_current_only_and_never_future_targets_to_features(monkeypatch):
    prepared, targets, parents = _batch_fixture()
    monkeypatch.setattr(views, 'gather_native_direction_feature_batch',
        lambda *args, **kwargs: {'features': {'seq_x': np.zeros((1, 1, 1), np.float32)}, 'witness': {}})
    raw, outcome = prefix._batch(prepared, targets, {'current_state_first_loss_weight': 1.0}, parents, np)
    assert set(raw['features']) == {'seq_x', 'current_state', 'current_observed', 'current_path_observed'}
    assert np.array_equal(raw['features']['current_state'], outcome['current_state'])
    assert np.array_equal(raw['features']['current_observed'], outcome['current_observed'])
    assert raw['features']['current_state'][0, 0] == target.FLAT
    assert 'state' not in raw['features'] and 'observed' not in raw['features']
    old_raw, _ = prefix._batch(prepared, targets, {}, parents, np)
    assert set(old_raw['features']) == {'seq_x'}


def test_batch_rejects_current_label_mismatch_and_feature_collision(monkeypatch):
    prepared, targets, parents = _batch_fixture()
    monkeypatch.setattr(views, 'gather_native_direction_feature_batch',
        lambda *args, **kwargs: {'features': {'seq_x': np.zeros((1, 1, 1), np.float32)}, 'witness': {}})
    original = target.query_dynamic_mtf_states
    def divergent(*args, **kwargs):
        result = original(*args, **kwargs)
        result['current_state'][0, 0] = target.DOWN
        return result
    monkeypatch.setattr(target, 'query_dynamic_mtf_states', divergent)
    with pytest.raises(RuntimeError, match='CURRENT_INPUT_TARGET_ALIGNMENT'):
        prefix._batch(prepared, targets, {'current_state_first_loss_weight': 1.0}, parents, np)
    monkeypatch.setattr(target, 'query_dynamic_mtf_states', original)
    monkeypatch.setattr(views, 'gather_native_direction_feature_batch',
        lambda *args, **kwargs: {'features': {'current_state': np.zeros((1, 5), np.int8)}, 'witness': {}})
    with pytest.raises(RuntimeError, match='CURRENT_INPUT_FEATURE_COLLISION'):
        prefix._batch(prepared, targets, {'current_state_first_loss_weight': 1.0}, parents, np)


@pytest.mark.parametrize('weight,accepted,activity', [(1.0, True, False), (0.0, False, False), (1, False, False), (True, False, False), (1.0, False, True)])
def test_launch_metadata_requires_explicit_valid_current_weight(tmp_path, monkeypatch, weight, accepted, activity):
    import hashlib
    import json
    from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as prefix_authority
    from gx1.contracts import entry_native_direction_research_campaign_v1 as authority
    from gx1.contracts import local_random_access_campaign_v2 as campaign
    numerical = recipes.native_direction_research_recipe(batch_size=16, max_updates=2,
        dynamic_state=True, current_state_first_loss_weight=1.0)
    numerical['current_state_first_loss_weight'] = weight
    if activity:
        numerical['first_flat_activity_weight'] = 0.25
    unsigned = {k: v for k, v in numerical.items() if k != 'contract_sha256'}
    numerical['contract_sha256'] = hashlib.sha256(json.dumps(unsigned, sort_keys=True,
        separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()).hexdigest()
    repo = tmp_path / 'source'
    repo.mkdir()
    runtime = tmp_path / 'runtime'
    dummy = {'path': str(tmp_path / 'unused.json'), 'sha256': 'a' * 64}
    monkeypatch.setattr(prefix_authority, 'CANONICAL_REPO', repo)
    monkeypatch.setattr(prefix_authority, 'REQUIRED_SOURCE_PATHS', ())
    monkeypatch.setattr(prefix_authority, 'DYNAMIC_STATE_EXTRA_SOURCE_PATHS', ())
    monkeypatch.setattr(prefix_authority, 'require_dynamic_state_input_manifest', lambda *a, **kw: None)
    monkeypatch.setattr(prefix_authority, 'require_dynamic_state_target_binding', lambda *a, **kw: None)
    launch = {'schema_version': authority.RECIPE_SCHEMA, 'source_repo': str(repo), 'runtime_root': str(runtime),
        'admission_stage': 'dynamic_mtf_state_synthetic_private', 'run_scope': authority.DYNAMIC_STATE_PRIVATE_SCOPE,
        'numerical_recipe': numerical, 'input_manifest': dummy, 'source_owners': {},
        'execution_binding_path': str(runtime / 'EXECUTION_BINDING.json'), 'fold': authority.BPS_FOLD,
        'max_checkpoint_bytes': 1073741824, 'max_published_json_bytes': 33554432,
        'training_enabled': True, 'full_epoch_training_allowed': False, 'test_data_used': False,
        'predictive_value_claimed': False, 'qualification_weights_for_main_fit': False,
        'target_manifest': dummy, 'target_sources': {'m1': dummy, 'm5_atr': dummy}, 'label_audit': dummy,
        'full_feature_profile': True, 'max_private_updates': 4, 'max_model_constructions': 3,
        'max_forwards': 21, 'max_backwards': 4, 'qualification_parent_rows': list(range(32))}
    launch['recipe_sha256'] = campaign.canonical_sha256(launch)
    if accepted:
        checked = prefix_authority.require_dynamic_state_launch_recipe(
            launch, source_repo=repo, source_commit='b' * 40, private_scope=True)
        assert checked['numerical_recipe']['current_state_first_loss_weight'] == 1.0
    else:
        expected = 'STATE_FIRST_ACTIVITY_TRAIN_AUDIT_NOT_ADMITTED' if activity else 'STATE_CURRENT_FIRST_LOSS_WEIGHT'
        with pytest.raises(RuntimeError, match=expected):
            prefix_authority.require_dynamic_state_launch_recipe(
                launch, source_repo=repo, source_commit='b' * 40, private_scope=True)
