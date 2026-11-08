"""Metadata-only S/F trainer route checks; no model or optimizer calls."""
import copy
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import pytest

from gx1.contracts import entry_native_direction_research_campaign_v1 as outer
from gx1.contracts import local_random_access_campaign_v2 as campaign
from gx1.contracts import market_state_native_F_private_campaign_v1 as scope
from gx1.contracts import market_state_native_F_private_state_v1 as state


def _recipe():
    old = json.load(open('/home/andre2/GX1_RUNS/MARKET_STATE_TIME_QUERY_PRIVATE_4STEP_20261107_003/RECIPE.json'))
    result = {k: copy.deepcopy(v) for k, v in old.items() if k != 'recipe_sha256'}
    result['schema_version'] = scope.RECIPE_SCHEMA
    result['run_scope'] = scope.SCOPE
    result['admission_stage'] = scope.STAGE
    result['runtime_root'] = str(scope.RUNTIME_ROOT)
    result['execution_binding_path'] = str(scope.RUNTIME_ROOT / 'EXECUTION_BINDING.json')
    result['numerical_recipe'] = scope.numerical_recipe()
    result['input_refs'] = scope.bound_input_refs()
    v2 = scope._read(scope._read(scope.ZERO_STEP_REFS['plan'], 'zero step plan')['inputs']['completed_v2_plan'], 'v2 plan')
    result['input_manifest'] = v2['input_refs']['mature_plan']
    result['expected_view_sha256'] = scope.VIEW_SHA256
    result['expected_model_contract_sha256'] = scope.MODEL_CONTRACT_SHA256
    result['expected_loss_contract_sha256'] = scope.LOSS_CONTRACT_SHA256
    result['max_private_updates'] = scope.MAX_PRIVATE_UPDATES
    result['max_model_constructions'] = scope.MAX_MODEL_CONSTRUCTIONS
    result['max_forwards'] = scope.MAX_FORWARDS
    result['max_backwards'] = scope.MAX_BACKWARDS
    result['max_checkpoint_bytes'] = scope.MAX_CHECKPOINT_BYTES
    result['max_published_json_bytes'] = scope.MAX_PUBLISHED_JSON_BYTES
    paths = set(outer.REQUIRED_SOURCE_PATHS) | set(scope.previous.EXTRA_SOURCE_PATHS) | set(scope.EXTRA_SOURCE_PATHS)
    result['source_owners'] = {name: {'path': str(scope.CANONICAL_REPO / name),
        'sha256': hashlib.sha256((scope.CANONICAL_REPO / name).read_bytes()).hexdigest()}
        for name in sorted(paths)}
    result['recipe_sha256'] = campaign.canonical_sha256(result)
    return result


def test_bridge_recipe_and_optimizer_numeric_binding():
    assert scope._bridge()[1]['status'] == 'COMPLETE'
    assert len(scope.bound_input_refs()) == 21
    assert scope.numerical_recipe() == state.numerical_recipe()
    recipe = _recipe()
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    checked = scope.require_recipe(recipe, source_repo=scope.CANONICAL_REPO,
                                   source_commit=head)
    assert checked['run_scope'] == scope.SCOPE
    assert checked['expected_view_sha256'] == scope.VIEW_SHA256


def test_recipe_rejects_weight_change_with_valid_seal():
    recipe = _recipe()
    recipe['numerical_recipe']['future_weight'] = 0.25
    recipe['recipe_sha256'] = campaign.canonical_sha256(
        {k: v for k, v in recipe.items() if k != 'recipe_sha256'})
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    with pytest.raises(Exception, match='RECIPE_SCOPE'):
        scope.require_recipe(recipe, source_repo=scope.CANONICAL_REPO,
                             source_commit=head)


def test_cursor_requires_exact_complete_counts_and_bound_result():
    ref = {'path': '/tmp/gx1-native-F-recipe.json', 'sha256': '1' * 64}
    result = {'path': '/tmp/gx1-native-F-result.json', 'sha256': '2' * 64}
    full = {'path': '/tmp/gx1-native-F-state.pt', 'sha256': '3' * 64}
    cursor = scope.build_cursor(recipe_ref=ref, outcome='COMPLETE',
        optimizer_updates=4, parent_cursor=32,
        parent_order_sha256='4' * 64, full_state=full, result=result)
    assert scope.require_cursor(cursor, expected_recipe=ref, verify_files=False,
                                batch_size=16)['outcome'] == 'COMPLETE'
    bad = copy.deepcopy(cursor)
    bad['optimizer_updates'] = 3
    bad['cursor_sha256'] = campaign.canonical_sha256(
        {k: v for k, v in bad.items() if k != 'cursor_sha256'})
    with pytest.raises(Exception, match='CURSOR_COMPLETE'):
        scope.require_cursor(bad, expected_recipe=ref, verify_files=False,
                             batch_size=16)


def test_existing_trainer_router_accepts_only_bound_native_F_recipe():
    recipe = _recipe()
    assert outer.guarded_research_power_limit(recipe) == 250
    with tempfile.TemporaryDirectory() as root:
        path = Path(root) / 'RECIPE.json'
        path.write_text(json.dumps(recipe, indent=2, ensure_ascii=False) + '\n')
        ref = {'path': str(path),
               'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        checked = outer.require_research_launch_recipe(
            ref, source_repo=scope.CANONICAL_REPO, source_commit=head)
        assert checked['run_scope'] == scope.SCOPE
