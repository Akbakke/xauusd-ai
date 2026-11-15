"""Fold 2 selected-native route metadata; no model or optimizer work."""
import copy
import subprocess

import pytest

from gx1.contracts import entry_native_direction_research_campaign_v1 as outer
from gx1.contracts import local_random_access_campaign_v2 as campaign
from gx1.contracts import market_state_fold2_selected_native_private_v1 as fold2


def _head():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()


def test_exact_fold2_full_gc_recipe_reuses_mature_source_and_targets():
    recipe = fold2.build_recipe(source_commit=_head())
    checked = fold2.require_recipe(recipe, source_repo=fold2.old.CANONICAL_REPO,
                                   source_commit=_head())
    assert checked['fold_id'] == 2
    assert checked['parent_rows'] == list(range(2, 34))
    assert checked['selected_state_endpoints'] is True
    assert checked['expected_view_sha256'] == fold2.old.FOLD2_FULL_GC_VIEW_SHA256
    assert checked['max_private_updates'] == 4
    assert checked['full_epoch_training_allowed'] is False
    assert {'frozen_priors', 'fold2_packet', 'fold2_sidecar_result'} <= set(checked['input_refs'])
    assert outer.guarded_research_power_limit(checked) == 250


def test_recipe_rejects_nonselected_state_with_valid_seal():
    recipe = fold2.build_recipe(source_commit=_head())
    changed = copy.deepcopy(recipe)
    changed['selected_state_endpoints'] = False
    changed['recipe_sha256'] = campaign.canonical_sha256({
        k: v for k, v in changed.items() if k != 'recipe_sha256'})
    with pytest.raises(Exception, match='RECIPE_BINDING'):
        fold2.require_recipe(changed, source_repo=fold2.old.CANONICAL_REPO,
                             source_commit=_head())


def test_policy_gate_and_cursor_require_exact_four_step_completion():
    recipe = fold2.build_recipe(source_commit=_head())
    gate = {'decision': 'ADMITTED', 'status': 'ADMITTED',
            'fold2_selected_native_private_authorized': True,
            'fold_id': 2, 'sample_parent_rows': list(range(2, 34)),
            'model_forwards_authorized': 6, 'backward_calls_authorized': 4,
            'optimizer_steps_authorized': 4,
            'genuine_predictor_fit_authorized': False, 'test_data_used': False}
    fold2.require_gate(gate, recipe)
    gate['optimizer_steps_authorized'] = 5
    with pytest.raises(Exception, match='POLICY_GATE'):
        fold2.require_gate(gate, recipe)
    recipe_ref = {'path': str(fold2.RECIPE_PATH), 'sha256': '1' * 64}
    cursor = fold2.build_cursor(recipe_ref=recipe_ref, outcome='COMPLETE',
        optimizer_updates=4, parent_cursor=32, parent_order_sha256='2' * 64,
        full_state={'path': '/tmp/fold2-state.pt', 'sha256': '3' * 64},
        result={'path': '/tmp/fold2-result.json', 'sha256': '4' * 64})
    assert outer.require_research_cursor(cursor, expected_recipe=recipe_ref,
                                         verify_files=False, batch_size=16) == cursor
    bad = copy.deepcopy(cursor)
    bad['optimizer_updates'] = 3
    bad['cursor_sha256'] = campaign.canonical_sha256({
        k: v for k, v in bad.items() if k != 'cursor_sha256'})
    with pytest.raises(Exception, match='CURSOR_COMPLETE'):
        fold2.require_cursor(bad, expected_recipe=recipe_ref, verify_files=False)


def test_current_policy_keeps_optimizer_closed():
    import json
    policy = json.loads((fold2.old.CANONICAL_REPO / 'NEXT_RUN_POLICY.json').read_text())
    assert policy['training_enabled'] is False
    assert policy['native_training_enabled'] is False
    assert fold2.STAGE not in policy


def test_outer_recipe_dispatch_requires_exact_one_use_path(monkeypatch):
    recipe = fold2.build_recipe(source_commit=_head())
    monkeypatch.setattr(fold2, 'build_recipe', lambda *, source_commit: copy.deepcopy(recipe))
    monkeypatch.setattr(campaign, 'require_binding',
                        lambda ref, **kwargs: dict(ref))
    monkeypatch.setattr(campaign, 'read_bound_json',
                        lambda path, sha: copy.deepcopy(recipe))
    correct = {'path': str(fold2.RECIPE_PATH), 'sha256': '1' * 64}
    checked = outer.require_research_launch_recipe(
        correct, source_repo=fold2.old.CANONICAL_REPO, source_commit=_head())
    assert checked['run_scope'] == fold2.SCOPE
    wrong = {'path': '/tmp/other-fold2-recipe.json', 'sha256': '1' * 64}
    with pytest.raises(Exception, match='FOLD2_SELECTED_RECIPE_PATH'):
        outer.require_research_launch_recipe(
            wrong, source_repo=fold2.old.CANONICAL_REPO, source_commit=_head())
