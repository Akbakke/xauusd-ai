"""Source and immutable metadata checks; no native model or target arrays."""
from copy import deepcopy
import json

import pytest

from gx1.contracts import entry_dynamic_m5_october_native_campaign_v1 as october
from gx1.contracts import entry_native_direction_research_campaign_v1 as authority


def test_private_arms_bind_same_fit_targets_and_separate_age_view():
    control = october.build_private_recipe(arm='control')
    candidate = october.build_private_recipe(arm='candidate')
    for recipe in (control, candidate):
        assert october.require_private_recipe(recipe, source_repo=october.REPO, source_commit='a' * 40) == recipe
        assert recipe['input_role'] == 'fit'
        assert recipe['fold']['prefix_rows'] == 29849
        assert recipe['qualification_parent_rows'] == sorted(recipe['qualification_parent_rows'])
        assert authority.guarded_research_power_limit(recipe) == 250
    for key in ('target_manifest', 'target_sources', 'qualification_parent_rows', 'initial_parent_rows',
                'endpoint_support_per_batch', 'prefix_normalization', 'prefix_prior', 'causal_inputs'):
        assert control[key] == candidate[key]
    assert control['feature_view'] != candidate['feature_view']
    assert control['numerical_recipe'].get('confirmed_run_age_input', False) is False
    assert candidate['numerical_recipe']['confirmed_run_age_input'] is True
    for key, value in (('input_role', 'score'), ('target_manifest', {'path': '/synthetic/later', 'sha256': '1' * 64}),
                       ('qualification_parent_rows', list(range(32)))):
        changed = deepcopy(candidate)
        changed[key] = value
        changed['recipe_sha256'] = october.campaign.canonical_sha256(
            {name: item for name, item in changed.items() if name != 'recipe_sha256'})
        with pytest.raises(RuntimeError, match='EXACT_PRIVATE_RECIPE'):
            october.require_private_recipe(changed, source_repo=october.REPO, source_commit='a' * 40)


def test_private_phase_gate_denies_development_and_wrong_arm():
    recipe = october.build_private_recipe(arm='candidate')
    gate = {'candidate_plan': october.roles.BATCH_PLAN, 'arm': 'candidate', 'input_role': 'fit',
            'held_out_features_authorized': False, 'held_out_outcomes_authorized': False}
    october.require_phase_gate(gate, recipe)
    for patch in ({'arm': 'control'}, {'input_role': 'score'},
                  {'held_out_features_authorized': True}, {'held_out_outcomes_authorized': True}):
        with pytest.raises(RuntimeError, match='PRIVATE_ROLE_GATE'):
            october.require_phase_gate({**gate, **patch}, recipe)


def test_launch_recipe_requires_exact_private_path(tmp_path, monkeypatch):
    runtime = tmp_path / 'RUNTIME'
    monkeypatch.setitem(october.PRIVATE_ROOTS, 'candidate', runtime)
    recipe = october.build_private_recipe(arm='candidate')
    correct = runtime.parent / 'NATIVE_RECIPE.json'
    correct.write_text(json.dumps(recipe))
    ref = {'path': str(correct), 'sha256': october.campaign.file_sha256(correct)}
    assert authority.require_research_launch_recipe(ref, source_repo=october.REPO,
                                                    source_commit='a' * 40) == recipe
    wrong = runtime.parent / 'RECIPE.json'
    wrong.write_text(json.dumps(recipe))
    with pytest.raises(authority.campaign.RandomAccessCampaignError, match='OCTOBER_PRIVATE_RECIPE_PATH'):
        authority.require_research_launch_recipe(
            {'path': str(wrong), 'sha256': october.campaign.file_sha256(wrong)},
            source_repo=october.REPO, source_commit='a' * 40)
