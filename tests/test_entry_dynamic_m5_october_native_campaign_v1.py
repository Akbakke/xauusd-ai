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


def test_fit_recipe_stays_on_private_arm_and_fit_target(monkeypatch):
    from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as prefix
    seen = []
    monkeypatch.setattr(prefix, 'require_dynamic_state_qualification',
                        lambda recipe, **kwargs: seen.append((recipe, kwargs)))
    fit = {}
    for arm in ('control', 'candidate'):
        review = {'path': str(october.PRIVATE_ROOTS[arm].parent / 'ACTUAL_ROOT_REVIEW.json'),
                  'sha256': '1' * 64}
        fit[arm] = october.build_fit_recipe(arm=arm, qualification_review=review,
                                            source_commit='a' * 40)
        assert october.require_fit_recipe(fit[arm], source_repo=october.REPO,
                                          source_commit='a' * 40) == fit[arm]
        assert fit[arm]['numerical_recipe']['max_updates'] == 1866
        assert fit[arm]['train_passes'] == 1
        assert fit[arm]['final_train_measurement_rows'] == 29849
        assert fit[arm]['input_role'] == 'fit'
        assert fit[arm]['target_manifest'] == october.build_private_recipe(arm=arm)['target_manifest']
    assert len(seen) == 4
    assert fit['control']['target_manifest'] == fit['candidate']['target_manifest']
    assert fit['control']['feature_view'] != fit['candidate']['feature_view']
    assert fit['control']['native_dynamic_state_qualification'] != fit['candidate']['native_dynamic_state_qualification']
    with pytest.raises(RuntimeError, match='FIT_PRIVATE_ORIGIN'):
        october.build_fit_recipe(arm='candidate', qualification_review=fit['control']['native_dynamic_state_qualification'],
                                 source_commit='a' * 40)
    tampered = deepcopy(fit['candidate'])
    tampered['arm'] = 'control'
    tampered['recipe_sha256'] = october.campaign.canonical_sha256(
        {k: v for k, v in tampered.items() if k != 'recipe_sha256'})
    with pytest.raises(RuntimeError):
        october.require_fit_recipe(tampered, source_repo=october.REPO, source_commit='a' * 40)


def test_october_fit_cursor_refuses_oof_and_cross_arm():
    from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as prefix
    batch = october._read(october.roles.BATCH_PLAN)
    control_view = october._read(batch['control_view'])
    identity = {'contract_sha256': control_view['contract_sha256'],
                'prefix_normalization': control_view['prefix_normalization'],
                'research_objective_sha256': control_view['research_objective']['contract_sha256']}
    def ref(name):
        return {'path': '/synthetic/' + name, 'sha256': '1' * 64}
    control_recipe = ref('control')
    control_recipe['path'] = str(october.FIT_ROOTS['control'].parent / 'RECIPE.json')
    args = dict(recipe_ref=control_recipe, fold=october.FOLD, feature_view_identity=identity,
                phase_state='COMPLETE', outcome='COMPLETE', optimizer_updates=1866,
                parent_cursor=29849, parent_order_sha256='0' * 64,
                full_state=ref('state'), own_initial_state=ref('initial'),
                initial_measurement=ref('initial-report'),
                final_train_measurement=ref('final-report'), oof_cursor=0,
                result=ref('result'))
    cursor = prefix.build_prefix_fit_cursor(**args)
    assert cursor['parent_cursor'] == 29849 and cursor['oof_cursor'] == 0
    invocation = {'invocation_sha256': '2' * 64, 'direction_prefix_fit_recipe': control_recipe,
                  'optimizer_step_budget': 1866, 'kind': prefix.KIND,
                  'checkpoint': {'pointer_path': '/synthetic/cursor'}}
    progress = prefix.build_prefix_fit_progress(
        window={'recipe': control_recipe}, plan={'plan_sha256': '3' * 64},
        invocation=invocation, checkpoint_pointer=ref('cursor'), cursor=cursor)
    assert progress['total_units'] == 29849 and progress['oof_cursor'] == 0
    with pytest.raises(Exception, match='CURSOR_COUNTER:oof_cursor'):
        prefix.build_prefix_fit_cursor(**{**args, 'oof_cursor': 1})
    candidate_recipe = {**control_recipe, 'path': str(october.FIT_ROOTS['candidate'].parent / 'RECIPE.json')}
    with pytest.raises(Exception, match='OCTOBER_CURSOR_ARM'):
        prefix.build_prefix_fit_cursor(**{**args, 'recipe_ref': candidate_recipe})


def test_fit_phase_gate_rejects_held_out_reads():
    gate = {'candidate_plan': october.roles.BATCH_PLAN, 'arm': 'candidate',
            'input_role': 'fit', 'held_out_features_authorized': False,
            'held_out_outcomes_authorized': False}
    october.require_fit_phase_gate(gate, {'arm': 'candidate'})
    for patch in ({'arm': 'control'}, {'input_role': 'score'},
                  {'held_out_features_authorized': True}, {'held_out_outcomes_authorized': True}):
        with pytest.raises(RuntimeError, match='FIT_ROLE_GATE'):
            october.require_fit_phase_gate({**gate, **patch}, {'arm': 'candidate'})


def test_october_fit_final_measurement_never_selects_reserved_rows():
    import numpy as np
    from gx1.scripts import run_entry_native_direction_prefix_fit_window_v1 as worker
    rows = np.arange(36172, dtype=np.int64)
    prepared = {'arrays': {'parent_rows': rows}}
    diagnostic = rows[:256]
    launch = {'run_scope': october.roles.FIT_SCOPE, 'fold': october.FOLD,
              'final_train_measurement_rows': 29849}
    assert worker._final_train_selection(prepared, diagnostic, launch, 1) is None
    selected = worker._final_train_selection(prepared, diagnostic, launch, 0)
    assert len(selected) == 29849 and selected[-1] == 29848


def test_prefix_launch_accepts_only_exact_october_fit_recipe_path(tmp_path, monkeypatch):
    from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as prefix
    monkeypatch.setattr(prefix, 'require_dynamic_state_qualification', lambda *args, **kwargs: None)
    runtime = tmp_path / 'RUNTIME'
    monkeypatch.setitem(october.FIT_ROOTS, 'candidate', runtime)
    review = {'path': str(october.PRIVATE_ROOTS['candidate'].parent / 'ACTUAL_ROOT_REVIEW.json'),
              'sha256': '1' * 64}
    recipe = october.build_fit_recipe(arm='candidate', qualification_review=review,
                                      source_commit='a' * 40)
    correct = runtime.parent / 'RECIPE.json'
    correct.write_text(json.dumps(recipe))
    assert prefix.require_prefix_fit_launch_recipe(
        {'path': str(correct), 'sha256': october.campaign.file_sha256(correct)},
        source_repo=october.REPO, source_commit='a' * 40) == recipe
    wrong = runtime.parent / 'UNBOUND.json'
    wrong.write_text(json.dumps(recipe))
    with pytest.raises(RuntimeError, match='OCTOBER_FIT_RECIPE_PATH'):
        prefix.require_prefix_fit_launch_recipe(
            {'path': str(wrong), 'sha256': october.campaign.file_sha256(wrong)},
            source_repo=october.REPO, source_commit='a' * 40)
    score = deepcopy(recipe)
    score['run_scope'] = october.roles.SCORE_SCOPE
    score['recipe_sha256'] = october.campaign.canonical_sha256(
        {k: v for k, v in score.items() if k != 'recipe_sha256'})
    score_path = runtime.parent / 'SCORE_RECIPE.json'
    score_path.write_text(json.dumps(score))
    with pytest.raises(RuntimeError):
        prefix.require_prefix_fit_launch_recipe(
            {'path': str(score_path), 'sha256': october.campaign.file_sha256(score_path)},
            source_repo=october.REPO, source_commit='a' * 40)


def test_score_recipe_requires_own_completed_fit_and_reserves_only_october(monkeypatch):
    from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as prefix
    monkeypatch.setattr(prefix, 'require_dynamic_state_qualification', lambda *args, **kwargs: None)
    original_read = october._read
    fit = {}
    origin = {}
    for arm in ('control', 'candidate'):
        private = {'path': str(october.PRIVATE_ROOTS[arm].parent/'ACTUAL_ROOT_REVIEW.json'),
                   'sha256': '1' * 64}
        fit[arm] = october.build_fit_recipe(arm=arm, qualification_review=private,
                                           source_commit='a' * 40)
        fit_ref = {'path': str(october.FIT_ROOTS[arm].parent/'RECIPE.json'),
                   'sha256': ('2' if arm == 'control' else '3') * 64}
        terminal = {'path': str(october.FIT_ROOTS[arm].parent/'ACTUAL_ROOT_REVIEW.json'),
                    'sha256': ('4' if arm == 'control' else '5') * 64}
        origin[arm] = {'fit_terminal_review': terminal, 'fit_recipe': fit_ref,
                       'fit_cursor': {'path': '/synthetic/cursor', 'sha256': '6' * 64},
                       'fit_checkpoint': {'path': '/synthetic/state', 'sha256': '7' * 64},
                       'own_initial_state': {'path': '/synthetic/initial', 'sha256': '8' * 64},
                       'initial_measurement': {'path': '/synthetic/initial-report', 'sha256': '9' * 64},
                       'final_train_measurement': {'path': '/synthetic/final-report', 'sha256': 'a' * 64},
                       'state_seal': 'b' * 64}
    monkeypatch.setattr(october, 'require_fit_origin',
                        lambda terminal, *, arm: origin[arm] if terminal == origin[arm]['fit_terminal_review']
                        else (_ for _ in ()).throw(RuntimeError('WRONG_FIT_ORIGIN')))
    monkeypatch.setattr(october, '_read',
                        lambda ref: fit['control'] if ref == origin['control']['fit_recipe'] else
                        fit['candidate'] if ref == origin['candidate']['fit_recipe'] else original_read(ref))
    recipes = {arm: october.build_score_recipe(arm=arm, fit_terminal=origin[arm]['fit_terminal_review'],
                                                source_commit='a' * 40)
               for arm in ('control', 'candidate')}
    for arm, recipe in recipes.items():
        assert october.require_score_recipe(recipe, source_repo=october.REPO,
                                            source_commit='a' * 40) == recipe
        assert recipe['input_role'] == 'score' and recipe['training_enabled'] is False
        assert recipe['max_windows'] == 1 and recipe['score_origin'] == origin[arm]
        assert recipe['target_manifest'] == original_read(october.roles.REFERENCE_PLAN)['october_series']
        assert recipe['score_storage_plan']['parents'] == 6323
    assert recipes['control']['feature_view'] != recipes['candidate']['feature_view']
    assert recipes['control']['target_manifest'] == recipes['candidate']['target_manifest']
    with pytest.raises(RuntimeError, match='SCORE_COORDINATES'):
        october.build_score_recipe(arm='candidate', fit_terminal=origin['control']['fit_terminal_review'],
                                   source_commit='a' * 40)
    tampered = deepcopy(recipes['candidate'])
    tampered['score_origin'] = origin['control']
    tampered['recipe_sha256'] = october.campaign.canonical_sha256(
        {key: value for key, value in tampered.items() if key != 'recipe_sha256'})
    with pytest.raises(RuntimeError, match='EXACT_SCORE_RECIPE'):
        october.require_score_recipe(tampered, source_repo=october.REPO, source_commit='a' * 40)


def test_score_phase_gate_and_cursor_are_own_arm_read_only():
    from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as prefix
    from gx1.scripts import run_entry_native_direction_prefix_fit_window_v1 as worker
    gate = {'candidate_plan': october.roles.BATCH_PLAN, 'arm': 'candidate', 'input_role': 'score',
            'held_out_features_authorized': True, 'held_out_outcomes_authorized': True,
            'checkpoint_reads_authorized': True, 'zero_update_reserved_scoring_authorized': True}
    october.require_score_phase_gate(gate, {'arm': 'candidate'})
    for patch in ({'arm': 'control'}, {'input_role': 'fit'}, {'held_out_outcomes_authorized': False},
                  {'checkpoint_reads_authorized': False},
                  {'zero_update_reserved_scoring_authorized': False}):
        with pytest.raises(RuntimeError, match='SCORE_ROLE_GATE'):
            october.require_score_phase_gate({**gate, **patch}, {'arm': 'candidate'})
    batch = october._read(october.roles.BATCH_PLAN)
    view = october._read(batch['candidate_view'])
    identity = {'contract_sha256': view['contract_sha256'],
                'prefix_normalization': view['prefix_normalization'],
                'research_objective_sha256': view['research_objective']['contract_sha256']}
    def ref(name):
        return {'path': '/synthetic/' + name, 'sha256': '1' * 64}
    recipe = {'path': str(october.SCORE_ROOTS['candidate'].parent/'RECIPE.json'), 'sha256': '2' * 64}
    args = dict(recipe_ref=recipe, fold=october.FOLD, feature_view_identity=identity,
                phase_state='COMPLETE', outcome='COMPLETE', optimizer_updates=1866,
                parent_cursor=29849, parent_order_sha256='0' * 64,
                full_state=ref('state'), own_initial_state=ref('initial'),
                initial_measurement=ref('initial-report'), final_train_measurement=ref('final-report'),
                oof_cursor=6323, result=ref('result'))
    cursor = prefix.build_prefix_fit_cursor(**args)
    assert cursor['oof_cursor'] == 6323
    invocation = {'invocation_sha256': '3' * 64, 'direction_prefix_fit_recipe': recipe,
                  'optimizer_step_budget': 0, 'kind': prefix.KIND,
                  'checkpoint': {'pointer_path': '/synthetic/cursor'}, 'invocation_number': 1,
                  'window_count': 1, 'expected_success_outcome': 'COMPLETE'}
    progress = prefix.build_prefix_fit_progress(window={'recipe': recipe},
        plan={'plan_sha256': '4' * 64}, invocation=invocation,
        checkpoint_pointer=ref('cursor'), cursor=cursor)
    assert progress['oof_cursor'] == 6323 and progress['optimizer_updates'] == 1866
    with pytest.raises(Exception, match='CURSOR_COUNTER:oof_cursor'):
        prefix.build_prefix_fit_cursor(**{**args, 'oof_cursor': 6324})
    with pytest.raises(Exception, match='OCTOBER_CURSOR_ARM'):
        prefix.build_prefix_fit_cursor(**{**args, 'recipe_ref':
            {'path': str(october.SCORE_ROOTS['control'].parent/'RECIPE.json'), 'sha256': '2' * 64}})
    assert worker._final_train_selection({}, None,
        {'run_scope': october.roles.SCORE_SCOPE}, 0) is None


def test_missing_fit_terminal_denies_score_before_recipe():
    terminal = {'path': str(october.FIT_ROOTS['candidate'].parent/'ACTUAL_ROOT_REVIEW.json'),
                'sha256': '0' * 64}
    with pytest.raises(Exception, match='bound JSON file unavailable'):
        october.build_score_recipe(arm='candidate', fit_terminal=terminal, source_commit='a' * 40)
