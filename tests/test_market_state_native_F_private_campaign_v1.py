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


def test_ta_mechanical_cursor_is_distinct_and_retains_four_step_transport():
    ref = {'path': '/tmp/gx1-ta-recipe.json', 'sha256': '1' * 64}
    report = {'path': '/tmp/gx1-ta-result.json', 'sha256': '2' * 64}
    state_ref = {'path': '/tmp/gx1-ta-state.pt', 'sha256': '3' * 64}
    value = scope.build_cursor(recipe_ref=ref, outcome='COMPLETE',
        optimizer_updates=4, parent_cursor=32, parent_order_sha256='4' * 64,
        full_state=state_ref, result=report, ta_sf=True)
    assert value['schema_version'] == scope.TA_CURSOR_SCHEMA
    assert outer.require_research_cursor(value, expected_recipe=ref,
        verify_files=False, batch_size=16) == value
    recipe = {'run_scope': scope.TA_SCOPE, 'fold_id': 1, 'training_enabled': True,
              'test_data_used': False, 'numerical_recipe': {'batch_size': 16}}
    assert outer.guarded_research_power_limit(recipe) == 250


def test_ta_mechanical_recipe_uses_fullfit_numerics_and_explicit_input_proof(monkeypatch):
    from gx1.contracts import market_state_time_query_fold1_fit_v1 as full
    from gx1.audit.verify_market_state_time_query_zero_step_v1 import TA_SOURCE_PATHS
    refs = {key: {'path': '/tmp/' + key + '.json', 'sha256': '1' * 64}
            for key in ('mature_plan', 'ta_candidate_view', 'ta_input_audit',
                        'ta_input_terminal', 'ta_endpoint_priors')}
    monkeypatch.setattr(scope, 'ta_input_refs', lambda: copy.deepcopy(refs))
    paths = full.required_source_paths(ta_sf=True)
    assert TA_SOURCE_PATHS <= paths
    audited = {path: campaign.file_sha256(scope.CANONICAL_REPO / path) for path in paths}
    monkeypatch.setattr(scope, '_require_ta_input_proof', lambda _: (
        {'contract_sha256': 'a' * 64}, {'source_owners': audited}))
    monkeypatch.setattr(campaign, 'require_binding', lambda ref, **kwargs: dict(ref))
    value = scope.build_ta_sf_mechanical_recipe(source_commit='a' * 40)
    assert value['numerical_recipe'] == full.ta_sf_numerical_recipe()
    assert value['numerical_recipe']['max_updates'] == 1849
    assert value['max_private_updates'] == 4
    assert value['parent_order_rows'] == 29574
    assert value['selected_fit_rows'] == 32
    assert value['parent_selection'] == scope.TA_PARENT_SELECTION
    assert value['qualification_weights_for_main_fit'] is False
    changed = copy.deepcopy(value)
    changed['numerical_recipe'] = scope.numerical_recipe()
    changed['recipe_sha256'] = campaign.canonical_sha256({
        key: item for key, item in changed.items() if key != 'recipe_sha256'})
    with pytest.raises(Exception, match='TA_RECIPE_SCOPE'):
        scope.require_ta_sf_mechanical_recipe(changed, source_repo=scope.CANONICAL_REPO,
                                              source_commit='a' * 40)
    audited['gx1/features/htf_features.py'] = '0' * 64
    with pytest.raises(Exception, match='TA_INPUT_SOURCE_DRIFT'):
        scope.require_ta_sf_mechanical_recipe(value, source_repo=scope.CANONICAL_REPO,
                                              source_commit='a' * 40)


def test_ta_input_proof_requires_completed_matching_terminal(monkeypatch):
    refs = {key: {'path': '/tmp/' + key + '.json', 'sha256': '1' * 64}
            for key in ('ta_candidate_view', 'ta_input_audit', 'ta_input_terminal')}
    values = {
        refs['ta_candidate_view']['path']: {'contract_sha256': 'a' * 64,
                                            'full_ta_feature_profile': True},
        refs['ta_input_audit']['path']: {'status': 'COMPLETE', 'test_data_used': False,
            'candidate_view': refs['ta_candidate_view'], 'input_audit': {
                'candidate_view_sha256': 'a' * 64,
                'gather': {'fit_rows': 29574, 'score_rows': 17753,
                          'all_fields_all_window_positions_bit_identical': True},
                'normalization_fits': 0, 'model_constructions': 1, 'model_forwards': 1,
                'backward_calls': 0, 'optimizer_steps': 0}},
        refs['ta_input_terminal']['path']: {'status': 'COMPLETE', 'exit_code': 0,
            'result': refs['ta_input_audit'], 'test_data_used': False,
            'no_relaunch_under_this_plan': True}}
    monkeypatch.setattr(scope, '_read', lambda ref, label: values[ref['path']])
    assert scope._require_ta_input_proof(refs)[0]['full_ta_feature_profile'] is True
    values[refs['ta_input_terminal']['path']]['exit_code'] = 1
    with pytest.raises(Exception, match='TA_INPUT_TERMINAL'):
        scope._require_ta_input_proof(refs)


def test_ta_materializer_emits_one_guarded_invocation_without_launch(tmp_path, monkeypatch):
    from gx1.scripts import materialize_market_state_time_query_fold1_fit_campaign_v1 as materializer
    root, runtime = tmp_path, tmp_path / 'RUNTIME'
    monkeypatch.setattr(scope, 'TA_ROOT', root)
    monkeypatch.setattr(scope, 'TA_RUNTIME_ROOT', runtime)
    monkeypatch.setattr(scope, 'TA_RECIPE_PATH', root / 'RECIPE.json')
    def ref(path):
        return {'path': str(path), 'sha256': '1' * 64}
    binding_path = root / 'PREPARATION_EXECUTION_BINDING.json'
    binding = {'materializer': ref(scope.CANONICAL_REPO /
        'gx1/scripts/materialize_market_state_time_query_fold1_fit_campaign_v1.py'),
        'recipe': ref(root / 'RECIPE.json'),
        'policy': ref(scope.CANONICAL_REPO / 'NEXT_RUN_POLICY.json'),
        'prepared_boot': ref(root / 'PREPARED_WINDOWS_BOOT.json'),
        'guard_template': ref(root / 'GUARD_TEMPLATE.json'), 'source_head': 'a' * 40}
    recipe = {'run_scope': scope.TA_SCOPE, 'admission_stage': scope.TA_STAGE,
        'runtime_root': str(runtime), 'recipe_sha256': '2' * 64,
        'numerical_recipe': {'max_updates': 1849}}
    gate = {'status': 'ADMITTED', 'decision': 'ADMITTED',
        'campaign_preparation_authorized': True, 'ta_sf_mechanical_authorized': True,
        'recipe': binding['recipe'], 'prepared_boot': binding['prepared_boot'],
        'guard_template': binding['guard_template'], 'materializer': binding['materializer'],
        'run_root': str(runtime), 'claimed_invocations': 0, 'completed_invocations': 0,
        'max_invocations': 1, 'relaunch_allowed': False, 'test_data_used': False,
        'fold_id': 1, 'parent_selection': scope.TA_PARENT_SELECTION, 'selected_fit_rows': 32,
        'model_forwards_authorized': 6, 'backward_calls_authorized': 4,
        'optimizer_steps_authorized': 4, 'genuine_predictor_fit_authorized': False}
    objects = {str(binding_path): binding,
        binding['policy']['path']: {scope.TA_STAGE: gate},
        binding['prepared_boot']['path']: {'synthetic_boot': True},
        binding['guard_template']['path']: {'phase': outer.PHASE,
            'test_data_used': False, 'gpu_uuid': 'GPU-synthetic',
            'signed_guard_sources': {}, 'controller_sources': {}}}
    published = {}
    def publish(path, value):
        published[str(path)] = copy.deepcopy(value)
        return ref(path)
    monkeypatch.setattr(materializer, '_ref', ref)
    monkeypatch.setattr(materializer, '_publish', publish)
    monkeypatch.setattr(materializer, 'require_capped_cpu_audit_execution', lambda: None)
    monkeypatch.setattr(materializer, 'require_capped_lock_ancestry', lambda: None)
    monkeypatch.setattr(materializer.subprocess, 'check_output',
        lambda args, **kwargs: 'a' * 40 if args[1] == 'rev-parse' else '')
    monkeypatch.setattr(materializer.os, 'sync', lambda: None)
    monkeypatch.setattr(campaign, 'file_sha256', lambda _: '1' * 64)
    monkeypatch.setattr(campaign, 'read_bound_json', lambda path, digest: objects[str(path)])
    monkeypatch.setattr(campaign, 'require_boot_identity', lambda value: value)
    monkeypatch.setattr(campaign, 'campaign_hardware_policy', lambda phase: {'phase': phase})
    monkeypatch.setattr(campaign, 'require_plan', lambda plan, **kwargs: {
        **plan, 'checked_invocations': [published[str(root / 'CAMPAIGN/INVOCATION_0001.json')]]})
    monkeypatch.setattr(campaign, 'next_action', lambda *args, **kwargs: {'decision': 'REBOOT_REQUIRED'})
    monkeypatch.setattr(outer, 'require_research_launch_recipe', lambda *args, **kwargs: recipe)
    monkeypatch.setattr(outer, 'require_native_direction_research_readiness',
        lambda *args, **kwargs: {'decision': 'READY_FOR_EXISTING_BOUND_CAMPAIGN_GATES'})
    result = materializer.materialize(binding_path)
    assert result['genuine_job_started'] is False
    assert result['decision'] == 'PASS_METADATA_PREPARED_REBOOT_REQUIRED'
    assert len(result['windows']) == 1
    invocation = published[str(root / 'CAMPAIGN/INVOCATION_0001.json')]
    window = published[str(root / 'CAMPAIGN/WINDOW_0001.json')]
    plan = published[str(root / 'CAMPAIGN/CAMPAIGN_PLAN.json')]
    assert invocation['kind'] == outer.KIND
    assert invocation['window_count'] == 1
    assert invocation['optimizer_step_budget'] == 4
    assert invocation['expected_success_outcome'] == 'COMPLETE'
    assert invocation['requires_fresh_windows_boot'] is True
    assert invocation['maximum_wall_seconds'] == 13800
    assert window['max_invocation_seconds'] == 12000
    assert window['output_directory'] == str(runtime / 'private-research')
    assert window['campaign_cursor_path'] == str(runtime / 'research-cursor/RESUME_CURSOR.json')
    assert plan['phase'] == outer.PHASE and plan['policy']['physical_power_limit_w'] == 250
    assert plan['test_data_used'] is False
