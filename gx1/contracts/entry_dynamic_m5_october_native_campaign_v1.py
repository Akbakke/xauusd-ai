"""Exact October private native recipes for the locked control and age candidate.

This source binds future campaign metadata; it grants no model or training run.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from gx1.contracts import entry_dynamic_m5_october_native_roles_v1 as roles
from gx1.contracts import local_random_access_campaign_v2 as campaign
from gx1.contracts.entry_model_native_train_recipe_v1 import native_direction_research_recipe

REPO = Path('/home/andre2/src/GX1_CURRENT')
BASE = Path('/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_NATIVE_CAMPAIGN_20261106_001')
PRIVATE_ROOTS = {arm: BASE / (arm.upper() + '_PRIVATE_002') / 'RUNTIME'
                 for arm in ('control', 'candidate')}
PRIVATE_STAGES = {arm: 'dynamic_m5_october_' + arm + '_private_20261106_002'
                  for arm in PRIVATE_ROOTS}
FIT_ROOTS = {arm: BASE / (arm.upper() + '_FIT_001') / 'RUNTIME'
             for arm in PRIVATE_ROOTS}
FIT_STAGES = {arm: 'dynamic_m5_october_' + arm + '_fit_20261106_001'
              for arm in PRIVATE_ROOTS}
SCORE_ROOTS = {arm: BASE / (arm.upper() + '_SCORE_001') / 'RUNTIME'
               for arm in PRIVATE_ROOTS}
SCORE_STAGES = {arm: 'dynamic_m5_october_' + arm + '_score_20261106_001'
                for arm in PRIVATE_ROOTS}
FOLD = {'fold_id': 'october_2024_confirmed_age_v1', 'prefix_rows': 29849,
        'oof_start_row': 29849, 'oof_end_row': 36172,
        'train_cutoff_ns': 1727740800000000000,
        'oof_cutoff_ns': 1730419200000000000}
COHORT_PLAN = {'path': str(REPO / 'configs/research/DYNAMIC_M5_OCTOBER_PRIVATE_COHORT_20261106_001.json'),
               'sha256': 'da4bc42d4e779251708276255987db47a6eb0af66d9f7781c8b107a54b72102d'}
COHORT_RESULT = {'path': '/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_PRIVATE_COHORT_20261106_001/RESULT.json',
                 'sha256': 'd71263a2f9b09a9f2a2baef5b6acbd300d50a5fb87629642ed4860d66d554811'}
COHORT_TERMINAL = {'path': '/home/andre2/GX1_RUNS/DYNAMIC_M5_OCTOBER_PRIVATE_COHORT_20261106_001/TERMINAL.json',
                   'sha256': '2aaf0bdc2dbeb92191afbfdf12f8d46e62012a2b6242478e54584d1ed301cdab'}
EXTRA_SOURCE_PATHS = (
    'gx1/contracts/entry_dynamic_m5_october_native_campaign_v1.py',
    'gx1/contracts/entry_dynamic_m5_october_native_roles_v1.py',
    'gx1/contracts/entry_dynamic_m5_october_native_identity_v1.py',
    'gx1/contracts/entry_dynamic_m5_october_causal_inputs_v1.py',
    'gx1/contracts/entry_native_direction_prefix_units_v1.py',
    'gx1/audit/dynamic_m5_october_native_inputs.py',
    'gx1/audit/dynamic_m5_october_native_batch_binding.py',
    'gx1/audit/dynamic_mtf_october_input_only.py',
    'gx1/audit/dynamic_mtf_october_feature_delivery.py',
    'gx1/audit/dynamic_mtf_december_native_inputs.py',
    'gx1/audit/dynamic_mtf_december_normalization.py',
    'gx1/audit/dynamic_mtf_train_prefix_inputs.py',
    'gx1/contracts/entry_dynamic_mtf_december_native_v1.py',
)


def _require(ok, reason):
    if not ok:
        raise RuntimeError('OCTOBER_NATIVE_CAMPAIGN_' + reason)


def _read(ref):
    return campaign.read_bound_json(Path(ref['path']), ref['sha256'])


def _owners(supplied=None):
    from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as prefix
    names = set(prefix.REQUIRED_SOURCE_PATHS) | set(prefix.DYNAMIC_STATE_EXTRA_SOURCE_PATHS) | set(EXTRA_SOURCE_PATHS)
    _require(supplied is None or (isinstance(supplied, Mapping) and set(supplied) == names),
             'SOURCE_OWNER_SET')
    result = {}
    for name in sorted(names):
        path = REPO / name
        _require(path.is_file() and not path.is_symlink(), 'SOURCE_PATH:' + name)
        if supplied is None:
            result[name] = {'path': str(path), 'sha256': campaign.file_sha256(path)}
        else:
            _require(supplied[name].get('path') == str(path), 'SOURCE_PATH:' + name)
            result[name] = campaign.require_binding(supplied[name], label='October native source ' + name)
    return result


def _registered():
    batch, refs = _read(roles.BATCH_PLAN), _read(roles.REFERENCE_PLAN)
    plan, cohort, terminal = _read(COHORT_PLAN), _read(COHORT_RESULT), _read(COHORT_TERMINAL)
    _require(batch['fit_prefix']['rows'] == FOLD['prefix_rows']
             and batch['fit_prefix']['cutoff_ns'] == FOLD['train_cutoff_ns']
             and batch['evaluation_axis']['parent_row_range'] == [FOLD['oof_start_row'], FOLD['oof_end_row']]
             and batch['evaluation_axis']['cutoff_ns'] == FOLD['oof_cutoff_ns']
             and batch['october_series'] == refs['october_series']
             and batch['target_source_refs'] == refs['target_source_refs']
             and plan['fit_series'] == refs['fit_series']
             and cohort['plan'] == COHORT_PLAN and terminal['result'] == COHORT_RESULT
             and terminal['exit_code'] == 0 and terminal['source_unchanged'] is True
             and cohort['fit_series'] == refs['fit_series']
             and cohort['fit_parent_rows'] == FOLD['prefix_rows']
             and cohort['fit_cutoff_ns_exclusive'] == FOLD['train_cutoff_ns']
             and cohort['private_support_pass'] is True and cohort['private_support_blockers'] == []
             and cohort['model_calls'] == 0 and cohort['october_outcomes_read'] is False
             and cohort['test_split_used'] is False
             and cohort['seed'] == 20260911 and cohort['lead_minutes'] == [1, 5, 15, 60, 240, 1440]
             and len(cohort['private_parent_rows_sorted']) == 32
             and len(cohort['initial_parent_rows_in_order']) == 256
             and len(set(cohort['private_parent_rows_sorted'])) == 32
             and len(set(cohort['initial_parent_rows_in_order'])) == 256
             and all(type(row) is int and 0 <= row < FOLD['prefix_rows']
                     for row in cohort['private_parent_rows_sorted'] + cohort['initial_parent_rows_in_order']),
             'FROZEN_FIT_COHORT')
    return batch, refs, cohort


def build_private_recipe(*, arm, source_owners=None):
    """Build an exact future private recipe from saved, fit-only evidence."""
    _require(arm in PRIVATE_ROOTS, 'ARM')
    with campaign.file_verification_scope():
        batch, refs, cohort = _registered()
        view_ref = batch[arm + '_view']
        view = _read(view_ref)
        _require(view['october_observer_axes'] == _read(batch['observer_axes'])
                 and view['prefix']['rows'] == FOLD['prefix_rows']
                 and view['prefix']['cutoff_ns'] == FOLD['train_cutoff_ns']
                 and view['native_endpoint_loss'] is True
                 and view.get('confirmed_run_age_input', False) is (arm == 'candidate')
                 and view['full_feature_profile'] is True, 'MATCHED_VIEW')
        numerical = native_direction_research_recipe(batch_size=16, max_updates=2,
            dynamic_state=True,
            current_state_first_loss_weight=view['current_state_first_loss_weight'],
            first_flat_activity_weight=view['first_flat_activity_weight'],
            native_endpoint_loss=True, confirmed_run_age_input=(arm == 'candidate'))
        _require(numerical['research_objective_sha256'] == view['research_objective']['contract_sha256'],
                 'OBJECTIVE_IDENTITY')
        root = PRIVATE_ROOTS[arm]
        result = {'schema_version': 'gx1_native_direction_research_launch_recipe_v1',
            'source_repo': str(REPO), 'run_scope': roles.PRIVATE_SCOPE,
            'admission_stage': PRIVATE_STAGES[arm], 'runtime_root': str(root),
            'execution_binding_path': str(root / 'EXECUTION_BINDING.json'),
            'arm': arm, 'candidate_plan': roles.BATCH_PLAN,
            'reference_plan': roles.REFERENCE_PLAN, 'input_role': 'fit',
            'fold': dict(FOLD), 'full_feature_profile': True,
            'feature_view': view_ref, 'prefix_normalization': view['prefix_normalization']['binding'],
            'prefix_prior': refs['fit_prior'], 'label_audit': refs['fit_counts'],
            'input_manifest': roles.BATCH_PLAN,
            'observer_axes': batch['observer_axes'], 'causal_inputs': batch['causal_inputs'],
            'target_manifest': refs['fit_series'], 'target_sources': batch['target_source_refs'],
            'numerical_recipe': numerical, 'source_owners': _owners(source_owners),
            'qualification_parent_rows': cohort['private_parent_rows_sorted'],
            'initial_parent_rows': cohort['initial_parent_rows_in_order'],
            'endpoint_support_per_batch': [
                [row[tf]['native']['observed'] for tf in ('M5', 'M15', 'H1', 'H4', 'D1')]
                for row in cohort['private_batch_support']],
            'max_private_updates': 4, 'max_model_constructions': 3,
            'max_forwards': 21, 'max_backwards': 4,
            'max_checkpoint_bytes': 1073741824, 'max_published_json_bytes': 33554432,
            'training_enabled': True, 'full_epoch_training_allowed': False,
            'qualification_weights_for_main_fit': False,
            'test_data_used': False, 'predictive_value_claimed': False}
        result['recipe_sha256'] = campaign.canonical_sha256(result)
        return result


def require_private_recipe(value, *, source_repo, source_commit):
    _require(Path(source_repo) == REPO and isinstance(source_commit, str)
             and len(source_commit) == 40 and all(char in '0123456789abcdef' for char in source_commit),
             'SOURCE_REPO')
    arm = value.get('arm') if isinstance(value, Mapping) else None
    _require(arm in PRIVATE_ROOTS and value.get('run_scope') == roles.PRIVATE_SCOPE,
             'PRIVATE_SCOPE')
    expected = build_private_recipe(arm=arm, source_owners=value.get('source_owners'))
    _require(value == expected and value.get('recipe_sha256') == campaign.canonical_sha256(
             {key: item for key, item in value.items() if key != 'recipe_sha256'}), 'EXACT_PRIVATE_RECIPE')
    return expected


def require_phase_gate(gate, recipe):
    _require(gate.get('candidate_plan') == roles.BATCH_PLAN
             and gate.get('arm') == recipe['arm']
             and gate.get('input_role') == 'fit'
             and gate.get('held_out_features_authorized') is False
             and gate.get('held_out_outcomes_authorized') is False,
             'PRIVATE_ROLE_GATE')


def build_fit_recipe(*, arm, qualification_review, source_commit, source_owners=None):
    """Build one pre-October fit recipe only after its own private mechanics pass."""
    _require(isinstance(source_commit, str) and len(source_commit) == 40
             and all(char in '0123456789abcdef' for char in source_commit), 'FIT_SOURCE_COMMIT')
    _require(arm in FIT_ROOTS and isinstance(qualification_review, Mapping)
             and qualification_review.get('path') == str(PRIVATE_ROOTS[arm].parent / 'ACTUAL_ROOT_REVIEW.json'),
             'FIT_PRIVATE_ORIGIN')
    from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as prefix
    with campaign.file_verification_scope():
        batch, refs, _cohort = _registered()
        view_ref = batch[arm + '_view']
        view = _read(view_ref)
        _require(view['prefix']['rows'] == FOLD['prefix_rows']
                 and view['prefix']['cutoff_ns'] == FOLD['train_cutoff_ns']
                 and view['october_observer_axes'] == _read(batch['observer_axes'])
                 and view['native_endpoint_loss'] is True
                 and view.get('confirmed_run_age_input', False) is (arm == 'candidate'),
                 'FIT_VIEW')
        updates = (FOLD['prefix_rows'] + 15) // 16
        numerical = native_direction_research_recipe(batch_size=16, max_updates=updates,
            dynamic_state=True,
            current_state_first_loss_weight=view['current_state_first_loss_weight'],
            first_flat_activity_weight=view['first_flat_activity_weight'],
            native_endpoint_loss=True, confirmed_run_age_input=(arm == 'candidate'))
        _require(numerical['research_objective_sha256'] == view['research_objective']['contract_sha256'],
                 'FIT_OBJECTIVE')
        root = FIT_ROOTS[arm]
        result = {'schema_version': 'gx1_native_direction_prefix_fit_launch_recipe_v1',
            'source_repo': str(REPO), 'run_scope': roles.FIT_SCOPE,
            'admission_stage': FIT_STAGES[arm], 'runtime_root': str(root),
            'execution_binding_path': str(root / 'EXECUTION_BINDING.json'),
            'arm': arm, 'candidate_plan': roles.BATCH_PLAN,
            'reference_plan': roles.REFERENCE_PLAN, 'input_role': 'fit',
            'fold': dict(FOLD), 'full_feature_profile': True,
            'feature_view': view_ref, 'prefix_normalization': view['prefix_normalization']['binding'],
            'prefix_prior': refs['fit_prior'], 'label_audit': refs['fit_counts'],
            'input_manifest': roles.BATCH_PLAN,
            'observer_axes': batch['observer_axes'], 'causal_inputs': batch['causal_inputs'],
            'target_manifest': refs['fit_series'], 'target_sources': batch['target_source_refs'],
            'numerical_recipe': numerical, 'source_owners': _owners(source_owners),
            'native_dynamic_state_qualification': dict(qualification_review),
            'train_passes': 1, 'initial_measurement_rows': 256,
            'final_train_measurement_rows': FOLD['prefix_rows'],
            'max_windows': 3, 'checkpoint_every_updates': 256,
            'parent_order_algorithm': prefix.PARENT_ORDER_ALGORITHM,
            'eval_batch_size': 16, 'max_checkpoint_bytes': 1073741824,
            'max_checkpoint_total_bytes': 4294967296, 'max_published_json_bytes': 33554432,
            'training_enabled': True, 'full_epoch_training_allowed': True,
            'qualification_weights_for_main_fit': False,
            'test_data_used': False, 'predictive_value_claimed': False}
        prefix.require_dynamic_state_qualification(result, source_repo=REPO, source_commit=source_commit)
        result['recipe_sha256'] = campaign.canonical_sha256(result)
        return result


def require_fit_recipe(value, *, source_repo, source_commit):
    _require(Path(source_repo) == REPO and isinstance(source_commit, str)
             and len(source_commit) == 40 and all(char in '0123456789abcdef' for char in source_commit),
             'FIT_SOURCE_REPO')
    arm = value.get('arm') if isinstance(value, Mapping) else None
    _require(arm in FIT_ROOTS and value.get('run_scope') == roles.FIT_SCOPE, 'FIT_SCOPE')
    expected = build_fit_recipe(arm=arm, source_commit=source_commit,
        qualification_review=value.get('native_dynamic_state_qualification'),
        source_owners=value.get('source_owners'))
    _require(value == expected and value.get('recipe_sha256') == campaign.canonical_sha256(
             {key: item for key, item in value.items() if key != 'recipe_sha256'}), 'EXACT_FIT_RECIPE')
    return expected


def require_fit_phase_gate(gate, recipe):
    _require(gate.get('candidate_plan') == roles.BATCH_PLAN
             and gate.get('arm') == recipe['arm']
             and gate.get('input_role') == 'fit'
             and gate.get('held_out_features_authorized') is False
             and gate.get('held_out_outcomes_authorized') is False,
             'FIT_ROLE_GATE')


def _fit_stage_file(ref, arm, name):
    root = FIT_ROOTS[arm].parent
    _require(isinstance(ref, Mapping) and ref.get('path') == str(root / name),
             'FIT_ARTIFACT_PATH:' + name)
    return _read(ref)


def require_fit_origin(fit_terminal, *, arm):
    """Verify a closed own-arm fit and its checkpoint before any October score read."""
    _require(arm in FIT_ROOTS, 'SCORE_ARM')
    from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as prefix
    root = FIT_ROOTS[arm]
    review = _fit_stage_file(fit_terminal, arm, 'ACTUAL_ROOT_REVIEW.json')
    _require(review.get('schema_version') == 'gx1_october_native_fit_terminal_review_v1'
             and review.get('status') == 'COMPLETE_CONSUMED_OCTOBER_FIT'
             and review.get('arm') == arm and review.get('candidate_plan') == roles.BATCH_PLAN
             and review.get('test_data_used') is False, 'FIT_TERMINAL_REVIEW')
    plan = _fit_stage_file(review['plan'], arm, 'CAMPAIGN/CAMPAIGN_PLAN.json')
    _require(plan['recipe'] == review['recipe']
             and plan['recipe']['path'] == str(root.parent/'RECIPE.json')
             and plan['runtime_root'] == str(root), 'FIT_TERMINAL_CAMPAIGN')
    checked = campaign.require_plan(plan, verify_files=True)
    receipt_paths = sorted((root/'receipts').glob('invocation-*.json'))
    _require(1 <= len(receipt_paths) <= 3
             and [path.name for path in receipt_paths] == [f'invocation-{n:04d}.json'
                 for n in range(1, len(receipt_paths)+1)]
             and review['receipt']['path'] == str(receipt_paths[-1])
             and not (root/'ACTIVE_INVOCATION.json').exists(), 'FIT_RECEIPT_SET')
    receipts = [campaign.read_bound_json(path, campaign.file_sha256(path)) for path in receipt_paths]
    chain = campaign.require_receipt_chain(checked, receipts)
    _require(chain[-1]['outcome'] == 'COMPLETE' and chain[-1]['guard_decision'] == 'PASS'
             and all(row['outcome'] == 'RESUMABLE' for row in chain[:-1])
             and chain[-1]['checkpoint_pointer_after'] == review['cursor']
             and _read(review['receipt'])['receipt_sha256'] == chain[-1]['receipt_sha256'],
             'FIT_TERMINAL_RECEIPTS')
    cursor = prefix.require_prefix_fit_cursor(
        _fit_stage_file(review['cursor'], arm, 'RUNTIME/prefix-cursor/RESUME_CURSOR.json'),
        expected_recipe=plan['recipe'], expected_fold=FOLD)
    result = _fit_stage_file(review['result'], arm,
                             f'RUNTIME/prefix-fit/window-{len(chain):04d}/RESULT.json')
    _require(cursor['outcome'] == cursor['phase_state'] == result['outcome'] == 'COMPLETE'
             and cursor['parent_cursor'] == result['parent_cursor'] == FOLD['prefix_rows']
             and 0 < cursor['optimizer_updates'] == result['optimizer_updates'] <= 1866
             and cursor['oof_cursor'] == result['oof_cursor'] == 0
             and result.get('oof_measurement') is None
             and result.get('error') is None and result.get('source_unchanged') is True
             and result.get('forbidden_open_paths') == []
             and result.get('qualification_weights_reused') is False
             and result.get('test_data_used') is False
             and result.get('candidate_plan') == roles.BATCH_PLAN
             and result.get('arm') == arm and result.get('input_role') == 'fit'
             and result.get('held_out_features_read') is False
             and result.get('held_out_outcomes_read') is False
             and result.get('physical_optimizer_updates') == cursor['optimizer_updates']
             and result.get('counts', {}).get('optimizer_step_returns') == cursor['optimizer_updates']
             and result.get('counts', {}).get('backward_returns') == cursor['optimizer_updates']
             and cursor['result'] == review['result'] and result['recipe'] == review['recipe'],
             'FIT_TERMINAL_OUTCOME')
    _require(roles.identity.require_checkpoint_view(cursor['feature_view_identity']) == arm,
             'FIT_CHECKPOINT_ARM')
    checkpoint = result['durable_checkpoint']
    _require(checkpoint['state'] == result['full_state'] == cursor['full_state']
             and checkpoint['parent_cursor'] == cursor['parent_cursor']
             and checkpoint['optimizer_updates'] == cursor['optimizer_updates']
             and checkpoint['state']['path'] == str(root/'prefix-fit/states'/
                 f"w{len(chain):04d}_TERMINAL_u{cursor['optimizer_updates']:06d}_p029849.pt"),
             'FIT_TERMINAL_CHECKPOINT')
    campaign._sha(checkpoint['contract_sha256'], 'October fit full-state seal')
    fit_recipe = _read(plan['recipe'])
    _require(fit_recipe['run_scope'] == roles.FIT_SCOPE and fit_recipe['arm'] == arm
             and fit_recipe['source_owners'] == _owners(fit_recipe['source_owners']),
             'FIT_RECIPE_SOURCE')
    for role, key, count in (('INITIAL_TRAIN', 'initial_measurement', 256),
                             ('FINAL_TRAIN', 'final_train_measurement', FOLD['prefix_rows'])):
        ref = cursor[key]
        _require(Path(ref['path']).is_relative_to(root/'prefix-fit'), 'FIT_MEASUREMENT_ROOT')
        report = _read(ref)
        _require(report['role'] == role and report['fold'] == FOLD
                 and report['target_manifest'] == fit_recipe['target_manifest']
                 and report['research_objective_sha256'] == fit_recipe['numerical_recipe']['research_objective_sha256']
                 and report['feature_view_identity'] == cursor['feature_view_identity']
                 and report['own_initial_state'] == cursor['own_initial_state']
                 and len(report['timeframes']) == 5
                 and report['test_data_used'] is False, 'FIT_MEASUREMENT:' + role)
        _require(Path(report['predictions']['path']).is_relative_to(root/'prefix-fit'),
                 'FIT_PREDICTIONS_ROOT')
        arrays = _read(report['predictions'])
        _require(arrays['arrays']['probabilities']['shape'] == [count, 5, 1440, 3],
                 'FIT_MEASUREMENT_ROWS:' + role)
    power = _fit_stage_file(review['power_after'], arm, 'POWER_AFTER.json')
    gpu = [part.strip() for part in power['gpu'][0].split(',')] if len(power.get('gpu', [])) == 1 else []
    _require(len(gpu) == 6 and gpu[0] == plan['gpu_uuid']
             and float(gpu[1]) == power.get('configured_power_limit_w') == 250
             and power.get('task_state') == 'Disabled'
             and power.get('test_data_used') is False
             and {row['TaskName'] for row in power['keeper'] if row.get('State') == 4}
                 == {'GX1GpuPowerLimit', 'GX1HostTelemetryBridge'},
             'FIT_TERMINAL_POWER')
    return {'fit_terminal_review': dict(fit_terminal), 'fit_recipe': review['recipe'],
            'fit_cursor': review['cursor'], 'fit_checkpoint': cursor['full_state'],
            'own_initial_state': cursor['own_initial_state'],
            'initial_measurement': cursor['initial_measurement'],
            'final_train_measurement': cursor['final_train_measurement'],
            'state_seal': checkpoint['contract_sha256']}


def score_storage_plan(*, max_published_json_bytes, max_checkpoint_total_bytes):
    """Reserve one staged October prediction set plus JSON and checkpoint headroom."""
    from gx1.contracts.entry_dynamic_mtf_state_report_v1 import dynamic_state_prediction_file_bytes
    _require(type(max_published_json_bytes) is int and max_published_json_bytes > 0
             and type(max_checkpoint_total_bytes) is int and max_checkpoint_total_bytes >= 0,
             'SCORE_STORAGE_LIMITS')
    rows = FOLD['oof_end_row'] - FOLD['oof_start_row']
    files = dynamic_state_prediction_file_bytes(rows, 1440)
    raw = sum(files.values())
    result = {'schema_version': 'gx1_october_score_storage_v1', 'parents': rows,
              'base_npy': files, 'total_npy_bytes': raw,
              'staged_sidecar_and_manifest_cap': max_published_json_bytes,
              'campaign_json_reserve': max_published_json_bytes,
              'checkpoint_reserve': max_checkpoint_total_bytes,
              'required_free_bytes': raw + 2 * max_published_json_bytes + max_checkpoint_total_bytes}
    result['contract_sha256'] = campaign.canonical_sha256(result)
    return result


def require_score_storage(directory, plan, *, expected_host=None):
    from datetime import datetime, timezone
    from gx1.contracts.entry_native_direction_prefix_fit_campaign_v1 import _observe_wsl_host_storage
    _require(plan == score_storage_plan(max_published_json_bytes=plan['campaign_json_reserve'],
        max_checkpoint_total_bytes=plan['checkpoint_reserve']), 'SCORE_STORAGE_PLAN')
    observed = _observe_wsl_host_storage(Path(directory))
    host = (observed['host_base_path'], observed['host_drive_root'])
    _require(expected_host is None or tuple(expected_host) == host, 'SCORE_HOST_IDENTITY')
    free = min(observed[key] for key in ('guest_available_bytes', 'host_available_bytes',
                                          'mounted_host_available_bytes'))
    _require(free >= plan['required_free_bytes'], 'SCORE_DISK_HEADROOM')
    return {'plan': plan, 'observation': observed, 'host_identity': list(host),
            'observed_utc': datetime.now(timezone.utc).isoformat(), 'reserved': False}


def build_score_recipe(*, arm, fit_terminal, source_commit, source_owners=None):
    """Bind one read-only October score to its own fully closed fit checkpoint."""
    _require(arm in SCORE_ROOTS and isinstance(source_commit, str)
             and len(source_commit) == 40 and all(char in '0123456789abcdef' for char in source_commit)
             and isinstance(fit_terminal, Mapping)
             and fit_terminal.get('path') == str(FIT_ROOTS[arm].parent/'ACTUAL_ROOT_REVIEW.json'),
             'SCORE_COORDINATES')
    from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as prefix
    with campaign.file_verification_scope():
        batch, refs, _cohort = _registered()
        origin = require_fit_origin(fit_terminal, arm=arm)
        fit_recipe = _read(origin['fit_recipe'])
        view_ref = batch[arm + '_view']
        _require(fit_recipe['run_scope'] == roles.FIT_SCOPE and fit_recipe['arm'] == arm
                 and fit_recipe['feature_view'] == view_ref
                 and fit_recipe['target_manifest'] == refs['fit_series']
                 and fit_recipe['fold'] == FOLD
                 and fit_recipe['source_owners'] == _owners(source_owners),
                 'SCORE_FIT_IDENTITY')
        root = SCORE_ROOTS[arm]
        result = {'schema_version': 'gx1_native_direction_prefix_fit_launch_recipe_v1',
            'source_repo': str(REPO), 'run_scope': roles.SCORE_SCOPE,
            'admission_stage': SCORE_STAGES[arm], 'runtime_root': str(root),
            'execution_binding_path': str(root/'EXECUTION_BINDING.json'),
            'arm': arm, 'candidate_plan': roles.BATCH_PLAN,
            'reference_plan': roles.REFERENCE_PLAN, 'input_role': 'score',
            'fold': dict(FOLD), 'full_feature_profile': True,
            'feature_view': view_ref,
            'prefix_normalization': fit_recipe['prefix_normalization'],
            'prefix_prior': fit_recipe['prefix_prior'], 'label_audit': fit_recipe['label_audit'],
            'input_manifest': roles.BATCH_PLAN,
            'observer_axes': batch['observer_axes'], 'causal_inputs': batch['causal_inputs'],
            'target_manifest': refs['october_series'], 'target_sources': batch['target_source_refs'],
            'numerical_recipe': fit_recipe['numerical_recipe'],
            'source_owners': fit_recipe['source_owners'],
            'native_dynamic_state_qualification': fit_recipe['native_dynamic_state_qualification'],
            'fit_terminal': dict(fit_terminal), 'score_origin': origin,
            'score_storage_plan': score_storage_plan(max_published_json_bytes=33554432,
                                                      max_checkpoint_total_bytes=4294967296),
            'train_passes': 1, 'initial_measurement_rows': 256,
            'final_train_measurement_rows': FOLD['prefix_rows'],
            'max_windows': 1, 'checkpoint_every_updates': 256,
            'parent_order_algorithm': prefix.PARENT_ORDER_ALGORITHM,
            'eval_batch_size': 16, 'max_checkpoint_bytes': 1073741824,
            'max_checkpoint_total_bytes': 4294967296, 'max_published_json_bytes': 33554432,
            'training_enabled': False, 'full_epoch_training_allowed': False,
            'qualification_weights_for_main_fit': False,
            'test_data_used': False, 'predictive_value_claimed': False}
        roles.require_role_binding(result)
        result['recipe_sha256'] = campaign.canonical_sha256(result)
        return result


def require_score_recipe(value, *, source_repo, source_commit):
    _require(Path(source_repo) == REPO and isinstance(source_commit, str)
             and len(source_commit) == 40 and all(char in '0123456789abcdef' for char in source_commit),
             'SCORE_SOURCE_REPO')
    arm = value.get('arm') if isinstance(value, Mapping) else None
    _require(arm in SCORE_ROOTS and value.get('run_scope') == roles.SCORE_SCOPE,
             'SCORE_SCOPE')
    expected = build_score_recipe(arm=arm, fit_terminal=value.get('fit_terminal'),
        source_commit=source_commit, source_owners=value.get('source_owners'))
    _require(value == expected and value.get('recipe_sha256') == campaign.canonical_sha256(
             {key: item for key, item in value.items() if key != 'recipe_sha256'}),
             'EXACT_SCORE_RECIPE')
    return expected


def require_score_phase_gate(gate, recipe):
    _require(gate.get('candidate_plan') == roles.BATCH_PLAN
             and gate.get('arm') == recipe['arm'] and gate.get('input_role') == 'score'
             and gate.get('held_out_features_authorized') is True
             and gate.get('held_out_outcomes_authorized') is True
             and gate.get('checkpoint_reads_authorized') is True
             and gate.get('zero_update_reserved_scoring_authorized') is True,
             'SCORE_ROLE_GATE')
