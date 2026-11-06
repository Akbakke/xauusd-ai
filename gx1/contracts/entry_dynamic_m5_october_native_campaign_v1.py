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
PRIVATE_ROOTS = {arm: BASE / (arm.upper() + '_PRIVATE_001') / 'RUNTIME'
                 for arm in ('control', 'candidate')}
PRIVATE_STAGES = {arm: 'dynamic_m5_october_' + arm + '_private_20261106_001'
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
