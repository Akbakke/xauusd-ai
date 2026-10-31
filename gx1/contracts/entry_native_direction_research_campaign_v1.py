"""Strict pure-direction research inside the existing guarded native campaign.

Metadata is stdlib-only. The shared campaign still owns source/physical-boot,
marker, launcher, signed hardware, receipts and the CURRENT lock. Numerical
recipe validation and model work belong after the canonical CUDA guard. This
qualification variant admits four private updates only, never forecast fit.
The recipe binds source bytes; the outer plan owns the committed Git identity,
so a policy can bind an exact recipe without a policy/recipe/commit hash cycle.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from gx1.contracts import local_random_access_campaign_v2 as campaign

PHASE = 'native_direction_research'
KIND = 'native_direction_research_window'
WINDOW_SCHEMA = 'gx1_native_direction_research_window_policy_v1'
RECIPE_SCHEMA = 'gx1_native_direction_research_launch_recipe_v1'
CURSOR_SCHEMA = 'gx1_native_direction_research_cursor_v1'
BPS_SCOPE = 'private_bps_b16_matched_initialization_resume_qualification'
WEIGHT_SCOPE = 'private_magnitude_weighted_b16_matched_initialization_resume_qualification'
ORDER_PRIVATE_SCOPE = 'private_sign_b16_chronological_initial_resume_qualification'
ORDER_SUPPORTED_SCOPE = 'private_sign_b16_chronological_supported_initial_resume_qualification'
DENSE_PRIVATE_SCOPE = 'private_dense_b16_initial_resume_qualification'
DECIMAL_PRIVATE_SCOPE = 'private_dense_decimal_b16_initial_resume_qualification'
FAMILY_PRIVATE_SCOPE = 'private_m5_family_sign_b16_initial_resume_qualification'
FAMILY_EXTRA_SOURCE_PATHS = ('gx1/contracts/entry_direction_family_intervention_v1.py',)
FAMILY_PROTOCOL_PATH = '/home/andre2/src/GX1_CURRENT/docs/DIRECTION_FAMILY_ADDED_VALUE_PROTOCOL_20261030.json'
DECIMAL_INPUT_AUDIT = {'path':'/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_DECIMAL_BOUND_INPUT_ACTUAL_20261028_001.json',
    'sha256':'8415d195256ea7dd9bc9cd35661c1536fe57c1db330b0969d16a963a3381be19'}
DECIMAL_INITIAL_PREDICTIONS = {'path':'/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DENSE_ONEPASS_FIT_20261028_001/RUNTIME/prefix-fit/window-0001/INITIAL_TRAIN.npz',
    'sha256':'d9f820f6d38b56879f6ae0811255e240603502ca08a0be257f92da1ca6499871'}
DENSE_INPUT_MANIFEST = {'path':'/home/andre2/GX1_RUNS/NATIVE_M5_DIRECTION_PREFIX_FIT_FOLD1_20261025_276/INPUT_MANIFEST.json',
    'sha256':'befabac7ebff4e5e8b6233b299b2fc8f03b87b94d99bc7ec4d43b0fb76f9246f'}
DENSE_TARGET_RESULT = {'path':'/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_DENSE_MINUTE_TARGETS_20261028_001/RESULT.json',
    'sha256':'1c832da918aee94ab2535898aab340e33e771c8ad9739e97ece21599be50128b'}
DENSE_TARGET = {'path':'/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_DENSE_MINUTE_TARGETS_20261028_001/DENSE_TARGETS.npz',
    'sha256':'07b504dfc95f39ff818f1bafd76ff9de3a4c3133f85e85e06ec09edae675676d'}
ORDER_SUPPORTED_DESIGN = {'path':'/home/andre2/src/GX1_CURRENT/docs/DIRECTION_ORDER_SUPPORTED_QUALIFICATION_20261028.json',
    'sha256':'d78335c7d089ff44beb2d3a470ec8e88fe204899e8ef6e1524cbcba089913cb1'}
ORDER_DESIGN = {'path':'/home/andre2/src/GX1_CURRENT/docs/DIRECTION_ORDER_ABLATION_DESIGN_20261028.json',
    'sha256':'336c7c5af9a0b5bf7759831394f27de06d46eb8ab9a22619a8c1856abc01fbf7'}
WEIGHT_DESIGN = {'path':'/home/andre2/src/GX1_CURRENT/docs/DIRECTION_MAG_WEIGHT_NATIVE_QUALIFICATION_DESIGN_20261028.json',
    'sha256':'fe5cd8dbb9f17e46c77821c5077c6ad2893719bc2204f01b963cfff90c4938c1'}
WEIGHT_INITIAL_PREDICTIONS = {'path':'/home/andre2/GX1_RUNS/NATIVE_M5_DIRECTION_PREFIX_FIT_FOLD1_20261025_276/RUNTIME/prefix-fit/window-0001/INITIAL_TRAIN.npz',
    'sha256':'f870c6eb0a6000b2be0197bfe650db3f35f41d125d91a90162df2d01aedf112a'}
BPS_CURSOR_SCHEMA = 'gx1_native_direction_bps_research_cursor_v1'
BPS_FOLD = {'fold_id':1,'prefix_rows':47602,'oof_start_row':47602,'oof_end_row':64890,
    'train_cutoff_ns':1735689600000000000,'oof_cutoff_ns':1743465600000000000}
BPS_GC_PARITY = {'path':'/home/andre2/GX1_RUNS/DIRECTION_DIAGNOSTIC_EXECUTION_PROGRAM_20261025_01a171d2/DIRECTION_GC_LAYOUT_REUSE_QUALIFICATION_ROOT_REVIEW_003.json',
    'sha256':'1d3347ee52f02af7afcac5d55c02532c0e2823e1ce20397633785a2512d39263'}
BPS_DESIGN = {'path':'/home/andre2/GX1_RUNS/DIRECTION_DIAGNOSTIC_EXECUTION_PROGRAM_20261025_01a171d2/DIRECTION_BPS_GENUINE_QUALIFICATION_SOURCE_DESIGN_001.json',
    'sha256':'a374d33275545127332b053aa3dced0430711951617288cca3419afa5b844c11'}
BPS_EXTRA_SOURCE_PATHS = ('gx1/contracts/entry_native_direction_bps_v1.py',
    'gx1/contracts/entry_native_direction_frozen_extraction_v1.py',
    'gx1/contracts/entry_native_direction_checkpoint_diagnostic_v1.py',
    'gx1/scripts/run_entry_native_direction_prefix_fit_window_v1.py',
    'gx1/scripts/run_entry_native_direction_bps_qualification_v1.py')
ORDER_EXTRA_SOURCE_PATHS = ('gx1/contracts/entry_native_direction_prefix_fit_campaign_v1.py',
    'gx1/contracts/entry_native_direction_frozen_extraction_v1.py',
    'gx1/scripts/run_entry_native_direction_prefix_fit_window_v1.py',
    'gx1/scripts/run_entry_native_direction_order_qualification_v1.py')
DENSE_EXTRA_SOURCE_PATHS = ('gx1/contracts/entry_native_direction_prefix_fit_campaign_v1.py',
    'gx1/contracts/entry_native_direction_frozen_extraction_v1.py',
    'gx1/contracts/entry_native_direction_context_report_v1.py',
    'gx1/contracts/entry_native_direction_dense_v1.py',
    'gx1/contracts/entry_native_direction_dense_data_v1.py',
    'gx1/contracts/entry_native_direction_dense_report_v1.py',
    'gx1/contracts/entry_direction_probability_reference_v1.py',
    'gx1/scripts/run_entry_native_direction_prefix_fit_window_v1.py',
    'gx1/scripts/run_entry_native_direction_dense_qualification_v1.py')
DECIMAL_EXTRA_SOURCE_PATHS = ('gx1/contracts/entry_native_direction_decimal_data_v1.py',
    'gx1/contracts/entry_native_direction_decimal_input_v1.py')
M1_PRICE_EXTRA_SOURCE_PATHS = ('gx1/features/entry_direction_momentum_profile_v1.py',
    'gx1/contracts/entry_closed_m1_ofi_window_v1.py',
    'gx1/contracts/entry_native_direction_m1_ofi_normalization_v1.py',
    'gx1/scripts/audit_raw_gc_m1_reference_v1.py')
M5_DENSE_BPS_STAGE = 'dynamic_m5_direction_dense_bps_private_20261030_001'
_M5_DENSE_BPS_ROOT = '/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63'
M5_DENSE_BPS_SCALE = {'path':_M5_DENSE_BPS_ROOT+'/DIRECTION_M5_DENSE_BPS_SCALE_20261030_001/OUTPUT/OWN_TRAIN_DENSE_BPS_SCALE.json', 'sha256':'b5f26bbda63f9fa02bc19eb370f17608d116432eef935b85d97270a37a7bedd9'}
M5_DENSE_BPS_SCALE_RESULT = {'path':_M5_DENSE_BPS_ROOT+'/DIRECTION_M5_DENSE_BPS_SCALE_20261030_001/OUTPUT/RESULT.json', 'sha256':'a3ca04a15b3d31af514e0a6fe1faccc6b144797a1f63f427d7d2f1588a91176e'}
M5_DENSE_BPS_SCALE_TERMINAL = {'path':_M5_DENSE_BPS_ROOT+'/DIRECTION_M5_DENSE_BPS_SCALE_20261030_001/OUTPUT/TERMINAL.json', 'sha256':'b9df328fbb446da62815fb72691ec7ff76437a2b8d87700701f67b96bbe57664'}
M5_DENSE_BPS_COUNT_REVIEW = {'path':_M5_DENSE_BPS_ROOT+'/DIRECTION_M5_DENSE_BPS_SCALE_20261030_001/OUTPUT/INDEPENDENT_PRIOR_AUDIT_COUNT_CHECK.json', 'sha256':'fb41cef41756c43706a18f57a7d842061b3036e7eb79e962074dbb5ed0d29b62'}
M5_DENSE_BPS_BASELINE_INITIAL = {'path':_M5_DENSE_BPS_ROOT+'/DIRECTION_M5_PRICE_ONLY_PRIVATE_QUALIFICATION_20261029_001/RUNTIME/private-research/INITIAL_TRAIN.npz', 'sha256':'90c298464bf2282ee17f5b71a02fc05c1fd942d9254a11e7edad63d631bf7792'}
M5_DENSE_BPS_BASELINE_REPORT = {'path':_M5_DENSE_BPS_ROOT+'/DIRECTION_M5_PRICE_ONLY_ONEPASS_FIT_20261029_001/RUNTIME/prefix-fit/window-0001/INITIAL_TRAIN.json', 'sha256':'a468c941d3d6246581e94c6ae4d728fc2689d51eee4d29e28f6302ac460d7005'}

M1_PRICE_SOURCE_TRANSITION = {'path':'/home/andre2/src/GX1_CURRENT/docs/DIRECTION_M1_PRICE_SOURCE_TRANSITION_20261029.json',
    'sha256':'d07be3985c931205f2482f566a3ec2087316346f31d84e34127590692a604524'}
M1_NORMALIZATION_PLAN = {'path':'/home/andre2/src/GX1_CURRENT/docs/DIRECTION_M1_TRAIN_NORMALIZATION_20261029.json',
    'sha256':'78f035271c8cf2e70fb224bff0656eba0044046942f7a407fd4a091301e0fc93'}
M1_NORMALIZATION_RESULT = {'path':'/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63/DIRECTION_M1_NORMALIZATION_TRAIN_20261029_001/OUTPUT/RESULT.json',
    'sha256':'e3cc9b2c99c5ab1afe404b6d642ae018fe708fbdfa824f8a0e502b0b292b6983'}
MODULE = 'gx1.scripts.run_unified_exit_native_candidate_window_v1'
CANONICAL_REPO = Path('/home/andre2/src/GX1_CURRENT')
REPO_SOURCE_PATHS = (
    'gx1/contracts/entry_exit_feature_surface_v1.py',
    'gx1/contracts/entry_exit_production_architecture_v1.py',
    'gx1/contracts/entry_gc_mtf_history_v1.py',
    'gx1/contracts/entry_model_native_input_normalization_v1.py',
    'gx1/contracts/entry_native_direction_prefix_normalization_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'gx1/features/basic_v1.py', 'gx1/features/entry_model_native_feature_layers_v1.py',
    'gx1/features/entry_specialist_feature_groups_v1.py', 'gx1/features/gc_native_context_v1.py',
    'gx1/features/htf_features.py', 'gx1/features/micro_structure_v1.py',
    'gx1/features/volatility_squeeze_state_v1.py',
    'gx1/models/entry_v10/entry_v10_input_normalization.py',
    'gx1/contracts/registry_hyperparameter_fit_v1.py', 'gx1/utils/artifact_primitives_v1.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'gx1/contracts/entry_native_direction_feature_view_v1.py',
    'gx1/contracts/entry_native_direction_research_v1.py',
    'gx1/contracts/entry_native_direction_research_state_v1.py',
    'gx1/contracts/entry_model_native_train_recipe_v1.py',
    'gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py',
    'gx1/contracts/entry_model_native_signal_v1.py',
    'gx1/contracts/entry_model_native_training_objective_v1.py',
    'gx1/contracts/entry_causal_m1_outcomes_v1.py', 'gx1/contracts/entry_exit_feature_base_v1.py',
    'gx1/models/entry_v10/entry_v10_ctx_train_v3.py',
    'gx1/scripts/run_unified_exit_random_access_full_train_v1.py',
    'gx1/contracts/entry_training_precision_v1.py', 'gx1/contracts/model_state_digest_v1.py',
    'gx1/contracts/unified_exit_random_access_state_view_v1.py',
    'gx1/contracts/local_random_access_campaign_v2.py',
    'gx1/contracts/entry_native_direction_research_campaign_v1.py',
    'gx1/scripts/local_random_access_campaign_v2.py',
    'gx1/scripts/run_unified_exit_native_candidate_window_v1.py',
    'gx1/scripts/run_entry_native_direction_research_window_v1.py',
    'scripts/gx1_capped_run.sh', 'scripts/gx1_guarded_trainer_exec.sh',
    'scripts/collect_gx1_handover_readonly.py',
)
RUNTIME_SOURCE_PATHS = tuple('.venv/lib/python3.10/site-packages/' + name for name in (
    'torch/_tensor.py', 'torch/autograd/__init__.py', 'numpy/lib/_npyio_impl.py',
    'torch/optim/adamw.py', 'torch/optim/optimizer.py', 'torch/utils/_foreach_utils.py', 'torch/version.py',
    'torch/serialization.py'))
REQUIRED_SOURCE_PATHS = REPO_SOURCE_PATHS + RUNTIME_SOURCE_PATHS


def _require(ok: bool, reason: str) -> None:
    if not ok:
        raise campaign.RandomAccessCampaignError('NATIVE_DIRECTION_RESEARCH_' + reason)


def _seal(value: Mapping[str, Any], key: str) -> dict[str, Any]:
    result = dict(value)
    result[key] = campaign.canonical_sha256(result)
    return result


def _identity(value: Any, fields: set[str], schema: str, seal: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping) and set(value) == fields | {seal}, 'FIELDS:' + schema)
    result = dict(value)
    claimed = campaign._sha(result.pop(seal), seal)
    _require(result.get('schema_version') == schema and claimed == campaign.canonical_sha256(result), 'IDENTITY:' + schema)
    result[seal] = claimed
    return result


def _path(value: Any, label: str) -> str:
    path = campaign._absolute(value, label)
    _require(path.resolve() == path and not any(item.is_symlink() for item in (path, *path.parents)), 'PATH:' + label)
    return str(path)


def require_research_launch_recipe(binding: Mapping[str, str], *, source_repo: Path,
        source_commit: str, verify_files: bool = True) -> dict[str, Any]:
    """Verify metadata and source bytes; never import Torch or grant training."""
    ref = campaign.require_binding(binding, label='direction research recipe', verify_file=verify_files)
    _require(verify_files, 'RECIPE_REQUIRES_PHYSICAL_METADATA')
    value = campaign.read_bound_json(Path(ref['path']), ref['sha256'])
    fields = {'schema_version', 'source_repo', 'runtime_root', 'admission_stage', 'run_scope',
        'numerical_recipe', 'input_manifest', 'source_owners', 'execution_binding_path',
        'max_private_updates', 'max_model_constructions', 'max_forwards', 'max_backwards',
        'max_checkpoint_bytes', 'max_published_json_bytes', 'training_enabled',
        'full_epoch_training_allowed', 'test_data_used', 'predictive_value_claimed',
        'qualification_weights_for_main_fit'}
    from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as prefix
    dense_bps = value.get('run_scope') == prefix.M5_DENSE_BPS_PRIVATE_SCOPE
    family = value.get('run_scope') == FAMILY_PRIVATE_SCOPE
    candidate = value.get('run_scope') in (prefix.M1_PRICE_PRIVATE_ADAPTER_SCOPE, prefix.M5_DENSE_BPS_PRIVATE_SCOPE, FAMILY_PRIVATE_SCOPE)
    priority = value.get('numerical_recipe', {}).get('five_minute_priority', False)
    _require(type(priority) is bool and (not priority or (candidate and value.get('candidate_mode') == 'price_only')), 'FIVE_PRIORITY_PRIVATE_SCOPE')
    bps = value.get('run_scope') == BPS_SCOPE
    weighted = value.get('run_scope') == WEIGHT_SCOPE
    supported = value.get('run_scope') == ORDER_SUPPORTED_SCOPE
    chronological = value.get('run_scope') == ORDER_PRIVATE_SCOPE or supported
    decimal = value.get('run_scope') == DECIMAL_PRIVATE_SCOPE or candidate
    dense = value.get('run_scope') == DENSE_PRIVATE_SCOPE or decimal
    b16 = bps or weighted or chronological or dense
    if b16: fields |= {'fold','gc_layout_metadata_parity','qualification_design'}
    if dense: fields |= {'dense_source_transition_proof','dense_target_result','dense_target'}
    if decimal: fields |= {'decimal_input_audit','decimal_target_refs'}
    if decimal and not candidate: fields.add('decimal_initial_predictions')
    if family:
        fields |= {'family_arm','family_intervention_sha256','family_protocol',
            'family_baseline_initial_report','family_baseline_initial_predictions'}
    if candidate:
        fields |= {'candidate_mode','parent_order_algorithm'}
        if priority: fields |= {'priority_design','priority_baseline_initial_report','priority_source_review'}
        if dense_bps: fields |= {'dense_bps_scale','dense_bps_scale_result','dense_bps_scale_terminal','dense_bps_count_review','dense_bps_initial_predictions','dense_bps_initial_report'}
        if value.get('candidate_mode') in ('price_and_m1',prefix.M1_PRICE_COARSENED_MODE): fields.add('candidate_m1_refs')
    recipe = _identity(value, fields, RECIPE_SCHEMA, 'recipe_sha256')
    _require(Path(_path(recipe['source_repo'], 'source repo')) == source_repo == CANONICAL_REPO
        and campaign._COMMIT.fullmatch(source_commit) is not None, 'RECIPE_SOURCE')
    _path(recipe['runtime_root'], 'runtime root')
    _require(type(recipe['admission_stage']) is str and recipe['admission_stage'].startswith('dynamic_m5_direction_')
        and recipe['run_scope'] in ('private_four_update_resume_qualification', BPS_SCOPE, WEIGHT_SCOPE, ORDER_PRIVATE_SCOPE, ORDER_SUPPORTED_SCOPE, DENSE_PRIVATE_SCOPE, DECIMAL_PRIVATE_SCOPE, prefix.M1_PRICE_PRIVATE_ADAPTER_SCOPE, prefix.M5_DENSE_BPS_PRIVATE_SCOPE, FAMILY_PRIVATE_SCOPE), 'RECIPE_SCOPE')
    if dense_bps:
        _require(recipe['admission_stage']==M5_DENSE_BPS_STAGE and recipe.get('candidate_mode')=='price_only'
            and not priority, 'DENSE_BPS_PRIVATE_SCOPE')
    if candidate:
        _require(recipe['candidate_mode'] in ('price_only','price_and_m1',prefix.M1_PRICE_COARSENED_MODE)
            and recipe['parent_order_algorithm']==prefix.PARENT_ORDER_ALGORITHM, 'M1_PRICE_MATCHED_SCOPE')
    for key, expected in {'max_private_updates':4, 'max_model_constructions':4 if bps or weighted else 3,
            'max_forwards':37 if family else 21 if chronological or dense else 6 if bps or weighted else 4,
            'max_backwards':5 if bps or weighted or dense_bps else 4,
            'max_checkpoint_bytes':1073741824, 'max_published_json_bytes':33554432}.items():
        _require(type(recipe[key]) is int and recipe[key] == expected, 'RECIPE_BUDGET:' + key)
    for key, expected in {'training_enabled':True, 'full_epoch_training_allowed':False,
            'test_data_used':False, 'predictive_value_claimed':False, 'qualification_weights_for_main_fit':False}.items():
        _require(recipe[key] is expected, 'RECIPE_AUTHORITY:' + key)
    numerical = recipe['numerical_recipe']
    _require(isinstance(numerical, Mapping) and numerical.get('schema_version') == 'gx1_native_direction_research_recipe_v1'
        and type(numerical.get('batch_size')) is int and numerical['batch_size'] == (16 if b16 else 2)
        and numerical.get('signed_bps_supervision',False) is bps
        and numerical.get('magnitude_weighted_direction',False) is weighted
        and numerical.get('dense_minute_supervision',False) is dense
        and numerical.get('dense_signed_bps_supervision', False) is dense_bps
        and type(numerical.get('max_updates')) is int and numerical['max_updates'] == 2
        and numerical.get('genuine_updates_authorized') is False and numerical.get('native_launch_authorized') is False,
        'NUMERICAL_RECIPE_ROLE')
    # This verifies the declared270 seal only. Its exact source-owned numerical
    # values are checked by the270 validator inside the canonical CUDA runner.
    unsigned = {k:v for k,v in numerical.items() if k != 'contract_sha256'}
    declared = hashlib.sha256(json.dumps(unsigned,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()).hexdigest()
    _require(numerical.get('contract_sha256') == declared, 'NUMERICAL_RECIPE_SEAL')
    campaign._sha(numerical.get('research_objective_sha256'), 'research objective')
    campaign.require_binding(recipe['input_manifest'], label='research input metadata', verify_file=True)
    _require(_path(recipe['execution_binding_path'], 'execution binding')
        == str(Path(recipe['runtime_root']) / 'EXECUTION_BINDING.json'), 'EXECUTION_BINDING_PATH')
    if bps:
        _require(recipe['fold'] == BPS_FOLD and recipe['gc_layout_metadata_parity'] == BPS_GC_PARITY
            and recipe['qualification_design'] == BPS_DESIGN, 'BPS_FIXED_INPUT_SCOPE')
        campaign.require_binding(recipe['gc_layout_metadata_parity'],label='bps GC metadata proof',verify_file=True)
        campaign.require_binding(recipe['qualification_design'],label='bps qualification design',verify_file=True)
    if weighted:
        _require(recipe['fold'] == BPS_FOLD and recipe['gc_layout_metadata_parity'] == BPS_GC_PARITY
            and recipe['qualification_design'] == WEIGHT_DESIGN, 'WEIGHT_FIXED_INPUT_SCOPE')
        campaign.require_binding(recipe['gc_layout_metadata_parity'],label='weight GC metadata proof',verify_file=True)
        campaign.require_binding(recipe['qualification_design'],label='weight qualification design',verify_file=True)
    if chronological:
        from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as prefix
        _require(recipe['fold'] == BPS_FOLD and recipe['gc_layout_metadata_parity'] == BPS_GC_PARITY
            and recipe['qualification_design'] == (ORDER_SUPPORTED_DESIGN if supported else ORDER_DESIGN), 'ORDER_FIXED_INPUT_SCOPE')
        campaign.require_binding(recipe['gc_layout_metadata_parity'],label='order GC metadata proof',verify_file=True)
        campaign.require_binding(recipe['qualification_design'],label='order qualification design',verify_file=True)
        refs = {key:campaign.require_binding(item,label='order original '+key,verify_file=True)
            for key,item in prefix.ORDER_ORIGIN.items()}
        original = campaign.read_bound_json(Path(refs['launch_recipe']['path']),refs['launch_recipe']['sha256'])
        initial = campaign.read_bound_json(Path(refs['initial_measurement']['path']),refs['initial_measurement']['sha256'])
        _require(original.get('fold') == recipe['fold'] and original.get('input_manifest') == recipe['input_manifest']
            and original.get('train_passes') == 1 and original.get('run_scope') == 'single_prefix_one_pass_direction_fit'
            and {k:v for k,v in original.get('numerical_recipe',{}).items() if k not in ('max_updates','contract_sha256')}
                == {k:v for k,v in numerical.items() if k not in ('max_updates','contract_sha256')}
            and initial.get('role') == 'INITIAL_TRAIN' and initial.get('fold') == recipe['fold']
            and initial.get('predictions') == refs['initial_predictions'], 'ORDER_ORIGINAL_IDENTITY')
    if dense:
        _require(recipe['fold']==BPS_FOLD and recipe['input_manifest']==DENSE_INPUT_MANIFEST
            and recipe['gc_layout_metadata_parity']==BPS_GC_PARITY
            and recipe['qualification_design']==ORDER_SUPPORTED_DESIGN, 'DENSE_FIXED_INPUT_SCOPE')
        if decimal:
            _require(recipe['decimal_input_audit']==DECIMAL_INPUT_AUDIT
                and (candidate or recipe['decimal_initial_predictions']==DECIMAL_INITIAL_PREDICTIONS), 'DECIMAL_INPUT_AUDIT_SCOPE')
            if not candidate:
                campaign.require_binding(recipe['decimal_initial_predictions'],label='decimal initial predictions',verify_file=True)
            audit_ref=campaign.require_binding(recipe['decimal_input_audit'],label='decimal input audit',verify_file=True)
            audit=campaign.read_bound_json(Path(audit_ref['path']),audit_ref['sha256'])
            refs=recipe['decimal_target_refs']
            _require(audit.get('schema_version')=='gx1_decimal_bound_input_actual_audit_v1'
                and audit.get('status')=='COMPLETE' and audit.get('parents')==118249
                and audit.get('horizons_minutes')==list(range(1,61)) and audit.get('test_data_used') is False
                and audit.get('new_model_forwards')==audit.get('optimizer_steps')==0
                and audit.get('physical_parent_and_bar_start_hashes_verified') is True
                and audit.get('corrected_archive_and_receipt_links_verified') is True
                and audit.get('original_legacy_anchors_verified') is True and audit.get('immutable_outputs') is True
                and isinstance(refs,Mapping) and refs==audit.get('inputs')
                and refs.get('original')==DENSE_TARGET and refs.get('original_result')==DENSE_TARGET_RESULT
                and recipe['dense_target']==refs.get('corrected')
                and recipe['dense_target_result']==refs.get('corrected_result'), 'DECIMAL_INPUT_EVIDENCE')
            for key,item in refs.items():campaign.require_binding(item,label='decimal '+key,verify_file=False)
        else:
            _require(recipe['dense_target_result']==DENSE_TARGET_RESULT and recipe['dense_target']==DENSE_TARGET, 'DENSE_FIXED_INPUT_SCOPE')
        for key in ('gc_layout_metadata_parity','qualification_design','dense_source_transition_proof','dense_target_result'):
            campaign.require_binding(recipe[key],label='dense '+key,verify_file=True)
        campaign.require_binding(recipe['dense_target'],label='dense target',verify_file=False)
        if candidate:
            _require(recipe['dense_source_transition_proof']==M1_PRICE_SOURCE_TRANSITION, 'M1_PRICE_TRANSITION_REF')
            if recipe['candidate_mode'] in ('price_and_m1',prefix.M1_PRICE_COARSENED_MODE):
                plan=campaign.read_bound_json(Path(M1_NORMALIZATION_PLAN['path']),M1_NORMALIZATION_PLAN['sha256'])
                result=campaign.read_bound_json(Path(M1_NORMALIZATION_RESULT['path']),M1_NORMALIZATION_RESULT['sha256'])
                refs=recipe['candidate_m1_refs']
                _require(result.get('status')=='COMPLETE' and result.get('plan')==M1_NORMALIZATION_PLAN
                    and result.get('train_parent_rows')==recipe['fold']['prefix_rows']
                    and result.get('normalization_fit_calls')==1 and result.get('target_values_used') is False
                    and result.get('model_forwards')==result.get('optimizer_steps')==0
                    and result.get('test_data_used') is False
                    and refs=={'source':plan['inputs']['m1_source'],'roll':plan['inputs']['roll'],
                        'normalization':result.get('normalization')}, 'M1_PRICE_NORMALIZATION_EVIDENCE')
                for key,ref in refs.items():
                    campaign.require_binding(ref,label='candidate M1 '+key,verify_file=(key=='normalization'))
                normalized=campaign.read_bound_json(Path(refs['normalization']['path']),refs['normalization']['sha256'])
                _require(normalized.get('schema_version')=='gx1_native_direction_m1_ofi_normalization_v1'
                    and normalized.get('scope')=='observed_closed_minutes_in_strict_train_prefix_only'
                    and normalized.get('contract_sha256')==result.get('normalization_contract_sha256')
                    and normalized.get('prefix_rows')==recipe['fold']['prefix_rows']
                    and normalized.get('cutoff_ns')==recipe['fold']['train_cutoff_ns']
                    and normalized.get('roll_sha256')==refs['roll']['sha256']
                    and normalized.get('model_contract_clock_bound') is True
                    and normalized.get('genuine_fit_authorized') is False
                    and normalized.get('native_launch_authorized') is False, 'M1_PRICE_NORMALIZATION_CONTRACT')
        else:
            _require(recipe['dense_source_transition_proof']['path']==str(source_repo/'docs/DIRECTION_DENSE_SOURCE_TRANSITION_20261028.json'), 'DENSE_PROOF_PATH')
    owners = recipe['source_owners']
    required_sources = REQUIRED_SOURCE_PATHS + (BPS_EXTRA_SOURCE_PATHS if bps or weighted else ORDER_EXTRA_SOURCE_PATHS if chronological else DENSE_EXTRA_SOURCE_PATHS if dense else ())
    if decimal: required_sources += DECIMAL_EXTRA_SOURCE_PATHS
    if candidate: required_sources += M1_PRICE_EXTRA_SOURCE_PATHS
    if dense_bps: required_sources += ('gx1/contracts/entry_native_direction_bps_v1.py',)
    if family: required_sources += FAMILY_EXTRA_SOURCE_PATHS
    if priority:
        from gx1.contracts.entry_native_direction_five_priority_private_v1 import EXTRA_SOURCE_PATHS
        required_sources += EXTRA_SOURCE_PATHS
    _require(isinstance(owners, Mapping) and set(owners) == set(required_sources), 'SOURCE_OWNER_SET')
    for key, item in owners.items():
        checked = campaign.require_binding(item, label='research source', verify_file=True)
        _require(checked['path'] == str(source_repo / key), 'SOURCE_OWNER_PATH:' + key)
    if decimal:
        _require(audit.get('source')==owners['gx1/contracts/entry_native_direction_decimal_input_v1.py'], 'DECIMAL_QUALIFIED_OWNER')
    if candidate:
        proof=campaign.read_bound_json(Path(M1_PRICE_SOURCE_TRANSITION['path']),M1_PRICE_SOURCE_TRANSITION['sha256'])
        manifest=campaign.read_bound_json(Path(recipe['input_manifest']['path']),recipe['input_manifest']['sha256'])
        names={'owner_profile_source':'gx1/features/entry_specialist_feature_groups_v1.py',
            'native_constructor_source':'gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py',
            'direction_component_source':'gx1/contracts/entry_native_direction_research_v1.py'}
        transitions=proof.get('source_transitions')
        _require(proof.get('schema_version')=='gx1_dense_source_transition_proof_v1'
            and proof.get('status')=='SOURCE_ONLY_VERIFIED_NO_DATA_OR_TRAINING'
            and proof.get('test_data_used') is False and proof.get('genuine_training_authorized') is False
            and proof.get('input_manifest')==recipe['input_manifest']
            and isinstance(transitions,Mapping) and set(transitions)==set(names), 'M1_PRICE_SOURCE_PROOF')
        for key,name in names.items():
            item=transitions[key]
            _require(isinstance(item,Mapping) and set(item)=={'original','current'}
                and item['original']==manifest['view_source_bindings'][key] and item['current']==owners[name]
                and item['original']['path']==item['current']['path'], 'M1_PRICE_SOURCE_TRANSITION:'+key)
    if family:
        _require(recipe['candidate_mode']=='price_only' and not priority and not dense_bps
            and type(recipe['family_arm']) is str and bool(recipe['family_arm'])
            and recipe['admission_stage'].startswith('dynamic_m5_direction_family_private_'),
            'FAMILY_PRIVATE_SCOPE')
        campaign._sha(recipe['family_intervention_sha256'], 'family intervention')
        _require(recipe['family_baseline_initial_predictions']==M5_DENSE_BPS_BASELINE_INITIAL
            and recipe['family_baseline_initial_report']==M5_DENSE_BPS_BASELINE_REPORT,
            'FAMILY_BASELINE_REFS')
        for key in ('family_baseline_initial_predictions','family_baseline_initial_report','family_protocol'):
            campaign.require_binding(recipe[key], label='family private '+key, verify_file=True)
        protocol=campaign.read_bound_json(Path(recipe['family_protocol']['path']),recipe['family_protocol']['sha256'])
        _require(recipe['family_protocol']['path']==FAMILY_PROTOCOL_PATH
            and protocol.get('schema_version')=='gx1_current_family_added_value_study_v1'
            and protocol.get('fixed_fold')==recipe['fold'] and protocol.get('targets')==recipe['dense_target']
            and protocol.get('horizons_minutes')==list(range(1,61))
            and protocol.get('test_data_used') is False and protocol.get('exit_evaluation') is False
            and protocol.get('broker_access_allowed') is False, 'FAMILY_PROTOCOL')
        expected=protocol.get('comparison',{}).get('fixed_training',{}).get('numerical_recipe')
        _require(isinstance(expected,Mapping)
            and {k:v for k,v in expected.items() if k not in ('max_updates','contract_sha256')}
                =={k:v for k,v in numerical.items() if k not in ('max_updates','contract_sha256')},
            'FAMILY_MATCHED_NUMERICS')
    if dense_bps:
        _require(recipe['dense_bps_scale']==M5_DENSE_BPS_SCALE and recipe['dense_bps_scale_result']==M5_DENSE_BPS_SCALE_RESULT
            and recipe['dense_bps_scale_terminal']==M5_DENSE_BPS_SCALE_TERMINAL
            and recipe['dense_bps_count_review']==M5_DENSE_BPS_COUNT_REVIEW
            and recipe['dense_bps_initial_predictions']==M5_DENSE_BPS_BASELINE_INITIAL
            and recipe['dense_bps_initial_report']==M5_DENSE_BPS_BASELINE_REPORT, 'DENSE_BPS_EVIDENCE_REFS')
        scale_ref=campaign.require_binding(recipe['dense_bps_scale'],label='dense bps scale',verify_file=True)
        result_ref=campaign.require_binding(recipe['dense_bps_scale_result'],label='dense bps scale result',verify_file=True)
        terminal_ref=campaign.require_binding(recipe['dense_bps_scale_terminal'],label='dense bps scale terminal',verify_file=True)
        review_ref=campaign.require_binding(recipe['dense_bps_count_review'],label='dense bps count review',verify_file=True)
        initial_ref=campaign.require_binding(recipe['dense_bps_initial_report'],label='dense bps initial report',verify_file=True)
        campaign.require_binding(recipe['dense_bps_initial_predictions'],label='dense bps initial predictions',verify_file=True)
        scale=campaign.read_bound_json(Path(scale_ref['path']),scale_ref['sha256'])
        sr=campaign.read_bound_json(Path(result_ref['path']),result_ref['sha256'])
        terminal=campaign.read_bound_json(Path(terminal_ref['path']),terminal_ref['sha256'])
        count_review=campaign.read_bound_json(Path(review_ref['path']),review_ref['sha256'])
        initial=campaign.read_bound_json(Path(initial_ref['path']),initial_ref['sha256'])
        sealed={k:v for k,v in scale.items() if k!='contract_sha256'}
        _require(scale.get('contract_sha256')==hashlib.sha256(json.dumps(sealed,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
            and scale.get('schema_version')=='gx1_direction_own_prefix_dense_signed_bps_scale_v1'
            and scale.get('fit_partition')=='train' and scale.get('horizons_minutes')==list(range(1,61))
            and scale.get('prefix_rows')==recipe['fold']['prefix_rows'] and scale.get('cutoff_ns')==recipe['fold']['train_cutoff_ns']
            and scale.get('target_source')==recipe['dense_target'], 'DENSE_BPS_SCALE_IDENTITY')
        _require(sr.get('status')=='COMPLETE_TRAIN_PREFIX_SCALE_ONLY' and sr.get('scale')==scale_ref
            and sr.get('model_forwards')==sr.get('optimizer_steps')==0 and sr.get('test_data_used') is False
            and sr.get('independent_counts_and_fsum_with_error_bounds') is True
            and terminal.get('status')=='COMPLETE' and terminal.get('result')==result_ref and terminal.get('scale')==scale_ref
            and count_review.get('status')=='PASS' and count_review.get('scale')==scale_ref
            and count_review.get('prior_decimal_sign_audit')==recipe['decimal_target_refs']['audit_result']
            and count_review.get('all_mature_observed_counts_equal') is True and count_review.get('horizons_checked')==60
            and count_review.get('test_data_used') is False, 'DENSE_BPS_SCALE_PROOF')
        _require(initial.get('role')=='INITIAL_TRAIN' and initial.get('fold')==recipe['fold']
            and initial.get('test_data_used') is False and initial.get('predictions',{}).get('sha256')==M5_DENSE_BPS_BASELINE_INITIAL['sha256']
            and all(type(initial.get('preservation',{}).get(k)) is str for k in ('model','all_buffers','rng')),
            'DENSE_BPS_INITIAL_REPORT')
    if priority:
        from gx1.contracts.entry_native_direction_five_priority_private_v1 import require_private_recipe
        require_private_recipe(recipe, owners, source_repo=source_repo, source_commit=source_commit)
    return recipe


def require_family_qualification(recipe, owners, *, source_repo, source_commit):
    """Bind a real same-arm private result; no historical qualification relabel."""
    def read(ref, label):
        checked=campaign.require_binding(ref,label=label,verify_file=True)
        return campaign.read_bound_json(Path(checked['path']),checked['sha256'])
    review=read(recipe['native_dense_qualification'],'family private review')
    _require(review.get('schema_version')=='gx1_m5_family_private_root_review_v1'
        and review.get('status')=='PASS_CONSUMED_FAMILY_PRIVATE_NATIVE_MECHANICS'
        and review.get('family_arm')==recipe.get('family_arm')
        and review.get('family_intervention_sha256')==recipe.get('family_intervention_sha256')
        and review.get('initial_predictions')==recipe.get('decimal_initial_predictions')
        and review.get('initial_report')==recipe.get('family_initial_report')
        and review.get('initial_parent_count')==256
        and review.get('private_optimizer_updates')==4
        and review.get('unperturbed_initial_prediction_shared_state_parity') is True
        and review.get('bit_exact_resume_reported_and_bound') is True
        and review.get('actual_input_intervention_checked') is True
        and review.get('qualification_weights_for_main_fit') is False
        and review.get('test_data_used') is False,'FAMILY_PRIVATE_REVIEW')
    plan=read(review['plan'],'family private plan')
    prior_ref=plan['recipe'];prior=read(prior_ref,'family private recipe')
    result=read(review['result'],'family private result')
    receipt=read(review['receipt'],'family private receipt')
    power=read(review['power_after'],'family private retained power')
    own=read(recipe['family_initial_report'],'family private own initial')
    campaign.require_binding(recipe['decimal_initial_predictions'],label='family private initial arrays',verify_file=True)
    _require(plan.get('source_commit')==review.get('source_commit')
        and receipt.get('plan_sha256')==plan.get('plan_sha256')
        and receipt.get('outcome')=='COMPLETE' and receipt.get('guard_decision')=='PASS'
        and receipt.get('trainer_guard_exit_code')==receipt.get('progress_observer_exit_code')==0
        and receipt.get('test_data_used') is False,'FAMILY_PRIVATE_TERMINAL')
    gpu=power.get('gpu')
    _require(isinstance(gpu,list) and len(gpu)==1 and isinstance(gpu[0],str),
        'FAMILY_PRIVATE_POWER')
    gpu_fields=[value.strip() for value in gpu[0].split(',')]
    try:
        actual_power=float(gpu_fields[1])
    except (IndexError,TypeError,ValueError):
        actual_power=None
    keepers=power.get('keeper')
    _require(power.get('task_state')=='Disabled'
        and power.get('configured_power_limit_w')==actual_power==250
        and len(gpu_fields)==6 and isinstance(plan.get('gpu_uuid'),str)
        and gpu_fields[0]==plan['gpu_uuid']
        and isinstance(keepers,list) and len(keepers)==2
        and all(isinstance(item,Mapping) and type(item.get('State')) is int
            and item['State']==4 for item in keepers)
        and {item.get('TaskName') for item in keepers}=={'GX1GpuPowerLimit','GX1HostTelemetryBridge'}
        and power.get('test_data_used') is False,
        'FAMILY_PRIVATE_POWER')
    _require(result.get('schema_version')=='gx1_native_m5_family_private_qualification_result_v1'
        and result.get('decision')=='PASS_M5_FAMILY_PRIVATE_MECHANICS'
        and result.get('recipe')==prior_ref
        and result.get('family_arm')==recipe['family_arm']
        and result.get('family_intervention_sha256')==recipe['family_intervention_sha256']
        and result.get('initial_predictions')==recipe['decimal_initial_predictions']
        and result.get('family_initial_report')==recipe['family_initial_report']
        and result.get('unperturbed_initial_prediction_shared_state_parity') is True
        and result.get('actual_input_intervention_checked') is True
        and result.get('decimal_prior_against_sign_audit') is True
        and result.get('bit_exact_resume') is True and result.get('physical_optimizer_updates')==4
        and result.get('source_unchanged') is True and result.get('test_data_used') is False
        and result.get('private_weights_for_main_fit') is False,'FAMILY_PRIVATE_RESULT')
    _require(own.get('schema_version')=='gx1_native_m5_family_initial_measurement_v1'
        and own.get('role')=='INITIAL_TRAIN' and own.get('fold')==recipe['fold']
        and own.get('family_arm')==recipe['family_arm']
        and own.get('family_intervention_sha256')==recipe['family_intervention_sha256']
        and own.get('predictions')==recipe['decimal_initial_predictions']
        and own.get('test_data_used') is False,'FAMILY_PRIVATE_INITIAL')
    keys=('input_manifest','fold','dense_target','dense_target_result','dense_source_transition_proof',
        'decimal_input_audit','decimal_target_refs','candidate_mode','parent_order_algorithm',
        'family_arm','family_intervention_sha256')
    _require(prior.get('run_scope')==FAMILY_PRIVATE_SCOPE
        and prior.get('candidate_mode')=='price_only'
        and all(prior.get(k)==recipe.get(k) for k in keys)
        and prior.get('qualification_weights_for_main_fit') is False
        and {k:v for k,v in prior.get('numerical_recipe',{}).items() if k not in ('max_updates','contract_sha256')}
            =={k:v for k,v in recipe['numerical_recipe'].items() if k not in ('max_updates','contract_sha256')},
        'FAMILY_PRIVATE_RECIPE')
    _require(set(prior['source_owners'])<=set(owners)
        and all(prior['source_owners'][k]==owners[k] for k in prior['source_owners']),
        'FAMILY_PRIVATE_UNCHANGED_SOURCES')
    return review


def guarded_research_power_limit(recipe):
    """Return the user-approved cap after the caller validates the bound recipe."""
    from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as prefix
    _require(isinstance(recipe, Mapping), 'POWER_RECIPE')
    if recipe.get('run_scope') == FAMILY_PRIVATE_SCOPE:
        _require(recipe.get('admission_stage','').startswith('dynamic_m5_direction_family_private_')
            and recipe.get('candidate_mode')=='price_only'
            and recipe.get('numerical_recipe',{}).get('dense_minute_supervision') is True
            and recipe.get('numerical_recipe',{}).get('dense_signed_bps_supervision',False) is False, 'FAMILY_POWER_SCOPE')
        return 250
    if recipe.get('run_scope') == prefix.M5_DENSE_BPS_PRIVATE_SCOPE:
        _require(recipe.get('admission_stage') == M5_DENSE_BPS_STAGE
            and recipe.get('candidate_mode') == 'price_only'
            and recipe.get('numerical_recipe', {}).get('dense_signed_bps_supervision') is True, 'DENSE_BPS_POWER_SCOPE')
        return 250
    _require(recipe.get('admission_stage') != M5_DENSE_BPS_STAGE, 'DENSE_BPS_POWER_SCOPE')
    return 300


def require_m1_structure_matched_recipes(coarse_ref, ordered_ref, *, source_repo: Path, source_commit: str):
    """Preflight both private recipes; no model, data payload or authority."""
    from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as prefix
    coarse=require_research_launch_recipe(coarse_ref,source_repo=source_repo,source_commit=source_commit)
    ordered=require_research_launch_recipe(ordered_ref,source_repo=source_repo,source_commit=source_commit)
    varying={'admission_stage','runtime_root','execution_binding_path','candidate_mode','recipe_sha256'}
    _require(coarse['run_scope']==ordered['run_scope']==prefix.M1_PRICE_PRIVATE_ADAPTER_SCOPE
        and coarse['candidate_mode']==prefix.M1_PRICE_COARSENED_MODE and ordered['candidate_mode']=='price_and_m1'
        and {k:v for k,v in coarse.items() if k not in varying}=={k:v for k,v in ordered.items() if k not in varying}
        and coarse['admission_stage']!=ordered['admission_stage'] and coarse['runtime_root']!=ordered['runtime_root']
        and coarse['execution_binding_path']!=ordered['execution_binding_path'] and coarse_ref!=ordered_ref,
        'M1_PRICE_MATCHED_PAIR')
    return {'coarsened':coarse,'ordered':ordered}


def require_research_window_policy(value: Any, *, verify_files: bool = True) -> dict[str, Any]:
    fields = {'schema_version', 'recipe', 'invocation_number', 'max_invocation_seconds',
        'progress_path', 'campaign_cursor_path', 'output_directory', 'test_data_used'}
    result = _identity(value, fields, WINDOW_SCHEMA, 'policy_sha256')
    _require(type(result['invocation_number']) is int and result['invocation_number'] == 1
        and type(result['max_invocation_seconds']) is int and result['max_invocation_seconds'] == 12000
        and result['test_data_used'] is False, 'WINDOW_SCOPE')
    campaign.require_binding(result['recipe'], label='research window recipe', verify_file=verify_files)
    paths = [_path(result[k], k) for k in ('progress_path', 'campaign_cursor_path', 'output_directory')]
    _require(len(set(paths)) == 3 and Path(paths[0]).parent != Path(paths[1]).parent, 'WINDOW_PATH_COLLISION')
    return result


def require_research_invocation_execution(result: dict[str, Any], execution: Mapping[str, Any], *,
        source_repo: Path, source_commit: str) -> None:
    recipe_ref = campaign.require_binding(execution['prelaunch_manifest'], label='research recipe')
    recipe = require_research_launch_recipe(recipe_ref, source_repo=source_repo, source_commit=source_commit)
    policy_ref = campaign.require_binding(execution['train_session_manifest'], label='research window')
    window = require_research_window_policy(campaign.read_bound_json(Path(policy_ref['path']),policy_ref['sha256']))
    _require(result['requires_fresh_windows_boot'] is True or recipe['run_scope'] == FAMILY_PRIVATE_SCOPE, 'FAMILY_ONLY_BOOT_MODE')
    separator = result['launcher_argv'].index('--')
    _require(result['launcher_argv'][separator+1:] == [str(source_repo/'.venv/bin/python'),'-m',MODULE,
        '--window-policy',policy_ref['path'],'--window-policy-file-sha256',policy_ref['sha256'],
        '--progress-path',window['progress_path']], 'INVOCATION_ARGV')
    _require(result['kind'] == KIND and result['batch_size'] == recipe['numerical_recipe']['batch_size'] and result['epoch_index'] is None
        and result['invocation_number'] == 1 and result['window_index'] == 0 and result['window_count'] == 1
        and type(result['optimizer_step_budget']) is int and result['optimizer_step_budget'] == 4
        and result['expected_success_outcome'] == 'COMPLETE' and result['maximum_wall_seconds'] == 13800
        and result['checkpoint'] == {'pointer_path':window['campaign_cursor_path'],'before_mode':'GENESIS',
            'predecessor_invocation_number':None,'write_mode':'ATOMIC_UPDATE'}
        and result['progress_path'] == window['progress_path']
        and execution['prelaunch_manifest_sha256'] == recipe['recipe_sha256']
        and execution['train_session_manifest_sha256'] == window['policy_sha256']
        and window['recipe'] == recipe_ref, 'INVOCATION_BINDING')
    result['direction_research_recipe'] = recipe_ref
    result['direction_research_window'] = window


def require_research_plan(value: Any, *, verify_files: bool = True) -> dict[str, Any]:
    fields = {'schema_version','decision','campaign_id','phase','created_utc','source_repo','source_commit',
        'runtime_root','gpu_uuid','prepared_windows_boot','recipe','invocations','signed_guard_sources',
        'controller_sources','policy','authority','test_data_used'}
    if isinstance(value, Mapping) and 'family_same_boot' in value:
        fields = fields | {'family_same_boot'}
    result = _identity(value, fields, campaign.PLAN_SCHEMA, 'plan_sha256')
    _require(result['phase'] == PHASE and result['decision'] == 'PASS_PREPARED'
        and type(result['campaign_id']) is str and bool(result['campaign_id'])
        and result['source_repo'] == str(CANONICAL_REPO)
        and type(result['source_commit']) is str and campaign._COMMIT.fullmatch(result['source_commit']) is not None
        and type(result['gpu_uuid']) is str and result['gpu_uuid'].startswith('GPU-')
        and result['test_data_used'] is False, 'PLAN_IDENTITY')
    campaign._utc(result['created_utc'], 'research plan creation')
    runtime = Path(_path(result['runtime_root'], 'runtime root'))
    result['prepared_windows_boot'] = campaign.require_boot_identity(result['prepared_windows_boot'])
    _require(campaign.canonical_sha256(result['policy']) == campaign.canonical_sha256(campaign.family_campaign_policy(result))
        and campaign.canonical_sha256(result['authority']) == campaign.canonical_sha256({'test':False,'promotion':False,'paper':False,'live':False,'cloud_spend':False}), 'PLAN_AUTHORITY')
    for key, keys in (('signed_guard_sources',{'runner','guard','query','certificate'}),
            ('controller_sources',{'controller','observer','campaign_cli'})):
        _require(isinstance(result[key],Mapping) and set(result[key]) == keys, 'PLAN_SOURCES:' + key)
        result[key] = {name:campaign.require_binding(item,label=key,verify_file=verify_files) for name,item in result[key].items()}
    result['recipe'] = campaign.require_binding(result['recipe'],label='research recipe',verify_file=verify_files)
    _require(isinstance(result['invocations'],list) and len(result['invocations']) == 1, 'PLAN_ONE_INVOCATION')
    result['invocations'] = [campaign.require_binding(result['invocations'][0],label='research invocation',verify_file=verify_files)]
    # Explicit absence of a selection artefact; required by the shared receipt transport.
    result['selection_artifact_sha256'] = None
    if verify_files:
        recipe = require_research_launch_recipe(result['recipe'],source_repo=CANONICAL_REPO,source_commit=result['source_commit'])
        _require(recipe['runtime_root'] == str(runtime), 'PLAN_RECIPE_RUNTIME')
        ref = result['invocations'][0]
        invocation = campaign.require_invocation(campaign.read_bound_json(Path(ref['path']),ref['sha256']),
            source_repo=CANONICAL_REPO,source_commit=result['source_commit'],verify_files=True)
        window = invocation.get('direction_research_window',{})
        _require(invocation['kind'] == KIND and invocation.get('direction_research_recipe') == result['recipe']
            and window.get('output_directory') == str(runtime/'private-research')
            and window.get('campaign_cursor_path') == str(runtime/'research-cursor'/'RESUME_CURSOR.json')
            and window.get('progress_path') == str(runtime/'progress'/'invocation-0001.json')
            and invocation['guard_log_path'] == str(runtime/'logs'/'invocation-0001.signed-guard.log'), 'PLAN_LAYOUT')
        campaign.require_family_plan_mode(result, recipe, [invocation])
        result['checked_invocations'] = [invocation]
    return result


def require_native_direction_research_readiness(repo: Path, *, window_path: Path,
        window_file_sha256: str) -> dict[str, Any]:
    """An exact bound qualification exception; unbound readiness remains closed."""
    window = require_research_window_policy(campaign.read_bound_json(window_path,window_file_sha256))
    policy_path = repo/'NEXT_RUN_POLICY.json'
    policy = campaign.read_bound_json(policy_path,campaign.file_sha256(policy_path))
    import subprocess
    head = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    branch = subprocess.check_output(['git','-C',str(repo),'branch','--show-current'],text=True).strip()
    dirty = subprocess.check_output(['git','-C',str(repo),'status','--porcelain=v1','--untracked-files=all'],text=True)
    recipe = require_research_launch_recipe(window['recipe'],source_repo=repo,source_commit=head)
    gate = policy.get(recipe['admission_stage'])
    _require(repo == CANONICAL_REPO and branch == 'work/gx1-current' and not dirty
        and policy.get('schema_version') == 'gx1_next_native_run_policy_v1'
        and policy.get('canonical_source_repo') == str(repo) and policy.get('canonical_branch') == branch
        and policy.get('training_module') == MODULE and policy.get('training_enabled') is True
        and policy.get('native_training_enabled') is True and policy.get('full_epoch_training_allowed') is False,
        'READINESS_GLOBAL')
    _require(isinstance(gate,Mapping) and gate.get('recipe') == window['recipe']
        and gate.get('run_root') == recipe['runtime_root'] and type(gate.get('max_invocations')) is int and gate['max_invocations'] == 1
        and type(gate.get('claimed_invocations')) is int and gate['claimed_invocations'] == 0
        and type(gate.get('completed_invocations')) is int and gate['completed_invocations'] == 0
        and gate.get('relaunch_allowed') is False, 'READINESS_ONCE')
    for key in ('execution_authorized','data_reads_authorized','genuine_constructor_authorized',
            'genuine_forward_authorized','genuine_backward_authorized','genuine_updates_authorized',
            'training_enabled','native_training_enabled'):
        _require(gate.get(key) is True, 'READINESS_GATE:' + key)
    for key in ('genuine_predictor_fit_authorized','native_launch_authorized','normalization_fit_authorized',
            'synthetic_test_execution_authorized','test_data_used','exit_evaluation','broker_access_allowed'):
        _require(gate.get(key) is False, 'READINESS_CLOSED_GATE:' + key)
    if recipe['run_scope'] == BPS_SCOPE:
        _require(gate.get('signed_bps_scale_fit_authorized') is True,'READINESS_BPS_SCALE')
    if recipe['run_scope'] == WEIGHT_SCOPE:
        _require(gate.get('magnitude_weight_scale_fit_authorized') is True,'READINESS_WEIGHT_SCALE')
    clock = campaign.require_binding(policy.get('gpu_clock_launcher'),label='existing GPU clock launcher')
    _require(clock == policy['gpu_clock_launcher'], 'READINESS_CLOCK')
    return {'decision':'READY_FOR_EXISTING_BOUND_CAMPAIGN_GATES','blocked_reasons':[],
        'canonical_source_repo':str(repo),'source_commit':head,'policy_sha256':campaign.file_sha256(policy_path),
        'training_started':False,'native_invocation_seconds':12000,'outer_guard_seconds':13800,
        'native_window_scope':{'research_only':True,'invocation_number':1,'optimizer_step_ceiling':4,
            'full_epoch_training_allowed':False,'window_policy_file_sha256':window_file_sha256}}


def require_native_direction_research_context(policy_path: Path, policy_file_sha256: str,
        progress_path: Path) -> tuple[dict[str,Any],dict[str,Any],dict[str,Any],dict[str,Any],dict[str,Any]]:
    with campaign.file_verification_scope():
        window = require_research_window_policy(campaign.read_bound_json(policy_path,policy_file_sha256))
        names = ('GX1_CAMPAIGN_PLAN_PATH','GX1_CAMPAIGN_PLAN_FILE_SHA256','GX1_CAMPAIGN_PLAN_SHA256','GX1_CAMPAIGN_INVOCATION_SHA256')
        _require(all(os.environ.get(name) for name in names), 'CAMPAIGN_ENV')
        plan_ref = {'path':os.environ[names[0]],'sha256':os.environ[names[1]]}
        plan = campaign.require_plan(campaign.read_bound_json(Path(plan_ref['path']),plan_ref['sha256']))
        invocation = plan['checked_invocations'][0]
        recipe = require_research_launch_recipe(window['recipe'],source_repo=CANONICAL_REPO,source_commit=plan['source_commit'])
        marker_path = Path(plan['runtime_root'])/'ACTIVE_INVOCATION.json'
        marker = campaign.read_bound_json(marker_path,campaign.file_sha256(marker_path))
        _require(plan['phase'] == PHASE and plan['plan_sha256'] == os.environ[names[2]]
            and invocation['invocation_sha256'] == os.environ[names[3]]
            and invocation['direction_research_window'] == window and plan['recipe'] == window['recipe']
            and str(progress_path) == window['progress_path'], 'WINDOW_CONTEXT')
        _require(marker.get('schema_version') == campaign.ACTIVE_SCHEMA
            and marker.get('marker_sha256') == campaign.canonical_sha256({k:v for k,v in marker.items() if k != 'marker_sha256'})
            and all(marker.get(key) == expected for key,expected in {
                'plan_sha256':plan['plan_sha256'],'invocation_sha256':invocation['invocation_sha256'],
                'invocation_number':1,'invocation_id':'invocation-0001','kind':KIND,
                'launcher_argv_sha256':invocation['launcher_argv_sha256'],'pointer_before_sha256':'GENESIS'}.items()), 'ACTIVE_MARKER')
        if 'family_same_boot' in plan:
            campaign.require_family_idle_admission(marker.get('idle_admission'), plan=plan,
                invocation=invocation, previous=None, current_boot=marker['boot'],
                at_utc=marker['started_utc'], expected_plan_binding=plan_ref)
            policy = campaign.read_bound_json(CANONICAL_REPO/'NEXT_RUN_POLICY.json', campaign.file_sha256(CANONICAL_REPO/'NEXT_RUN_POLICY.json'))
            _require(policy.get(recipe['admission_stage'], {}).get('same_boot_after_idle_review_authorized') is True, 'FAMILY_BOOT_READINESS')
        binding_path = Path(recipe['execution_binding_path'])
        binding = campaign.read_bound_json(binding_path,campaign.file_sha256(binding_path))
        _require(set(binding) == {'plan','source_head','policy'} and binding['plan'] == plan_ref
            and binding['source_head'] == plan['source_commit']
            and binding['policy'] == {'path':str(CANONICAL_REPO/'NEXT_RUN_POLICY.json'),
                'sha256':campaign.file_sha256(CANONICAL_REPO/'NEXT_RUN_POLICY.json')}, 'FROZEN_EXECUTION_BINDING')
        campaign.require_clean_source(plan)
        require_native_direction_research_readiness(CANONICAL_REPO,window_path=policy_path,window_file_sha256=policy_file_sha256)
        _require(not Path(window['output_directory']).exists() and not Path(window['campaign_cursor_path']).parent.exists()
            and not progress_path.exists(), 'GENESIS_NOT_FRESH')
    return window,plan,invocation,marker,recipe


def build_research_cursor(*, recipe_ref, outcome, optimizer_updates, parent_cursor,
        parent_order_sha256, full_state, result, batch_size=2) -> dict[str, Any]:
    value = _seal({'schema_version':BPS_CURSOR_SCHEMA if batch_size == 16 else CURSOR_SCHEMA,'recipe':dict(recipe_ref),'outcome':outcome,
        'optimizer_updates':optimizer_updates,'parent_cursor':parent_cursor,'parent_order_sha256':parent_order_sha256,
        'full_state':full_state,'result':result,'test_data_used':False}, 'cursor_sha256')
    return require_research_cursor(value,expected_recipe=recipe_ref,verify_files=False,batch_size=batch_size)


def require_research_cursor(value, *, expected_recipe, verify_files=True, batch_size=2) -> dict[str, Any]:
    fields = {'schema_version','recipe','outcome','optimizer_updates','parent_cursor',
        'parent_order_sha256','full_state','result','test_data_used'}
    _require(type(batch_size) is int and batch_size in (2,16),'CURSOR_BATCH_SIZE')
    result = _identity(value,fields,BPS_CURSOR_SCHEMA if batch_size == 16 else CURSOR_SCHEMA,'cursor_sha256')
    _require(result['recipe'] == expected_recipe and result['test_data_used'] is False
        and result['outcome'] in {'COMPLETE','FAILED'}, 'CURSOR_ROLE')
    for key in ('optimizer_updates','parent_cursor'):
        _require(type(result[key]) is int and 0 <= result[key] <= (2*batch_size if key=='parent_cursor' else 4), 'CURSOR_COUNTER:' + key)
    campaign._sha(result['parent_order_sha256'],'parent order')
    campaign.require_binding(result['result'],label='research terminal result',verify_file=verify_files)
    if result['full_state'] is not None:
        campaign.require_binding(result['full_state'],label='research full state',verify_file=verify_files)
    _require(result['outcome'] != 'COMPLETE' or (result['optimizer_updates'] == 4 and result['parent_cursor'] == 2*batch_size
        and result['full_state'] is not None), 'CURSOR_COMPLETE')
    return result


def build_research_progress(*, window, plan, invocation, checkpoint_pointer,
        optimizer_updates, parent_cursor, parent_order_sha256, outcome) -> dict[str,Any]:
    from datetime import datetime, timezone
    value = _seal({'schema_version':campaign.PROGRESS_SCHEMA,'plan_sha256':plan['plan_sha256'],
        'invocation_sha256':invocation['invocation_sha256'],'phase':KIND,'recipe':window['recipe'],
        'completed_units':optimizer_updates,'total_units':4,'optimizer_updates':optimizer_updates,
        'parent_cursor':parent_cursor,'parent_order_sha256':parent_order_sha256,
        'checkpoint_pointer':checkpoint_pointer,'terminal':True,'outcome':outcome,
        'observed_utc':datetime.now(timezone.utc).isoformat()},'progress_sha256')
    return require_research_progress(value,plan_sha256=plan['plan_sha256'],invocation=invocation,
        expected_selection_receipt_sha256=None,verify_file=False)


def require_research_progress(value, *, plan_sha256, invocation,
        expected_selection_receipt_sha256, verify_file=True) -> dict[str,Any]:
    fields = {'schema_version','plan_sha256','invocation_sha256','phase','recipe','completed_units','total_units',
        'optimizer_updates','parent_cursor','parent_order_sha256','checkpoint_pointer','terminal','outcome','observed_utc'}
    result = _identity(value,fields,campaign.PROGRESS_SCHEMA,'progress_sha256')
    _require(expected_selection_receipt_sha256 is None and invocation['kind'] == KIND
        and result['plan_sha256'] == plan_sha256 and result['invocation_sha256'] == invocation['invocation_sha256']
        and result['phase'] == KIND and result['recipe'] == invocation['direction_research_recipe']
        and result['terminal'] is True and result['outcome'] in {'COMPLETE','FAILED'}, 'PROGRESS_IDENTITY')
    campaign._utc(result['observed_utc'],'research progress')
    batch_size=invocation['batch_size']
    _require(type(batch_size) is int and batch_size in (2,16),'PROGRESS_BATCH_SIZE')
    for key in ('completed_units','total_units','optimizer_updates','parent_cursor'):
        _require(type(result[key]) is int and 0 <= result[key] <= (2*batch_size if key=='parent_cursor' else 4),'PROGRESS_COUNTER:' + key)
    _require(result['total_units'] == 4 and result['completed_units'] == result['optimizer_updates']
        and (result['outcome'] != 'COMPLETE' or result['optimizer_updates'] == 4 and result['parent_cursor'] == 2*batch_size), 'PROGRESS_COMPLETE')
    campaign._sha(result['parent_order_sha256'],'parent order')
    ref = campaign.require_binding(result['checkpoint_pointer'],label='research cursor',verify_file=verify_file)
    _require(ref['path'] == invocation['checkpoint']['pointer_path'],'PROGRESS_CURSOR_PATH')
    if verify_file:
        cursor = require_research_cursor(campaign.read_bound_json(Path(ref['path']),ref['sha256']),
            expected_recipe=result['recipe'],verify_files=True,batch_size=batch_size)
        _require(all(cursor[k] == result[k] for k in ('outcome','optimizer_updates','parent_cursor','parent_order_sha256')),
            'PROGRESS_DURABLE_CURSOR')
    return result
