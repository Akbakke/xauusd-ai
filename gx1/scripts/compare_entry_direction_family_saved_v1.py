"""One-shot, capped authority for the ten-arm saved TRAIN/Q1 comparison.

No model, checkpoint or raw input is opened. Historical evidence is consumed
unchanged; statistical work belongs to compare_family_value_predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys
import time
import zipfile
import xml.etree.ElementTree as ET

import numpy as np

from gx1.scripts import compare_entry_direction_matched_exposure_v1 as matched

REPO = matched.REPO
STUDY = Path('/home/andre2/GX1_RUNS/DIRECTION_FAMILY_ADDED_VALUE_20261030_01a18c19')
STAGE = 'direction_family_saved_comparison_20261101_001'
PROTOCOL = {'path': str(REPO/'docs/DIRECTION_FAMILY_ADDED_VALUE_PROTOCOL_20261030.json'),
            'sha256': '151f633dd401b950741f48e02faa62177725d47d9eaf8d1c4d08d28a44d24446'}
AXIS_SOURCE = {'path': str(STUDY/'INPUT_FULL_001/RESULT.json'),
               'sha256': '7361baad30ea2f0cc21699952b6336a49e701a6b64c86d1401d2828bdfc83c64'}
ARMS = ('baseline', 'structure_swing_encoder', 'smc_liquidity_encoder',
        'trend_ema_encoder', 'vol_compression_encoder', 'momentum_flow_encoder',
        'session_regime_encoder', 'chart_geometry_encoder',
        'price_action_candle_encoder', 'macro_context')
ROWS = {'train': 47602, 'later': 17288}
SOURCE_PATHS = matched.SOURCE_PATHS + (
    'gx1/scripts/compare_entry_direction_family_saved_v1.py', 'scripts/gx1_capped_run.sh')
TEST_PATHS = ('tests/test_compare_entry_direction_family_saved_v1.py',
              'tests/test_direction_family_saved_comparison_v1.py')
MAX_BYTES = 64 * 1024 * 1024
NAMES = {'parent_rows', 'bar_start_ns', 'decision_ns', 'terminal_time_ns',
         'observed', 'sign', 'logit_up'} | {'context__'+n for n, _ in matched.CONTEXT_SPEC}


def require(ok, reason):
    if not ok:
        raise RuntimeError('FAMILY_SAVED_' + reason)


def ref_path(ref, *, suffix):
    require(isinstance(ref, dict) and set(ref) == {'path', 'sha256'}, 'REF')
    path = Path(ref['path'])
    require(path.is_absolute() and path.suffix == suffix
            and '..' not in path.parts and str(path) == ref['path'], 'REF_PATH')
    require(isinstance(ref['sha256'], str) and len(ref['sha256']) == 64
            and all(c in '0123456789abcdef' for c in ref['sha256']), 'REF_HASH')
    return path


def identity(path):
    st = path.lstat()
    require(stat.S_ISREG(st.st_mode) and path.resolve() == path, 'REGULAR_NO_SYMLINK')
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def read_bytes(ref):
    path = ref_path(ref, suffix=Path(ref['path']).suffix)
    before = identity(path)
    require(0 < before[2] <= MAX_BYTES, 'FILE_SIZE')
    with path.open('rb') as reader:
        st = os.fstat(reader.fileno())
        require((st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns) == before, 'OPEN_IDENTITY')
        raw = reader.read(MAX_BYTES + 1)
    require(identity(path) == before and len(raw) == before[2]
            and hashlib.sha256(raw).hexdigest() == ref['sha256'], 'HASH_OR_FREEZE')
    return raw


def read_metadata(ref, tracked):
    path = ref_path(ref, suffix='.json')
    require(path.is_relative_to(STUDY) or path.is_relative_to(REPO/'docs')
            or path == REPO/'NEXT_RUN_POLICY.json', 'METADATA_ROOT')
    value = json.loads(read_bytes(ref))
    require(isinstance(value, dict), 'METADATA_OBJECT')
    if str(path) in tracked:
        require(tracked[str(path)][0] == ref, 'CONFLICTING_BINDING')
    tracked[str(path)] = (dict(ref), identity(path))
    return value


def require_plan(plan):
    fields = {'schema_version', 'stage', 'source_head', 'protocol', 'canonical_axis_source',
              'sources', 'arms', 'output_dir', 'test_data_used'}
    require(set(plan) == fields and plan['schema_version'] == 'gx1_family_saved_comparison_plan_v1'
            and plan['stage'] == STAGE and plan['test_data_used'] is False, 'PLAN')
    require(plan['protocol'] == PROTOCOL and plan['canonical_axis_source'] == AXIS_SOURCE, 'ANCHORS')
    require(isinstance(plan['arms'], dict) and set(plan['arms']) == set(ARMS), 'ALL_TEN_ARMS')
    for arm in ARMS:
        entry = plan['arms'][arm]
        require(isinstance(entry, dict) and set(entry) == {'terminal'}, 'ARM_ENTRY')
        path = ref_path(entry['terminal'], suffix='.json')
        require(path.parent.parent == STUDY and path.parent.name.startswith('FIT_')
                and path.name == ('VERIFIED_TERMINAL_002.json' if arm == 'baseline' else 'VERIFIED_TERMINAL.json')
                and (arm != 'baseline' or path.parent == STUDY/'FIT_BASELINE_001'), 'TERMINAL_PATH')
    require(len({x['terminal']['path'] for x in plan['arms'].values()}) == 10, 'DISTINCT_ARMS')
    require(set(plan['sources']) == set(SOURCE_PATHS), 'SOURCE_SET')
    for name, ref in plan['sources'].items():
        require(ref_path(ref, suffix=Path(name).suffix) == REPO/name, 'SOURCE_PATH')
    output = Path(plan['output_dir'])
    require(output.is_absolute() and output.parent.parent == STUDY
            and output.name == 'OUTPUT' and output.parent.name.startswith('FAMILY_SAVED_COMPARISON_')
            and '..' not in output.parts, 'OUTPUT_ROOT')


def require_protocol(protocol):
    comparison = protocol.get('comparison', {})
    uncertainty = protocol.get('uncertainty', {})
    require(protocol.get('schema_version') == 'gx1_current_family_added_value_study_v1'
            and comparison.get('arm_order') == list(ARMS)
            and comparison.get('maximum_refit_arms') == 10
            and protocol.get('horizons_minutes') == list(range(1, 61))
            and protocol.get('primary_horizon_minutes') == 5
            and protocol.get('test_data_used') is False, 'STUDY_PROTOCOL')
    require(uncertainty.get('replicates') == 10000 and uncertainty.get('seed') == 20260911
            and uncertainty.get('familywise_alpha') == .05
            and uncertainty.get('primary_metrics') == ['auc', 'logloss', 'brier']
            and uncertainty.get('block_schemes') == ['UTC_day', 'UTC_week'], 'UNCERTAINTY')
    fold = protocol['fixed_fold']
    require(fold['prefix_rows'] == ROWS['train']
            and fold['oof_end_row']-fold['oof_start_row'] == ROWS['later']
            and fold['train_cutoff_ns'] == matched.TRAIN_END
            and fold['oof_cutoff_ns'] == matched.LATER_END, 'FOLD')


def require_admission(policy, plan, plan_ref):
    stage = policy.get(STAGE, {})
    require(stage.get('status') == 'ADMITTED' and stage.get('plan') == plan_ref
            and stage.get('execution_authorized') is True
            and stage.get('genuine_prediction_reads_authorized') is True
            and type(stage.get('max_invocations')) is int and stage['max_invocations'] == 1
            and type(stage.get('claimed_invocations')) is int and stage['claimed_invocations'] == 0
            and type(stage.get('completed_invocations')) is int and stage['completed_invocations'] == 0
            and stage.get('relaunch_allowed') is False
            and stage.get('source_head') == plan['source_head'], 'ADMISSION')
    require(all(policy.get(k) is False for k in ('training_enabled', 'native_training_enabled',
            'full_epoch_training_allowed', 'full_val_allowed', 'native_launch_authorized'))
            and all(stage.get(k) is False for k in ('model_forwards_authorized',
            'optimizer_steps_authorized', 'checkpoint_reads_authorized', 'raw_input_reads_authorized',
            'test_data_used', 'broker_access_allowed')), 'CLOSED_GATES')
    return stage


def require_source_qualification(value, sources, tracked):
    require(value.get('status') == 'PASS' and value.get('sources') == sources
            and value.get('genuine_payloads_read') == 0 and value.get('test_data_used') is False
            and set(value.get('tests', {})) == set(TEST_PATHS), 'SOURCE_QUALIFICATION')
    evidence = value.get('pytest', {})
    require(evidence.get('exit_code') == 0 and evidence.get('test_files') == list(TEST_PATHS),
            'EXACT_FOCUSED_TESTS')
    for name, ref in value['tests'].items():
        require(ref_path(ref, suffix='.py') == REPO/name, 'TEST_SOURCE')
        read_bytes(ref)
        tracked[ref['path']] = (ref, identity(Path(ref['path'])))
    for key, suffix in (('junit', '.xml'), ('stdout', '.log')):
        ref = evidence[key]
        path = ref_path(ref, suffix=suffix)
        require(path.is_relative_to(STUDY), 'TEST_EVIDENCE_ROOT')
        raw = read_bytes(ref)
        tracked[str(path)] = (ref, identity(path))
        if key == 'junit':
            report = ET.fromstring(raw)
            cases = list(report.iter('testcase'))
            require(cases and not list(report.iter('failure')) and not list(report.iter('error'))
                    and not list(report.iter('skipped')), 'TEST_RESULT')
            expected = {name[:-3].replace('/', '.') for name in TEST_PATHS}
            require({case.get('classname') for case in cases} == expected, 'TEST_MODULES')


def require_recovered_scoring_lineage(arm, terminal, review, recipe, campaign, result, read):
    """Keep completed original training and fresh scoring as two real chains."""
    origin = recipe.get('scoring_recovery_origin')
    fields = {'schema_version', 'family_arm', 'original_source_head', 'original_recipe',
              'original_plan', 'training_receipt', 'training_cursor', 'training_review',
              'state_integrity', 'interruption_review', 'closure_review'}
    identities = {
        'session_regime_encoder': ('gx1_family_session_scoring_recovery_origin_v1', 'FIT_SESSION_REGIME_001'),
        'chart_geometry_encoder': ('gx1_family_chart_scoring_recovery_origin_v1', 'FIT_CHART_GEOMETRY_001'),
    }
    require(arm in identities, 'RECOVERY_ARM')
    origin_schema, original_name = identities[arm]
    require(isinstance(origin, dict)
            and set(origin) == fields and origin.get('family_arm') == arm
            and origin.get('schema_version') == origin_schema
            and recipe.get('max_windows') == 1 and recipe.get('training_enabled') is False
            and recipe.get('full_epoch_training_allowed') is False,
            'RECOVERY_ORIGIN')
    require(terminal.get('scoring_recovery_origin') == review.get('scoring_recovery_origin')
            == result.get('scoring_recovery_origin') == origin
            and terminal.get('exit_code_scope') == 'recovery_scoring_invocation'
            and terminal.get('training_source_head') == origin['original_source_head']
            and 'original_receipts' not in terminal and 'original_receipt' not in terminal,
            'RECOVERY_NOT_ORIGINAL_RECEIPT_CHAIN')
    old_recipe = read(origin['original_recipe'])
    old_campaign = read(origin['original_plan'])
    original_root = Path(old_recipe['runtime_root'])
    require(original_root == STUDY/original_name/'RUNTIME'
            and old_campaign['recipe'] == origin['original_recipe']
            and old_campaign['source_commit'] == origin['original_source_head']
            and old_campaign['runtime_root'] == str(original_root)
            and campaign['runtime_root'] != str(original_root)
            and campaign['plan_sha256'] != old_campaign['plan_sha256'],
            'DISTINCT_RECOVERY_CAMPAIGN')
    require(old_recipe.get('recipe_sha256') == matched.binding.canonical_sha256(
            {k: v for k, v in old_recipe.items() if k != 'recipe_sha256'}),
            'ORIGINAL_RECIPE_SEAL')
    same = ('run_scope', 'family_arm', 'dense_evaluation_protocol',
            'family_intervention_sha256', 'fold', 'numerical_recipe', 'train_passes',
            'final_train_measurement_rows', 'input_manifest', 'dense_target',
            'dense_target_result', 'dense_source_transition_proof', 'decimal_input_audit',
            'decimal_target_refs', 'candidate_mode', 'parent_order_algorithm',
            'native_dense_qualification', 'decimal_initial_predictions', 'family_initial_report')
    require(all(recipe[k] == old_recipe[k] for k in same), 'RECOVERY_SAME_NUMERICAL_INPUTS')
    require(terminal.get('training_receipt') == origin['training_receipt']
            and terminal.get('scoring_receipt') == review['receipt']
            and review.get('training_window_review') == origin['training_review']
            and origin['training_receipt']['path'] == str(original_root/'receipts/invocation-0001.json')
            and terminal['scoring_receipt']['path'] == str(
                Path(campaign['runtime_root'])/'receipts/invocation-0001.json'),
            'RECOVERY_RECEIPT_ROLES')
    training = read(origin['training_receipt'])
    scoring = read(terminal['scoring_receipt'])
    for receipt, plan_sha, outcome in ((training, old_campaign['plan_sha256'], 'RESUMABLE'),
                                       (scoring, campaign['plan_sha256'], 'COMPLETE')):
        require(receipt.get('invocation_number') == 1 and receipt.get('plan_sha256') == plan_sha
                and receipt.get('guard_decision') == 'PASS'
                and receipt.get('trainer_guard_exit_code') == receipt.get('progress_observer_exit_code') == 0
                and receipt.get('outcome') == outcome and receipt.get('test_data_used') is False,
                'RECOVERY_TWO_NATIVE_CHAINS')
    require(origin['training_cursor'] == training.get('checkpoint_pointer_snapshot')
            and Path(origin['training_cursor']['path']) ==
                original_root/'invocation-evidence/invocation-0001/CHECKPOINT_POINTER.json',
            'RECOVERY_IMMUTABLE_ORIGIN_CURSOR')
    cursor = read(origin['training_cursor'])
    training_review = read(origin['training_review'])
    integrity = read(origin['state_integrity'])
    peer = read(origin['interruption_review'])
    closure = read(origin['closure_review'])
    require(cursor.get('recipe') == origin['original_recipe']
            and cursor.get('phase_state') == 'SCORING' and cursor.get('outcome') == 'RESUMABLE'
            and cursor.get('optimizer_updates') == 2976 and cursor.get('parent_cursor') == ROWS['train']
            and cursor.get('oof_cursor') == 0 and cursor.get('final_train_measurement') is None,
            'RECOVERY_TRAINING_COMPLETE')
    require(training_review.get('status') == 'PASS_TRAINING_COMPLETE_SCORING_PENDING'
            and training_review.get('family_arm') == arm
            and training_review.get('source_head') == origin['original_source_head']
            and training_review.get('receipt') == origin['training_receipt']
            and training_review.get('full_state') == cursor['full_state']
            and training_review.get('cursor', {}).get('sha256') == origin['training_cursor']['sha256'],
            'RECOVERY_TRAINING_REVIEW')
    require(integrity.get('status') == 'TRAINING_STATE_BYTES_PRESERVED_SCORING_INTERRUPTED'
            and integrity.get('source_head') == origin['original_source_head']
            and integrity.get('training_receipt') == origin['training_receipt']
            and integrity.get('training_window_review') == origin['training_review']
            and integrity.get('full_state') == cursor['full_state']
            and integrity.get('own_initial_state') == cursor['own_initial_state']
            and integrity.get('cursor', {}).get('sha256') == origin['training_cursor']['sha256']
            and integrity.get('native_exit_code_known') is False
            and integrity.get('completed_training_updates') == 2976
            and integrity.get('training_parents') == ROWS['train']
            and integrity.get('originals_modified') is False and integrity.get('test_data_used') is False,
            'RECOVERY_INTEGRITY')
    require(peer.get('status') == 'PASS_SCOPED_METADATA_WORKER_REVIEW'
            and peer.get('references', {}).get('integrity') == origin['state_integrity'], 'RECOVERY_PEER')
    if arm == 'chart_geometry_encoder':
        require(peer.get('schema_version') == 'gx1_family_interruption_independent_metadata_peer_review_v1'
                and peer.get('family_arm') == arm and peer.get('test_data_used') is False
                and peer.get('original_native_exit_code_known') is False
                and peer.get('original_native_exit_code') is None
                and integrity.get('schema_version') == 'gx1_family_interrupted_scoring_integrity_v1'
                and integrity.get('family_arm') == arm, 'RECOVERY_CHART_PEER')
    else:
        require(peer.get('scope', {}).get('checkpoint_or_payload_bytes_read_by_peer') is False
                and peer.get('scope', {}).get('test_data_used') is False, 'RECOVERY_PEER')
    require(closure.get('schema_version') == 'gx1_family_interrupted_scoring_closure_v1'
            and closure.get('status') == 'CLOSED_INTERRUPTED_SCORING_ORIGINALS_PRESERVED'
            and closure.get('family_arm') == arm
            and closure.get('original_source_head') == origin['original_source_head']
            and closure.get('state_integrity') == origin['state_integrity']
            and closure.get('interruption_review') == origin['interruption_review']
            and closure.get('training_window_review') == origin['training_review']
            and closure.get('interrupted_active_marker') == integrity.get('interrupted_active_marker')
            and closure.get('interrupted_claim') == integrity.get('interrupted_claim')
            and closure.get('original_relaunch_allowed') is False
            and closure.get('interrupted_native_exit_code_known') is False
            and closure.get('interrupted_native_exit_code') is None
            and closure.get('claimed_invocations') == 2 and closure.get('completed_invocations') == 1
            and closure.get('revoked_invocations') == [3, 4]
            and closure.get('test_data_used') is False, 'RECOVERY_CLOSURE')
    old_result = read(cursor['result'])
    require(old_result.get('recipe') == origin['original_recipe']
            and old_result.get('outcome') == 'RESUMABLE'
            and old_result.get('phase_state') == 'SCORING'
            and old_result.get('error') is None and old_result.get('source_unchanged') is True
            and old_result.get('optimizer_updates') == 2976
            and old_result.get('parent_cursor') == ROWS['train']
            and old_result.get('test_data_used') is False, 'RECOVERY_ORIGINAL_TRAIN_RESULT')
    seal = old_result.get('durable_checkpoint', {}).get('contract_sha256')
    require(isinstance(seal, str) and len(seal) == 64
            and old_result.get('durable_checkpoint', {}).get('state') == cursor['full_state']
            and result.get('durable_checkpoint', {}).get('contract_sha256') == seal
            and result.get('own_initial_state') == cursor['own_initial_state']
            and result.get('initial_measurement') == cursor['initial_measurement']
            and result.get('optimizer_updates') == cursor['optimizer_updates']
            and result.get('parent_cursor') == cursor['parent_cursor']
            and result.get('error') is None and result.get('qualification_weights_reused') is False,
            'RECOVERY_STATE_UNCHANGED')
    counts = result.get('counts', {})
    require(all(counts.get(k) == 0 for k in ('backward_calls', 'backward_returns',
            'optimizer_step_calls', 'optimizer_step_returns', 'empty_batches', 'forbidden_legacy_calls'))
            and counts.get('forward_calls') == counts.get('forward_returns') == 4057
            and counts.get('forward_kind_calls') == {'FINAL_TRAIN': 2976, 'OOF': 1081},
            'RECOVERY_ZERO_UPDATES_FULL_SCORING')


def require_arm_evidence(arm, entry, protocol, read):
    """Read an explicit metadata chain, never recursively open arbitrary refs."""
    terminal = read(entry['terminal'])
    baseline = arm == 'baseline'
    recovered = terminal.get('schema_version') == 'gx1_family_recovered_scoring_verified_terminal_v1'
    require(not recovered or arm in ('session_regime_encoder', 'chart_geometry_encoder'), 'RECOVERY_ARM')
    require(terminal.get('schema_version') == ('gx1_family_baseline_verified_terminal_v1'
            if baseline else ('gx1_family_recovered_scoring_verified_terminal_v1'
                              if recovered else 'gx1_family_verified_terminal_v1'))
            and terminal.get('status') == 'COMPLETE' and terminal.get('exit_code') == 0
            and terminal.get('family_arm') == arm
            and terminal.get('source_unchanged') is True and terminal.get('test_data_used') is False
            and terminal.get('optimizer_updates') == 2976 and terminal.get('scoring_optimizer_updates') == 0
            and terminal.get('train_parent_rows' if baseline else 'train_parents') == ROWS['train']
            and terminal.get('later_parent_rows' if baseline else 'later_parents') == ROWS['later'], 'TERMINAL:'+arm)
    root = Path(entry['terminal']['path']).parent
    require(terminal['root_review']['path'] == str(root/'ACTUAL_ROOT_REVIEW_001.json'), 'OWN_REVIEW')
    review = read(terminal['root_review'])
    require(review.get('schema_version') == ('gx1_family_baseline_training_window_review_v1'
            if baseline else ('gx1_family_recovered_scoring_review_v1'
                              if recovered else 'gx1_family_training_window_review_v1'))
            and (baseline or review.get('family_arm') == arm)
            and review.get('status', '').startswith('PASS_')
            and review.get('status', '').endswith('_ONEPASS_FULLTRAIN_LATER_TERMINAL')
            and review.get('source_head') == terminal['source_head']
            and review.get('next_action') == {'decision': 'COMPLETE'}
            and review.get('full_train_scored') is True and review.get('later_scored') is True
            and review.get('training_passes') == 1 and review.get('optimizer_updates') == 2976
            and review.get('scoring_optimizer_updates') == 0 and review.get('test_data_used') is False,
            'ROOT_REVIEW:'+arm)
    payloads = {'train': review['full_train_predictions'], 'later': review['later_predictions']}
    require(terminal['train_predictions' if baseline else 'full_train_predictions'] == payloads['train']
            and terminal['later_predictions'] == payloads['later'], 'TERMINAL_ARCHIVES')
    for part, ref in payloads.items():
        path = ref_path(ref, suffix='.npz')
        require(path.is_relative_to(root/'RUNTIME/prefix-fit')
                and path.name == ('FINAL_TRAIN.npz' if part == 'train' else 'OOF.npz'), 'ARCHIVE_ROLE')
    verification = read(terminal['saved_prediction_verification'])
    vt_ref = terminal['saved_prediction_terminal' if baseline else 'saved_verification_terminal']
    vt = read(vt_ref)
    require(vt.get('status') == 'COMPLETE' and vt.get('exit_code') == 0
            and vt.get('result') == terminal['saved_prediction_verification']
            and vt.get('plan') == verification.get('plan') and vt.get('error') is None
            and vt.get('denied_events') == [] and vt.get('test_data_used') is False, 'VERIFICATION_TERMINAL')
    require(verification.get('schema_version') == ('gx1_family_baseline_saved_verification_result_v1'
            if baseline else 'gx1_family_saved_verification_result_v1')
            and verification.get('status') == 'PASS'
            and verification.get('source_head') == terminal['source_head']
            and verification.get('canonical_axis_source') == AXIS_SOURCE
            and verification.get('same_metric_owner_recomputation') is True
            and verification.get('model_or_checkpoint_deserialized') is False
            and verification.get('test_data_used') is False
            and all(verification['inputs'].get(ref['path']) == ref
                    for ref in [terminal['root_review'], *payloads.values()]), 'SAVED_VERIFICATION')
    execution = read(review['execution_binding'])
    campaign = read(execution['plan'])
    recipe = read(campaign['recipe'])
    require(execution['source_head'] == campaign['source_commit'] == terminal['source_head']
            and recipe.get('run_scope') == 'single_prefix_m5_family_value_sign_one_pass'
            and recipe.get('family_arm') == arm and recipe.get('runtime_root') == str(root/'RUNTIME')
            and recipe.get('dense_evaluation_protocol') == PROTOCOL
            and recipe.get('family_intervention_sha256') == protocol['comparison']['intervention_contract_sha256']
            and recipe.get('fold') == protocol['fixed_fold']
            and recipe.get('numerical_recipe') == protocol['comparison']['fixed_training']['numerical_recipe']
            and recipe.get('train_passes') == 1 and recipe.get('final_train_measurement_rows') == ROWS['train']
            and recipe.get('test_data_used') is False, 'MATCHED_RECIPE')
    require(recipe.get('recipe_sha256') == matched.binding.canonical_sha256(
            {k: v for k, v in recipe.items() if k != 'recipe_sha256'}), 'RECIPE_SEAL')
    private = read(recipe['native_dense_qualification'])
    require(private.get('family_arm') == arm
            and private.get('status') == 'PASS_CONSUMED_FAMILY_PRIVATE_NATIVE_MECHANICS'
            and private.get('unperturbed_initial_prediction_shared_state_parity') is True
            and private.get('initial_predictions') == recipe['decimal_initial_predictions']
            and private.get('initial_report') == recipe['family_initial_report']
            and private.get('qualification_weights_for_main_fit') is False
            and private.get('test_data_used') is False, 'OWN_PRIVATE_PARITY')
    if not recovered:
        receipts = ([read(review['training_window_review'])['receipt'], terminal['original_receipt']]
                    if baseline else terminal['original_receipts'])
        require(isinstance(receipts, list) and 1 <= len(receipts) <= 4
                and receipts[-1] == review['receipt'], 'ORIGINAL_RECEIPTS')
        for number, ref in enumerate(receipts, 1):
            receipt = read(ref)
            require(receipt.get('invocation_number') == number
                    and receipt.get('plan_sha256') == campaign['plan_sha256']
                    and receipt.get('guard_decision') == 'PASS'
                    and receipt.get('trainer_guard_exit_code') == receipt.get('progress_observer_exit_code') == 0
                    and receipt.get('outcome') == ('COMPLETE' if number == len(receipts) else 'RESUMABLE')
                    and receipt.get('test_data_used') is False, 'NATIVE_RECEIPT')
    result = read(review['result'])
    require(result.get('outcome') == 'COMPLETE' and result.get('recipe') == campaign['recipe']
            and result.get('source_unchanged') is True and result.get('test_data_used') is False, 'FIT_RESULT')
    if recovered:
        require_recovered_scoring_lineage(arm, terminal, review, recipe, campaign, result, read)
    reports = {part: read(result[key]) for part, key in
               (('train', 'final_train_measurement'), ('later', 'oof_measurement'))}
    for part, report in reports.items():
        require(report.get('predictions') == payloads[part] and report.get('fold') == recipe['fold']
                and report.get('role') == ('FINAL_TRAIN' if part == 'train' else 'OOF')
                and report.get('test_data_used') is False
                and verification['panels'][part]['parents'] == ROWS[part]
                and verification['panels'][part]['horizons'] == 60
                and len(report['model_metrics']['per_horizon']) == 60, 'REPORT_BINDING')
    for key in ('feature_view_identity', 'own_prefix_prior', 'context_map', 'own_initial_state', 'fresh_identity_sha256'):
        require(reports['train'][key] == reports['later'][key], 'REPORT_PAIR:'+key)
    require(verification['axis_lineage_manifest'] == recipe['input_manifest'], 'AXIS_LINEAGE')
    return {'payloads': payloads, 'reports': reports, 'recipe': recipe, 'verification': verification}


def read_prediction(ref, rows):
    raw = read_bytes(ref)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries = archive.infolist()
        require(len(entries) == len(NAMES) and {x.filename for x in entries} == {k+'.npy' for k in NAMES}
                and sum(x.file_size for x in entries) <= MAX_BYTES, 'ARCHIVE_BOUNDS')
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    require(set(data) == NAMES, 'ARCHIVE_MEMBERS')
    for key, value in data.items():
        shape = (rows,) if key in ('parent_rows', 'bar_start_ns', 'decision_ns') or key.startswith('context__') else (rows, 60)
        dtype = np.int8 if key == 'sign' or key.startswith('context__') else np.bool_ if key == 'observed' else np.float32 if key == 'logit_up' else np.int64
        require(value.shape == shape and value.dtype == dtype and np.isfinite(value).all(), 'ARRAY_DOMAIN:'+key)
    for key, labels in matched.CONTEXT_SPEC:
        require(np.isin(data['context__'+key], [x for x, _ in labels]).all(), 'CONTEXT_DOMAIN')
    require(np.all(data['parent_rows'] >= 0) and np.all(data['bar_start_ns'] > 0)
            and np.all(data['bar_start_ns'] % 300_000_000_000 == 0), 'CLOCK_DOMAIN')
    return data


def atomic_json(path, value):
    """Exclusive publication: a durable complete inode is linked into place."""
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
    temporary = path.with_name('.'+path.name+'.tmp')
    with temporary.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.chmod(0o444)
    os.link(temporary, path)
    temporary.unlink()
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}


def compare_archives(evidence, axis, protocol, *, reader=read_prediction):
    """Admission has already bound every arm; no fallback to a partial study."""
    require(set(evidence) == set(ARMS), 'ALL_TEN_ARMS')
    partitions = {'train': {}, 'later': {}}
    for arm in ARMS:
        for part in partitions:
            partitions[part][arm] = reader(evidence[arm]['payloads'][part], ROWS[part])
    hashes = {}
    for part, data in partitions.items():
        base = data['baseline']
        anchor = axis['partitions'][part]
        require(anchor['selected_parent_rows'] == ROWS[part], 'AXIS_ROWS')
        for key in ('parent_rows', 'decision_ns'):
            require(matched.saved.digest(base[key], np) == anchor[key+'_sha256'], 'CANONICAL_AXIS')
        # Bar start is deterministically bound to the independently anchored decision clock.
        require(np.array_equal(base['bar_start_ns'], base['decision_ns']-300_000_000_000), 'BAR_START_AXIS')
        hashes[part] = {k: matched.saved.digest(base[k], np) for k in ('parent_rows', 'bar_start_ns', 'decision_ns')}
    fold = protocol['fixed_fold']
    result = matched.compare_family_value_predictions(
        partitions['train'], partitions['later'], arm_order=list(ARMS),
        train_cutoff=fold['train_cutoff_ns'], later_cutoff=fold['oof_cutoff_ns'],
        expected_train_rows=ROWS['train'], expected_later_rows=ROWS['later'],
        expected_axis_hashes=hashes, replicates=10000, seed=20260911, familywise_alpha=.05)
    require(len(result['primary_five_minute']) == 9 and result['per_interval_alpha'] == .05/54, 'CONTRAST_COUNT')
    for part in partitions:
        for arm in ARMS:
            original = evidence[arm]['reports'][part]['model_metrics']['per_horizon']
            for panel, report in zip(result['panels'][part], original):
                metrics = panel['arms'][arm]
                require(panel['horizon_minutes'] == report['horizon_minutes']
                        and panel['counts']['eligible'] == report['counts']['eligible_cells']
                        and all(np.isclose(metrics[k], report[k], rtol=1e-12, atol=1e-12)
                                for k in ('accuracy', 'logloss', 'brier'))
                        and np.isclose(metrics['auc'], report['rank']['auc'], rtol=1e-12, atol=1e-12),
                        'ORIGINAL_ALL60_SCORES')
    return result


def main(argv=None):
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'plan-sha256', 'policy-sha256', 'source-head'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args(argv)
    started = time.monotonic()
    scope = matched.cap.require_capped_cpu_audit_execution()
    lock = matched.cap.require_capped_lock_ancestry()
    require(lock['scope'] == scope['cgroup_path'][len('/sys/fs/cgroup'):], 'CAPPED_LOCK')
    expected_git = (args.source_head, 'work/gx1-current', '')
    require(matched._git_state() == expected_git, 'CLEAN_SOURCE')
    tracked = {}
    read = lambda ref: read_metadata(ref, tracked)
    plan_ref = {'path': args.plan, 'sha256': args.plan_sha256}
    plan = read(plan_ref)
    require_plan(plan)
    require(plan['source_head'] == args.source_head, 'PLAN_SOURCE')
    policy_ref = {'path': str(REPO/'NEXT_RUN_POLICY.json'), 'sha256': args.policy_sha256}
    policy = read(policy_ref)
    stage = require_admission(policy, plan, plan_ref)
    qualified = read(stage['source_qualification'])
    require_source_qualification(qualified, plan['sources'], tracked)
    for ref in plan['sources'].values():
        read_bytes(ref)
        tracked[ref['path']] = (ref, identity(Path(ref['path'])))
    protocol = read(plan['protocol'])
    require_protocol(protocol)
    axis = read(plan['canonical_axis_source'])
    require(axis.get('status') == 'COMPLETE' and axis.get('full_population_measured') is True
            and axis.get('source_unchanged') is True and axis.get('test_data_used') is False, 'AXIS_ANCHOR')
    evidence = {arm: require_arm_evidence(arm, plan['arms'][arm], protocol, read) for arm in ARMS}
    baseline = evidence['baseline']['recipe']
    for item in evidence.values():
        require(all(item['recipe'][k] == baseline[k] for k in
            ('input_manifest', 'dense_target', 'dense_target_result', 'dense_source_transition_proof',
             'decimal_input_audit', 'decimal_target_refs', 'candidate_mode', 'parent_order_algorithm',
             'fold', 'numerical_recipe')), 'MATCHED_INPUT_RECIPE')
    payloads = [evidence[a]['payloads'][p] for a in ARMS for p in ('train', 'later')]
    require(len({x['path'] for x in payloads}) == 20, 'TWENTY_DISTINCT_ARCHIVES')
    output = Path(plan['output_dir'])
    require(not output.exists() and not output.is_symlink()
            and output.parent.is_dir() and output.parent.resolve() == output.parent, 'UNCLAIMED_OUTPUT')
    output.mkdir(mode=0o700)
    atomic_json(output/'CLAIM.json', {'plan': plan_ref, 'policy': policy_ref,
                'source_head': args.source_head, 'test_data_used': False})
    reads, denials = matched._install_guard(payloads, set(tracked), output)
    result_ref = None
    error = None
    try:
        result = compare_archives(evidence, axis, protocol)
        require(set(reads) == {x['path'] for x in payloads} and all(n == 1 for n in reads.values())
                and not denials, 'PAYLOAD_OPEN_SCOPE')
        for ref, before in tracked.values():
            require(identity(Path(ref['path'])) == before, 'METADATA_BOOKEND')
            read_bytes(ref)
        require(matched._git_state() == expected_git and matched.cap.require_capped_lock_ancestry() == lock,
                'FINAL_SOURCE_LOCK')
        result.update(status='COMPLETE_ALL_TEN_REUSED_DEVELOPMENT_NO_EDGE_CLAIM',
            plan=plan_ref, policy=policy_ref, source_head=args.source_head,
            canonical_axis_source=AXIS_SOURCE, original_evidence=plan['arms'],
            source_qualification=stage['source_qualification'], execution=scope, lock=lock,
            original_all60_scores_reproduced=True, primary_contrasts=54,
            payload_archives_read=20, raw_input_reads=0, checkpoint_reads=0,
            model_forwards=0, optimizer_steps=0, elapsed_seconds=time.monotonic()-started,
            independent_metric_algebra=False, target_label_source_replay=False)
        result_ref = atomic_json(output/'RESULT.json', result)
    except Exception as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)}
    terminal = atomic_json(output/'TERMINAL.json', {
        'status': 'COMPLETE' if error is None else 'FAILED', 'exit_code': 0 if error is None else 1,
        'plan': plan_ref, 'result': result_ref, 'error': error, 'denied_paths': denials,
        'payload_open_counts': reads, 'test_data_used': False})
    print(json.dumps({'terminal': terminal, 'result': result_ref, 'error': error}, sort_keys=True))
    return 0 if error is None else 1


if __name__ == '__main__':
    raise SystemExit(main())
