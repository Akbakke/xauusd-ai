"""Bind and independently check outputs after an intervention publication failure.

No model forward or training occurs. Original predictions and corrected labels
are re-scored. Ablated predictions were not retained, so their metrics cannot be
independently recomputed by this audit; the original failed run stays failed.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import sys
import time
import zipfile

import numpy as np

REPO = Path('/home/andre2/src/GX1_CURRENT')
ROOT = Path('/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63')
STAGE = 'independent_direction_gc_mtf_intervention_recovery_20261028_001'
SOURCE_PATHS = (
    'gx1/scripts/audit_entry_direction_gc_intervention_recovery_v1.py',
    'gx1/audit/direction_gc_mtf_intervention_v1.py',
    'gx1/scripts/audit_entry_direction_candle_midpoints_v1.py',
    'gx1/contracts/entry_direction_probability_reference_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'gx1/contracts/local_random_access_campaign_v2.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'gx1/contracts/entry_model_native_bundle_commit_v1.py',
    'scripts/gx1_capped_run.sh',
    'tests/test_audit_entry_direction_gc_intervention_recovery_v1.py',
)


def require(ok, reason):
    if not bool(ok):
        raise RuntimeError('DIRECTION_GC_INTERVENTION_RECOVERY_' + reason)


def verify_original_scores(saved, corrected, result, *, first, last, cutoff, select, probability):
    """Independently recompute every saved-original sampled score on corrected labels."""
    n = last - first
    for name in ('parent_rows', 'bar_start_ns', 'decision_ns'):
        require(name in saved and name in corrected and np.array_equal(saved[name], corrected[name][first:last]), 'AXIS:' + name)
    for name in ('terminal_time_ns', 'observed'):
        require(name in saved and name in corrected and np.array_equal(saved[name], corrected[name][first:last]), 'TARGET:' + name)
    require(saved['logit_up'].dtype == np.float32 and saved['logit_up'].shape == (n, 60), 'SAVED_LOGITS')
    chunks = select(n); indices = np.concatenate(chunks); rows = first + indices
    require(result['selection'] == {'policy': 'every tenth original OOF batch', 'batch_size': 16,
        'batch_stride': 10, 'selected_parents': len(indices), 'selected_batches': len(chunks)}, 'SELECTION')
    panels = result['metrics']['per_horizon']; require(len(panels) == 60, 'SIXTY_PANELS')
    max_difference = 0.0
    for j in range(60):
        take = corrected['observed'][rows, j] & (corrected['sign'][rows, j] != 0) & (corrected['terminal_time_ns'][rows, j] < cutoff)
        labels = np.ascontiguousarray((corrected['sign'][rows[take], j] == 1).astype(np.int8))
        logits = np.ascontiguousarray(saved['logit_up'][indices[take], j], dtype=np.float64)
        require(len(labels) > 0 and np.unique(labels).size == 2, 'BOTH_CLASSES:' + str(j + 1))
        metrics = probability.direction_probability_metrics(labels, logits)
        actual = {'auc': probability.direction_rank_diagnostics(labels, logits, score_kind='logit_up')['auc'],
            'logloss': float(np.mean(metrics['logloss_by_row'])), 'brier': float(np.mean(metrics['brier_by_row']))}
        panel = panels[j]; require(panel['horizon_minutes'] == j + 1 and panel['eligible'] == len(labels), 'PANEL_COUNTS')
        for key, value in actual.items():
            difference = abs(value - panel['arms']['original'][key]); max_difference = max(max_difference, difference)
            require(np.isfinite(difference) and difference <= 1e-12, 'ORIGINAL_SCORE:' + str(j + 1) + ':' + key)
    return {'sampled_parents': len(indices), 'sampled_batches': len(chunks),
        'horizons_verified': 60, 'max_absolute_score_difference': max_difference}


def validate_failed_run(result, inner, failed, claim, plan_ref, stdio, guard):
    require(result['schema_version'] == 'gx1_direction_gc_mtf_intervention_result_v1'
        and result['status'] == 'COMPLETE' and result['plan'] == claim['plan'] == plan_ref
        and result['source_head'] == claim['source_head']
        and result['exact_saved_oof_logit_bytes_reproduced'] is True
        and result['forward_calls'] == 218 and result['optimizer_steps'] == 0
        and result['model_unchanged'] is True and result['test_data_used'] is False
        and result['corrected_target_identity'] == 'exact_original_decimal_midpoint_v1', 'INNER_RESULT')
    require(inner['status'] == 'COMPLETE' and inner['exit_code'] == 0
        and inner['inputs_unchanged'] is inner['source_unchanged'] is True and inner['test_data_used'] is False, 'INNER_TERMINAL')
    require(failed['status'] == 'FAILED' and failed['exit_code'] == 1 and failed['relaunch_allowed'] is False
        and 'NATIVE_DIRECTION_RESEARCH_WINDOW_UNBOUND_OPEN:' in failed['error'], 'FAILED_TERMINAL')
    require('publish_bundle_directory_noreplace' in stdio and 'NATIVE_DIRECTION_RESEARCH_WINDOW_UNBOUND_OPEN:' in stdio
        and 'event=exit child_status=1' in guard and 'peak_core_temp_c=53' in guard
        and 'peak_memory_temp_c=56' in guard and 'peak_power_draw_w=132.17' in guard, 'FAILURE_AND_GUARD')
    intervals = result['metrics']['paired_intervals']
    require(len(intervals) == 4 and result['metrics']['per_interval_alpha'] == 0.05 / 12
        and {(item['horizon_minutes'], item['scheme']) for item in intervals}
            == {(h, scheme) for h in (5, 60) for scheme in ('UTC_DAY', 'UTC_WEEK_MONDAY')}, 'INTERVAL_SCOPE')
    for item in intervals:
        require(item['horizon_minutes'] in (5, 60) and item['scheme'] in ('UTC_DAY', 'UTC_WEEK_MONDAY'), 'INTERVAL_ID')
        for value in item['intervals'].values():
            require(value['valid_replicates'] == 10000 and value['undefined_replicates'] == 0
                and np.isfinite(value['low']) and np.isfinite(value['high']) and value['low'] <= value['high'], 'INTERVAL_VALUES')


def main(argv=None):
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('plan', 'plan-sha256', 'policy-sha256', 'source-head'):
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args(argv)
    from gx1.contracts import gx1_capped_execution_v1 as cap
    from gx1.contracts import immutable_event_authority_v1 as publication
    from gx1.contracts import local_random_access_campaign_v2 as binding
    from gx1.contracts import entry_direction_probability_reference_v1 as probability
    from gx1.contracts.entry_model_native_bundle_commit_v1 import publish_bundle_directory_noreplace
    from gx1.scripts import audit_entry_direction_candle_midpoints_v1 as candle
    from gx1.audit.direction_gc_mtf_intervention_v1 import selected_batches
    require('torch' in sys.modules and not sys.modules['torch'].cuda.is_initialized(), 'NO_CUDA')
    initial = candle.git_state(); require(initial == (args.source_head, 'work/gx1-current', ''), 'INITIAL_SOURCE')
    plan_ref = {'path': str(candle.regular(args.plan)), 'sha256': args.plan_sha256}
    policy_ref = {'path': str(REPO / 'NEXT_RUN_POLICY.json'), 'sha256': args.policy_sha256}
    with binding.fresh_file_verification_scope():
        plan = binding.read_bound_json(Path(plan_ref['path']), plan_ref['sha256'])
        require(set(plan) == {'schema_version', 'stage', 'design', 'inputs', 'sources', 'output_dir', 'wall_seconds', 'test_data_used'}
            and plan['schema_version'] == 'gx1_direction_gc_intervention_recovery_plan_v1' and plan['stage'] == STAGE
            and plan['wall_seconds'] == 600 and plan['test_data_used'] is False, 'PLAN')
        require(plan['design']['path'] == str(REPO / 'docs/DIRECTION_GC_INTERVENTION_RECOVERY_20261028.json'), 'DESIGN_PATH')
        design = binding.read_bound_json(Path(plan['design']['path']), plan['design']['sha256'])
        require(design['schema_version'] == 'gx1_direction_gc_intervention_recovery_design_v1'
            and design['inputs'] == plan['inputs'] and design['test_data_used'] is False, 'DESIGN')
        policy = binding.read_bound_json(Path(policy_ref['path']), policy_ref['sha256']); gate = policy[STAGE]
        require(gate['status'] == 'ADMITTED' and gate['plan'] == plan_ref and gate['execution_authorized'] is True
            and gate['max_invocations'] == 1 and gate['claimed_invocations'] == gate['completed_invocations'] == 0
            and gate['relaunch_allowed'] is False, 'ONE_SHOT_POLICY')
        require(all(policy[k] is False for k in ('training_enabled', 'native_training_enabled',
            'full_epoch_training_allowed', 'full_val_allowed', 'native_launch_authorized'))
            and all(gate[k] is False for k in ('model_forwards_authorized', 'optimizer_steps_authorized',
                'checkpoint_reads_authorized', 'test_data_used', 'broker_access_allowed')), 'CLOSED_GATES')
        require(set(plan['sources']) == set(SOURCE_PATHS)
            and all(ref['path'] == str(REPO / name) for name, ref in plan['sources'].items()), 'SOURCE_SET')
        for ref in plan['sources'].values():
            candle.regular(ref['path']); binding.require_binding(ref, label='source')
        require(set(plan['inputs']) == {'original_plan', 'original_result', 'inner_terminal', 'failed_terminal', 'claim',
            'stdio_log', 'guard_log', 'oof_predictions', 'corrected_target', 'oof_report'}, 'INPUT_SET')
        meta = {}
        for key in ('original_plan', 'original_result', 'inner_terminal', 'failed_terminal', 'claim', 'oof_report'):
            ref = plan['inputs'][key]; meta[key] = binding.read_bound_json(Path(ref['path']), ref['sha256'])
        prior = policy['independent_direction_gc_mtf_intervention_20261028_003']
        require(prior['status'] == 'FAILED_ONE_SHOT_CONSUMED' and prior['execution_authorized'] is False
            and prior['plan'] == plan['inputs']['original_plan'] and prior['claim'] == plan['inputs']['claim']
            and prior['failed_terminal'] == plan['inputs']['failed_terminal'], 'FAILED_PRIOR_BINDING')
        original_plan, original, inner, failed, claim, oof = (meta[k] for k in
            ('original_plan', 'original_result', 'inner_terminal', 'failed_terminal', 'claim', 'oof_report'))
        require(original_plan['inputs']['oof_predictions'] == plan['inputs']['oof_predictions']
            and original_plan['inputs']['corrected_target'] == plan['inputs']['corrected_target']
            and original_plan['inputs']['oof_report'] == plan['inputs']['oof_report']
            and oof['predictions'] == plan['inputs']['oof_predictions'] and oof['fold'] == original['fold'], 'PRIOR_INPUTS')
        control_snapshot = binding.capture_verified_file_snapshot()
    proof, lock = cap.require_capped_cpu_audit_execution(), cap.require_capped_lock_ancestry()
    require(lock['scope'] == proof['cgroup_path'][len('/sys/fs/cgroup'):], 'CAPPED_LOCK_SCOPE')
    out = Path(plan['output_dir']); require(out.parent == ROOT and out.name == 'DIRECTION_GC_INTERVENTION_RECOVERY_20261028_001'
        and ROOT.resolve(strict=True) == ROOT and not out.exists() and not out.is_symlink(), 'OUTPUT_SCOPE')
    original_dir = Path(original_plan['output_dir']); require(original_dir.is_dir() and not original_dir.is_symlink()
        and original_dir.parent == ROOT, 'ORIGINAL_DIRECTORY')
    out.mkdir(exist_ok=False)
    def publish(path, value):
        publication._publish_bytes(path, binding.canonical_bytes(value))
    publish(out / 'INVOCATION_CLAIM.json', {'plan': plan_ref, 'policy': policy_ref,
        'source_head': args.source_head, 'execution': proof, 'lock': lock})
    payloads = [plan['inputs'][k] for k in ('oof_predictions', 'corrected_target')]
    logrefs = [plan['inputs'][k] for k in ('stdio_log', 'guard_log')]
    metadata = [plan_ref['path'], policy_ref['path'], plan['design']['path'], str(original_dir), str(original_dir / 'OUTPUT')]
    metadata += [plan['inputs'][k]['path'] for k in ('original_plan', 'original_result', 'inner_terminal', 'failed_terminal', 'claim', 'oof_report')]
    metadata += [ref['path'] for ref in logrefs]
    sys.addaudithook(candle.make_guard(out, [ref['path'] for ref in payloads], metadata,
        source_paths=[ref['path'] for ref in plan['sources'].values()]))
    started = time.monotonic(); previous_handler = signal.getsignal(signal.SIGALRM); previous_profile = sys.getprofile()
    def expired(*unused):
        raise RuntimeError('DIRECTION_GC_INTERVENTION_RECOVERY_WALL_LIMIT')
    signal.signal(signal.SIGALRM, expired); signal.alarm(plan['wall_seconds']); sys.setprofile(candle.no_queries_or_models)
    stage = out / '.OUTPUT.staging'
    try:
        stage.mkdir(exist_ok=False)
        with binding.fresh_file_verification_scope():
            for ref in payloads + logrefs:
                candle.regular(ref['path']); binding.require_binding(ref, label='input')
            stdio, guard = (Path(ref['path']).read_text() for ref in logrefs)
            validate_failed_run(original, inner, failed, claim, plan['inputs']['original_plan'], stdio, guard)
            with np.load(plan['inputs']['oof_predictions']['path'], allow_pickle=False) as archive:
                names = ('parent_rows', 'bar_start_ns', 'decision_ns', 'terminal_time_ns', 'observed', 'sign', 'logit_up')
                require(all(name in archive.files for name in names), 'OOF_FIELDS')
                saved = {name: archive[name].copy() for name in names}
            with np.load(plan['inputs']['corrected_target']['path'], allow_pickle=False) as archive:
                names = ('parent_rows', 'bar_start_ns', 'decision_ns', 'terminal_time_ns', 'observed', 'sign')
                require(all(name in archive.files for name in names), 'CORRECTED_FIELDS')
                corrected = {name: archive[name].copy() for name in names}
            fold = original['fold']
            verified = verify_original_scores(saved, corrected, original, first=fold['oof_start_row'],
                last=fold['oof_end_row'], cutoff=fold['oof_cutoff_ns'], select=selected_batches, probability=probability)
            input_snapshot = binding.capture_verified_file_snapshot()
        input_snapshot.require_unchanged(); control_snapshot.require_unchanged()
        # Complete only the directory sync that the failed producer could not do.
        for directory in (original_dir / 'OUTPUT', original_dir):
            descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try: os.fsync(descriptor)
            finally: os.close(descriptor)
        require(candle.git_state() == initial and cap.require_capped_lock_ancestry() == lock, 'FINAL_SOURCE_LOCK')
        output = {'schema_version': 'gx1_direction_gc_intervention_recovery_result_v1', 'status': 'COMPLETE',
            'plan': plan_ref, 'source_head': args.source_head, 'original_failed_plan': plan['inputs']['original_plan'],
            'original_failed_run_result': plan['inputs']['original_result'], 'original_inner_terminal': plan['inputs']['inner_terminal'],
            'original_failed_terminal': plan['inputs']['failed_terminal'], 'original_run_remains_failed': True,
            'original_directory_fsync_completed': True, 'independently_recomputed_original_scores': verified,
            'ablated_logits_retained': False, 'ablated_scores_independently_recomputed': False,
            'limits': 'Integrity recovery and original-score replay only. Ablation metrics remain from the failed producer, not independently recomputed. One reused development fold; no economic edge.',
            'new_model_forwards': 0, 'optimizer_steps': 0, 'test_data_used': False}
        publish(stage / 'RESULT.json', output)
        publish(stage / 'TERMINAL.json', {'status': 'COMPLETE', 'exit_code': 0, 'source_unchanged': True,
            'inputs_unchanged': True, 'elapsed_seconds': time.monotonic() - started, 'test_data_used': False})
        publish_bundle_directory_noreplace(stage, out / 'OUTPUT')
        publish(out / 'TERMINAL.json', {'status': 'COMPLETE', 'exit_code': 0, 'plan': plan_ref,
            'result_path': str(out / 'OUTPUT/RESULT.json')})
        return 0
    except Exception as exc:
        publish(out / 'FAILED_TERMINAL.json', {'status': 'FAILED', 'exit_code': 1,
            'error': type(exc).__name__ + ': ' + str(exc), 'relaunch_allowed': False})
        raise
    finally:
        sys.setprofile(previous_profile); signal.alarm(0); signal.signal(signal.SIGALRM, previous_handler)


if __name__ == '__main__':
    raise SystemExit(main())
