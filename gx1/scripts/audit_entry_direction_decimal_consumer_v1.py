"""Qualify a separate decimal-target structural consumer; no model or training."""
from __future__ import annotations

import argparse
from pathlib import Path
import signal
import sys
import time
import zipfile

import numpy as np

REPO = Path('/home/andre2/src/GX1_CURRENT')
ROOT = Path('/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63')
STAGE = 'independent_direction_decimal_consumer_audit_20261028_001'
SOURCE_PATHS = (
    'gx1/scripts/audit_entry_direction_decimal_consumer_v1.py',
    'gx1/scripts/audit_entry_direction_decimal_sign_v1.py',
    'gx1/scripts/audit_entry_direction_candle_midpoints_v1.py',
    'gx1/scripts/materialize_entry_direction_decimal_targets_v1.py',
    'gx1/contracts/entry_native_direction_decimal_data_v1.py',
    'gx1/contracts/entry_native_direction_dense_data_v1.py',
    'gx1/contracts/entry_native_direction_dense_v1.py',
    'gx1/contracts/entry_native_direction_research_v1.py',
    'gx1/contracts/entry_model_native_training_objective_v1.py',
    'gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'gx1/contracts/local_random_access_campaign_v2.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'gx1/contracts/entry_model_native_bundle_commit_v1.py',
    'scripts/gx1_capped_run.sh',
    'tests/test_entry_native_direction_decimal_data_v1.py',
    'tests/test_audit_entry_direction_decimal_consumer_v1.py',
)


def require(ok, reason):
    if not bool(ok):
        raise RuntimeError('DIRECTION_DECIMAL_CONSUMER_' + reason)


def require_receipts(plan, materialization, materialization_plan, audit, oof, *, decimal):
    """Check semantic linkage after the caller authenticates all JSON bytes."""
    refs = plan['inputs']
    require(materialization.get('schema_version') == 'gx1_direction_decimal_targets_result_v1'
        and materialization.get('status') == 'COMPLETE'
        and materialization.get('plan') == refs['materialization_plan']
        and materialization.get('target') == refs['corrected']
        and materialization.get('target_identity') == 'exact_original_decimal_midpoint_v1'
        and materialization.get('parent_rows') == decimal.PARENTS
        and materialization.get('horizons_minutes') == list(range(1, 61))
        and materialization.get('source_clock_bid_ask_and_decimal_grid_parity') is True
        and materialization.get('independent_audit_confusions_reproduced') is True
        and materialization.get('all_false_moves_removed') == 976
        and materialization.get('nonzero_sign_flips') == 0
        and materialization.get('test_data_used') is False, 'MATERIALIZATION_RECEIPT')
    originals = materialization_plan['inputs']
    require(materialization_plan.get('schema_version') == 'gx1_direction_decimal_targets_plan_v1'
        and originals.get('legacy_targets') == refs['legacy']
        and originals.get('dense_targets') == refs['original']
        and originals.get('audit_result') == refs['audit_result'], 'ORIGINAL_LINEAGE')
    require(audit.get('schema_version') == 'gx1_direction_decimal_sign_audit_result_v1'
        and audit.get('status') == 'COMPLETE' and audit.get('test_data_used') is False
        and audit.get('vendor_to_canonical_float64_and_clock_parity') is True
        and audit.get('target_source_rows_and_masks_parity') is True, 'AUDIT_RECEIPT')
    require(oof.get('schema_version') == 'gx1_native_direction_prefix_measurement_v1'
        and oof.get('role') == 'OOF' and oof.get('test_data_used') is False
        and oof.get('predictions') == refs['oof_predictions']
        and oof.get('fold') == {'fold_id': 1, 'prefix_rows': decimal.FOLD_TRAIN,
            'oof_start_row': decimal.FOLD_TRAIN, 'oof_end_row': decimal.FOLD_LATER_END,
            'train_cutoff_ns': decimal.TRAIN_CUTOFF, 'oof_cutoff_ns': decimal.LATER_CUTOFF}, 'OOF_RECEIPT')


def compare_oof_rows(oof, original, *, first, last):
    take = slice(first, last)
    for key in ('parent_rows', 'bar_start_ns', 'decision_ns', 'terminal_time_ns', 'observed', 'sign'):
        require(key in oof and oof[key].dtype == original[key].dtype
            and np.array_equal(oof[key], original[key][take]), 'OOF_JOIN:' + key)
    return last - first


def bps_difference(original, corrected):
    mask = original['observed']
    require(np.array_equal(mask, corrected['observed']), 'BPS_MASK')
    diff = np.abs(original['mid_return_bps'][mask] - corrected['mid_return_bps'][mask])
    require(len(diff) > 0 and np.isfinite(diff).all(), 'BPS_DOMAIN')
    return {'observed_cells': int(len(diff)), 'different_float_values': int(np.count_nonzero(diff)),
        'absolute_difference_bps_quantiles': {str(q): float(np.quantile(diff, q, method='linear'))
            for q in (0.5, 0.95, 0.99, 1.0)}}


def main(argv=None):
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('plan', 'plan-sha256', 'policy-sha256', 'source-head'):
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args(argv)
    from gx1.contracts import gx1_capped_execution_v1 as cap
    from gx1.contracts import immutable_event_authority_v1 as publication
    from gx1.contracts import local_random_access_campaign_v2 as binding
    from gx1.contracts.entry_model_native_bundle_commit_v1 import publish_bundle_directory_noreplace
    from gx1.contracts import entry_native_direction_decimal_data_v1 as consumer
    from gx1.scripts import audit_entry_direction_candle_midpoints_v1 as candle
    from gx1.scripts import audit_entry_direction_decimal_sign_v1 as decimal
    from gx1.scripts import materialize_entry_direction_decimal_targets_v1 as materialize
    require(not any(name.split('.')[0] in ('sklearn', 'joblib') for name in sys.modules)
        and 'torch' in sys.modules and not sys.modules['torch'].cuda.is_initialized(), 'PRELOADED_MODEL')
    initial = candle.git_state()
    require(initial == (args.source_head, 'work/gx1-current', ''), 'INITIAL_SOURCE')
    plan_ref = {'path': str(candle.regular(args.plan)), 'sha256': args.plan_sha256}
    policy_ref = {'path': str(REPO / 'NEXT_RUN_POLICY.json'), 'sha256': args.policy_sha256}
    with binding.fresh_file_verification_scope():
        plan = binding.read_bound_json(Path(plan_ref['path']), plan_ref['sha256'])
        require(set(plan) == {'schema_version', 'stage', 'design', 'inputs', 'sources', 'output_dir', 'wall_seconds', 'test_data_used'}
            and plan['schema_version'] == 'gx1_direction_decimal_consumer_plan_v1' and plan['stage'] == STAGE
            and plan['wall_seconds'] == 600 and plan['test_data_used'] is False, 'PLAN')
        require(plan['design']['path'] == str(REPO / 'docs/DIRECTION_DECIMAL_CONSUMER_AUDIT_20261028.json'), 'DESIGN_PATH')
        design = binding.read_bound_json(Path(plan['design']['path']), plan['design']['sha256'])
        require(design['schema_version'] == 'gx1_direction_decimal_consumer_design_v1'
            and design['inputs'] == plan['inputs'] and design['test_data_used'] is False, 'DESIGN')
        policy = binding.read_bound_json(Path(policy_ref['path']), policy_ref['sha256'])
        gate = policy[STAGE]
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
        require(set(plan['inputs']) == {'legacy', 'original', 'corrected', 'audit_result',
            'materialization_result', 'materialization_plan', 'oof_report', 'oof_predictions'}, 'INPUT_SET')
        meta = {}
        for key in ('audit_result', 'materialization_result', 'materialization_plan', 'oof_report'):
            ref = plan['inputs'][key]
            meta[key] = binding.read_bound_json(Path(ref['path']), ref['sha256'])
        require_receipts(plan, meta['materialization_result'], meta['materialization_plan'], meta['audit_result'],
            meta['oof_report'], decimal=decimal)
        control_snapshot = binding.capture_verified_file_snapshot()
    proof, lock = cap.require_capped_cpu_audit_execution(), cap.require_capped_lock_ancestry()
    require(lock['scope'] == proof['cgroup_path'][len('/sys/fs/cgroup'):], 'CAPPED_LOCK_SCOPE')
    out = Path(plan['output_dir'])
    require(ROOT.resolve(strict=True) == ROOT and out.parent == ROOT
        and out.name == 'DIRECTION_DECIMAL_CONSUMER_AUDIT_20261028_001'
        and not out.exists() and not out.is_symlink(), 'OUTPUT_SCOPE')
    out.mkdir(exist_ok=False)
    def publish(path, value):
        publication._publish_bytes(path, binding.canonical_bytes(value))
    publish(out / 'INVOCATION_CLAIM.json', {'plan': plan_ref, 'policy': policy_ref,
        'source_head': args.source_head, 'execution': proof, 'lock': lock})
    payloads = [plan['inputs'][k] for k in ('legacy', 'original', 'corrected', 'oof_predictions')]
    require(len({ref['path'] for ref in payloads}) == len(payloads), 'DISTINCT_INPUTS')
    metadata = [plan_ref['path'], policy_ref['path'], plan['design']['path']]
    metadata += [plan['inputs'][k]['path'] for k in ('audit_result', 'materialization_result', 'materialization_plan', 'oof_report')]
    sys.addaudithook(candle.make_guard(out, [ref['path'] for ref in payloads], metadata,
        source_paths=[ref['path'] for ref in plan['sources'].values()]))
    started = time.monotonic(); previous_handler = signal.getsignal(signal.SIGALRM); previous_profile = sys.getprofile()
    def expired(*unused):
        raise RuntimeError('DIRECTION_DECIMAL_CONSUMER_WALL_LIMIT')
    def deny_model_calls(frame, event, arg):
        candle.no_queries_or_models(frame, event, arg)
        candle.no_native_models(frame, event, arg)
    signal.signal(signal.SIGALRM, expired); signal.alarm(plan['wall_seconds']); sys.setprofile(deny_model_calls)
    stage = out / '.OUTPUT.staging'
    try:
        stage.mkdir(exist_ok=False)
        with binding.fresh_file_verification_scope():
            for ref in payloads:
                candle.regular(ref['path']); binding.require_binding(ref, label='input')
            legacy = candle.read_targets(plan['inputs']['legacy']['path'], np=np)
            original = decimal._read_target(plan['inputs']['original']['path'])
            corrected = decimal._read_target(plan['inputs']['corrected']['path'])
            joined = consumer.bind_decimal_grid_dense_targets(axes={k: legacy[k] for k in consumer.AXES},
                legacy=legacy, original_dense=original, corrected_dense=corrected,
                target_identity=meta['materialization_result']['target_identity'])
            parity = materialize.require_corrected_target(original, corrected, meta['audit_result'], decimal=decimal)
            require(parity == {'all_false_moves_removed': 976, 'nonzero_sign_flips': 0}, 'AUDIT_COUNTS')
            with np.load(plan['inputs']['oof_predictions']['path'], allow_pickle=False) as archive:
                names = ('parent_rows', 'bar_start_ns', 'decision_ns', 'terminal_time_ns', 'observed', 'sign')
                require(all(k in archive.files for k in names), 'OOF_FIELDS')
                oof = {k: archive[k].copy() for k in names}
            oof_rows = compare_oof_rows(oof, original, first=decimal.FOLD_TRAIN, last=decimal.FOLD_LATER_END)
            differences = bps_difference(original, corrected)
            require(all(not value.flags.writeable for value in joined.values()), 'READ_ONLY_OUTPUT')
            input_snapshot = binding.capture_verified_file_snapshot()
        input_snapshot.require_unchanged(); control_snapshot.require_unchanged()
        require(candle.git_state() == initial and cap.require_capped_lock_ancestry() == lock, 'FINAL_SOURCE_LOCK')
        result = {'schema_version': 'gx1_direction_decimal_consumer_result_v1', 'status': 'COMPLETE',
            'plan': plan_ref, 'source_head': args.source_head, 'target_identity': consumer.DECIMAL_TARGET_IDENTITY,
            'parents': decimal.PARENTS, 'horizons_minutes': list(range(1, 61)),
            'legacy_anchor_bytes_verified_on_original_only': True,
            'corrected_axes_clocks_rows_masks_and_domain_verified': True,
            'independent_60_horizon_confusion_matrices_reproduced': True,
            'saved_oof_rows_rejoined_to_original': oof_rows, 'bps_difference': differences, **parity,
            'scope': 'Hash-bound original and corrected pre-2026 targets and saved fold1 OOF identities; no model.',
            'limits': 'Structural and lineage qualification only; future use requires a separately admitted authenticated consumer, fresh initial baseline and no TEST.',
            'new_model_forwards': 0, 'optimizer_steps': 0, 'test_data_used': False}
        publish(stage / 'RESULT.json', result)
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
