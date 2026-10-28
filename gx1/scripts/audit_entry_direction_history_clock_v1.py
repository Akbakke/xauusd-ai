"""Measure real M5 input-window gaps and fixed direction scores without a model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import sys
import time
import zipfile

import numpy as np

REPO = Path('/home/andre2/src/GX1_CURRENT')
ROOT = Path('/home/andre2/GX1_RUNS/INDEPENDENT_DIRECTION_EDGE_20261027_01a17d63')
STAGE = 'independent_direction_history_clock_audit_20261028_001'
BAR_NS = 300 * 10**9
MINUTE_NS = 60 * 10**9
PARENTS, TRAIN_END, LATER_END = 118249, 47602, 64890
CUTOFF_NS, TEST_NS = 1767225600000000000, 1782864000000000000
LATER_CUTOFF_NS = 1743465600000000000
SOURCE_PATHS = (
    'gx1/scripts/audit_entry_direction_history_clock_v1.py',
    'gx1/scripts/audit_entry_direction_candle_midpoints_v1.py',
    'gx1/contracts/entry_direction_probability_reference_v1.py',
    'gx1/contracts/entry_native_direction_context_report_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'gx1/contracts/local_random_access_campaign_v2.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'gx1/contracts/entry_model_native_bundle_commit_v1.py',
    'scripts/gx1_capped_run.sh',
    'tests/test_audit_entry_direction_history_clock_v1.py',
)


def require(ok, reason):
    if not bool(ok):
        raise RuntimeError('DIRECTION_HISTORY_CLOCK_' + reason)


def history_clock_gaps(source_times, starts, *, window_rows=96):
    """Measure actual elapsed time between ordered source rows, preserving gaps."""
    require(isinstance(source_times, np.ndarray) and source_times.dtype == np.int64 and source_times.ndim == 1
        and isinstance(starts, np.ndarray) and starts.dtype == np.int64 and starts.ndim == 1
        and type(window_rows) is int and window_rows >= 2 and len(starts) > 0
        and len(source_times) >= window_rows and np.all(source_times % BAR_NS == 0)
        and np.all(np.diff(source_times) > 0) and np.all(np.diff(starts) > 0), 'CLOCK_DOMAIN')
    pos = np.searchsorted(source_times, starts)
    require(np.all(pos >= window_rows-1) and np.all(pos < len(source_times))
        and np.array_equal(source_times[pos], starts), 'PARENT_SOURCE_JOIN')
    delta = np.diff(source_times)
    gap_prefix = np.r_[0, np.cumsum(delta > BAR_NS, dtype=np.int64)]
    count = gap_prefix[pos] - gap_prefix[pos-window_rows+1]
    short_rows = min(12, window_rows-1)
    recent_count = gap_prefix[pos] - gap_prefix[pos-short_rows]
    largest_missing = np.zeros(len(pos), np.int64)
    for lag in range(1, window_rows):
        absent = (source_times[pos-lag+1] - source_times[pos-lag] - BAR_NS) // MINUTE_NS
        largest_missing = np.maximum(largest_missing, absent)
    elapsed = (source_times[pos] - source_times[pos-window_rows+1] + BAR_NS) // MINUTE_NS
    return {'gap_count': count, 'recent_gap_count': recent_count,
        'largest_missing_minutes': largest_missing, 'elapsed_minutes': elapsed}


def summarize(values):
    require(len(values['gap_count']) > 0, 'EMPTY_GROUP')
    gap, recent, largest, elapsed = (values[k] for k in
        ('gap_count', 'recent_gap_count', 'largest_missing_minutes', 'elapsed_minutes'))
    return {'parents': len(gap), 'with_any_gap': int((gap > 0).sum()),
        'with_gap_in_last_12_intervals': int((recent > 0).sum()),
        'gap_count_distribution': {'0': int((gap == 0).sum()), '1': int((gap == 1).sum()),
            '2_or_more': int((gap >= 2).sum())},
        'largest_missing_minutes_quantiles': {str(q): float(np.quantile(largest, q, method='linear')) for q in (0.5, 0.9, 0.99, 1.0)},
        'elapsed_minutes_quantiles': {str(q): float(np.quantile(elapsed, q, method='linear')) for q in (0.5, 0.9, 0.99, 1.0)}}


def cohort_scores(candidate, gap_counts, report):
    from gx1.contracts.entry_native_direction_context_report_v1 import _panel
    from gx1.contracts.entry_direction_probability_reference_v1 import direction_rank_diagnostics
    require(len(gap_counts) == len(candidate['parent_rows']) and gap_counts.dtype == np.int64, 'COHORT_AXIS')
    result = []
    for j in (4, 59):
        mature = candidate['terminal_time_ns'][:, j] < LATER_CUTOFF_NS
        entire = _panel(candidate['logit_up'][:, j], candidate['sign'][:, j], candidate['observed'][:, j], mature)
        original = report['model_metrics']['per_horizon'][j]
        require(original['counts']['eligible_cells'] == entire['counts']['eligible_cells']
            and all(np.isclose(original[k], entire[k], rtol=0, atol=1e-12) for k in ('accuracy','logloss','brier')), 'ORIGINAL_SCORE_REPLAY')
        groups = []
        for name, take in (('no_gap', gap_counts == 0), ('one_or_more_gaps', gap_counts > 0)):
            panel = _panel(candidate['logit_up'][take, j], candidate['sign'][take, j],
                candidate['observed'][take, j], mature[take])
            eligible = take & candidate['observed'][:, j] & (candidate['sign'][:, j] != 0) & mature
            labels = np.ascontiguousarray((candidate['sign'][eligible, j] == 1).astype(np.int8))
            scores = np.ascontiguousarray(candidate['logit_up'][eligible, j], dtype=np.float64)
            auc = direction_rank_diagnostics(labels, scores, score_kind='logit_up')['auc']
            groups.append({'name': name, 'parents': int(take.sum()), 'eligible': len(labels),
                'auc': auc, 'accuracy': panel['accuracy'], 'logloss': panel['logloss'], 'brier': panel['brier']})
        result.append({'horizon_minutes': j+1, 'groups': groups})
    return result


def main(argv=None):
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('plan', 'plan-sha256', 'policy-sha256', 'source-head'):
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args(argv)
    import pyarrow as arrow
    import pyarrow.parquet as parquet
    import pyarrow.dataset
    from gx1.contracts import gx1_capped_execution_v1 as cap
    from gx1.contracts import immutable_event_authority_v1 as publication
    from gx1.contracts import local_random_access_campaign_v2 as binding
    from gx1.contracts.entry_model_native_bundle_commit_v1 import publish_bundle_directory_noreplace
    from gx1.contracts.entry_native_direction_context_report_v1 import _panel
    from gx1.contracts.entry_direction_probability_reference_v1 import direction_rank_diagnostics
    from gx1.scripts import audit_entry_direction_candle_midpoints_v1 as candle
    require(not any(name.split('.')[0] in ('torch', 'sklearn', 'joblib') for name in sys.modules), 'PRELOADED_MODEL')
    initial = candle.git_state()
    require(initial == (args.source_head, 'work/gx1-current', ''), 'INITIAL_SOURCE')
    plan_ref = {'path': str(candle.regular(args.plan)), 'sha256': args.plan_sha256}
    policy_ref = {'path': str(REPO/'NEXT_RUN_POLICY.json'), 'sha256': args.policy_sha256}
    with binding.fresh_file_verification_scope():
        plan = binding.read_bound_json(Path(plan_ref['path']), plan_ref['sha256'])
        require(set(plan) == {'schema_version', 'stage', 'design', 'inputs', 'sources', 'output_dir', 'wall_seconds', 'test_data_used'}
            and plan['schema_version'] == 'gx1_direction_history_clock_plan_v1' and plan['stage'] == STAGE
            and plan['wall_seconds'] == 600 and plan['test_data_used'] is False, 'PLAN')
        require(plan['design']['path'] == str(REPO/'docs/DIRECTION_HISTORY_CLOCK_AUDIT_20261028.json'), 'DESIGN_PATH')
        design = binding.read_bound_json(Path(plan['design']['path']), plan['design']['sha256'])
        require(design['schema_version'] == 'gx1_direction_history_clock_design_v1'
            and design['parents'] == PARENTS and design['sequence_rows'] == 96
            and design['test_data_used'] is False and plan['inputs'] == design['inputs'], 'DESIGN')
        policy = binding.read_bound_json(Path(policy_ref['path']), policy_ref['sha256'])
        gate = policy[STAGE]
        require(gate['status'] == 'ADMITTED' and gate['plan'] == plan_ref and gate['execution_authorized'] is True
            and gate['max_invocations'] == 1 and gate['claimed_invocations'] == gate['completed_invocations'] == 0
            and gate['relaunch_allowed'] is False, 'ONE_SHOT_POLICY')
        require(all(policy[key] is False for key in ('training_enabled','native_training_enabled',
            'full_epoch_training_allowed','full_val_allowed','native_launch_authorized')), 'GLOBAL_GATES')
        require(set(plan['sources']) == set(SOURCE_PATHS)
            and all(ref['path'] == str(REPO/name) for name,ref in plan['sources'].items()), 'SOURCE_SET')
        for ref in plan['sources'].values():
            candle.regular(ref['path']); binding.require_binding(ref, label='source')
        require(set(plan['inputs']) == {'surface_metadata','surface_manifest','surface','target','candidate_report','candidate_predictions'}, 'INPUT_SET')
        meta = {}
        for key in ('surface_metadata','surface_manifest','candidate_report'):
            ref = plan['inputs'][key]; meta[key] = binding.read_bound_json(Path(ref['path']), ref['sha256'])
        audit, surface_manifest, report = (meta[k] for k in ('surface_metadata','surface_manifest','candidate_report'))
        surface_ref, target_ref = plan['inputs']['surface'], plan['inputs']['target']
        require(audit['schema_version'] == 'entry_model_native_sequence_source_reconstruction_audit_v1'
            and audit['decision'] == 'PASS' and audit['feature_surface_path'] == surface_ref['path']
            and audit['feature_surface_sha256'] == surface_ref['sha256'] and audit['feature_surface_rows'] == 1140763
            and audit['feature_surface_manifest_path'] == plan['inputs']['surface_manifest']['path']
            and audit['feature_surface_manifest_sha256'] == plan['inputs']['surface_manifest']['sha256']
            and surface_manifest['output_parquet'] == surface_ref['path']
            and surface_manifest['output_parquet_sha256'] == surface_ref['sha256']
            and surface_manifest['anchor_timeframe'] == 'M5', 'SURFACE_BINDING')
        require(report['predictions'] == plan['inputs']['candidate_predictions'] and report['test_data_used'] is False
            and report['fold'] == {'fold_id':1,'prefix_rows':TRAIN_END,'oof_start_row':TRAIN_END,
                'oof_end_row':LATER_END,'train_cutoff_ns':1735689600000000000,'oof_cutoff_ns':LATER_CUTOFF_NS}, 'REPORT_BINDING')
        control_snapshot = binding.capture_verified_file_snapshot()
    proof, lock = cap.require_capped_cpu_audit_execution(), cap.require_capped_lock_ancestry()
    require(lock['scope'] == proof['cgroup_path'][len('/sys/fs/cgroup'):], 'CAPPED_LOCK_SCOPE')
    out = Path(plan['output_dir'])
    require(ROOT.resolve(strict=True) == ROOT and out.parent == ROOT
        and out.name == 'DIRECTION_HISTORY_CLOCK_AUDIT_20261028_001'
        and not out.exists() and not out.is_symlink(), 'OUTPUT_SCOPE')
    out.mkdir(exist_ok=False)
    def publish(path, value):
        publication._publish_bytes(path, binding.canonical_bytes(value))
    publish(out/'INVOCATION_CLAIM.json', {'plan':plan_ref,'policy':policy_ref,
        'source_head':args.source_head,'execution':proof,'lock':lock})
    payloads = [plan['inputs'][k] for k in ('surface','target','candidate_predictions')]
    require(len({x['path'] for x in payloads}) == 3, 'DISTINCT_PAYLOADS')
    metadata = [plan_ref['path'],policy_ref['path'],plan['design']['path']]
    metadata += [plan['inputs'][k]['path'] for k in ('surface_metadata','surface_manifest','candidate_report')]
    sys.addaudithook(candle.make_guard(out, [x['path'] for x in payloads], metadata,
        source_paths=[x['path'] for x in plan['sources'].values()]))
    started = time.monotonic();previous_handler=signal.getsignal(signal.SIGALRM);previous_profile=sys.getprofile()
    def expired(*unused):
        raise RuntimeError('DIRECTION_HISTORY_CLOCK_WALL_LIMIT')
    signal.signal(signal.SIGALRM,expired);signal.alarm(plan['wall_seconds']);sys.setprofile(candle.no_queries_or_models)
    stage=out/'.OUTPUT.staging'
    try:
        stage.mkdir(exist_ok=False)
        with binding.fresh_file_verification_scope():
            for ref in payloads:
                candle.regular(ref['path']);binding.require_binding(ref,label='input')
            with Path(surface_ref['path']).open('rb') as handle:
                table = parquet.read_table(handle, columns=['time'], use_threads=False)
            column = table['time']
            require(table.column_names == ['time'] and table.num_rows == audit['feature_surface_rows']
                and column.null_count == 0 and arrow.types.is_timestamp(column.type)
                and column.type.tz in (None,'UTC'), 'SOURCE_CLOCK_HEADER')
            times = np.ascontiguousarray(column.cast(arrow.timestamp('ns',tz=column.type.tz)).cast(arrow.int64()).to_numpy())
            require(times.dtype == np.int64 and np.all(times < TEST_NS), 'SOURCE_TEST_SEAL')
            with np.load(target_ref['path'],allow_pickle=False) as archive:
                parents, starts = archive['parent_rows'].copy(), archive['bar_start_ns'].copy()
            require(parents.dtype == starts.dtype == np.int64 and parents.shape == starts.shape == (PARENTS,)
                and np.array_equal(parents,np.arange(PARENTS,dtype=np.int64))
                and np.all(starts + BAR_NS < CUTOFF_NS), 'TARGET_AXES')
            gaps = history_clock_gaps(times,starts,window_rows=96)
            populations = {'all_pre2026':summarize(gaps),
                'fold1_train':summarize({k:v[:TRAIN_END] for k,v in gaps.items()}),
                'fold1_later':summarize({k:v[TRAIN_END:LATER_END] for k,v in gaps.items()})}
            with np.load(plan['inputs']['candidate_predictions']['path'],allow_pickle=False) as archive:
                names=('parent_rows','bar_start_ns','decision_ns','terminal_time_ns','observed','sign','logit_up')
                require(all(k in archive.files for k in names),'PREDICTION_FIELDS')
                candidate={k:archive[k].copy() for k in names}
            sl=slice(TRAIN_END,LATER_END)
            require(np.array_equal(candidate['parent_rows'],parents[sl]) and np.array_equal(candidate['bar_start_ns'],starts[sl])
                and np.array_equal(candidate['decision_ns'],starts[sl]+BAR_NS)
                and candidate['terminal_time_ns'].dtype==np.int64 and candidate['terminal_time_ns'].shape==(LATER_END-TRAIN_END,60)
                and np.array_equal(candidate['terminal_time_ns'],candidate['decision_ns'][:,None]+np.arange(1,61,dtype=np.int64)[None,:]*MINUTE_NS)
                and candidate['observed'].dtype==np.bool_ and candidate['sign'].dtype==np.int8
                and candidate['observed'].shape==candidate['sign'].shape==candidate['logit_up'].shape==(LATER_END-TRAIN_END,60)
                and np.isfinite(candidate['logit_up']).all(), 'CANDIDATE_JOIN')
            cohorts = cohort_scores(candidate,gaps['gap_count'][sl],report)
            input_snapshot=binding.capture_verified_file_snapshot()
        input_snapshot.require_unchanged();control_snapshot.require_unchanged()
        require(candle.git_state()==initial and cap.require_capped_lock_ancestry()==lock,'FINAL_SOURCE_LOCK')
        result={'schema_version':'gx1_direction_history_clock_result_v1','status':'COMPLETE','plan':plan_ref,
            'source_head':args.source_head,'source_rows':len(times),'window_rows':96,'nominal_elapsed_minutes':480,
            'populations':populations,'fixed_later_cohort_scores':cohorts,'original_scores_reproduced':True,
            'scope':'M5 input surface clock, 118249 pre-2026 parents; fixed 5m/60m logits on reused fold1.',
            'limits':'Gap group comparisons are descriptive and confounded; no interval, causality, trading or economic edge inference.',
            'new_model_forwards':0,'optimizer_steps':0,'test_data_used':False}
        publish(stage/'RESULT.json',result)
        publish(stage/'TERMINAL.json',{'status':'COMPLETE','exit_code':0,'source_unchanged':True,
            'inputs_unchanged':True,'elapsed_seconds':time.monotonic()-started,'test_data_used':False})
        publish_bundle_directory_noreplace(stage,out/'OUTPUT')
        publish(out/'TERMINAL.json',{'status':'COMPLETE','exit_code':0,'plan':plan_ref,'result_path':str(out/'OUTPUT/RESULT.json')})
        return 0
    except Exception as exc:
        publish(out/'FAILED_TERMINAL.json',{'status':'FAILED','exit_code':1,
            'error':type(exc).__name__+': '+str(exc),'relaunch_allowed':False})
        raise
    finally:
        sys.setprofile(previous_profile);signal.alarm(0);signal.signal(signal.SIGALRM,previous_handler)


if __name__ == '__main__':
    raise SystemExit(main())
