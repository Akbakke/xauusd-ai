"""Evaluate saved 60-output direction predictions; no model, fit or market reads."""
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
STAGE = 'independent_direction_dense_onepass_uncertainty_20261028_001'
ROWS = 17288
START, END = 47602, 64890
TRAIN_CUTOFF = 1735689600000000000
LATER_CUTOFF = 1743465600000000000
DAY_NS = 86_400_000_000_000
MINUTE_NS = 60 * 10**9
SOURCE_PATHS = (
    'gx1/scripts/score_entry_direction_dense_onepass_v1.py',
    'gx1/scripts/score_entry_direction_dense_minutes_v1.py',
    'gx1/scripts/compare_entry_direction_matched_exposure_v1.py',
    'gx1/scripts/compare_entry_direction_saved_bps_v1.py',
    'gx1/scripts/audit_entry_direction_candle_midpoints_v1.py',
    'gx1/scripts/materialize_entry_direction_dense_v1.py',
    'gx1/contracts/entry_direction_block_rank_v1.py',
    'gx1/contracts/entry_direction_probability_reference_v1.py',
    'gx1/contracts/entry_native_direction_context_report_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'gx1/contracts/local_random_access_campaign_v2.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'gx1/contracts/entry_model_native_bundle_commit_v1.py',
    'scripts/gx1_capped_run.sh',
    'tests/test_score_entry_direction_dense_onepass_v1.py',
)


def require(ok, reason):
    if not bool(ok):
        raise RuntimeError('DENSE_ONEPASS_UNCERTAINTY_' + reason)


def compare_saved_predictions(candidate, baseline, *, cutoff, replicates=10000, minimum_blocks=8,
                              candidate_report=None, baseline_report=None):
    from gx1.contracts import entry_direction_block_rank_v1 as rank
    from gx1.contracts import entry_direction_probability_reference_v1 as probability
    from gx1.contracts.entry_native_direction_context_report_v1 import _panel
    from gx1.scripts import compare_entry_direction_matched_exposure_v1 as matched
    from gx1.scripts import compare_entry_direction_saved_bps_v1 as saved
    from gx1.scripts.score_entry_direction_dense_minutes_v1 import coarsen_block_pairs, simultaneous_intervals

    n = len(candidate['parent_rows']); anchors = np.array([0, 4, 14], np.int64)
    for key in ('parent_rows', 'bar_start_ns', 'decision_ns'):
        require(candidate[key].dtype == baseline[key].dtype == np.int64
            and candidate[key].shape == baseline[key].shape == (n,)
            and np.array_equal(candidate[key], baseline[key]), 'AXIS:' + key)
    require(n > 0 and np.all(np.diff(candidate['parent_rows']) > 0)
        and np.all(np.diff(candidate['decision_ns']) > 0)
        and np.array_equal(candidate['decision_ns'], candidate['bar_start_ns'] + 5 * MINUTE_NS)
        and np.all(candidate['decision_ns'] < cutoff), 'DECISION_CLOCK')
    for key, dtype in (('terminal_time_ns', np.int64), ('observed', np.bool_), ('sign', np.int8)):
        require(candidate[key].dtype == baseline[key].dtype == dtype
            and candidate[key].shape == (n, 60) and baseline[key].shape == (n, 3)
            and np.array_equal(candidate[key][:, anchors], baseline[key]), 'ANCHOR:' + key)
    require(np.array_equal(candidate['terminal_time_ns'], candidate['decision_ns'][:, None]
        + np.arange(1, 61, dtype=np.int64)[None, :] * MINUTE_NS)
        and np.isin(candidate['sign'], (-1, 0, 1)).all()
        and np.all(candidate['sign'][~candidate['observed']] == 0), 'TARGET_CLOCK_DOMAIN')
    for value, width in ((candidate['logit_up'], 60), (baseline['logit_up'], 3)):
        require(value.dtype in (np.float32, np.float64) and value.shape == (n, width)
            and np.isfinite(value).all(), 'LOGITS')
    days = candidate['decision_ns'] // DAY_NS
    day_values, day_ids = np.unique(days, return_inverse=True)
    week_values, week_ids = np.unique((days + 3) // 7, return_inverse=True)
    require(len(day_values) >= minimum_blocks and len(week_values) >= minimum_blocks, 'MINIMUM_BLOCKS')
    day_to_week = np.searchsorted(week_values, (day_values + 3) // 7)
    day_weights = saved.block_weights(len(day_values), seed=20261027, np=np, replicates=replicates)
    week_weights = saved.block_weights(len(week_values), seed=20261028, np=np, replicates=replicates)
    points = np.empty(60, np.float64)
    day_draws = np.empty((replicates, 60), np.float64); week_draws = np.empty_like(day_draws)
    panels = []; primary = None
    for j in range(60):
        mature = candidate['terminal_time_ns'][:, j] < cutoff
        take = candidate['observed'][:, j] & (candidate['sign'][:, j] != 0) & mature
        labels = np.ascontiguousarray((candidate['sign'][take, j] == 1).astype(np.int8))
        z = np.ascontiguousarray(candidate['logit_up'][take, j], dtype=np.float64)
        require(len(labels) > 0 and np.unique(labels).size == 2, 'BOTH_CLASSES')
        used, compact = np.unique(day_ids[take], return_inverse=True)
        summary = rank.block_rank_pairs(labels, z, np.ascontiguousarray(compact, dtype=np.int64))
        dn, dd = rank.weighted_rank_pairs(summary, day_weights[:, used])
        used_weeks, week_summary = coarsen_block_pairs(summary, day_to_week[used])
        wn, wd = rank.weighted_rank_pairs(week_summary, week_weights[:, used_weeks])
        require(np.all(dd > 0) and np.all(wd > 0), 'UNDEFINED_BOOTSTRAP')
        auc = probability.direction_rank_diagnostics(labels, z, score_kind='logit_up')['auc']
        points[j] = auc; day_draws[:, j] = dn / (2 * dd); week_draws[:, j] = wn / (2 * wd)
        require(np.isclose(auc, summary['twice_wins'].sum() /
            (2 * summary['positive'].sum() * summary['negative'].sum()), rtol=0, atol=1e-15), 'RANK_PARITY')
        panel = _panel(candidate['logit_up'][:, j], candidate['sign'][:, j], candidate['observed'][:, j], mature)
        if candidate_report is not None:
            original = candidate_report['model_metrics']['per_horizon'][j]
            require(original['horizon_minutes'] == j + 1 and original['counts']['eligible_cells'] == len(labels)
                and np.isclose(original['rank']['auc'], auc, rtol=0, atol=1e-12)
                and all(np.isclose(original[k], panel[k], rtol=0, atol=1e-12)
                    for k in ('accuracy', 'logloss', 'brier')), 'CANDIDATE_SCORE_REPLAY')
        entry = {'horizon_minutes': j + 1, 'eligible': len(labels), 'auc': float(auc),
            'accuracy': panel['accuracy'], 'logloss': panel['logloss'], 'brier': panel['brier'],
            'counts': panel['counts']}
        if candidate_report is not None:
            entry['train_prior_up'] = candidate_report['own_prefix_prior']['probability_up'][j]
            entry['prior_logloss'] = candidate_report['prior_metrics']['per_horizon'][j]['logloss']
            entry['prior_brier'] = candidate_report['prior_metrics']['per_horizon'][j]['brier']
        panels.append(entry)
        if j == 4:
            old = np.ascontiguousarray(baseline['logit_up'][take, 1], dtype=np.float64)
            old_rank = probability.direction_rank_diagnostics(labels, old, score_kind='logit_up')['auc']
            old_panel = _panel(baseline['logit_up'][:, 1], baseline['sign'][:, 1], baseline['observed'][:, 1], mature)
            if baseline_report is not None:
                prior = baseline_report['model_metrics']['per_horizon'][1]
                require(prior['horizon_minutes'] == 5 and prior['counts']['eligible_cells'] == len(labels)
                    and all(np.isclose(prior[k], old_panel[k], rtol=0, atol=1e-12)
                        for k in ('accuracy', 'logloss', 'brier')), 'BASELINE_SCORE_REPLAY')
            groups = []
            for name, ids, weights in (('UTC_DAY', day_ids, day_weights), ('UTC_WEEK_MONDAY', week_ids, week_weights)):
                draws = matched.paired_draws(labels, old, z, ids[take], weights)
                groups.append({'scheme': name, 'blocks': weights.shape[1],
                    'candidate_minus_baseline': {metric: matched.interval(draws[metric], .05/4)
                        for metric in ('auc', 'logloss', 'brier')},
                    'candidate_auc': matched.interval(draws['three_auc'], .05/4)})
            primary = {'horizon_minutes': 5, 'eligible': len(labels),
                'baseline': {'auc': float(old_rank), 'accuracy': old_panel['accuracy'],
                    'logloss': old_panel['logloss'], 'brier': old_panel['brier']},
                'candidate': {'auc': float(auc), 'accuracy': panel['accuracy'],
                    'logloss': panel['logloss'], 'brier': panel['brier']},
                'candidate_minus_baseline': {'auc': float(auc-old_rank),
                    'logloss': float(panel['logloss']-old_panel['logloss']),
                    'brier': float(panel['brier']-old_panel['brier'])}, 'intervals': groups}
    day_band = simultaneous_intervals(points, day_draws); week_band = simultaneous_intervals(points, week_draws)
    for j, panel in enumerate(panels):
        panel['auc_simultaneous_day'] = {'low': float(day_band['low'][j]), 'high': float(day_band['high'][j])}
        panel['auc_simultaneous_week'] = {'low': float(week_band['low'][j]), 'high': float(week_band['high'][j])}
        panel['both_lower_bounds_above_half'] = bool(day_band['low'][j] > .5 and week_band['low'][j] > .5)
    primary['auc_improvement_supported'] = all(x['candidate_minus_baseline']['auc']['low'] > 0 for x in primary['intervals'])
    primary['candidate_above_half_supported'] = all(x['candidate_auc']['low'] > .5 for x in primary['intervals'])
    return {'panels': panels, 'primary_five_minute': primary,
        'simultaneous_day_half_width': day_band['half_width'], 'simultaneous_week_half_width': week_band['half_width'],
        'day_blocks': len(day_values), 'week_blocks': len(week_values), 'replicates': replicates,
        'scope': 'One reused development fold, fixed original float labels; no horizon or threshold selection.',
        'limits': 'Calendar block exchangeability unproved; combined geometry and 60-head changes cannot be attributed separately. Labels include the measured decimal ties; no fills, FLAT quality or economic edge.',
        'test_data_used': False, 'economic_edge_qualified': False}


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
    from gx1.contracts.entry_native_direction_context_report_v1 import CONTEXT_SPEC, _panel
    from gx1.contracts.entry_direction_probability_reference_v1 import direction_rank_diagnostics
    from gx1.contracts.entry_direction_block_rank_v1 import block_rank_pairs, weighted_rank_pairs
    from gx1.scripts import audit_entry_direction_candle_midpoints_v1 as candle
    from gx1.scripts import compare_entry_direction_matched_exposure_v1 as matched
    from gx1.scripts import compare_entry_direction_saved_bps_v1 as saved
    from gx1.scripts.score_entry_direction_dense_minutes_v1 import coarsen_block_pairs, simultaneous_intervals
    require(not any(name.split('.')[0] in ('torch', 'sklearn', 'joblib') for name in sys.modules), 'PRELOADED_MODEL')
    initial = candle.git_state()
    require(initial == (args.source_head, 'work/gx1-current', ''), 'INITIAL_SOURCE')
    plan_ref = {'path': str(candle.regular(args.plan)), 'sha256': args.plan_sha256}
    policy_ref = {'path': str(REPO/'NEXT_RUN_POLICY.json'), 'sha256': args.policy_sha256}
    with binding.fresh_file_verification_scope():
        plan = binding.read_bound_json(Path(plan_ref['path']), plan_ref['sha256'])
        require(set(plan) == {'schema_version', 'stage', 'design', 'inputs', 'sources', 'output_dir', 'wall_seconds', 'test_data_used'}
            and plan['schema_version'] == 'gx1_dense_onepass_uncertainty_plan_v1' and plan['stage'] == STAGE
            and plan['wall_seconds'] == 600 and plan['test_data_used'] is False, 'PLAN')
        require(plan['design']['path'] == str(REPO/'docs/DIRECTION_DENSE_ONEPASS_EVALUATION_20261028.json'), 'DESIGN_PATH')
        design = binding.read_bound_json(Path(plan['design']['path']), plan['design']['sha256'])
        require(design['schema_version'] == 'gx1_dense_onepass_direction_evaluation_protocol_v1'
            and design['test_data_used'] is False and design['fold'] == {
                'fold_id': 1, 'prefix_rows': START, 'oof_start_row': START, 'oof_end_row': END,
                'train_cutoff_ns': TRAIN_CUTOFF, 'oof_cutoff_ns': LATER_CUTOFF}, 'DESIGN')
        policy = binding.read_bound_json(Path(policy_ref['path']), policy_ref['sha256'])
        gate = policy[STAGE]
        require(gate['status'] == 'ADMITTED' and gate['plan'] == plan_ref and gate['execution_authorized'] is True
            and gate['max_invocations'] == 1 and gate['claimed_invocations'] == gate['completed_invocations'] == 0
            and gate['relaunch_allowed'] is False, 'ONE_SHOT_POLICY')
        require(all(policy[key] is False for key in ('training_enabled', 'native_training_enabled',
            'full_epoch_training_allowed', 'full_val_allowed', 'native_launch_authorized')), 'GLOBAL_GATES')
        require(set(plan['sources']) == set(SOURCE_PATHS)
            and all(ref['path'] == str(REPO/name) for name, ref in plan['sources'].items()), 'SOURCE_SET')
        for ref in plan['sources'].values():
            candle.regular(ref['path']); binding.require_binding(ref, label='source')
        require(set(plan['inputs']) == {'candidate_report', 'baseline_report', 'baseline_rank_review',
            'candidate_predictions', 'baseline_predictions'}, 'INPUT_SET')
        reports = {}
        for key in ('candidate_report', 'baseline_report', 'baseline_rank_review'):
            ref = plan['inputs'][key]; reports[key] = binding.read_bound_json(Path(ref['path']), ref['sha256'])
        candidate_report, baseline_report, rank_review = (reports[k] for k in
            ('candidate_report', 'baseline_report', 'baseline_rank_review'))
        require(candidate_report['predictions'] == plan['inputs']['candidate_predictions']
            and baseline_report['predictions'] == plan['inputs']['baseline_predictions']
            and candidate_report['fold'] == baseline_report['fold'] == design['fold']
            and all(r['test_data_used'] is False for r in reports.values()), 'REPORT_BINDINGS')
        reference = next(p for p in rank_review['panels'] if p['horizon_minutes'] == 5)['arms']['mixed']
        require(rank_review['original_scores_reproduced'] is True and reference['auc'] is not None, 'BASELINE_RANK_REFERENCE')
        control_snapshot = binding.capture_verified_file_snapshot()
    proof, lock = cap.require_capped_cpu_audit_execution(), cap.require_capped_lock_ancestry()
    require(lock['scope'] == proof['cgroup_path'][len('/sys/fs/cgroup'):], 'CAPPED_LOCK_SCOPE')
    out = Path(plan['output_dir'])
    require(ROOT.resolve(strict=True) == ROOT and out.parent == ROOT
        and out.name == 'DENSE_ONEPASS_UNCERTAINTY_20261028_001'
        and not out.exists() and not out.is_symlink(), 'OUTPUT_SCOPE')
    out.mkdir(exist_ok=False)
    def publish(path, value):
        publication._publish_bytes(path, binding.canonical_bytes(value))
    publish(out/'INVOCATION_CLAIM.json', {'plan': plan_ref, 'policy': policy_ref,
        'source_head': args.source_head, 'execution': proof, 'lock': lock})
    payloads = [plan['inputs']['candidate_predictions'], plan['inputs']['baseline_predictions']]
    require(len({r['path'] for r in payloads}) == 2, 'DISTINCT_PAYLOADS')
    metadata = [plan_ref['path'], policy_ref['path'], plan['design']['path']]
    metadata += [plan['inputs'][k]['path'] for k in ('candidate_report', 'baseline_report', 'baseline_rank_review')]
    sys.addaudithook(candle.make_guard(out, [x['path'] for x in payloads], metadata,
        source_paths=[x['path'] for x in plan['sources'].values()]))
    started = time.monotonic(); previous_handler = signal.getsignal(signal.SIGALRM); previous_profile = sys.getprofile()
    def expired(*unused):
        raise RuntimeError('DENSE_ONEPASS_UNCERTAINTY_WALL_LIMIT')
    signal.signal(signal.SIGALRM, expired); signal.alarm(plan['wall_seconds'])
    sys.setprofile(candle.no_queries_or_models)
    stage = out/'.OUTPUT.staging'
    try:
        stage.mkdir(exist_ok=False)
        with binding.fresh_file_verification_scope():
            for ref in payloads:
                candle.regular(ref['path']); binding.require_binding(ref, label='prediction')
            baseline = matched._read_prediction(plan['inputs']['baseline_predictions'], context_spec=CONTEXT_SPEC)
            with Path(plan['inputs']['candidate_predictions']['path']).open('rb') as handle:
                with zipfile.ZipFile(handle) as archive:
                    require(sum(x.file_size for x in archive.infolist()) <= 128 * 1024**2, 'CANDIDATE_ARCHIVE_SIZE')
                handle.seek(0)
                with np.load(handle, allow_pickle=False) as archive:
                    names = {'parent_rows', 'bar_start_ns', 'decision_ns', 'terminal_time_ns', 'observed', 'sign', 'logit_up'}
                    require(set(archive.files) == names | {'context__' + name for name, _ in CONTEXT_SPEC}, 'CANDIDATE_MEMBERS')
                    candidate = {name: archive[name].copy() for name in names}
            require(np.array_equal(candidate['parent_rows'], np.arange(START, END, dtype=np.int64))
                and np.all(candidate['decision_ns'] >= TRAIN_CUTOFF) and np.all(candidate['decision_ns'] < LATER_CUTOFF), 'EXACT_FOLD')
            result = compare_saved_predictions(candidate, baseline, cutoff=LATER_CUTOFF,
                candidate_report=candidate_report, baseline_report=baseline_report)
            primary = result['primary_five_minute']
            require(np.isclose(primary['baseline']['auc'], reference['auc'], rtol=0, atol=1e-12), 'BASELINE_RANK_REPLAY')
            input_snapshot = binding.capture_verified_file_snapshot()
        input_snapshot.require_unchanged(); control_snapshot.require_unchanged()
        require(candle.git_state() == initial and cap.require_capped_lock_ancestry() == lock, 'FINAL_SOURCE_LOCK')
        result.update(schema_version='gx1_dense_onepass_uncertainty_result_v1', status='COMPLETE',
            plan=plan_ref, source_head=args.source_head, original_scores_reproduced=True,
            original_baseline_rank_reproduced=True, new_model_forwards=0, optimizer_steps=0)
        publish(stage/'RESULT.json', result)
        publish(stage/'TERMINAL.json', {'status': 'COMPLETE', 'exit_code': 0, 'source_unchanged': True,
            'inputs_unchanged': True, 'elapsed_seconds': time.monotonic()-started, 'test_data_used': False})
        publish_bundle_directory_noreplace(stage, out/'OUTPUT')
        publish(out/'TERMINAL.json', {'status': 'COMPLETE', 'exit_code': 0,
            'plan': plan_ref, 'result_path': str(out/'OUTPUT/RESULT.json')})
        return 0
    except Exception as exc:
        publish(out/'FAILED_TERMINAL.json', {'status': 'FAILED', 'exit_code': 1,
            'error': type(exc).__name__ + ': ' + str(exc), 'relaunch_allowed': False})
        raise
    finally:
        sys.setprofile(previous_profile); signal.alarm(0); signal.signal(signal.SIGALRM, previous_handler)


if __name__ == '__main__':
    raise SystemExit(main())
