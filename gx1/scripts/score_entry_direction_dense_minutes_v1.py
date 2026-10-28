"""Score a fixed saved five-minute signal against each 1–60-minute target.

This is a four-fold development audit, not 60 independently trained outputs.
All AUC pairs stay within fold; block resamples are shared across all horizons.
"""
from __future__ import annotations

import io
import json
import numpy as np

from gx1.contracts import entry_direction_block_rank_v1 as rank
from gx1.contracts import entry_direction_probability_reference_v1 as probability
from gx1.scripts import compare_entry_direction_saved_bps_v1 as saved
from gx1.scripts import materialize_entry_direction_dense_v1 as dense

DAY_NS = 86_400_000_000_000
REPLICATES = 10000
DAY_SEED, WEEK_SEED, FOLD_SEED_OFFSET = 20261027, 20261028, 100


def require(ok, reason):
    if not ok:
        raise RuntimeError('DIRECTION_DENSE_SCORE_' + reason)


def coarsen_block_pairs(summary, day_to_week):
    """Sum the exact within-day pair table into week blocks, including ties."""
    day_to_week = np.asarray(day_to_week)
    k = len(summary['positive'])
    require(day_to_week.ndim == 1 and day_to_week.shape == (k,)
        and day_to_week.dtype.kind == 'i' and np.all(day_to_week >= 0), 'WEEK_MAP')
    weeks, inverse = np.unique(day_to_week, return_inverse=True)
    transform = np.zeros((k, len(weeks)), dtype=np.int64)
    transform[np.arange(k), inverse] = 1
    return weeks, {'positive': transform.T @ summary['positive'],
        'negative': transform.T @ summary['negative'],
        'twice_wins': transform.T @ summary['twice_wins'] @ transform}


def simultaneous_intervals(points, draws):
    """One 97.5% maximum-deviation band for all fixed horizons in one scheme."""
    require(isinstance(points, np.ndarray) and points.dtype == np.float64 and points.ndim == 1
        and isinstance(draws, np.ndarray) and draws.dtype == np.float64 and draws.ndim == 2
        and draws.shape[1] == len(points) and draws.shape[0] > 0
        and np.isfinite(points).all() and np.isfinite(draws).all(), 'BAND_INPUT')
    max_deviation = np.max(np.abs(draws - points[None, :]), axis=1)
    half_width = float(np.quantile(max_deviation, .975, method='linear'))
    return {'half_width': half_width, 'low': np.maximum(0., points - half_width),
        'high': np.minimum(1., points + half_width), 'replicates': len(draws),
        'per_scheme_alpha': .025, 'horizons_simultaneous': len(points)}


def score_fixed_signal(target, folds, predictions, reports, *, replicates=REPLICATES, minimum_blocks=8):
    require(len(folds) == len(predictions) == len(reports) == 4 and type(replicates) is int
        and replicates > 0 and type(minimum_blocks) is int and minimum_blocks > 0, 'SCOPE')
    prepared = []
    anchor_indices = np.array([0, 4, 14], np.int64)
    for fold, pred, report in zip(folds, predictions, reports):
        first, last, prefix = fold['oof_start_row'], fold['oof_end_row'], fold['prefix_rows']
        rows = np.arange(first, last, dtype=np.int64)
        require(prefix == first and report['fold'] == fold and report['test_data_used'] is False
            and np.array_equal(pred['parent_rows'], target['parent_rows'][rows])
            and np.array_equal(pred['bar_start_ns'], target['bar_start_ns'][rows])
            and np.array_equal(pred['decision_ns'], target['decision_ns'][rows]), 'PARENT_JOIN')
        for name in ('terminal_time_ns', 'observed', 'sign'):
            require(np.array_equal(pred[name], target[name][rows][:, anchor_indices]), 'ANCHOR_JOIN:' + name)
        train = dense.causal_target_batch(target, np.arange(prefix, dtype=np.int64), cutoff_ns=fold['train_cutoff_ns'])
        eligible_train = train['observed'] & (train['sign'] != 0)
        train_up = (eligible_train & (train['sign'] == 1)).sum(axis=0)
        train_down = (eligible_train & (train['sign'] == -1)).sum(axis=0)
        require(np.all(train_up > 0) and np.all(train_down > 0), 'TRAIN_PRIOR_CLASSES')
        prior = train_up / (train_up + train_down)
        del train, eligible_train
        oof = dense.causal_target_batch(target, rows, cutoff_ns=fold['oof_cutoff_ns'])
        days = oof['decision_ns'] // DAY_NS
        day_values, day_ids = np.unique(days, return_inverse=True)
        week_values, week_ids = np.unique((days + 3) // 7, return_inverse=True)
        require(len(day_values) >= minimum_blocks and len(week_values) >= minimum_blocks, 'MIN_BLOCKS')
        day_to_week = np.searchsorted(week_values, (day_values + 3) // 7)
        prepared.append({'fold': fold, 'report': report, 'oof': oof,
            'logits': np.ascontiguousarray(pred['logit_up'][:, 1], dtype=np.float64), 'prior': prior,
            'day_ids': day_ids, 'day_to_week': day_to_week,
            'day_weights': saved.block_weights(len(day_values), seed=DAY_SEED + FOLD_SEED_OFFSET * fold['fold_id'], np=np, replicates=replicates),
            'week_weights': saved.block_weights(len(week_values), seed=WEEK_SEED + FOLD_SEED_OFFSET * fold['fold_id'], np=np, replicates=replicates),
            'day_blocks': len(day_values), 'week_blocks': len(week_values)})
    points = np.zeros(60, np.float64)
    day_draws, week_draws = np.zeros((replicates, 60), np.float64), np.zeros((replicates, 60), np.float64)
    panels = []
    for j, horizon in enumerate(dense.HORIZONS):
        all_num = all_den = 0
        day_num = np.zeros(replicates, np.float64); day_den = np.zeros(replicates, np.float64)
        week_num = np.zeros(replicates, np.float64); week_den = np.zeros(replicates, np.float64)
        fold_panels = []; total_rows = 0; total_scores = np.zeros(4, np.float64)
        for part in prepared:
            oof = part['oof']; eligible = oof['observed'][:, j] & (oof['sign'][:, j] != 0)
            labels = np.ascontiguousarray((oof['sign'][eligible, j] == 1).astype(np.int8))
            z = np.ascontiguousarray(part['logits'][eligible], dtype=np.float64)
            require(len(labels) > 0 and np.unique(labels).size == 2, 'OOF_CLASSES')
            used_days, compact = np.unique(part['day_ids'][eligible], return_inverse=True)
            summary = rank.block_rank_pairs(labels, z, np.ascontiguousarray(compact, dtype=np.int64))
            numerator = int(summary['twice_wins'].sum()); denominator = int(summary['positive'].sum() * summary['negative'].sum())
            require(denominator > 0, 'PAIR_DENOMINATOR')
            all_num += numerator; all_den += denominator
            dn, dd = rank.weighted_rank_pairs(summary, part['day_weights'][:, used_days])
            used_weeks, week_summary = coarsen_block_pairs(summary, part['day_to_week'][used_days])
            wn, wd = rank.weighted_rank_pairs(week_summary, part['week_weights'][:, used_weeks])
            day_num += dn; day_den += dd; week_num += wn; week_den += wd
            model = probability.direction_probability_metrics(labels, z)
            prior_logit = float(np.log(part['prior'][j] / (1 - part['prior'][j])))
            prior = probability.direction_probability_metrics(labels, np.full(len(labels), prior_logit, np.float64))
            values = np.asarray([model['logloss'], model['brier'], prior['logloss'], prior['brier']], np.float64)
            total_scores += values * len(labels); total_rows += len(labels)
            item = {'fold_id': part['fold']['fold_id'], 'eligible': len(labels),
                'up': int(labels.sum()), 'down': int((labels == 0).sum()),
                'observed_zero': int((oof['observed'][:, j] & (oof['sign'][:, j] == 0)).sum()),
                'mature_missing': int((oof['mature'][:, j] & ~oof['observed'][:, j]).sum()),
                'immature': int((~oof['mature'][:, j]).sum()), 'train_prior_up': float(part['prior'][j]),
                'auc': numerator / (2 * denominator), 'logloss': float(values[0]), 'brier': float(values[1]),
                'prior_logloss': float(values[2]), 'prior_brier': float(values[3])}
            require(item['eligible'] + item['observed_zero'] + item['mature_missing'] + item['immature'] == len(oof['parent_rows']), 'COUNT_PARTITION')
            if horizon == 5:
                original = part['report']['model_metrics']['per_horizon'][1]
                require(original['horizon_minutes'] == 5 and original['counts']['eligible_cells'] == item['eligible']
                    and np.isclose(item['logloss'], original['logloss'], rtol=1e-12, atol=1e-12)
                    and np.isclose(item['brier'], original['brier'], rtol=1e-12, atol=1e-12), 'FIVE_MINUTE_REPLAY')
            fold_panels.append(item)
        require(all_den > 0 and np.all(day_den > 0) and np.all(week_den > 0), 'UNDEFINED_BOOTSTRAP')
        points[j] = all_num / (2 * all_den)
        day_draws[:, j] = day_num / (2 * day_den)
        week_draws[:, j] = week_num / (2 * week_den)
        panels.append({'horizon_minutes': horizon, 'eligible': total_rows,
            'within_fold_pair_weighted_auc': float(points[j]), 'logloss': float(total_scores[0] / total_rows),
            'brier': float(total_scores[1] / total_rows), 'prior_logloss': float(total_scores[2] / total_rows),
            'prior_brier': float(total_scores[3] / total_rows), 'folds': fold_panels})
    day_band, week_band = simultaneous_intervals(points, day_draws), simultaneous_intervals(points, week_draws)
    for j, panel in enumerate(panels):
        panel['auc_simultaneous_day'] = {'low': float(day_band['low'][j]), 'high': float(day_band['high'][j])}
        panel['auc_simultaneous_week'] = {'low': float(week_band['low'][j]), 'high': float(week_band['high'][j])}
        panel['both_band_lower_above_half'] = bool(day_band['low'][j] > .5 and week_band['low'][j] > .5)
    return {'panels': panels, 'day_band_half_width': day_band['half_width'], 'week_band_half_width': week_band['half_width'],
        'bootstrap_replicates': replicates, 'fold_blocks': [{'fold_id': p['fold']['fold_id'],
            'day': p['day_blocks'], 'week': p['week_blocks']} for p in prepared],
        'scope': 'Fixed 5-minute score against 60 distinct endpoints; four reused development folds; no cross-fold rank pairs.',
        'test_data_used': False, 'economic_edge_qualified': False}


def main(argv=None):
    import argparse
    import hashlib
    import os
    from pathlib import Path
    import signal
    import stat
    import sys
    import time
    import zipfile

    from gx1.contracts import gx1_capped_execution_v1 as cap
    from gx1.contracts import local_random_access_campaign_v2 as binding
    from gx1.contracts.entry_native_direction_context_report_v1 import CONTEXT_SPEC
    from gx1.scripts import audit_entry_direction_candle_midpoints_v1 as candle
    from gx1.scripts import compare_entry_direction_matched_exposure_v1 as matched

    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'plan-sha256', 'policy-sha256', 'source-head'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args(argv)
    stage_name = 'direction_dense_fixed_signal_actual_20261028_001'
    plan_ref = {'path': str(Path(args.plan)), 'sha256': args.plan_sha256}
    plan = binding.read_bound_json(Path(plan_ref['path']), plan_ref['sha256'])
    policy = binding.read_bound_json(candle.REPO / 'NEXT_RUN_POLICY.json', args.policy_sha256)
    fields = {'schema_version', 'stage', 'design', 'target_result', 'fold_reports', 'inputs',
        'sources', 'output_dir', 'wall_seconds', 'test_data_used'}
    require(set(plan) == fields and plan['schema_version'] == 'gx1_direction_dense_fixed_signal_plan_v1'
        and plan['stage'] == stage_name and plan['wall_seconds'] == 900 and plan['test_data_used'] is False, 'PLAN')
    gate = policy[stage_name]
    require(gate['status'] == 'ADMITTED' and gate['plan'] == plan_ref and gate['execution_authorized'] is True
        and gate['prediction_and_target_reads_authorized'] is True and gate['max_invocations'] == 1
        and gate['claimed_invocations'] == gate['completed_invocations'] == 0 and gate['relaunch_allowed'] is False
        and policy['independent_direction_edge_20261027']['current_unit'] == stage_name, 'ADMISSION')
    require(all(policy.get(k) is False for k in ('training_enabled', 'native_training_enabled',
        'full_epoch_training_allowed', 'full_val_allowed', 'native_launch_authorized'))
        and all(gate.get(k) is False for k in ('model_forwards_authorized', 'optimizer_steps_authorized',
            'checkpoint_reads_authorized', 'test_data_used', 'broker_access_allowed')), 'CLOSED_GATES')
    expected_git = (args.source_head, 'work/gx1-current', '')
    require(matched._git_state() == expected_git, 'CLEAN_SOURCE')
    design = binding.read_bound_json(Path(plan['design']['path']), plan['design']['sha256'])
    target_report = binding.read_bound_json(Path(plan['target_result']['path']), plan['target_result']['sha256'])
    protocol = design['uncertainty']
    require(design['status'] == 'PREDECLARED_NO_EXECUTION_AUTHORITY'
        and design['horizons_minutes'] == list(range(1, 61)) and design['folds'] == [1, 2, 3, 4]
        and design['score_axis_minutes'] == 5 and design['test_data_used'] is False
        and protocol['replicates'] == REPLICATES and protocol['day_seed'] == DAY_SEED
        and protocol['week_seed'] == WEEK_SEED and protocol['fold_seed_offset'] == FOLD_SEED_OFFSET, 'PROTOCOL')
    require(target_report['status'] == 'COMPLETE_REAL_TARGETS_ONLY' and target_report['test_data_used'] is False
        and target_report['horizons_minutes'] == list(range(1, 61))
        and target_report['legacy_anchor_byte_parity'] == {'1': True, '5': True, '15': True}, 'TARGET_REPORT')
    require(set(plan['inputs']) == {'target', 'predictions'} and len(plan['inputs']['predictions']) == 4
        and len(plan['fold_reports']) == 4
        and plan['inputs']['target'] == {k: target_report['target'][k] for k in ('path', 'sha256')}, 'INPUTS')
    reports = [binding.read_bound_json(Path(ref['path']), ref['sha256']) for ref in plan['fold_reports']]
    folds = [report['fold'] for report in reports]
    for index, (report, ref) in enumerate(zip(reports, plan['inputs']['predictions']), 1):
        require(report['fold']['fold_id'] == index and report['test_data_used'] is False
            and report['predictions'] == ref, 'FOLD_REPORT')
    expected_sources = set(matched.SOURCE_PATHS) | set(candle.SOURCE_PATHS) | {
        'gx1/scripts/materialize_entry_direction_dense_v1.py', 'gx1/scripts/score_entry_direction_dense_minutes_v1.py'}
    require(set(plan['sources']) == expected_sources, 'SOURCE_SET')
    for name, ref in plan['sources'].items():
        require(ref['path'] == str(candle.REPO / name), 'SOURCE_PATH')
        binding.require_binding(ref, label='dense score source', verify_file=True)
    proof, lock = cap.require_capped_cpu_audit_execution(), cap.require_capped_lock_ancestry()
    require(lock['scope'] == proof['cgroup_path'][len('/sys/fs/cgroup'):], 'CAPPED_LOCK')
    output = Path(plan['output_dir'])
    require(output.is_absolute() and output.parent == candle.ROOT and not output.exists()
        and not output.is_symlink() and candle.ROOT.resolve(strict=True) == candle.ROOT, 'OUTPUT')
    output.mkdir(mode=0o700)
    claim = matched._write_new(output / 'CLAIM.json', {'plan': plan_ref, 'source_head': args.source_head, 'test_data_used': False})
    refs = [plan['inputs']['target'], *plan['inputs']['predictions']]
    require(len({ref['path'] for ref in refs}) == 5 and all(Path(ref['path']).suffix == '.npz' for ref in refs), 'FIVE_PAYLOADS')
    allowed_json = {plan_ref['path'], plan['design']['path'], plan['target_result']['path'],
        *(ref['path'] for ref in plan['fold_reports'])}
    reads, denials = matched._install_guard(refs, allowed_json, output)
    started = time.monotonic()
    def expired(*unused):
        raise RuntimeError('DIRECTION_DENSE_SCORE_WALL_LIMIT')
    signal.signal(signal.SIGALRM, expired); signal.alarm(plan['wall_seconds'])
    def read_bytes(ref, maximum):
        path = Path(ref['path']); before = path.lstat()
        require(stat.S_ISREG(before.st_mode) and not path.is_symlink() and 0 < before.st_size <= maximum, 'INPUT_FILE')
        identity = lambda x: (x.st_dev, x.st_ino, x.st_size, x.st_mtime_ns, x.st_ctime_ns)
        with path.open('rb') as f:
            require(identity(os.fstat(f.fileno())) == identity(before), 'INPUT_OPEN_IDENTITY')
            raw = f.read(maximum + 1)
            require(identity(os.fstat(f.fileno())) == identity(before), 'INPUT_READ_IDENTITY')
        require(identity(path.lstat()) == identity(before) and len(raw) == before.st_size
            and hashlib.sha256(raw).hexdigest() == ref['sha256'], 'INPUT_SHA')
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            require(sum(x.file_size for x in archive.infolist()) <= 512 * 1024**2, 'ARCHIVE_SIZE')
        return raw
    try:
        raw = read_bytes(refs[0], 128 * 1024**2)
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            expected = {'parent_rows', 'bar_start_ns', 'decision_ns', 'dynamic_entry_m1_row',
                'terminal_time_ns', 'terminal_source_row', 'observed', 'mid_move', 'mid_return_bps', 'sign'}
            require(set(archive.files) == expected and len(archive.files) == len(expected), 'TARGET_MEMBERS')
            target = {name: archive[name] for name in ('parent_rows', 'bar_start_ns', 'decision_ns',
                'terminal_time_ns', 'observed', 'sign')}
        del raw
        size = target_report['parents']
        require(size == candle.PARENTS and all(target[k].shape == (size,) and target[k].dtype == np.int64
            for k in ('parent_rows', 'bar_start_ns', 'decision_ns'))
            and target['terminal_time_ns'].shape == target['observed'].shape == target['sign'].shape == (size, 60)
            and target['terminal_time_ns'].dtype == np.int64 and target['observed'].dtype == np.bool_
            and target['sign'].dtype == np.int8 and np.array_equal(target['parent_rows'], np.arange(size, dtype=np.int64)), 'TARGET_HEADER')
        predictions = []
        for ref, fold in zip(refs[1:], folds):
            raw = read_bytes(ref, 64 * 1024**2)
            predictions.append(saved.read_predictions(io.BytesIO(raw), signed_bps=False,
                rows=fold['oof_end_row'] - fold['oof_start_row'], np=np, context_spec=CONTEXT_SPEC))
            del raw
        require(set(reads) == {ref['path'] for ref in refs}
            and all(count == 1 for count in reads.values()) and not denials, 'OPEN_SCOPE')
        result = score_fixed_signal(target, folds, predictions, reports)
        for panel in result['panels']:
            j = panel['horizon_minutes'] - 1
            for item, source in zip(panel['folds'], target_report['fold_summaries']):
                group = next(x for x in source['groups'] if x['role'] == 'oof')
                require(item['fold_id'] == source['fold_id'] and all(item[key] == group[key][j]
                    for key in ('eligible', 'up', 'down', 'observed_zero', 'mature_missing', 'immature')), 'TARGET_COUNT_REPLAY')
        require(matched._git_state() == expected_git and cap.require_capped_lock_ancestry() == lock, 'FINAL_SOURCE_LOCK')
        result.update(schema_version='gx1_direction_dense_fixed_signal_result_v1',
            status='COMPLETE_REUSED_DEVELOPMENT_NOT_60_OUTPUT_MODEL', plan=plan_ref, source_head=args.source_head,
            exact_original_anchor_and_parent_joins=True, target_counts_replayed=True, five_minute_scores_replayed=True,
            payload_archives_read=5, model_forwards=0, optimizer_steps=0, checkpoint_reads=0,
            elapsed_seconds=time.monotonic()-started,
            uncertainty='97.5% max-absolute-deviation bootstrap band across 60 horizons per day/week scheme; block exchangeability unproved.')
        result_ref = matched._write_new(output / 'RESULT.json', result)
        terminal_ref = matched._write_new(output / 'TERMINAL.json', {'status': 'COMPLETE', 'exit_code': 0,
            'plan': plan_ref, 'claim': claim, 'result': result_ref, 'source_head': args.source_head,
            'source_unchanged': True, 'execution': proof, 'lock': lock, 'test_data_used': False})
        print(json.dumps({'status': result['status'], 'result': result_ref, 'terminal': terminal_ref,
            'anchors': [p for p in result['panels'] if p['horizon_minutes'] in (1, 5, 10, 15, 30, 60)],
            'horizons_with_both_lower_above_half': [p['horizon_minutes'] for p in result['panels'] if p['both_band_lower_above_half']]}, sort_keys=True))
    except Exception as exc:
        matched._write_new(output / 'TERMINAL.json', {'status': 'FAILED', 'exit_code': 1, 'plan': plan_ref,
            'claim': claim, 'source_head': args.source_head, 'error': type(exc).__name__ + ': ' + str(exc),
            'test_data_used': False})
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
