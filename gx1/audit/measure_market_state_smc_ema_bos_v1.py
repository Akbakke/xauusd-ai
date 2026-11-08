"""One fixed TRAIN-only SMC BOS plus EMA20 direction reference; no model or trade."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from gx1.contracts import market_state_five_endpoint_target_binding_v1 as state
from gx1.contracts import market_state_future_displacement_candidate_v1 as future
from gx1.contracts.gx1_capped_execution_v1 import (
    require_capped_cpu_audit_execution, require_capped_lock_ancestry,
)

REPO = Path('/home/andre2/src/GX1_CURRENT')
PLAN = REPO / 'configs/research/MARKET_STATE_SMC_EMA_BOS_20261108_001.json'
POLICY = REPO / 'NEXT_RUN_POLICY.json'
SOURCE = REPO / 'gx1/audit/measure_market_state_smc_ema_bos_v1.py'
DAY_NS = 86_400_000_000_000
HOUR_NS = 3_600_000_000_000


def need(ok, why):
    if not bool(ok):
        raise RuntimeError('SMC_EMA_BOS_' + why)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x') as out:
        json.dump(value, out, sort_keys=True, indent=2, allow_nan=False)
        out.write('\n')
        out.flush()
        os.fsync(out.fileno())


def verify(item):
    path = Path(item['path'])
    need(path.is_absolute() and path.is_file() and not path.is_symlink()
         and sha(path) == item['sha256'], 'INPUT_HASH:' + str(path))
    return path


def frozen(claim, plan):
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=REPO, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO)
    need(head == claim['source_head'] and branch == 'work/gx1-current' and not dirty,
         'SOURCE_HEAD')
    need(sha(PLAN) == claim['plan_sha256'] and sha(POLICY) == claim['policy_sha256']
         and sha(SOURCE) == plan['source_sha256'], 'SOURCE_HASH')


def summarize(issues, slope, bos_up, bos_down, bound):
    """Prebound fold/week/EMA-side event-versus-control statistic."""
    need(np.array_equal(bound.decision_ns, issues), 'F_ISSUES')
    direction = np.zeros(len(slope), dtype=np.int8)
    finite = np.isfinite(slope)
    direction[finite] = np.sign(slope[finite]).astype(np.int8)
    event = ((bos_up & ~bos_down & (direction == 1))
             | (bos_down & ~bos_up & (direction == -1)))
    control = ~bos_up & ~bos_down & (direction != 0)
    eligible = bound.valid[:, 2] & np.isfinite(slope)
    signed_f = direction * bound.sign[:, 2] * bound.magnitude[:, 2]
    long_f = bound.sign[:, 2] * bound.magnitude[:, 2]
    # Epoch day 4 is Monday 1970-01-05. Each 60-minute outcome must mature
    # inside its own later TRAIN score fold, never across a boundary.
    week = ((issues // DAY_NS) - 4) // 7
    records = []
    totals = []
    event_union = np.zeros(len(issues), dtype=np.bool_)
    for fold_id in range(1, 6):
        lo, hi = state.FOLD_BOUNDARIES_NS[fold_id - 1:fold_id + 1]
        in_fold = (issues >= lo) & (issues < hi) & (issues + HOUR_NS < hi)
        e = eligible & in_fold & event
        c = eligible & in_fold & control
        event_union |= e
        totals.append({'fold_id': fold_id, 'eligible_events': int(e.sum()),
                       'eligible_controls': int(c.sum()),
                       'event_up': int((e & (direction == 1)).sum()),
                       'event_down': int((e & (direction == -1)).sum()),
                       'event_gap_fraction': float(bound.gap_between[e, 2].mean()) if e.any() else None})
        for w in np.unique(week[e]):
            for side in (-1, 1):
                es = e & (week == w) & (direction == side)
                if not es.any():
                    continue
                cs = c & (week == w) & (direction == side)
                need(cs.any(), 'NO_SAME_WEEK_SIDE_CONTROL')
                n = int(es.sum())
                event_mean = float(signed_f[es].mean())
                control_mean = float(signed_f[cs].mean())
                records.append({'fold_id': fold_id, 'utc_week_id': int(w),
                                'ema_direction': side, 'events': n, 'controls': int(cs.sum()),
                                'event_mean_signed_F_over_H1_ATR': event_mean,
                                'control_mean_signed_F_over_H1_ATR': control_mean,
                                'delta': event_mean - control_mean})
    need(records and len(totals) == 5, 'EMPTY_SCORE')
    all_events = sum(x['eligible_events'] for x in totals)
    matched = sum(x['events'] for x in records)
    need(all_events == matched, 'EVENT_MATCH_COVERAGE')
    fold_delta = []
    for fold_id in range(1, 6):
        rows = [x for x in records if x['fold_id'] == fold_id]
        n = sum(x['events'] for x in rows)
        fold_delta.append(sum(x['events'] * x['delta'] for x in rows) / n if n else None)
    weeks = sorted(set(x['utc_week_id'] for x in records))
    numer = np.asarray([sum(x['events'] * x['delta'] for x in records
                            if x['utc_week_id'] == w) for w in weeks], dtype=np.float64)
    denom = np.asarray([sum(x['events'] for x in records if x['utc_week_id'] == w)
                        for w in weeks], dtype=np.int64)
    pooled = float(numer.sum() / denom.sum())
    rng = np.random.default_rng(20261108)
    draws = rng.integers(0, len(weeks), size=(2000, len(weeks)))
    samples = numer[draws].sum(axis=1) / denom[draws].sum(axis=1)
    ci = [float(x) for x in np.quantile(samples, [0.025, 0.975])]
    ev = event_union
    need(int(ev.sum()) == all_events, 'EVENT_TOTAL')
    positive = sum(x is not None and x > 0 for x in fold_delta)
    gate = bool(all(x['eligible_events'] >= 200 for x in totals)
                and positive >= 4 and pooled > 0 and ci[0] > 0)
    return {'decision': 'POSITIVE_DIRECTIONAL_ASSOCIATION_ONLY' if gate else 'NO_POSITIVE_GATE',
            'scope': 'H1_native_60min_F_endpoint_after_closed_M5_BOS_and_EMA20_slope',
            'primary': {'event_minus_same_UTC_week_and_EMA_side_control': pooled,
                        'week_block_bootstrap_95pct_CI': ci, 'bootstrap_seed': 20261108,
                        'bootstrap_replicates': 2000, 'fold_deltas': fold_delta,
                        'positive_folds': positive, 'matched_utc_weeks': len(weeks)},
            'coverage': totals, 'matched_strata': len(records),
            'event_mean_signed_F_over_H1_ATR': float(signed_f[ev].mean()),
            'always_long_mean_F_over_H1_ATR_on_events': float(long_f[ev].mean()),
            'candidate_half_H1_ATR_side_hit_rate': float((signed_f[ev & (bound.magnitude[:, 2] >= 0.5)] > 0).mean())
                if np.any(ev & (bound.magnitude[:, 2] >= 0.5)) else None,
            'model_forwards': 0, 'optimizer_steps': 0, 'test_data_used': False,
            'entry_or_economic_edge_demonstrated': False}


def run():
    plan = json.loads(PLAN.read_text())
    need(plan.get('schema_version') == 'gx1_market_state_smc_ema_bos_plan_v1'
         and plan.get('status') == 'PREPARED_NOT_CLAIMED'
         and plan.get('columns') == ['time', 'smc_bos_up', 'smc_bos_down', 'ema20_slope_atr']
         and plan.get('primary_horizon_minutes') == 60
         and plan.get('score_folds') == 5 and plan.get('one_shot_invocations') == 1
         and plan.get('max_wall_seconds') == 600
         and plan.get('model_forwards') == plan.get('optimizer_steps') == 0
         and plan.get('training_enabled') is False and plan.get('test_data_used') is False,
         'PLAN_SCOPE')
    root = Path(plan['run_root'])
    need(root == Path('/home/andre2/GX1_RUNS/MARKET_STATE_SMC_EMA_BOS_20261108_001'), 'RUN_ROOT')
    claim = json.loads((root / 'CLAIM.json').read_text())
    need(claim['plan_sha256'] == sha(PLAN) and claim['source_head'], 'CLAIM')
    frozen(claim, plan)
    policy = json.loads(POLICY.read_text())
    gate = policy['market_state_smc_ema_bos_20261108_001']
    need(policy['training_enabled'] is False and gate['status'] == 'EXECUTION_CLAIMED'
         and gate['claimed_invocations'] == 1 and gate['completed_invocations'] == 0
         and gate['plan'] == {'path': str(PLAN), 'sha256': sha(PLAN)}
         and gate['source_sha256'] == plan['source_sha256']
         and gate['execution_authorized'] is True and gate['data_reads_authorized'] is True
         and gate['research_fit_authorized'] is False and gate['test_authorized'] is False
         and gate['relaunch_allowed'] is False, 'POLICY')
    need(not (root / 'RESULT.json').exists() and not (root / 'TERMINAL.json').exists(),
         'ALREADY_COMPLETED')
    limits = require_capped_cpu_audit_execution()
    limits['lock'] = require_capped_lock_ancestry()
    need(limits['memory_max'] == limits['memory_high'] == 4 * 1024**3
         and limits['swap'] == 512 * 1024**2 and limits['pids'] == 64
         and limits['cpu_affinity'] == list(range(8))
         and limits['numerical_threads'] == 1, 'CAP_LIMITS')
    data_path = verify(plan['m5_source'])
    axis_path = verify(plan['model_axis'])
    table = pq.read_table(data_path, columns=plan['columns'])
    source_time = table['time'].to_numpy(zero_copy_only=False).astype('datetime64[ns]').astype(np.int64)
    need(np.all(np.diff(source_time) > 0), 'SOURCE_CLOCK')
    with np.load(axis_path, allow_pickle=False) as axes:
        parent = np.asarray(axes['parent_rows'], dtype=np.int64)
        bar = np.asarray(axes['bar_start_ns'], dtype=np.int64)
        issues = np.asarray(axes['decision_ns'], dtype=np.int64)
    need(len(parent) == len(bar) == len(issues) == 118249
         and np.array_equal(parent, np.arange(118249, dtype=np.int64))
         and np.array_equal(issues, bar + 300_000_000_000), 'MODEL_AXIS')
    at = np.searchsorted(source_time, bar)
    need(np.all(at < len(source_time)) and np.array_equal(source_time[at], bar), 'EXACT_M5_JOIN')
    slope = np.asarray(table['ema20_slope_atr'].to_numpy(zero_copy_only=False)[at], dtype=np.float64)
    up = np.asarray(table['smc_bos_up'].to_numpy(zero_copy_only=False)[at] > 0, dtype=np.bool_)
    down = np.asarray(table['smc_bos_down'].to_numpy(zero_copy_only=False)[at] > 0, dtype=np.bool_)
    bound = future.load_bound_future_displacement()
    need(np.array_equal(bound.parent_rows, parent), 'F_PARENT')
    result = summarize(issues, slope, up, down, bound)
    result.update({'status': 'COMPLETE_TRAIN_ONLY', 'plan': {'path': str(PLAN), 'sha256': sha(PLAN)},
                   'source_head': claim['source_head'], 'source': plan['m5_source'],
                   'model_axis': plan['model_axis'], 'limits': limits})
    frozen(claim, plan)
    verify(plan['m5_source'])
    verify(plan['model_axis'])
    write(root / 'RESULT.json', result)
    write(root / 'TERMINAL.json', {'status': 'COMPLETE', 'exit_code': 0,
                                  'result_sha256': sha(root / 'RESULT.json'),
                                  'source_unchanged': True, 'test_data_used': False})
    print(json.dumps({'decision': result['decision'], 'events': sum(x['eligible_events'] for x in result['coverage']),
                      'effect': result['primary']['event_minus_same_UTC_week_and_EMA_side_control'],
                      'ci': result['primary']['week_block_bootstrap_95pct_CI']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.run:
        run()
