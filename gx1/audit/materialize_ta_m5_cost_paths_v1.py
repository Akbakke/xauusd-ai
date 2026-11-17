"""Exact Fold2 fit and Apr-Jun development M5 price/cost/path targets.

No model, Entry rule, fixed holding rule, TEST or trading lives here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True

REPO = Path('/home/andre2/src/GX1_CURRENT')
PLAN = REPO / 'configs/research/TA_M5_COST_PATH_TARGETS_20261117_001.json'
ROOT = Path('/home/andre2/GX1_RUNS/TA_M5_COST_PATH_TARGETS_20261117_001')
STAGE = 'ta_m5_cost_path_targets_20261117_001'
HORIZONS = (1, 5, 15, 60, 240, 480)
FIT_ROWS, SCORE_ROWS = 47338, 17536
FIT_END, SCORE_END = 1735689600000000000, 1751328000000000000
MINUTE_NS = 60_000_000_000


def need(value, code):
    if not bool(value):
        raise RuntimeError('TA_M5_COST_PATH_' + code)


def ref(path):
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), 'REGULAR_FILE')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path), 'sha256': digest.hexdigest()}


def read(binding):
    need(ref(binding['path']) == binding, 'INPUT_SHA')
    return json.loads(Path(binding['path']).read_text())


def publish(name, payload):
    from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
    path = ROOT / name
    raw = payload if isinstance(payload, bytes) else (json.dumps(
        payload, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    _publish_bytes(path, raw)
    return ref(path)


def inspect(plan_sha, *, admitted):
    import numpy as np
    from gx1.contracts.entry_observed_market_v1 import require_entry_observed_market_scope
    need(not ROOT.exists() and ref(PLAN)['sha256'] == plan_sha, 'PLAN_OR_OUTPUT')
    plan = json.loads(PLAN.read_text())
    need(plan['stage'] == STAGE and plan['status'] == 'PREPARED_NOT_ADMITTED'
         and plan['horizon_minutes'] == list(HORIZONS)
         and plan['fit_rows'] == FIT_ROWS and plan['score_rows'] == SCORE_ROWS
         and plan['fit_target_end_ns'] == FIT_END and plan['score_target_end_ns'] == SCORE_END
         and plan['max_seconds'] == 900 and plan['test_data_used'] is False
         and plan['model_forwards'] == plan['optimizer_updates'] == 0
         and plan['target_fields'] == ['net_bps', 'mfe_bps', 'pre_mfe_mae_upper_bps',
                                       'terminal_observed', 'path_observed'],
         'FIXED_SCOPE')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    need(subprocess.run(['git', 'merge-base', '--is-ancestor', plan['source_base_head'], head],
         cwd=REPO).returncode == 0
         and not subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True)
         and all(ref(REPO / name)['sha256'] == digest
                 for name, digest in plan['source_owners'].items()), 'SOURCE_CLEAN_BOUND')
    for binding in plan['inputs'].values():
        need(ref(binding['path']) == binding, 'INPUT_REF')
    view = read(plan['inputs']['view'])
    result = read(plan['inputs']['paired_fit_result'])
    cost = read(plan['inputs']['cost_result'])
    need(result['status'] == 'COMPLETE_ONE_PASS_PAIRED_S_F_STRUCTURE_APR_JUN_SCORE'
         and result['test_data_used'] is False
         and result['captures']['FINAL'] == plan['inputs']['score_capture']
         and result['fit_rows'] == FIT_ROWS and result['final']['rows'] == SCORE_ROWS
         and cost['status'] == 'COMPLETE_FIXED_COSTED_M5_DIRECTION_MARKOUT'
         and cost['final_capture'] == plan['inputs']['score_capture']
         and cost['cost_recipe'] == plan['inputs']['cost_recipe']
         and cost['test_data_used'] is False, 'REUSED_RESULT')
    with np.load(plan['inputs']['model_axes']['path'], allow_pickle=False) as z:
        fit_rows = z['parent_rows'][:FIT_ROWS].copy()
        fit_issues = z['decision_ns'][:FIT_ROWS].copy()
        need(len(z['parent_rows']) == view['full_axes']['rows']
             and hashlib.sha256(z['parent_rows'].tobytes()).hexdigest()
                 == view['full_axes']['parent_rows_sha256']
             and hashlib.sha256(z['decision_ns'].tobytes()).hexdigest()
                 == view['full_axes']['decision_ns_sha256']
             and hashlib.sha256(fit_rows.tobytes()).hexdigest()
                 == view['prefix']['parent_rows_sha256']
             and hashlib.sha256(fit_issues.tobytes()).hexdigest()
                 == view['prefix']['decision_ns_sha256'], 'FIT_AXES')
    with np.load(plan['inputs']['score_capture']['path'], allow_pickle=False) as z:
        score_rows, score_issues = z['parent_rows'].copy(), z['issue_ns'].copy()
        need(score_rows.shape == score_issues.shape == (SCORE_ROWS,)
             and np.all(np.diff(score_rows) > 0)
             and np.all(np.diff(score_issues) > 0), 'SCORE_AXES')
    need(np.all((fit_issues >= 1714521600000000000) & (fit_issues < FIT_END))
         and np.all((score_issues >= 1743465600000000000) & (score_issues < SCORE_END)),
         'INTERVALS')
    scope = require_entry_observed_market_scope(read(plan['inputs']['cost_recipe']))
    need(scope['economics'] == plan['economics'] == cost['cost_assumptions']
         and scope['target_m1_source_sha256'] == plan['inputs']['m1_quotes']['sha256']
         and scope['economics']['historical_cost_truth_qualified'] is False,
         'COST_SCOPE')
    policy = json.loads((REPO / 'NEXT_RUN_POLICY.json').read_text())
    gate = policy.get(STAGE, {})
    need(policy['training_enabled'] is policy['native_training_enabled'] is False
         and policy['native_launch_authorized'] is False
         and gate.get('plan') == ref(PLAN)
         and gate.get('remaining_invocations') == (1 if admitted else 0)
         and gate.get('execution_authorized') is admitted
         and gate.get('training_authorized') is False
         and gate.get('test_authorized') is False
         and gate.get('broker_access_allowed') is False
         and gate.get('model_forwards_authorized') == 0,
         'POLICY')
    return plan, head, scope, {'FIT': (fit_rows, fit_issues, FIT_END),
                               'SCORE': (score_rows, score_issues, SCORE_END)}


def materialize(source, rows, issues, cutoff, economics, *, preflight=False):
    import numpy as np
    import pandas as pd
    import torch
    from gx1.contracts.entry_causal_m1_outcomes_v1 import query_dynamic_m1_prefix
    from gx1.contracts.entry_observed_market_v1 import build_entry_observed_market_targets
    if preflight:
        rows, issues = rows[:16], issues[:16]
    bars = pd.DatetimeIndex(pd.to_datetime(
        source['frame']['time'].iloc[rows], utc=True)).as_unit('ns').asi8
    need(np.array_equal(bars + 5 * MINUTE_NS, issues), 'PHYSICAL_M5_CLOCK')
    n = len(rows)
    net = np.full((n, len(HORIZONS), 2), np.nan, dtype=np.float64)
    mfe = np.full_like(net, np.nan)
    mae = np.full_like(net, np.nan)
    terminal = np.zeros((n, len(HORIZONS)), dtype=np.bool_)
    path = np.zeros_like(terminal)
    for j, horizon in enumerate(HORIZONS):
        eligible = issues + horizon * MINUTE_NS < cutoff
        at = np.flatnonzero(eligible)
        if not len(at):
            continue
        obs = query_dynamic_m1_prefix(source['index'], fill_surface=source['fill'],
            query_entry_rows=rows[at],
            horizon_minutes=np.full(len(at), horizon, dtype=np.int64),
            endpoint_kind='terminal_open_inclusive')
        term, complete = obs['terminal_gross_observed'], obs['path_observed']
        need(np.array_equal(obs['query_entry_rows'], rows[at])
             and np.array_equal(obs['terminal_time_ns'], issues[at] + horizon * MINUTE_NS)
             and np.array_equal(term[:, 0], term[:, 1])
             and np.array_equal(complete[:, 0], complete[:, 1])
             and np.all(~complete[:, 0] | term[:, 0]), 'QUERY_PARITY')
        terminal[at, j] = term[:, 0]
        path[at, j] = complete[:, 0]
        t = at[term[:, 0]]
        if len(t):
            values = build_entry_observed_market_targets(
                gross_return_bps=torch.as_tensor(
                    obs['terminal_gross_return_bps'][term[:, 0]], dtype=torch.float64),
                elapsed_wall_clock_seconds=torch.as_tensor(
                    obs['elapsed_wall_seconds'][term[:, 0]], dtype=torch.int64),
                economics=economics)['target_bps'].numpy()
            need(values.shape == (len(t), 3) and np.all(values[:, 2] == 0), 'COST_TARGET')
            net[t, j] = values[:, :2]
        v = at[complete[:, 0]]
        if len(v):
            mfe[v, j] = obs['mfe_bps'][complete[:, 0]]
            mae[v, j] = obs['mae_before_first_mfe_upper_bps'][complete[:, 0]]
    need(np.array_equal(np.isfinite(net).all(axis=2), terminal)
         and np.array_equal(np.isfinite(mfe).all(axis=2), path)
         and np.array_equal(np.isfinite(mae).all(axis=2), path)
         and np.all(~path | terminal), 'TARGET_MASKS')
    values = {'parent_rows': rows, 'issue_ns': issues, 'horizon_minutes':
              np.asarray(HORIZONS, dtype=np.int64), 'net_bps': net,
              'mfe_bps': mfe, 'pre_mfe_mae_upper_bps': mae,
              'terminal_observed': terminal, 'path_observed': path}
    summary = {'rows': n, 'per_horizon': {str(h): {
        'terminal_observed': int(terminal[:, j].sum()),
        'path_observed': int(path[:, j].sum()),
        'either_side_positive_net_on_path': int((
            path[:, j] & (np.nan_to_num(net[:, j], nan=-1e300).max(axis=1) > 0)).sum())}
        for j, h in enumerate(HORIZONS)}}
    return values, summary


def run(plan_sha, *, preflight):
    import numpy as np
    from gx1.contracts.entry_observed_market_v1 import read_dynamic_m5_target_source
    plan, head, scope, cohorts = inspect(plan_sha, admitted=not preflight)
    def expire(*_):
        raise TimeoutError('TA_M5_COST_PATH_MAX_SECONDS')
    previous = signal.signal(signal.SIGALRM, expire)
    signal.alarm(plan['max_seconds'])
    if preflight:
        try:
            source = read_dynamic_m5_target_source(scope, split='train')
            checks = {name: materialize(source, *parts, scope['economics'], preflight=True)[1]
                      for name, parts in cohorts.items()}
            print(json.dumps({'status': 'PASS_ACTUAL_TARGET_16_PER_COHORT_NO_CLAIM',
                              'checks': checks, 'model_forwards': 0,
                              'optimizer_updates': 0}))
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
        return
    ROOT.mkdir()
    claim = publish('CLAIM.json', {'plan': ref(PLAN), 'source_commit': head,
        'claimed_utc': datetime.now(timezone.utc).isoformat(), 'test_data_used': False})
    started = time.monotonic()
    try:
        source = read_dynamic_m5_target_source(scope, split='train')
        outputs, summaries = {}, {}
        for name, parts in cohorts.items():
            output, summary = materialize(source, *parts, scope['economics'])
            stream = io.BytesIO()
            np.savez_compressed(stream, **output)
            outputs[name] = publish(name + '_TARGETS.npz', stream.getvalue())
            summaries[name] = summary
        need(not subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True)
             and all(ref(REPO / name)['sha256'] == digest
                     for name, digest in plan['source_owners'].items()), 'SOURCE_BOOKEND')
        result = publish('RESULT.json', {'status': 'COMPLETE_FIXED_FOLD2_APR_JUN_COST_PATH_TARGETS',
            'plan': ref(PLAN), 'claim': claim, 'source_commit': head,
            'targets': outputs, 'summaries': summaries,
            'economics': scope['economics'], 'elapsed_seconds': time.monotonic() - started,
            'model_forwards': 0, 'optimizer_updates': 0, 'test_data_used': False,
            'economic_edge_proven': False})
        publish('TERMINAL.json', {'status': 'COMPLETE', 'exit_code': 0,
            'claim': claim, 'result': result, 'test_data_used': False})
    except BaseException as exc:
        publish('TERMINAL.json', {'status': 'FAILED_CONSUMED', 'exit_code': 1,
            'claim': claim, 'reason': type(exc).__name__ + ':' + str(exc),
            'test_data_used': False})
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    run(args.plan_sha256, preflight=args.preflight)


if __name__ == '__main__':
    main()
