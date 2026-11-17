"""One fixed, costed M5 direction markout from the consumed Apr-Jun score capture.

Observation horizons are not holding rules or a trade simulation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True

REPO = Path('/home/andre2/src/GX1_CURRENT')
PLAN = REPO / 'configs/research/TA_APR_JUN_M5_FIXED_COST_20261117_001.json'
ROOT = Path('/home/andre2/GX1_RUNS/TA_APR_JUN_M5_FIXED_COST_20261117_001')
STAGE = 'ta_apr_jun_m5_fixed_cost_20261117_001'
HORIZONS = (1, 5, 15, 60, 240, 480)
CUTOFF = 1751328000000000000  # 2025-07-01 exclusive
MINUTE_NS = 60_000_000_000


def need(ok, reason):
    if not bool(ok):
        raise RuntimeError('TA_APR_JUN_M5_COST_' + reason)


def ref(path):
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), 'REGULAR_FILE')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return {'path': str(path), 'sha256': h.hexdigest()}


def read(binding):
    need(ref(binding['path']) == binding, 'INPUT_SHA')
    return json.loads(Path(binding['path']).read_text())


def publish(name, value):
    path = ROOT / name
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    with path.open('xb') as out:
        out.write(raw)
    return ref(path)


def inspect(plan_sha, *, admitted):
    import numpy as np
    from gx1.contracts.entry_observed_market_v1 import require_entry_observed_market_scope
    need(ref(PLAN)['sha256'] == plan_sha and not ROOT.exists(), 'PLAN_OR_OUTPUT')
    plan = json.loads(PLAN.read_text())
    need(plan['stage'] == STAGE and plan['status'] == 'PREPARED_NOT_ADMITTED'
         and plan['horizon_minutes'] == list(HORIZONS)
         and plan['score_interval'] == ['2025-04-01', '2025-07-01']
         and plan['decision'] == 'native_F_M5_factorized_three_class_argmax_tie_flat'
         and plan['diagnostic'] == 'native_F_M5_forced_side_up_at_half'
         and plan['cost_comparators'] == ['always_LONG', 'always_SHORT', 'always_FLAT']
         and plan['max_seconds'] == 600 and plan['test_data_used'] is False
         and plan['model_forwards'] == plan['optimizer_updates'] == 0,
         'FIXED_SCOPE')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    need(subprocess.run(['git', 'merge-base', '--is-ancestor', plan['source_base_head'], head],
         cwd=REPO).returncode == 0
         and not subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True),
         'CLEAN_SOURCE')
    need(all(ref(REPO / name)['sha256'] == digest
             for name, digest in plan['source_owners'].items()), 'SOURCE_OWNERS')
    for binding in plan['inputs'].values():
        need(ref(binding['path']) == binding, 'INPUT_REF')
    fit = read(plan['inputs']['fit_result'])
    need(fit['status'] == 'COMPLETE_ONE_PASS_PAIRED_S_F_STRUCTURE_APR_JUN_SCORE'
         and fit['test_data_used'] is False and fit['economic_edge_proven'] is False
         and fit['captures']['FINAL'] == plan['inputs']['final_capture']
         and fit['final']['rows'] == 17536, 'FIT_RESULT')
    with np.load(plan['inputs']['final_capture']['path'], allow_pickle=False) as z:
        rows, issues, raw = z['parent_rows'], z['issue_ns'], z['native_F_raw']
        need(rows.shape == issues.shape == (17536,) and raw.shape == (17536, 5, 2)
             and np.isfinite(raw).all() and np.all(np.diff(issues) > 0)
             and np.all((issues >= 1743465600000000000) & (issues < CUTOFF))
             and np.all(z['native_query_k'][:, 0] == 1), 'CAPTURE')
    recipe = read(plan['inputs']['cost_recipe'])
    scope = require_entry_observed_market_scope(recipe)
    need(scope['economics'] == plan['economics']
         and scope['economics']['historical_cost_truth_qualified'] is False
         and scope['train_control_cutoff'] == '2026-01-01T00:00:00+00:00',
         'DECLARED_COST')
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
    return plan, head, scope


def decisions(raw):
    import numpy as np
    activity = np.exp(-np.logaddexp(0., -raw[:, 0].astype(np.float64)))
    up = np.exp(-np.logaddexp(0., -raw[:, 1].astype(np.float64)))
    probs = np.stack((activity * up, activity * (1. - up), 1. - activity), axis=1)
    primary = np.argmax(probs, axis=1).astype(np.int8)
    primary[(probs == probs.max(axis=1, keepdims=True)).sum(axis=1) > 1] = 2
    forced = np.where(up >= 0.5, 0, 1).astype(np.int8)
    return primary, forced


def one_panel(source, rows, issues, actions, horizon, economics):
    import numpy as np
    import torch
    from gx1.contracts.entry_causal_m1_outcomes_v1 import query_dynamic_m1_prefix
    from gx1.contracts.entry_observed_market_v1 import build_entry_observed_market_targets
    allowed = issues + horizon * MINUTE_NS < CUTOFF
    qrows, qissues = rows[allowed], issues[allowed]
    obs = query_dynamic_m1_prefix(source['index'], fill_surface=source['fill'],
        query_entry_rows=qrows,
        horizon_minutes=np.full(len(qrows), horizon, dtype=np.int64),
        endpoint_kind='terminal_open_inclusive')
    term, path = obs['terminal_gross_observed'], obs['path_observed']
    need(np.array_equal(obs['query_entry_rows'], qrows)
         and np.array_equal(obs['terminal_time_ns'], qissues + horizon * MINUTE_NS)
         and np.array_equal(term[:, 0], term[:, 1])
         and np.array_equal(path[:, 0], path[:, 1]), 'OUTCOME_AXIS')
    common = term[:, 0] & path[:, 0]
    need(bool(common.any()), 'OUTCOME_COVERAGE')
    net = build_entry_observed_market_targets(
        gross_return_bps=torch.as_tensor(obs['terminal_gross_return_bps'][common], dtype=torch.float64),
        elapsed_wall_clock_seconds=torch.as_tensor(obs['elapsed_wall_seconds'][common], dtype=torch.int64),
        economics=economics)['target_bps'].numpy()
    need(net.shape == (int(common.sum()), 3) and np.all(net[:, 2] == 0), 'NET')
    selected = actions[allowed][common]
    def measure(action):
        value = net[np.arange(len(net)), action]
        held = action != 2
        return {'mean_net_bps_all_issues': float(value.mean()),
                'mean_net_bps_selected': float(value[held].mean()) if held.any() else None,
                'LONG': int((action == 0).sum()), 'SHORT': int((action == 1).sum()),
                'FLAT': int((action == 2).sum())}
    return {'eligible_rows': int(len(qrows)), 'terminal_observed': int(term[:, 0].sum()),
        'common_rows': int(common.sum()),
        'common_rows_sha256': hashlib.sha256(qrows[common].tobytes()).hexdigest(),
        'primary': measure(selected[:, 0]), 'forced_side': measure(selected[:, 1]),
        'always_LONG': measure(np.zeros(len(net), dtype=np.int8)),
        'always_SHORT': measure(np.ones(len(net), dtype=np.int8)),
        'always_FLAT': measure(np.full(len(net), 2, dtype=np.int8))}


def run(plan_sha, *, preflight):
    import numpy as np
    import pandas as pd
    from gx1.contracts.entry_observed_market_v1 import read_dynamic_m5_target_source
    plan, head, scope = inspect(plan_sha, admitted=not preflight)
    if preflight:
        print(json.dumps({'status': 'PASS_BOUND_CAPTURE_COST_SOURCE_NO_OUTCOME_QUERY',
                          'rows': 17536, 'model_forwards': 0, 'optimizer_updates': 0}))
        return
    ROOT.mkdir()
    claim = publish('CLAIM.json', {'plan': ref(PLAN), 'source_commit': head,
        'claimed_utc': datetime.now(timezone.utc).isoformat(), 'test_data_used': False})
    started = time.monotonic()
    def expire(*_):
        raise TimeoutError('TA_APR_JUN_M5_COST_MAX_SECONDS')
    previous = signal.signal(signal.SIGALRM, expire)
    signal.alarm(plan['max_seconds'])
    try:
        with np.load(plan['inputs']['final_capture']['path'], allow_pickle=False) as z:
            rows, issues, raw = (z[k].copy() for k in ('parent_rows', 'issue_ns', 'native_F_raw'))
        primary, forced = decisions(raw[:, 0])
        actions = np.stack((primary, forced), axis=1)
        source = read_dynamic_m5_target_source(scope, split='train')
        bars = pd.DatetimeIndex(pd.to_datetime(
            source['frame']['time'].iloc[rows], utc=True)).as_unit('ns').asi8
        need(np.array_equal(bars + 5 * MINUTE_NS, issues), 'PHYSICAL_M5_CLOCK')
        panels = {str(h): one_panel(source, rows, issues, actions, h, scope['economics'])
                  for h in HORIZONS}
        need(not subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True)
             and all(ref(REPO / name)['sha256'] == digest
                     for name, digest in plan['source_owners'].items()), 'SOURCE_BOOKEND')
        result = publish('RESULT.json', {'status': 'COMPLETE_FIXED_COSTED_M5_DIRECTION_MARKOUT',
            'plan': ref(PLAN), 'claim': claim, 'source_commit': head,
            'final_capture': plan['inputs']['final_capture'], 'cost_recipe': plan['inputs']['cost_recipe'],
            'cost_assumptions': scope['economics'], 'panels': panels,
            'elapsed_seconds': time.monotonic() - started,
            'model_forwards': 0, 'optimizer_updates': 0,
            'test_data_used': False, 'trade_economics_proven': False,
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
