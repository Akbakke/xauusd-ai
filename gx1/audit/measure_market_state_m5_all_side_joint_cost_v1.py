"""One guarded Fold 2 M5 Entry target-coverage audit for both executable sides.

Replay the consumed hindsight-S joint-cost receipt, then measure LONG and
SHORT on the same terminal-and-path mask. The six predeclared horizons are
observation points, never a holding rule or hindsight horizon selection.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import torch

from gx1.audit import measure_market_state_s_m5_joint_cost_v1 as joint
from gx1.audit.measure_market_state_fold2_exact_prior_later_v1 import axes
from gx1.contracts import entry_causal_m1_outcomes_v1 as path
from gx1.contracts import market_state_five_endpoint_target_binding_v1 as state
from gx1.contracts import market_state_time_query_native_F_objective_v1 as objective
from gx1.contracts.entry_observed_market_v1 import (
    build_entry_observed_market_targets, read_dynamic_m5_target_source,
    require_entry_observed_market_scope,
)
from gx1.contracts.immutable_event_authority_v1 import _publish_bytes

REPO = joint.REPO
STAGE = 'market_state_m5_all_side_joint_cost_20261115_001'
PLAN = REPO / 'configs/research/MARKET_STATE_M5_ALL_SIDE_JOINT_COST_20261115_001.json'
POLICY = joint.POLICY
ROOT = Path('/home/andre2/GX1_RUNS/MARKET_STATE_M5_ALL_SIDE_JOINT_COST_20261115_001')
HORIZONS = joint.HORIZONS
SIDES = ('long', 'short')
CODE = tuple(sorted(set(joint.CODE) | {
    'gx1/audit/measure_market_state_m5_all_side_joint_cost_v1.py',
}))
INPUTS = {
    **joint.INPUTS,
    'joint_plan': (joint.PLAN, '480e2b54363051cfc72733d5239603358c5af2412298e9347c4754045ad737a4'),
    'joint_result': (joint.ROOT / 'RESULT.json',
                     '6fe578eeedb99a80e8bc833668d3efaab1fdb8a6a28afbb16c5ed2f9e97b8c7d'),
    'joint_terminal': (joint.ROOT / 'TERMINAL.json',
                       '85e851b481dccc8e0e55360fa4a0f39bf79b02021ac32b27e2b592e87a40dcbf'),
}


def need(ok: object, reason: str) -> None:
    if not bool(ok):
        raise RuntimeError('M5_ALL_SIDE_JOINT_COST_' + reason)


def _summary(net: np.ndarray, mfe: np.ndarray, mae: np.ndarray) -> dict:
    need(net.shape == mfe.shape == mae.shape and net.ndim == 2
         and net.shape[1] == 2 and len(net) > 0
         and np.isfinite(net).all() and np.isfinite(mfe).all()
         and np.isfinite(mae).all() and np.all(mae >= 0), 'SUMMARY_DOMAIN')
    out = {}
    for j, side in enumerate(SIDES):
        part = joint.paired(net[:, j], mfe[:, j], mae[:, j])
        part['positive_terminal_net_count'] = int(np.count_nonzero(net[:, j] > 0))
        part['positive_quality_count'] = int(np.count_nonzero(net[:, j] - mae[:, j] > 0))
        out[side] = part
    out['common_rows'] = len(net)
    out['either_side_positive_terminal_net_count'] = int(np.count_nonzero(
        np.any(net > 0, axis=1)))
    out['either_side_positive_quality_count'] = int(np.count_nonzero(
        np.any(net - mae > 0, axis=1)))
    return out


def inspect(plan_sha: str) -> tuple:
    need(Path.cwd().resolve() == REPO
         and joint.git('branch', '--show-current') == 'work/gx1-current'
         and joint.git('status', '--porcelain') == '', 'CLEAN_SOURCE')
    head = joint.git('rev-parse', 'HEAD')
    need(not ROOT.exists() and joint.sha(PLAN) == plan_sha, 'PLAN_OUTPUT')
    plan = json.loads(PLAN.read_text())
    need(set(plan) == {'schema_version', 'stage', 'status', 'source_base_head',
                       'input_refs', 'source_owners', 'scope', 'limits', 'output_root'}
         and plan['schema_version'] == 'gx1_m5_all_side_joint_cost_plan_v1'
         and plan['stage'] == STAGE and plan['status'] == 'PREPARED_NOT_ADMITTED'
         and plan['scope'] == {
             'fold_id': 2, 'fit_rows': 47338, 'later_rows': 4096,
             'horizon_minutes': list(HORIZONS),
             'side_order': list(SIDES),
             'common_mask': 'exact_terminal_and_complete_m1_path_both_sides',
             'comparison': 'consumed_hindsight_S_side_replay_only',
             'entry_timeframe': 'M5', 'exit_evaluation': False,
             'trade_selection': False, 'test_data_used': False}
         and plan['limits'] == {
             'class': 'audit', 'memory': '4G', 'swap': '512M',
             'wall_seconds': 1800, 'max_invocations': 1,
             'model_forwards': 0, 'optimizer_steps': 0,
             'training_authorized': False}
         and plan['output_root'] == str(ROOT), 'PLAN')
    need(subprocess.run(('git', 'merge-base', '--is-ancestor',
                         plan['source_base_head'], head), cwd=REPO).returncode == 0,
         'ANCESTOR')
    need(set(plan['source_owners']) == set(CODE)
         and all(joint.sha(REPO / name) == digest
                 for name, digest in plan['source_owners'].items()), 'SOURCE_HASH')
    expected = {name: {'path': str(p), 'sha256': digest}
                for name, (p, digest) in INPUTS.items()}
    need(plan['input_refs'] == expected
         and all(joint.sha(p) == digest for p, digest in INPUTS.values()),
         'INPUT_HASH')
    old = json.loads(INPUTS['joint_result'][0].read_text())
    terminal = json.loads(INPUTS['joint_terminal'][0].read_text())
    need(old['status'] == terminal['status'] == 'COMPLETE'
         and terminal['exit_code'] == 0
         and terminal['result'] == expected['joint_result']
         and old['plan'] == expected['joint_plan']
         and old['horizon_minutes'] == list(HORIZONS)
         and old['model_forwards'] == old['optimizer_steps'] == 0
         and old['test_data_used'] is False
         and old['trade_selection'] is False, 'JOINT_RECEIPT')
    bound = state.load_bound_five_endpoint_targets()
    coverage = json.loads(joint.old_worker.COVERAGE.read_text())
    fit, rows, issues = axes(coverage, bound)
    need(len(fit['fit_parent_rows']) == 47338 and len(rows) == 4096,
         'AXIS_LENGTH')
    scope = require_entry_observed_market_scope(
        json.loads(joint.old_worker.RECIPE.read_text()))
    need(scope['economics'] == old['economics']
         and scope['target_m1_source_sha256'] == joint.old_worker.M1_SHA,
         'COST_SOURCE')
    policy = json.loads(POLICY.read_text())
    need(policy.get('training_enabled') is False
         and policy.get('native_training_enabled') is False
         and policy.get('native_launch_authorized') is False
         and policy.get(STAGE) == {
             'status': 'ADMITTED', 'plan': joint.ref(PLAN),
             'output_root': str(ROOT), 'max_invocations': 1,
             'claimed_invocations': 0, 'completed_invocations': 0,
             'execution_authorized': True, 'data_reads_authorized': True,
             'model_forwards_authorized': 0, 'optimizer_steps_authorized': 0,
             'training_authorized': False, 'test_authorized': False,
             'broker_access_allowed': False, 'relaunch_allowed': False},
         'POLICY')
    return joint.ref(POLICY), head, bound, fit, rows, issues, scope, old


def one_cohort(source: dict, bound: object, old: dict, *, rows: np.ndarray,
               issues: np.ndarray, partition: str, cutoff: int,
               economics: dict) -> dict:
    need(partition in ('fit', 'score') and len(rows) == len(issues)
         and len(rows) in (47338, 4096), 'COHORT')
    bars = pd.DatetimeIndex(pd.to_datetime(
        source['frame']['time'].iloc[rows], utc=True)).as_unit('ns').asi8
    need(np.array_equal(bars + joint.M5_NS, issues), 'PHYSICAL_CLOCK')
    truth = bound.gather(rows, issues, fold_id=2, partition=partition)
    bits = truth['allowed'][:, 0]
    known = np.isin(bits, (1, 4)) & truth['eligible'][:, 0] & truth['mature'][:, 0]
    output = {}
    for horizon in HORIZONS:
        allowed = issues + horizon * joint.MINUTE_NS < cutoff
        qrows, qissues = rows[allowed], issues[allowed]
        previous = old[str(horizon)]
        obs = path.query_dynamic_m1_prefix(
            source['index'], fill_surface=source['fill'],
            query_entry_rows=qrows,
            horizon_minutes=np.full(len(qrows), horizon, dtype=np.int64),
            endpoint_kind='terminal_open_inclusive')
        need(np.array_equal(obs['query_entry_rows'], qrows)
             and np.array_equal(obs['terminal_time_ns'],
                                qissues + horizon * joint.MINUTE_NS)
             and bool(np.all(obs['terminal_time_ns'] < cutoff)), 'QUERY')
        term, pathmask = obs['terminal_gross_observed'], obs['path_observed']
        need(term.shape == pathmask.shape == (len(qrows), 2)
             and np.array_equal(term[:, 0], term[:, 1])
             and np.array_equal(pathmask[:, 0], pathmask[:, 1])
             and np.all(~pathmask[:, 0] | term[:, 0]), 'MASK')
        common = term[:, 0] & pathmask[:, 0]
        need(int(common.sum()) >= previous['common_mask']['n']
             and bool(common.any()), 'COMMON_COVERAGE')
        net = np.full((len(qrows), 2), np.nan, dtype=np.float64)
        if np.any(term[:, 0]):
            targets = build_entry_observed_market_targets(
                gross_return_bps=torch.as_tensor(
                    obs['terminal_gross_return_bps'][term[:, 0]],
                    dtype=torch.float64),
                elapsed_wall_clock_seconds=torch.as_tensor(
                    obs['elapsed_wall_seconds'][term[:, 0]],
                    dtype=torch.int64), economics=economics)['target_bps'].numpy()
            need(targets.shape == (int(term[:, 0].sum()), 3)
                 and np.all(targets[:, 2] == 0), 'TARGET')
            net[term[:, 0]] = targets[:, :2]
        values = _summary(net[common], obs['mfe_bps'][common],
                          obs['mae_before_first_mfe_upper_bps'][common])
        # Replay the consumed hindsight-S subset exactly before interpreting
        # the new all-side population. This comparison remains oracle-only.
        old_joint = common & known[allowed]
        side = np.where(bits[allowed][old_joint] == 4, 0, 1)
        i = np.arange(int(old_joint.sum()))
        old_net = net[old_joint][i, side]
        old_mfe = obs['mfe_bps'][old_joint][i, side]
        old_mae = obs['mae_before_first_mfe_upper_bps'][old_joint][i, side]
        replay = joint.paired(old_net, old_mfe, old_mae)
        need(replay['n'] == previous['common_mask']['n']
             and joint.axis_sha(qrows[old_joint]) == previous['joint_parent_rows_sha256']
             and joint.axis_sha(qissues[old_joint]) == previous['joint_issue_ns_sha256']
             and all(np.isclose(replay[key]['mean_bps'],
                                previous['common_mask'][key]['mean_bps'],
                                rtol=0, atol=1e-10)
                     for key in ('terminal_net_bps', 'mfe_bps',
                                 'pre_mfe_mae_upper_bps',
                                 'net_minus_pre_mfe_mae_upper_bps')),
             'JOINT_REPLAY')
        weeks = objective._weeks(qissues[common])
        weekly = {str(w): _summary(net[common][weeks == w],
                                   obs['mfe_bps'][common][weeks == w],
                                   obs['mae_before_first_mfe_upper_bps'][common][weeks == w])
                  for w in np.unique(weeks)}
        need(sum(x['common_rows'] for x in weekly.values()) == values['common_rows'],
             'WEEK_PARITY')
        output[str(horizon)] = {
            'horizon_minutes': horizon,
            'eligible_rows': len(qrows),
            'terminal_observed_rows': int(term[:, 0].sum()),
            'path_observed_rows': int(pathmask[:, 0].sum()),
            'common_parent_rows_sha256': joint.axis_sha(qrows[common]),
            'common_issue_ns_sha256': joint.axis_sha(qissues[common]),
            'common_mask': values,
            'by_utc_issue_week': weekly,
            'hindsight_s_subset_replay': True,
            'terminal_only_excluded_rows': int(term[:, 0].sum() - common.sum()),
        }
    return output


def run(plan_sha: str) -> None:
    from gx1.contracts.gx1_capped_execution_v1 import (
        require_capped_cpu_audit_execution, require_capped_lock_ancestry)
    context = inspect(plan_sha)
    cap = require_capped_cpu_audit_execution()
    lock = require_capped_lock_ancestry()
    need(cap['memory_max'] == cap['memory_high'] == 4 * 1024**3
         and cap['swap'] == 512 * 1024**2 and lock is not None, 'CAP')
    ROOT.mkdir(parents=False, exist_ok=False)
    _publish_bytes(ROOT / 'CLAIM.json', joint.encode({
        'schema_version': 'gx1_m5_all_side_joint_cost_claim_v1',
        'stage': STAGE, 'plan': joint.ref(PLAN),
        'policy': context[0], 'source_head': context[1],
        'test_data_used': False}))
    claim = joint.ref(ROOT / 'CLAIM.json')
    started = time.monotonic()
    try:
        _, head, bound, fit, rows, issues, scope, old = context
        source = read_dynamic_m5_target_source(scope, split='train')
        fit_result = one_cohort(
            source, bound, old['fit'], rows=fit['fit_parent_rows'],
            issues=fit['fit_decision_ns'], partition='fit',
            cutoff=joint.FIT_CUTOFF, economics=scope['economics'])
        later_result = one_cohort(
            source, bound, old['later'], rows=rows, issues=issues,
            partition='score', cutoff=joint.LATER_CUTOFF,
            economics=scope['economics'])
        result = {
            'schema_version': 'gx1_m5_all_side_joint_cost_result_v1',
            'status': 'COMPLETE', 'stage': STAGE,
            'plan': joint.ref(PLAN), 'policy': context[0], 'claim': claim,
            'source_head': head,
            'observed_utc': datetime.now(timezone.utc).isoformat(),
            'economics': scope['economics'],
            'horizon_minutes': list(HORIZONS),
            'fit': fit_result, 'later': later_result,
            'joint_s_result': joint.ref(INPUTS['joint_result'][0]),
            'model_forwards': 0, 'optimizer_steps': 0,
            'test_data_used': False, 'trade_selection': False,
            'elapsed_seconds': time.monotonic() - started}
        need(joint.git('rev-parse', 'HEAD') == head
             and joint.git('status', '--porcelain') == ''
             and joint.ref(POLICY) == context[0]
             and all(joint.sha(p) == digest for p, digest in INPUTS.values()),
             'FINAL_SOURCE')
        _publish_bytes(ROOT / 'RESULT.json', joint.encode(result))
        _publish_bytes(ROOT / 'TERMINAL.json', joint.encode({
            'schema_version': 'gx1_m5_all_side_joint_cost_terminal_v1',
            'status': 'COMPLETE', 'exit_code': 0, 'stage': STAGE,
            'claim': claim, 'result': joint.ref(ROOT / 'RESULT.json'),
            'source_head': head, 'test_data_used': False}))
    except BaseException as exc:
        _publish_bytes(ROOT / 'TERMINAL.json', joint.encode({
            'schema_version': 'gx1_m5_all_side_joint_cost_terminal_v1',
            'status': 'FAILED_CONSUMED', 'exit_code': 1, 'stage': STAGE,
            'claim': claim, 'result': None, 'source_head': context[1],
            'reason': type(exc).__name__ + ':' + str(exc),
            'test_data_used': False}))
        raise


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan-sha256', required=True)
    ap.add_argument('--preflight', action='store_true')
    ap.add_argument('--run', action='store_true')
    args = ap.parse_args()
    need(args.preflight != args.run, 'MODE')
    if args.preflight:
        context = inspect(args.plan_sha256)
        toy = _summary(np.array([[1.0, -2.0], [-1.0, 2.0]]),
                       np.array([[2.0, 1.0], [1.0, 2.0]]),
                       np.array([[0.5, 1.0], [1.0, 0.5]]))
        need(toy['common_rows'] == 2
             and toy['either_side_positive_terminal_net_count'] == 2
             and toy['long']['positive_quality_count'] == 1
             and toy['short']['positive_quality_count'] == 1,
             'SYNTHETIC_SUCCESS')
        try:
            _summary(np.array([[1.0, 2.0]]),
                     np.array([[1.0, 2.0]]),
                     np.array([[0.0, -1.0]]))
        except RuntimeError as exc:
            need(str(exc) == 'M5_ALL_SIDE_JOINT_COST_SUMMARY_DOMAIN',
                 'SYNTHETIC_FAILURE')
        else:
            need(False, 'SYNTHETIC_FAILURE')
        print(json.dumps({'preflight': 'PASS_NO_CLAIM_NO_PRICE_ARRAY_DECODE',
                          'fit_rows': 47338, 'later_rows': 4096,
                          'source_head': context[1]}, sort_keys=True))
    else:
        run(args.plan_sha256)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
