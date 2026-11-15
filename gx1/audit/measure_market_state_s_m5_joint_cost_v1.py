"""Common-mask terminal net, MFE and pre-MFE MAE for fixed S/M5 audits.

The side is known only in hindsight. No model decision, Entry rule, Exit,
TEST, broker, trade, or horizon selection is performed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import torch

from gx1.audit.measure_market_state_fold2_exact_prior_later_v1 import axes
from gx1.audit.measure_market_state_s_m5_quote_alignment_v1 import (
    encode, git, ref, sha, summary,
)
from gx1.audit import measure_market_state_s_m5_path_cost_alignment_v1 as old_worker
from gx1.contracts import entry_causal_m1_outcomes_v1 as path
from gx1.contracts import market_state_five_endpoint_target_binding_v1 as state
from gx1.contracts import market_state_time_query_native_F_objective_v1 as objective
from gx1.contracts.entry_observed_market_v1 import (
    build_entry_observed_market_targets, read_dynamic_m5_target_source,
    require_entry_observed_market_scope,
)
from gx1.contracts.immutable_event_authority_v1 import _publish_bytes

REPO = Path('/home/andre2/src/GX1_CURRENT')
STAGE = 'market_state_s_m5_joint_cost_20261115_001'
PLAN = REPO / 'configs/research/MARKET_STATE_S_M5_JOINT_COST_20261115_001.json'
POLICY = REPO / 'NEXT_RUN_POLICY.json'
ROOT = Path('/home/andre2/GX1_RUNS/MARKET_STATE_S_M5_JOINT_COST_20261115_001')
OLD_ROOT = old_worker.ROOT
HORIZONS = old_worker.HORIZONS
MINUTE_NS = old_worker.MINUTE_NS
M5_NS = old_worker.M5_NS
FIT_CUTOFF = old_worker.FIT_CUTOFF
LATER_CUTOFF = old_worker.LATER_CUTOFF
CODE = (
    'gx1/audit/measure_market_state_s_m5_joint_cost_v1.py',
    'gx1/audit/measure_market_state_s_m5_path_cost_alignment_v1.py',
    'gx1/audit/measure_market_state_fold2_exact_prior_later_v1.py',
    'gx1/audit/measure_market_state_s_m5_quote_alignment_v1.py',
    'gx1/contracts/entry_causal_m1_outcomes_v1.py',
    'gx1/contracts/entry_observed_market_v1.py',
    'gx1/contracts/market_state_five_endpoint_target_binding_v1.py',
    'gx1/contracts/market_state_time_query_native_F_objective_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'scripts/gx1_capped_run.sh',
)
INPUTS = {
    'old_result': (OLD_ROOT / 'RESULT.json',
                   '22d3a4f7e0311afa2c1bbd940a15b70d1c3514339a3be40b8d79c4e6ab4d5866'),
    'old_terminal': (OLD_ROOT / 'TERMINAL.json',
                     '6997d52c887c18d86f5e751ff3250a14d038e777951e441b799e183a480e5961'),
    'old_plan': (old_worker.PLAN,
                 'e24fc5971c158e7490338ae1f95e29d30555f1946ce7de852f800a945ac77c7d'),
    'coverage': old_worker.INPUTS['coverage'],
    'recipe': old_worker.INPUTS['recipe'],
    'physical_receipt': old_worker.INPUTS['physical_receipt'],
    'm1': old_worker.INPUTS['m1'],
    'state_mask': old_worker.INPUTS['state_mask'],
    'model_axes': old_worker.INPUTS['model_axes'],
}


def need(ok: object, reason: str) -> None:
    if not bool(ok):
        raise RuntimeError('S_M5_JOINT_COST_' + reason)


def axis_sha(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype='<i8').tobytes()).hexdigest()


def paired(net: np.ndarray, mfe: np.ndarray, mae: np.ndarray) -> dict:
    need(net.shape == mfe.shape == mae.shape
         and net.ndim == 1 and len(net) > 0
         and bool(np.isfinite(net).all())
         and bool(np.isfinite(mfe).all())
         and bool(np.isfinite(mae).all())
         and bool(np.all(mae >= 0)), 'PAIR_DOMAIN')
    return {'n': len(net), 'terminal_net_bps': summary(net),
            'mfe_bps': summary(mfe),
            'pre_mfe_mae_upper_bps': summary(mae),
            'net_minus_pre_mfe_mae_upper_bps': summary(net - mae)}


def inspect(plan_sha: str) -> tuple[dict, dict, str, object, dict, dict, dict, dict]:
    need(Path.cwd().resolve() == REPO
         and git('branch', '--show-current') == 'work/gx1-current'
         and git('status', '--porcelain') == '', 'CLEAN_SOURCE')
    head = git('rev-parse', 'HEAD')
    need(not ROOT.exists() and sha(PLAN) == plan_sha, 'PLAN_OUTPUT')
    plan = json.loads(PLAN.read_text())
    need(set(plan) == {'schema_version', 'stage', 'status', 'source_base_head',
                       'input_refs', 'source_owners', 'scope', 'limits', 'output_root'}
         and plan['schema_version'] == 'gx1_s_m5_joint_cost_plan_v1'
         and plan['stage'] == STAGE and plan['status'] == 'PREPARED_NOT_ADMITTED'
         and plan['scope'] == {
             'fold_id': 2, 'fit_rows': 47338, 'later_rows': 4096,
             'horizon_minutes': list(HORIZONS),
             'fit_cutoff_ns_exclusive': FIT_CUTOFF,
             'later_cutoff_ns_exclusive': LATER_CUTOFF,
             'side_selection': 'hindsight_certified_S_only',
             'entry_timeframe': 'M5', 'exit_evaluation': False,
             'test_data_used': False}
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
         and all(sha(REPO / rel) == digest
                 for rel, digest in plan['source_owners'].items()), 'SOURCE_HASH')
    expected = {name: {'path': str(p), 'sha256': digest}
                for name, (p, digest) in INPUTS.items()}
    need(plan['input_refs'] == expected
         and all(sha(p) == digest for p, digest in INPUTS.values()), 'INPUT_HASH')
    old = json.loads((OLD_ROOT / 'RESULT.json').read_text())
    terminal = json.loads((OLD_ROOT / 'TERMINAL.json').read_text())
    need(old['status'] == terminal['status'] == 'COMPLETE'
         and terminal['exit_code'] == 0
         and terminal['result'] == expected['old_result']
         and old['plan'] == expected['old_plan']
         and old['horizon_minutes'] == list(HORIZONS)
         and old['fit']['rows'] == 47338
         and old['later']['rows'] == 4096
         and old['model_forwards'] == old['optimizer_steps'] == 0
         and old['test_data_used'] is False
         and old['trade_selection'] is False, 'OLD_RECEIPT')
    bound = state.load_bound_five_endpoint_targets()
    coverage = json.loads(old_worker.COVERAGE.read_text())
    fit, rows, issues = axes(coverage, bound)
    need(old['fit']['parent_rows_sha256'] == fit['fit_parent_rows_sha256']
         and old['fit']['decision_ns_sha256'] == fit['fit_decision_ns_sha256']
         and old['later']['parent_rows_sha256']
             == coverage['selected_parent_rows_sha256']
         and old['later']['decision_ns_sha256']
             == coverage['selected_decision_ns_sha256'], 'EXACT_AXES')
    scope = require_entry_observed_market_scope(
        json.loads(old_worker.RECIPE.read_text()))
    need(scope['economics'] == old['economics']
         and scope['target_m1_source_sha256'] == old_worker.M1_SHA
         and scope['train_control_cutoff'] == '2026-01-01T00:00:00+00:00',
         'COST_SOURCE')
    policy = json.loads(POLICY.read_text())
    gate = policy.get(STAGE)
    need(policy.get('training_enabled') is False
         and policy.get('native_training_enabled') is False
         and policy.get('native_launch_authorized') is False
         and gate == {
             'status': 'ADMITTED', 'plan': ref(PLAN),
             'output_root': str(ROOT), 'max_invocations': 1,
             'claimed_invocations': 0, 'completed_invocations': 0,
             'execution_authorized': True, 'data_reads_authorized': True,
             'model_forwards_authorized': 0, 'optimizer_steps_authorized': 0,
             'training_authorized': False, 'test_authorized': False,
             'broker_access_allowed': False, 'relaunch_allowed': False},
         'POLICY')
    return plan, ref(POLICY), head, bound, fit, {
        'rows': rows, 'issues': issues}, scope, old


def one_cohort(source: dict, bound: object, old: dict, *,
               rows: np.ndarray, issues: np.ndarray,
               partition: str, cutoff: int, economics: dict) -> dict:
    need(partition in ('fit', 'score') and len(rows) == len(issues)
         and len(rows) == old['rows'], 'COHORT')
    bars = pd.DatetimeIndex(pd.to_datetime(
        source['frame']['time'].iloc[rows], utc=True)).as_unit('ns').asi8
    need(np.array_equal(bars + M5_NS, issues), 'PHYSICAL_CLOCK')
    truth = bound.gather(rows, issues, fold_id=2, partition=partition)
    bits = truth['allowed'][:, 0]
    known = np.isin(bits, (1, 4)) & truth['eligible'][:, 0] & truth['mature'][:, 0]
    need(int(known.sum()) == old['s_m5_known_side_rows'], 'KNOWN_SIDE')
    output = {}
    for horizon in HORIZONS:
        previous = old['horizons'][str(horizon)]
        allowed = issues + horizon * MINUTE_NS < cutoff
        qrows, qissues, qbits, qknown = rows[allowed], issues[allowed], bits[allowed], known[allowed]
        need(len(qrows) == previous['eligible_rows'], 'ELIGIBLE')
        obs = path.query_dynamic_m1_prefix(
            source['index'], fill_surface=source['fill'],
            query_entry_rows=qrows,
            horizon_minutes=np.full(len(qrows), horizon, dtype=np.int64),
            endpoint_kind='terminal_open_inclusive')
        need(np.array_equal(obs['query_entry_rows'], qrows)
             and np.array_equal(obs['terminal_time_ns'],
                                qissues + horizon * MINUTE_NS)
             and bool(np.all(obs['terminal_time_ns'] < cutoff)), 'QUERY')
        term = obs['terminal_gross_observed']
        pathmask = obs['path_observed']
        need(term.shape == pathmask.shape == (len(qrows), 2)
             and np.array_equal(term[:, 0], term[:, 1])
             and np.array_equal(pathmask[:, 0], pathmask[:, 1])
             and bool(np.all(~pathmask[:, 0] | term[:, 0])), 'MASK')
        known_term = qknown & term[:, 0]
        joint = known_term & pathmask[:, 0]
        counts = previous['counts']
        need(int(term[:, 0].sum()) == counts['terminal_observed']
             and int(pathmask[:, 0].sum()) == counts['path_observed']
             and int(known_term.sum()) == counts['known_s_terminal']
             and int(joint.sum()) == counts['known_s_joint']
             and bool(joint.any()), 'COUNT_REPLAY')
        selected = np.flatnonzero(known_term)
        gross = torch.tensor(obs['terminal_gross_return_bps'][known_term], dtype=torch.float64)
        elapsed = torch.tensor(obs['elapsed_wall_seconds'][known_term], dtype=torch.int64)
        net = build_entry_observed_market_targets(
            gross_return_bps=gross,
            elapsed_wall_clock_seconds=elapsed,
            economics=economics)['target_bps'][:, :2].numpy()
        side = np.where(qbits[known_term] == 4, 0, 1)
        selected_net = net[np.arange(len(net)), side]
        need(np.isclose(float(selected_net.mean()),
                        previous['hindsight_s_side_net_bps']['mean_bps'],
                        rtol=0, atol=1e-10), 'NET_REPLAY')
        inside = pathmask[selected, 0]
        net_joint = selected_net[inside]
        joint_rows = np.flatnonzero(joint)
        joint_side = np.where(qbits[joint] == 4, 0, 1)
        mfe = obs['mfe_bps'][joint_rows, joint_side]
        mae = obs['mae_before_first_mfe_upper_bps'][joint_rows, joint_side]
        measured = paired(net_joint, mfe, mae)
        need(measured['n'] == counts['known_s_joint']
             and np.isclose(measured['mfe_bps']['mean_bps'],
                            previous['hindsight_s_side_mfe_bps']['mean_bps'],
                            rtol=0, atol=1e-10)
             and np.isclose(measured['pre_mfe_mae_upper_bps']['mean_bps'],
                            previous['hindsight_s_side_pre_mfe_mae_upper_bps']['mean_bps'],
                            rtol=0, atol=1e-10), 'PATH_REPLAY')
        weeks = objective._weeks(qissues[joint])
        weekly = {str(week): paired(net_joint[weeks == week],
                                    mfe[weeks == week], mae[weeks == week])
                  for week in np.unique(weeks)}
        need(sum(x['n'] for x in weekly.values()) == measured['n'], 'WEEKS')
        output[str(horizon)] = {
            'horizon_minutes': horizon,
            'joint_parent_rows_sha256': axis_sha(qrows[joint]),
            'joint_issue_ns_sha256': axis_sha(qissues[joint]),
            'common_mask': measured,
            'by_utc_issue_week': weekly,
            'terminal_only_excluded_rows': int(known_term.sum() - joint.sum()),
            'old_result_counts_replayed': True,
            'old_terminal_net_mean_replayed': True,
            'old_path_means_replayed': True}
    return output


def run(plan_sha: str) -> None:
    from gx1.contracts.gx1_capped_execution_v1 import (
        require_capped_cpu_audit_execution, require_capped_lock_ancestry)
    plan, policy_ref, head, bound, fit, later, scope, old = inspect(plan_sha)
    cap = require_capped_cpu_audit_execution()
    lock = require_capped_lock_ancestry()
    need(cap['memory_max'] == cap['memory_high'] == 4 * 1024**3
         and cap['swap'] == 512 * 1024**2 and lock is not None, 'CAP')
    ROOT.mkdir(parents=False, exist_ok=False)
    _publish_bytes(ROOT / 'CLAIM.json', encode({
        'schema_version': 'gx1_s_m5_joint_cost_claim_v1',
        'stage': STAGE, 'plan': ref(PLAN), 'policy': policy_ref,
        'source_head': head, 'test_data_used': False}))
    started = time.monotonic()
    try:
        source = read_dynamic_m5_target_source(scope, split='train')
        fit_result = one_cohort(
            source, bound, old['fit'], rows=fit['fit_parent_rows'],
            issues=fit['fit_decision_ns'], partition='fit',
            cutoff=FIT_CUTOFF, economics=scope['economics'])
        later_result = one_cohort(
            source, bound, old['later'], rows=later['rows'],
            issues=later['issues'], partition='score',
            cutoff=LATER_CUTOFF, economics=scope['economics'])
        result = {
            'schema_version': 'gx1_s_m5_joint_cost_result_v1',
            'status': 'COMPLETE', 'stage': STAGE,
            'plan': ref(PLAN), 'policy': policy_ref,
            'claim': ref(ROOT / 'CLAIM.json'),
            'old_result': ref(OLD_ROOT / 'RESULT.json'),
            'source_head': head,
            'observed_utc': datetime.now(timezone.utc).isoformat(),
            'economics': scope['economics'],
            'horizon_minutes': list(HORIZONS),
            'fit': fit_result, 'later': later_result,
            'model_forwards': 0, 'optimizer_steps': 0,
            'test_data_used': False, 'trade_selection': False,
            'elapsed_seconds': time.monotonic() - started}
        need(git('rev-parse', 'HEAD') == head
             and git('status', '--porcelain') == ''
             and ref(POLICY) == policy_ref
             and all(sha(p) == digest for p, digest in INPUTS.values()),
             'FINAL_SOURCE')
        _publish_bytes(ROOT / 'RESULT.json', encode(result))
        _publish_bytes(ROOT / 'TERMINAL.json', encode({
            'schema_version': 'gx1_s_m5_joint_cost_terminal_v1',
            'status': 'COMPLETE', 'exit_code': 0,
            'stage': STAGE, 'claim': ref(ROOT / 'CLAIM.json'),
            'result': ref(ROOT / 'RESULT.json'),
            'source_head': head, 'test_data_used': False}))
    except BaseException as exc:
        _publish_bytes(ROOT / 'TERMINAL.json', encode({
            'schema_version': 'gx1_s_m5_joint_cost_terminal_v1',
            'status': 'FAILED_CONSUMED', 'exit_code': 1,
            'stage': STAGE, 'claim': ref(ROOT / 'CLAIM.json'),
            'result': None, 'source_head': head,
            'reason': type(exc).__name__ + ':' + str(exc),
            'test_data_used': False}))
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    need(args.preflight != args.run, 'MODE')
    if args.preflight:
        _, _, head, _, fit, later, scope, old = inspect(args.plan_sha256)
        need(len(fit['fit_parent_rows']) == old['fit']['rows']
             and len(later['rows']) == old['later']['rows']
             and scope['economics'] == old['economics'], 'PREFLIGHT')
        toy = paired(np.array([1.0, -1.0]),
                     np.array([2.0, 3.0]), np.array([0.5, 1.0]))
        need(toy['n'] == 2
             and toy['net_minus_pre_mfe_mae_upper_bps']['mean_bps'] == -0.75,
             'SYNTHETIC_SUCCESS')
        try:
            paired(np.array([1.0]), np.array([2.0]), np.array([-0.5]))
        except RuntimeError as exc:
            need(str(exc) == 'S_M5_JOINT_COST_PAIR_DOMAIN', 'SYNTHETIC_FAILURE')
        else:
            need(False, 'SYNTHETIC_FAILURE')
        print(json.dumps({'preflight': 'PASS_NO_CLAIM_NO_PRICE_ARRAY_DECODE',
                          'fit_rows': 47338, 'later_rows': 4096,
                          'source_head': head}, sort_keys=True))
    else:
        run(args.plan_sha256)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
