"""One guarded Fold 2 probabilistic current-state transition reference.

Fit a fixed categorical S(t) table on mature earlier TRAIN only, then score
selected later S/F native endpoints against the frozen Fold 2 prior. A prior
state is used only when its physical M1 row is available at the issue time.
No native model, TEST, Entry, broker, order or trade is involved.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from gx1.audit import measure_market_state_causal_persistence_reference_v1 as persistence
from gx1.audit import measure_market_state_fold2_exact_prior_later_v1 as prior
from gx1.contracts import market_state_five_endpoint_target_binding_v1 as state
from gx1.contracts import market_state_future_displacement_candidate_v1 as future
from gx1.contracts import market_state_time_query_native_F_objective_v1 as objective
from gx1.contracts.immutable_event_authority_v1 import _publish_bytes

REPO = prior.REPO
POLICY = prior.POLICY
STAGE = 'market_state_fold2_causal_transition_20261115_001'
PLAN = REPO / 'configs/research/MARKET_STATE_FOLD2_CAUSAL_TRANSITION_20261115_001.json'
ROOT = Path('/home/andre2/GX1_RUNS/MARKET_STATE_FOLD2_CAUSAL_TRANSITION_20261115_001')
BITS = (1, 2, 4, 5)
TASKS = ('state', 'future')
TIMEFRAMES = ('M5', 'M15', 'H1', 'H4', 'D1')
INPUTS = {
    **prior.INPUTS,
    'prior_result': (prior.ROOT / 'RESULT.json',
                     '252b397f794bb7663e7836b941ba69f33fcc58f31bb6e0796af448b408140b07'),
    'prior_terminal': (prior.ROOT / 'TERMINAL.json',
                       '304b7204faab52d9440f659a4e46bcd9da2f4de856963dcf37adb8a265a6b497'),
    'prior_frozen': (prior.ROOT / 'FROZEN_PRIORS.json',
                     'bd8ecf1d47d067d8ef3aee1f4bd5fa2241e618f615edc4792b95dbe194fa0eb4'),
}
CODE = tuple(sorted(set(prior.CODE) | {
    'gx1/audit/measure_market_state_causal_persistence_reference_v1.py',
    'gx1/audit/measure_market_state_fold2_causal_transition_v1.py',
    'gx1/contracts/market_state_five_endpoint_contract_v1.py',
}))


def need(ok: object, reason: str) -> None:
    if not bool(ok):
        raise RuntimeError('FOLD2_CAUSAL_TRANSITION_' + reason)


def terminal(status: str, *, claim: dict, result: dict | None,
             head: str, reason: str | None = None) -> dict:
    need(status in ('COMPLETE', 'FAILED_CONSUMED')
         and (result is not None) == (status == 'COMPLETE')
         and (reason is not None) == (status == 'FAILED_CONSUMED'),
         'TERMINAL_ARGUMENTS')
    value = {'schema_version': 'gx1_fold2_causal_transition_terminal_v1',
             'status': status, 'exit_code': 0 if status == 'COMPLETE' else 1,
             'stage': STAGE, 'claim': claim, 'result': result,
             'source_head': head, 'test_data_used': False}
    if reason is not None:
        value['reason'] = reason
    return value


def _probabilities(frozen: dict) -> dict:
    objective.require_endpoint_priors(frozen, fold_id=2)
    need(frozen['fit_rows'] == 47338 and
         frozen['native_query_k'] == [1, 3, 12, 48, 288], 'FROZEN_PRIOR')
    out = {}
    for task in TASKS:
        raw = np.asarray(frozen['tasks'][task]['raw_logits'], dtype=np.float64)
        need(raw.shape == (5, 2) and np.isfinite(raw).all(), 'PRIOR_LOGITS')
        p = 1 / (1 + np.exp(-raw))
        need(bool(np.all((p > 0) & (p < 1))), 'PRIOR_PROBABILITY')
        out[task] = p
    return out


def fit_table(now: np.ndarray, truths: dict, frozen: dict) -> dict:
    """One fixed prior-centered pseudo-observation per current state cell."""
    need(now.dtype == np.uint8 and now.ndim == 2 and now.shape[1] == 5
         and np.isin(now, (0, *BITS)).all()
         and set(truths) == set(TASKS), 'FIT_INPUT')
    p0 = _probabilities(frozen)
    out = {}
    for task in TASKS:
        truth = truths[task]
        allowed = truth['allowed']
        eligible = truth['eligible'] & truth['mature']
        need(allowed.shape == now.shape and eligible.shape == now.shape,
             'FIT_TRUTH_SHAPE')
        out[task] = {}
        for j, tf in enumerate(TIMEFRAMES):
            out[task][tf] = {}
            for bit in BITS:
                row = now[:, j] == bit
                activity = row & eligible[:, j] & np.isin(allowed[:, j], (1, 2, 4, 5))
                side = row & eligible[:, j] & np.isin(allowed[:, j], (1, 4))
                n_a, y_a = int(activity.sum()), int(np.count_nonzero(
                    activity & (allowed[:, j] != 2)))
                n_s, y_s = int(side.sum()), int(np.count_nonzero(
                    side & (allowed[:, j] == 4)))
                pa = (y_a + float(p0[task][j, 0])) / (n_a + 1)
                ps = (y_s + float(p0[task][j, 1])) / (n_s + 1)
                need(0 < pa < 1 and 0 < ps < 1, 'FIT_PROBABILITY')
                out[task][tf][str(bit)] = {
                    'activity': {'count': n_a, 'positive': y_a,
                                 'probability': pa},
                    'conditional_side': {'count': n_s, 'positive': y_s,
                                         'probability': ps}}
    return out


def logits(now: np.ndarray, frozen: dict, table: dict) -> dict:
    p0 = _probabilities(frozen)
    need(now.dtype == np.uint8 and now.ndim == 2 and now.shape[1] == 5,
         'APPLY_AXIS')
    out = {}
    for task in TASKS:
        p = np.broadcast_to(p0[task][None, :, :],
                            (len(now), 5, 2)).copy()
        for j, tf in enumerate(TIMEFRAMES):
            for bit in BITS:
                chosen = now[:, j] == bit
                p[chosen, j, 0] = table[task][tf][str(bit)]['activity']['probability']
                p[chosen, j, 1] = table[task][tf][str(bit)]['conditional_side']['probability']
        need(bool(np.isfinite(p).all() and np.all((p > 0) & (p < 1))),
             'APPLY_PROBABILITY')
        out[task] = np.log(p / (1 - p)).astype(np.float64)
    return out


def _current(bound: state.BoundFiveEndpointTargets, issues: np.ndarray) -> tuple:
    raw = state._checked_bytes('mask', 4 * 1024 * 1024)
    with np.load(io.BytesIO(raw), allow_pickle=False) as mask:
        source_issues = np.array(mask['issue_ns'], copy=True)
        source_m1 = np.array(mask['m1_source_row'], copy=True)
        source_allowed = np.array(mask['allowed_set'], copy=True)
    at = np.searchsorted(source_issues, bound.decision_ns)
    need(len(bound.parent_rows) == 118249
         and bool(np.all(at < len(source_issues)))
         and np.array_equal(bound.decision_ns, source_issues[at]),
         'CURRENT_SOURCE_AXIS')
    return persistence._current_states(source_issues, source_m1,
                                       source_allowed, issues)


def _scores(raw: dict, prior_raw: dict, truths: dict,
            issues: np.ndarray) -> dict:
    query = np.broadcast_to(objective.NATIVE_QUERY,
                            (len(issues), 5)).copy()
    out = {}
    for name, values in (('transition', raw), ('prior', prior_raw)):
        out[name] = {task: objective._endpoint_task_sums(
            values[task], truths[task], query) for task in TASKS}
    return out


def _subset(truths: dict, mask: np.ndarray) -> dict:
    return {task: {key: value[mask] if isinstance(value, np.ndarray)
                 and value.shape[0] == len(mask) else value
                 for key, value in truth.items()}
            for task, truth in truths.items()}


def _verify_prior(scores: dict, expected: dict) -> None:
    for task in TASKS:
        for tf in TIMEFRAMES:
            for metric in objective.METRICS:
                measured = scores['prior'][task][tf][metric]
                fixed = expected[task][tf][metric]
                need(measured['count'] == fixed['count']
                     and abs(measured['brier_sum'] - fixed['brier_sum']) < 1e-5
                     and abs(measured['log_sum'] - fixed['log_sum']) < 1e-5,
                     'PRIOR_SCORE_PARITY:' + task + ':' + tf + ':' + metric)


def inspect(plan_sha: str) -> tuple:
    need(Path.cwd().resolve() == REPO
         and prior.git('branch', '--show-current') == 'work/gx1-current'
         and prior.git('status', '--porcelain') == '', 'CLEAN_SOURCE')
    head = prior.git('rev-parse', 'HEAD')
    need(not ROOT.exists() and prior.sha(PLAN) == plan_sha, 'PLAN_OUTPUT')
    plan = json.loads(PLAN.read_text())
    need(set(plan) == {'schema_version', 'stage', 'status', 'source_base_head',
                       'input_refs', 'source_owners', 'scope', 'limits', 'output_root'}
         and plan['schema_version'] == 'gx1_fold2_causal_transition_plan_v1'
         and plan['stage'] == STAGE and plan['status'] == 'PREPARED_NOT_ADMITTED'
         and plan['scope'] == {
             'fold_id': 2, 'fit_rows': 47338, 'later_rows': 4096,
             'timeframes': list(TIMEFRAMES), 'tasks': list(TASKS),
             'primary_cell': 'state_M5_conditional_side',
             'current_state_bits': list(BITS),
             'missing_current_fallback': 'frozen_prior',
             'smoothing': 'one_frozen_prior_pseudo_observation',
             'test_data_used': False}
         and plan['limits'] == {
             'class': 'audit', 'memory': '4G', 'swap': '512M',
             'wall_seconds': 900, 'max_invocations': 1,
             'model_forwards': 0, 'optimizer_steps': 0,
             'native_training_authorized': False}
         and plan['output_root'] == str(ROOT), 'PLAN')
    need(subprocess.run(('git', 'merge-base', '--is-ancestor',
                         plan['source_base_head'], head), cwd=REPO).returncode == 0,
         'ANCESTOR')
    need(set(plan['source_owners']) == set(CODE)
         and all(prior.sha(REPO / path) == digest
                 for path, digest in plan['source_owners'].items()),
         'SOURCE_HASH')
    expected = {name: {'path': str(path), 'sha256': digest}
                for name, (path, digest) in INPUTS.items()}
    need(plan['input_refs'] == expected
         and all(prior.sha(path) == digest for path, digest in INPUTS.values()),
         'INPUT_HASH')
    prior_result = json.loads(INPUTS['prior_result'][0].read_text())
    prior_terminal = json.loads(INPUTS['prior_terminal'][0].read_text())
    frozen = json.loads(INPUTS['prior_frozen'][0].read_text())
    need(prior_result['status'] == prior_terminal['status'] == 'COMPLETE'
         and prior_terminal['exit_code'] == 0
         and prior_terminal['result'] == expected['prior_result']
         and prior_result['frozen_priors'] == expected['prior_frozen']
         and prior_result['fit_rows'] == 47338
         and prior_result['selected_score_rows'] == 4096
         and prior_result['model_forwards'] == prior_result['optimizer_steps'] == 0
         and prior_result['test_data_used'] is False, 'PRIOR_RECEIPTS')
    _probabilities(frozen)
    bound = state.load_bound_five_endpoint_targets()
    fbound = future.load_bound_future_displacement()
    need(np.array_equal(bound.parent_rows, fbound.parent_rows)
         and np.array_equal(bound.decision_ns, fbound.decision_ns),
         'TARGET_AXIS_JOIN')
    fit, later_rows, later_issues = prior.axes(
        json.loads(prior.COVERAGE.read_text()), bound)
    need(frozen['fit_parent_rows_sha256'] == fit['fit_parent_rows_sha256']
         and frozen['fit_decision_ns_sha256'] == fit['fit_decision_ns_sha256']
         and prior_result['selected_parent_rows_sha256']
             == json.loads(prior.COVERAGE.read_text())['selected_parent_rows_sha256'],
         'FROZEN_AXES')
    policy = json.loads(POLICY.read_text())
    need(policy.get('training_enabled') is False
         and policy.get('native_training_enabled') is False
         and policy.get('native_launch_authorized') is False
         and policy.get(STAGE) == {
             'status': 'ADMITTED', 'plan': prior.ref(PLAN),
             'output_root': str(ROOT), 'max_invocations': 1,
             'claimed_invocations': 0, 'completed_invocations': 0,
             'execution_authorized': True, 'data_reads_authorized': True,
             'baseline_estimation_authorized': True,
             'model_forwards_authorized': 0, 'optimizer_steps_authorized': 0,
             'native_training_authorized': False, 'test_authorized': False,
             'broker_access_allowed': False, 'relaunch_allowed': False},
         'POLICY')
    return prior.ref(POLICY), head, bound, fbound, frozen, prior_result, fit, later_rows, later_issues


def measure(context: tuple) -> dict:
    policy_ref, head, bound, fbound, frozen, prior_result, fit, rows, issues = context
    fit_rows, fit_issues = fit['fit_parent_rows'], fit['fit_decision_ns']
    fit_truth = {task: owner.gather(fit_rows, fit_issues,
                 fold_id=2, partition='fit')
                 for task, owner in (('state', bound), ('future', fbound))}
    later_truth = {task: owner.gather(rows, issues, fold_id=2,
                   partition='score')
                   for task, owner in (('state', bound), ('future', fbound))}
    need(all(bool(x['mature'].all()) for x in fit_truth.values()),
         'FIT_MATURITY')
    fit_now, fit_reasons = _current(bound, fit_issues)
    later_now, later_reasons = _current(bound, issues)
    table = fit_table(fit_now, fit_truth, frozen)
    raw = logits(later_now, frozen, table)
    prior_raw = {task: np.broadcast_to(
        np.asarray(frozen['tasks'][task]['raw_logits'], dtype=np.float64)[None],
        (len(issues), 5, 2)).copy() for task in TASKS}
    overall = _scores(raw, prior_raw, later_truth, issues)
    _verify_prior(overall, prior_result['scores']['overall'])
    week_ids = objective._weeks(issues)
    weekly = {}
    for week in np.unique(week_ids):
        mask = week_ids == week
        item = _scores({task: value[mask] for task, value in raw.items()},
                       {task: value[mask] for task, value in prior_raw.items()},
                       _subset(later_truth, mask), issues[mask])
        _verify_prior(item, prior_result['scores']['by_utc_week'][str(week)])
        weekly[str(week)] = item
    need(len(weekly) == 14
         and np.array_equal(fit_now == 0, fit_reasons != 0)
         and np.array_equal(later_now == 0, later_reasons != 0),
         'CURRENT_COVERAGE')
    coverage = {}
    for name, values, reasons in (('fit', fit_now, fit_reasons),
                                  ('later', later_now, later_reasons)):
        coverage[name] = {tf: {
            'rows': len(values),
            'known_current': int(np.count_nonzero(values[:, j] != 0)),
            'current_bits': {str(bit): int(np.count_nonzero(values[:, j] == bit))
                             for bit in BITS},
            'no_current_reasons': {str(reason): int(np.count_nonzero(
                reasons[:, j] == reason)) for reason in range(1, 6)}}
            for j, tf in enumerate(TIMEFRAMES)}
    return {'schema_version': 'gx1_fold2_causal_transition_result_v1',
            'status': 'COMPLETE', 'stage': STAGE,
            'plan': prior.ref(PLAN), 'policy': policy_ref,
            'source_head': head,
            'observed_utc': datetime.now(timezone.utc).isoformat(),
            'frozen_priors': prior.ref(INPUTS['prior_frozen'][0]),
            'fit_rows': len(fit_rows), 'selected_later_rows': len(rows),
            'table': table, 'coverage': coverage,
            'scores': {'overall': overall, 'by_utc_week': weekly},
            'prior_score_parity': True, 'model_forwards': 0,
            'optimizer_steps': 0, 'native_model_fit': False,
            'baseline_estimation': True, 'test_data_used': False,
            'trade_selection': False}


def run(plan_sha: str) -> None:
    from gx1.contracts.gx1_capped_execution_v1 import (
        require_capped_cpu_audit_execution, require_capped_lock_ancestry)
    context = inspect(plan_sha)
    cap = require_capped_cpu_audit_execution()
    lock = require_capped_lock_ancestry()
    need(cap['memory_max'] == cap['memory_high'] == 4 * 1024**3
         and cap['swap'] == 512 * 1024**2 and lock is not None, 'CAP')
    ROOT.mkdir(parents=False, exist_ok=False)
    _publish_bytes(ROOT / 'CLAIM.json', prior.encode({
        'schema_version': 'gx1_fold2_causal_transition_claim_v1',
        'stage': STAGE, 'plan': prior.ref(PLAN),
        'policy': context[0], 'source_head': context[1],
        'test_data_used': False}))
    claim = prior.ref(ROOT / 'CLAIM.json')
    started = time.monotonic()
    try:
        result = measure(context)
        result['claim'] = claim
        result['elapsed_seconds'] = time.monotonic() - started
        need(prior.git('rev-parse', 'HEAD') == context[1]
             and prior.git('status', '--porcelain') == ''
             and prior.ref(POLICY) == context[0]
             and all(prior.sha(path) == digest
                     for path, digest in INPUTS.values()),
             'FINAL_SOURCE')
        _publish_bytes(ROOT / 'RESULT.json', prior.encode(result))
        _publish_bytes(ROOT / 'TERMINAL.json', prior.encode(
            terminal('COMPLETE', claim=claim,
                     result=prior.ref(ROOT / 'RESULT.json'),
                     head=context[1])))
    except BaseException as exc:
        _publish_bytes(ROOT / 'TERMINAL.json', prior.encode(
            terminal('FAILED_CONSUMED', claim=claim, result=None,
                     head=context[1],
                     reason=type(exc).__name__ + ':' + str(exc))))
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
        for sample in (context[6]['fit_decision_ns'][:16], context[-1][:16]):
            current, reasons = _current(context[2], sample)
            need(current.shape == reasons.shape == (16, 5)
                 and np.array_equal(current == 0, reasons != 0),
                 'REAL_CURRENT_CLOCK_PREFLIGHT')
        p = np.asarray([[1]*5, [4]*5, [0]*5, [5]*5], dtype=np.uint8)
        truth = {task: {'allowed': np.asarray([[4]*5,[1]*5,[2]*5,[5]*5],dtype=np.uint8),
                        'eligible': np.ones((4,5),dtype=np.bool_),
                        'mature': np.ones((4,5),dtype=np.bool_)} for task in TASKS}
        table = fit_table(p, truth, context[4])
        raw = logits(p, context[4], table)
        need(all(raw[task].shape == (4,5,2) and np.isfinite(raw[task]).all()
                 for task in TASKS)
             and np.allclose(raw['state'][2], np.asarray(
                 context[4]['tasks']['state']['raw_logits'], dtype=np.float64),
                 rtol=0, atol=1e-12)
             and table['state']['M5']['5']['activity']['count'] == 1
             and table['state']['M5']['5']['conditional_side']['count'] == 0,
             'SYNTHETIC_SUCCESS')
        for status, result, reason in (
            ('COMPLETE', {'path': 'synthetic', 'sha256': '0'*64}, None),
            ('FAILED_CONSUMED', None, 'synthetic_failure')):
            x = terminal(status, claim={'path': 'synthetic', 'sha256': '1'*64},
                         result=result, head=context[1], reason=reason)
            need((x['exit_code'] == 0) == (status == 'COMPLETE'),
                 'SYNTHETIC_TERMINAL')
        print(json.dumps({'preflight': 'PASS_NO_CLAIM_NO_TARGET_SCORE',
                          'source_head': context[1], 'fit_rows': 47338,
                          'later_rows': len(context[-2]),
                          'prior_contract_sha256': context[4]['contract_sha256']},
                         sort_keys=True))
    else:
        run(args.plan_sha256)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
