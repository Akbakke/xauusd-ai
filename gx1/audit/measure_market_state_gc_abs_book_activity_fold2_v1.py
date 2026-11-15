"""One guarded Fold 2 GC absolute-book-imbalance activity reference.

Fit only nonnegative univariate activity slopes on mature earlier TRAIN,
with the frozen prior as intercept. Score fixed later S/F native endpoints.
No native model, Entry, TEST, broker or trade is involved.
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
from scipy.optimize import minimize
from scipy.special import expit

from gx1.audit import measure_market_state_gc_book_tw_fold2_v1 as book
from gx1.audit import measure_market_state_fold2_causal_transition_v1 as transition
from gx1.contracts import market_state_five_endpoint_target_binding_v1 as state
from gx1.contracts import market_state_future_displacement_candidate_v1 as future
from gx1.contracts import market_state_time_query_native_F_objective_v1 as objective
from gx1.contracts.immutable_event_authority_v1 import _publish_bytes

REPO = book.REPO
STAGE = 'market_state_gc_abs_book_activity_fold2_20261115_001'
PLAN = REPO / 'configs/research/MARKET_STATE_GC_ABS_BOOK_ACTIVITY_FOLD2_20261115_001.json'
POLICY = book.POLICY
ROOT = Path('/home/andre2/GX1_RUNS/MARKET_STATE_GC_ABS_BOOK_ACTIVITY_FOLD2_20261115_001')
TASKS = ('state', 'future')
TIMEFRAMES = ('M5', 'M15', 'H1', 'H4', 'D1')
CODE = tuple(sorted(set(book.CODE) | {
    'gx1/audit/measure_market_state_fold2_causal_transition_v1.py',
    'gx1/audit/measure_market_state_gc_abs_book_activity_fold2_v1.py',
    'gx1/contracts/market_state_time_query_scorer_v2.py',
}))
INPUTS = {
    **book.INPUTS,
    'book_plan': (book.PLAN,
                  'e24a27042f799852f417a1d2e2caef1fa238be8caa8de20fe27365a1c3986e0a'),
    'book_result': (book.ROOT / 'RESULT.json',
                    'b06066116a451204fab812afdeeaad8885cd8858fced70142cb91e0de83263c5'),
    'book_terminal': (book.ROOT / 'TERMINAL.json',
                      '9e989ca67e7a54365998d067be795b711b14e305c27dd3b1bc88ca45b7010b7e'),
}


def need(value: object, reason: str) -> None:
    if not bool(value):
        raise RuntimeError('GC_ABS_BOOK_ACTIVITY_FOLD2_' + reason)


def fit_slope(z: np.ndarray, active: np.ndarray, intercept: float) -> dict:
    need(z.ndim == active.ndim == 1 and z.shape == active.shape
         and len(z) > 0 and active.dtype == np.bool_
         and np.isfinite(z).all() and np.isfinite(intercept),
         'SLOPE_INPUT')
    y = active.astype(np.float64)
    def objective_and_gradient(x: np.ndarray) -> tuple[float, np.ndarray]:
        logits = intercept + x[0] * z
        loss = np.logaddexp(0.0, logits) - y * logits
        gradient = np.dot(expit(logits) - y, z)
        return float(loss.sum(dtype=np.float64)), np.asarray([gradient], dtype=np.float64)
    result = minimize(objective_and_gradient, np.array([0.0]),
                      method='L-BFGS-B', jac=True, bounds=((0.0, None),),
                      options={'maxiter': 200, 'ftol': 1e-12})
    slope = float(result.x[0])
    need(result.success and np.isfinite(slope) and slope >= 0
         and np.isfinite(result.fun), 'SLOPE_FIT')
    prior_loss = objective_and_gradient(np.array([0.0]))[0]
    need(result.fun <= prior_loss + 1e-6, 'SLOPE_TRAIN_LOSS')
    return {'count': len(z), 'active': int(active.sum()),
            'slope_nonnegative': slope,
            'fit_log_mean': float(result.fun / len(z)),
            'prior_log_mean_on_observed_fit': float(prior_loss / len(z))}


def inspect(plan_sha: str) -> tuple:
    need(Path.cwd().resolve() == REPO
         and book.git('branch', '--show-current') == 'work/gx1-current'
         and book.git('status', '--porcelain') == '', 'CLEAN_SOURCE')
    head = book.git('rev-parse', 'HEAD')
    need(not ROOT.exists() and book.sha(PLAN) == plan_sha, 'PLAN_OUTPUT')
    plan = json.loads(PLAN.read_text())
    need(set(plan) == {'schema_version', 'stage', 'status', 'source_base_head',
                       'input_refs', 'source_owners', 'scope', 'limits', 'output_root'}
         and plan['schema_version'] == 'gx1_gc_abs_book_activity_fold2_plan_v1'
         and plan['stage'] == STAGE and plan['status'] == 'PREPARED_NOT_ADMITTED'
         and plan['scope'] == {
             'fold_id': 2, 'fit_rows': 47338, 'later_rows': 4096,
             'tasks': list(TASKS), 'timeframes': list(TIMEFRAMES),
             'primary_cell': 'state_M5_activity',
             'source_field': book.FIELD,
             'transform': 'absolute_value',
             'positive_orientation': 'higher_absolute_imbalance_higher_activity',
             'fit_standardization': 'observed_earlier_fit_rows_only',
             'probability_model': 'frozen_prior_intercept_plus_nonnegative_univariate_slope',
             'missing_field_fallback': 'frozen_prior',
             'test_data_used': False}
         and plan['limits'] == {
             'class': 'audit', 'memory': '4G', 'swap': '512M',
             'wall_seconds': 1800, 'max_invocations': 1,
             'baseline_estimation': True, 'model_forwards': 0,
             'optimizer_steps': 0, 'native_training_authorized': False}
         and plan['output_root'] == str(ROOT), 'PLAN')
    need(subprocess.run(('git', 'merge-base', '--is-ancestor',
                         plan['source_base_head'], head), cwd=REPO).returncode == 0,
         'ANCESTOR')
    need(set(plan['source_owners']) == set(CODE)
         and all(book.sha(REPO / name) == digest
                 for name, digest in plan['source_owners'].items()),
         'SOURCE_HASH')
    expected = {name: {'path': str(p), 'sha256': digest}
                for name, (p, digest) in INPUTS.items()}
    need(plan['input_refs'] == expected
         and all(book.sha(p) == digest for p, digest in INPUTS.values()),
         'INPUT_HASH')
    prior_result = json.loads(INPUTS['prior_result'][0].read_text())
    prior_terminal = json.loads(INPUTS['prior_terminal'][0].read_text())
    book_result = json.loads(INPUTS['book_result'][0].read_text())
    book_terminal = json.loads(INPUTS['book_terminal'][0].read_text())
    frozen = json.loads(INPUTS['prior_frozen'][0].read_text())
    need(prior_result['status'] == prior_terminal['status'] == 'COMPLETE'
         and prior_terminal['result'] == expected['prior_result']
         and book_result['status'] == book_terminal['status'] == 'COMPLETE'
         and book_terminal['result'] == expected['book_result']
         and book_result['source_field']['field'] == book.FIELD
         and book_result['fit']['rows'] == 47338
         and book_result['later']['rows'] == 4096
         and book_result['model_forwards'] == book_result['optimizer_steps'] == 0
         and book_result['test_data_used'] is False,
         'RECEIPTS')
    transition._probabilities(frozen)
    meta = book.legacy_source_metadata()
    registry = meta['registry']['field_report'][book.FIELD]
    need(registry['finite_observed'] == 117628
         and registry['masked_float32_train_sha256']
             == book_result['source_field']['masked_float32_train_sha256'],
         'SOURCE_FIELD')
    bound = state.load_bound_five_endpoint_targets()
    fbound = future.load_bound_future_displacement()
    need(np.array_equal(bound.parent_rows, fbound.parent_rows)
         and np.array_equal(bound.decision_ns, fbound.decision_ns),
         'TARGET_JOIN')
    fit, later_rows, later_issues = book.prior.axes(
        json.loads(book.prior.COVERAGE.read_text()), bound)
    need(len(fit['fit_parent_rows']) == 47338 and len(later_rows) == 4096,
         'TARGET_AXES')
    policy = json.loads(POLICY.read_text())
    need(policy.get('training_enabled') is False
         and policy.get('native_training_enabled') is False
         and policy.get('native_launch_authorized') is False
         and policy.get(STAGE) == {
             'status': 'ADMITTED', 'plan': book.ref(PLAN),
             'output_root': str(ROOT), 'max_invocations': 1,
             'claimed_invocations': 0, 'completed_invocations': 0,
             'execution_authorized': True, 'data_reads_authorized': True,
             'baseline_estimation_authorized': True,
             'model_forwards_authorized': 0, 'optimizer_steps_authorized': 0,
             'native_training_authorized': False,
             'test_authorized': False, 'broker_access_allowed': False,
             'relaunch_allowed': False}, 'POLICY')
    return book.ref(POLICY), head, meta, registry, bound, fbound, frozen, prior_result, fit, later_rows, later_issues


def measure(context: tuple) -> dict:
    policy_ref, head, meta, registry, bound, fbound, frozen, prior_result, fit, later_rows, later_issues = context
    gc, observed, _, field = book.extract(meta, registry)
    magnitude = np.abs(gc)
    fit_rows, fit_issues = fit['fit_parent_rows'], fit['fit_decision_ns']
    fit_observed = observed[fit_rows]
    need(bool(fit_observed.any()) and np.isfinite(magnitude[fit_rows[fit_observed]]).all(),
         'FIT_FIELD')
    mean = float(magnitude[fit_rows[fit_observed]].mean(dtype=np.float64))
    std = float(magnitude[fit_rows[fit_observed]].std(dtype=np.float64))
    need(np.isfinite(mean) and np.isfinite(std) and std > 0, 'FIT_SCALE')
    truth_fit = {name: owner.gather(fit_rows, fit_issues, fold_id=2,
                                    partition='fit')
                 for name, owner in (('state', bound), ('future', fbound))}
    truth_later = {name: owner.gather(later_rows, later_issues, fold_id=2,
                                      partition='score')
                   for name, owner in (('state', bound), ('future', fbound))}
    need(all(bool(v['mature'].all()) for v in truth_fit.values()),
         'FIT_MATURITY')
    p0 = transition._probabilities(frozen)
    slopes = {}
    fit_magnitude = magnitude[fit_rows]
    for name in TASKS:
        slopes[name] = {}
        truth = truth_fit[name]
        for j, tf in enumerate(TIMEFRAMES):
            bits = truth['allowed'][:, j]
            mask = fit_observed & truth['eligible'][:, j] & truth['mature'][:, j] & (bits != 7)
            need(bool(mask.any()), 'EMPTY_FIT:' + name + ':' + tf)
            z = (fit_magnitude[mask] - mean) / std
            active = bits[mask] != 2
            intercept = float(frozen['tasks'][name]['raw_logits'][j][0])
            slopes[name][tf] = fit_slope(z, active, intercept)
    prior_raw = {name: np.broadcast_to(
        np.asarray(frozen['tasks'][name]['raw_logits'], dtype=np.float64)[None],
        (len(later_rows), 5, 2)).copy() for name in TASKS}
    raw = {name: value.copy() for name, value in prior_raw.items()}
    later_observed = observed[later_rows]
    z_later = (magnitude[later_rows[later_observed]] - mean) / std
    for name in TASKS:
        for j, tf in enumerate(TIMEFRAMES):
            raw[name][later_observed, j, 0] += slopes[name][tf]['slope_nonnegative'] * z_later
        need(bool(np.isfinite(raw[name]).all()), 'PREDICTION_FINITE:' + name)
    overall = transition._scores(raw, prior_raw, truth_later, later_issues)
    transition._verify_prior(overall, prior_result['scores']['overall'])
    weekly = {}
    weeks = objective._weeks(later_issues)
    for week in np.unique(weeks):
        mask = weeks == week
        item = transition._scores(
            {name: value[mask] for name, value in raw.items()},
            {name: value[mask] for name, value in prior_raw.items()},
            transition._subset(truth_later, mask), later_issues[mask])
        transition._verify_prior(item, prior_result['scores']['by_utc_week'][str(week)])
        weekly[str(week)] = item
    need(len(weekly) == 14
         and all(overall['transition'][name][tf]['conditional_side']
                 == overall['prior'][name][tf]['conditional_side']
                 for name in TASKS for tf in TIMEFRAMES),
         'SIDE_UNCHANGED')
    return {
        'schema_version': 'gx1_gc_abs_book_activity_fold2_result_v1',
        'status': 'COMPLETE', 'stage': STAGE,
        'plan': book.ref(PLAN), 'policy': policy_ref,
        'source_head': head,
        'observed_utc': datetime.now(timezone.utc).isoformat(),
        'field': field, 'transform': 'absolute_value',
        'fit_standardization': {'mean': mean, 'std': std,
                                'observed_fit_rows': int(fit_observed.sum())},
        'slopes': slopes,
        'coverage': {'fit_observed_rows': int(fit_observed.sum()),
                     'later_observed_rows': int(later_observed.sum()),
                     'later_rows': len(later_rows)},
        'scores': {'overall': overall, 'by_utc_week': weekly},
        'prior_score_parity': True,
        'side_logits_unchanged': True,
        'model_forwards': 0, 'optimizer_steps': 0,
        'native_model_fit': False, 'baseline_estimation': True,
        'test_data_used': False, 'trade_selection': False,
    }


def run(plan_sha: str) -> None:
    from gx1.contracts.gx1_capped_execution_v1 import (
        require_capped_cpu_audit_execution, require_capped_lock_ancestry)
    context = inspect(plan_sha)
    cap = require_capped_cpu_audit_execution()
    lock = require_capped_lock_ancestry()
    need(cap['memory_max'] == cap['memory_high'] == 4 * 1024**3
         and cap['swap'] == 512 * 1024**2 and lock is not None, 'CAP')
    ROOT.mkdir(parents=False, exist_ok=False)
    _publish_bytes(ROOT / 'CLAIM.json', book.encode({
        'schema_version': 'gx1_gc_abs_book_activity_fold2_claim_v1',
        'stage': STAGE, 'plan': book.ref(PLAN),
        'policy': context[0], 'source_head': context[1],
        'test_data_used': False}))
    claim = book.ref(ROOT / 'CLAIM.json')
    started = time.monotonic()
    try:
        result = measure(context)
        result['claim'] = claim
        result['elapsed_seconds'] = time.monotonic() - started
        need(book.git('rev-parse', 'HEAD') == context[1]
             and book.git('status', '--porcelain') == ''
             and book.ref(POLICY) == context[0]
             and all(book.sha(p) == digest for p, digest in INPUTS.values()),
             'FINAL_SOURCE')
        _publish_bytes(ROOT / 'RESULT.json', book.encode(result))
        _publish_bytes(ROOT / 'TERMINAL.json', book.encode({
            'schema_version': 'gx1_gc_abs_book_activity_fold2_terminal_v1',
            'status': 'COMPLETE', 'exit_code': 0,
            'stage': STAGE, 'claim': claim,
            'result': book.ref(ROOT / 'RESULT.json'),
            'source_head': context[1], 'test_data_used': False}))
    except BaseException as exc:
        _publish_bytes(ROOT / 'TERMINAL.json', book.encode({
            'schema_version': 'gx1_gc_abs_book_activity_fold2_terminal_v1',
            'status': 'FAILED_CONSUMED', 'exit_code': 1,
            'stage': STAGE, 'claim': claim, 'result': None,
            'source_head': context[1],
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
        gc, observed, _, _ = book.extract(context[2], context[3])
        rows = context[8]['fit_parent_rows']
        x = np.abs(gc[rows[observed[rows]]])
        need(len(x) > 0 and np.isfinite(x).all()
             and np.std(x, dtype=np.float64) > 0,
             'REAL_FIT_SCALE_PREFLIGHT')
        toy = fit_slope(np.array([-2., -1., 0., 1., 2.]),
                        np.array([False, False, True, True, False]), 0.0)
        need(toy['slope_nonnegative'] > 0
             and toy['fit_log_mean'] < toy['prior_log_mean_on_observed_fit'],
             'SYNTHETIC_SUCCESS')
        try:
            fit_slope(np.array([np.nan]), np.array([True]), 0.0)
        except RuntimeError as exc:
            need(str(exc) == 'GC_ABS_BOOK_ACTIVITY_FOLD2_SLOPE_INPUT',
                 'SYNTHETIC_FAILURE')
        else:
            need(False, 'SYNTHETIC_FAILURE')
        print(json.dumps({'preflight': 'PASS_NO_CLAIM_NO_TARGET_SCORE',
                          'fit_rows': 47338, 'later_rows': 4096,
                          'source_head': context[1]}, sort_keys=True))
    else:
        run(args.plan_sha256)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
