"""One-use exact Fold2 S/F TRAIN priors and fixed later development scores.

Only the prior is estimated. No model, feature gather, TEST, broker or trade.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True

import numpy as np

from gx1.contracts import market_state_five_endpoint_target_binding_v1 as state
from gx1.contracts import market_state_future_displacement_candidate_v1 as future
from gx1.contracts import market_state_time_query_native_F_objective_v1 as objective
from gx1.contracts.immutable_event_authority_v1 import _publish_bytes

REPO = Path('/home/andre2/src/GX1_CURRENT')
STAGE = 'market_state_fold2_exact_prior_later_4096_20261115_001'
PLAN = REPO / 'configs/research/MARKET_STATE_FOLD2_EXACT_PRIOR_LATER_4096_20261115_001.json'
POLICY = REPO / 'NEXT_RUN_POLICY.json'
ROOT = Path('/home/andre2/GX1_RUNS/MARKET_STATE_FOLD2_EXACT_PRIOR_LATER_4096_20261115_001')
COVERAGE = REPO / 'configs/research/MARKET_STATE_FULL_GC_FOLD2_LATER_COVERAGE_20261115_001.json'
LATER_ROOT = Path('/home/andre2/GX1_RUNS/MARKET_STATE_FULL_GC_FOLD2_LATER_INPUTS_20261115_001')
OLD_ROOT = Path('/home/andre2/GX1_RUNS/MARKET_STATE_FIVE_ENDPOINT_PRIOR_20261107_001')
FOLD = 2
FIT_ROWS = 47338
SCORE_ROWS = 4096
PARENT_SHA = 'db43609ad6bd10e8d1a13b216243855c499189f8d2f08ef0a67d49de66e29fe9'
ISSUE_SHA = '380184a9c95a02f2716326765783746b1834371a7266545113fb1eae6722da47'
INPUTS = {
    'coverage': (COVERAGE, 'ff18b8fcde8b09c41cbec5c216c532ef688a6be29767e456475cef941495c640'),
    'later_result': (LATER_ROOT / 'RESULT.json', '820c41c10faddefc9ad38622168eb2080e6913ee68ebba6f0e0dff8fa0edf37b'),
    'later_terminal': (LATER_ROOT / 'TERMINAL.json', '47267210715f01d6f4dc03e30b207344a7ccde23b87e0867e73b4330665ff950'),
    'old_prior_result': (OLD_ROOT / 'RESULT.json', '5304433c42b43ea2bdf593066e627c8904604a751aa27c6fa5dab6349efefa48'),
    'old_prior_terminal': (OLD_ROOT / 'TERMINAL.json', '8f9804b384baf0c1b3d5d321465f00424da17956c06ef39929f107767bc4e9a4'),
    'state_mask': (Path(state.REFS['mask'][0]), state.REFS['mask'][1]),
    'future_tape': future.REFS['tape'],
}
CODE = (
    'gx1/audit/measure_market_state_fold2_exact_prior_later_v1.py',
    'gx1/contracts/market_state_five_endpoint_target_binding_v1.py',
    'gx1/contracts/market_state_future_displacement_candidate_v1.py',
    'gx1/contracts/market_state_five_endpoint_prior_v1.py',
    'gx1/contracts/market_state_time_query_native_F_objective_v1.py',
    'gx1/contracts/market_state_time_query_scorer_v2.py',
    'gx1/contracts/market_state_time_query_target_v1.py',
    'gx1/contracts/market_state_time_query_side_normalized_loss_v1.py',
    'gx1/contracts/market_state_time_query_native_F_model_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'scripts/gx1_capped_run.sh',
)


def need(ok: object, code: str) -> None:
    if not bool(ok):
        raise RuntimeError('FOLD2_EXACT_PRIOR_' + code)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def ref(path: Path) -> dict:
    return {'path': str(path.resolve()), 'sha256': sha(path)}


def encode(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def git(*args: str) -> str:
    return subprocess.check_output(('git', *args), cwd=REPO, text=True).strip()


def axes(coverage: dict, bound: state.BoundFiveEndpointTargets) -> tuple[dict, np.ndarray, np.ndarray]:
    fit = bound.mature_oof_axes(fold_id=FOLD)
    need(fit['joined_fit_rows'] == FIT_ROWS
         and fit['fit_parent_rows_sha256'] == PARENT_SHA
         and fit['fit_decision_ns_sha256'] == ISSUE_SHA, 'FIT_AXIS')
    need(coverage['score_parent_start'] == fit['old_joined_fit_rows']
         and coverage['score_rows'] == len(fit['score_parent_rows'])
         and coverage['score_rows'] == 17288
         and coverage['score_start_ns'] == state.FOLD_BOUNDARIES_NS[1]
         and coverage['score_end_exclusive_ns'] == state.FOLD_BOUNDARIES_NS[2]
         and coverage['batch_size'] == 16
         and len(coverage['batch_indices']) == SCORE_ROWS // 16,
         'COVERAGE_AXIS')
    at = (coverage['score_parent_start']
          + np.asarray(coverage['batch_indices'], dtype=np.int64)[:, None] * 16
          + np.arange(16, dtype=np.int64)).reshape(-1)
    need(at.shape == (SCORE_ROWS,)
         and bool(np.all(np.diff(at) > 0))
         and int(at[-1]) < coverage['score_parent_start'] + coverage['score_rows'],
         'SELECTED_INDEX')
    rows, issues = bound.parent_rows[at].copy(), bound.decision_ns[at].copy()
    def axis_sha(value: np.ndarray) -> str:
        return hashlib.sha256(np.ascontiguousarray(value, dtype='<i8').tobytes()).hexdigest()
    need(axis_sha(rows) == coverage['selected_parent_rows_sha256']
         and axis_sha(issues) == coverage['selected_decision_ns_sha256']
         and bool(np.all((issues >= coverage['score_start_ns'])
                         & (issues < coverage['score_end_exclusive_ns']))),
         'SELECTED_HASH')
    return fit, rows, issues


def inspect(plan_sha: str) -> tuple[dict, dict, str, dict, object, object, dict, np.ndarray, np.ndarray]:
    need(Path.cwd().resolve() == REPO
         and git('branch', '--show-current') == 'work/gx1-current'
         and git('status', '--porcelain') == '', 'CLEAN_SOURCE')
    head = git('rev-parse', 'HEAD')
    need(not ROOT.exists() and sha(PLAN) == plan_sha, 'PLAN_OUTPUT')
    plan = json.loads(PLAN.read_text())
    need(set(plan) == {'schema_version', 'stage', 'status', 'source_base_head',
                       'input_refs', 'source_owners', 'scope', 'limits', 'output_root'}
         and plan['schema_version'] == 'gx1_fold2_exact_prior_later_plan_v1'
         and plan['stage'] == STAGE and plan['status'] == 'PREPARED_NOT_ADMITTED'
         and plan['scope'] == {'fold_id': FOLD, 'fit_rows': FIT_ROWS,
                               'fit_parent_rows_sha256': PARENT_SHA,
                               'fit_decision_ns_sha256': ISSUE_SHA,
                               'later_selected_rows': SCORE_ROWS,
                               'timeframes': ['M5', 'M15', 'H1', 'H4', 'D1'],
                               'test_data_used': False}
         and plan['limits'] == {'class': 'audit', 'memory': '4G', 'swap': '512M',
                                'wall_seconds': 600, 'max_invocations': 1,
                                'model_forwards': 0, 'optimizer_steps': 0,
                                'training_authorized': False}
         and plan['output_root'] == str(ROOT), 'PLAN')
    need(subprocess.run(('git', 'merge-base', '--is-ancestor',
                         plan['source_base_head'], head), cwd=REPO).returncode == 0,
         'ANCESTOR')
    need(set(plan['source_owners']) == set(CODE)
         and all(sha(REPO / rel) == digest
                 for rel, digest in plan['source_owners'].items()), 'CODE_HASH')
    expected = {name: {'path': str(path), 'sha256': digest}
                for name, (path, digest) in INPUTS.items()}
    need(plan['input_refs'] == expected
         and all(sha(path) == digest for path, digest in INPUTS.values()),
         'INPUT_HASH')
    later = json.loads(INPUTS['later_result'][0].read_text())
    terminal = json.loads(INPUTS['later_terminal'][0].read_text())
    old = json.loads(INPUTS['old_prior_result'][0].read_text())
    old_terminal = json.loads(INPUTS['old_prior_terminal'][0].read_text())
    coverage = json.loads(COVERAGE.read_text())
    need(later['status'] == 'COMPLETE'
         and later['selected_parent_rows'] == SCORE_ROWS
         and later['target_arrays_decoded'] == 0
         and later['model_forwards'] == later['optimizer_steps'] == 0
         and later['test_data_used'] is False
         and coverage['view_sha256'] == later['view_sha256']
         and terminal['status'] == 'COMPLETE' and terminal['exit_code'] == 0
         and terminal['result'] == expected['later_result']
         and old['status'] == 'COMPLETE'
         and old['folds'][1]['fit_rows'] == 47602
         and old_terminal['status'] == 'COMPLETE' and old_terminal['exit_code'] == 0
         and old_terminal['result'] == expected['old_prior_result']
         and coverage['status'] == 'COMPLETE'
         and coverage['qualification'] == expected['later_result']
         and coverage['selected_parent_rows_sha256'] == later['parent_rows_sha256']
         and coverage['selected_decision_ns_sha256'] == later['decision_ns_sha256'],
         'RECEIPT_CHAIN')
    state_bound = state.load_bound_five_endpoint_targets()
    future_bound = future.load_bound_future_displacement()
    need(np.array_equal(state_bound.parent_rows, future_bound.parent_rows)
         and np.array_equal(state_bound.decision_ns, future_bound.decision_ns),
         'TARGET_JOIN')
    fit, rows, issues = axes(coverage, state_bound)
    policy_ref = ref(POLICY)
    policy = json.loads(POLICY.read_text())
    gate = policy.get(STAGE)
    need(policy.get('training_enabled') is False
         and policy.get('native_training_enabled') is False
         and policy.get('native_launch_authorized') is False
         and isinstance(gate, dict)
         and gate == {'status': 'ADMITTED', 'plan': ref(PLAN),
                      'output_root': str(ROOT), 'max_invocations': 1,
                      'claimed_invocations': 0, 'completed_invocations': 0,
                      'execution_authorized': True, 'data_reads_authorized': True,
                      'baseline_estimation_authorized': True,
                      'model_forwards_authorized': 0, 'optimizer_steps_authorized': 0,
                      'training_authorized': False, 'test_authorized': False,
                      'broker_access_allowed': False, 'relaunch_allowed': False},
         'POLICY')
    return plan, policy_ref, head, coverage, state_bound, future_bound, fit, rows, issues


def score(frozen: dict, state_truth: dict, future_truth: dict,
          issues: np.ndarray) -> dict:
    raw = {name: np.asarray(frozen['tasks'][name]['raw_logits'], dtype=np.float64)
           for name in ('state', 'future')}
    query = np.broadcast_to(objective.NATIVE_QUERY, (len(issues), 5)).copy()
    def one(mask: np.ndarray) -> dict:
        need(mask.dtype == np.bool_ and mask.shape == (len(issues),)
             and bool(mask.any()), 'SCORE_MASK')
        out = {}
        for name, truth in (('state', state_truth), ('future', future_truth)):
            selected = {key: value[mask] for key, value in truth.items()
                        if isinstance(value, np.ndarray) and value.shape[0] == len(issues)}
            repeated = np.broadcast_to(raw[name][None, :, :],
                                       (int(mask.sum()), 5, 2))
            out[name] = objective._endpoint_task_sums(repeated, selected, query[mask])
        return out
    all_mask = np.ones(len(issues), dtype=np.bool_)
    overall = one(all_mask)
    weeks = objective._weeks(issues)
    weekly = {str(week): one(weeks == week) for week in np.unique(weeks)}
    for name in ('state', 'future'):
        for tf in objective.TIMEFRAMES:
            for metric in objective.METRICS:
                full = overall[name][tf][metric]
                for key in ('count', 'brier_sum', 'log_sum'):
                    if key in full:
                        total = sum(week[name][tf][metric][key]
                                    for week in weekly.values())
                        need(np.isclose(total, full[key], atol=1e-7, rtol=0),
                             'WEEK_REDUCTION')
    return {'overall': overall, 'by_utc_week': weekly,
            'week_count': len(weekly)}


def run(plan_sha: str) -> None:
    from gx1.contracts.gx1_capped_execution_v1 import (
        require_capped_cpu_audit_execution, require_capped_lock_ancestry)
    plan, policy_ref, head, coverage, state_bound, future_bound, fit, rows, issues = inspect(plan_sha)
    cap = require_capped_cpu_audit_execution()
    lock = require_capped_lock_ancestry()
    need(cap['memory_max'] == cap['memory_high'] == 4 * 1024**3
         and cap['swap'] == 512 * 1024**2 and lock is not None,
         'CAP')
    ROOT.mkdir(parents=False, exist_ok=False)
    claim = {'schema_version': 'gx1_fold2_exact_prior_later_claim_v1',
             'stage': STAGE, 'plan': ref(PLAN), 'policy': policy_ref,
             'source_head': head, 'test_data_used': False}
    _publish_bytes(ROOT / 'CLAIM.json', encode(claim))
    began = time.monotonic()
    try:
        frozen = objective.fit_endpoint_priors(
            state_bound, future_bound, axes=fit, fold_id=FOLD)
        need(objective.require_endpoint_priors(frozen, fold_id=FOLD) == frozen
             and frozen['fit_rows'] == FIT_ROWS
             and frozen['fit_parent_rows_sha256'] == PARENT_SHA
             and frozen['fit_decision_ns_sha256'] == ISSUE_SHA,
             'FROZEN_PRIOR')
        _publish_bytes(ROOT / 'FROZEN_PRIORS.json', encode(frozen))
        frozen_ref = ref(ROOT / 'FROZEN_PRIORS.json')
        s_truth = state_bound.gather(rows, issues, fold_id=FOLD, partition='score')
        f_truth = future_bound.gather(rows, issues, fold_id=FOLD, partition='score')
        need(np.array_equal(s_truth['issue_ns'], f_truth['issue_ns'])
             and np.array_equal(s_truth['parent_rows'], f_truth['parent_rows']),
             'SCORE_JOIN')
        result = {
            'schema_version': 'gx1_fold2_exact_prior_later_result_v1',
            'status': 'COMPLETE', 'stage': STAGE,
            'plan': ref(PLAN), 'policy': policy_ref,
            'claim': ref(ROOT / 'CLAIM.json'), 'frozen_priors': frozen_ref,
            'source_head': head, 'fit_rows': FIT_ROWS,
            'old_reference_fit_rows': 47602,
            'selected_score_rows': SCORE_ROWS,
            'selected_parent_rows_sha256': coverage['selected_parent_rows_sha256'],
            'selected_decision_ns_sha256': coverage['selected_decision_ns_sha256'],
            'scores': score(frozen, s_truth, f_truth, issues),
            'model_forwards': 0, 'optimizer_steps': 0,
            'test_data_used': False, 'elapsed_seconds': time.monotonic() - began}
        need(git('rev-parse', 'HEAD') == head
             and git('status', '--porcelain') == ''
             and ref(POLICY) == policy_ref
             and all(sha(path) == digest for path, digest in INPUTS.values()),
             'FINAL_SOURCE')
        _publish_bytes(ROOT / 'RESULT.json', encode(result))
        terminal = {'schema_version': 'gx1_fold2_exact_prior_later_terminal_v1',
                    'status': 'COMPLETE', 'exit_code': 0, 'stage': STAGE,
                    'claim': ref(ROOT / 'CLAIM.json'),
                    'frozen_priors': frozen_ref,
                    'result': ref(ROOT / 'RESULT.json'),
                    'source_head': head, 'test_data_used': False}
        _publish_bytes(ROOT / 'TERMINAL.json', encode(terminal))
    except BaseException as exc:
        terminal = {'schema_version': 'gx1_fold2_exact_prior_later_terminal_v1',
                    'status': 'FAILED_CONSUMED', 'exit_code': 1, 'stage': STAGE,
                    'claim': ref(ROOT / 'CLAIM.json'),
                    'result': None, 'source_head': head,
                    'reason': type(exc).__name__ + ':' + str(exc),
                    'test_data_used': False}
        _publish_bytes(ROOT / 'TERMINAL.json', encode(terminal))
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    need(args.preflight != args.run, 'MODE')
    if args.preflight:
        x = inspect(args.plan_sha256)
        need(x[6]['joined_fit_rows'] == FIT_ROWS and len(x[7]) == SCORE_ROWS,
             'PREFLIGHT_AXES')
        need(json.loads(encode({'ok': 1})) == {'ok': 1},
             'SYNTHETIC_REPORT_SUCCESS')
        toy_issues = np.array([1735689600000000000 + i * 300000000000
                               for i in range(4)], dtype=np.int64)
        toy_truth = {'allowed': np.tile(np.array([[1, 2, 4, 5, 7]],
                                                dtype=np.uint8), (4, 1)),
                     'eligible': np.ones((4, 5), dtype=np.bool_),
                     'mature': np.ones((4, 5), dtype=np.bool_)}
        toy_prior = {'tasks': {name: {'raw_logits': [[0.0, 0.0]] * 5}
                               for name in ('state', 'future')}}
        toy = score(toy_prior, toy_truth, toy_truth, toy_issues)
        need(toy['week_count'] == 1
             and toy['overall']['state']['M5']['conditional_side']['count'] == 4
             and toy['overall']['future']['M15']['activity']['count'] == 4,
             'SYNTHETIC_SCORE_SUCCESS')
        try:
            encode({'bad': float('nan')})
        except ValueError:
            pass
        else:
            need(False, 'SYNTHETIC_REPORT_FAILURE')
        try:
            score({}, {}, {}, np.array([], dtype=np.int64))
        except (KeyError, ValueError, RuntimeError):
            pass
        else:
            need(False, 'SYNTHETIC_FAILURE')
        print(json.dumps({'preflight': 'PASS_NO_CLAIM_NO_PRIOR_FIT',
                          'source_head': x[2], 'fit_rows': FIT_ROWS,
                          'later_rows': SCORE_ROWS}, sort_keys=True))
    else:
        run(args.plan_sha256)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
