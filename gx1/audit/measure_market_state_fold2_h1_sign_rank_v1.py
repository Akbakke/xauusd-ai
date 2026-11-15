"""One predeclared, read-only H1-current-sign rank check for Fold 2 S/F.

No model fit, sign flip, threshold search, TEST, Entry or trade. Current H1
state is obtained only from the observed M1 row at each M5 decision.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from gx1.audit import measure_market_state_fold2_exact_prior_later_v1 as prior
from gx1.audit import measure_market_state_fold2_causal_transition_v1 as transition
from gx1.contracts import market_state_five_endpoint_target_binding_v1 as state
from gx1.contracts import market_state_future_displacement_candidate_v1 as future
from gx1.contracts import market_state_time_query_native_F_objective_v1 as objective
from gx1.contracts.immutable_event_authority_v1 import _publish_bytes

REPO = prior.REPO
POLICY = prior.POLICY
STAGE = 'market_state_fold2_h1_sign_rank_20261115_001'
PLAN = REPO / 'configs/research/MARKET_STATE_FOLD2_H1_SIGN_RANK_20261115_001.json'
ROOT = Path('/home/andre2/GX1_RUNS/MARKET_STATE_FOLD2_H1_SIGN_RANK_20261115_001')
TIMEFRAMES = ('M5', 'M15', 'H1', 'H4', 'D1')
TASKS = ('state', 'future')
CODE = (
    'gx1/audit/measure_market_state_fold2_h1_sign_rank_v1.py',
    'gx1/audit/measure_market_state_fold2_causal_transition_v1.py',
    'gx1/audit/measure_market_state_fold2_exact_prior_later_v1.py',
    'gx1/audit/measure_market_state_causal_persistence_reference_v1.py',
    'gx1/contracts/market_state_five_endpoint_target_binding_v1.py',
    'gx1/contracts/market_state_future_displacement_candidate_v1.py',
    'gx1/contracts/market_state_time_query_native_F_objective_v1.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'scripts/gx1_capped_run.sh',
)


def need(ok: object, reason: str) -> None:
    if not bool(ok):
        raise RuntimeError('FOLD2_H1_SIGN_RANK_' + reason)


def rank_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Exact Mann-Whitney AUC for the declared {-1, 0, +1} sign score."""
    need(labels.ndim == scores.ndim == 1 and len(labels) == len(scores)
         and np.isin(labels, (0, 1)).all() and np.isin(scores, (-1, 0, 1)).all(),
         'AUC_INPUT')
    pos = int(np.count_nonzero(labels == 1))
    neg = len(labels) - pos
    if not pos or not neg:
        return None
    lower_neg = 0
    wins = 0.0
    for score in (-1, 0, 1):
        mask = scores == score
        p = int(np.count_nonzero(mask & (labels == 1)))
        n = int(np.count_nonzero(mask & (labels == 0)))
        wins += p * (lower_neg + 0.5 * n)
        lower_neg += n
    return wins / (pos * neg)


def sign(now: np.ndarray, column: int) -> np.ndarray:
    out = np.zeros(len(now), dtype=np.int8)
    out[now[:, column] == 1] = -1
    out[now[:, column] == 4] = 1
    return out


def cell(truth: dict, now: np.ndarray, j: int, subset: np.ndarray | None = None) -> dict:
    allowed = truth['allowed'][:, j]
    mask = truth['eligible'][:, j] & truth['mature'][:, j] & np.isin(allowed, (1, 4))
    if subset is not None:
        need(subset.shape == mask.shape, 'SUBSET')
        mask &= subset
    labels = (allowed[mask] == 4).astype(np.uint8)
    h1 = sign(now, 2)[mask]
    m5 = sign(now, 0)[mask]
    return {
        'known_future_side_rows': int(mask.sum()),
        'up_rows': int(labels.sum()),
        'down_rows': int(len(labels) - labels.sum()),
        'h1_current_direction_known_rows': int(np.count_nonzero(h1)),
        'm5_current_direction_known_rows': int(np.count_nonzero(m5)),
        'h1_sign_auc': rank_auc(labels, h1),
        'm5_sign_control_auc': rank_auc(labels, m5),
        'constant_rank_reference_auc': 0.5 if 0 < labels.sum() < len(labels) else None,
    }


def preflight(plan_sha: str) -> tuple:
    need(Path.cwd().resolve() == REPO
         and prior.git('branch', '--show-current') == 'work/gx1-current'
         and prior.git('status', '--porcelain') == '', 'CLEAN_SOURCE')
    head = prior.git('rev-parse', 'HEAD')
    need(not ROOT.exists() and prior.sha(PLAN) == plan_sha, 'PLAN_ROOT')
    plan = json.loads(PLAN.read_text())
    need(plan['schema_version'] == 'gx1_fold2_h1_sign_rank_plan_v1'
         and plan['stage'] == STAGE and plan['status'] == 'ADMITTED_READ_ONLY'
         and plan['source_base_head'] == '5e2dca7cf2c19a3762bc1e388d28b6a5cdf48e9b'
         and plan['output_root'] == str(ROOT)
         and plan['scope'] == {
             'fold_id': 2, 'fit_rows': 47338, 'later_rows': 4096,
             'tasks': list(TASKS), 'timeframes': list(TIMEFRAMES),
             'primary_cell': 'state_M5_known_side',
             'predictor': 'current_H1_bit4_plus1_bit1_minus1_else_zero',
             'control': 'current_M5_same_sign_mapping_same_rows',
             'target': 'future_allowed_bit4_up_vs_bit1_down_eligible_mature',
             'metric': 'tie_aware_rank_auc_no_fit',
             'weekly_primary': True, 'test_data_used': False}
         and plan['limits'] == {
             'class': 'audit', 'memory': '4G', 'swap': '512M',
             'max_invocations': 1, 'model_forwards': 0,
             'optimizer_steps': 0, 'native_training_authorized': False},
         'PLAN')
    need(subprocess.run(('git', 'merge-base', '--is-ancestor',
                         plan['source_base_head'], head), cwd=REPO).returncode == 0,
         'ANCESTOR')
    need(set(plan['source_owners']) == set(CODE)
         and all(prior.sha(REPO / path) == digest
                 for path, digest in plan['source_owners'].items()), 'SOURCE_HASH')
    need(all(Path(ref['path']).is_file() and prior.sha(Path(ref['path'])) == ref['sha256']
             for ref in plan['input_refs'].values()), 'INPUT_HASH')
    policy = json.loads(POLICY.read_text())
    need(policy['training_enabled'] is False
         and policy['native_training_enabled'] is False
         and policy[STAGE] == {
             'status': 'ADMITTED_READ_ONLY', 'plan': prior.ref(PLAN),
             'output_root': str(ROOT), 'max_invocations': 1,
             'claimed_invocations': 0, 'completed_invocations': 0,
             'data_reads_authorized': True, 'model_forwards_authorized': 0,
             'optimizer_steps_authorized': 0, 'training_authorized': False,
             'test_authorized': False, 'broker_access_allowed': False,
             'relaunch_allowed': False}, 'POLICY')
    return head, prior.ref(POLICY), plan


def measure(head: str, policy_ref: dict) -> dict:
    bound = state.load_bound_five_endpoint_targets()
    fbound = future.load_bound_future_displacement()
    need(np.array_equal(bound.parent_rows, fbound.parent_rows)
         and np.array_equal(bound.decision_ns, fbound.decision_ns), 'TARGET_AXIS')
    fit, later_rows, later_issues = prior.axes(json.loads(prior.COVERAGE.read_text()), bound)
    fit_rows, fit_issues = fit['fit_parent_rows'], fit['fit_decision_ns']
    need(len(fit_rows) == 47338 and len(later_rows) == 4096, 'COHORT_SIZE')
    now_fit, reasons_fit = transition._current(bound, fit_issues)
    now_later, reasons_later = transition._current(bound, later_issues)
    need(np.array_equal(now_fit == 0, reasons_fit != 0)
         and np.array_equal(now_later == 0, reasons_later != 0), 'CURRENT_COVERAGE')
    cohorts = {}
    for name, rows, issues, now, partition in (
        ('fit', fit_rows, fit_issues, now_fit, 'fit'),
        ('later', later_rows, later_issues, now_later, 'score')):
        truths = {task: owner.gather(rows, issues, fold_id=2, partition=partition)
                  for task, owner in (('state', bound), ('future', fbound))}
        if name == 'fit':
            need(all(bool(x['mature'].all()) for x in truths.values()), 'FIT_MATURITY')
        cohorts[name] = {task: {tf: cell(truths[task], now, j)
                                for j, tf in enumerate(TIMEFRAMES)}
                         for task in TASKS}
        if name == 'later':
            weeks = objective._weeks(issues)
            weekly = {str(week): cell(truths['state'], now, 0, weeks == week)
                      for week in np.unique(weeks)}
            need(len(weekly) == 14, 'WEEK_COUNT')
    return {
        'schema_version': 'gx1_fold2_h1_sign_rank_result_v1',
        'status': 'COMPLETE', 'stage': STAGE, 'source_head': head,
        'plan': prior.ref(PLAN), 'policy': policy_ref,
        'observed_utc': datetime.now(timezone.utc).isoformat(),
        'fit_rows': len(fit_rows), 'later_rows': len(later_rows),
        'scores': cohorts, 'later_primary_by_utc_week': weekly,
        'model_forwards': 0, 'optimizer_steps': 0,
        'model_fit': False, 'test_data_used': False,
        'trade_selection': False,
    }


def run(plan_sha: str) -> None:
    from gx1.contracts.gx1_capped_execution_v1 import (
        require_capped_cpu_audit_execution, require_capped_lock_ancestry)
    head, policy_ref, _ = preflight(plan_sha)
    cap = require_capped_cpu_audit_execution()
    lock = require_capped_lock_ancestry()
    need(cap['memory_max'] == cap['memory_high'] == 4 * 1024**3
         and cap['swap'] == 512 * 1024**2 and lock is not None, 'CAP')
    ROOT.mkdir(parents=False, exist_ok=False)
    _publish_bytes(ROOT / 'CLAIM.json', prior.encode({
        'schema_version': 'gx1_fold2_h1_sign_rank_claim_v1',
        'stage': STAGE, 'plan': prior.ref(PLAN),
        'policy': policy_ref, 'source_head': head, 'test_data_used': False}))
    claim = prior.ref(ROOT / 'CLAIM.json')
    started = time.monotonic()
    try:
        result = measure(head, policy_ref)
        result['claim'] = claim
        result['elapsed_seconds'] = time.monotonic() - started
        need(prior.git('rev-parse', 'HEAD') == head
             and prior.git('status', '--porcelain') == ''
             and prior.ref(POLICY) == policy_ref, 'FINAL_SOURCE')
        _publish_bytes(ROOT / 'RESULT.json', prior.encode(result))
        terminal = {
            'schema_version': 'gx1_fold2_h1_sign_rank_terminal_v1',
            'status': 'COMPLETE', 'exit_code': 0, 'stage': STAGE,
            'claim': claim, 'result': prior.ref(ROOT / 'RESULT.json'),
            'source_head': head, 'test_data_used': False}
        _publish_bytes(ROOT / 'TERMINAL.json', prior.encode(terminal))
    except BaseException as exc:
        terminal = {
            'schema_version': 'gx1_fold2_h1_sign_rank_terminal_v1',
            'status': 'FAILED_CONSUMED', 'exit_code': 1, 'stage': STAGE,
            'claim': claim, 'result': None, 'source_head': head,
            'reason': type(exc).__name__ + ':' + str(exc),
            'test_data_used': False}
        _publish_bytes(ROOT / 'TERMINAL.json', prior.encode(terminal))
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan-sha', required=True)
    args = parser.parse_args()
    run(args.plan_sha)


if __name__ == '__main__':
    main()
