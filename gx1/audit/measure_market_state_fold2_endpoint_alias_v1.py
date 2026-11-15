"""One read-only Fold 2 audit of local S states hidden between native endpoints.

This measures certified M1 ending states, not intraminute ordering, a new
target, model quality, Entry, Exit, or economic edge.
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
from gx1.audit import verify_market_state_m1_duration_v1 as duration
from gx1.audit import build_full_train_market_state_endpoint_candidate_v1 as base
from gx1.contracts import market_state_five_endpoint_target_binding_v1 as binding
from gx1.contracts.immutable_event_authority_v1 import _publish_bytes

REPO = prior.REPO
POLICY = prior.POLICY
STAGE = 'market_state_fold2_endpoint_alias_20261115_001'
PLAN = REPO / 'configs/research/MARKET_STATE_FOLD2_ENDPOINT_ALIAS_20261115_001.json'
ROOT = Path('/home/andre2/GX1_RUNS/MARKET_STATE_FOLD2_ENDPOINT_ALIAS_20261115_001')
TIMEFRAMES = ('M5', 'M15', 'H1', 'H4', 'D1')
LEADS = (5, 15, 60, 240, 1440)
CODE = (
    'gx1/audit/measure_market_state_fold2_endpoint_alias_v1.py',
    'gx1/audit/measure_market_state_fold2_exact_prior_later_v1.py',
    'gx1/audit/verify_market_state_m1_duration_v1.py',
    'gx1/audit/verify_market_state_endpoint_persistence_v1.py',
    'gx1/audit/build_full_train_market_state_endpoint_candidate_v1.py',
    'gx1/audit/verify_market_state_endpoint_candidate_v2.py',
    'gx1/contracts/entry_dynamic_mtf_state_targets_v1.py',
    'gx1/contracts/market_state_five_endpoint_target_binding_v1.py',
    'gx1/contracts/unified_exit_market_closure_authority_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'scripts/gx1_capped_run.sh',
)


def need(ok: object, code: str) -> None:
    if not bool(ok):
        raise RuntimeError('FOLD2_ENDPOINT_ALIAS_' + code)


def exact(axis: np.ndarray, wanted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    loc = np.searchsorted(axis, wanted)
    safe = np.minimum(loc, len(axis) - 1)
    return safe, (loc < len(axis)) & (axis[safe] == wanted)


def alias_cell(axis: np.ndarray, bits: np.ndarray, unknown_gap: np.ndarray,
               issues: np.ndarray, truth: dict, j: int, lead: int) -> dict:
    need(axis.ndim == issues.ndim == unknown_gap.ndim == 1
         and bits.shape == (len(axis), 5) and len(unknown_gap) == len(axis)
         and truth['allowed'].shape == truth['eligible'].shape
         == truth['mature'].shape == (len(issues), 5)
         and np.all(np.diff(axis) > 0) and lead > 0, 'SHAPE')
    cur, cur_exact = exact(axis, issues)
    end, end_exact = exact(axis, issues + lead * duration.MINUTE_NS)
    future = truth['allowed'][:, j]
    eligible = truth['eligible'][:, j] & truth['mature'][:, j]
    chosen = eligible & cur_exact & end_exact & np.isin(future, (1, 2, 4))
    need(np.array_equal(bits[end[chosen], j], future[chosen]), 'TARGET_PARITY')
    now = bits[cur, j]
    both = chosen & np.isin(now, (1, 2, 4))
    same = both & (now == future)
    c, e, state = cur[same], end[same], now[same]
    prefix = {v: np.r_[0, np.cumsum(bits[:, j] == v, dtype=np.int64)]
              for v in (1, 2, 4)}
    different = np.zeros(len(c), bool)
    opposite = np.zeros(len(c), bool)
    for v in (1, 2, 4):
        at = state == v
        if not np.any(at):
            continue
        other = sum(prefix[w][e[at]] - prefix[w][c[at] + 1]
                    for w in (1, 2, 4) if w != v)
        different[at] = other > 0
        if v in (1, 4):
            w = 4 if v == 1 else 1
            opposite[at] = (prefix[w][e[at]] - prefix[w][c[at] + 1]) > 0
    gap_prefix = np.r_[0, np.cumsum(unknown_gap, dtype=np.int64)]
    clean = (gap_prefix[e + 1] - gap_prefix[c + 1]) == 0
    return {
        'cohort_rows': len(issues),
        'eligible_mature_future_singleton': int((eligible & np.isin(future, (1, 2, 4))).sum()),
        'both_exact_current_future_singleton': int(both.sum()),
        'same_certified_current_and_endpoint': len(c),
        'same_with_certified_different_intermediate': int(different.sum()),
        'same_with_certified_opposite_side_intermediate': int(opposite.sum()),
        'same_without_unknown_source_gap': int(clean.sum()),
        'same_without_unknown_gap_with_different_intermediate': int((clean & different).sum()),
        'different_intermediate_fraction_of_same': (float(different.mean()) if len(c) else None),
        'unknown_or_non_singleton_intermediate_rows_not_treated_as_negative_proof': True,
    }


def preflight(plan_sha: str) -> tuple:
    need(Path.cwd().resolve() == REPO
         and prior.git('branch', '--show-current') == 'work/gx1-current'
         and prior.git('status', '--porcelain') == '', 'CLEAN_SOURCE')
    head = prior.git('rev-parse', 'HEAD')
    need(not ROOT.exists() and prior.sha(PLAN) == plan_sha, 'PLAN_ROOT')
    plan = json.loads(PLAN.read_text())
    need(plan['schema_version'] == 'gx1_fold2_endpoint_alias_plan_v1'
         and plan['stage'] == STAGE and plan['status'] == 'ADMITTED_READ_ONLY'
         and plan['source_base_head'] == 'b84993c128257af5b89b3344fccb7f2ff98e508d'
         and plan['output_root'] == str(ROOT)
         and plan['scope'] == {
             'fold_id': 2, 'fit_rows': 47338, 'later_rows': 4096,
             'timeframes': list(TIMEFRAMES), 'lead_minutes': list(LEADS),
             'primary_cell': 'state_M5_same_current_and_endpoint',
             'metric': 'fraction_with_certified_different_intermediate_M1_state',
             'control': 'same_current_endpoint_without_unknown_source_gap',
             'singletons': [1, 2, 4], 'no_model_fit': True, 'test_data_used': False}
         and plan['limits'] == {
             'class': 'audit', 'memory': '4G', 'swap': '512M',
             'max_invocations': 1, 'model_forwards': 0,
             'optimizer_steps': 0, 'target_published': False}, 'PLAN')
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


def measure(head: str, policy_ref: dict, plan: dict) -> dict:
    deadline = time.monotonic() + 1800
    old_plan = json.loads(Path(plan['input_refs']['m1_duration_plan']['path']).read_text())
    full_plan = duration.endpoint._checked_small_json(old_plan['input_refs']['target_source_plan'])
    m1, issues, m5_source_rows, stats, a05, a10, gap_kind, _ = duration._temporary_tapes(
        full_plan, deadline)
    bits, _, cold = duration._classify_all_m1(m1, a05, a10, gap_kind, deadline)
    selected = duration._selected_full_mask()
    parity = duration._selected_endpoint_parity(
        selected, issues, m5_source_rows, a05['validity_ns'], a05['source_row'], bits)
    need(all(v['bit_mismatches'] == v['m1_source_row_mismatches'] == 0
             for v in parity.values()), 'ENDPOINT_PARITY')
    bound = binding.load_bound_five_endpoint_targets()
    fit, later_rows, later_issues = prior.axes(json.loads(prior.COVERAGE.read_text()), bound)
    fit_rows, fit_issues = fit['fit_parent_rows'], fit['fit_decision_ns']
    need(len(fit_rows) == 47338 and len(later_rows) == 4096, 'COHORT_SIZE')
    scores = {}
    for name, rows, issue_ns, partition in (
        ('fit', fit_rows, fit_issues, 'fit'),
        ('later', later_rows, later_issues, 'score')):
        truth = bound.gather(rows, issue_ns, fold_id=2, partition=partition)
        scores[name] = {tf: alias_cell(a05['validity_ns'], bits,
                                      gap_kind == 2, issue_ns, truth, j, lead)
                        for j, (tf, lead) in enumerate(zip(TIMEFRAMES, LEADS))}
    need(all(base.regular(Path(path)) == recorded for path, recorded in stats.items()),
         'SOURCE_BOOKEND')
    return {
        'schema_version': 'gx1_fold2_endpoint_alias_result_v1',
        'status': 'COMPLETE', 'stage': STAGE, 'source_head': head,
        'plan': prior.ref(PLAN), 'policy': policy_ref,
        'observed_utc': datetime.now(timezone.utc).isoformat(),
        'scores': scores, 'selected_endpoint_parity': parity,
        'cold_side_report': cold, 'model_forwards': 0,
        'optimizer_steps': 0, 'model_fit': False, 'target_published': False,
        'test_data_used': False, 'trade_selection': False,
    }


def run(plan_sha: str) -> None:
    from gx1.contracts.gx1_capped_execution_v1 import (
        require_capped_cpu_audit_execution, require_capped_lock_ancestry)
    head, policy_ref, plan = preflight(plan_sha)
    cap = require_capped_cpu_audit_execution()
    lock = require_capped_lock_ancestry()
    need(cap['memory_max'] == cap['memory_high'] == 4 * 1024**3
         and cap['swap'] == 512 * 1024**2 and lock is not None, 'CAP')
    ROOT.mkdir(parents=False, exist_ok=False)
    _publish_bytes(ROOT / 'CLAIM.json', prior.encode({
        'schema_version': 'gx1_fold2_endpoint_alias_claim_v1',
        'stage': STAGE, 'plan': prior.ref(PLAN),
        'policy': policy_ref, 'source_head': head, 'test_data_used': False}))
    claim = prior.ref(ROOT / 'CLAIM.json')
    started = time.monotonic()
    try:
        result = measure(head, policy_ref, plan)
        result['claim'] = claim
        result['elapsed_seconds'] = time.monotonic() - started
        need(prior.git('rev-parse', 'HEAD') == head
             and prior.git('status', '--porcelain') == ''
             and prior.ref(POLICY) == policy_ref, 'FINAL_SOURCE')
        _publish_bytes(ROOT / 'RESULT.json', prior.encode(result))
        terminal = {
            'schema_version': 'gx1_fold2_endpoint_alias_terminal_v1',
            'status': 'COMPLETE', 'exit_code': 0, 'stage': STAGE,
            'claim': claim, 'result': prior.ref(ROOT / 'RESULT.json'),
            'source_head': head, 'test_data_used': False}
        _publish_bytes(ROOT / 'TERMINAL.json', prior.encode(terminal))
    except BaseException as exc:
        terminal = {
            'schema_version': 'gx1_fold2_endpoint_alias_terminal_v1',
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
