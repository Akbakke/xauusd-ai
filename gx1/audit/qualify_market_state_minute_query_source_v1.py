"""One-use read-only real-tape qualification of minute S query source semantics."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from gx1.audit import measure_market_state_fold2_exact_prior_later_v1 as prior
from gx1.audit import build_market_state_time_query_tape_v1 as producer
from gx1.contracts import market_state_five_endpoint_target_binding_v1 as binding
from gx1.contracts.market_state_minute_query_candidate_v1 import (
    MinuteQueryTargetTape, NATIVE_MINUTES,
)
from gx1.contracts.immutable_event_authority_v1 import _publish_bytes

REPO = prior.REPO
POLICY = prior.POLICY
STAGE = 'market_state_fold2_minute_query_source_20261115_001'
PLAN = REPO / 'configs/research/MARKET_STATE_FOLD2_MINUTE_QUERY_SOURCE_20261115_001.json'
ROOT = Path('/home/andre2/GX1_RUNS/MARKET_STATE_FOLD2_MINUTE_QUERY_SOURCE_20261115_001')
TIMEFRAMES = ('M5', 'M15', 'H1', 'H4', 'D1')
CODE = (
    'gx1/audit/qualify_market_state_minute_query_source_v1.py',
    'gx1/contracts/market_state_minute_query_candidate_v1.py',
    'gx1/contracts/market_state_time_query_target_v1.py',
    'gx1/contracts/market_state_time_query_model_v2.py',
    'gx1/contracts/market_state_time_query_loss_v2.py',
    'gx1/contracts/market_state_time_query_scorer_v2.py',
    'gx1/contracts/market_state_five_endpoint_contract_v1.py',
    'gx1/contracts/market_state_five_endpoint_target_binding_v1.py',
    'gx1/audit/measure_market_state_fold2_exact_prior_later_v1.py',
    'gx1/audit/build_market_state_time_query_tape_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'scripts/gx1_capped_run.sh',
)


def need(ok: object, code: str) -> None:
    if not bool(ok):
        raise RuntimeError('FOLD2_MINUTE_QUERY_' + code)


def preflight(plan_sha: str) -> tuple:
    need(Path.cwd().resolve() == REPO
         and prior.git('branch', '--show-current') == 'work/gx1-current'
         and prior.git('status', '--porcelain') == '', 'CLEAN_SOURCE')
    head = prior.git('rev-parse', 'HEAD')
    need(not ROOT.exists() and prior.sha(PLAN) == plan_sha, 'PLAN_ROOT')
    plan = json.loads(PLAN.read_text())
    need(plan['schema_version'] == 'gx1_fold2_minute_query_source_plan_v1'
         and plan['stage'] == STAGE and plan['status'] == 'ADMITTED_READ_ONLY'
         and plan['source_base_head'] == '6861ca95d4e655855415875dc131a12a7a6b9c82'
         and plan['output_root'] == str(ROOT)
         and plan['scope'] == {
             'fold_id': 2, 'fit_rows': 47338, 'later_rows': 4096,
             'timeframes': list(TIMEFRAMES),
             'primary': 'M5_minute_1_2_3_4_exact_source_coverage_and_bits',
             'control': 'native_5_15_60_240_1440_minute_endpoint_parity',
             'minute_1_all_timeframes': True,
             'model_fit': False, 'test_data_used': False}
         and plan['limits'] == {
             'class': 'audit', 'memory': '4G', 'swap': '512M',
             'max_invocations': 1, 'model_forwards': 0,
             'optimizer_steps': 0, 'target_builds': 0}, 'PLAN')
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


def _counts(truth: dict) -> dict:
    return {tf: {
        'mature': int(truth['mature'][:, j].sum()),
        'eligible': int(truth['eligible'][:, j].sum()),
        'bits': {str(bit): int((truth['eligible'][:, j]
                                & (truth['allowed'][:, j] == bit)).sum())
                 for bit in (1, 2, 4, 5, 7)},
        'path_observed': int((truth['eligible'][:, j]
                              & truth['path_observed'][:, j]).sum()),
        'unknown_source_gap_at_query': int((truth['source_gap_kind'][:, j] == 2).sum()),
    } for j, tf in enumerate(TIMEFRAMES)}


def measure(head: str, policy_ref: dict, plan: dict) -> dict:
    tape_ref = plan['input_refs']['candidate_tape']
    raw = Path(tape_ref['path']).read_bytes()
    need(prior.sha(Path(tape_ref['path'])) == tape_ref['sha256'], 'TAPE_HASH')
    with np.load(io.BytesIO(raw), allow_pickle=False) as saved:
        need(set(saved.files) == producer.TAPE_KEYS, 'TAPE_KEYS')
        tape = MinuteQueryTargetTape(
            np.array(saved['validity_ns'], copy=True),
            np.array(saved['m1_source_row'], copy=True),
            np.array(saved['allowed_set'], copy=True),
            np.array(saved['path_observed'], copy=True),
            np.array(saved['gap_kind'], copy=True))
    bound = binding.load_bound_five_endpoint_targets()
    fit, later_rows, later_issues = prior.axes(
        json.loads(prior.COVERAGE.read_text()), bound)
    fit_rows, fit_issues = fit['fit_parent_rows'], fit['fit_decision_ns']
    need(len(fit_rows) == 47338 and len(later_rows) == 4096, 'COHORT_SIZE')
    result = {}
    native = np.asarray(NATIVE_MINUTES, np.int64)
    for name, rows, issues, partition, upper in (
        ('fit', fit_rows, fit_issues, 'fit', binding.FOLD_BOUNDARIES_NS[1]),
        ('later', later_rows, later_issues, 'score', binding.FOLD_BOUNDARIES_NS[2])):
        bound_truth = bound.gather(rows, issues, fold_id=2, partition=partition)
        endpoint = tape.gather_minutes(
            issues, np.broadcast_to(native, (len(issues), 5)).copy(),
            fold_upper_ns=int(upper))
        comparable = bound_truth['mature']
        mismatches = {
            'allowed_on_mature': int(np.count_nonzero(
                (endpoint['allowed'] != bound_truth['allowed']) & comparable)),
            'mature': int(np.count_nonzero(
                endpoint['mature'] != bound_truth['mature'])),
            'effective_eligible': int(np.count_nonzero(
                endpoint['eligible'] != (bound_truth['eligible']
                                         & bound_truth['mature']))),
        }
        early = {}
        for minute in (1, 2, 3, 4):
            query = np.broadcast_to(native, (len(issues), 5)).copy()
            query[:, 0] = minute
            got = tape.gather_minutes(issues, query, fold_upper_ns=int(upper))
            early[str(minute)] = _counts(got)['M5']
        one = tape.gather_minutes(
            issues, np.ones((len(issues), 5), np.int64),
            fold_upper_ns=int(upper))
        result[name] = {'endpoint_mismatches': mismatches,
                        'native_endpoint': _counts(endpoint),
                        'minute_1_all_timeframes': _counts(one),
                        'M5_minutes_1_to_4': early}
    need(prior.sha(Path(tape_ref['path'])) == tape_ref['sha256']
         and prior.ref(POLICY) == policy_ref
         and prior.git('rev-parse', 'HEAD') == head
         and prior.git('status', '--porcelain') == '', 'BOOKEND')
    return {
        'schema_version': 'gx1_fold2_minute_query_source_result_v1',
        'status': 'COMPLETE', 'stage': STAGE, 'source_head': head,
        'plan': prior.ref(PLAN), 'policy': policy_ref,
        'observed_utc': datetime.now(timezone.utc).isoformat(),
        'scores': result, 'model_forwards': 0, 'optimizer_steps': 0,
        'model_fit': False, 'target_builds': 0,
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
        'schema_version': 'gx1_fold2_minute_query_source_claim_v1',
        'stage': STAGE, 'plan': prior.ref(PLAN),
        'policy': policy_ref, 'source_head': head, 'test_data_used': False}))
    claim = prior.ref(ROOT / 'CLAIM.json')
    started = time.monotonic()
    try:
        result = measure(head, policy_ref, plan)
        result['claim'] = claim
        result['elapsed_seconds'] = time.monotonic() - started
        _publish_bytes(ROOT / 'RESULT.json', prior.encode(result))
        terminal = {
            'schema_version': 'gx1_fold2_minute_query_source_terminal_v1',
            'status': 'COMPLETE', 'exit_code': 0, 'stage': STAGE,
            'claim': claim, 'result': prior.ref(ROOT / 'RESULT.json'),
            'source_head': head, 'test_data_used': False}
        _publish_bytes(ROOT / 'TERMINAL.json', prior.encode(terminal))
    except BaseException as exc:
        terminal = {
            'schema_version': 'gx1_fold2_minute_query_source_terminal_v1',
            'status': 'FAILED_CONSUMED', 'exit_code': 1, 'stage': STAGE,
            'claim': claim, 'result': None, 'source_head': head,
            'reason': type(exc).__name__ + ':' + str(exc),
            'test_data_used': False}
        _publish_bytes(ROOT / 'TERMINAL.json', prior.encode(terminal))
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan-sha', required=True)
    run(parser.parse_args().plan_sha)


if __name__ == '__main__':
    main()
