"""One-use read-only Fold 2 audit of a fixed multiminute S query path."""
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
from gx1.audit import measure_market_state_fold2_causal_transition_v1 as transition
from gx1.audit import measure_market_state_fold2_h1_sign_rank_v1 as sign_owner
from gx1.audit import measure_market_state_fold2_paired_minute_path_v1 as paired
from gx1.audit import build_market_state_time_query_tape_v1 as tape_owner
from gx1.contracts import market_state_five_endpoint_target_binding_v1 as state
from gx1.contracts import market_state_time_query_native_F_objective_v1 as objective
from gx1.contracts.market_state_fold2_multiminute_path_v1 import (
    QUERY_MINUTES, QUERY_SHA256, gather_fixed_path,
    observed_transition_intervals, same_tf_persistence_reference,
)
from gx1.contracts.market_state_minute_query_candidate_v1 import MinuteQueryTargetTape
from gx1.contracts.immutable_event_authority_v1 import _publish_bytes

REPO = prior.REPO
POLICY = prior.POLICY
STAGE = 'market_state_fold2_multiminute_path_20261115_001'
PLAN = REPO / 'configs/research/MARKET_STATE_FOLD2_MULTIMINUTE_PATH_20261115_001.json'
ROOT = Path('/home/andre2/GX1_RUNS/MARKET_STATE_FOLD2_MULTIMINUTE_PATH_20261115_001')
TIMEFRAMES = ('M5', 'M15', 'H1', 'H4', 'D1')
CODE = (
    'gx1/audit/measure_market_state_fold2_multiminute_path_v1.py',
    'gx1/audit/measure_market_state_fold2_exact_prior_later_v1.py',
    'gx1/audit/measure_market_state_fold2_causal_transition_v1.py',
    'gx1/audit/measure_market_state_fold2_h1_sign_rank_v1.py',
    'gx1/audit/measure_market_state_fold2_paired_minute_path_v1.py',
    'gx1/audit/measure_market_state_causal_persistence_reference_v1.py',
    'gx1/audit/build_market_state_time_query_tape_v1.py',
    'gx1/contracts/market_state_fold2_multiminute_path_v1.py',
    'gx1/contracts/market_state_minute_query_candidate_v1.py',
    'gx1/contracts/market_state_five_endpoint_target_binding_v1.py',
    'gx1/contracts/market_state_time_query_native_F_objective_v1.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'scripts/gx1_capped_run.sh',
)
LIMITS = {'class': 'audit', 'memory': '4G', 'swap': '512M',
          'wall_seconds': 900, 'max_invocations': 1}
SCOPE = {
    'fold_id': 2, 'fit_rows': 47338, 'later_rows': 4096,
    'timeframes': list(TIMEFRAMES),
    'query_minutes': QUERY_MINUTES.tolist(), 'query_sha256': QUERY_SHA256,
    'transition_rule': 'different_adjacent_certified_singletons_prove_change_within_interval',
    'reference': 'exact_causal_same_TF_current_sign_on_same_rows',
    'primary': 'M5_intermediate_state_changes_and_same_TF_persistence_by_checkpoint',
    'weekly_primary': True, 'model_forwards': 0,
    'optimizer_steps': 0, 'test_data_used': False,
}


def need(ok: object, code: str) -> None:
    if not bool(ok):
        raise RuntimeError('FOLD2_MULTIMINUTE_PATH_' + code)


def preflight(plan_sha: str) -> tuple:
    need(Path.cwd().resolve() == REPO
         and prior.git('branch', '--show-current') == 'work/gx1-current'
         and prior.git('status', '--porcelain') == '', 'CLEAN_SOURCE')
    head = prior.git('rev-parse', 'HEAD')
    need(not ROOT.exists() and prior.sha(PLAN) == plan_sha, 'PLAN_ROOT')
    plan = json.loads(PLAN.read_text())
    need(plan.get('schema_version') == 'gx1_fold2_multiminute_path_plan_v1'
         and plan.get('stage') == STAGE and plan.get('status') == 'ADMITTED_READ_ONLY'
         and plan.get('source_base_head') == 'b04a7c860e9f481c2c88885e14fb4759ce34ab6b'
         and plan.get('output_root') == str(ROOT)
         and plan.get('scope') == SCOPE and plan.get('limits') == LIMITS,
         'PLAN')
    need(subprocess.run(('git', 'merge-base', '--is-ancestor',
                         plan['source_base_head'], head), cwd=REPO).returncode == 0,
         'ANCESTOR')
    need(set(plan.get('source_owners', {})) == set(CODE)
         and all(prior.sha(REPO / path) == digest
                 for path, digest in plan['source_owners'].items()), 'SOURCE_HASH')
    need(all(Path(ref['path']).is_file()
             and prior.sha(Path(ref['path'])) == ref['sha256']
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


def _coverage(path: dict) -> dict:
    result = {}
    for j, tf in enumerate(TIMEFRAMES):
        result[tf] = []
        for q in range(len(QUERY_MINUTES)):
            bits = path['allowed'][:, q, j]
            result[tf].append({
                'minute': int(QUERY_MINUTES[q, j]),
                'rows': len(bits),
                'mature': int(path['mature'][:, q, j].sum()),
                'eligible': int(path['eligible'][:, q, j].sum()),
                'allowed_counts': {str(bit): int(np.count_nonzero(bits == bit))
                                   for bit in (1, 2, 4, 5, 7)},
            })
    return result


def _slice(path: dict, subset: np.ndarray) -> dict:
    return {key: value[subset] if isinstance(value, np.ndarray) else value
            for key, value in path.items()}


def _same_rows_known_side(path: dict, now: np.ndarray) -> dict:
    """Keep all five AUCs on identical certified side rows within each TF."""
    out = {}
    for j, tf in enumerate(TIMEFRAMES):
        bits = path['allowed'][:, :, j]
        certified = (path['eligible'][:, :, j] & path['mature'][:, :, j]
                     & np.isin(bits, (1, 4)))
        same = np.all(certified, axis=1)
        labels = bits[same] == 4
        score = sign_owner.sign(now, j)[same]
        out[tf] = {
            'all_five_known_side_rows': int(same.sum()),
            'current_certified_direction_rows': int(np.count_nonzero(score)),
            'first_last_side_flip_rows': int(np.count_nonzero(
                labels[:, 0] != labels[:, -1])),
            'at_least_one_sampled_adjacent_side_flip_rows': int(np.count_nonzero(
                np.any(labels[:, 1:] != labels[:, :-1], axis=1))),
            'same_tf_current_sign_auc_by_checkpoint': [
                sign_owner.rank_auc(labels[:, q].astype(np.uint8), score)
                for q in range(len(QUERY_MINUTES))],
        }
    return out


def measure(head: str, policy_ref: dict, plan: dict) -> dict:
    bound = state.load_bound_five_endpoint_targets()
    coverage = json.loads(Path(plan['input_refs']['coverage']['path']).read_text())
    fit, later_rows, later_issues = prior.axes(coverage, bound)
    fit_rows, fit_issues = fit['fit_parent_rows'], fit['fit_decision_ns']
    need(len(fit_rows) == 47338 and len(later_rows) == 4096
         and coverage['score_start_ns'] == 1735689600000000000
         and coverage['score_end_exclusive_ns'] == 1743465600000000000,
         'COHORT_AXIS')
    now_fit, reasons_fit = transition._current(bound, fit_issues)
    now_later, reasons_later = transition._current(bound, later_issues)
    need(np.array_equal(now_fit == 0, reasons_fit != 0)
         and np.array_equal(now_later == 0, reasons_later != 0),
         'CAUSAL_CURRENT')
    tape_path = Path(plan['input_refs']['candidate_tape']['path'])
    with np.load(io.BytesIO(tape_path.read_bytes()), allow_pickle=False) as archive:
        need(set(archive.files) == tape_owner.TAPE_KEYS, 'TAPE_MEMBERS')
        tape = MinuteQueryTargetTape(
            np.array(archive['validity_ns'], copy=True),
            np.array(archive['m1_source_row'], copy=True),
            np.array(archive['allowed_set'], copy=True),
            np.array(archive['path_observed'], copy=True),
            np.array(archive['gap_kind'], copy=True))
    source = json.loads(Path(plan['input_refs']['minute_source_review']['path']).read_text())
    cohorts = {}
    weekly = {}
    for name, rows, issues, now, cutoff, partition in (
        ('fit', fit_rows, fit_issues, now_fit, coverage['score_start_ns'], 'fit'),
        ('later', later_rows, later_issues, now_later,
         coverage['score_end_exclusive_ns'], 'score')):
        path = gather_fixed_path(tape, issues, fold_upper_ns=cutoff)
        endpoint = {key: path[key][:, -1, :]
                    for key in ('validity_ns', 'mature', 'allowed', 'eligible')}
        old = bound.gather(rows, issues, fold_id=2, partition=partition)
        paired.check_endpoint_parity(endpoint, old, name)
        first = {tf: {
            'mature': int(path['mature'][:, 0, j].sum()),
            'eligible': int(path['eligible'][:, 0, j].sum()),
            'known_side': int(np.count_nonzero(
                path['eligible'][:, 0, j]
                & np.isin(path['allowed'][:, 0, j], (1, 4))))}
            for j, tf in enumerate(TIMEFRAMES)}
        need(first == source['minute1_all_TF'][name],
             'REUSED_MINUTE_COUNTS:' + name)
        cohorts[name] = {
            'rows': len(issues), 'coverage': _coverage(path),
            'transition_intervals': observed_transition_intervals(path),
            'same_tf_persistence': same_tf_persistence_reference(path, now),
            'same_rows_known_side': _same_rows_known_side(path, now),
        }
        if name == 'later':
            weeks = objective._weeks(issues)
            for week in np.unique(weeks):
                subset = weeks == week
                selected = _slice(path, subset)
                weekly[str(week)] = {
                    'rows': int(subset.sum()),
                    'M5_persistence': same_tf_persistence_reference(
                        selected, now[subset])['per_tf']['M5'],
                    'M5_transition_intervals': observed_transition_intervals(
                        selected)['intervals_by_tf']['M5'],
                }
            need(len(weekly) == 14, 'WEEK_COUNT')
    need(prior.sha(tape_path) == plan['input_refs']['candidate_tape']['sha256']
         and prior.git('rev-parse', 'HEAD') == head
         and prior.git('status', '--porcelain') == ''
         and prior.ref(POLICY) == policy_ref, 'BOOKEND')
    return {
        'schema_version': 'gx1_fold2_multiminute_path_result_v1',
        'status': 'COMPLETE', 'stage': STAGE, 'source_head': head,
        'plan': prior.ref(PLAN), 'policy': policy_ref,
        'observed_utc': datetime.now(timezone.utc).isoformat(),
        'query_sha256': QUERY_SHA256, 'fit_rows': len(fit_issues),
        'later_rows': len(later_issues), 'cohorts': cohorts,
        'later_M5_by_utc_week': weekly,
        'model_forwards': 0, 'optimizer_steps': 0, 'model_fit': False,
        'test_data_used': False, 'trade_selection': False,
        'joint_path_probability_claimed': False,
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
        'schema_version': 'gx1_fold2_multiminute_path_claim_v1',
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
            'schema_version': 'gx1_fold2_multiminute_path_terminal_v1',
            'status': 'COMPLETE', 'exit_code': 0, 'stage': STAGE,
            'claim': claim, 'result': prior.ref(ROOT / 'RESULT.json'),
            'source_head': head, 'test_data_used': False}
        _publish_bytes(ROOT / 'TERMINAL.json', prior.encode(terminal))
    except BaseException as exc:
        terminal = {
            'schema_version': 'gx1_fold2_multiminute_path_terminal_v1',
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
