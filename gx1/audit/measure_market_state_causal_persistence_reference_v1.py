"""One-use, unfitted TRAIN audit of exact causal market-state persistence.

A current state comes only from an earlier issuance whose endpoint is this
issuance's closed M5 clock. Unknown inputs abstain; nothing is fitted or filled.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path
import signal
import subprocess
import time

import numpy as np

from gx1.audit import measure_market_state_five_endpoint_prior_v1 as prior
from gx1.contracts import market_state_five_endpoint_target_binding_v1 as target
from gx1.contracts.market_state_five_endpoint_contract_v1 import (
    LEAD_MINUTES, TIMEFRAMES, proper_score_sums,
)
from gx1.contracts.immutable_event_authority_v1 import _fsync_directory, _publish_bytes

REPO, RUNS, POLICY = prior.REPO, prior.RUNS, prior.POLICY
STAGE = 'market_state_causal_persistence_reference_20261107_001'
SCHEMA = 'gx1_market_state_causal_persistence_plan_v1'
PLAN_PATH = REPO / 'configs/research' / (STAGE.upper() + '.json')
OUT_ROOT = RUNS / STAGE.upper()
RULE = 'exact_predecessor_same_m1_row_no_fit_no_fallback_v1'
LIMITS = dict(prior.LIMITS)
EXECUTION = {
    'class': 'audit', 'runner': 'scripts/gx1_capped_run.sh',
    'memory': '2G', 'swap': '512M', 'timeout_seconds': 600,
    'kill_after_seconds': 10,
    'module': 'gx1.audit.measure_market_state_causal_persistence_reference_v1',
    'model_forwards': 0, 'optimizer_steps': 0,
}
SOURCE_PATHS = prior.SOURCE_PATHS | frozenset({
    'gx1/audit/measure_market_state_causal_persistence_reference_v1.py',
})
PRIOR_RESULT = {
    'path': str(RUNS / 'MARKET_STATE_FIVE_ENDPOINT_PRIOR_20261107_001/RESULT.json'),
    'sha256': '5304433c42b43ea2bdf593066e627c8904604a751aa27c6fa5dab6349efefa48',
}
PRIOR_TERMINAL = {
    'path': str(RUNS / 'MARKET_STATE_FIVE_ENDPOINT_PRIOR_20261107_001/TERMINAL.json'),
    'sha256': '8f9804b384baf0c1b3d5d321465f00424da17956c06ef39929f107767bc4e9a4',
}
PRIOR_PLAN = {
    'path': str(REPO / 'configs/research/MARKET_STATE_FIVE_ENDPOINT_PRIOR_20261107_001.json'),
    'sha256': '7710bd98c7e14b1f2a6d4506f3f3499b02a0c04078f95c8b279b9cd033f00170',
}
# Zero means forecastable. Every other code is an explicit abstention reason.
REASONS = ('FORECAST', 'NO_M5_ANCHOR', 'M5_ANCHOR_M1_MISSING',
           'NO_TF_PREDECESSOR', 'TF_PREDECESSOR_M1_MISSING',
           'CURRENT_ALL_UNKNOWN')


def _need(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError('CAUSAL_PERSISTENCE_' + reason)


def _sha(path: Path) -> str:
    return prior._sha(path)


def _read_ref(ref: dict, *, max_bytes: int = 1024 * 1024) -> dict:
    path = Path(ref['path'])
    _need(path.is_absolute() and path.is_file()
          and 0 < path.stat().st_size <= max_bytes
          and _sha(path) == ref['sha256'], 'REF_HASH:' + path.name)
    return json.loads(path.read_text())


def _write(root: Path, name: str, value: dict) -> dict:
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    _need(len(data) <= 1024 * 1024 and json.loads(data) == value, 'PUBLICATION')
    path = root / name
    _publish_bytes(path, data)
    return {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest()}


def _receipt_preflight() -> dict:
    # The prior owner checks all named target receipts and hashes both payloads
    # without decoding them before CLAIM.
    prior._receipt_preflight()
    _read_ref(PRIOR_PLAN)
    result = _read_ref(PRIOR_RESULT)
    terminal = _read_ref(PRIOR_TERMINAL)
    _need(result.get('schema_version') == 'gx1_market_state_five_endpoint_prior_result_v1'
          and result.get('decision') == 'PASS_TRAIN_PRIOR_REFERENCE_ONLY'
          and result.get('status') == 'COMPLETE'
          and result.get('plan') == PRIOR_PLAN
          and result.get('target_rows_joined') == 118249
          and result.get('unjoined_target_rows') == 457
          and result.get('smoothing') == prior.SMOOTHING
          and result.get('model_forwards') == result.get('optimizer_steps') == 0
          and result.get('test_data_used') is False
          and isinstance(result.get('folds'), list)
          and [fold.get('fold_id') for fold in result['folds']] == [1, 2, 3, 4, 5]
          and terminal.get('result') == PRIOR_RESULT
          and terminal.get('status') == 'COMPLETE'
          and terminal.get('exit_code') == 0
          and terminal.get('no_relaunch_under_this_plan') is True,
          'PRIOR_RECEIPTS')
    return result


def _qualify_plan(plan: dict, plan_ref: dict, head: str) -> None:
    refs = {name: {'path': path, 'sha256': digest}
            for name, (path, digest) in target.REFS.items()}
    _need(isinstance(plan, dict)
          and plan.get('schema_version') == SCHEMA
          and plan.get('stage') == STAGE
          and plan.get('status') == 'PREPARED_NOT_ADMITTED'
          and plan.get('output_root') == str(OUT_ROOT)
          and not OUT_ROOT.exists()
          and plan.get('fold_ids') == [1, 2, 3, 4, 5]
          and plan.get('reference_rule') == RULE
          and plan.get('max_invocations') == 1
          and plan.get('input_refs') == refs
          and plan.get('prior_result') == PRIOR_RESULT
          and plan.get('prior_terminal') == PRIOR_TERMINAL
          and plan.get('limits') == LIMITS
          and plan.get('execution') == EXECUTION
          and plan.get('model_forwards_authorized') == 0
          and plan.get('optimizer_steps_authorized') == 0
          and plan.get('training_authorized') is False
          and plan.get('baseline_estimation_authorized') is False
          and plan.get('test_data_used') is False
          and isinstance(plan.get('source_base_head'), str)
          and len(plan['source_base_head']) == 40
          and subprocess.run(['git', 'merge-base', '--is-ancestor',
                              plan['source_base_head'], head],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL).returncode == 0,
          'PLAN_SCOPE')
    owners = plan.get('source_owners')
    _need(isinstance(owners, dict) and set(owners) == SOURCE_PATHS
          and all(isinstance(digest, str) and len(digest) == 64
                  and (REPO / path).is_file()
                  and _sha(REPO / path) == digest
                  for path, digest in owners.items()), 'SOURCE_OWNERS')
    _need(plan_ref['path'] == str(PLAN_PATH), 'PLAN_PATH')


def preflight_read_only(plan_path: Path, plan_sha256: str,
                        out_dir: Path) -> dict:
    head = prior._clean_head()
    _need(Path(plan_path) == PLAN_PATH and Path(out_dir) == OUT_ROOT
          and PLAN_PATH.is_file() and not OUT_ROOT.exists()
          and isinstance(plan_sha256, str) and len(plan_sha256) == 64
          and _sha(PLAN_PATH) == plan_sha256, 'PLAN_IDENTITY')
    plan = json.loads(PLAN_PATH.read_text())
    ref = {'path': str(PLAN_PATH), 'sha256': plan_sha256}
    _qualify_plan(plan, ref, head)
    _receipt_preflight()
    return {'status': 'READY_NO_CLAIM_NO_PAYLOAD', 'plan': ref,
            'source_head': head, 'source_owners_checked': len(SOURCE_PATHS),
            'claim_created': False, 'model_forwards': 0,
            'optimizer_steps': 0, 'test_data_used': False}


def _preflight(plan_path: Path, plan_sha256: str,
               out_dir: Path) -> tuple[dict, dict, dict, str, dict]:
    from gx1.contracts.gx1_capped_execution_v1 import (
        require_capped_cpu_audit_execution, require_capped_lock_ancestry,
    )
    head = prior._clean_head()
    _need(Path(plan_path) == PLAN_PATH and Path(out_dir) == OUT_ROOT
          and PLAN_PATH.is_file() and not OUT_ROOT.exists()
          and isinstance(plan_sha256, str) and len(plan_sha256) == 64
          and _sha(PLAN_PATH) == plan_sha256, 'PLAN_IDENTITY')
    plan = json.loads(PLAN_PATH.read_text())
    plan_ref = {'path': str(PLAN_PATH), 'sha256': plan_sha256}
    _qualify_plan(plan, plan_ref, head)
    policy_ref = {'path': str(POLICY), 'sha256': _sha(POLICY)}
    policy = _read_ref(policy_ref, max_bytes=4 * 1024 * 1024)
    cap = require_capped_cpu_audit_execution()
    lock = require_capped_lock_ancestry()
    _need(all(cap.get(key) == value for key, value in LIMITS.items()
              if key != 'max_wall_seconds')
          and type(cap.get('pids')) is int and cap['pids'] <= 64
          and isinstance(lock.get('scope'), str)
          and cap.get('cgroup_path') == '/sys/fs/cgroup' + lock['scope'],
          'CAP_LOCK')
    gate = policy.get(STAGE, {})
    _need(policy.get('training_enabled') is False
          and policy.get('native_training_enabled') is False
          and policy.get('native_launch_authorized') is False
          and policy.get('full_val_allowed') is False
          and gate.get('decision') == 'ADMITTED'
          and gate.get('plan') == plan_ref
          and gate.get('execution_authorized') is True
          and gate.get('data_reads_authorized') is True
          and gate.get('deterministic_reference_authorized') is True
          and gate.get('baseline_estimation_authorized') is False
          and gate.get('claimed_invocations') == gate.get('completed_invocations') == 0
          and gate.get('max_invocations') == 1
          and gate.get('model_forwards_authorized') == 0
          and gate.get('optimizer_steps_authorized') == 0
          and gate.get('training_authorized') is False
          and gate.get('test_authorized') is False
          and gate.get('relaunch_allowed') is False
          and gate.get('output_root') == str(OUT_ROOT)
          and gate.get('memory_max') == '2G'
          and gate.get('swap_max') == '512M'
          and gate.get('wall_seconds') == 600,
          'POLICY_GATE')
    result = _receipt_preflight()
    return plan, plan_ref, policy_ref, head, result


def _current_states(mask_issues: np.ndarray, mask_m1: np.ndarray,
                    mask_allowed: np.ndarray,
                    query_issues: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return current allowed bits and abstention reason codes on exact clocks."""
    n = len(mask_issues)
    q = len(query_issues)
    _need(mask_issues.dtype == query_issues.dtype == np.int64
          and mask_issues.ndim == query_issues.ndim == 1 and n > 0 and q > 0
          and mask_m1.dtype == np.int64 and mask_m1.shape == (n, 5)
          and mask_allowed.dtype == np.uint8 and mask_allowed.shape == (n, 5)
          and bool(np.all(np.diff(mask_issues) > 0))
          and bool(np.all(np.diff(query_issues) > 0))
          and bool(np.isin(mask_allowed, (1, 2, 4, 5, 7)).all()),
          'CURRENT_AXES')
    bits = np.zeros((q, 5), dtype=np.uint8)
    reasons = np.zeros((q, 5), dtype=np.uint8)
    anchor_clock = query_issues - target.M5_NS
    anchor_at = np.searchsorted(mask_issues, anchor_clock)
    anchor_exists = (anchor_at < n) & (mask_issues[np.minimum(anchor_at, n - 1)] == anchor_clock)
    anchor_m1 = np.full(q, -1, dtype=np.int64)
    anchor_m1[anchor_exists] = mask_m1[anchor_at[anchor_exists], 0]
    for j, minutes in enumerate(LEAD_MINUTES):
        predecessor_clock = query_issues - minutes * target.MINUTE_NS
        at = np.searchsorted(mask_issues, predecessor_clock)
        exists = (at < n) & (mask_issues[np.minimum(at, n - 1)] == predecessor_clock)
        predecessor_m1 = np.full(q, -1, dtype=np.int64)
        predecessor_bits = np.zeros(q, dtype=np.uint8)
        predecessor_m1[exists] = mask_m1[at[exists], j]
        predecessor_bits[exists] = mask_allowed[at[exists], j]
        mismatch = (anchor_m1 >= 0) & exists & (predecessor_m1 >= 0) & (predecessor_m1 != anchor_m1)
        _need(not bool(mismatch.any()), 'PHYSICAL_M1_ROW_MISMATCH:' + TIMEFRAMES[j])
        reason = np.zeros(q, dtype=np.uint8)
        reason[~anchor_exists] = 1
        reason[anchor_exists & (anchor_m1 < 0)] = 2
        ready = anchor_exists & (anchor_m1 >= 0)
        reason[ready & ~exists] = 3
        reason[ready & exists & (predecessor_m1 < 0)] = 4
        reason[ready & exists & (predecessor_m1 >= 0) & (predecessor_bits == 7)] = 5
        good = reason == 0
        bits[good, j] = predecessor_bits[good]
        reasons[:, j] = reason
    _need(bool(np.array_equal(bits == 0, reasons != 0)), 'ABSTENTION_PARITY')
    return bits, reasons


def _metric(count: int, wrong: int, prior_score: dict, *, brier_factor: int = 1,
            has_log: bool = False) -> dict:
    _need(0 <= wrong <= count and prior_score['count'] == count, 'PAIRED_SUPPORT')
    total = brier_factor * wrong
    out = {'count': count, 'wrong_count': wrong,
           'persistence_brier_sum': total,
           'persistence_brier_mean': total / count if count else None,
           'prior_brier_sum': prior_score['brier_sum'],
           'prior_brier_mean': prior_score['brier_sum'] / count if count else None}
    if has_log:
        out.update(persistence_log_sum='Infinity' if wrong else 0.0,
                   persistence_log_mean='Infinity' if wrong else (0.0 if count else None),
                   persistence_zero_truth_probability_rows=wrong,
                   prior_log_sum=prior_score['log_sum'],
                   prior_log_mean=prior_score['log_sum'] / count if count else None)
    return out


def _score_rows(now: np.ndarray, reasons: np.ndarray, truth: dict,
                prior_probabilities: np.ndarray, issues: np.ndarray) -> dict:
    allowed, eligible, mature = truth['allowed'], truth['eligible'], truth['mature']
    n = len(issues)
    _need(now.shape == reasons.shape == allowed.shape == eligible.shape == mature.shape == (n, 5)
          and prior_probabilities.shape == (5, 3)
          and bool(np.isfinite(prior_probabilities).all())
          and bool(np.all(prior_probabilities > 0))
          and bool(np.allclose(prior_probabilities.sum(axis=1), 1, rtol=0, atol=1e-15)),
          'SCORE_INPUT')
    predicted = np.broadcast_to(prior_probabilities, (n, 5, 3))
    act_input = np.isin(now, (1, 2, 4, 5))
    side_input = np.isin(now, (1, 4))
    three_input = np.isin(now, (1, 2, 4))
    act_prior = proper_score_sums(predicted, allowed, eligible & act_input, mature)['per_tf']
    side_prior = proper_score_sums(predicted, allowed, eligible & side_input, mature)['per_tf']
    three_prior = proper_score_sums(predicted, allowed, eligible & three_input, mature)['per_tf']
    ready = eligible & mature
    act_support = ready & (allowed != 7) & act_input
    side_support = ready & np.isin(allowed, (1, 4)) & side_input
    three_support = ready & np.isin(allowed, (1, 2, 4)) & three_input
    months = issues.astype('datetime64[ns]').astype('datetime64[M]').astype(str)
    report = {}
    for j, tf in enumerate(TIMEFRAMES):
        a = act_support[:, j]
        s = side_support[:, j]
        c = three_support[:, j]
        a_wrong = int(np.count_nonzero((now[a, j] != 2) != (allowed[a, j] != 2)))
        s_wrong = int(np.count_nonzero(now[s, j] != allowed[s, j]))
        c_wrong = int(np.count_nonzero(now[c, j] != allowed[c, j]))
        reason_counts = {REASONS[k]: int(np.count_nonzero(reasons[:, j] == k))
                         for k in range(1, len(REASONS))}
        state_counts = {str(bit): int(np.count_nonzero(now[:, j] == bit))
                        for bit in (1, 2, 4, 5)}
        monthly = {}
        for month in np.unique(months):
            m = months == month
            monthly[str(month)] = {
                'rows': int(m.sum()),
                'no_forecast': {REASONS[k]: int(np.count_nonzero(m & (reasons[:, j] == k)))
                                for k in range(1, len(REASONS))},
                'current_bits': {str(bit): int(np.count_nonzero(m & (now[:, j] == bit)))
                                 for bit in (1, 2, 4, 5)},
                'activity': {'count': int(np.count_nonzero(m & a)),
                             'wrong_count': int(np.count_nonzero(m & a & ((now[:, j] != 2) != (allowed[:, j] != 2))))},
                'conditional_side': {'count': int(np.count_nonzero(m & s)),
                                     'wrong_count': int(np.count_nonzero(m & s & (now[:, j] != allowed[:, j])))},
                'three_class': {'count': int(np.count_nonzero(m & c)),
                                'wrong_count': int(np.count_nonzero(m & c & (now[:, j] != allowed[:, j])))},
            }
        report[tf] = {
            'score_rows': n,
            'no_forecast': reason_counts,
            'current_bits': state_counts,
            'future_activity_known': int(np.count_nonzero(ready[:, j] & (allowed[:, j] != 7))),
            'future_side_certified': int(np.count_nonzero(ready[:, j] & np.isin(allowed[:, j], (1, 4)))),
            'future_singleton': int(np.count_nonzero(ready[:, j] & np.isin(allowed[:, j], (1, 2, 4)))),
            'activity': _metric(int(a.sum()), a_wrong, act_prior[tf]['activity']),
            'conditional_side': _metric(int(s.sum()), s_wrong,
                                        side_prior[tf]['conditional_side'], has_log=True),
            'three_class': _metric(int(c.sum()), c_wrong,
                                   three_prior[tf]['three_class'], brier_factor=2, has_log=True),
            'months': monthly,
        }
        _need(sum(reason_counts.values()) + sum(state_counts.values()) == n
              and sum(x['activity']['count'] for x in monthly.values()) == int(a.sum())
              and sum(x['conditional_side']['count'] for x in monthly.values()) == int(s.sum())
              and sum(x['three_class']['count'] for x in monthly.values()) == int(c.sum()),
              'COVERAGE_PARITY:' + tf)
    return report


def _measure(prior_result: dict) -> dict:
    bound = target.load_bound_five_endpoint_targets()
    raw = target._checked_bytes('mask', 4 * 1024 * 1024)
    with np.load(io.BytesIO(raw), allow_pickle=False) as mask:
        mask_issues = np.array(mask['issue_ns'], copy=True)
        mask_m1 = np.array(mask['m1_source_row'], copy=True)
        mask_allowed = np.array(mask['allowed_set'], copy=True)
    folds = []
    for fold_id in range(1, 6):
        lower = target.FOLD_BOUNDARIES_NS[fold_id - 1]
        upper = target.FOLD_BOUNDARIES_NS[fold_id]
        score = (bound.decision_ns >= lower) & (bound.decision_ns < upper)
        issues = bound.decision_ns[score]
        parents = bound.parent_rows[score]
        _need(len(issues) == prior_result['folds'][fold_id - 1]['score_rows']
              and int(issues[0]) == prior_result['folds'][fold_id - 1]['score_first_issue_ns']
              and int(issues[-1]) == prior_result['folds'][fold_id - 1]['score_last_issue_ns'],
              'PRIOR_SCORE_AXES')
        truth = bound.gather(parents, issues, fold_id=fold_id, partition='score')
        now, reasons = _current_states(mask_issues, mask_m1, mask_allowed, issues)
        probs = np.asarray(prior_result['folds'][fold_id - 1]['prior_probabilities'], dtype=np.float64)
        scores = _score_rows(now, reasons, truth, probs, issues)
        for tf in TIMEFRAMES:
            old = prior_result['folds'][fold_id - 1]['score_sums'][tf]
            _need(scores[tf]['future_activity_known'] == old['activity']['count']
                  and scores[tf]['future_side_certified'] == old['conditional_side']['count']
                  and scores[tf]['future_singleton'] == old['three_class']['count'],
                  'PRIOR_DENOMINATORS:' + tf)
        folds.append({'fold_id': fold_id, 'score_rows': len(issues),
                      'score_first_issue_ns': int(issues[0]),
                      'score_last_issue_ns': int(issues[-1]),
                      'timeframes': scores})
    _need(sum(fold['score_rows'] for fold in folds) == 88400
          and bound.unjoined_target_rows == 457, 'TOTAL_ROWS')
    pooled = {}
    for tf in TIMEFRAMES:
        rows = [fold['timeframes'][tf] for fold in folds]
        pooled[tf] = {'score_rows': sum(row['score_rows'] for row in rows),
                      'no_forecast': {name: sum(row['no_forecast'][name] for row in rows)
                                      for name in REASONS[1:]},
                      'current_bits': {str(bit): sum(row['current_bits'][str(bit)] for row in rows)
                                       for bit in (1, 2, 4, 5)}}
        for metric in ('activity', 'conditional_side', 'three_class'):
            count = sum(row[metric]['count'] for row in rows)
            wrong = sum(row[metric]['wrong_count'] for row in rows)
            brier = sum(row[metric]['persistence_brier_sum'] for row in rows)
            prior_brier = sum(row[metric]['prior_brier_sum'] for row in rows)
            pooled[tf][metric] = {
                'count': count, 'wrong_count': wrong,
                'persistence_brier_sum': brier,
                'persistence_brier_mean': brier / count if count else None,
                'prior_brier_sum': prior_brier,
                'prior_brier_mean': prior_brier / count if count else None,
            }
            if metric != 'activity':
                pooled[tf][metric]['persistence_log_mean'] = 'Infinity' if wrong else (0.0 if count else None)
                pooled[tf][metric]['persistence_zero_truth_probability_rows'] = wrong
                prior_log = sum(row[metric]['prior_log_sum'] for row in rows)
                pooled[tf][metric]['prior_log_mean'] = prior_log / count if count else None
    return {'scope_limit': 'Unfitted persistence from earlier TRAIN target masks on five score folds. Same pinned M1 row index is checked, not raw M1 replay or live input availability. Abstention limits coverage; no transition fit, model skill, TEST generalization, Entry, or economics.',
            'reference_rule': RULE, 'target_rows_joined': len(bound.parent_rows),
            'unjoined_target_rows': bound.unjoined_target_rows,
            'folds': folds, 'pooled': pooled,
            'model_forwards': 0, 'optimizer_steps': 0,
            'baseline_parameters_fitted': False, 'test_data_used': False}


@contextmanager
def _bounded_worker(started: float):
    previous = signal.getsignal(signal.SIGALRM)
    remaining = signal.getitimer(signal.ITIMER_REAL)
    _need(previous == signal.SIG_DFL and remaining == (0.0, 0.0), 'TIMER_OWNER')
    seconds = LIMITS['max_wall_seconds'] - (time.monotonic() - started) - 10.0
    _need(seconds > 0, 'PREFLIGHT_WALL')
    def expire(_signum, _frame):
        raise TimeoutError('CAUSAL_PERSISTENCE_WORKER_WALL_LIMIT')
    signal.signal(signal.SIGALRM, expire)
    try:
        signal.setitimer(signal.ITIMER_REAL, seconds)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def run(plan_path: Path, plan_sha256: str, out_dir: Path) -> dict:
    began = time.monotonic()
    plan, plan_ref, policy_ref, head, old = _preflight(plan_path, plan_sha256, out_dir)
    OUT_ROOT.mkdir(parents=True, exist_ok=False)
    claim = None
    try:
        _fsync_directory(OUT_ROOT.parent)
        claim = _write(OUT_ROOT, 'CLAIM.json', {
            'schema_version': 'gx1_market_state_causal_persistence_claim_v1',
            'plan': plan_ref, 'policy': policy_ref, 'source_head': head,
            'invocation': 1, 'fold_ids': plan['fold_ids']})
        with _bounded_worker(began):
            data = _measure(old)
        _need(_sha(PLAN_PATH) == plan_sha256
              and _sha(POLICY) == policy_ref['sha256']
              and prior._clean_head() == head, 'FINAL_SOURCE_IDENTITY')
        result = _write(OUT_ROOT, 'RESULT.json', {
            'schema_version': 'gx1_market_state_causal_persistence_result_v1',
            'status': 'COMPLETE',
            'decision': 'PASS_DETERMINISTIC_TRAIN_PERSISTENCE_REFERENCE_ONLY',
            'plan': plan_ref, 'claim': claim, 'source_head': head,
            'elapsed_seconds': time.monotonic() - began, **data})
        terminal = _write(OUT_ROOT, 'TERMINAL.json', {
            'schema_version': 'gx1_market_state_causal_persistence_terminal_v1',
            'status': 'COMPLETE', 'exit_code': 0,
            'plan': plan_ref, 'claim': claim, 'result': result,
            'model_forwards': 0, 'optimizer_steps': 0,
            'test_data_used': False, 'no_relaunch_under_this_plan': True,
            'output_inventory': ['CLAIM.json', 'RESULT.json', 'TERMINAL.json']})
        return {'result': result, 'terminal': terminal}
    except Exception as exc:
        failure = _write(OUT_ROOT, 'FAILED.json', {
            'schema_version': 'gx1_market_state_causal_persistence_failure_v1',
            'plan': plan_ref, 'claim': claim, 'source_head': head,
            'error_type': type(exc).__name__, 'error': str(exc)[:2048],
            'no_relaunch_under_this_plan': True})
        _write(OUT_ROOT, 'TERMINAL.json', {
            'schema_version': 'gx1_market_state_causal_persistence_terminal_v1',
            'status': 'FAILED', 'exit_code': 1,
            'plan': plan_ref, 'claim': claim, 'failure': failure,
            'model_forwards': 0, 'optimizer_steps': 0,
            'test_data_used': False, 'no_relaunch_under_this_plan': True,
            'output_inventory': (['CLAIM.json', 'FAILED.json', 'TERMINAL.json']
                                 if claim else ['FAILED.json', 'TERMINAL.json'])})
        raise


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        print(json.dumps(preflight_read_only(args.plan, args.plan_sha256,
                                             args.out_dir), sort_keys=True))
    else:
        print(json.dumps(run(args.plan, args.plan_sha256, args.out_dir), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
