"""One-use TRAIN-only description of sampled market-state endpoint persistence.

This measures certified M5-issue-grid endpoints, not continuous M1 state
running time, predictive skill, target choice, Entry value or trading value.
No target construction, model fit, TEST or broker access is present.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import signal
import subprocess
import time

import numpy as np

from gx1.contracts import market_state_five_endpoint_target_binding_v1 as target
from gx1.contracts.gx1_capped_execution_v1 import (
    require_capped_cpu_audit_execution, require_capped_lock_ancestry,
)
from gx1.contracts.immutable_event_authority_v1 import _fsync_directory, _publish_bytes

REPO = Path('/home/andre2/src/GX1_CURRENT')
RUNS = Path('/home/andre2/GX1_RUNS')
STAGE = 'market_state_endpoint_persistence_20261107_001'
SCHEMA = 'gx1_market_state_endpoint_persistence_plan_v1'
PLAN_PATH = REPO / 'configs/research' / (STAGE.upper() + '.json')
OUT_ROOT = RUNS / STAGE.upper()
POLICY = REPO / 'NEXT_RUN_POLICY.json'
MINUTE_NS = 60_000_000_000
M5_NS = 5 * MINUTE_NS
TIMEFRAMES = ('M5', 'M15', 'H1', 'H4', 'D1')
LEADS = (5, 15, 60, 240, 1440)
STATE_BITS = {'DOWN': 1, 'FLAT': 2, 'UP': 4}
SIDE_BITS = (1, 4)
INPUT_REFS = {name: {'path': path, 'sha256': digest}
              for name, (path, digest) in target.REFS.items()}
SOURCE_PATHS = frozenset({
    'gx1/audit/verify_market_state_endpoint_persistence_v1.py',
    'gx1/contracts/market_state_five_endpoint_target_binding_v1.py',
    'gx1/contracts/market_state_five_endpoint_contract_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'scripts/gx1_capped_run.sh',
})
EXECUTION = {'class': 'audit', 'runner': 'scripts/gx1_capped_run.sh',
             'memory': '4G', 'swap': '512M', 'timeout_seconds': 600,
             'kill_after_seconds': 10,
             'module': 'gx1.audit.verify_market_state_endpoint_persistence_v1',
             'model_fits': 0, 'target_builds': 0}
LIMITS = {'memory_max': 4 * 1024**3, 'memory_high': 4 * 1024**3,
          'swap': 512 * 1024**2, 'pids': 64, 'numerical_threads': 1,
          'internal_wall_seconds': 540, 'max_output_bytes': 1024**2}
METRIC_CONTRACT = {
    'scope': 'full_selected_TRAIN_target_118706_issues_with_118249_model_join_reported',
    'issue_cadence_minutes': 5, 'lead_minutes': list(LEADS),
    'certified_bits': STATE_BITS,
    'pointwise_side': 'exact_own_lead_UTC_both_certified_SIDE_same_or_opposite',
    'pointwise_intermediate_gaps': 'counted_separately_not_continuity',
    'run': 'same_singleton_at_consecutive_5min_issues_same_UTC_validity_month',
    'known_run_end': 'next_exact_5min_singleton_different_including_FLAT',
    'censor': 'ACTIVE_or_ALL_unknown_side_untyped_clock_gap_validity_month_data_edge',
    'sampled_span_minutes': '(endpoint_count_minus_one)_times_five',
    'continuous_M1_duration_proven': False,
    'overlapping_pairs_independent': False,
    'decision': 'DESCRIPTIVE_ONLY_NO_TARGET_OR_MODEL_SELECTION',
}
PUBLISHED_BYTES = 0


def _need(ok: bool, reason: str) -> None:
    if not ok:
        raise RuntimeError('MARKET_STATE_ENDPOINT_PERSISTENCE_' + reason)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checked_small_json(item: dict) -> dict:
    path = Path(item['path'])
    _need(path.is_absolute() and path.is_file() and not path.is_symlink()
          and 0 < path.stat().st_size <= 4 * 1024**2
          and _sha(path) == item['sha256'], 'SMALL_REF:' + str(path))
    return json.loads(path.read_text())


def _small_reference_chain() -> None:
    receipts = {name: _checked_small_json(item)
                for name, item in INPUT_REFS.items()
                if name not in ('mask', 'model_axes')}
    result, terminal = receipts['mask_result'], receipts['mask_terminal']
    corrected, selected = receipts['mask_plan'], receipts['selected_plan']
    source, model = receipts['target_source_plan'], receipts['model_plan']
    _need(result.get('row_masks') == INPUT_REFS['mask']
          and result.get('plan') == INPUT_REFS['mask_plan']
          and result.get('status') == 'COMPLETE'
          and result.get('decision') == 'PASS_SEVEN_LOCAL_SIDE_DEMOTIONS_SOURCE_ONLY'
          and result.get('test_data_used') is False
          and terminal.get('result') == INPUT_REFS['mask_result']
          and terminal.get('row_masks') == INPUT_REFS['mask']
          and terminal.get('status') == 'COMPLETE'
          and terminal.get('exit_code') == 0
          and terminal.get('test_data_used') is False
          and corrected.get('inputs', {}).get('selected', {}).get('plan')
              == INPUT_REFS['selected_plan']
          and selected.get('inputs', {}).get('full_plan')
              == INPUT_REFS['target_source_plan']
          and source.get('lead_minutes') == list(LEADS)
          and source.get('test_data_used') is False
          and model.get('axes', {}).get('source') == INPUT_REFS['model_axes']
          and model.get('axes', {}).get('rows') == 118249
          and model.get('axes', {}).get('decision_rule')
              == 'bar_start_ns_plus_300000000000',
          'SELECTED_MASK_RECEIPTS')


def _clean_head() -> str:
    _need(Path.cwd().resolve() == REPO
          and subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip()
              == 'work/gx1-current'
          and subprocess.check_output(['git', 'status', '--porcelain'], text=True) == '',
          'CLEAN_SOURCE')
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()


def _policy_gate(policy: dict, plan_ref: dict) -> None:
    gate = policy.get(STAGE, {})
    _need(all(policy.get(key) is False for key in
              ('training_enabled', 'native_training_enabled',
               'native_launch_authorized', 'full_val_allowed'))
          and gate.get('status') == gate.get('decision') == 'ADMITTED'
          and gate.get('plan') == plan_ref
          and gate.get('output_root') == str(OUT_ROOT)
          and gate.get('execution_authorized') is True
          and gate.get('data_reads_authorized') is True
          and gate.get('model_fits_authorized') == 0
          and gate.get('target_builds_authorized') == 0
          and gate.get('training_authorized') is False
          and gate.get('test_authorized') is False
          and gate.get('checkpoint_reads_authorized') is False
          and gate.get('broker_access_allowed') is False
          and gate.get('test_data_used') is False
          and gate.get('claimed_invocations') == gate.get('completed_invocations') == 0
          and gate.get('max_invocations') == 1
          and gate.get('relaunch_allowed') is False
          and gate.get('memory_max') == '4G'
          and gate.get('swap_max') == '512M'
          and gate.get('wall_seconds') == 600,
          'POLICY_GATE')


def planned_command(plan_sha256: str) -> list[str]:
    return ['timeout', '--signal=TERM', '--kill-after=10s', '600s',
            EXECUTION['runner'], '--class', 'audit', '--mem', '4G',
            '--swap', '512M', '--', '.venv/bin/python', '-m',
            EXECUTION['module'], '--plan', str(PLAN_PATH),
            '--plan-sha256', plan_sha256, '--out-dir', str(OUT_ROOT)]


def preflight(plan_path: Path, plan_sha256: str, out_dir: Path, *,
              admitted: bool = False) -> tuple[dict, dict, str, dict | None]:
    head = _clean_head()
    _need(plan_path == PLAN_PATH and out_dir == OUT_ROOT and not out_dir.exists()
          and len(plan_sha256) == 64 and _sha(plan_path) == plan_sha256,
          'PLAN_IDENTITY')
    plan = json.loads(plan_path.read_text())
    _small_reference_chain()
    _need(plan.get('schema_version') == SCHEMA
          and plan.get('stage') == STAGE
          and plan.get('status') == 'PREPARED_NOT_ADMITTED'
          and plan.get('output_root') == str(OUT_ROOT)
          and plan.get('max_invocations') == 1
          and plan.get('input_refs') == INPUT_REFS
          and plan.get('source_owners') is not None
          and set(plan['source_owners']) == SOURCE_PATHS
          and all(_sha(REPO / path) == digest
                  for path, digest in plan['source_owners'].items())
          and plan.get('execution') == EXECUTION
          and plan.get('limits') == LIMITS
          and plan.get('metric_contract') == METRIC_CONTRACT
          and plan.get('command_template') == planned_command('{plan_sha256}')
          and plan.get('expected_issue_rows') == 118706
          and plan.get('expected_joined_model_rows') == 118249
          and plan.get('read_only') is True
          and plan.get('test_data_used') is False
          and isinstance(plan.get('source_base_head'), str)
          and len(plan['source_base_head']) == 40
          and subprocess.run(['git', 'merge-base', '--is-ancestor',
                              plan['source_base_head'], head],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL).returncode == 0,
          'PLAN_SCOPE')
    plan_ref = {'path': str(plan_path), 'sha256': plan_sha256}
    policy_ref = None
    if admitted:
        cap = require_capped_cpu_audit_execution()
        lock = require_capped_lock_ancestry()
        _need(cap['memory_max'] == cap['memory_high'] == LIMITS['memory_max']
              and cap['swap'] == LIMITS['swap']
              and cap['pids'] == LIMITS['pids']
              and cap['numerical_threads'] == 1
              and lock['scope'] == cap['cgroup_path'][len('/sys/fs/cgroup'):],
              'CAP_LIMITS')
        policy_ref = {'path': str(POLICY), 'sha256': _sha(POLICY)}
        policy = json.loads(POLICY.read_text())
        _need(_sha(POLICY) == policy_ref['sha256'], 'POLICY_STABLE_READ')
        _policy_gate(policy, plan_ref)
    return plan, plan_ref, head, policy_ref


def _quantiles(values: np.ndarray) -> dict:
    if not len(values):
        return {'count': 0, 'minimum': None, 'p10': None, 'median': None,
                'p90': None, 'maximum': None}
    points = np.quantile(values.astype(np.float64), (0.1, 0.5, 0.9), method='linear')
    return {'count': int(len(values)), 'minimum': int(values.min()),
            'p10': float(points[0]), 'median': float(points[1]),
            'p90': float(points[2]), 'maximum': int(values.max())}


def _boundary_reason(bits: np.ndarray, issues: np.ndarray, months: np.ndarray,
                     index: int, neighbor: int, state: int) -> str:
    if neighbor < 0 or neighbor >= len(bits):
        return 'data_edge'
    if months[index] != months[neighbor]:
        return 'validity_month'
    if abs(int(issues[index]) - int(issues[neighbor])) != M5_NS:
        return 'untyped_clock_gap'
    if int(bits[neighbor]) in STATE_BITS.values():
        _need(int(bits[neighbor]) != state, 'RUN_CONTINUATION_SPLIT')
        return 'known_different'
    return 'unknown_side_or_state'


def _run_report(bits: np.ndarray, issues: np.ndarray, months: np.ndarray,
                lead_minutes: int) -> dict:
    singleton = np.isin(bits, tuple(STATE_BITS.values()))
    adjacent = (np.diff(issues) == M5_NS) & (months[1:] == months[:-1])
    continuation = adjacent & singleton[1:] & singleton[:-1] & (bits[1:] == bits[:-1])
    starts = np.flatnonzero(singleton & ~np.r_[False, continuation])
    ends = np.flatnonzero(singleton & ~np.r_[continuation, False])
    _need(len(starts) == len(ends) and bool(np.all(starts <= ends)), 'RUN_AXES')
    records = []
    for start, end in zip(starts, ends):
        state = int(bits[start])
        left = _boundary_reason(bits, issues, months, int(start), int(start) - 1, state)
        right = _boundary_reason(bits, issues, months, int(end), int(end) + 1, state)
        records.append((state, int(end - start + 1), left, right))
    def summarize(rows: list[tuple]) -> dict:
        lengths = np.asarray([row[1] for row in rows], dtype=np.int64)
        spans = (lengths - 1) * 5
        bounded = [row for row in rows if row[2] == row[3] == 'known_different']
        bounded_spans = np.asarray([(row[1] - 1) * 5 for row in bounded], dtype=np.int64)
        reasons = ('known_different', 'unknown_side_or_state',
                   'untyped_clock_gap', 'validity_month', 'data_edge')
        return {'runs': len(rows), 'sampled_endpoint_count': int(lengths.sum()),
                'length_endpoints': _quantiles(lengths),
                'sampled_span_minutes': _quantiles(spans),
                'left_boundary': {key: sum(row[2] == key for row in rows) for key in reasons},
                'right_boundary': {key: sum(row[3] == key for row in rows) for key in reasons},
                'fully_bounded_runs': len(bounded),
                'fully_bounded_sampled_span_minutes': _quantiles(bounded_spans),
                'fully_bounded_sampled_span_below_own_lead': int((bounded_spans < lead_minutes).sum()),
                'fully_bounded_sampled_span_at_least_own_lead': int((bounded_spans >= lead_minutes).sum())}
    by_state = {name: summarize([r for r in records if r[0] == bit])
                for name, bit in STATE_BITS.items()}
    return {'by_state': by_state,
            'certified_side_combined': summarize([r for r in records if r[0] in SIDE_BITS])}


def measure_endpoint_persistence(issue_ns: np.ndarray, allowed_set: np.ndarray,
                                 m1_source_row: np.ndarray) -> dict:
    """Pure descriptive metrics on full selected issue axis; no M1 continuity claim."""
    _need(isinstance(issue_ns, np.ndarray) and issue_ns.dtype == np.int64
          and issue_ns.ndim == 1 and len(issue_ns) > 1
          and bool(np.all(np.diff(issue_ns) > 0))
          and bool(np.all(issue_ns % M5_NS == 0))
          and isinstance(allowed_set, np.ndarray) and allowed_set.dtype == np.uint8
          and allowed_set.shape == (len(issue_ns), 5)
          and isinstance(m1_source_row, np.ndarray) and m1_source_row.dtype == np.int64
          and m1_source_row.shape == allowed_set.shape
          and bool(np.isin(allowed_set, (1, 2, 4, 5, 7)).all())
          and bool(np.all(m1_source_row[allowed_set != 7] >= 0)),
          'TARGET_AXES')
    result = {}
    for col, (tf, lead) in enumerate(zip(TIMEFRAMES, LEADS)):
        bits = allowed_set[:, col]
        validity = issue_ns + lead * MINUTE_NS
        months = validity.view('datetime64[ns]').astype('datetime64[M]').astype(np.int64)
        certified = np.isin(bits, SIDE_BITS)
        wanted = issue_ns + lead * MINUTE_NS
        pos = np.searchsorted(issue_ns, wanted)
        safe = np.minimum(pos, len(issue_ns) - 1)
        exact = (pos < len(issue_ns)) & (issue_ns[safe] == wanted)
        future_side = certified[safe]
        paired = certified & exact & future_side
        same = paired & (bits == bits[safe])
        opposite = paired & (bits != bits[safe])
        gap_prefix = np.r_[0, np.cumsum(np.diff(issue_ns) != M5_NS)]
        gap_between = np.zeros(len(issue_ns), dtype=bool)
        gap_between[paired] = (gap_prefix[safe[paired]] - gap_prefix[np.flatnonzero(paired)]) > 0
        result[tf] = {
            'lead_minutes': lead, 'issues': len(issue_ns),
            'certified_side_endpoints': int(certified.sum()),
            'certified_side_coverage': float(certified.mean()),
            'flat_endpoints': int((bits == 2).sum()),
            'active_unknown_side_endpoints': int((bits == 5).sum()),
            'all_unknown_endpoints': int((bits == 7).sum()),
            'pointwise_own_lead': {
                'starting_certified_side': int(certified.sum()),
                'missing_exact_future_issue': int((certified & ~exact).sum()),
                'future_not_certified_side': int((certified & exact & ~future_side).sum()),
                'both_certified': int(paired.sum()),
                'same_side': int(same.sum()), 'opposite_side': int(opposite.sum()),
                'same_side_rate': (float(same.sum() / paired.sum()) if paired.any() else None),
                'paired_with_intermediate_clock_gap': int(gap_between.sum()),
                'paired_across_validity_month': int((paired & (months != months[safe])).sum()),
            },
            'sampled_runs': _run_report(bits, issue_ns, months, lead),
        }
        pair = result[tf]['pointwise_own_lead']
        _need(pair['starting_certified_side'] == pair['missing_exact_future_issue']
              + pair['future_not_certified_side'] + pair['both_certified']
              and pair['both_certified'] == pair['same_side'] + pair['opposite_side'],
              'PAIR_PARTITION:' + tf)
    return {'schema_version': 'gx1_market_state_endpoint_persistence_metrics_v1',
            'scope': 'full_selected_TRAIN_endpoints_descriptive_only',
            'issues': len(issue_ns), 'timeframes': result,
            'continuous_M1_duration_proven': False,
            'overlapping_pairs_independent': False}


def _load_full_selected() -> tuple[dict, dict]:
    bound = target.load_bound_five_endpoint_targets()
    with np.load(io.BytesIO(target._checked_bytes('mask', 4 * 1024**2)),
                 allow_pickle=False) as saved:
        _need(set(saved.files) == target.MASK_KEYS, 'SELECTED_MASK_KEYS')
        issues = np.array(saved['issue_ns'], copy=True)
        allowed = np.array(saved['allowed_set'], copy=True)
        m1_rows = np.array(saved['m1_source_row'], copy=True)
    _need(len(issues) == 118706 and len(bound.parent_rows) == 118249
          and bound.unjoined_target_rows == 457, 'EXPECTED_ROWS')
    at = np.searchsorted(issues, bound.decision_ns)
    _need(bool(np.all(at < len(issues)))
          and np.array_equal(issues[at], bound.decision_ns)
          and np.array_equal(allowed[at], bound.allowed_set)
          and np.array_equal(m1_rows[at], bound.m1_source_row),
          'MODEL_JOIN')
    report = measure_endpoint_persistence(issues, allowed, m1_rows)
    return report, {'full_target_issues': len(issues),
                    'joined_model_issues': len(bound.parent_rows),
                    'excluded_target_issues': bound.unjoined_target_rows,
                    'joined_parent_rows_sha256': target._axis_hash(bound.parent_rows),
                    'joined_decision_ns_sha256': target._axis_hash(bound.decision_ns)}


def _write(root: Path, name: str, value: dict) -> dict:
    global PUBLISHED_BYTES
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    _need(name in ('FAILED.json', 'TERMINAL.json')
          or PUBLISHED_BYTES + len(data) <= LIMITS['max_output_bytes'], 'OUTPUT_BUDGET')
    _publish_bytes(root / name, data)
    PUBLISHED_BYTES += len(data)
    return {'path': str(root / name), 'sha256': hashlib.sha256(data).hexdigest()}


def _source_bookend(plan: dict, plan_sha256: str, policy_sha256: str, head: str) -> None:
    _need(_clean_head() == head and _sha(PLAN_PATH) == plan_sha256
          and _sha(POLICY) == policy_sha256
          and all(_sha(REPO / path) == digest for path, digest in plan['source_owners'].items()),
          'SOURCE_BOOKEND')
    for name, item in INPUT_REFS.items():
        _need(_sha(Path(item['path'])) == item['sha256'], 'INPUT_BOOKEND:' + name)


def _wall_limit(_signum, _frame) -> None:
    raise TimeoutError('MARKET_STATE_ENDPOINT_PERSISTENCE_INTERNAL_WALL_LIMIT')


def run(plan_path: Path, plan_sha256: str, out_dir: Path) -> dict:
    global PUBLISHED_BYTES
    started = time.monotonic()
    plan, plan_ref, head, policy_ref = preflight(plan_path, plan_sha256, out_dir,
                                                  admitted=True)
    _need(policy_ref is not None, 'POLICY_REQUIRED')
    deadline = started + LIMITS['internal_wall_seconds']
    remaining = deadline - time.monotonic()
    _need(remaining > 0 and signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0),
          'WALL_BUDGET_PRECLAIM')
    out_dir.mkdir(parents=True, exist_ok=False)
    PUBLISHED_BYTES = 0
    claim = None
    old_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _wall_limit)
        signal.setitimer(signal.ITIMER_REAL, remaining)
        _fsync_directory(out_dir.parent)
        claim = _write(out_dir, 'CLAIM.json', {
            'schema_version': 'gx1_market_state_endpoint_persistence_claim_v1',
            'invocation': 1, 'plan': plan_ref, 'policy': policy_ref,
            'source_head': head, 'read_only': True, 'model_fits': 0,
            'target_builds': 0, 'test_data_used': False})
        metrics, join = _load_full_selected()
        _source_bookend(plan, plan_sha256, policy_ref['sha256'], head)
        result = _write(out_dir, 'RESULT.json', {
            'schema_version': 'gx1_market_state_endpoint_persistence_result_v1',
            'status': 'COMPLETE_DESCRIPTIVE_NO_TARGET_SELECTION',
            'plan': plan_ref, 'policy': policy_ref, 'claim': claim,
            'source_head': head, 'elapsed_seconds': time.monotonic() - started,
            'input_refs': INPUT_REFS, 'join': join, 'metrics': metrics,
            'metric_contract': METRIC_CONTRACT,
            'read_only': True, 'model_fits': 0, 'target_builds': 0,
            'continuous_M1_duration_proven': False,
            'test_data_used': False, 'entry_exit_or_trading_authority': False})
        signal.setitimer(signal.ITIMER_REAL, 0)
        terminal = _write(out_dir, 'TERMINAL.json', {
            'schema_version': 'gx1_market_state_endpoint_persistence_terminal_v1',
            'status': 'COMPLETE', 'exit_code': 0, 'plan': plan_ref,
            'claim': claim, 'result': result,
            'no_relaunch_under_this_plan': True, 'test_data_used': False})
        return {'result': result, 'terminal': terminal}
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL, 0)
        failed = _write(out_dir, 'FAILED.json', {
            'schema_version': 'gx1_market_state_endpoint_persistence_failure_v1',
            'plan': plan_ref, 'claim': claim, 'source_head': head,
            'error_type': type(exc).__name__, 'error': str(exc)[:2048],
            'elapsed_seconds': time.monotonic() - started,
            'no_relaunch_under_this_plan': True})
        _write(out_dir, 'TERMINAL.json', {
            'schema_version': 'gx1_market_state_endpoint_persistence_terminal_v1',
            'status': 'FAILED', 'exit_code': 1, 'plan': plan_ref,
            'claim': claim, 'failure': failed,
            'no_relaunch_under_this_plan': True, 'test_data_used': False})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        plan, plan_ref, head, _ = preflight(args.plan, args.plan_sha256, args.out_dir)
        print(json.dumps({'status': 'READY_NO_CLAIM_NO_PAYLOAD',
                          'plan': plan_ref, 'source_head': head,
                          'issue_rows': plan['expected_issue_rows'],
                          'joined_model_rows': plan['expected_joined_model_rows'],
                          'claim_created': False, 'payload_read': False,
                          'planned_command': planned_command(args.plan_sha256),
                          'test_data_used': False}, sort_keys=True))
        return 0
    print(json.dumps(run(args.plan, args.plan_sha256, args.out_dir), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
