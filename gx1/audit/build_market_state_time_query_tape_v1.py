"""One-use TRAIN candidate tape for future market-state time queries.

Reuses the audited local M1 classifier and requires exact selected-endpoint
parity before publishing a compact target-only tape. No model, TEST or trades.
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

from gx1.audit import verify_market_state_m1_duration_v1 as duration
from gx1.contracts.gx1_capped_execution_v1 import (
    require_capped_cpu_audit_execution, require_capped_lock_ancestry,
)
from gx1.contracts.immutable_event_authority_v1 import _fsync_directory, _publish_bytes

REPO = duration.REPO
RUNS = duration.RUNS
STAGE = 'market_state_time_query_tape_20261107_001'
SCHEMA = 'gx1_market_state_time_query_tape_plan_v1'
PLAN_PATH = REPO / 'configs/research' / (STAGE.upper() + '.json')
OUT_ROOT = RUNS / STAGE.upper()
POLICY = duration.POLICY
INPUT_REFS = duration.INPUT_REFS
SOURCE_PATHS = duration.SOURCE_PATHS | {
    'gx1/audit/build_market_state_time_query_tape_v1.py',
}
TAPE_NAME = 'TRAIN_CANDIDATE_TAPE.npz'
TAPE_KEYS = frozenset({
    'validity_ns', 'm1_source_row', 'allowed_set', 'path_observed', 'gap_kind',
})
EXECUTION = {
    'class': 'audit', 'runner': 'scripts/gx1_capped_run.sh',
    'memory': '4G', 'swap': '512M', 'timeout_seconds': 1800,
    'kill_after_seconds': 10,
    'module': 'gx1.audit.build_market_state_time_query_tape_v1',
    'temporary_target_tapes': 2, 'published_target_tapes': 1,
    'model_fits': 0,
}
LIMITS = {**duration.LIMITS, 'max_tape_bytes': 32 * 1024**2}
CONTRACT = {
    'scope': 'TRAIN_only_candidate_all_observed_M1_minutes_five_TF',
    'state_source': 'selected_0.5_ATR_activity_1.0_ATR_local_side_formula',
    'gray_middle': 'ACTIVE_side_unknown',
    'allowed_bits': {'DOWN': 1, 'FLAT': 2, 'UP': 4,
                     'ACTIVE_SIDE_UNKNOWN': 5, 'ALL_UNKNOWN': 7},
    'tape_members': sorted(TAPE_KEYS),
    'required_selected_endpoint_bit_and_source_row_mismatches': 0,
    'unsampled_minutes': 'new_local_diagnostic_extension_candidate_not_old_published_target',
    'future_labels_as_model_inputs': False,
    'selected_training_target': False,
    'test_data_used': False,
}


def _need(ok: bool, code: str) -> None:
    if not ok:
        raise RuntimeError('MARKET_STATE_TIME_QUERY_TAPE_' + code)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def planned_command(plan_sha256: str) -> list[str]:
    return ['timeout', '--signal=TERM', '--kill-after=10s', '1800s',
            EXECUTION['runner'], '--class', 'audit', '--mem', '4G',
            '--swap', '512M', '--', '.venv/bin/python', '-m', EXECUTION['module'],
            '--plan', str(PLAN_PATH), '--plan-sha256', plan_sha256,
            '--out-dir', str(OUT_ROOT)]


def preflight(plan_path: Path, digest: str, out_dir: Path,
              *, admitted: bool = False) -> tuple[dict, dict, str, dict | None]:
    head = duration._clean_head()
    _need(plan_path == PLAN_PATH and out_dir == OUT_ROOT and not out_dir.exists()
          and len(digest) == 64 and _sha(plan_path) == digest,
          'PLAN_IDENTITY')
    plan = json.loads(plan_path.read_text())
    duration.endpoint._small_reference_chain()
    _need(plan.get('schema_version') == SCHEMA
          and plan.get('stage') == STAGE
          and plan.get('status') == 'PREPARED_NOT_ADMITTED'
          and plan.get('output_root') == str(OUT_ROOT)
          and plan.get('max_invocations') == 1
          and plan.get('input_refs') == INPUT_REFS
          and plan.get('source_owners', {}).keys() == SOURCE_PATHS
          and all(_sha(REPO / name) == hash_ for name, hash_ in
                  plan['source_owners'].items())
          and plan.get('execution') == EXECUTION
          and plan.get('limits') == LIMITS
          and plan.get('metric_contract') == CONTRACT
          and plan.get('numba_version') == duration.NUMBA_VERSION
          == duration.numba.__version__
          and plan.get('expected_m1_rows') == duration.base.M1_ROWS == 591682
          and plan.get('expected_issue_rows') == 118706
          and plan.get('command_template') == planned_command('{plan_sha256}')
          and plan.get('read_only') is True
          and plan.get('test_data_used') is False
          and isinstance(plan.get('source_base_head'), str)
          and len(plan['source_base_head']) == 40
          and subprocess.run(['git', 'merge-base', '--is-ancestor',
                              plan['source_base_head'], head],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL).returncode == 0,
          'PLAN_SCOPE')
    plan_ref = {'path': str(plan_path), 'sha256': digest}
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
        gate = policy.get(STAGE, {})
        _need(_sha(POLICY) == policy_ref['sha256']
              and all(policy.get(k) is False for k in
                      ('training_enabled', 'native_training_enabled',
                       'native_launch_authorized', 'full_val_allowed'))
              and gate.get('status') == gate.get('decision') == 'ADMITTED'
              and gate.get('plan') == plan_ref
              and gate.get('output_root') == str(OUT_ROOT)
              and gate.get('execution_authorized') is True
              and gate.get('data_reads_authorized') is True
              and gate.get('target_tape_builds_authorized') == 2
              and gate.get('target_publish_authorized') is True
              and gate.get('model_fits_authorized') == 0
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
              and gate.get('wall_seconds') == 1800,
              'POLICY_GATE')
    return plan, plan_ref, head, policy_ref


def encode_tape(axis: np.ndarray, source_rows: np.ndarray,
                allowed: np.ndarray, path: np.ndarray,
                gap_kind: np.ndarray) -> tuple[bytes, dict]:
    """Pack target-only values; reject wrong clocks, masks or archive schema."""
    n = duration.base.M1_ROWS
    _need(axis.dtype == source_rows.dtype == np.int64
          and axis.shape == source_rows.shape == gap_kind.shape == (n,)
          and allowed.shape == path.shape == (n, len(duration.TIMEFRAMES))
          and allowed.dtype == gap_kind.dtype == np.uint8
          and path.dtype == np.bool_
          and np.all(np.diff(axis) > 0)
          and np.array_equal(source_rows,
                             np.arange(5193205, 5193205 + n, dtype=np.int64))
          and np.isin(allowed, (1, 2, 4, 5, 7)).all()
          and np.isin(gap_kind, (0, 1, 2)).all()
          and np.all(path <= np.isin(allowed, (1, 2, 4))),
          'TAPE_SCHEMA')
    buffer = io.BytesIO()
    np.savez_compressed(buffer, validity_ns=axis, m1_source_row=source_rows,
                        allowed_set=allowed, path_observed=path,
                        gap_kind=gap_kind)
    payload = buffer.getvalue()
    _need(0 < len(payload) <= LIMITS['max_tape_bytes'], 'TAPE_BYTES')
    with np.load(io.BytesIO(payload), allow_pickle=False) as saved:
        _need(set(saved.files) == TAPE_KEYS
              and np.array_equal(saved['validity_ns'], axis)
              and np.array_equal(saved['m1_source_row'], source_rows)
              and np.array_equal(saved['allowed_set'], allowed)
              and np.array_equal(saved['path_observed'], path)
              and np.array_equal(saved['gap_kind'], gap_kind),
              'TAPE_ROUNDTRIP')
    report = {
        'm1_rows': n, 'timeframes': list(duration.TIMEFRAMES),
        'allowed_counts': {tf: {str(bit): int((allowed[:, col] == bit).sum())
                                for bit in (1, 2, 4, 5, 7)}
                           for col, tf in enumerate(duration.TIMEFRAMES)},
        'path_confirmed_counts': {tf: int(path[:, col].sum())
                                  for col, tf in enumerate(duration.TIMEFRAMES)},
        'documented_closure_gaps': int((gap_kind == 1).sum()),
        'unknown_source_gaps': int((gap_kind == 2).sum()),
    }
    return payload, report


def _write(root: Path, name: str, value: dict) -> dict:
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    _need(len(payload) <= LIMITS['max_output_bytes'], 'RECEIPT_BYTES')
    _publish_bytes(root / name, payload)
    return {'path': str(root / name), 'sha256': hashlib.sha256(payload).hexdigest()}


def _timeout(_signum, _frame) -> None:
    raise TimeoutError('MARKET_STATE_TIME_QUERY_TAPE_INTERNAL_WALL_LIMIT')


def run(plan_path: Path, digest: str, out_dir: Path) -> dict:
    started = time.monotonic()
    plan, plan_ref, head, policy_ref = preflight(plan_path, digest, out_dir,
                                                  admitted=True)
    deadline = started + LIMITS['internal_wall_seconds']
    _need(policy_ref is not None and deadline > time.monotonic() + 60,
          'WALL_BUDGET_PRECLAIM')
    out_dir.mkdir(parents=True, exist_ok=False)
    old_handler = signal.getsignal(signal.SIGALRM)
    claim = None
    try:
        signal.signal(signal.SIGALRM, _timeout)
        signal.setitimer(signal.ITIMER_REAL, deadline - time.monotonic())
        _fsync_directory(out_dir.parent)
        claim = _write(out_dir, 'CLAIM.json', {
            'schema_version': 'gx1_market_state_time_query_tape_claim_v1',
            'invocation': 1, 'plan': plan_ref, 'policy': policy_ref,
            'source_head': head, 'read_only': True, 'test_data_used': False})
        full_plan = duration.endpoint._checked_small_json(INPUT_REFS['target_source_plan'])
        m1, issues, m5_rows, stats, a05, a10, gaps, authority = (
            duration._temporary_tapes(full_plan, deadline))
        selected = duration._selected_full_mask()
        allowed, path, cold = duration._classify_all_m1(
            m1, a05, a10, gaps, deadline)
        parity = duration._selected_endpoint_parity(
            selected, issues, m5_rows, a05['validity_ns'],
            a05['source_row'], allowed)
        _need(len(parity) == len(duration.TIMEFRAMES)
              and all(p['bit_mismatches'] == p['m1_source_row_mismatches'] == 0
                      for p in parity.values()), 'ENDPOINT_PARITY')
        tape_bytes, tape_report = encode_tape(
            a05['validity_ns'], a05['source_row'], allowed, path, gaps)
        _need(duration._clean_head() == head and _sha(PLAN_PATH) == digest
              and _sha(POLICY) == policy_ref['sha256']
              and all(_sha(REPO / name) == hash_ for name, hash_ in
                      plan['source_owners'].items())
              and all(_sha(Path(v['path'])) == v['sha256']
                      for v in INPUT_REFS.values())
              and all(duration.base.regular(Path(path_)) == stamp
                      for path_, stamp in stats.items()),
              'SOURCE_BOOKEND')
        _publish_bytes(out_dir / TAPE_NAME, tape_bytes)
        tape_ref = {'path': str(out_dir / TAPE_NAME),
                    'sha256': hashlib.sha256(tape_bytes).hexdigest(),
                    'bytes': len(tape_bytes)}
        result = _write(out_dir, 'RESULT.json', {
            'schema_version': 'gx1_market_state_time_query_tape_result_v1',
            'status': 'COMPLETE_CANDIDATE_TAPE_ENDPOINT_PARITY',
            'plan': plan_ref, 'policy': policy_ref, 'claim': claim,
            'source_head': head, 'elapsed_seconds': time.monotonic() - started,
            'input_refs': INPUT_REFS, 'metric_contract': CONTRACT,
            'source_clock_sha256': authority['m1_clock_sha256'],
            'm5_issues': len(issues), 'candidate_tape': tape_ref,
            'tape_report': tape_report,
            'selected_endpoint_parity': parity, 'cold_local_side': cold,
            'read_only': True, 'temporary_target_tapes': 2,
            'published_candidate_target_tapes': 1, 'model_fits': 0,
            'test_data_used': False, 'target_selection': False,
            'limit': 'New local M1 extension with selected-endpoint parity, not yet a selected training target or proof of model value.'})
        signal.setitimer(signal.ITIMER_REAL, 0)
        terminal = _write(out_dir, 'TERMINAL.json', {
            'schema_version': 'gx1_market_state_time_query_tape_terminal_v1',
            'status': 'COMPLETE', 'exit_code': 0,
            'plan': plan_ref, 'claim': claim, 'result': result,
            'candidate_tape': tape_ref,
            'no_relaunch_under_this_plan': True, 'test_data_used': False})
        return {'candidate_tape': tape_ref, 'result': result, 'terminal': terminal}
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL, 0)
        failed = _write(out_dir, 'FAILED.json', {
            'schema_version': 'gx1_market_state_time_query_tape_failure_v1',
            'plan': plan_ref, 'claim': claim, 'source_head': head,
            'error_type': type(exc).__name__, 'error': str(exc)[:2048],
            'no_relaunch_under_this_plan': True})
        _write(out_dir, 'TERMINAL.json', {
            'schema_version': 'gx1_market_state_time_query_tape_terminal_v1',
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
        plan, ref, head, _ = preflight(args.plan, args.plan_sha256, args.out_dir)
        print(json.dumps({
            'status': 'READY_NO_CLAIM_NO_PAYLOAD', 'plan': ref,
            'source_head': head, 'm1_rows': plan['expected_m1_rows'],
            'issues': plan['expected_issue_rows'],
            'planned_command': planned_command(args.plan_sha256),
            'claim_created': False, 'payload_read': False,
            'test_data_used': False}, sort_keys=True))
        return 0
    print(json.dumps(run(args.plan, args.plan_sha256, args.out_dir), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
