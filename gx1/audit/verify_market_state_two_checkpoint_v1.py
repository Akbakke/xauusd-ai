"""TRAIN-only midpoint/endpoint coverage and missed-path lower bounds.

Reuses the selected local M1 classifier. This does not publish a target or fit a
model. Interior evidence is observed-minute evidence, never a full episode.
"""
from __future__ import annotations

import argparse
import hashlib
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
STAGE = 'market_state_two_checkpoint_20261107_001'
PLAN_PATH = REPO / 'configs/research' / (STAGE.upper() + '.json')
OUT_ROOT = RUNS / STAGE.upper()
POLICY = duration.POLICY
INPUT_REFS = duration.INPUT_REFS
SOURCE_PATHS = duration.SOURCE_PATHS | {
    'gx1/audit/verify_market_state_two_checkpoint_v1.py',
}
EXECUTION = {
    'class': 'audit', 'runner': 'scripts/gx1_capped_run.sh',
    'memory': '4G', 'swap': '512M', 'timeout_seconds': 1800,
    'kill_after_seconds': 10,
    'module': 'gx1.audit.verify_market_state_two_checkpoint_v1',
    'temporary_target_tapes': 2, 'published_target_tapes': 0, 'model_fits': 0,
}
LIMITS = {**duration.LIMITS, 'max_output_bytes': 1024**2}
CONTRACT = {
    'scope': 'TRAIN_only_118706_issues_five_TF_selected_state_formula',
    'midpoint_minutes': 'ceil(native_TF_lead_minutes/2)',
    'endpoint_minutes': list(duration.LEADS),
    'required_selected_endpoint_bit_and_source_row_mismatches': 0,
    'gray_middle': 'ACTIVE_side_unknown',
    'missed_activity': 'both_checkpoints_FLAT_interior_observed_ACTIVE',
    'missed_opposite_side': 'both_checkpoints_same_singleton_side_interior_opposite_path_confirmed',
    'interior_clock': 'observed_M1_validity_minutes_strictly_after_issue_and_before_endpoint',
    'censor': 'immature_first_then_missing_M1_unknown_labels_gaps_and_unconfirmed_side',
    'interpretation': 'observed_lower_bounds_not_complete_episodes_or_joint_path_probability',
    'test_data_used': False,
}
MINUTE_NS = duration.MINUTE_NS


def _need(ok: bool, code: str) -> None:
    if not ok:
        raise RuntimeError('MARKET_STATE_TWO_CHECKPOINT_' + code)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _count_between(mask: np.ndarray, left: np.ndarray,
                   right: np.ndarray) -> np.ndarray:
    prefix = np.r_[0, np.cumsum(mask, dtype=np.int64)]
    return prefix[right] - prefix[left]


def _partition(issues: np.ndarray, mid: np.ndarray, end: np.ndarray,
               mid_present: np.ndarray, end_present: np.ndarray,
               mid_immature: np.ndarray, end_immature: np.ndarray,
               interior_active: np.ndarray, interior_opposite: np.ndarray,
               observed: np.ndarray, unknown: np.ndarray,
               ambiguous: np.ndarray, unconfirmed_side: np.ndarray,
               closure: np.ndarray,
               source_gap: np.ndarray, lead: int) -> dict:
    both = (mid != 7) & (end != 7)
    flat = (mid == 2) & (end == 2)
    same_side = ((mid == 1) & (end == 1)) | ((mid == 4) & (end == 4))
    mature_interval = ~end_immature
    complete_clock = (observed == lead) & mature_interval
    activity_observable = flat & complete_clock & (unknown == 0)
    side_observable = (same_side & complete_clock & (unknown == 0)
                       & (ambiguous == 0) & (unconfirmed_side == 0))
    missed_activity = flat & interior_active
    missed_side = same_side & interior_opposite
    return {
        'issues': int(len(issues)),
        'midpoint_exact_M1': int(mid_present.sum()),
        'midpoint_immature': int(mid_immature.sum()),
        'midpoint_missing_exact_M1': int((~mid_present & ~mid_immature).sum()),
        'endpoint_exact_M1': int(end_present.sum()),
        'endpoint_immature': int(end_immature.sum()),
        'endpoint_missing_exact_M1': int((~end_present & ~end_immature).sum()),
        'midpoint_bits': {str(bit): int((mid == bit).sum())
                          for bit in (1, 2, 4, 5, 7)},
        'endpoint_bits': {str(bit): int((end == bit).sum())
                          for bit in (1, 2, 4, 5, 7)},
        'both_informative_checkpoints': int(both.sum()),
        'both_flat': int(flat.sum()),
        'both_same_singleton_side': int(same_side.sum()),
        'activity_observable_both_flat': int(activity_observable.sum()),
        'side_observable_both_same_singleton': int(side_observable.sum()),
        'missed_activity_on_activity_observable': int(
            (missed_activity & activity_observable).sum()),
        'missed_opposite_on_side_observable': int(
            (missed_side & side_observable).sum()),
        'missed_activity_observed_lower_bound': int(missed_activity.sum()),
        'missed_activity_with_full_M1_clock': int((missed_activity & complete_clock).sum()),
        'missed_opposite_path_confirmed_lower_bound': int(missed_side.sum()),
        'missed_opposite_with_full_M1_clock': int((missed_side & complete_clock).sum()),
        'full_M1_clock': int(complete_clock.sum()),
        'missing_M1_minutes': int(((lead - observed) * mature_interval).sum()),
        'intervals_with_missing_M1': int(((observed < lead) & mature_interval).sum()),
        'intervals_with_unknown_label': int(((unknown > 0) & mature_interval).sum()),
        'intervals_with_active_side_unknown': int(((ambiguous > 0) & mature_interval).sum()),
        'intervals_with_unconfirmed_singleton_side': int(
            ((unconfirmed_side > 0) & mature_interval).sum()),
        'intervals_crossing_documented_closure': int(((closure > 0) & mature_interval).sum()),
        'intervals_crossing_unknown_source_gap': int(((source_gap > 0) & mature_interval).sum()),
        'missing_M1_without_gap_successor_inside_interval': int(
            ((observed < lead) & (closure == 0) & (source_gap == 0)
             & mature_interval).sum()),
    }


def measure_tf(issues: np.ndarray, axis: np.ndarray, bits: np.ndarray,
               path_confirmed: np.ndarray, gap_kind: np.ndarray,
               lead: int, cutoff: int) -> dict:
    """Pure vectorized diagnostic; an unknown interior never proves absence."""
    _need(issues.ndim == axis.ndim == bits.ndim == path_confirmed.ndim
          == gap_kind.ndim == 1
          and len(axis) == len(bits) == len(path_confirmed) == len(gap_kind)
          and np.all(np.diff(axis) > 0)
          and np.isin(bits, (1, 2, 4, 5, 7)).all()
          and np.isin(gap_kind, (0, 1, 2)).all()
          and np.all(path_confirmed <= np.isin(bits, (1, 2, 4)))
          and type(lead) is int and lead in duration.LEADS
          and np.all(np.diff(issues) > 0) and len(issues) > 0
          and isinstance(cutoff, int),
          'METRIC_INPUT')
    half = (lead + 1) // 2
    mid_at, mid_present = duration.classify.exact_positions(
        axis, issues, half, cutoff, np)
    end_at, end_present = duration.classify.exact_positions(
        axis, issues, lead, cutoff, np)
    mid_immature = issues + half * MINUTE_NS >= cutoff
    end_immature = issues + lead * MINUTE_NS >= cutoff
    _need(np.all(~(mid_present & mid_immature))
          and np.all(~(end_present & end_immature)), 'CUTOFF_PRECEDENCE')
    mid = np.full(len(issues), 7, np.uint8)
    end = np.full(len(issues), 7, np.uint8)
    mid[mid_present] = bits[mid_at[mid_present]]
    end[end_present] = bits[end_at[end_present]]
    cutoff_row = np.searchsorted(axis, cutoff, side='left')
    left = np.minimum(np.searchsorted(axis, issues, side='right'), cutoff_row)
    interior_right = np.minimum(
        np.searchsorted(axis, issues + lead * MINUTE_NS, side='left'), cutoff_row)
    full_right = np.minimum(
        np.searchsorted(axis, issues + lead * MINUTE_NS, side='right'), cutoff_row)
    _need(np.all(left <= interior_right) and np.all(interior_right <= full_right),
          'INTERIOR_CLOCK')
    observed = full_right - left
    _need(np.all((0 <= observed) & (observed <= lead)), 'OBSERVED_CLOCK')
    active = _count_between(np.isin(bits, (1, 4, 5)), left, interior_right) > 0
    opposite_down = _count_between((bits == 1) & path_confirmed,
                                   left, interior_right) > 0
    opposite_up = _count_between((bits == 4) & path_confirmed,
                                 left, interior_right) > 0
    opposite = np.where(mid == 1, opposite_up, opposite_down)
    unknown = _count_between(bits == 7, left, full_right)
    ambiguous = _count_between(bits == 5, left, full_right)
    unconfirmed_side = _count_between(
        np.isin(bits, (1, 4)) & ~path_confirmed, left, full_right)
    closure = _count_between(gap_kind == 1, left, full_right)
    source_gap = _count_between(gap_kind == 2, left, full_right)
    months = issues.astype('datetime64[ns]').astype('datetime64[M]').astype(str)
    def group(select: np.ndarray) -> dict:
        return _partition(issues[select], mid[select], end[select],
                          mid_present[select], end_present[select],
                          mid_immature[select], end_immature[select],
                          active[select], opposite[select], observed[select],
                          unknown[select], ambiguous[select],
                          unconfirmed_side[select], closure[select],
                          source_gap[select], lead)
    return {
        'lead_minutes': lead, 'midpoint_minutes': half,
        'midpoint_exact_M1': int(mid_present.sum()),
        'endpoint_exact_M1': int(end_present.sum()),
        'all': group(np.ones(len(issues), bool)),
        'months': {str(month): group(months == month) for month in np.unique(months)},
    }


def planned_command(plan_sha256: str) -> list[str]:
    return ['timeout', '--signal=TERM', '--kill-after=10s', '1800s',
            EXECUTION['runner'], '--class', 'audit', '--mem', '4G',
            '--swap', '512M', '--', '.venv/bin/python', '-m', EXECUTION['module'],
            '--plan', str(PLAN_PATH), '--plan-sha256', plan_sha256,
            '--out-dir', str(OUT_ROOT)]


def preflight(plan_path: Path, digest: str, out_dir: Path,
              admitted: bool = False) -> tuple[dict, dict, str, dict | None]:
    head = duration._clean_head()
    _need(plan_path == PLAN_PATH and out_dir == OUT_ROOT and not out_dir.exists()
          and len(digest) == 64 and _sha(plan_path) == digest,
          'PLAN_IDENTITY')
    plan = json.loads(plan_path.read_text())
    duration.endpoint._small_reference_chain()
    _need(plan.get('schema_version') == 'gx1_market_state_two_checkpoint_plan_v1'
          and plan.get('stage') == STAGE
          and plan.get('status') == 'PREPARED_NOT_ADMITTED'
          and plan.get('output_root') == str(OUT_ROOT)
          and plan.get('max_invocations') == 1
          and plan.get('input_refs') == INPUT_REFS
          and plan.get('source_owners', {}).keys() == SOURCE_PATHS
          and all(_sha(REPO / p) == h for p, h in plan['source_owners'].items())
          and plan.get('execution') == EXECUTION
          and plan.get('limits') == LIMITS
          and plan.get('metric_contract') == CONTRACT
          and plan.get('expected_m1_rows') == duration.base.M1_ROWS == 591682
          and plan.get('expected_issue_rows') == 118706
          and plan.get('numba_version') == duration.NUMBA_VERSION
              == duration.numba.__version__
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
              and cap['swap'] == LIMITS['swap'] and cap['pids'] == LIMITS['pids']
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
              and gate.get('target_publish_authorized') is False
              and gate.get('model_fits_authorized') == 0
              and gate.get('training_authorized') is False
              and gate.get('test_authorized') is False
              and gate.get('checkpoint_reads_authorized') is False
              and gate.get('broker_access_allowed') is False
              and gate.get('test_data_used') is False
              and gate.get('claimed_invocations') == gate.get('completed_invocations') == 0
              and gate.get('max_invocations') == 1
              and gate.get('relaunch_allowed') is False
              and gate.get('memory_max') == '4G' and gate.get('swap_max') == '512M'
              and gate.get('wall_seconds') == 1800,
              'POLICY_GATE')
    return plan, plan_ref, head, policy_ref


def _write(root: Path, name: str, value: dict) -> dict:
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    _need(len(data) <= LIMITS['max_output_bytes'], 'OUTPUT_BUDGET')
    _publish_bytes(root / name, data)
    return {'path': str(root / name), 'sha256': hashlib.sha256(data).hexdigest()}


def _timeout(_signum, _frame) -> None:
    raise TimeoutError('MARKET_STATE_TWO_CHECKPOINT_INTERNAL_WALL_LIMIT')


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
            'schema_version': 'gx1_market_state_two_checkpoint_claim_v1',
            'invocation': 1, 'plan': plan_ref, 'policy': policy_ref,
            'source_head': head, 'read_only': True, 'test_data_used': False})
        full_plan = duration.endpoint._checked_small_json(INPUT_REFS['target_source_plan'])
        m1, issues, m5_rows, stats, a05, a10, gaps, authority = (
            duration._temporary_tapes(full_plan, deadline))
        selected = duration._selected_full_mask()
        bits, path, cold = duration._classify_all_m1(
            m1, a05, a10, gaps, deadline)
        parity = duration._selected_endpoint_parity(
            selected, issues, m5_rows, a05['validity_ns'],
            a05['source_row'], bits)
        cutoff = duration.base.pd_ns(duration.base.TRAIN_END)
        metrics = {tf: measure_tf(
            issues, a05['validity_ns'], bits[:, col], path[:, col],
            gaps, duration.LEADS[col], cutoff)
            for col, tf in enumerate(duration.TIMEFRAMES)}
        _need(all(v['all']['issues'] == len(issues)
                  and sum(m['issues'] for m in v['months'].values()) == len(issues)
                  for v in metrics.values()), 'METRIC_PARTITION')
        _need(duration._clean_head() == head and _sha(PLAN_PATH) == digest
              and _sha(POLICY) == policy_ref['sha256']
              and all(_sha(REPO / p) == h for p, h in plan['source_owners'].items())
              and all(_sha(Path(v['path'])) == v['sha256']
                      for v in INPUT_REFS.values())
              and all(duration.base.regular(Path(p)) == stamp
                      for p, stamp in stats.items()), 'SOURCE_BOOKEND')
        result = _write(out_dir, 'RESULT.json', {
            'schema_version': 'gx1_market_state_two_checkpoint_result_v1',
            'status': 'COMPLETE_READ_ONLY_TWO_CHECKPOINT_DIAGNOSTIC',
            'plan': plan_ref, 'policy': policy_ref, 'claim': claim,
            'source_head': head, 'elapsed_seconds': time.monotonic() - started,
            'input_refs': INPUT_REFS, 'metric_contract': CONTRACT,
            'm1_rows': len(a05['validity_ns']), 'm5_issues': len(issues),
            'documented_closure_gaps': int((gaps == 1).sum()),
            'unknown_source_gaps': int((gaps == 2).sum()),
            'source_clock_sha256': authority['m1_clock_sha256'],
            'selected_endpoint_parity': parity,
            'cold_local_side': cold, 'metrics': metrics,
            'read_only': True, 'temporary_target_tapes': 2,
            'published_target_tapes': 0, 'model_fits': 0,
            'test_data_used': False, 'target_selection': False,
            'limit': 'Two checkpoints can miss turns and returns. Counts are observed lower bounds; missing source, unknown state, and intraminute order remain censored.'})
        signal.setitimer(signal.ITIMER_REAL, 0)
        terminal = _write(out_dir, 'TERMINAL.json', {
            'schema_version': 'gx1_market_state_two_checkpoint_terminal_v1',
            'status': 'COMPLETE', 'exit_code': 0,
            'plan': plan_ref, 'claim': claim, 'result': result,
            'no_relaunch_under_this_plan': True, 'test_data_used': False})
        return {'result': result, 'terminal': terminal}
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL, 0)
        failed = _write(out_dir, 'FAILED.json', {
            'schema_version': 'gx1_market_state_two_checkpoint_failure_v1',
            'plan': plan_ref, 'claim': claim, 'source_head': head,
            'error_type': type(exc).__name__, 'error': str(exc)[:2048],
            'no_relaunch_under_this_plan': True})
        _write(out_dir, 'TERMINAL.json', {
            'schema_version': 'gx1_market_state_two_checkpoint_terminal_v1',
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
