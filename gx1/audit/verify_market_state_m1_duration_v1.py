"""One-use TRAIN-only description of selected state runs on observed M1 minutes.

The two existing fixed-factor tapes are temporary audit inputs. They are
reconciled to every published selected M5-issued endpoint before run statistics
are published. No target tape, model, TEST, Entry, or trading output is created.
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

import numba
from numba import njit
import numpy as np

from gx1.audit import build_full_train_market_state_endpoint_candidate_v1 as base
from gx1.audit import verify_market_state_endpoint_candidate_v2 as classify
from gx1.audit import verify_market_state_endpoint_persistence_v1 as endpoint
from gx1.contracts import entry_dynamic_mtf_state_targets_v1 as target
from gx1.contracts import market_state_five_endpoint_target_binding_v1 as binding
from gx1.contracts.gx1_capped_execution_v1 import (
    require_capped_cpu_audit_execution, require_capped_lock_ancestry,
)
from gx1.contracts.immutable_event_authority_v1 import _fsync_directory, _publish_bytes
from gx1.scripts.audit_dynamic_mtf_state_closure_observability_v1 import early_active_proof

REPO = Path('/home/andre2/src/GX1_CURRENT')
RUNS = Path('/home/andre2/GX1_RUNS')
STAGE = 'market_state_m1_duration_20261107_001'
SCHEMA = 'gx1_market_state_m1_duration_plan_v1'
PLAN_PATH = REPO / 'configs/research' / (STAGE.upper() + '.json')
OUT_ROOT = RUNS / STAGE.upper()
POLICY = REPO / 'NEXT_RUN_POLICY.json'
INPUT_REFS = {name: ref for name, ref in endpoint.INPUT_REFS.items()
              if name != 'model_axes'}
SOURCE_PATHS = frozenset({
    'gx1/audit/verify_market_state_m1_duration_v1.py',
    'gx1/audit/verify_market_state_endpoint_persistence_v1.py',
    'gx1/audit/build_full_train_market_state_endpoint_candidate_v1.py',
    'gx1/audit/verify_market_state_endpoint_candidate_v2.py',
    'gx1/contracts/entry_dynamic_mtf_state_targets_v1.py',
    'gx1/contracts/market_state_five_endpoint_target_binding_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'gx1/contracts/unified_exit_market_closure_authority_v1.py',
    'gx1/scripts/audit_dynamic_mtf_state_closure_observability_v1.py',
    'scripts/gx1_capped_run.sh',
})
NUMBA_VERSION = '0.65.0'
EXECUTION = {
    'class': 'audit', 'runner': 'scripts/gx1_capped_run.sh',
    'memory': '4G', 'swap': '512M', 'timeout_seconds': 1800,
    'kill_after_seconds': 10,
    'module': 'gx1.audit.verify_market_state_m1_duration_v1',
    'temporary_target_tapes': 2, 'published_target_tapes': 0,
    'model_fits': 0,
}
LIMITS = {
    'memory_max': 4 * 1024**3, 'memory_high': 4 * 1024**3,
    'swap': 512 * 1024**2, 'pids': 64, 'numerical_threads': 1,
    'internal_wall_seconds': 1680, 'cold_batch_rows': 8192,
    'max_output_bytes': 1024**2,
}
METRIC_CONTRACT = {
    'scope': 'selected_formula_on_all_591682_observed_TRAIN_M1_rows',
    'factors': [0.5, 1.0], 'gray_middle': 'ACTIVE_side_unknown',
    'side_proof': 'last_W_observed_M1_cold_replay_of_existing_step_direction',
    'required_selected_endpoint_bit_mismatches': 0,
    'unsampled_minute_status': 'new_local_extension_not_unique_selected_tape',
    'state_bits': {'DOWN': 1, 'FLAT': 2, 'UP': 4, 'ACTIVE': 5, 'ALL': 7},
    'run_clock': 'exact_consecutive_observed_M1_validity_minutes',
    'run_duration': 'inclusive_observed_M1_bar_count_not_intraminute_time',
    'censor': 'unknown_label_documented_closure_unknown_source_gap_data_edge',
    'path_confirmed': 'singleton_and_corresponding_existing_path_observed',
    'own_lead_comparison': 'fully_known_bounded_runs_below_or_at_lead_minutes',
    'atr_update': 'exact_adjacent_known_state_changes_with_or_without_closed_TF_ATR_update',
    'decision': 'DESCRIPTIVE_ONLY_NO_TARGET_OR_MODEL_SELECTION',
    'test_data_used': False,
}
TIMEFRAMES = ('M5', 'M15', 'H1', 'H4', 'D1')
LEADS = (5, 15, 60, 240, 1440)
STATE_BITS = {'DOWN': 1, 'FLAT': 2, 'UP': 4}
SINGLETON = (1, 2, 4)
MINUTE_NS = 60_000_000_000
PUBLISHED_BYTES = 0

# Compile the existing audited direction formula, without copying or changing it.
_JIT_STEP = njit(cache=False, fastmath=False)(target._step_direction)


def _need(ok: bool, code: str) -> None:
    if not ok:
        raise RuntimeError('MARKET_STATE_M1_DURATION_' + code)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clean_head() -> str:
    return endpoint._clean_head()


def _policy_gate(policy: dict, plan_ref: dict) -> None:
    gate = policy.get(STAGE, {})
    _need(all(policy.get(key) is False for key in (
        'training_enabled', 'native_training_enabled',
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
        and gate.get('memory_max') == '4G'
        and gate.get('swap_max') == '512M'
        and gate.get('wall_seconds') == 1800, 'POLICY_GATE')


def planned_command(plan_sha256: str) -> list[str]:
    return ['timeout', '--signal=TERM', '--kill-after=10s', '1800s',
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
    endpoint._small_reference_chain()
    _need(numba.__version__ == NUMBA_VERSION
          and plan.get('schema_version') == SCHEMA
          and plan.get('stage') == STAGE
          and plan.get('status') == 'PREPARED_NOT_ADMITTED'
          and plan.get('output_root') == str(OUT_ROOT)
          and plan.get('max_invocations') == 1
          and plan.get('input_refs') == INPUT_REFS
          and plan.get('execution') == EXECUTION
          and plan.get('limits') == LIMITS
          and plan.get('metric_contract') == METRIC_CONTRACT
          and plan.get('numba_version') == NUMBA_VERSION
          and plan.get('expected_m1_rows') == base.M1_ROWS == 591682
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
    owners = plan.get('source_owners')
    _need(isinstance(owners, dict) and set(owners) == SOURCE_PATHS
          and all(_sha(REPO / path) == digest for path, digest in owners.items()),
          'SOURCE_OWNERS')
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


@njit(cache=False, fastmath=False)
def _cold_replay_many(rows: np.ndarray, window: int, open_: np.ndarray,
                      high: np.ndarray, low: np.ndarray, close: np.ndarray,
                      threshold: np.ndarray, unknown_gap: np.ndarray) -> np.ndarray:
    """Existing step formula, reset at the last W observed M1 bars."""
    result = np.empty(len(rows), dtype=np.int8)
    for i in range(len(rows)):
        row = rows[i]
        start = row - window + 1
        predecessor = open_[start] if start == 0 or unknown_gap[start] else close[start - 1]
        direction = -1
        lo = predecessor
        hi = predecessor
        for k in range(start, row + 1):
            if unknown_gap[k] or not (threshold[k] > 0):
                direction = -1
                lo = open_[k]
                hi = open_[k]
            if threshold[k] > 0:
                direction, lo, hi, _ = _JIT_STEP(
                    direction, lo, hi, high[k], low[k], close[k], threshold[k])
        result[i] = direction
    return result


def _cold_in_chunks(rows: np.ndarray, window: int, open_: np.ndarray,
                    high: np.ndarray, low: np.ndarray, close: np.ndarray,
                    threshold: np.ndarray, unknown_gap: np.ndarray,
                    deadline: float) -> np.ndarray:
    out = np.empty(len(rows), dtype=np.int8)
    for start in range(0, len(rows), LIMITS['cold_batch_rows']):
        _need(time.monotonic() + 60 < deadline, 'DEADLINE_RESERVE')
        stop = min(start + LIMITS['cold_batch_rows'], len(rows))
        out[start:stop] = _cold_replay_many(
            rows[start:stop], window, open_, high, low, close, threshold,
            unknown_gap)
    return out


def _boundary(bits: np.ndarray, clock: np.ndarray, gap_kind: np.ndarray,
              at: int, neighbor: int) -> str:
    if neighbor < 0 or neighbor >= len(bits):
        return 'data_edge'
    successor = max(at, neighbor)
    if clock[successor] - clock[successor - 1] != MINUTE_NS:
        return ('documented_closure' if gap_kind[successor] == 1
                else 'unknown_source_gap')
    return ('known_different' if int(bits[neighbor]) in SINGLETON
            else 'unknown_label')


def _quantiles(values: np.ndarray) -> dict:
    if len(values) == 0:
        return {'count': 0, 'min': None, 'p10': None, 'median': None,
                'p90': None, 'max': None}
    points = np.quantile(values.astype(np.float64),
                         (0.1, 0.5, 0.9), method='linear')
    return {'count': int(len(values)), 'min': int(values.min()),
            'p10': float(points[0]), 'median': float(points[1]),
            'p90': float(points[2]), 'max': int(values.max())}


def _run_summary(bits: np.ndarray, clock: np.ndarray,
                 gap_kind: np.ndarray, lead: int) -> dict:
    singleton = np.isin(bits, SINGLETON)
    linked = (np.diff(clock) == MINUTE_NS) & singleton[1:] & singleton[:-1]
    continuation = linked & (bits[1:] == bits[:-1])
    starts = np.flatnonzero(singleton & ~np.r_[False, continuation])
    ends = np.flatnonzero(singleton & ~np.r_[continuation, False])
    _need(len(starts) == len(ends) and np.all(starts <= ends),
          'RUN_AXES')
    records = []
    for start, end in zip(starts, ends):
        left = _boundary(bits, clock, gap_kind, int(start), int(start)-1)
        right = _boundary(bits, clock, gap_kind, int(end), int(end)+1)
        records.append((int(bits[start]), int(end-start+1), left, right))
    reasons = ('known_different', 'unknown_label', 'documented_closure',
               'unknown_source_gap', 'data_edge')
    def describe(rows: list[tuple]) -> dict:
        lengths = np.asarray([r[1] for r in rows], dtype=np.int64)
        bounded = np.asarray([r[1] for r in rows
                              if r[2] == r[3] == 'known_different'], dtype=np.int64)
        return {
            'runs': len(rows), 'certified_m1_rows': int(lengths.sum()),
            'inclusive_observed_minutes': _quantiles(lengths),
            'fully_known_bounded_runs': int(len(bounded)),
            'fully_known_bounded_minutes': _quantiles(bounded),
            'fully_known_bounded_below_own_lead': int((bounded < lead).sum()),
            'fully_known_bounded_at_least_own_lead': int((bounded >= lead).sum()),
            'left_boundary': {k: sum(r[2] == k for r in rows) for k in reasons},
            'right_boundary': {k: sum(r[3] == k for r in rows) for k in reasons},
        }
    by_state = {name: describe([r for r in records if r[0] == bit])
                for name, bit in STATE_BITS.items()}
    _need(sum(x['certified_m1_rows'] for x in by_state.values())
          == int(singleton.sum()), 'RUN_COVERAGE')
    return {'by_state': by_state, 'all_singleton': describe(records)}


def _atr_change_report(bits: np.ndarray, clock: np.ndarray,
                       atr_available_ns: np.ndarray) -> dict:
    adjacent = np.diff(clock) == MINUTE_NS
    singleton = np.isin(bits, SINGLETON)
    comparable = adjacent & singleton[1:] & singleton[:-1]
    changed = comparable & (bits[1:] != bits[:-1])
    atr_changed = atr_available_ns[1:] != atr_available_ns[:-1]
    return {
        'adjacent_comparable_pairs': int(comparable.sum()),
        'state_changes': int(changed.sum()),
        'changes_with_new_closed_TF_ATR': int((changed & atr_changed).sum()),
        'changes_without_new_closed_TF_ATR': int((changed & ~atr_changed).sum()),
        'atr_updates_in_comparable_pairs': int((comparable & atr_changed).sum()),
    }


def summarize_m1_states(bits: np.ndarray, path_confirmed: np.ndarray,
                        clock: np.ndarray, gap_kind: np.ndarray,
                        atr_available_ns: np.ndarray, lead: int) -> dict:
    """Pure total and path-confirmed censored runs, no forecast scoring."""
    _need(bits.dtype == np.uint8 and bits.ndim == 1
          and path_confirmed.dtype == np.bool_
          and clock.dtype == np.int64 and atr_available_ns.dtype == np.int64
          and gap_kind.dtype == np.uint8
          and len(bits) == len(path_confirmed) == len(clock) == len(gap_kind)
              == len(atr_available_ns) > 1
          and np.all(np.diff(clock) > 0)
          and np.isin(bits, (1, 2, 4, 5, 7)).all()
          and np.isin(gap_kind, (0, 1, 2)).all()
          and np.all(gap_kind[1:][np.diff(clock) != MINUTE_NS] > 0)
          and np.all(gap_kind[1:][np.diff(clock) == MINUTE_NS] == 0)
          and np.all(path_confirmed <= np.isin(bits, SINGLETON))
          and type(lead) is int and lead > 0, 'SUMMARY_INPUT')
    path_bits = np.where(path_confirmed, bits, np.uint8(7)).astype(np.uint8)
    counts = {name: int((bits == bit).sum()) for name, bit in
              {**STATE_BITS, 'ACTIVE': 5, 'ALL': 7}.items()}
    _need(sum(counts.values()) == len(bits), 'BIT_PARTITION')
    return {
        'm1_rows': len(bits), 'lead_minutes': lead, 'label_counts': counts,
        'path_confirmed_singleton_rows': int(path_confirmed.sum()),
        'label_only_runs': _run_summary(bits, clock, gap_kind, lead),
        'path_confirmed_runs': _run_summary(path_bits, clock, gap_kind, lead),
        'atr_update_coincident_state_changes': _atr_change_report(
            bits, clock, atr_available_ns),
        'continuous_between_M1_bars_or_intraminute_order_proven': False,
    }


def _temporary_tapes(full_plan: dict, deadline: float):
    schedule, authority = base.read_closure(full_plan)
    m1, m5, issues, m5_source_rows, stats = base.load_sources(full_plan, authority)
    _need(time.monotonic() + 60 < deadline, 'DEADLINE_AFTER_SOURCES')
    cutoff = base.pd_ns(base.TRAIN_END)
    _need(target.MINIMUM_MOVE_ATR == 0.5
          and tuple(target.TIMEFRAMES) == TIMEFRAMES
          and tuple(base.LEADS) == LEADS, 'TARGET_RECIPE')
    atr = target.build_causal_tf_atr(m5, cutoff_ns=cutoff)
    a05_series = target.build_dynamic_mtf_state_series(
        m1, tf_atr=atr, source_ref=full_plan['m1']['source'],
        atr_source_ref=full_plan['m5']['source'], cutoff_ns=cutoff,
        source_row_offset=5193205, closure_authority=authority,
        exact_schedule=schedule)
    target.MINIMUM_MOVE_ATR = 1.0
    try:
        atr10 = {**atr, 'definition_sha256':
                 target.dynamic_mtf_state_target_contract()['contract_sha256']}
        a10_series = target.build_dynamic_mtf_state_series(
            m1, tf_atr=atr10, source_ref=full_plan['m1']['source'],
            atr_source_ref=full_plan['m5']['source'], cutoff_ns=cutoff,
            source_row_offset=5193205, closure_authority=authority,
            exact_schedule=schedule)
    finally:
        target.MINIMUM_MOVE_ATR = 0.5
    _need(time.monotonic() + 60 < deadline, 'DEADLINE_AFTER_TAPES')
    a05, a10 = a05_series.arrays, a10_series.arrays
    axis = a05['validity_ns']
    _need(len(axis) == base.M1_ROWS and np.array_equal(axis, a10['validity_ns'])
          and np.array_equal(a05['source_row'],
                             np.arange(5193205, 5193205 + base.M1_ROWS))
          and np.array_equal(issues, np.asarray(issues, np.int64)),
          'TAPE_AXES')
    times = target._clock(m1, target.EXIT_DECISION_BAR_SECONDS)
    known, _ = target._known_closures(
        times, full_plan['m1']['source'], authority, schedule)
    gap_kind = np.zeros(len(axis), np.uint8)
    gap_rows = np.flatnonzero(np.diff(axis) != MINUTE_NS) + 1
    for row in gap_rows:
        gap_kind[row] = 1 if int(row) in known else 2
    _need(int((gap_kind == 1).sum()) == authority['known_market_closure_count']
          and int((gap_kind == 2).sum()) == authority['unknown_source_gap_count'],
          'GAP_AUTHORITY')
    return (m1, issues, m5_source_rows, stats, a05, a10, gap_kind, authority)


def _selected_full_mask() -> dict:
    with np.load(io.BytesIO(binding._checked_bytes('mask', 4 * 1024**2)),
                 allow_pickle=False) as saved:
        _need(set(saved.files) == binding.MASK_KEYS, 'SELECTED_MEMBERS')
        return {name: np.array(saved[name], copy=True) for name in
                ('issue_ns', 'm5_source_row', 'lead_minutes', 'm1_source_row',
                 'allowed_set')}


def _classify_all_m1(m1, a05: dict, a10: dict, gap_kind: np.ndarray,
                     deadline: float) -> tuple[np.ndarray, np.ndarray, dict]:
    n = len(a05['validity_ns'])
    rows = np.arange(n, dtype=np.int64)
    values = target._ohlc(m1)
    open_, high, low, close = (np.ascontiguousarray(values[k], dtype=np.float64)
                               for k in ('open', 'high', 'low', 'close'))
    unknown_gap = gap_kind == 2
    bits = np.full((n, len(TIMEFRAMES)), 7, np.uint8)
    path = np.zeros_like(bits, dtype=bool)
    cold_report = {}
    blocked = target.ATR_UNAVAILABLE | target.LOCAL_SPAN_INCOMPLETE | target.UNKNOWN_SOURCE_GAP
    for col, tf in enumerate(TIMEFRAMES):
        _need(time.monotonic() + 60 < deadline, 'DEADLINE_TF:' + tf)
        window = target.LOCAL_WINDOW_M1_BARS[tf]
        theta05 = a05['threshold'][:, col]
        theta10 = a10['threshold'][:, col]
        early, _ = early_active_proof(
            high, low, theta10, unknown_gap, window, np)
        allowed, _ = classify.classify_rows(
            a05, a10, early, rows, col, target, np)
        complete = ((a10['reason'][:, col] & blocked) == 0) & (theta10 > 0)
        middle = complete & (a10['local_range'][:, col] >= theta05) & (
            a10['local_range'][:, col] < theta10)
        _need(np.all(allowed[middle] == 7), 'MIDDLE_BEFORE_PROMOTION:' + tf)
        allowed[middle] = 5
        candidates = np.flatnonzero(np.isin(allowed, (1, 4))).astype(np.int64)
        _need(np.all(candidates >= window - 1), 'COLD_WINDOW_SUPPORT:' + tf)
        local = _cold_in_chunks(
            candidates, window, open_, high, low, close, theta10,
            unknown_gap, deadline)
        old_side = np.where(allowed[candidates] == 1,
                            target.DOWN, target.UP)
        _need(np.isin(local, (-1, target.DOWN, target.UP)).all(),
              'LOCAL_SIDE_DOMAIN:' + tf)
        demote = local != old_side
        allowed[candidates[demote]] = 5
        bits[:, col] = allowed
        path[:, col] = np.where(
            allowed == 2, a05['path_observed'][:, col],
            a10['path_observed'][:, col]) & np.isin(allowed, SINGLETON)
        cold_report[tf] = {
            'candidate_certified_side_m1_rows': int(len(candidates)),
            'local_same': int((~demote).sum()),
            'local_unknown': int((local == -1).sum()),
            'local_opposite': int((demote & (local != -1)).sum()),
            'gray_promoted_active_m1_rows': int(middle.sum()),
        }
        _need(cold_report[tf]['candidate_certified_side_m1_rows']
              == cold_report[tf]['local_same']
              + cold_report[tf]['local_unknown']
              + cold_report[tf]['local_opposite'], 'COLD_PARTITION:' + tf)
    return bits, path, cold_report


def _selected_endpoint_parity(selected: dict, issues: np.ndarray,
                              m5_source_rows: np.ndarray, axis: np.ndarray,
                              source_rows: np.ndarray, m1_bits: np.ndarray) -> dict:
    _need(len(issues) == len(selected['issue_ns']) == 118706
          and np.array_equal(issues, selected['issue_ns'])
          and np.array_equal(m5_source_rows, selected['m5_source_row'])
          and np.array_equal(selected['lead_minutes'],
                             np.asarray(LEADS, np.int16)), 'ISSUE_IDENTITY')
    report = {}
    cutoff = base.pd_ns(base.TRAIN_END)
    for col, tf in enumerate(TIMEFRAMES):
        at, present = classify.exact_positions(
            axis, issues, LEADS[col], cutoff, np)
        got = np.full(len(issues), 7, np.uint8)
        source = np.full(len(issues), -1, np.int64)
        got[present] = m1_bits[at[present], col]
        source[present] = source_rows[at[present]]
        mismatches = int(np.count_nonzero(got != selected['allowed_set'][:, col]))
        source_mismatch = int(np.count_nonzero(
            source != selected['m1_source_row'][:, col]))
        _need(mismatches == source_mismatch == 0,
              'SELECTED_ENDPOINT_PARITY:' + tf + ':' + str(mismatches)
              + ':' + str(source_mismatch))
        report[tf] = {'issues': len(issues), 'exact_m1_endpoint': int(present.sum()),
                      'bit_mismatches': mismatches,
                      'm1_source_row_mismatches': source_mismatch}
    return report


def _source_bookend(plan: dict, plan_sha256: str, policy_sha256: str,
                    head: str, stats: dict) -> None:
    _need(_clean_head() == head and _sha(PLAN_PATH) == plan_sha256
          and _sha(POLICY) == policy_sha256
          and all(_sha(REPO / path) == digest
                  for path, digest in plan['source_owners'].items()),
          'SOURCE_BOOKEND')
    for name, item in INPUT_REFS.items():
        _need(_sha(Path(item['path'])) == item['sha256'], 'INPUT_BOOKEND:' + name)
    for path, original in stats.items():
        _need(base.regular(Path(path)) == original, 'RAW_SOURCE_CHANGED:' + path)


def _write(root: Path, name: str, value: dict) -> dict:
    global PUBLISHED_BYTES
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    _need(name in ('FAILED.json', 'TERMINAL.json')
          or PUBLISHED_BYTES + len(data) <= LIMITS['max_output_bytes'],
          'OUTPUT_BUDGET')
    _publish_bytes(root / name, data)
    PUBLISHED_BYTES += len(data)
    return {'path': str(root / name), 'sha256': hashlib.sha256(data).hexdigest()}


def _wall_limit(_signum, _frame) -> None:
    raise TimeoutError('MARKET_STATE_M1_DURATION_INTERNAL_WALL_LIMIT')


def run(plan_path: Path, plan_sha256: str, out_dir: Path) -> dict:
    global PUBLISHED_BYTES
    started = time.monotonic()
    plan, plan_ref, head, policy_ref = preflight(
        plan_path, plan_sha256, out_dir, admitted=True)
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
            'schema_version': 'gx1_market_state_m1_duration_claim_v1',
            'invocation': 1, 'plan': plan_ref, 'policy': policy_ref,
            'source_head': head, 'read_only': True,
            'temporary_target_tapes': 2, 'published_target_tapes': 0,
            'model_fits': 0, 'test_data_used': False})
        full_plan = endpoint._checked_small_json(INPUT_REFS['target_source_plan'])
        m1, issues, m5_rows, stats, a05, a10, gap_kind, authority = (
            _temporary_tapes(full_plan, deadline))
        selected = _selected_full_mask()
        bits, path, cold_report = _classify_all_m1(
            m1, a05, a10, gap_kind, deadline)
        parity = _selected_endpoint_parity(
            selected, issues, m5_rows, a05['validity_ns'],
            a05['source_row'], bits)
        metrics = {
            tf: summarize_m1_states(
                bits[:, col], path[:, col], a05['validity_ns'], gap_kind,
                a10['atr_available_ns'][:, col], LEADS[col])
            for col, tf in enumerate(TIMEFRAMES)
        }
        _need(all(metrics[tf]['m1_rows'] == base.M1_ROWS
                  for tf in TIMEFRAMES), 'METRIC_ROWS')
        _source_bookend(plan, plan_sha256, policy_ref['sha256'], head, stats)
        result = _write(out_dir, 'RESULT.json', {
            'schema_version': 'gx1_market_state_m1_duration_result_v1',
            'status': 'COMPLETE_DESCRIPTIVE_SELECTED_ENDPOINT_PARITY',
            'plan': plan_ref, 'policy': policy_ref, 'claim': claim,
            'source_head': head, 'elapsed_seconds': time.monotonic() - started,
            'input_refs': INPUT_REFS, 'metric_contract': METRIC_CONTRACT,
            'm1_rows': base.M1_ROWS, 'm5_issues': len(issues),
            'documented_closure_gaps': int((gap_kind == 1).sum()),
            'unknown_source_gaps': int((gap_kind == 2).sum()),
            'source_clock_sha256': authority['m1_clock_sha256'],
            'selected_endpoint_parity': parity,
            'cold_local_side': cold_report, 'metrics': metrics,
            'read_only': True, 'temporary_target_tapes': 2,
            'published_target_tapes': 0, 'model_fits': 0,
            'test_data_used': False,
            'target_selection': False,
            'entry_exit_or_trading_authority': False,
            'limit': 'This is a local extension of the selected endpoint rule: endpoint parity does not prove unique labels on unsampled M1 minutes. M1 bars are observed samples; intraminute transitions and economic value are unproven.'})
        signal.setitimer(signal.ITIMER_REAL, 0)
        terminal = _write(out_dir, 'TERMINAL.json', {
            'schema_version': 'gx1_market_state_m1_duration_terminal_v1',
            'status': 'COMPLETE', 'exit_code': 0, 'plan': plan_ref,
            'claim': claim, 'result': result,
            'no_relaunch_under_this_plan': True, 'test_data_used': False})
        return {'result': result, 'terminal': terminal}
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL, 0)
        failed = _write(out_dir, 'FAILED.json', {
            'schema_version': 'gx1_market_state_m1_duration_failure_v1',
            'plan': plan_ref, 'claim': claim, 'source_head': head,
            'error_type': type(exc).__name__, 'error': str(exc)[:2048],
            'elapsed_seconds': time.monotonic() - started,
            'no_relaunch_under_this_plan': True})
        _write(out_dir, 'TERMINAL.json', {
            'schema_version': 'gx1_market_state_m1_duration_terminal_v1',
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
        plan, plan_ref, head, _ = preflight(
            args.plan, args.plan_sha256, args.out_dir)
        print(json.dumps({
            'status': 'READY_NO_CLAIM_NO_PAYLOAD', 'plan': plan_ref,
            'source_head': head, 'm1_rows': plan['expected_m1_rows'],
            'issues': plan['expected_issue_rows'],
            'planned_command': planned_command(args.plan_sha256),
            'claim_created': False, 'payload_read': False,
            'test_data_used': False}, sort_keys=True))
        return 0
    print(json.dumps(run(args.plan, args.plan_sha256, args.out_dir),
                     sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
