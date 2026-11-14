"""One-use six-lane anchor-span extension for five development folds.

Reuse the immutable fold-2 span sidecars, replay only later source rows with
all 31 old trendline fields checked bitwise, and publish complete sidecars.
No target labels, normalization fit, model, sealed TEST or trading path.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import signal
import subprocess
import time

import numpy as np

from gx1.audit import qualify_market_state_ta_anchor_span_inputs_v1 as replay
from gx1.contracts.gx1_capped_execution_v1 import (
    require_capped_cpu_audit_execution, require_capped_lock_ancestry,
)
from gx1.contracts.immutable_event_authority_v1 import _fsync_directory, _publish_bytes

REPO = Path('/home/andre2/src/GX1_CURRENT')
RUNS = Path('/home/andre2/GX1_RUNS')
STAGE = 'market_state_ta_fivefold_span_extension_20261114_001'
PLAN = REPO / 'configs/research' / (STAGE.upper() + '.json')
ROOT = RUNS / STAGE.upper()
POLICY = REPO / 'NEXT_RUN_POLICY.json'
LANES = ('local', 'M5', 'M15', 'H1', 'H4', 'D1')
KEYS = frozenset(('row_indices', 'times_ns', 'native_bar_positions', 'anchor_span_bars'))
CUTOFF_NS = 1767225600000000000
OLD_PLAN = REPO / 'configs/research/MARKET_STATE_TA_TRENDLINE_ANCHOR_SPAN_INPUTS_20261112_001.json'
FOLD2_RESULT = RUNS / 'MARKET_STATE_TA_FOLD2_SPAN_REUSE_20261113_002/RESULT.json'
COVERAGE = REPO / 'docs/TA_FIVEFOLD_ANCHOR_SPAN_COVERAGE_20261114.json'
SOURCE_PATHS = frozenset({
    'gx1/audit/qualify_market_state_ta_fivefold_span_extension_v1.py',
    'gx1/audit/qualify_market_state_ta_anchor_span_inputs_v1.py',
    'gx1/scripts/audit_direction_d1_trendline_replay_v1.py',
    'gx1/features/trendline_registry_v1.py',
    'gx1/features/technical_indicators_v1.py',
    'gx1/features/htf_features.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'scripts/gx1_capped_run.sh',
})
MEMORY_MAX = 4 * 1024**3
SWAP_MAX = 512 * 1024**2
WALL_SECONDS = 3600
INTERNAL_SECONDS = 3480
SCOPE = {'new_rows_total': 173854, 'merged_rows_total': 290522,
         'producer_calls': 6, 'model_constructions': 0, 'model_forwards': 0,
         'normalization_fits': 0, 'optimizer_steps': 0,
         'target_labels_used': False, 'test_data_used': False}


def need(condition: bool, code: str) -> None:
    if not condition:
        raise RuntimeError('FIVEFOLD_SPAN_EXTENSION_' + code)


def sha(path: Path) -> str:
    return replay.sha(path)


def ref(path: Path) -> dict:
    need(path.is_absolute() and path.is_file() and not path.is_symlink(),
         'REF_PATH:' + str(path))
    return {'path': str(path), 'sha256': sha(path)}


def checked(item: dict, *, limit: int = 64 * 1024**2) -> dict:
    path = Path(item['path'])
    need(set(item) == {'path', 'sha256'} and path.is_absolute()
         and path.is_file() and not path.is_symlink()
         and 0 < path.stat().st_size <= limit and ref(path) == item,
         'REF_HASH:' + str(path))
    return json.loads(path.read_text())


def publish(name: str, value: dict) -> dict:
    need('/' not in name and name not in ('', '.', '..'), 'OUTPUT_NAME')
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    need(len(payload) <= 4 * 1024**2, 'JSON_OUTPUT_BUDGET')
    _publish_bytes(ROOT / name, payload)
    return ref(ROOT / name)


def planned_command(plan_sha: str) -> list[str]:
    return ['timeout', '--signal=TERM', '--kill-after=10s', '3600s',
            'scripts/gx1_capped_run.sh', '--class', 'audit', '--mem', '4G',
            '--swap', '512M', '--', '.venv/bin/python', '-m',
            'gx1.audit.qualify_market_state_ta_fivefold_span_extension_v1',
            '--plan', str(PLAN), '--plan-sha256', plan_sha]


def clean_head() -> str:
    need(Path.cwd().resolve() == REPO, 'WORKDIR')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    branch = subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], text=True)
    need(branch == 'work/gx1-current' and not dirty and len(head) == 40,
         'CLEAN_SOURCE')
    return head


def archive(item: dict, *, expected_range: list[int]) -> dict[str, np.ndarray]:
    path = Path(item['path'])
    need(ref(path) == item, 'SIDECAR_HASH:' + str(path))
    with np.load(path, allow_pickle=False) as data:
        need(set(data.files) == KEYS, 'SIDECAR_KEYS')
        block = {key: data[key].copy() for key in KEYS}
    rows = block['row_indices']
    n = expected_range[1] - expected_range[0]
    need(rows.dtype == np.int64 and rows.shape == (n,)
         and np.array_equal(rows, np.arange(*expected_range, dtype=np.int64)),
         'SIDECAR_ROWS')
    need(all(block[key].dtype == np.int64 and block[key].shape == (n,)
             for key in ('times_ns', 'native_bar_positions')),
         'SIDECAR_CLOCKS')
    spans = block['anchor_span_bars']
    need(spans.dtype == np.float32 and spans.shape == (n, 2)
         and np.isfinite(spans).all() and np.all(spans >= 0)
         and np.all(spans == np.floor(spans)), 'SIDECAR_SPANS')
    return block


def merge(old: dict, new: dict, *, lane: str, scope: dict) -> dict:
    old_range, new_range = scope['old_range'], scope['new_range']
    full_range = scope['required_superrange']
    need(old_range[0] == full_range[0]
         and old_range[1] == new_range[0]
         and new_range[1] == full_range[1], 'ADJACENT_RANGES:' + lane)
    out = {key: np.concatenate((old[key], new[key]), axis=0) for key in KEYS}
    rows, times = out['row_indices'], out['times_ns']
    population = scope['full_population']
    available = times + replay.BAR_NS['M5' if lane == 'local' else lane] * 10**9
    need(np.array_equal(rows, np.arange(*full_range, dtype=np.int64))
         and len(rows) == population['selected_unique_row_count']
         and replay.row_sha(rows, lane) == population['selected_row_indices_sha256']
         and replay.axis_sha(available) == population['available_ns_sha256']
         and np.all(np.diff(times) > 0) and np.all(available < CUTOFF_NS),
         'FULL_POPULATION:' + lane)
    return out


def preflight(plan_path: Path, plan_sha: str, *, admitted: bool = False) -> tuple:
    head = clean_head()
    need(plan_path == PLAN and len(plan_sha) == 64 and sha(PLAN) == plan_sha
         and not ROOT.exists(), 'PLAN_IDENTITY')
    plan = json.loads(PLAN.read_text())
    coverage = checked(plan['coverage'])
    fold2 = checked(plan['fold2_result'])
    old = checked(plan['old_plan'])
    need(plan['schema_version'] == 'gx1_ta_fivefold_span_extension_plan_v1'
         and plan['stage'] == STAGE and plan['status'] == 'PREPARED_NOT_ADMITTED'
         and plan['output_root'] == str(ROOT)
         and plan['command_template'] == planned_command('{plan_sha256}')
         and plan['cutoff_ns'] == CUTOFF_NS
         and plan['scope'] == SCOPE
         and plan['max_invocations'] == 1 and plan['test_data_used'] is False
         and fold2['status'] == 'COMPLETE' and fold2['source_unchanged'] is True
         and fold2['test_data_used'] is False
         and coverage['new_rows_total'] == SCOPE['new_rows_total']
         and coverage['fit_proof_checks_total'] == 30
         and coverage['score_end_exclusive_utc'] == '2026-01-01T00:00:00+00:00'
         and old['scope']['ranges']['local'][0] == coverage['lanes']['local']['old_range'][0]
         and plan['lanes'] == list(LANES)
         and plan['lane_scopes'] == coverage['lanes']
         and plan['source_base_head'] and len(plan['source_base_head']) == 40
         and subprocess.run(['git', 'merge-base', '--is-ancestor',
                             plan['source_base_head'], head],
                            capture_output=True).returncode == 0,
         'PLAN_SCOPE')
    owners = plan['source_owners']
    need(set(owners) == SOURCE_PATHS
         and all(sha(REPO / name) == digest for name, digest in owners.items()),
         'SOURCE_OWNERS')
    need(Path(__file__).resolve() == REPO /
         'gx1/audit/qualify_market_state_ta_fivefold_span_extension_v1.py',
         'IMPORT_PATH')
    for lane in LANES:
        item = coverage['lanes'][lane]
        need(item['old_sidecar'] == fold2['surfaces'][lane]['artifact']
             and item['old_range'][1] == item['new_range'][0]
             and item['new_rows'] == item['new_range'][1] - item['new_range'][0]
             and item['full_population']['selected_unique_row_count']
                 == item['required_superrange'][1] - item['required_superrange'][0]
             and all(check['row_hash_match'] is True for check in item['fit_proof_checks']),
             'LANE_SCOPE:' + lane)
        archive(item['old_sidecar'], expected_range=item['old_range'])
    policy_ref = None
    if admitted:
        cap = require_capped_cpu_audit_execution()
        require_capped_lock_ancestry()
        need(cap['memory_max'] == cap['memory_high'] == MEMORY_MAX
             and cap['swap'] == SWAP_MAX, 'CAP_LIMITS')
        policy_ref = ref(POLICY)
        policy = checked(policy_ref)
        gate = policy.get(STAGE, {})
        need(all(policy[k] is False for k in ('training_enabled',
             'native_training_enabled', 'full_epoch_training_allowed',
             'full_val_allowed', 'native_launch_authorized'))
             and gate.get('status') == gate.get('decision') == 'ADMITTED'
             and gate.get('plan') == {'path': str(PLAN), 'sha256': plan_sha}
             and gate.get('output_root') == str(ROOT)
             and gate.get('execution_authorized') is True
             and gate.get('data_reads_authorized') is True
             and gate.get('training_authorized') is False
             and gate.get('max_invocations') == 1
             and gate.get('claimed_invocations') == gate.get('completed_invocations') == 0
             and gate.get('relaunch_allowed') is False
             and gate.get('memory_max') == '4G'
             and gate.get('swap_max') == '512M'
             and gate.get('wall_seconds') == WALL_SECONDS
             and gate.get('test_data_used') is False
             and gate.get('broker_access_allowed') is False,
             'POLICY_GATE')
    return plan, coverage, fold2, old, head, policy_ref


def bookend(plan: dict, plan_sha: str, policy_ref: dict, head: str) -> None:
    need(clean_head() == head and sha(PLAN) == plan_sha
         and ref(POLICY) == policy_ref
         and all(sha(REPO / name) == digest
                 for name, digest in plan['source_owners'].items())
         and all(ref(Path(plan[key]['path'])) == plan[key]
                 for key in ('coverage', 'fold2_result', 'old_plan')),
         'SOURCE_BOOKEND')
    for lane in LANES:
        item = plan['lane_scopes'][lane]['old_sidecar']
        need(ref(Path(item['path'])) == item, 'OLD_SIDECAR_BOOKEND:' + lane)


def wall_limit(_signum, _frame) -> None:
    raise TimeoutError('FIVEFOLD_SPAN_EXTENSION_INTERNAL_WALL_LIMIT')


def run(plan_path: Path, plan_sha: str) -> dict:
    started = time.monotonic()
    plan, coverage, _, old, head, policy_ref = preflight(
        plan_path, plan_sha, admitted=True)
    need(policy_ref is not None, 'ADMISSION_REQUIRED')
    ROOT.mkdir(parents=True, exist_ok=False)
    _fsync_directory(ROOT.parent)
    claim = None
    completed = {}
    old_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, wall_limit)
        signal.setitimer(signal.ITIMER_REAL, INTERNAL_SECONDS)
        claim = publish('CLAIM.json', {
            'schema_version': 'gx1_ta_fivefold_span_extension_claim_v1',
            'plan': {'path': str(PLAN), 'sha256': plan_sha},
            'policy': policy_ref, 'source_head': head,
            'invocation': 1, 'test_data_used': False})
        delta_dir = ROOT / 'DELTA'
        delta_dir.mkdir()
        wrapper = copy.deepcopy(old)
        wrapper['populations'] = {lane: coverage['lanes'][lane]['new_population']
                                  for lane in LANES}
        wrapper['scope'] = {'ranges': {lane: coverage['lanes'][lane]['new_range']
                                      for lane in LANES},
                            'history_end_exclusive_ns': CUTOFF_NS,
                            'replay_lanes': list(LANES),
                            'array_output_budget_bytes': 4 * 1024**2}
        wrapper['output_root'] = str(delta_dir)
        delta = replay.measure(wrapper, ranges=wrapper['scope']['ranges'],
                               cutoff=CUTOFF_NS, output_root=delta_dir,
                               lanes=LANES)
        need(set(delta) == set(LANES), 'SIX_REPLAYS')
        for lane in LANES:
            scope = coverage['lanes'][lane]
            item = delta[lane]
            need(item['population'] == scope['new_population']
                 and item['parity']['float32_bit_mismatched_cells'] == 0
                 and item['parity']['fields'] == 31
                 and item['fresh_state_at_source_row_zero'] is True,
                 'NEW_PARITY:' + lane)
            first = archive(scope['old_sidecar'], expected_range=scope['old_range'])
            second = archive(item['artifact'], expected_range=scope['new_range'])
            merged = merge(first, second, lane=lane, scope=scope)
            buffer = io.BytesIO()
            np.savez(buffer, **merged)
            need(0 < buffer.tell() <= 8 * 1024**2,
                 'MERGED_ARRAY_BUDGET:' + lane)
            path = ROOT / (lane + '_ANCHOR_SPANS.npz')
            _publish_bytes(path, buffer.getvalue())
            completed[lane] = {'delta': item, 'full_artifact': ref(path),
                               'full_rows': len(merged['row_indices']),
                               'old_rows_reused': len(first['row_indices']),
                               'new_rows': len(second['row_indices']),
                               'old_values_bitexact': True}
            del first, second, merged, buffer
        need(sum(x['new_rows'] for x in completed.values()) == SCOPE['new_rows_total']
             and sum(x['full_rows'] for x in completed.values()) == SCOPE['merged_rows_total'],
             'TOTAL_COUNTS')
        bookend(plan, plan_sha, policy_ref, head)
        need(time.monotonic() - started < INTERNAL_SECONDS, 'WALL_BOOKEND')
        result = publish('RESULT.json', {
            'schema_version': 'gx1_ta_fivefold_span_extension_result_v1',
            'status': 'COMPLETE_SIX_LANES_INPUT_ONLY',
            'plan': {'path': str(PLAN), 'sha256': plan_sha},
            'policy': policy_ref, 'claim': claim,
            'source_head': head, 'source_unchanged': True,
            'coverage': plan['coverage'], 'lanes': completed,
            'counts': SCOPE, 'elapsed_seconds': time.monotonic() - started,
            'normalization_fitted': False, 'model_fit': False,
            'test_data_used': False})
        signal.setitimer(signal.ITIMER_REAL, 0)
        terminal = publish('TERMINAL.json', {
            'schema_version': 'gx1_ta_fivefold_span_extension_terminal_v1',
            'status': 'COMPLETE', 'exit_code': 0,
            'plan': {'path': str(PLAN), 'sha256': plan_sha},
            'claim': claim, 'result': result, 'no_relaunch': True,
            'test_data_used': False})
        return {'result': result, 'terminal': terminal}
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL, 0)
        failed = publish('FAILED.json', {
            'schema_version': 'gx1_ta_fivefold_span_extension_failure_v1',
            'plan': {'path': str(PLAN), 'sha256': plan_sha},
            'claim': claim, 'completed_lanes': completed,
            'error_type': type(exc).__name__, 'error': str(exc)[:2048],
            'elapsed_seconds': time.monotonic() - started,
            'no_relaunch': True, 'test_data_used': False})
        publish('TERMINAL.json', {
            'schema_version': 'gx1_ta_fivefold_span_extension_terminal_v1',
            'status': 'FAILED_CONSUMED', 'exit_code': 1,
            'plan': {'path': str(PLAN), 'sha256': plan_sha},
            'claim': claim, 'failure': failed,
            'completed_lanes': len(completed), 'no_relaunch': True,
            'test_data_used': False})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.preflight:
        plan, coverage, _, _, head, _ = preflight(args.plan, args.plan_sha256)
        # Exercise merge acceptance and rejection without physical payload reads.
        rows = np.arange(2, dtype=np.int64)
        left = {'row_indices': rows[:1], 'times_ns': np.array([10], np.int64),
                'native_bar_positions': np.array([1], np.int64),
                'anchor_span_bars': np.array([[0., 1.]], np.float32)}
        right = {'row_indices': rows[1:], 'times_ns': np.array([20], np.int64),
                 'native_bar_positions': np.array([2], np.int64),
                 'anchor_span_bars': np.array([[2., 3.]], np.float32)}
        full = {'old_range': [0, 1], 'new_range': [1, 2],
                'required_superrange': [0, 2],
                'full_population': {'selected_unique_row_count': 2,
                                    'selected_row_indices_sha256': replay.row_sha(rows, 'local'),
                                    'available_ns_sha256': replay.axis_sha(np.array([10, 20])
                                        + replay.BAR_NS['M5'] * 10**9)}}
        # Use actual nanosecond times so the availability check remains causal.
        left['times_ns'] += 1_000_000_000_000
        right['times_ns'] += 1_000_000_000_000
        full['full_population']['available_ns_sha256'] = replay.axis_sha(
            np.array([left['times_ns'][0], right['times_ns'][0]], np.int64)
            + replay.BAR_NS['M5'] * 10**9)
        need(len(merge(left, right, lane='local', scope=full)['row_indices']) == 2,
             'SYNTHETIC_MERGE')
        broken = copy.deepcopy(full)
        broken['new_range'] = [2, 3]
        try:
            merge(left, right, lane='local', scope=broken)
        except RuntimeError as exc:
            need('ADJACENT_RANGES' in str(exc), 'SYNTHETIC_FAILURE')
        else:
            raise RuntimeError('FIVEFOLD_SPAN_EXTENSION_SYNTHETIC_FAILURE_MISSED')
        print(json.dumps({'status': 'READY_NO_CLAIM_NO_PAYLOAD',
                          'source_head': head, 'plan': {'path': str(PLAN),
                              'sha256': args.plan_sha256},
                          'new_rows': coverage['new_rows_total'],
                          'planned_command': planned_command(args.plan_sha256),
                          'test_data_used': False}, sort_keys=True))
        return 0
    print(json.dumps(run(args.plan, args.plan_sha256), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
