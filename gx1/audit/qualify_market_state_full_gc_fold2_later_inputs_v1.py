"""One capped, read-only full-GC gather of the fixed fold2 later panel.

Reuses six qualified later anchor-span sidecars and the fold2 TRAIN packet.
No target arrays, model construction, fit, TEST, broker or trading.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import signal
import subprocess
import time

import numpy as np

from gx1.audit import qualify_market_state_mature_prefix_normalization_v1 as mature
from gx1.audit import verify_market_state_native_F_cpu_zero_step_v1 as base
from gx1.contracts import entry_native_direction_feature_view_v1 as views

REPO, RUNS, POLICY = base.REPO, base.RUNS, base.POLICY
STAGE = 'market_state_full_gc_fold2_later_inputs_20261115_001'
PLAN_PATH = REPO / 'configs/research/MARKET_STATE_FULL_GC_FOLD2_LATER_INPUTS_20261115_001.json'
OUT_ROOT = RUNS / 'MARKET_STATE_FULL_GC_FOLD2_LATER_INPUTS_20261115_001'
LATER_ROOT = RUNS / 'MARKET_STATE_TA_FOLD2_LATER_INPUTS_20261113_002'
WORKER_PATH = 'gx1/audit/qualify_market_state_full_gc_fold2_later_inputs_v1.py'
SOURCE_PATHS = tuple(sorted(set(base.SOURCE_PATHS) | {WORKER_PATH}))
MEMORY_BYTES = 8 * 1024**3
SWAP_BYTES = 512 * 1024**2
WALL_SECONDS = 900
SCORE_ROWS = 4096
BATCH_ROWS = 16
BATCHES = 256
LANES = ('local', 'M5', 'M15', 'H1', 'H4', 'D1')


def need(ok: object, reason: str) -> None:
    if not bool(ok):
        raise RuntimeError('FULL_GC_FOLD2_LATER_' + reason)


def planned_command(plan_sha256: str) -> list[str]:
    return ['timeout', '--signal=TERM', '--kill-after=10s', '900s',
            'scripts/gx1_capped_run.sh', '--class', 'producer', '--mem', '8G',
            '--swap', '512M', '--', '.venv/bin/python', '-m',
            'gx1.audit.qualify_market_state_full_gc_fold2_later_inputs_v1',
            '--plan', str(PLAN_PATH), '--plan-sha256', plan_sha256]


def ref(path: Path) -> dict:
    return base.ref(path)


def checked(item: dict) -> dict:
    return base.checked_json(item)


def inspect(plan_ref: dict, *, require_policy: bool) -> dict:
    from gx1.contracts.entry_anchor_span_feature_inputs_v1 import later_panel
    head, _ = base.current_state()
    need(plan_ref == ref(PLAN_PATH) and not OUT_ROOT.exists(), 'PLAN_OR_OUTPUT')
    plan = json.loads(PLAN_PATH.read_text())
    expected_scope = {'fold_id': 2, 'partition': 'later_development',
                      'selected_parent_rows': SCORE_ROWS, 'batch_rows': BATCH_ROWS,
                      'feature_batches': BATCHES, 'target_arrays_decoded': 0,
                      'model_constructions': 0, 'model_forwards': 0,
                      'optimizer_steps': 0, 'test_data_used': False}
    need(set(plan) == {'schema_version', 'stage', 'status', 'source_base_head',
                       'source_owners', 'inputs', 'scope', 'limits',
                       'output_root', 'command_template', 'expected_view_sha256'}
         and plan['schema_version'] == 'gx1_market_state_full_gc_fold2_later_inputs_plan_v1'
         and plan['stage'] == STAGE and plan['status'] == 'PREPARED_NOT_ADMITTED'
         and plan['scope'] == expected_scope
         and plan['limits'] == {'memory_max': MEMORY_BYTES,
                                'memory_high': MEMORY_BYTES,
                                'swap': SWAP_BYTES, 'wall_seconds': WALL_SECONDS}
         and plan['output_root'] == str(OUT_ROOT)
         and plan['command_template'] == planned_command('{plan_sha256}')
         and subprocess.run(('git', 'merge-base', '--is-ancestor',
                             plan['source_base_head'], head), cwd=REPO,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL).returncode == 0,
         'PLAN_SCOPE')
    owners = plan['source_owners']
    need(set(owners) == set(SOURCE_PATHS)
         and all(base.sha(REPO / name) == digest for name, digest in owners.items()),
         'SOURCE_OWNERS')
    inp = plan['inputs']
    expected_refs = {
        'train_view_plan': ref(base.FOLD2_VIEW_PLAN),
        'train_view_result': ref(base.FOLD2_ROOT / 'RESULT.json'),
        'train_view_validation': ref(base.FOLD2_ROOT / 'VALIDATION.json'),
        'later_plan': ref(LATER_ROOT / 'PLAN.json'),
        'later_result': ref(LATER_ROOT / 'RESULT.json'),
        'later_terminal': ref(LATER_ROOT / 'TERMINAL.json'),
        'later_coverage': ref(LATER_ROOT / 'COVERAGE.json'),
        'later_gather': ref(LATER_ROOT / 'GATHER_RESULT.json'),
        'later_validation': ref(LATER_ROOT / 'VALIDATION.json'),
        'initial_result': ref(base.OUT_ROOT / 'RESULT.json'),
    }
    need(inp == expected_refs, 'INPUT_REFS')
    train_plan = checked(inp['train_view_plan'])
    train_result = checked(inp['train_view_result'])
    train_validation = checked(inp['train_view_validation'])
    later_plan = checked(inp['later_plan'])
    prior_later_view = checked(later_plan['inputs']['view'])
    later_result = checked(inp['later_result'])
    later_terminal = checked(inp['later_terminal'])
    prior_coverage = checked(inp['later_coverage'])
    later_gather = checked(inp['later_gather'])
    later_validation = checked(inp['later_validation'])
    initial = checked(inp['initial_result'])
    packet = checked(train_plan['inputs']['packet'])
    constructor = checked(train_plan['inputs']['constructor'])
    prior_view = checked(train_plan['inputs']['view'])
    physical = checked(train_plan['inputs']['physical_result'])
    physical_plan = checked(physical['plan'])
    old_span_plan = checked(physical_plan['evidence']['input_plan'])
    provider_plan = checked(mature.INPUT_REFS['provider_plan'])
    base.checked_file(mature.INPUT_REFS['provider_worker'])
    need(train_result['status'] == 'COMPLETE_FULL_FOLD2_TRAIN_INPUT_GATHER'
         and train_validation['status'] == 'PASS_FULL_FOLD2_TRAIN_INPUT_GATHER'
         and later_result['status'] == later_terminal['status'] == 'COMPLETE'
         and later_terminal['exit_code'] == 0
         and later_terminal['result'] == inp['later_result']
         and later_terminal['coverage'] == inp['later_coverage']
         and later_terminal['gather_result'] == inp['later_gather']
         and later_gather['status'] == 'COMPLETE'
         and later_gather['selected_later_inputs_qualified'] is True
         and later_validation['status'] == 'PASS_JSON_HASH_JOIN_AND_EXACT_COUNTS'
         and later_plan['inputs']['packet'] == train_plan['inputs']['packet']
         and later_plan['inputs']['constructor'] == train_plan['inputs']['constructor']
         and later_plan['inputs']['provider_plan'] == mature.INPUT_REFS['provider_plan']
         and later_plan['inputs']['fold2_physical_result'] == train_plan['inputs']['physical_result']
         and later_plan['replay_plan']['caches'] == old_span_plan['caches']
         and later_plan['replay_plan']['local_clock_path'] == old_span_plan['local_clock_path']
         and later_result['test_data_used'] is False
         and later_terminal['test_data_used'] is False
         and later_gather['test_data_used'] is False
         and initial['status'] == 'COMPLETE'
         and initial['decision'] == 'PASS_REAL_FEATURE_S_F_FORWARD_BACKWARD_NO_UPDATE'
         and initial['test_data_used'] is False,
         'INPUT_LINEAGE')
    for lane in LANES:
        base.checked_file(later_result['surfaces'][lane]['artifact'])
    for lane in LANES[1:]:
        for kind in ('feats', 'ts'):
            base.checked_file(old_span_plan['caches'][lane][kind])
    sources = {key: dict(item) for key, item in prior_view['source_bindings'].items()}
    bridge = base.old.HISTORICAL_RUNTIME_BRIDGE['gx1/contracts/entry_gc_mtf_history_v1.py']
    need(sources['gc_gather_source']['sha256'] == bridge[0]
         and owners['gx1/contracts/entry_gc_mtf_history_v1.py'] == bridge[1],
         'HISTORICAL_GC_BRIDGE')
    for key, item in sources.items():
        path = Path(item['path'])
        if path.is_relative_to(REPO) and key != 'gc_gather_source':
            relative = str(path.relative_to(REPO))
            need(relative in owners, 'CURRENT_VIEW_SOURCE:' + key)
            sources[key] = {'path': str(path), 'sha256': owners[relative]}
        elif key != 'gc_gather_source':
            base.checked_file(item)
    view = views.bind_native_direction_feature_view(
        prefix_normalization=packet,
        prefix_normalization_ref=train_plan['inputs']['packet'],
        constructor_metadata=constructor,
        constructor_metadata_ref=train_plan['inputs']['constructor'],
        source_bindings=sources, max_batch_rows=16,
        full_feature_profile=True, market_state_time_query_v2=True)
    need(view['contract_sha256'] == plan['expected_view_sha256']
         and view['contract_sha256'] == initial['view_sha256']
         and view['physical_widths'] == {'signal': 2727, 'ctx_cont': 2542,
                                         'ctx_cat': 1, 'mtf': 255},
         'CURRENT_FULL_GC_VIEW')
    populations = {lane: later_result['surfaces'][lane]['population'] for lane in LANES}
    need(set(prior_coverage) == {'schema_version', 'status', 'qualification',
          'view_sha256', 'full_axes', 'normalization_prefix', 'score_parent_start',
          'score_rows', 'score_start_ns', 'score_end_exclusive_ns', 'batch_size',
          'batch_indices', 'selected_parent_rows_sha256',
          'selected_decision_ns_sha256', 'physical_populations'}
         and prior_coverage['status'] == 'COMPLETE'
         and prior_coverage['view_sha256'] == prior_later_view['contract_sha256']
         and prior_coverage['qualification'] == inp['later_result']
         and prior_coverage['physical_populations'] == populations
         and prior_coverage['normalization_prefix'] == view['prefix']
         and prior_coverage['full_axes'] == view['full_axes'],
         'PRIOR_COVERAGE')
    panel = later_panel(fold_id=2)
    need(prior_coverage['score_parent_start'] == panel['score_parent_start']
         and prior_coverage['score_rows'] == panel['score_rows']
         and prior_coverage['batch_indices'] == panel['batch_indices']
         and len(panel['batch_indices']) == BATCHES,
         'FIXED_LATER_PANEL')
    coverage = dict(prior_coverage, view_sha256=view['contract_sha256'])
    expected_shapes = {key: [BATCH_ROWS, *shape[1:]]
                       for key, shape in initial['feature_shapes'].items()}
    need(len(expected_shapes) == 14 and 'gc_seq_d1' in expected_shapes,
         'FULL_GC_SHAPES')
    policy_ref = ref(POLICY)
    if require_policy:
        policy = json.loads(POLICY.read_text())
        gate = policy.get(STAGE)
        need(policy.get('training_enabled') is False
             and policy.get('native_training_enabled') is False
             and policy.get('native_launch_authorized') is False
             and isinstance(gate, dict) and gate.get('status') == 'ADMITTED'
             and gate.get('plan') == plan_ref
             and gate.get('worker') == ref(REPO / WORKER_PATH)
             and gate.get('max_invocations') == 1
             and gate.get('claimed_invocations') == gate.get('completed_invocations') == 0
             and gate.get('execution_authorized') is True
             and gate.get('data_reads_authorized') is True
             and gate.get('model_forwards_authorized') == 0
             and gate.get('optimizer_steps_authorized') == 0
             and gate.get('training_authorized') is False
             and gate.get('relaunch_allowed') is False
             and gate.get('test_data_used') is False,
             'POLICY')
    return dict(plan=plan, policy_ref=policy_ref, view=view, packet=packet,
                coverage=coverage, populations=populations, panel=panel,
                expected_shapes=expected_shapes, later_result=later_result,
                old_span_plan=old_span_plan, provider_plan=provider_plan)


def preflight(plan_ref: dict) -> dict:
    x = inspect(plan_ref, require_policy=False)
    return {'status': 'READY_NO_CLAIM_NO_ARRAY_DECODE',
            'plan': plan_ref, 'view_sha256': x['view']['contract_sha256'],
            'source_owners_checked': len(x['plan']['source_owners']),
            'later_parent_rows': SCORE_ROWS, 'feature_batches': BATCHES,
            'model_forwards': 0, 'optimizer_steps': 0, 'test_data_used': False}


def gather(x: dict, started: float) -> dict:
    from gx1.contracts.entry_anchor_span_feature_inputs_v1 import (
        build_anchor_span_inputs, TIMEFRAMES,
    )
    from gx1.contracts.entry_exit_feature_surface_v1 import load_m1_feature_surface_times
    from gx1.contracts.entry_gc_mtf_history_v1 import load_gc_mtf_history_raw_artifact
    provider = mature._provider_module()
    provider_plan = x['provider_plan']
    info = provider.validate_metadata(provider_plan)
    axes, _ = provider.read_axes(provider_plan, info, np)
    view = x['view']
    for key in ('parent_rows', 'bar_start_ns', 'decision_ns'):
        value = axes[key]
        need(value.dtype == np.int64 and value.shape == (view['full_axes']['rows'],)
             and hashlib.sha256(value.astype('<i8', copy=False).tobytes()).hexdigest()
                 == view['full_axes'][key + '_sha256'], 'AXIS:' + key)
        value.setflags(write=False)
    groups = provider.map_reuse(info['reuse'], np)
    source, current = groups['source050'], groups['current051']
    old_span_plan = x['old_span_plan']
    need(provider.regular(old_span_plan['local_clock_path'])
         == old_span_plan['local_clock_stat'], 'LOCAL_CLOCK_STAT')
    times = load_m1_feature_surface_times(
        Path(old_span_plan['local_clock_path']),
        context='FULL_GC_FOLD2_LATER_CLOCK_ONLY',
        expected_bar_seconds=300).asi8
    positions = np.searchsorted(times, axes['bar_start_ns']).astype(np.int64)
    need(np.all(positions < len(times))
         and np.array_equal(times[positions], axes['bar_start_ns']),
         'LOCAL_POSITIONS')
    for array in (times, positions, source['signal'], current['snap'],
                  current['ctx_cont'], current['ctx_cat']):
        array.setflags(write=False)
    sequence = dict(signal=source['signal'], times_ns=times,
                    sample_positions=positions, audit=info['audit'])
    payloads, values, clocks = {}, {}, {}
    for lane in LANES:
        sidecar = x['later_result']['surfaces'][lane]['artifact']
        base.checked_file(sidecar)
        with np.load(sidecar['path'], allow_pickle=False) as archive:
            payloads[lane] = {key: archive[key] for key in archive.files}
        if lane != 'local':
            cache = old_span_plan['caches'][lane]
            values[lane] = np.load(cache['feats']['path'], mmap_mode='r',
                                   allow_pickle=False)
            clocks[lane] = np.load(cache['ts']['path'], mmap_mode='r',
                                   allow_pickle=False)
    parent = x['packet']['parent_prefix_normalization']['base_normalization']
    bound = build_anchor_span_inputs(
        view, entry_sequence_source=sequence,
        snap_x=current['snap'], ctx_cont=current['ctx_cont'],
        ctx_cat=current['ctx_cat'],
        parent_signal_names=parent['surfaces']['signal']['field_names'],
        parent_mtf_names=parent['surfaces']['mtf_m5']['field_names'],
        parent_mtf_values=values, parent_mtf_times=clocks,
        span_payloads=payloads, physical_populations=x['populations'],
        unobserved_units_profile=None, later_coverage=x['coverage'],
        later_axes=axes)
    gc = load_gc_mtf_history_raw_artifact(provider_plan['gc_surfaces'])
    need(gc['layout'] == mature._receipt_chain()['gc_layout']
         and gc['layout']['contract_sha256']
             == view['prefix_normalization']['gc_layout_sha256'],
         'CURRENT_GC_LAYOUT')
    panel = x['panel']
    chosen = (panel['score_parent_start']
              + np.asarray(panel['batch_indices'], dtype=np.int64)[:, None] * BATCH_ROWS
              + np.arange(BATCH_ROWS, dtype=np.int64)).reshape(-1)
    need(chosen.shape == (SCORE_ROWS,)
         and int(chosen[-1]) < panel['score_parent_start'] + panel['score_rows'],
         'CHOSEN_ROWS')
    visited, issues_seen = hashlib.sha256(), hashlib.sha256()
    shapes = x['expected_shapes']
    clock_checks = {lane: 0 for lane in TIMEFRAMES}
    for left in range(0, SCORE_ROWS, BATCH_ROWS):
        need(time.monotonic() - started < WALL_SECONDS - 100, 'TIME_BUDGET')
        at = chosen[left:left + BATCH_ROWS]
        rows, issues = axes['parent_rows'][at], axes['decision_ns'][at]
        batch = views.gather_native_direction_feature_batch(
            view, parent_rows=axes['parent_rows'],
            bar_start_ns=axes['bar_start_ns'], decision_ns=axes['decision_ns'],
            entry_sequence_source=sequence, snap_x=current['snap'],
            ctx_cont=current['ctx_cont'], ctx_cat=current['ctx_cat'],
            multi_tf_sources=bound.mtf_sources, gc_surfaces=gc['surfaces'],
            gc_layout=gc['layout'], batch_parent_rows=rows,
            unobserved_units_profile=None, anchor_span_inputs=bound)
        need(batch['witness']['parent_rows'] == rows.tolist()
             and batch['witness']['decision_ns'] == issues.tolist()
             and set(batch['features']) == set(shapes), 'BATCH_IDENTITY')
        visited.update(rows.astype('<i8', copy=False).tobytes())
        issues_seen.update(issues.astype('<i8', copy=False).tobytes())
        for key, expected in shapes.items():
            value = batch['features'][key]
            dtype = np.int64 if key == 'ctx_cat' else np.float32
            need(value.dtype == dtype and list(value.shape) == expected
                 and np.isfinite(value).all(), 'FEATURE:' + key)
        for lane, witnesses in batch['witness']['multi_tf'].items():
            need(len(witnesses) == BATCH_ROWS
                 and all(max(row['available_ns']) <= int(issue)
                         for row, issue in zip(witnesses, issues)),
                 'CAUSAL_CLOCK:' + lane)
            clock_checks[lane] += BATCH_ROWS
    need(visited.hexdigest() == x['coverage']['selected_parent_rows_sha256']
         and issues_seen.hexdigest() == x['coverage']['selected_decision_ns_sha256']
         and all(n == SCORE_ROWS for n in clock_checks.values()),
         'COMPLETE_PANEL')
    for array in (*source.values(), *current.values(),
                  *values.values(), *clocks.values()):
        provider.flush_pages(array)
    for surface in gc['surfaces'].values():
        for key in ('native_row_indices', 'available_ns', 'features'):
            provider.flush_pages(surface[key])
    provider.validate_metadata(provider_plan)
    return {'decision': 'PASS_FULL_GC_FOLD2_LATER_INPUT_GATHER',
            'view_sha256': view['contract_sha256'],
            'selected_parent_rows': SCORE_ROWS,
            'feature_batches': BATCHES,
            'feature_shapes_per_batch': shapes,
            'parent_rows_sha256': visited.hexdigest(),
            'decision_ns_sha256': issues_seen.hexdigest(),
            'causal_mtf_clock_rows': clock_checks,
            'gc_surface_count': len(gc['surfaces']),
            'all_raw_feature_values_finite': True,
            'cpu_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'target_arrays_decoded': 0, 'model_constructions': 0,
            'model_forwards': 0, 'optimizer_steps': 0,
            'test_data_used': False}


def run(plan_ref: dict) -> dict:
    from gx1.contracts.gx1_capped_execution_v1 import (
        require_capped_cpu_research_producer_execution,
        require_capped_lock_ancestry,
    )
    started = time.monotonic()
    x = inspect(plan_ref, require_policy=True)
    cap = require_capped_cpu_research_producer_execution()
    cap['lock'] = require_capped_lock_ancestry()
    need(cap['memory_max'] == cap['memory_high'] == MEMORY_BYTES
         and cap['swap'] == SWAP_BYTES and cap['pids'] == 64, 'CAP')
    head, _ = base.current_state()
    OUT_ROOT.mkdir(parents=True, exist_ok=False)
    claim = None
    try:
        claim = base.old._write(OUT_ROOT, 'CLAIM.json', {
            'schema_version': 'gx1_market_state_full_gc_fold2_later_claim_v1',
            'plan': plan_ref, 'policy': x['policy_ref'],
            'source_head': head, 'test_data_used': False},
            limit=base.old.MAX_JSON_BYTES)
        old_handler = signal.getsignal(signal.SIGALRM)
        def timeout(_signum, _frame):
            raise TimeoutError('FULL_GC_FOLD2_LATER_WALL')
        signal.signal(signal.SIGALRM, timeout)
        signal.setitimer(signal.ITIMER_REAL, WALL_SECONDS - 100)
        try:
            result = gather(x, started)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)
        need(base.current_state()[0] == head
             and ref(PLAN_PATH) == plan_ref
             and ref(POLICY) == x['policy_ref']
             and all(base.sha(REPO / name) == digest
                     for name, digest in x['plan']['source_owners'].items()),
             'SOURCE_BOOKEND')
        result_ref = base.old._write(OUT_ROOT, 'RESULT.json', {
            'schema_version': 'gx1_market_state_full_gc_fold2_later_result_v1',
            'status': 'COMPLETE', 'plan': plan_ref, 'claim': claim,
            'policy': x['policy_ref'], 'source_head': head,
            'cap': cap, 'elapsed_seconds': time.monotonic() - started,
            **result}, limit=base.old.MAX_JSON_BYTES)
        terminal = base.old._write(OUT_ROOT, 'TERMINAL.json', {
            'schema_version': 'gx1_market_state_full_gc_fold2_later_terminal_v1',
            'status': 'COMPLETE', 'exit_code': 0,
            'plan': plan_ref, 'claim': claim, 'result': result_ref,
            'feature_batches': BATCHES, 'target_arrays_decoded': 0,
            'model_forwards': 0, 'optimizer_steps': 0,
            'test_data_used': False, 'no_relaunch_under_this_plan': True},
            limit=base.old.MAX_JSON_BYTES)
        return {'result': result_ref, 'terminal': terminal}
    except BaseException as exc:
        failed = base.old._write(OUT_ROOT, 'FAILED.json', {
            'schema_version': 'gx1_market_state_full_gc_fold2_later_failure_v1',
            'plan': plan_ref, 'claim': claim, 'source_head': head,
            'error_type': type(exc).__name__, 'error': str(exc)[:2048],
            'elapsed_seconds': time.monotonic() - started,
            'no_relaunch_under_this_plan': True},
            limit=base.old.MAX_JSON_BYTES)
        base.old._write(OUT_ROOT, 'TERMINAL.json', {
            'schema_version': 'gx1_market_state_full_gc_fold2_later_terminal_v1',
            'status': 'FAILED', 'exit_code': 1,
            'plan': plan_ref, 'claim': claim, 'failure': failed,
            'feature_batches_upper_bound': BATCHES,
            'target_arrays_decoded': 0, 'model_forwards': 0,
            'optimizer_steps': 0, 'test_data_used': False,
            'no_relaunch_under_this_plan': True},
            limit=base.old.MAX_JSON_BYTES)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    plan_ref = {'path': args.plan, 'sha256': args.plan_sha256}
    report = preflight(plan_ref) if args.preflight else run(plan_ref)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
