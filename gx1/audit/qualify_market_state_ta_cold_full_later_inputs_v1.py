"""One-use, read-only qualification of cold M5/MTF later physical coverage.

Reuses the qualified Fold2 raw span sidecars and all-row native input gather.
No target payload, normalization fit, model, checkpoint or optimizer is read.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import signal
import subprocess
import time

import numpy as np

from gx1.contracts import entry_anchor_span_feature_inputs_v1 as overlay
from gx1.contracts import entry_native_direction_feature_view_v1 as views
from gx1.contracts.entry_gc_mtf_history_v1 import WINDOW_ROWS
from gx1.contracts.entry_model_native_signal_v1 import MODEL_NATIVE_SEQ_LEN
from gx1.contracts.gx1_capped_execution_v1 import require_capped_cpu_audit_execution
from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
from gx1.features import htf_features as htf
from gx1.models.entry_v10.entry_v10_input_normalization import _hash_int64_indices

REPO = Path('/home/andre2/src/GX1_CURRENT')
RUNS = Path('/home/andre2/GX1_RUNS')
STAGE = 'ta_m5_cold_full_later_inputs_20261117_001'
ROOT = RUNS / STAGE.upper()
PLAN = REPO / 'configs/research' / (STAGE.upper() + '.json')
POLICY = REPO / 'NEXT_RUN_POLICY.json'
SOURCE_BASE = 'a1f11cd333c7f0cbe3c008bd98fab8bf3c3c1bdc'
COLD_VIEW_SHA = '7cdcc64b80ba8a1ac92aae24ac7fcd2b62252da31cf96e57ad7a86fc943c3293'
CURRENT_VIEW_SHA = '8c58f3c8b13a6d13143c1b5580051f0386fc5201452493a32ae072cd54b7cfd9'
MEMORY = '4G'
SWAP = '512M'
WALL_SECONDS = 1800
LANES = ('local', *overlay.TIMEFRAMES)
SOURCE_PATHS = (
    'gx1/audit/qualify_market_state_ta_cold_full_later_inputs_v1.py',
    'gx1/contracts/entry_anchor_span_feature_inputs_v1.py',
    'gx1/contracts/entry_native_direction_feature_view_v1.py',
    'gx1/contracts/entry_anchor_span_feature_view_v1.py',
    'gx1/contracts/entry_gc_mtf_history_v1.py',
    'gx1/contracts/entry_model_native_signal_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'gx1/features/htf_features.py',
    'gx1/models/entry_v10/entry_v10_input_normalization.py',
    'tests/test_qualify_market_state_ta_cold_full_later_inputs_v1.py',
)
INPUTS = {
    'fold1_view': dict(path=str(RUNS / 'MARKET_STATE_TA_M5_OBJECTIVE_BENCHMARK_20261112_001/FEATURE_VIEW.json'),
                       sha256='2b15bcfe7fec09d83dfc6af71fe406b6c8153aa5d0a5d9262d112fea3ace3d60'),
    'fold1_packet': dict(path=str(RUNS / 'MARKET_STATE_TA_M5_OBJECTIVE_BENCHMARK_20261112_001/PREFIX_PACKET.json'),
                         sha256='900fd2ca1b571b68bbea05e94cda739d718fa6167aaaa9290c8980f60fbac9fb'),
    'constructor': dict(path=str(RUNS / 'MARKET_STATE_TA_M5_OBJECTIVE_BENCHMARK_20261112_001/CONSTRUCTOR_METADATA.json'),
                        sha256='84883657faac58baa62250528b74bd38b3625ac91031677062acb16080aeaf88'),
    'current_view': dict(path=str(REPO / 'configs/research/TA_FOLD2_APR_JUN_FEATURE_VIEW_20261117_001.json'),
                         sha256='2e8d3e2aec63c423a1a85d1391e02edbe8dc2864fa687ea11e73a9af1d2e6eda'),
    'fold2_view': dict(path=str(RUNS / 'MARKET_STATE_TA_FOLD2_SPAN_VIEW_20261113_001/FEATURE_VIEW.json'),
                       sha256='81ea31a124f45d18adfd35958b90ebb1124d6bb8d256dbe29f32e5b9213cbff0'),
    'fold2_gather': dict(path=str(RUNS / 'MARKET_STATE_TA_FOLD2_SPAN_VIEW_20261113_001/RESULT.json'),
                         sha256='a108db7b1b56f5d7321befbcf6412c8d0f3116a88a9bda31059270d09970d7ec'),
    'fold2_gather_terminal': dict(path=str(RUNS / 'MARKET_STATE_TA_FOLD2_SPAN_VIEW_20261113_001/TERMINAL.json'),
                                  sha256='b71df0c340b409994602893201b24a4a0cbd51cce578deb26bc2933db64ab84b'),
    'span_result': dict(path=str(RUNS / 'MARKET_STATE_TA_FOLD2_SPAN_REUSE_20261113_002/RESULT.json'),
                        sha256='8f2acfafc459cc9a2fbcb7b588c40eb008d1e75c7e47835bbc55f24b68f32cc2'),
    'span_terminal': dict(path=str(RUNS / 'MARKET_STATE_TA_FOLD2_SPAN_REUSE_20261113_002/TERMINAL.json'),
                          sha256='3de8d27ac093e9d06a5b473bd7d76cfaa8eaea8c33cfdeaa35bda1720094b807'),
    'axis_archive': dict(path=str(RUNS / 'NATIVE_M5_DIRECTION_BASELINE_DATA_20261024_245/OUTPUT/PRICE_TARGETS.npz'),
                         sha256='9d4a5fb3c473b088aff5a5d5a176c199a64053b679229c14f125697b93af577d'),
}


def need(ok, reason):
    if not bool(ok):
        raise RuntimeError('TA_COLD_FULL_LATER_' + reason)


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def file_ref(path):
    path = Path(path)
    need(path.is_absolute() and path.is_file() and not path.is_symlink(), 'FILE_REF')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return dict(path=str(path), sha256=h.hexdigest())


def checked_json(ref):
    need(type(ref) is dict and set(ref) == {'path', 'sha256'}
         and type(ref['path']) is str and type(ref['sha256']) is str
         and PurePosixPath(ref['path']).is_absolute()
         and '..' not in PurePosixPath(ref['path']).parts
         and file_ref(ref['path']) == ref, 'INPUT_REF')
    return json.loads(Path(ref['path']).read_text())


def cold_views():
    old, packet, constructor, current = (checked_json(INPUTS[k]) for k in
        ('fold1_view', 'fold1_packet', 'constructor', 'current_view'))
    need(old['prefix_normalization']['binding'] == INPUTS['fold1_packet']
         and old['constructor_metadata']['binding'] == INPUTS['constructor']
         and current['full_axes'] == old['full_axes'], 'VIEW_ANCESTRY')
    def bind(base, normalization):
        return views.bind_native_direction_feature_view(
            prefix_normalization=normalization,
            prefix_normalization_ref=base['prefix_normalization']['binding'],
            constructor_metadata=constructor,
            constructor_metadata_ref=INPUTS['constructor'],
            source_bindings=base['source_bindings'],
            max_batch_rows=base['max_batch_rows'],
            unobserved_units_profile=normalization.get('unobserved_units_profile'),
            full_feature_profile=True, full_ta_feature_profile=True,
            market_state_time_query_v2=True, dedicated_m5_side=True,
            dedicated_tf_activity=True, dedicated_non_m5_side=True,
            direct_tf_hidden=True)
    cold = bind(old, packet)
    current_packet = checked_json(current['prefix_normalization']['binding'])
    warm = bind(current, current_packet)
    need(cold['contract_sha256'] == COLD_VIEW_SHA
         and warm['contract_sha256'] == CURRENT_VIEW_SHA
         and cold['full_axes'] == warm['full_axes']
         and cold['physical_widths'] == warm['physical_widths']
         and cold['physical_field_names_sha256'] == warm['physical_field_names_sha256']
         and cold['research_objective'] == warm['research_objective']
         and cold['prefix']['rows'] == overlay.cold_full_later_panel()['prefix_rows'],
         'COLD_VIEW_BINDING')
    return cold, warm


def reuse_proof(span, gather, span_terminal, gather_terminal, fold2_view, cold):
    """Existing all-row gather supplies the physical input proof, not a new fit."""
    need(span_terminal['status'] == gather_terminal['status'] == 'COMPLETE'
         and span_terminal['exit_code'] == gather_terminal['exit_code'] == 0
         and span_terminal['result'] == INPUTS['span_result']
         and gather_terminal['result'] == INPUTS['fold2_gather']
         and span_terminal['source_unchanged'] is gather_terminal['source_unchanged'] is True,
         'TERMINAL_PROOF')
    need(span['status'] == 'COMPLETE' and span['source_unchanged'] is True
         and span['test_data_used'] is False
         and span['counts']['model_forwards'] == span['counts']['optimizer_steps'] == 0
         and set(span['surfaces']) == set(LANES)
         and all(s['all_overlap_columns_bitexact'] is True for s in span['surfaces'].values()),
         'SPAN_PROOF')
    need(gather['status'] == 'COMPLETE_FULL_FOLD2_TRAIN_INPUT_GATHER'
         and gather['source_unchanged'] is True and gather['test_data_used'] is False
         and gather['view'] == INPUTS['fold2_view']
         and gather['counts']['parent_rows'] == 47338
         and gather['counts']['feature_batches'] == 2959
         and gather['counts']['model_forwards'] == gather['counts']['optimizer_steps'] == 0
         and gather['counts']['normalization_fits'] == 0
         and gather['all_raw_native_clock_alias_and_finite_guards_passed'] is True
         and gather['all_normalized_values_finite'] is True
         and gather['visited_parent_rows_sha256'] == fold2_view['prefix']['parent_rows_sha256'],
         'FULL_GATHER_PROOF')
    panel = overlay.cold_full_later_panel()
    need(panel['score_parent_start'] + panel['score_rows'] == gather['counts']['parent_rows']
         and cold['full_axes'] == fold2_view['full_axes']
         and cold['physical_widths'] == fold2_view['physical_widths']
         and cold['physical_field_names_sha256'] == fold2_view['physical_field_names_sha256'],
         'COLD_SUBSET_AND_PHYSICAL_IDENTITY')
    sidecars, populations = {}, {}
    for lane in LANES:
        surface = span['surfaces'][lane]
        need(set(surface) >= {'artifact', 'population'}
             and type(surface['artifact']) is dict
             and set(surface['artifact']) == {'path', 'sha256'}
             and surface['artifact']['path'] == str(RUNS / 'MARKET_STATE_TA_FOLD2_SPAN_REUSE_20261113_002' / (lane + '_ANCHOR_SPANS.npz')),
             'SIDECAR_PROVENANCE:' + lane)
        sidecars[lane] = surface['artifact']
        populations[lane] = surface['population']
    return sidecars, populations


def build_plan():
    cold, _ = cold_views()
    span, gather, st, gt, fold2_view = (checked_json(INPUTS[k]) for k in
        ('span_result', 'fold2_gather', 'span_terminal', 'fold2_gather_terminal', 'fold2_view'))
    sidecars, populations = reuse_proof(span, gather, st, gt, fold2_view, cold)
    panel = overlay.cold_full_later_panel()
    need(cold['full_axes']['source'] == INPUTS['axis_archive'], 'AXIS_SOURCE')
    return dict(schema_version='gx1_ta_m5_cold_full_later_input_plan_v1', stage=STAGE,
        source_repo=str(REPO), source_base_commit=SOURCE_BASE,
        source_owners={p:file_ref(REPO / p) for p in SOURCE_PATHS},
        inputs=deepcopy(INPUTS), physical_sidecars=deepcopy(sidecars),
        physical_populations=deepcopy(populations), cold_view_contract_sha256=COLD_VIEW_SHA,
        cold_view_file=dict(path=str(ROOT / 'FEATURE_VIEW.json'),
                            sha256=hashlib.sha256(encoded(cold)).hexdigest()),
        full_axes=deepcopy(cold['full_axes']), normalization_prefix=deepcopy(cold['prefix']),
        panel=panel, scope=dict(selected_parent_rows=panel['score_rows'],
            full_batches=panel['full_batches'], tail_rows=panel['tail_rows'],
            physical_surfaces=6, reused_full_gather_rows=47338, clock_probe_rows=3,
            target_arrays_decoded=0, normalization_fits=0, producer_calls=0,
            model_constructions=0, model_forwards=0, optimizer_steps=0,
            test_data_used=False, economic_edge_proven=False),
        execution=dict(module='gx1.audit.qualify_market_state_ta_cold_full_later_inputs_v1',
            cwd=str(REPO), runner='scripts/gx1_capped_run.sh', job_class='audit',
            memory=MEMORY, swap=SWAP, timeout_seconds=WALL_SECONDS,
            kill_after_seconds=10), output_root=str(ROOT))


def require_physical_refs(plan):
    refs = (plan['inputs']['axis_archive'], *plan['physical_sidecars'].values())
    need(all(file_ref(ref['path']) == ref for ref in refs), 'PHYSICAL_INPUT_SHA')


def inspect_plan(plan_sha, *, execute):
    plan_ref = dict(path=str(PLAN), sha256=plan_sha)
    plan = checked_json(plan_ref)
    need(plan == build_plan(), 'PLAN_IDENTITY')
    require_physical_refs(plan)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=REPO, text=True).strip()
    clean = subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True) == ''
    ancestry = subprocess.run(['git', 'merge-base', '--is-ancestor', SOURCE_BASE, head],
                              cwd=REPO, capture_output=True).returncode == 0
    need(Path.cwd().resolve() == REPO and branch == 'work/gx1-current' and ancestry,
         'SOURCE_BRANCH_AND_CWD')
    need(not ROOT.exists(), 'ONE_USE_ROOT_EXISTS')
    if execute:
        need(clean, 'CLEAN_SOURCE')
        policy_ref = file_ref(POLICY)
        policy = checked_json(policy_ref)
        need(all(policy.get(k) is False for k in
                 ('training_enabled', 'native_training_enabled', 'native_launch_authorized',
                  'full_val_allowed', 'full_epoch_training_allowed')), 'TRAINING_CLOSED')
        gate = policy.get(STAGE, {})
        need(gate.get('status') == 'ADMITTED' and gate.get('plan') == plan_ref
             and gate.get('execution_authorized') is True
             and gate.get('data_reads_authorized') is True
             and gate.get('training_authorized') is False
             and gate.get('test_authorized') is False
             and gate.get('scope') == plan['scope']
             and gate.get('max_invocations') == gate.get('remaining_invocations') == 1
             and gate.get('claimed_invocations') == gate.get('completed_invocations') == 0
             and gate.get('relaunch_allowed') is False, 'EXACT_POLICY_ADMISSION')
        return plan, plan_ref, policy_ref, head, clean
    return plan, plan_ref, None, head, clean


def load_physical(plan):
    """Read only three source axes and the six bound span archives."""
    need(file_ref(INPUTS['axis_archive']['path']) == INPUTS['axis_archive'], 'AXIS_ARCHIVE_SHA')
    with np.load(INPUTS['axis_archive']['path'], allow_pickle=False) as z:
        axes = {}
        for key in ('parent_rows', 'bar_start_ns', 'decision_ns'):
            raw = z[key]
            need(raw.dtype == np.int64, 'FULL_AXIS_DTYPE:' + key)
            axes[key] = np.array(raw, copy=True, order='C')
    for key, value in axes.items():
        need(value.shape == (118249,) and np.all(np.diff(value) > 0)
             and overlay._axis_sha(value) == plan['full_axes'][key + '_sha256'],
             'FULL_AXIS:' + key)
        value.setflags(write=False)
    payloads = {}
    for lane in LANES:
        ref = plan['physical_sidecars'][lane]
        need(file_ref(ref['path']) == ref, 'SIDECAR_SHA:' + lane)
        with np.load(ref['path'], allow_pickle=False) as z:
            need(set(z.files) == {'row_indices', 'times_ns', 'native_bar_positions', 'anchor_span_bars'},
                 'SIDECAR_KEYS:' + lane)
            payload = {k:np.array(z[k], copy=True, order='C') for k in z.files}
        rows, times, native, spans = (payload[k] for k in
            ('row_indices', 'times_ns', 'native_bar_positions', 'anchor_span_bars'))
        pop = plan['physical_populations'][lane]
        shift = 300 * 10**9 if lane == 'local' else int(htf.MULTI_TF_SHIFT[lane].value)
        namespace = 'entry_local_m5_rows' if lane == 'local' else 'mtf:' + lane
        need(rows.dtype == times.dtype == native.dtype == np.int64
             and spans.dtype == np.float32 and spans.shape == (len(rows), 2)
             and len(rows) == pop['selected_unique_row_count']
             and np.all(np.diff(rows) == 1) and np.all(np.diff(times) > 0)
             and np.all(np.diff(native) > 0) and np.isfinite(spans).all()
             and np.all(spans >= 0) and np.all(spans == np.floor(spans))
             and _hash_int64_indices(rows, namespace=namespace) == pop['selected_row_indices_sha256']
             and hashlib.sha256(np.ascontiguousarray(times + shift, dtype='<i8').tobytes()).hexdigest()
                 == pop['available_ns_sha256']
             and np.all(times + shift < plan['panel']['score_end_exclusive_ns']),
             'SIDECAR_POPULATION:' + lane)
        for a in payload.values():a.setflags(write=False)
        payloads[lane] = payload
    return axes, payloads


def clock_probes(axes, payloads):
    """Boundary and tail clocks; prior Fold2 gather proves values on all rows."""
    panel = overlay.cold_full_later_panel()
    start = panel['score_parent_start']
    stop = start + panel['score_rows']
    probes = (start, stop - 2, stop - 1)
    local = payloads['local']
    for position in probes:
        decision, bar_start = (int(axes[k][position]) for k in ('decision_ns', 'bar_start_ns'))
        at = int(np.searchsorted(local['times_ns'], bar_start))
        need(at < len(local['times_ns']) and local['times_ns'][at] == bar_start
             and at >= MODEL_NATIVE_SEQ_LEN - 1, 'LOCAL_PROBE_CLOCK')
        lower = at - MODEL_NATIVE_SEQ_LEN + 1
        need(np.all(np.diff(local['row_indices'][lower:at + 1]) == 1)
             and np.all(local['times_ns'][lower:at + 1] + 300 * 10**9 <= decision),
             'LOCAL_PROBE_WINDOW')
        for tf in overlay.TIMEFRAMES:
            payload = payloads[tf]
            shift = int(htf.MULTI_TF_SHIFT[tf].value)
            right = int(np.searchsorted(payload['times_ns'], decision - shift, side='right'))
            left = right - WINDOW_ROWS[tf]
            need(left >= 0 and right <= len(payload['times_ns'])
                 and np.all(np.diff(payload['row_indices'][left:right]) == 1)
                 and np.all(payload['times_ns'][left:right] + shift <= decision),
                 'MTF_PROBE_WINDOW:' + tf)
    return dict(parent_positions=list(probes), local_windows=3, mtf_windows=15,
                fresh_native_value_gather=0)


def coverage_record(plan, view, axes, result_ref):
    panel = plan['panel']
    start, stop = panel['score_parent_start'], panel['score_parent_start'] + panel['score_rows']
    selected = np.arange(start, stop, dtype=np.int64)
    coverage = dict(schema_version=overlay.COLD_FULL_LATER_COVERAGE_SCHEMA,
        status='COMPLETE', qualification=result_ref,
        view_sha256=view['contract_sha256'], full_axes=deepcopy(view['full_axes']),
        normalization_prefix=deepcopy(view['prefix']),
        score_parent_start=start, score_rows=panel['score_rows'],
        score_start_ns=panel['score_start_ns'],
        score_end_exclusive_ns=panel['score_end_exclusive_ns'],
        batch_size=panel['batch_size'], full_batches=panel['full_batches'],
        tail_rows=panel['tail_rows'],
        selected_parent_rows_sha256=overlay._axis_sha(axes['parent_rows'][selected]),
        selected_decision_ns_sha256=overlay._axis_sha(axes['decision_ns'][selected]),
        physical_populations=deepcopy(plan['physical_populations']),
        physical_sidecars=deepcopy(plan['physical_sidecars']))
    bound = overlay._require_later_coverage(view, coverage, axes,
        plan['physical_populations'], sidecar_refs=plan['physical_sidecars'])
    need(len(bound['later_parent_positions']) == panel['score_rows']
         and int(bound['later_parent_positions'][-1]) == selected[-1], 'COVERAGE_GATE')
    return coverage


def terminal_record(status, *, plan_ref, claim, head, result, coverage, failure, source_unchanged):
    need((status == 'COMPLETE' and result is not None and coverage is not None and failure is None)
         or (status == 'FAILED_CONSUMED' and failure is not None), 'TERMINAL_STATE')
    return dict(status=status, exit_code=0 if status == 'COMPLETE' else 1,
        no_relaunch=True, source_head=head, plan=plan_ref, claim=claim,
        result=result, coverage=coverage, failure=failure, source_unchanged=source_unchanged)


def publish(name, value):
    data = encoded(value)
    need(len(data) <= 2 * 1024**2, 'JSON_BUDGET')
    path = ROOT / name
    _publish_bytes(path, data)
    return file_ref(path)


def bookend(plan_ref, policy_ref, head, plan):
    need(file_ref(PLAN) == plan_ref and file_ref(POLICY) == policy_ref
         and subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip() == head
         and subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True) == ''
         and all(file_ref(REPO / p) == ref for p, ref in plan['source_owners'].items())
         and all(file_ref(ref['path']) == ref for ref in (*plan['inputs'].values(),
                  *plan['physical_sidecars'].values())), 'SOURCE_INPUT_BOOKEND')


def run(plan_sha):
    began = time.monotonic()
    cap = require_capped_cpu_audit_execution()
    plan, plan_ref, policy_ref, head, _ = inspect_plan(plan_sha, execute=True)
    need(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), 'AMBIENT_TIMER')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    ROOT.mkdir()
    claim = publish('CLAIM.json', dict(plan=plan_ref, policy=policy_ref, source_head=head,
        cap=cap, scope=plan['scope']))
    result = coverage = failure = None
    status = 'FAILED_CONSUMED'
    old_alarm = signal.getsignal(signal.SIGALRM)
    def expired(*_):raise TimeoutError('TA_COLD_FULL_LATER_1800_SECONDS')
    try:
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, max(1, WALL_SECONDS - 30 - (time.monotonic() - began)))
        cold, _ = cold_views()
        need(cold['contract_sha256'] == plan['cold_view_contract_sha256'], 'RUN_VIEW')
        view_ref = publish('FEATURE_VIEW.json', cold)
        need(view_ref == plan['cold_view_file'], 'VIEW_FILE_SHA')
        span, gather, st, gt, fold2_view = (checked_json(INPUTS[k]) for k in
            ('span_result', 'fold2_gather', 'span_terminal', 'fold2_gather_terminal', 'fold2_view'))
        sidecars, populations = reuse_proof(span, gather, st, gt, fold2_view, cold)
        need(sidecars == plan['physical_sidecars'] and populations == plan['physical_populations'],
             'RUN_PHYSICAL_BINDING')
        axes, payloads = load_physical(plan)
        probes = clock_probes(axes, payloads)
        bookend(plan_ref, policy_ref, head, plan)
        result = publish('RESULT.json', dict(status='COMPLETE', plan=plan_ref, claim=claim,
            policy=policy_ref, source_head=head, source_unchanged=True,
            view=view_ref, prior_full_gather=INPUTS['fold2_gather'],
            prior_span_result=INPUTS['span_result'],
            physical_sidecars=sidecars, physical_populations=populations,
            scope=plan['scope'], clock_probes=probes,
            full_score_parent_positions=list((plan['panel']['score_parent_start'],
                plan['panel']['score_parent_start'] + plan['panel']['score_rows'] - 1)),
            coverage_path=str(ROOT / 'COVERAGE.json'),
            target_arrays_decoded=0, normalization_fits=0, producer_calls=0,
            model_forwards=0, optimizer_steps=0, test_data_used=False,
            economic_edge_proven=False,
            limits=['Physical cold later coverage only; no fresh cold normalized/model gather.',
                    'Historical Oct-Dec score is reused development, not independent Edge.']))
        coverage_value = coverage_record(plan, cold, axes, result)
        coverage = publish('COVERAGE.json', coverage_value)
        status = 'COMPLETE'
    except BaseException as exc:
        failure = publish('FAILED.json', dict(status='FAILED_CONSUMED', plan=plan_ref,
            claim=claim, error_type=type(exc).__name__, error=str(exc),
            result=result, coverage=coverage))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_alarm)
        terminal = publish('TERMINAL.json', terminal_record(status, plan_ref=plan_ref,
            claim=claim, head=head, result=result, coverage=coverage, failure=failure,
            source_unchanged=(subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==head)))
    return terminal, status


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight', action='store_true')
    mode.add_argument('--execute', action='store_true')
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    if args.preflight:
        plan, plan_ref, _, head, clean = inspect_plan(args.plan_sha256, execute=False)
        print(json.dumps(dict(status='PREFLIGHT_PASS' if clean else 'PREFLIGHT_SOURCE_DIRTY',
            plan=plan_ref, source_head=head, source_clean=clean, output_root_absent=True,
            scope=plan['scope'], cold_view_sha256=plan['cold_view_contract_sha256'],
            sidecar_count=len(plan['physical_sidecars']), claim_created=False), sort_keys=True))
        return
    terminal, status = run(args.plan_sha256)
    print(json.dumps(dict(status=status, terminal=terminal), sort_keys=True))
    if status != 'COMPLETE':raise SystemExit(1)


if __name__ == '__main__':
    main()
