"""One-use, target-free raw and normalized gather of the new 318/256 view.

The immutable old matrices are overlaid with six qualified EMA-delta sidecars.
Every selected FIT and later row is gathered; the 275 intervening rows are not.
No model, normalization fit, checkpoint or target value is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np

from gx1.contracts import entry_anchor_span_feature_inputs_v1 as overlay
from gx1.contracts import entry_model_native_input_normalization_v1 as norm
from gx1.contracts import entry_native_direction_feature_view_v1 as views
from gx1.contracts.entry_gc_mtf_history_v1 import WINDOW_ROWS
from gx1.contracts.gx1_capped_execution_v1 import (
    require_capped_cpu_audit_execution, require_capped_lock_ancestry,
)
from gx1.contracts.immutable_event_authority_v1 import _publish_bytes

REPO = Path('/home/andre2/src/GX1_CURRENT')
RUNS = Path('/home/andre2/GX1_RUNS')
STAGE = 'ta_m5_ema_delta_full_gather_20261118_001'
ROOT = RUNS / STAGE.upper()
PLAN = REPO / 'configs/research' / (STAGE.upper() + '.json')
POLICY = REPO / 'NEXT_RUN_POLICY.json'
MATERIAL = RUNS / 'TA_M5_EMA_DELTA_SIDECARS_20261117_002'
NORMAL = RUNS / 'TA_M5_EMA_DELTA_FIT_NORMALIZATION_20261118_001'
VIEW = RUNS / 'TA_M5_EMA_DELTA_FEATURE_VIEW_20261118_001'
OLD_GATHER = RUNS / 'MARKET_STATE_TA_FOLD2_SPAN_VIEW_20261113_001'
FIT_SPANS = RUNS / 'MARKET_STATE_TA_TRENDLINE_ANCHOR_SPAN_INPUTS_20261112_001'
LATER_SPANS = RUNS / 'MARKET_STATE_TA_FOLD2_SPAN_REUSE_20261113_002'
LANES = ('local', *overlay.TIMEFRAMES)
SURFACES = {'seq_x': 'signal', 'snap_x': 'signal', 'ctx_cont': 'ctx_cont',
    **{'seq_' + tf.lower(): 'mtf_' + tf.lower() for tf in overlay.TIMEFRAMES}}
BATCH = 16
FIT_ROWS = 29574
LATER_START = 29849
LATER_STOP = 47338
LATER_ROWS = LATER_STOP - LATER_START
WALL_SECONDS = 2400
PHASES = {'FIT': (0, FIT_ROWS), 'LATER': (LATER_START, LATER_STOP)}
SCOPE = {'fit_rows': FIT_ROWS, 'later_parent_start': LATER_START,
    'later_rows': LATER_ROWS, 'gap_rows_excluded': LATER_START - FIT_ROWS,
    'selected_parent_rows': FIT_ROWS + LATER_ROWS, 'batch_rows': BATCH,
    'fit_batches': 1849, 'later_batches': 1094, 'feature_batches': 2943,
    'normalization_calls': 23544, 'raw_and_normalized_full_selected_values': True,
    'target_arrays_decoded': 0, 'model_constructions': 0, 'model_forwards': 0,
    'normalization_fits': 0, 'checkpoint_reads': 0, 'optimizer_steps': 0,
    'test_data_used': False, 'economic_edge_proven': False}
SOURCE_PATHS = (
    'gx1/audit/qualify_ta_ema_delta_full_gather_v1.py',
    'gx1/contracts/entry_ema_delta_feature_view_v1.py',
    'gx1/contracts/entry_anchor_span_feature_view_v1.py',
    'gx1/contracts/entry_anchor_span_feature_inputs_v1.py',
    'gx1/contracts/entry_anchor_span_normalization_v1.py',
    'gx1/contracts/entry_native_direction_feature_view_v1.py',
    'gx1/contracts/entry_model_native_input_normalization_v1.py',
    'gx1/contracts/entry_gc_mtf_history_v1.py',
    'gx1/contracts/entry_exit_feature_surface_v1.py',
    'gx1/audit/qualify_market_state_mature_prefix_normalization_v1.py',
    'gx1/contracts/gx1_capped_execution_v1.py',
    'gx1/contracts/immutable_event_authority_v1.py',
    'gx1/features/entry_specialist_feature_groups_v1.py',
    'gx1/features/htf_features.py',
    'scripts/gx1_capped_run.sh',
    'tests/test_qualify_ta_ema_delta_full_gather_v1.py',
)


def need(ok, reason):
    if not bool(ok):
        raise RuntimeError('TA_EMA_DELTA_FULL_GATHER_' + reason)


def ref(path):
    path = Path(path)
    need(path.is_absolute() and path.is_file() and not path.is_symlink(), 'FILE_REF')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return {'path': str(path), 'sha256': digest.hexdigest()}


def read(binding):
    need(type(binding) is dict and set(binding) == {'path', 'sha256'}
         and ref(binding['path']) == binding, 'JSON_REF')
    return json.loads(Path(binding['path']).read_text())


def axis_sha(values):
    return hashlib.sha256(np.ascontiguousarray(values, dtype='<i8').tobytes()).hexdigest()


def phase_positions():
    """The only selectable positions; no gap or later TEST positions."""
    need(FIT_ROWS == 29574 and LATER_START == 29849 and LATER_STOP == 47338
         and BATCH == 16 and LATER_ROWS == 17489, 'FIXED_RANGES')
    return {name: np.arange(start, stop, dtype=np.int64)
            for name, (start, stop) in PHASES.items()}


def batch_slices(name):
    start, stop = PHASES[name]
    for left in range(start, stop, BATCH):
        yield slice(left, min(left + BATCH, stop))


def input_refs():
    """Resolve current SHA only after the three predecessor stages complete."""
    paths = {'material_result': MATERIAL / 'OUTPUT/RESULT.json',
        'material_terminal': MATERIAL / 'TERMINAL.json',
        'normal_result': NORMAL / 'RESULT.json',
        'normal_terminal': NORMAL / 'TERMINAL.json',
        'packet': VIEW / 'PREFIX_PACKET.json',
        'constructor': VIEW / 'CONSTRUCTOR_METADATA.json',
        'view': VIEW / 'FEATURE_VIEW.json',
        'view_result': VIEW / 'RESULT.json',
        'view_terminal': VIEW / 'TERMINAL.json',
        'old_gather_plan': OLD_GATHER / 'PLAN.json',
        'fit_span_result': FIT_SPANS / 'RESULT.json',
        'later_span_result': LATER_SPANS / 'RESULT.json'}
    refs = {name: ref(path) for name, path in paths.items()}
    material, normal, old_plan = (read(refs[name]) for name in
        ('material_result', 'normal_result', 'old_gather_plan'))
    refs['material_manifest'] = material['manifest']
    need(refs['material_manifest'] == ref(MATERIAL / 'OUTPUT/MANIFEST.json'),
         'MATERIAL_MANIFEST_REF')
    refs['normal_base'] = normal['artifacts']['EXTENDED_BASE_NORMALIZATION']
    refs['normal_ta'] = normal['artifacts']['TA_INPUT_NORMALIZATION']
    refs['normal_units'] = normal['artifacts']['NEW_UNITS_PROFILE']
    for key in ('normal_base', 'normal_ta', 'normal_units'):
        need(ref(refs[key]['path']) == refs[key], 'NORMAL_ARTIFACT:' + key)
    for key in ('provider_plan', 'provider_worker', 'old_physical_plan'):
        refs[key] = old_plan['inputs'][key]
        need(ref(refs[key]['path']) == refs[key], 'OLD_INPUT:' + key)
    return refs


def checked_dependencies(plan):
    refs = plan['inputs']
    material, manifest = read(refs['material_result']), read(refs['material_manifest'])
    normal_result = read(refs['normal_result'])
    packet, constructor, view = (read(refs[k]) for k in ('packet', 'constructor', 'view'))
    base, ta, units = (read(refs[k]) for k in ('normal_base', 'normal_ta', 'normal_units'))
    material_terminal, normal_terminal, view_terminal = (read(refs[k]) for k in
        ('material_terminal', 'normal_terminal', 'view_terminal'))
    view_result = read(refs['view_result'])
    need(material_terminal['status'] == normal_terminal['status'] ==
         view_terminal['status'] == 'COMPLETE'
         and all(t['exit_code'] == 0 for t in
                 (material_terminal, normal_terminal, view_terminal))
         and material_terminal['result'] == refs['material_result']
         and normal_terminal['result'] == refs['normal_result']
         and view_terminal['result'] == refs['view_result']
         and material_terminal.get('source_unchanged') is True
         and view_terminal.get('source_unchanged') is True,
         'PREDECESSOR_TERMINALS')
    need(material['status'] == manifest['status'] == normal_result['status'] == 'COMPLETE'
         and view_result['status'] == 'COMPLETE'
         and material['manifest'] == refs['material_manifest']
         and material['surfaces'] == manifest['surfaces']
         and manifest['span_reuse_result'] == refs['later_span_result']
         and normal_result['inputs']['materialized_result'] == refs['material_result']
         and normal_result['artifacts']['EXTENDED_BASE_NORMALIZATION'] == refs['normal_base']
         and normal_result['artifacts']['TA_INPUT_NORMALIZATION'] == refs['normal_ta']
         and normal_result['artifacts']['NEW_UNITS_PROFILE'] == refs['normal_units']
         and normal_result['source_unchanged'] is True
         and material['fit_rows'] == FIT_ROWS
         and material['later_parent_start'] == LATER_START
         and material['later_rows'] == LATER_ROWS
         and material['cutoff_ns'] == 1735603200000000000
         and material['test_data_used'] is normal_result['test_data_used'] is False
         and material['model_forwards'] == material['optimizer_steps'] ==
             normal_result['model_forwards'] == 0,
         'PREDECESSOR_RESULTS')
    need(packet['base_normalization'] == base
         and packet['unobserved_units_profile'] == units
         and view['prefix_normalization']['binding'] == refs['packet']
         and view['constructor_metadata']['binding'] == refs['constructor']
         and view['ema_delta_input_extension']['physical_input_result'] == refs['material_result']
         and view['ema_delta_input_extension']['sidecars'] == plan['ema_sidecars']
         and view['ema_delta_input_extension']['execution_authorized'] is False
         and view['prefix']['rows'] == FIT_ROWS
         and view['prefix']['cutoff_ns'] == 1727654400000000000
         and view['full_axes']['rows'] == 118249
         and view['physical_widths'] == {'signal': 318, 'ctx_cont': 132,
                                         'ctx_cat': 1, 'mtf': 256}
         and view['max_batch_rows'] == BATCH
         and normal_result['base_contract_sha256'] == base['contract_sha256']
         and normal_result['ta_contract_sha256'] == ta['contract_sha256']
         and normal_result['units_contract_sha256'] == units['contract_sha256']
         and view['prefix_normalization']['ta_base_normalization_sha256'] == ta['contract_sha256'],
         'NEW_VIEW_NORMALIZATION_BINDING')
    views._require_view(view, unobserved_units_profile=units)
    need(ta == norm.project_ta_input_normalization_contract(
         base, unobserved_units_profile=units), 'TA_NORMALIZATION_RECONSTRUCTION')
    need(constructor['seq_input_dim'] == 318
         and constructor['multi_tf']['m5_seq_dim'] == 256, 'CONSTRUCTOR_WIDTHS')
    fit_result, later_result = (read(refs[k]) for k in
        ('fit_span_result', 'later_span_result'))
    need(fit_result['status'] == 'COMPLETE_INPUT_EXTENSION_ONLY'
         and later_result['status'] == 'COMPLETE'
         and fit_result['source_unchanged'] is later_result['source_unchanged'] is True
         and set(fit_result['surfaces']) == set(later_result['surfaces']) == set(LANES),
         'OLD_SPAN_RESULTS')
    for lane in LANES:
        need(fit_result['surfaces'][lane]['artifact'] == plan['fit_sidecars'][lane]
             and later_result['surfaces'][lane]['artifact'] == plan['later_sidecars'][lane]
             and manifest['surfaces'][lane]['artifact'] == plan['ema_sidecars'][lane]
             and manifest['surfaces'][lane]['span_artifact'] == plan['later_sidecars'][lane]
             and manifest['surfaces'][lane]['old_slope_parity_rows'] ==
                 manifest['surfaces'][lane]['rows'], 'SIX_SIDECAR_BINDINGS:' + lane)
    return packet, view, ta, material, manifest, fit_result, later_result


def build_plan():
    refs = input_refs()
    material, fit_result, later_result = (read(refs[k]) for k in
        ('material_result', 'fit_span_result', 'later_span_result'))
    plan = {'schema_version': 'gx1_ta_ema_delta_full_gather_plan_v1',
        'stage': STAGE, 'output_root': str(ROOT),
        'source_base_commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        'source_owners': {name: ref(REPO / name) for name in SOURCE_PATHS},
        'inputs': refs,
        'fit_sidecars': {lane: fit_result['surfaces'][lane]['artifact'] for lane in LANES},
        'later_sidecars': {lane: later_result['surfaces'][lane]['artifact'] for lane in LANES},
        'ema_sidecars': {lane: material['surfaces'][lane]['artifact'] for lane in LANES},
        'scope': dict(SCOPE),
        'execution': {'runner': 'scripts/gx1_capped_run.sh', 'job_class': 'audit',
            'memory': '4G', 'swap': '512M', 'timeout_seconds': WALL_SECONDS,
            'kill_after_seconds': 10,
            'module': 'gx1.audit.qualify_ta_ema_delta_full_gather_v1'}}
    checked_dependencies(plan)
    return plan


def inspect(plan_sha, *, execute):
    plan_ref = ref(PLAN)
    need(plan_ref['sha256'] == plan_sha and not ROOT.exists(), 'PLAN_OR_ONE_USE_ROOT')
    plan = read(plan_ref)
    expected = build_plan()
    expected['source_base_commit'] = plan.get('source_base_commit')
    need(plan == expected and plan['scope'] == SCOPE, 'PLAN_IDENTITY')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=REPO, text=True).strip()
    clean = subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True) == ''
    ancestor = subprocess.run(['git', 'merge-base', '--is-ancestor',
        plan['source_base_commit'], head], cwd=REPO, capture_output=True).returncode == 0
    need(Path.cwd().resolve() == REPO and branch == 'work/gx1-current' and ancestor,
         'SOURCE_BRANCH_CWD')
    if not execute:
        return plan, plan_ref, None, head, clean
    need(clean, 'CLEAN_SOURCE')
    policy_ref = ref(POLICY)
    policy = read(policy_ref)
    need(all(policy.get(k) is False for k in ('training_enabled',
         'native_training_enabled', 'native_launch_authorized',
         'full_epoch_training_allowed', 'full_val_allowed')),
         'TRAINING_AND_TEST_CLOSED')
    gate = policy.get(STAGE, {})
    need(gate.get('status') == 'ADMITTED' and gate.get('plan') == plan_ref
         and gate.get('scope') == SCOPE
         and gate.get('execution_authorized') is True
         and gate.get('data_reads_authorized') is True
         and gate.get('training_authorized') is False
         and gate.get('test_authorized') is False
         and gate.get('model_forwards_authorized') == 0
         and gate.get('optimizer_steps_authorized') == 0
         and gate.get('max_invocations') == gate.get('remaining_invocations') == 1
         and gate.get('claimed_invocations') == gate.get('completed_invocations') == 0
         and gate.get('relaunch_allowed') is False, 'ONE_USE_ADMISSION')
    return plan, plan_ref, policy_ref, head, clean


def coverage_record(plan, view, axes, material_ref, populations):
    panel = overlay.cold_full_later_panel()
    positions = phase_positions()['LATER']
    need(panel['score_parent_start'] == LATER_START
         and panel['score_rows'] == LATER_ROWS
         and panel['score_end_exclusive_ns'] == 1735603200000000000,
         'LATER_PANEL')
    coverage = {'schema_version': overlay.COLD_FULL_LATER_COVERAGE_SCHEMA,
        'status': 'COMPLETE', 'qualification': material_ref,
        'view_sha256': view['contract_sha256'], 'full_axes': view['full_axes'],
        'normalization_prefix': view['prefix'],
        'score_parent_start': LATER_START, 'score_rows': LATER_ROWS,
        'score_start_ns': panel['score_start_ns'],
        'score_end_exclusive_ns': panel['score_end_exclusive_ns'],
        'batch_size': BATCH, 'full_batches': panel['full_batches'],
        'tail_rows': panel['tail_rows'],
        'selected_parent_rows_sha256': axis_sha(axes['parent_rows'][positions]),
        'selected_decision_ns_sha256': axis_sha(axes['decision_ns'][positions]),
        'physical_populations': populations,
        'physical_sidecars': plan['later_sidecars']}
    overlay._require_later_coverage(view, coverage, axes, populations,
                                    sidecar_refs=plan['later_sidecars'])
    return coverage


def _load_payloads(bindings, keys):
    payloads = {}
    for lane in LANES:
        binding = bindings[lane]
        need(ref(binding['path']) == binding, 'SIDECAR_FILE:' + lane)
        with np.load(binding['path'], allow_pickle=False) as archive:
            need(set(archive.files) == set(keys), 'SIDECAR_MEMBERS:' + lane)
            payload = {key: np.array(archive[key], copy=True, order='C') for key in keys}
        for array in payload.values():
            array.setflags(write=False)
        payloads[lane] = payload
    return payloads


def _ema_bundle(plan, material, manifest):
    return {'result_ref': plan['inputs']['material_result'], 'result': material,
        'manifest_ref': plan['inputs']['material_manifest'], 'manifest': manifest,
        'payloads': _load_payloads(plan['ema_sidecars'],
                                   ('row_indices', 'times_ns', 'delta_5_atr'))}


def _shape(name, batch, width):
    if name == 'seq_x':
        return (batch, 96, width)
    if name.startswith('seq_'):
        return (batch, WINDOW_ROWS[name[4:].upper()], width)
    return (batch, width)


def _hash_update(digest, value, dtype):
    digest.update(np.ascontiguousarray(value, dtype=dtype).tobytes())


def gather_phase(name, *, view, normal, axes, seq, current, bound, started):
    """Hash every raw and normalized float32 cell in deterministic parent order."""
    parent_hash = hashlib.sha256()
    decision_hash = hashlib.sha256()
    bar_hash = hashlib.sha256()
    category_hash = hashlib.sha256()
    categories = [0, 0, 0, 0]
    streams = {surface: {'raw': hashlib.sha256(), 'normalized': hashlib.sha256()}
               for surface in SURFACES}
    rows_seen = batches = normalizations = 0
    for selection in batch_slices(name):
        need(time.monotonic() - started < WALL_SECONDS - 30, 'TIME_BUDGET')
        rows, issues = axes['parent_rows'][selection], axes['decision_ns'][selection]
        raw = views.gather_native_direction_feature_batch(
            view, parent_rows=axes['parent_rows'], bar_start_ns=axes['bar_start_ns'],
            decision_ns=axes['decision_ns'], entry_sequence_source=seq,
            snap_x=current['snap'], ctx_cont=current['ctx_cont'], ctx_cat=current['ctx_cat'],
            multi_tf_sources=bound.mtf_sources, gc_surfaces=None, gc_layout=None,
            batch_parent_rows=rows, unobserved_units_profile=view['unobserved_units_profile'],
            anchor_span_inputs=bound)
        need(raw['witness']['parent_rows'] == rows.tolist()
             and raw['witness']['decision_ns'] == issues.tolist()
             and set(raw['features']) == set(SURFACES) | {'ctx_cat'},
             'NATIVE_BATCH_WITNESS:' + name)
        cat = raw['features']['ctx_cat']
        need(cat.shape == (len(rows), 1) and cat.dtype == np.int64
             and np.isin(cat, (0, 1, 2, 3)).all(), 'CATEGORY_DOMAIN')
        for index in range(4):
            categories[index] += int(np.count_nonzero(cat == index))
        _hash_update(category_hash, cat, '<i8')
        _hash_update(parent_hash, rows, '<i8')
        _hash_update(decision_hash, issues, '<i8')
        _hash_update(bar_hash, axes['bar_start_ns'][selection], '<i8')
        for key, surface in SURFACES.items():
            stats = normal['surfaces'][surface]
            x = raw['features'][key]
            need(x.shape == _shape(key, len(rows), len(stats['field_names']))
                 and x.dtype == np.float32 and np.isfinite(x).all(),
                 'RAW_SURFACE:' + name + ':' + key)
            z = norm.apply_surface_normalization(x, stats)
            need(z.shape == x.shape and z.dtype == np.float32
                 and np.isfinite(z).all(),
                 'NORMALIZED_SURFACE:' + name + ':' + key)
            _hash_update(streams[key]['raw'], x, '<f4')
            _hash_update(streams[key]['normalized'], z, '<f4')
            normalizations += 1
        rows_seen += len(rows)
        batches += 1
    start, stop = PHASES[name]
    expected_rows = stop - start
    expected_batches = (expected_rows + BATCH - 1) // BATCH
    expected_parent = axis_sha(axes['parent_rows'][start:stop])
    expected_decision = axis_sha(axes['decision_ns'][start:stop])
    need(rows_seen == expected_rows and batches == expected_batches
         and normalizations == expected_batches * len(SURFACES)
         and parent_hash.hexdigest() == expected_parent
         and decision_hash.hexdigest() == expected_decision
         and sum(categories) == expected_rows, 'EXACT_PHASE_COMPLETION:' + name)
    return {'rows': rows_seen, 'batches': batches, 'tail_rows': expected_rows % BATCH,
        'normalization_calls': normalizations,
        'parent_rows_sha256': expected_parent,
        'bar_start_ns_sha256': bar_hash.hexdigest(),
        'decision_ns_sha256': expected_decision,
        'ctx_cat_sha256': category_hash.hexdigest(),
        'category_counts': categories,
        'surfaces': {key: {'shape_per_row': list(_shape(key, 1,
            len(normal['surfaces'][surface]['field_names']))[1:]),
            'raw_sha256': digests['raw'].hexdigest(),
            'normalized_sha256': digests['normalized'].hexdigest()}
            for key, surface in SURFACES.items()
            for digests in (streams[key],)}}


def publish(name, value):
    data = (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    need(len(data) < 2 * 1024**2, 'JSON_BUDGET')
    _publish_bytes(ROOT / name, data)
    return ref(ROOT / name)


def bookend(plan, plan_ref, policy_ref, head, file_stats, provider):
    need(ref(PLAN) == plan_ref and ref(POLICY) == policy_ref
         and subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip() == head
         and subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True) == ''
         and all(ref(REPO / name) == binding for name, binding in plan['source_owners'].items())
         and all(ref(binding['path']) == binding for binding in plan['inputs'].values())
         and all(ref(binding['path']) == binding for bucket in
                 ('fit_sidecars', 'later_sidecars', 'ema_sidecars')
                 for binding in plan[bucket].values())
         and all(provider.regular(path) == value for path, value in file_stats.items()),
         'SOURCE_INPUT_BOOKEND')


def run(plan_sha):
    started = time.monotonic()
    cap = require_capped_cpu_audit_execution()
    require_capped_lock_ancestry()
    need(cap['memory_max'] == cap['memory_high'] == 4 * 1024**3
         and cap['swap'] == 512 * 1024**2, 'CAP')
    plan, plan_ref, policy_ref, head, _ = inspect(plan_sha, execute=True)
    need(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), 'AMBIENT_TIMER')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    ROOT.mkdir()
    claim = publish('CLAIM.json', {'plan': plan_ref, 'policy': policy_ref,
        'source_commit': head, 'scope': SCOPE, 'cap': cap})
    old_alarm = signal.getsignal(signal.SIGALRM)
    groups = None
    provider = None
    maps = []
    status = 'FAILED_CONSUMED'
    result_ref = coverage_ref = None
    try:
        def expired(*_):
            raise TimeoutError('TA_EMA_DELTA_FULL_GATHER_TIMEOUT')
        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, max(1, WALL_SECONDS - 30))
        import torch
        torch.nn.Module.__init__ = lambda *a, **k: need(False, 'MODEL_CONSTRUCTION_FORBIDDEN')
        torch.load = lambda *a, **k: need(False, 'CHECKPOINT_READ_FORBIDDEN')
        torch.optim.Optimizer.__init__ = lambda *a, **k: need(False, 'OPTIMIZER_FORBIDDEN')
        from gx1.audit import qualify_market_state_mature_prefix_normalization_v1 as mature
        from gx1.contracts.entry_exit_feature_surface_v1 import load_m1_feature_surface_times
        provider = mature._provider_module()
        info = provider.validate_metadata(read(plan['inputs']['provider_plan']))
        old_gather_plan = read(plan['inputs']['old_gather_plan'])
        old_physical_plan = read(plan['inputs']['old_physical_plan'])
        need(old_gather_plan['inputs']['old_physical_plan'] == plan['inputs']['old_physical_plan']
             and old_gather_plan['inputs']['provider_plan'] == plan['inputs']['provider_plan']
             and old_gather_plan['inputs']['provider_worker'] == plan['inputs']['provider_worker']
             and provider.regular(old_physical_plan['local_clock_path']) ==
                 old_physical_plan['local_clock_stat']
             and old_physical_plan['local_clock_path'] == info['source']['output_parquet'],
             'OLD_PHYSICAL_PROVIDER')
        packet, view, normal, material, manifest, fit_result, later_result = checked_dependencies(plan)
        axes, _ = provider.read_axes(read(plan['inputs']['provider_plan']), info, np)
        need(set(axes) == {'parent_rows', 'bar_start_ns', 'decision_ns'}, 'AXIS_MEMBERS')
        for key, value in axes.items():
            need(value.dtype == np.int64 and value.shape == (118249,)
                 and axis_sha(value) == view['full_axes'][key + '_sha256'],
                 'FULL_AXIS:' + key)
            value.setflags(write=False)
        need(np.array_equal(axes['bar_start_ns'] + 300 * 10**9, axes['decision_ns'])
             and axis_sha(axes['parent_rows'][:FIT_ROWS]) == view['prefix']['parent_rows_sha256']
             and axis_sha(axes['decision_ns'][:FIT_ROWS]) == view['prefix']['decision_ns_sha256']
             and np.all(axes['decision_ns'][:FIT_ROWS] < view['prefix']['cutoff_ns'])
             and np.all(axes['decision_ns'][LATER_START:LATER_STOP] <
                        material['cutoff_ns'])
             and axes['decision_ns'][LATER_START] >= 1727740800000000000,
             'SELECTED_AXES_AND_CLOCKS')
        groups = provider.map_reuse(info['reuse'], np)
        source, current = groups['source050'], groups['current051']
        times = load_m1_feature_surface_times(Path(old_physical_plan['local_clock_path']),
            context='EMA_DELTA_FULL_GATHER_CLOCK_ONLY', expected_bar_seconds=300).asi8
        positions = np.searchsorted(times, axes['bar_start_ns']).astype(np.int64)
        need(np.all(positions < len(times))
             and np.array_equal(times[positions], axes['bar_start_ns']),
             'EXACT_LOCAL_POSITIONS')
        for array in (*axes.values(), times, positions):
            array.setflags(write=False)
        seq = {'signal': source['signal'], 'times_ns': times,
               'sample_positions': positions, 'audit': info['audit']}
        file_stats = dict(info['reuse']['scratch_files'])
        file_stats[old_physical_plan['local_clock_path']] = old_physical_plan['local_clock_stat']
        values, clocks = {}, {}
        for lane in overlay.TIMEFRAMES:
            for key in ('feats', 'ts'):
                binding = old_physical_plan['caches'][lane][key]
                need(ref(binding['path']) == binding, 'OLD_MTF_SHA:' + lane + ':' + key)
                file_stats[binding['path']] = provider.regular(binding['path'])
            values[lane] = np.load(old_physical_plan['caches'][lane]['feats']['path'],
                                   mmap_mode='r', allow_pickle=False)
            clocks[lane] = np.load(old_physical_plan['caches'][lane]['ts']['path'],
                                   mmap_mode='r', allow_pickle=False)
            maps.extend((values[lane], clocks[lane]))
        old_packet = read(old_gather_plan['inputs']['packet'])
        old_base = old_packet['parent_prefix_normalization']['base_normalization']
        ema = _ema_bundle(plan, material, manifest)
        common = {'view': view, 'entry_sequence_source': seq,
            'snap_x': current['snap'], 'ctx_cont': current['ctx_cont'],
            'ctx_cat': current['ctx_cat'],
            'parent_signal_names': old_base['surfaces']['signal']['field_names'],
            'parent_mtf_names': old_base['surfaces']['mtf_m5']['field_names'],
            'parent_mtf_values': values, 'parent_mtf_times': clocks,
            'unobserved_units_profile': packet['unobserved_units_profile'],
            'ema_delta_bundle': ema}
        fit_payloads = _load_payloads(plan['fit_sidecars'],
            ('row_indices', 'times_ns', 'native_bar_positions', 'anchor_span_bars'))
        fit_bound = overlay.build_anchor_span_inputs(
            **common, span_payloads=fit_payloads,
            physical_populations={lane: fit_result['surfaces'][lane]['population']
                                  for lane in LANES})
        later_payloads = _load_payloads(plan['later_sidecars'],
            ('row_indices', 'times_ns', 'native_bar_positions', 'anchor_span_bars'))
        populations = {lane: later_result['surfaces'][lane]['population'] for lane in LANES}
        coverage = coverage_record(plan, view, axes, plan['inputs']['material_result'], populations)
        later_bound = overlay.build_anchor_span_inputs(
            **common, span_payloads=later_payloads, physical_populations=populations,
            later_coverage=coverage, later_axes=axes,
            later_sidecar_refs=plan['later_sidecars'])
        print('EMA_DELTA_INPUTS_BOUND: 47063 selected rows, 2943 batches, no model',
              flush=True)
        fit = gather_phase('FIT', view=view, normal=normal, axes=axes,
            seq=seq, current=current, bound=fit_bound, started=started)
        later = gather_phase('LATER', view=view, normal=normal, axes=axes,
            seq=seq, current=current, bound=later_bound, started=started)
        need(fit['rows'] + later['rows'] == SCOPE['selected_parent_rows']
             and fit['batches'] + later['batches'] == SCOPE['feature_batches']
             and fit['normalization_calls'] + later['normalization_calls'] ==
                 SCOPE['normalization_calls'], 'FULL_COMPLETION')
        bookend(plan, plan_ref, policy_ref, head, file_stats, provider)
        result_ref = publish('RESULT.json', {'status': 'COMPLETE',
            'plan': plan_ref, 'claim': claim, 'policy': policy_ref,
            'source_commit': head, 'source_unchanged': True,
            'view': plan['inputs']['view'], 'normalization': plan['inputs']['normal_ta'],
            'materialized_result': plan['inputs']['material_result'],
            'full_raw_and_normalized_selected_values': True,
            'scope': SCOPE, 'fit': fit, 'later': later,
            'gap_rows_excluded': LATER_START - FIT_ROWS,
            'target_arrays_decoded': 0, 'model_forwards': 0,
            'normalization_fits': 0, 'optimizer_steps': 0,
            'test_data_used': False, 'model_baseline_qualified': False,
            'predictive_value_claimed': False, 'economic_edge_proven': False,
            'elapsed_seconds': time.monotonic() - started})
        coverage_ref = publish('COVERAGE.json', coverage)
        status = 'COMPLETE'
    except BaseException as exc:
        publish('FAILED.json', {'status': 'FAILED_CONSUMED', 'plan': plan_ref,
            'claim': claim, 'error_type': type(exc).__name__,
            'error': str(exc)[:2000], 'result': result_ref,
            'coverage': coverage_ref})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_alarm)
        publish('TERMINAL.json', {'status': status,
            'exit_code': 0 if status == 'COMPLETE' else 1,
            'plan': plan_ref, 'claim': claim, 'result': result_ref,
            'coverage': coverage_ref, 'source_commit': head,
            'source_unchanged': subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip() == head,
            'relaunch_allowed': False})
        if provider is not None:
            for array in maps:
                provider.flush_pages(array)
        if groups is not None and provider is not None:
            for group in groups.values():
                for array in group.values():
                    provider.flush_pages(array)
    return result_ref


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--build-plan', action='store_true')
    mode.add_argument('--preflight', action='store_true')
    parser.add_argument('--plan-sha256')
    args = parser.parse_args()
    if args.build_plan:
        need(not args.preflight and args.plan_sha256 is None and not PLAN.exists()
             and not ROOT.exists(), 'BUILD_CLI')
        _publish_bytes(PLAN, (json.dumps(build_plan(), sort_keys=True,
            indent=2, allow_nan=False) + '\n').encode())
        print(json.dumps(ref(PLAN)))
        return
    need(args.plan_sha256 is not None, 'PLAN_SHA_REQUIRED')
    if args.preflight:
        plan, plan_ref, _, head, clean = inspect(args.plan_sha256, execute=False)
        print(json.dumps({'status': 'PREFLIGHT_PASS' if clean else 'PREFLIGHT_SOURCE_DIRTY',
            'plan': plan_ref, 'source_commit': head,
            'scope': plan['scope'], 'claim_created': False}, sort_keys=True))
        return
    run(args.plan_sha256)


if __name__ == '__main__':
    main()
