"""Reuse qualified Fold2 TRAIN and Apr-Jun score inputs for one TA campaign."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from gx1.audit import qualify_market_state_mature_prefix_normalization_v1 as mature
from gx1.contracts import entry_native_direction_feature_view_v1 as views
from gx1.contracts.entry_anchor_span_feature_inputs_v1 import (
    build_anchor_span_inputs, later_panel,
)
from gx1.contracts.entry_exit_feature_surface_v1 import load_m1_feature_surface_times
from gx1.contracts.market_state_ta_decoder_view_rebind_v1 import rebind_decoder_only
from gx1.contracts.market_state_ta_fold2_m5_fit_v1 import (
    INITIAL_PLAN_REF, require_reused_input_refs,
)
from gx1.contracts.market_state_time_query_native_F_objective_v1 import (
    require_m5_equal_observation_axes,
)

REPO = Path('/home/andre2/src/GX1_CURRENT')
PHYSICAL = Path('/home/andre2/GX1_RUNS/TA_FOLD2_APR_JUN_INPUTS_20261117_001')
ISSUE = Path('/home/andre2/GX1_RUNS/TA_INTEGRATED_APR_JUN_ISSUE_INPUTS_20261117_002')
BASE_VIEW = REPO / 'configs/research/TA_FOLD2_APR_JUN_FEATURE_VIEW_20261117_001.json'
MODEL_VIEW = REPO / 'configs/research/TA_FOLD2_APR_JUN_INTEGRATED_VIEW_20261117_001.json'
FIT_REVIEW = REPO / 'docs/TA_INTEGRATED_FOLD2_FIT_INPUTS_REVIEW_20261116.json'
TF = ('M5', 'M15', 'H1', 'H4', 'D1')
TRAIN_ROWS = 47338
SCORE_ROWS = 17536


def need(ok, reason):
    if not bool(ok):
        raise RuntimeError('TA_APR_JUN_CAMPAIGN_INPUT_' + reason)


def ref(path):
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), 'FILE')
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path), 'sha256': digest.hexdigest()}


def read(binding):
    need(ref(binding['path']) == binding, 'REF')
    return json.loads(Path(binding['path']).read_text())


def axis_hash(value):
    return hashlib.sha256(value.astype('<i8', copy=False).tobytes()).hexdigest()


def _tape(binding, parents, issues):
    need(ref(binding['path']) == {key: binding[key] for key in ('path', 'sha256')},
         'TAPE_REF')
    with np.load(binding['path'], allow_pickle=False) as archive:
        need(set(archive.files) == {'parent_rows', 'issue_ns', 'm1_mid_ohlc_bps',
                                    'm1_observed', 'structure_hit', 'structure_observed'},
             'TAPE_MEMBERS')
        value = {key: archive[key] for key in archive.files}
    n = len(parents)
    need(np.array_equal(value['parent_rows'], parents)
         and np.array_equal(value['issue_ns'], issues)
         and value['m1_mid_ohlc_bps'].dtype == np.float32
         and value['m1_mid_ohlc_bps'].shape == (n, 5, 4)
         and value['m1_observed'].dtype == np.bool_
         and value['m1_observed'].shape == (n, 5)
         and value['structure_hit'].dtype == value['structure_observed'].dtype == np.bool_
         and value['structure_hit'].shape == value['structure_observed'].shape == (n, 2)
         and np.isfinite(value['m1_mid_ohlc_bps']).all()
         and np.all(value['m1_mid_ohlc_bps'][~value['m1_observed']] == 0)
         and np.all(value['m1_mid_ohlc_bps'][:, -1, 3] == 0)
         and np.all(~value['structure_hit'] | value['structure_observed']),
         'TAPE_CONTENT')
    for array in value.values():
        array.setflags(write=False)
    return value


def prepare(*, device: torch.device):
    """Read only immutable inputs. Caller owns admission and releases mappings."""
    need(isinstance(device, torch.device) and device.type in ('cpu', 'cuda'), 'DEVICE')
    inputs = require_reused_input_refs(INITIAL_PLAN_REF)
    prior = read(inputs['later_plan'])
    train_result = read(inputs['physical_result'])
    score_result = read(ref(PHYSICAL / 'RESULT.json'))
    coverage = read(ref(PHYSICAL / 'COVERAGE.json'))
    score_terminal = read(ref(PHYSICAL / 'TERMINAL.json'))
    issue_result = read(ref(ISSUE / 'RESULT.json'))
    issue_terminal = read(ref(ISSUE / 'TERMINAL.json'))
    need(score_result['status'] == 'COMPLETE_APR_JUN_TA_SCORE_INPUTS_NO_MODEL'
         and score_result['source_unchanged'] is True
         and coverage['status'] == score_terminal['status'] == 'COMPLETE'
         and score_terminal['exit_code'] == 0
         and score_terminal['result'] == ref(PHYSICAL / 'RESULT.json')
         and issue_result['status'] == 'COMPLETE_APR_JUN_M1_STRUCTURE_INPUTS_NO_MODEL'
         and issue_result['source_unchanged'] is True
         and issue_terminal['status'] == 'COMPLETE' and issue_terminal['exit_code'] == 0
         and issue_terminal['result'] == ref(ISSUE / 'RESULT.json')
         and all(x['test_data_used'] is False for x in (score_result, issue_result)),
         'QUALIFIED_SCORE_RECEIPTS')
    base_view, view = read(ref(BASE_VIEW)), read(ref(MODEL_VIEW))
    packet, constructor = read(inputs['packet']), read(inputs['constructor'])
    need(base_view['contract_sha256'] == coverage['view_sha256']
         and view['prefix'] == base_view['prefix']
         and view['full_axes'] == base_view['full_axes']
         and view['prefix_normalization'] == base_view['prefix_normalization']
         and view['source_bindings'] == base_view['source_bindings']
         and view['full_ta_feature_profile'] is True, 'VIEW_BINDING')
    provider_plan = read(inputs['provider_plan'])
    provider = mature._provider_module()
    info = provider.validate_metadata(provider_plan)
    axes, _ = provider.read_axes(provider_plan, info, np)
    need(set(axes) == {'parent_rows', 'bar_start_ns', 'decision_ns'}
         and all(axis_hash(axes[k]) == view['full_axes'][k + '_sha256'] for k in axes),
         'FULL_AXES')
    for value in axes.values():
        value.setflags(write=False)
    require_m5_equal_observation_axes(axes['parent_rows'][:TRAIN_ROWS],
                                      axes['decision_ns'][:TRAIN_ROWS], fold_id=2)
    panel = later_panel(fold_id=3)
    selected = (panel['score_parent_start']
                + np.asarray(panel['batch_indices'], dtype=np.int64)[:, None] * 16
                + np.arange(16, dtype=np.int64)).reshape(-1)
    need(len(selected) == SCORE_ROWS
         and axis_hash(axes['parent_rows'][selected]) == coverage['selected_parent_rows_sha256']
         and axis_hash(axes['decision_ns'][selected]) == coverage['selected_decision_ns_sha256']
         and issue_result['score_rows'] == SCORE_ROWS, 'SCORE_AXIS')
    groups = provider.map_reuse(info['reuse'], np)
    source, current = groups['source050'], groups['current051']
    runtime = prior['replay_plan']
    times = load_m1_feature_surface_times(Path(runtime['local_clock_path']),
        context='TA_APR_JUN_CAMPAIGN_FULL_PARENT_CLOCK', expected_bar_seconds=300).asi8
    positions = np.searchsorted(times, axes['bar_start_ns']).astype(np.int64)
    need(np.all(positions < len(times))
         and np.array_equal(times[positions], axes['bar_start_ns']), 'LOCAL_CLOCK')
    for value in (times, positions, source['signal'], current['snap'],
                  current['ctx_cont'], current['ctx_cat']):
        value.setflags(write=False)
    seq = {'signal': source['signal'], 'times_ns': times,
           'sample_positions': positions, 'audit': info['audit']}
    values, clocks, mapped = {}, {}, []
    for tf in TF:
        values[tf] = np.load(runtime['caches'][tf]['feats']['path'], mmap_mode='r',
                             allow_pickle=False)
        clocks[tf] = np.load(runtime['caches'][tf]['ts']['path'], mmap_mode='r',
                             allow_pickle=False)
        mapped.extend((values[tf], clocks[tf]))
    names = packet['parent_prefix_normalization']['base_normalization']['surfaces']
    overlays = {}
    for label, result in (('TRAIN', train_result), ('SCORE', score_result)):
        payloads = {}
        for lane in ('local', *TF):
            binding = result['surfaces'][lane]['artifact']
            need(ref(binding['path']) == binding, 'SIDECAR_' + label + '_' + lane)
            with np.load(binding['path'], allow_pickle=False) as archive:
                payloads[lane] = {key: archive[key] for key in archive.files}
        base = build_anchor_span_inputs(base_view, entry_sequence_source=seq,
            snap_x=current['snap'], ctx_cont=current['ctx_cont'], ctx_cat=current['ctx_cat'],
            parent_signal_names=names['signal']['field_names'],
            parent_mtf_names=names['mtf_m5']['field_names'],
            parent_mtf_values=values, parent_mtf_times=clocks,
            span_payloads=payloads,
            physical_populations=(coverage['physical_populations'] if label == 'SCORE'
                                  else {lane: item['population']
                                        for lane, item in result['surfaces'].items()}),
            unobserved_units_profile=None,
            **({'later_coverage': coverage, 'later_axes': axes}
               if label == 'SCORE' else {}))
        overlays[label] = rebind_decoder_only(base_view, view, base)
    fit_ref = json.loads(FIT_REVIEW.read_text())['input_archive']
    fit_tape = _tape(fit_ref, axes['parent_rows'][:TRAIN_ROWS],
                     axes['decision_ns'][:TRAIN_ROWS])
    score_tape = _tape(issue_result['input_archive'], axes['parent_rows'][selected],
                       axes['decision_ns'][selected])
    need(issue_result['issue_axis_sha256'] == axis_hash(score_tape['issue_ns']),
         'SCORE_TAPE_AXIS_SHA')

    def features(label, rows, issues):
        need(label in overlays and len(rows) == len(issues), 'GATHER_REQUEST')
        batch = views.gather_native_direction_feature_batch(view,
            parent_rows=axes['parent_rows'], bar_start_ns=axes['bar_start_ns'],
            decision_ns=axes['decision_ns'], entry_sequence_source=seq,
            snap_x=current['snap'], ctx_cont=current['ctx_cont'], ctx_cat=current['ctx_cat'],
            multi_tf_sources=overlays[label].mtf_sources, gc_surfaces=None,
            gc_layout=None, batch_parent_rows=rows,
            unobserved_units_profile=None, anchor_span_inputs=overlays[label])
        need(batch['witness']['parent_rows'] == rows.tolist()
             and batch['witness']['decision_ns'] == issues.tolist(), 'GATHER_AXIS')
        return {key: torch.from_numpy(value).to(device) for key, value in batch['features'].items()}

    def release():
        for group in groups.values():
            for value in group.values():
                provider.flush_pages(value)
        for value in mapped:
            provider.flush_pages(value)

    return {'view': view, 'base_view': base_view, 'packet': packet,
            'constructor': constructor, 'axes': axes, 'selected': selected,
            'fit_tape': fit_tape, 'score_tape': score_tape,
            'features': features, 'release': release,
            'input_refs': {'physical_result': ref(PHYSICAL / 'RESULT.json'),
                           'coverage': ref(PHYSICAL / 'COVERAGE.json'),
                           'score_tape_result': ref(ISSUE / 'RESULT.json'),
                           'fit_review': ref(FIT_REVIEW),
                           'model_view': ref(MODEL_VIEW), 'base_view': ref(BASE_VIEW)}}
