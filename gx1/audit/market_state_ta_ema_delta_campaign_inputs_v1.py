"""Exact 318/256 cold FIT and later input adapter for one M5 decision campaign.

The completed full gather owns the physical value proof. This adapter only
reconstructs the same immutable inputs and exposes the existing fit/score
context. It fits no statistics and constructs no model.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from gx1.audit import market_state_ta_apr_jun_campaign_inputs_v1 as legacy
from gx1.audit import qualify_market_state_mature_prefix_normalization_v1 as mature
from gx1.audit import qualify_ta_ema_delta_full_gather_v1 as gathered
from gx1.contracts import entry_anchor_span_feature_inputs_v1 as overlay
from gx1.contracts import entry_native_direction_feature_view_v1 as views
from gx1.contracts.entry_exit_feature_surface_v1 import load_m1_feature_surface_times
from gx1.contracts.market_state_ta_integrated_paired_objective_v1 import contract as paired_contract
from gx1.contracts.market_state_time_query_native_F_objective_v1 import require_endpoint_priors

REPO = gathered.REPO
FIT_ROWS, LATER_START, LATER_STOP = gathered.FIT_ROWS, gathered.LATER_START, gathered.LATER_STOP
LATER_ROWS = gathered.LATER_ROWS
TARGETS = Path('/home/andre2/GX1_RUNS/TA_M5_COST_PATH_TARGETS_20261117_001/FIT_TARGETS.npz')
PRIORS = Path('/home/andre2/GX1_RUNS/TA_M5_EMA_DELTA_FOLD1_PRIORS_20261118_001/ENDPOINT_PRIORS.json')
FIT_REVIEW = legacy.FIT_REVIEW
LANES = gathered.LANES


def need(ok, reason):
    if not bool(ok):
        raise RuntimeError('TA_EMA_DELTA_CAMPAIGN_INPUT_' + reason)


def successor_input_refs():
    """All exact model-input dependencies, with no mutable latest path."""
    refs = gathered.input_refs()
    review = legacy.read(legacy.ref(FIT_REVIEW))
    archive = review['input_archive']
    tape = {key: archive[key] for key in ('path', 'sha256')}
    need(legacy.ref(tape['path']) == tape
         and Path(tape['path']).stat().st_size == archive['bytes'], 'FIT_TAPE_REF')
    extra = {
        'gather_plan': legacy.ref(gathered.PLAN),
        'gather_result': legacy.ref(gathered.ROOT / 'RESULT.json'),
        'gather_coverage': legacy.ref(gathered.ROOT / 'COVERAGE.json'),
        'gather_terminal': legacy.ref(gathered.ROOT / 'TERMINAL.json'),
        'priors': legacy.ref(PRIORS),
        'prior_result': legacy.ref(PRIORS.parent / 'RESULT.json'),
        'prior_terminal': legacy.ref(PRIORS.parent / 'TERMINAL.json'),
        'fit_review': legacy.ref(FIT_REVIEW),
        'fit_tape': tape,
        'fit_targets': legacy.ref(TARGETS),
        'target_result': legacy.ref(TARGETS.parent / 'RESULT.json'),
        'target_terminal': legacy.ref(TARGETS.parent / 'TERMINAL.json'),
    }
    need(not set(refs).intersection(extra), 'INPUT_KEY_COLLISION')
    return {**refs, **extra}


def _receipts(inputs):
    plan = legacy.read(inputs['gather_plan'])
    result = legacy.read(inputs['gather_result'])
    coverage = legacy.read(inputs['gather_coverage'])
    terminal = legacy.read(inputs['gather_terminal'])
    need(plan['schema_version'] == 'gx1_ta_ema_delta_full_gather_plan_v1'
         and plan['stage'] == gathered.STAGE and plan['output_root'] == str(gathered.ROOT)
         and plan['inputs'] == {key: inputs[key] for key in gathered.input_refs()}
         and plan['scope'] == gathered.SCOPE
         and result['status'] == terminal['status'] == coverage['status'] == 'COMPLETE'
         and result['plan'] == terminal['plan'] == inputs['gather_plan']
         and terminal['exit_code'] == 0
         and terminal['result'] == inputs['gather_result']
         and terminal['coverage'] == inputs['gather_coverage']
         and terminal['source_unchanged'] is result['source_unchanged'] is True
         and result['scope'] == gathered.SCOPE
         and result['full_raw_and_normalized_selected_values'] is True
         and result['fit']['rows'] == FIT_ROWS
         and result['fit']['batches'] == gathered.SCOPE['fit_batches']
         and result['fit']['tail_rows'] == 6
         and result['later']['rows'] == LATER_ROWS
         and result['later']['batches'] == gathered.SCOPE['later_batches']
         and result['later']['tail_rows'] == 1
         and result['gap_rows_excluded'] == LATER_START - FIT_ROWS
         and result['target_arrays_decoded'] == result['model_forwards']
             == result['normalization_fits'] == result['optimizer_steps'] == 0
         and result['test_data_used'] is False
         and result['view'] == inputs['view']
         and result['normalization'] == inputs['normal_ta']
         and result['materialized_result'] == inputs['material_result']
         and coverage['qualification'] == inputs['material_result']
         and coverage['physical_sidecars'] == plan['later_sidecars']
         and coverage['view_sha256'] == gathered.read(inputs['view'])['contract_sha256'],
         'GATHER_RECEIPTS')
    for name, phase in (('FIT', result['fit']), ('LATER', result['later'])):
        need(phase['normalization_calls'] == phase['batches'] * len(gathered.SURFACES)
             and set(phase['surfaces']) == set(gathered.SURFACES)
             and all(set(item) == {'shape_per_row', 'raw_sha256', 'normalized_sha256'}
                     for item in phase['surfaces'].values())
             and phase['parent_rows_sha256'] == (
                 coverage['normalization_prefix']['parent_rows_sha256'] if name == 'FIT'
                 else coverage['selected_parent_rows_sha256'])
             and phase['decision_ns_sha256'] == (
                 coverage['normalization_prefix']['decision_ns_sha256'] if name == 'FIT'
                 else coverage['selected_decision_ns_sha256']),
             'GATHER_PHASE_' + name)
    prior_result = legacy.read(inputs['prior_result'])
    prior_terminal = legacy.read(inputs['prior_terminal'])
    priors = legacy.read(inputs['priors'])
    need(prior_result['status'] == prior_terminal['status'] == 'COMPLETE'
         and prior_result['endpoint_priors'] == inputs['priors']
         and prior_result['fold_id'] == 1 and prior_result['fit_rows'] == FIT_ROWS
         and prior_result['model_forwards'] == prior_result['optimizer_steps'] == 0
         and prior_result['test_data_used'] is False
         and prior_terminal['exit_code'] == 0
         and prior_terminal['result'] == inputs['prior_result']
         and prior_terminal['endpoint_priors'] == inputs['priors'],
         'FOLD1_PRIOR_RECEIPTS')
    require_endpoint_priors(priors, fold_id=1, ta_only=True, dedicated_m5_side=True)
    need(paired_contract(fold_id=1)['contract_sha256'] != paired_contract()['contract_sha256'],
         'FOLD1_AUX_IDENTITY')
    target_result = legacy.read(inputs['target_result'])
    target_terminal = legacy.read(inputs['target_terminal'])
    need(target_result['status'] == 'COMPLETE_FIXED_FOLD2_APR_JUN_COST_PATH_TARGETS'
         and target_result['targets']['FIT'] == inputs['fit_targets']
         and target_result['test_data_used'] is False
         and target_terminal['status'] == 'COMPLETE'
         and target_terminal['exit_code'] == 0
         and target_terminal['result'] == inputs['target_result'],
         'COST_TARGET_RECEIPTS')
    review = legacy.read(inputs['fit_review'])
    need({key: review['input_archive'][key] for key in ('path', 'sha256')}
         == inputs['fit_tape']
         and Path(inputs['fit_tape']['path']).stat().st_size
             == review['input_archive']['bytes'], 'FIT_TAPE_RECEIPT')
    packet, view, normal, material, manifest, fit_result, later_result = (
        gathered.checked_dependencies(plan))
    need(coverage['normalization_prefix'] == view['prefix']
         and coverage['full_axes'] == view['full_axes']
         and coverage['score_parent_start'] == LATER_START
         and coverage['score_rows'] == LATER_ROWS
         and coverage['tail_rows'] == 1
         and view['prefix_normalization']['binding'] == inputs['packet']
         and view['constructor_metadata']['binding'] == inputs['constructor']
         and view['prefix_normalization']['ta_base_normalization_sha256']
             == normal['contract_sha256']
         and packet['unobserved_units_profile'] == view['unobserved_units_profile'],
         'SUCCESSOR_VIEW')
    return plan, result, coverage, packet, view, material, manifest, fit_result, later_result, priors


def prepare(*, device: torch.device, inputs: dict):
    """Reconstruct the qualified cold context; no model or target fit."""
    need(isinstance(device, torch.device) and device.type in ('cpu', 'cuda'), 'DEVICE')
    need(type(inputs) is dict and inputs == successor_input_refs(), 'INPUT_REFS')
    (plan, result, coverage, packet, view, material, manifest,
     fit_result, later_result, priors) = _receipts(inputs)
    provider = None
    groups = {}
    mapped = []
    released = False

    def release():
        nonlocal released
        if released:
            return
        released = True
        if provider is not None:
            for array in mapped:
                provider.flush_pages(array)
            for group in groups.values():
                for array in group.values():
                    provider.flush_pages(array)

    try:
        provider = mature._provider_module()
        info = provider.validate_metadata(gathered.read(inputs['provider_plan']))
        old_physical = gathered.read(inputs['old_physical_plan'])
        old_gather = gathered.read(inputs['old_gather_plan'])
        need(old_gather['inputs']['old_physical_plan'] == inputs['old_physical_plan']
             and old_gather['inputs']['provider_plan'] == inputs['provider_plan']
             and old_gather['inputs']['provider_worker'] == inputs['provider_worker']
             and provider.regular(old_physical['local_clock_path'])
                 == old_physical['local_clock_stat'],
             'PROVIDER_CLOCK')
        axes, _ = provider.read_axes(gathered.read(inputs['provider_plan']), info, np)
        need(set(axes) == {'parent_rows', 'bar_start_ns', 'decision_ns'}, 'AXES')
        for key, value in axes.items():
            need(value.dtype == np.int64 and value.shape == (view['full_axes']['rows'],)
                 and gathered.axis_sha(value) == view['full_axes'][key + '_sha256'],
                 'FULL_AXIS_' + key)
            value.setflags(write=False)
        selected = gathered.phase_positions()['LATER']
        need(gathered.axis_sha(axes['parent_rows'][:FIT_ROWS])
                 == result['fit']['parent_rows_sha256']
             and gathered.axis_sha(axes['decision_ns'][:FIT_ROWS])
                 == result['fit']['decision_ns_sha256']
             and gathered.axis_sha(axes['parent_rows'][selected])
                 == result['later']['parent_rows_sha256']
             and gathered.axis_sha(axes['decision_ns'][selected])
                 == result['later']['decision_ns_sha256']
             and np.array_equal(axes['bar_start_ns'] + 300 * 10**9, axes['decision_ns']),
             'SELECTED_AXES')
        with np.load(inputs['fit_targets']['path'], allow_pickle=False) as archive:
            target_parents, target_issues = archive['parent_rows'], archive['issue_ns']
        need(len(target_parents) == len(target_issues) == LATER_STOP
             and np.array_equal(axes['parent_rows'][:FIT_ROWS], target_parents[:FIT_ROWS])
             and np.array_equal(axes['decision_ns'][:FIT_ROWS], target_issues[:FIT_ROWS])
             and np.array_equal(axes['parent_rows'][selected], target_parents[LATER_START:LATER_STOP])
             and np.array_equal(axes['decision_ns'][selected], target_issues[LATER_START:LATER_STOP]),
             'TARGET_AXES')
        groups = provider.map_reuse(info['reuse'], np)
        source, current = groups['source050'], groups['current051']
        times = load_m1_feature_surface_times(Path(old_physical['local_clock_path']),
            context='EMA_DELTA_CAMPAIGN_LOCAL_CLOCK', expected_bar_seconds=300).asi8
        positions = np.searchsorted(times, axes['bar_start_ns']).astype(np.int64)
        need(np.all(positions < len(times))
             and np.array_equal(times[positions], axes['bar_start_ns']),
             'LOCAL_CLOCK')
        for array in (times, positions, source['signal'], current['snap'],
                      current['ctx_cont'], current['ctx_cat']):
            array.setflags(write=False)
        seq = {'signal': source['signal'], 'times_ns': times,
               'sample_positions': positions, 'audit': info['audit']}
        values, clocks = {}, {}
        for lane in overlay.TIMEFRAMES:
            for key in ('feats', 'ts'):
                binding = old_physical['caches'][lane][key]
                need(gathered.ref(binding['path']) == binding,
                     'MTF_REF_' + lane + '_' + key)
            values[lane] = np.load(old_physical['caches'][lane]['feats']['path'],
                                   mmap_mode='r', allow_pickle=False)
            clocks[lane] = np.load(old_physical['caches'][lane]['ts']['path'],
                                   mmap_mode='r', allow_pickle=False)
            values[lane].setflags(write=False)
            clocks[lane].setflags(write=False)
            mapped.extend((values[lane], clocks[lane]))
        old_base = packet['parent_anchor_packet']['parent_prefix_normalization']['base_normalization']
        common = {'view': view, 'entry_sequence_source': seq,
            'snap_x': current['snap'], 'ctx_cont': current['ctx_cont'],
            'ctx_cat': current['ctx_cat'],
            'parent_signal_names': old_base['surfaces']['signal']['field_names'],
            'parent_mtf_names': old_base['surfaces']['mtf_m5']['field_names'],
            'parent_mtf_values': values, 'parent_mtf_times': clocks,
            'unobserved_units_profile': packet['unobserved_units_profile'],
            'ema_delta_bundle': gathered._ema_bundle(plan, material, manifest)}
        fit_bound = overlay.build_anchor_span_inputs(
            **common,
            span_payloads=gathered._load_payloads(plan['fit_sidecars'],
                ('row_indices', 'times_ns', 'native_bar_positions', 'anchor_span_bars')),
            physical_populations={lane: fit_result['surfaces'][lane]['population']
                                  for lane in LANES})
        populations = {lane: later_result['surfaces'][lane]['population'] for lane in LANES}
        need(coverage == gathered.coverage_record(
            plan, view, axes, inputs['material_result'], populations),
            'LATER_COVERAGE_RECONSTRUCTION')
        score_bound = overlay.build_anchor_span_inputs(
            **common,
            span_payloads=gathered._load_payloads(plan['later_sidecars'],
                ('row_indices', 'times_ns', 'native_bar_positions', 'anchor_span_bars')),
            physical_populations=populations, later_coverage=coverage,
            later_axes=axes, later_sidecar_refs=plan['later_sidecars'])
        fit_tape = legacy._tape(inputs['fit_tape'],
            axes['parent_rows'][:FIT_ROWS], axes['decision_ns'][:FIT_ROWS],
            stop=FIT_ROWS)
        score_tape = legacy._tape(inputs['fit_tape'],
            axes['parent_rows'][selected], axes['decision_ns'][selected],
            start=LATER_START, stop=LATER_STOP)

        def features(label, rows, issues):
            need(label in ('TRAIN', 'SCORE')
                 and len(rows) == len(issues) > 0, 'FEATURE_REQUEST')
            bound = fit_bound if label == 'TRAIN' else score_bound
            batch = views.gather_native_direction_feature_batch(view,
                parent_rows=axes['parent_rows'], bar_start_ns=axes['bar_start_ns'],
                decision_ns=axes['decision_ns'], entry_sequence_source=seq,
                snap_x=current['snap'], ctx_cont=current['ctx_cont'],
                ctx_cat=current['ctx_cat'], multi_tf_sources=bound.mtf_sources,
                gc_surfaces=None, gc_layout=None, batch_parent_rows=rows,
                unobserved_units_profile=packet['unobserved_units_profile'],
                anchor_span_inputs=bound)
            need(batch['witness']['parent_rows'] == rows.tolist()
                 and batch['witness']['decision_ns'] == issues.tolist(),
                 'FEATURE_WITNESS')
            return {key: torch.from_numpy(value).to(device)
                    for key, value in batch['features'].items()}

        return {'view': view, 'packet': packet,
            'constructor': gathered.read(inputs['constructor']),
            'priors': priors, 'units_profile': packet['unobserved_units_profile'],
            'axes': axes, 'selected': selected, 'fit_tape': fit_tape,
            'score_tape': score_tape, 'features': features, 'release': release,
            'fit_rows': FIT_ROWS, 'score_rows': LATER_ROWS,
            'fit_fold_id': 1, 'score_fold_id': 1,
            'score_partition': 'score', 'input_refs': inputs}
    except BaseException:
        release()
        raise
