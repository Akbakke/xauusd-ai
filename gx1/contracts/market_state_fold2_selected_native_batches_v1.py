"""Reuse completed Fold 2 full-GC TRAIN inputs for two selected S/F B16 batches.

Called only inside an admitted guarded native campaign. No model or optimizer here.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from gx1.audit import qualify_market_state_mature_prefix_normalization_v1 as mature
from gx1.audit import verify_market_state_native_F_cpu_zero_step_v1 as initial
from gx1.audit import verify_market_state_time_query_zero_step_v1 as old
from gx1.contracts import entry_native_direction_feature_view_v1 as views
from gx1.contracts import market_state_native_F_private_campaign_v1 as scope


def _need(ok, reason):
    if not ok:
        raise RuntimeError('FOLD2_SELECTED_NATIVE_BATCHES_' + reason)


def load_two_batches(*, source_owners, stream=False, later_full_gc_ref=None):
    """Return view, normalization, constructor, two raw batches and provider.

    Receipts, current source bindings, five causal clocks and all 32 TRAIN axes
    are checked before any model construction by the caller.
    """
    from gx1.contracts.entry_anchor_span_feature_inputs_v1 import (
        TIMEFRAMES, build_anchor_span_inputs,
    )
    from gx1.contracts.entry_exit_feature_surface_v1 import load_m1_feature_surface_times
    from gx1.contracts.entry_gc_mtf_history_v1 import load_gc_mtf_history_raw_artifact
    from gx1.contracts.market_state_time_query_native_F_batch_v1 import FEATURE_KEYS

    proof = scope.fold2_selected_native_source_evidence()
    _need(proof['decision'] == 'PASS_FOLD2_FULL_GC_SELECTED_S_F_SOURCE_TARGET_ONLY',
          'SOURCE_TARGET_PROOF')
    review = scope._read(scope.FOLD2_SELECTED_INITIAL_REVIEW, 'Fold 2 initial review')
    plan = scope._read(review['plan'], 'Fold 2 initial plan')
    inputs = plan['inputs']
    view_plan = initial.checked_json(inputs['fold2_view_plan'])
    packet = initial.checked_json(inputs['fold2_packet'])
    constructor = initial.checked_json(inputs['fold2_constructor'])
    prior_view = initial.checked_json(inputs['fold2_prior_ta_view'])
    physical = initial.checked_json(inputs['fold2_sidecar_result'])
    physical_plan = initial.checked_json(physical['plan'])
    old_span_plan = initial.checked_json(physical_plan['evidence']['input_plan'])
    provider_plan = initial.checked_json(mature.INPUT_REFS['provider_plan'])
    _need(view_plan['inputs']['packet'] == inputs['fold2_packet']
          and view_plan['inputs']['constructor'] == inputs['fold2_constructor']
          and view_plan['inputs']['view'] == inputs['fold2_prior_ta_view']
          and physical['status'] == 'COMPLETE', 'INPUT_LINEAGE')
    sources = {name: dict(ref) for name, ref in prior_view['source_bindings'].items()}
    bridge = old.HISTORICAL_RUNTIME_BRIDGE['gx1/contracts/entry_gc_mtf_history_v1.py']
    _need(sources['gc_gather_source']['sha256'] == bridge[0]
          and source_owners['gx1/contracts/entry_gc_mtf_history_v1.py']['sha256'] == bridge[1],
          'HISTORICAL_GC_BRIDGE')
    for name, ref in sources.items():
        path = Path(ref['path'])
        if path.is_relative_to(scope.CANONICAL_REPO) and name != 'gc_gather_source':
            relative = str(path.relative_to(scope.CANONICAL_REPO))
            _need(relative in source_owners, 'SOURCE_OWNER:' + relative)
            sources[name] = source_owners[relative]
        elif name != 'gc_gather_source':
            initial.checked_file(ref)
    view = views.bind_native_direction_feature_view(
        prefix_normalization=packet, prefix_normalization_ref=inputs['fold2_packet'],
        constructor_metadata=constructor, constructor_metadata_ref=inputs['fold2_constructor'],
        source_bindings=sources, max_batch_rows=scope.BATCH_SIZE,
        full_feature_profile=True, market_state_time_query_v2=True)
    _need(view['contract_sha256'] == proof['view_sha256']
          and view['physical_widths'] == {'signal': 2727, 'ctx_cont': 2542,
                                          'ctx_cat': 1, 'mtf': 255}, 'FULL_GC_VIEW')

    provider = mature._provider_module()
    info = provider.validate_metadata(provider_plan)
    axes, _ = provider.read_axes(provider_plan, info, np)
    for key in ('parent_rows', 'bar_start_ns', 'decision_ns'):
        values = axes[key]
        _need(values.dtype == np.int64 and values.shape == (view['full_axes']['rows'],)
              and hashlib.sha256(values.astype('<i8', copy=False).tobytes()).hexdigest()
                  == view['full_axes'][key + '_sha256']
              and hashlib.sha256(values[:47338].astype('<i8', copy=False).tobytes()).hexdigest()
                  == view['prefix'][key + '_sha256'], 'AXIS:' + key)
        values.setflags(write=False)
    selected = np.asarray(scope.PARENT_ROWS, dtype=np.int64)
    at = np.searchsorted(axes['parent_rows'], selected)
    _need(bool(np.all(at < len(axes['parent_rows'])))
          and np.array_equal(axes['parent_rows'][at], selected)
          and int(at[-1]) < 47338
          and bool(np.all(axes['decision_ns'][at] < view['prefix']['cutoff_ns'])),
          'SELECTED_TRAIN_ROWS')
    issues = axes['decision_ns'][at]
    from gx1.contracts.market_state_five_endpoint_target_binding_v1 import (
        load_bound_five_endpoint_targets,
    )
    from gx1.contracts.market_state_future_displacement_candidate_v1 import (
        load_bound_future_displacement,
    )
    for truth in (load_bound_five_endpoint_targets().gather(
            selected, issues, fold_id=2, partition='fit'),
            load_bound_future_displacement().gather(
            selected, issues, fold_id=2, partition='fit')):
        _need(bool(truth['eligible'].all() and truth['mature'].all()),
              'SELECTED_TARGETS')
    groups = provider.map_reuse(info['reuse'], np)
    source, current = groups['source050'], groups['current051']
    _need(provider.regular(old_span_plan['local_clock_path'])
          == old_span_plan['local_clock_stat'], 'LOCAL_CLOCK_STAT')
    times = load_m1_feature_surface_times(
        Path(old_span_plan['local_clock_path']), context='FOLD2_SELECTED_TRAIN_CLOCK_ONLY',
        expected_bar_seconds=300).asi8
    positions = np.searchsorted(times, axes['bar_start_ns']).astype(np.int64)
    _need(np.all(positions < len(times))
          and np.array_equal(times[positions], axes['bar_start_ns']), 'LOCAL_POSITIONS')
    for array in (times, positions, source['signal'], current['snap'],
                  current['ctx_cont'], current['ctx_cat']):
        array.setflags(write=False)
    sequence = dict(signal=source['signal'], times_ns=times,
                    sample_positions=positions, audit=info['audit'])
    payloads, values, clocks = {}, {}, {}
    for lane in ('local', *TIMEFRAMES):
        sidecar = physical['surfaces'][lane]['artifact']
        initial.checked_file(sidecar)
        with np.load(sidecar['path'], allow_pickle=False) as archive:
            payloads[lane] = {key: archive[key] for key in archive.files}
        if lane != 'local':
            cache = old_span_plan['caches'][lane]
            for kind in ('feats', 'ts'):
                initial.checked_file(cache[kind])
            values[lane] = np.load(cache['feats']['path'], mmap_mode='r', allow_pickle=False)
            clocks[lane] = np.load(cache['ts']['path'], mmap_mode='r', allow_pickle=False)
    parent = packet['parent_prefix_normalization']['base_normalization']
    bound = build_anchor_span_inputs(
        view, entry_sequence_source=sequence, snap_x=current['snap'],
        ctx_cont=current['ctx_cont'], ctx_cat=current['ctx_cat'],
        parent_signal_names=parent['surfaces']['signal']['field_names'],
        parent_mtf_names=parent['surfaces']['mtf_m5']['field_names'],
        parent_mtf_values=values, parent_mtf_times=clocks, span_payloads=payloads,
        physical_populations={lane: item['population']
                              for lane, item in physical['surfaces'].items()},
        unobserved_units_profile=None)
    gc = load_gc_mtf_history_raw_artifact(provider_plan['gc_surfaces'])
    _need(gc['layout'] == mature._receipt_chain()['gc_layout']
          and gc['layout']['contract_sha256'] == view['prefix_normalization']['gc_layout_sha256'],
          'CURRENT_GC_LAYOUT')
    if stream:
        _need(isinstance(later_full_gc_ref, dict)
              and set(later_full_gc_ref) == {'path', 'sha256'}, 'LATER_REF')
        later_full = initial.checked_json(later_full_gc_ref)
        _need(later_full['status'] == 'COMPLETE'
              and later_full['decision'] == 'PASS_FULL_GC_FOLD2_LATER_INPUT_GATHER'
              and later_full['view_sha256'] == view['contract_sha256']
              and later_full['selected_parent_rows'] == 4096
              and later_full['feature_batches'] == 256
              and later_full['test_data_used'] is False, 'LATER_RECEIPT')
        later_plan = initial.checked_json(later_full['plan'])
        later_physical = initial.checked_json(later_plan['inputs']['later_result'])
        coverage = initial.checked_json(later_plan['inputs']['later_coverage'])
        _need(later_physical['status'] == 'COMPLETE'
              and coverage['status'] == 'COMPLETE'
              and coverage['score_rows'] >= 4096
              and len(coverage['batch_indices']) == 256
              and later_full['parent_rows_sha256']
                  == coverage['selected_parent_rows_sha256']
              and later_full['decision_ns_sha256']
                  == coverage['selected_decision_ns_sha256'], 'LATER_LINEAGE')
        chosen = (coverage['score_parent_start']
                  + np.asarray(coverage['batch_indices'], dtype=np.int64)[:, None] * 16
                  + np.arange(16, dtype=np.int64)).reshape(-1)
        _need(chosen.shape == (4096,)
              and int(chosen[-1]) < coverage['score_parent_start'] + coverage['score_rows']
              and hashlib.sha256(axes['parent_rows'][chosen].astype('<i8', copy=False).tobytes()).hexdigest()
                  == coverage['selected_parent_rows_sha256']
              and hashlib.sha256(axes['decision_ns'][chosen].astype('<i8', copy=False).tobytes()).hexdigest()
                  == coverage['selected_decision_ns_sha256'], 'LATER_SELECTED_AXES')
        later_payloads = {}
        for lane in ('local', *TIMEFRAMES):
            artifact = later_physical['surfaces'][lane]['artifact']
            initial.checked_file(artifact)
            with np.load(artifact['path'], allow_pickle=False) as archive:
                later_payloads[lane] = {key: archive[key] for key in archive.files}
        later_coverage = dict(coverage, view_sha256=view['contract_sha256'])
        later_bound = build_anchor_span_inputs(
            view, entry_sequence_source=sequence, snap_x=current['snap'],
            ctx_cont=current['ctx_cont'], ctx_cat=current['ctx_cat'],
            parent_signal_names=parent['surfaces']['signal']['field_names'],
            parent_mtf_names=parent['surfaces']['mtf_m5']['field_names'],
            parent_mtf_values=values, parent_mtf_times=clocks,
            span_payloads=later_payloads,
            physical_populations={lane: item['population']
                                  for lane, item in later_physical['surfaces'].items()},
            unobserved_units_profile=None, later_coverage=later_coverage,
            later_axes=axes)
        selected_set = set(chosen.tolist())

        def gather(label, rows, batch_issues):
            _need(label in ('TRAIN', 'LATER') and isinstance(rows, np.ndarray)
                  and rows.dtype == np.int64 and isinstance(batch_issues, np.ndarray)
                  and batch_issues.dtype == np.int64 and 1 <= len(rows) <= 16
                  and len(rows) == len(batch_issues), 'STREAM_REQUEST')
            at = np.searchsorted(axes['parent_rows'], rows)
            _need(bool(np.all(at < len(axes['parent_rows'])))
                  and np.array_equal(axes['parent_rows'][at], rows)
                  and np.array_equal(axes['decision_ns'][at], batch_issues)
                  and (bool(np.all(at < 47338)) if label == 'TRAIN'
                       else all(int(pos) in selected_set for pos in at)),
                  'STREAM_AXES')
            overlay = bound if label == 'TRAIN' else later_bound
            raw = views.gather_native_direction_feature_batch(
                view, parent_rows=axes['parent_rows'], bar_start_ns=axes['bar_start_ns'],
                decision_ns=axes['decision_ns'], entry_sequence_source=sequence,
                snap_x=current['snap'], ctx_cont=current['ctx_cont'], ctx_cat=current['ctx_cat'],
                multi_tf_sources=overlay.mtf_sources, gc_surfaces=gc['surfaces'],
                gc_layout=gc['layout'], batch_parent_rows=rows,
                unobserved_units_profile=None, anchor_span_inputs=overlay)
            _need(set(raw['features']) == FEATURE_KEYS
                  and raw['witness']['parent_rows'] == rows.tolist()
                  and raw['witness']['decision_ns'] == batch_issues.tolist(),
                  'STREAM_BATCH_IDENTITY')
            for name, value in raw['features'].items():
                dtype = np.int64 if name == 'ctx_cat' else np.float32
                _need(value.dtype == dtype and value.shape[0] == len(rows)
                      and bool(np.isfinite(value).all()), 'STREAM_FEATURE:' + name)
            for lane, witnesses in raw['witness']['multi_tf'].items():
                _need(len(witnesses) == len(rows)
                      and all(max(item['available_ns']) <= int(issue)
                              for item, issue in zip(witnesses, batch_issues)),
                      'STREAM_CLOCK:' + lane)
            return raw['features']

        def release():
            for array in (*source.values(), *current.values(),
                          *values.values(), *clocks.values()):
                provider.flush_pages(array)
            for surface in gc['surfaces'].values():
                for key in ('native_row_indices', 'available_ns', 'features'):
                    provider.flush_pages(surface[key])
            provider.validate_metadata(provider_plan)

        return {'view': view, 'packet': packet, 'constructor': constructor,
                'axes': axes, 'train_parent_rows': axes['parent_rows'][:47338],
                'train_issue_ns': axes['decision_ns'][:47338],
                'later_parent_rows': axes['parent_rows'][chosen],
                'later_issue_ns': axes['decision_ns'][chosen],
                'later_score_upper_ns': coverage['score_end_exclusive_ns'],
                'gather': gather, 'release': release,
                'provider': provider, 'provider_plan': provider_plan}

    batches, reports = [], []
    for left in (0, scope.BATCH_SIZE):
        rows = selected[left:left + scope.BATCH_SIZE]
        batch_issues = issues[left:left + scope.BATCH_SIZE]
        raw = views.gather_native_direction_feature_batch(
            view, parent_rows=axes['parent_rows'], bar_start_ns=axes['bar_start_ns'],
            decision_ns=axes['decision_ns'], entry_sequence_source=sequence,
            snap_x=current['snap'], ctx_cont=current['ctx_cont'], ctx_cat=current['ctx_cat'],
            multi_tf_sources=bound.mtf_sources, gc_surfaces=gc['surfaces'],
            gc_layout=gc['layout'], batch_parent_rows=rows,
            unobserved_units_profile=None, anchor_span_inputs=bound)
        _need(set(raw['features']) == FEATURE_KEYS
              and raw['witness']['parent_rows'] == rows.tolist()
              and raw['witness']['decision_ns'] == batch_issues.tolist(),
              'BATCH_IDENTITY')
        for name, value in raw['features'].items():
            expected_dtype = np.int64 if name == 'ctx_cat' else np.float32
            _need(value.dtype == expected_dtype and value.shape[0] == scope.BATCH_SIZE
                  and bool(np.isfinite(value).all()), 'FEATURE:' + name)
        for lane, witnesses in raw['witness']['multi_tf'].items():
            _need(len(witnesses) == scope.BATCH_SIZE
                  and all(max(item['available_ns']) <= int(issue)
                          for item, issue in zip(witnesses, batch_issues)),
                  'CAUSAL_CLOCK:' + lane)
        batches.append((raw['features'], rows, batch_issues))
        reports.append({'parent_rows': rows.tolist(), 'decision_ns': batch_issues.tolist(),
                        'feature_shapes': {name: list(value.shape)
                                           for name, value in raw['features'].items()},
                        'causal_lanes': sorted(raw['witness']['multi_tf'])})
    for array in (*source.values(), *current.values(), *values.values(), *clocks.values()):
        provider.flush_pages(array)
    for surface in gc['surfaces'].values():
        for key in ('native_row_indices', 'available_ns', 'features'):
            provider.flush_pages(surface[key])
    provider.validate_metadata(provider_plan)
    return view, packet, constructor, batches, reports, provider, provider_plan
