"""Bind the registered December TRAIN input prefix to the canonical normalization owners."""
import hashlib
import runpy
from pathlib import Path

import numpy as np
import pandas as pd

from gx1.audit.dynamic_mtf_train_prefix_inputs import _check_payload_stats, _read, _require
from gx1.contracts import entry_gc_mtf_history_v1 as history
from gx1.contracts import entry_native_direction_prefix_normalization_v1 as prefix
from gx1.contracts import local_random_access_campaign_v2 as campaign
from gx1.contracts.entry_exit_feature_surface_v1 import load_m1_feature_surface_times
from gx1.features import htf_features as htf


def _axis_digest(values):
    return hashlib.sha256(values.astype('<i8', copy=False).tobytes()).hexdigest()


def _raw_array(ref, *, shape, dtype):
    campaign.require_binding({'path': ref['path'], 'sha256': ref['sha256']}, label='December raw input')
    a = np.load(ref['path'], mmap_mode='r', allow_pickle=False)
    _require(a.shape == tuple(shape) and a.dtype == np.dtype(dtype) and not a.flags.writeable,
             'ARRAY_HEADER:' + ref['path'])
    return a


def _closed_rows(bar_axis, shift_ns, last_issue_ns):
    """Rightmost closed native bar; comparisons here read timestamps only."""
    _require(bar_axis.ndim == 1 and bar_axis.dtype == np.int64 and np.all(np.diff(bar_axis) > 0),
             'SOURCE_CLOCK')
    stop = int(np.searchsorted(bar_axis, last_issue_ns - shift_ns, side='right'))
    _require(stop > 0 and int(bar_axis[stop - 1]) + shift_ns <= last_issue_ns, 'EMPTY_CAUSAL_PREFIX')
    return stop


def _fit_axes(source_ref, target_ref, geometry, cutoff, reserved_geometry, reserved_cutoff):
    # Recheck the physical clocks. The price archive is never indexed by a target or price key.
    campaign.require_binding(source_ref, label='December physical parent clocks')
    target = _read(target_ref)
    n = geometry['parents']
    _require(geometry['parent_range'] == [0, n] and target['metadata']['parents'] == n
             and target['metadata']['cutoff_ns'] == cutoff, 'FIT_TARGET_GEOMETRY')
    total = n + reserved_geometry['parents']
    _require(reserved_geometry['parent_range'] == [n, total] and cutoff < reserved_cutoff, 'RESERVED_GEOMETRY')
    observer = {'source': source_ref, 'rows': total,
                'fit': {'rows': n, 'cutoff_ns': cutoff},
                'held_out': {'rows': total - n, 'cutoff_ns': reserved_cutoff}}
    result = {}
    with np.load(source_ref['path'], allow_pickle=False) as archive:
        for field, digest_key in (('parent_rows', 'parent_rows'), ('bar_start_ns', 'bar_start_ns'),
                                  ('decision_ns', 'issue_ns')):
            axis = archive[field]
            _require(axis.dtype == np.int64 and axis.ndim == 1 and len(axis) >= total
                     and np.all(np.diff(axis) > 0), 'PHYSICAL_AXIS:' + field)
            if field == 'decision_ns':
                _require(np.searchsorted(axis, cutoff, side='left') == n
                         and np.searchsorted(axis, reserved_cutoff, side='left') == total, 'EXACT_PREFIX_COUNT')
            key = field + '_sha256'
            observer[key] = _axis_digest(axis[:total])
            observer['fit'][key] = _axis_digest(axis[:n])
            observer['held_out'][key] = _axis_digest(axis[n:total])
            _require(observer['fit'][key] == geometry[digest_key + '_sha256_le_i64']
                     and observer['held_out'][key] == reserved_geometry[digest_key + '_sha256_le_i64'],
                     'AXIS_SHA:' + field)
            result[field] = axis[:n].copy()
    for physical, target_field in (('parent_rows', 'parent_rows'), ('decision_ns', 'issue_ns')):
        desc = target['arrays'][target_field]
        _require(Path(desc['path']).parent == Path(target_ref['path']).parent, 'TARGET_CLOCK_PATH')
        bound = _raw_array(desc, shape=(n,), dtype='int64')
        _require(np.array_equal(result[physical], bound), 'TARGET_PHYSICAL_AXIS:' + target_field)
    _require(np.array_equal(result['bar_start_ns'] + prefix.BAR_NS, result['decision_ns']), 'DECISION_OFFSET')
    expected = {'source': target_ref, 'rows': n, **{k + '_sha256': _axis_digest(v) for k, v in result.items()}}
    return result, expected, observer


def _history_views(plan, transition_ref, layout_ref, last_issue):
    # The bound historical qualification already checked these exact bytes, including the full
    # V4 cache contract. Rehash the files we map, then expose only closed prefix rows to the fitter.
    lineage = plan['base_lineage']
    xau_ref = {'path': lineage['mtf_cache_manifest_path'], 'sha256': lineage['mtf_cache_manifest_sha256']}
    xau = _read(xau_ref)
    names = list(htf.MULTI_TF_PER_BAR_FEATURES_V4)
    _require(xau['cache_identity_sha256'] == htf.compute_htf_v4_cache_identity(xau)
             and xau['builder_version'] == lineage['mtf_builder_version'] == htf.HTF_V4_CACHE_BUILDER_VERSION
             and xau['feature_names'] == names and xau['m5_prebuilt_source'] == lineage['m5_prebuilt_path']
             and xau['m5_prebuilt_source_sha256'] == lineage['m5_prebuilt_sha256'], 'XAU_CACHE')
    transition = _read(transition_ref)
    raw_ref = plan['gc_surfaces']
    gc_source = _read(raw_ref)
    gc_layout = history.require_gc_mtf_history_layout(_read(layout_ref))
    _require(transition['binding'] == raw_ref and transition['normalization_applied'] is False
             and transition['current_layout_sha256'] == gc_layout['contract_sha256']
             and transition['raw_value_validation']['performed'] is True
             and transition['array_descriptors'] == gc_source['surfaces'], 'RAW_GC_BINDING')
    mtf, gc = {}, {}
    for tf in prefix.WINDOWS:
        info = xau['tfs'][tf]
        root = Path(plan['native_cache_dir'])
        _require(root == Path(xau_ref['path']).parent and info['ts_npy'] == tf + '_ts.npy'
                 and info['feats_npy'] == tf + '_feats.npy' and info['feature_count'] == len(names), 'XAU_DESCRIPTOR:' + tf)
        time_ref = {'path': str(root / info['ts_npy']), 'sha256': info['ts_npy_sha256']}
        feat_ref = {'path': str(root / info['feats_npy']), 'sha256': info['feats_npy_sha256']}
        _require(all(plan['payloads'][r['path']]['sha256'] == r['sha256'] for r in (time_ref, feat_ref)),
                 'XAU_INVENTORY:' + tf)
        times = _raw_array(time_ref, shape=(info['n_bars'],), dtype='int64')
        shift = int(htf.MULTI_TF_SHIFT[tf].value)
        stop = _closed_rows(times, shift, last_issue)
        feat = _raw_array(feat_ref, shape=(len(times), len(names)), dtype='float32')[:stop]
        frame = pd.DataFrame(feat, index=pd.DatetimeIndex(times[:stop].astype('datetime64[ns]'), tz='UTC'),
                             columns=names, copy=False)
        _require(np.shares_memory(frame.to_numpy(copy=False), feat), 'XAU_VIEW:' + tf)
        frame.attrs.update(ts_int64=times[:stop], feats_np=feat, causal_warmup_rows=info['causal_warmup_rows'],
                           htf_feature_contract=htf.HTF_V4_MATRIX_CONTRACT)
        mtf[tf] = frame
        descs = gc_source['surfaces'][tf]
        _require(all(plan['payloads'][r['path']]['sha256'] == r['sha256'] for r in descs.values()), 'GC_INVENTORY:' + tf)
        raw = {k: _raw_array(d, shape=d['shape'], dtype=d['dtype']) for k, d in descs.items()}
        size = _closed_rows(raw['available_ns'], 0, last_issue)
        view = {k: a[:size] for k, a in raw.items()}
        ids = view['native_row_indices']
        _require(np.all((ids >= 0) & (ids < stop)) and np.array_equal(times[ids] + shift, view['available_ns']),
                 'GC_XAU_EXACT_CLOCK:' + tf)
        gc[tf] = {**view, 'field_names': gc_layout['field_names'][tf]}
    return mtf, gc, gc_layout


def fit_masked_december_prefix(policy, stage):
    """Source-bound data-only operation. The calling capped runner owns admission and publication."""
    previous = policy['dynamic_mtf_train_prefix_dec01_20261104_001']
    geometry = previous['geometry']['fit']
    cutoff = previous['fit_cutoff_ns']
    _require(policy['training_enabled'] is stage['training_enabled'] is stage['model_calls_authorized'] is False
             and stage['normalization_fit_authorized'] is True and stage['fit_cutoff_ns'] == cutoff
             and stage['fit_target_view'] == previous['fit_target_view'], 'DECEMBER_NORMALIZATION_SCOPE')
    spec = prefix.december_zero_support_profile()
    _require(geometry['parents'] == spec['fit_prefix']['rows'] and cutoff == spec['fit_prefix']['cutoff_ns'],
             'REGISTERED_NORMALIZATION_PREFIX')
    with campaign.file_verification_scope():
        native = policy['dynamic_mtf_state_onepass_fit_20261103_001']
        recipe = _read(native['recipe'])
        inputs = _read(recipe['input_manifest'])
        parents = policy['dynamic_m5_direction_native_prefix_input_binding_repair_264']
        physical = _read(inputs['physical_provider_plan'])
        ref = physical['input_provider_plan']
        plan = _read(ref)
        source_refs = inputs['view_source_bindings']
        _require(recipe['full_feature_profile'] is True and recipe['run_scope'] == 'single_prefix_dynamic_mtf_state_one_pass'
                 and not any('m1_ofi' in name for name in (*recipe, *inputs, *source_refs))
                 and ref == parents['support_plan'] and physical['input_provider_worker'] == parents['support_worker']
                 and inputs['physical_provider_plan'] == policy['dynamic_m5_direction_gc_boolean_physical_prefix_normalization_267']['plan']
                 and inputs['axes'] == previous['physical_axes'] == plan['axes']['source']
                 and physical['payloads'] == plan['payloads'], 'RAW_PHYSICAL_PROVIDER')
        _check_payload_stats(plan['payloads'])
        worker = physical['input_provider_worker']
        campaign.require_binding(worker, label='December verified raw mmap provider')
        provider = runpy.run_path(worker['path'], run_name='gx1_december_raw_mmap_provider')
        # These two pure retained provider operations verify 261's completed decode and map the
        # exact six read-only files. None of its plan, loader or qualification entry points is run.
        reuse = provider['validate_reuse'](plan)
        mapped = provider['map_reuse'](reuse, np)
        audit = _read(plan['source_reconstruction'])
        source = _read(plan['m5_feature_manifest'])
        _require(audit['decision'] == source['decision'] == 'PASS' and len(audit['checks']) == 8
                 and all(v is True for v in audit['checks'].values())
                 and audit['feature_surface_manifest_sha256'] == plan['m5_feature_manifest']['sha256']
                 and audit['feature_surface_path'] == source['output_parquet']
                 and audit['feature_surface_sha256'] == source['output_parquet_sha256'] == plan['payloads'][source['output_parquet']]['sha256']
                 and audit['rows'] == len(mapped['current051']['snap'])
                 and audit['feature_surface_rows'] == source['rows'] == len(mapped['source050']['signal']),
                 'DECODED_SOURCE_PARITY')
        axes, expected, observer = _fit_axes(inputs['axes'], previous['fit_target_view'], geometry, cutoff,
                                             previous['geometry']['held_out'], previous['held_out_cutoff_ns'])
        starts, issues = axes['bar_start_ns'], axes['decision_ns']
        times = load_m1_feature_surface_times(Path(source['output_parquet']), context='DECEMBER_PREFIX_CLOCK_ONLY',
                                             expected_bar_seconds=prefix.BAR_NS // 1_000_000_000).asi8
        positions = np.searchsorted(times, starts)
        _require(len(times) == source['rows'] and (positions < len(times)).all()
                 and np.array_equal(times[positions], starts), 'SOURCE050_CLOCK_JOIN')
        stop = int(positions[-1]) + 1
        seq = {'times_ns': times[:stop], 'signal': mapped['source050']['signal'][:stop],
               'sample_positions': positions, 'audit': audit}
        cur = {name: arr[:len(issues)] for name, arr in mapped['current051'].items()}
        mtf, gc, layout = _history_views(plan, source_refs['raw_gc_transition'], source_refs['raw_gc_layout'], int(issues[-1]))
        bindings = {'original_recipe': native['recipe'], 'input_manifest': recipe['input_manifest'],
                    'physical_provider_plan': inputs['physical_provider_plan'], 'ancestral264_plan': ref,
                    'raw_mmap_provider': worker, 'decode_reuse': plan['decode_reuse_binding'],
                    'source_reconstruction': plan['source_reconstruction'], 'current_gc_layout': source_refs['raw_gc_layout'],
                    'current_gc_raw_transition': source_refs['raw_gc_transition'],
                    'current_source_owners': stage['source_owners'], 'reserved_observer_axes': observer,
                    'physical_axes': inputs['axes'],
                    'fit_target_view': previous['fit_target_view']}
        qualified = prefix.qualify_native_direction_prefix_population(
            **axes, cutoff_ns=cutoff, expected_axes=expected, expected_prefix_parent_rows=axes['parent_rows'],
            entry_sequence_source=seq, snap_x=cur['snap'], ctx_cont=cur['ctx_cont'], ctx_cat=cur['ctx_cat'],
            ordered_signal_names=source['feature_field_order'], multi_tf_sources=mtf, gc_surfaces=gc,
            gc_layout=layout, base_lineage=plan['base_lineage'], source_bindings=bindings, zero_support_profile=spec)
        report = qualified['report']
        norm = prefix.fit_native_direction_prefix_normalization(qualified) if report['fit_allowed'] else None
        _check_payload_stats(plan['payloads'])
        campaign.capture_verified_file_snapshot()
    return report, norm, {'physical_parent_axis_keys': ['parent_rows', 'bar_start_ns', 'decision_ns'],
                          'fit_parent_rows': len(issues), 'last_issue_ns': int(issues[-1]),
                          'fit_cutoff_ns_exclusive': cutoff, 'source050_rows_through_last_issue': stop,
                          'xau_raw_rows_through_last_issue': {k: len(v) for k, v in mtf.items()},
                          'gc_raw_rows_through_last_issue': {k: len(v['available_ns']) for k, v in gc.items()},
                          'source_stat_bookends': len(plan['payloads']), 'clock_and_mmap_bindings': bindings,
                          'old_normalization_statistics_reused': False, 'separate_m1_ofi_branch': False,
                          'market_outcomes_decoded': 0, 'model_calls': 0}
