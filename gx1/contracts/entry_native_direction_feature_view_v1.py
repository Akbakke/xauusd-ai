"""Research-only native feature rows and direct constructor bindings.

Callers verify file bytes, source/axis hashes and immutable raw backings before
binding or gathering. Internal JSON content seals are not pretty-file SHA256s.
This owner performs no I/O, fits or admission. A batch validates selected rows,
not the complete source again. Features remain raw until the unchanged native
backbone normalizes them. Legacy dynamic heads still compute; their outputs do
not supervise the separate direction head. Callers own deterministic seeding
and the existing FP32/MHA context on the returned wrapper.backbone.
"""
from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
import hashlib
import json
import math
from pathlib import PurePosixPath

import numpy as np
import pandas as pd

from gx1.contracts import entry_gc_mtf_history_v1 as history
from gx1.contracts import entry_model_native_input_normalization_v1 as norm
from gx1.contracts.entry_model_native_signal_v1 import MODEL_NATIVE_SEQ_LEN
from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
from gx1.contracts.entry_causal_m1_outcomes_v1 import DYNAMIC_M1_PREFIX_HORIZON_MINUTES
from gx1.contracts.entry_native_direction_research_v1 import NativeEntryDirectionResearch, native_direction_research_contract
from gx1.features import entry_specialist_feature_groups_v1 as owners
from gx1.features import entry_direction_momentum_profile_v1 as price_momentum_owner
from gx1.features.htf_features import MULTI_TF_SHIFT, get_model_native_multi_tf_route_windows
from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import EntryV10CtxHybridTransformer

SCHEMA_VERSION = 'gx1_entry_native_direction_feature_view_v1'
BAR_NS = 300_000_000_000
MAX_NS = 2**63 - 1
SOURCE_REF_KEYS = frozenset({'feature_view_design', 'input_qualification', 'raw_input_provider_plan',
    'raw_gc_layout', 'raw_gc_transition', 'direction_component_source', 'native_constructor_source',
    'mtf_window_source', 'gc_gather_source', 'owner_profile_source'})
DENSE_SOURCE_REF_KEYS = frozenset({'dense_direction_component_source', 'dense_profile_source', 'dense_native_constructor_source'})
PRICE_MOMENTUM_SOURCE_REF_KEYS = frozenset({'price_momentum_profile_source'})
M1_OFI_SOURCE_REF_KEYS = frozenset({'m1_ofi_window_source', 'm1_ofi_normalization_source',
    'm1_ofi_roll_owner_source', 'm1_ofi_roll_schedule'})
VIEW_KEYS = frozenset({'schema_version', 'prefix_normalization', 'prefix', 'full_axes', 'base_lineage',
    'prefix_source_bindings_sha256', 'physical_widths', 'physical_field_names_sha256', 'sequence_bars',
    'window_rows', 'canonical_routing_sha256', 'constructor_scalar_config', 'constructor_metadata',
    'entry_execution_profile', 'research_objective', 'source_bindings', 'max_batch_rows', 'evidence_scope', 'contract_sha256'})
EVIDENCE_SCOPE = 'metadata_binding_and_selected_raw_rows_only_no_file_or_native_execution_qualification'


def _require(value, reason):
    if not value:
        raise RuntimeError('NATIVE_DIRECTION_FEATURE_VIEW_' + reason)


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _copy(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _sealed(value, reason):
    _require(isinstance(value, Mapping) and value.get('contract_sha256') == _sha(
        {k: v for k, v in value.items() if k != 'contract_sha256'}), reason)


def _ref(value):
    _require(isinstance(value, Mapping) and set(value) == {'path', 'sha256'}, 'REF_KEYS')
    p, digest = value['path'], value['sha256']
    _require(isinstance(p, str) and PurePosixPath(p).is_absolute() and str(PurePosixPath(p)) == p
        and '..' not in PurePosixPath(p).parts and isinstance(digest, str) and len(digest) == 64
        and all(c in '0123456789abcdef' for c in digest), 'REF')
    return dict(value)


@lru_cache(maxsize=1)
def _canonical():
    fields = owners.compile_compact_entry_execution_profile()['physical_field_names']
    families = tuple(owners.MODEL_NATIVE_TRAINING_SPECIALISTS)
    signal = {family: [] for family in families}
    for index, name in enumerate(fields['signal']):
        family = owners.classify_entry_specialist_feature(name)
        _require(family in signal, 'SIGNAL_OWNER')
        signal[family].append(index)
    context = owners.model_native_context_specialist_routing_contract()
    aliases = owners.model_native_context_temporal_alias_policy(fields['signal'])
    mtf = {name: list(indices) for name, indices in owners.require_multi_tf_specialist_routing_v4(fields['mtf']).items()}
    _require(len(families) == 8 and set(signal) == set(mtf) == set(context['ctx_cont_indices']) == set(families), 'FULL_EIGHT_OWNERS')
    routes = {'specialist_input_indices': signal,
        'specialist_ctx_cont_indices': context['ctx_cont_indices'],
        'specialist_ctx_cont_nominal_indices': context['ctx_cont_nominal_indices'],
        'specialist_ctx_cat_indices': context['ctx_cat_indices'], 'multi_tf_specialist_input_indices': mtf,
        'temporal_alias_signal_indices': aliases['signal_indices'],
        'temporal_alias_ctx_cont_indices': aliases['ctx_cont_indices']}
    widths = {name: len(names) for name, names in fields.items()}
    return fields, routes, widths


def _configuration(metadata):
    _require(isinstance(metadata, Mapping) and isinstance(metadata.get('multi_tf'), Mapping)
        and isinstance(metadata.get('specialist_fusion'), Mapping), 'CONSTRUCTOR_METADATA')
    fields, routes, widths = _canonical()
    mtf, specialist = metadata['multi_tf'], metadata['specialist_fusion']
    _require(all(type(metadata.get(key)) is int and metadata[key] == expected for key, expected in (
        ('seq_input_dim', widths['signal']), ('snap_input_dim', widths['signal']),
        ('seq_len', MODEL_NATIVE_SEQ_LEN), ('ctx_cont_dim', widths['ctx_cont']), ('ctx_cat_dim', widths['ctx_cat'])))
        and list(mtf.get('feature_names', ())) == list(fields['mtf']), 'CONSTRUCTOR_PHYSICAL_FIELDS')
    for tf, length in history.WINDOW_ROWS.items():
        _require(type(mtf.get(tf.lower() + '_seq_dim')) is int and mtf[tf.lower() + '_seq_dim'] == widths['mtf']
            and type(mtf.get(tf.lower() + '_seq_len')) is int and mtf[tf.lower() + '_seq_len'] == length, 'CONSTRUCTOR_TF_GEOMETRY')
    context = specialist.get('context_routing', {})
    _require(specialist.get('input_indices') == routes['specialist_input_indices']
        and all(context.get(key) == routes[target] for key, target in (
            ('ctx_cont_indices', 'specialist_ctx_cont_indices'), ('ctx_cont_nominal_indices', 'specialist_ctx_cont_nominal_indices'),
            ('ctx_cat_indices', 'specialist_ctx_cat_indices')))
        and context.get('temporal_alias_policy', {}).get('signal_indices') == routes['temporal_alias_signal_indices']
        and context.get('temporal_alias_policy', {}).get('ctx_cont_indices') == routes['temporal_alias_ctx_cont_indices'], 'CONSTRUCTOR_FULL_OWNER_ROUTING')
    values = {'dropout': metadata.get('dropout'), 'multi_tf_num_layers': mtf.get('multi_tf_num_layers'),
        'multi_tf_scale': mtf.get('multi_tf_scale'), 'specialist_num_layers': specialist.get('num_layers'),
        'specialist_fusion_scale': specialist.get('fusion_scale'), 'cross_family_fusion_scale': specialist.get('cross_family_fusion_scale')}
    for key, value in values.items():
        if key.endswith('num_layers'):
            _require(type(value) is int and value > 0, 'CONSTRUCTOR_SCALAR:' + key)
        else:
            _require(type(value) in (float, int) and math.isfinite(value)
                and (0 <= value < 1 if key == 'dropout' else value > 0), 'CONSTRUCTOR_SCALAR:' + key)
            values[key] = float(value)
    return values


def _require_view(view):
    _require(isinstance(view, Mapping), 'VIEW_KEYS')
    dense = 'dense_minute_supervision' in view
    price_momentum = 'price_momentum_profile' in view
    m1_ofi = 'm1_ofi_normalization' in view
    coarsened = 'm1_ofi_coarsened' in view
    _require(set(view) == VIEW_KEYS | ({'bps_scale'} if 'bps_scale' in view else set()) | ({'magnitude_weight_contract'} if 'magnitude_weight_contract' in view else set()) | ({'dense_minute_supervision'} if dense else set()) | ({'price_momentum_profile'} if price_momentum else set()) | ({'m1_ofi_normalization'} if m1_ofi else set()) | ({'m1_ofi_coarsened'} if coarsened else set())
        and sum((dense, 'bps_scale' in view, 'magnitude_weight_contract' in view)) <= 1
        and (not dense or view['dense_minute_supervision'] is True)
        and (not price_momentum or (dense and view['price_momentum_profile'] is True))
        and (not m1_ofi or dense) and (not coarsened or (m1_ofi and view['m1_ofi_coarsened'] is True)), 'VIEW_KEYS')
    if dense:
        from gx1.contracts.entry_native_direction_dense_v1 import dense_direction_research_contract
        objective = dense_direction_research_contract()
    else:
        objective = native_direction_research_contract(signed_bps='bps_scale' in view, magnitude_weighted='magnitude_weight_contract' in view)
    _sealed(view, 'VIEW_SEAL')
    fields, routes, widths = _canonical()
    _require(view['schema_version'] == SCHEMA_VERSION and view['physical_widths'] == widths
        and view['physical_field_names_sha256'] == {key: _sha(names) for key, names in fields.items()}
        and view['sequence_bars'] == MODEL_NATIVE_SEQ_LEN and view['window_rows'] == history.WINDOW_ROWS
        and view['canonical_routing_sha256'] == _sha(routes)
        and view['research_objective'] == objective
        and view['evidence_scope'] == EVIDENCE_SCOPE
        and type(view['max_batch_rows']) is int and view['max_batch_rows'] > 0, 'VIEW_CANONICAL_IDENTITY')
    _require(set(view['source_bindings']) == SOURCE_REF_KEYS | (DENSE_SOURCE_REF_KEYS if dense else set()) | (PRICE_MOMENTUM_SOURCE_REF_KEYS if price_momentum else set()) | (M1_OFI_SOURCE_REF_KEYS if m1_ofi else set()), 'SOURCE_REF_KEYS')
    for item in view['source_bindings'].values(): _ref(item)
    if m1_ofi:
        from gx1.contracts.entry_native_direction_m1_ofi_normalization_v1 import require as require_m1_ofi
        fitted = require_m1_ofi(view['m1_ofi_normalization'])
        _require(fitted['model_contract_clock_bound'] is True
            and fitted['roll_sha256'] == view['source_bindings']['m1_ofi_roll_schedule']['sha256'], 'M1_OFI_ROLL_JOIN')
        _require(fitted['prefix_rows'] == view['prefix']['rows']
            and fitted['cutoff_ns'] == view['prefix']['cutoff_ns'], 'M1_OFI_OWN_PREFIX')
        _require(fitted['full_rows'] == view['full_axes']['rows']
            and all(fitted['full_' + key] == view['full_axes'][key]
                and fitted['prefix_' + key] == view['prefix'][key]
                for key in ('parent_rows_sha256', 'decision_ns_sha256')), 'M1_OFI_AXIS_JOIN')
    if 'bps_scale' in view:
        from gx1.contracts.entry_native_direction_bps_v1 import require_direction_bps_scale
        scale = require_direction_bps_scale(view['bps_scale'])
        _require(scale['prefix_rows'] == view['prefix']['rows']
            and scale['cutoff_ns'] == view['prefix']['cutoff_ns'], 'BPS_OWN_PREFIX')
    if 'magnitude_weight_contract' in view:
        from gx1.contracts.entry_native_direction_bps_v1 import require_direction_magnitude_weight_contract
        weight = require_direction_magnitude_weight_contract(view['magnitude_weight_contract'])
        _require(weight['base_scale']['prefix_rows'] == view['prefix']['rows']
            and weight['base_scale']['cutoff_ns'] == view['prefix']['cutoff_ns'], 'MAG_WEIGHT_OWN_PREFIX')
    return view


def bind_native_direction_feature_view(*, prefix_normalization, prefix_normalization_ref,
    constructor_metadata, constructor_metadata_ref, source_bindings, max_batch_rows, expected=None, bps_scale=None, magnitude_weight_contract=None, dense_minutes=False, price_momentum=False, m1_ofi_normalization=None, m1_ofi_coarsened=False):
    """Bind verified metadata; ``expected`` requires exact reconstruction of a view.

    Plainrefs identify caller-verified files. Content seals bind their decoded
    objects separately. No file hash or genuine-native qualification occurs here.
    Old constructor metadata may retain old artifact provenance; only its explicit
    scalar configuration/full raw routing are used, never its old GC stats join.
    """
    _sealed(prefix_normalization, 'PREFIX_NORMALIZATION_SEAL')
    _require(set(prefix_normalization) == {'schema_version', 'base_normalization', 'gc_mtf_history',
        'prefix_qualification', 'old055_or223_normalization_identity_reused', 'native_launch_authorized', 'genuine_model_fit_authorized'}
        | {'contract_sha256'} and prefix_normalization['schema_version'] == 'gx1_native_direction_prefix_normalization_v1'
        and prefix_normalization['old055_or223_normalization_identity_reused'] is False
        and prefix_normalization['native_launch_authorized'] is prefix_normalization['genuine_model_fit_authorized'] is False, 'PREFIX_NORMALIZATION_SCOPE')
    report = prefix_normalization['prefix_qualification']; _sealed(report, 'PREFIX_QUALIFICATION_SEAL')
    _require(report.get('fit_allowed') is True and report.get('blockers') == [] and report.get('statistics_fits') == 0, 'SUPPORTED_PREFIX')
    proof = report['proof']; fields, routes, widths = _canonical()
    base = norm.require_input_normalization_contract(prefix_normalization['base_normalization'],
        expected_field_names={name: list(fields['signal'] if name == 'signal' else fields['ctx_cont'] if name == 'ctx_cont' else fields['mtf'])
            for name in norm.EXPECTED_SURFACES}, expected_ctx_cat_names=fields['ctx_cat'],
        expected_entry_training_phase=ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE)
    gc_history = history.require_gc_mtf_history(prefix_normalization['gc_mtf_history'])
    _require(base['lineage']['train_row_count'] == proof['prefix']['rows']
        and base['lineage']['context_fit_row_count'] == proof['prefix']['rows']
        and base['lineage']['local_fit_row_count'] == proof['local']['selected_unique_row_count']
        and all(base['lineage'][key] == value for key, value in proof['base_lineage'].items())
        and gc_history['normalization']['provenance']['train_end_ns'] == proof['prefix']['cutoff_ns'], 'PREFIX_LINEAGE_JOIN')
    for tf, item in gc_history['normalization']['provenance']['row_union'].items():
        _require(all(item[key] == proof['gc'][tf][key] for key in ('rows', 'native_row_indices_sha256', 'available_ns_sha256')), 'GC_PREFIX_ROW_UNION:' + tf)
    _require(type(dense_minutes) is bool and type(price_momentum) is bool and type(m1_ofi_coarsened) is bool
        and not (dense_minutes and (bps_scale is not None or magnitude_weight_contract is not None))
        and (not price_momentum or dense_minutes)
        and (m1_ofi_normalization is None or dense_minutes)
        and (not m1_ofi_coarsened or m1_ofi_normalization is not None), 'OBJECTIVE_MODE')
    _require(isinstance(source_bindings, Mapping) and set(source_bindings) == SOURCE_REF_KEYS | (DENSE_SOURCE_REF_KEYS if dense_minutes else set()) | (PRICE_MOMENTUM_SOURCE_REF_KEYS if price_momentum else set()) | (M1_OFI_SOURCE_REF_KEYS if m1_ofi_normalization is not None else set()), 'SOURCE_REF_KEYS')
    source_bindings = {key: _ref(value) for key, value in source_bindings.items()}
    provenance = proof['source_bindings']
    _require(source_bindings['raw_input_provider_plan'] == provenance['ancestral264_plan']
        and source_bindings['raw_gc_layout'] == provenance['current_gc_layout']
        and source_bindings['raw_gc_transition'] == provenance['current_gc_raw_transition']
        and all(source_bindings[key] == provenance['current_source_owners'][owner] for key, owner in (
            ('mtf_window_source', 'gx1/features/htf_features.py'), ('gc_gather_source', 'gx1/contracts/entry_gc_mtf_history_v1.py'),
            ('owner_profile_source', 'gx1/features/entry_specialist_feature_groups_v1.py'))), 'SOURCE_PROVENANCE_JOINS')
    scalars = _configuration(constructor_metadata)
    compiled = (price_momentum_owner.compile_price_momentum_profile(gc_history) if price_momentum
        else owners.compile_gc_mtf_geometry_entry_execution_profile(gc_history) if dense_minutes
        else owners.compile_gc_mtf_entry_execution_profile(gc_history))
    _require(type(max_batch_rows) is int and max_batch_rows > 0, 'MAX_BATCH_ROWS')
    if dense_minutes:
        from gx1.contracts.entry_native_direction_dense_v1 import dense_direction_research_contract
        objective = dense_direction_research_contract()
    else:
        objective = native_direction_research_contract(signed_bps=bps_scale is not None, magnitude_weighted=magnitude_weight_contract is not None)
    value = {'schema_version': SCHEMA_VERSION,
        'prefix_normalization': {'binding': _ref(prefix_normalization_ref), 'contract_sha256': prefix_normalization['contract_sha256'],
            'base_sha256': base['contract_sha256'], 'gc_layout_sha256': gc_history['layout']['contract_sha256'],
            'gc_normalization_sha256': gc_history['normalization']['contract_sha256'], 'qualification_sha256': report['contract_sha256']},
        'prefix': _copy(proof['prefix']), 'full_axes': _copy(proof['full_axes']), 'base_lineage': _copy(proof['base_lineage']),
        'prefix_source_bindings_sha256': _sha(provenance), 'physical_widths': dict(widths),
        'physical_field_names_sha256': {key: _sha(names) for key, names in fields.items()},
        'sequence_bars': MODEL_NATIVE_SEQ_LEN, 'window_rows': dict(history.WINDOW_ROWS), 'canonical_routing_sha256': _sha(routes),
        'constructor_scalar_config': scalars,
        'constructor_metadata': {'binding': _ref(constructor_metadata_ref), 'content_sha256': _sha(constructor_metadata)},
        'entry_execution_profile': {'name': compiled['name'], 'contract_sha256': compiled['contract_sha256']},
        'research_objective': objective, 'source_bindings': source_bindings,
        'max_batch_rows': max_batch_rows, 'evidence_scope': EVIDENCE_SCOPE}
    if bps_scale is not None:
        value['bps_scale'] = _copy(bps_scale)
    if magnitude_weight_contract is not None:
        value['magnitude_weight_contract'] = _copy(magnitude_weight_contract)
    if dense_minutes:
        value['dense_minute_supervision'] = True
    if price_momentum:
        value['price_momentum_profile'] = True
    if m1_ofi_normalization is not None:
        value['m1_ofi_normalization'] = _copy(m1_ofi_normalization)
    if m1_ofi_coarsened:
        value['m1_ofi_coarsened'] = True
    value['contract_sha256'] = _sha(value)
    _require_view(value)
    _require(expected is None or expected == value, 'EXPECTED_VIEW_MISMATCH')
    return value


def gather_native_direction_feature_batch(view, *, parent_rows, bar_start_ns, decision_ns,
    entry_sequence_source, snap_x, ctx_cont, ctx_cat, multi_tf_sources, gc_surfaces, gc_layout, batch_parent_rows,
    m1_ofi_source=None, m1_ofi_clock=None):
    """Gather one bounded raw batch from caller-verified immutable full backings.

    The normalization cutoff describes fitted statistics; later OOF feature rows
    remain legal. Training target maturity is handled only by the separate loss.
    No labels, future endpoints, padding or resampling enter these feature arrays.
    """
    _require_view(view)
    fields, routes, widths = _canonical(); rows = view['full_axes']['rows']
    for axis in (parent_rows, bar_start_ns, decision_ns):
        _require(isinstance(axis, np.ndarray) and axis.dtype == np.int64 and axis.shape == (rows,)
            and not axis.flags.writeable, 'FULL_AXIS_HEADER')
    selected = np.asarray(batch_parent_rows)
    _require(selected.dtype == np.int64 and selected.ndim == 1 and 0 < len(selected) <= view['max_batch_rows']
        and len(set(selected.tolist())) == len(selected), 'BOUNDED_PARENT_BATCH')
    at = np.searchsorted(parent_rows, selected)
    _require(np.all(at < rows) and np.array_equal(parent_rows[at], selected), 'EXACT_PARENT_IDS')
    starts, decisions = bar_start_ns[at], decision_ns[at]
    _require(np.all(starts >= 0) and np.all(starts % BAR_NS == 0) and np.all(starts <= MAX_NS - BAR_NS)
        and np.array_equal(decisions, starts + BAR_NS), 'EXACT_DECISION_CLOCK')
    _require(isinstance(entry_sequence_source, Mapping) and set(entry_sequence_source) == {'signal', 'times_ns', 'sample_positions', 'audit'}, 'SEQUENCE_SOURCE_KEYS')
    source, times, positions = (entry_sequence_source[key] for key in ('signal', 'times_ns', 'sample_positions'))
    _require(isinstance(source, np.ndarray) and source.dtype == np.float32 and source.ndim == 2
        and source.shape[1] == widths['signal'] and not source.flags.writeable
        and isinstance(times, np.ndarray) and times.dtype == np.int64 and times.shape == (len(source),) and not times.flags.writeable
        and isinstance(positions, np.ndarray) and positions.dtype == np.int64 and positions.shape == (rows,) and not positions.flags.writeable, 'SOURCE_HEADERS')
    for array, width, dtype in ((snap_x, widths['signal'], np.float32), (ctx_cont, widths['ctx_cont'], np.float32), (ctx_cat, widths['ctx_cat'], np.int64)):
        _require(isinstance(array, np.ndarray) and array.dtype == dtype and array.shape == (rows, width)
            and not array.flags.writeable, 'CURRENT_HEADERS')
    take = positions[at]
    _require(np.all(take >= MODEL_NATIVE_SEQ_LEN - 1) and np.all(take < len(source))
        and np.array_equal(times[take], starts), 'EXACT_SOURCE_POINTERS')
    sequences, clock_rows = [], []
    for position, decision in zip(take, decisions):
        left = int(position) - MODEL_NATIVE_SEQ_LEN + 1
        clock = times[left:int(position) + 1]
        _require(np.all(clock[1:] > clock[:-1]) and np.all(clock % BAR_NS == 0)
            and np.all(clock <= MAX_NS - BAR_NS) and np.all(clock + BAR_NS <= decision), 'CAUSAL_LOCAL_SEQUENCE_CLOCK')
        sequences.append(source[left:int(position) + 1]); clock_rows.append(clock)
    features = {'seq_x': np.stack(sequences), 'snap_x': np.ascontiguousarray(snap_x[at]),
        'ctx_cont': np.ascontiguousarray(ctx_cont[at]), 'ctx_cat': np.ascontiguousarray(ctx_cat[at])}
    _require(np.array_equal(features['seq_x'][:, -1], features['snap_x'])
        and np.array_equal(features['snap_x'][:, routes['temporal_alias_signal_indices']],
            features['ctx_cont'][:, routes['temporal_alias_ctx_cont_indices']]), 'CURRENT_SEQUENCE_ALIAS_PARITY')
    layout = history.require_gc_mtf_history_layout(gc_layout)
    _require(layout['contract_sha256'] == view['prefix_normalization']['gc_layout_sha256'], 'CURRENT_GC_LAYOUT')
    gc_surfaces = history.require_gc_mtf_history_surfaces(gc_surfaces, layout=layout)
    collected = {tf: [] for tf in history.TIMEFRAMES}; gc_collected = {tf: [] for tf in history.TIMEFRAMES}
    witnesses = {tf: [] for tf in history.TIMEFRAMES}
    for start, decision in zip(starts, decisions):
        windows = get_model_native_multi_tf_route_windows(multi_tf_sources, decision_bar_start=pd.Timestamp(int(start), tz='UTC'),
            per_tf_seq_lens=dict(history.WINDOW_ROWS), route_timeframes=history.TIMEFRAMES,
            base_bar_duration=pd.Timedelta(seconds=300), return_row_indices=True)
        for tf, (window, ids) in windows.items():
            available = np.asarray(multi_tf_sources[tf].attrs['ts_int64'][ids], dtype=np.int64) + int(MULTI_TF_SHIFT[tf].value)
            _require(np.all(available <= decision), 'CAUSAL_TF_CLOCK:' + tf)
            collected[tf].append(window)
            gc_collected[tf].append(history.gather_gc_mtf_history_rows(gc_surfaces[tf], tf=tf,
                native_row_indices=ids, available_ns=available, layout=layout))
            witnesses[tf].append({'native_row_indices': ids.tolist(), 'available_ns': available.tolist()})
    for tf in history.TIMEFRAMES:
        features['seq_' + tf.lower()] = np.stack(collected[tf])
        features['gc_seq_' + tf.lower()] = np.stack(gc_collected[tf])
    if 'm1_ofi_normalization' in view:
        from gx1.contracts import entry_closed_m1_ofi_window_v1 as m1
        fitted = view['m1_ofi_normalization']
        _require(isinstance(m1_ofi_source, m1.QualifiedSource) and isinstance(m1_ofi_clock, m1.QualifiedClock)
            and m1_ofi_source.source_sha256 == fitted['source_sha256']
            and m1_ofi_clock.clock_sha256 == fitted['clock_sha256'], 'M1_OFI_SOURCE_IDENTITY')
        minute = m1.gather(m1_ofi_source, m1_ofi_clock, batch_parent_rows=selected)
        _require(np.array_equal(minute['parent_rows'], selected) and np.array_equal(minute['decision_ns'], decisions), 'M1_OFI_EXACT_PARENT_CLOCK')
        features['m1_ofi_contracts'] = minute['ofi_contracts'].copy()
        features['m1_ofi_observed'] = minute['ofi_observed'].copy()
    else:
        _require(m1_ofi_source is None and m1_ofi_clock is None, 'UNBOUND_M1_OFI_INPUT')
    _require(all(array.flags.c_contiguous and np.isfinite(array).all() for array in features.values()), 'FINITE_CONTIGUOUS_RAW_BATCH')
    return {'features': features, 'witness': {'view_sha256': view['contract_sha256'], 'parent_rows': selected.tolist(),
        'bar_start_ns': starts.tolist(), 'decision_ns': decisions.tolist(), 'source_positions': take.tolist(),
        'local_sequence_clock_sha256': hashlib.sha256(np.stack(clock_rows).astype('<i8', copy=False).tobytes()).hexdigest(),
        'multi_tf': witnesses, 'raw_values_not_normalized': True}}


def build_native_direction_research_model(view, *, prefix_normalization, constructor_metadata, device):
    """Use the exact native type plus260head; caller controls fresh RNG/policy.

    No old artifact reader, Dataset, target provider, checkpoint or factory is
    called. This returns a fresh combined ONLINE; actual initialization and
    gradients still need separately admitted genuine qualification.
    """
    bind_native_direction_feature_view(prefix_normalization=prefix_normalization,
        prefix_normalization_ref=view['prefix_normalization']['binding'], constructor_metadata=constructor_metadata,
        constructor_metadata_ref=view['constructor_metadata']['binding'], source_bindings=view['source_bindings'],
        max_batch_rows=view['max_batch_rows'], expected=view, bps_scale=view.get('bps_scale'), magnitude_weight_contract=view.get('magnitude_weight_contract'),
        dense_minutes=view.get('dense_minute_supervision', False), price_momentum=view.get('price_momentum_profile', False),
        m1_ofi_normalization=view.get('m1_ofi_normalization'), m1_ofi_coarsened=view.get('m1_ofi_coarsened', False))
    _, routes, widths = _canonical()
    kwargs = _copy(routes)
    kwargs.update(view['constructor_scalar_config'])
    kwargs.update(seq_input_dim=widths['signal'], snap_input_dim=widths['signal'], seq_len=MODEL_NATIVE_SEQ_LEN,
        ctx_cont_dim=widths['ctx_cont'], ctx_cat_dim=widths['ctx_cat'],
        input_normalization=prefix_normalization['base_normalization'],
        gc_mtf_history=prefix_normalization['gc_mtf_history'], entry_training_phase=ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE,
        dynamic_m5_query_minutes=tuple(DYNAMIC_M1_PREFIX_HORIZON_MINUTES), entry_execution_profile=view['entry_execution_profile'])
    for tf, length in history.WINDOW_ROWS.items():
        kwargs[tf.lower() + '_seq_dim'] = widths['mtf']; kwargs[tf.lower() + '_seq_len'] = length
    backbone = EntryV10CtxHybridTransformer(**kwargs).to(device)
    _require(type(backbone) is EntryV10CtxHybridTransformer and backbone.entry_execution_profile == view['entry_execution_profile']
        and backbone.gc_mtf_history == prefix_normalization['gc_mtf_history'], 'CONSTRUCTED_NATIVE_BINDING')
    backbone.require_input_normalization_state()
    if view.get('dense_minute_supervision', False):
        from gx1.contracts.entry_native_direction_dense_v1 import DenseNativeEntryDirectionResearch
        wrapper = DenseNativeEntryDirectionResearch(backbone, m1_ofi_normalization=view.get('m1_ofi_normalization'),
            m1_ofi_coarsened=view.get('m1_ofi_coarsened', False))
    else:
        wrapper = NativeEntryDirectionResearch(backbone, bps_scale=view.get('bps_scale'), magnitude_weight_contract=view.get('magnitude_weight_contract'))
    wrapper.feature_view_identity = {'contract_sha256': view['contract_sha256'],
        'prefix_normalization': _copy(view['prefix_normalization']), 'research_objective_sha256': view['research_objective']['contract_sha256']}
    return wrapper
