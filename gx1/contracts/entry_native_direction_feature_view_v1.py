"""Research-only native feature rows and direct constructor bindings.

Callers verify file bytes, source/axis hashes and immutable raw backings before
binding or gathering. Internal JSON content seals are not pretty-file SHA256s.
This owner performs no I/O, fits or admission. A batch validates selected rows,
not the complete source again. Features remain raw until the unchanged native
backbone normalizes them. Legacy direction wrappers retain their old forward.
The explicit state-chain wrapper requests representation-only output and skips
unused legacy heads. Callers own deterministic seeding
and the existing FP32/MHA context on the returned wrapper.backbone.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
from pathlib import PurePosixPath

import numpy as np
import pandas as pd

from gx1.contracts import entry_gc_mtf_history_v1 as history
from gx1.contracts import entry_model_native_input_normalization_v1 as norm
from gx1.contracts import entry_native_direction_prefix_normalization_v1 as prefix_owner
from gx1.contracts import entry_native_direction_prefix_units_v1 as units
from gx1.contracts.entry_model_native_signal_v1 import MODEL_NATIVE_SEQ_LEN
from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
from gx1.contracts.entry_causal_m1_outcomes_v1 import DYNAMIC_M1_PREFIX_HORIZON_MINUTES
from gx1.contracts.entry_native_direction_research_v1 import (
    NativeEntryDirectionResearch, native_direction_research_contract,
    NativeDynamicMTFStateResearch, dynamic_mtf_state_research_contract,
)
from gx1.contracts.market_state_five_endpoint_model_v1 import (
    FiveEndpointMarketStateModel, five_endpoint_model_contract,
)
from gx1.contracts.market_state_time_query_model_v1 import (
    TimeQueryMarketStateModel, time_query_model_contract,
)
from gx1.contracts.market_state_time_query_model_v2 import (
    FactorizedTimeQueryMarketStateModel, factorized_time_query_model_contract,
)
from gx1.contracts import market_state_five_endpoint_target_binding_v1 as five_endpoint_targets
from gx1.contracts import market_state_time_query_binding_v1 as time_query_targets
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
DYNAMIC_STATE_SOURCE_REF_KEYS = frozenset({'dynamic_state_target_source'})
FIVE_ENDPOINT_SOURCE_REF_KEYS = frozenset({'five_endpoint_model_source', 'five_endpoint_contract_source',
    'five_endpoint_target_binding_source', 'five_endpoint_corrected_mask'})
TIME_QUERY_SOURCE_REF_KEYS = frozenset({'time_query_model_source', 'time_query_loss_source',
    'time_query_target_source', 'time_query_target_binding_source', 'time_query_candidate_tape',
    'time_query_mature_prefix_plan', 'five_endpoint_contract_source'})
TIME_QUERY_V2_SOURCE_REF_KEYS = (TIME_QUERY_SOURCE_REF_KEYS -
    {'time_query_model_source', 'time_query_loss_source'}) | frozenset({
    'time_query_v2_model_source', 'time_query_v2_loss_source'})
ANCHOR_SPAN_SOURCE_REF_KEYS = frozenset({'anchor_span_inputs_source'})
TIME_QUERY_MATURE_FIT_ROWS = (47338, 64615, 82152, 100224)
TIME_QUERY_MODEL_AXIS_ROWS = 118249
TIME_QUERY_MATURE_FOLD1_PLAN_PATH = str(time_query_targets.ROOT /
    'configs/research/MARKET_STATE_MATURE_FOLD1_NORMALIZATION_20261107_001.json')
TIME_QUERY_MATURE_PLAN_REF = {'path': str(time_query_targets.ROOT /
    'configs/research/MARKET_STATE_MATURE_PREFIX_NORMALIZATION_20261107_001.json'),
    'sha256': 'b77795e606b35424d6189b7866337ca74c9f7dfc22371168452f5a6f6d665029'}
CONFIRMED_AGE_SOURCE_REF_KEYS = frozenset({'confirmed_age_candidate_design'})
OCTOBER_SOURCE_REF_KEYS = frozenset({'october_input_plan'})
GC_RELATIVE_SOURCE_REF_KEYS = frozenset({'gc_relative_coordinates_source', 'gc_price_coordinate_source_map_source'})
FAMILY_SOURCE_REF_KEYS = frozenset({'family_intervention_source'})
FIVE_PRIORITY_SOURCE_REF_KEYS = frozenset({'five_priority_source'})
DENSE_BPS_SOURCE_REF_KEYS = frozenset({'dense_bps_component_source'})
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


@lru_cache(maxsize=2)
def _canonical(*, ta_only=False):
    profile = owners.compile_full_ta_entry_execution_profile() if ta_only else None
    fields = (profile['physical_field_names'] if ta_only
              else owners.compile_compact_entry_execution_profile()['physical_field_names'])
    families = tuple(owners.MODEL_NATIVE_TRAINING_SPECIALISTS)
    signal = {family: [] for family in families}
    for index, name in enumerate(fields['signal']):
        family = owners.classify_entry_specialist_feature(name)
        _require(family in signal, 'SIGNAL_OWNER')
        signal[family].append(index)
    context = owners.model_native_context_specialist_routing_contract()
    aliases = owners.model_native_context_temporal_alias_policy(
        profile['source_field_names']['signal'] if ta_only else fields['signal'])
    if ta_only:
        context = _copy(context)
        mapping = {old: new for new, old in enumerate(profile['source_ctx_cont_indices'])}
        for key in ('ctx_cont_indices', 'ctx_cont_nominal_indices'):
            context[key] = {family: [mapping[i] for i in indices if i in mapping]
                            for family, indices in context[key].items()}
        aliases = {'signal_indices': list(profile['temporal_alias_signal_indices']),
                   'ctx_cont_indices': list(profile['temporal_alias_ctx_cont_indices'])}
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


_OBSERVER_AXIS_KEYS = ('parent_rows', 'bar_start_ns', 'decision_ns')


def _require_observer_scope(scope, fit):
    hashes = {key + '_sha256' for key in _OBSERVER_AXIS_KEYS}
    _require(isinstance(scope, Mapping) and set(scope) == {'source', 'rows', 'fit', 'held_out'} | hashes,
             'OBSERVER_SCOPE')
    _ref(scope['source'])
    for part in (scope['fit'], scope['held_out']):
        _require(isinstance(part, Mapping) and set(part) == hashes | {'rows', 'cutoff_ns'}
                 and type(part['rows']) is int and part['rows'] > 0 and type(part['cutoff_ns']) is int,
                 'OBSERVER_PART')
    for block in (scope, scope['fit'], scope['held_out']):
        _require(all(isinstance(block[k], str) and len(block[k]) == 64
                     and all(c in '0123456789abcdef' for c in block[k]) for k in hashes), 'OBSERVER_DIGEST')
    _require(all(fit.get(k) == v for k, v in scope['fit'].items())
             and type(scope['rows']) is int and scope['rows'] == scope['fit']['rows'] + scope['held_out']['rows']
             and 0 < scope['fit']['cutoff_ns'] < scope['held_out']['cutoff_ns'] <= history.TEST_BOUNDARY_NS,
             'OBSERVER_PREFIX')
    return {k: _copy(v) for k, v in scope.items() if k not in ('fit', 'held_out')}


@dataclass(frozen=True, eq=False, init=False)
class NativeDirectionObserverAxes:
    """One verification per use of the immutable fit or fit+held-out axes. Contains no features."""
    view_sha256: str
    parent_rows: np.ndarray
    bar_start_ns: np.ndarray
    decision_ns: np.ndarray

    def __init__(self, view, *, source, parent_rows, bar_start_ns, decision_ns, role=None,
                 unobserved_units_profile=None):
        _require_view(view, unobserved_units_profile=unobserved_units_profile)
        scope = view.get('reserved_observer_axes', view.get('october_observer_axes'))
        _require(isinstance(scope, Mapping) and _ref(source) == scope['source'], 'OBSERVER_SOURCE')
        fit_rows = scope['fit']['rows']
        n = len(parent_rows)
        _require(role in (None, 'fit', 'score'), 'OBSERVER_ROLE')
        suffix_only = role == 'score'
        _require((n == scope['held_out']['rows'] if suffix_only else
                  n == fit_rows if role == 'fit' else n in (fit_rows, scope['rows'])), 'OBSERVER_ROW_COUNT')
        for key, supplied in zip(_OBSERVER_AXIS_KEYS, (parent_rows, bar_start_ns, decision_ns)):
            _require(isinstance(supplied, np.ndarray) and supplied.dtype == np.int64 and supplied.shape == (n,),
                     'OBSERVER_ARRAY:' + key)
            # A caller-owned read-only flag can be reversed. These small, privately
            # copied axes have immutable bytes backing for every later gather.
            axis = np.frombuffer(supplied.astype('<i8', copy=False).tobytes(), dtype='<i8')
            digest_key = key + '_sha256'
            bound, portion = (scope['held_out'], axis) if suffix_only else (scope['fit'], axis[:fit_rows])
            _require(hashlib.sha256(portion).hexdigest() == bound[digest_key]
                     and np.all(axis >= 0) and np.all(axis[1:] > axis[:-1]), 'OBSERVER_FIT_AXIS:' + key)
            if not suffix_only and n > fit_rows:
                _require(hashlib.sha256(axis[fit_rows:]).hexdigest() == scope['held_out'][digest_key]
                         and hashlib.sha256(axis).hexdigest() == scope[digest_key], 'OBSERVER_HELD_OUT_AXIS:' + key)
            object.__setattr__(self, key, axis)
        _require(np.all(self.bar_start_ns <= MAX_NS - BAR_NS) and np.all(self.bar_start_ns % BAR_NS == 0)
                 and np.array_equal(self.bar_start_ns + BAR_NS, self.decision_ns)
                 and np.searchsorted(self.decision_ns, scope['fit']['cutoff_ns']) == (0 if suffix_only else fit_rows)
                 and self.decision_ns[-1] < scope['held_out']['cutoff_ns'], 'OBSERVER_CLOCK')
        object.__setattr__(self, 'view_sha256', view['contract_sha256'])


def _require_view(view, *, unobserved_units_profile=None):
    _require(isinstance(view, Mapping), 'VIEW_KEYS')
    units_profile = unobserved_units_profile is not None
    time_query_v2 = 'market_state_time_query_v2' in view
    anchor_span = 'anchor_span_input_extension' in view
    _require(not anchor_span or units_profile, 'ANCHOR_SPAN_EXPLICIT_UNITS')
    time_query = 'market_state_time_query' in view or time_query_v2
    _require(units_profile == ('unobserved_units_profile' in view), 'EXPLICIT_UNOBSERVED_UNITS_PROFILE')
    if units_profile:
        profile = (units.require_anchor_span_fold1_prefix_units_profile(unobserved_units_profile) if anchor_span
                   else units.require_mature_fold1_prefix_units_profile(unobserved_units_profile) if time_query
                   else units.require_october_prefix_units_profile(unobserved_units_profile))
        _require(view['unobserved_units_profile'] == profile, 'UNOBSERVED_UNITS_VIEW_BINDING')
    if anchor_span:
        from gx1.contracts.entry_anchor_span_feature_view_v1 import require_anchor_span_view_provenance
        require_anchor_span_view_provenance(view, profile)
    dense = 'dense_minute_supervision' in view
    price_momentum = 'price_momentum_profile' in view
    full = 'full_feature_profile' in view
    ta_only = 'full_ta_feature_profile' in view
    dedicated_m5_side = 'dedicated_m5_side' in view
    dedicated_non_m5_side = 'dedicated_non_m5_side' in view
    dedicated_tf_activity = 'dedicated_tf_activity' in view
    dynamic_state = 'dynamic_state_supervision' in view
    five_endpoint = 'market_state_five_endpoint' in view
    current_state = 'current_state_first_loss_weight' in view
    confirmed_age = 'confirmed_run_age_input' in view
    october_source = 'october_input_plan' in view.get('source_bindings', {})
    first_activity = 'first_flat_activity_weight' in view
    endpoint = 'native_endpoint_loss' in view
    masked_profile = 'zero_support_profile' in view
    scoped_october = 'october_observer_axes' in view
    extended_state = 'extended_forecast_minutes' in view
    five_priority = 'five_minute_priority' in view
    dense_bps = 'dense_bps_scale' in view
    m1_ofi = 'm1_ofi_normalization' in view
    coarsened = 'm1_ofi_coarsened' in view
    family = 'family_intervention' in view
    initialization = view.get('price_momentum_initialization', 'zero_residual')
    relative = 'gc_relative_coordinates_sha256' in view
    _require(set(view) == VIEW_KEYS | ({'bps_scale'} if 'bps_scale' in view else set()) | ({'magnitude_weight_contract'} if 'magnitude_weight_contract' in view else set()) | ({'dense_minute_supervision'} if dense else set()) | ({'price_momentum_profile'} if price_momentum else set()) | ({'full_feature_profile'} if full else set()) | ({'full_ta_feature_profile'} if ta_only else set()) | ({'dedicated_m5_side'} if dedicated_m5_side else set()) | ({'dedicated_non_m5_side'} if dedicated_non_m5_side else set()) | ({'dedicated_tf_activity'} if dedicated_tf_activity else set()) | ({'five_minute_priority'} if five_priority else set()) | ({'dense_bps_scale'} if dense_bps else set()) | ({'m1_ofi_normalization'} if m1_ofi else set()) | ({'m1_ofi_coarsened'} if coarsened else set()) | ({'family_intervention'} if family else set()) | ({'price_momentum_initialization'} if initialization != 'zero_residual' else set()) | ({'gc_relative_coordinates_sha256'} if relative else set()) | ({'dynamic_state_supervision'} if dynamic_state else set()) | ({'market_state_five_endpoint'} if five_endpoint else set()) | ({'market_state_time_query'} if 'market_state_time_query' in view else set()) | ({'market_state_time_query_v2'} if time_query_v2 else set()) | ({'current_state_first_loss_weight'} if current_state else set()) | ({'confirmed_run_age_input'} if confirmed_age else set()) | ({'extended_forecast_minutes'} if extended_state else set()) | ({'first_flat_activity_weight'} if first_activity else set())
        | ({'native_endpoint_loss'} if endpoint else set()) | ({'zero_support_profile', 'reserved_observer_axes'} if masked_profile else set())
        | ({'unobserved_units_profile'} if units_profile else set()) | ({'october_observer_axes'} if scoped_october else set())
        | ({'anchor_span_input_extension'} if anchor_span else set())
        and not (masked_profile and units_profile)
        and (not ta_only or (view['full_ta_feature_profile'] is True and full and time_query_v2))
        and (not dedicated_m5_side or (view['dedicated_m5_side'] is True and ta_only and time_query_v2))
        and (not dedicated_non_m5_side or (view['dedicated_non_m5_side'] is True
            and dedicated_m5_side and ta_only and time_query_v2))
        and (not dedicated_tf_activity or (view['dedicated_tf_activity'] is True
            and dedicated_m5_side and ta_only and time_query_v2))
        and not (time_query_v2 and 'market_state_time_query' in view)
        and (not scoped_october or (units_profile and not masked_profile))
        and (not units_profile or (full and ((dynamic_state and endpoint) or five_endpoint or time_query)))
        and (not masked_profile or endpoint)
        and (not endpoint or (view['native_endpoint_loss'] is True and first_activity and not extended_state))
        and (not first_activity or current_state)
        and (not extended_state or current_state)
        and (not current_state or dynamic_state)
        and (not confirmed_age or (view['confirmed_run_age_input'] is True and current_state and endpoint and units_profile))
        and sum((dense, 'bps_scale' in view, 'magnitude_weight_contract' in view)) <= 1
        and (not dense or view['dense_minute_supervision'] is True)
        and (not price_momentum or (dense and view['price_momentum_profile'] is True))
        and (not full or (view['full_feature_profile'] is True and not price_momentum and not family))
        and (not dynamic_state or (view['dynamic_state_supervision'] is True and full
            and not any((dense, price_momentum, five_priority, dense_bps, m1_ofi, coarsened, family, relative))
            and 'bps_scale' not in view and 'magnitude_weight_contract' not in view
            and initialization == 'zero_residual'))
        and (not five_endpoint or (view['market_state_five_endpoint'] is True and full
            and not any((dynamic_state, dense, price_momentum, five_priority, dense_bps, m1_ofi,
                coarsened, family, relative, current_state, confirmed_age, first_activity,
                endpoint, masked_profile, extended_state))
            and 'bps_scale' not in view and 'magnitude_weight_contract' not in view
            and initialization == 'zero_residual'))
        and (not time_query or (view['market_state_time_query_v2' if time_query_v2 else 'market_state_time_query'] is True and full
            and not any((dynamic_state, five_endpoint, dense, price_momentum, five_priority, dense_bps,
                m1_ofi, coarsened, family, relative, current_state, confirmed_age, first_activity,
                endpoint, masked_profile, extended_state, scoped_october))
            and 'bps_scale' not in view and 'magnitude_weight_contract' not in view
            and initialization == 'zero_residual'))
        and initialization in ('zero_residual', 'linear_default')
        and (initialization == 'zero_residual' or (price_momentum and not family))
        and (not relative or (not family and initialization == 'zero_residual'))
        and (not five_priority or (price_momentum and not m1_ofi and view['five_minute_priority'] is True))
        and (not dense_bps or (dense and price_momentum and not five_priority and not m1_ofi))
        and (not family or (dense and price_momentum and not five_priority and not dense_bps and not m1_ofi))
        and (not m1_ofi or dense) and (not coarsened or (m1_ofi and view['m1_ofi_coarsened'] is True)), 'VIEW_KEYS')
    if time_query_v2:
        objective = factorized_time_query_model_contract(ta_only=ta_only, dedicated_m5_side=dedicated_m5_side,
            dedicated_non_m5_side=dedicated_non_m5_side, dedicated_tf_activity=dedicated_tf_activity)
    elif time_query:
        objective = time_query_model_contract()
    elif five_endpoint:
        objective = five_endpoint_model_contract()
    elif dynamic_state:
        objective = (dynamic_mtf_state_research_contract() if not current_state
            else dynamic_mtf_state_research_contract(current_state_input=True,
                first_state_loss_weight=view['current_state_first_loss_weight'],
                confirmed_run_age_input=confirmed_age,
                extended_forecast_minutes=view.get('extended_forecast_minutes'),
                first_flat_activity_weight=view.get('first_flat_activity_weight'),
                native_endpoint_loss=view.get('native_endpoint_loss', False)))
    elif dense_bps:
        from gx1.contracts.entry_native_direction_dense_v1 import dense_direction_bps_research_contract
        objective = dense_direction_bps_research_contract()
    elif five_priority:
        from gx1.contracts.entry_native_direction_five_priority_v1 import five_priority_direction_contract
        objective = five_priority_direction_contract()
    elif dense:
        from gx1.contracts.entry_native_direction_dense_v1 import dense_direction_research_contract
        objective = dense_direction_research_contract()
    else:
        objective = native_direction_research_contract(signed_bps='bps_scale' in view, magnitude_weighted='magnitude_weight_contract' in view)
    _sealed(view, 'VIEW_SEAL')
    fields, routes, widths = _canonical(ta_only=ta_only)
    _require(view['schema_version'] == SCHEMA_VERSION and view['physical_widths'] == widths
        and view['physical_field_names_sha256'] == {key: _sha(names) for key, names in fields.items()}
        and view['sequence_bars'] == MODEL_NATIVE_SEQ_LEN and view['window_rows'] == history.WINDOW_ROWS
        and view['canonical_routing_sha256'] == _sha(routes)
        and view['research_objective'] == objective
        and view['evidence_scope'] == EVIDENCE_SCOPE
        and type(view['max_batch_rows']) is int and view['max_batch_rows'] > 0, 'VIEW_CANONICAL_IDENTITY')
    _require(set(view['source_bindings']) == SOURCE_REF_KEYS | (DENSE_SOURCE_REF_KEYS if dense else set()) | (PRICE_MOMENTUM_SOURCE_REF_KEYS if price_momentum else set()) | (FIVE_PRIORITY_SOURCE_REF_KEYS if five_priority else set()) | (DENSE_BPS_SOURCE_REF_KEYS if dense_bps else set()) | (M1_OFI_SOURCE_REF_KEYS if m1_ofi else set()) | (FAMILY_SOURCE_REF_KEYS if family else set()) | (GC_RELATIVE_SOURCE_REF_KEYS if relative else set()) | (DYNAMIC_STATE_SOURCE_REF_KEYS if dynamic_state else set()) | (FIVE_ENDPOINT_SOURCE_REF_KEYS if five_endpoint else set()) | (TIME_QUERY_V2_SOURCE_REF_KEYS if time_query_v2 else TIME_QUERY_SOURCE_REF_KEYS if time_query else set()) | (CONFIRMED_AGE_SOURCE_REF_KEYS if confirmed_age else set()) | (OCTOBER_SOURCE_REF_KEYS if october_source else set()) | (ANCHOR_SPAN_SOURCE_REF_KEYS if anchor_span else set()), 'SOURCE_REF_KEYS')
    for item in view['source_bindings'].values(): _ref(item)
    if five_endpoint:
        path, sha256 = five_endpoint_targets.REFS['mask']
        _require(view['source_bindings']['five_endpoint_corrected_mask'] == {'path': path, 'sha256': sha256},
                 'FIVE_ENDPOINT_CORRECTED_MASK')
    if time_query:
        path, sha256 = time_query_targets.CANDIDATE_REFS['tape']
        _require(view['source_bindings']['time_query_candidate_tape'] == {'path': str(path), 'sha256': sha256},
                 'TIME_QUERY_CANDIDATE_TAPE')
        plan_ref = view['source_bindings']['time_query_mature_prefix_plan']
        _require((plan_ref['path'] == TIME_QUERY_MATURE_FOLD1_PLAN_PATH if units_profile
                  else plan_ref == TIME_QUERY_MATURE_PLAN_REF), 'TIME_QUERY_MATURE_PLAN')
        if time_query_v2:
            _require(view['source_bindings']['time_query_v2_model_source']['path'] == str(
                time_query_targets.ROOT / 'gx1/contracts/market_state_time_query_model_v2.py')
                and view['source_bindings']['time_query_v2_loss_source']['path'] == str(
                time_query_targets.ROOT / 'gx1/contracts/market_state_time_query_loss_v2.py'),
                'TIME_QUERY_V2_SOURCE_PATHS')
    _require(masked_profile == ('zero_support_profile_sha256' in view['prefix_normalization'])
             == ('reserved_observer_axes_sha256' in view['prefix_normalization']), 'ZERO_SUPPORT_VIEW_BINDING')
    _require(units_profile == ('unobserved_units_profile_sha256' in view['prefix_normalization']),
             'UNOBSERVED_UNITS_VIEW_BINDING')
    _require(scoped_october == ('october_observer_axes_sha256' in view['prefix_normalization']),
             'OCTOBER_OBSERVER_VIEW_BINDING')
    if units_profile:
        _require(view['prefix_normalization']['unobserved_units_profile_sha256'] == profile['contract_sha256'],
                 'UNOBSERVED_UNITS_VIEW_BINDING')
        units.require_prefix_units_fit_identity(view['prefix'], profile)
    if scoped_october:
        scope = view['october_observer_axes']
        _require(view['full_axes'] == _require_observer_scope(scope, view['prefix'])
                 and view['prefix_normalization']['october_observer_axes_sha256'] == _sha(scope),
                 'OCTOBER_OBSERVER_VIEW_BINDING')
    if masked_profile:
        profile = view['zero_support_profile']
        prefix_owner.require_december_zero_support_profile(profile)
        _require(view['prefix_normalization']['zero_support_profile_sha256'] == profile['contract_sha256'],
                 'ZERO_SUPPORT_VIEW_BINDING')
        scope = view['reserved_observer_axes']
        _require(view['full_axes'] == _require_observer_scope(scope, view['prefix'])
                 and view['prefix_normalization']['reserved_observer_axes_sha256'] == _sha(scope),
                 'OBSERVER_VIEW_BINDING')
    if relative:
        digest = view['gc_relative_coordinates_sha256']
        _require(type(digest) is str and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest), 'GC_COORDINATE_IDENTITY')
    full_name = (owners.FULL_TA_ENTRY_EXECUTION_PROFILE_NAME if ta_only
                 else owners.FULL_GC_MTF_ENTRY_EXECUTION_PROFILE_NAME)
    _require((view['entry_execution_profile']['name'] == full_name) == full, 'FULL_PROFILE_IDENTITY')
    _require(ta_only == ('ta_base_normalization_sha256' in view['prefix_normalization'])
             == ('source_owner_profile_sha256' in view['prefix_normalization']),
             'TA_NORMALIZATION_VIEW_BINDING')
    if ta_only:
        compiled_ta = owners.compile_full_ta_entry_execution_profile()
        _require(view['entry_execution_profile'] == {
            key: compiled_ta[key] for key in ('name', 'contract_sha256')},
            'TA_PROFILE_IDENTITY')
        for key in ('ta_base_normalization_sha256', 'source_owner_profile_sha256'):
            digest = view['prefix_normalization'][key]
            _require(type(digest) is str and len(digest) == 64
                     and all(c in '0123456789abcdef' for c in digest), 'TA_NORMALIZATION_HASH')
    if price_momentum:
        expected_name = (price_momentum_owner.PROFILE_NAME if initialization == 'zero_residual'
            else price_momentum_owner.LINEAR_INIT_PROFILE_NAME)
        _require(view['entry_execution_profile']['name'] == expected_name, 'PRICE_INITIALIZATION_PROFILE')
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
    if family:
        _require_family_reference(view['family_intervention'])
    if dense_bps:
        from gx1.contracts.entry_native_direction_bps_v1 import require_dense_direction_bps_scale
        scale = require_dense_direction_bps_scale(view['dense_bps_scale'])
        _require(scale['prefix_rows'] == view['prefix']['rows'] and scale['cutoff_ns'] == view['prefix']['cutoff_ns']
            and all(scale[key] == view['prefix'][key] for key in ('parent_rows_sha256', 'decision_ns_sha256')), 'DENSE_BPS_OWN_PREFIX')
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


def _require_family_reference(value):
    from gx1.contracts.entry_direction_family_intervention_v1 import BASELINE, MACRO_ARM
    _require(isinstance(value, Mapping) and set(value) == {'arm', 'contract_sha256'}
        and value['arm'] in (BASELINE, MACRO_ARM, *owners.MODEL_NATIVE_TRAINING_SPECIALISTS)
        and type(value['contract_sha256']) is str and len(value['contract_sha256']) == 64
        and all(c in '0123456789abcdef' for c in value['contract_sha256']), 'FAMILY_REFERENCE')
    return dict(value)


def family_reference_view(view):
    """Same sealed family study view with the explicit no-op reference arm."""
    _require_view(view)
    _require('family_intervention' in view, 'FAMILY_REFERENCE_VIEW')
    from gx1.contracts.entry_direction_family_intervention_v1 import BASELINE
    value = _copy(view)
    value['family_intervention']['arm'] = BASELINE
    del value['contract_sha256']
    value['contract_sha256'] = _sha(value)
    return _require_view(value)


def bind_native_direction_feature_view(*, prefix_normalization, prefix_normalization_ref,
    constructor_metadata, constructor_metadata_ref, source_bindings, max_batch_rows, expected=None, bps_scale=None, magnitude_weight_contract=None, dense_minutes=False, price_momentum=False, m1_ofi_normalization=None, m1_ofi_coarsened=False, five_minute_priority=False, dense_bps_scale=None, family_intervention=None, price_momentum_initialization='zero_residual', full_feature_profile=False, dynamic_state_supervision=False, market_state_five_endpoint=False, current_state_first_loss_weight=None, extended_forecast_minutes=None, first_flat_activity_weight=None, native_endpoint_loss=False, confirmed_run_age_input=False, zero_support_profile=None, unobserved_units_profile=None, october_observer_axes=None, market_state_time_query=False, market_state_time_query_v2=False, full_ta_feature_profile=False, dedicated_m5_side=False, dedicated_non_m5_side=False, dedicated_tf_activity=False):
    """Bind verified metadata; ``expected`` requires exact reconstruction of a view.

    Plainrefs identify caller-verified files. Content seals bind their decoded
    objects separately. No file hash or genuine-native qualification occurs here.
    Old constructor metadata may retain old artifact provenance; only its explicit
    scalar configuration/full raw routing are used, never its old GC stats join.
    """
    arguments = locals().copy()
    if prefix_normalization.get('schema_version') == 'gx1_anchor_span_feature_prefix_packet_v1':
        from gx1.contracts.entry_anchor_span_feature_view_v1 import bind_anchor_span_feature_view
        return bind_anchor_span_feature_view(**arguments)
    time_query_mode = market_state_time_query or market_state_time_query_v2
    _require(type(dedicated_m5_side) is bool and (not dedicated_m5_side
             or (full_ta_feature_profile is True and market_state_time_query_v2 is True)),
             'DEDICATED_M5_SIDE_SCOPE')
    _require(type(dedicated_non_m5_side) is bool and (not dedicated_non_m5_side
             or (dedicated_m5_side is True and full_ta_feature_profile is True
                 and market_state_time_query_v2 is True)), 'DEDICATED_NON_M5_SIDE_SCOPE')
    _require(type(dedicated_tf_activity) is bool and (not dedicated_tf_activity
             or (dedicated_m5_side is True and full_ta_feature_profile is True
                 and market_state_time_query_v2 is True)),
             'DEDICATED_TF_ACTIVITY_SCOPE')
    _require(type(full_ta_feature_profile) is bool
             and (not full_ta_feature_profile or (full_feature_profile is True
                  and market_state_time_query_v2 is True)), 'TA_PROFILE_SCOPE')
    _require(zero_support_profile is None or unobserved_units_profile is None, 'PROFILE_MIX')
    _require(october_observer_axes is None or (unobserved_units_profile is not None
             and zero_support_profile is None), 'OCTOBER_OBSERVER_PROFILE')
    if unobserved_units_profile is not None:
        unobserved_units_profile = (units.require_mature_fold1_prefix_units_profile(unobserved_units_profile)
            if time_query_mode else units.require_october_prefix_units_profile(unobserved_units_profile))
    units_kwargs = ({} if unobserved_units_profile is None
                    else {'unobserved_units_profile': unobserved_units_profile})
    _sealed(prefix_normalization, 'PREFIX_NORMALIZATION_SEAL')
    _require(set(prefix_normalization) == {'schema_version', 'base_normalization', 'gc_mtf_history',
        'prefix_qualification', 'old055_or223_normalization_identity_reused', 'native_launch_authorized', 'genuine_model_fit_authorized'}
        | {'contract_sha256'} | ({'zero_support_profile'} if zero_support_profile is not None else set())
        | ({'unobserved_units_profile'} if unobserved_units_profile is not None else set())
        and prefix_normalization['schema_version'] == 'gx1_native_direction_prefix_normalization_v1'
        and prefix_normalization['old055_or223_normalization_identity_reused'] is False
        and prefix_normalization['native_launch_authorized'] is prefix_normalization['genuine_model_fit_authorized'] is False, 'PREFIX_NORMALIZATION_SCOPE')
    report = prefix_normalization['prefix_qualification']; _sealed(report, 'PREFIX_QUALIFICATION_SEAL')
    if zero_support_profile is not None:
        _require(native_endpoint_loss is True and prefix_normalization['zero_support_profile'] == zero_support_profile
                 and report.get('zero_support_profile') == zero_support_profile, 'ZERO_SUPPORT_FIT_BINDING')
        prefix_owner.require_december_normalization_bindings(prefix_normalization, zero_support_profile)
    else:
        _require('zero_support_profile' not in report, 'UNBOUND_ZERO_SUPPORT_PROFILE')
    if unobserved_units_profile is not None:
        _require(full_feature_profile is True
                 and ((dynamic_state_supervision and native_endpoint_loss) or market_state_five_endpoint
                      or time_query_mode)
                 and prefix_normalization['unobserved_units_profile'] == unobserved_units_profile
                 and report.get('unobserved_units_profile') == unobserved_units_profile, 'UNOBSERVED_UNITS_FIT_BINDING')
        if time_query_mode:
            prefix_owner.require_prefix_units_normalization_bindings(prefix_normalization, unobserved_units_profile)
        else:
            prefix_owner.require_october_units_normalization_bindings(prefix_normalization, unobserved_units_profile)
    else:
        _require('unobserved_units_profile' not in report, 'UNBOUND_UNOBSERVED_UNITS_PROFILE')
    _require(report.get('fit_allowed') is True and report.get('blockers') == [] and report.get('statistics_fits') == 0, 'SUPPORTED_PREFIX')
    proof = report['proof']; fields, routes, widths = _canonical()
    base = norm.require_input_normalization_contract(prefix_normalization['base_normalization'],
        expected_field_names={name: list(fields['signal'] if name == 'signal' else fields['ctx_cont'] if name == 'ctx_cont' else fields['mtf'])
            for name in norm.EXPECTED_SURFACES}, expected_ctx_cat_names=fields['ctx_cat'],
        expected_entry_training_phase=ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE, **units_kwargs)
    gc_history = history.require_gc_mtf_history(prefix_normalization['gc_mtf_history'], **units_kwargs)
    if zero_support_profile is None:
        surfaces = (*base['surfaces'].values(), *gc_history['normalization']['surfaces'].values())
        _require(all('zero_support_profile_sha256' not in surface for surface in surfaces), 'UNBOUND_ZERO_SUPPORT_PROFILE')
    if unobserved_units_profile is None:
        surfaces = (*base['surfaces'].values(), *gc_history['normalization']['surfaces'].values())
        _require(all('unobserved_units_profile_sha256' not in surface for surface in surfaces), 'UNBOUND_UNOBSERVED_UNITS_PROFILE')
    relative = gc_history['normalization'].get('relative_price')
    _require(price_momentum_initialization in ('zero_residual', 'linear_default')
        and (price_momentum_initialization == 'zero_residual' or (price_momentum and family_intervention is None))
        and (relative is None or (family_intervention is None and price_momentum_initialization == 'zero_residual')), 'SEPARATE_DESIGN_VARIANT')
    _require(base['lineage']['train_row_count'] == proof['prefix']['rows']
        and base['lineage']['context_fit_row_count'] == proof['prefix']['rows']
        and base['lineage']['local_fit_row_count'] == proof['local']['selected_unique_row_count']
        and all(base['lineage'][key] == value for key, value in proof['base_lineage'].items())
        and gc_history['normalization']['provenance']['train_end_ns'] == proof['prefix']['cutoff_ns'], 'PREFIX_LINEAGE_JOIN')
    for tf, item in gc_history['normalization']['provenance']['row_union'].items():
        _require(all(item[key] == proof['gc'][tf][key] for key in ('rows', 'native_row_indices_sha256', 'available_ns_sha256')), 'GC_PREFIX_ROW_UNION:' + tf)
    _require(type(dense_minutes) is bool and type(price_momentum) is bool and type(m1_ofi_coarsened) is bool and type(five_minute_priority) is bool
        and type(full_feature_profile) is bool and (not full_feature_profile or (not price_momentum and family_intervention is None))
        and type(dynamic_state_supervision) is bool and type(native_endpoint_loss) is bool
        and type(market_state_five_endpoint) is bool and type(market_state_time_query) is bool
        and type(market_state_time_query_v2) is bool and not (market_state_time_query and market_state_time_query_v2)
        and type(confirmed_run_age_input) is bool
        and (not market_state_five_endpoint or (full_feature_profile and not any((
            dynamic_state_supervision, dense_minutes, price_momentum, m1_ofi_coarsened,
            five_minute_priority, confirmed_run_age_input, native_endpoint_loss))
            and current_state_first_loss_weight is None and first_flat_activity_weight is None
            and extended_forecast_minutes is None and bps_scale is None
            and magnitude_weight_contract is None and dense_bps_scale is None
            and m1_ofi_normalization is None and family_intervention is None
            and relative is None and zero_support_profile is None
            and price_momentum_initialization == 'zero_residual'))
        and (not time_query_mode or (full_feature_profile
            and not any((market_state_five_endpoint, dynamic_state_supervision, dense_minutes,
                price_momentum, m1_ofi_coarsened, five_minute_priority, confirmed_run_age_input,
                native_endpoint_loss)) and current_state_first_loss_weight is None
            and first_flat_activity_weight is None and extended_forecast_minutes is None
            and bps_scale is None and magnitude_weight_contract is None and dense_bps_scale is None
            and m1_ofi_normalization is None and family_intervention is None and relative is None
            and zero_support_profile is None and october_observer_axes is None
            and price_momentum_initialization == 'zero_residual'))
        and (not confirmed_run_age_input or (current_state_first_loss_weight is not None
            and native_endpoint_loss and unobserved_units_profile is not None))
        and (not native_endpoint_loss or (first_flat_activity_weight is not None and extended_forecast_minutes is None))
        and (current_state_first_loss_weight is None or dynamic_state_supervision)
        and (first_flat_activity_weight is None or current_state_first_loss_weight is not None)
        and (extended_forecast_minutes is None or current_state_first_loss_weight is not None)
        and (not dynamic_state_supervision or (full_feature_profile and not any((dense_minutes, price_momentum,
            m1_ofi_coarsened, five_minute_priority)) and bps_scale is None and magnitude_weight_contract is None
            and dense_bps_scale is None and m1_ofi_normalization is None and family_intervention is None
            and relative is None and price_momentum_initialization == 'zero_residual'))
        and not (dense_minutes and (bps_scale is not None or magnitude_weight_contract is not None))
        and (not price_momentum or dense_minutes)
        and (not five_minute_priority or (price_momentum and m1_ofi_normalization is None))
        and (dense_bps_scale is None or (dense_minutes and price_momentum and not five_minute_priority and m1_ofi_normalization is None))
        and (m1_ofi_normalization is None or dense_minutes)
        and (not m1_ofi_coarsened or m1_ofi_normalization is not None), 'OBJECTIVE_MODE')
    provenance = proof['source_bindings']
    october_provenance = unobserved_units_profile is not None and set(provenance) == {
            'ancestral_264_plan', 'ancestral_267_plan', 'current_plan', 'prior_october_support'}
    mature_provenance = time_query_mode and isinstance(provenance, Mapping) and set(provenance) == {
        'boolean_raw_domain_transition', 'boolean_raw_gc_layout', 'boolean_raw_result',
        'cohort_rule', 'current_plan', 'current_policy', 'current_source_head',
        'current_source_owners', 'model_axes', 'provider_plan', 'provider_worker'}
    _require(not time_query_mode or mature_provenance, 'TIME_QUERY_MATURE_PROVENANCE')
    _require(isinstance(source_bindings, Mapping) and set(source_bindings) == SOURCE_REF_KEYS | (DENSE_SOURCE_REF_KEYS if dense_minutes else set()) | (PRICE_MOMENTUM_SOURCE_REF_KEYS if price_momentum else set()) | (FIVE_PRIORITY_SOURCE_REF_KEYS if five_minute_priority else set()) | (DENSE_BPS_SOURCE_REF_KEYS if dense_bps_scale is not None else set()) | (M1_OFI_SOURCE_REF_KEYS if m1_ofi_normalization is not None else set()) | (FAMILY_SOURCE_REF_KEYS if family_intervention is not None else set()) | (GC_RELATIVE_SOURCE_REF_KEYS if relative is not None else set()) | (DYNAMIC_STATE_SOURCE_REF_KEYS if dynamic_state_supervision else set()) | (FIVE_ENDPOINT_SOURCE_REF_KEYS if market_state_five_endpoint else set()) | (TIME_QUERY_V2_SOURCE_REF_KEYS if market_state_time_query_v2 else TIME_QUERY_SOURCE_REF_KEYS if market_state_time_query else set()) | (CONFIRMED_AGE_SOURCE_REF_KEYS if confirmed_run_age_input else set()) | (OCTOBER_SOURCE_REF_KEYS if october_provenance else set()), 'SOURCE_REF_KEYS')
    source_bindings = {key: _ref(value) for key, value in source_bindings.items()}
    if mature_provenance:
        fold1_units = unobserved_units_profile is not None
        cohorts = {five_endpoint_targets.FOLD_BOUNDARIES_NS[0] - 288 * BAR_NS: (1, 29574)}
        cohorts.update({boundary - 288 * BAR_NS: (fold_id, rows) for fold_id, (boundary, rows) in
            enumerate(zip(five_endpoint_targets.FOLD_BOUNDARIES_NS[1:5], TIME_QUERY_MATURE_FIT_ROWS), start=2)})
        cohort = cohorts.get(proof['prefix']['cutoff_ns'])
        prefix_root = time_query_targets.RUNS / ('MARKET_STATE_MATURE_FOLD1_NORMALIZATION_20261107_001'
            if fold1_units else 'MARKET_STATE_MATURE_PREFIX_NORMALIZATION_20261107_001')
        owner_refs = provenance['current_source_owners']
        owner_paths = ('gx1/features/htf_features.py',
                       'gx1/contracts/entry_gc_mtf_history_v1.py',
                       'gx1/features/entry_specialist_feature_groups_v1.py')
        _require(isinstance(owner_refs, Mapping) and all(path in owner_refs for path in owner_paths),
                 'TIME_QUERY_MATURE_SOURCE_OWNERS')
        _require((provenance['current_plan']['path'] == TIME_QUERY_MATURE_FOLD1_PLAN_PATH if fold1_units
                  else provenance['current_plan'] == TIME_QUERY_MATURE_PLAN_REF)
            and source_bindings['time_query_mature_prefix_plan'] == provenance['current_plan']
            and provenance['cohort_rule'] == 'decision_ns_strictly_below_official_cutoff_minus_1440_minutes'
            and cohort is not None and (cohort[0] == 1) == fold1_units
            and proof['prefix']['rows'] == cohort[1]
            and prefix_normalization_ref['path'] == str(prefix_root / f'PREFIX_NORMALIZATION_FOLD_{cohort[0]}.json')
            and source_bindings['input_qualification']['path'] == str(prefix_root / f'QUALIFICATION_FOLD_{cohort[0]}.json')
            and proof['full_axes']['rows'] == TIME_QUERY_MODEL_AXIS_ROWS
            and proof['full_axes']['source'] == provenance['model_axes']
            and source_bindings['raw_input_provider_plan'] == provenance['provider_plan']
            and source_bindings['raw_gc_layout'] == provenance['boolean_raw_gc_layout']
            and source_bindings['raw_gc_transition'] == provenance['boolean_raw_domain_transition']
            and all(source_bindings[key]['path'] == str(time_query_targets.ROOT / owner)
                    and ((full_ta_feature_profile and key == 'owner_profile_source')
                         or source_bindings[key]['sha256'] == owner_refs[owner])
                    for key, owner in (
                ('mtf_window_source', 'gx1/features/htf_features.py'),
                ('gc_gather_source', 'gx1/contracts/entry_gc_mtf_history_v1.py'),
                ('owner_profile_source', 'gx1/features/entry_specialist_feature_groups_v1.py')))
            and all(_ref(provenance[key]) for key in ('current_policy', 'boolean_raw_result', 'provider_worker'))
            and isinstance(provenance['current_source_head'], str)
            and len(provenance['current_source_head']) == 40
            and all(c in '0123456789abcdef' for c in provenance['current_source_head']),
            'TIME_QUERY_MATURE_PROVENANCE_JOINS')
    elif october_provenance:
        # October's qualified input-only proof binds its own plan and ancestry.
        # The caller verifies every file ref before this metadata-only binding.
        _require(source_bindings['raw_input_provider_plan'] == provenance['ancestral_264_plan']
                 and source_bindings['october_input_plan'] == provenance['current_plan']
                 and source_bindings['input_qualification']['path'] ==
                     str(PurePosixPath(prefix_normalization_ref['path']).with_name('QUALIFICATION.json')),
                 'OCTOBER_SOURCE_PROVENANCE_JOINS')
    else:
        _require(source_bindings['raw_input_provider_plan'] == provenance['ancestral264_plan']
            and source_bindings['raw_gc_layout'] == provenance['current_gc_layout']
            and source_bindings['raw_gc_transition'] == provenance['current_gc_raw_transition']
            and all(source_bindings[key] == provenance['current_source_owners'][owner] for key, owner in (
                ('mtf_window_source', 'gx1/features/htf_features.py'), ('gc_gather_source', 'gx1/contracts/entry_gc_mtf_history_v1.py'),
                ('owner_profile_source', 'gx1/features/entry_specialist_feature_groups_v1.py'))), 'SOURCE_PROVENANCE_JOINS')
    scalars = _configuration(constructor_metadata)
    compiled = (owners.compile_full_ta_entry_execution_profile() if full_ta_feature_profile
        else owners.compile_full_gc_mtf_entry_execution_profile(gc_history, **units_kwargs) if full_feature_profile
        else price_momentum_owner.compile_price_momentum_profile(gc_history, initialization=price_momentum_initialization, **units_kwargs) if price_momentum
        else owners.compile_gc_mtf_geometry_entry_execution_profile(gc_history, **units_kwargs) if dense_minutes
        else owners.compile_gc_mtf_entry_execution_profile(gc_history, **units_kwargs))
    if family_intervention is not None:
        from gx1.contracts.entry_direction_family_intervention_v1 import compile_family_intervention
        reference = _require_family_reference(family_intervention)
        _require(dense_minutes and price_momentum and not five_minute_priority and dense_bps_scale is None
            and m1_ofi_normalization is None, 'FAMILY_RESEARCH_PROFILE')
        intervention = compile_family_intervention(profile_reference={k: compiled[k] for k in ('name', 'contract_sha256')},
            gc_mtf_history=gc_history, base_normalization=base, gc_normalization=gc_history['normalization'])
        _require(reference['contract_sha256'] == intervention['contract_sha256'], 'FAMILY_NORMALIZATION_PROFILE_BINDING')
    _require(type(max_batch_rows) is int and max_batch_rows > 0, 'MAX_BATCH_ROWS')
    if market_state_time_query_v2:
        objective = factorized_time_query_model_contract(ta_only=full_ta_feature_profile, dedicated_m5_side=dedicated_m5_side,
            dedicated_non_m5_side=dedicated_non_m5_side, dedicated_tf_activity=dedicated_tf_activity)
    elif market_state_time_query:
        objective = time_query_model_contract()
    elif market_state_five_endpoint:
        objective = five_endpoint_model_contract()
    elif dynamic_state_supervision:
        objective = (dynamic_mtf_state_research_contract() if current_state_first_loss_weight is None
            else dynamic_mtf_state_research_contract(current_state_input=True,
                first_state_loss_weight=current_state_first_loss_weight, confirmed_run_age_input=confirmed_run_age_input,
                extended_forecast_minutes=extended_forecast_minutes,
                first_flat_activity_weight=first_flat_activity_weight, native_endpoint_loss=native_endpoint_loss))
    elif dense_bps_scale is not None:
        from gx1.contracts.entry_native_direction_dense_v1 import dense_direction_bps_research_contract
        objective = dense_direction_bps_research_contract()
    elif five_minute_priority:
        from gx1.contracts.entry_native_direction_five_priority_v1 import five_priority_direction_contract
        objective = five_priority_direction_contract()
    elif dense_minutes:
        from gx1.contracts.entry_native_direction_dense_v1 import dense_direction_research_contract
        objective = dense_direction_research_contract()
    else:
        objective = native_direction_research_contract(signed_bps=bps_scale is not None, magnitude_weighted=magnitude_weight_contract is not None)
    if full_ta_feature_profile:
        ta_base = norm.project_ta_input_normalization_contract(base, **units_kwargs)
        fields, routes, widths = _canonical(ta_only=True)
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
    if dense_bps_scale is not None:
        value['dense_bps_scale'] = _copy(dense_bps_scale)
    if magnitude_weight_contract is not None:
        value['magnitude_weight_contract'] = _copy(magnitude_weight_contract)
    if dense_minutes:
        value['dense_minute_supervision'] = True
    if price_momentum:
        value['price_momentum_profile'] = True
    if full_feature_profile:
        value['full_feature_profile'] = True
    if full_ta_feature_profile:
        value['full_ta_feature_profile'] = True
        value['prefix_normalization']['ta_base_normalization_sha256'] = ta_base['contract_sha256']
        value['prefix_normalization']['source_owner_profile_sha256'] = (
            provenance['current_source_owners']['gx1/features/entry_specialist_feature_groups_v1.py'])
    if dynamic_state_supervision:
        value['dynamic_state_supervision'] = True
    if market_state_five_endpoint:
        value['market_state_five_endpoint'] = True
    if market_state_time_query:
        value['market_state_time_query'] = True
    if market_state_time_query_v2:
        value['market_state_time_query_v2'] = True
    if dedicated_m5_side:
        value['dedicated_m5_side'] = True
    if dedicated_non_m5_side:
        value['dedicated_non_m5_side'] = True
    if dedicated_tf_activity:
        value['dedicated_tf_activity'] = True
    if current_state_first_loss_weight is not None:
        value['current_state_first_loss_weight'] = current_state_first_loss_weight
    if confirmed_run_age_input:
        value['confirmed_run_age_input'] = True
    if first_flat_activity_weight is not None:
        value['first_flat_activity_weight'] = first_flat_activity_weight
    if native_endpoint_loss:
        value['native_endpoint_loss'] = True
    if zero_support_profile is not None:
        value['zero_support_profile'] = _copy(zero_support_profile)
        value['prefix_normalization']['zero_support_profile_sha256'] = zero_support_profile['contract_sha256']
        scope = provenance.get('reserved_observer_axes')
        full = _require_observer_scope(scope, proof['prefix'])
        _require(full['source'] == provenance.get('physical_axes')
                 and proof['full_axes'].get('source') == provenance.get('fit_target_view')
                 and all(proof['full_axes'].get(k) == v for k, v in scope['fit'].items() if k != 'cutoff_ns'),
                 'OBSERVER_FIT_LINEAGE')
        value['full_axes'] = full
        value['reserved_observer_axes'] = _copy(scope)
        value['prefix_normalization']['reserved_observer_axes_sha256'] = _sha(scope)
    if unobserved_units_profile is not None:
        value['unobserved_units_profile'] = _copy(unobserved_units_profile)
        value['prefix_normalization']['unobserved_units_profile_sha256'] = unobserved_units_profile['contract_sha256']
    if october_observer_axes is not None:
        full = _require_observer_scope(october_observer_axes, proof['prefix'])
        value['october_observer_axes'] = _copy(october_observer_axes)
        value['full_axes'] = full
        value['prefix_normalization']['october_observer_axes_sha256'] = _sha(october_observer_axes)
    if extended_forecast_minutes is not None:
        value['extended_forecast_minutes'] = extended_forecast_minutes
    if price_momentum_initialization != 'zero_residual':
        value['price_momentum_initialization'] = price_momentum_initialization
    if relative is not None:
        value['gc_relative_coordinates_sha256'] = relative['layout']['contract_sha256']
    if five_minute_priority:
        value['five_minute_priority'] = True
    if m1_ofi_normalization is not None:
        value['m1_ofi_normalization'] = _copy(m1_ofi_normalization)
    if m1_ofi_coarsened:
        value['m1_ofi_coarsened'] = True
    if family_intervention is not None:
        value['family_intervention'] = _copy(reference)
    value['contract_sha256'] = _sha(value)
    _require_view(value, **units_kwargs)
    _require(expected is None or expected == value, 'EXPECTED_VIEW_MISMATCH')
    return value


def gather_native_direction_feature_batch(view, *, parent_rows, bar_start_ns, decision_ns,
    entry_sequence_source, snap_x, ctx_cont, ctx_cat, multi_tf_sources, gc_surfaces, gc_layout, batch_parent_rows,
    m1_ofi_source=None, m1_ofi_clock=None, observer_axes=None, unobserved_units_profile=None,
    anchor_span_inputs=None):
    """Gather one bounded raw batch from caller-verified immutable full backings.

    The normalization cutoff describes fitted statistics; later OOF feature rows
    remain legal. Training target maturity is handled only by the separate loss.
    No labels, future endpoints, padding or resampling enter these feature arrays.
    """
    _require_view(view, unobserved_units_profile=unobserved_units_profile)
    fields, routes, widths = _canonical(); rows = view['full_axes']['rows']
    ta_only = view.get('full_ta_feature_profile', False)
    ta_profile = owners.compile_full_ta_entry_execution_profile() if ta_only else None
    anchor_span = 'anchor_span_input_extension' in view
    if anchor_span:
        from gx1.contracts.entry_anchor_span_feature_inputs_v1 import QualifiedAnchorSpanInputs
        _require(type(anchor_span_inputs) is QualifiedAnchorSpanInputs, 'QUALIFIED_ANCHOR_SPAN_INPUTS')
        anchor_span_inputs.require_binding(view, entry_sequence_source=entry_sequence_source,
            snap_x=snap_x, ctx_cont=ctx_cont, ctx_cat=ctx_cat, multi_tf_sources=multi_tf_sources)
    else:
        _require(anchor_span_inputs is None, 'UNBOUND_ANCHOR_SPAN_INPUTS')
    source_width = anchor_span_inputs.parent_signal_width if anchor_span else widths['signal']
    if ta_only:
        _require(gc_surfaces is None and gc_layout is None, 'TA_GC_INPUT_FORBIDDEN')
    if 'zero_support_profile' in view or 'october_observer_axes' in view:
        _require(type(observer_axes) is NativeDirectionObserverAxes and observer_axes.view_sha256 == view['contract_sha256']
                 and parent_rows is observer_axes.parent_rows and bar_start_ns is observer_axes.bar_start_ns
                 and decision_ns is observer_axes.decision_ns, 'BOUND_OBSERVER_AXES_REQUIRED')
        rows = len(parent_rows)
    else:
        _require(observer_axes is None, 'UNBOUND_OBSERVER_AXES')
    for axis in (parent_rows, bar_start_ns, decision_ns):
        _require(isinstance(axis, np.ndarray) and axis.dtype == np.int64 and axis.shape == (rows,)
            and not axis.flags.writeable, 'FULL_AXIS_HEADER')
    selected = np.asarray(batch_parent_rows)
    _require(selected.dtype == np.int64 and selected.ndim == 1 and 0 < len(selected) <= view['max_batch_rows']
        and len(set(selected.tolist())) == len(selected), 'BOUNDED_PARENT_BATCH')
    at = np.searchsorted(parent_rows, selected)
    _require(np.all(at < rows) and np.array_equal(parent_rows[at], selected), 'EXACT_PARENT_IDS')
    if anchor_span:
        if anchor_span_inputs.later_parent_positions is None:
            _require(np.all(at < anchor_span_inputs.fit_parent_rows), 'ANCHOR_SPAN_PREFIX_ONLY')
        else:
            anchor_span_inputs.require_later_batch(at, selected, decision_ns[at])
    starts, decisions = bar_start_ns[at], decision_ns[at]
    _require(np.all(starts >= 0) and np.all(starts % BAR_NS == 0) and np.all(starts <= MAX_NS - BAR_NS)
        and np.array_equal(decisions, starts + BAR_NS), 'EXACT_DECISION_CLOCK')
    _require(isinstance(entry_sequence_source, Mapping) and set(entry_sequence_source) == {'signal', 'times_ns', 'sample_positions', 'audit'}, 'SEQUENCE_SOURCE_KEYS')
    source, times, positions = (entry_sequence_source[key] for key in ('signal', 'times_ns', 'sample_positions'))
    _require(isinstance(source, np.ndarray) and source.dtype == np.float32 and source.ndim == 2
        and source.shape[1] == source_width and not source.flags.writeable
        and isinstance(times, np.ndarray) and times.dtype == np.int64 and times.shape == (len(source),) and not times.flags.writeable
        and isinstance(positions, np.ndarray) and positions.dtype == np.int64 and positions.shape == (rows,) and not positions.flags.writeable, 'SOURCE_HEADERS')
    for array, width, dtype in ((snap_x, source_width, np.float32), (ctx_cont, widths['ctx_cont'], np.float32), (ctx_cat, widths['ctx_cat'], np.int64)):
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
        if anchor_span:
            ids = np.arange(left, int(position) + 1, dtype=np.int64)
            window = anchor_span_inputs.local_batch(source, ids, ids, clock)
        else:
            window = source[left:int(position) + 1]
            if ta_only:
                window = np.ascontiguousarray(window[:, ta_profile['source_signal_indices']])
        sequences.append(window)
        clock_rows.append(clock)
    if ta_only:
        # Select canonical non-GC columns directly, before any batch copy.
        snap = (anchor_span_inputs.local_batch(snap_x, at, take, starts) if anchor_span
                else snap_x[np.ix_(at, ta_profile['source_signal_indices'])])
        context = ctx_cont[np.ix_(at, ta_profile['source_ctx_cont_indices'])]
        _, routes, _ = _canonical(ta_only=True)
    else:
        snap = snap_x[at]
        context = ctx_cont[at]
    features = {'seq_x': np.stack(sequences), 'snap_x': np.ascontiguousarray(snap),
        'ctx_cont': np.ascontiguousarray(context), 'ctx_cat': np.ascontiguousarray(ctx_cat[at])}
    _require(np.array_equal(features['seq_x'][:, -1], features['snap_x'])
        and np.array_equal(features['snap_x'][:, routes['temporal_alias_signal_indices']],
            features['ctx_cont'][:, routes['temporal_alias_ctx_cont_indices']]), 'CURRENT_SEQUENCE_ALIAS_PARITY')
    if not ta_only:
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
            if not ta_only:
                gc_collected[tf].append(history.gather_gc_mtf_history_rows(gc_surfaces[tf], tf=tf,
                    native_row_indices=ids, available_ns=available, layout=layout))
            original_ids = anchor_span_inputs.original_mtf_rows(tf, ids) if anchor_span else ids
            witnesses[tf].append({'native_row_indices': original_ids.tolist(), 'available_ns': available.tolist()})
    for tf in history.TIMEFRAMES:
        features['seq_' + tf.lower()] = np.stack(collected[tf])
        if not ta_only:
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
    if 'zero_support_profile' in view:
        # Raw alias, source and finite checks run before projecting selected copies.
        # One sealed field inventory applies on both sides of the fit cutoff.
        for key, surface in (('seq_x', 'signal'), ('snap_x', 'signal'), ('ctx_cont', 'ctx_cont'),
                             ('gc_seq_d1', 'gc_mtf_d1')):
            features[key] = prefix_owner.project_december_zero_support(
                features[key], surface=surface, profile=view['zero_support_profile'])
    return {'features': features, 'witness': {'view_sha256': view['contract_sha256'], 'parent_rows': selected.tolist(),
        'bar_start_ns': starts.tolist(), 'decision_ns': decisions.tolist(), 'source_positions': take.tolist(),
        'local_sequence_clock_sha256': hashlib.sha256(np.stack(clock_rows).astype('<i8', copy=False).tobytes()).hexdigest(),
        'multi_tf': witnesses, 'raw_values_not_normalized': True}}


def build_native_direction_research_model(
    view, *, prefix_normalization, constructor_metadata, device, unobserved_units_profile=None,
):
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
        m1_ofi_normalization=view.get('m1_ofi_normalization'), m1_ofi_coarsened=view.get('m1_ofi_coarsened', False),
        five_minute_priority=view.get('five_minute_priority', False), dense_bps_scale=view.get('dense_bps_scale'),
        family_intervention=view.get('family_intervention'),
        price_momentum_initialization=view.get('price_momentum_initialization', 'zero_residual'),
        full_feature_profile=view.get('full_feature_profile', False),
        full_ta_feature_profile=view.get('full_ta_feature_profile', False),
        dedicated_m5_side=view.get('dedicated_m5_side', False),
        dedicated_non_m5_side=view.get('dedicated_non_m5_side', False),
        dedicated_tf_activity=view.get('dedicated_tf_activity', False),
        dynamic_state_supervision=view.get('dynamic_state_supervision', False),
        market_state_five_endpoint=view.get('market_state_five_endpoint', False),
        market_state_time_query=view.get('market_state_time_query', False),
        market_state_time_query_v2=view.get('market_state_time_query_v2', False),
        current_state_first_loss_weight=view.get('current_state_first_loss_weight'),
        confirmed_run_age_input=view.get('confirmed_run_age_input', False),
        extended_forecast_minutes=view.get('extended_forecast_minutes'),
        first_flat_activity_weight=view.get('first_flat_activity_weight'),
        native_endpoint_loss=view.get('native_endpoint_loss', False), zero_support_profile=view.get('zero_support_profile'),
        october_observer_axes=view.get('october_observer_axes'),
        **({} if unobserved_units_profile is None else {'unobserved_units_profile': unobserved_units_profile}))
    ta_only = view.get('full_ta_feature_profile', False)
    _, routes, widths = _canonical(ta_only=ta_only)
    runtime_norm = (norm.project_ta_input_normalization_contract(
        prefix_normalization['base_normalization'], unobserved_units_profile=unobserved_units_profile)
        if ta_only else prefix_normalization['base_normalization'])
    if ta_only:
        _require(runtime_norm['contract_sha256']
                 == view['prefix_normalization']['ta_base_normalization_sha256'],
                 'TA_NORMALIZATION_CONSTRUCTION_BINDING')
    runtime_gc = None if ta_only else prefix_normalization['gc_mtf_history']
    kwargs = _copy(routes)
    kwargs.update(view['constructor_scalar_config'])
    kwargs.update(seq_input_dim=widths['signal'], snap_input_dim=widths['signal'], seq_len=MODEL_NATIVE_SEQ_LEN,
        ctx_cont_dim=widths['ctx_cont'], ctx_cat_dim=widths['ctx_cat'],
        input_normalization=runtime_norm,
        gc_mtf_history=runtime_gc, entry_training_phase=ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE,
        dynamic_m5_query_minutes=tuple(DYNAMIC_M1_PREFIX_HORIZON_MINUTES), entry_execution_profile=view['entry_execution_profile'])
    for tf, length in history.WINDOW_ROWS.items():
        kwargs[tf.lower() + '_seq_dim'] = widths['mtf']; kwargs[tf.lower() + '_seq_len'] = length
    if 'zero_support_profile' in view:
        kwargs['zero_support_profile'] = view['zero_support_profile']
    if unobserved_units_profile is not None and not ta_only:
        kwargs['research_unobserved_units_profile'] = _copy(unobserved_units_profile)
    backbone = EntryV10CtxHybridTransformer(**kwargs).to(device)
    _require(type(backbone) is EntryV10CtxHybridTransformer and backbone.entry_execution_profile == view['entry_execution_profile']
        and backbone.gc_mtf_history == runtime_gc, 'CONSTRUCTED_NATIVE_BINDING')
    backbone.require_input_normalization_state()
    if view.get('market_state_time_query_v2', False):
        wrapper = FactorizedTimeQueryMarketStateModel(
            backbone, dedicated_m5_side=view.get('dedicated_m5_side', False),
            dedicated_non_m5_side=view.get('dedicated_non_m5_side', False),
            dedicated_tf_activity=view.get('dedicated_tf_activity', False))
    elif view.get('market_state_time_query', False):
        wrapper = TimeQueryMarketStateModel(backbone)
    elif view.get('market_state_five_endpoint', False):
        wrapper = FiveEndpointMarketStateModel(backbone)
    elif view.get('dynamic_state_supervision', False):
        first_weight = view.get('current_state_first_loss_weight')
        wrapper = (NativeDynamicMTFStateResearch(backbone) if first_weight is None
            else NativeDynamicMTFStateResearch(backbone, current_state_input=True, first_state_loss_weight=first_weight,
                confirmed_run_age_input=view.get('confirmed_run_age_input', False),
                extended_forecast_minutes=view.get('extended_forecast_minutes'),
                first_flat_activity_weight=view.get('first_flat_activity_weight'),
                native_endpoint_loss=view.get('native_endpoint_loss', False)))
    elif view.get('dense_minute_supervision', False):
        from gx1.contracts.entry_native_direction_dense_v1 import DenseNativeEntryDirectionResearch
        wrapper = DenseNativeEntryDirectionResearch(backbone, m1_ofi_normalization=view.get('m1_ofi_normalization'),
            m1_ofi_coarsened=view.get('m1_ofi_coarsened', False), five_minute_priority=view.get('five_minute_priority', False),
            dense_bps_scale=view.get('dense_bps_scale'))
    else:
        wrapper = NativeEntryDirectionResearch(backbone, bps_scale=view.get('bps_scale'), magnitude_weight_contract=view.get('magnitude_weight_contract'))
    wrapper.feature_view_identity = {'contract_sha256': view['contract_sha256'],
        'prefix_normalization': _copy(view['prefix_normalization']), 'research_objective_sha256': view['research_objective']['contract_sha256']}
    return wrapper
