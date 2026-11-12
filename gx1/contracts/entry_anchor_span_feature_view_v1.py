"""One exact additive TRAIN-input view; immutable old prefix/GC stay parents.

These metadata functions read no files and construct no model. Callers verify
file references and obtain separate execution authority. The existing native
factory owns any later construction after its exact metadata rebind.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json

from gx1.contracts import entry_anchor_span_normalization_v1 as extension
from gx1.contracts import entry_gc_mtf_history_v1 as history
from gx1.contracts import entry_model_native_input_normalization_v1 as norm
from gx1.contracts import entry_native_direction_prefix_units_v1 as units
from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
from gx1.contracts.market_state_time_query_model_v2 import factorized_time_query_model_contract

PACKET_SCHEMA = 'gx1_anchor_span_feature_prefix_packet_v1'
MARKER_KEY = 'anchor_span_input_extension'
ANCHOR_SPAN_SOURCE_REF_KEYS = frozenset({'anchor_span_inputs_source'})
OVERLAY_OWNER_PATH = '/home/andre2/src/GX1_CURRENT/gx1/contracts/entry_anchor_span_feature_inputs_v1.py'
PARENT_PACKET_SHA = 'bde697043f2609ce76905ca85d86bc87530bb002c25bb78d80cb9cd4b0031948'
PARENT_QUALIFICATION_SHA = '3f4309c26e8d802f7694f5aaaf8f8c301ab82e1c4bf272eb7540c8718e65dc1f'
PARENT_QUALIFICATION_REF = {
    'path': '/home/andre2/GX1_RUNS/MARKET_STATE_MATURE_FOLD1_NORMALIZATION_20261107_001/QUALIFICATION_FOLD_1.json',
    'sha256': '0fad1ac1476205e0485207da4b5e6be6b0de5585a8cb4170f87679c63580fb94',
}
PARENT_GC_LAYOUT_SHA = 'b67b9b94b78b4cf933c23e6738666911cd2c262ae3ab696448d42571e13bec6f'
PARENT_GC_NORMALIZATION_SHA = 'e3a0b4421e2ee5501a5750f102856652aa4361cf51f4e1c780f73d8f11b974ec'
BASE_CONTRACT_SHA = '58e9e9bb63b65f05ae2b95ee04c6338623eb4489b75c0626c87978d41323fead'
TA_CONTRACT_SHA = '6c882086902701d5c9503d0c77c72c8e291ed65c44d065171c2d27afa500c3f6'
UNITS_CONTRACT_SHA = 'accae7dfbc704c2f1af33d4fe71ffa7c270b675c5f1f13ce0d18d15381a93e1e'
PARENT_CONSTRUCTOR_REF = {
    'path': '/home/andre2/GX1_RUNS/NATIVE_M5_EARLY_MFE_FRESH_PREPARATION_20261022_187/CONSTRUCTOR_METADATA.json',
    'sha256': '8d0fc19b646b142dced37ad86dd0e43a71bb8efe299a9ba4fd2de08e324db2bf',
}
PARENT_CONSTRUCTOR_CONTENT_SHA = '3a3f59a768774fc1f81ff4bd6c54393b5ed62f1d6816e50cd1b6f042d1d3fe1b'
SCALARS = dict(dropout=.05, multi_tf_num_layers=2, multi_tf_scale=.5,
               specialist_num_layers=1, specialist_fusion_scale=.25, cross_family_fusion_scale=.25)
# Exact completed artifacts and parent-proof identities; populated from their
# retained JSON metadata, never from a checkpoint or market-value payload.
NORMALIZATION_EVIDENCE = {'physical_result': {'path': '/home/andre2/GX1_RUNS/MARKET_STATE_TA_TRENDLINE_ANCHOR_SPAN_INPUTS_20261112_001/RESULT.json',
                     'sha256': '8878d17a5bf114fcf9cd5eb0e74792ddc5df068b95b67951b594759d5b2b05bb'},
 'normalization_result': {'path': '/home/andre2/GX1_RUNS/MARKET_STATE_TA_TRENDLINE_ANCHOR_SPAN_NORMALIZATION_20261112_001/RESULT.json',
                          'sha256': '4476c63b5b70237adf6c67db7f125c6eefb95963cab9418a46ec2d3f23a77f9f'},
 'normalization_terminal': {'path': '/home/andre2/GX1_RUNS/MARKET_STATE_TA_TRENDLINE_ANCHOR_SPAN_NORMALIZATION_20261112_001/TERMINAL.json',
                            'sha256': '568bed0ad27f7a467f4ca15453314347baf9406c1c472f9719d6e56e625c9ede'},
 'expanded_base': {'path': '/home/andre2/GX1_RUNS/MARKET_STATE_TA_TRENDLINE_ANCHOR_SPAN_NORMALIZATION_20261112_001/EXTENDED_BASE_NORMALIZATION.json',
                   'sha256': '71e201b690101b7a0748fe8594fe35cae6d88288f74deee46897a54adfda69c1'},
 'ta_normalization': {'path': '/home/andre2/GX1_RUNS/MARKET_STATE_TA_TRENDLINE_ANCHOR_SPAN_NORMALIZATION_20261112_001/TA_INPUT_NORMALIZATION.json',
                      'sha256': '02eca0cceec636a39b57cb12d55aa07f458653e289d6c5f57b16811e61438693'},
 'units_profile': {'path': '/home/andre2/GX1_RUNS/MARKET_STATE_TA_TRENDLINE_ANCHOR_SPAN_NORMALIZATION_20261112_001/NEW_UNITS_PROFILE.json',
                   'sha256': '050f93ad7850b9b94872cb785fb7510ec59b8a39c006bc6f0b729a145204d358'}}
PARENT_PROOF_HASHES = {'prefix': 'b49ac583954f741f95fa33046db7af5136d1a43f2a62ceff1d3a5d55254c4ee4',
 'full_axes': '1b53b9e779c8d5458f7e1aa08f426075ee3fd68aae246972a8449224120931e9',
 'base_lineage': 'b798c42f7b500bc21a50770e4093b5e54e5627d4fc5593f0fa10f2d67fa99a39',
 'source_bindings': '854b085f6eada35a816ac0c703d859c052f8e283045400c49ad4533f05443f17'}
PARENT_OWNER_PROFILE_SHA = '6070f033f3046f30c01666b02226155eb032fd66bf1364621697e43cfebdbef3'
PACKET_CONTRACT_SHA = '7a6958c0bad17bccd3746c0df199b7a64248cdbb874520f7ccb886331584b6c0'

_MODE_DEFAULTS = dict(bps_scale=None, magnitude_weight_contract=None, dense_minutes=False,
    price_momentum=False, m1_ofi_normalization=None, m1_ofi_coarsened=False,
    five_minute_priority=False, dense_bps_scale=None, family_intervention=None,
    price_momentum_initialization='zero_residual', full_feature_profile=False,
    dynamic_state_supervision=False, market_state_five_endpoint=False,
    current_state_first_loss_weight=None, extended_forecast_minutes=None,
    first_flat_activity_weight=None, native_endpoint_loss=False, confirmed_run_age_input=False,
    zero_support_profile=None, october_observer_axes=None, market_state_time_query=False,
    market_state_time_query_v2=False, full_ta_feature_profile=False, dedicated_m5_side=False,
    dedicated_non_m5_side=False, dedicated_tf_activity=False)
_ACTIVITY_MODES = {**_MODE_DEFAULTS, 'full_feature_profile': True, 'full_ta_feature_profile': True,
    'market_state_time_query_v2': True, 'dedicated_m5_side': True, 'dedicated_tf_activity': True}


def _need(ok, reason):
    if not bool(ok):
        raise RuntimeError('ANCHOR_SPAN_FEATURE_VIEW_' + reason)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _sha(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _copy(value):
    return json.loads(_canonical(value))


def _views():
    # Lazy import keeps the existing factory's dispatch free of import cycles.
    from gx1.contracts import entry_native_direction_feature_view_v1
    return entry_native_direction_feature_view_v1


def _exact_seal(value, expected, reason):
    _need(isinstance(value, Mapping) and value.get('contract_sha256') == expected
          and _sha({k: v for k, v in value.items() if k != 'contract_sha256'}) == expected, reason)


def _parent(parent):
    _exact_seal(parent, PARENT_PACKET_SHA, 'PARENT_PACKET')
    old_units = units.mature_fold1_prefix_units_profile()
    report = parent['prefix_qualification']
    _exact_seal(report, PARENT_QUALIFICATION_SHA, 'PARENT_QUALIFICATION')
    _need(parent['unobserved_units_profile'] == report['unobserved_units_profile'] == old_units
          and parent['native_launch_authorized'] is parent['genuine_model_fit_authorized'] is False
          and report['fit_allowed'] is True and report['blockers'] == []
          and report['statistics_fits'] == 0, 'PARENT_SCOPE')
    units.require_prefix_units_fit_identity(report['proof']['prefix'], old_units)
    old = parent['base_normalization']
    norm.require_input_normalization_contract(old,
        expected_field_names={k: s['field_names'] for k, s in old['surfaces'].items()},
        expected_ctx_cat_names=old['ctx_cat']['field_names'],
        expected_entry_training_phase=ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE,
        unobserved_units_profile=old_units)
    # This GC object retains its old profile identity. It is never rebound to
    # expanded signal indices and is not supplied to the TA-only backbone.
    gc = history.require_gc_mtf_history(parent['gc_mtf_history'], unobserved_units_profile=old_units)
    _need(gc['layout']['contract_sha256'] == PARENT_GC_LAYOUT_SHA
          and gc['normalization']['contract_sha256'] == PARENT_GC_NORMALIZATION_SHA,
          'PARENT_GC')
    return report['proof']


def build_anchor_span_prefix_packet(parent_prefix_normalization, base_normalization, *,
                                    unobserved_units_profile):
    """Seal the exact new base together with the unchanged qualified parent."""
    _parent(parent_prefix_normalization)
    profile = units.require_anchor_span_fold1_prefix_units_profile(unobserved_units_profile)
    _need(profile['contract_sha256'] == UNITS_CONTRACT_SHA, 'UNITS_IDENTITY')
    fields, _, _ = _views()._canonical()
    base = norm.require_input_normalization_contract(base_normalization,
        expected_field_names={s: fields['mtf' if s.startswith('mtf_') else s]
                              for s in norm.EXPECTED_SURFACES},
        expected_ctx_cat_names=fields['ctx_cat'],
        expected_entry_training_phase=ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE,
        unobserved_units_profile=profile)
    _need(base['contract_sha256'] == BASE_CONTRACT_SHA
          and _canonical(base['anchor_span_extension']['source_contract'])
          == _canonical(parent_prefix_normalization['base_normalization']), 'BASE_PARENT_JOIN')
    ta = norm.project_ta_input_normalization_contract(base, unobserved_units_profile=profile)
    _need(ta['contract_sha256'] == TA_CONTRACT_SHA, 'TA_IDENTITY')
    value = dict(schema_version=PACKET_SCHEMA,
        parent_prefix_reference=_copy(extension.PARENT_ARTIFACT),
        parent_prefix_normalization=_copy(parent_prefix_normalization),
        base_normalization=_copy(base), gc_mtf_history=_copy(parent_prefix_normalization['gc_mtf_history']),
        unobserved_units_profile=_copy(profile), normalization_evidence=_copy(NORMALIZATION_EVIDENCE),
        native_launch_authorized=False, genuine_model_fit_authorized=False,
        full_feature_view_gather_qualified=False, actual_model_baseline_complete=False)
    _need(_sha(value) == PACKET_CONTRACT_SHA, 'PACKET_IDENTITY')
    return {**value, 'contract_sha256': PACKET_CONTRACT_SHA}


def require_anchor_span_prefix_packet(packet):
    _exact_seal(packet, PACKET_CONTRACT_SHA, 'PACKET_SEAL')
    expected = build_anchor_span_prefix_packet(packet['parent_prefix_normalization'],
        packet['base_normalization'], unobserved_units_profile=packet['unobserved_units_profile'])
    _need(_canonical(packet) == _canonical(expected), 'PACKET_IDENTITY')
    return expected


def _constructor():
    fields, routes, widths = _views()._canonical()
    mtf = dict(feature_names=list(fields['mtf']), multi_tf_num_layers=SCALARS['multi_tf_num_layers'],
               multi_tf_scale=SCALARS['multi_tf_scale'])
    for tf, length in history.WINDOW_ROWS.items():
        mtf[tf.lower() + '_seq_dim'] = widths['mtf']
        mtf[tf.lower() + '_seq_len'] = length
    context = {key: _copy(routes['specialist_' + key])
               for key in ('ctx_cont_indices', 'ctx_cont_nominal_indices', 'ctx_cat_indices')}
    context['temporal_alias_policy'] = {
        'signal_indices': _copy(routes['temporal_alias_signal_indices']),
        'ctx_cont_indices': _copy(routes['temporal_alias_ctx_cont_indices'])}
    return dict(schema_version='gx1_anchor_span_constructor_metadata_v1',
        parent_constructor=dict(binding=_copy(PARENT_CONSTRUCTOR_REF), content_sha256=PARENT_CONSTRUCTOR_CONTENT_SHA),
        physical_layout='current_full_raw_before_ta_projection', evidence_scope='metadata_only_no_model_or_fit',
        seq_input_dim=widths['signal'], snap_input_dim=widths['signal'], seq_len=_views().MODEL_NATIVE_SEQ_LEN,
        ctx_cont_dim=widths['ctx_cont'], ctx_cat_dim=widths['ctx_cat'], dropout=SCALARS['dropout'], multi_tf=mtf,
        specialist_fusion=dict(num_layers=SCALARS['specialist_num_layers'],
            fusion_scale=SCALARS['specialist_fusion_scale'],
            cross_family_fusion_scale=SCALARS['cross_family_fusion_scale'],
            input_indices=_copy(routes['specialist_input_indices']), context_routing=context))


def build_anchor_span_constructor_metadata(parent_constructor_metadata):
    """Reuse six unchanged scalars; emit fresh current physical routing metadata."""
    _need(isinstance(parent_constructor_metadata, Mapping)
          and _sha(parent_constructor_metadata) == PARENT_CONSTRUCTOR_CONTENT_SHA, 'PARENT_CONSTRUCTOR')
    result = _constructor()
    _need(_views()._configuration(result) == SCALARS, 'CONSTRUCTOR_SCALARS')
    return result


def _marker(overlay):
    return dict(schema_version='gx1_anchor_span_input_view_extension_v1',
        parent_prefix_reference=_copy(extension.PARENT_ARTIFACT), parent_prefix_contract_sha256=PARENT_PACKET_SHA,
        parent_qualification=_copy(PARENT_QUALIFICATION_REF), packet_contract_sha256=PACKET_CONTRACT_SHA,
        normalization_evidence=_copy(NORMALIZATION_EVIDENCE),
        parent_constructor=dict(binding=_copy(PARENT_CONSTRUCTOR_REF), content_sha256=PARENT_CONSTRUCTOR_CONTENT_SHA),
        overlay_owner=_copy(overlay),
        storage='immutable_parent_arrays_plus_qualified_span_sidecars', execution_authorized=False)


def require_anchor_span_view_provenance(view, profile):
    """Additional exact metadata gates; generic geometry/objective gates remain."""
    profile = units.require_anchor_span_fold1_prefix_units_profile(profile)
    _need(profile['contract_sha256'] == UNITS_CONTRACT_SHA, 'VIEW_UNITS')
    _need(all(view.get(k) is True for k in ('full_feature_profile', 'full_ta_feature_profile',
          'market_state_time_query_v2', 'dedicated_m5_side', 'dedicated_tf_activity'))
          and 'dedicated_non_m5_side' not in view, 'ACTIVITY_ONLY_VIEW')
    overlay = _views()._ref(view.get('source_bindings', {}).get('anchor_span_inputs_source'))
    _need(overlay['path'] == OVERLAY_OWNER_PATH, 'OVERLAY_OWNER_PATH')
    _need(_canonical(view.get(MARKER_KEY)) == _canonical(_marker(overlay)), 'VIEW_PROVENANCE')
    prefix = view['prefix_normalization']
    _need(isinstance(prefix, Mapping) and 'binding' in prefix, 'VIEW_NORMALIZATION')
    expected = dict(binding=_views()._ref(prefix['binding']), contract_sha256=PACKET_CONTRACT_SHA,
        base_sha256=BASE_CONTRACT_SHA, gc_layout_sha256=PARENT_GC_LAYOUT_SHA,
        gc_normalization_sha256=PARENT_GC_NORMALIZATION_SHA, qualification_sha256=PARENT_QUALIFICATION_SHA,
        ta_base_normalization_sha256=TA_CONTRACT_SHA, source_owner_profile_sha256=PARENT_OWNER_PROFILE_SHA,
        unobserved_units_profile_sha256=UNITS_CONTRACT_SHA)
    _need(_canonical(prefix) == _canonical(expected), 'VIEW_NORMALIZATION')
    _need(all(_sha(view[key]) == PARENT_PROOF_HASHES[key]
              for key in ('prefix', 'full_axes', 'base_lineage'))
          and view['prefix_source_bindings_sha256'] == PARENT_PROOF_HASHES['source_bindings'],
          'VIEW_PARENT_PROOF')
    _need(view['constructor_scalar_config'] == SCALARS
          and view['constructor_metadata']['content_sha256'] == _sha(_constructor()), 'VIEW_CONSTRUCTOR')
    _views()._ref(view['constructor_metadata']['binding'])
    return _copy(view[MARKER_KEY])


def bind_anchor_span_feature_view(*, prefix_normalization, prefix_normalization_ref,
    constructor_metadata, constructor_metadata_ref, source_bindings, max_batch_rows,
    expected=None, unobserved_units_profile=None, **modes):
    """Accept the existing binder kwargs, permitting only activity-only defaults."""
    _need(set(modes) <= set(_MODE_DEFAULTS)
          and _canonical({**_MODE_DEFAULTS, **modes}) == _canonical(_ACTIVITY_MODES), 'ACTIVITY_ONLY_ARGUMENTS')
    packet = require_anchor_span_prefix_packet(prefix_normalization)
    profile = units.require_anchor_span_fold1_prefix_units_profile(unobserved_units_profile)
    _need(profile == packet['unobserved_units_profile'], 'EXPLICIT_UNITS')
    _need(_canonical(constructor_metadata) == _canonical(_constructor()), 'CONSTRUCTOR_IDENTITY')
    views = _views()
    _need(isinstance(source_bindings, Mapping)
          and set(source_bindings) == views.SOURCE_REF_KEYS | views.TIME_QUERY_V2_SOURCE_REF_KEYS
                                      | ANCHOR_SPAN_SOURCE_REF_KEYS, 'SOURCE_KEYS')
    refs = {k: views._ref(v) for k, v in source_bindings.items()}
    proof = packet['parent_prefix_normalization']['prefix_qualification']['proof']
    provenance = proof['source_bindings']
    immutable = dict(input_qualification=PARENT_QUALIFICATION_REF,
        raw_input_provider_plan=provenance['provider_plan'], raw_gc_layout=provenance['boolean_raw_gc_layout'],
        raw_gc_transition=provenance['boolean_raw_domain_transition'], time_query_mature_prefix_plan=provenance['current_plan'])
    _need(all(refs[k] == v for k, v in immutable.items()), 'PARENT_SOURCE_JOINS')
    for key, path in (
        ('mtf_window_source', 'gx1/features/htf_features.py'),
        ('gc_gather_source', 'gx1/contracts/entry_gc_mtf_history_v1.py'),
        ('owner_profile_source', 'gx1/features/entry_specialist_feature_groups_v1.py'),
    ):
        _need(refs[key]['path'] == str(views.time_query_targets.ROOT / path)
              and (key == 'owner_profile_source'
                   or refs[key]['sha256'] == provenance['current_source_owners'][path]),
              'PARENT_SOURCE_OWNERS')
    _need(refs['anchor_span_inputs_source']['path'] == OVERLAY_OWNER_PATH, 'OVERLAY_OWNER_PATH')
    _need(type(max_batch_rows) is int and max_batch_rows > 0, 'MAX_BATCH_ROWS')
    fields, routes, widths = views._canonical(ta_only=True)
    compiled = views.owners.compile_full_ta_entry_execution_profile()
    value = dict(schema_version=views.SCHEMA_VERSION,
        prefix_normalization=dict(binding=views._ref(prefix_normalization_ref), contract_sha256=PACKET_CONTRACT_SHA,
            base_sha256=BASE_CONTRACT_SHA, gc_layout_sha256=PARENT_GC_LAYOUT_SHA,
            gc_normalization_sha256=PARENT_GC_NORMALIZATION_SHA, qualification_sha256=PARENT_QUALIFICATION_SHA,
            ta_base_normalization_sha256=TA_CONTRACT_SHA, source_owner_profile_sha256=PARENT_OWNER_PROFILE_SHA,
            unobserved_units_profile_sha256=UNITS_CONTRACT_SHA),
        prefix=_copy(proof['prefix']), full_axes=_copy(proof['full_axes']), base_lineage=_copy(proof['base_lineage']),
        prefix_source_bindings_sha256=_sha(provenance), physical_widths=dict(widths),
        physical_field_names_sha256={k: views._sha(v) for k, v in fields.items()},
        sequence_bars=views.MODEL_NATIVE_SEQ_LEN, window_rows=dict(history.WINDOW_ROWS),
        canonical_routing_sha256=views._sha(routes), constructor_scalar_config=dict(SCALARS),
        constructor_metadata=dict(binding=views._ref(constructor_metadata_ref), content_sha256=_sha(constructor_metadata)),
        entry_execution_profile={k: compiled[k] for k in ('name','contract_sha256')},
        research_objective=factorized_time_query_model_contract(ta_only=True, dedicated_m5_side=True,
            dedicated_non_m5_side=False, dedicated_tf_activity=True), source_bindings=refs,
        max_batch_rows=max_batch_rows, evidence_scope=views.EVIDENCE_SCOPE,
        full_feature_profile=True, full_ta_feature_profile=True, market_state_time_query_v2=True,
        dedicated_m5_side=True, dedicated_tf_activity=True, unobserved_units_profile=_copy(profile))
    value[MARKER_KEY] = _marker(refs['anchor_span_inputs_source'])
    value['contract_sha256'] = _sha(value)
    views._require_view(value, unobserved_units_profile=profile)
    if expected is not None:
        _need(_canonical(expected) == _canonical(value), 'EXPECTED_VIEW')
    return value
