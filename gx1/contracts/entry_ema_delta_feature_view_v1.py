"""Exact metadata successor for the restored EMA spread delta on M5 and MTF.

This is a new 318/256 input identity. It preserves the sealed Fold1 anchor-span
packet as a parent and binds the one completed FIT-only normalization and six
causal physical sidecars. It does not authorize execution or read market values.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path

from gx1.contracts import entry_anchor_span_feature_view_v1 as anchor
from gx1.contracts import entry_anchor_span_normalization_v1 as extension
from gx1.contracts import entry_model_native_input_normalization_v1 as norm
from gx1.contracts import entry_native_direction_prefix_units_v1 as units
from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
from gx1.contracts.market_state_time_query_model_v2 import factorized_time_query_model_contract

PACKET_SCHEMA = 'gx1_ema_delta_feature_prefix_packet_v1'
MARKER_KEY = 'ema_delta_input_extension'
ROOT = Path('/home/andre2/GX1_RUNS/TA_M5_EMA_DELTA_FEATURE_VIEW_20261118_001')
PARENT_PACKET_REF = {
    'path': '/home/andre2/GX1_RUNS/MARKET_STATE_TA_ANCHOR_SPAN_SHORT_LEARNING_20261112_001/PREFIX_PACKET.json',
    'sha256': '900fd2ca1b571b68bbea05e94cda739d718fa6167aaaa9290c8980f60fbac9fb',
}
NORMALIZATION_ROOT = '/home/andre2/GX1_RUNS/TA_M5_EMA_DELTA_FIT_NORMALIZATION_20261118_001'
NORMALIZATION_EVIDENCE = {
    'result': {'path': NORMALIZATION_ROOT + '/RESULT.json',
               'sha256': 'dfc3f5b9e021a7e122ae87a6ca04136a7b960f86ba4930df46354150ba25973a'},
    'terminal': {'path': NORMALIZATION_ROOT + '/TERMINAL.json',
                 'sha256': '47eeacd82df0a18f5da5693382131aa62c15561e67c542572ad59f7ec2d260d8'},
    'base': {'path': NORMALIZATION_ROOT + '/EXTENDED_BASE_NORMALIZATION.json',
             'sha256': 'e3a643872ad69ea7999dabcb2cf200189948f933729b0c72fb3c87dbe9d2dead'},
    'ta': {'path': NORMALIZATION_ROOT + '/TA_INPUT_NORMALIZATION.json',
           'sha256': 'cd1b972dd08d060b89b93b92d9c3770fdaae99418ba1f3f9bb58f8986d84bd2f'},
    'units': {'path': NORMALIZATION_ROOT + '/NEW_UNITS_PROFILE.json',
              'sha256': 'd062de090dbf974d093a2d7bec909904b346f66fbd25212ab65a69d309232ebb'},
}
MATERIALIZER_ROOT = '/home/andre2/GX1_RUNS/TA_M5_EMA_DELTA_SIDECARS_20261117_002'
MATERIALIZED_RESULT_REF = {
    'path': MATERIALIZER_ROOT + '/OUTPUT/RESULT.json',
    'sha256': 'aea480092efea5b0f3f97dfe2be34dda97d9da377434c2a78d8712174a1e4c20',
}
MATERIALIZED_TERMINAL_REF = {
    'path': MATERIALIZER_ROOT + '/TERMINAL.json',
    'sha256': '2b697d3c86a61a134de2e41dee7ff7f0f21b70b85e9a37da257a8c3b44f0dbb5',
}
SIDECAR_SHA = {
    'local': 'f23c60aad63a5e1217e3c02b671dae3826b701acca5acba1f7f0ecec1677da12',
    'M5': 'f0992b9580cb3fb10010f9590468c24fae4c359efb878f6914f14a97d775036d',
    'M15': 'b46b75986d02fc1078107e4f7ec25a4020fed1388cdaad9ed67378c92b73861e',
    'H1': 'ff933970fbb795ef16b9646bc1c7c151b1fb407e47b0bc2cb1e2e3a24ce26c27',
    'H4': 'ea2c4052e9b472826c8771d51dee3386adf4735abb97cf88a56b03f615b0814b',
    'D1': 'b01ca297f8ccfb1913bea6e1e6d03bb0110542137c54a49b721b65958cbe1681',
}
SIDECAR_ROWS = {'local': 47614, 'M5': 47534, 'M15': 15903, 'H1': 4058, 'H4': 1134, 'D1': 425}
SIDECAR_REFS = {lane: {'path': MATERIALIZER_ROOT + '/OUTPUT/' + lane + '_EMA_DELTA.npz',
                       'sha256': digest} for lane, digest in SIDECAR_SHA.items()}
BASE_CONTRACT_SHA = 'bcb51f7a1345f31b3bd1bf4227014c78f3563e3bbdf9d4f31750882589a5b737'
TA_CONTRACT_SHA = '78c959430bf9a13998d0fd52af41d37330a3a4bddac94aa61c58d4237e8f6c8f'
UNITS_CONTRACT_SHA = '593febb9a3a006790242e92a9b141bde28c2aafa310d64afae5cf1ea917356a0'
PACKET_CONTRACT_SHA = 'a864cb9501019e99fdaa88dc13b068c55048c1cff5ae6d6a3e959ebeb1cc701b'
PACKET_REF = {
    'path': str(ROOT / 'PREFIX_PACKET.json'),
    'sha256': '1d572ff79c14483dea947870948a03d1ccc2dd83f157b76b24483bc30c82fc01',
}
CONSTRUCTOR_REF = {
    'path': str(ROOT / 'CONSTRUCTOR_METADATA.json'),
    'sha256': '5541ef5db0606e12f767ebbe590a55594cbdb71a091c459ddc43e61e114ece6d',
}
PARENT_SOURCE_REFS = {
    'input_qualification': anchor.PARENT_QUALIFICATION_REF,
    'raw_input_provider_plan': {
        'path': '/home/andre2/GX1_RUNS/NATIVE_M5_DIRECTION_NATIVE_PREFIX_INPUT_BINDING_REPAIR_20261024_264/PLAN_SUPPORT.json',
        'sha256': '96ff82533042ab069292505b3a44315c13223a9b9d9e9ee669cd757e3f84c086'},
    'raw_gc_layout': {
        'path': '/home/andre2/GX1_RUNS/NATIVE_M5_DIRECTION_GC_BOOLEAN_PHYSICAL_PREFIX_NORMALIZATION_20261024_267/CURRENT_GC_LAYOUT.json',
        'sha256': 'c0cdc60447ceb303ebdcdc790716789585fb32c0c30918f5d58d030ad6baffde'},
    'raw_gc_transition': {
        'path': '/home/andre2/GX1_RUNS/NATIVE_M5_DIRECTION_GC_BOOLEAN_PHYSICAL_PREFIX_NORMALIZATION_20261024_267/RAW_DOMAIN_TRANSITION.json',
        'sha256': '98bc67279431199725ff64630dcce0687e6b0908bef0595724b8e861a7306a00'},
    'time_query_mature_prefix_plan': {
        'path': '/home/andre2/src/GX1_CURRENT/configs/research/MARKET_STATE_MATURE_FOLD1_NORMALIZATION_20261107_001.json',
        'sha256': '2e74f6369d3b3bf685d9cdaa606db84ed0e5cadf6f4afe1fdb59e7550ef68616'},
}


def _need(ok, reason):
    if not bool(ok):
        raise RuntimeError('EMA_DELTA_FEATURE_VIEW_' + reason)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _sha(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _copy(value):
    return json.loads(_canonical(value))


def artifact_bytes(value):
    """The immutable metadata publisher uses this exact pretty JSON encoding."""
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def artifact_ref(name, value):
    _need(name in ('PREFIX_PACKET.json', 'CONSTRUCTOR_METADATA.json', 'FEATURE_VIEW.json'),
          'OUTPUT_NAME')
    return {'path': str(ROOT / name), 'sha256': hashlib.sha256(artifact_bytes(value)).hexdigest()}


def _fixed(value, binding, reason, *, ordered=False):
    _need(isinstance(value, Mapping) and isinstance(binding, Mapping), reason)
    raw = ((json.dumps(value, indent=2, allow_nan=False) + '\n').encode() if ordered
           else artifact_bytes(value))
    _need(hashlib.sha256(raw).hexdigest() == binding['sha256'], reason)
    return value


def _views():
    from gx1.contracts import entry_native_direction_feature_view_v1
    return entry_native_direction_feature_view_v1


def _normalization_inputs(parent_packet, result, terminal, materialized, materialized_terminal,
                          base, ta, profile):
    _fixed(parent_packet, PARENT_PACKET_REF, 'PARENT_PACKET_FILE')
    anchor._exact_seal(parent_packet, anchor.PACKET_CONTRACT_SHA, 'PARENT_PACKET_SEAL')
    _need(parent_packet['schema_version'] == anchor.PACKET_SCHEMA
          and parent_packet['normalization_evidence'] == anchor.NORMALIZATION_EVIDENCE
          and parent_packet['unobserved_units_profile']['contract_sha256'] == anchor.UNITS_CONTRACT_SHA
          and parent_packet['base_normalization']['contract_sha256'] == anchor.BASE_CONTRACT_SHA,
          'PARENT_PACKET_IDENTITY')
    anchor._parent(parent_packet['parent_prefix_normalization'])
    _fixed(result, NORMALIZATION_EVIDENCE['result'], 'FIT_RESULT_FILE')
    _fixed(terminal, NORMALIZATION_EVIDENCE['terminal'], 'FIT_TERMINAL_FILE')
    _fixed(materialized, MATERIALIZED_RESULT_REF, 'MATERIALIZED_RESULT_FILE', ordered=True)
    _fixed(materialized_terminal, MATERIALIZED_TERMINAL_REF, 'MATERIALIZED_TERMINAL_FILE', ordered=True)
    _fixed(base, NORMALIZATION_EVIDENCE['base'], 'BASE_FILE')
    _fixed(ta, NORMALIZATION_EVIDENCE['ta'], 'TA_FILE')
    _fixed(profile, NORMALIZATION_EVIDENCE['units'], 'UNITS_FILE')
    _need(result['status'] == terminal['status'] == 'COMPLETE'
          and terminal['exit_code'] == 0 and terminal['result'] == NORMALIZATION_EVIDENCE['result']
          and result['artifacts']['EXTENDED_BASE_NORMALIZATION'] == NORMALIZATION_EVIDENCE['base']
          and result['artifacts']['TA_INPUT_NORMALIZATION'] == NORMALIZATION_EVIDENCE['ta']
          and result['artifacts']['NEW_UNITS_PROFILE'] == NORMALIZATION_EVIDENCE['units']
          and result['inputs']['materialized_result'] == MATERIALIZED_RESULT_REF
          and result['inputs']['materialized_terminal'] == MATERIALIZED_TERMINAL_REF
          and result['source_unchanged'] is True and result['test_data_used'] is False
          and result['surface_fit_calls'] == 6 and result['old_field_fit_calls'] == 0
          and result['model_forwards'] == 0
          and materialized['status'] == materialized_terminal['status'] == 'COMPLETE'
          and materialized_terminal['exit_code'] == 0
          and materialized_terminal['result'] == MATERIALIZED_RESULT_REF
          and materialized_terminal['source_unchanged'] is True
          and materialized['fit_rows'] == 29574 and materialized['later_rows'] == 17489
          and materialized['test_data_used'] is False
          and materialized['model_forwards'] == materialized['optimizer_steps'] == 0
          and set(materialized['surfaces']) == set(SIDECAR_REFS)
          and all(materialized['surfaces'][lane]['artifact'] == SIDECAR_REFS[lane]
                  and materialized['surfaces'][lane]['rows']
                  == materialized['surfaces'][lane]['old_slope_parity_rows']
                  == SIDECAR_ROWS[lane] for lane in SIDECAR_REFS), 'FIT_PHYSICAL_RECEIPTS')
    new_units = units.require_ema_delta_fold1_prefix_units_profile(profile)
    _need(new_units['contract_sha256'] == UNITS_CONTRACT_SHA
          and result['units_contract_sha256'] == UNITS_CONTRACT_SHA, 'UNITS_IDENTITY')
    fields = _views().owners.compile_full_ta_entry_execution_profile()['source_field_names']
    names = {surface: list(fields['mtf' if surface.startswith('mtf_') else surface])
             for surface in norm.EXPECTED_SURFACES}
    norm.require_input_normalization_contract(base, expected_field_names=names,
        expected_ctx_cat_names=fields['ctx_cat'],
        expected_entry_training_phase=ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE,
        unobserved_units_profile=new_units)
    _need(base['contract_sha256'] == result['base_contract_sha256'] == BASE_CONTRACT_SHA
          and base['anchor_span_extension']['schema_version'] == extension.EMA_DELTA_SCHEMA
          and base['anchor_span_extension']['source_contract'] == parent_packet['base_normalization']
          and base['anchor_span_extension']['physical_input_result'] == MATERIALIZED_RESULT_REF
          and base['anchor_span_extension']['preservation']['all_old_stat_vectors_exact'] is True
          and base['anchor_span_extension']['old_statistics_refitted'] is False,
          'BASE_PARENT_JOIN')
    projected = norm.project_ta_input_normalization_contract(base, unobserved_units_profile=new_units)
    _need(projected == ta and ta['contract_sha256'] == result['ta_contract_sha256']
          == TA_CONTRACT_SHA, 'TA_IDENTITY')


def build_ema_delta_prefix_packet(*, parent_packet, normalization_result,
                                  normalization_terminal, materialized_result,
                                  materialized_terminal, base_normalization,
                                  ta_normalization, unobserved_units_profile):
    _normalization_inputs(parent_packet, normalization_result, normalization_terminal,
        materialized_result, materialized_terminal, base_normalization, ta_normalization,
        unobserved_units_profile)
    value = dict(schema_version=PACKET_SCHEMA, parent_anchor_packet_reference=_copy(PARENT_PACKET_REF),
        parent_anchor_packet=_copy(parent_packet), base_normalization=_copy(base_normalization),
        gc_mtf_history=_copy(parent_packet['gc_mtf_history']),
        unobserved_units_profile=_copy(unobserved_units_profile),
        normalization_evidence=_copy(NORMALIZATION_EVIDENCE),
        materialized_result=_copy(MATERIALIZED_RESULT_REF),
        materialized_terminal=_copy(MATERIALIZED_TERMINAL_REF),
        sidecars=_copy(SIDECAR_REFS),
        native_launch_authorized=False, genuine_model_fit_authorized=False,
        full_feature_view_gather_qualified=False, actual_model_baseline_complete=False)
    _need(_sha(value) == PACKET_CONTRACT_SHA, 'PACKET_IDENTITY')
    return {**value, 'contract_sha256': PACKET_CONTRACT_SHA}


def require_ema_delta_prefix_packet(packet):
    _need(isinstance(packet, Mapping) and packet.get('schema_version') == PACKET_SCHEMA
          and packet.get('contract_sha256') == PACKET_CONTRACT_SHA
          and packet.get('contract_sha256') == _sha({
              k: v for k, v in packet.items() if k != 'contract_sha256'}), 'PACKET_SEAL')
    _fixed(packet, PACKET_REF, 'PACKET_FILE')
    _need(packet['parent_anchor_packet_reference'] == PARENT_PACKET_REF
          and packet['normalization_evidence'] == NORMALIZATION_EVIDENCE
          and packet['materialized_result'] == MATERIALIZED_RESULT_REF
          and packet['materialized_terminal'] == MATERIALIZED_TERMINAL_REF
          and packet['sidecars'] == SIDECAR_REFS
          and packet['base_normalization']['contract_sha256'] == BASE_CONTRACT_SHA
          and packet['unobserved_units_profile']['contract_sha256'] == UNITS_CONTRACT_SHA
          and packet['native_launch_authorized'] is packet['genuine_model_fit_authorized']
          is packet['full_feature_view_gather_qualified']
          is packet['actual_model_baseline_complete'] is False, 'PACKET_SCOPE')
    parent = packet['parent_anchor_packet']
    _fixed(parent, PARENT_PACKET_REF, 'PARENT_PACKET_FILE')
    anchor._exact_seal(parent, anchor.PACKET_CONTRACT_SHA, 'PARENT_PACKET_SEAL')
    _need(packet['gc_mtf_history'] == parent['gc_mtf_history']
          and packet['base_normalization']['anchor_span_extension']['source_contract']
          == parent['base_normalization']
          and packet['base_normalization']['anchor_span_extension']['physical_input_result']
          == MATERIALIZED_RESULT_REF, 'PACKET_PARENT_JOIN')
    return _copy(packet)


def _normalization_identity(packet, packet_ref):
    return dict(binding=_views()._ref(packet_ref), contract_sha256=packet['contract_sha256'],
        base_sha256=BASE_CONTRACT_SHA, gc_layout_sha256=anchor.PARENT_GC_LAYOUT_SHA,
        gc_normalization_sha256=anchor.PARENT_GC_NORMALIZATION_SHA,
        qualification_sha256=anchor.PARENT_QUALIFICATION_SHA,
        ta_base_normalization_sha256=TA_CONTRACT_SHA,
        source_owner_profile_sha256=anchor.PARENT_OWNER_PROFILE_SHA,
        unobserved_units_profile_sha256=UNITS_CONTRACT_SHA)


def _marker():
    # The physical overlay owns this exact eight-key protocol. Other immutable
    # provenance remains in the sealed packet, prefix identity and source refs.
    return dict(schema_version='gx1_ema_delta_input_extension_v1',
        physical_input_result=_copy(MATERIALIZED_RESULT_REF),
        sidecars=_copy(SIDECAR_REFS), fit_rows=29574,
        score_parent_start=29849, score_rows=17489,
        score_end_exclusive_ns=1735603200000000000,
        execution_authorized=False)


def require_ema_delta_view_provenance(view, profile):
    _need(isinstance(view, Mapping), 'VIEW')
    prefix = view.get('prefix_normalization')
    _need(isinstance(prefix, Mapping), 'VIEW_NORMALIZATION')
    _need(profile is not None and profile['contract_sha256'] == UNITS_CONTRACT_SHA
          and view.get('full_feature_profile') is True
          and view.get('full_ta_feature_profile') is True
          and all(view.get(k) is True for k in ('market_state_time_query_v2',
              'dedicated_m5_side', 'dedicated_non_m5_side', 'dedicated_tf_activity',
              'direct_tf_hidden')), 'VIEW_MODE')
    sources = view.get('source_bindings', {})
    _need(sources.get('anchor_span_inputs_source') == sources.get('ema_delta_inputs_source')
          and sources.get('ema_delta_inputs_source', {}).get('path')
          == anchor.OVERLAY_OWNER_PATH, 'OVERLAY_OWNER')
    _need(view.get('anchor_span_input_extension')
          == anchor._marker(sources['anchor_span_inputs_source'], anchor.FOLD1_BINDING),
          'PARENT_SPAN_MARKER')
    expected_marker = _marker()
    _need(view.get(MARKER_KEY) == expected_marker, 'EMA_MARKER')
    _need(prefix['contract_sha256'] == PACKET_CONTRACT_SHA
          and prefix == _normalization_identity(
              {'contract_sha256': prefix['contract_sha256']}, prefix['binding'])
          and prefix['binding'] == PACKET_REF, 'VIEW_NORMALIZATION')
    _need(all(_sha(view[key]) == anchor.PARENT_PROOF_HASHES[key]
              for key in ('prefix', 'full_axes', 'base_lineage'))
          and view['prefix_source_bindings_sha256']
              == anchor.PARENT_PROOF_HASHES['source_bindings'], 'PARENT_PROOF')
    _need(view['constructor_scalar_config'] == anchor.SCALARS
          and view['constructor_metadata']['content_sha256'] == _sha(anchor._constructor())
          and view['constructor_metadata']['binding'] == CONSTRUCTOR_REF, 'CONSTRUCTOR')
    _need(all(sources.get(key) == ref for key, ref in PARENT_SOURCE_REFS.items()),
          'PARENT_SOURCES')
    return _copy(view[MARKER_KEY])


def bind_ema_delta_feature_view(*, prefix_normalization, prefix_normalization_ref,
                                constructor_metadata, constructor_metadata_ref,
                                source_bindings, max_batch_rows, expected=None,
                                unobserved_units_profile=None, **modes):
    expected_modes = {**anchor._ACTIVITY_MODES,
        'dedicated_non_m5_side': True, 'direct_tf_hidden': True}
    _need(set(modes) <= set(anchor._MODE_DEFAULTS)
          and {**anchor._MODE_DEFAULTS, **modes} == expected_modes, 'VIEW_MODES')
    packet = require_ema_delta_prefix_packet(prefix_normalization)
    _need(unobserved_units_profile == packet['unobserved_units_profile']
          and constructor_metadata == anchor._constructor()
          and type(max_batch_rows) is int and max_batch_rows == 16, 'VIEW_INPUTS')
    parent = packet['parent_anchor_packet']
    proof = parent['parent_prefix_normalization']['prefix_qualification']['proof']
    provenance = proof['source_bindings']
    refs = {key: _views()._ref(item) for key, item in source_bindings.items()}
    immutable = dict(input_qualification=anchor.PARENT_QUALIFICATION_REF,
        raw_input_provider_plan=provenance['provider_plan'],
        raw_gc_layout=provenance['boolean_raw_gc_layout'],
        raw_gc_transition=provenance['boolean_raw_domain_transition'],
        time_query_mature_prefix_plan=provenance['current_plan'])
    _need(all(refs.get(k) == v for k, v in immutable.items())
          and refs['anchor_span_inputs_source'] == refs['ema_delta_inputs_source']
          and refs['anchor_span_inputs_source']['path'] == anchor.OVERLAY_OWNER_PATH,
          'SOURCE_BINDINGS')
    fields, routes, widths = _views()._canonical(ta_only=True)
    _need(widths['signal'] == 318 and widths['mtf'] == 256, 'CURRENT_WIDTHS')
    compiled = _views().owners.compile_full_ta_entry_execution_profile()
    value = dict(schema_version=_views().SCHEMA_VERSION,
        prefix_normalization=_normalization_identity(packet, prefix_normalization_ref),
        prefix=_copy(proof['prefix']), full_axes=_copy(proof['full_axes']),
        base_lineage=_copy(proof['base_lineage']),
        prefix_source_bindings_sha256=_sha(provenance), physical_widths=dict(widths),
        physical_field_names_sha256={k: _views()._sha(v) for k, v in fields.items()},
        sequence_bars=_views().MODEL_NATIVE_SEQ_LEN,
        window_rows=dict(_views().history.WINDOW_ROWS),
        canonical_routing_sha256=_views()._sha(routes),
        constructor_scalar_config=dict(anchor.SCALARS),
        constructor_metadata=dict(binding=_views()._ref(constructor_metadata_ref),
                                  content_sha256=_sha(constructor_metadata)),
        entry_execution_profile={k: compiled[k] for k in ('name', 'contract_sha256')},
        research_objective=factorized_time_query_model_contract(ta_only=True,
            dedicated_m5_side=True, dedicated_non_m5_side=True,
            dedicated_tf_activity=True, direct_tf_hidden=True),
        source_bindings=refs, max_batch_rows=max_batch_rows,
        evidence_scope=_views().EVIDENCE_SCOPE,
        full_feature_profile=True, market_state_time_query_v2=True,
        full_ta_feature_profile=True, dedicated_m5_side=True,
        dedicated_non_m5_side=True, dedicated_tf_activity=True,
        direct_tf_hidden=True,
        unobserved_units_profile=_copy(unobserved_units_profile))
    value['anchor_span_input_extension'] = anchor._marker(
        refs['anchor_span_inputs_source'], anchor.FOLD1_BINDING)
    value[MARKER_KEY] = _marker()
    value['contract_sha256'] = _sha(value)
    _views()._require_view(value, unobserved_units_profile=unobserved_units_profile)
    if expected is not None:
        _need(expected == value, 'EXPECTED_VIEW')
    return value


def build_ema_delta_metadata_bundle(*, parent_packet, normalization_result,
                                    normalization_terminal, materialized_result,
                                    materialized_terminal, base_normalization,
                                    ta_normalization, unobserved_units_profile,
                                    parent_constructor_metadata, source_bindings,
                                    max_batch_rows=16):
    """Pure builder; caller verifies/publishes its three exact file refs in order."""
    packet = build_ema_delta_prefix_packet(
        parent_packet=parent_packet, normalization_result=normalization_result,
        normalization_terminal=normalization_terminal,
        materialized_result=materialized_result,
        materialized_terminal=materialized_terminal,
        base_normalization=base_normalization, ta_normalization=ta_normalization,
        unobserved_units_profile=unobserved_units_profile)
    _need(_sha(parent_constructor_metadata) == anchor.PARENT_CONSTRUCTOR_CONTENT_SHA,
          'PARENT_CONSTRUCTOR')
    constructor = anchor.build_anchor_span_constructor_metadata(parent_constructor_metadata)
    packet_ref = artifact_ref('PREFIX_PACKET.json', packet)
    constructor_ref = artifact_ref('CONSTRUCTOR_METADATA.json', constructor)
    _need(packet_ref == PACKET_REF and constructor_ref == CONSTRUCTOR_REF,
          'BUNDLE_ARTIFACT_IDENTITY')
    view = bind_ema_delta_feature_view(
        prefix_normalization=packet, prefix_normalization_ref=packet_ref,
        constructor_metadata=constructor, constructor_metadata_ref=constructor_ref,
        source_bindings=source_bindings, max_batch_rows=max_batch_rows,
        unobserved_units_profile=unobserved_units_profile,
        full_feature_profile=True, market_state_time_query_v2=True,
        full_ta_feature_profile=True, dedicated_m5_side=True,
        dedicated_non_m5_side=True, dedicated_tf_activity=True,
        direct_tf_hidden=True)
    return {'prefix_packet': packet, 'constructor_metadata': constructor,
            'feature_view': view, 'artifact_refs': {
                'PREFIX_PACKET.json': packet_ref,
                'CONSTRUCTOR_METADATA.json': constructor_ref,
                'FEATURE_VIEW.json': artifact_ref('FEATURE_VIEW.json', view)}}
