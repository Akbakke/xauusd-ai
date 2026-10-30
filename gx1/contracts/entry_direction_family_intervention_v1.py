"""Bounded research value interventions; no model, data IO, fit or launch.

Observation masks stay unchanged: value contribution conditional on coverage,
not total information removal or a causal claim. Native contracts are untouched.
"""
from __future__ import annotations
import hashlib
import json
from collections.abc import Mapping
import numpy as np
from gx1.contracts import entry_gc_mtf_history_v1 as history
from gx1.contracts import entry_model_native_input_normalization_v1 as norm
from gx1.contracts.entry_model_native_signal_v1 import MODEL_NATIVE_SEQ_LEN
from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
from gx1.features import entry_direction_momentum_profile_v1 as momentum
from gx1.features import entry_specialist_feature_groups_v1 as owners
from gx1.features import gc_native_context_v1 as gc
from gx1.features import micro_structure_v1 as micro
from gx1.features import htf_features as htf
from gx1.features.d1_macro_context_v1 import d1_macro_context_contract

SCHEMA_VERSION = 'gx1_direction_family_value_intervention_v1'
BASELINE = 'baseline'
MACRO_ARM = 'macro_context'
CONVENTION = {
    'estimand': 'family_values_conditional_on_unchanged_observation_masks',
    'continuous': 'exact_float32_TRAIN_normalization_center',
    'unobserved': 'source_masked_storage_zero',
    'binary_market_event': 'source_binary_false_zero',
    'categorical': 'first_canonical_declared_domain_constant_reference_not_natural_neutral',
    'ema_stack_aligned_v2': 'source_declared_flat_zero_preserves_ternary_domain',
    'source_constants': 'preserve_declared_source_constant_byte_identically',
    'history_normalization': 'own_TF_TRAIN_owner_not_current_context_stats',
    'claim': 'frozen_sensitivity_only_unless_separately_matched_retraining',
}

def _require(ok, reason):
    if not ok:
        raise RuntimeError('FAMILY_INTERVENTION_' + reason)

def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def _plain(value):
    return json.loads(json.dumps(value, allow_nan=False))

def _surface(names, stats, routes, *, rows, categorical=False, source_constants=None):
    names = list(names)
    assigned = {int(i): f for f, indices in routes.items() for i in indices}
    _require(sum(map(len, routes.values())) == len(assigned), 'DUPLICATE_FAMILY_OWNER')
    pairs = list(dict.fromkeys((*gc.masking_pairs(names), *micro.range_observation_masking_pairs(names)))) if not categorical else []
    _require(len(dict(pairs)) == len(pairs), 'CONFLICTING_MASK_OWNER')
    masks = {m for _, m in pairs} | {i for i, n in enumerate(names)
        if gc.field_role(n) == 'gc_observation_mask' or n in micro.range_observation_binary_fields(names)}
    constants = {names.index(n): value for n, value in (source_constants or {}).items()}
    replacement, modes = [], []
    for i, name in enumerate(names):
        if i in masks:
            value, mode = None, 'preserve_observation_mask'
        elif i in constants:
            value, mode = None, CONVENTION['source_constants']
        elif categorical or bool(stats['categorical_mask'][i]):
            domain = (stats['domains'] if categorical else stats['categorical_domains'])[name]
            value, mode = domain[0], CONVENTION['categorical']
        elif bool(stats['binary_mask'][i]):
            value, mode = 0.0, CONVENTION['binary_market_event']
        elif name == 'ema_stack_aligned_v2':
            value, mode = 0.0, CONVENTION['ema_stack_aligned_v2']
        else:
            value, mode = float(np.float32(stats['center'][i])), CONVENTION['continuous']
        replacement.append(value); modes.append(mode)
    cross = [{'value_index': v, 'mask_index': m, 'value_family': assigned.get(v), 'mask_family': assigned.get(m)}
             for v, m in pairs if v in assigned and assigned.get(v) != assigned.get(m)]
    return {'field_names': names, 'rows': rows, 'dtype': 'int64' if categorical else 'float32',
        'normalization_sha256': stats['contract_sha256'] if categorical else stats['stats_sha256'],
        'family_indices': {f: [int(i) for i in indices] for f, indices in routes.items()},
        'mask_pairs': [list(p) for p in pairs], 'preserved_mask_indices': sorted(masks),
        'source_constants': {str(i): value for i, value in constants.items()},
        'preserved_constant_indices': sorted(constants),
        'shared_mask_dependencies': cross, 'replacement_raw': replacement, 'replacement_convention': modes,
        'binary_indices': [] if categorical else [i for i, v in enumerate(stats['binary_mask']) if v],
        'categorical_domains': _plain(stats['domains'] if categorical else stats['categorical_domains'])}

def compile_family_intervention(*, profile_reference, gc_mtf_history, base_normalization, gc_normalization):
    """Compile already-loaded TRAIN metadata; caller must bind digest in recipe."""
    bound_history = history.require_gc_mtf_history(gc_mtf_history)
    layout = bound_history['layout']
    gc_norm = history.require_gc_mtf_history_normalization(gc_normalization, layout=layout)
    _require(gc_norm == bound_history['normalization'], 'GC_NORMALIZATION_IDENTITY')
    profile = momentum.compile_price_momentum_profile(bound_history)
    _require(dict(profile_reference) == {k: profile[k] for k in ('name', 'contract_sha256')}, 'PROFILE_IDENTITY')
    fields = profile['physical_field_names']
    base = norm.require_input_normalization_contract(base_normalization,
        expected_field_names={s: fields['signal'] if s == 'signal' else fields['ctx_cont'] if s == 'ctx_cont' else fields['mtf'] for s in norm.EXPECTED_SURFACES},
        expected_ctx_cat_names=fields['ctx_cat'], expected_entry_training_phase=ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE)
    families = list(owners.MODEL_NATIVE_TRAINING_SPECIALISTS)
    _require(len(families) == 8 and list(profile['local_specialists']) == families
        and list(profile['mtf_specialists']) == families, 'ALL_EIGHT_FAMILIES')
    routes = owners.model_native_context_specialist_routing_contract()
    surfaces = {
        'seq_x': _surface(fields['signal'], base['surfaces']['signal'], profile['local_signal_indices_by_specialist'], rows=MODEL_NATIVE_SEQ_LEN),
        'snap_x': _surface(fields['signal'], base['surfaces']['signal'], profile['local_signal_indices_by_specialist'], rows=None),
        'ctx_cont': _surface(fields['ctx_cont'], base['surfaces']['ctx_cont'], profile['ctx_cont_indices_by_specialist'], rows=None),
        'ctx_cat': _surface(fields['ctx_cat'], base['ctx_cat'], routes['ctx_cat_indices'], rows=None, categorical=True),
    }
    for tf in layout['timeframes']:
        suffix = tf.lower()
        surfaces['seq_' + suffix] = _surface(fields['mtf'], base['surfaces']['mtf_' + suffix], profile['mtf_indices_by_specialist'], rows=layout['window_rows'][tf],
            source_constants=htf.MULTI_TF_SOURCE_CONSTANT_BINARY_EVENTS_V4.get(tf, {}))
        surfaces['gc_seq_' + suffix] = _surface(layout['field_names'][tf], gc_norm['surfaces']['gc_mtf_' + suffix], layout['family_indices'][tf], rows=layout['window_rows'][tf])
    aliases = owners.model_native_context_temporal_alias_policy(fields['signal'])['aliases']
    for a in aliases:
        i, j = int(a['signal_index']), int(a['ctx_cont_index'])
        x, y = surfaces['seq_x'], surfaces['ctx_cont']
        _require(x['replacement_raw'][i] == y['replacement_raw'][j]
            and x['replacement_convention'][i] == y['replacement_convention'][j]
            and all((i in x['family_indices'].get(f, [])) == (j in y['family_indices'].get(f, [])) for f in families), 'ALIAS_REPLACEMENT_OR_OWNER')
    arms = {BASELINE: {key: [] for key in surfaces}}
    for family in families:
        arms[family] = {key: [i for i in item['family_indices'].get(family, []) if i not in item['preserved_mask_indices'] and i not in item['preserved_constant_indices']]
                       for key, item in surfaces.items()}
    macro_contract = d1_macro_context_contract()
    _require(list(profile['macro_context_fields']) == macro_contract['fields']
        and macro_contract['specialist_owner'] == 'session_regime_encoder', 'MACRO_SOURCE_OWNER')
    macro = set(macro_contract['fields'])
    arms[MACRO_ARM] = {key: [i for i, n in enumerate(item['field_names']) if n.removeprefix('ctx_cont.') in macro
        and i not in item['preserved_mask_indices'] and i not in item['preserved_constant_indices']] if key in ('seq_x', 'snap_x', 'ctx_cont') else []
        for key, item in surfaces.items()}
    _require(len(arms[MACRO_ARM]['ctx_cont']) == len([n for i, n in enumerate(surfaces['ctx_cont']['field_names']) if n in macro
        and i not in surfaces['ctx_cont']['preserved_mask_indices'] and i not in surfaces['ctx_cont']['preserved_constant_indices']]) and all(set(indices) <= set(arms['session_regime_encoder'][key])
        for key, indices in arms[MACRO_ARM].items()), 'MACRO_OWNER_CLOSURE')
    result = _plain({'schema_version': SCHEMA_VERSION, 'profile_reference': profile_reference,
        'base_normalization_sha256': base['contract_sha256'], 'gc_normalization_sha256': gc_norm['contract_sha256'],
        'gc_layout_sha256': layout['contract_sha256'], 'conventions': CONVENTION,
        'families': families, 'surfaces': surfaces, 'aliases': aliases,
        'macro_context_fields': sorted(macro), 'macro_source_contract': macro_contract, 'arms': arms,
        'same_named_gc_history_is_not_declared_current_alias': True,
        'training_authorized': False, 'test_accessed': False})
    return {**result, 'contract_sha256': _sha(result)}

def require_family_intervention(contract, *, expected_sha256):
    _require(isinstance(contract, Mapping) and contract.get('schema_version') == SCHEMA_VERSION
        and contract.get('contract_sha256') == expected_sha256
        and _sha({k: v for k, v in contract.items() if k != 'contract_sha256'}) == expected_sha256, 'CONTRACT_HASH')
    _require(contract['conventions'] == CONVENTION and contract['training_authorized'] is False
        and contract['test_accessed'] is False, 'CONTRACT_SCOPE')
    return contract

def _validate_inputs(features, contract):
    _require(isinstance(features, Mapping) and set(features) == set(contract['surfaces']), 'INPUT_KEYS_NO_TARGETS')
    batch = None
    for key, item in contract['surfaces'].items():
        x = features[key]
        _require(isinstance(x, np.ndarray) and x.dtype == np.dtype(item['dtype'])
            and x.ndim == (2 if item['rows'] is None else 3), 'INPUT_DTYPE_OR_RANK:' + key)
        if batch is None:
            batch = x.shape[0]
        _require(batch > 0 and x.shape == ((batch, len(item['field_names'])) if item['rows'] is None else (batch, item['rows'], len(item['field_names'])))
            and np.isfinite(x).all(), 'INPUT_SHAPE_OR_FINITE:' + key)
        for i, value in item['source_constants'].items():
            _require(np.all(x[..., int(i)] == value), 'SOURCE_CONSTANT_DOMAIN:' + key)
        binary = x[..., item['binary_indices']]
        _require(((binary == 0.0) | (binary == 1.0)).all(), 'BINARY_DOMAIN:' + key)
        for name, domain in item['categorical_domains'].items():
            _require(np.isin(x[..., item['field_names'].index(name)], domain).all(), 'CATEGORICAL_DOMAIN:' + key)
        if item['mask_pairs']:
            values, masks = np.asarray(item['mask_pairs'], dtype=np.intp).T
            observed = x[..., masks]
            _require(((observed == 0.0) | (observed == 1.0)).all()
                and ((observed != 0.0) | (x[..., values] == 0.0)).all(), 'MASKED_STORAGE:' + key)
        if key.startswith('seq_') and 'ema_stack_aligned_v2' in item['field_names']:
            _require(np.isin(x[..., item['field_names'].index('ema_stack_aligned_v2')], (-1.0, 0.0, 1.0)).all(), 'TERNARY_DOMAIN:' + key)
    _require(np.array_equal(features['seq_x'][:, -1], features['snap_x']), 'SNAPSHOT_ALIAS')
    signal_aliases = [int(a['signal_index']) for a in contract['aliases']]
    context_aliases = [int(a['ctx_cont_index']) for a in contract['aliases']]
    _require(np.array_equal(features['snap_x'][:, signal_aliases], features['ctx_cont'][:, context_aliases]), 'CONTEXT_ALIAS')

def apply_family_intervention_numpy(features, contract, *, arm, expected_sha256):
    """Independent raw arrays, only declared value columns changed; no labels."""
    contract = require_family_intervention(contract, expected_sha256=expected_sha256)
    _require(arm in contract['arms'], 'UNKNOWN_ARM')
    _validate_inputs(features, contract)
    output = {key: value.copy() for key, value in features.items()}
    for key, indices in contract['arms'][arm].items():
        item, target, raw = contract['surfaces'][key], output[key], features[key]
        pairs = dict(item['mask_pairs'])
        for i in indices:
            _require(i not in item['preserved_mask_indices'] and i not in item['preserved_constant_indices'], 'MASK_OR_CONSTANT_MUTATION')
            value = item['replacement_raw'][i]
            target[..., i] = np.where(raw[..., pairs[i]] == 1, value, 0.0) if i in pairs else value
        protected = sorted(set(range(raw.shape[-1])) - set(indices))
        _require(target[..., protected].tobytes() == raw[..., protected].tobytes(), 'UNRELATED_VALUE_CHANGED:' + key)
    _validate_inputs(output, contract)
    return output
