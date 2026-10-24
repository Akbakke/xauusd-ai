"""239 fixed, price-free current/previous book witness for analysis only.

Callers own223 file verification, canonical native selection and original parent
binding. This module loads no artifacts and does not encode S/ret_1 or alter any
native feature, normalization or model route.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json

import numpy as np

TIMEFRAMES = ('M5', 'M15', 'H1', 'H4', 'D1')
WITNESS_SCHEMA = 'gx1_direction_temporal_book_witness_v1'
PREFIX_SCHEMA = 'gx1_direction_temporal_book_prefix_v1'
# Exact223 columns, not a field-name or timeframe inference over numeric inputs.
BOOK_SOURCE = {
    'M5': (639, 502, 503, 'gc_m5_observed_book_tw_imbalance', 'gc_m5_observed_book_tw_imbalance_observed'),
    'M15': (493, 356, 357, 'gc_m15_observed_book_tw_imbalance', 'gc_m15_observed_book_tw_imbalance_observed'),
    'H1': (453, 316, 317, 'gc_h1_observed_book_tw_imbalance', 'gc_h1_observed_book_tw_imbalance_observed'),
    'H4': (431, 294, 295, 'gc_h4_observed_book_tw_imbalance', 'gc_h4_observed_book_tw_imbalance_observed'),
    'D1': (399, 262, 263, 'gc_d1_observed_book_tw_imbalance', 'gc_d1_observed_book_tw_imbalance_observed'),
}


def _require(ok, reason):
    if not ok:
        raise RuntimeError('DIRECTION_TEMPORAL_BOOK_' + reason)


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _array_sha(value):
    return hashlib.sha256(memoryview(np.ascontiguousarray(value)).cast('B')).hexdigest()


def _readonly(value):
    result = np.array(value, copy=True, order='C')
    result.setflags(write=False)
    return result


def _axis(value, reason, *, strict=False):
    _require(isinstance(value, np.ndarray) and value.dtype == np.int64
             and value.ndim == 1 and len(value) > 0 and np.all(value >= 0), reason)
    if strict:
        _require(np.all(value[1:] > value[:-1]), reason)
    return value


def _pairs(value, rows, reason):
    _require(isinstance(value, np.ndarray) and value.dtype == np.float32
             and value.shape == (rows, 2) and np.isfinite(value).all(), reason)
    _require(np.isin(value[:, 1], (0, 1)).all()
             and np.all(value[value[:, 1] == 0, 0] == 0), reason + '_MASKED_STORAGE')
    return value


def _spec(tf):
    return {'field_names': list(BOOK_SOURCE[tf][3:]),
            'binary_fields': [BOOK_SOURCE[tf][4]],
            'categorical_domains': {}, 'masking_pairs': [[0, 1]]}


def _lane(tf):
    return 'direction_book_' + tf.lower()


def project_direction_temporal_book_witness(surfaces, parent_decision_ns, *,
        current_native_row_indices, current_206_pairs):
    """Select exact causal native rows and prove supplied206 current-pair parity.

    ``surfaces`` has the existing verified223 four-key surface format. The
    caller's current IDs must come from the canonical native latest-closed
    route, not a search over a potentially sparse223 union. All parents remain.
    """
    decisions = _axis(parent_decision_ns, 'PARENT_CLOCK', strict=True)
    _require(np.all(decisions % 300_000_000_000 == 0), 'PARENT_M5_GRID')
    for value, reason in ((surfaces, 'SURFACES'),
                          (current_native_row_indices, 'CANONICAL_CURRENT_IDS'),
                          (current_206_pairs, 'CURRENT206_PAIRS')):
        _require(isinstance(value, Mapping) and set(value) == set(TIMEFRAMES), reason)
    rows = len(decisions)
    current, previous = np.empty((rows, 10), np.float32), np.empty((rows, 10), np.float32)
    ci, pi, ca, pa, hashes = {}, {}, {}, {}, {}
    for slot, tf in enumerate(TIMEFRAMES):
        surface = surfaces[tf]
        _require(isinstance(surface, Mapping) and set(surface) == {
            'native_row_indices', 'available_ns', 'features', 'field_names'}, 'SURFACE:' + tf)
        ids = _axis(surface['native_row_indices'], 'SOURCE_IDS:' + tf, strict=True)
        available = _axis(surface['available_ns'], 'SOURCE_CLOCK:' + tf, strict=True)
        features = surface['features']
        width, vi, mi, value_name, mask_name = BOOK_SOURCE[tf]
        _require(len(ids) == len(available) and np.all(available % 60_000_000_000 == 0)
                 and isinstance(features, np.ndarray) and features.dtype == np.float32
                 and features.shape == (len(ids), width)
                 and len(surface['field_names']) == width
                 and surface['field_names'][vi] == value_name
                 and surface['field_names'][mi] == mask_name
                 and all(not a.flags.writeable for a in (ids, available, features)),
                 'VERIFIED_SOURCE_HEADER:' + tf)
        canonical = _axis(current_native_row_indices[tf], 'CANONICAL_IDS:' + tf)
        _require(len(canonical) == rows and np.all(canonical > 0), 'CANONICAL_ROWS:' + tf)
        right = np.searchsorted(available, decisions, side='right') - 1
        _require(np.all(right >= 0), 'NO_CLOSED_CURRENT:' + tf)
        _require(np.array_equal(ids[right], canonical), 'CURRENT_ID_ASOF:' + tf)
        predecessor = canonical - 1
        left = np.searchsorted(ids, predecessor)
        _require(np.all(left < len(ids)), 'PREVIOUS_ID_MISSING:' + tf)
        _require(np.array_equal(ids[left], predecessor), 'PREVIOUS_ID_MISSING:' + tf)
        _require(np.all(available[left] < available[right])
                 and np.all(available[right] <= decisions), 'SELECTED_CLOCK:' + tf)
        now = _pairs(np.ascontiguousarray(features[np.ix_(right, [vi, mi])]), rows, 'CURRENT:' + tf)
        before = _pairs(np.ascontiguousarray(features[np.ix_(left, [vi, mi])]), rows, 'PREVIOUS:' + tf)
        peer = _pairs(current_206_pairs[tf], rows, 'CURRENT206:' + tf)
        _require(np.array_equal(now.view(np.uint32), peer.view(np.uint32)), 'CURRENT206_PAIR_BITS:' + tf)
        current[:, 2 * slot:2 * slot + 2], previous[:, 2 * slot:2 * slot + 2] = now, before
        ci[tf], pi[tf] = _readonly(canonical), _readonly(predecessor)
        ca[tf], pa[tf] = _readonly(available[right]), _readonly(available[left])
        hashes[tf] = _array_sha(peer)
    witness = {'schema_version': WITNESS_SCHEMA, 'parent_decision_ns': _readonly(decisions),
               'current': _readonly(current), 'previous': _readonly(previous),
               'current_native_row_indices': ci, 'previous_native_row_indices': pi,
               'current_available_ns': ca, 'previous_available_ns': pa,
               'proof': {'parent_rows': rows, 'parent_decision_ns_sha256': _array_sha(decisions),
                         'current_206_pairs_sha256': hashes, 'current_206_bit_parity': True,
                         'canonical_current_id_parity': True}}
    return _require_witness(witness)


def _require_witness(witness):
    _require(isinstance(witness, Mapping) and set(witness) == {
        'schema_version', 'parent_decision_ns', 'current', 'previous',
        'current_native_row_indices', 'previous_native_row_indices',
        'current_available_ns', 'previous_available_ns', 'proof'}
        and witness['schema_version'] == WITNESS_SCHEMA, 'WITNESS_SCHEMA')
    decisions = _axis(witness['parent_decision_ns'], 'WITNESS_CLOCK', strict=True)
    rows = len(decisions)
    proof = witness['proof']
    _require(isinstance(proof, Mapping) and proof.get('parent_rows') == rows
             and proof.get('parent_decision_ns_sha256') == _array_sha(decisions)
             and proof.get('current_206_bit_parity') is True
             and proof.get('canonical_current_id_parity') is True, 'WITNESS_PROOF')
    for slot in ('current', 'previous'):
        matrix = witness[slot]
        _require(isinstance(matrix, np.ndarray) and matrix.dtype == np.float32
                 and matrix.shape == (rows, 10) and not matrix.flags.writeable, 'WITNESS_VALUES')
    for slot, tf in enumerate(TIMEFRAMES):
        ci = _axis(witness['current_native_row_indices'][tf], 'WITNESS_CURRENT_IDS')
        pi = _axis(witness['previous_native_row_indices'][tf], 'WITNESS_PREVIOUS_IDS')
        ca = _axis(witness['current_available_ns'][tf], 'WITNESS_CURRENT_CLOCK')
        pa = _axis(witness['previous_available_ns'][tf], 'WITNESS_PREVIOUS_CLOCK')
        _require(all(len(x) == rows and not x.flags.writeable for x in (ci, pi, ca, pa))
                 and np.all(ci > 0) and np.array_equal(pi, ci - 1)
                 and np.all(pa < ca) and np.all(ca <= decisions), 'WITNESS_ROW_GEOMETRY')
        for name in ('current', 'previous'):
            _pairs(witness[name][:, 2 * slot:2 * slot + 2], rows, 'WITNESS_PAIRS')
        _require(proof['current_206_pairs_sha256'][tf] ==
                 _array_sha(witness['current'][:, 2 * slot:2 * slot + 2]), 'WITNESS_CURRENT206_HASH')
    return witness


def _require_prefix(normalization):
    from gx1.contracts.entry_relative_price_snapshot_v1 import _validate_surface_stats
    _require(isinstance(normalization, Mapping) and set(normalization) == {
        'schema_version', 'cutoff_ns', 'fit_row_count', 'first_decision_ns', 'last_decision_ns',
        'parent_decision_ns_sha256', 'current_fit_proofs', 'surfaces', 'contract_sha256'}
        and normalization['schema_version'] == PREFIX_SCHEMA, 'PREFIX_SCHEMA')
    _require(normalization['contract_sha256'] == _sha({k: v for k, v in normalization.items()
             if k != 'contract_sha256'}), 'PREFIX_HASH')
    rows, cutoff = normalization['fit_row_count'], normalization['cutoff_ns']
    _require(type(rows) is int and rows >= 2 and type(cutoff) is int
             and 0 <= normalization['first_decision_ns'] < normalization['last_decision_ns'] < cutoff
             and set(normalization['surfaces']) == set(TIMEFRAMES)
             and set(normalization['current_fit_proofs']) == set(TIMEFRAMES), 'PREFIX_GEOMETRY')
    for tf in TIMEFRAMES:
        stats = normalization['surfaces'][tf]
        _validate_surface_stats(stats, spec=_spec(tf), lane=_lane(tf), rows=rows)
        support = normalization['current_fit_proofs'][tf]['observed_value_rows']
        _require(type(support) is int and support > 0
                 and stats['observed_fit_counts'][0] == support, 'PREFIX_CURRENT_SUPPORT:' + tf)
    return normalization


def _prefix_proofs(witness, rows):
    return {tf: {
        'current_pairs_sha256': _array_sha(witness['current'][:rows, 2 * slot:2 * slot + 2]),
        'native_row_indices_sha256': _array_sha(witness['current_native_row_indices'][tf][:rows]),
        'available_ns_sha256': _array_sha(witness['current_available_ns'][tf][:rows]),
        'observed_value_rows': int(np.count_nonzero(witness['current'][:rows, 2 * slot + 1] == 1)),
    } for slot, tf in enumerate(TIMEFRAMES)}


def fit_direction_temporal_book_prefix(witness, *, cutoff_ns):
    """Five tiny current-slot fits; all original D<cutoff parents, no label mask."""
    from gx1.contracts.entry_relative_price_snapshot_v1 import _fit_surface
    witness = _require_witness(witness)
    _require(type(cutoff_ns) is int and cutoff_ns > 0, 'PREFIX_CUTOFF')
    decisions = witness['parent_decision_ns']
    rows = int(np.searchsorted(decisions, cutoff_ns, side='left'))
    _require(rows >= 2, 'PREFIX_PARENT_COUNT')
    fitted, proofs = {}, _prefix_proofs(witness, rows)
    # Preflight all five domains before any fit; count0 is not a fitted scale.
    for tf in TIMEFRAMES:
        _require(proofs[tf]['observed_value_rows'] > 0, 'PREFIX_NO_CURRENT_OBSERVATIONS:' + tf)
    for slot, tf in enumerate(TIMEFRAMES):
        values = np.ascontiguousarray(witness['current'][:rows, 2 * slot:2 * slot + 2])
        fitted[tf] = _fit_surface(values, _spec(tf), _lane(tf))
    value = {'schema_version': PREFIX_SCHEMA, 'cutoff_ns': cutoff_ns, 'fit_row_count': rows,
             'first_decision_ns': int(decisions[0]), 'last_decision_ns': int(decisions[rows - 1]),
             'parent_decision_ns_sha256': _array_sha(decisions[:rows]),
             'current_fit_proofs': proofs, 'surfaces': fitted}
    return _require_prefix({**value, 'contract_sha256': _sha(value)})


def apply_direction_temporal_book_prefix(witness, normalization):
    """Apply the same frozen TF statistics to both slots; never fit at apply."""
    from gx1.contracts.entry_model_native_input_normalization_v1 import apply_surface_normalization
    witness, normalization = _require_witness(witness), _require_prefix(normalization)
    decisions = witness['parent_decision_ns']
    rows = int(np.searchsorted(decisions, normalization['cutoff_ns'], side='left'))
    _require(rows == normalization['fit_row_count']
             and int(decisions[0]) == normalization['first_decision_ns']
             and int(decisions[rows - 1]) == normalization['last_decision_ns']
             and _array_sha(decisions[:rows]) == normalization['parent_decision_ns_sha256']
             and _prefix_proofs(witness, rows) == normalization['current_fit_proofs'],
             'PREFIX_PARENT_BINDING')
    before_hash = _sha(normalization)
    current, previous = np.empty_like(witness['current']), np.empty_like(witness['previous'])
    for slot, tf in enumerate(TIMEFRAMES):
        columns = slice(2 * slot, 2 * slot + 2)
        stats = normalization['surfaces'][tf]
        current[:, columns] = apply_surface_normalization(witness['current'][:, columns], stats)
        previous[:, columns] = apply_surface_normalization(witness['previous'][:, columns], stats)
    _require(_sha(normalization) == before_hash, 'PREFIX_CHANGED_DURING_APPLY')
    return {'current': _readonly(current),
            'current_previous': _readonly(np.column_stack((current, previous))),
            'normalization_sha256': normalization['contract_sha256'],
            'parent_decision_ns_sha256': witness['proof']['parent_decision_ns_sha256']}
