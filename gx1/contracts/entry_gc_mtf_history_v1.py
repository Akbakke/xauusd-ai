"""Additive218 own-TF GC history metadata and pure array contracts.

No payload loader, policy, native launch or fitting admission lives here. The
producer binds verified024 tables and the existing native TF row selector.
The original049 context,055 normalization and base tensors remain unchanged.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import re
import numpy as np

from gx1.features import gc_native_context_v1 as gc

SCHEMA_VERSION = 'gx1_gc_mtf_history_layout_v1'
NORMALIZATION_SCHEMA_VERSION = 'gx1_gc_mtf_history_normalization_v1'
TIMEFRAMES = ('M5', 'M15', 'H1', 'H4', 'D1')
# Fixed218 geometry; the architecture/producer must match the existing native
# selector. No alternate window selector or source-clock inference is exposed.
WINDOW_ROWS = {'M5': 16, 'M15': 64, 'H1': 96, 'H4': 96, 'D1': 252}
TIMEFRAME_MINUTES = {'M5': 5, 'M15': 15, 'H1': 60, 'H4': 240, 'D1': 1440}
MINUTE_NS = 60_000_000_000
TEST_BOUNDARY_NS = 1_782_864_000_000_000_000
FIT_POPULATION = 'unique_native_tf_rows_consumed_by_train_windows'
# Exact immutable registry table identities, never a field-name heuristic.
SOURCE_TABLE_TIMEFRAMES = {
    'BASIS_H1.parquet': 'H1', 'BASIS_H4.parquet': 'H4',
    'BASIS_M15.parquet': 'M15', 'BASIS_M5.parquet': 'M5',
    'M5_GC_FEATURES.parquet': 'M5',
    'TRADE_FEATURES_D1.parquet': 'D1', 'TRADE_FEATURES_H1.parquet': 'H1',
    'TRADE_FEATURES_H4.parquet': 'H4', 'TRADE_FEATURES_M15.parquet': 'M15',
    'TRADE_FEATURES_M5.parquet': 'M5',
    'retained::TRADE_FEATURES_H1.parquet': 'H1',
    'retained::TRADE_FEATURES_H4.parquet': 'H4',
    'retained::TRADE_FEATURES_M15.parquet': 'M15',
    'retained::TRADE_FEATURES_M5.parquet': 'M5',
}


def _require(value, reason):
    if not value:
        raise RuntimeError('GC_MTF_HISTORY_' + reason)


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def _array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value, dtype='<i8').tobytes()).hexdigest()


def _hash(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def gc_mtf_history_layout():
    """Return the complete ordered049 inventory partitioned by bound source TF."""
    from gx1.features.entry_specialist_feature_groups_v1 import MODEL_NATIVE_TRAINING_SPECIALISTS
    _require(set(gc.REGISTRY['tables']) == set(SOURCE_TABLE_TIMEFRAMES), 'SOURCE_TABLE_INVENTORY')
    families = list(MODEL_NATIVE_TRAINING_SPECIALISTS)
    _require(len(families) == len(set(families)) == 8, 'FAMILY_OWNER_INVENTORY')
    source = {tf: [] for tf in TIMEFRAMES}
    names = {tf: [] for tf in TIMEFRAMES}
    owners = {tf: {family: [] for family in families} for tf in TIMEFRAMES}
    for original in gc.REGISTRY['fields']:
        item = json.loads(json.dumps(original))
        tf = SOURCE_TABLE_TIMEFRAMES[item['table']]
        _require(item['owner'] in families, 'SOURCE_SPECIALIST')
        source[tf].append(item)
        for name in (item['name'], item['availability_field']):
            owners[tf][item['owner']].append(len(names[tf]))
            names[tf].append(name)
    _require(sum(len(items) for items in source.values()) == len(gc.VALUE_FIELDS) == 1205,
        'ALL049_VALUES_PRESERVED')
    for original in gc.TRAILING_SUPPORT_FIELDS:
        item = dict(original)
        tf = SOURCE_TABLE_TIMEFRAMES[item['table']]
        _require(item['availability_field'] in names[tf]
            and item['availability_field'] in gc.STRUCTURAL_VALUE_FIELDS, 'EXISTING_BBO_BINARY_GATE')
        source[tf].append(item)
        owners[tf][item['owner']].append(len(names[tf]))
        names[tf].append(item['name'])
    _require(sum(map(len, names.values())) == 2 * len(gc.VALUE_FIELDS) + 5
        and all(len(v) == len(set(v)) for v in names.values()), 'NO_PADDING_OR_REPLICATION')
    value = {'schema_version': SCHEMA_VERSION, 'timeframes': list(TIMEFRAMES),
        'families': families, 'field_names': names, 'family_indices': owners,
        'value_mask_pairs': {tf: [list(pair) for pair in gc.masking_pairs(names[tf])] for tf in TIMEFRAMES},
        'widths': {tf: len(names[tf]) for tf in TIMEFRAMES}, 'window_rows': dict(WINDOW_ROWS),
        'binary_fields': {tf: [name for name in names[tf] if name in gc.mtf_history_binary_fields(names[tf])]
            for tf in TIMEFRAMES}, 'categorical_domains': {tf: {} for tf in TIMEFRAMES},
        'source_registry': {'path': gc.REGISTRY_PATH, 'sha256': gc.REGISTRY_SHA256},
        'source_tables': json.loads(json.dumps(gc.REGISTRY['tables'])), 'source_fields': source}
    return {**value, 'contract_sha256': _sha(value)}


def require_gc_mtf_history_layout(value):
    _require(isinstance(value, Mapping) and dict(value) == gc_mtf_history_layout(), 'LAYOUT_CONTRACT')
    _require(value['contract_sha256'] == _sha({k: v for k, v in value.items()
        if k != 'contract_sha256'}), 'LAYOUT_CANONICAL_HASH')
    return dict(value)


def _axis(values, reason, *, nonnegative=False):
    _require(isinstance(values, np.ndarray) and values.dtype == np.int64
        and values.ndim == 1 and len(values) > 0
        and np.all(values[1:] > values[:-1])
        and (not nonnegative or np.all(values >= 0)), reason)
    return values


def project_gc_mtf_source_rows(*, timeframe, historical_close_ns, source_tables, layout=None):
    """Project caller-verified tables at each historical TF close, without I/O.

    The producer owns file hashes, incident/source identity and native row-clock
    parity. Unknown pre-source rows receive zeros with false observation gates.
    """
    layout = gc_mtf_history_layout() if layout is None else require_gc_mtf_history_layout(layout)
    _require(timeframe in TIMEFRAMES, 'PROJECT_TIMEFRAME')
    closes = _axis(historical_close_ns, 'HISTORICAL_CLOSE_CLOCK', nonnegative=True)
    _require(np.all(closes % MINUTE_NS == 0) and closes[-1] <= TEST_BOUNDARY_NS, 'HISTORICAL_CLOSE_BOUNDARY')
    fields = layout['source_fields'][timeframe]
    required = set(item['table'] for item in fields)
    _require(isinstance(source_tables, Mapping) and set(source_tables) == required, 'PROJECT_TABLE_KEYS')
    joins = {}
    for key in required:
        clock, table = source_tables[key]
        _require(isinstance(clock, np.ndarray) and clock.dtype == np.int64 and clock.ndim == 1
            and np.all(clock[1:] > clock[:-1])
            and len(clock) == table.num_rows
            and np.array_equal(clock, table['available_ns'].to_numpy())
            and set(gc._needed_columns([f for f in fields if f['table'] == key])) <= set(table.column_names),
            'PROJECT_TABLE_CLOCK_OR_COLUMNS:' + key)
        index = np.searchsorted(clock, closes, side='right') - 1
        # An own-TF history row names this exact closed bucket. A missing
        # bucket is unknown, not the previous available GC bucket. Individual
        # values within a matching row retain their declared source clocks.
        matched = (index >= 0)
        if len(clock):
            matched &= clock[np.maximum(index, 0)] == closes
        joins[key] = np.where(matched, index, -1)
    names = layout['field_names'][timeframe]
    positions = {name: i for i, name in enumerate(names)}
    output = np.zeros((len(closes), len(names)), np.float32)
    for item in fields:
        values, observed = gc._field_values(item, *source_tables[item['table']], closes, joins[item['table']])
        output[:, positions[item['name']]] = values
        # The five new values reuse an existing source BBO-valid value. They
        # must not replace that value with a newly manufactured observation.
        if item.get('transform') != gc.BBO_TRAILING_SUPPORT_TRANSFORM:
            output[:, positions[item['availability_field']]] = observed.astype(np.float32)
        else:
            _require(np.array_equal(output[:, positions[item['availability_field']]], observed.astype(np.float32)),
                'TRAILING_SUPPORT_SHARED_GATE')
    gc.require_masked_storage_numpy(output, names)
    return output


def require_gc_mtf_history_surfaces(value, *, layout):
    """Validate five immutable sidecar headers/clocks once, never scan all values.

    Payload hashes and effective provenance remain loader-owned. Selected raw
    windows and fitted TRAIN rows receive their own finite/domain validation.
    """
    layout = require_gc_mtf_history_layout(layout)
    _require(isinstance(value, Mapping) and set(value) == set(TIMEFRAMES), 'SURFACE_INVENTORY')
    result = {}
    for tf in TIMEFRAMES:
        surface = value[tf]
        _require(isinstance(surface, Mapping) and set(surface) == {
            'native_row_indices', 'available_ns', 'features', 'field_names'}, 'SURFACE_KEYS:' + tf)
        rows = _axis(surface['native_row_indices'], 'SURFACE_NATIVE_ROWS:' + tf, nonnegative=True)
        available = _axis(surface['available_ns'], 'SURFACE_CLOSED_CLOCK:' + tf, nonnegative=True)
        features = surface['features']
        _require(len(rows) == len(available) and np.all(available % MINUTE_NS == 0)
            and available[-1] <= TEST_BOUNDARY_NS
            and isinstance(features, np.ndarray) and features.dtype == np.float32
            and features.shape == (len(rows), layout['widths'][tf])
            and surface['field_names'] == layout['field_names'][tf], 'SURFACE_SHAPE_CLOCK_FIELDS:' + tf)
        result[tf] = {'field_names': list(surface['field_names'])}
        for key in ('native_row_indices', 'available_ns', 'features'):
            readonly = surface[key].view()
            readonly.setflags(write=False)
            result[tf][key] = readonly
    return result


def gather_gc_mtf_history_rows(surface, *, tf, native_row_indices, available_ns, layout):
    """Gather the exact already-selected native window, with no missing-row fill.

    The Dataset validates layout/surface headers once. This hot path checks only
    the selected native IDs/clocks and value/mask cells, not the full matrix.
    """
    _require(isinstance(layout, Mapping) and layout.get('schema_version') == SCHEMA_VERSION
        and tf in TIMEFRAMES and layout.get('window_rows', {}).get(tf) == WINDOW_ROWS[tf], 'GATHER_LAYOUT')
    indices = _axis(native_row_indices, 'GATHER_NATIVE_ROWS', nonnegative=True)
    closes = _axis(available_ns, 'GATHER_CLOSED_CLOCK', nonnegative=True)
    _require(len(indices) == len(closes) == WINDOW_ROWS[tf]
        and np.all(np.diff(indices) == 1), 'GATHER_NATIVE_WINDOW')
    _require(surface['field_names'] == layout['field_names'][tf]
        and all(not surface[k].flags.writeable for k in ('native_row_indices', 'available_ns', 'features')),
        'GATHER_UNVALIDATED_SURFACE')
    position = np.searchsorted(surface['native_row_indices'], indices)
    _require(np.all(position < len(surface['native_row_indices'])), 'GATHER_MISSING_NATIVE_ROW')
    _require(np.array_equal(surface['native_row_indices'][position], indices)
        and np.array_equal(surface['available_ns'][position], closes), 'GATHER_NATIVE_ROW_CLOCK_MISMATCH')
    output = np.asarray(surface['features'][position], dtype=np.float32)
    _require(output.shape == (WINDOW_ROWS[tf], layout['widths'][tf]) and np.isfinite(output).all(), 'GATHER_VALUES')
    gc.require_masked_storage_numpy(output, layout['field_names'][tf])
    return output


def _normalization_provenance(value, layout, surfaces):
    _require(isinstance(value, Mapping) and set(value) == {
        'split', 'fit_population', 'train_start_ns', 'train_end_ns', 'row_union', 'source_registry'}, 'NORMALIZATION_PROVENANCE_KEYS')
    p = dict(value)
    _require(p['split'] == 'train' and p['fit_population'] == FIT_POPULATION
        and type(p['train_start_ns']) is int and type(p['train_end_ns']) is int
        and 0 <= p['train_start_ns'] < p['train_end_ns'] <= TEST_BOUNDARY_NS
        and p['source_registry'] == layout['source_registry']
        and isinstance(p['row_union'], Mapping) and set(p['row_union']) == set(TIMEFRAMES), 'NORMALIZATION_TRAIN_PROVENANCE')
    for tf in TIMEFRAMES:
        row = p['row_union'][tf]
        _require(isinstance(row, Mapping) and set(row) == {
            'rows', 'native_row_indices_sha256', 'available_ns_sha256'}
            and type(row['rows']) is int and row['rows'] >= 2
            and row['rows'] == surfaces['gc_mtf_' + tf.lower()]['fit_row_count']
            and _hash(row['native_row_indices_sha256']) and _hash(row['available_ns_sha256']),
            'NORMALIZATION_ROW_UNION:' + tf)
    return p


def require_gc_mtf_history_normalization(value, *, layout):
    from gx1.contracts.entry_model_native_input_normalization_v1 import require_surface_normalization
    layout = require_gc_mtf_history_layout(layout)
    _require(isinstance(value, Mapping) and set(value) == {
        'schema_version', 'layout_sha256', 'surfaces', 'provenance', 'contract_sha256'}, 'NORMALIZATION_KEYS')
    data = dict(value)
    _require(data['schema_version'] == NORMALIZATION_SCHEMA_VERSION
        and data['layout_sha256'] == layout['contract_sha256']
        and isinstance(data['surfaces'], Mapping)
        and set(data['surfaces']) == {'gc_mtf_' + tf.lower() for tf in TIMEFRAMES}, 'NORMALIZATION_LAYOUT')
    for tf in TIMEFRAMES:
        name = 'gc_mtf_' + tf.lower()
        require_surface_normalization(data['surfaces'][name], surface=name, field_names=layout['field_names'][tf])
        _require(data['surfaces'][name]['categorical_domains'] == layout['categorical_domains'][tf],
            'NORMALIZATION_CATEGORICAL_DOMAINS:' + tf)
    _normalization_provenance(data['provenance'], layout, data['surfaces'])
    _require(_hash(data['contract_sha256']) and data['contract_sha256'] == _sha(
        {k: v for k, v in data.items() if k != 'contract_sha256'}), 'NORMALIZATION_HASH')
    return data


def fit_gc_mtf_history_normalization(surfaces, train_row_indices_by_tf, *, train_start_ns, train_end_ns, layout):
    """Fit only the explicit unique TRAIN union chosen by the existing selector.

    Warmup history before TRAIN start is legal. CONTROL rows are never read by
    this fit, and no unobserved-field domain fallback is introduced.
    """
    from gx1.contracts.entry_model_native_input_normalization_v1 import fit_surface_normalization
    layout = require_gc_mtf_history_layout(layout)
    surfaces = require_gc_mtf_history_surfaces(surfaces, layout=layout)
    _require(type(train_start_ns) is int and type(train_end_ns) is int
        and 0 <= train_start_ns < train_end_ns <= TEST_BOUNDARY_NS, 'FIT_TRAIN_BOUNDARIES')
    _require(isinstance(train_row_indices_by_tf, Mapping)
        and set(train_row_indices_by_tf) == set(TIMEFRAMES), 'FIT_TRAIN_UNION_KEYS')
    fitted, unions = {}, {}
    for tf in TIMEFRAMES:
        rows = _axis(train_row_indices_by_tf[tf], 'FIT_UNIQUE_TRAIN_ROWS:' + tf, nonnegative=True)
        surface = surfaces[tf]
        position = np.searchsorted(surface['native_row_indices'], rows)
        _require(len(rows) >= 2 and np.all(position < len(surface['native_row_indices'])), 'FIT_TRAIN_ROW_BOUNDS:' + tf)
        _require(np.array_equal(surface['native_row_indices'][position], rows), 'FIT_TRAIN_ROW_IDENTITY:' + tf)
        closes = surface['available_ns'][position]
        _require(np.all(closes < train_end_ns), 'FIT_CONTROL_OR_LATER_CLOCK:' + tf)
        selected = np.asarray(surface['features'][position], dtype=np.float32)
        gc.require_masked_storage_numpy(selected, layout['field_names'][tf])
        name = 'gc_mtf_' + tf.lower()
        fitted[name] = fit_surface_normalization(selected, surface=name,
            field_names=layout['field_names'][tf], row_count=len(rows),
            semantic_categorical_domains=layout['categorical_domains'][tf])
        unions[tf] = {'rows': len(rows), 'native_row_indices_sha256': _array_sha(rows),
            'available_ns_sha256': _array_sha(closes)}
    value = {'schema_version': NORMALIZATION_SCHEMA_VERSION, 'layout_sha256': layout['contract_sha256'],
        'surfaces': fitted, 'provenance': {'split': 'train', 'fit_population': FIT_POPULATION,
            'train_start_ns': train_start_ns, 'train_end_ns': train_end_ns,
            'row_union': unions, 'source_registry': dict(layout['source_registry'])}}
    return require_gc_mtf_history_normalization({**value, 'contract_sha256': _sha(value)}, layout=layout)


def require_gc_mtf_history(value):
    """The one candidate wrapper schema consumed by model/profile/bundle."""
    _require(isinstance(value, Mapping) and set(value) == {'layout', 'normalization'}, 'WRAPPER_KEYS')
    layout = require_gc_mtf_history_layout(value['layout'])
    return {'layout': layout, 'normalization': require_gc_mtf_history_normalization(value['normalization'], layout=layout)}



def _gc_mtf_history_artifact_ref(value):
    from pathlib import Path
    _require(isinstance(value, Mapping) and set(value) == {'path', 'sha256'}, 'ARTIFACT_REF_KEYS')
    path = value['path']
    _require(isinstance(path, str) and Path(path).is_absolute()
        and str(Path(path)) == path and '..' not in Path(path).parts
        and _hash(value['sha256']), 'ARTIFACT_REF')
    return {'path': path, 'sha256': value['sha256']}


def _read_gc_mtf_history_artifact_json(binding):
    import os
    import stat
    from pathlib import Path
    binding = _gc_mtf_history_artifact_ref(binding)
    path = Path(binding['path'])
    _require(path.resolve(strict=True) == path, 'ARTIFACT_JSON_PATH')
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        _require(stat.S_ISREG(info.st_mode) and 0 < info.st_size <= 2 * 1024 * 1024, 'ARTIFACT_JSON_SIZE')
        body = stream.read(2 * 1024 * 1024 + 1)
    _require(len(body) == info.st_size and hashlib.sha256(body).hexdigest() == binding['sha256'],
        'ARTIFACT_JSON_HASH')
    def unique_object(pairs):
        result = dict(pairs)
        _require(len(result) == len(pairs), 'ARTIFACT_JSON_DUPLICATE_KEYS')
        return result
    value = json.loads(body, object_pairs_hook=unique_object,
        parse_constant=lambda _: _require(False, 'ARTIFACT_JSON_NONFINITE'))
    _require(isinstance(value, Mapping), 'ARTIFACT_JSON_OBJECT')
    return value


def require_gc_mtf_history_artifact_identity(value, *, history=None):
    """Validate durable identity without opening metadata or array files."""
    from pathlib import Path
    _require(isinstance(value, Mapping) and set(value) == {
        'binding', 'layout_sha256', 'normalization_sha256'}, 'ARTIFACT_IDENTITY_KEYS')
    binding = _gc_mtf_history_artifact_ref(value['binding'])
    _require(Path(binding['path']).name == 'SURFACES.json'
        and _hash(value['layout_sha256']) and _hash(value['normalization_sha256']), 'ARTIFACT_IDENTITY')
    if history is not None:
        history = require_gc_mtf_history(history)
        _require(value['layout_sha256'] == history['layout']['contract_sha256']
            and value['normalization_sha256'] == history['normalization']['contract_sha256'],
            'ARTIFACT_IDENTITY_HISTORY')
    return {'binding': binding, 'layout_sha256': value['layout_sha256'],
        'normalization_sha256': value['normalization_sha256']}


def read_gc_mtf_history_artifact_metadata(binding):
    """Read only the three bound JSON files, never NPY, base cache or targets."""
    from pathlib import Path
    binding = _gc_mtf_history_artifact_ref(binding)
    directory = Path(binding['path']).parent
    _require(Path(binding['path']).name == 'SURFACES.json', 'ARTIFACT_MANIFEST_NAME')
    manifest = _read_gc_mtf_history_artifact_json(binding)
    _require(manifest.get('schema_version') == 'gx1_gc_mtf_history_physical_surfaces_v1'
        and type(manifest.get('CONTROL_fit_rows')) is int and manifest['CONTROL_fit_rows'] == 0
        and type(manifest.get('TEST_fit_rows')) is int and manifest['TEST_fit_rows'] == 0
        and manifest.get('predictive_value_measured') is False
        and isinstance(manifest.get('surfaces'), Mapping)
        and set(manifest['surfaces']) == set(TIMEFRAMES), 'ARTIFACT_MANIFEST_SCOPE')
    history = {}
    for key, filename in (('layout', 'LAYOUT.json'), ('normalization', 'NORMALIZATION.json')):
        ref = _gc_mtf_history_artifact_ref(manifest.get(key))
        _require(ref['path'] == str(directory / filename), 'ARTIFACT_METADATA_PATH:' + key)
        history[key] = _read_gc_mtf_history_artifact_json(ref)
    history = require_gc_mtf_history(history)
    for tf in TIMEFRAMES:
        surface = manifest['surfaces'][tf]
        _require(isinstance(surface, Mapping) and set(surface) == {
            'native_row_indices', 'available_ns', 'features'}, 'ARTIFACT_ARRAY_INVENTORY:' + tf)
        rows = None
        for key in ('native_row_indices', 'available_ns', 'features'):
            descriptor = surface[key]
            _require(isinstance(descriptor, Mapping) and set(descriptor) == {
                'path', 'sha256', 'size_bytes', 'dtype', 'shape'}, 'ARTIFACT_ARRAY_DESCRIPTOR:' + tf)
            ref = _gc_mtf_history_artifact_ref({k: descriptor[k] for k in ('path', 'sha256')})
            shape = descriptor['shape']
            _require(ref['path'] == str(directory / ('GC_' + tf + '_' + key + '.npy'))
                and type(descriptor['size_bytes']) is int and descriptor['size_bytes'] > 0
                and isinstance(shape, list) and all(type(n) is int and n > 0 for n in shape)
                and descriptor['dtype'] == ('float32' if key == 'features' else 'int64')
                and len(shape) == (2 if key == 'features' else 1), 'ARTIFACT_ARRAY_HEADER:' + tf)
            rows = shape[0] if rows is None else rows
            _require(shape[0] == rows and (key != 'features'
                or shape[1] == history['layout']['widths'][tf]), 'ARTIFACT_ARRAY_SHAPE:' + tf)
    identity = require_gc_mtf_history_artifact_identity({'binding': binding,
        'layout_sha256': history['layout']['contract_sha256'],
        'normalization_sha256': history['normalization']['contract_sha256']}, history=history)
    return {'history': history, 'identity': identity, 'manifest': manifest}


def load_gc_mtf_history_artifact(binding):
    """Verify all fifteen immutable descriptors through the canonical mmap owner."""
    import os
    from pathlib import Path
    from gx1.features import htf_features as htf
    result = read_gc_mtf_history_artifact_metadata(binding)
    directory = Path(result['identity']['binding']['path']).parent
    fd = os.open(directory, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) | getattr(os, 'O_NOFOLLOW', 0))
    surfaces = {}
    try:
        for tf in TIMEFRAMES:
            surface = {'field_names': result['history']['layout']['field_names'][tf]}
            for key in ('native_row_indices', 'available_ns', 'features'):
                descriptor = result['manifest']['surfaces'][tf][key]
                array = htf._load_verified_cache_npy(fd, Path(descriptor['path']).name,
                    expected_sha256=descriptor['sha256'], expected_size_bytes=descriptor['size_bytes'],
                    label='gc_mtf_' + tf.lower() + '/' + key)
                _require(array.dtype.name == descriptor['dtype'] and list(array.shape) == descriptor['shape']
                    and not array.flags.writeable, 'ARTIFACT_ARRAY_LOADED_HEADER:' + tf)
                surface[key] = array
            surfaces[tf] = surface
    finally:
        os.close(fd)
    surfaces = require_gc_mtf_history_surfaces(surfaces, layout=result['history']['layout'])
    return {**result, 'surfaces': surfaces}
