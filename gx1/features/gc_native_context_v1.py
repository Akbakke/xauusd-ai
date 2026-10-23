"""One native GC field, availability, source and specialist owner.

A missing value has no market value. Zero is the additive storage identity
behind an explicit observed=0 gate, and must be removed after normalization
before any learned projection. Statistics use only observed physical TRAIN.
"""
from __future__ import annotations
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SCHEMA_VERSION = 'gx1_native_gc_context_v1'
REGISTRY_PATH = 'configs/research/NATIVE_GC_CONTEXT_FIELDS_20261018_049.json'
REGISTRY_SHA256 = '3e444777099a848cd8795688b57a19122d1ac5454f42902a3495e64aad73a757'


def _stream_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):digest.update(chunk)
    return digest.hexdigest()


def _read_registry():
    path = Path(__file__).resolve().parents[2] / REGISTRY_PATH
    if _stream_sha256(path) != REGISTRY_SHA256:
        raise RuntimeError('NATIVE_GC_REGISTRY_HASH')
    result = json.loads(path.read_text())
    if result['schema_version'] != 'gx1_native_gc_context_registry_v1' or result['outcomes_accessed'] is not False:
        raise RuntimeError('NATIVE_GC_REGISTRY_SCOPE')
    return result


REGISTRY = _read_registry()
VALUE_FIELDS = tuple(item['name'] for item in REGISTRY['fields'])
STRUCTURAL_VALUE_FIELDS = frozenset(item['name'] for item in REGISTRY['fields']
    if item.get('role') == 'structural_observation_mask')
OBSERVED_FIELDS = tuple(item['availability_field'] for item in REGISTRY['fields'])
CONTEXT_FIELDS = tuple(name for item in REGISTRY['fields'] for name in (item['name'], item['availability_field']))
SPECIALIST_OWNERS = {name: item['owner'] for item in REGISTRY['fields']
                     for name in (item['name'], item['availability_field'])}
if len(set(CONTEXT_FIELDS)) != len(CONTEXT_FIELDS):
    raise RuntimeError('NATIVE_GC_DUPLICATE_FIELDS')


BBO_TRAILING_SUPPORT_TRANSFORM = 'bbo_trailing_support_gap_minutes'
# Additive218 side inputs only. The immutable049 registry and legacy context
# dimensions are unchanged; the existing structural BBO-valid value is the gate.
TRAILING_SUPPORT_FIELDS = tuple({
    'name': 'gc_' + tf.lower() + '_observed_book_trailing_support_gap_minutes',
    'availability_field': 'gc_' + tf.lower() + '_observed_book_valid',
    'table': 'TRADE_FEATURES_' + tf + '.parquet',
    'column': 'observed_book_last_minute_ns',
    'validity_column': 'observed_book_valid',
    'source_clock_column': 'available_ns',
    'transform': BBO_TRAILING_SUPPORT_TRANSFORM,
    'role': 'bbo_trailing_support_gap',
    'owner': 'momentum_flow_encoder',
    'bucket_minutes': minutes,
} for tf, minutes in (('M5', 5), ('M15', 15), ('H1', 60), ('H4', 240), ('D1', 1440)))
_TRAILING_SUPPORT_BY_NAME = {item['name']: item for item in TRAILING_SUPPORT_FIELDS}


def _require_trailing_support_field(item):
    expected = _TRAILING_SUPPORT_BY_NAME.get(item.get('name'))
    if expected is None or any(item.get(key) != value for key, value in expected.items()):
        raise RuntimeError('NATIVE_GC_TRAILING_SUPPORT_FIELD_CONTRACT')
    return expected


def field_role(name):
    name = str(name).removeprefix('ctx_cont.')
    if name in OBSERVED_FIELDS or name in STRUCTURAL_VALUE_FIELDS:
        return 'gc_observation_mask'
    if name in VALUE_FIELDS or name in _TRAILING_SUPPORT_BY_NAME:
        return 'gc_masked_source_value'
    return 'dense_market_feature'


def masking_pairs(names):
    """Indices on the actual ordered signal or context surface, never an alias copy."""
    positions = {name: i for i, name in enumerate(names)}
    pairs = []
    declared_pairs = tuple(zip(VALUE_FIELDS, OBSERVED_FIELDS)) + tuple(
        (item['name'], item['availability_field']) for item in TRAILING_SUPPORT_FIELDS)
    for value, observed in declared_pairs:
        for prefix in ('', 'ctx_cont.'):
            if prefix + value in positions:
                if prefix + observed not in positions:
                    raise RuntimeError('NATIVE_GC_OBSERVATION_FIELD_MISSING')
                pairs.append((positions[prefix + value], positions[prefix + observed]))
    return tuple(pairs)


def binary_fields(names):
    declared = set(OBSERVED_FIELDS) | STRUCTURAL_VALUE_FIELDS
    return {name for name in names if str(name).removeprefix('ctx_cont.') in declared}


def mtf_history_binary_fields(names):
    """Source-formula first-retouch bits on additive GC history surfaces only."""
    columns = {'ob_bull_first_retouch', 'ob_bear_first_retouch',
        'observed_proxy_ob_bull_first_retouch', 'observed_proxy_ob_bear_first_retouch'}
    declared = {item['name'] for item in REGISTRY['fields'] if item['column'] in columns}
    return binary_fields(names) | {name for name in names
        if str(name).removeprefix('ctx_cont.') in declared}




def _needed_columns(fields):
    columns = {'available_ns'}
    for item in fields:
        columns.add(item['column'])
        if item.get('transform') == BBO_TRAILING_SUPPORT_TRANSFORM:
            _require_trailing_support_field(item)
            columns.add('bar_start_ns')
        for key in ('validity_column', 'source_clock_column'):
            if item.get(key):columns.add(item[key])
    return sorted(columns)


def _load_sources(registry, *, decision_stop_ns=None):
    tables = {}
    for key in dict.fromkeys(item['table'] for item in registry['fields']):
        item = registry['tables'][key];path = Path(item['path'])
        if _stream_sha256(path) != item['sha256']:raise RuntimeError('NATIVE_GC_SOURCE_HASH:' + key)
        fields = [field for field in registry['fields'] if field['table'] == key]
        stop = pd.Timestamp('2026-07-01T00:00:00Z').value
        if decision_stop_ns is not None:stop = min(stop, int(decision_stop_ns))
        table = pq.read_table(path, columns=_needed_columns(fields), filters=[('available_ns', '<=', stop)])
        clock = table['available_ns'].to_numpy().astype(np.int64)
        if np.any(np.diff(clock) <= 0):raise RuntimeError('NATIVE_GC_SOURCE_CLOCK:' + key)
        tables[key] = (clock, table)
    return tables


@lru_cache(maxsize=1)
def _sources():
    return _load_sources(REGISTRY)


def _decision_stamps(decisions):
    times = pd.DatetimeIndex(decisions)
    if times.tz is None or not times.is_monotonic_increasing or times.has_duplicates:
        raise RuntimeError('NATIVE_GC_DECISION_CLOCK')
    stamps = times.as_unit('ns').asi8
    if len(stamps) and stamps[-1] > pd.Timestamp('2026-07-01T00:00:00Z').value:
        raise RuntimeError('NATIVE_GC_TEST_FORBIDDEN')
    return times, stamps


DECISION_AGE_COLUMNS = frozenset(('observed_profile_snapshot_age_minutes', 'observed_spot_basis_age_minutes'))


def _field_values(item, clock, table, stamps, index):
    # The latest bucket is authoritative even if its field/domain is missing.
    # Never search backward for an earlier valid value across a gap or roll.
    n = len(stamps)
    if not len(clock):return np.zeros(n, np.float32), np.zeros(n, bool)
    take = np.maximum(index, 0)
    values = table[item['column']].to_numpy().astype(np.float64)[take]
    observed = (index >= 0) & np.isfinite(values)
    validity = item.get('validity_column')
    if validity:
        mask = table[validity].to_numpy()[take]
        if not np.isin(mask[np.isfinite(mask)], (0, 1)).all():raise RuntimeError('NATIVE_GC_SOURCE_VALIDITY_DOMAIN')
        observed &= np.isfinite(mask) & (mask == 1)
    source_clock = item.get('source_clock_column') or 'available_ns'
    source_raw = table[source_clock].to_numpy()[take]
    source_defined = np.isfinite(source_raw)
    source = np.where(source_defined, source_raw, 0).astype(np.int64)
    if np.any(source_defined & (source_raw != source)):
        raise RuntimeError('NATIVE_GC_SOURCE_CLOCK_PRECISION')
    if np.any((index >= 0) & source_defined & ((source > stamps) | (source > clock[take]))):
        raise RuntimeError('NATIVE_GC_FUTURE_SOURCE_CLOCK:' + item['name'])
    observed &= source_defined
    transform = item.get('transform')
    if transform is None and item['column'] in DECISION_AGE_COLUMNS:
        transform = 'decision_age_minutes'
    if transform == BBO_TRAILING_SUPPORT_TRANSFORM:
        spec = _require_trailing_support_field(item)
        # last is the START of the final admitted, fully covered BBO minute.
        # The output describes support location, never last actual quote age.
        promised = (index >= 0) & np.isfinite(mask) & (mask == 1)
        # Select admitted Arrow endpoints before conversion: nullable int64
        # columns otherwise become float64 and can hide a one-nanosecond fault.
        # The physical owner also uses exact float64 minute timestamps with NaN.
        last_column = table[item['column']].take(take[promised])
        bar_column = table['bar_start_ns'].take(take[promised])
        if last_column.null_count or bar_column.null_count or not source_defined[promised].all():
            raise RuntimeError('NATIVE_GC_TRAILING_SUPPORT_MISSING_ENDPOINT')
        raw_last, raw_bar = last_column.to_numpy(), bar_column.to_numpy()
        if len(raw_last) and (raw_last.dtype not in (np.dtype('int64'), np.dtype('float64'))
                or raw_bar.dtype not in (np.dtype('int64'), np.dtype('float64'))
                or not np.isfinite(raw_last).all() or not np.isfinite(raw_bar).all()):
            raise RuntimeError('NATIVE_GC_TRAILING_SUPPORT_ENDPOINT_PRECISION')
        last, bar = raw_last.astype(np.int64), raw_bar.astype(np.int64)
        available = clock[take][promised]
        minute = 60 * 1_000_000_000
        if (np.any(raw_last != last) or np.any(raw_bar != bar)
                or np.any(last % minute) or np.any(bar % minute)
                or np.any(available - bar != spec['bucket_minutes'] * minute)
                or np.any(last < bar) or np.any(last > available - minute)):
            raise RuntimeError('NATIVE_GC_TRAILING_SUPPORT_ENDPOINT_GRID')
        values = np.full(n, np.nan, np.float64)
        values[promised] = (available - last - minute) / minute
        observed = promised
    elif transform:
        if transform != 'decision_age_minutes' or item['column'] not in DECISION_AGE_COLUMNS:
            raise RuntimeError('NATIVE_GC_SOURCE_TRANSFORM')
        values = (stamps - source) / (60 * 1_000_000_000)
    if item.get('role') == 'structural_observation_mask' and not np.isin(values[observed], (0, 1)).all():
        raise RuntimeError('NATIVE_GC_STRUCTURAL_VALUE_DOMAIN')
    stored = np.zeros(n, np.float32)
    stored[observed] = values[observed].astype(np.float32)
    if not np.isfinite(stored).all():raise RuntimeError('NATIVE_GC_NONFINITE')
    return stored, observed


def build_context(decisions):
    times, stamps = _decision_stamps(decisions)
    sources = _sources()
    joins = {key: np.searchsorted(clock, stamps, side='right') - 1 for key, (clock, _) in sources.items()}
    output = np.zeros((len(times), len(CONTEXT_FIELDS)), np.float32)
    for j, item in enumerate(REGISTRY['fields']):
        values, observed = _field_values(item, *sources[item['table']], stamps, joins[item['table']])
        output[:, 2 * j] = values
        output[:, 2 * j + 1] = observed.astype(np.float32)
    return pd.DataFrame(output, index=times, columns=CONTEXT_FIELDS, copy=False)


def _completion_candidate_map():
    from gx1.features.gc_order_flow_v1 import completion_native_candidates_v1
    return completion_native_candidates_v1()


def build_registry_from_train(candidate_map, table_bindings, train_decisions, *, train_start, train_end,
                              source_terminal, retained_registry=None):
    """A new registry from declared formulas and exact TRAIN decision clocks.

    No targets/evaluation data enter this API. Finite observed constants stay.
    Equal sampled vectors are reported, never used to claim formula aliases.
    Only the same bound source column/domain/clock/transform can be aliased.
    This returns a reviewable object; the caller owns immutable publication.
    """
    declared = _completion_candidate_map()
    if candidate_map != declared:raise RuntimeError('NATIVE_GC_CANDIDATE_MAP_NOT_OWNER_DEFINED')
    times, stamps = _decision_stamps(train_decisions)
    start, end = pd.Timestamp(train_start), pd.Timestamp(train_end)
    if (start.tz is None or end.tz is None or not len(stamps) or start >= end or
            end.value > pd.Timestamp('2026-07-01T00:00:00Z').value or
            stamps[0] < start.value or stamps[-1] >= end.value):
        raise RuntimeError('NATIVE_GC_REGISTRY_TRAIN_SCOPE')
    terminal_path = Path(source_terminal['path'])
    if _stream_sha256(terminal_path) != source_terminal['sha256']:raise RuntimeError('NATIVE_GC_COMPLETION_TERMINAL_HASH')
    terminal = json.loads(terminal_path.read_text())
    if (terminal.get('status') != 'COMPLETE' or terminal.get('exit_code') != 0 or
            terminal.get('source_unchanged') is not True or terminal.get('test_data_used') is not False):
        raise RuntimeError('NATIVE_GC_COMPLETION_NOT_PUBLISHED')
    required = set(item['table'] for item in candidate_map['fields'])
    if set(table_bindings) != required:raise RuntimeError('NATIVE_GC_COMPLETION_TABLE_INVENTORY')
    tables = {}
    for key, binding in table_bindings.items():
        path = Path(binding['path']);entry = terminal['artifacts'].get(key, {})
        if (not path.is_absolute() or path.is_symlink() or path.resolve(strict=True) != path or
                path != terminal_path.parent / key or entry.get('sha256') != binding['sha256'] or
                entry.get('bytes') != path.stat().st_size):
            raise RuntimeError('NATIVE_GC_COMPLETION_TABLE_BINDING')
        tables[key] = dict(binding)
    retained = REGISTRY if retained_registry is None else retained_registry
    fields = []
    for original in retained['fields']:
        item = dict(original);key = item['table'];binding = retained['tables'][key]
        if key in tables and tables[key] != binding:
            key = 'retained::' + key
        if key in tables and tables[key] != binding:raise RuntimeError('NATIVE_GC_RETAINED_SOURCE_COLLISION')
        tables[key] = dict(binding);item['table'] = key;fields.append(item)
    existing = {item['name'] for item in fields}
    for original in candidate_map['fields']:
        if original['name'] in existing:raise RuntimeError('NATIVE_GC_CANDIDATE_NAME_COLLISION')
        existing.add(original['name']);item = dict(original)
        item['availability_field'] = item['name'] + '_observed'
        if item['column'] in DECISION_AGE_COLUMNS:item['transform'] = 'decision_age_minutes'
        fields.append(item)
    pending = {'fields': fields, 'tables': tables}
    sources = _load_sources(pending, decision_stop_ns=stamps[-1])
    joins = {key: np.searchsorted(clock, stamps, side='right') - 1 for key, (clock, _) in sources.items()}
    admitted = [];unsupported = [];aliases = [];stats = {};formula_keys = {};vector_groups = {}
    for item in fields:
        values, observed = _field_values(item, *sources[item['table']], stamps, joins[item['table']])
        finite = values[observed];count = int(observed.sum())
        constant = bool(count and np.all(finite == finite[0]))
        digest = hashlib.sha256(observed.astype(np.uint8).tobytes() + values.tobytes()).hexdigest()
        stats[item['name']] = {'decision_rows':len(stamps),'finite_observed':count,'missing':len(stamps)-count,
            'constant_observed':constant,'observed_min':float(finite.min()) if count else None,
            'observed_max':float(finite.max()) if count else None,'masked_float32_train_sha256':digest}
        vector_groups.setdefault(digest, []).append(item['name'])
        if not count:
            unsupported.append({'name':item['name'],'reason':'NO_GENUINE_TRAIN_OBSERVATION'});continue
        transform = item.get('transform')
        formula_key = (tables[item['table']]['sha256'], item['column'], item.get('validity_column'),
            item.get('source_clock_column') or 'available_ns', transform, item.get('role'))
        if formula_key in formula_keys:
            canonical = formula_keys[formula_key]
            if canonical['owner'] != item['owner']:raise RuntimeError('NATIVE_GC_FORMULA_SPECIALIST_CONFLICT')
            aliases.append({'removed':item['name'],'canonical':canonical['name'],
                'proof':'SAME_FULL_SOURCE_HASH_COLUMN_VALIDITY_CLOCK_TRANSFORM_ROLE','formula_key':list(formula_key)})
            continue
        formula_keys[formula_key] = item
        item.update(train_observed=count,train_missing=len(stamps)-count,train_constant=constant)
        admitted.append(item)
    formula_path = Path(__file__).with_name('gc_order_flow_v1.py')
    return {'schema_version':'gx1_native_gc_context_registry_v1','tables':tables,'fields':admitted,
        'unsupported':unsupported,'exact_duplicate_aliases':aliases,'outcomes_accessed':False,
        'train_window':[start.isoformat(),end.isoformat()], 'train_decision_rows':len(stamps),
        'train_decision_clock_sha256':hashlib.sha256(stamps.tobytes()).hexdigest(),
        'candidate_map_sha256':hashlib.sha256(json.dumps(candidate_map,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        'formula_owner':{'path':'gx1/features/gc_order_flow_v1.py','sha256':_stream_sha256(formula_path)},
        'source_terminal':dict(source_terminal),'retained_source_registry':retained,
        'field_report':stats,'equal_train_vector_groups':[names for names in vector_groups.values() if len(names)>1],
        'selection_rule':'retain every observed declared source and observed constant; only proven same-source formula/clock aliases removed',
        'duplicate_sample_equality_is_formula_proof':False,'historical_revision_or_feed_completeness_proven':False,
        'new_physical_generation_exact_train_clock_match_required':True,'native_training_admitted':False}


def require_registry_train_clock_match(decisions, registry=None):
    """Whole physical TRAIN must exactly match the registry's fitted clock."""
    registry = REGISTRY if registry is None else registry
    _, stamps = _decision_stamps(decisions)
    digest = hashlib.sha256(stamps.tobytes()).hexdigest()
    if (registry.get('train_decision_rows') != len(stamps) or
            registry.get('train_decision_clock_sha256') != digest):
        raise RuntimeError('NATIVE_GC_REGISTRY_PHYSICAL_TRAIN_CLOCK_MISMATCH')
    return {'rows':len(stamps),'train_decision_clock_sha256':digest,'exact_match':True}


def require_masked_storage_numpy(values, names):
    data = np.asarray(values)
    for index, name in enumerate(names):
        if str(name).removeprefix('ctx_cont.') in STRUCTURAL_VALUE_FIELDS and not np.isin(data[..., index], (0.0, 1.0)).all():
            raise RuntimeError('NATIVE_GC_STRUCTURAL_VALUE_DOMAIN')
    for value, observed in masking_pairs(names):
        mask = data[..., observed]
        if not np.isin(mask, (0.0, 1.0)).all() or np.any(data[..., value][mask == 0.0] != 0.0):
            raise RuntimeError('NATIVE_GC_MASKED_STORAGE_INVALID')


def mask_normalized_numpy(raw, normalized, names):
    require_masked_storage_numpy(raw, names)
    for value, observed in masking_pairs(names):
        normalized[..., value] = np.where(raw[..., observed] == 1.0, normalized[..., value], 0.0)
    return normalized


@lru_cache(maxsize=16)
def _torch_indices(names, device):
    import torch
    pairs = masking_pairs(names)
    return (torch.tensor([v for v, _ in pairs], dtype=torch.long, device=device),
            torch.tensor([m for _, m in pairs], dtype=torch.long, device=device))


def mask_normalized_torch(raw, normalized, names):
    import torch
    if not masking_pairs(names):
        return normalized
    vi, mi = _torch_indices(tuple(names), raw.device)
    binary = [i for i,name in enumerate(names) if str(name).removeprefix('ctx_cont.') in STRUCTURAL_VALUE_FIELDS]
    if binary:
        values = raw.index_select(-1, torch.tensor(binary, dtype=torch.long, device=raw.device))
        if bool(((values != 0.0) & (values != 1.0)).any()):raise RuntimeError('NATIVE_GC_STRUCTURAL_VALUE_DOMAIN')
    mask = raw.index_select(-1, mi)
    value = raw.index_select(-1, vi)
    if bool((((mask != 0.0) & (mask != 1.0)) | ((mask == 0.0) & (value != 0.0))).any()):
        raise RuntimeError('NATIVE_GC_MASKED_STORAGE_INVALID')
    result = normalized.clone()
    result.index_copy_(-1, vi, torch.where(mask == 1.0, normalized.index_select(-1, vi), 0.0))
    return result


def contract():
    return {'schema_version': SCHEMA_VERSION, 'registry_path': REGISTRY_PATH,
            'registry_sha256': REGISTRY_SHA256, 'value_fields': list(VALUE_FIELDS),
            'observed_fields': list(OBSERVED_FIELDS), 'specialist_owners': SPECIALIST_OWNERS,
            'structural_binary_identity_value_fields': sorted(STRUCTURAL_VALUE_FIELDS),
            'missing_market_values_imputed': False, 'normalization_fit': 'all observed physical TRAIN rows only',
            'runtime': 'explicit observation gate after normalization before learned projections',
            'test_source_read': False}
