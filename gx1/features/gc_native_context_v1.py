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
REGISTRY_PATH = 'configs/research/NATIVE_GC_CONTEXT_FIELDS_20261018_001.json'
REGISTRY_SHA256 = '4ebe7a989a7313d607140ec7025109e0a94de9185d2b360b92abfdaa2d1ebf61'


def _read_registry():
    path = Path(__file__).resolve().parents[2] / REGISTRY_PATH
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != REGISTRY_SHA256:
        raise RuntimeError('NATIVE_GC_REGISTRY_HASH')
    result = json.loads(data)
    if result['schema_version'] != 'gx1_native_gc_context_registry_v1' or result['outcomes_accessed'] is not False:
        raise RuntimeError('NATIVE_GC_REGISTRY_SCOPE')
    return result


REGISTRY = _read_registry()
VALUE_FIELDS = tuple(item['name'] for item in REGISTRY['fields'])
OBSERVED_FIELDS = tuple(item['availability_field'] for item in REGISTRY['fields'])
CONTEXT_FIELDS = tuple(name for item in REGISTRY['fields'] for name in (item['name'], item['availability_field']))
SPECIALIST_OWNERS = {name: item['owner'] for item in REGISTRY['fields']
                     for name in (item['name'], item['availability_field'])}
if len(set(CONTEXT_FIELDS)) != len(CONTEXT_FIELDS):
    raise RuntimeError('NATIVE_GC_DUPLICATE_FIELDS')


def masking_pairs(names):
    """Indices on the actual ordered signal or context surface, never an alias copy."""
    positions = {name: i for i, name in enumerate(names)}
    pairs = []
    for value, observed in zip(VALUE_FIELDS, OBSERVED_FIELDS):
        for prefix in ('', 'ctx_cont.'):
            if prefix + value in positions:
                if prefix + observed not in positions:
                    raise RuntimeError('NATIVE_GC_OBSERVATION_FIELD_MISSING')
                pairs.append((positions[prefix + value], positions[prefix + observed]))
    return tuple(pairs)


def binary_fields(names):
    return {name for name in names if name.removeprefix('ctx_cont.') in OBSERVED_FIELDS}


@lru_cache(maxsize=1)
def _sources():
    tables = {}
    for key in dict.fromkeys(item['table'] for item in REGISTRY['fields']):
        item = REGISTRY['tables'][key]
        path = Path(item['path'])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != item['sha256']:
            raise RuntimeError('NATIVE_GC_SOURCE_HASH:' + key)
        table = pq.read_table(path)
        clock = table['available_ns'].to_numpy().astype(np.int64)
        if len(clock) < 1 or np.any(np.diff(clock) <= 0):
            raise RuntimeError('NATIVE_GC_SOURCE_CLOCK:' + key)
        before_seal = clock <= pd.Timestamp('2026-07-01T00:00:00Z').value
        table = table.filter(before_seal)
        clock = clock[before_seal]
        tables[key] = (clock, table)
    return tables


def build_context(decisions):
    times = pd.DatetimeIndex(decisions)
    if times.tz is None or not times.is_monotonic_increasing or times.has_duplicates:
        raise RuntimeError('NATIVE_GC_DECISION_CLOCK')
    stamps = times.as_unit('ns').asi8
    if len(stamps) and stamps[-1] > pd.Timestamp('2026-07-01T00:00:00Z').value:
        raise RuntimeError('NATIVE_GC_TEST_FORBIDDEN')
    sources = _sources()
    joins = {key: np.searchsorted(clock, stamps, side='right') - 1
             for key, (clock, _) in sources.items()}
    output = np.zeros((len(times), len(CONTEXT_FIELDS)), np.float32)
    for j, item in enumerate(REGISTRY['fields']):
        clock, table = sources[item['table']]
        index = joins[item['table']]
        values = table[item['column']].to_numpy().astype(float)[np.maximum(index, 0)]
        # Use the latest closed bucket, including its missingness. Never search
        # backward for an earlier valid value across a gap or contract roll.
        observed = (index >= 0) & np.isfinite(values)
        output[observed, 2 * j] = values[observed].astype(np.float32)
        output[:, 2 * j + 1] = observed.astype(np.float32)
    if not np.isfinite(output).all():
        raise RuntimeError('NATIVE_GC_NONFINITE')
    return pd.DataFrame(output, index=times, columns=CONTEXT_FIELDS, copy=False)


def require_masked_storage_numpy(values, names):
    data = np.asarray(values)
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
            'missing_market_values_imputed': False, 'normalization_fit': 'all observed physical TRAIN rows only',
            'runtime': 'explicit observation gate after normalization before learned projections',
            'test_source_read': False}
