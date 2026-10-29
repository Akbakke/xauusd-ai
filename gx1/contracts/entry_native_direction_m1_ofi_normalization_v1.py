"""Source-bound M1 OFI training-prefix statistics, research only.

A caller must separately qualify source files, authoritative contract identity,
and obtain admission for any genuine fit. This module performs no file I/O and
confers no training, scoring, launch, or trading authority.
"""
from __future__ import annotations
from collections.abc import Mapping
import hashlib
import json
import numpy as np
from gx1.contracts import entry_closed_m1_ofi_window_v1 as windows
from gx1.contracts import entry_model_native_input_normalization_v1 as native_norm

SURFACE = 'research_m1_ofi'
FIELDS = ['m1_ofi_contracts']


def _require(value, reason):
    if not value:
        raise RuntimeError('NATIVE_DIRECTION_M1_OFI_NORMALIZATION_' + reason)


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _axis_sha256(value):
    return hashlib.sha256(memoryview(np.ascontiguousarray(value, dtype='<i8')).cast('B')).hexdigest()


def fit(source, clock, *, prefix_rows, cutoff_ns):
    """Fit only observed minutes of decisions strictly before own TRAIN cutoff."""
    _require(isinstance(source, windows.QualifiedSource) and isinstance(clock, windows.QualifiedClock), 'SOURCE_CLOCK')
    _require(type(prefix_rows) is int and 0 < prefix_rows <= len(clock.parent_rows)
        and type(cutoff_ns) is int and cutoff_ns > 0
        and bool(np.all(clock.decision_ns[:prefix_rows] < cutoff_ns))
        and bool(np.all(clock.decision_ns[prefix_rows:] >= cutoff_ns)), 'STRICT_PREFIX_CUTOFF')
    batch = windows.gather(source, clock, batch_parent_rows=clock.parent_rows[:prefix_rows])
    observed = batch['ofi_contracts'][batch['ofi_observed']]
    _require(len(observed) >= 2 and bool(np.isfinite(observed).all()), 'OBSERVED_MINUTES')
    surface = native_norm.fit_surface_normalization(np.ascontiguousarray(observed[:, None]),
        surface=SURFACE, field_names=FIELDS, semantic_categorical_domains={}, semantic_binary_fields=[],
        allow_constant_train_fields=True)
    value = {'schema_version':'gx1_native_direction_m1_ofi_normalization_v1',
        'source_sha256':source.source_sha256, 'clock_sha256':clock.clock_sha256,
        'roll_sha256':clock.roll_sha256, 'model_contract_clock_bound':clock.roll_sha256 is not None,
        'prefix_rows':prefix_rows, 'cutoff_ns':cutoff_ns, 'observed_minute_count':len(observed),
        'full_rows':len(clock.parent_rows),
        'full_parent_rows_sha256':_axis_sha256(clock.parent_rows),
        'full_decision_ns_sha256':_axis_sha256(clock.decision_ns),
        'prefix_parent_rows_sha256':_axis_sha256(clock.parent_rows[:prefix_rows]),
        'prefix_decision_ns_sha256':_axis_sha256(clock.decision_ns[:prefix_rows]),
        'surface':surface, 'scope':'observed_closed_minutes_in_strict_train_prefix_only',
        'genuine_fit_authorized':False, 'native_launch_authorized':False}
    value['contract_sha256'] = _sha(value)
    return require(value)


def require(value):
    keys={'schema_version','source_sha256','clock_sha256','roll_sha256','model_contract_clock_bound','prefix_rows','cutoff_ns',
        'observed_minute_count','full_rows','full_parent_rows_sha256','full_decision_ns_sha256',
        'prefix_parent_rows_sha256','prefix_decision_ns_sha256','surface','scope',
        'genuine_fit_authorized','native_launch_authorized','contract_sha256'}
    _require(isinstance(value,Mapping) and set(value)==keys
        and value['contract_sha256']==_sha({k:v for k,v in value.items() if k!='contract_sha256'}), 'CONTRACT_SEAL')
    _require(value['schema_version']=='gx1_native_direction_m1_ofi_normalization_v1'
        and value['scope']=='observed_closed_minutes_in_strict_train_prefix_only'
        and value['genuine_fit_authorized'] is False and value['native_launch_authorized'] is False
        and all(type(value[k]) is int and value[k]>0 for k in ('prefix_rows','cutoff_ns','observed_minute_count','full_rows'))
        and value['full_rows'] >= value['prefix_rows'] and value['observed_minute_count'] <= 5*value['prefix_rows']
        and type(value['model_contract_clock_bound']) is bool
        and value['model_contract_clock_bound']==(value['roll_sha256'] is not None), 'CONTRACT_SCOPE')
    if value['roll_sha256'] is not None:
        _require(type(value['roll_sha256']) is str and len(value['roll_sha256'])==64
            and all(c in '0123456789abcdef' for c in value['roll_sha256']), 'ROLL_IDENTITY')
    for key in ('source_sha256','clock_sha256','full_parent_rows_sha256','full_decision_ns_sha256',
        'prefix_parent_rows_sha256','prefix_decision_ns_sha256'):
        digest=value[key]
        _require(type(digest) is str and len(digest)==64 and all(c in '0123456789abcdef' for c in digest), 'SOURCE_IDENTITY')
    surface=native_norm.require_surface_normalization(value['surface'],surface=SURFACE,field_names=FIELDS)
    _require(surface['fit_row_count']==value['observed_minute_count'] and surface['binary_field_count']==0
        and surface['categorical_field_count']==0, 'SURFACE_SCOPE')
    return json.loads(json.dumps(value, allow_nan=False))
