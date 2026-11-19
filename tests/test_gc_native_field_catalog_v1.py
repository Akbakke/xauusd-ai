"""Separate the pure layout/masks from explicit historical source inspection."""
from pathlib import Path
import subprocess
import sys


def test_static_catalog_exactly_preserves_legacy_fields_owners_and_binary_domains():
    from gx1.features import gc_native_context_v1 as legacy
    from gx1.features import gc_native_field_catalog_v1 as catalog

    # An explicit historical inspection, outside the native TA import path.
    fields = legacy.REGISTRY['fields']
    source_binary = {'structure_available', 'bos_up', 'bos_down', 'sweep_up_event',
        'sweep_down_event', 'ob_bull_active', 'ob_bull_created', 'ob_bull_first_retouch',
        'ob_bear_active', 'ob_bear_created', 'ob_bear_first_retouch'}
    source_binary |= {'observed_proxy_' + name for name in tuple(source_binary)}
    expected = tuple((item['name'], item['owner'],
        item.get('role') == 'structural_observation_mask', item['column'] in source_binary)
        for item in fields)
    assert catalog.FIELD_DESCRIPTORS == expected
    assert catalog.VALUE_FIELDS == tuple(item['name'] for item in fields)
    assert catalog.OBSERVED_FIELDS == tuple(item['availability_field'] for item in fields)
    assert catalog.CONTEXT_FIELDS == tuple(name for item in fields
        for name in (item['name'], item['availability_field']))
    assert catalog.SPECIALIST_OWNERS == {name: item['owner'] for item in fields
        for name in (item['name'], item['availability_field'])}
    assert catalog.STRUCTURAL_VALUE_FIELDS == frozenset(item['name'] for item in fields
        if item.get('role') == 'structural_observation_mask')
    assert catalog.SOURCE_BINARY_VALUE_FIELDS == frozenset(item['name'] for item in fields
        if item['column'] in source_binary)
    assert catalog.TRAILING_SUPPORT_MASK_PAIRS == tuple(
        (item['name'], item['availability_field']) for item in legacy.TRAILING_SUPPORT_FIELDS)
    assert catalog.masking_pairs(catalog.CONTEXT_FIELDS) == tuple((2*i, 2*i+1) for i in range(1205))
    for name in ('VALUE_FIELDS', 'OBSERVED_FIELDS', 'CONTEXT_FIELDS', 'SPECIALIST_OWNERS',
                 'STRUCTURAL_VALUE_FIELDS', 'SOURCE_BINARY_VALUE_FIELDS', 'field_role',
                 'masking_pairs', 'binary_fields', 'mtf_history_binary_fields',
                 'require_masked_storage_numpy', 'mask_normalized_numpy', 'mask_normalized_torch'):
        assert getattr(legacy, name) is getattr(catalog, name)
    payload = repr(catalog.catalog_contract())
    assert not any(text in payload for text in ('path', 'registry', '049', 'M1', 'train_'))


def _run_fresh(script):
    result = subprocess.run([sys.executable, '-c', script], cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_native_ta_import_profile_fit_and_numpy_torch_runtime_never_read_legacy_metadata():
    _run_fresh(r'''
import builtins
import pathlib
import os

def guarded(original):
    def read(file, *args, **kwargs):
        try:
            name = os.fspath(file)
        except TypeError:
            name = ''
        if isinstance(name, str) and ('NATIVE_GC_CONTEXT_FIELDS' in name or 'XAU_M1' in name):
            raise AssertionError('native TA followed legacy metadata: ' + name)
        return original(file, *args, **kwargs)
    return read
builtins.open = guarded(builtins.open)
pathlib.Path.open = guarded(pathlib.Path.open)
from gx1.features import gc_native_context_v1 as legacy
assert 'REGISTRY' not in vars(legacy)
def forbidden(*args, **kwargs):
    raise AssertionError('legacy source reader called')
legacy._read_registry = forbidden
legacy._load_sources = forbidden
legacy.pq.read_table = forbidden
from gx1.features import gc_native_field_catalog_v1 as catalog
from gx1.features.entry_specialist_feature_groups_v1 import compile_full_ta_entry_execution_profile
from gx1.contracts.market_state_native_m5_feature_inputs_v1 import native_m5_feature_provider_contract
from gx1.contracts import entry_model_native_input_normalization_v1 as norm
from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import EntryV10CtxHybridTransformer
import numpy as np
import torch
profile = compile_full_ta_entry_execution_profile()
assert {key: len(value) for key, value in profile['physical_field_names'].items()} == {
    'signal': 318, 'ctx_cont': 132, 'ctx_cat': 1, 'mtf': 256}
assert native_m5_feature_provider_contract()['field_names']['signal'] == list(profile['physical_field_names']['signal'])
raw = np.array([[1.], [2.], [3.]], dtype=np.float32)
stats = norm.fit_surface_normalization(raw, surface='signal', field_names=['ret_1'])
normalized = norm.apply_surface_normalization(raw, stats)
assert np.isfinite(normalized).all()
# Genuine mask domain/runtime exercise also needs no legacy source binding.
names = [catalog.VALUE_FIELDS[0], catalog.OBSERVED_FIELDS[0]]
raw = np.array([[0., 0.], [10., 1.]], dtype=np.float32)
expected = np.array([[0., 0.], [3., 1.]], dtype=np.float32)
assert np.array_equal(catalog.mask_normalized_numpy(raw, np.array([[5.,0.],[3.,1.]],np.float32),names), expected)
actual = catalog.mask_normalized_torch(torch.tensor(raw), torch.tensor([[5.,0.],[3.,1.]]), names)
assert torch.equal(actual, torch.tensor(expected))
assert 'REGISTRY' not in vars(legacy)
''')


def test_legacy_registry_is_read_only_on_explicit_access_and_respects_override():
    _run_fresh(r'''
from gx1.features import gc_native_context_v1 as legacy
assert 'REGISTRY' not in vars(legacy)
calls = []
registry = {'fields': [], 'tables': {}}
def read():
    calls.append(True)
    return registry
legacy._read_registry = read
assert len(legacy.VALUE_FIELDS) == 1205
assert legacy.masking_pairs(['ret_1']) == ()
assert legacy.contract()['value_fields'] == list(legacy.VALUE_FIELDS)
assert calls == []
assert legacy.REGISTRY is registry
assert calls == [True]
override = {'fields': ['explicit override']}
legacy.REGISTRY = override
assert legacy._legacy_registry() is override
assert calls == [True]
''')
