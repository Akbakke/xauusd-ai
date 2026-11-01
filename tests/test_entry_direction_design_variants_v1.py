"""Synthetic coordinate/identity wiring only; no native model is constructed."""
from copy import deepcopy
from types import SimpleNamespace, MethodType

import numpy as np
import pytest
import torch

from gx1.contracts import entry_native_direction_feature_view_v1 as view_owner
from gx1.contracts import entry_native_direction_prefix_normalization_v1 as prefix
from gx1.contracts import entry_gc_mtf_history_v1 as history
from gx1.contracts import gc_relative_price_coordinates_v1 as coordinates
from gx1.contracts.gc_price_coordinate_contract_v1 import build_gc_price_coordinate_source_contract
from gx1.features import entry_direction_momentum_profile_v1 as profiles
from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import EntryV10CtxHybridTransformer
from tests.test_entry_native_direction_feature_view_v1 import bound_case, _ref


def _price_arguments(bound):
    args = deepcopy(bound['arguments'])
    args['source_bindings'].update({key: _ref(key + '.py') for key in
        view_owner.DENSE_SOURCE_REF_KEYS | view_owner.PRICE_MOMENTUM_SOURCE_REF_KEYS})
    return dict(args, dense_minutes=True, price_momentum=True)


def test_initialization_variant_is_explicit_and_leaves_inputs_targets_and_stats_identical(bound_case):
    args = _price_arguments(bound_case)
    old = view_owner.bind_native_direction_feature_view(**args)
    new = view_owner.bind_native_direction_feature_view(**args, price_momentum_initialization='linear_default')
    assert 'price_momentum_initialization' not in old
    assert old['entry_execution_profile']['name'] == profiles.PROFILE_NAME
    assert new['entry_execution_profile']['name'] == profiles.LINEAR_INIT_PROFILE_NAME
    assert new['contract_sha256'] != old['contract_sha256']
    for key in ('prefix_normalization', 'research_objective', 'physical_widths', 'canonical_routing_sha256'):
        assert new[key] == old[key]
    assert view_owner.bind_native_direction_feature_view(**args,
        price_momentum_initialization='linear_default', expected=new) == new
    with pytest.raises(RuntimeError):
        view_owner.bind_native_direction_feature_view(**args, expected=new)
    with pytest.raises(RuntimeError):
        view_owner.bind_native_direction_feature_view(**bound_case['arguments'], price_momentum_initialization='linear_default')


def test_constructor_receives_new_profile_but_is_never_executed(bound_case, monkeypatch):
    args = _price_arguments(bound_case)
    view = view_owner.bind_native_direction_feature_view(**args, price_momentum_initialization='linear_default')
    calls = []
    class StoppedBeforeConstruction(Exception):
        pass
    def stop(**kwargs):
        calls.append(kwargs)
        raise StoppedBeforeConstruction
    monkeypatch.setattr(view_owner, 'EntryV10CtxHybridTransformer', stop)
    with pytest.raises(StoppedBeforeConstruction):
        view_owner.build_native_direction_research_model(view,
            prefix_normalization=args['prefix_normalization'], constructor_metadata=args['constructor_metadata'], device='cpu')
    assert len(calls) == 1
    assert calls[0]['entry_execution_profile'] == view['entry_execution_profile']
    assert calls[0]['gc_mtf_history'] == args['prefix_normalization']['gc_mtf_history']


@pytest.fixture(scope='module')
def relative_case(bound_case):
    case = deepcopy(bound_case['case'])
    layout = coordinates.compile_gc_native_relative_price_layout(history_layout=case['gc_layout'],
        source_contract=build_gc_price_coordinate_source_contract(source_mode='native_current'), tick_size_raw=100000000)
    qualified = prefix.qualify_native_direction_prefix_population(**case)
    fitted = prefix.fit_native_direction_prefix_normalization(qualified, gc_relative_coordinates=layout)
    args = _price_arguments(bound_case)
    args['prefix_normalization'] = fitted
    args['source_bindings'].update({key: _ref(key + '.py') for key in view_owner.GC_RELATIVE_SOURCE_REF_KEYS})
    return case, layout, fitted, args


def test_gc_variant_has_own_stats_and_retains_raw_population_and_old_price_initialization(bound_case, relative_case):
    case, layout, fitted, args = relative_case
    view = view_owner.bind_native_direction_feature_view(**args)
    assert view['gc_relative_coordinates_sha256'] == layout['contract_sha256']
    assert 'price_momentum_initialization' not in view
    assert view['entry_execution_profile']['name'] == profiles.PROFILE_NAME
    assert fitted['base_normalization'] == bound_case['arguments']['prefix_normalization']['base_normalization']
    assert fitted['prefix_qualification'] == bound_case['arguments']['prefix_normalization']['prefix_qualification']
    assert fitted['gc_mtf_history']['layout'] == case['gc_layout']
    assert view['prefix_normalization']['gc_normalization_sha256'] != bound_case['view']['prefix_normalization']['gc_normalization_sha256']
    compiled = profiles.compile_price_momentum_profile(fitted['gc_mtf_history'])
    for tf in history.TIMEFRAMES:
        surface = layout['surfaces'][tf]
        assert compiled['gc_mtf_widths'][tf] == surface['raw_width'] + 2
        assert compiled['gc_mtf_family_indices'][tf] == surface['family_indices']
        assert set(surface['family_indices']) == set(case['gc_layout']['families'])
    with pytest.raises(RuntimeError, match='SEPARATE_DESIGN_VARIANT'):
        view_owner.bind_native_direction_feature_view(**args, price_momentum_initialization='linear_default')
    bad = deepcopy(args)
    del bad['source_bindings'][next(iter(view_owner.GC_RELATIVE_SOURCE_REF_KEYS))]
    with pytest.raises(RuntimeError, match='SOURCE_REF_KEYS'):
        view_owner.bind_native_direction_feature_view(**bad)


def _normalization_spy(gc_history):
    # Actual stat application methods on a plain object: no Module/constructor/forward.
    obj = SimpleNamespace(gc_mtf_history=gc_history,
        _input_normalization_field_names={}, _input_norm_categorical_domains={},
        _input_norm_categorical_index_tuples={})
    for name, stats in history.gc_mtf_normalization_surfaces(gc_history).items():
        obj._input_normalization_field_names[name] = stats['field_names']
        obj._input_norm_categorical_domains[name] = stats['categorical_domains']
        obj._input_norm_categorical_index_tuples[name] = ()
        for suffix, dtype in (('center', torch.float32), ('scale', torch.float32),
            ('binary_mask', torch.bool), ('categorical_mask', torch.bool)):
            setattr(obj, 'input_norm_' + name + '_' + suffix, torch.tensor(stats[suffix], dtype=dtype))
    obj._normalize_input_surface = MethodType(EntryV10CtxHybridTransformer._normalize_input_surface, obj)
    return obj


@pytest.mark.parametrize('tf', history.TIMEFRAMES)
def test_actual_native_preprojection_coordinate_method_matches_numpy_and_preserves_masks(relative_case, tf):
    case, layout, fitted, _ = relative_case
    gc_history = fitted['gc_mtf_history']
    raw = np.ascontiguousarray(case['gc_surfaces'][tf]['features'][:4])
    before = raw.copy()
    tensor = torch.from_numpy(raw).reshape(1, len(raw), raw.shape[1])
    obj = _normalization_spy(gc_history)
    actual = EntryV10CtxHybridTransformer._normalize_gc_history_surface(obj, tensor, timeframe=tf)
    encoded = coordinates.transform_gc_native_relative_price_block(raw, layout=layout, timeframe=tf)['values']
    width = raw.shape[1]
    expected = np.empty_like(encoded)
    for surface, start, end in (('gc_mtf_' + tf.lower(), 0, width), ('gc_anchor_' + tf.lower(), width, width + 1)):
        stats = history.gc_mtf_normalization_surfaces(gc_history)[surface]
        center, scale = np.asarray(stats['center'], np.float64), np.asarray(stats['scale'], np.float64)
        expected[:, start:end] = np.arcsinh((encoded[:, start:end].astype(np.float64) - center) / scale).astype(np.float32)
        binary = np.array(stats['binary_mask'], bool)
        expected[:, np.arange(start, end)[binary]] = encoded[:, np.arange(start, end)[binary]]
    for vi, mi in case['gc_layout']['value_mask_pairs'][tf]:
        expected[encoded[:, mi] == 0, vi] = 0
    expected[:, width + 1] = encoded[:, width + 1]
    expected[encoded[:, width + 1] == 0, width] = 0
    eps = 4 * np.finfo(np.float32).eps
    np.testing.assert_allclose(actual.numpy()[0], expected, rtol=eps, atol=eps)
    np.testing.assert_array_equal(raw.view(np.uint32), before.view(np.uint32))
    assert actual.shape[-1] == layout['surfaces'][tf]['model_width']
    assert not actual.requires_grad
