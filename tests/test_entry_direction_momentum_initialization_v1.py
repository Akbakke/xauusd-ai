"""Pure profile and isolated Linear checks; never construct the native model."""
from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from gx1.features import entry_direction_momentum_profile_v1 as owner


MODEL_SOURCE = (Path(__file__).resolve().parents[1]
    / 'gx1/models/entry_v10/entry_v10_ctx_hybrid_transformer.py')


@pytest.fixture
def synthetic_parent(monkeypatch):
    """Isolate profile identity from data loading and native model allocation."""
    mtf = tuple(f'field_{i:03d}' for i in range(249)) + owner.PRICE_MOMENTUM_FIELDS
    family = next(f for f in owner.owners.MODEL_NATIVE_TRAINING_SPECIALISTS
                  if f != owner.MOMENTUM_FAMILY)
    base_indices = tuple(range(178))
    routes = {family: base_indices, owner.MOMENTUM_FAMILY: tuple(range(249, 253))}
    payload = {
        'schema_version': 'synthetic_parent_v1', 'name': 'synthetic_parent',
        'physical_field_names': {'mtf': mtf},
        'active_field_names': {'mtf': mtf[:178]},
        'mtf_indices': base_indices, 'base_mtf_specialists': (family,),
        'mtf_indices_by_specialist': {family: base_indices},
        'preserved_parent_contract': {'sentinel': 'unchanged'},
    }
    parent = {**payload, 'contract_sha256': owner.owners._canonical_sha256(payload)}
    monkeypatch.setattr(owner.owners, 'compile_gc_mtf_geometry_entry_execution_profile',
                        lambda _history: deepcopy(parent))
    monkeypatch.setattr(owner.owners, 'require_multi_tf_specialist_routing_v4',
                        lambda _fields: deepcopy(routes))
    return {}, parent


def _reference(profile):
    return {key: profile[key] for key in ('name', 'contract_sha256')}


def test_linear_profile_changes_only_initialization_identity(synthetic_parent):
    history, parent = synthetic_parent
    legacy = owner.compile_price_momentum_profile(history)
    assert legacy == owner.compile_price_momentum_profile(history, initialization='zero_residual')
    assert legacy['name'] == owner.PROFILE_NAME
    assert 'price_momentum_initialization' not in legacy
    linear = owner.compile_price_momentum_profile(history, initialization='linear_default')
    assert linear['name'] == owner.LINEAR_INIT_PROFILE_NAME
    assert linear['price_momentum_initialization'] == 'linear_default_rng_isolated_v1'
    assert linear['contract_sha256'] != legacy['contract_sha256']
    changed = {'name', 'contract_sha256', 'price_momentum_initialization'}
    assert {k: v for k, v in legacy.items() if k not in changed} == {
        k: v for k, v in linear.items() if k not in changed}
    assert linear['research_parent_entry_execution_profile'] == _reference(parent)
    assert linear['physical_field_names'] == parent['physical_field_names']
    assert linear['mtf_indices_by_specialist'][owner.MOMENTUM_FAMILY] == tuple(range(249, 253))


def test_references_bind_initializer_and_reject_cross_profile_hash(synthetic_parent):
    history, _ = synthetic_parent
    legacy = owner.compile_price_momentum_profile(history)
    linear = owner.compile_price_momentum_profile(history, initialization='linear_default')
    for profile, other in ((legacy, linear), (linear, legacy)):
        reference = _reference(profile)
        assert owner.require_entry_execution_profile_reference(reference, gc_mtf_history=history) == reference
        with pytest.raises(RuntimeError, match='PROFILE_REFERENCE_MISMATCH'):
            owner.require_entry_execution_profile_reference(
                {**reference, 'contract_sha256': other['contract_sha256']}, gc_mtf_history=history)
        with pytest.raises(RuntimeError, match='PROFILE_REFERENCE_INVALID'):
            owner.require_entry_execution_profile_reference(reference)
        with pytest.raises(RuntimeError, match='PROFILE_REFERENCE_INVALID'):
            owner.require_entry_execution_profile_reference(
                {**reference, 'initialization': 'linear_default'}, gc_mtf_history=history)


@pytest.mark.parametrize('bad', [None, False, 0, '', 'normal', 'linear_default_rng_isolated_v1'])
def test_unknown_initialization_fails_before_parent_compilation(monkeypatch, bad):
    def forbidden(*_args, **_kwargs):
        pytest.fail('invalid initializer reached parent compilation')
    monkeypatch.setattr(owner.owners, 'compile_gc_mtf_geometry_entry_execution_profile', forbidden)
    with pytest.raises(RuntimeError, match='PRICE_MOMENTUM_INITIALIZATION_INVALID'):
        owner.compile_price_momentum_profile({}, initialization=bad)


def _constructor_ast():
    tree = ast.parse(MODEL_SOURCE.read_text())
    model = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                 and node.name == 'EntryV10CtxHybridTransformer')
    return next(node for node in model.body if isinstance(node, ast.FunctionDef)
                and node.name == '__init__')


def _execute_nodes(nodes, scope):
    module = ast.Module(body=deepcopy(nodes), type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(MODEL_SOURCE), 'exec'), scope)


def _assignment(constructor, name):
    return next(node for node in ast.walk(constructor) if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == name for target in node.targets))


@pytest.mark.parametrize('mode', ['zero_residual', 'linear_default'])
def test_native_profile_selection_uses_own_identity_without_native_construction(synthetic_parent, mode):
    history, _ = synthetic_parent
    profile = owner.compile_price_momentum_profile(history, initialization=mode)
    scope = {
        'profile_reference': _reference(profile), 'gc_mtf_history': history,
        'PRICE_MOMENTUM_PROFILE_NAMES': owner.PRICE_MOMENTUM_PROFILE_NAMES,
        'price_momentum_initialization': owner.price_momentum_initialization,
        'compile_price_momentum_profile': owner.compile_price_momentum_profile,
    }
    constructor = _constructor_ast()
    _execute_nodes([_assignment(constructor, name) for name in
                    ('price_momentum_mode', 'compiled_profile', 'architecture_profile')], scope)
    assert scope['price_momentum_mode'] == mode
    assert scope['compiled_profile'] == profile
    assert scope['architecture_profile'] == profile['research_parent_entry_execution_profile']


@pytest.mark.parametrize('mode', ['zero_residual', 'linear_default'])
def test_isolated_price_projection_preserves_rng_and_declared_initializer(mode):
    """Execute only the actual nested initializer, never a native constructor."""
    initializer = next(node for node in ast.walk(_constructor_ast())
                       if isinstance(node, ast.FunctionDef) and node.name == '_mtf_projection')
    scope = {
        'torch': torch, 'nn': torch.nn, 'price_momentum': True,
        'price_momentum_mode': mode, 'PRICE_MOMENTUM_FAMILY': owner.MOMENTUM_FAMILY,
        'cleaned_mtf': {owner.MOMENTUM_FAMILY: tuple(range(4)), 'existing': tuple(range(7))},
        'd_model': 128,
    }
    _execute_nodes([initializer], scope)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(8193)
        state = torch.random.get_rng_state().clone()
        actual = scope['_mtf_projection'](owner.MOMENTUM_FAMILY)
        assert torch.equal(torch.random.get_rng_state(), state)
        ordinary = torch.nn.Linear(4, 128)
        if mode == 'zero_residual':
            assert torch.count_nonzero(actual.weight) == 0
            assert torch.count_nonzero(actual.bias) == 0
        else:
            assert torch.equal(actual.weight, ordinary.weight)
            assert torch.equal(actual.bias, ordinary.bias)
            assert torch.count_nonzero(actual.weight) > 0
        # Existing family initialization remains the ordinary consuming path.
        torch.random.set_rng_state(state)
        existing = scope['_mtf_projection']('existing')
        consumed = torch.random.get_rng_state().clone()
        torch.random.set_rng_state(state)
        expected_existing = torch.nn.Linear(7, 128)
        assert torch.equal(existing.weight, expected_existing.weight)
        assert torch.equal(existing.bias, expected_existing.bias)
        assert torch.equal(torch.random.get_rng_state(), consumed)
