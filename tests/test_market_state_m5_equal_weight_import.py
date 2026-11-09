"""Four-tensor import boundary; synthetic decoder only, no market data or fit."""
import copy

import pytest
import torch

from gx1.contracts import market_state_time_query_fold1_fit_state_v1 as state
from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256 as digest
from tests.test_market_state_ta_dedicated_m5_binding_v1 import _model

NAMES = (
    'state_model.m5_side_head.weight', 'state_model.m5_side_head.bias',
    'future_m5_side_head.weight', 'future_m5_side_head.bias',
)
ARTIFACT = {'path': '/synthetic/coefficients.npz', 'sha256': 'a' * 64}
RECIPE = {'path': '/synthetic/PLAN.json', 'sha256': 'b' * 64}


def _case(monkeypatch):
    source, model = _model(monkeypatch), _model(monkeypatch, True)
    state.copy_legacy_weights_to_dedicated_m5(
        source, model, expected_legacy_model_sha256=digest(source.state_dict()))
    coefficients = {
        name: torch.full_like(model.state_dict()[name], 0.125 * (index + 1))
        for index, name in enumerate(NAMES)}
    return model, coefficients


def _import(model, coefficients, **kwargs):
    return state.import_equal_weight_m5_side_heads(model, coefficients,
        expected_model_state_sha256=kwargs.pop('expected_model_state_sha256',
                                               digest(model.state_dict())),
        coefficient_artifact=kwargs.pop('coefficient_artifact', ARTIFACT),
        coefficient_recipe=RECIPE, **kwargs)


def _snapshot(model):
    _, old, _, _, _, trainer = state._deps()
    return {
        'state': copy.deepcopy(model.state_dict()),
        'buffers': old._tree_sha(dict(model.named_buffers())),
        'rng': old._tree_sha(trainer._attended_session_rng_state(device=torch.device('cpu'))),
        'modes': {name: module.training for name, module in model.named_modules()},
        'parameters': [(name, id(p), p.requires_grad)
                       for name, p in model.named_parameters(remove_duplicate=False)],
    }


def test_import_changes_only_four_tensors_and_preserves_identity_modes_buffers_rng(monkeypatch):
    model, coefficients = _case(monkeypatch)
    model.eval()
    model.state_model.m5_side_head.train()  # Preserve mixed module modes too.
    model.future_m5_side_head.weight.requires_grad_(False)
    model.register_buffer('nonpersistent_import_witness', torch.tensor([3.0]), persistent=False)
    before = _snapshot(model)
    originals = {name: value.clone() for name, value in coefficients.items()}
    result = _import(model, coefficients)
    after = _snapshot(model)
    assert result['source_model_state_sha256'] == digest(before['state'])
    assert result['model_state_sha256'] == digest(after['state'])
    assert result['source_model_state_sha256'] != result['model_state_sha256']
    assert result['coefficient_artifact'] == ARTIFACT
    assert result['coefficient_recipe'] == RECIPE
    assert result['optimizer_steps_performed'] == 0
    assert result['fresh_actual_output_baseline_required']
    assert result['parameter_identity_unchanged']
    assert set(result['tensor_changes']) == set(NAMES)
    assert before['state'].keys() == after['state'].keys()
    for name, value in after['state'].items():
        assert torch.equal(value, coefficients[name] if name in NAMES else before['state'][name]), name
    for key in ('buffers', 'rng', 'modes', 'parameters'):
        assert before[key] == after[key], key
    for name in NAMES:
        assert torch.equal(coefficients[name], originals[name])
        change = result['tensor_changes'][name]
        assert change['after_sha256'] == change['coefficient_sha256'] == digest({name: coefficients[name]})
        assert change['before_sha256'] == digest({name: before['state'][name]})
        assert change['changed_elements'] == coefficients[name].numel()
    # Directly exercise the native Linear consumers of the frozen decoder input.
    representation = torch.arange(3 * model.state_model._hidden_width, dtype=torch.float32).reshape(3, -1) / 9
    for head, prefix in ((model.state_model.m5_side_head, 'state_model.m5_side_head'),
                         (model.future_m5_side_head, 'future_m5_side_head')):
        expected = torch.nn.functional.linear(representation, coefficients[prefix + '.weight'],
                                               coefficients[prefix + '.bias'])
        assert torch.equal(head(representation), expected)


@pytest.mark.parametrize('fault', [
    'source_digest', 'dtype_fp64', 'dtype_fp16', 'shape', 'nan', 'inf',
    'missing', 'extra', 'pending_grad', 'reference',
])
def test_bad_import_is_rejected_before_any_weight_change(monkeypatch, fault):
    model, coefficients = _case(monkeypatch)
    kwargs = {}
    name = NAMES[-1]  # A late failure must not partially copy the preceding keys.
    if fault == 'source_digest':
        kwargs['expected_model_state_sha256'] = '0' * 64
    elif fault.startswith('dtype'):
        coefficients[name] = coefficients[name].to(torch.float64 if fault == 'dtype_fp64' else torch.float16)
    elif fault == 'shape':
        coefficients[name] = torch.zeros(2)
    elif fault in ('nan', 'inf'):
        coefficients[name][0] = float(fault)
    elif fault == 'missing':
        coefficients.pop(name)
    elif fault == 'extra':
        coefficients['future_head.bias'] = model.future_head.bias.detach().clone()
    elif fault == 'pending_grad':
        model.future_head.bias.grad = torch.zeros_like(model.future_head.bias)
    elif fault == 'reference':
        kwargs['coefficient_artifact'] = {'path': 'relative.npz', 'sha256': 'a' * 64}
    before = _snapshot(model)
    with pytest.raises(RuntimeError, match='M5_IMPORT|PARTIAL_GRADIENTS'):
        _import(model, coefficients, **kwargs)
    after = _snapshot(model)
    assert digest(before['state']) == digest(after['state'])
    for key in ('buffers', 'rng', 'modes', 'parameters'):
        assert before[key] == after[key]


def test_default_model_is_rejected(monkeypatch):
    model = _model(monkeypatch)
    before = digest(model.state_dict())
    with pytest.raises(RuntimeError, match='M5_IMPORT_MODE'):
        _import(model, {})
    assert digest(model.state_dict()) == before


def test_invalid_binding_is_rejected_before_import(monkeypatch):
    model, coefficients = _case(monkeypatch)
    model.native_F_objective_sha256[0] ^= 1
    before = digest(model.state_dict())
    with pytest.raises(RuntimeError, match='NATIVE_F_HEAD_BINDING'):
        _import(model, coefficients)
    assert digest(model.state_dict()) == before
