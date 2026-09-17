"""Synthetic owner-chain checks; these do not demonstrate market learning."""
from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from gx1.contracts.unified_exit_reference_policy_v1 import reference_policy_contract
from gx1.contracts.unified_exit_random_access_state_view_v1 import _structured_sha256
from gx1.contracts.unified_exit_random_access_training_v1 import (
    collate_random_access_training_items, run_random_access_training_step,
)
from tests import test_unified_exit_random_access_state_view_v1 as states
from tests.test_liquidation_relative_learning_v1 import relative_fixture, _TwoRowHead, _validate
from tests.test_unified_exit_random_access_training_v1 import _item, _normalization
from tests.test_unified_exit_reference_data_flow_v1 import _batch as legacy_batch


CUTOFF = 2_000_000_000_000_000_000


def _inputs(*, counts=(600, 600), terminals=(False, False), state_index=0):
    contract, item = _item(state_index=state_index)
    options = dict(counts=counts, terminals=terminals,
                   reference_policy=reference_policy_contract(), reference_cutoff_time_ns=CUTOFF)
    view = states._materialize(state_index, **options)
    item['transitions'][0]['state_view'] = view
    item['anchor']['state_view'] = states._materialize(0, anchor=True, **options)
    bindings = {name: view[name] for name in (
        'm1_source_sha256', 'market_closure_authority_sha256',
        'economic_step_manifest_sha256', 'economics_objective_contract_sha256')}
    bindings.update(sampler_contract=contract, normalization_artifact=_normalization(),
                    reference_policy=reference_policy_contract(), reference_cutoff_time_ns=CUTOFF)
    return item, bindings


def _collate(item, bindings):
    return collate_random_access_training_items(
        [item], outer_batch_size=3, device=torch.device('cpu'),
        **{('expected_' + k if k.endswith('_sha256') else k): v for k, v in bindings.items()})


def _run(batch, *, cutoff=CUTOFF):
    model = _TwoRowHead()
    target = copy.deepcopy(model).requires_grad_(False).eval()
    entry = torch.tensor([[2.], [3.], [4.]], requires_grad=True)
    result = run_random_access_training_step(
        model=model, target_model=target, entry_decision_representations=entry,
        target_entry_decision_representations=entry.detach(), batch=batch, grad_accum_steps=1,
        reference_policy=reference_policy_contract(), reference_cutoff_time_ns=cutoff)
    assert model.calls == target.calls == result['backward_calls'] == 1
    assert all(p.grad is None for p in target.parameters())
    assert not result['entry_targets'].requires_grad
    assert not result['targets'].requires_grad
    return result


@pytest.mark.parametrize('count', [2, 3, 121, 122])
@pytest.mark.parametrize('terminal', [False, True])
def test_same_state_zero_reference_drives_entry_and_exit(relative_fixture, count, terminal):
    item, bindings = _inputs(counts=(count, count + 1), terminals=(terminal, False))
    batch = _collate(item, bindings)
    result = _run(batch)
    old = _run(legacy_batch(counts=(count, count + 1), terminals=(terminal, False))[0], cutoff=None)
    q = result['entry_reference_target_evidence']['hold_target_bps']
    torch.testing.assert_close(q, result['targets'][..., 0], rtol=0, atol=0)
    torch.testing.assert_close(result['entry_targets'][1, :2],
                               batch['entry_liquidation_value_bps'][0] + q[0].clamp_min(0), rtol=0, atol=0)
    assert result['entry_targets'][1, 1] > 0  # Observed favourable side beats the entry cost.
    assert old['entry_targets'][1, :2].max() < 0  # Legacy raw teacher remains cost-only here.
    assert result['entry_targets'][:, 2].tolist() == [0, 0, 0]
    assert result['entry_valid_mask'].tolist() == [[False, False, True], [True]*3, [False, False, True]]
    for name in ('targets', 'prediction', 'entry_gradients'):
        torch.testing.assert_close(result[name], old[name], rtol=0, atol=0)
    assert batch['transition_count'] == 1
    assert batch['online_entry_batch_index'].tolist() == [1]
    assert batch['target_entry_batch_index'].tolist() == [1, 1]
    torch.testing.assert_close(batch['importance_weight'], torch.ones(1), rtol=0, atol=0)
    assert item['anchor']['state_view']['loss_weight'] == 0
    anchor = batch['entry_reference_policy_trace']
    assert anchor['boundary_action_valid_mask'][0, 0, 0].item() == (not (terminal and count <= 121))
    assert anchor['boundary_right_censored_mask'][0, 0].item() == (not terminal and count <= 121)
    assert result['entry_bridge_semantics'] == 'observed_reference_anchor_Q_mu_with_greedy_first_action'
    assert result['entry_bridge_binding']['reference_cutoff_time_ns'] == CUTOFF


def test_sampled_exit_does_not_replace_entry_state_zero(relative_fixture):
    item, bindings = _inputs(state_index=4)
    batch = _collate(item, bindings)
    result = _run(batch)
    anchor_result = _run(_collate(*_inputs()))
    torch.testing.assert_close(result['entry_targets'], anchor_result['entry_targets'], rtol=0, atol=0)
    assert item['anchor']['state_view']['current']['state_index'] == 0
    assert batch['reference_boundary_time_ns'][0] > batch['reference_boundary_time_ns'][1]


@pytest.mark.parametrize('anchor', [False, True])
def test_cutoff_is_exact_and_rejects_before_model_input_construction(relative_fixture, anchor):
    options = dict(anchor=anchor, reference_policy=reference_policy_contract())
    view = states._materialize(0, reference_cutoff_time_ns=CUTOFF, **options)
    boundary = view['reference_policy_trace']['boundary']['decision_time_ns']
    exact = states._materialize(0, reference_cutoff_time_ns=boundary, **options)
    assert len(exact['reference_policy_trace']['steps']) == 120
    assert exact['reference_policy_trace']['boundary']['decision_time_ns'] == boundary
    calls = []
    with pytest.raises(RuntimeError, match='REFERENCE_CUTOFF_EXCEEDED'):
        states._materialize(0, reference_cutoff_time_ns=boundary - 1, model_state_times=calls, **options)
    assert calls == []


def test_resealed_trace_cannot_cross_cutoff(relative_fixture):
    item, _ = _inputs()
    view = dict(item['transitions'][0]['state_view'])
    view['reference_policy_trace'] = dict(view['reference_policy_trace'])
    view['reference_policy_trace']['supervision_end_time_ns'] = view['reference_policy_trace']['boundary']['decision_time_ns'] - 1
    view.pop('state_view_sha256')
    view['state_view_sha256'] = _structured_sha256(view)
    with pytest.raises(RuntimeError, match='REFERENCE_CUTOFF_EXCEEDED'):
        _validate(view)


@pytest.mark.parametrize('mutation', ['drop_cutoff', 'different_cutoff', 'drop_policy', 'legacy_anchor', 'legacy_transition'])
def test_collator_requires_explicit_consistent_binding(relative_fixture, mutation):
    item, bindings = _inputs()
    if mutation == 'drop_cutoff':
        bindings.pop('reference_cutoff_time_ns')
    elif mutation == 'different_cutoff':
        bindings['reference_cutoff_time_ns'] -= 1
    elif mutation == 'drop_policy':
        bindings.pop('reference_policy')
    elif mutation == 'legacy_anchor':
        item['anchor']['state_view'] = states._materialize(0, anchor=True)
    else:
        item['transitions'][0]['state_view'] = states._materialize(0, reference_policy=reference_policy_contract())
    with pytest.raises(RuntimeError, match='REFERENCE_'):
        _collate(item, bindings)


@pytest.mark.parametrize('mutation', ['not_opted_in', 'missing_field', 'wrong_cutoff', 'future_boundary', 'bad_clock_type', 'wrong_policy'])
def test_training_binding_rejection_precedes_any_forward(relative_fixture, mutation):
    batch = _collate(*_inputs())
    cutoff = CUTOFF
    if mutation == 'not_opted_in':
        cutoff = None
    elif mutation == 'missing_field':
        batch.pop('entry_reference_policy_trace')
    elif mutation == 'wrong_cutoff':
        cutoff -= 1
    elif mutation == 'future_boundary':
        batch['reference_boundary_time_ns'][1] = CUTOFF + 1
    elif mutation == 'bad_clock_type':
        batch['reference_boundary_time_ns'] = [CUTOFF, CUTOFF]
    else:
        batch['entry_reference_policy_trace']['policy'] = {}
    model = _TwoRowHead()
    target = copy.deepcopy(model).requires_grad_(False).eval()
    with pytest.raises(RuntimeError, match='REFERENCE_ENTRY_'):
        run_random_access_training_step(
            model=model, target_model=target, entry_decision_representations=torch.ones(3, 1),
            target_entry_decision_representations=torch.ones(3, 1), batch=batch, grad_accum_steps=1,
            reference_policy=reference_policy_contract(), reference_cutoff_time_ns=cutoff)
    assert model.calls == target.calls == 0
    assert all(p.grad is None for p in model.parameters())


def test_native_bridge_propagates_binding_and_uses_existing_target_owners(relative_fixture, monkeypatch):
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as native
    item, bindings = _inputs()
    expected = _run(_collate(item, bindings))
    # Only unrelated gate telemetry is stubbed; native collation/targets/backward run.
    monkeypatch.setattr(native, '_unified_exit_gate_view', lambda output: {})
    monkeypatch.setattr(native, '_accumulate_cooperation_gate_epoch', lambda *a, **k: None)
    monkeypatch.setattr(native, '_accumulate_feature_tf_gate_epoch', lambda *a, **k: None)
    adapter = SimpleNamespace(
        random_access_training_bindings_v1=lambda: bindings,
        materialize_random_access_training_item_v1=lambda row, outer_batch_index: item if outer_batch_index == 1 else None)
    model = _TwoRowHead()
    target = copy.deepcopy(model).requires_grad_(False).eval()
    entry = torch.tensor([[2.], [3.], [4.]])
    gradients, stats, targets, valid = native._episode_native_exit_train_v2(
        model=model, target_model=target, entry_decision_representations=entry,
        target_entry_decision_representations=entry.clone(), entry_row_indices=torch.zeros(3, dtype=torch.long),
        dataset=SimpleNamespace(_unified_exit_lifecycle_v2=adapter), device=torch.device('cpu'),
        grad_accum_steps=1, exit_cooperation_gate_epoch={}, exit_feature_tf_gate_epoch={})
    torch.testing.assert_close(targets, expected['entry_targets'], rtol=0, atol=0)
    torch.testing.assert_close(gradients, expected['entry_gradients'], rtol=0, atol=0)
    assert torch.equal(valid, expected['entry_valid_mask'])
    assert stats['entry_bridge_semantics'] == expected['entry_bridge_semantics']
    assert model.calls == target.calls == stats['random_access_backward_calls'] == 1
