"""Sampled Exit measurements must use the same target as native TRAIN."""
import copy
from types import SimpleNamespace

import pytest
import torch
import pandas as pd

from gx1.scripts import run_unified_exit_random_access_val_v1 as val
from gx1.contracts.unified_exit_reference_policy_v1 import reference_policy_contract
from tests.test_liquidation_relative_learning_v1 import relative_fixture, _TwoRowHead
from tests.test_coherent_reference_entry_v1 import _inputs, _collate, _run, CUTOFF
from tests.test_chronological_control_targets import _factory, _context


@pytest.mark.parametrize('count,offset', [(6, 4), (125, 4), (126, 4), (600, 7)])
def test_sampled_exit_target_is_bit_exact_native_train(relative_fixture, count, offset):
    batch = _collate(*_inputs(counts=(count, count), state_index=offset))
    trained = _run(batch)
    factory = _factory(batch, count)
    model = _TwoRowHead().eval()
    teacher = copy.deepcopy(model).requires_grad_(False)
    output = {val.UNIFIED_EXIT_MODEL_REPRESENTATION_KEY: torch.tensor([[3.]])}
    with torch.inference_mode():
        rows, targets = val._bounded_reference_exit_observations(
            model=model, boundary_model=teacher, entry_output=output,
            boundary_entry_output=output, state_factory=factory, child_rows=[0],
            device=torch.device('cpu'), reference_policy=reference_policy_contract(),
            reference_cutoff_time_ns=CUTOFF, return_reference_targets=True,
            start_state_indices=[offset])
    length = min(120, count - 1 - offset)
    torch.testing.assert_close(targets['hold_target_bps'], trained['targets'][..., 0], rtol=0, atol=0)
    torch.testing.assert_close(torch.tensor(rows[0]['prediction_hold_bps']), trained['prediction'][0, :, 0], rtol=0, atol=0)
    assert rows[0]['state_index'] == offset
    assert rows[0]['observed_backup_steps'] == length
    assert rows[0]['boundary_censored'] == [offset + length == count - 1] * 2
    assert min(rows[0]['boundary_bootstrap_weight']) > 0
    assert factory.calls == [offset, offset + length]
    assert model.calls == teacher.calls == 1
    assert not targets['hold_target_bps'].requires_grad
    assert all(p.grad is None for p in teacher.parameters())


def test_multiple_sampled_states_keep_order_and_repeat_entry_identity(relative_fixture):
    offsets = [7, 4, 7]
    batch = _collate(*_inputs())
    factory = _factory(batch, 600)
    model = _TwoRowHead().eval()
    teacher = copy.deepcopy(model).requires_grad_(False)
    output = {val.UNIFIED_EXIT_MODEL_REPRESENTATION_KEY: torch.full((3, 1), 3.)}
    with torch.inference_mode():
        rows, targets = val._bounded_reference_exit_observations(
            model=model, boundary_model=teacher, entry_output=output,
            boundary_entry_output=output, state_factory=factory, child_rows=[0, 0, 0],
            device=torch.device('cpu'), reference_policy=reference_policy_contract(),
            reference_cutoff_time_ns=CUTOFF, return_reference_targets=True,
            start_state_indices=offsets)
    expected = torch.cat([_run(_collate(*_inputs(state_index=i)))['targets'][..., 0] for i in offsets])
    torch.testing.assert_close(targets['hold_target_bps'], expected, rtol=0, atol=0)
    assert [r['state_index'] for r in rows] == offsets
    assert factory.calls == offsets + [i + 120 for i in offsets]


@pytest.mark.parametrize('offsets', [[], [0, 1], [-1], [1.5], [True], [599], [600]])
def test_invalid_sampled_offsets_fail_before_prices_or_model(relative_fixture, offsets):
    factory = _factory(_collate(*_inputs()), 600)
    def forbidden(*args, **kwargs):
        raise AssertionError('invalid sample reached prices or model state')
    factory.economic_step_provider = SimpleNamespace(materialize_training_projection=forbidden)
    factory.materialize_state = forbidden
    with pytest.raises(RuntimeError, match='BOUNDED_REFERENCE_(STATE_INDICES_INVALID|SUCCESSOR_REQUIRED)'):
        val._bounded_reference_exit_observations(
            model=None, boundary_model=None, entry_output={}, boundary_entry_output={},
            state_factory=factory, child_rows=[0], device=torch.device('cpu'),
            reference_policy=reference_policy_contract(), reference_cutoff_time_ns=CUTOFF,
            start_state_indices=offsets)


def test_sampled_cutoff_uses_shifted_boundary_close_before_prices(relative_fixture):
    factory = _factory(_collate(*_inputs()), 600)
    offset = 7
    close = int(factory.times.asi8[479 + offset + 120]) + 60_000_000_000
    lengths, _, _, trace = val._bounded_reference_trace(
        state_factory=factory, child_rows=[0], device=torch.device('cpu'),
        reference_policy=reference_policy_contract(), reference_cutoff_time_ns=close,
        start_state_indices=[offset])
    assert lengths == [120]
    assert not bool(trace['boundary_right_censored_mask'].any())
    def forbidden(*args, **kwargs):
        raise AssertionError('future prices or model state accessed')
    factory.economic_step_provider = SimpleNamespace(materialize_training_projection=forbidden)
    factory.materialize_state = forbidden
    with pytest.raises(RuntimeError, match='SUPPORT_CROSSES_CUTOFF'):
        val._bounded_reference_exit_observations(
            model=None, boundary_model=None, entry_output={}, boundary_entry_output={},
            state_factory=factory, child_rows=[0], device=torch.device('cpu'),
            reference_policy=reference_policy_contract(), reference_cutoff_time_ns=close - 1,
            start_state_indices=[offset])


def _sampled_context(tmp_path):
    args = _context(tmp_path)
    factory = args['candidate_state_factory']
    for entry in factory.entries:
        entry.update(entry_m1_start_row=0, available_state_count=600)
    factory.times = pd.date_range('2026-05-20T00:00Z', periods=600, freq='min')
    args['candidate_sampled_state_indices'] = [[7, 4, 7, 1] for _ in args['parent_rows']]
    return args


def test_entry_report_includes_all_fixed_samples_and_keeps_anchor_targets(tmp_path, monkeypatch):
    args = _sampled_context(tmp_path)
    calls = []
    sampled_records = []
    entry_forwards = []
    def forward(model, seq, snap, **kw):
        entry_forwards.append(model)
        return {val.UNIFIED_EXIT_MODEL_REPRESENTATION_KEY: seq,
                'entry_action_q_bps': seq.expand(-1, 3).contiguous()}
    def reference(**kw):
        children = kw['child_rows']
        offsets = kw.get('start_state_indices')
        if offsets is None:
            calls.append(children)
            q = torch.tensor([[float(i % 13), -2.] for i in children])
            return ([{'entry_row_index': i, 'state_index': 0, 'target_hold_bps': x}
                     for i, x in zip(children, q.tolist())], {'hold_target_bps': q})
        assert children == [i for i in calls[-1] for _ in range(4)]
        assert offsets == [7, 4, 7, 1] * len(calls[-1])
        lookup = dict(zip(args['candidate_child_rows'], args['parent_rows']))
        expected_reps = [float(lookup[i]) for i in children]
        assert kw['entry_output'][val.UNIFIED_EXIT_MODEL_REPRESENTATION_KEY][:, 0].tolist() == expected_reps
        assert kw['boundary_entry_output'][val.UNIFIED_EXIT_MODEL_REPRESENTATION_KEY][:, 0].tolist() == expected_reps
        assert kw['boundary_model'] is args['candidate_target_model']
        assert kw['reference_policy'] == reference_policy_contract()
        assert kw['reference_cutoff_time_ns'] == int(pd.Timestamp('2026-06-01T00:00Z').value)
        rows = [{'entry_row_index': i, 'state_index': offset, 'target_hold_bps': [float(offset), -2.]}
                for i, offset in zip(children, offsets)]
        sampled_records.extend(rows)
        return rows
    monkeypatch.setattr(val, '_model_forward_fp32', forward)
    monkeypatch.setattr(val, '_multi_tf_kwargs_from_batch', lambda *a: {})
    monkeypatch.setattr(val, '_bounded_reference_exit_observations', reference)
    monkeypatch.setattr(val, '_new_active_head_epoch_accumulator', lambda: {})
    monkeypatch.setattr(val, '_accumulate_active_head_epoch', lambda *a: None)
    monkeypatch.setattr(val, '_active_head_epoch_diagnostics', lambda *a: ({}, None))
    monkeypatch.setattr(val, 'accumulate_route_diagnostics_v1', lambda *a: None)
    monkeypatch.setattr(val, 'finalize_route_diagnostics_v1', lambda *a: {})
    _, diag, _ = val._entry_representations(**args)
    assert len(entry_forwards) == 8  # Entry representations reused for all four samples.
    assert len(sampled_records) == 4 * 256
    assert diag['bounded_exit_sampled_observations'] == sampled_records
    assert diag['sampled_state_coordinates_sha256'] == val.canonical_sha256({
        'entry_row_indices': args['candidate_child_rows'],
        'state_indices': args['candidate_sampled_state_indices']})
    assert len(diag['bounded_exit_anchor_observations']) == 256
    for row in diag['bounded_entry_observations']:
        assert row['target_q_bps'] == [-4 + float(row['entry_row_index'] % 13), -4., 0.]


@pytest.mark.parametrize('fault', ['rows', 'samples', 'negative', 'boolean', 'no_successor', 'cutoff'])
def test_invalid_fixed_sample_selection_stops_before_any_entry_forward(tmp_path, monkeypatch, fault):
    args = _sampled_context(tmp_path)
    selected = args['candidate_sampled_state_indices']
    if fault == 'rows':
        selected.pop()
    elif fault == 'samples':
        selected[0].pop()
    elif fault == 'negative':
        selected[0][0] = -1
    elif fault == 'boolean':
        selected[0][0] = True
    elif fault == 'no_successor':
        selected[0][0] = 599
    else:
        args['candidate_state_factory'].times = pd.date_range('2026-06-01T00:00Z', periods=600, freq='min')
    def forbidden(*a, **kw):
        raise AssertionError('invalid selection reached model')
    monkeypatch.setattr(val, '_model_forward_fp32', forbidden)
    with pytest.raises(RuntimeError, match='CHRONOLOGICAL_SAMPLED_'):
        val._entry_representations(**args)
