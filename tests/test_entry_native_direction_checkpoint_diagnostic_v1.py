"""Synthetic C mechanics only: no saved states, physical inputs, fit or CUDA."""
from copy import deepcopy
import hashlib
from types import SimpleNamespace
import sys

import pytest
import numpy as np
import torch

from gx1.contracts import entry_native_direction_checkpoint_diagnostic_v1 as contract
from gx1.scripts import audit_entry_native_direction_checkpoints_v1 as worker


def _ref(name):
    return {'path': contract.PROGRAM + '/' + name, 'sha256': '0' * 64}


def _plan():
    return {'schema_version': contract.SCHEMA, 'operation': contract.OPERATION,
        'admission_stage': contract.STAGE, 'proposal': deepcopy(contract.PROPOSAL),
        'source_head': 'a' * 40, 'policy': {'path': contract.REPO + '/NEXT_RUN_POLICY.json', 'sha256': '0' * 64},
        'source_owners': {name: {'path': contract.REPO + '/' + name, 'sha256': '0' * 64} for name in contract.NEW_FILES},
        'run_root': contract.PROGRAM + '/C_CHECKPOINT_DIAGNOSTIC_SYNTHETIC',
        'budget': deepcopy(contract.RESOURCE_LIMITS), 'operation_limits': deepcopy(contract.CALL_LIMITS),
        'initial_report': deepcopy(contract.INITIAL_REPORT),
        'initial_predictions': deepcopy(contract.INITIAL_PREDICTIONS), 'max_invocations': 1,
        'execution_authorized': True, 'data_reads_authorized': True, 'checkpoint_reads_authorized': True,
        'genuine_forwards_authorized': True, 'gradient_reverse_authorized': True,
        'optimizer_steps_authorized': False, 'test_data_used': False}


def _proposal():
    return {'schema_version': 'gx1_direction_C_metadata_proposal_v1',
        'planned_saved_states': [{'role': role, 'consumed_parent_cursor': cursor, 'updates': updates,
            'planned_PT_ref': {'path': contract.BASE + '/RUNTIME/prefix-fit/states/' + name, 'sha256': digest},
            'declaring_JSON': _ref('SYNTHETIC_DECLARATION_' + role + '.json')}
            for role, cursor, updates, name, digest in contract.STATE_SPEC],
        'fixed_panels': {'prefix_rows': 47602, 'seed': 20260911,
            'slices': {name: [left, right] for name, left, right in contract.PANELS}},
        'minimal_operations': {'OOF_or_TEST_forward_calls': 0, 'autograd_reverse_calls': 6,
            'checkpoint_body_loads_total': 4, 'checkpoint_selection': False, 'gradient_forward_calls': 2,
            'model_source_or_architecture_changes': 0, 'new_probe_fits': 0, 'optimizer_steps': 0,
            'panel_batch_size': 16, 'panel_forward_calls': 192,
            'reuse_each_decoded_state_across_phases': True, 'total_forward_calls': 194}}


def test_exact_plan_proposal_and_bound_initial_are_plain_metadata():
    plan, proposal = _plan(), _proposal()
    assert contract.require_plan(plan) == plan
    assert contract.require_proposal(proposal) == proposal
    for ref in (contract.INITIAL_REPORT, contract.INITIAL_PREDICTIONS):
        assert contract.plain_ref(ref) == ref
    changed = contract.require_plan(plan)
    changed['operation_limits']['checkpoint_load'] = 9
    assert plan['operation_limits']['checkpoint_load'] == 4


@pytest.mark.parametrize('fault', ['initial', 'proposal', 'step', 'budget', 'root', 'boolean_invocations'])
def test_corrupt_plan_is_rejected_without_a_loader(fault):
    plan = _plan()
    if fault == 'initial': plan['initial_predictions']['sha256'] = '1' * 64
    elif fault == 'proposal': plan['proposal']['path'] += '.other'
    elif fault == 'step': plan['optimizer_steps_authorized'] = True
    elif fault == 'budget': plan['operation_limits']['panel_forward'] += 1
    elif fault == 'root': plan['run_root'] = contract.BASE + '/RUNTIME/new'
    elif fault == 'boolean_invocations': plan['max_invocations'] = True
    with pytest.raises(RuntimeError, match='NATIVE_DIRECTION_CHECKPOINT_DIAGNOSTIC_'):
        contract.require_plan(plan)


@pytest.mark.parametrize('fault', ['state_sha', 'state_order', 'cursor', 'panels', 'reverse'])
def test_proposal_identity_cannot_choose_states_or_panels(fault):
    proposal = _proposal()
    if fault == 'state_sha': proposal['planned_saved_states'][0]['planned_PT_ref']['sha256'] = '1' * 64
    elif fault == 'state_order': proposal['planned_saved_states'].reverse()
    elif fault == 'cursor': proposal['planned_saved_states'][1]['consumed_parent_cursor'] += 1
    elif fault == 'panels': proposal['fixed_panels']['slices']['training_order_middle'][0] -= 1
    elif fault == 'reverse': proposal['minimal_operations']['autograd_reverse_calls'] = 7
    with pytest.raises(RuntimeError):
        contract.require_proposal(proposal)


@pytest.mark.parametrize('left,right,cursor,seen', [
    (0, 256, 0, 0), (0, 256, 4096, 256), (23801, 24057, 4096, 0),
    (23801, 24057, 24576, 256), (47346, 47602, 24576, 0),
    (47346, 47602, 47602, 256), (23801, 24057, 23901, 100)])
def test_presentation_seen_status(left, right, cursor, seen):
    result = contract.panel_seen_status(left, right, cursor)
    assert result['seen_rows'] == seen
    assert result['unseen_rows'] == right - left - seen


def test_completed_counts_are_all_actual_calls_and_returns():
    counts = contract.CallBudget()
    with pytest.raises(RuntimeError, match='EXACT_COMPLETED_CALLS'):
        counts.complete()
    for key, limit in contract.CALL_LIMITS.items():
        for _ in range(limit):
            counts.start(key)
            counts.returned(key)
    assert counts.complete()['calls'] == contract.CALL_LIMITS
    assert counts.complete()['optimizer_steps'] == 0
    with pytest.raises(RuntimeError, match='CALL_CEILING_torch_load'):
        counts.start('torch_load')
    with pytest.raises(RuntimeError, match='RETURN_reverse'):
        counts.returned('reverse')


def test_profile_limits_a_real_delegated_call_before_the_fifth_body():
    counts = contract.CallBudget()
    entered = []
    def delegated():
        entered.append(1)
        return 'synthetic'
    old = sys.getprofile()
    try:
        sys.setprofile(worker._profile_budget(counts, {delegated.__code__: 'torch_load'}, {'name': 'LOAD'}))
        for _ in range(4):
            delegated()
        with pytest.raises(RuntimeError, match='CALL_CEILING_torch_load'):
            delegated()
    finally:
        sys.setprofile(old)
    assert len(entered) == 4
    assert counts.calls['torch_load'] == counts.returns['torch_load'] == 4


@pytest.mark.parametrize('filename,name', [
    (contract.REPO + '/gx1/anything.py', 'fit_something'),
    ('/synthetic/site-packages/torch/optim/adamw.py', 'step'),
    ('/synthetic/site-packages/torch/_tensor.py', 'backward'),
    ('/synthetic/site-packages/torch/serialization.py', 'save')])
def test_profile_denies_updates_fit_and_extra_serialization(filename, name):
    def body(): pass
    frame = SimpleNamespace(f_code=body.__code__.replace(co_filename=filename, co_name=name))
    with pytest.raises(RuntimeError, match='FORBIDDEN_OPERATION'):
        worker._profile_budget(contract.CallBudget(), {}, {'name': 'PANEL'})(frame, 'call', None)


def test_gradient_none_zero_opposition_and_zero_norm_are_distinct():
    p, q = torch.zeros(2, requires_grad=True), torch.zeros(1, requires_grad=True)
    gradients = [(torch.tensor([1., 0.]), None),
        (torch.tensor([-1., 0.]), torch.zeros(1)), (torch.zeros(2), None)]
    result = worker.gradient_geometry([('backbone.route.weight', p), ('backbone.other.bias', q)], gradients, torch)
    route = result['backbone.route']
    assert route['cosines']['0_1'] == -1.
    assert route['cosines']['0_2'] is route['cosines']['1_2'] is None
    assert route['zero_parameters'] == [0, 0, 1]
    other = result['backbone.other']
    assert other['none_parameters'] == [1, 0, 1]
    assert other['zero_parameters'] == [0, 1, 0]
    assert worker._module_scope('backbone.mtf_family_encoder.family.layers.0.weight')[1] == 'shared_across_timeframes'
    assert worker._module_scope('backbone.mtf_family_proj.family.weight')[1] == 'shared_across_timeframes'
    assert worker._module_scope('backbone.gc_mtf_family_proj.m5__family.weight')[1] == 'named_parameter_route'


def test_horizon_contributions_use_total_eligible_not_each_horizon_mean():
    cutoff = 2_000_000_000_000
    decision = torch.tensor([1, 2], dtype=torch.int64)
    logits = torch.zeros((2, 3), dtype=torch.float32, requires_grad=True)
    target = {'sign': torch.tensor([[1, 0, -1], [-1, 1, 0]], dtype=torch.int8),
        'observed': torch.tensor([[True, True, True], [True, True, False]]),
        'decision_ns': decision,
        'terminal_time_ns': decision[:, None] + torch.tensor([1, 5, 15])[None, :] * 60_000_000_000}
    pieces, counts, per_horizon = worker.horizon_contributions(logits, target, cutoff, torch)
    assert counts['eligible_cells'] == 4
    assert [item['eligible_cells'] for item in per_horizon] == [2, 1, 1]
    torch.testing.assert_close(torch.stack(pieces), torch.log(torch.tensor(2.)) * torch.tensor([.5, .25, .25]))
    assert all(piece.requires_grad for piece in pieces)
    assert logits.grad is None


@pytest.fixture
def measurement_case():
    with torch.random.fork_rng(devices=[]):
        model = torch.nn.Linear(2, 1)
        model.register_buffer('synthetic_buffer', torch.zeros(1))
        model.train()
        state = {'step': 0}
        optimizer = SimpleNamespace(state_dict=lambda: dict(state))
        def identity(model, _device, _tree, _trainer):
            return {'parameters': [p.detach().tolist() for p in model.parameters()],
                'buffer': model.synthetic_buffer.tolist(),
                'modes': {name: m.training for name, m in model.named_modules()},
                'rng': hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()}
        prefix = SimpleNamespace(_online_identity=identity)
        trainer = SimpleNamespace(_attended_session_rng_state=lambda **_kw: torch.get_rng_state(),
            _restore_attended_session_rng_state=lambda value, **_kw: torch.set_rng_state(value))
        yield model, optimizer, state, prefix, trainer


def test_eval_measurement_restores_modes_and_training_rng(measurement_case):
    model, optimizer, _, prefix, trainer = measurement_case
    initial_rng = torch.get_rng_state().clone()
    with worker._measurement(model, optimizer, device='cpu', trainer=trainer,
            prefix=prefix, tree_sha=lambda value: value, evaluation=True):
        assert not model.training
    assert model.training
    with worker._measurement(model, optimizer, device='cpu', trainer=trainer,
            prefix=prefix, tree_sha=lambda value: value, evaluation=False):
        torch.rand(4)
    torch.testing.assert_close(torch.get_rng_state(), initial_rng, rtol=0, atol=0)


@pytest.mark.parametrize('fault', ['weight', 'buffer', 'optimizer', 'gradient'])
def test_measurement_does_not_hide_mutations_by_restoring_weights(measurement_case, fault):
    model, optimizer, state, prefix, trainer = measurement_case
    with pytest.raises(RuntimeError, match='MEASUREMENT_STATE_CHANGED'):
        with worker._measurement(model, optimizer, device='cpu', trainer=trainer,
                prefix=prefix, tree_sha=lambda value: value, evaluation=True):
            if fault == 'weight':
                with torch.no_grad(): model.weight.add_(1)
            elif fault == 'buffer': model.synthetic_buffer.add_(1)
            elif fault == 'optimizer': state['step'] += 1
            elif fault == 'gradient': model.weight.grad = torch.zeros_like(model.weight)


def _recorder_shell():
    value = object.__new__(worker.ActivationRecorder)
    value.torch, value.records, value.seen = torch, {}, set()
    value.expected = {'route': (16, 2, 3)}
    return value


def test_hook_shape_duplicate_and_missing_route_are_fatal():
    recorder = _recorder_shell()
    with pytest.raises(RuntimeError, match='HOOK_SHAPE_OR_DUPLICATE'):
        recorder._capture('route', torch.zeros(16, 3, 2), (16, 2, 3))
    recorder._capture('route', torch.zeros(16, 2, 3), (16, 2, 3))
    with pytest.raises(RuntimeError, match='HOOK_SHAPE_OR_DUPLICATE'):
        recorder._capture('route', torch.zeros(16, 2, 3), (16, 2, 3))
    recorder.end()
    with pytest.raises(RuntimeError, match='MISSING_OR_EXTRA_HOOK_ROUTES'):
        recorder.end()
    assert recorder.records['route']['rows'] == 16
    assert recorder.records['route']['extras']['last_minus_mean_square_mean'] == 0.


def test_actual_monitored_code_owners_and_reverse_counts_without_native_calls():
    from gx1.scripts import run_entry_native_direction_prefix_fit_window_v1 as prefix
    from gx1.contracts import entry_native_direction_feature_view_v1 as view
    from gx1.contracts import entry_native_direction_research_state_v1 as state
    from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
    def physical_prepare(): pass
    def axes_loader(): pass
    def reuse_mapper(): pass
    context = {'physical': SimpleNamespace(prepare_inputs=physical_prepare),
        'ancestor': SimpleNamespace(read_axes=axes_loader, map_reuse=reuse_mapper)}
    codes = worker._monitored_codes(context, np, torch, prefix, view, trainer, state)
    assert len(codes) == 20
    assert codes[torch.load.__code__] == 'torch_load'
    assert codes[np.load.__code__] == 'numpy_load'
    assert codes[torch.autograd.grad.__code__] == 'autograd_grad'
    assert codes[torch.autograd.graph._engine_run_backward.__code__] == 'engine_reverse'
    counts = contract.CallBudget()
    old = sys.getprofile()
    try:
        sys.setprofile(worker._profile_budget(counts, codes, {'name': 'SYNTHETIC_REVERSE'}))
        for _ in range(2):
            x = torch.zeros(3, requires_grad=True)
            losses = [x[h].square() for h in range(3)]
            for h, loss in enumerate(losses):
                gradient, = torch.autograd.grad(loss, [x], retain_graph=h < 2)
                assert gradient.shape == (3,)
            assert x.grad is None
    finally:
        sys.setprofile(old)
    assert counts.calls['autograd_grad'] == counts.returns['autograd_grad'] == 6
    assert counts.calls['engine_reverse'] == counts.returns['engine_reverse'] == 6
    assert counts.calls['native_forward'] == counts.calls['research_constructor'] == 0
    assert counts.calls['torch_load'] == counts.calls['physical_prepare'] == 0
    assert contract.RUNTIME_ROUTE_QUALIFIED is False
